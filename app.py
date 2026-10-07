"""
Resume Analyzer - RAG-powered resume gap analysis tool.
Hackathon Problem: Students struggle to identify missing skills,
relevant experience, and areas to improve in their resumes.

Architecture:
  - Local knowledge base (role benchmarks + rubric) indexed into ChromaDB
  - Uploaded resume parsed (PDF/DOCX), chunked, and indexed into a temp Chroma collection
  - Dual-stage semantic retrieval feeds a LangChain LCEL chain
  - Gradio minimal UI surfaces the markdown report
"""

import os
import re
import shutil
import tempfile
from dotenv import load_dotenv

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# ⚙️  CONFIGURATION
# ---------------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

API_KEY = os.environ.get("NEXUS_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "❌ API key not found.\n\n"
        "Create a .env file in the project root with:\n"
        "  NEXUS_API_KEY=your-api-key-here\n"
        "  BASE_URL=https://nexusapi.navigatelabs.ai\n"
        "  MODEL_NAME=nova-micro\n\n"
        "Or set the NEXUS_API_KEY environment variable."
    )

BASE_URL = os.environ.get("BASE_URL", "https://nexusapi.navigatelabs.ai")
MODEL_NAME = os.environ.get("MODEL_NAME", "nova-micro")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# 🔌  LLM & EMBEDDINGS
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.2,
)

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=EMBEDDING_MODEL,
)

# -- Fallback: comment out the two lines above and use these instead if
#    Nexus AI does not support embeddings (chat-only key):
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# 📂  KNOWLEDGE BASE CONTENT
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = "./knowledge_base"

KB_FILES = {
    "cloud_devsecops_engineer.txt": """\
ROLE: Cloud & DevSecOps Engineer
ROLE_LABEL: Cloud & DevSecOps Engineer

CORE SKILLS REQUIRED:
- CI/CD pipeline design and maintenance (Jenkins, GitHub Actions, GitLab CI, CircleCI)
- Container orchestration: Kubernetes (kubectl, Helm, Kustomize), Docker
- Infrastructure as Code: Terraform, Pulumi, AWS CloudFormation, Azure Bicep
- Security integration: SAST, DAST, SCA tools (Snyk, Checkov, Trivy, SonarQube)
- Secrets management: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault
- Monitoring & observability: Prometheus, Grafana, ELK Stack, Datadog, Jaeger
- Cloud platforms: AWS (EC2, EKS, ECS, Lambda, RDS, S3, IAM, VPC), Azure, GCP
- Networking: VPNs, firewalls, zero-trust architecture, service mesh (Istio, Linkerd)
- Scripting: Bash, Python, PowerShell for automation

TOOLS & TECHNOLOGIES:
- Source control: Git, GitOps workflows (ArgoCD, Flux)
- Artifact registries: ECR, Docker Hub, JFrog Artifactory, Nexus
- Policy as Code: OPA/Gatekeeper, AWS Config, Azure Policy
- Incident response: PagerDuty, Opsgenie, runbook automation
- Compliance frameworks awareness: SOC2, ISO 27001, NIST, CIS Benchmarks

CLOUD/SYSTEM COMPETENCIES:
- Design multi-region, highly available architectures
- Implement least-privilege IAM roles and RBAC
- Automate vulnerability scanning in CI pipelines (shift-left security)
- Configure auto-scaling, load balancing, and cost optimisation
- Disaster recovery planning: RTO/RPO definition, backup automation

EXPECTED PROJECT DEPTH:
- End-to-end CI/CD pipeline with integrated security gates (SAST + container scanning)
- Kubernetes cluster setup with network policies and RBAC
- Terraform modules managing multi-environment cloud infrastructure
- Real incident postmortem demonstrating observability tooling usage
- Automated compliance reporting dashboard
""",

    "software_engineer.txt": """\
ROLE: Software Engineer
ROLE_LABEL: Software Engineer

CORE SKILLS REQUIRED:
- Proficiency in at least one primary language: Python, Java, Go, C++, TypeScript/JavaScript
- Data structures and algorithms: trees, graphs, dynamic programming, sorting, hashing
- System design: microservices, REST/gRPC APIs, message queues (Kafka, RabbitMQ)
- Databases: relational (PostgreSQL, MySQL) and NoSQL (MongoDB, Redis, DynamoDB)
- Testing: unit, integration, E2E tests; TDD/BDD practices; code coverage tooling
- Version control: Git branching strategies (GitFlow, trunk-based), code review practices
- Web frameworks: FastAPI, Django, Spring Boot, Express, NestJS (language-dependent)
- Frontend basics (if full-stack): React, Next.js, state management

TOOLS & TECHNOLOGIES:
- Build tools: Maven, Gradle, Poetry, npm/yarn, Make
- Containerisation: Docker, Docker Compose
- Cloud basics: deploying apps to AWS/GCP/Azure, serverless functions
- Observability: structured logging, distributed tracing, APM tools
- API design: OpenAPI/Swagger, Postman, GraphQL

CLOUD/SYSTEM COMPETENCIES:
- Design scalable, low-latency services handling concurrent requests
- Implement caching strategies (in-memory, CDN, distributed cache)
- Write idempotent, fault-tolerant distributed systems
- Database indexing, query optimisation, connection pooling
- OAuth 2.0 / JWT authentication flows

EXPECTED PROJECT DEPTH:
- Full-stack or backend application with documented API (OpenAPI spec)
- Demonstrates handling of edge cases, error states, and retries
- Includes a test suite with >70% coverage and CI integration
- Performance benchmark or load test results discussed
- At least one project showing database schema design decisions
""",

    "data_analyst.txt": """\
ROLE: Data Analyst
ROLE_LABEL: Data Analyst

CORE SKILLS REQUIRED:
- SQL: complex joins, window functions, CTEs, query optimisation
- Python for data analysis: pandas, NumPy, matplotlib, seaborn, Plotly
- Statistics: hypothesis testing, A/B testing, regression, correlation, distributions
- Data visualisation: Tableau, Power BI, Looker, or equivalent
- Spreadsheet proficiency: advanced Excel/Google Sheets (VLOOKUP, pivot tables, macros)
- ETL/ELT pipelines: dbt, Airflow, Fivetran, Stitch, or custom Python scripts
- Data warehousing concepts: star/snowflake schema, dimensional modelling
- Business intelligence: KPI definition, dashboard design, stakeholder reporting

TOOLS & TECHNOLOGIES:
- Cloud data platforms: BigQuery, Snowflake, Redshift, Databricks
- Notebook environments: Jupyter, Google Colab, Databricks Notebooks
- Version control for analytics: Git, dbt projects
- Data quality: Great Expectations, dbt tests, anomaly detection
- Communication tools: Confluence, Notion, slide decks for non-technical audiences

CLOUD/SYSTEM COMPETENCIES:
- Ingest, clean, and transform multi-source datasets (APIs, CSVs, databases)
- Build automated reporting pipelines with scheduling
- Perform cohort analysis, funnel analysis, and churn prediction
- Ensure data governance: PII handling, GDPR awareness, access controls

EXPECTED PROJECT DEPTH:
- End-to-end analysis project: data collection → cleaning → EDA → insight → recommendation
- Interactive dashboard published and shared (Tableau Public, Looker Studio)
- A/B test design and results interpretation with statistical significance
- SQL queries demonstrating window functions and performance awareness
- Written narrative explaining findings to a non-technical audience
""",

    "ai_ml_engineer.txt": """\
ROLE: AI/ML Engineer
ROLE_LABEL: AI/ML Engineer

CORE SKILLS REQUIRED:
- Machine learning fundamentals: supervised, unsupervised, reinforcement learning
- Deep learning frameworks: PyTorch, TensorFlow/Keras
- NLP: tokenisation, embeddings, transformers (BERT, GPT family), fine-tuning LLMs
- Computer vision (if applicable): CNNs, object detection (YOLO, Detectron2)
- Feature engineering, data preprocessing, class imbalance handling
- Model evaluation: precision/recall/F1, ROC-AUC, BLEU, perplexity, NDCG
- MLOps: experiment tracking (MLflow, W&B), model versioning, serving (TorchServe, BentoML)
- Vector databases & RAG pipelines: Chroma, Pinecone, Weaviate, FAISS
- Prompt engineering and LLM integration: LangChain, LlamaIndex, OpenAI API

TOOLS & TECHNOLOGIES:
- Data: pandas, NumPy, HuggingFace Datasets, NLTK, spaCy
- Compute: CUDA, distributed training (DDP, DeepSpeed), cloud GPUs (AWS SageMaker, GCP Vertex AI)
- Deployment: FastAPI/Flask model APIs, Docker, Kubernetes for ML workloads
- Monitoring: data drift detection (Evidently AI), model performance dashboards

CLOUD/SYSTEM COMPETENCIES:
- Design end-to-end ML pipelines from raw data to deployed inference endpoint
- Implement efficient batching, quantisation (INT8/FP16), and model compression
- Build retrieval-augmented generation (RAG) systems with hybrid search
- Fine-tune open-source LLMs (LoRA/QLoRA) on domain-specific data
- Ensure reproducibility: seed setting, config management, experiment logging

EXPECTED PROJECT DEPTH:
- Trained and deployed model with documented training pipeline (scripts + configs)
- Demonstrates hyperparameter tuning with tracked experiments (MLflow/W&B)
- RAG system or LLM application with evaluation metrics reported
- Model card describing architecture, training data, limitations, and bias considerations
- Comparison of at least two modelling approaches with analysis
""",

    "cybersecurity_engineer.txt": """\
ROLE: Cybersecurity Engineer
ROLE_LABEL: Cybersecurity Engineer

CORE SKILLS REQUIRED:
- Network security: firewalls, IDS/IPS, VPNs, zero-trust architecture
- Security monitoring & SIEM: Splunk, QRadar, Sentinel, Elastic Stack
- Incident response: forensics, malware analysis, containment, remediation
- Vulnerability management: scanning, patching, risk assessment (CVSS)
- Encryption & PKI: AES, RSA, TLS/SSL, certificate management
- Security frameworks & compliance: NIST CSF, ISO 27001, CIS Benchmarks, SOC2
- Cloud security: AWS Security Hub, Azure Security Center, GCP Security Command Center
- Identity & access management: SSO, MFA, OAuth 2.0, SAML, RBAC/ABAC

TOOLS & TECHNOLOGIES:
- Security tools: Burp Suite, OWASP ZAP, Metasploit, Nmap, Wireshark
- Scripting: Python, PowerShell, Bash for automation and forensics
- Endpoint protection: CrowdStrike, SentinelOne, Microsoft Defender
- Secure coding: SAST/DAST integration (SonarQube, Checkmarx, Veracode)
- Container security: Twistlock, Sysdig, Falco

CLOUD/SYSTEM COMPETENCIES:
- Design and implement security controls for cloud infrastructure (IaC security)
- Conduct red team/blue team exercises and penetration testing
- Develop security policies, procedures, and incident response playbooks
- Implement DevSecOps pipelines with embedded security gates
- Security architecture review and threat modeling (STRIDE, DREAD)

EXPECTED PROJECT DEPTH:
- Full incident response plan with tabletop exercise documentation
- Security audit report with findings and remediation roadmap
- Security automation script (e.g., automated log analysis, threat detection)
- Vulnerability assessment of a live environment with prioritized findings
- Security hardening guide for a specific system or application
""",

    "game_developer.txt": """\
ROLE: Game Developer
ROLE_LABEL: Game Developer

CORE SKILLS REQUIRED:
- Game engines: Unity (C#), Unreal Engine (C++/Blueprints), Godot (GDScript)
- Graphics programming: WebGL, Vulkan, DirectX, OpenGL, shader programming (HLSL, GLSL)
- Physics engines: PhysX, Havok, bullet physics integration
- Networking for games: replication, lag compensation, matchmaking (Photon, Mirror, FishNet)
- Performance optimisation: memory management, draw calls, batching, occlusion culling
- Audio integration: FMOD, Wwise, spatial audio, sound design integration
- Cross-platform development: mobile (iOS/Android), console, PC, VR/AR

TOOLS & TECHNOLOGIES:
- Version control for large assets: Perforce, Plastic SCM, Git LFS
- CI/CD for games: automated builds, regression testing, performance profiling
- Level design tools: Tiled, World Machine, Substance Designer
- Animation: Maya, Blender, Mixamo, state machines, inverse kinematics
- Testing: automated QA, bug tracking, playtest coordination

CLOUD/SYSTEM COMPETENCIES:
- Design scalable multiplayer game architecture (client-server, peer-to-peer)
- Implement anti-cheat systems and server-side validation
- Optimize game performance for target hardware (console, mobile, VR)
- Integrate in-app purchases, analytics, and cloud save systems
- Build and maintain dedicated game servers

EXPECTED PROJECT DEPTH:
- Completed game with polished gameplay loop and UX
- Multiplayer implementation with player synchronization
- Performance profiling report identifying and fixing bottlenecks
- Level design document with gameplay flow and mechanics
- Technical specification for engine integration or custom engine features
""",

    "resume_improvement_rubric.txt": """\
DOCUMENT TYPE: Resume Improvement Rubric
DOC_TYPE: rubric

=== XYZ BULLET FORMULA ===
The gold standard for resume bullet points (from Google's hiring guide):
  "Accomplished [X] as measured by [Y] by doing [Z]"

  X = What you did (the outcome or achievement)
  Y = Quantified impact (metric, percentage, dollar amount, time saved, users affected)
  Z = How you did it (method, tool, technology, approach)

Examples of WEAK → STRONG rewrites:
  WEAK:  "Worked on backend APIs"
  STRONG: "Reduced API response latency by 40% (from 250ms to 150ms) by implementing Redis caching and query indexing on a FastAPI service handling 10K daily requests."

  WEAK:  "Helped improve the CI/CD pipeline"
  STRONG: "Cut deployment cycle time by 60% by redesigning the GitHub Actions CI/CD pipeline to run parallel test suites and caching Docker layers, reducing build time from 18 to 7 minutes."

  WEAK:  "Built a machine learning model"
  STRONG: "Achieved 94.2% F1-score on a multi-class NLP classifier by fine-tuning DistilBERT on 50K labelled customer support tickets, deployed via FastAPI serving 2K daily predictions."

=== ATS FORMATTING GUIDELINES ===
- Use standard section headers: Summary, Skills, Experience, Education, Projects, Certifications
- Avoid tables, text boxes, headers/footers, columns — ATS parsers often fail on these
- Use standard fonts: Arial, Calibri, Times New Roman (10–12pt body, 14–16pt name)
- Save/submit as .pdf (unless employer specifies .docx)
- Keep to 1 page for <3 years experience; 2 pages max for senior roles
- Spell out acronyms at first use: "Kubernetes (K8s)"
- Place most recent experience first (reverse chronological)
- Include a Skills section with comma-separated keywords (ATS keyword matching)
- Avoid personal pronouns (I, me, my)

=== STRONG ACTION VERBS BY CATEGORY ===
Engineering/Development:
  Architected, Engineered, Developed, Implemented, Optimised, Refactored, Automated,
  Designed, Deployed, Integrated, Migrated, Scaled, Debugged, Containerised

Analysis/Research:
  Analysed, Evaluated, Benchmarked, Identified, Investigated, Modelled, Validated,
  Quantified, Synthesised, Forecasted

Leadership/Collaboration:
  Led, Mentored, Coordinated, Collaborated, Facilitated, Presented, Communicated,
  Delivered, Spearheaded, Championed

=== QUANTIFICATION STANDARDS ===
Always try to include at least ONE number per bullet:
  - Performance gains: "reduced by X%", "improved by Xms", "X× faster"
  - Scale: "X users", "X requests/second", "X GB of data", "X microservices"
  - Time: "delivered in X weeks", "saved X hours/week", "within X-month timeline"
  - Team/scope: "led a team of X", "across X repositories", "spanning X countries"
  - Business impact: "$X cost savings", "X% increase in conversion", "X new clients"

If no metric is available, scope it:
  "Built X for a team of 8 engineers, used in production for 3 months"

=== COMMON RESUME MISTAKES TO FLAG ===
1. Generic objective statements instead of a targeted summary
2. Responsibilities listed instead of achievements ("Responsible for..." → rewrite as impact)
3. Missing tech stack in project descriptions
4. Inconsistent tense (use past tense for past roles, present for current)
5. No GitHub/portfolio link for technical roles
6. Soft skills listed without evidence ("Good communicator" → cite presentation/talk)
7. Education section overshadowing sparse work experience (flip order if you have experience)
8. Missing dates on experiences/projects
""",
}

# Map from display name → metadata "role" tag used in ChromaDB
ROLE_DISPLAY_TO_TAG = {
    "Cloud & DevSecOps Engineer": "Cloud & DevSecOps Engineer",
    "Software Engineer":          "Software Engineer",
    "Data Analyst":               "Data Analyst",
    "AI/ML Engineer":             "AI/ML Engineer",
    "Cybersecurity Engineer":     "Cybersecurity Engineer",
    "Game Developer":             "Game Developer",
}

# ---------------------------------------------------------------------------
# 📁  STAGE 1 — KNOWLEDGE BASE AUTO-SETUP
# ---------------------------------------------------------------------------

def setup_knowledge_base() -> None:
    """
    Creates ./knowledge_base/ if absent and writes all benchmark + rubric files.
    Idempotent: safe to call multiple times.
    """
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    for filename, content in KB_FILES.items():
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
    print(f"[KB] Knowledge base ready at '{KNOWLEDGE_BASE_DIR}'")


def _tag_metadata(filename: str) -> dict:
    """
    Derive metadata tags from the knowledge-base filename.
    Rubric file → {"doc_type": "rubric"}
    Role files  → {"doc_type": "role_benchmark", "role": "<Role Label>"}
    """
    if "rubric" in filename:
        return {"doc_type": "rubric"}

    # Extract role label from file content header (ROLE_LABEL field)
    filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    role_label = filename.replace("_", " ").replace(".txt", "").title()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ROLE_LABEL:"):
                    role_label = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"doc_type": "role_benchmark", "role": role_label}


def build_kb_vectorstore() -> Chroma:
    """
    Loads all knowledge-base files, splits into chunks with metadata,
    and indexes them into a persistent ChromaDB collection.

    Returns the Chroma retriever-ready vectorstore.
    """
    print("[KB] Loading and indexing knowledge base into ChromaDB …")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    all_docs = []

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not (filename.endswith(".txt") or filename.endswith(".md")):
            continue
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        metadata = _tag_metadata(filename)
        chunks = splitter.split_text(raw_text)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata=metadata))

    kb_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="kb_collection",
        # persist_directory="./chroma_kb",  # uncomment to persist across restarts
    )
    print(f"[KB] Indexed {len(all_docs)} chunks into ChromaDB.")
    return kb_store


# ---------------------------------------------------------------------------
# 📄  STAGE 2 — MULTI-FORMAT RESUME INGESTION
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalise whitespace artifacts common in PDF/DOCX exports.
    Collapses excessive blank lines and strips leading/trailing spaces per line.
    """
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse multiple spaces (multi-column artefacts)
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()


def ingest_resume(file_path: str) -> Chroma:
    """
    Parses a PDF or DOCX resume, cleans text, splits into semantic chunks,
    tags with {"doc_type": "student_resume"}, and indexes into an in-memory
    per-session Chroma collection.

    Returns the Chroma vectorstore for the resume.
    """
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[Resume] Loading file: {file_path} (type={ext})")

    # --- Load raw documents ---
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Upload a .pdf or .docx file.")

    raw_docs = loader.load()

    # --- Clean and re-wrap as tagged Documents ---
    cleaned_docs = []
    for doc in raw_docs:
        cleaned_text = _clean_text(doc.page_content)
        if cleaned_text:
            cleaned_docs.append(
                Document(
                    page_content=cleaned_text,
                    metadata={"doc_type": "student_resume"},
                )
            )

    if not cleaned_docs:
        raise ValueError("Could not extract any text from the uploaded resume. "
                         "Ensure the file is not scanned/image-only.")

    # --- Chunk with semantic-friendly settings ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    resume_chunks = splitter.split_documents(cleaned_docs)

    # Preserve doc_type tag through splitting
    for chunk in resume_chunks:
        chunk.metadata["doc_type"] = "student_resume"

    # --- Index into a temporary in-memory ChromaDB collection ---
    resume_store = Chroma.from_documents(
        documents=resume_chunks,
        embedding=embeddings,
        collection_name="resume_session",  # overwritten each upload
    )
    print(f"[Resume] Indexed {len(resume_chunks)} chunks from the uploaded resume.")
    return resume_store


# ---------------------------------------------------------------------------
# 🔍  STAGE 3 — DUAL-STAGE SEMANTIC RETRIEVAL
# ---------------------------------------------------------------------------

def retrieve_context(
    kb_store: Chroma,
    resume_store: Chroma,
    selected_role: str,
    k_benchmark: int = 6,
    k_rubric: int = 4,
    k_resume: int = 8,
) -> tuple[str, str, str]:
    """
    Two-pass retrieval:

    Pass 1 — From KB:
      a) Role benchmark chunks filtered by metadata role == selected_role
      b) Rubric chunks filtered by doc_type == rubric

    Pass 2 — From Resume:
      Uses the concatenated benchmark text as a semantic query against the
      student's resume chunks to surface the most relevant experience sections.

    Returns (benchmark_context, resume_context, rubric_context) as plain strings.
    """
    role_tag = ROLE_DISPLAY_TO_TAG.get(selected_role, selected_role)

    # --- Retrieval 1a: Role benchmark ---
    benchmark_docs = kb_store.similarity_search(
        query=f"required skills and competencies for {role_tag}",
        k=k_benchmark,
        filter={"role": role_tag},
    )
    benchmark_context = "\n\n".join(d.page_content for d in benchmark_docs)

    # --- Retrieval 1b: Rubric ---
    rubric_docs = kb_store.similarity_search(
        query="resume improvement bullet point formula quantification ATS",
        k=k_rubric,
        filter={"doc_type": "rubric"},
    )
    rubric_context = "\n\n".join(d.page_content for d in rubric_docs)

    # --- Retrieval 2: Resume — queried with benchmark as semantic signal ---
    semantic_query = (
        f"projects, experience, and skills related to {role_tag}: "
        + benchmark_context[:800]  # use first 800 chars of benchmark as the query
    )
    resume_docs = resume_store.similarity_search(
        query=semantic_query,
        k=k_resume,
        filter={"doc_type": "student_resume"},
    )
    resume_context = "\n\n".join(d.page_content for d in resume_docs)

    return benchmark_context, resume_context, rubric_context


# ---------------------------------------------------------------------------
# 🤖  STAGE 4 — LCEL CHAIN (GROUNDED GENERATION)
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert technical resume reviewer and career coach.
Your task is to analyse a student's resume against a target role benchmark and produce
a concise, actionable report grounded ONLY in the provided context — do not hallucinate
skills or experiences that are not present in the resume text.

=== TARGET ROLE ===
{selected_role}

=== ROLE BENCHMARK (retrieved from knowledge base) ===
{benchmark_context}

=== STUDENT RESUME CONTENT (retrieved relevant sections) ===
{resume_context}

=== RESUME IMPROVEMENT RUBRIC (retrieved guidelines) ===
{rubric_context}

---
Using ONLY the information above, produce a Markdown report with EXACTLY these three sections:

## 1. Missing Skills
List the key skills/tools/competencies required for the **{selected_role}** role (per the benchmark)
that are **absent or not clearly demonstrated** in the student's resume.
Group them under sub-headings (e.g., Core Technical, Tools & Platforms, Cloud/System).
For each missing item, add a one-line note on why it matters for this role.

## 2. Relevant Experience
Identify which specific projects, roles, or experiences from the student's resume **most
closely align** with the target role. For each, explain the semantic match — what skills
or responsibilities it demonstrates and how it maps to the role's requirements.
Be specific; quote or closely paraphrase the resume text.

## 3. Areas That Could Be Improved
For at least 3 weak or unquantified bullet points found in the resume:
- Show the **original text** (labelled "Before:")
- Provide a rewritten version using the **XYZ formula** (labelled "After:")
- Add a one-line note citing which rubric principle was applied.
Also include 2–3 general formatting or ATS observations if relevant.

Keep the tone constructive and specific. Do not include generic advice not grounded in the resume.
"""
)

rag_chain = ANALYSIS_PROMPT | llm | StrOutputParser()


def run_analysis(
    benchmark_context: str,
    resume_context: str,
    rubric_context: str,
    selected_role: str,
) -> str:
    """
    Invokes the LCEL chain with retrieved context and returns the markdown report.
    """
    print("[LLM] Running grounded analysis chain …")
    result = rag_chain.invoke(
        {
            "selected_role":      selected_role,
            "benchmark_context":  benchmark_context,
            "resume_context":     resume_context,
            "rubric_context":     rubric_context,
        }
    )
    return result


# ---------------------------------------------------------------------------
# 🔄  FULL PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

# KB vectorstore is built once at startup and reused across sessions
_kb_store: Chroma | None = None


def get_kb_store() -> Chroma:
    global _kb_store
    if _kb_store is None:
        setup_knowledge_base()
        _kb_store = build_kb_vectorstore()
    return _kb_store


def _split_report(report: str) -> tuple[str, str, str]:
    """
    Splits the LLM markdown report into the three labelled sections.
    Falls back gracefully if section headers are missing.
    """
    import re as _re
    # Match section headers like "## 1. Missing Skills" or "## Missing Skills"
    parts = _re.split(r"(?m)^##\s+\d*\.?\s*", report)
    # parts[0] is anything before the first ##, parts[1..3] are the sections
    sections = [p.strip() for p in parts if p.strip()]

    def _extract(keyword: str) -> str:
        for s in sections:
            if s.lower().startswith(keyword.lower()):
                # Remove the heading line, return body
                lines = s.splitlines()
                return "\n".join(lines[1:]).strip()
        return ""

    missing   = _extract("Missing Skills")   or (sections[0] if len(sections) > 0 else "")
    relevant  = _extract("Relevant Exp")     or (sections[1] if len(sections) > 1 else "")
    improve   = _extract("Areas")            or (sections[2] if len(sections) > 2 else "")

    # Final fallback: if splitting failed just dump full report in first box
    if not any([missing, relevant, improve]):
        return report, "", ""

    return missing, relevant, improve


def analyze_resume(file_obj, selected_role: str) -> tuple[str, str, str, str]:
    """
    Main pipeline function called by Gradio.

    Returns four values:
        status_msg   — short status line shown at the top
        missing      — Missing Skills section markdown
        relevant     — Relevant Experience section markdown
        improve      — Areas to Improve section markdown
    """
    if file_obj is None:
        return "⚠️ Please upload a resume file (.pdf or .docx).", "", "", ""
    if not selected_role:
        return "⚠️ Please select a target role from the dropdown.", "", "", ""

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        # STAGE 2: Parse and index the resume
        resume_store = ingest_resume(file_path)

        # STAGE 3: Dual-stage retrieval
        kb_store = get_kb_store()
        benchmark_ctx, resume_ctx, rubric_ctx = retrieve_context(
            kb_store, resume_store, selected_role
        )

        # STAGE 4: Grounded LLM generation
        report = run_analysis(benchmark_ctx, resume_ctx, rubric_ctx, selected_role)

        # Split into three panels
        missing, relevant, improve = _split_report(report)

        status = f"✅ Analysis complete for **{selected_role}**"
        return status, missing, relevant, improve

    except ValueError as ve:
        msg = f"⚠️ {ve}"
        return msg, "", "", ""
    except Exception as e:
        msg = f"❌ Error: `{e}`\n\nCheck API key / base URL / model name at the top of `app.py`."
        return msg, "", "", ""


# ---------------------------------------------------------------------------
# 🖥️  STAGE 5 — GRADIO UI
# ---------------------------------------------------------------------------

ROLE_OPTIONS = [
    "Cloud & DevSecOps Engineer",
    "Software Engineer",
    "Data Analyst",
    "AI/ML Engineer",
    "Cybersecurity Engineer",
    "Game Developer",
]

_CSS = """
/* ── Page & outer wrapper background only ── */
.gradio-container { background: #0f1117 !important; }
body { background: #0f1117 !important; }

/* ── Header card ── */
#header-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 32px 20px;
    margin-bottom: 8px;
}
#header-card h1 { color: #f1f5f9; font-size: 1.9rem; margin: 0 0 6px; }
#header-card p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }

/* ── Input panel ── */
#input-panel {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
}

/* ── Analyse button ── */
#analyze-btn {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    cursor: pointer;
    transition: opacity 0.2s;
}
#analyze-btn:hover { opacity: 0.88; }

/* ── Status bar ── */
#status-bar { border-radius: 8px; padding: 0 4px; }

/* ── Result column cards — all three identical base style ── */
.result-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    min-height: 220px;
}

/* ── Coloured top accent per card ── */
#card-missing  { border-top: 3px solid #f87171 !important; }
#card-relevant { border-top: 3px solid #34d399 !important; }
#card-improve  { border-top: 3px solid #fbbf24 !important; }

/* ── Section labels inside each card ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 12px !important;
    padding: 0 !important;
}
.label-missing  { color: #f87171 !important; }
.label-relevant { color: #34d399 !important; }
.label-improve  { color: #fbbf24 !important; }

/* ── Markdown text inside cards ── */
.result-card p, .result-card li  { color: #cbd5e1 !important; font-size: 0.9rem; }
.result-card h2, .result-card h3 { color: #f1f5f9 !important; }
.result-card strong              { color: #e2e8f0 !important; }
.result-card code                { background: #0f172a; color: #a5f3fc; padding: 1px 5px; border-radius: 4px; }

/* ── Dropdown cursor ── */
.gradio-dropdown,
.gradio-dropdown *,
.gradio-dropdown input,
.gradio-dropdown svg,
.gradio-dropdown li { cursor: pointer !important; }
"""

with gr.Blocks(title="Resume Analyzer") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Group(elem_id="header-card"):
        gr.HTML("""
            <h1>📄 Resume Analyzer</h1>
            <p>Upload your resume and pick a target role — get a gap analysis in seconds.</p>
        """)

    # ── Inputs ──────────────────────────────────────────────────────────────
    with gr.Group(elem_id="input-panel"):
        with gr.Row():
            file_input = gr.File(
                label="Upload Resume",
                file_types=[".pdf", ".docx"],
                scale=3,
            )
            role_dropdown = gr.Dropdown(
                choices=ROLE_OPTIONS,
                value=ROLE_OPTIONS[0],
                label="Target Role",
                scale=2,
            )

        analyze_btn = gr.Button(
            "🔍  Analyze Resume",
            elem_id="analyze-btn",
            variant="primary",
        )

    # ── Status ──────────────────────────────────────────────────────────────
    status_bar = gr.Markdown(elem_id="status-bar")

    # ── Results (three side-by-side cards) ─────────────────────────────────
    with gr.Row(equal_height=True):

        with gr.Column(elem_id="card-missing", elem_classes=["result-card"]):
            gr.HTML('<p class="section-label label-missing">🔴 Missing Skills</p>')
            out_missing = gr.Markdown()

        with gr.Column(elem_id="card-relevant", elem_classes=["result-card"]):
            gr.HTML('<p class="section-label label-relevant">🟢 Relevant Experience</p>')
            out_relevant = gr.Markdown()

        with gr.Column(elem_id="card-improve", elem_classes=["result-card"]):
            gr.HTML('<p class="section-label label-improve">🟡 Areas to Improve</p>')
            out_improve = gr.Markdown()

    # ── Wire up ─────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_resume,
        inputs=[file_input, role_dropdown],
        outputs=[status_bar, out_missing, out_relevant, out_improve],
    )

# ---------------------------------------------------------------------------
# 🚀  ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pre-warm: create knowledge base files and index into ChromaDB before
    # the first user request so the initial analysis is fast.
    get_kb_store()
    demo.launch(css=_CSS)
