# 🌌 Cosmic Compass
### An Agentic, Multi-Modal RAG System for Astronomy & Cosmology

> *Team KnowlAgents — Gargi Shroff & Ameya Rathod | IIIT Hyderabad*

---

## Overview

**Cosmic Compass** is a production-grade, **agentic Retrieval-Augmented Generation (RAG)** system that functions as a Subject Matter Expert (SME) in Astronomy and Cosmology. It goes far beyond a simple Q&A bot: it plans multi-step tasks, delegates to specialist sub-agents, retrieves multi-modal context (text + images), generates formatted documents (PDF via LaTeX, DOCX), and sends email notifications — all under **human-in-the-loop (HITL) oversight**.

Built with **LangChain**, **LangGraph**, **Elasticsearch**, **CLIP embeddings**, and a **FastAPI** backend, this project demonstrates a complete, real-world deployment of a supervised multi-agent system with interruptible, resumable workflows.

---

## Key Features

### Multi-Agent Architecture (LangGraph + LangChain)
- **Supervisor Agent** — orchestrates task planning, creates structured to-do lists, routes sub-tasks to specialist agents, and chains their outputs
- **Content Generator Agent** — RAG-powered specialist for Q&A, report writing, and quiz generation; handles both PDF (LaTeX) and DOCX output formats with format-specific prompting strategies
- **Email Agent** — strictly-scoped utility agent for composing and dispatching emails with attachments via SMTP

### Human-in-the-Loop (HITL) Workflow
- Agent execution is **paused before any sensitive action** (document generation, email dispatch) using LangGraph's `InMemorySaver` checkpointing
- A **modal UI** surfaces the pending tool call and its arguments for user review
- Users can **Approve**, **Reject**, or **Edit** arguments inline before resuming
- The workflow resumes deterministically from the saved checkpoint via the `/api/chat/resume` endpoint

### Multi-Modal RAG Pipeline
- **Hybrid Retrieval**: combines BM25 keyword search with CLIP (`clip-ViT-B-32`) semantic k-NN search over Elasticsearch
- **Cross-Encoder Reranking** (`ms-marco-MiniLM-L-6-v2`) for precision-optimized result ordering
- **Image-aware retrieval**: dedicated image-only k-NN fallback via `min_images` parameter when visual context is sparse
- **Multi-modal embedding**: CLIP encodes both raw images and text chunks into a shared vector space, enabling unified semantic search

### Sophisticated Preprocessing & Chunking
- Ingests **PDF, DOCX, PPTX, MD, TXT** formats via PyMuPDF, docx2txt, and python-pptx
- **Hierarchical chunking**: 256-token child chunks nested within 1024-token parent chunks, balancing granularity with context retention
- Images are extracted, saved to `outputs/images/`, and referenced via text placeholders in the chunk stream
- JSON metadata manifests per document enable reproducible, incremental indexing

### Document Generation
- **PDF (LaTeX)**: compiles full LaTeX source (including Beamer slides) via `pdflatex` subprocess with correct image dependency resolution
- **DOCX**: generates structured Word documents with embedded images via `python-docx`
- **Email**: multi-attachment dispatch via Gmail SMTP (`smtplib`)

### Middleware & Guardrails
| Middleware | Function |
|---|---|
| `HumanInTheLoopMiddleware` | Intercepts sensitive tool calls for user approval |
| `ContentFilterMiddleware` | Blocks banned keywords (e.g., weapons, drugs) |
| `PIIMiddleware` | Masks/blocks credit card numbers, API keys, and other PII |
| `ToolRetryMiddleware` | Auto-retries failed tool calls up to 3 times |

### Prompting Strategy
- **Structured workflow prompts**: supervisor prompt encodes an explicit step-by-step algorithm ("1. Plan → 2. Delegate → 3. Inspect & Chain") rather than a vague role description
- **Zero-shot vs. one-shot differentiation**: DOCX generation uses zero-shot instruction; LaTeX/Beamer generation uses a one-shot template to enforce syntactic correctness
- **Proactive memory injection**: user email and format preference are captured at login and injected as system context, solving the "forgotten email" failure mode observed in early iterations

---

## System Architecture

```
.
├── api/
│   ├── main2.py          # FastAPI backend — /api/chat and /api/chat/resume
│   └── index2.html       # Interactive frontend with HITL modal UI
├── core/
│   └── agent_self2.py    # Supervisor, Content Generator, Email agents + all middleware
├── data_processing/
│   └── process.py        # Multi-format ingestion, extraction, hierarchical chunking
├── rag/
│   ├── indexer.py        # CLIP embedding + Elasticsearch indexing
│   └── rag_tools.py      # HybridRetriever (BM25 + k-NN + CrossEncoder reranking)
├── tools/
│   └── sme_tools.py      # compile_latex_to_pdf, create_docx_report, send_email
├── docs/                 # Source corpus (astronomy textbooks)
├── outputs/              # chunks/, images/, metadata/ (auto-generated)
└── utils/
    └── scrape_articles.py
```

### Request Lifecycle

```
User Message
    │
    ▼
FastAPI /api/chat
    │
    ▼
supervisor_agent (LangGraph)
    │
    ├─► Plans task, creates to-do list
    │
    ├─► Calls content_generator_agent
    │       │
    │       └─► HybridRetriever (BM25 + k-NN + Rerank)
    │               └─► compile_latex_to_pdf / create_docx_report
    │
    ├─► [HITL INTERRUPT] ◄── InMemorySaver checkpoint
    │       │
    │       └─► User: Approve / Reject / Edit
    │
    ├─► Resumes via /api/chat/resume (LangGraph Command)
    │
    └─► email_agent ──► send_email (SMTP)
            │
            └─► [HITL INTERRUPT] ──► User approval ──► Final Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangChain, LangGraph |
| LLM | DeepSeek (`deepseek/deepseek-chat` via OpenRouter) |
| Vector DB | Elasticsearch 8.x |
| Embeddings | `clip-ViT-B-32` (sentence-transformers) |
| Reranker | `ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| Keyword Search | BM25 (Elasticsearch) |
| PDF Ingestion | PyMuPDF (fitz) |
| PDF Generation | pdflatex (TeX Live / MiKTeX) |
| DOCX | python-docx, docx2txt |
| PPTX | python-pptx |
| Backend | FastAPI + Uvicorn |
| Email | smtplib (Gmail SMTP) |
| Checkpointing | LangGraph InMemorySaver |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Running Elasticsearch instance (Docker recommended)
- `pdflatex` installed (TeX Live or MiKTeX)
- Git

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd <your-repo-name>

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY="your-openrouter-api-key"
SMTP_API="your-gmail-app-password"
```

### 3. Verify Elasticsearch

Ensure Elasticsearch is running at `http://localhost:9200`:

```bash
curl http://localhost:9200  # Should return cluster info
```

### 4. Process Documents & Build Index

Run from the project root using Python's module flag:

```bash
# Process each source document
python -m data_processing.process docs/astronomy_openstax.pdf
python -m data_processing.process docs/astronomy_for_mere_mortals.pdf
python -m data_processing.process docs/introduction_to_astronomy_and_cosmology.pdf

# Build the Elasticsearch index (run once)
python -m rag.indexer
```

### 5. Launch the Application

```bash
uvicorn api.main2:app --reload --port 8000
```

Navigate to `http://localhost:8000`. Enter your email (used as `thread_id`) and preferred output format, then start chatting.

---

## Example Workflows

**Multi-step agentic task:**
> *"Create a detailed quiz on black holes in PDF format and email it to me."*

The system will:
1. Supervisor plans: `[generate quiz] → [email quiz]`
2. Content Generator retrieves relevant chunks + images via hybrid RAG
3. Formats content as LaTeX, compiles to PDF
4. **HITL pause** — user approves document generation
5. Email Agent prepares the email with the PDF attached
6. **HITL pause** — user approves email dispatch
7. Email sent; final confirmation returned

**Pure Q&A:**
> *"What is the evidence for dark matter in galaxy clusters?"*

Retrieves top-k text and image results, reranks via CrossEncoder, synthesizes a grounded response.

---

## Iterative Development & Lessons Learned

The final system reflects several cycles of failure analysis and prompt engineering:

- **Model selection**: Smaller open-source models failed at multi-step instruction following. DeepSeek (`deepseek-chat`) was chosen for consistent agentic reasoning and LaTeX generation quality.
- **Prompt structure**: Naive role-description prompts caused agents to drop the second half of chained tasks. Explicit numbered workflow prompts ("Plan → Delegate → Inspect & Chain") resolved this.
- **Memory injection**: Capturing user email/format at login and injecting as system context solved the "forgotten instruction" failure mode across long workflows.
- **Format-specific prompting**: DOCX content was reliably plain-text with a zero-shot instruction; LaTeX required a one-shot Beamer template in the prompt to enforce syntactic validity.

---

## Bonus Features Implemented

- [x] **CrossEncoder reranking** on hybrid retrieval results
- [x] **PIIMiddleware** — masks credit card numbers, API keys
- [x] **ContentFilterMiddleware** — keyword-based guardrail layer
- [x] **HumanInTheLoopMiddleware** — interruptible, resumable agentic workflows with user-editable tool arguments

---

## Future Work

- **Persistent checkpointing**: swap `InMemorySaver` for `AsyncPostgresSaver` to survive server restarts
- **PPTX generation**: add `create_pptx_report` tool using python-pptx
- **Advanced error recovery**: supervisor retries with alternate format on tool failure (e.g., PDF → DOCX fallback)
- **Streaming responses**: switch to SSE for real-time token streaming in the frontend

---

## Demo

[Watch the full demo](https://drive.google.com/file/d/1E4oGmzJFmuBC8XkyFSf1ZruJytPwInON/view?usp=sharing)

---
