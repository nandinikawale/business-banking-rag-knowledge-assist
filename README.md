# Business Banking RAG Knowledge Assistant

A submission-ready retrieval-augmented generation project for Business Banking
Relationship Managers. It retrieves from a versioned, entirely fictional policy and
product knowledge base and returns answers with inspectable citations.

> **Safety and data statement:** All documents, policies, thresholds, entities, and
> examples are fictional. The project contains no customer data, internal bank
> documents, confidential business rules, or real eligibility guidance.

## Business problem

Relationship Managers often need to locate product, documentation, onboarding, and
service guidance while preparing for customer conversations. This assistant reduces
search friction while keeping the evidence visible and preserving human accountability.
It does not approve credit, determine compliance outcomes, or initiate transactions.

## What the MVP demonstrates

- Markdown ingestion with required document metadata and versioning.
- Heading-aware chunking with citation metadata.
- Reproducible offline embeddings using feature hashing.
- Persistent vector retrieval in Chroma with category filters.
- Deterministic grounded synthesis that works without an API key.
- Source excerpts, relevance scores, and unsupported-question handling.
- FastAPI `/health` and `/query` endpoints.
- Streamlit user interface.
- Retrieval evaluation using Hit Rate@4 and Mean Reciprocal Rank.
- Automated tests, Ruff checks, and GitHub Actions.

## Architecture

```text
Fictional Markdown documents
          │
          ▼
metadata validation → heading-aware chunks → local embeddings → Chroma
                                                              │
question → optional category filter → top-k retrieval ─────────┘
                                      │
                                      ▼
                         grounded answer + citations
                            │                    │
                       Streamlit             FastAPI
```

The local hashing embedder is a deliberate submission-stage choice: it is deterministic,
has no model-download or API-key dependency, and keeps the live demo inexpensive. A
semantic embedding provider and optional LLM answer generator are planned enhancements.

## Run locally

Prerequisites: Python 3.13 and Internet access for the initial package installation.

```powershell
git clone https://github.com/nandinikawale/business-banking-rag-knowledge-assistant.git
cd business-banking-rag-knowledge-assistant
py -3.13 -m pip install uv==0.12.24
py -3.13 -m uv sync
py -3.13 -m uv run streamlit run streamlit_app/app.py
```

Build the vector index and evaluate retrieval:

```powershell
py -3.13 -m uv run banking-rag-index
py -3.13 -m uv run banking-rag-evaluate
```

### Baseline retrieval results

The submission benchmark contains ten banking questions with an expected source
document. With `top_k=4`, the current deterministic baseline achieves:

| Metric | Result |
| --- | ---: |
| Hit Rate@4 | 1.000 |
| Mean Reciprocal Rank | 0.825 |

This small benchmark verifies the pipeline and is not evidence of production accuracy.
Tomorrow's enhancement scope includes a larger question set, negative cases, semantic
embeddings, answer-groundedness scoring, and adversarial tests.

Run the API:

```powershell
py -3.13 -m uv run uvicorn banking_rag.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Test and quality checks

```powershell
py -3.13 -m uv run pytest
py -3.13 -m uv run ruff check .
py -3.13 -m uv run ruff format --check .
```

## Repository structure

```text
data/knowledge/       Fictional versioned knowledge documents
data/evaluation/      Retrieval benchmark questions
src/banking_rag/      Ingestion, embeddings, vector store, RAG, API, evaluation
streamlit_app/        Interactive cited-answer interface
tests/                Unit and retrieval integration tests
.github/workflows/    Continuous integration
```

## Relationship to the portfolio series

1. `business-banking-rm-intelligence-hub` — analytics and RM opportunities.
2. **This repository** — knowledge retrieval and grounded answers.
3. `business-banking-rm-copilot` — planned controlled agent orchestration combining
   structured analytics with policy retrieval and human approval.

## Current limitations

- The knowledge corpus is intentionally small and fictional.
- Hashing embeddings emphasize reproducibility over state-of-the-art semantics.
- Deterministic synthesis quotes and compresses retrieved evidence; it is not presented as
  generative reasoning.
- Authentication, document-level access control, prompt-injection evaluation, and a hosted
  API are future enhancements.

## License

MIT License. Copyright (c) 2026 nandinikawale.
