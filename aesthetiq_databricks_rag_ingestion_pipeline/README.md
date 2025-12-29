# AESTHETIQ Assistant — Databricks RAG Ingestion Pipeline

A production-style **document ingestion pipeline** built in **Databricks** that converts raw documents (e.g., PDFs) into **RAG-ready** structured data.  
It extracts text, chunks content, and writes normalized **Delta tables** with incremental processing, hashing, and failure logging — designed for downstream **embeddings + vector search**.

---

## Recruiter Quick Look
- **What this demonstrates:** production-ready pipeline design, incremental ingestion patterns, Delta modeling, reliability/observability, and RAG preprocessing.
- **Tech:** Databricks, PySpark, Delta Lake, Python
- **Outputs:** curated Delta tables (`documents`, `chunks`, `failures`) ready for embedding generation and vector indexing.

---

## Architecture

### Visual (GitHub Mermaid)
```mermaid
flowchart TD
  A[Source Docs - PDFs] --> B[Text Extraction]
  B --> C[Chunking - size and overlap]
  C --> D[Delta - documents]
  C --> E[Delta - chunks]
  C --> F[Delta - failures]
  E --> G[Next - Embeddings]
  G --> H[Next - Vector Index]
  H --> I[Next - Retrieval Evaluation]
 
---

## Project Status / Progress

### ✅ Completed (Current)
- Ingestion foundation in Databricks (PySpark)
- Text extraction (page-aware when available)
- Chunking into retrieval-friendly segments (size + overlap)
- Delta Lake table outputs:
  - documents (status + metadata for incremental ingestion)
  - chunks (RAG-ready text units + hashes)
  - failures (error events for debugging/monitoring)
- Reliability patterns: incremental processing + hashing + failure logging
- Architecture documented (Mermaid diagram)

### 🟡 In Progress
- Add `requirements.txt` to repo (if not already present)
- Optional: add `local_dry_run.py` + `sample_docs/` for a quick non-Databricks demo
- Add a small “Example Output” section (docs processed, chunks written, failures logged)

### 🔜 Next to Complete the RAG Pipeline (Roadmap)

**Step 1 — Embeddings**
- Add an embeddings job (batch) that reads from `chunks`
- Write embeddings to a Delta table (e.g., `chunk_embeddings`)
- Track embedding run metadata (timestamp, model name, version)

**Step 2 — Vector Index**
- Build/refresh a vector index from `chunk_embeddings`
- Store index metadata (index name, last refresh, row counts)

**Step 3 — Retrieval + Evaluation**
- Create a small evaluation set (20–50 queries)
- Measure retrieval quality (e.g., recall@k, MRR) before/after tuning chunking
- Add simple reporting (notebook output or markdown summary)

**Step 4 — Operationalization**
- Turn ingestion + embedding + indexing into scheduled jobs
- Add monitoring: failure rate, docs processed, chunks written, latency
- Add alerting rules (optional)

### 🎯 Definition of “Done”
This project is complete when:
- ingestion produces clean Delta tables reliably
- embeddings are generated and stored incrementally
- a vector index is built/refreshed from embeddings
- retrieval quality is measured with basic evaluation metrics
- jobs are runnable end-to-end with minimal manual steps
