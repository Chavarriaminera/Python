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
> If you ever see a Mermaid error, it usually means the code fence wasn’t closed.  
> This block starts with ```mermaid and ends with ``` on its own line.

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
