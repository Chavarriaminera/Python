# AESTHETIQ Assistant — Databricks RAG Ingestion Pipeline

A production-style **document ingestion pipeline** built in **Databricks** that converts raw documents (e.g., PDFs) into **RAG-ready** structured data.  
It extracts text, chunks content, and writes normalized **Delta tables** with incremental processing, hashing, and failure logging — designed for downstream **embeddings + vector search**.


## Recruiter Quick Look
- **What this demonstrates:** production-ready pipeline design, incremental ingestion patterns, Delta modeling, reliability/observability, and RAG preprocessing.
- **Tech:** Databricks, PySpark, Delta Lake, Python
- **Outputs:** curated Delta tables (`documents`, `chunks`, `failures`) ready for embedding generation and vector indexing.

---

## Problem
RAG systems are only as good as the ingestion layer. Without consistent chunking, deduplication, and failure handling, retrieval quality and system reliability degrade quickly.

This project focuses on creating a robust ingestion foundation:
- predictable chunk boundaries
- incremental processing (avoid duplicates / reprocessing)
- traceability + debugging via failure logging

---

## What the Pipeline Does
1. **Discovers documents** in the configured storage location (incremental ingestion supported)
2. **Extracts text** (page-aware when available)
3. **Chunks text** into retrieval-friendly segments
4. **Writes Delta tables** for analytics + downstream embedding jobs
5. **Logs failures** with enough metadata to debug and monitor runs

---

## Data Model (Delta Tables)

### 1) `documents`
One row per document.
- Tracks ingestion status, metadata, and timestamps
- Enables incremental loads and reprocessing logic

### 2) `chunks`
One row per chunk (RAG-ready unit of retrieval).
- Includes identifiers (doc_id, page_number, chunk_index)
- Stores chunk text and hashing for deduplication / traceability

### 3) `failures`
One row per error event.
- Captures error context for debugging and dashboards
- Helps monitor pipeline reliability over time

---

## Repository Contents
- `aesthetiq_workbook1_ingestion_databricks.py`  
  Databricks notebook source (`# COMMAND ----------`) with “micro-cells” for step-by-step execution and validation.
- `requirements.txt`  
  Minimal dependencies (may vary by cluster/runtime).
- `README.md`  
  Project documentation.

---

## How to Run (Databricks)
1. Import or upload `aesthetiq_workbook1_ingestion_databricks.py` into Databricks (Workspace or Repos)
2. Attach the notebook to a cluster with access to your storage path
3. Run cells top-to-bottom

**Design note:** The notebook is intentionally broken into small “micro-cells” to make each step easy to run, verify, and debug.

---

## Configuration
Update these values in the notebook to match your environment:
- storage input path (where PDFs/docs live)
- catalog/schema/table names for Delta outputs
- any job/run identifiers used for tracking

---

## Why This Matters for RAG
This ingestion pipeline produces a clean and reliable corpus for:
- **embedding generation**
- **vector indexing**
- **LLM retrieval (RAG)**

The result is higher-quality retrieval, fewer duplicates, and easier debugging.

---

## Next Steps (Planned Enhancements)
- Add embedding generation (batch job + incremental updates)
- Add vector index build/refresh logic
- Add retrieval evaluation (gold set + metrics)
- Add dashboards/alerts for failures and ingestion SLAs
- Add CI checks (linting + basic unit tests for chunking utilities)

---

## License
Choose a license that fits your goals (MIT is common for portfolio projects).
