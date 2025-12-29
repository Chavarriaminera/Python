# Python Projects

This repository contains selected Python projects focused on applied
data science, natural language processing (NLP), exploratory data
analysis (EDA), and automation. Each project emphasizes clear analytical
workflows, thoughtful visualization, and real-world data challenges.

## Portfolio Highlights
- **Data Science / NLP:** sentiment analysis, imbalanced distributions, interpretable language signals
- **Data Engineering / RAG:** Databricks + Delta Lake ingestion pipeline (PDF → chunks → Delta tables) built for embeddings + vector search
- **Focus:** reproducible workflows, clarity, and production-minded reliability patterns

## Featured Projects

### Bitcoin Sentiment Analysis

Exploratory analysis of sentiment dynamics in Bitcoin-related tweets
using VADER compound sentiment scores. This project focuses on
understanding highly imbalanced sentiment distributions and surfacing
rare but extreme emotional responses through log-scaled visualization.

**Key concepts demonstrated:**
- Natural language processing (NLP)
- Sentiment analysis with VADER
- Exploratory data analysis (EDA)
- Log-scaled visualization for skewed distributions
- Analytical interpretation of social media data

📂 **Project directory:**  
[`bitcoin_sentiment_analysis/`](bitcoin_sentiment_analysis/)

---

### AESTHETIQ Assistant — Databricks RAG Ingestion Pipeline

A production-style document ingestion pipeline built in Databricks that
converts raw documents (e.g., PDFs) into RAG-ready structured data.
The workflow extracts text, chunks content, and writes normalized Delta
tables with incremental processing, hashing, and failure logging — designed
to support downstream embeddings + vector search for LLM retrieval.

**Key concepts demonstrated:**
- Data engineering pipeline design (ETL/ELT)
- Databricks + PySpark workflows
- Delta Lake table modeling (`documents` / `chunks` / `failures`)
- Incremental ingestion + deduplication patterns (hashing)
- Observability and error logging for reliability
- RAG preprocessing (chunking for embeddings + retrieval)

📂 **Project directory:**  
[`aesthetiq_databricks_rag_ingestion_pipeline/`](aesthetiq_databricks_rag_ingestion_pipeline/)

> Optional local demo (no Databricks): add `local_dry_run.py` in the project folder and run it locally.

## Why This Matters

Sentiment data is often dominated by neutral expressions, which can
mask meaningful but infrequent emotional extremes. This project shows
how careful exploratory analysis and visualization choices can uncover
signals that are relevant for downstream modeling, risk analysis, and
event-driven insights.

Separately, RAG systems are only as reliable as their ingestion layer.
A consistent chunking strategy, incremental processing, and failure
logging are essential for building trustworthy corpora for embeddings,
vector indexing, and retrieval quality.

## Other Work

- **Pokémon Team Builder**  
  AI-assisted experimentation project exploring decision logic and
  recommendation-style workflows.

- **Python → R Automation Scripts**  
  Utility scripts demonstrating cross-language automation and workflow
  orchestration.

## Repository Structure

