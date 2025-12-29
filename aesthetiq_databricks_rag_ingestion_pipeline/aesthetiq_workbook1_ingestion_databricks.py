# Databricks notebook source
# ============================================================
# AESTHETIQ Assistant™ — Workbook 1: Ingestion (Dummy‑Proof • Micro‑Cells)
# Powered by AESTHETIQ-RAG™
# Tagline: Where aesthetic intelligence meets clinical precision.
#
# Scope of THIS notebook (Workbook 1):
#   - Knowledge ingestion only:
#       PDFs -> page text -> token-based chunks -> Delta tables
#       + Document Registry (incremental ingestion)
#       + Failure Quarantine (observable errors)
#
# Scope of OTHER workbook (Workbook 2):
#   - Embeddings + Vector Search live in a different Databricks workbook.
#
# How to use:
#   - Run cells top to bottom.
#   - After each "STOP & CHECK" cell, confirm output matches expectations.
#   - Start in DRY_RUN mode (1 file) so you learn safely.
# ============================================================

# COMMAND ----------
# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# WHY:
# - pypdf: read PDF text
# - transformers + torch: tokenizer to measure chunk size in TOKENS (how LLMs read)
#
# Databricks NOTE:
# - After %pip you MUST restart Python in the next cell
# ============================================================
# MAGIC %pip install -U pypdf transformers torch

# COMMAND ----------
# ============================================================
# CELL 2 — Restart Python
# ============================================================
# WHY:
# Databricks needs a kernel restart to load newly installed packages.
# ============================================================
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# ============================================================
# CELL 3 — Imports (keep imports centralized)
# ============================================================
# WHY:
# - predictable notebook execution
# - easy debugging (one place to check missing libraries)
# ============================================================
import os
import re
import uuid
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timezone

import pypdf
from transformers import AutoTokenizer

from pyspark.sql import functions as F  # noqa: F401 (kept for future use)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
    LongType,
)

# COMMAND ----------
# ============================================================
# CELL 4 — Beginner Glossary (Read-Only)
# ============================================================
# Token: How the model counts input.
#   A “piece” of text the AI reads. Models have limits in tokens.
#
# Chunk: A small piece of a document, sized so the AI can read it.
# Delta table: reliable ACID table in Databricks
# Idempotent: You can run the pipeline again without making duplicates.
# Hash: A fingerprint. If content changes, the fingerprint changes.
# (CDF): Change Data Feed
#   Delta feature that records row-level changes, useful for incremental sync later.
# ============================================================

# COMMAND ----------
# ============================================================
# CELL 5 — Config (single source of truth)
# ============================================================
# WHY:
# - keeps all knobs in one place
# - safer refactors
# - easier to explain to non-technical stakeholders
# ============================================================
@dataclass(frozen=True)
class Config:
    # Unity Catalog tables
    chunks_table: str = "ragmicroneedling.default.aesthetiq_knowledge_chunks"
    documents_table: str = "ragmicroneedling.default.aesthetiq_documents"
    failures_table: str = "ragmicroneedling.default.aesthetiq_ingestion_failures"

    # Input PDFs
    pdf_dir_path: str = "/Volumes/ragmicroneedling/default/microneedling_documents"

    # Chunking
    tokenizer_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_tokens_per_chunk: int = 500
    separators: Tuple[str, ...] = ("\n\n", "\n", " ", "")

    # Learning-friendly knobs
    # During production you will make DRY_RUN False and MAX_FILES None
    DRY_RUN: bool = True
    MAX_FILES: int = 1

    # Pipeline behavior
    compute_content_hash: bool = False
    fail_fast: bool = False


cfg = Config()

# COMMAND ----------
# ============================================================
# CELL 6 — Create tables (DDL)
# ============================================================
# WHY:
# We create tables BEFORE ingestion so the pipeline can write reliably.
#
# Tables:
# 1) chunks_table:
#    - each row is a chunk of text used later for embedding and retrieval
# 2) documents_table:
#    - one row per document with status & metadata for incremental ingestion
# 3) failures_table:
#    - one row per error event for debugging & dashboards
# ============================================================
spark.sql(
    f'''
CREATE TABLE IF NOT EXISTS {cfg.chunks_table} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  doc_id STRING,
  source_file STRING,
  page_number INT,
  chunk_index INT,
  chunk_hash STRING,
  chunk_text STRING,
  ingestion_run_id STRING,
  processed_timestamp TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
'''
)

spark.sql(
    f'''
CREATE TABLE IF NOT EXISTS {cfg.documents_table} (
  doc_id STRING,
  source_file STRING,
  full_path STRING,
  file_size_bytes BIGINT,
  file_mtime_epoch BIGINT,
  file_hash_sha256 STRING,
  status STRING,                 -- success | failed | skipped
  pages_extracted INT,
  chunks_written INT,
  last_error_stage STRING,
  last_error_message STRING,
  last_ingestion_run_id STRING,
  last_ingested_timestamp TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
'''
)

spark.sql(
    f'''
CREATE TABLE IF NOT EXISTS {cfg.failures_table} (
  failure_id STRING,
  ingestion_run_id STRING,
  doc_id STRING,
  source_file STRING,
  full_path STRING,
  stage STRING,
  error_message STRING,
  error_timestamp TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
'''
)

print("✅ Tables created/verified.")

# COMMAND ----------
# ============================================================
# CELL 7 — STOP & CHECK: Confirm tables exist
# ============================================================
# EXPECTED:
# - DESCRIBE TABLE should succeed for all 3 tables
# - If it fails: check Unity Catalog permissions or table names
# ============================================================
spark.sql(f"DESCRIBE TABLE {cfg.chunks_table}").show(truncate=False)
spark.sql(f"DESCRIBE TABLE {cfg.documents_table}").show(truncate=False)
spark.sql(f"DESCRIBE TABLE {cfg.failures_table}").show(truncate=False)

# COMMAND ----------
# ============================================================
# CELL 8 — Load tokenizer (once)
# ============================================================
# WHY:
# - We chunk by TOKENS (how models count input), not characters.
# - Loading once is faster and consistent across documents.
#
# EXPECTED:
# - A confirmation print.
# - If it fails: the model name may require access; swap tokenizer_id temporarily.
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, cache_dir="/tmp/hf_cache")
print("Tokenizer loaded:", cfg.tokenizer_id)

# COMMAND ----------
# ============================================================
# CELL 9 — STOP & CHECK: tokenizer sanity
# ============================================================
# EXPECTED:
# - token_count should be a small positive integer
# - if token_count is 0, something is wrong with the tokenizer or text
# ============================================================
sample = "Botox dilution and reconstitution should follow manufacturer guidance."
token_count = len(tokenizer.encode(sample))
print("Sample token_count =", token_count)

# COMMAND ----------
# ============================================================
# CELL 10 — Utility: normalize text
# ============================================================
# WHY:
# - avoids tiny whitespace differences creating “new” hashes
# - improves stability across runs
# ============================================================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# COMMAND ----------
# ============================================================
# CELL 11 — Utility: hashing helpers
# ============================================================
# WHY:
# - doc_id provides a stable identifier for a document
# - chunk_hash provides a stable identifier for each chunk
# - file_hash_sha256 optionally fingerprints the entire PDF content (strongest)
# ============================================================
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_doc_id(source_file: str) -> str:
    # If you might have duplicate filenames, hash full_path instead.
    return sha256_hex(source_file.lower())


def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

# COMMAND ----------
# ============================================================
# CELL 12 — Utility: file metadata (size + modified time)
# ============================================================
# WHY:
# - size + mtime are fast signals to detect “no change”
# - we use these for incremental ingestion skip logic
# ============================================================
def get_file_metadata(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return int(st.st_size), int(st.st_mtime)

# COMMAND ----------
# ============================================================
# CELL 13 — TextProcessor: extract text page-by-page
# ============================================================
# WHY:
# - page-level traceability supports citations and debugging later
# - lets you store page_number with each chunk
# ============================================================
class TextProcessor:
    def __init__(self, cfg: Config, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            pages: List[Tuple[int, str]] = []
            for i, page in enumerate(reader.pages, start=1):
                txt = (page.extract_text() or "").strip()
                if txt:
                    pages.append((i, txt))
            return pages

# COMMAND ----------
# ============================================================
# CELL 14 — STOP & CHECK: confirm PDFs exist in your directory
# ============================================================
# EXPECTED:
# - You should see at least 1 PDF name printed.
# - If you see 0:
#     * confirm cfg.pdf_dir_path
#     * confirm the Volume is mounted and contains PDFs
# ============================================================
pdfs: List[str] = []
with os.scandir(cfg.pdf_dir_path) as entries:
    for e in entries:
        if e.is_file() and e.name.lower().endswith(".pdf"):
            pdfs.append(e.name)

print("PDF count:", len(pdfs))
print("First few PDFs:", pdfs[:5])

# COMMAND ----------
# ============================================================
# CELL 15 — Token length cache helper
# ============================================================
# WHY:
# tokenizer.encode() can be slow if called repeatedly.
# caching speeds up chunking dramatically.
# ============================================================
def token_len_cached(tokenizer: AutoTokenizer, cache: Dict[str, int], s: str) -> int:
    if s not in cache:
        cache[s] = len(tokenizer.encode(s))
    return cache[s]

# COMMAND ----------
# ============================================================
# CELL 16 — Hard token split (final failsafe)
# ============================================================
# WHY:
# If a single piece of text is too large even after splitting,
# we must guarantee it becomes <= max_tokens_per_chunk.
# ============================================================
def hard_token_split(tokenizer: AutoTokenizer, max_tokens: int, text: str) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks: List[str] = []
    start = 0

    while start < len(tokens):
        sub = tokens[start : start + max_tokens]
        chunks.append(normalize_text(tokenizer.decode(sub)))
        start += max_tokens

    return [c for c in chunks if c]

# COMMAND ----------
# ============================================================
# CELL 17 — Progressive chunking (meaning-preserving)
# ============================================================
# STRATEGY:
# Try splitting in this order:
#   1) paragraph breaks  ("\n\n")
#   2) line breaks       ("\n")
#   3) spaces            (" ")
#   4) character level   ("")  <- last resort
#
# WHY:
# We want chunks that keep ideas together. Paragraphs are best.
# ============================================================
def chunk_text(cfg: Config, tokenizer: AutoTokenizer, text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    cache: Dict[str, int] = {}
    chunks: List[str] = [text]

    for sep in cfg.separators:
        if all(token_len_cached(tokenizer, cache, c) <= cfg.max_tokens_per_chunk for c in chunks):
            break

        new_chunks: List[str] = []
        for c in chunks:
            if token_len_cached(tokenizer, cache, c) <= cfg.max_tokens_per_chunk:
                new_chunks.append(c)
                continue

            parts = c.split(sep) if sep else list(c)
            current = ""

            for part in parts:
                if sep and not part:
                    continue

                combined = (current + sep + part) if current else part

                if token_len_cached(tokenizer, cache, combined) <= cfg.max_tokens_per_chunk:
                    current = combined
                else:
                    if current:
                        new_chunks.append(normalize_text(current))

                    if token_len_cached(tokenizer, cache, part) > cfg.max_tokens_per_chunk:
                        new_chunks.extend(hard_token_split(tokenizer, cfg.max_tokens_per_chunk, part))
                        current = ""
                    else:
                        current = part

            if current:
                new_chunks.append(normalize_text(current))

        chunks = [normalize_text(x) for x in new_chunks if normalize_text(x)]

    out: List[str] = []
    for c in chunks:
        if token_len_cached(tokenizer, cache, c) > cfg.max_tokens_per_chunk:
            out.extend(hard_token_split(tokenizer, cfg.max_tokens_per_chunk, c))
        else:
            out.append(c)

    return [normalize_text(x) for x in out if normalize_text(x)]

# COMMAND ----------
# ============================================================
# CELL 18 — STOP & CHECK: chunking sanity test
# ============================================================
# EXPECTED:
# - The max_tokens_in_chunks should be <= cfg.max_tokens_per_chunk
# - The number of chunks should be >= 1
# ============================================================
test_text = " ".join(["Microneedling technique and depth vary by area."] * 400)
chunks = chunk_text(cfg, tokenizer, test_text)

max_tokens = max((len(tokenizer.encode(c)) for c in chunks), default=0)
print("chunks =", len(chunks), "max_tokens_in_chunks =", max_tokens, "limit =", cfg.max_tokens_per_chunk)

# COMMAND ----------
# ============================================================
# CELL 19 — Instantiate TextProcessor
# ============================================================
processor = TextProcessor(cfg, tokenizer)
print("■ TextProcessor ready.")

# COMMAND ----------
# ============================================================
# CELL 20 — Explicit schemas (prevents type drift)
# ============================================================
# IMPORTANT:
# - We do NOT insert `id` because Delta generates it via IDENTITY.
# ============================================================
CHUNKS_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("source_file", StringType(), False),
        StructField("page_number", IntegerType(), True),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_hash", StringType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("ingestion_run_id", StringType(), False),
        StructField("processed_timestamp", TimestampType(), False),
    ]
)

DOCS_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("source_file", StringType(), False),
        StructField("full_path", StringType(), False),
        StructField("file_size_bytes", LongType(), False),
        StructField("file_mtime_epoch", LongType(), False),
        StructField("file_hash_sha256", StringType(), True),
        StructField("status", StringType(), False),
        StructField("pages_extracted", IntegerType(), False),
        StructField("chunks_written", IntegerType(), False),
        StructField("last_error_stage", StringType(), True),
        StructField("last_error_message", StringType(), True),
        StructField("last_ingestion_run_id", StringType(), False),
        StructField("last_ingested_timestamp", TimestampType(), False),
    ]
)

FAIL_SCHEMA = StructType(
    [
        StructField("failure_id", StringType(), False),
        StructField("ingestion_run_id", StringType(), False),
        StructField("doc_id", StringType(), False),
        StructField("source_file", StringType(), False),
        StructField("full_path", StringType(), False),
        StructField("stage", StringType(), False),
        StructField("error_message", StringType(), False),
        StructField("error_timestamp", TimestampType(), False),
    ]
)

# COMMAND ----------
# ============================================================
# CELL 21 — Ingestor helper: should_skip (incremental ingestion)
# ============================================================
# WHY:
# - If the PDF has not changed, do not re-process it.
# - We treat "success + same size + same mtime" as unchanged.
# - Optionally, you can also compare file_hash for strongest guarantee.
# ============================================================
def should_skip(doc_id: str, size_bytes: int, mtime_epoch: int) -> bool:
    q = (
        spark.sql(
            f'''
      SELECT status, file_size_bytes, file_mtime_epoch
      FROM {cfg.documents_table}
      WHERE doc_id = "{doc_id}"
      ORDER BY last_ingested_timestamp DESC
      LIMIT 1
    '''
        )
        .collect()
    )

    if not q:
        return False

    r = q[0]
    return (
        r["status"] == "success"
        and int(r["file_size_bytes"]) == int(size_bytes)
        and int(r["file_mtime_epoch"]) == int(mtime_epoch)
    )

# COMMAND ----------
# ============================================================
# CELL 22 — Ingestor helper: log_failure (structured error logging)
# ============================================================
# WHY:
# - Print statements are not queryable.
# - A failure table lets you dashboard failures and debug systematically.
# ============================================================
def log_failure(
    ingestion_run_id: str,
    doc_id: str,
    source_file: str,
    full_path: str,
    stage: str,
    error: Exception,
) -> None:
    now_ts = datetime.now(timezone.utc)
    failure_id = str(uuid.uuid4())

    df = spark.createDataFrame(
        [
            (
                failure_id,
                ingestion_run_id,
                doc_id,
                source_file,
                full_path,
                stage,
                str(error)[:5000],
                now_ts,
            )
        ],
        schema=FAIL_SCHEMA,
    )

    df.write.format("delta").mode("append").saveAsTable(cfg.failures_table)

# COMMAND ----------
# ============================================================
# CELL 23 — Ingestor helper: upsert_document (document registry)
# ============================================================
# WHY:
# - Documents table is your ingestion “control panel”.
# - It answers: success/failed/skipped, counts, last error, last run.
# ============================================================
def upsert_document(doc_row: tuple) -> None:
    df = spark.createDataFrame([doc_row], schema=DOCS_SCHEMA)
    df.createOrReplaceTempView("staging_docs")

    spark.sql(
        f'''
      MERGE INTO {cfg.documents_table} AS t
      USING staging_docs AS s
      ON t.doc_id = s.doc_id
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    '''
    )

# COMMAND ----------
# ============================================================
# CELL 24 — Ingestor helper: merge_chunks (idempotent write)
# ============================================================
# WHY:
# - chunk_hash is the stable key
# - MERGE prevents duplicates on re-run
# ============================================================
def merge_chunks(rows: List[tuple]) -> int:
    if not rows:
        return 0

    df = spark.createDataFrame(rows, schema=CHUNKS_SCHEMA)
    df.createOrReplaceTempView("staging_chunks")

    spark.sql(
        f'''
      MERGE INTO {cfg.chunks_table} AS t
      USING staging_chunks AS s
      ON t.chunk_hash = s.chunk_hash
      WHEN NOT MATCHED THEN
        INSERT (
          doc_id, source_file, page_number, chunk_index,
          chunk_hash, chunk_text, ingestion_run_id, processed_timestamp
        )
        VALUES (
          s.doc_id, s.source_file, s.page_number, s.chunk_index,
          s.chunk_hash, s.chunk_text, s.ingestion_run_id, s.processed_timestamp
        )
    '''
    )

    # Count of the staging rows (not necessarily newly inserted, but consistent with original workbook)
    return df.count()

# COMMAND ----------
# ============================================================
# CELL 25 — Build chunk rows for one PDF (pure transformation)
# ============================================================
# WHY:
# - Separate “data transformation” from “writing to tables”
# - Easier to test, reason about, and debug
# ============================================================
def build_chunk_rows(
    doc_id: str,
    source_file: str,
    full_path: str,
    ingestion_run_id: str,
) -> Tuple[List[tuple], int]:
    now_ts = datetime.now(timezone.utc)
    pages = processor.extract_pages(full_path)

    if not pages:
        raise RuntimeError("No pages extracted (empty/unreadable PDF).")

    rows: List[tuple] = []
    for page_num, page_text in pages:
        chunks = chunk_text(cfg, tokenizer, page_text)
        for idx, chunk in enumerate(chunks):
            chash = sha256_hex(f"{doc_id}|{page_num}|{idx}|{normalize_text(chunk)}")
            rows.append((doc_id, source_file, int(page_num), int(idx), chash, chunk, ingestion_run_id, now_ts))

    return rows, len(pages)

# COMMAND ----------
# ============================================================
# CELL 26 — Process one PDF (end-to-end)
# ============================================================
# WHAT this does:
# 1) collect metadata (size, mtime, optional file hash)
# 2) skip if unchanged
# 3) extract pages + chunk
# 4) MERGE chunks (idempotent)
# 5) upsert documents registry (success/skipped/failed)
# 6) log failures to failures_table
# ============================================================
def process_one_pdf(entry, ingestion_run_id: str) -> int:
    source_file = entry.name
    full_path = os.path.join(cfg.pdf_dir_path, entry.name)

    doc_id = make_doc_id(source_file)
    size_bytes, mtime_epoch = get_file_metadata(full_path)

    # Fast skip
    if should_skip(doc_id, size_bytes, mtime_epoch):
        doc_row = (
            doc_id,
            source_file,
            full_path,
            size_bytes,
            mtime_epoch,
            None,  # file_hash_sha256
            "skipped",
            0,  # pages_extracted
            0,  # chunks_written
            None,  # last_error_stage
            None,  # last_error_message
            ingestion_run_id,
            datetime.now(timezone.utc),
        )
        upsert_document(doc_row)
        print(f"SKIP unchanged: {source_file}")
        return 0

    # Optional strongest detection (slower)
    file_hash = sha256_file(full_path) if cfg.compute_content_hash else None

    try:
        rows, pages_extracted = build_chunk_rows(doc_id, source_file, full_path, ingestion_run_id)
        n_chunks = merge_chunks(rows)

        doc_row = (
            doc_id,
            source_file,
            full_path,
            size_bytes,
            mtime_epoch,
            file_hash,
            "success",
            int(pages_extracted),
            int(n_chunks),
            None,
            None,
            ingestion_run_id,
            datetime.now(timezone.utc),
        )
        upsert_document(doc_row)

        print(f"OK: {source_file} pages={pages_extracted} chunks={n_chunks}")
        return n_chunks

    except Exception as e:
        log_failure(ingestion_run_id, doc_id, source_file, full_path, "process_one_pdf", e)

        doc_row = (
            doc_id,
            source_file,
            full_path,
            size_bytes,
            mtime_epoch,
            file_hash,
            "failed",
            0,
            0,
            "process_one_pdf",
            str(e)[:5000],
            ingestion_run_id,
            datetime.now(timezone.utc),
        )
        upsert_document(doc_row)

        print(f"FAIL: {source_file} -> {e}")
        if cfg.fail_fast:
            raise
        return 0

# COMMAND ----------
# ============================================================
# CELL 27 — Build the PDF list (respect DRY_RUN)
# ============================================================
# WHY:
# - DRY_RUN makes learning safer (process only 1 file)
# - You can later set DRY_RUN=False to process everything
# ============================================================
with os.scandir(cfg.pdf_dir_path) as entries:
    pdf_entries = [e for e in entries if e.is_file() and e.name.lower().endswith(".pdf")]

pdf_entries = sorted(pdf_entries, key=lambda e: e.name.lower())

if cfg.DRY_RUN:
    pdf_entries = pdf_entries[: cfg.MAX_FILES]
    print(f"DRY_RUN=True -> processing first {len(pdf_entries)} file(s).")
else:
    print(f"DRY_RUN=False -> processing {len(pdf_entries)} files.")

# COMMAND ----------
# ============================================================
# CELL 28 — STOP & CHECK: show which PDFs will be processed
# ============================================================
# EXPECTED:
# - In DRY_RUN mode, you should see 1 file.
# - If list is empty: confirm cfg.pdf_dir_path and file extensions.
# ============================================================
print("Files to process:")
for e in pdf_entries:
    print(" -", e.name)

# COMMAND ----------
# ============================================================
# CELL 29 — Run ingestion (the batch loop)
# ============================================================
# WHY:
# - ingestion_run_id ties this entire run together for auditability
# - we can later query exactly what happened in this run
# ============================================================
ingestion_run_id = str(uuid.uuid4())
print("Ingestion run id:", ingestion_run_id)

total_chunks = 0
start = time.time()

for e in pdf_entries:
    total_chunks += process_one_pdf(e, ingestion_run_id)

elapsed_ms = int((time.time() - start) * 1000)
print(f"■ DONE. chunks_inserted={total_chunks} elapsed_ms={elapsed_ms} run_id={ingestion_run_id}")

# COMMAND ----------
# ============================================================
# CELL 30 — STOP & CHECK: verify chunks landed
# ============================================================
# EXPECTED:
# - total_chunks should be >= 1 after a successful run
# - If 0:
#     * PDF might be scanned images (no extractable text)
#     * or extraction returned empty
# ============================================================
spark.sql(f"SELECT count(*) AS total_chunks FROM {cfg.chunks_table}").show()

# COMMAND ----------
# ============================================================
# CELL 31 — STOP & CHECK: sample chunk preview
# ============================================================
# EXPECTED:
# - You should see chunk_text previews.
# - If chunk_text is empty: PDF extraction is failing (likely image-based PDF).
# ============================================================
spark.sql(
    f'''
  SELECT
    source_file,
    page_number,
    chunk_index,
    substring(chunk_text, 1, 180) AS preview
  FROM {cfg.chunks_table}
  ORDER BY processed_timestamp DESC
  LIMIT 10
'''
).show(truncate=False)

# COMMAND ----------
# ============================================================
# CELL 32 — STOP & CHECK: documents registry status
# ============================================================
# EXPECTED:
# - Each processed file should have a row with status:
#     success OR skipped OR failed
# ============================================================
spark.sql(
    f'''
  SELECT status, count(*) AS n
  FROM {cfg.documents_table}
  GROUP BY status
  ORDER BY n DESC
'''
).show()

# COMMAND ----------
# ============================================================
# CELL 33 — STOP & CHECK: recent documents
# ============================================================
# EXPECTED:
# - You should see the file name, status, and counts.
# ============================================================
spark.sql(
    f'''
  SELECT
    source_file,
    status,
    pages_extracted,
    chunks_written,
    last_ingested_timestamp
  FROM {cfg.documents_table}
  ORDER BY last_ingested_timestamp DESC
  LIMIT 20
'''
).show(truncate=False)

# COMMAND ----------
# ============================================================
# CELL 34 — STOP & CHECK: failures table (should be empty usually)
# ============================================================
# EXPECTED:
# - Ideally 0 rows.
# - If rows exist, that is OK: it tells you exactly what broke and where.
# ============================================================
spark.sql(f"SELECT count(*) AS failure_events FROM {cfg.failures_table}").show()

spark.sql(
    f'''
  SELECT
    error_timestamp,
    source_file,
    stage,
    substring(error_message, 1, 220) AS msg
  FROM {cfg.failures_table}
  ORDER BY error_timestamp DESC
  LIMIT 10
'''
).show(truncate=False)

# COMMAND ----------
# ============================================================
# CELL 35 — Re-run test (idempotency)
# ============================================================
# WHY:
# - A “perfect” ingestion pipeline can be re-run safely.
# - If your PDFs have not changed, status should become "skipped"
#   and total_chunks should NOT explode with duplicates.
#
# Do this:
# 1) Re-run CELL 29 (Run ingestion)
# 2) Then run this cell again.
#
# EXPECTED:
# - duplicates should be 0 (by chunk_hash)
# ============================================================
spark.sql(
    f'''
  SELECT count(*) - count(DISTINCT chunk_hash) AS duplicate_chunk_hash_count
  FROM {cfg.chunks_table}
'''
).show()

# COMMAND ----------
# ============================================================
# CELL 36 — Optional: strongest change detection (content hash)
# ============================================================
# If you want the strongest “unchanged detection”, set:
#   cfg.compute_content_hash = True
# Then rerun ingestion.
#
# Tradeoff:
# - more CPU / I/O (reads full file bytes)
# - but detects changes even if mtime/size are unreliable
# ============================================================

# COMMAND ----------
# ============================================================
# CELL 37 — Optional: OPTIMIZE / ZORDER
# ============================================================
# WHY:
# - MERGE can create small files.
# - OPTIMIZE compacts files; ZORDER helps performance for common filters.
#
# Use after big ingestion batches (not necessarily every run).
# ============================================================
# spark.sql(f"OPTIMIZE {cfg.chunks_table} ZORDER BY (doc_id, source_file)")
# spark.sql(f"OPTIMIZE {cfg.documents_table} ZORDER BY (doc_id)")
# print("■ OPTIMIZE complete.")

# COMMAND ----------
# ============================================================
# CELL 38 — End of Workbook 1 (Read-Only)
# ============================================================
# Congratulations — you have built the knowledge foundation for AESTHETIQ Assistant™.
#
# NEXT workbook (Workbook 2):
# - Create embeddings for chunk_text
# - Build / sync Databricks Vector Search index (using CDF)
#
# This separation is intentional and professional:
# - Ingestion is batch + I/O heavy
# - Vector search is ML infra + indexing heavy
# ============================================================
