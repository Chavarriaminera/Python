import argparse
import hashlib
from pathlib import Path


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = end - overlap

    return chunks


def main():
    parser = argparse.ArgumentParser(description="AESTHETIQ ingestion dry-run (local chunking demo).")
    parser.add_argument("--input", required=True, help="Path to a .txt file to chunk")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    path = Path(args.input)
    text = path.read_text(encoding="utf-8", errors="ignore")

    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

    print(f"\nInput: {path.name}")
    print(f"Characters: {len(text):,}")
    print(f"Chunks: {len(chunks):,}\n")

    for i, c in enumerate(chunks[:2]):
        print(f"--- Chunk {i} (len={len(c)}) hash={sha256(c)[:12]} ---")
        print(c[:500] + ("..." if len(c) > 500 else ""))
        print()

    print("Sample output rows (illustrative):")
    for i, c in enumerate(chunks[:3]):
        row = {
            "doc_id": path.stem,
            "source_file": path.name,
            "page_number": None,
            "chunk_index": i,
            "chunk_hash": sha256(c),
            "chunk_text_preview": c[:120] + ("..." if len(c) > 120 else "")
        }
        print(row)


if __name__ == "__main__":
    main()
