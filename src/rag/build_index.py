"""
Pre-build the FAISS index and persist it to disk.

Run at Docker build time so the Lambda container ships with a ready index and
never re-encodes the regulatory docs on a cold start:

    python -m src.rag.build_index
"""
from pathlib import Path

from src.rag.pipeline import build_index, save_index

DOCS_DIR  = Path(__file__).resolve().parent / "docs"
INDEX_DIR = Path(__file__).resolve().parent / "index"

if __name__ == "__main__":
    index, chunks = build_index(str(DOCS_DIR))
    save_index(index, chunks, str(INDEX_DIR))
