from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_document(docs_dir):
    TextFiles = Path(docs_dir).rglob("*.txt")
    Files = []

    for file in TextFiles:
        text = file.read_text(encoding="utf-8")
        Files = Files.append((text,file.stem))
    print(Files)

load_document("src/rag/docs")

