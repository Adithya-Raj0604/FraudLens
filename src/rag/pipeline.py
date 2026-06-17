from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def save_index(index, chunks, out_dir):
    """Persist a built FAISS index + its chunks to disk (run at Docker build time)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index ({index.ntotal} vectors, {len(chunks)} chunks) to {out}")


def load_index(out_dir):
    """Load a pre-built FAISS index + chunks from disk (avoids re-encoding at startup)."""
    out = Path(out_dir)
    index = faiss.read_index(str(out / "faiss.index"))
    with open(out / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def load_documents(docs_dir):
    TextFiles = Path(docs_dir).rglob("*.txt")
    Files = []

    for file in TextFiles:
        text = file.read_text(encoding="utf-8")
        Files.append((text, file.stem))

    return Files

def chunk_documents(docs):
    chunks = []

    for text, source in docs:
        text = text.replace("\r\n", "\n")
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(paragraph.strip()) >= 50:
                chunks.append((paragraph.strip(), source))

    return chunks

def build_index(docs_dir):
    docs   = load_documents(docs_dir)
    chunks = chunk_documents(docs)

    if not chunks:
        print(f"Warning: no documents found in {docs_dir}. RAG tool will return empty results.")
        # Return a minimal valid index so the agent can still start
        dim   = 384  # all-MiniLM-L6-v2 output dimension
        index = faiss.IndexFlatL2(dim)
        return index, []

    texts      = [chunk for chunk, source in chunks]
    embeddings = _model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(f"Index built: {index.ntotal} vectors from {len(chunks)} chunks")
    return index, chunks

def query_index(index, chunks, query, top_k=3):
    if not chunks or index.ntotal == 0:
        return ["No regulatory documents loaded. Add .txt files to src/rag/docs/ to enable RAG."]

    q_vec = _model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for idx in indices[0]:
        chunk_text, source = chunks[idx]
        results.append(chunk_text)

    return results

if __name__ == "__main__":
    index, chunks = build_index("src/rag/docs")

    test_queries = [
        "suspicious transaction reporting deadline",
        "account balance drained to zero fraud",
        "velocity more than 5 transactions escalate",
    ]

    for query in test_queries:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
        results = query_index(index, chunks, query)
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} ---\n{r[:300]}")
