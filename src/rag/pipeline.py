from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

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
