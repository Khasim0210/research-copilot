"""
Vector store: build FAISS index from chunks, persist to disk, and retrieve.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"


def build_vectorstore(chunks, embeddings, persist_dir: Path = VECTORSTORE_DIR):
    """Embed all chunks and save FAISS index to disk."""
    print(f"Building FAISS index from {len(chunks)} chunks...")
    vs = FAISS.from_documents(chunks, embeddings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(persist_dir))
    print(f"Saved vector store to {persist_dir}")
    return vs


def load_vectorstore(embeddings, persist_dir: Path = VECTORSTORE_DIR):
    """Load FAISS index from disk."""
    print(f"Loading FAISS index from {persist_dir}...")
    vs = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # OK: we created this file ourselves
    )
    return vs


def search(vectorstore, query: str, k: int = 5):
    """Retrieve top-k most similar chunks for a query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    # Build the index, then test retrieval
    from ingest import load_documents, chunk_documents, get_embeddings

    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    vs = build_vectorstore(chunks, embeddings)

    # Test retrieval
    query = "What is self-attention?"
    print(f"\n--- Retrieval test ---")
    print(f"Query: {query}\n")
    results = search(vs, query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"[Result {i}] Score: {score:.4f}")
        print(f"Source page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...\n")