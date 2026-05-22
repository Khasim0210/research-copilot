"""
Document ingestion pipeline:
1. Load PDFs from data/documents/
2. Split into chunks (~500 tokens, 50-token overlap)
3. Generate embeddings using sentence-transformers
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = Path(__file__).parent.parent / "data" / "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: Path = DATA_DIR):
    """Load all PDFs from the data directory."""
    docs = []
    for pdf_path in data_dir.glob("*.pdf"):
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages total.")
    return docs


def chunk_documents(docs, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def _pick_device() -> str:
    """Pick the best available device: mps (M-series Mac) > cuda > cpu."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Load the embedding model on the best available device."""
    device = _pick_device()
    print(f"Loading embedding model: {model_name} (device={device})")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def run_pipeline():
    """End-to-end test of ingestion."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    # Quick sanity test: embed the first chunk
    sample_vector = embeddings.embed_query(chunks[0].page_content)
    print(f"\n--- Sanity check ---")
    print(f"First chunk preview: {chunks[0].page_content[:150]}...")
    print(f"Embedding vector dimension: {len(sample_vector)}")
    print(f"First 5 values: {sample_vector[:5]}")
    return chunks, embeddings


if __name__ == "__main__":
    run_pipeline()