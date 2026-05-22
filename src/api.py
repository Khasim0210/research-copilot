"""
FastAPI wrapper for the RAG copilot.
Endpoints:
  GET  /health  - service health check
  POST /chat    - ask a question (with optional history)
  POST /ingest  - upload a PDF and rebuild the vector store
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.ingest import load_documents, chunk_documents, get_embeddings, DATA_DIR
from src.vectorstore import build_vectorstore, load_vectorstore, VECTORSTORE_DIR
from src.memory import build_conversational_chain

app = FastAPI(
    title="Research Copilot API",
    description="RAG-powered document Q&A using LangChain + FAISS + Qwen2.5",
    version="0.1.0",
)

# Lazy globals (initialized on startup)
embeddings = None
vectorstore = None
chain = None


@app.on_event("startup")
def startup():
    """Load embeddings + vector store + chain once when the server starts."""
    global embeddings, vectorstore, chain
    embeddings = get_embeddings()
    if VECTORSTORE_DIR.exists():
        vectorstore = load_vectorstore(embeddings, VECTORSTORE_DIR)
        chain = build_conversational_chain(vectorstore)
        print("✅ RAG chain ready.")
    else:
        print("⚠️  No vector store found. POST a PDF to /ingest to build one.")


# ---------- Pydantic schemas ----------

class HistoryTurn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = []


class ChatResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources_used: str


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store_loaded": vectorstore is not None,
        "chain_ready": chain is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if chain is None:
        raise HTTPException(503, "RAG chain not ready. Ingest a PDF first via /ingest.")

    # Convert history to LangChain message objects
    history = []
    for turn in req.history:
        if turn.role == "user":
            history.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            history.append(AIMessage(content=turn.content))

    result = chain.invoke({"question": req.question, "history": history})
    return ChatResponse(
        question=req.question,
        standalone_question=result["standalone"],
        answer=result["answer"],
        sources_used=result["context"][:500] + "...",
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Save uploaded file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild index from ALL documents in DATA_DIR
    global vectorstore, chain
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, embeddings)
    chain = build_conversational_chain(vectorstore)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "status": "Vector store rebuilt and chain refreshed.",
    }