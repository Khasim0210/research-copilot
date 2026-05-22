"""
RAG chain: retriever + LLM with a grounded prompt.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from ingest import get_embeddings
from vectorstore import load_vectorstore, VECTORSTORE_DIR

# Load environment variables from .env
load_dotenv()

LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"


def get_llm():
    """Initialize the LLM via Hugging Face Inference API."""
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.3,         # Lower = more grounded, less creative
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
    return ChatHuggingFace(llm=endpoint)


def format_docs(docs):
    """Concatenate retrieved chunks into one context string with sources."""
    return "\n\n".join(
        f"[Source: page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(vectorstore, k: int = 5):
    """Compose retrieval + prompt + LLM into a single chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Answer the user's question using ONLY "
         "the provided context. If the answer is not in the context, say "
         "'I cannot answer based on the provided documents.' "
         "Cite the source page number for each claim."),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    embeddings = get_embeddings()
    vs = load_vectorstore(embeddings, VECTORSTORE_DIR)
    chain, retriever = build_rag_chain(vs)

    questions = [
        "What is the Transformer architecture?",
        "Explain self-attention in simple terms.",
        "What datasets were used for training?",
    ]

    for q in questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        answer = chain.invoke(q)
        print(f"A: {answer}")