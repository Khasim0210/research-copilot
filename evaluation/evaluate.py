"""
Evaluate the RAG copilot on a small held-out Q&A set.

Metrics:
  - Retrieval Precision@5  -> fraction of retrieved chunks containing relevant keywords
  - Answer Relevancy        -> cosine similarity between answer embedding and ground_truth_answer embedding
  - Hallucination Rate      -> fraction of generated sentences whose content is NOT supported by retrieved context

Usage:
    python -m evaluation.evaluate
"""

import json
from pathlib import Path
import numpy as np

from src.ingest import get_embeddings
from src.vectorstore import load_vectorstore, VECTORSTORE_DIR
from src.rag_chain import build_rag_chain

EVAL_FILE = Path(__file__).parent / "eval_dataset.json"


def precision_at_k(retrieved_docs, relevant_keywords, k=5):
    """A chunk counts as relevant if it contains >=1 of the relevant keywords (case-insensitive)."""
    if not retrieved_docs:
        return 0.0
    hits = 0
    for doc in retrieved_docs[:k]:
        text = doc.page_content.lower()
        if any(kw.lower() in text for kw in relevant_keywords):
            hits += 1
    return hits / min(k, len(retrieved_docs))


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def answer_relevancy(answer, ground_truth, embeddings):
    """Semantic similarity between the model answer and the gold answer."""
    a_emb = embeddings.embed_query(answer)
    g_emb = embeddings.embed_query(ground_truth)
    return cosine_sim(a_emb, g_emb)


def hallucination_rate(answer, retrieved_docs, embeddings, threshold=0.55):
    """
    Hallucination = generated sentence not supported by ANY retrieved chunk.
    We skip very short and math-heavy sentences (they distort the metric)
    but keep the semantic-similarity check strict.
    """
    import re

    raw = re.split(r"(?<=[.!?])\s+", answer.replace("\n", " "))
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 30:                                       # too short to evaluate
            continue
        if sum(c.isalpha() for c in s) < len(s) * 0.5:        # mostly symbols/math
            continue
        sentences.append(s)

    if not sentences:
        return 0.0

    doc_embs = [embeddings.embed_query(d.page_content) for d in retrieved_docs]
    ungrounded = 0
    for sent in sentences:
        s_emb = embeddings.embed_query(sent)
        max_sim = max((cosine_sim(s_emb, d_emb) for d_emb in doc_embs), default=0)
        if max_sim < threshold:
            ungrounded += 1
    return ungrounded / len(sentences)


def run_eval():
    print("Loading embeddings + vector store...")
    embeddings = get_embeddings()
    vs = load_vectorstore(embeddings, VECTORSTORE_DIR)
    chain, retriever = build_rag_chain(vs, k=5)

    with EVAL_FILE.open() as f:
        eval_set = json.load(f)

    precisions, relevancies, hallucinations = [], [], []

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        print(f"\n[{i}/{len(eval_set)}] Q: {q}")

        retrieved = retriever.invoke(q)
        answer = chain.invoke(q)

        p = precision_at_k(retrieved, item["relevant_keywords"], k=5)
        r = answer_relevancy(answer, item["ground_truth_answer"], embeddings)
        h = hallucination_rate(answer, retrieved, embeddings)

        precisions.append(p)
        relevancies.append(r)
        hallucinations.append(h)

        print(f"   Precision@5: {p:.2f}   Relevancy: {r:.2f}   Hallucination: {h:.2f}")

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Retrieval Precision@5      : {np.mean(precisions):.2%}")
    print(f"Answer Relevancy (cosine)  : {np.mean(relevancies):.2%}")
    print(f"Hallucination Rate         : {np.mean(hallucinations):.2%}")
    print("=" * 60)

    # Save results
    results = {
        "precision_at_5": float(np.mean(precisions)),
        "answer_relevancy": float(np.mean(relevancies)),
        "hallucination_rate": float(np.mean(hallucinations)),
        "n_samples": len(eval_set),
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run_eval()