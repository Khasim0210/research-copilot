# 🧠 Agentic AI Research Copilot

A document-grounded conversational AI assistant built with **RAG (Retrieval-Augmented Generation)**, **LangChain**, **FAISS**, and **LLMs**. It ingests PDFs, retrieves relevant context, and generates accurate, source-cited answers — exposed as a **FastAPI** REST service and containerized with **Docker**.

---

## 📊 Evaluation Results

| Metric | Score |
|---|---|
| **Retrieval Precision@5** | **85.00%** |
| **Answer Relevancy** (cosine similarity vs. ground truth) | **83.28%** |
| **Hallucination Rate** | **17.50%** |

Evaluated on 8 hand-curated Q&A pairs from the *Attention Is All You Need* paper using semantic-similarity grounding checks.

---

## 🏗️ Architecture

```text
PDF Document
     │
     ▼
 ① CHUNK         (RecursiveCharacterTextSplitter, 500 tokens, 50 overlap)
     │
     ▼
 ② EMBED         (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
     │
     ▼
 ③ STORE         (FAISS local index, persisted to disk)
     │
     ▼
[User Question]
     │
     ▼
 ④ CONDENSE      (rewrite follow-up + history → standalone question)
     │
     ▼
 ⑤ RETRIEVE      (top-5 cosine-similar chunks)
     │
     ▼
 ⑥ AUGMENT       (stuff chunks into prompt with citations)
     │
     ▼
 ⑦ GENERATE      (Qwen2.5-7B-Instruct via HuggingFace Inference API)
     │
     ▼
[Grounded Answer + Source Page Citations]
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain 0.3 | Composable chains, retriever abstraction, memory primitives |
| Vector DB | FAISS | Fast local similarity search, persistent index |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | 384-dim, free, runs on MPS/CPU |
| LLM | Qwen2.5-7B-Instruct (HF Inference API) | Strong open-source instruct model, free tier |
| Document Loading | pypdf | Robust PDF parsing |
| API | FastAPI + Uvicorn | Async, auto-generated OpenAPI docs |
| Deployment | Docker + docker-compose | Reproducible, environment-independent |
| Evaluation | NumPy + custom metrics | Precision@K, semantic relevancy, hallucination detection |

---

## 📂 Project Structure

```text
research-copilot/
├── data/documents/          # Uploaded PDFs
├── src/
│   ├── ingest.py            # PDF loading, chunking, embedding
│   ├── vectorstore.py       # FAISS build/load/search
│   ├── rag_chain.py         # LangChain retrieval + generation chain
│   ├── memory.py            # Conversational chain with history condensation
│   └── api.py               # FastAPI endpoints (/health /chat /ingest)
├── evaluation/
│   ├── eval_dataset.json    # 8 Q&A pairs for evaluation
│   ├── evaluate.py          # Precision@5, relevancy, hallucination metrics
│   └── results.json         # Saved metrics
├── Dockerfile               # Container recipe
├── docker-compose.yml       # Easy orchestration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Full Setup & Run Guide

### Prerequisites

- macOS / Linux / Windows
- Python 3.11+
- Conda or venv
- Docker Desktop (for containerized deployment)
- A free [Hugging Face account + access token](https://huggingface.co/settings/tokens)

### 1. Clone the repository

```bash
git clone https://github.com/Khasim0210/research-copilot.git
cd research-copilot
```

### 2. Add your Hugging Face token

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

> 💡 Get a free token at https://huggingface.co/settings/tokens (select "Read" scope).

### 3. Add documents to query

Drop any PDFs into `data/documents/`. To use the included demo (the original Transformer paper):

```bash
curl -L -o data/documents/attention.pdf https://arxiv.org/pdf/1706.03762.pdf
```

---

### 4A. Run locally (with conda)

```bash
# Create + activate the environment
conda create -n copilot python=3.11 -y
conda activate copilot

# Install dependencies
pip install -r requirements.txt
pip install sentencepiece python-multipart

# Build the FAISS vector store from your PDFs (one-time)
python -m src.vectorstore

# Start the API server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The server will be live at **http://localhost:8000**.

### 4B. Run with Docker (recommended)

```bash
docker compose up --build
```

The first build takes ~5–15 min. Subsequent starts are near-instant.

---

### 5. Try it

Open the interactive API docs in your browser:

**http://localhost:8000/docs**

You'll see Swagger UI with all 3 endpoints (`/health`, `/chat`, `/ingest`). Click **"Try it out"** on `/chat`, paste a question, and execute.

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/chat` | Ask a question with optional conversation history |
| `POST` | `/ingest` | Upload a PDF and rebuild the index |

### Example: `/chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is multi-head attention?",
    "history": []
  }'
```

Response:

```json
{
  "question": "What is multi-head attention?",
  "standalone_question": "What is multi-head attention?",
  "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It uses h=8 parallel attention layers [Source: page 4].",
  "sources_used": "[Source: page 4]..."
}
```

### Multi-turn example

The chain rewrites follow-up questions using history before retrieval:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "And what about the decoder?",
    "history": [
      {"role": "user",      "content": "How many layers does the encoder have?"},
      {"role": "assistant", "content": "The encoder has N = 6 identical layers."}
    ]
  }'
```

---

## 🧪 Running the Evaluation

```bash
python -m evaluation.evaluate
```

Outputs per-question metrics + aggregate scores, saved to `evaluation/results.json`.

Sample output:

```text
[1/8] Q: What is the Transformer architecture?
   Precision@5: 1.00   Relevancy: 0.91   Hallucination: 0.00
...
==============================================================
AGGREGATE RESULTS
==============================================================
Retrieval Precision@5      : 85.00%
Answer Relevancy (cosine)  : 83.28%
Hallucination Rate         : 17.50%
==============================================================
```

---

## 🔬 Key Design Decisions

- **MiniLM-L6-v2 for embeddings** — small (90MB), fast on CPU, 384 dims is enough for paragraph-scale retrieval.
- **Two-stage conversational chain** — follow-ups are first rewritten into standalone questions using chat history, *then* retrieval runs. This avoids the common bug where "what about its limits?" retrieves nothing relevant.
- **Strict grounded prompt** — system prompt forces the LLM to refuse if context is insufficient, mitigating hallucination.
- **Device-agnostic embeddings** — auto-detects MPS (Apple Silicon) / CUDA / CPU so the same code runs on dev laptop and Docker container.
- **Sentence-level hallucination metric** — uses semantic similarity with math-content filtering for robust grounding detection.

---

## 🛑 Stopping the App

**Local (conda):** `Ctrl + C` in the terminal running uvicorn.

**Docker:**

```bash
docker compose down
```

---

## 🚧 Future Improvements

1. **Hybrid retrieval** — combine dense (FAISS) with sparse (BM25) for better recall on keyword queries.
2. **Reranking** — add a cross-encoder reranker (e.g., bge-reranker) on top of FAISS top-20 → top-5.
3. **DPO / PPO fine-tuning** — collect preference pairs and align the generator with `trl.DPOTrainer` for better instruction-following.
4. **Streaming responses** — switch FastAPI endpoint to Server-Sent Events for token-by-token output.
5. **Multi-document with metadata filters** — filter retrieval by document, author, or date.

---