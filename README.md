# Agentic RAG Copilot

An AI-powered IT Support Assistant built using **LangGraph, FastAPI, Groq, Pinecone, and Tavily**. The system intelligently routes user queries, retrieves relevant knowledge, evaluates document relevance, and generates context-aware responses.

## Key Features
- **Agentic RAG:** Dynamic query routing using LangGraph.
- **Semantic Search:** Hugging Face embeddings with Pinecone.
- **Web Search:** Tavily integration for external information.
- **Self-Correction:** Evidence grading, query rewriting, and retrieval retries.
- **AI Responses:** Groq-powered LLM inference.
- **Observability:** Agent execution traces and SQLite audit logs.
- **Web Interface:** Responsive HTML, CSS, and JavaScript frontend.

## Tech Stack
**Python | LangGraph | LangChain | FastAPI | Groq | Pinecone | Hugging Face | Tavily | SQLite | Docker**

## Run Locally

```bash
pip install -r requirements.txt
python ingest_sample_kb.py
python run.py
```

Open: `http://127.0.0.1:8080`

Configure the required API keys in a local `.env` file before running.

## Architecture

**User Query → LangGraph Router → Knowledge Base Retrieval → Evidence Grading → Web Search (if needed) → LLM → Response**

## Attribution
Adapted from the open-source [Enterprise IT Support Agentic RAG Copilot](https://github.com/entbappy/Enterprise-IT-Support-Agentic-RAG-Copilot) project (Apache 2.0). Customized with Groq inference, Hugging Face embeddings, and additional configuration improvements.
