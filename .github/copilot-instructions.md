# Gandalf — Copilot Instructions

## Project Overview
Gandalf is a Retrieval-Augmented Generation (RAG) chatbot powered by Tolkien's legendarium (The Hobbit, The Lord of the Rings, The Silmarillion). It uses FAISS for vector search, HuggingFace models for generation, and Gradio for the web UI. The live demo runs on HuggingFace Spaces.

## Architecture
```
Gandalf/
├── app.py              # Gradio web app (entry point for both local & HF Spaces)
├── config.py           # Shared constants, prompts, model settings
├── indexer.py           # Unified PDF → FAISS indexing pipeline
├── requirements.txt     # Python dependencies
├── gandalf_index/       # FAISS vectorstore (index.faiss + index.pkl)
├── books/               # Source PDFs (not committed — see README)
├── models/              # Optional local GGUF models (not committed)
├── notebooks/           # Archived Jupyter experiments
├── archive/             # Legacy scripts kept for reference
├── .github/workflows/   # CI: auto-sync to HuggingFace Spaces
└── README.md
```

## Key Conventions

### Python
- Target **Python 3.10+**
- Use `langchain_community` and `langchain_huggingface` (not deprecated `langchain.embeddings`, `langchain.vectorstores`)
- All imports at top of file, grouped: stdlib → third-party → local
- Use type hints for function signatures
- Use `logging` module instead of print statements in library code (print is OK in CLI scripts)

### LangChain
- Embeddings: `langchain_huggingface.HuggingFaceEmbeddings`
- Vectorstore: `langchain_community.vectorstores.FAISS`
- LLM endpoint: `huggingface_hub.InferenceClient` (chat_completion API)
- LLM model: `Qwen/Qwen2.5-7B-Instruct` (via HF Inference Providers)
- Always pass `allow_dangerous_deserialization=True` when loading FAISS indexes
- Keep chunk_size=500, chunk_overlap=100 for consistency with existing index

### Configuration
- All model names, prompt templates, and tunable parameters live in `config.py`
- Environment variables via `python-dotenv`; token key is `HUGGINGFACEHUB_API_TOKEN`
- Never hardcode API tokens

### HuggingFace Spaces Deployment
- The GitHub Action in `.github/workflows/sync-to-hf.yml` auto-syncs to `CupaTroopa/gandalf`
- HF Space expects `app.py`, `requirements.txt`, and `gandalf_index/` at repo root
- The `app.py` must work both locally and on HF Spaces (use `dotenv` with graceful fallback)
- Space SDK: Gradio

### FAISS Index
- Canonical index is `gandalf_index/` (all 3 books combined)
- To rebuild: `python indexer.py` (requires PDFs in `books/`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Metadata per chunk: `book_name`, `chapter_number` (Hobbit only), `chapter_name`

### Gradio UI
- Keep the interface simple — single textbox input, text output
- Include source citation (book + chapter) in every response
- Fallback Gandalf quotes when the model says "I don't know"

## Do NOT
- Commit `.env`, `books/`, `models/`, or `__pycache__/`
- Use `langchain.embeddings` or `langchain.vectorstores` (deprecated)
- Change the FAISS index folder name (HF Space depends on `gandalf_index/`)
- Remove the `allow_dangerous_deserialization=True` flag (required for FAISS)
