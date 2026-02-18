---
title: gandalf
emoji: 🧙
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Gandalf 🧙
**Tolkien Lore RAG Chatbot — Powered by FAISS, Qwen & Gradio**

Gandalf is a Retrieval-Augmented Generation (RAG) chatbot grounded in J.R.R. Tolkien's core legendarium:

- 📘 **The Hobbit** (1937)
- 📗 **The Lord of the Rings** (1954–1955)
- 📙 **The Silmarillion** (1977)

It combines semantic vector search over the full text with the **Qwen2.5-7B-Instruct** LLM to deliver canonical, chapter-referenced answers — all in Gandalf's voice, wrapped in a Middle-earth themed UI.

🔗 **Live Demo**: [huggingface.co/spaces/CupaTroopa/gandalf](https://huggingface.co/spaces/CupaTroopa/gandalf)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-Book RAG** | Searches across all three books simultaneously via FAISS |
| **Source Citations** | Every answer includes book + chapter reference |
| **Gandalf Persona** | Responds with ancient wisdom, wit, and poetic cadence |
| **Fallback Quotes** | Graceful "I don't know" with in-character Gandalf lines |
| **Middle-earth UI** | Dark parchment theme with Cinzel & Crimson Text fonts, gold accents |
| **Auto-Deploy** | Push to `main` → GitHub Action syncs to HuggingFace Spaces |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (via `langchain-community`) |
| **LLM** | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) via HF Inference API |
| **LLM Interface** | `huggingface_hub.InferenceClient.chat_completion()` |
| **Web UI** | [Gradio 5](https://www.gradio.app/) Blocks API with custom CSS |
| **PDF Parsing** | `pdfminer.six` via LangChain's `PyPDFLoader` |
| **CI/CD** | GitHub Actions → `huggingface_hub.upload_folder()` |

---

## 📁 Project Structure

```
Gandalf/
├── app.py                  # Gradio web app (local & HF Spaces entry point)
├── config.py               # Constants, prompts, model settings, UI theme
├── indexer.py              # Unified PDF → FAISS indexing pipeline
├── requirements.txt        # Python dependencies
├── gandalf_index/          # FAISS vectorstore (index.faiss + index.pkl)
│   ├── index.faiss
│   └── index.pkl
├── archive/                # Legacy scripts kept for reference
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       └── sync-to-hf.yml # Auto-sync to HuggingFace Spaces
├── .gitignore
└── README.md
```

**Not committed** (see `.gitignore`):
- `books/` — source PDFs (copyrighted)
- `models/` — local GGUF models
- `notebooks/` — Jupyter experiments
- `.env` — API tokens

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/JTMarcu/Gandalf.git
cd Gandalf
pip install -r requirements.txt
```

### 2. Set Your API Token
Create a `.env` file:
```properties
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Launch the Chatbot
```bash
python app.py
```
Open the Gradio link in your browser and speak, friend!

### 4. (Optional) Rebuild the Vector Index
If you want to re-index from source PDFs, place them in `books/` and run:
```bash
python indexer.py                   # All three books
python indexer.py --book hobbit     # Just The Hobbit
python indexer.py --book lotr silmarillion
```

---

## 🔍 How It Works

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FAISS       │────▶│  Top-k chunks    │────▶│  Qwen 2.5-7B    │
│  Vector      │     │  + metadata      │     │  Instruct        │
│  Search      │     │  (book, chapter) │     │  (chat_completion)│
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              Gandalf-style answer
                                              + source citation
```

1. **Embed the question** — The user's query is vectorized with `all-MiniLM-L6-v2`
2. **Retrieve context** — FAISS returns the most relevant text chunks (500 chars each) with book/chapter metadata
3. **Generate answer** — The context + question are sent to Qwen2.5-7B-Instruct via `InferenceClient.chat_completion()` with a Gandalf persona system prompt
4. **Cite sources** — The response includes the book name and chapter from the top retrieved chunk
5. **Fallback** — If the model says "I don't know", a random in-character Gandalf quote is returned instead

---

## 🚢 Deployment

The repo auto-syncs to [HuggingFace Spaces](https://huggingface.co/spaces/CupaTroopa/gandalf) via GitHub Actions on every push to `main`.

Only these files are uploaded to the Space:
- `app.py`, `config.py`, `requirements.txt`, `README.md`, `gandalf_index/**`

**Setup** (one-time):
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Add a secret named `HF_TOKEN` with a HuggingFace write token
3. Push to `main` — the workflow handles the rest

---

## 📌 Requirements

- Python 3.10+
- HuggingFace API token (free tier works)
- ~500 MB disk for the FAISS index + dependencies

---

## 🔮 Future Ideas

- Chat history / multi-turn conversation
- Support for *Unfinished Tales* and *The Letters of J.R.R. Tolkien*
- Gandalf-style voice synthesis
- Source text preview alongside answers
- Streaming responses

---

> *"All we have to decide is what to do with the time that is given us."*
> ― Gandalf, *The Fellowship of the Ring*

---

Built by [Jonathan Marcu](https://github.com/JTMarcu) · [Live Demo](https://huggingface.co/spaces/CupaTroopa/gandalf)
