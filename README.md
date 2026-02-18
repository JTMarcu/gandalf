---
title: gandalf
emoji: 🧙
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Gandalf 🧙
**Tolkien Lore RAG Chatbot — Powered by LangChain, FAISS & HuggingFace**

Gandalf is a Retrieval-Augmented Generation (RAG) chatbot grounded in J.R.R. Tolkien's core legendarium:

- 📘 **The Hobbit** (1937)
- 📗 **The Lord of the Rings** (1954–1955)
- 📙 **The Silmarillion** (1977)

It combines semantic vector search over the full text with the **Zephyr-7B** LLM to deliver canonical, chapter-referenced answers — all in Gandalf's voice.

🔗 **Live Demo**: [huggingface.co/spaces/CupaTroopa/gandalf](https://huggingface.co/spaces/CupaTroopa/gandalf)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-Book RAG** | Searches across all three books simultaneously |
| **Source Citations** | Every answer includes book + chapter reference |
| **Gandalf Persona** | Responds with ancient wisdom, wit, and poetic cadence |
| **Fallback Quotes** | Graceful "I don't know" with in-character Gandalf lines |
| **Auto-Deploy** | Push to `main` → GitHub Action syncs to HuggingFace Spaces |
| **Gradio UI** | Clean web interface that works locally and on HF Spaces |

---

## 📁 Project Structure

```
Gandalf/
├── app.py                  # Gradio web app (local & HF Spaces entry point)
├── config.py               # All constants, prompts, model settings
├── indexer.py              # Unified PDF → FAISS indexing pipeline
├── requirements.txt        # Python dependencies
├── gandalf_index/          # FAISS vectorstore (index.faiss + index.pkl)
│   ├── index.faiss
│   └── index.pkl
├── books/                  # Source PDFs (not committed)
├── models/                 # Optional local GGUF models (not committed)
├── notebooks/              # Archived Jupyter experiments
├── archive/                # Legacy scripts kept for reference
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       └── sync-to-hf.yml # Auto-sync to HuggingFace Spaces
├── .gitignore
└── README.md
```

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

### 3. (Optional) Rebuild the Vector Index
Place PDFs in `books/` and run:
```bash
python indexer.py                   # All three books
python indexer.py --book hobbit     # Just The Hobbit
python indexer.py --book lotr silmarillion
```

### 4. Launch the Chatbot
```bash
python app.py
```
Open the Gradio link in your browser and speak, friend!

---

## 🧪 Example

```
Q: What is the origin of the Silmarils?

🧙 Gandalf says:
The Silmarils were wrought by Fëanor, greatest of the Noldor, in the days
before the Darkening of Valinor. Within them he captured the light of the
Two Trees of Valinor — Telperion and Laurelin — and no craft since has
equalled their making...

📖 Source: The Silmarillion, Of the Silmarils and the Unrest of the Noldor
```

---

## 🛠 How It Works

1. **Text Extraction** — PDFs are parsed with `pdfminer.six` via LangChain's `PyPDFLoader`
2. **Chunking + Metadata** — Text is split into 500-char chunks with chapter/book metadata
3. **Embedding + Storage** — Each chunk is vectorized with `all-MiniLM-L6-v2` and stored in FAISS
4. **Retrieval + Generation** — Zephyr-7B uses top-k retrieved chunks to generate in-character answers

---

## 🚢 Deployment

The repo auto-syncs to [HuggingFace Spaces](https://huggingface.co/spaces/CupaTroopa/gandalf) via GitHub Actions on every push to `main`.

**Setup** (one-time):
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Add a secret named `HF_TOKEN` with a HuggingFace write token
3. Push to `main` — the workflow handles the rest

---

## 📌 Requirements

- Python 3.10+
- HuggingFace API token (free tier works)
- ~8–16 GB RAM for local model inference (optional)

---

## 🔮 Future Enhancements

- Add support for *Unfinished Tales* and *The Letters of J.R.R. Tolkien*
- Gandalf-style voice with ElevenLabs or Bark
- Chat history / multi-turn conversation
- Chapter highlighting or source text preview
- Offline-only mode with GGUF-compatible local models

---

> *"All we have to decide is what to do with the time that is given us."*
> ― Gandalf, *The Fellowship of the Ring*

---

Built by [Jonathan Marcu](https://github.com/JTMarcu) · [Live Demo](https://huggingface.co/spaces/CupaTroopa/gandalf)
