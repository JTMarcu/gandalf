# Gandalf 🧙  
**Tolkien Lore Chatbot — Powered by RAG**

Gandalf is a Retrieval-Augmented Generation (RAG) chatbot trained on the full text of J.R.R. Tolkien's core legendarium:

- 📘 [The Hobbit (1937)](https://archive.org/details/TheHobbit_201905)
- 📗 [The Lord of the Rings (1954–1955)](https://archive.org/details/tolkien-j.-the-lord-of-the-rings-harper-collins-ebooks-2010)
- 📙 [The Silmarillion (1977)](https://archive.org/details/TheSilmarillionIllustratedJ.R.R.TolkienTedNasmith)

Built with LangChain, FAISS, and the Mistral-7B-Instruct model, this project offers canonical, chapter-referenced answers to your Middle-earth questions — either locally or via Hugging Face Spaces.

🔗 **Live Demo**: [Ask Gandalf on Hugging Face](https://huggingface.co/spaces/CupaTroopa/gandalf)

---

## ✨ Features

- **📚 Multi-Book Integration**: Pulls from The Hobbit, LOTR, and The Silmarillion.
- **🧠 RAG Pipeline**: Combines semantic search and generation.
- **📎 Source-Aware Responses**: Includes book/chapter metadata in answers.
- **💻 Local Execution (optional)**: No API calls after setup.
- **🌐 Gradio UI**: Web interface for local or Hugging Face deployment.

---

## 📁 Project Structure

```
Gandalf/
├── books/                       # PDF files for indexing
│   ├── The Hobbit.pdf
│   ├── The Lord of the Rings.pdf
│   └── The Silmarillion.pdf
├── gandalf_index.py            # Merged indexing script (Hobbit + LOTR + Silmarillion)
├── app.py                      # Gradio chatbot interface
├── gandalf_index/              # FAISS vectorstore
│   ├── index.faiss
│   └── index.pkl
├── models/                     # Optional local LLM folder
├── .env                        # Hugging Face API token
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Add Environment Variable
Create a `.env` file:
```properties
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 3. Add Tolkien PDFs
Place all 3 books inside a `/books` folder:
- `The Hobbit.pdf`
- `The Lord of the Rings.pdf`
- `The Silmarillion.pdf`

### 4. Generate the Vector Index
```bash
python gandalf_index.py
```

### 5. Launch the Chatbot
```bash
python app.py
```

Open the Gradio link in your browser and speak, friend!

---

## 🧪 Example Usage

```python
question = "What is the origin of the Silmarils?"
result = qa_chain.invoke({"query": question})
print("🧙 Gandalf says:\n", result['result'])
```

---

## 🛠 How It Works

1. **Text Extraction** — PDFs are parsed with `pdfminer.six`.
2. **Chunking + Metadata** — LangChain splits the text by chapter and book.
3. **Embedding + Storage** — Each chunk is vectorized (MiniLM) and stored in FAISS.
4. **Retrieval + Generation** — Mistral-7B uses top-k chunks to answer contextually.

---

## 📌 Requirements

- Python 3.8+
- ~8–16 GB RAM for local model inference (optional)
- Access to Hugging Face for inference or deployment

---

## 🛠️ Future Enhancements

- Add support for *Unfinished Tales* and *The Letters of J.R.R. Tolkien*
- Gandalf-style voice with ElevenLabs or Bark
- Chapter highlighting or source text preview
- Offline-only mode with GGUF-compatible local models

---

> “The burned hand teaches best. After that advice about fire goes to the heart.”  
> ― J.R.R. Tolkien, *The Two Towers*

---

🛡️ Built with care by [Jonathan Marcu](https://github.com/JTMarcu) · [Live Demo](https://huggingface.co/spaces/CupaTroopa/gandalf)
```