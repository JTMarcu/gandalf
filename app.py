"""Gandalf — Tolkien Lore RAG Chatbot.

Works both locally (reads .env) and on HuggingFace Spaces (reads secrets).
"""

from __future__ import annotations

import os
import random
import warnings

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    APP_DESCRIPTION,
    APP_TITLE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    GANDALF_QUOTES,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_MESSAGE,
    USER_TEMPLATE,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Environment ───────────────────────────────────────────────────────────
load_dotenv()  # no-op on HF Spaces (no .env file present)

hf_token: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN environment variable.")

# ── Vectorstore ───────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(
    FAISS_INDEX_DIR, embedding_model, allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

# ── LLM ───────────────────────────────────────────────────────────────────
client = InferenceClient(model=LLM_MODEL, token=hf_token)


# ── Chat function ─────────────────────────────────────────────────────────

def ask_gandalf(question: str) -> str:
    """Retrieve relevant lore and generate a Gandalf-style answer."""
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build chat messages
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]

    # Generate answer via chat completion
    response = client.chat_completion(
        messages=messages,
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    answer: str = response.choices[0].message.content
    sources: list = docs

    # Fallback when the model punts
    if "i don't know" in answer.lower():
        answer = random.choice(GANDALF_QUOTES)

    # Build source citation from first retrieved chunk
    if sources:
        meta = sources[0].metadata
        book = meta.get("book_name", "Unknown book")
        chapter_num = meta.get("chapter_number", "")
        chapter_name = meta.get("chapter_name", "Unknown chapter")
        parts = [book]
        if chapter_num:
            parts.append(chapter_num)
        if chapter_name and chapter_name != "Unknown":
            parts.append(chapter_name)
        reference = f"📖 Source: {', '.join(parts)}"
    else:
        reference = "📖 Source: Unknown"

    return f"{answer}\n\n{reference}"


# ── Gradio UI ─────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=ask_gandalf,
    inputs=gr.Textbox(placeholder="Ask Gandalf anything about Middle-earth..."),
    outputs="text",
    title=APP_TITLE,
    description=APP_DESCRIPTION,
)

if __name__ == "__main__":
    demo.launch()