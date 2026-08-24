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
    CUSTOM_CSS,
    EMBEDDING_MODEL,
    EXAMPLE_QUESTIONS,
    FAISS_INDEX_DIR,
    GANDALF_QUOTES,
    GANDALF_THEME,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_MESSAGE,
    USER_TEMPLATE,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
retriever = db.as_retriever(search_kwargs={"k": 6})

# ── LLM ───────────────────────────────────────────────────────────────────
client = InferenceClient(model=LLM_MODEL, api_key=hf_token)


# ── Chat function ─────────────────────────────────────────────────────────

def ask_gandalf(question: str) -> str:
    """Retrieve relevant lore and generate a Gandalf-style answer."""
    if not question or not question.strip():
        return "*Speak, friend, and ask your question.*"

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build chat messages
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]

    # Generate answer via chat completion
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        answer: str = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 — surface a friendly message, not a stack trace
        return (
            "⚠️ The palantír is clouded — I could not reach the model.\n\n"
            f"*Details: {exc}*\n\n"
            "Check that a valid `HUGGINGFACEHUB_API_TOKEN` (with Inference "
            "Providers access) is configured."
        )

    sources: list = docs

    # Fallback when the model punts
    if "i don't know" in answer.lower():
        answer = random.choice(GANDALF_QUOTES)

    # Build source citations from all retrieved chunks (deduplicated, ordered)
    seen: set = set()
    citations: list[str] = []
    for doc in sources:
        meta = doc.metadata
        book = meta.get("book_name", "Unknown book")
        chapter_num = meta.get("chapter_number", "")
        chapter_name = meta.get("chapter_name", "Unknown chapter")
        parts = [book]
        if chapter_num:
            parts.append(chapter_num)
        if chapter_name and chapter_name != "Unknown":
            parts.append(chapter_name)
        label = ", ".join(parts)
        if label not in seen:
            seen.add(label)
            citations.append(label)

    if citations:
        reference = "📖 Sources:\n" + "\n".join(f"- {c}" for c in citations)
    else:
        reference = "📖 Source: Unknown"

    return f"{answer}\n\n{reference}"


# ── Gradio UI ─────────────────────────────────────────────────────────────

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=GANDALF_THEME,
    title="Gandalf — Tolkien Lore Chatbot",
) as demo:

    # Header
    gr.Markdown(f"# {APP_TITLE}", elem_id="title")
    gr.Markdown(APP_DESCRIPTION, elem_id="description")

    # Input
    gr.Markdown("**What would you ask the Grey Wizard?**", elem_id="input-label")
    question = gr.Textbox(
        placeholder="e.g. Who is Belladonna Took?",
        lines=3,
        label="Question",
        show_label=False,
        elem_id="question",
    )

    # Buttons
    with gr.Row():
        clear_btn = gr.ClearButton(value="Clear")
        submit_btn = gr.Button("Ask Gandalf", variant="primary")

    # Output
    answer = gr.Markdown(
        value="*Gandalf's answer will appear here…*",
        elem_id="answer",
    )

    # Examples
    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUESTIONS],
        inputs=question,
    )

    # Footer
    gr.Markdown(
        "Built with FAISS · Sentence-Transformers · Qwen · Gradio",
        elem_id="footer",
    )

    # Events
    submit_btn.click(ask_gandalf, inputs=question, outputs=answer)
    question.submit(ask_gandalf, inputs=question, outputs=answer)
    clear_btn.add([question, answer])

if __name__ == "__main__":
    # ssr_mode=False avoids Gradio's experimental SSR layer, which emits
    # benign SvelteKit 405 errors in the HF Space logs.
    demo.launch(ssr_mode=False)