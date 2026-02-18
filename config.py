"""Shared configuration for the Gandalf RAG chatbot."""

# ---------------------------------------------------------------------------
# Embedding & Vectorstore
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_DIR: str = "gandalf_index"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_NEW_TOKENS: int = 256

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE: str = (
    "You are Gandalf the Grey, wise and powerful wizard of Middle-earth. "
    "You MUST respond ONLY in English — never use any other language. "
    "Speak with the tone of ancient wisdom, poetic cadence, and occasional wit. "
    "Use the provided lore context to answer the user's question. "
    "If the answer is not found in the lore, speak as Gandalf would — "
    "with insight, mystery, or gentle deflection."
)

USER_TEMPLATE: str = """\
Context:
{context}

Question:
{question}"""

# ---------------------------------------------------------------------------
# Fallback quotes (used when the LLM says "I don't know")
# ---------------------------------------------------------------------------
GANDALF_QUOTES: list[str] = [
    "Even the wisest cannot answer all questions.",
    "I fear the answer lies beyond my knowledge, dear friend.",
    "That is a riddle I cannot unravel.",
    "I do not know — and I do not make guesses lightly.",
    "Ask me again after second breakfast.",
    "Not all those who wander are lost — but my answer certainly is.",
    "A wizard is never late, nor is he early — he answers precisely when he means to.",
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
APP_TITLE: str = "🧙 Gandalf"
APP_DESCRIPTION: str = (
    "*Speak, friend, and enter.*\n\n"
    "A RAG chatbot grounded in the lore of **The Hobbit**, "
    "**The Lord of the Rings**, and **The Silmarillion**.\n\n"
    "Ask anything about Middle-earth and Gandalf shall answer."
)

EXAMPLE_QUESTIONS: list[str] = [
    "Who is Belladonna Took?",
    "What happened at the Battle of Helm's Deep?",
    "Tell me about the Silmarils.",
    "What is the One Ring?",
    "Who were the Istari?",
    "What is the history of Gondolin?",
]

CUSTOM_CSS: str = """\
/* ── Middle-earth theme ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* Dark parchment background */
.gradio-container {
    background: radial-gradient(ellipse at top, #1a1510 0%, #0d0b08 70%) !important;
    font-family: 'Crimson Text', Georgia, serif !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

/* Title styling */
#title {
    font-family: 'Cinzel', serif !important;
    color: #c8a84e !important;
    text-align: center !important;
    font-size: 3rem !important;
    letter-spacing: 0.15em !important;
    text-shadow: 0 0 20px rgba(200, 168, 78, 0.3) !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Description */
#description {
    text-align: center !important;
    color: #9a8c7a !important;
    font-style: italic !important;
    font-size: 1.1rem !important;
    margin-top: 0 !important;
}
#description em, #description strong {
    color: #c8a84e !important;
}

/* Decorative divider */
.divider {
    text-align: center;
    color: #5a4a32;
    font-size: 1.4rem;
    letter-spacing: 0.5em;
    margin: 0.5rem 0;
}

/* Input label */
#input-label {
    color: #c8a84e !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-align: center !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Input textbox — target all nested elements in Gradio 5 */
#question {
    background: #1e1a14 !important;
    border: 2px solid #c8a84e !important;
    border-radius: 8px !important;
    min-height: 120px !important;
}
#question * {
    background: #1e1a14 !important;
    color: #d4c5a9 !important;
    font-family: 'Crimson Text', Georgia, serif !important;
    font-size: 1.15rem !important;
}
#question textarea {
    min-height: 100px !important;
    height: 100px !important;
    padding: 16px !important;
    border: none !important;
    caret-color: #c8a84e !important;
}
#question textarea:focus {
    box-shadow: none !important;
}
#question:focus-within {
    border-color: #c8a84e !important;
    box-shadow: 0 0 16px rgba(200, 168, 78, 0.25) !important;
}
#question textarea::placeholder {
    color: #7a6b55 !important;
    font-style: italic !important;
}
#question label {
    display: none !important;
}

/* Submit button */
#submit {
    background: linear-gradient(135deg, #5a4a32, #3d3424) !important;
    border: 1px solid #c8a84e !important;
    color: #c8a84e !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    padding: 10px 32px !important;
    transition: all 0.3s ease !important;
}
#submit:hover {
    background: linear-gradient(135deg, #6b5d4a, #5a4a32) !important;
    box-shadow: 0 0 16px rgba(200, 168, 78, 0.25) !important;
}

/* Clear button */
#clear {
    background: transparent !important;
    border: 1px solid #3d3424 !important;
    color: #6b5d4a !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
}
#clear:hover {
    border-color: #6b5d4a !important;
    color: #9a8c7a !important;
}

/* Output area */
#answer {
    background: #1e1a14 !important;
    border: 1px solid #3d3424 !important;
    border-radius: 6px !important;
    padding: 20px !important;
    min-height: 60px !important;
}
#answer .prose {
    color: #d4c5a9 !important;
    font-family: 'Crimson Text', Georgia, serif !important;
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
}
#answer label {
    display: none !important;
}

/* Example buttons */
.example-btn button {
    background: #1e1a14 !important;
    border: 1px solid #3d3424 !important;
    color: #9a8c7a !important;
    font-family: 'Crimson Text', Georgia, serif !important;
    font-size: 0.95rem !important;
    font-style: italic !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}
.example-btn button:hover {
    border-color: #c8a84e !important;
    color: #c8a84e !important;
    background: #251f17 !important;
}

/* Footer */
#footer {
    text-align: center !important;
    color: #4a3f30 !important;
    font-size: 0.85rem !important;
    margin-top: 1.5rem !important;
    font-style: italic !important;
}

/* Hide default footer */
footer { display: none !important; }
"""
