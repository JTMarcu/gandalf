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

# ---------------------------------------------------------------------------
# Gradio Theme (handles inputs, buttons, borders natively — no CSS hacks)
# ---------------------------------------------------------------------------
import gradio as gr  # noqa: E402
from gradio.themes.utils import sizes  # noqa: E402

_gold = gr.themes.Color(
    c50="#faf6eb", c100="#f0e6c8", c200="#e4d5a0", c300="#d4c085",
    c400="#c8a84e", c500="#b8963a", c600="#a07e2e", c700="#8a6b24",
    c800="#70561c", c900="#5a4516", c950="#4a3812", name="gold",
)
_brown = gr.themes.Color(
    c50="#d4c5a9", c100="#c4b494", c200="#a89670", c300="#8c7a56",
    c400="#6e5f40", c500="#564a30", c600="#3d3520", c700="#2a2518",
    c800="#1e1a14", c900="#12100b", c950="#0d0b08", name="brown",
)

GANDALF_THEME = gr.themes.Base(
    primary_hue=_gold,
    secondary_hue=_gold,
    neutral_hue=_brown,
    font=(gr.themes.GoogleFont("Crimson Text"), "Georgia", "serif"),
    font_mono=("Consolas", "monospace"),
).set(
    # Body
    body_background_fill="#0d0b08",
    body_background_fill_dark="#0d0b08",
    body_text_color="#d4c5a9",
    body_text_color_dark="#d4c5a9",
    body_text_color_subdued="#8c7a56",
    body_text_color_subdued_dark="#8c7a56",
    # Backgrounds
    background_fill_primary="#12100b",
    background_fill_primary_dark="#12100b",
    background_fill_secondary="#1e1a14",
    background_fill_secondary_dark="#1e1a14",
    # Blocks
    block_background_fill="#1e1a14",
    block_background_fill_dark="#1e1a14",
    block_border_color="#3d3520",
    block_border_color_dark="#3d3520",
    block_label_text_color="#c8a84e",
    block_label_text_color_dark="#c8a84e",
    block_title_text_color="#c8a84e",
    block_title_text_color_dark="#c8a84e",
    # Inputs
    input_background_fill="#12100b",
    input_background_fill_dark="#12100b",
    input_background_fill_focus="#1e1a14",
    input_background_fill_focus_dark="#1e1a14",
    input_border_color="#c8a84e",
    input_border_color_dark="#c8a84e",
    input_border_color_focus="#c8a84e",
    input_border_color_focus_dark="#c8a84e",
    input_border_width="2px",
    input_placeholder_color="#6e5f40",
    input_placeholder_color_dark="#6e5f40",
    # Borders
    border_color_primary="#3d3520",
    border_color_primary_dark="#3d3520",
    border_color_accent="#c8a84e",
    border_color_accent_dark="#c8a84e",
    # Primary button
    button_primary_background_fill="linear-gradient(135deg, #5a4a32, #3d3424)",
    button_primary_background_fill_dark="linear-gradient(135deg, #5a4a32, #3d3424)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6b5d4a, #5a4a32)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #6b5d4a, #5a4a32)",
    button_primary_text_color="#c8a84e",
    button_primary_text_color_dark="#c8a84e",
    button_primary_border_color="#c8a84e",
    button_primary_border_color_dark="#c8a84e",
    # Secondary button
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="#2a2518",
    button_secondary_background_fill_hover_dark="#2a2518",
    button_secondary_text_color="#6b5d4a",
    button_secondary_text_color_dark="#6b5d4a",
    button_secondary_border_color="#3d3520",
    button_secondary_border_color_dark="#3d3520",
    # Accent
    color_accent="#c8a84e",
    color_accent_soft="#2a2518",
    color_accent_soft_dark="#2a2518",
    loader_color="#c8a84e",
    loader_color_dark="#c8a84e",
    shadow_drop="none",
    shadow_drop_lg="none",
    block_shadow="none",
)

# CSS only for things the theme API can't handle
CUSTOM_CSS: str = """\
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');

.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
}

/* Title */
#title {
    overflow: visible !important;
}
#title > * {
    overflow: visible !important;
}
#title h1 {
    font-family: 'Cinzel', serif !important;
    color: #c8a84e !important;
    text-align: center !important;
    font-size: 2.8rem !important;
    letter-spacing: 0.1em !important;
    text-shadow: 0 0 20px rgba(200, 168, 78, 0.3) !important;
    white-space: nowrap !important;
    overflow: visible !important;
}

/* Description */
#description {
    text-align: center !important;
    color: #9a8c7a !important;
}
#description em, #description strong {
    color: #c8a84e !important;
}

/* Input label */
#input-label {
    text-align: center !important;
}
#input-label strong {
    font-family: 'Cinzel', serif !important;
    color: #c8a84e !important;
    letter-spacing: 0.05em !important;
}

/* Footer */
#footer {
    text-align: center !important;
    color: #4a3f30 !important;
    font-style: italic !important;
}
footer { display: none !important; }

/* Hide dark/light mode toggle */
.dark-mode-toggle, .gradio-container > .flex { display: none !important; }
"""
