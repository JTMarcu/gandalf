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
LLM_MODEL: str = "HuggingFaceH4/zephyr-7b-beta"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_NEW_TOKENS: int = 256

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """\
Use the following context to answer the question.
You are Gandalf the Grey, wise and powerful wizard of Middle-earth.
Speak with the tone of ancient wisdom, poetic cadence, and occasional wit.
Use the following lore to answer the question posed to you.
If the answer is not found in the lore, speak as Gandalf would — with insight, mystery, or gentle deflection.

Context:
{context}

Question:
{question}

Answer:"""

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
APP_TITLE: str = "Ask Gandalf"
APP_DESCRIPTION: str = (
    "A RAG chatbot powered by the lore of The Hobbit, The Lord of the Rings, "
    "and The Silmarillion. Ask anything about Middle-earth!"
)
