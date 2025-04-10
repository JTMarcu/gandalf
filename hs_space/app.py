import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load API token from .env file (for local testing, remove this line when deploying to Hugging Face Spaces)
from dotenv import load_dotenv
load_dotenv()

# 🔐 Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in environment variables.")

# 🧠 Load FAISS vectorstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # updated import

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("gandalf_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# 🤖 Load LLM from Hugging Face Inference API
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=hf_token)

llm = HuggingFaceEndpoint(
    client=client,
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=256,
    task="text-generation"  # REQUIRED to avoid `task='unknown'` error
)

# 📜 Prompt and QA chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question. 
If the answer is not in the context, reply as Gandalf would — something wise or witty, and make it clear that the answer is unknown.
Context:
{context}
Question:
{question}
Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# 🎛️ Gradio Interface
import gradio as gr
import random

gandalf_quotes = [
    "Even the wisest cannot answer all questions.",
    "I fear the answer lies beyond my knowledge, dear friend.",
    "That is a riddle I cannot unravel.",
    "I do not know — and I do not make guesses lightly.",
    "Ask me again after second breakfast."
]

def ask_gandalf(question):
    result = qa_chain.invoke({"query": question})
    answer = result['result']
    sources = result['source_documents']

    if "I don't know" in answer.lower():
        answer = random.choice(gandalf_quotes)

    # Extract metadata from the first source document
    if sources:
        source = sources[0].metadata
        chapter_number = source.get("chapter_number", "Unknown chapter number")
        chapter_name = source.get("chapter_name", "Unknown chapter name")
        book_name = source.get("book_name", "Unknown book")
        reference = f"📖 Source: {book_name}, {chapter_number} - {chapter_name}"
    else:
        reference = "📖 Source: Unknown"

    return f"{answer}\n\n{reference}"

demo = gr.Interface(
    fn=ask_gandalf,
    inputs=gr.Textbox(placeholder="Ask Gandalf anything about Middle-earth..."),
    outputs="text",
    title="Ask Gandalf",
    description="🧙 Ask questions based on the lore from The Lord of the Rings."
)

if __name__ == "__main__":
    demo.launch()