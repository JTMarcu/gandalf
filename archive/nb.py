from pathlib import Path
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import nbformat

# Define a cloud-friendly RAG chatbot notebook using Hugging Face Hub
notebook_code = [
    {
        "cell_type": "markdown",
        "source": "# 🧙‍♂️ Gandalf RAG Chatbot using Hugging Face Hub\n\nThis notebook runs a Retrieval-Augmented Generation (RAG) chatbot using a vectorstore index + a remote LLM from Hugging Face Inference API."
    },
    {
        "cell_type": "code",
        "source": """# 📦 Install required packages (if running in Colab or HF Spaces)
# !pip install langchain langchain-community sentence-transformers faiss-cpu python-dotenv
"""
    },
    {
        "cell_type": "markdown",
        "source": "## 🔐 Setup Environment",
    },
    {
        "cell_type": "code",
        "source": """from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token. Add HUGGINGFACEHUB_API_TOKEN to your .env file.")
"""
    },
    {
        "cell_type": "markdown",
        "source": "## 🔎 Load Embeddings + Vectorstore",
    },
    {
        "cell_type": "code",
        "source": """from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("gandalf_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()"""
    },
    {
        "cell_type": "markdown",
        "source": "## 🤖 Load a Hugging Face Model for Inference",
    },
    {
        "cell_type": "code",
        "source": """from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Or try "tiiuae/falcon-7b-instruct"
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
    huggingfacehub_api_token=hf_token
)"""
    },
    {
        "cell_type": "markdown",
        "source": "## 🔁 Ask Questions (RAG)",
    },
    {
        "cell_type": "code",
        "source": """from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

question = "What happened in the mines of Moria?"
result = qa_chain.invoke({"query": question})

print("🧙 Gandalf says:\\n", result['result'])"""
    }
]

# Build the notebook
nb = new_notebook(
    cells=[
        new_markdown_cell(cell["source"]) if cell["cell_type"] == "markdown"
        else new_code_cell(cell["source"])
        for cell in notebook_code
    ]
)

# Save the notebook to file
output_path = Path("gandalf_hfhub_rag.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

output_path.absolute()
