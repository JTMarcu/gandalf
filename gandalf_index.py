import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load Hugging Face API token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

# Load and process the PDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# After splitting the docs
for i, doc in enumerate(splits):
    # Simple heuristic: scan text and assign chapter if found
    text = doc.page_content
    if "Chapter" in text:
        chapter_line = [line for line in text.splitlines() if "Chapter" in line]
        if chapter_line:
            doc.metadata["chapter"] = chapter_line[0]
    else:
        # Carry forward last known chapter
        doc.metadata["chapter"] = splits[i-1].metadata.get("chapter", "Unknown")

# Create and save the FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("gandalf_index")

print("✅ Vectorstore saved as 'gandalf_index'")
