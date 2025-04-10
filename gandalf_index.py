import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # ✅ Import this for custom docs

# Load Hugging Face API token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

# Load and split PDFs from the "books" folder
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

books_folder = "books"
book_files = [f for f in os.listdir(books_folder) if f.endswith(".pdf")]

final_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for book_file in book_files:
    book_path = os.path.join(books_folder, book_file)
    loader = PyPDFLoader(book_path)
    docs = loader.load()

    book_name = os.path.splitext(book_file)[0]  # Extract book name from file name
    last_known_chapter = "Unknown"

    for doc in text_splitter.split_documents(docs):
        text = doc.page_content
        chapter_number = "Unknown"
        chapter_name = "Unknown"

        # Detect chapter number and name
        if "Chapter" in text:
            chapter_line = [line.strip() for line in text.splitlines() if "Chapter" in line]
            if chapter_line:
                last_known_chapter = chapter_line[0]
                parts = last_known_chapter.split(" ", 1)
                if len(parts) > 1:
                    chapter_number = parts[0]
                    chapter_name = parts[1]

        new_doc = Document(
            page_content=text,
            metadata={
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "book_name": book_name
            }
        )
        final_docs.append(new_doc)

# Create and save FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_docs, embeddings)
vectorstore.save_local("gandalf_index")

print("✅ Vectorstore saved as 'gandalf_index'")
