import os
import re
import warnings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=FutureWarning)

# Load Hugging Face API token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

# Only process The Hobbit
books_folder = "books"
hobbit_file = "The Hobbit.pdf"
book_path = os.path.join(books_folder, hobbit_file)

# Load and split
loader = PyPDFLoader(book_path)
docs = loader.load()

final_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Track last known chapter metadata
last_known_chapter_number = "Unknown"
last_known_chapter_name = "Unknown"
last_known_book_name = "The Hobbit"

for doc in split_docs:
    text = doc.page_content
    lines = text.splitlines()

    chapter_number = "Unknown"
    chapter_name = "Unknown"
    book_name = "The Hobbit"

    for i, line in enumerate(lines):
        clean_line = line.strip()

        # Match "Chapter I", "CHAPTER 2", etc.
        chapter_match = re.match(r"(?i)^chapter\s+([ivxlcdm\d]+)", clean_line)
        if chapter_match:
            chapter_number = f"Chapter {chapter_match.group(1).upper()}"

            # Look ahead for title (next non-empty, not another "chapter")
            for j in range(i+1, min(i+6, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.lower().startswith("chapter"):
                    chapter_name = next_line.title() if next_line.isupper() else next_line
                    break

            # Update fallback cache
            last_known_chapter_number = chapter_number
            last_known_chapter_name = chapter_name
            last_known_book_name = book_name
            break

    # Fallback to last known chapter if this chunk doesn’t contain one
    if chapter_number == "Unknown":
        chapter_number = last_known_chapter_number
        chapter_name = last_known_chapter_name
        book_name = last_known_book_name

    new_doc = Document(
        page_content=text,
        metadata={
            "chapter_number": chapter_number,
            "chapter_name": chapter_name,
            "book_name": book_name
        }
    )
    final_docs.append(new_doc)

    print(f"📘 {book_name} | 📖 {chapter_number} - {chapter_name}")

# Save to FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_docs, embeddings)
vectorstore.save_local("gandalf_index_hobbit")

print("✅ Vectorstore saved as 'gandalf_index_hobbit'")
