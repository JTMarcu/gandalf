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

# Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

# PDF path
book_path = os.path.join("books", "The Silmarillion.pdf")
loader = PyPDFLoader(book_path)
docs = loader.load()

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Book and known chapter titles
book_name = "The Silmarillion"

chapter_titles = {
    "Ainulindalë",
    "Valaquenta",
    "Akallabêth",
    "Of the Rings of Power and the Third Age",
    "Of the Beginning of Days",
    "Of Aulë and Yavanna",
    "Of the Coming of the Elves and the Captivity of Melkor",
    "Of Thingol and Melian",
    "Of Eldamar and the Princes of the Eldalië",
    "Of Fëanor and the Unchaining of Melkor",
    "Of the Silmarils and the Unrest of the Noldor",
    "Of the Darkening of Valinor",
    "Of the Flight of the Noldor",
    "Of the Sindar",
    "Of the Sun and Moon and the Hiding of Valinor",
    "Of Men",
    "Of the Return of the Noldor",
    "Of Beleriand and Its Realms",
    "Of the Noldor in Beleriand",
    "Of Maeglin",
    "Of the Coming of Men into the West",
    "Of the Ruin of Beleriand and the Fall of Fingolfin",
    "Of Beren and Lúthien",
    "Of the Fifth Battle: Nirnaeth Arnoediad",
    "Of Túrin Turambar",
    "Of the Ruin of Doriath",
    "Of Tuor and the Fall of Gondolin",
    "Of the Voyage of Eärendil and the War of Wrath"
}

# Track current section
current_chapter = "Unknown"
final_docs = []

for doc in chunks:
    lines = doc.page_content.splitlines()
    assigned = False

    for line in lines:
        clean = line.strip()

        for title in chapter_titles:
            if clean.lower() == title.lower():
                current_chapter = title
                assigned = True
                break

        if assigned:
            break

    final_docs.append(Document(
        page_content=doc.page_content,
        metadata={
            "book_name": book_name,
            "chapter_name": current_chapter
        }
    ))

    print(f"📘 {book_name} | 📖 {current_chapter}")

# Save FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_docs, embeddings)
vectorstore.save_local("gandalf_index_silmarillion")

print("✅ Vectorstore saved as 'gandalf_index_silmarillion'")
