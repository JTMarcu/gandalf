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
book_path = os.path.join("books", "The Lord of The Rings.pdf")
loader = PyPDFLoader(book_path)
docs = loader.load()

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Book section transitions
book_sections = {
    "THE FELLOWSHIP OF THE RING": "The Fellowship of the Ring",
    "THE TWO TOWERS": "The Two Towers",
    "THE RETURN OF THE KING": "The Return of the King"
}

# Known extra sections
extra_sections = {
    "NOTE ON THE TEXT": ("Front Matter", "Note on the Text"),
    "NOTE ON THE 50TH ANNIVERSARY EDITION": ("Front Matter", "Note on the 50th Anniversary Edition"),
    "FOREWORD TO THE SECOND EDITION": ("Front Matter", "Foreword"),
    "PROLOGUE CONCERNING HOBBITS, AND OTHER MATTERS": ("Front Matter", "Prologue"),
    "APPENDICES": ("Appendices", "Appendices"),
    "ANNALS OF THE KINGS AND RULERS": ("Appendices", "Annals of the Kings and Rulers"),
    "THE TALE OF YEARS": ("Appendices", "The Tale of Years"),
    "FAMILY TREES": ("Appendices", "Family Trees"),
    "CALENDARS": ("Appendices", "Calendars"),
    "WRITING AND SPELLING": ("Appendices", "Writing and Spelling"),
    "THE LANGUAGES AND PEOPLES OF THE THIRD AGE": ("Appendices", "Languages and Peoples"),
    "ON TRANSLATION": ("Appendices", "On Translation"),
    "INDEXES": ("Indexes", "Indexes"),
    "POEMS AND SONGS": ("Indexes", "Poems and Songs"),
    "PERSONS, PLACES AND THINGS": ("Indexes", "Persons, Places and Things")
}

# Chapter titles for the 3 books
chapter_titles = {
    "The Fellowship of the Ring": {
        "A Long-expected Party",
        "The Shadow of the Past",
        "Three is Company",
        "A Short Cut to Mushrooms",
        "A Conspiracy Unmasked",
        "The Old Forest",
        "In the House of Tom Bombadil",
        "Fog on the Barrow-downs",
        "At the Sign of The Prancing Pony",
        "Strider",
        "A Knife in the Dark",
        "Flight to the Ford",
        "Many Meetings",
        "The Council of Elrond",
        "The Ring Goes South",
        "A Journey in the Dark",
        "The Bridge of Khazad-dûm",
        "Lothlórien",
        "The Mirror of Galadriel",
        "Farewell to Lórien",
        "The Great River",
        "The Breaking of the Fellowship"
    },
    "The Two Towers": {
        "The Departure of Boromir",
        "The Riders of Rohan",
        "The Uruk-hai",
        "Treebeard",
        "The White Rider",
        "The King of the Golden Hall",
        "Helm’s Deep",
        "The Road to Isengard",
        "Flotsam and Jetsam",
        "The Voice of Saruman",
        "The Palantír",
        "The Taming of Sméagol",
        "The Passage of the Marshes",
        "The Black Gate is Closed",
        "Of Herbs and Stewed Rabbit",
        "The Window on the West",
        "The Forbidden Pool",
        "Journey to the Cross-roads",
        "The Stairs of Cirith Ungol",
        "Shelob’s Lair",
        "The Choices of Master Samwise"
    },
    "The Return of the King": {
        "Minas Tirith",
        "The Passing of the Grey Company",
        "The Muster of Rohan",
        "The Siege of Gondor",
        "The Ride of the Rohirrim",
        "The Battle of the Pelennor Fields",
        "The Pyre of Denethor",
        "The Houses of Healing",
        "The Last Debate",
        "The Black Gate Opens",
        "The Tower of Cirith Ungol",
        "The Land of Shadow",
        "Mount Doom",
        "The Field of Cormallen",
        "The Steward and the King",
        "Many Partings",
        "Homeward Bound",
        "The Scouring of the Shire",
        "The Grey Havens"
    }
}

# Track current state
current_book = "The Fellowship of the Ring"
current_chapter = "Unknown"
final_docs = []

for doc in chunks:
    lines = doc.page_content.splitlines()
    assigned = False

    for line in lines:
        clean = line.strip()
        upper = clean.upper()

        # Detect book title
        for section_key, section_val in book_sections.items():
            if section_key in upper:
                current_book = section_val

        # Detect extra sections
        if upper in extra_sections:
            current_book, current_chapter = extra_sections[upper]
            assigned = True
            break

        # Match chapter titles from known list
        for title in chapter_titles.get(current_book, []):
            if clean.lower() == title.lower():
                current_chapter = title
                assigned = True
                break

        if assigned:
            break

    final_docs.append(Document(
        page_content=doc.page_content,
        metadata={
            "book_name": current_book,
            "chapter_name": current_chapter
        }
    ))

    print(f"📘 {current_book} | 📖 {current_chapter}")

# Save index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_docs, embeddings)
vectorstore.save_local("gandalf_index_lotr")

print("✅ Vectorstore saved as 'gandalf_index_lotr'")
