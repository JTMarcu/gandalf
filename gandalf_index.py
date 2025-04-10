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
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
final_docs = []

### --- The Hobbit ---
hobbit_path = os.path.join("books", "The Hobbit.pdf")
hobbit_loader = PyPDFLoader(hobbit_path)
hobbit_docs = hobbit_loader.load()
hobbit_chunks = text_splitter.split_documents(hobbit_docs)

hobbit_chapter_number = "Unknown"
hobbit_chapter_name = "Unknown"
hobbit_book_name = "The Hobbit"

for doc in hobbit_chunks:
    text = doc.page_content
    lines = text.splitlines()

    chapter_number = "Unknown"
    chapter_name = "Unknown"

    for i, line in enumerate(lines):
        clean = line.strip()
        match = re.match(r"(?i)^chapter\s+([ivxlcdm\d]+)", clean)
        if match:
            chapter_number = f"Chapter {match.group(1).upper()}"
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.lower().startswith("chapter"):
                    chapter_name = next_line.title() if next_line.isupper() else next_line
                    break
            hobbit_chapter_number = chapter_number
            hobbit_chapter_name = chapter_name
            break

    if chapter_number == "Unknown":
        chapter_number = hobbit_chapter_number
        chapter_name = hobbit_chapter_name

    final_docs.append(Document(
        page_content=text,
        metadata={
            "book_name": hobbit_book_name,
            "chapter_number": chapter_number,
            "chapter_name": chapter_name
        }
    ))


### --- The Lord of the Rings ---
lotr_path = os.path.join("books", "The Lord of the Rings.pdf")
lotr_loader = PyPDFLoader(lotr_path)
lotr_docs = lotr_loader.load()
lotr_chunks = text_splitter.split_documents(lotr_docs)

book_sections = {
    "THE FELLOWSHIP OF THE RING": "The Fellowship of the Ring",
    "THE TWO TOWERS": "The Two Towers",
    "THE RETURN OF THE KING": "The Return of the King"
}

chapter_titles_by_book = {
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

current_lotr_book = "The Fellowship of the Ring"
current_lotr_chapter = "Unknown"

for doc in lotr_chunks:
    lines = doc.page_content.splitlines()
    assigned = False

    for line in lines:
        clean = line.strip()
        upper = clean.upper()

        for section_key, section_val in book_sections.items():
            if section_key in upper:
                current_lotr_book = section_val

        for title in chapter_titles_by_book.get(current_lotr_book, []):
            if clean.lower() == title.lower():
                current_lotr_chapter = title
                assigned = True
                break

        if assigned:
            break

    final_docs.append(Document(
        page_content=doc.page_content,
        metadata={
            "book_name": current_lotr_book,
            "chapter_name": current_lotr_chapter
        }
    ))


### --- The Silmarillion ---
sil_path = os.path.join("books", "The Silmarillion.pdf")
sil_loader = PyPDFLoader(sil_path)
sil_docs = sil_loader.load()
sil_chunks = text_splitter.split_documents(sil_docs)

silmarillion_book = "The Silmarillion"
silmarillion_chapter = "Unknown"
chapter_titles_sil = {
    "Ainulindalë", "Valaquenta", "Akallabêth", "Of the Rings of Power and the Third Age",
    "Of the Beginning of Days", "Of Aulë and Yavanna", "Of the Coming of the Elves and the Captivity of Melkor",
    "Of Thingol and Melian", "Of Eldamar and the Princes of the Eldalië", "Of Fëanor and the Unchaining of Melkor",
    "Of the Silmarils and the Unrest of the Noldor", "Of the Darkening of Valinor", "Of the Flight of the Noldor",
    "Of the Sindar", "Of the Sun and Moon and the Hiding of Valinor", "Of Men", "Of the Return of the Noldor",
    "Of Beleriand and Its Realms", "Of the Noldor in Beleriand", "Of Maeglin", "Of the Coming of Men into the West",
    "Of the Ruin of Beleriand and the Fall of Fingolfin", "Of Beren and Lúthien", "Of the Fifth Battle: Nirnaeth Arnoediad",
    "Of Túrin Turambar", "Of the Ruin of Doriath", "Of Tuor and the Fall of Gondolin",
    "Of the Voyage of Eärendil and the War of Wrath"
}

for doc in sil_chunks:
    lines = doc.page_content.splitlines()
    assigned = False

    for line in lines:
        clean = line.strip()
        for title in chapter_titles_sil:
            if clean.lower() == title.lower():
                silmarillion_chapter = title
                assigned = True
                break
        if assigned:
            break

    final_docs.append(Document(
        page_content=doc.page_content,
        metadata={
            "book_name": silmarillion_book,
            "chapter_name": silmarillion_chapter
        }
    ))


### --- Save FAISS vectorstore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(final_docs, embeddings)
vectorstore.save_local("gandalf_index")

print("✅ All three books indexed and saved as 'gandalf_index'")
