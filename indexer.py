"""Unified PDF → FAISS indexing pipeline for all three Tolkien books.

Usage:
    python indexer.py              # Index all three books
    python indexer.py --book hobbit # Index only The Hobbit
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from typing import Optional

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, FAISS_INDEX_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Text splitter (shared) ────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# ── Book metadata ─────────────────────────────────────────────────────────

# LOTR sub-book detection
LOTR_BOOK_SECTIONS: dict[str, str] = {
    "THE FELLOWSHIP OF THE RING": "The Fellowship of the Ring",
    "THE TWO TOWERS": "The Two Towers",
    "THE RETURN OF THE KING": "The Return of the King",
}

# LOTR chapter titles by sub-book
LOTR_CHAPTERS: dict[str, set[str]] = {
    "The Fellowship of the Ring": {
        "A Long-expected Party", "The Shadow of the Past", "Three is Company",
        "A Short Cut to Mushrooms", "A Conspiracy Unmasked", "The Old Forest",
        "In the House of Tom Bombadil", "Fog on the Barrow-downs",
        "At the Sign of The Prancing Pony", "Strider", "A Knife in the Dark",
        "Flight to the Ford", "Many Meetings", "The Council of Elrond",
        "The Ring Goes South", "A Journey in the Dark",
        "The Bridge of Khazad-dûm", "Lothlórien", "The Mirror of Galadriel",
        "Farewell to Lórien", "The Great River", "The Breaking of the Fellowship",
    },
    "The Two Towers": {
        "The Departure of Boromir", "The Riders of Rohan", "The Uruk-hai",
        "Treebeard", "The White Rider", "The King of the Golden Hall",
        "Helm's Deep", "The Road to Isengard", "Flotsam and Jetsam",
        "The Voice of Saruman", "The Palantír", "The Taming of Sméagol",
        "The Passage of the Marshes", "The Black Gate is Closed",
        "Of Herbs and Stewed Rabbit", "The Window on the West",
        "The Forbidden Pool", "Journey to the Cross-roads",
        "The Stairs of Cirith Ungol", "Shelob's Lair",
        "The Choices of Master Samwise",
    },
    "The Return of the King": {
        "Minas Tirith", "The Passing of the Grey Company", "The Muster of Rohan",
        "The Siege of Gondor", "The Ride of the Rohirrim",
        "The Battle of the Pelennor Fields", "The Pyre of Denethor",
        "The Houses of Healing", "The Last Debate", "The Black Gate Opens",
        "The Tower of Cirith Ungol", "The Land of Shadow", "Mount Doom",
        "The Field of Cormallen", "The Steward and the King", "Many Partings",
        "Homeward Bound", "The Scouring of the Shire", "The Grey Havens",
    },
}

# Silmarillion chapter titles
SILMARILLION_CHAPTERS: set[str] = {
    "Ainulindalë", "Valaquenta", "Akallabêth",
    "Of the Rings of Power and the Third Age",
    "Of the Beginning of Days", "Of Aulë and Yavanna",
    "Of the Coming of the Elves and the Captivity of Melkor",
    "Of Thingol and Melian", "Of Eldamar and the Princes of the Eldalië",
    "Of Fëanor and the Unchaining of Melkor",
    "Of the Silmarils and the Unrest of the Noldor",
    "Of the Darkening of Valinor", "Of the Flight of the Noldor",
    "Of the Sindar", "Of the Sun and Moon and the Hiding of Valinor",
    "Of Men", "Of the Return of the Noldor",
    "Of Beleriand and Its Realms", "Of the Noldor in Beleriand",
    "Of Maeglin", "Of the Coming of Men into the West",
    "Of the Ruin of Beleriand and the Fall of Fingolfin",
    "Of Beren and Lúthien", "Of the Fifth Battle: Nirnaeth Arnoediad",
    "Of Túrin Turambar", "Of the Ruin of Doriath",
    "Of Tuor and the Fall of Gondolin",
    "Of the Voyage of Eärendil and the War of Wrath",
}


# ── Per-book indexing functions ───────────────────────────────────────────

def _load_and_split(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    return splitter.split_documents(loader.load())


def index_hobbit(books_dir: str = "books") -> list[Document]:
    """Index The Hobbit with chapter-level metadata."""
    path = os.path.join(books_dir, "The Hobbit.pdf")
    chunks = _load_and_split(path)
    docs: list[Document] = []

    last_chapter_number = "Unknown"
    last_chapter_name = "Unknown"

    for chunk in chunks:
        lines = chunk.page_content.splitlines()
        chapter_number = "Unknown"
        chapter_name = "Unknown"

        for i, line in enumerate(lines):
            match = re.match(r"(?i)^chapter\s+([ivxlcdm\d]+)", line.strip())
            if match:
                chapter_number = f"Chapter {match.group(1).upper()}"
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.lower().startswith("chapter"):
                        chapter_name = next_line.title() if next_line.isupper() else next_line
                        break
                last_chapter_number = chapter_number
                last_chapter_name = chapter_name
                break

        if chapter_number == "Unknown":
            chapter_number = last_chapter_number
            chapter_name = last_chapter_name

        docs.append(Document(
            page_content=chunk.page_content,
            metadata={
                "book_name": "The Hobbit",
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
            },
        ))

    log.info("The Hobbit: %d chunks", len(docs))
    return docs


def index_lotr(books_dir: str = "books") -> list[Document]:
    """Index The Lord of the Rings with book/chapter metadata."""
    path = os.path.join(books_dir, "The Lord of The Rings.pdf")
    chunks = _load_and_split(path)
    docs: list[Document] = []

    current_book = "The Fellowship of the Ring"
    current_chapter = "Unknown"

    for chunk in chunks:
        lines = chunk.page_content.splitlines()

        for line in lines:
            clean = line.strip()
            upper = clean.upper()

            for key, val in LOTR_BOOK_SECTIONS.items():
                if key in upper:
                    current_book = val

            for title in LOTR_CHAPTERS.get(current_book, set()):
                if clean.lower() == title.lower():
                    current_chapter = title
                    break

        docs.append(Document(
            page_content=chunk.page_content,
            metadata={
                "book_name": current_book,
                "chapter_name": current_chapter,
            },
        ))

    log.info("The Lord of the Rings: %d chunks", len(docs))
    return docs


def index_silmarillion(books_dir: str = "books") -> list[Document]:
    """Index The Silmarillion with chapter metadata."""
    path = os.path.join(books_dir, "The Silmarillion.pdf")
    chunks = _load_and_split(path)
    docs: list[Document] = []

    current_chapter = "Unknown"

    for chunk in chunks:
        lines = chunk.page_content.splitlines()

        for line in lines:
            clean = line.strip()
            for title in SILMARILLION_CHAPTERS:
                if clean.lower() == title.lower():
                    current_chapter = title
                    break

        docs.append(Document(
            page_content=chunk.page_content,
            metadata={
                "book_name": "The Silmarillion",
                "chapter_name": current_chapter,
            },
        ))

    log.info("The Silmarillion: %d chunks", len(docs))
    return docs


# ── Main entry point ─────────────────────────────────────────────────────

BOOK_INDEXERS = {
    "hobbit": index_hobbit,
    "lotr": index_lotr,
    "silmarillion": index_silmarillion,
}


def build_index(
    books: Optional[list[str]] = None,
    books_dir: str = "books",
    output_dir: Optional[str] = None,
) -> None:
    """Build and save a FAISS vectorstore from one or more books.

    Args:
        books: List of book keys to index (default: all three).
        books_dir: Directory containing the source PDFs.
        output_dir: Where to save the FAISS index (default: config value).
    """
    output_dir = output_dir or FAISS_INDEX_DIR
    books = books or list(BOOK_INDEXERS.keys())

    all_docs: list[Document] = []
    for key in books:
        indexer = BOOK_INDEXERS.get(key)
        if indexer is None:
            log.warning("Unknown book key: %s (skipping)", key)
            continue
        all_docs.extend(indexer(books_dir))

    if not all_docs:
        log.error("No documents indexed. Check that PDFs exist in %s/", books_dir)
        return

    log.info("Total chunks: %d — building FAISS index...", len(all_docs))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(output_dir)
    log.info("Saved FAISS index to %s/", output_dir)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build the Gandalf FAISS index")
    parser.add_argument(
        "--book",
        choices=list(BOOK_INDEXERS.keys()),
        nargs="+",
        default=None,
        help="Specific book(s) to index (default: all)",
    )
    parser.add_argument(
        "--books-dir",
        default="books",
        help="Directory with source PDFs (default: books/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output index directory (default: {FAISS_INDEX_DIR})",
    )
    args = parser.parse_args()
    build_index(books=args.book, books_dir=args.books_dir, output_dir=args.output)
