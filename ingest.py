"""
ingest.py — Document ingestion pipeline for the RAG system.

What this script does (the RAG "indexing" phase):
  1. Loads documents from ./documents/ (.txt, .pdf, .docx)
  2. Splits them into overlapping chunks so context is preserved
  3. Embeds each chunk into a vector (a list of numbers capturing meaning)
  4. Stores those vectors in ChromaDB so query.py can search them later

Interview tip: This runs ONCE (or whenever you add new documents).
query.py reads from the stored vectors — it does NOT re-embed every time.
"""

import os
import sys
from pathlib import Path

# LangChain document loaders — each handles a specific file type
from langchain_community.document_loaders import (
    TextLoader,        # handles .txt files
    PyPDFLoader,       # handles .pdf files (uses pypdf under the hood)
    Docx2txtLoader,    # handles .docx files (uses python-docx under the hood)
)

# RecursiveCharacterTextSplitter: tries to split on paragraphs → sentences →
# words → characters, in that order, so chunks stay semantically meaningful.
# In LangChain 1.x this lives in its own package: langchain-text-splitters.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OllamaEmbeddings: sends text to the locally running Ollama server and gets
# back a vector (embedding). No internet required — it runs on your machine.
from langchain_ollama import OllamaEmbeddings

# Chroma: the vector store. "Persistent" means it saves to disk so vectors
# survive between runs. Without persistence you'd re-embed everything each time.
from langchain_chroma import Chroma


# ── Configuration ─────────────────────────────────────────────────────────────

DOCUMENTS_DIR = Path("./documents")   # where your source files live
CHROMA_DIR    = Path("./chroma_db")   # where ChromaDB will persist vectors
COLLECTION    = "rag_capstone"        # logical name for this set of documents

# Embedding model running locally via Ollama.
# nomic-embed-text is optimised for retrieval tasks — it turns text into a
# 768-dimensional vector that captures semantic meaning.
EMBED_MODEL = "nomic-embed-text"

# Chunk size: how many characters per chunk.
# 500 is a sweet spot — large enough to hold a complete idea, small enough
# that the embedding captures that one idea well (not a mix of many topics).
CHUNK_SIZE = 500

# Overlap: how many characters the next chunk shares with the previous one.
# 75-character overlap means a sentence that straddles a chunk boundary
# will appear in both chunks, so we never lose context at the seams.
CHUNK_OVERLAP = 75


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_documents(directory: Path) -> list:
    """
    Walk the documents directory and load every supported file.

    Returns a list of LangChain Document objects. Each Document has:
      - page_content: the raw text
      - metadata: dict with at least {"source": "<file path>"}

    Supported formats: .txt, .pdf, .docx
    """
    loaders = {
        ".txt":  TextLoader,
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

    all_docs = []
    files_found = list(directory.iterdir()) if directory.exists() else []

    if not files_found:
        print(f"  [!] No files found in {directory}. Add documents and re-run.")
        sys.exit(1)

    for file_path in sorted(files_found):
        suffix = file_path.suffix.lower()
        if suffix not in loaders:
            print(f"  [skip] {file_path.name} — unsupported type '{suffix}'")
            continue

        print(f"  [load] {file_path.name} ...")
        try:
            loader = loaders[suffix](str(file_path))
            docs = loader.load()
            print(f"         → {len(docs)} page(s) loaded")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  [ERROR] Could not load {file_path.name}: {e}")

    return all_docs


def split_documents(docs: list) -> list:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Why overlap? Imagine a key sentence falls right at a chunk boundary.
    Without overlap, neither chunk has the full context. With 75-character
    overlap, that sentence appears (partially) in both chunks, so retrieval
    can still find it.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # length_function: how to measure "size". len() counts characters.
        # Some setups use a token-counting function here instead.
        length_function=len,
        # separators tried in order — paragraph breaks first, then sentence
        # breaks, then word breaks, then raw character splits as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    return chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  RAG INGEST PIPELINE")
    print("="*60)

    # Step 1: Load documents
    print("\n[Step 1/4] Loading documents from ./documents/ ...")
    docs = load_documents(DOCUMENTS_DIR)
    print(f"  ✓ Loaded {len(docs)} document page(s) total")

    # Step 2: Split into chunks
    print("\n[Step 2/4] Splitting into chunks ...")
    print(f"  chunk_size={CHUNK_SIZE} chars, chunk_overlap={CHUNK_OVERLAP} chars")
    chunks = split_documents(docs)
    print(f"  ✓ Created {len(chunks)} chunks from {len(docs)} page(s)")

    # Show a preview of the first chunk so you can see what's stored
    if chunks:
        preview = chunks[0].page_content[:200].replace("\n", " ")
        print(f"\n  Preview of chunk #1:\n  \"{preview}...\"")

    # Step 3: Set up the embedding model
    print("\n[Step 3/4] Connecting to Ollama embedding model ...")
    print(f"  Model: {EMBED_MODEL} (running locally — no internet needed)")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    print("  ✓ Embedding model ready")

    # Step 4: Store in ChromaDB
    # Chroma.from_documents() does two things in one call:
    #   a) Calls the embedding model on every chunk to get its vector
    #   b) Saves both the vector and the original text to disk
    print(f"\n[Step 4/4] Embedding chunks and storing in ChromaDB ...")
    print(f"  Destination: {CHROMA_DIR}/")
    print(f"  Collection:  {COLLECTION}")
    print(f"  (Embedding {len(chunks)} chunks via Ollama — this may take a moment...)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )

    print(f"  ✓ Stored {len(chunks)} vectors in ChromaDB")

    print("\n" + "="*60)
    print("  INGEST COMPLETE")
    print(f"  {len(chunks)} chunks ready for querying.")
    print(f"  Run:  python query.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
