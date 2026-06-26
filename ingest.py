"""Step 2: Data ingestion — PDF parsing, chunking, and ChromaDB population."""

import os
import glob

from pypdf import PdfReader
import chromadb


def parse_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest_corpus(corpus_dir: str, db_dir: str, collection_name: str = "navy_regs"):
    chroma_client = chromadb.PersistentClient(path=db_dir)
    # Drop any stale collection so re-ingesting is idempotent (no duplicate chunks).
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=collection_name)

    pdf_files = [
        f for f in glob.glob(os.path.join(corpus_dir, "**/*.pdf"), recursive=True)
        if not f.endswith(":Zone.Identifier")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {corpus_dir}")

    all_ids, all_docs, all_metadatas = [], [], []
    doc_id = 0

    for pdf_path in sorted(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"  Parsing: {filename}")
        raw_text = parse_pdf(pdf_path)
        chunks = chunk_text(raw_text)
        print(f"    -> {len(chunks)} chunks")

        for chunk in chunks:
            all_ids.append(f"chunk_{doc_id}")
            all_docs.append(chunk)
            all_metadatas.append({"source": filename})
            doc_id += 1

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
    print(f"\nIngested {len(all_docs)} chunks from {len(pdf_files)} file(s) into '{collection_name}'.")
    return collection


if __name__ == "__main__":
    CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
    DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(DB_DIR, exist_ok=True)

    ingest_corpus(CORPUS_DIR, DB_DIR)
