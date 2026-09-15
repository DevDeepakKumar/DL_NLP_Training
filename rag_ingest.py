"""
RAG from Scratch — Ingest a PDF into ChromaDB
-------------------------------------------------
Reads a PDF page by page, embeds each page, and stores the embeddings in a
local ChromaDB collection. This script only INGESTS — it does not answer
any questions. Once it finishes, use rag_chat.py to ask questions about
the document.

Setup:
    pip install pdfplumber sentence-transformers chromadb

Run:
    python rag_ingest.py --pdf my_report.pdf
    python rag_ingest.py --pdf my_report.pdf --db_path ./my_chromadb --collection finance

Re-running this script on the same PDF is safe — pages already stored are
skipped, so you won't get duplicates or errors.
"""

import os
import sys
import argparse

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_pdf(pdf_path, collection, embed_model):
    """Read the PDF page by page, embed each page, and store it in ChromaDB.
    Skips pages that were already ingested in a previous run."""

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        added = 0
        skipped_empty = 0
        skipped_existing = 0

        for page_number, page in enumerate(pdf.pages, start=1):
            page_id = f"{os.path.basename(pdf_path)}_page_{page_number}"

            if page_id in existing_ids:
                skipped_existing += 1
                continue

            text = page.extract_text()
            if not text or not text.strip():
                skipped_empty += 1
                continue
            text = text.strip()

            embedding = embed_model.encode(text, normalize_embeddings=True).tolist()

            collection.add(
                ids=[page_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"page": page_number, "source": pdf_path}],
            )
            added += 1

    print(f"\nDone with '{pdf_path}':")
    print(f"  Total pages in PDF:        {total_pages}")
    print(f"  New pages added:           {added}")
    print(f"  Already ingested (skipped):{skipped_existing:>4}")
    print(f"  Empty/unreadable (skipped):{skipped_empty:>4}")
    print(f"  Total chunks now in collection: {collection.count()}")


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB for later Q&A.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest.")
    parser.add_argument("--db_path", default="./my_chromadb", help="Where to store/find the ChromaDB folder.")
    parser.add_argument("--collection", default="finance", help="Name of the collection to store chunks in.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found at '{args.pdf}'.")
        sys.exit(1)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=args.db_path)
    collection = chroma_client.get_or_create_collection(name=args.collection, embedding_function=None)

    print(f"Ingesting '{args.pdf}' into collection '{args.collection}' at '{args.db_path}'...")
    ingest_pdf(args.pdf, collection, embed_model)

    print("\nIngestion finished. You can now run rag_chat.py to ask questions about this document.")


if __name__ == "__main__":
    main()
