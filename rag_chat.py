"""
RAG from Scratch — Chat with an EXISTING ChromaDB Collection
---------------------------------------------------------------
Connects to a ChromaDB collection that was already populated (e.g. by
rag_terminal.py or the notebook) and lets you ask questions about it.
This script does NOT read any PDF and does NOT create or store embeddings —
it only queries what's already in the vector database.

Setup:
    1. Install dependencies:
       pip install sentence-transformers chromadb groq python-dotenv

    2. Create a file named .env in the same folder as this script with:
       GROQ_API_KEY="your_key_here"

    3. Make sure ./my_chromadb (or your --db_path) already exists and has
       data in it — run the ingestion script/notebook first if it doesn't.

Run:
    python rag_chat.py
    python rag_chat.py --db_path ./my_chromadb --collection finance --top_k 5

Then type questions at the prompt. Type 'exit' or 'quit' to stop.
"""

import os
import sys
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


def retrieve_docs(query, collection, embed_model, top_k=3):
    """Embed the query and return the top_k most similar stored chunks, joined together."""
    q_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=top_k)
    return "\n\n".join(results["documents"][0])


def generate_answer(context, query, groq_client):
    """Send the retrieved context + question to the LLM and return its answer."""
    chat_completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise technical assistant. Answer questions using only the provided context.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}",
            },
        ],
    )
    return chat_completion.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Chat with an existing ChromaDB collection.")
    parser.add_argument("--db_path", default="./my_chromadb", help="Path to the existing ChromaDB folder.")
    parser.add_argument("--collection", default="finance", help="Name of the existing collection to query.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve per question.")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Error: no ChromaDB folder found at '{args.db_path}'. "
              f"Run the ingestion script/notebook first to create and populate it.")
        sys.exit(1)

    # --- Load API key ---
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=\"your_key_here\".")
        sys.exit(1)
    groq_client = Groq(api_key=api_key)

    # --- Connect to the existing vector database ---
    chroma_client = chromadb.PersistentClient(path=args.db_path)
    try:
        collection = chroma_client.get_collection(name=args.collection)
    except Exception:
        print(f"Error: collection '{args.collection}' not found in '{args.db_path}'.")
        sys.exit(1)

    if collection.count() == 0:
        print(f"Error: collection '{args.collection}' is empty. Ingest documents first.")
        sys.exit(1)

    print(f"Connected to collection '{args.collection}' ({collection.count()} chunks stored).")

    # --- Load embedding model (needed only to embed your questions) ---
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Interactive Q&A loop ---
    print("\nReady! Ask a question about the document (type 'exit' or 'quit' to stop).\n")

    while True:
        query = input("Q: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            continue

        context = retrieve_docs(query, collection, embed_model, top_k=args.top_k)
        answer = generate_answer(context, query, groq_client)

        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
