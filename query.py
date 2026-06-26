"""Step 3: Retrieval & Q&A — vector search + Gemini generation with strict grounding."""

import os
import sys

from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

# Model is configurable via env so the pipeline isn't pinned to one (possibly
# deprecated) ID. Default is a current, free-tier-friendly Gemini model.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
TOP_K = int(os.environ.get("TOP_K", "5"))
COLLECTION_NAME = "navy_regs"

# Phrase the model must emit when the context does not contain the answer.
REFUSAL = "I do not know based on the corpus"


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: Set GEMINI_API_KEY (e.g. in a .env file — see .env.example).", file=sys.stderr)
        sys.exit(1)
    return key


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a strict Q&A assistant for the Regulations for the Navy corpus. "
        "Answer using ONLY the context block below. "
        f"If the answer is not explicitly stated in the provided context, reply with exactly '{REFUSAL}' "
        "and nothing else. Do not use outside knowledge and do not guess. "
        "When you do answer, cite the source file(s) named in the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def retrieve_context(collection, question: str) -> tuple[str, str]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found.", "Unknown"

    docs = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
    ids = results["ids"][0] if results.get("ids") else [f"chunk_{i}" for i in range(len(docs))]
    sources = set()

    context_parts = []
    for doc, meta, cid in zip(docs, metadatas, ids):
        src = meta.get("source", "Unknown") if meta else "Unknown"
        sources.add(src)
        context_parts.append(f"[Source: {src} | id: {cid}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    source_str = ", ".join(sorted(sources))
    return context, source_str


def answer_question(question: str, client) -> str:
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    context, sources = retrieve_context(collection, question)
    prompt = build_prompt(context, question)

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    answer = response.text.strip()

    # Don't print misleading citations when the model refused to answer.
    if REFUSAL.lower() in answer.lower():
        return f"{answer}\n\n[CITED SOURCES: none — answer not found in corpus]"
    return f"{answer}\n\n[CITED SOURCES: {sources}]"


if __name__ == "__main__":
    client = genai.Client(api_key=get_api_key())

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    print("\n" + answer_question(question, client))
