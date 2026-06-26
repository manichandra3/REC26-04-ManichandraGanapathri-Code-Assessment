"""Step 4: Automated groundedness evaluation — checks for hallucinations."""

import os
import sys
import json
import time

from dotenv import load_dotenv
from google import genai
import chromadb

from query import get_api_key, retrieve_context, build_prompt, MODEL_NAME, COLLECTION_NAME, REFUSAL

load_dotenv()


def evaluate_groundedness(question: str, answer: str, context: str, client, generate=None) -> dict:
    eval_prompt = (
        "You are a strict groundedness judge for a RAG system. Your task is to determine "
        "whether the given ANSWER is fully supported by the provided CONTEXT.\n\n"
        "Rules:\n"
        "- Output exactly one word: GROUNDED or NOT GROUNDED\n"
        "- GROUNDED means every factual claim in the answer is present in or directly "
        "inferable from the context.\n"
        "- NOT GROUNDED means the answer contains information NOT found in the context "
        "(hallucination) or contradicts the context.\n"
        "- If the answer says 'I do not know based on the corpus', it is always GROUNDED.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "JUDGMENT (GROUNDED or NOT GROUNDED):"
    )

    # Use the rate-limit-aware caller when provided (falls back to a plain call).
    if generate is not None:
        response = generate(client, eval_prompt)
    else:
        response = client.models.generate_content(model=MODEL_NAME, contents=eval_prompt)
    verdict = response.text.strip().upper()

    grounded = "GROUNDED" in verdict and "NOT GROUNDED" not in verdict

    return {
        "question": question,
        "answer": answer,
        "verdict": "GROUNDED" if grounded else "NOT GROUNDED",
    }


def generate_with_retry(client, prompt, model=MODEL_NAME, max_retries=6):
    # Retry transient failures: 429 (rate limit) and 503 (server overload).
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            msg = str(e)
            transient = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "503" in msg or "UNAVAILABLE" in msg
            if transient and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Transient error (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exhausted")


def main():
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    judge_client = genai.Client(api_key=api_key)

    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # In-domain questions (answerable from the Regulations for the Navy corpus)
    # plus deliberately out-of-domain "trick" questions that MUST be refused.
    # The judge below checks groundedness, so a correct refusal counts as GROUNDED.
    test_questions = [
        # --- in-domain ---
        "What is the procedure for summary punishment in the Navy?",
        "Who has the authority to convene a court martial?",
        "What are the regulations regarding arrest and naval custody?",
        "How is a Board of Inquiry constituted?",
        "What are the responsibilities of a Commanding Officer under these regulations?",
        # --- out-of-domain / unanswerable (must be refused) ---
        "What is the capital of France?",
        "What is the current price of Bitcoin?",
        "What is the prize money for winning a video game tournament?",
    ]

    results = []
    for i, q in enumerate(test_questions):
        print(f"Q: {q}")
        context, sources = retrieve_context(collection, q)
        prompt = build_prompt(context, q)

        # One question's API failure must not kill the whole evaluation run.
        try:
            time.sleep(6)
            response = generate_with_retry(client, prompt)
            answer = response.text.strip()

            time.sleep(6)
            eval_result = evaluate_groundedness(q, answer, context, judge_client, generate=generate_with_retry)
        except Exception as e:
            print(f"  ERROR after retries: {str(e)[:120]}")
            results.append({"question": q, "answer": None, "verdict": "ERROR", "sources": sources, "refused": False})
            print()
            continue

        eval_result["sources"] = sources
        # Did the model correctly refuse? (used for the unanswerable-handling metric)
        eval_result["refused"] = REFUSAL.lower() in answer.lower()
        results.append(eval_result)

        status = "PASS" if eval_result["verdict"] == "GROUNDED" else "FAIL"
        print(f"  A: {answer[:120]}...")
        print(f"  Verdict: {eval_result['verdict']} [{status}]  Refused: {eval_result['refused']}")
        print()

    grounded_count = sum(1 for r in results if r["verdict"] == "GROUNDED")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")
    total = len(results)
    scored = total - errors  # questions that actually produced a verdict
    accuracy = (grounded_count / scored * 100) if scored else 0.0

    summary = {
        "total_questions": total,
        "scored": scored,
        "grounded": grounded_count,
        "not_grounded": scored - grounded_count,
        "errors": errors,
        "grounding_accuracy_pct": round(accuracy, 1),
        "refused_count": sum(1 for r in results if r.get("refused")),
    }
    results.append(summary)

    print(f"{'='*50}")
    print(f"Groundedness: {grounded_count}/{scored} grounded ({accuracy:.1f}%)" + (f"  [{errors} API error(s) skipped]" if errors else ""))
    print(f"Refusals (unanswerable handling): {summary['refused_count']} question(s) refused")
    print(f"{'='*50}")

    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    return accuracy


if __name__ == "__main__":
    main()
