# Plain-Vanilla RAG System

A minimal, framework-free Retrieval-Augmented Generation pipeline over a policy
corpus — the **Regulations for the Navy, Part I & Part II** (~675 pages of
administrative, disciplinary, and statutory regulations). No LangChain /
LlamaIndex; just `pypdf`, `chromadb`, and the Gemini API.

---

## 1. How to Run

### Setup
```bash
python3 -m venv venv
source venv/bin/activate            # Linux/Mac
# .\venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt

cp .env.example .env                # then edit .env and paste your key
# .env must contain: GEMINI_API_KEY=...   (free key: https://aistudio.google.com/apikey)
```

### Step 1 — Ingest / Index the corpus
Parses every PDF in `corpus/`, chunks it, embeds the chunks, and stores them in a
local ChromaDB at `./chroma_db`. Re-running is idempotent (it rebuilds cleanly).
```bash
python ingest.py
# -> Ingested 1918 chunks from 2 file(s) into 'navy_regs'.
```

### Step 2 — Query
```bash
python query.py "Who has the authority to convene a court martial?"
```
Output is the grounded answer followed by `[CITED SOURCES: ...]`.

### Step 3 — Evaluate
```bash
python evaluate.py
```
Runs the LLM-as-judge groundedness metric over the sample question set and writes
`eval_results.json`.

---

## 2. Pipeline Architecture & Key Choices

| Component | Choice | Why |
|---|---|---|
| **PDF parsing** | `pypdf` | Pure-Python, no system deps; handles the multi-hundred-page regulation PDFs. |
| **Chunking** | Character sliding-window, **1000 chars / 200 overlap** | Regulations are written as numbered clauses; ~1000 chars keeps a clause (or a few) intact, and the 200-char overlap stops an answer from being split across a chunk boundary. Simple and deterministic — no tokenizer dependency. |
| **Embeddings** | ChromaDB default **all-MiniLM-L6-v2** (ONNX, runs locally) | Zero extra cost / no API round-trips for embedding, good enough semantic recall for this corpus, ships with Chroma. |
| **Vector store** | **ChromaDB** persistent client (`./chroma_db`) | Pure Python, no Docker/cloud/server. Persists to disk so ingest runs once. |
| **Retrieval** | Top-**k = 5** (`TOP_K` env-configurable) | k=5 gives the LLM enough surrounding clauses to answer without flooding the context with noise that dilutes grounding. |
| **LLM** | `gemini-2.5-flash-lite` (`GEMINI_MODEL` env-configurable) | Fast, free-tier-friendly, generous rate limits. Model ID is **not hardcoded** — overridable via env so the pipeline survives model deprecations. |
| **Orchestration** | Native Python (3 scripts) | Transparent and auditable for a 75-minute build — no framework abstraction to reason about. |

**Prompt design** (`build_prompt` in `query.py`): a strict system instruction that
(a) restricts the model to the provided context block, (b) forbids outside
knowledge and guessing, (c) mandates an exact refusal string when the answer is
absent, and (d) requires citing the source file. Each retrieved chunk is injected
with a `[Source: <file> | id: <chunk_id>]` header so the model can attribute its
answer.

---

## 3. Unanswerable Questions

Handling out-of-corpus questions is enforced at two layers:

1. **Prompt-level refusal.** The system prompt instructs the model: *“If the answer
   is not explicitly stated in the provided context, reply with exactly
   `I do not know based on the corpus` and nothing else. Do not use outside
   knowledge and do not guess.”* Vector search **always** returns the top-k chunks
   (even for an irrelevant question), so the LLM — not the retriever — is the gate
   that recognizes the chunks don't contain the answer and refuses.
2. **Citation suppression.** When the answer is a refusal, `query.py` does **not**
   print misleading source citations; it prints
   `[CITED SOURCES: none — answer not found in corpus]` instead.

Verified live — e.g. *“What is the current price of Bitcoin?”* and *“What is the
capital of France?”* both return `I do not know based on the corpus` with no
citation, while in-domain questions return a grounded, cited answer.

---

## 4. Evaluation Notes

**Metrics proposed:**
- Retrieval precision/recall (needs a labelled relevant-chunk set — not built, as
  the corpus ships without gold labels).
- Context relevance (LLM scoring of retrieved chunks vs. question).
- **LLM-as-judge groundedness** — *implemented*.
- **Unanswerable-refusal rate** on out-of-domain questions — *implemented*.

**Implemented metric (`evaluate.py`):** for each sample question the pipeline
retrieves context, generates an answer, then a **separate judge LLM call**
classifies the answer as `GROUNDED` / `NOT GROUNDED` (a correct refusal counts as
GROUNDED). It also records whether each out-of-domain question was correctly
**refused**. Results are written to `eval_results.json`.

The sample set mixes **5 in-domain** Navy-regulation questions with **3
out-of-domain** trick questions that must be refused.

**Results** (live run, `gemini-2.5-flash-lite`):

<!-- EVAL_RESULTS -->

---

## 5. Future Improvements

Given more than 75 minutes:
- **Page-aware chunking & metadata**: store page numbers / chapter / clause numbers
  so citations point to an exact regulation (e.g. *“RegsNavyII.pdf, Ch. V, §0512”*)
  rather than just the file.
- **Token-based chunking** (e.g. `tiktoken`) instead of raw characters, with
  recursive splitting on clause boundaries.
- **Hybrid retrieval**: combine dense (MiniLM) with BM25/keyword search and a
  **re-ranker** (cross-encoder) — important for a legalistic corpus full of exact
  terms ("court martial", section numbers).
- **Labelled eval set** with gold relevant chunks to compute real retrieval
  precision/recall@k, plus answer-correctness scoring (not just groundedness).
- **Query expansion / multi-query retrieval** to improve recall on paraphrased
  questions.
- **Streaming responses** and a thin web UI / API instead of a CLI.
- **Caching & batching** of embeddings and judge calls to cut cost and avoid free-tier
  rate limits.
