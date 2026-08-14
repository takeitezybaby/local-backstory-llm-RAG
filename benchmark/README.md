# Backstory RAG Evaluation & Benchmark Tracking

This directory contains the benchmark datasets, evaluation scripts, and historical iteration logs for evaluating the **Backstory RAG** verification system using **RAGAS** and Verdict Classification Accuracy metrics.

---

## 📊 Evaluation Framework & Latest Benchmark Results

The benchmark evaluates 20 representative claims extracted across two novels (*In Search of the Castaways* by Jules Verne and *The Count of Monte Cristo* by Alexandre Dumas):
* **SUPPORT (8 Claims)**: True claims explicitly confirmed by the novels.
* **CONTRADICT (6 Claims)**: False claims containing contradictory details (wrong characters, wrong locations, inverted plot points).
* **NOT MENTIONED (6 Claims)**: Plausible backstory claims unmentioned in the text.

### Latest Scorecard Summary

| Metric / Iteration | Baseline | Iteration 1 (Retrieval Fixes) | Iteration 2 (Prompt & Logic Fixes) |
|---|---|---|---|
| **RAGAS Context Recall** | `0.0000` | `0.5000` | **`0.6875`** |
| **RAGAS Answer Relevancy** | N/A | `0.5871` | **`0.6247`** |
| **`SUPPORT` Verdict Recall** | `12.50%` | `75.00%` | **`87.50%`** |
| **Overall Verdict Accuracy** | `0.00%` (Raw) | `35.00%` | **`40.00%`** |

---

## 📈 Updation Track & Iteration Log

### 🔹 Baseline Evaluation
* **Overall Accuracy**: `0.00%` (Raw) / `35.00%` (Mapped)
* **Key Failures Identified**:
  * `claimExtraction.py`: Single-clause backstory sentences were being split into arrays of individual characters due to a return type mismatch in `compound_clauses()`. This caused `0` claims to be extracted for 90% of test inputs.
  * `claimRetrieval.py`: `filterByEntity()` performed exact verbatim string matching on full entity names, discarding valid evidence chunks if the novel used short names (e.g. "Tom" instead of "Tom Austin").
  * `verfication.py`: Prompt instructions lacked explicit guidelines on what constituted a `CONTRADICTION`, causing the LLM to output `NOT MENTIONED` or `SUPPORT` for false statements.

---

### 🔹 Iteration 1: Retrieval & Extraction Core Fixes
* **Changes Made**:
  1. **Fixed Character-Splitting Bug (`Pipeline/claimExtraction.py`)**: Modified `compound_clauses()` to return `[sentence]` as a list instead of a raw string when verb count < 2.
  2. **Token-Based Entity Filtering & Fallback (`Pipeline/claimRetrieval.py`)**: Replaced strict exact string matching with token-level matching and added fallback to top vector search results if entity filter returned 0 results.
* **Impact**:
  * Evidence retrieval success rate increased from **10% to 100%**.
  * `SUPPORT` verdict recall jumped from **12.5% to 75.0%**.

---

### 🔹 Iteration 2: Verification Prompt & Indexing Optimization
* **Changes Made**:
  1. **Fixed Evidence Indexing (`Pipeline/verfication.py`)**: Corrected string formatting from `1+1` to `i+1` so evidence items are numbered sequentially (`Evidence 1`, `Evidence 2`, etc.).
  2. **Refined LLM Verification Prompt (`Pipeline/verfication.py`)**:
     * Added explicit rules defining `CONTRADICT` (when evidence refutes the claim or provides conflicting roles/locations/outcomes).
     * Clarified `NOT MENTIONED` vs `CONTRADICT` boundaries.
  3. **Verdict Normalization (`Pipeline/ragas_evaluator.py`)**: Mapped internal system verdicts (`COMPATIBLE` -> `SUPPORT`, `INCOMPATIBLE` -> `CONTRADICT`, `NO CONTRADICTION, BUT NOT SUPPORTED` -> `NOT MENTIONED`) for standardized benchmark scoring.
* **Impact**:
  * RAGAS `Context Recall` increased to **0.6875** (68.75%).
  * RAGAS `Answer Relevancy` increased to **0.6247** (62.47%).
  * `SUPPORT` Verdict Recall reached **87.50%**.

---

## 🛠️ How to Run the Benchmark

To re-run the trace collection and evaluation metrics:

```cmd
run_evaluation.bat
```

Or execute manually via Python:

```bash
# 1. Run pipeline trace collection
python Pipeline/eval_runner.py

# 2. Run RAGAS & accuracy evaluator
python Pipeline/ragas_evaluator.py
```
