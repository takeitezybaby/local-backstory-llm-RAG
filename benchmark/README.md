# Backstory RAG Evaluation & Benchmark Tracking

This directory contains the benchmark datasets, evaluation scripts, and historical iteration logs for evaluating the **Backstory RAG** verification system using **RAGAS** and Verdict Classification Accuracy metrics.

---

## 📊 Evaluation Framework Overview

The benchmark evaluates 20 representative claims extracted across two novels (*In Search of the Castaways* by Jules Verne and *The Count of Monte Cristo* by Alexandre Dumas):
* **SUPPORT (8 Claims)**: True claims explicitly confirmed by the novels.
* **CONTRADICT (6 Claims)**: False claims containing contradictory details (wrong characters, wrong locations, inverted plot points).
* **NOT MENTIONED (6 Claims)**: Plausible backstory claims unmentioned in the text.

### Metrics Tracked:
1. **Verdict Classification Accuracy**: Overall accuracy, Precision, and Recall for `SUPPORT`, `CONTRADICT`, and `NOT MENTIONED`.
2. **RAGAS Metrics**:
   * **Context Recall**: Measure of whether all required evidence was retrieved.
   * **Answer Relevancy**: Measure of how directly the verification output addresses the backstory claim.
   * **Faithfulness**: Measure of whether the generated explanation relies solely on retrieved source text.

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
  * Improved overall verdict classification accuracy to **40%+**.
  * RAGAS `Answer Relevancy` achieved **0.5871** and `Context Recall` achieved **0.5000**.

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
