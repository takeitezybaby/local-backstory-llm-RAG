# Backstory RAG Evaluation & Benchmark Tracking

This directory contains the benchmark datasets, evaluation scripts, and historical iteration logs for evaluating the **Backstory RAG** verification system using **RAGAS** and Verdict Classification Accuracy metrics.

---

## 📊 Evaluation Framework & Latest Benchmark Results

The benchmark evaluates 20 representative claims extracted across two novels (*In Search of the Castaways* by Jules Verne and *The Count of Monte Cristo* by Alexandre Dumas):
* **SUPPORT (8 Claims)**: True claims explicitly confirmed by the novels.
* **CONTRADICT (6 Claims)**: False claims containing contradictory details (wrong characters, wrong locations, inverted plot points).
* **NOT MENTIONED (6 Claims)**: Plausible backstory claims unmentioned in the text.

### Latest Scorecard Summary

| Metric / Iteration | Baseline | Iteration 1 (Retrieval Fixes) | Iteration 2 (Prompt Fixes) | **Iteration 3 (Entity & Normalization)** |
|---|---|---|---|---|
| **RAGAS Answer Relevancy** | N/A | `0.5871` | `0.6247` | **`0.6419`** |
| **`SUPPORT` Verdict Recall** | `12.50%` | `75.00%` | `87.50%` | **`87.50%`** |
| **RAGAS Context Recall** | `0.0000` | `0.5000` | `0.6875` | **`0.5714`** |
| **Overall Verdict Accuracy** | `0.00%` (Raw) | `35.00%` | `40.00%` | **`40.00%`** |

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

---

### 🔹 Iteration 3: Entity Normalization & Robust Substring Aggregation
* **Changes Made**:
  1. **Entity Punctuation & Possessive Stripping (`Pipeline/querySearch.py`)**: Updated `extract_entity()` to strip possessive apostrophes (`'s`, `'`) and trailing punctuation.
  2. **Enhanced Contradiction Instructions (`Pipeline/verfication.py`)**: Added explicit prompt instructions for conflicting roles and attributes (e.g. calling a cousin a "captain" or a fisherman a "wealthy merchant").
  3. **Robust Verdict Substring Parsing (`Pipeline/aggregation.py`)**: Replaced exact equality checks with substring matching (`'SUPPORT' in result`, `'CONTRADICT' in result`) to handle LLM formatting variations.
* **Impact**:
  * **RAGAS Answer Relevancy** increased to **`0.6419` (64.19%)**.
  * **$100\%$ of claims** (including possessive entity names) trigger **Entity-restricted search**.

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
