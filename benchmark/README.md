# Backstory RAG Evaluation & Benchmark Tracking

This directory contains the benchmark datasets, evaluation scripts, and historical iteration logs for evaluating the **Backstory RAG** verification system using **RAGAS** and Verdict Classification Accuracy metrics.

---

## 📊 Evaluation Framework & Latest Benchmark Results

The benchmark evaluates 20 representative claims extracted across two novels (*In Search of the Castaways* by Jules Verne and *The Count of Monte Cristo* by Alexandre Dumas):
* **SUPPORT (8 Claims)**: True claims explicitly confirmed by the novels.
* **CONTRADICT (6 Claims)**: False claims containing contradictory details (wrong characters, wrong locations, inverted plot points).
* **NOT MENTIONED (6 Claims)**: Plausible backstory claims unmentioned in the text.

### Benchmark Scorecard Progression

| Metric / Iteration | Baseline | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 | **Iteration 5 (Current)** |
|---|---|---|---|---|---|---|
| **Overall Verdict Accuracy** | `0.00%` (Raw) | `35.00%` | `40.00%` | `60.00%` | `70.00%` | **`80.00%` (16/20)** 🚀 |
| **`CONTRADICT` Recall** | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | **`100.00%` (6/6)** 🎯 |
| **`NOT MENTIONED` Recall** | `0.00%` | `16.67%` | `50.00%` | `100.00%` | `100.00%` | **`83.33%` (5/6)** 🎯 |
| **`SUPPORT` Recall** | `12.50%` | `75.00%` | `87.50%` | `75.00%` | `100.00%` | **`62.50%` (5/8)** |
| **RAGAS Answer Relevancy** | N/A | `0.5871` | `0.6247` | `0.6419` | `0.6130` | **`0.6779`** 📈 |
| **RAGAS Context Precision** | N/A | `0.1428` | `0.1550` | `0.1600` | `0.1667` | **`0.2102`** 📈 |
| **Default Model** | `mistral-7b` | `mistral-7b` | `mistral-7b` | `mistral-7b` | `mistral-7b` | **`phi3.5:latest` (3.8B)** ⚡ |
| **Inference Time (20 claims)**| ~10 mins | ~8 mins | ~8 mins | ~7 mins | ~8 mins | **`~45 seconds`** ⚡ |

---

### 🔹 Iteration 5: Model Optimization & Book-Aware Retrieval (Accuracy: 70% ➔ 80%)
* **Model Migration (`phi3.5:latest`)**:
  - Migrated production pipeline and evaluation layers from `mistral-7b` to `phi3.5:latest` (3.8B parameter SLM).
  - Model fits 100% in 4GB GPU VRAM on RTX 3050 Laptop GPU with 0 offloading and 0 timeout errors.
  - End-to-end 20-claim verification latency dropped from **8-12 minutes to under 45 seconds** (10x faster).
* **Book-Aware Filtered Search (`Pipeline/querySearch.py` & `Pipeline/claimRetrieval.py`)**:
  - Automatically resolves entity $\rightarrow$ target novel ID (`Book 1`: Verne, `Book 2`: Dumas) and performs deep FAISS search (`k_search = 120`).
  - Eliminates cross-novel entity confusion and cross-novel hallucinations entirely.
* **Explicit Role-Conflict Verification Logic (`Pipeline/verfication.py`)**:
  - Formats prompts with `<|user|>` / `<|assistant|>` and strict criteria for distinguishing between false role claims (`CONTRADICT`) and unmentioned backstory claims (`NOT MENTIONED`).
* **Major Milestone**:
  - **`CONTRADICT` Accuracy skyrocketed from 0/6 (0.00%) to 6/6 (100.00% PASS)**!
  - **Overall Verdict Accuracy reached 80.00% (16/20 PASS)**.

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

### 🔹 Iteration 3: Possessive Resolution & Regex Parsing (Accuracy: 40% ➔ 60%)
* **Changes Made**:
  1. **Fixed Possessive Truncation Bug (`Pipeline/querySearch.py`)**:
     * Fixed string-stripping bug where `strip(" '’s.,")` was stripping valid trailing `'s'` characters from names (e.g., turning `Mercédès` into `mercédé` and `Dantès` into `dantè`).
     * Added `find_entity_in_index()` with accent normalization (`leclère` $\leftrightarrow$ `leclere`), achieving **100% entity match rate**.
  2. **Fixed Relative Pronoun Scrambling (`Pipeline/claimExtraction.py`)**:
     * Prevented pronoun resolver from corrupting relative clauses (`who`/`which`) with distant entity names.
     * Preserved coordinate verb phrases (e.g., *"hooked and landed the shark"*) from being broken into dangling fragments.
  3. **Regex-Based Verdict Parsing (`Pipeline/aggregation.py`)**:
     * Replaced substring checks (`'SUPPORT' in result`, `'CONTRADICT' in result`) with strict regex extraction (`re.findall(r'verdict\s*:\s*["\']?\s*(SUPPORT|CONTRADICT|NOT MENTIONED)...')`).
     * Eliminated false-positive collisions from reasoning phrases like *"nor does it directly contradict this claim"*.

---

### 🔹 Iteration 4: Subject-First & Balanced Hybrid Retrieval (Accuracy: 60% ➔ 70%)
* **Changes Made**:
  1. **Subject-First Grammatical Entity Resolution (`Pipeline/querySearch.py`)**:
     * Updated `extract_entity()` to prioritize the grammatical subject (`nsubj` / `nsubjpass`) over trailing mentioned names (e.g. accurately identifying `Danglars` in Claim 8 and `Fernand Mondego` in Claim 13 instead of downstream objects).
     * Implemented descending key length sorting in `find_entity_in_index()` to avoid false matching against short sub-tokens.
  2. **Balanced Global-First Hybrid Retrieval (`Pipeline/claimRetrieval.py`)**:
     * Structured candidate evidence as `global_results[:10] + entity_results[:5]`.
     * Global semantic vector search prioritizes key event and action descriptions, while subset entity search supplies core character biographical background and titles.
  3. **Entity Key Pooling**:
     * Merged partitioned sub-keys in `entity.json` (e.g., `m. danglars` + `danglars`), eliminating blind spots in initial chapters.
* **Impact**:
  * **Overall Verdict Accuracy reached 70.00% (14/20 PASS)**.
  * **`SUPPORT` Recall reached 100.00% (8/8 PASS)**.
  * **`NOT MENTIONED` Recall reached 100.00% (6/6 PASS)**.
  * **RAGAS Context Recall increased to 74.17% (0.7417)**.

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
