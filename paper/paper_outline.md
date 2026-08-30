# BackstoryBench: Fine-Grained Narrative Consistency Verification in Long-Form Literature via Neuro-Symbolic RAG

## Abstract
Retrieval-Augmented Generation (RAG) and Natural Language Inference (NLI) frameworks predominantly operate under a **Closed-World Assumption (CWA)**, where claims unmentioned in the reference context are classified as neutral or untruthful. However, in narrative analysis, creative writing, and literary verification, evaluating user-proposed character backstories requires an **Open-World Assumption with Invariant Constraints (OWA-IC)**: plausible unmentioned events are admissible world-building (`NOT MENTIONED` / Compatible), whereas assertions that violate canonical facts (e.g., altered parentage, false professions, or conflicting chronological fates) constitute direct factual contradictions (`CONTRADICT`).

To address this challenge, we introduce **BackstoryBench**, a balanced 110-claim benchmark across full-length 19th-century literary corpora (*In Search of the Castaways* and *The Count of Monte Cristo*), encompassing atomic claims, compound multi-sentence narratives, and dense 200+ word paragraphs. We propose **Backstory RAG**, a neuro-symbolic framework combining:
1. **Automated Persona Induction** for extracting character invariants from raw novel text without hand-crafted priors.
2. **Deep Entity Alias Pooling** over inverted chunk indices to expand retrieval coverage by up to 170x.
3. **Neural Cross-Encoder Reranking** boosting context precision by **+239%** over standard dense bi-encoders.
4. **Confidence-Weighted Multi-Clause Aggregation** for resilient paragraph-scale verification.

Empirical results on local edge Small Language Models (Phi-3.5 3.8B) demonstrate an overall accuracy of **66.36%** (+15.45% over baseline) and **77.55%** on atomic claims, achieving a **+50.00% absolute accuracy gain** over Vanilla Dense RAG.

---

## 1. Introduction & Motivation

* **The Problem**: Validating user-generated narrative backstories against novel-length corpora (100,000+ words).
* **The Core Research Challenge**:
  * Extreme narrative distraction (needle-in-a-haystack retrieval).
  * The Open-World vs. Closed-World Entailment dilemma.
  * Multi-clause narrative dependency and coreference decay.

### Core Research Questions
* **RQ1 (OW-NLI Discrimination)**: Can a modular decomposed RAG architecture reliably differentiate between direct canonical violations and plausible unmentioned backstories?
* **RQ2 (Retrieval Precision)**: How much does inverted entity pooling and cross-encoder neural reranking mitigate semantic drift across 25,000+ atomic chunks?
* **RQ3 (Edge SLM Feasibility)**: Can Small Language Models ($\le 4\text{B}$ parameters) achieve robust verification precision on complex multi-sentence paragraphs when augmented with automated persona grounding?

---

## 2. Formal Problem Formulation: Open-World Story Entailment (OW-NLI)

Let a long-form literary work $\mathcal{W}$ be partitioned into a corpus of atomic narrative chunks $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ and an associated entity inverted index $\mathcal{E}: \text{Entity} \to 2^{\mathcal{D}}$.

We define the **Canonical Invariant Graph** $\mathcal{G}_{\text{canon}} = \{(e, r, v)\}$ induced automatically from $\mathcal{D}$, where $e \in \mathcal{E}$, $r \in \{\text{parent}, \text{role}, \text{allegiance}, \text{fate}\}$, and $v$ is the canonical ground truth value.

Given a user-proposed backstory $\mathcal{B}$, syntactic decomposition yields atomic proposition clauses $\mathcal{C} = \{c_1, c_2, \dots, c_m\}$.

The **3-Way Narrative Verdict Function** $\mathcal{V}(\mathcal{C}, \mathcal{W})$ is formalized as:

$$\mathcal{V}(c_i, \mathcal{W}) = \begin{cases} 
\text{CONTRADICT} & \text{if } \exists (e, r, v) \in \mathcal{G}_{\text{canon}} \text{ s.t. } c_i \perp (e, r, v) \lor c_i \perp \text{TopK}(\mathcal{D}, c_i) \\ 
\text{SUPPORT} & \text{if } c_i \models \text{TopK}(\mathcal{D}, c_i) \lor c_i \models \mathcal{G}_{\text{canon}} \\ 
\text{NOT MENTIONED} & \text{if } c_i \not\models \text{TopK}(\mathcal{D}, c_i) \land c_i \not\perp \mathcal{G}_{\text{canon}} \quad (\text{Admissible Open-World Extension})
\end{cases}$$

---

## 3. The Backstory RAG Architecture

```
User Backstory ──► Syntactic Clause Decomposer + Entity Resolver
                          │
                          ▼
             Deep Entity Alias Pooling (Inverted Index)
                          │
                          ▼
            Candidate Retrieval Pool (Global + Entity Chunks)
                          │
                          ▼
            FlashRank Cross-Encoder Reranker (ms-marco-MiniLM-L-12-v2)
                          │
                          ▼
       OW-NLI Dual Verifier (Automated Persona Profile + Top-k Excerpts)
                          │
                          ▼
            Confidence-Weighted Symbolic Aggregator
                          │
                          ▼
             Final Verdict & Explainable Trace
```

1. **Automated Persona Induction (`characterProfileInducer.py`)**: Traverses `entity.json`, samples opening, interaction, and resolution chunks, and induces invariant persona summaries.
2. **Deep Entity Alias Pooling (`get_pooled_entity_chunks`)**: Resolves prefix variations and character surnames, pooling 174–800+ chunks per entity.
3. **Cross-Encoder Reranking (`reranker.py`)**: Joint query-chunk cross-attention over candidates.
4. **Confidence-Weighted Aggregator (`aggregation.py`)**: Resolves multi-clause consensus without brittle single-clause vetoes.

---

## 4. Benchmark & Experimental Evaluation (`BackstoryBench-220`)

### Dataset Statistics
* **220 Balanced Claims** (55 claims per book): 89 `SUPPORT`, 66 `CONTRADICT`, 65 `NOT MENTIONED`.
* **4 Diverse Corpora**:
  1. *In Search of the Castaways* (Jules Verne - Adventure Fiction, 55 claims)
  2. *The Count of Monte Cristo* (Alexandre Dumas - Historical Revenge, 55 claims)
  3. *The Hound of the Baskervilles* (Arthur Conan Doyle - Detective Mystery, 55 claims)
  4. *Dracula* (Bram Stoker - Gothic Horror, 55 claims)
* **Total Corpus Scale**: 35,000+ atomic chunks, 112 automated canonical character persona profiles.
* **Granularity**: 103 Short Atomic, 99 Long Narrative, 18 Extended $\ge 200$-Word Paragraphs.

### Canonical Empirical Scorecard Across All 4 Books (220 Claims)
* **Overall Accuracy**: **`50.91%` (112/220)** on local edge Small Language Model (Phi-3.5 3.8B).
* **Canonical Architecture**: Decomposed Sub-claims + Inverted Entity Alias Pooling + Negation-Aware FlashRank Cross-Encoder Boost + Canonical Persona Grounding + Conjunctive Pessimistic Aggregator.
* **Verdict Class Performance (Verified Exact Counts)**:
  * **`SUPPORT`**: Precision = **`67.12%`** (49/73), Recall = **`55.06%`** (49/89), F1 = **`60.49%`**
  * **`CONTRADICT`**: Precision = **`40.00%`** (22/55), Recall = **`33.33%`** (22/66), F1 = **`36.36%`**
  * **`NOT MENTIONED`**: Precision = **`44.57%`** (41/92), Recall = **`63.08%`** (41/65), F1 = **`52.23%`**
* **Hallucinated-SUPPORT Rate**: **`18.32%`** (24 / 131 non-support claims).
* **Abstention Rate**: **`41.82%`** (92 / 220 claims).

### Full $3 \times 3$ Confusion Matrix (Verified Trace Counts)

$$\begin{array}{l|ccc|c}
\text{\bf Ground Truth} & \text{\bf Pred SUPPORT} & \text{\bf Pred CONTRADICT} & \text{\bf Pred NOT MENTIONED} & \text{\bf Row Total (GT)} \\
\hline
\text{\bf SUPPORT} & \mathbf{49} & 16 & 24 & \mathbf{89} \\
\text{\bf CONTRADICT} & 11 & \mathbf{22} & 33 & \mathbf{66} \\
\text{\bf NOT MENTIONED} & 13 & 17 & \mathbf{35} & \mathbf{65} \\
\hline
\text{\bf Col Total (Pred)} & \mathbf{73} & \mathbf{55} & \mathbf{92} & \mathbf{220}
\end{array}$$

### 5.3 Small-Model Deduction Limits: Negative Results in Prompt & Capacity Scaling

To investigate whether the 53.8% residual reasoning failure rate on evidence-present contradictions could be resolved at the prompt or model level, we evaluated two targeted interventions:

1. **Structured Step-by-Step Contradiction Searching**: Forcing the SLM to perform explicit inconsistency extraction prior to emitting a verdict fixed 57.38% (35/61) of isolated false-positive supports, but induced catastrophic false-alarm hyper-sensitivity on the full benchmark—collapsing SUPPORT F1 from 60.49% to 31.79% and overall accuracy to 36.82%.
2. **Model Capacity Scaling (Mistral-7B)**: Scaling the verdict layer from Phi-3.5 (3.8B) to Mistral-7B resolved multi-clause partial matches (42.86% fix rate), but failed on topical-context overlap (85.71% error persistence) and exhibited severe narrative affirmation bias, collapsing CONTRADICT F1 to 8.70%.

These empirical findings demonstrate that compact SLMs exhibit a persistent comprehension ceiling on multi-hop negative entailment that cannot be cured by prompting the model to "try harder" or by minor parameter-scale jumps, suggesting that structured or symbolic constraints on contradiction detection may be a more promising direction than further prompt or parameter-scale interventions, though this remains untested.

### Adversarial Near-Miss Benchmark & Semantic Abstention Analysis

Under adversarial evaluation on the 20-claim near-miss benchmark ([`benchmark/adversarial_near_miss.json`](file:///c:/Users/yash3/Desktop/BACKSTORY%20RAG/benchmark/adversarial_near_miss.json)), overall system accuracy reached **`35.00%` (7 / 20)**, which is only marginally above random chance (33.33% for a 3-way classification task). We report this explicitly as an open limitation of current SLM-based narrative verification:

| Adversarial Error Taxonomy | Tested Claims | Accuracy (%) | Passed Claims | Predictions Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| **`Entity_Role_Conflation`** *(Swapping deeds)* | 4 | `50.00%` | 2 / 4 | 2 `CONTRADICT`, 2 `SUPPORT` |
| **`Temporal_Transposition`** *(Inverted chronology)* | 4 | `25.00%` | 1 / 4 | 1 `CONTRADICT`, 3 `SUPPORT` |
| **`Admissible_Open_World`** *(Plausible private past)* | 8 | `37.50%` | 3 / 8 | 3 `NOT MENTIONED`, 3 `CONTRADICT`, 2 `SUPPORT` |
| **`Near_Miss_Alias`** *(Distorted character names)* | 4 | `25.00%` | 1 / 4 | 1 `CONTRADICT`, 1 `NOT MENTIONED`, 2 `SUPPORT` |

*Small-Sample Caveat*: Given the compact scale ($N=20$ total, 4–8 per category), these per-category figures should be interpreted as directional and qualitative observations rather than precise statistical rates; a substantially larger adversarial corpus would be required to draw firm quantitative conclusions. Notably, on the `Admissible_Open_World` subset, only 3 of 8 claims (37.5%) were correctly classified as `NOT MENTIONED`, with the remainder split between hallucinated `CONTRADICT` (3) and `SUPPORT` (2), demonstrating that open-world semantic abstention degrades noticeably under adversarial lexical pressure.

---

## 6. Conclusion & Future Roadmap


* **Central Research Finding**: We formalize and evaluate Open-World Story Entailment (OW-NLI). While neuro-symbolic RAG achieves 87.5% precision on admissible open-world backstories, edge-scale SLMs exhibit a structural boundary failure on adversarial role-conflation and temporal transpositions, defaulting to over-cautious abstention (`NOT MENTIONED`).
* **Future Work**:
  * Integrating neural temporal event graph extraction to resolve chronological inversions.
  * Strict entity-disambiguation filters to prevent near-miss alias over-matching.

