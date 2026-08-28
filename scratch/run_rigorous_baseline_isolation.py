import os
import sys
import json
import time
import requests
import faiss
import numpy as np

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks, get_canonical_profile
from claimExtraction import extract_atomic_claims
from reranker import rerank_candidates
from aggregation import aggregate_results

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

def query_llm(prompt, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120}
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            time.sleep(2)
    return ""

# Prompt A: Binary Closed-World Prompt (SUPPORT vs CONTRADICT only)
def prompt_binary(claim, evidence_list):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list[:8])])
    return f"""<|user|>
Evaluate if the Claim is TRUE (SUPPORT) or FALSE (CONTRADICT) based on the excerpts.
Claim: "{claim}"
Source Excerpts:
{ev_text}

End on the last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT<|end|>
<|assistant|>"""

# Prompt B: 3-Way Open-World NLI Prompt (SUPPORT vs CONTRADICT vs NOT MENTIONED)
def prompt_3way(claim, evidence_list, entity="", use_canonical=False):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list[:8])])
    prof_sec = ""
    if use_canonical and entity:
        prof = get_canonical_profile(entity)
        if prof:
            prof_sec = f"Canonical Knowledge about {entity}:\n{prof}\n\n"
            
    return f"""<|user|>
Evaluate whether the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on the excerpts.
Claim: "{claim}"
Character: "{entity}"

{prof_sec}Source Excerpts:
{ev_text}

RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with canonical facts, identity, parentage, role, or fate in the text.
2. SUPPORT: The claim is confirmed true by the excerpts.
3. NOT MENTIONED: The claim describes unmentioned private history, hobbies, or details absent from the text without direct contradiction.

End on the last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

def parse_verdict_clean(raw_resp):
    lines = [l.strip() for l in raw_resp.split("\n") if l.strip()]
    for l in reversed(lines):
        up = l.upper()
        if "VERDICT:" in up:
            val = up.split("VERDICT:")[1].strip()
            if "CONTRADICT" in val or "INCOMPATIBLE" in val:
                return "CONTRADICT"
            elif "SUPPORT" in val and "NOT" not in val:
                return "SUPPORT"
            elif "NOT MENTIONED" in val or "UNMENTIONED" in val:
                return "NOT MENTIONED"
        if up in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            return up
    if "CONTRADICT" in raw_resp.upper():
        return "CONTRADICT"
    if "NOT MENTIONED" in raw_resp.upper():
        return "NOT MENTIONED"
    return "SUPPORT"

# Load Assets
print("Loading corpus assets...")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Sample 40 balanced claims (10 per book: 4 Support, 3 Contradict, 3 Not Mentioned) for rigorous 4-condition baseline isolation
test_subset = []
book_counts = {}
for item in dataset:
    b = item["book"]
    book_counts.setdefault(b, 0)
    if book_counts[b] < 10:
        test_subset.append(item)
        book_counts[b] += 1

print(f"Running 4-Condition Baseline Isolation across {len(test_subset)} balanced claims (10 per book across all 4 books)...\n")

def run_experiment(name, retrieval_type="vanilla", prompt_type="binary", use_canonical=False):
    print(f"--- Condition: {name} ---")
    traces = []
    t_start = time.time()
    
    # 3x3 Confusion Matrix: [GT][PRED]
    matrix = {
        "SUPPORT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "NOT MENTIONED": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
    }
    
    for idx, item in enumerate(test_subset, 1):
        claim_text = item["user_input"]
        gt = item["ground_truth_verdict"].strip().upper()
        
        if retrieval_type == "vanilla":
            q_embed = normalize(createEmbeddings(claim_text))
            scores, indices = faiss_index.search(q_embed, 5)
            evidence = [chunks[i] for i in indices[0] if i < len(chunks)]
            
            if prompt_type == "binary":
                p = prompt_binary(claim_text, evidence)
            else:
                p = prompt_3way(claim_text, evidence, "", use_canonical=False)
            resp = query_llm(p)
            pred = parse_verdict_clean(resp)
            
        else: # Backstory RAG retrieval
            sub_claims = extract_atomic_claims(claim_text)
            sub_verifs = []
            for sub in sub_claims:
                ent = extract_entity(sub, entity_index)
                pooled_cids = get_pooled_entity_chunks(ent, entity_index) if ent else []
                target_book = chunks[pooled_cids[0]].get("Book") if (pooled_cids and pooled_cids[0] < len(chunks)) else None
                
                g_res = global_search(sub, faiss_index, chunks, target_book=target_book, top_k=25)
                e_res = subset_search(sub, pooled_cids, faiss_index, chunks, top_k=25) if pooled_cids else []
                
                seen = set()
                cand = []
                for r in g_res + e_res:
                    txt = r["text"].strip()
                    if txt not in seen:
                        seen.add(txt)
                        cand.append(r)
                        
                evidence = rerank_candidates(sub, cand, top_k=8)
                
                if prompt_type == "binary":
                    p = prompt_binary(sub, evidence)
                else:
                    p = prompt_3way(sub, evidence, ent, use_canonical=use_canonical)
                resp = query_llm(p)
                sub_verifs.append({"Claim": sub, "Evidence": evidence, "Verification_result": resp})
                
            agg = aggregate_results(sub_verifs)
            pred_raw = agg["Final Verdict"]
            pred = "SUPPORT" if "COMPATIBLE" in pred_raw else ("CONTRADICT" if "INCOMPATIBLE" in pred_raw else "NOT MENTIONED")
            
        is_match = (pred == gt)
        matrix[gt][pred] = matrix[gt].get(pred, 0) + 1
        traces.append({"id": item["id"], "gt": gt, "pred": pred, "match": is_match})
        
    total = len(traces)
    correct = sum(1 for t in traces if t["match"])
    elapsed = time.time() - t_start
    acc = correct / total * 100
    
    print(f"Accuracy: {acc:.2f}% ({correct}/{total}) | Total Time: {elapsed:.1f}s (Avg {elapsed/total:.2f}s/claim)")
    print("Confusion Matrix [Row=Ground Truth, Col=Predicted]:")
    print(f"               Pred SUPPORT | Pred CONTRADICT | Pred NOT MENTIONED")
    print(f"GT SUPPORT    :    {matrix['SUPPORT']['SUPPORT']:<8} |    {matrix['SUPPORT']['CONTRADICT']:<10} |    {matrix['SUPPORT']['NOT MENTIONED']:<10}")
    print(f"GT CONTRADICT :    {matrix['CONTRADICT']['SUPPORT']:<8} |    {matrix['CONTRADICT']['CONTRADICT']:<10} |    {matrix['CONTRADICT']['NOT MENTIONED']:<10}")
    print(f"GT NOT MENTION:    {matrix['NOT MENTIONED']['SUPPORT']:<8} |    {matrix['NOT MENTIONED']['CONTRADICT']:<10} |    {matrix['NOT MENTIONED']['NOT MENTIONED']:<10}\n")
    
    return {
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "avg_latency": round(elapsed/total, 2),
        "confusion_matrix": matrix
    }

isolation_results = {}

# 1. Condition 1: Vanilla Dense Retrieval + Binary Prompt (Naive Baseline)
isolation_results["[1] Vanilla Retrieval + Binary Prompt"] = run_experiment(
    "Vanilla Dense Retrieval + Binary Prompt", retrieval_type="vanilla", prompt_type="binary", use_canonical=False
)

# 2. Condition 2: Vanilla Dense Retrieval + 3-Way OW-NLI Prompt (Isolated Verdict Layer)
isolation_results["[2] Vanilla Retrieval + 3-Way OW-NLI Prompt"] = run_experiment(
    "Vanilla Dense Retrieval + 3-Way OW-NLI Prompt", retrieval_type="vanilla", prompt_type="3way", use_canonical=False
)

# 3. Condition 3: Backstory RAG Retrieval + Binary Prompt (Isolated Retrieval Stack)
isolation_results["[3] Backstory Retrieval + Binary Prompt"] = run_experiment(
    "Backstory Retrieval + Binary Prompt", retrieval_type="backstory", prompt_type="binary", use_canonical=False
)

# 4. Condition 4: Full Backstory RAG (Backstory Retrieval + Canonical Persona + 3-Way OW-NLI)
isolation_results["[4] Full Backstory RAG (Retrieval + Persona + 3-Way OW-NLI)"] = run_experiment(
    "Full Backstory RAG (Ours)", retrieval_type="backstory", prompt_type="3way", use_canonical=True
)

with open("benchmark/baseline_isolation_results.json", "w", encoding="utf-8") as f:
    json.dump(isolation_results, f, ensure_ascii=False, indent=2)

print("Saved baseline isolation results to 'benchmark/baseline_isolation_results.json'.")
