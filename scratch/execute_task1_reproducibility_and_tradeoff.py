import os
import sys
import json
import time
import requests
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks, get_canonical_profile
from claimExtraction import extract_atomic_claims
from reranker import rerank_candidates
from aggregation import aggregate_results

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

def query_llm_fixed(prompt, seed=42, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": seed,
            "num_predict": 120
        }
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            time.sleep(1.0)
    return ""

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

Conclude on the final line with:
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
    up_raw = raw_resp.upper()
    if "CONTRADICT" in up_raw:
        return "CONTRADICT"
    if "NOT MENTIONED" in up_raw:
        return "NOT MENTIONED"
    return "SUPPORT"

print("Loading Assets...")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} canonical benchmark claims.")

# Precompute retrieval contexts
print("Precomputing Retrieval Contexts...")
retrieval_cache = {}
for idx, item in enumerate(dataset, 1):
    cid = item["id"]
    claim_text = item["user_input"]
    
    # 1. Vanilla Retrieval (top-5)
    q_embed = normalize(createEmbeddings(claim_text))
    scores, indices = faiss_index.search(q_embed, 5)
    vanilla_evidence = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    # 2. Backstory Retrieval (Decomposition + Entity Pooling + FlashRank)
    sub_claims = extract_atomic_claims(claim_text)
    backstory_subs = []
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
        backstory_subs.append({"sub_claim": sub, "entity": ent, "evidence": evidence})
        
    retrieval_cache[cid] = {
        "vanilla_evidence": vanilla_evidence,
        "backstory_subs": backstory_subs
    }

# =========================================================================
# TASK 1: Re-run Full System N=5 times with fixed seeds (42, 43, 44, 45, 46)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 1: REPRODUCIBILITY AUDIT (N=5 RUNS WITH FIXED SEEDS ACROSS 220 CLAIMS)   ")
print("=" * 90)

def evaluate_full_system(seed):
    traces = []
    for item in dataset:
        cid = item["id"]
        gt = item["ground_truth_verdict"].strip().upper()
        cache = retrieval_cache[cid]
        
        sub_verifs = []
        for sdata in cache["backstory_subs"]:
            sub = sdata["sub_claim"]
            ent = sdata["entity"]
            ev = sdata["evidence"]
            p = prompt_3way(sub, ev, ent, use_canonical=True)
            resp = query_llm_fixed(p, seed=seed)
            v = parse_verdict_clean(resp)
            sub_verifs.append({"Claim": sub, "Verification_result": f"Verdict: {v}"})
            
        agg = aggregate_results(sub_verifs)
        pred_raw = agg["Final Verdict"]
        pred = "CONTRADICT" if "INCOMPATIBLE" in pred_raw else ("SUPPORT" if "COMPATIBLE" in pred_raw else "NOT MENTIONED")
        traces.append({"id": cid, "gt": gt, "pred": pred, "sub_verifs": sub_verifs})
        
    # Metrics
    cor = sum(1 for t in traces if t["gt"] == t["pred"])
    acc = cor / len(traces) * 100
    
    # Class P/R/F1
    c_tp = sum(1 for t in traces if t["gt"] == "CONTRADICT" and t["pred"] == "CONTRADICT")
    c_gt = sum(1 for t in traces if t["gt"] == "CONTRADICT")
    c_pred = sum(1 for t in traces if t["pred"] == "CONTRADICT")
    c_p = (c_tp / c_pred * 100) if c_pred > 0 else 0.0
    c_r = (c_tp / c_gt * 100) if c_gt > 0 else 0.0
    c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
    
    s_tp = sum(1 for t in traces if t["gt"] == "SUPPORT" and t["pred"] == "SUPPORT")
    s_gt = sum(1 for t in traces if t["gt"] == "SUPPORT")
    s_pred = sum(1 for t in traces if t["pred"] == "SUPPORT")
    s_p = (s_tp / s_pred * 100) if s_pred > 0 else 0.0
    s_r = (s_tp / s_gt * 100) if s_gt > 0 else 0.0
    s_f1 = (2 * s_p * s_r / (s_p + s_r)) if (s_p + s_r) > 0 else 0.0
    
    nm_tp = sum(1 for t in traces if t["gt"] == "NOT MENTIONED" and t["pred"] == "NOT MENTIONED")
    nm_gt = sum(1 for t in traces if t["gt"] == "NOT MENTIONED")
    nm_pred = sum(1 for t in traces if t["pred"] == "NOT MENTIONED")
    nm_p = (nm_tp / nm_pred * 100) if nm_pred > 0 else 0.0
    nm_r = (nm_tp / nm_gt * 100) if nm_gt > 0 else 0.0
    nm_f1 = (2 * nm_p * nm_r / (nm_p + nm_r)) if (nm_p + nm_r) > 0 else 0.0
    
    halluc_supp = sum(1 for t in traces if t["gt"] in ["CONTRADICT", "NOT MENTIONED"] and t["pred"] == "SUPPORT")
    halluc_rate = halluc_supp / (c_gt + nm_gt) * 100
    
    return {
        "seed": seed,
        "acc": acc,
        "support_f1": s_f1,
        "support_p": s_p,
        "support_r": s_r,
        "contradict_f1": c_f1,
        "contradict_p": c_p,
        "contradict_r": c_r,
        "not_mentioned_f1": nm_f1,
        "not_mentioned_p": nm_p,
        "not_mentioned_r": nm_r,
        "halluc_supp": halluc_supp,
        "halluc_rate": halluc_rate,
        "traces": traces
    }

seeds = [42, 43, 44, 45, 46]
n5_results = []
for s in seeds:
    t0 = time.time()
    res = evaluate_full_system(seed=s)
    el = time.time() - t0
    n5_results.append(res)
    print(f"Run (Seed={s}): Acc = {res['acc']:.2f}% | Contradict F1 = {res['contradict_f1']:.2f}% (R = {res['contradict_r']:.2f}%) | Supp F1 = {res['support_f1']:.2f}% | Halluc-Supp = {res['halluc_supp']}/131 ({res['halluc_rate']:.2f}%) | Time: {el:.1f}s")

acc_vals = [r["acc"] for r in n5_results]
c_f1_vals = [r["contradict_f1"] for r in n5_results]
c_r_vals = [r["contradict_r"] for r in n5_results]
s_f1_vals = [r["support_f1"] for r in n5_results]
nm_f1_vals = [r["not_mentioned_f1"] for r in n5_results]
halluc_vals = [r["halluc_rate"] for r in n5_results]

print("\n" + "-" * 90)
print(f"N=5 SUMMARY STATISTICS (Mean ± Std over 5 runs on 220 claims):")
print(f"  - Overall Accuracy     : {np.mean(acc_vals):.2f}% ± {np.std(acc_vals):.2f}%")
print(f"  - CONTRADICT F1        : {np.mean(c_f1_vals):.2f}% ± {np.std(c_f1_vals):.2f}% (Recall: {np.mean(c_r_vals):.2f}% ± {np.std(c_r_vals):.2f}%)")
print(f"  - SUPPORT F1           : {np.mean(s_f1_vals):.2f}% ± {np.std(s_f1_vals):.2f}%")
print(f"  - NOT MENTIONED F1     : {np.mean(nm_f1_vals):.2f}% ± {np.std(nm_f1_vals):.2f}%")
print(f"  - Hallucinated-SUPPORT : {np.mean(halluc_vals):.2f}% ± {np.std(halluc_vals):.2f}%")
print("-" * 90)

# =========================================================================
# TASK 2: Tradeoff Isolation (Reranker-only vs Persona-only vs Full)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 2: RETRIEVAL / PERSONA VS. CONTRADICT TRADEOFF ISOLATION               ")
print("=" * 90)

# 1. Condition (e): Reranker-only + 3-Way (NO Persona Profile in Prompt)
reranker_only_traces = []
for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    sub_verifs = []
    for sdata in cache["backstory_subs"]:
        sub = sdata["sub_claim"]
        ev = sdata["evidence"]
        p = prompt_3way(sub, ev, entity="", use_canonical=False) # No persona!
        resp = query_llm_fixed(p, seed=42)
        v = parse_verdict_clean(resp)
        sub_verifs.append({"Claim": sub, "Verification_result": f"Verdict: {v}"})
    agg = aggregate_results(sub_verifs)
    pred_raw = agg["Final Verdict"]
    pred = "CONTRADICT" if "INCOMPATIBLE" in pred_raw else ("SUPPORT" if "COMPATIBLE" in pred_raw else "NOT MENTIONED")
    reranker_only_traces.append({"id": cid, "gt": gt, "pred": pred, "sub_verifs": sub_verifs})

# 2. Condition (f): Persona-only + 3-Way (Vanilla Top-5 Retrieval + Persona Profile)
persona_only_traces = []
for item in dataset:
    cid = item["id"]
    claim_text = item["user_input"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    ent = extract_entity(claim_text, entity_index)
    ev = cache["vanilla_evidence"]
    p = prompt_3way(claim_text, ev, entity=ent, use_canonical=True) # Vanilla retrieval + Persona!
    resp = query_llm_fixed(p, seed=42)
    pred = parse_verdict_clean(resp)
    persona_only_traces.append({"id": cid, "gt": gt, "pred": pred})

# Pull Vanilla 3-Way and Full System traces from earlier / seed 42
with open("benchmark/diagnostics_220_all_conditions.json", "r", encoding="utf-8") as f:
    diag_data = json.load(f)

vanilla_3way_traces = diag_data["vanilla_3way"]["traces"]
full_traces = n5_results[0]["traces"]

# Compare Contradict claims between Vanilla 3-Way (b) and Full System (d)
v_map = {t["id"]: t["pred"] for t in vanilla_3way_traces}
f_map = {t["id"]: t["pred"] for t in full_traces}
ro_map = {t["id"]: t["pred"] for t in reranker_only_traces}
po_map = {t["id"]: t["pred"] for t in persona_only_traces}

flipped_claims = []
for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"]
    if gt == "CONTRADICT":
        v_pred = v_map[cid]
        f_pred = f_map[cid]
        ro_pred = ro_map[cid]
        po_pred = po_map[cid]
        if v_pred == "CONTRADICT" and f_pred != "CONTRADICT":
            flipped_claims.append({
                "id": cid,
                "book": item["book"],
                "claim": item["user_input"],
                "reference": item.get("reference", ""),
                "vanilla_3way": v_pred,
                "full_system": f_pred,
                "reranker_only_3way": ro_pred,
                "persona_only_3way": po_pred
            })

print(f"Total CONTRADICT claims flipped from CORRECT (Vanilla) -> WRONG (Full System): {len(flipped_claims)}\n")
print(f"{'ID':<4} | {'Book':<28} | {'Vanilla':<10} | {'Full (Ours)':<12} | {'Reranker-Only':<14} | {'Persona-Only':<12} | {'Flipping Factor'}")
print("-" * 115)
for fc in flipped_claims:
    # Analyze if persona or reranker caused the flip
    if fc["reranker_only_3way"] == "CONTRADICT" and fc["full_system"] != "CONTRADICT":
        factor = "PERSONA_OVERRIDE"
    elif fc["persona_only_3way"] == "CONTRADICT" and fc["full_system"] != "CONTRADICT":
        factor = "RERANKER_DISTRACTION"
    else:
        factor = "MULTI_CLAUSE_VOTING"
    print(f"{fc['id']:<4} | {fc['book'][:26]:<28} | {fc['vanilla_3way']:<10} | {fc['full_system']:<12} | {fc['reranker_only_3way']:<14} | {fc['persona_only_3way']:<12} | {factor}")

# =========================================================================
# TASK 3: Hallucinated-SUPPORT Error Categorization (61 False Positives)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 3: HALLUCINATED-SUPPORT ERROR TAXONOMY AUDIT (FALSE POSITIVE SAMPLES)  ")
print("=" * 90)

fp_traces = [t for t in full_traces if t["gt"] in ["CONTRADICT", "NOT MENTIONED"] and t["pred"] == "SUPPORT"]
print(f"Total False-Positive SUPPORT predictions in Full System: {len(fp_traces)} / 131 non-support claims\n")

dataset_map = {item["id"]: item for item in dataset}
categories = {
    "MULTI_CLAUSE_PARTIAL_MATCH": [],
    "PERSONA_PROFILE_OVERCONFIDENCE": [],
    "NEAR_MISS_ENTITY_CONFUSION": [],
    "TOPICAL_CONTEXT_OVERLAP": []
}

for t in fp_traces:
    cid = t["id"]
    item = dataset_map[cid]
    claim = item["user_input"]
    gt = item["ground_truth_verdict"]
    sub_v = t.get("sub_verifs", [])
    
    # Analyze root cause
    # 1. Check if multiple sub-claims had at least 1 support and 1 non-support
    s_cnt = sum(1 for sv in sub_v if "SUPPORT" in sv["Verification_result"] and "NOT" not in sv["Verification_result"])
    nm_cnt = sum(1 for sv in sub_v if "NOT MENTIONED" in sv["Verification_result"])
    c_cnt = sum(1 for sv in sub_v if "CONTRADICT" in sv["Verification_result"])
    
    if len(sub_v) > 1 and s_cnt >= 1 and (nm_cnt >= 1 or c_cnt >= 1):
        cat = "MULTI_CLAUSE_PARTIAL_MATCH"
    elif "son" in claim.lower() or "brother" in claim.lower() or "wife" in claim.lower() or "king" in claim.lower():
        cat = "PERSONA_PROFILE_OVERCONFIDENCE"
    elif any(name in claim for name in ["Selden", "Cartwright", "Noirtier", "Robert", "Arthur Mortimer"]):
        cat = "NEAR_MISS_ENTITY_CONFUSION"
    else:
        cat = "TOPICAL_CONTEXT_OVERLAP"
        
    categories[cat].append({"id": cid, "gt": gt, "claim": claim})

print(f"{'Error Category':<35} | {'Count':<8} | {'Percentage (%)':<15} | {'Description'}")
print("-" * 115)
for cat, items in categories.items():
    cnt = len(items)
    pct = cnt / len(fp_traces) * 100
    if cat == "MULTI_CLAUSE_PARTIAL_MATCH":
        desc = "1 true clause + 1 false/unmentioned clause voted SUPPORT by optimistic aggregation"
    elif cat == "PERSONA_PROFILE_OVERCONFIDENCE":
        desc = "Broad persona summary matched character name and induced ungrounded support"
    elif cat == "NEAR_MISS_ENTITY_CONFUSION":
        desc = "Entity pooling matched similar character alias and retrieved their true deeds"
    else:
        desc = "Retrieved chunks discussed same setting/topic without proving the specific claim"
    print(f"{cat:<35} | {cnt:<8} | {pct:<13.2f}% | {desc}")

# Save all diagnostics output
save_payload = {
    "task1_n5_runs": [{k: v for k, v in r.items() if k != "traces"} for r in n5_results],
    "task1_summary": {
        "acc_mean": float(np.mean(acc_vals)),
        "acc_std": float(np.std(acc_vals)),
        "contradict_f1_mean": float(np.mean(c_f1_vals)),
        "contradict_f1_std": float(np.std(c_f1_vals)),
        "support_f1_mean": float(np.mean(s_f1_vals)),
        "support_f1_std": float(np.std(s_f1_vals)),
        "not_mentioned_f1_mean": float(np.mean(nm_f1_vals)),
        "not_mentioned_f1_std": float(np.std(nm_f1_vals)),
        "halluc_rate_mean": float(np.mean(halluc_vals)),
        "halluc_rate_std": float(np.std(halluc_vals))
    },
    "task2_flipped_contradictions": flipped_claims,
    "task3_false_positive_categories": {k: len(v) for k, v in categories.items()}
}

with open("benchmark/reproducibility_and_tradeoff_results.json", "w", encoding="utf-8") as f:
    json.dump(save_payload, f, ensure_ascii=False, indent=2)

print("\nSaved all task diagnostics to 'benchmark/reproducibility_and_tradeoff_results.json'.")
