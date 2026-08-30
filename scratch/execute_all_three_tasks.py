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

API_URL = "http://localhost:11434/api/generate"

def query_llm_model(model_name, prompt, seed=42, max_retries=3):
    payload = {
        "model": model_name,
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
            r = requests.post(API_URL, json=payload, timeout=90)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            time.sleep(1.5)
    return ""

def prompt_3way(claim, evidence_list, entity="", use_canonical=True):
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

def parse_verdict(raw_resp):
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

# Load Assets
print("Loading Assets...")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Precompute retrieval contexts
print("Precomputing Retrieval Contexts for all 220 claims...")
retrieval_cache = {}
for idx, item in enumerate(dataset, 1):
    cid = item["id"]
    claim_text = item["user_input"]
    
    # 1. Whole claim retrieval (no decomposition)
    ent_whole = extract_entity(claim_text, entity_index)
    pooled_whole = get_pooled_entity_chunks(ent_whole, entity_index) if ent_whole else []
    target_book_whole = chunks[pooled_whole[0]].get("Book") if (pooled_whole and pooled_whole[0] < len(chunks)) else None
    
    g_res_w = global_search(claim_text, faiss_index, chunks, target_book=target_book_whole, top_k=25)
    e_res_w = subset_search(claim_text, pooled_whole, faiss_index, chunks, top_k=25) if pooled_whole else []
    
    seen = set()
    cand_w = []
    for r in g_res_w + e_res_w:
        txt = r["text"].strip()
        if txt not in seen:
            seen.add(txt)
            cand_w.append(r)
    evidence_whole = rerank_candidates(claim_text, cand_w, top_k=8)
    
    # 2. Decomposed sub-claims
    sub_claims = extract_atomic_claims(claim_text)
    backstory_subs = []
    for sub in sub_claims:
        ent = extract_entity(sub, entity_index)
        pooled_cids = get_pooled_entity_chunks(ent, entity_index) if ent else []
        target_book = chunks[pooled_cids[0]].get("Book") if (pooled_cids and pooled_cids[0] < len(chunks)) else None
        
        g_res = global_search(sub, faiss_index, chunks, target_book=target_book, top_k=25)
        e_res = subset_search(sub, pooled_cids, faiss_index, chunks, top_k=25) if pooled_cids else []
        
        seen_s = set()
        cand = []
        for r in g_res + e_res:
            txt = r["text"].strip()
            if txt not in seen_s:
                seen_s.add(txt)
                cand.append(r)
                
        evidence = rerank_candidates(sub, cand, top_k=8)
        backstory_subs.append({"sub_claim": sub, "entity": ent, "evidence": evidence, "all_candidates": cand})
        
    retrieval_cache[cid] = {
        "evidence_whole": evidence_whole,
        "entity_whole": ent_whole,
        "backstory_subs": backstory_subs
    }

# =========================================================================
# TASK 1: CONJUNCTIVE AGGREGATOR & WHOLE-CLAIM TEST (PHI-3.5)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 1: CONJUNCTIVE AGGREGATOR & WHOLE-CLAIM ISOLATION (PHI-3.5)            ")
print("=" * 90)

def aggregate_conjunctive(sub_verdicts):
    c_cnt = sum(1 for v in sub_verdicts if v == "CONTRADICT")
    s_cnt = sum(1 for v in sub_verdicts if v == "SUPPORT")
    nm_cnt = sum(1 for v in sub_verdicts if v == "NOT MENTIONED")
    
    if c_cnt >= 1:
        return "CONTRADICT"
    elif s_cnt >= 1 and nm_cnt == 0:
        return "SUPPORT"
    else:
        return "NOT MENTIONED"

# 1A. Full System with Conjunctive Aggregator (Decomposed)
t0 = time.time()
conjunctive_traces = []
for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    
    sub_verdicts = []
    sub_details = []
    for sdata in cache["backstory_subs"]:
        sub = sdata["sub_claim"]
        ent = sdata["entity"]
        ev = sdata["evidence"]
        p = prompt_3way(sub, ev, ent, use_canonical=True)
        resp = query_llm_model("phi3.5:latest", p, seed=42)
        v = parse_verdict(resp)
        sub_verdicts.append(v)
        sub_details.append({"sub_claim": sub, "verdict": v, "resp": resp})
        
    pred = aggregate_conjunctive(sub_verdicts)
    conjunctive_traces.append({"id": cid, "gt": gt, "pred": pred, "sub_details": sub_details})

el_1a = time.time() - t0
print(f"Executed Conjunctive Full System in {el_1a:.1f}s")

# 1B. Whole Claim (No Decomposition) with Direct Verdict
t0 = time.time()
whole_claim_traces = []
for item in dataset:
    cid = item["id"]
    claim_text = item["user_input"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    
    ev = cache["evidence_whole"]
    ent = cache["entity_whole"]
    p = prompt_3way(claim_text, ev, ent, use_canonical=True)
    resp = query_llm_model("phi3.5:latest", p, seed=42)
    pred = parse_verdict(resp)
    whole_claim_traces.append({"id": cid, "gt": gt, "pred": pred, "resp": resp})

el_1b = time.time() - t0
print(f"Executed Whole-Claim (No Decomposition) in {el_1b:.1f}s")

def compute_metrics(traces, name):
    total = len(traces)
    correct = sum(1 for t in traces if t["gt"] == t["pred"])
    acc = correct / total * 100
    
    matrix = {
        "SUPPORT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "NOT MENTIONED": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
    }
    for t in traces:
        matrix[t["gt"]][t["pred"]] += 1
        
    class_stats = {}
    for cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        gt_cnt = sum(matrix[cls].values())
        pred_cnt = sum(matrix[r][cls] for r in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"])
        tp = matrix[cls][cls]
        prec = (tp / pred_cnt * 100) if pred_cnt > 0 else 0.0
        rec = (tp / gt_cnt * 100) if gt_cnt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        class_stats[cls] = {"gt": gt_cnt, "pred": pred_cnt, "tp": tp, "p": prec, "r": rec, "f1": f1}
        
    halluc_supp = matrix["CONTRADICT"]["SUPPORT"] + matrix["NOT MENTIONED"]["SUPPORT"]
    non_supp_gt = sum(matrix["CONTRADICT"].values()) + sum(matrix["NOT MENTIONED"].values())
    halluc_rate = halluc_supp / non_supp_gt * 100
    abstain_cnt = sum(matrix[r]["NOT MENTIONED"] for r in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"])
    abstain_rate = abstain_cnt / total * 100
    
    return {
        "name": name,
        "acc": acc,
        "correct": correct,
        "total": total,
        "class_stats": class_stats,
        "halluc_supp": halluc_supp,
        "non_supp_gt": non_supp_gt,
        "halluc_rate": halluc_rate,
        "abstain_cnt": abstain_cnt,
        "abstain_rate": abstain_rate,
        "matrix": matrix
    }

m_baseline = {
    "name": "Baseline (Condition d: Optimistic Aggregation)",
    "acc": 52.27,
    "class_stats": {
        "SUPPORT": {"p": 53.03, "r": 79.78, "f1": 63.68},
        "CONTRADICT": {"p": 33.33, "r": 16.67, "f1": 22.22},
        "NOT MENTIONED": {"p": 60.00, "r": 52.31, "f1": 55.93}
    },
    "halluc_supp": 63,
    "halluc_rate": 48.09,
    "abstain_rate": 25.00
}

m_conj = compute_metrics(conjunctive_traces, "Task 1: Conjunctive Aggregator (Phi-3.5)")
m_whole = compute_metrics(whole_claim_traces, "Task 1: Whole Claim (No Decomposition, Phi-3.5)")

print("\nTASK 1 COMPARATIVE RESULTS AGAINST STABILIZED BASELINE:")
print(f"{'Configuration':<45} | {'Acc (%)':<8} | {'Supp F1':<8} | {'Cont F1':<8} | {'NotM F1':<8} | {'Halluc-Supp Rate':<18} | {'Abstain Rate'}")
print("-" * 125)
print(f"{m_baseline['name']:<45} | {m_baseline['acc']:<6.2f}% | {m_baseline['class_stats']['SUPPORT']['f1']:<6.2f}% | {m_baseline['class_stats']['CONTRADICT']['f1']:<6.2f}% | {m_baseline['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {m_baseline['halluc_supp']}/131 ({m_baseline['halluc_rate']:.2f}%)  | {m_baseline['abstain_rate']:.2f}%")
print(f"{m_conj['name']:<45} | {m_conj['acc']:<6.2f}% | {m_conj['class_stats']['SUPPORT']['f1']:<6.2f}% | {m_conj['class_stats']['CONTRADICT']['f1']:<6.2f}% | {m_conj['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {m_conj['halluc_supp']}/131 ({m_conj['halluc_rate']:.2f}%)  | {m_conj['abstain_rate']:.2f}%")
print(f"{m_whole['name']:<45} | {m_whole['acc']:<6.2f}% | {m_whole['class_stats']['SUPPORT']['f1']:<6.2f}% | {m_whole['class_stats']['CONTRADICT']['f1']:<6.2f}% | {m_whole['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {m_whole['halluc_supp']}/131 ({m_whole['halluc_rate']:.2f}%)  | {m_whole['abstain_rate']:.2f}%")

# =========================================================================
# TASK 2: SLM CAPACITY ISOLATION (MISTRAL-7B EVALUATION)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 2: SLM CAPACITY ISOLATION (MISTRAL-7B ON 63 FALSE POSITIVES & 220 BENCHMARK)   ")
print("=" * 90)

# 2A. Test Mistral-7B on the 63 False-Positive SUPPORT claims
with open("benchmark/reproducibility_and_tradeoff_results.json", "r", encoding="utf-8") as f:
    rep_data = json.load(f)

fp_categories = rep_data["task3_false_positive_categories"]
# Load baseline traces to identify exact 63 claims
with open("benchmark/diagnostics_220_all_conditions.json", "r", encoding="utf-8") as f:
    diag = json.load(f)

baseline_traces = diag["backstory_3way"]["traces"]
fp_claim_ids = [t["id"] for t in baseline_traces if t["gt"] in ["CONTRADICT", "NOT MENTIONED"] and t["pred"] == "SUPPORT"]
print(f"Evaluating {len(fp_claim_ids)} False-Positive SUPPORT claims on Mistral-7B...")

mistral_fp_flips = {}
for cid in fp_claim_ids:
    item = next(it for it in dataset if it["id"] == cid)
    gt = item["ground_truth_verdict"]
    cache = retrieval_cache[cid]
    
    sub_verdicts = []
    for sdata in cache["backstory_subs"]:
        sub = sdata["sub_claim"]
        ent = sdata["entity"]
        ev = sdata["evidence"]
        p = prompt_3way(sub, ev, ent, use_canonical=True)
        resp = query_llm_model("mistral:7b", p, seed=42)
        v = parse_verdict(resp)
        sub_verdicts.append(v)
        
    pred_m = aggregate_conjunctive(sub_verdicts)
    is_fixed = (pred_m == gt)
    
    # Categorize
    sub_v = [s["verdict"] for s in conjunctive_traces[cid-1]["sub_details"]]
    s_cnt = sum(1 for v in sub_v if v == "SUPPORT")
    nm_cnt = sum(1 for v in sub_v if v == "NOT MENTIONED")
    c_cnt = sum(1 for v in sub_v if v == "CONTRADICT")
    
    if len(sub_v) > 1 and s_cnt >= 1 and (nm_cnt >= 1 or c_cnt >= 1):
        cat = "MULTI_CLAUSE_PARTIAL_MATCH"
    elif "son" in item["user_input"].lower() or "brother" in item["user_input"].lower() or "wife" in item["user_input"].lower():
        cat = "PERSONA_PROFILE_OVERCONFIDENCE"
    elif any(name in item["user_input"] for name in ["Selden", "Cartwright", "Noirtier", "Robert", "Arthur Mortimer"]):
        cat = "NEAR_MISS_ENTITY_CONFUSION"
    else:
        cat = "TOPICAL_CONTEXT_OVERLAP"
        
    mistral_fp_flips.setdefault(cat, {"total": 0, "fixed": 0, "pred_breakdown": {}})
    mistral_fp_flips[cat]["total"] += 1
    if is_fixed:
        mistral_fp_flips[cat]["fixed"] += 1
    mistral_fp_flips[cat]["pred_breakdown"][pred_m] = mistral_fp_flips[cat]["pred_breakdown"].get(pred_m, 0) + 1

print("\nMISTRAL-7B FIX RATE ACROSS THE 4 ERROR CATEGORIES:")
print(f"{'Error Category':<35} | {'Count':<8} | {'Fixed by Mistral':<18} | {'Fix Rate (%)':<14} | {'Mistral Predictions'}")
print("-" * 115)
tot_fp = 0
tot_fixed = 0
for cat, s in mistral_fp_flips.items():
    tot_fp += s["total"]
    tot_fixed += s["fixed"]
    rate = s["fixed"] / s["total"] * 100
    print(f"{cat:<35} | {s['total']:<8} | {s['fixed']:<18} | {rate:<12.2f}% | {s['pred_breakdown']}")
print("-" * 115)
print(f"{'TOTAL FALSE POSITIVES':<35} | {tot_fp:<8} | {tot_fixed:<18} | {tot_fixed/tot_fp*100:<12.2f}%")

# 2B. Run Full 220 Benchmark with Mistral-7B + Conjunctive Aggregator
print("\nRunning Full 220 Benchmark with Mistral-7B (Conjunctive Aggregator)...")
t0 = time.time()
mistral_full_traces = []
for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    
    sub_verdicts = []
    for sdata in cache["backstory_subs"]:
        sub = sdata["sub_claim"]
        ent = sdata["entity"]
        ev = sdata["evidence"]
        p = prompt_3way(sub, ev, ent, use_canonical=True)
        resp = query_llm_model("mistral:7b", p, seed=42)
        v = parse_verdict(resp)
        sub_verdicts.append(v)
        
    pred = aggregate_conjunctive(sub_verdicts)
    mistral_full_traces.append({"id": cid, "gt": gt, "pred": pred})

el_2b = time.time() - t0
print(f"Executed Full Mistral-7B 220 Benchmark in {el_2b:.1f}s")
m_mistral = compute_metrics(mistral_full_traces, "Task 2: Mistral-7B + Conjunctive Aggregator")

print("\nMISTRAL-7B FULL BENCHMARK SCORECARD VS BASELINE:")
print(f"{'Configuration':<45} | {'Acc (%)':<8} | {'Supp F1':<8} | {'Cont F1':<8} | {'NotM F1':<8} | {'Halluc-Supp Rate':<18} | {'Abstain Rate'}")
print("-" * 125)
print(f"{m_baseline['name']:<45} | {m_baseline['acc']:<6.2f}% | {m_baseline['class_stats']['SUPPORT']['f1']:<6.2f}% | {m_baseline['class_stats']['CONTRADICT']['f1']:<6.2f}% | {m_baseline['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {m_baseline['halluc_supp']}/131 ({m_baseline['halluc_rate']:.2f}%)  | {m_baseline['abstain_rate']:.2f}%")
print(f"{m_mistral['name']:<45} | {m_mistral['acc']:<6.2f}% | {m_mistral['class_stats']['SUPPORT']['f1']:<6.2f}% | {m_mistral['class_stats']['CONTRADICT']['f1']:<6.2f}% | {m_mistral['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {m_mistral['halluc_supp']}/131 ({m_mistral['halluc_rate']:.2f}%)  | {m_mistral['abstain_rate']:.2f}%")

# =========================================================================
# TASK 3: SCOPED CHUNK TRACE (DRACULA & BASKERVILLES RETRIEVAL FAILURES)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 3: SCOPED CHUNK TRACE FOR DRACULA (8 CLAIMS) & BASKERVILLES (5 CLAIMS)  ")
print("=" * 90)

# Dracula Retrieval Failures (8 claims) & Baskervilles Retrieval Failures (5 claims)
scoped_claims_to_trace = [
    # Dracula (8 claims)
    120, 121, 130, 186, 187, 189, 199, 205,
    # Baskervilles (5 claims)
    114, 115, 147, 148, 150
]

trace_results = []
for cid in scoped_claims_to_trace:
    item = next(it for it in dataset if it["id"] == cid)
    bname = item["book"]
    claim = item["user_input"]
    ref = item.get("reference", "")
    gt = item["ground_truth_verdict"]
    cache = retrieval_cache[cid]
    
    # 1. Inspect Top-8 chunks fed to model
    top8_texts = []
    for s in cache["backstory_subs"]:
        for e in s["evidence"]:
            top8_texts.append(e["text"])
    top8_combined = " ".join(top8_texts).lower()
    
    # 2. Inspect Full Candidate Pool (All global + entity chunks before reranking, ~35-50 chunks)
    full_pool_texts = []
    for s in cache["backstory_subs"]:
        for e in s.get("all_candidates", []):
            full_pool_texts.append(e["text"])
    full_pool_combined = " ".join(full_pool_texts).lower()
    
    # Check if reference facts exist in Top-8 vs Full Pool
    ref_keywords = [w.lower().strip(".,!?:;\"'") for w in ref.split() if len(w) > 3 and w.lower() not in {"this", "that", "with", "from", "were", "been", "died", "claim", "novel"}]
    in_top8 = sum(1 for kw in ref_keywords if kw in top8_combined) >= 2
    in_full_pool = sum(1 for kw in ref_keywords if kw in full_pool_combined) >= 2
    
    if in_top8:
        diagnosis = "PRESENT_IN_TOP_8 (Model Reasoning Error)"
    elif in_full_pool:
        diagnosis = "PRESENT_IN_CANDIDATE_POOL_BUT_DROPPED_BY_RERANKER"
    else:
        diagnosis = "ABSENT_FROM_RETRIEVAL_POOL (Entity/Index Miss)"
        
    trace_results.append({
        "id": cid,
        "book": bname,
        "claim": claim,
        "reference": ref,
        "diagnosis": diagnosis
    })
    print(f"[{cid:03d}] {bname:<30} | {diagnosis}")
    print(f"      Claim: {claim[:80]}...")
    print(f"      Ref  : {ref}")
    print("-" * 115)

# Save entire payload
all_tasks_payload = {
    "task1_conjunctive": {k: v for k, v in m_conj.items() if k != "matrix"},
    "task1_whole_claim": {k: v for k, v in m_whole.items() if k != "matrix"},
    "task2_mistral_fp_flips": mistral_fp_flips,
    "task2_mistral_full": {k: v for k, v in m_mistral.items() if k != "matrix"},
    "task3_scoped_trace": trace_results
}

with open("benchmark/all_three_tasks_results.json", "w", encoding="utf-8") as f:
    json.dump(all_tasks_payload, f, ensure_ascii=False, indent=2)

print("\nSaved all results to 'benchmark/all_three_tasks_results.json'.")
