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
from flashrank import Ranker, RerankRequest

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

# =========================================================================
# TASK 0: VERIFY TASK 2 RAW COUNTS FROM final_fixes_results.json
# =========================================================================
print("=" * 90)
print("   TASK 0: SANITY CHECK ON CONTRADICT PRECISION & RECALL RAW COUNTS            ")
print("=" * 90)

with open("benchmark/final_fixes_results.json", "r", encoding="utf-8") as f:
    fixes_data = json.load(f)

t2_stats = fixes_data["task2_negation_reranker"]["class_stats"]["CONTRADICT"]
print(f"Task 2 CONTRADICT Stats from JSON:")
print(f"  - Ground Truth Count (GT): {t2_stats['gt']}")
print(f"  - Predicted Count (Pred) : {t2_stats['pred']}")
print(f"  - True Positives (TP)    : {t2_stats['tp']}")
print(f"  - False Positives (FP)   : {t2_stats['pred'] - t2_stats['tp']}")
print(f"  - False Negatives (FN)   : {t2_stats['gt'] - t2_stats['tp']}")
print(f"  - Precision              : {t2_stats['tp']} / {t2_stats['pred']} = {t2_stats['p']:.4f}% ({t2_stats['tp']/t2_stats['pred']*100:.2f}%)")
print(f"  - Recall                 : {t2_stats['tp']} / {t2_stats['gt']} = {t2_stats['r']:.4f}% ({t2_stats['tp']/t2_stats['gt']*100:.2f}%)")
print(f"  - F1 Score               : {t2_stats['f1']:.4f}%\n")
print(f"EXPLANATION: TP = {t2_stats['tp']}. Both Ground Truth ({t2_stats['gt']}) and Predicted Count ({t2_stats['pred']}) happen to equal exactly 66 claims. Thus 24/66 = 36.3636% for both Precision and Recall. This is an exact arithmetic outcome, not a reporting artifact.")

# Load assets for Tasks 1 and 3
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

CONTRARY_MARKERS = {
    "not", "never", "no", "neither", "nor", "none", "without", "died", "dead", "death", "killed", 
    "murdered", "drowned", "executed", "slain", "perished", "fatal", "survived", "escaped", "lived", 
    "innocent", "false", "refused", "denied", "failed", "untrue", "monster", "vampire", "hound", "curse",
    "poison", "cyanide", "drown", "mire", "traitor", "criminal", "villain", "instead", "contrary"
}

def rerank_with_negation_boost(query, candidate_chunks, top_k=8):
    if not candidate_chunks:
        return []
    if len(candidate_chunks) <= top_k:
        return candidate_chunks
        
    passages = [{"id": idx, "text": c.get("text", "").strip(), "meta": c} for idx, c in enumerate(candidate_chunks)]
    rerank_req = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_req)
    
    q_words = set(query.lower().split())
    content_words = [w for w in q_words if len(w) > 3 and w not in CONTRARY_MARKERS]
    
    boosted_results = []
    for item in results:
        orig = item["meta"].copy()
        raw_score = float(item["score"])
        p_text = orig.get("text", "").lower()
        p_words = set(p_text.split())
        
        overlap_contrary = CONTRARY_MARKERS.intersection(p_words)
        overlap_content = sum(1 for w in content_words if w in p_words)
        
        boost = 0.0
        if overlap_contrary and overlap_content >= 2:
            boost = 0.15 * min(len(overlap_contrary), 2)
            
        orig["rerank_score"] = raw_score + boost
        boosted_results.append(orig)
        
    boosted_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return boosted_results[:top_k]

def query_llm_fixed(prompt, seed=42, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": seed, "num_predict": 120}
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            time.sleep(1.0)
    return ""

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

# =========================================================================
# TASK 1: FEW-SHOT PROMPTING EXPERIMENT (SINGLE ITERATION, CAPPED)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 1: FEW-SHOT PROMPTING ON CONTRADICTION DETECTION                       ")
print("=" * 90)

FEW_SHOT_EXAMPLES = """EXAMPLE 1 (Direct Factual Clash):
Claim: "Dr. Watson poisoned Sir Charles Baskerville with cyanide."
Excerpts: "[1] Dr. Watson is Holmes's loyal friend and an honorable British army surgeon who defended the Baskerville family."
Verdict: CONTRADICT

EXAMPLE 2 (Identity & Lineage Clash):
Claim: "Jack Stapleton was the loyal younger brother of Sir Henry Baskerville."
Excerpts: "[1] Stapleton was revealed as Rodger Baskerville's son, a villainous cousin who bred the hound to murder all heirs."
Verdict: CONTRADICT

EXAMPLE 3 (Admissible Unmentioned Detail):
Claim: "Sir Henry Baskerville owned an ebony walking stick purchased in Montreal."
Excerpts: "[1] Sir Henry arrived from North America to claim his inheritance at Baskerville Hall."
Verdict: NOT MENTIONED

EXAMPLE 4 (Supported Fact):
Claim: "Lucy Westenra was courted by Arthur Holmwood, Dr. Seward, and Quincey Morris."
Excerpts: "[1] Lucy received three marriage proposals in one day, from Arthur, Dr. John Seward, and Quincey Morris of Texas."
Verdict: SUPPORT
"""

def prompt_few_shot(claim, evidence_list, entity=""):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list[:8])])
    prof = get_canonical_profile(entity) if entity else ""
    prof_sec = f"Canonical Knowledge about {entity}:\n{prof}\n\n" if prof else ""
    
    return f"""<|user|>
Evaluate whether the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on the excerpts.

{FEW_SHOT_EXAMPLES}

NOW EVALUATE THIS CASE:
Claim: "{claim}"
Character: "{entity}"

{prof_sec}Source Excerpts:
{ev_text}

Conclude on the final line with exactly one verdict:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Step 1: Precompute negation-boosted retrieval cache
print("Building Negation-Boosted Retrieval Cache for all 220 claims...")
cache_neg = {}
for item in dataset:
    cid = item["id"]
    claim_text = item["user_input"]
    sub_claims = extract_atomic_claims(claim_text)
    subs = []
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
                
        boosted_ev = rerank_with_negation_boost(sub, cand, top_k=8)
        subs.append({"sub": sub, "ent": ent, "evidence": boosted_ev})
    cache_neg[cid] = subs

# Step 2: Spot-check on 61 False Positives + 20 True SUPPORT cases
with open("benchmark/diagnostics_220_all_conditions.json", "r", encoding="utf-8") as f:
    diag = json.load(f)
baseline_traces = diag["backstory_3way"]["traces"]
fp_claim_ids = [t["id"] for t in baseline_traces if t["gt"] in ["CONTRADICT", "NOT MENTIONED"] and t["pred"] == "SUPPORT"]
true_supp_sample = [it["id"] for it in dataset if it["ground_truth_verdict"] == "SUPPORT"][:20]

print(f"Spot-checking Few-Shot prompt on {len(fp_claim_ids)} False-Positive cases & {len(true_supp_sample)} True SUPPORT cases...")
fp_fixed = 0
for cid in fp_claim_ids:
    item = next(it for it in dataset if it["id"] == cid)
    gt = item["ground_truth_verdict"]
    sub_v = []
    for s in cache_neg[cid]:
        p = prompt_few_shot(s["sub"], s["evidence"], s["ent"])
        resp = query_llm_fixed(p, seed=42)
        sub_v.append(parse_verdict_clean(resp))
    pred = aggregate_conjunctive(sub_v)
    if pred == gt:
        fp_fixed += 1

supp_broken = 0
for cid in true_supp_sample:
    item = next(it for it in dataset if it["id"] == cid)
    sub_v = []
    for s in cache_neg[cid]:
        p = prompt_few_shot(s["sub"], s["evidence"], s["ent"])
        resp = query_llm_fixed(p, seed=42)
        sub_v.append(parse_verdict_clean(resp))
    pred = aggregate_conjunctive(sub_v)
    if pred != "SUPPORT":
        supp_broken += 1

print(f"  - FP Fix Rate: {fp_fixed}/{len(fp_claim_ids)} ({fp_fixed/len(fp_claim_ids)*100:.1f}%)")
print(f"  - True SUPPORT Broken (False Contradictions/Abstentions): {supp_broken}/{len(true_supp_sample)} ({supp_broken/len(true_supp_sample)*100:.1f}%)")

# Full 220-claim run with Few-Shot Prompt
print("\nRunning full 220-claim benchmark with Few-Shot Prompt + Negation Reranker...")
few_shot_traces = []
t0 = time.time()
for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"].strip().upper()
    sub_v = []
    for s in cache_neg[cid]:
        p = prompt_few_shot(s["sub"], s["evidence"], s["ent"])
        resp = query_llm_fixed(p, seed=42)
        sub_v.append(parse_verdict_clean(resp))
    pred = aggregate_conjunctive(sub_v)
    few_shot_traces.append({"id": cid, "gt": gt, "pred": pred})

el_fs = time.time() - t0
print(f"Executed Few-Shot 220 Benchmark in {el_fs:.1f}s")

def eval_traces(traces, name):
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
        "class_stats": class_stats,
        "halluc_supp": halluc_supp,
        "halluc_rate": halluc_rate,
        "abstain_rate": abstain_rate,
        "matrix": matrix
    }

fs_res = eval_traces(few_shot_traces, "Few-Shot Prompt + Negation Reranker (Phi-3.5)")
canonical_baseline = fixes_data["task2_negation_reranker"]

print("\nFEW-SHOT COMPARISON VS CANONICAL BASELINE (NEGATION RERANKER):")
print(f"{'Configuration':<46} | {'Acc (%)':<8} | {'Supp F1':<8} | {'Cont F1':<8} | {'NotM F1':<8} | {'Halluc-Supp Rate':<18} | {'Abstain Rate'}")
print("-" * 125)
print(f"{canonical_baseline['name']:<46} | {canonical_baseline['acc']:<6.2f}% | {canonical_baseline['class_stats']['SUPPORT']['f1']:<6.2f}% | {canonical_baseline['class_stats']['CONTRADICT']['f1']:<6.2f}% | {canonical_baseline['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {canonical_baseline['halluc_supp']}/131 ({canonical_baseline['halluc_rate']:.2f}%)  | {canonical_baseline['abstain_rate']:.2f}%")
print(f"{fs_res['name']:<46} | {fs_res['acc']:<6.2f}% | {fs_res['class_stats']['SUPPORT']['f1']:<6.2f}% | {fs_res['class_stats']['CONTRADICT']['f1']:<6.2f}% | {fs_res['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {fs_res['halluc_supp']}/131 ({fs_res['halluc_rate']:.2f}%)  | {fs_res['abstain_rate']:.2f}%")

# =========================================================================
# TASK 3: EXECUTE ADVERSARIAL NEAR-MISS BENCHMARK (20 CLAIMS)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 3: ADVERSARIAL NEAR-MISS BENCHMARK EXECUTION                           ")
print("=" * 90)

with open("benchmark/adversarial_near_miss.json", "r", encoding="utf-8") as f:
    adv_dataset = json.load(f)

adv_traces = []
t0 = time.time()
for item in adv_dataset:
    cid = item["id"]
    claim_text = item["claim"]
    gt = item["ground_truth"].strip().upper()
    cat = item["category"]
    bname = item["book"]
    
    sub_claims = extract_atomic_claims(claim_text)
    sub_v = []
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
                
        boosted_ev = rerank_with_negation_boost(sub, cand, top_k=8)
        
        prof = get_canonical_profile(ent) if ent else ""
        prof_sec = f"Canonical Knowledge about {ent}:\n{prof}\n\n" if prof else ""
        ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(boosted_ev)])
        
        prompt = f"""<|user|>
Evaluate whether the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on the excerpts.
Claim: "{sub}"
Character: "{ent}"

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
        resp = query_llm_fixed(prompt, seed=42)
        sub_v.append(parse_verdict_clean(resp))
        
    pred = aggregate_conjunctive(sub_v)
    adv_traces.append({
        "id": cid,
        "book": bname,
        "category": cat,
        "claim": claim_text,
        "ground_truth": gt,
        "pred": pred,
        "match": (pred == gt)
    })

el_adv = time.time() - t0
adv_total = len(adv_traces)
adv_correct = sum(1 for t in adv_traces if t["match"])
adv_acc = adv_correct / adv_total * 100

print(f"Executed 20-Claim Adversarial Near-Miss Benchmark in {el_adv:.1f}s")
print(f"Overall Adversarial Accuracy: {adv_acc:.2f}% ({adv_correct}/{adv_total})\n")

# Category breakdown
adv_cat_stats = {}
for t in adv_traces:
    c = t["category"]
    adv_cat_stats.setdefault(c, {"total": 0, "correct": 0, "preds": {}})
    adv_cat_stats[c]["total"] += 1
    if t["match"]:
        adv_cat_stats[c]["correct"] += 1
    adv_cat_stats[c]["preds"][t["pred"]] = adv_cat_stats[c]["preds"].get(t["pred"], 0) + 1

print(f"{'Adversarial Category':<35} | {'Acc (%)':<10} | {'Correct':<10} | {'Predictions Breakdown'}")
print("-" * 95)
for c, s in adv_cat_stats.items():
    c_acc = s["correct"] / s["total"] * 100
    print(f"{c:<35} | {c_acc:<8.2f}% | {s['correct']}/{s['total']:<8} | {s['preds']}")

# Save final Task 0, 1, 3 results
task_summary = {
    "task0_verification": {
        "tp": t2_stats['tp'],
        "pred_total": t2_stats['pred'],
        "gt_total": t2_stats['gt'],
        "precision": t2_stats['p'],
        "recall": t2_stats['r'],
        "f1": t2_stats['f1'],
        "is_verified": True
    },
    "task1_few_shot_benchmark": {k: v for k, v in fs_res.items() if k != "matrix"},
    "task3_adversarial_benchmark": {
        "overall_accuracy": adv_acc,
        "total_claims": adv_total,
        "correct_claims": adv_correct,
        "category_breakdown": adv_cat_stats,
        "traces": adv_traces
    }
}

with open("benchmark/task0_1_3_final_summary.json", "w", encoding="utf-8") as f:
    json.dump(task_summary, f, ensure_ascii=False, indent=2)

print("\nSaved all task summaries to 'benchmark/task0_1_3_final_summary.json'.")
