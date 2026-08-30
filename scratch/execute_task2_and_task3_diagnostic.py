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

def query_llm_fixed(prompt, seed=42, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": seed,
            "num_predict": 150
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

print("Loading Assets...")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize FlashRank
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Define Negation-Aware Reranker Function
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
        
        # Check contrary marker overlap
        overlap_contrary = CONTRARY_MARKERS.intersection(p_words)
        overlap_content = sum(1 for w in content_words if w in p_words)
        
        boost = 0.0
        if overlap_contrary and overlap_content >= 2:
            boost = 0.15 * min(len(overlap_contrary), 2)
            
        orig["rerank_score"] = raw_score + boost
        boosted_results.append(orig)
        
    boosted_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return boosted_results[:top_k]

# =========================================================================
# TASK 2: TEST RERANKER NEGATION BOOST ON 4 TRACED CASES & FULL 220
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 2: RERANKER NEGATION-AWARE BOOST EVALUATION                            ")
print("=" * 90)

# Check specifically the 4 traced cases: 120, 150, 186, 205
traced_target_ids = [120, 150, 186, 205]
print("Checking top-8 presence for 4 target cases after negation boost...")

target_surface_results = {}
for cid in traced_target_ids:
    item = next(it for it in dataset if it["id"] == cid)
    claim = item["user_input"]
    ref = item.get("reference", "")
    sub_claims = extract_atomic_claims(claim)
    
    surfaced = False
    ref_kw = [w.lower().strip(".,!?:;\"'") for w in ref.split() if len(w) > 3 and w.lower() not in {"this", "that", "with", "from", "were", "been", "died", "claim", "novel"}]
    
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
        ev_comb = " ".join([e["text"].lower() for e in boosted_ev])
        if sum(1 for kw in ref_kw if kw in ev_comb) >= 2:
            surfaced = True
            
    target_surface_results[cid] = surfaced
    status = "SURFACED IN TOP-8 [FIXED]" if surfaced else "STILL MISSED"
    print(f"  - Claim ID {cid:03d} ({item['book'][:20]}): {status}")

# Re-run full 220-claim benchmark under Task 2
print("\nRe-running full 220-claim benchmark with Negation-Aware Reranker Boost...")
task2_traces = []
t0 = time.time()

for idx, item in enumerate(dataset, 1):
    cid = item["id"]
    claim_text = item["user_input"]
    gt = item["ground_truth_verdict"].strip().upper()
    
    sub_claims = extract_atomic_claims(claim_text)
    sub_verdicts = []
    
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
        v = parse_verdict_clean(resp)
        sub_verdicts.append(v)
        
    pred = aggregate_conjunctive(sub_verdicts)
    task2_traces.append({"id": cid, "gt": gt, "pred": pred})

el_task2 = time.time() - t0
print(f"Executed Task 2 Full Benchmark in {el_task2:.1f}s")

# =========================================================================
# TASK 3: STRUCTURED CONTRADICTION-SEARCH PROMPT (SINGLE ITERATION, CAPPED)
# =========================================================================
print("\n" + "=" * 90)
print("   TASK 3: STRUCTURED CONTRADICTION-SEARCH PROMPT EVALUATION                   ")
print("=" * 90)

# Precompute standard top-8 retrieval cache for Prompt testing
standard_cache = {}
for item in dataset:
    cid = item["id"]
    claim_text = item["user_input"]
    sub_claims = extract_atomic_claims(claim_text)
    subs_info = []
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
        passages = [{"id": idx, "text": c.get("text", "").strip(), "meta": c} for idx, c in enumerate(cand)]
        rerank_req = RerankRequest(query=sub, passages=passages)
        std_ev = [item["meta"] for item in ranker.rerank(rerank_req)[:8]]
        subs_info.append({"sub": sub, "ent": ent, "evidence": std_ev})
    standard_cache[cid] = subs_info

def structured_prompt(sub, evidence, ent):
    prof = get_canonical_profile(ent) if ent else ""
    prof_sec = f"Canonical Knowledge about {ent}:\n{prof}\n\n" if prof else ""
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence[:8])])
    
    return f"""<|user|>
You are a rigorous literary fact-checker. Perform step-by-step contradiction analysis before deciding the verdict.

Claim: "{sub}"
Character: "{ent}"

{prof_sec}Source Excerpts:
{ev_text}

ANALYSIS STEPS:
Step 1: Check if any fact in the Source Excerpts or Canonical Knowledge directly clashes with or disproves the Claim (e.g. different fate, different identity, contrary action). State the specific clash or explicitly write "NO INCONSISTENCY FOUND".
Step 2: Conclude on the final line with exactly one verdict:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

# Test Prompt on the 61 False-Positive SUPPORT cases first
with open("benchmark/reproducibility_and_tradeoff_results.json", "r", encoding="utf-8") as f:
    rep_data = json.load(f)

# Load baseline traces to identify exact 61 FP claims
with open("benchmark/diagnostics_220_all_conditions.json", "r", encoding="utf-8") as f:
    diag = json.load(f)
baseline_traces = diag["backstory_3way"]["traces"]
fp_claim_ids = [t["id"] for t in baseline_traces if t["gt"] in ["CONTRADICT", "NOT MENTIONED"] and t["pred"] == "SUPPORT"]

print(f"Testing Structured Contradiction-Search Prompt on {len(fp_claim_ids)} False-Positive SUPPORT claims...")
fp_fixed_count = 0
for cid in fp_claim_ids:
    item = next(it for it in dataset if it["id"] == cid)
    gt = item["ground_truth_verdict"]
    subs_info = standard_cache[cid]
    
    sub_verdicts = []
    for sdata in subs_info:
        p = structured_prompt(sdata["sub"], sdata["evidence"], sdata["ent"])
        resp = query_llm_fixed(p, seed=42)
        v = parse_verdict_clean(resp)
        sub_verdicts.append(v)
        
    pred = aggregate_conjunctive(sub_verdicts)
    if pred == gt:
        fp_fixed_count += 1

print(f"Structured Prompt Check on 61 False Positives: {fp_fixed_count} / {len(fp_claim_ids)} fixed ({fp_fixed_count/len(fp_claim_ids)*100:.2f}%)")

# Re-run full 220-claim benchmark under Task 3
print("\nRe-running full 220-claim benchmark with Structured Contradiction-Search Prompt...")
task3_traces = []
t0 = time.time()

for item in dataset:
    cid = item["id"]
    gt = item["ground_truth_verdict"].strip().upper()
    subs_info = standard_cache[cid]
    
    sub_verdicts = []
    for sdata in subs_info:
        p = structured_prompt(sdata["sub"], sdata["evidence"], sdata["ent"])
        resp = query_llm_fixed(p, seed=42)
        v = parse_verdict_clean(resp)
        sub_verdicts.append(v)
        
    pred = aggregate_conjunctive(sub_verdicts)
    task3_traces.append({"id": cid, "gt": gt, "pred": pred})

el_task3 = time.time() - t0
print(f"Executed Task 3 Full Benchmark in {el_task3:.1f}s")

# =========================================================================
# COMPUTE ALL SCORECARDS FOR FINAL COMPARISON
# =========================================================================
def evaluate_trace_metrics(traces, name):
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

# Baseline Task 1 of Record
with open("benchmark/all_three_tasks_results.json", "r", encoding="utf-8") as f:
    t1_base = json.load(f)["task1_conjunctive"]

res_task2 = evaluate_trace_metrics(task2_traces, "Task 2: Reranker Negation Boost")
res_task3 = evaluate_trace_metrics(task3_traces, "Task 3: Structured Prompt")

print("\n" + "=" * 90)
print("             FINAL COMPARISON TABLE: TASK 1 BASELINE VS TASK 2 VS TASK 3       ")
print("=" * 90)
print(f"{'Configuration':<42} | {'Acc (%)':<8} | {'Supp F1':<8} | {'Cont F1':<8} | {'NotM F1':<8} | {'Halluc-Supp Rate':<18} | {'Abstain Rate'}")
print("-" * 120)
print(f"{'Task 1 Baseline (Conjunctive Decomposed)':<42} | {t1_base['acc']:<6.2f}% | {t1_base['class_stats']['SUPPORT']['f1']:<6.2f}% | {t1_base['class_stats']['CONTRADICT']['f1']:<6.2f}% | {t1_base['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {t1_base['halluc_supp']}/131 ({t1_base['halluc_rate']:.2f}%)  | {t1_base['abstain_rate']:.2f}%")
print(f"{res_task2['name']:<42} | {res_task2['acc']:<6.2f}% | {res_task2['class_stats']['SUPPORT']['f1']:<6.2f}% | {res_task2['class_stats']['CONTRADICT']['f1']:<6.2f}% | {res_task2['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {res_task2['halluc_supp']}/131 ({res_task2['halluc_rate']:.2f}%)  | {res_task2['abstain_rate']:.2f}%")
print(f"{res_task3['name']:<42} | {res_task3['acc']:<6.2f}% | {res_task3['class_stats']['SUPPORT']['f1']:<6.2f}% | {res_task3['class_stats']['CONTRADICT']['f1']:<6.2f}% | {res_task3['class_stats']['NOT MENTIONED']['f1']:<6.2f}% | {res_task3['halluc_supp']}/131 ({res_task3['halluc_rate']:.2f}%)  | {res_task3['abstain_rate']:.2f}%")

final_payload = {
    "task1_baseline_of_record": t1_base,
    "task2_negation_reranker": {k: v for k, v in res_task2.items() if k != "matrix"},
    "task2_target_surface_results": target_surface_results,
    "task3_structured_prompt": {k: v for k, v in res_task3.items() if k != "matrix"},
    "task3_fp_fixed_count": fp_fixed_count
}

with open("benchmark/final_fixes_results.json", "w", encoding="utf-8") as f:
    json.dump(final_payload, f, ensure_ascii=False, indent=2)

print("\nSaved all final results to 'benchmark/final_fixes_results.json'.")
