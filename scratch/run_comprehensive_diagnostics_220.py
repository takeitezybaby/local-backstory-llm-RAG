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
            return r.json().get("response", "").strip()
        except Exception:
            time.sleep(1.5)
    return ""

# Prompt A: Binary Closed-World
def prompt_binary(claim, evidence_list):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list[:8])])
    return f"""<|user|>
Evaluate if the Claim is TRUE (SUPPORT) or FALSE (CONTRADICT) based strictly on the excerpts.
Claim: "{claim}"
Source Excerpts:
{ev_text}

Conclude on the final line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT<|end|>
<|assistant|>"""

# Prompt B: 3-Way Open-World NLI
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

def parse_verdict_clean(raw_resp, is_binary=False):
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
                return "NOT MENTIONED" if not is_binary else "CONTRADICT"
        if up in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            if is_binary and up == "NOT MENTIONED":
                return "CONTRADICT"
            return up
    up_raw = raw_resp.upper()
    if "CONTRADICT" in up_raw:
        return "CONTRADICT"
    if not is_binary and "NOT MENTIONED" in up_raw:
        return "NOT MENTIONED"
    return "SUPPORT"

print("Loading Assets...")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} benchmark claims.\n")

# Step 1: Precompute Retrieval for all 220 claims to guarantee identical contexts for fair comparison
print("[1/3] Precomputing Retrieval Contexts for all 220 claims...")
retrieval_cache = {}

for idx, item in enumerate(dataset, 1):
    cid = item["id"]
    claim_text = item["user_input"]
    
    # 1. Vanilla Retrieval (512-char global top-5)
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
    if idx % 50 == 0 or idx == len(dataset):
        print(f"  Precomputed {idx}/220 claim contexts...")

print("\n[2/3] Executing All 4 Experimental Conditions across 220 claims with ThreadPoolExecutor...")

def evaluate_single_claim_condition(item, condition):
    cid = item["id"]
    claim_text = item["user_input"]
    gt = item["ground_truth_verdict"].strip().upper()
    cache = retrieval_cache[cid]
    
    if condition == "vanilla_binary":
        evidence = cache["vanilla_evidence"]
        p = prompt_binary(claim_text, evidence)
        resp = query_llm(p)
        pred = parse_verdict_clean(resp, is_binary=True)
        return {"id": cid, "gt": gt, "pred": pred, "resp": resp, "evidence": [e["text"] for e in evidence]}
        
    elif condition == "vanilla_3way":
        evidence = cache["vanilla_evidence"]
        p = prompt_3way(claim_text, evidence, "", use_canonical=False)
        resp = query_llm(p)
        pred = parse_verdict_clean(resp, is_binary=False)
        return {"id": cid, "gt": gt, "pred": pred, "resp": resp, "evidence": [e["text"] for e in evidence]}
        
    elif condition == "backstory_binary":
        sub_verifs = []
        all_ev = []
        for sdata in cache["backstory_subs"]:
            sub = sdata["sub_claim"]
            ev = sdata["evidence"]
            p = prompt_binary(sub, ev)
            resp = query_llm(p)
            v = parse_verdict_clean(resp, is_binary=True)
            sub_verifs.append({"Claim": sub, "Verification_result": f"Verdict: {v}"})
            all_ev.extend([e["text"] for e in ev])
        # Aggregate binary
        c_cnt = sum(1 for sv in sub_verifs if "CONTRADICT" in sv["Verification_result"])
        pred = "CONTRADICT" if c_cnt > 0 else "SUPPORT"
        return {"id": cid, "gt": gt, "pred": pred, "resp": str(sub_verifs), "evidence": all_ev}
        
    elif condition == "backstory_3way": # Full System (Ours)
        sub_verifs = []
        all_ev = []
        for sdata in cache["backstory_subs"]:
            sub = sdata["sub_claim"]
            ent = sdata["entity"]
            ev = sdata["evidence"]
            p = prompt_3way(sub, ev, ent, use_canonical=True)
            resp = query_llm(p)
            v = parse_verdict_clean(resp, is_binary=False)
            sub_verifs.append({"Claim": sub, "Verification_result": f"Verdict: {v}"})
            all_ev.extend([e["text"] for e in ev])
            
        agg = aggregate_results(sub_verifs)
        pred_raw = agg["Final Verdict"]
        pred = "CONTRADICT" if "INCOMPATIBLE" in pred_raw else ("SUPPORT" if "COMPATIBLE" in pred_raw else "NOT MENTIONED")
        return {"id": cid, "gt": gt, "pred": pred, "resp": f"Verdict: {pred_raw}. Breakdown: {agg['Breakdown']}", "evidence": all_ev}

results_by_condition = {}

for cond_key, cond_name in [
    ("vanilla_binary", "Condition (a): Vanilla Retrieval + Binary Verdict"),
    ("vanilla_3way", "Condition (b): Vanilla Retrieval + 3-Way OW-NLI Verdict"),
    ("backstory_binary", "Condition (c): Backstory Retrieval + Binary Verdict"),
    ("backstory_3way", "Condition (d): Backstory Retrieval + Persona + 3-Way OW-NLI (Full System)")
]:
    print(f"\nRunning {cond_name}...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        cond_traces = list(executor.map(lambda it: evaluate_single_claim_condition(it, cond_key), dataset))
    elapsed = time.time() - t0
    print(f"Finished {cond_name} in {elapsed:.1f}s (Avg {elapsed/len(dataset):.2f}s/claim)")
    results_by_condition[cond_key] = {
        "name": cond_name,
        "elapsed_sec": elapsed,
        "traces": cond_traces
    }

# Save all raw experimental traces
with open("benchmark/diagnostics_220_all_conditions.json", "w", encoding="utf-8") as f:
    json.dump(results_by_condition, f, ensure_ascii=False, indent=2)

print("\nSaved all 4-condition traces to 'benchmark/diagnostics_220_all_conditions.json'.")
