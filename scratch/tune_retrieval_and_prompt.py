import sys
import json
import requests
import faiss
import re

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings
from querySearch import loadEntityIndex, extract_entity, normalize, subset_search, strip_accents
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def smart_find_entity_keys(raw_entity, index):
    if not raw_entity or not index:
        return []
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    unacc = strip_accents(cleaned)
    words = [w for w in re.split(r"\s+", unacc) if len(w) > 3 and w not in {"lord", "lady", "captain", "major", "french", "scottish"}]
    
    matched_keys = []
    # 1. Exact match
    for k in index:
        k_unacc = strip_accents(k)
        if unacc == k_unacc:
            matched_keys.append(k)
            
    # 2. Word matches
    for k in index:
        if k in matched_keys:
            continue
        k_unacc = strip_accents(k)
        for w in words:
            if w in k_unacc.split() or k_unacc == w:
                matched_keys.append(k)
                break
    return matched_keys

def get_entity_book_from_keys(keys, entity_index, metadata):
    for k in keys:
        cids = entity_index.get(k, [])
        if cids:
            first_cid = cids[0]
            if first_cid < len(metadata):
                return metadata[first_cid].get("Book")
    return None

def deep_global_search(query, faiss_index, metadata, target_book=None, top_n=15):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    k_search = 150 if target_book else 30
    scores, indices = faiss_index.search(query_embed, k_search)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        Parentdata = metadata[idx]
        if target_book and Parentdata.get("Book") != target_book:
            continue
        results.append({
            "Score": float(score),
            "text": Parentdata["text"],
            "Book": Parentdata["Book"],
            "Chapter": Parentdata["Chapter"],
            "Parent Chunk id": Parentdata["Parent Chunk id"],
            "Atomic id": Parentdata["Atomic id"]
        })
        if len(results) >= top_n:
            break
    return results

def optimized_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        claim_entity = extract_entity(claim, entity_index)
        matched_keys = smart_find_entity_keys(claim_entity, entity_index)
        target_book = get_entity_book_from_keys(matched_keys, entity_index, metadata)
        
        # 1. Deep Global Search (book filtered)
        global_results = deep_global_search(claim, faiss_index, metadata, target_book=target_book, top_n=12)
        
        # 2. Entity Subset Search for all matched keys
        entity_results = []
        for k in matched_keys[:2]:
            cids = entity_index.get(k, [])
            if cids:
                entity_results.extend(subset_search(claim, cids, faiss_index, metadata))
                
        # Deduplicate & rank
        seen_texts = set()
        combined = []
        for r in global_results[:10] + entity_results[:6]:
            t = r["text"].strip()
            if t not in seen_texts:
                seen_texts.add(t)
                combined.append(r)
                
        retrievals.append({
            "Claim": claim,
            "Entity": claim_entity,
            "Target_Book": target_book,
            "Evidence": combined[:14]
        })
    return retrievals

def prompt_generation_phi(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""<|user|>
You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The core statement in the claim is confirmed true by the evidence excerpts.
2. CONTRADICT: The claim directly contradicts source evidence (e.g. asserts someone was captain when someone else held that title; asserts someone was a Parisian merchant when they were a Catalan fisherman; asserts someone died vs survived/commanded ship; asserts different parents).
3. NOT MENTIONED: The specific asserted fact or event is completely unmentioned in the evidence excerpts and no conflicting source evidence exists (e.g. claims someone served in Parliament or was a Crimean war nurse, but the text never mentions such a position or period).

Evaluate concisely, then conclude on the very last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
    return prompt

print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

correct = 0
total = len(dataset)

for s in dataset:
    rets = optimized_retrieval(s["user_input"], chunks, faiss_index, entity_index)
    verifs = []
    for r in rets:
        p = prompt_generation_phi(r["Claim"], r["Evidence"], r["Entity"])
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3.5:latest",
            "prompt": p,
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.0}
        }, timeout=60).json()["response"]
        verifs.append({"Claim": r["Claim"], "Verification_result": resp})
        
    agg = aggregate_results(verifs)
    pred = "SUPPORT" if agg["Final Verdict"] == "COMPATIBLE" else ("CONTRADICT" if agg["Final Verdict"] == "INCOMPATIBLE" else "NOT MENTIONED")
    match = (pred == s["ground_truth_verdict"])
    if match:
        correct += 1
    print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}", flush=True)

print("-" * 50)
print(f"OPTIMIZED PHI-3.5 PIPELINE ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
