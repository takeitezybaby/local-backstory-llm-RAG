import requests
import json
import sys
import re
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, global_search, subset_search
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def get_entity_book(matched_key, entity_index, metadata):
    if not matched_key or matched_key not in entity_index:
        return None
    cids = entity_index[matched_key]
    if not cids:
        return None
    first_cid = cids[0]
    if first_cid < len(metadata):
        return metadata[first_cid].get("Book")
    return None

def book_aware_claim_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        claim_entity = extract_entity(claim, entity_index)
        matched_key = find_entity_in_index(claim_entity, entity_index)
        target_book = get_entity_book(matched_key, entity_index, metadata)
        
        raw_global = global_search(claim, faiss_index, metadata)
        if target_book:
            filtered_global = [r for r in raw_global if r.get("Book") == target_book]
        else:
            filtered_global = raw_global
            
        entity_results = []
        if matched_key and matched_key in entity_index:
            entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata)
            
        seen_texts = set()
        combined = []
        for r in filtered_global[:10] + entity_results[:5]:
            t = r["text"].strip()
            if t not in seen_texts:
                seen_texts.add(t)
                combined.append(r)
                
        retrievals.append({
            "Claim": claim,
            "Entity": claim_entity,
            "Target_Book": target_book,
            "Evidence": combined[:15]
        })
    return retrievals

def prompt_generation_phi(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""<|user|>
You are an expert fact-checker evaluating a backstory claim against source novel excerpts.

Claim: "{claim}"
Entity: "{entity}"

Source Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The claim is confirmed true by the source excerpts.
2. CONTRADICT: The claim contradicts or conflicts with facts in the source excerpts (e.g. asserts someone was captain when the excerpts show someone else was captain; asserts someone was a merchant when excerpts show they were a fisherman; asserts someone died vs arrived safely; asserts a character has different parents).
3. NOT MENTIONED: The asserted event/fact is completely unmentioned in the source excerpts.

Evaluate concisely, then conclude on the last line with exactly:
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
    rets = book_aware_claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
    print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}")

print("-" * 50)
print(f"BOOK-AWARE PHI-3.5 ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
