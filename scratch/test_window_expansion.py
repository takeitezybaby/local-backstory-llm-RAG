import sys
import json
import requests
import faiss
import numpy as np

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, normalize, subset_search
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

atomic_chunks = loadChunks("Data/atomicChunks.json")
parent_chunks = loadChunks("Data/chunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

# Map parent chunk id to parent text
parent_dict = {}
for p in parent_chunks:
    parent_dict[p["Chunk id"]] = p["text"]

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

def deep_global_search(query, faiss_index, metadata, target_book=None, top_n=15):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    k_search = 120 if target_book else 20
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

def window_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        claim_entity = extract_entity(claim, entity_index)
        matched_key = find_entity_in_index(claim_entity, entity_index)
        target_book = get_entity_book(matched_key, entity_index, metadata)
        
        global_results = deep_global_search(claim, faiss_index, metadata, target_book=target_book, top_n=15)
        
        entity_results = []
        if matched_key and matched_key in entity_index:
            entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata)
            
        seen_parents = set()
        combined = []
        for r in global_results[:10] + entity_results[:6]:
            pid = r.get("Parent Chunk id")
            if pid and pid in parent_dict:
                if pid not in seen_parents:
                    seen_parents.add(pid)
                    combined.append({
                        "text": parent_dict[pid],
                        "Book": r.get("Book"),
                        "Chapter": r.get("Chapter"),
                        "Parent Chunk id": pid
                    })
            else:
                t = r["text"].strip()
                if t not in seen_parents:
                    seen_parents.add(t)
                    combined.append(r)
                
        retrievals.append({
            "Claim": claim,
            "Entity": claim_entity,
            "Target_Book": target_book,
            "Evidence": combined[:8]
        })
    return retrievals

def prompt_generation_phi(claim, evidence_list, entity):
    top_evidence = evidence_list[:8]
    Evidence = "\n\n".join([f"Excerpt {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""<|user|>
You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The claim's core facts are confirmed true by the novel excerpts.
2. CONTRADICT: The claim contradicts facts in the novel excerpts (e.g. asserts someone was captain when excerpts show someone else was captain; asserts someone was a merchant when excerpts show they were a fisherman; asserts someone died vs arrived safely; asserts a character has different parents or background).
3. NOT MENTIONED: The asserted event/fact is completely unmentioned in the novel excerpts and no conflicting facts exist.

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
    rets = window_retrieval(s["user_input"], atomic_chunks, faiss_index, entity_index)
    verifs = []
    for r in rets:
        p = prompt_generation_phi(r["Claim"], r["Evidence"], r["Entity"])
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3.5:latest",
            "prompt": p,
            "stream": False,
            "options": {"num_ctx": 3072, "temperature": 0.0}
        }, timeout=60).json()["response"]
        verifs.append({"Claim": r["Claim"], "Verification_result": resp})
        
    agg = aggregate_results(verifs)
    pred = "SUPPORT" if agg["Final Verdict"] == "COMPATIBLE" else ("CONTRADICT" if agg["Final Verdict"] == "INCOMPATIBLE" else "NOT MENTIONED")
    match = (pred == s["ground_truth_verdict"])
    if match:
        correct += 1
    print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}", flush=True)

print("-" * 50)
print(f"PARENT-WINDOW PHI-3.5 ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
