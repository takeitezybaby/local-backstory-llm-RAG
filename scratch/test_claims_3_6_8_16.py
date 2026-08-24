import sys
import json
import requests
import faiss
import numpy as np

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, normalize, subset_search
from claimExtraction import extract_atomic_claims

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

def deep_global_search(query, faiss_index, metadata, target_book=None, top_n=12):
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

def enhanced_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        claim_entity = extract_entity(claim, entity_index)
        matched_key = find_entity_in_index(claim_entity, entity_index)
        target_book = get_entity_book(matched_key, entity_index, metadata)
        
        global_results = deep_global_search(claim, faiss_index, metadata, target_book=target_book, top_n=12)
        
        entity_results = []
        if matched_key and matched_key in entity_index:
            entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata)
            
        seen_texts = set()
        combined = []
        for r in global_results[:10] + entity_results[:5]:
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
You are an expert fact-checker evaluating a claim against source novel excerpts.

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

for target_id in [3, 6, 8, 16]:
    s = [item for item in dataset if item["id"] == target_id][0]
    rets = enhanced_retrieval(s["user_input"], chunks, faiss_index, entity_index)
    print(f"\n================ CLAIM {target_id} (GT: {s['ground_truth_verdict']}) ================", flush=True)
    print(f"Input: {s['user_input']}", flush=True)
    print(f"Entity: {rets[0]['Entity']}", flush=True)
    print("Top 3 retrieved excerpts:", flush=True)
    for i, ev in enumerate(rets[0]["Evidence"][:3]):
        print(f"  [{i+1}] {ev['text']}", flush=True)
    
    p = prompt_generation_phi(rets[0]["Claim"], rets[0]["Evidence"], rets[0]["Entity"])
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "phi3.5:latest",
        "prompt": p,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    }, timeout=60).json()["response"]
    print(f"\nModel Output:\n{resp}\n", flush=True)
