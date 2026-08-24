import requests
import json
import sys
import re
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, normalize, subset_search, strip_accents
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def get_all_entity_chunks(raw_entity, index):
    if not raw_entity or not index:
        return [], None
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    unacc = strip_accents(cleaned)
    all_chunks = set()
    best_key = None
    for key, chunk_list in index.items():
        key_unacc = strip_accents(key)
        if len(unacc) > 3 and (unacc in key_unacc or key_unacc in unacc):
            all_chunks.update(chunk_list)
            if not best_key:
                best_key = key
        elif unacc == key_unacc:
            all_chunks.update(chunk_list)
            best_key = key
    return sorted(list(all_chunks)), best_key

def get_entity_book(chunk_ids, metadata):
    if not chunk_ids:
        return None
    first_cid = chunk_ids[0]
    if first_cid < len(metadata):
        return metadata[first_cid].get("Book")
    return None

def deep_global_search(query, faiss_index, metadata, target_book=None, top_n=10):
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

def full_pipeline_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        claim_entity = extract_entity(claim, entity_index)
        entity_chunks, matched_key = get_all_entity_chunks(claim_entity, entity_index)
        target_book = get_entity_book(entity_chunks, metadata)
        
        # 1. Global semantic search
        global_results = deep_global_search(claim, faiss_index, metadata, target_book=target_book, top_n=8)
        
        # 2. Entity subset search
        entity_results = []
        if entity_chunks:
            entity_results = subset_search(claim, entity_chunks, faiss_index, metadata)
            
        # 3. Entity Anchor Chunks (introductory chunks for character role definition)
        anchor_results = []
        if entity_chunks:
            for cid in entity_chunks[:3]:
                if cid < len(metadata):
                    cd = metadata[cid]
                    anchor_results.append({
                        "Score": 1.0,
                        "text": cd["text"],
                        "Book": cd.get("Book"),
                        "Chapter": cd.get("Chapter")
                    })
                    
        # Combine without duplicates: Global top 8 + Entity top 4 + Anchor top 2
        seen_texts = set()
        combined = []
        for r in global_results[:8] + entity_results[:4] + anchor_results[:2]:
            t = r["text"].strip()
            if t not in seen_texts:
                seen_texts.add(t)
                combined.append(r)
                
        retrievals.append({
            "Claim": claim,
            "Entity": claim_entity,
            "Evidence": combined[:12]
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
1. SUPPORT: The claim is confirmed as true by the evidence excerpts.
2. CONTRADICT: The claim contradicts facts in the evidence excerpts. If the claim attributes a role, title, deed, parentage, or outcome to a character (e.g. asserts someone was captain, merchant, or arrived safely), but the evidence shows someone else held that role, someone else performed the deed, or the character died/had different background, this is an explicit conflict. Classify any such conflict as CONTRADICT.
3. NOT MENTIONED: Used ONLY when the asserted fact or action is completely unmentioned in the evidence excerpts and no conflicting facts exist.

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
    rets = full_pipeline_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
print(f"COMPLETE PHI-3.5 PIPELINE ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
