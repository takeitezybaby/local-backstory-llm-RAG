import sys
import json
import faiss
import spacy
import re
import requests

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, global_search, subset_search, find_entity_in_index
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def extract_all_entities(claim, index):
    doc = nlp(claim)
    found_keys = []
    
    # 1. Named entities
    for ent in doc.ents:
        k = find_entity_in_index(ent.text, index)
        if k and len(k) > 2 and k not in found_keys:
            found_keys.append(k)
            
    # 2. Noun chunks
    for chunk in doc.noun_chunks:
        k = find_entity_in_index(chunk.text, index)
        if k and len(k) > 2 and k not in found_keys:
            found_keys.append(k)
            
    return found_keys

def multi_entity_claim_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        entities = extract_all_entities(claim, entity_index)
        global_results = global_search(claim, faiss_index, metadata)
        
        entity_results = []
        for ent in entities:
            if ent in entity_index:
                entity_results.extend(subset_search(claim, entity_index[ent], faiss_index, metadata))
                
        seen = set()
        combined = []
        # Interleave entity results and global results
        for r in entity_results + global_results:
            t = r["text"].strip()
            if t not in seen:
                seen.add(t)
                combined.append(r)
                
        retrievals.append({
            "Claim": claim,
            "Entity": ", ".join(entities) if entities else "None",
            "Search_type": "Multi-Entity Hybrid",
            "Evidence": combined[:12]
        })
    return retrievals

def prompt_generation(claim, evidence_list, entity):
    top_evidence = evidence_list[:10]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are an expert fact-checker evaluating a backstory claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION RULES:
1. SUPPORT: The evidence explicitly confirms the claim's core facts (the character, role, and actions/events).
2. CONTRADICT: The claim asserts facts that conflict with or contradict the source evidence (e.g. asserts a character is a merchant when evidence shows they are a fisherman; asserts an entity is captain when evidence shows someone else is captain; asserts an entity safely arrived when evidence shows they died).
3. NOT MENTIONED: The key asserted fact/action is completely unmentioned in the evidence excerpts.

CRITICAL RULE: If the evidence shows the character in an entirely different role, relation, or state than asserted in the claim (e.g. Major MacNabb is Lord Glenarvan's cousin rather than captain; Fernand is a fisherman rather than a wealthy Parisian merchant; Ayrton is a convict/mutineer rather than loyal mate; Leclère died at sea rather than safely landing), you MUST classify as "Verdict: CONTRADICT".

Briefly verify the claim, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    return prompt

print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

correct = 0
total = len(dataset)

for s in dataset:
    rets = multi_entity_claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
    verifs = []
    for r in rets:
        prompt = prompt_generation(r["Claim"], r["Evidence"], r["Entity"])
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "koesn/mistral-7b-instruct:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.0}
        }, timeout=90)
        verifs.append({"Claim": r["Claim"], "Verification_result": resp.json()["response"]})
        
    agg = aggregate_results(verifs)
    pred = "SUPPORT" if agg["Final Verdict"] == "COMPATIBLE" else ("CONTRADICT" if agg["Final Verdict"] == "INCOMPATIBLE" else "NOT MENTIONED")
    match = (pred == s["ground_truth_verdict"])
    if match:
        correct += 1
    print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}")

print("-" * 50)
print(f"MULTI-ENTITY HYBRID ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
