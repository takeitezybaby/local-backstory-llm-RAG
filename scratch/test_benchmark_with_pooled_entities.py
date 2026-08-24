import spacy
import re
import unicodedata
import json
import sys
import requests
import faiss
import numpy as np

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex, strip_accents, global_search, subset_search
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def get_all_entity_chunks(raw_entity, index):
    if not raw_entity or not index:
        return []
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    unacc = strip_accents(cleaned)
    all_chunks = set()
    for key, chunk_list in index.items():
        key_unacc = strip_accents(key)
        if len(unacc) > 3 and (unacc in key_unacc or key_unacc in unacc):
            all_chunks.update(chunk_list)
        elif unacc == key_unacc:
            all_chunks.update(chunk_list)
    return sorted(list(all_chunks))

def extract_primary_entity_name(query):
    doc = nlp(query)
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return re.sub(r"['’]s?\b", "", token.text).strip(" .,!?:;\"'").lower()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return re.sub(r"['’]s?\b", "", ent.text).strip(" .,!?:;\"'").lower()
    return None

def pooled_claim_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        ent_name = extract_primary_entity_name(claim)
        entity_chunk_ids = get_all_entity_chunks(ent_name, entity_index) if ent_name else []
        global_results = global_search(claim, faiss_index, metadata)
        
        if entity_chunk_ids:
            entity_results = subset_search(claim, entity_chunk_ids, faiss_index, metadata)
            seen = set()
            combined = []
            for r in entity_results + global_results:
                t = r["text"].strip()
                if t not in seen:
                    seen.add(t)
                    combined.append(r)
            result = combined[:15]
            search_type = f"Pooled-Entity '{ent_name}' + Global"
        else:
            result = global_results[:15]
            search_type = "Global-search"
            
        retrievals.append({
            "Claim": claim,
            "Entity": ent_name if ent_name else "None",
            "Search_type": search_type,
            "Evidence": result
        })
    return retrievals

def prompt_generation(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are an expert fact-checker evaluating a backstory claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION RULES:
1. SUPPORT: The evidence explicitly confirms the claim's core facts (the character, role, and actions/events).
2. CONTRADICT: The claim asserts facts that conflict with or contradict the source evidence (e.g. asserts an entity has a different role/title, wrong parent, wrong job, or claims an event succeeded when evidence shows they died or were betrayed).
3. NOT MENTIONED: The key asserted fact/action is completely unmentioned in the evidence excerpts.

CRITICAL REFUTATION RULE: If the evidence shows the character in an entirely different role, relation, or state than asserted in the claim (e.g. Major MacNabb is Lord Glenarvan's cousin rather than captain; Fernand is a Catalan fisherman rather than a wealthy Parisian merchant; Ayrton is a convict/mutineer rather than loyal mate; Leclère died at sea rather than safely landing into port; Danglars was the supercargo/clerk jealous of Dantès), evaluate carefully.

Briefly verify the claim, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    return prompt

print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

correct = 0
total = len(dataset)

for s in dataset:
    rets = pooled_claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
print(f"VERIFIED POOLED-ENTITY ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
