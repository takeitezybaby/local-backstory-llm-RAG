import spacy
import re
import unicodedata
import json
import sys
import requests
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, global_search, subset_search, strip_accents
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def find_entity_in_index(raw_entity, index):
    if not raw_entity or not index:
        return None
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    if cleaned in index:
        return cleaned
    unaccented = strip_accents(cleaned)
    if unaccented in index:
        return unaccented
        
    sorted_keys = sorted(index.keys(), key=lambda x: len(x), reverse=True)
    for key in sorted_keys:
        if len(key) <= 3:
            continue
        key_unacc = strip_accents(key)
        if cleaned == key or unaccented == key_unacc:
            return key
        if cleaned in key or key in cleaned:
            return key
        if unaccented in key_unacc or key_unacc in unaccented:
            return key
    return None

def extract_primary_entity(query, index):
    doc = nlp(query)
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            subj_text = " ".join([t.text for t in token.subtree if not t.is_punct])
            k = find_entity_in_index(subj_text, index)
            if k:
                return k
            k = find_entity_in_index(token.text, index)
            if k:
                return k
                
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            k = find_entity_in_index(ent.text, index)
            if k:
                return k
                
    for ent in doc.ents:
        k = find_entity_in_index(ent.text, index)
        if k:
            return k
    return None

def subject_first_retrieval(backstory, metadata, faiss_index, entity_index):
    claims = extract_atomic_claims(backstory)
    retrievals = []
    for claim in claims:
        primary_ent = extract_primary_entity(claim, entity_index)
        global_results = global_search(claim, faiss_index, metadata)
        
        if primary_ent and primary_ent in entity_index:
            entity_results = subset_search(claim, entity_index[primary_ent], faiss_index, metadata)
            seen = set()
            combined = []
            for r in entity_results + global_results:
                t = r["text"].strip()
                if t not in seen:
                    seen.add(t)
                    combined.append(r)
            result = combined[:15]
            search_type = f"Hybrid (Entity '{primary_ent}' + Global)"
        else:
            result = global_results[:15]
            search_type = "Global-search"
            
        retrievals.append({
            "Claim": claim,
            "Entity": primary_ent if primary_ent else "None",
            "Search_type": search_type,
            "Evidence": result
        })
    return retrievals

def prompt_generation(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are an expert literary fact-checking judge evaluating a backstory claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The evidence confirms the claim's core facts (character identity, role, and actions/events). Minor differences in phrasing do not prevent support.
2. CONTRADICT: The claim asserts facts that conflict with or contradict the source evidence.
   - Refutation Examples: 
     * The claim says someone safely commanded the ship into port, but evidence states they died of fever at sea -> CONTRADICT.
     * The claim says someone was a wealthy Parisian merchant, but evidence states they were a poor Catalan fisherman -> CONTRADICT.
     * The claim says someone was the captain of the yacht, but evidence states they were Lord Glenarvan's cousin or that someone else was captain -> CONTRADICT.
     * The claim says someone was the loyal first mate who built the yacht, but evidence states they were a convict/traitor -> CONTRADICT.
     * The claim says two characters married immediately upon arrival, but evidence states they were separated by arrest at the betrothal feast -> CONTRADICT.
3. NOT MENTIONED: The specific asserted fact or event is completely absent and unmentioned in the evidence excerpts.

Reason briefly, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    return prompt

print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

correct = 0
total = len(dataset)

for s in dataset:
    rets = subject_first_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
print(f"VERIFIED SUBJECT-FIRST ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
