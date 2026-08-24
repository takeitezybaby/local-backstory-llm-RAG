import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from aggregation import extract_single_verdict

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def test_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(evidence_list[:8])])
    prompt = f"""[INST] You are a strict, objective fact-checking judge evaluating a backstory claim against novel excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{evidence_text}

STRICT EVALUATION RULES:
1. SUPPORT: The evidence explicitly confirms BOTH the entity AND the exact specific action, occupation, date, or event asserted in the claim. The mere appearance of the character's name is NOT ENOUGH.
2. CONTRADICT: The evidence refutes the claim or states conflicting facts (e.g., states a different occupation, different father/relative, wrong title, or opposite outcome).
3. NOT MENTIONED: The evidence does NOT contain information about the specific asserted action/event, even if the character is mentioned in other contexts.

First, identify what specific fact/action is claimed.
Second, check if that specific fact/action is present in the evidence.
Finally, conclude your response on the last line with exactly:
Verdict: SUPPORT, Verdict: CONTRADICT, or Verdict: NOT MENTIONED [/INST]"""
    
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "koesn/mistral-7b-instruct:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    })
    return resp.json()["response"]

test_cases = [
    (10, "Ayrton was the loyal first mate of Lord Glenarvan who originally built the yacht Duncan in Glasgow."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon."),
    (16, "Lady Helena served as a military nurse in the Crimean War before marrying Lord Glenarvan."),
    (17, "Thalcave visited London in 1855 to work as an interpreter for the Royal Geographical Society.")
]

for cid, text in test_cases:
    print(f"\n================ CLAIM {cid} ================")
    print("Claim:", text)
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    for r in rets:
        raw = test_prompt(r["Claim"], r["Evidence"], r["Entity"])
        print("\nLLM Output:\n", raw.strip())
        print("\nExtracted Verdict:", extract_single_verdict(raw))
