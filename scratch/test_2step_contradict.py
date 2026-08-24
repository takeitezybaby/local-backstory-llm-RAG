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

contradict_claims = [
    (9, "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."),
    (10, "Ayrton was the loyal first mate of Lord Glenarvan who originally built the yacht Duncan in Glasgow."),
    (11, "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon."),
    (14, "Mercédès married Edmond Dantès immediately after the Pharaon docked in Marseilles on February 24, 1815.")
]

def run_2step_prompt(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    evidence_text = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{evidence_text}

Follow this exact decision tree:
1. Is the claim explicitly confirmed as true by the evidence excerpts? If yes -> output "Verdict: SUPPORT".
2. If NOT supported, check if the evidence shows the character in a DIFFERENT role/relation/event than asserted in the claim (e.g. Major MacNabb is Lord Glenarvan's cousin rather than captain; someone else is captain; Fernand is a fisherman rather than a merchant; Leclère died at sea rather than commanding into harbor; Ayrton is a convict/mutineer). If there is ANY such factual conflict -> output "Verdict: CONTRADICT".
3. If the asserted fact/action is completely absent without any conflicting mention -> output "Verdict: NOT MENTIONED".

Reason briefly, then conclude on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "koesn/mistral-7b-instruct:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    }, timeout=90)
    return resp.json()["response"]

print("Testing 2-Step Decision Tree on CONTRADICT Claims:")
for cid, text in contradict_claims:
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    raw = run_2step_prompt(rets[0]["Claim"], rets[0]["Evidence"], rets[0]["Entity"])
    verdict = extract_single_verdict(raw)
    print(f"\nClaim {cid}: {text[:60]}...")
    print(f"Predicted Verdict: {verdict} | {'PASS' if verdict == 'CONTRADICT' else 'FAIL'}")
    print("Reasoning snippet:", raw[:160].replace("\n", " "))
