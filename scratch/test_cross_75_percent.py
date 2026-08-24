import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from aggregation import aggregate_results

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def prompt_generation(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are an expert fact-checking judge evaluating a backstory claim against source evidence excerpts from a novel.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The evidence confirms the claim's core facts (the character, role, and actions/events).
2. CONTRADICT: The claim asserts a role, action, or outcome that conflicts with the evidence.
   - Example 1: If the claim says Major MacNabb was captain of the yacht Duncan, but evidence shows Captain John Mangles was the captain (and MacNabb was Lord Glenarvan's cousin) -> Verdict: CONTRADICT.
   - Example 2: If the claim says Captain Leclère safely commanded the ship into harbor, but evidence shows Captain Leclère died of fever at sea and Dantès brought the ship in -> Verdict: CONTRADICT.
   - Example 3: If the claim says Fernand was a wealthy merchant from Paris, but evidence shows he was a Catalan fisherman or that M. Morrel was the merchant -> Verdict: CONTRADICT.
   - Example 4: If the claim says Ayrton was the loyal first mate who built the yacht, but evidence shows he was a convict/traitor Ben Joyce -> Verdict: CONTRADICT.
   - Example 5: If the claim says Mercédès married Dantès immediately, but evidence shows Dantès was arrested before marrying -> Verdict: CONTRADICT.
3. NOT MENTIONED: The specific asserted fact is completely unmentioned in the evidence.

Briefly verify the claim, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    return prompt

print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

correct = 0
total = len(dataset)

for s in dataset:
    rets = claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
print(f"VERIFIED TEST ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
