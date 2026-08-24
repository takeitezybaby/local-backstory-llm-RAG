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

def run_predicate_prompt(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    evidence_text = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""[INST] You are a strict factual fact-checking judge evaluating a backstory claim against source evidence excerpts from a novel.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{evidence_text}

EVALUATION INSTRUCTIONS:
1. Identify the essential factual statements asserted in the claim (the character, their occupation/title, and the specific event/action).
2. Check if EVERY factual statement in the claim is explicitly backed by the evidence excerpts.
3. Decision Rules:
   - "Verdict: SUPPORT" : ONLY if ALL key facts in the claim (including occupations, events, and actions) are explicitly confirmed by the evidence excerpts. If ANY key fact in the claim is missing from the evidence, it CANNOT be SUPPORT.
   - "Verdict: CONTRADICT" : If the claim makes an assertion that conflicts with facts in the evidence (e.g. states a different occupation, wrong captain, wrong father, or opposite outcome).
   - "Verdict: NOT MENTIONED" : If a key fact/action in the claim is absent or unmentioned in the evidence excerpts.

Reason briefly, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "koesn/mistral-7b-instruct:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    }, timeout=90)
    return resp.json()["response"]

correct = 0
total = len(dataset)
print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
print("-" * 50)

for s in dataset:
    rets = claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
    verifs = []
    for r in rets:
        raw = run_predicate_prompt(r["Claim"], r["Evidence"], r["Entity"])
        verifs.append({"Claim": r["Claim"], "Verification_result": raw})
    
    agg = aggregate_results(verifs)
    pred = "SUPPORT" if agg["Final Verdict"] == "COMPATIBLE" else ("CONTRADICT" if agg["Final Verdict"] == "INCOMPATIBLE" else "NOT MENTIONED")
    match = (pred == s["ground_truth_verdict"])
    if match:
        correct += 1
    print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}")

print("-" * 50)
print(f"VERIFIED TEST ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
