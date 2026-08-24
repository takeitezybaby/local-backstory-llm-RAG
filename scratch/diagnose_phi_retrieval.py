import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from aggregation import aggregate_results, extract_single_verdict

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def prompt_generation_phi(claim, evidence_list, entity):
    top_evidence = evidence_list[:12]
    Evidence = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(top_evidence)])
    prompt = f"""<|user|>
You are an expert fact-checking evaluator checking a backstory claim against excerpts from a novel.

Claim: "{claim}"
Target Entity: "{entity}"

Evidence Excerpts from Novel:
{Evidence}

EVALUATION RULES:
1. SUPPORT: The core statement in the claim is confirmed true by the evidence excerpts.
2. CONTRADICT: The claim asserts a fact that contradicts or conflicts with the evidence excerpts (e.g. asserts someone was captain when someone else held that role; asserts someone was a Parisian merchant when they were a Catalan fisherman; asserts someone died vs commanded into port; asserts a character was someone's child when they have different parents).
3. NOT MENTIONED: The specific asserted fact or event is completely unmentioned in the evidence excerpts.

Evaluate carefully, then conclude on the very last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
    return prompt

for s in dataset:
    rets = claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
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
    print(f"Claim {s['id']:<2} | GT: {s['ground_truth_verdict']:<14} | Pred: {pred:<14} | {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"   Input: {s['user_input']}")
        print(f"   Reasoning: {verifs[0]['Verification_result'][:180].replace(chr(10), ' ')}")
