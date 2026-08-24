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
    prompt = f"""You are an expert fact-checking judge evaluating a backstory claim against source evidence excerpts from a novel.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The evidence explicitly confirms the core claim (the character, role, and actions/events).
2. CONTRADICT: The claim asserts facts that conflict with the evidence (e.g., claiming someone was captain when someone else was captain; claiming someone was a merchant when they were a fisherman; claiming someone arrived safely when they died).
3. NOT MENTIONED: The specific asserted fact or event is completely unmentioned in the evidence excerpts.

Reason step-by-step concisely, then conclude your answer on the last line with exactly:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED"""
    return prompt

def test_phi_benchmark():
    print(f"{'ID':<3} | {'GT':<14} | {'Predicted':<14} | {'Match?':<8}")
    print("-" * 50)
    correct = 0
    total = len(dataset)

    for s in dataset:
        rets = claim_retrieval(s["user_input"], chunks, faiss_index, entity_index)
        verifs = []
        for r in rets:
            prompt = prompt_generation_phi(r["Claim"], r["Evidence"], r["Entity"])
            resp = requests.post("http://localhost:11434/api/generate", json={
                "model": "phi3.5:latest",
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": 2048, "temperature": 0.0}
            }, timeout=60)
            verifs.append({"Claim": r["Claim"], "Verification_result": resp.json()["response"]})
            
        agg = aggregate_results(verifs)
        pred = "SUPPORT" if agg["Final Verdict"] == "COMPATIBLE" else ("CONTRADICT" if agg["Final Verdict"] == "INCOMPATIBLE" else "NOT MENTIONED")
        match = (pred == s["ground_truth_verdict"])
        if match:
            correct += 1
        print(f"{s['id']:<3} | {s['ground_truth_verdict']:<14} | {pred:<14} | {'PASS' if match else 'FAIL':<8}")

    print("-" * 50)
    print(f"PHI-3.5 BENCHMARK ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")

if __name__ == "__main__":
    test_phi_benchmark()
