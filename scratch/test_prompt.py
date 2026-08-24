import requests
import json

API = "http://localhost:11434/api/generate"

claim = "Lady Helena served as a military nurse in the Crimean War before marrying Lord Glenarvan."
entity = "lady helena"
evidence = """Evidence 1: Almost immediately Lady Helena , accompanied by the major , appeared in the court - yard , and flew to meet her husband .
Evidence 2: Lord Edward Glenarvan was on board with his young wife , Lady Helena , and one of his cousins , Major MacNabb .
Evidence 3: Lady Helena would have rushed towards her husband , but the major restrained her ."""

prompt = f"""[INST] You are an expert factual verification judge.
Your task is to verify if a Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based STRICTLY on the provided Evidence excerpts from a novel.

Claim: "{claim}"
Target Entity: "{entity}"

Evidence Excerpts:
{evidence}

Evaluation Criteria:
- SUPPORT: The evidence explicitly and directly confirms the specific actions/facts asserted in the claim. If any core fact (like an occupation, place, or event) is not in the text, it is NOT supported.
- CONTRADICT: The evidence directly contradicts the claim (e.g. states a different job, different relationship, or conflicting event).
- NOT MENTIONED: The evidence does NOT contain proof of the key action/fact in the claim (e.g. mentions Lady Helena, but says nothing about serving as a nurse in the Crimean War).

Reason step by step, then conclude with: "Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""

resp = requests.post(API, json={
    "model": "koesn/mistral-7b-instruct:latest",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 4096}
})

print("Raw model response:")
print(resp.json()["response"])
