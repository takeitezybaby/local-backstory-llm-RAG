import requests
import json

API = "http://localhost:11434/api/generate"

claim = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."
entity = "macnabb"
evidence = """Evidence 1: Lord Edward Glenarvan was on board with his young wife , Lady Helena , and one of his cousins , Major MacNabb .
Evidence 2: Tom obeyed ; and the bottle found under such singular circumstances was placed on the cabin - table , around which Lord Glenarvan , Major MacNabb , and Captain John Mangles took their seats .
Evidence 3: He is the captain of the Duncan , and must not , therefore , expose himself . At this moment the boat , commanded by Captain Mangles , started ."""

prompt = f"""[INST] You are an expert factual verification judge.
Your task is to verify if a Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based STRICTLY on the provided Evidence excerpts from a novel.

Claim: "{claim}"
Target Entity: "{entity}"

Evidence Excerpts:
{evidence}

Evaluation Criteria:
- SUPPORT: The evidence explicitly and directly confirms the specific actions/facts asserted in the claim. If any core fact (like an occupation, place, or event) is not in the text, it is NOT supported.
- CONTRADICT: The evidence directly contradicts the claim (e.g. claim says MacNabb is captain, but evidence identifies John Mangles as captain and MacNabb as cousin).
- NOT MENTIONED: The evidence does NOT contain proof to confirm or refute the key action/fact in the claim.

Reason step by step, then conclude with: "Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""

resp = requests.post(API, json={
    "model": "koesn/mistral-7b-instruct:latest",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 4096}
})

print("Raw model response:")
print(resp.json()["response"])
