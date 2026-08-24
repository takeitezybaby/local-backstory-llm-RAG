import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

text7 = "Abbé Faria was an Italian priest imprisoned in the Château d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dantès."
text8 = "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain."

for text in [text7, text8]:
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    evs = rets[0]["Evidence"][:15]
    evidence_text = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(evs)])
    prompt = f"""[INST] You are an expert fact-checker evaluating a backstory claim against novel evidence excerpts.

Claim: "{text}"
Entity: "{rets[0]['Entity']}"

Evidence Excerpts:
{evidence_text}

EVALUATION RULES:
1. SUPPORT: The evidence explicitly confirms the claim's core facts (the character, role, and actions/events).
2. CONTRADICT: The claim asserts facts that conflict with or contradict the source evidence.
3. NOT MENTIONED: The key asserted fact/action is completely unmentioned in the evidence excerpts.

Briefly verify the claim, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
    
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "koesn/mistral-7b-instruct:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    }, timeout=90)
    print("\nClaim:", text)
    print("LLM Response:\n", resp.json()["response"])
