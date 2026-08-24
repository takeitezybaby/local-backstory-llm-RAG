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

def test_checklist_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"Evidence {i+1}:\n{e['text']}" for i, e in enumerate(evidence_list[:10])])
    prompt = f"""[INST] You are a rigorous fact-checking judge evaluating a backstory claim against excerpts from a novel.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{evidence_text}

Follow this exact 3-step verification checklist:
Step 1: Identify the main entity and the specific fact/action/attribute stated in the claim.
Step 2: Check if that specific fact/action/attribute is explicitly stated in the evidence excerpts.
Step 3: Determine the verdict:
- If the specific fact/action is explicitly confirmed in the evidence -> "Verdict: SUPPORT"
- If the evidence states a conflicting or contradictory fact -> "Verdict: CONTRADICT"
- If the evidence does NOT mention or confirm the specific fact/action (even if the entity appears in other contexts) -> "Verdict: NOT MENTIONED"

Write your brief checklist, and conclude your last line with:
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
    (11, "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (16, "Lady Helena served as a military nurse in the Crimean War before marrying Lord Glenarvan."),
    (18, "M. Morrel previously served as a secret diplomat for the King of Spain in Madrid before founding his shipping business."),
    (19, "Gérard de Villefort wrote a published memoir detailing his childhood experiences during the French Revolution.")
]

for cid, text in test_cases:
    print(f"\n================ CLAIM {cid} ================")
    print("Claim:", text)
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    for r in rets:
        raw = test_checklist_prompt(r["Claim"], r["Evidence"], r["Entity"])
        print("\nLLM Output:\n", raw.strip())
        print("\nExtracted Verdict:", extract_single_verdict(raw))
