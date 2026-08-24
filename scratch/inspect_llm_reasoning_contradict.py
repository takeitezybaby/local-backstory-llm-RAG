import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import prompt_generation

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

test_claims = [
    (9, "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."),
    (10, "Ayrton was the loyal first mate of Lord Glenarvan who originally built the yacht Duncan in Glasgow."),
    (11, "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon."),
    (14, "Mercédès married Edmond Dantès immediately after the Pharaon docked in Marseilles on February 24, 1815.")
]

for cid, text in test_claims:
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    print(f"\n================ CLAIM {cid} ================")
    print("Claim:", text)
    for r in rets:
        prompt = prompt_generation(r["Claim"], r["Evidence"], r["Entity"])
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "koesn/mistral-7b-instruct:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.0}
        }, timeout=90)
        print("LLM Output:\n", resp.json()["response"])
