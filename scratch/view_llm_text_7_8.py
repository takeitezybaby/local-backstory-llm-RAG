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

for cid, text in [(7, "Abbé Faria was an Italian priest imprisoned in the Château d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dantès."),
                  (8, "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain.")]:
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    print(f"\n================ CLAIM {cid} ================")
    for r in rets:
        prompt = prompt_generation(r["Claim"], r["Evidence"], r["Entity"])
        print("Prompt Snippet Evidence 1 & 2:")
        for ev in r["Evidence"][:2]:
            print("  *", ev["text"][:120])
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "koesn/mistral-7b-instruct:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.0}
        }, timeout=90)
        print("LLM Output:\n", resp.json()["response"])
