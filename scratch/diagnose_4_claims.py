import sys
import json
import requests
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
sys.path.append("scratch")
from test_enhanced_phi import enhanced_retrieval, prompt_generation_phi

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for target_id in [3, 6, 8, 16]:
    s = [item for item in dataset if item["id"] == target_id][0]
    rets = enhanced_retrieval(s["user_input"], chunks, faiss_index, entity_index)
    print(f"\n================ CLAIM {target_id} (GT: {s['ground_truth_verdict']}) ================")
    print(f"Input: {s['user_input']}")
    print(f"Entity: {rets[0]['Entity']}")
    print(f"Target Book: {rets[0]['Target_Book']}")
    print(f"Top 5 Evidence Excerpts:")
    for i, ev in enumerate(rets[0]["Evidence"][:5]):
        print(f"  [{i+1}] {ev['text']}")
    
    p = prompt_generation_phi(rets[0]["Claim"], rets[0]["Evidence"], rets[0]["Entity"])
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "phi3.5:latest",
        "prompt": p,
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.0}
    }, timeout=60).json()["response"]
    print(f"\nModel Output:\n{resp}")
