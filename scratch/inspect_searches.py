import sys
import os
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity
from claimRetrieval import claim_retrieval

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"{'ID':<3} | {'Extracted Entity':<22} | {'In Index?':<10} | {'Search Type Used':<28} | {'GT Verdict':<15}")
print("-" * 85)

for s in dataset:
    claim_text = s["user_input"]
    ent = extract_entity(claim_text)
    in_idx = ent in entity_index if ent else False
    
    retrievals = claim_retrieval(claim_text, chunks, faiss_index, entity_index)
    search_type = retrievals[0]["Search_type"] if retrievals else "None"
    
    print(f"{s['id']:<3} | {str(ent):<22} | {str(in_idx):<10} | {search_type:<28} | {s['ground_truth_verdict']:<15}")
