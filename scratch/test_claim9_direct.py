import sys
import json
import requests
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import prompt_generation, generate_response

atomic_chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

claim_text = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."

rets = claim_retrieval(claim_text, atomic_chunks, faiss_index, entity_index)
for r in rets:
    print(f"Claim: {r['Claim']}")
    print(f"Entity: {r['Entity']}")
    print(f"Target_Book: {r.get('Target_Book')}")
    print(f"Search type: {r['Search_type']}")
    print(f"Evidence count: {len(r['Evidence'])}")
    for i, e in enumerate(r['Evidence'][:5]):
        print(f"  [{i+1}] {e['text']}")
        
    p = prompt_generation(r["Claim"], r["Evidence"], r["Entity"])
    resp = generate_response(p)
    print("\n--- PROMPT SENT ---")
    print(p)
    print("\n--- LLM RAW RESPONSE ---")
    print(resp)
