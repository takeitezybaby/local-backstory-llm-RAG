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

claim9 = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."
rets = claim_retrieval(claim9, chunks, faiss_index, entity_index)
print("Evidence count:", len(rets[0]["Evidence"]))
for i, ev in enumerate(rets[0]["Evidence"]):
    print(f"\n--- Evidence {i+1} ---")
    print(ev["text"])
