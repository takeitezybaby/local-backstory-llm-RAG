import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
sys.path.append("scratch")
from test_complete_phi_pipeline import full_pipeline_retrieval

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

c9 = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."
rets = full_pipeline_retrieval(c9, chunks, faiss_index, entity_index)
print("Entity:", rets[0]["Entity"])
print(f"Retrieved {len(rets[0]['Evidence'])} chunks:")
for i, ev in enumerate(rets[0]["Evidence"]):
    print(f"\n[{i+1}] {ev['text']}")
