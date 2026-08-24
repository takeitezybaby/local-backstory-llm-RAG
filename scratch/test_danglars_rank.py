import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, subset_search

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

claim8 = "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain."

res = subset_search(claim8, entity_index["danglars"], faiss_index, chunks)
print("Total subset search results:", len(res))
for i, r in enumerate(res):
    print(f"\nRank {i+1} [Chunk {r['Atomic id']}]:")
    print(r["text"][:140])
