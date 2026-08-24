import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, global_search, subset_search

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

claim7 = "Abb Faria was an Italian priest imprisoned in the Chteau d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dants."

# Let's search with subset search and global search
faria_chunks = subset_search(claim7, entity_index.get("abb faria", entity_index.get("faria", [])), faiss_index, chunks)
global_chunks = global_search(claim7, faiss_index, chunks)

print("Top 5 Faria entity chunks:")
for c in faria_chunks[:5]:
    print("  *", c["text"][:150])

print("\nTop 5 Global chunks:")
for c in global_chunks[:5]:
    print("  *", c["text"][:150])
