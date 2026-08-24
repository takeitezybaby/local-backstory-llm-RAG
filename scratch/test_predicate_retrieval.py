import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import global_search

chunks = loadChunks("Data/atomicChunks.json")
faiss_index = faiss.read_index("Data/atomic.index")

queries = [
    "captain of the yacht Duncan",
    "hooked and landed the hammerhead shark",
    "builder of the yacht Duncan in Glasgow",
    "father of Mary Grant",
    "Captain Leclere Pharaon arrival Marseilles",
    "Fernand Mondego occupation merchant Paris",
    "Mercedes married Edmond Dantes wedding feast arrest"
]

for q in queries:
    print(f"\n================ QUERY: '{q}' ================")
    res = global_search(q, faiss_index, chunks)
    for r in res[:3]:
        print(f"  * {r['text'][:140]}...")
