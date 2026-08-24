import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

contradict_claims = [
    (9, "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."),
    (10, "Ayrton was the loyal first mate of Lord Glenarvan who originally built the yacht Duncan in Glasgow."),
    (11, "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon."),
    (14, "Mercédès married Edmond Dantès immediately after the Pharaon docked in Marseilles on February 24, 1815.")
]

for cid, text in contradict_claims:
    print(f"\n================ CLAIM {cid} ================")
    print("Claim:", text)
    rets = claim_retrieval(text, chunks, faiss_index, entity_index)
    for r in rets:
        print("  Atomic:", r["Claim"])
        print("  Entity:", r["Entity"])
        print("  Search Type:", r["Search_type"])
        print(f"  Evidence Count: {len(r['Evidence'])}")
        print("  Top 3 Snippets:")
        for ev in r["Evidence"][:3]:
            print(f"    * {ev['text'][:140]}...")
