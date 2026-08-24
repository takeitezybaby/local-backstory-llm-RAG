import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from claimExtraction import extract_atomic_claims

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

claims_to_check = [
    (2, "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark."),
    (3, "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India."),
    (4, "Captain Harry Grant was the commander of the brig Britannia which was shipwrecked in the Southern Seas."),
    (5, "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the three-masted ship the Pharaon."),
    (6, "Gérard de Villefort was the deputy crown prosecutor in Marseilles who interrogated Edmond Dantès.")
]

for cid, text in claims_to_check:
    print(f"\n================ CLAIM {cid} ================")
    print("Full text:", text)
    atomics = extract_atomic_claims(text)
    print("Extracted atomic claims:", atomics)
    retrievals = claim_retrieval(text, chunks, faiss_index, entity_index)
    for r in retrievals:
        print("  - Atomic:", r["Claim"])
        print("    Entity:", r["Entity"])
        print("    Search Type:", r["Search_type"])
        print(f"    Evidence count: {len(r['Evidence'])}")
        print("    Top 2 Evidence Snippets:")
        for ev in r["Evidence"][:2]:
            print(f"      * {ev['text'][:120]}...")
