import sys
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from verfication import verify_claim

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

for claim_id, claim_text in [
    (2, "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark."),
    (5, "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the three-masted ship the Pharaon."),
    (16, "Lady Helena served as a military nurse in the Crimean War before marrying Lord Glenarvan.")
]:
    print(f"\n================ CLAIM {claim_id} ================")
    print("Claim:", claim_text)
    res = verify_claim(claim_text, chunks, faiss_index, entity_index)
    for r in res:
        print("\n--- Atomic Claim:", r["Claim"])
        print("Raw LLM Output:\n", r["Verification_result"])
