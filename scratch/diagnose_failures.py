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

failed_ids = [7, 8, 9, 10, 11, 12, 13, 14]

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for sample in dataset:
    if sample["id"] in failed_ids:
        print(f"\n================ CLAIM {sample['id']} ================")
        print("Question:", sample["user_input"])
        print("GT Verdict:", sample["ground_truth_verdict"])
        print("Reference Truth:", sample["reference"])
        rets = claim_retrieval(sample["user_input"], chunks, faiss_index, entity_index)
        for r in rets:
            print("  Atomic:", r["Claim"])
            print("  Entity:", r["Entity"])
            print(f"  Evidence Count: {len(r['Evidence'])}")
            print("  Top 3 Snippets:")
            for ev in r["Evidence"][:3]:
                print(f"    * {ev['text'][:120]}...")
