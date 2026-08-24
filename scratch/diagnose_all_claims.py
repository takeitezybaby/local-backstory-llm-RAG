import sys
import os
import json
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import verify_claim
from aggregation import aggregate_results

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

print(f"Total traces: {len(traces)}")
for t in traces:
    gt = t["ground_truth_verdict"]
    mapped = t.get("mapped_actual_verdict", t["actual_verdict"])
    status = "CORRECT" if gt == mapped else "WRONG"
    print(f"\n[{t['id']}] {status} | GT: {gt} | Actual: {mapped} ({t['actual_verdict']})")
    print(f"  Question: {t['question']}")
    print(f"  Response: {t['response']}")
    print(f"  Context count: {len(t['contexts'])}")
    if status == "WRONG":
        print("  Top Contexts:")
        for c in t['contexts'][:3]:
            print(f"    - {c[:120]}...")
