import json
import os
import faiss
import sys

# Ensure Pipeline module imports work cleanly
sys.path.append(os.path.dirname(__file__))

from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import verify_claim
from aggregation import aggregate_results

def run_evaluation_traces(dataset_path="Data/eval_dataset.json", output_path="Data/eval_traces.json", limit=None):
    print("Loading index and chunk metadata...")
    atomic_chunks = loadChunks(os.path.join("Data", "atomicChunks.json"))
    entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
    faiss_index = faiss.read_index(os.path.join("Data", "atomic.index"))
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if limit is not None:
        dataset = dataset[:limit]
        
    traces = []
    total = len(dataset)
    print(f"Running pipeline traces for {total} evaluation claims from '{dataset_path}'...\n")
    
    for idx, sample in enumerate(dataset, 1):
        claim_text = sample["user_input"]
        claim_type = sample.get("claim_type", "short")
        print(f"[{idx}/{total}] ({claim_type.upper()}) Processing: {claim_text[:60]}...")
        
        # 1. Retrieve evidence chunks
        retrievals = claim_retrieval(claim_text, atomic_chunks, faiss_index, entity_index)
        
        contexts = []
        for ret in retrievals:
            for evid in ret.get("Evidence", []):
                if "text" in evid:
                    contexts.append(evid["text"])
                    
        # Remove duplicate text snippets while preserving order
        unique_contexts = list(dict.fromkeys(contexts))
        
        # 2. Run LLM verification & result aggregation
        llm_verification = verify_claim(claim_text, atomic_chunks, faiss_index, entity_index)
        aggregated = aggregate_results(llm_verification)
        
        system_response = f"Verdict: {aggregated['Final Verdict']}. Score: {aggregated['Normalized Score']:.2f}. Breakdown: {aggregated['Breakdown']}"
        
        traces.append({
            "id": sample["id"],
            "book": sample.get("book", ""),
            "claim_type": claim_type,
            "question": claim_text,
            "user_input": claim_text,
            "contexts": unique_contexts,
            "response": system_response,
            "ground_truth": sample["reference"],
            "ground_truth_verdict": sample["ground_truth_verdict"],
            "actual_verdict": aggregated["Final Verdict"],
            "normalized_score": aggregated["Normalized Score"]
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
        
    print(f"\nTrace collection complete! Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Data/eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output", default="Data/eval_traces.json", help="Path to save traces")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims to evaluate")
    args = parser.parse_args()
    
    run_evaluation_traces(dataset_path=args.dataset, output_path=args.output, limit=args.limit)

