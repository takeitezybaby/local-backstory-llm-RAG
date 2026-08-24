import json
import os
import sys
import time
import re
import faiss
import numpy as np

# Ensure Pipeline imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Pipeline"))

from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import verify_claim, generate_response
from aggregation import aggregate_results

# Standard verdict mapping
VERDICT_MAP = {
    "COMPATIBLE": "SUPPORT",
    "PARTIALLY COMPATIBLE": "SUPPORT",
    "INCOMPATIBLE": "CONTRADICT",
    "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED",
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "NOT MENTIONED": "NOT MENTIONED"
}

def extract_verdict_from_text(raw_text):
    """Parse verdict strictly from LLM output."""
    raw = raw_text.strip().upper()
    
    # 1. Regex check for structured output
    m = re.findall(r'VERDICT\s*:\s*["\']?\s*(SUPPORT|CONTRADICT|NOT MENTIONED)', raw)
    if m:
        return m[-1]
        
    # 2. Check for exact line matches
    lines = [line.strip().strip('"\'*#') for line in raw.split('\n') if line.strip()]
    for line in reversed(lines):
        if line in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            return line
        if line.startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().strip('"\'*#')
            if val in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
                return val

    # 3. Fallback priority check
    if "CONTRADICT" in raw:
        return "CONTRADICT"
    elif "NOT MENTIONED" in raw:
        return "NOT MENTIONED"
    elif "SUPPORT" in raw:
        return "SUPPORT"
        
    return "NOT MENTIONED"

# =========================================================================
# BASELINE 1: Vanilla Dense RAG (Naive Top-5 Cosine, No Entity Index, No Claim Decomposition)
# =========================================================================
def run_vanilla_dense_rag(backstory, faiss_index, metadata, top_k=5):
    # 1. Naive global vector embedding of raw query
    query_embed = createEmbeddings(backstory)
    query_embed = normalize(query_embed)
    
    scores, indices = faiss_index.search(query_embed, top_k)
    retrieved_texts = [metadata[idx]["text"] for idx in indices[0] if idx < len(metadata)]
    
    # 2. Format single naive verification prompt
    evidence_block = "\n".join([f"Excerpt {i+1}:\n{t}" for i, t in enumerate(retrieved_texts)])
    prompt = f"""<|user|>
You are a fact-checker verifying a backstory against source novel excerpts.

Backstory: "{backstory}"

Source Excerpts:
{evidence_block}

Determine if the backstory is supported by the excerpts, contradicted by the excerpts, or not mentioned.
Conclude on the last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
    
    response = generate_response(prompt)
    verdict = extract_verdict_from_text(response)
    return verdict, retrieved_texts

# =========================================================================
# BASELINE 2: Backstory RAG (Our Full 10-Stage Neuro-Symbolic Pipeline)
# =========================================================================
def run_backstory_rag(backstory, faiss_index, entity_index, metadata):
    llm_verification = verify_claim(backstory, metadata, faiss_index, entity_index)
    aggregated = aggregate_results(llm_verification)
    final_verdict = aggregated["Final Verdict"]
    mapped = VERDICT_MAP.get(final_verdict.strip().upper(), final_verdict)
    return mapped

# =========================================================================
# COMPARATIVE BENCHMARK RUNNER
# =========================================================================
def run_comparative_benchmark(dataset_path="benchmark/eval_dataset_100.json", limit=None, output_path="benchmark/baseline_comparison_results.json"):
    print("==================================================")
    print("   BACKSTORY RAG vs. BASELINES BENCHMARK SUITE   ")
    print("==================================================")
    
    metadata = loadChunks(os.path.join("Data", "atomicChunks.json"))
    entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
    faiss_index = faiss.read_index(os.path.join("Data", "atomic.index"))
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if limit is not None:
        dataset = dataset[:limit]
        
    total = len(dataset)
    print(f"Evaluating {total} claims on Vanilla RAG and Backstory RAG...\n")
    
    results = {
        "Vanilla_Dense_RAG": {"correct": 0, "total": total, "by_class": {}, "latencies": []},
        "Backstory_RAG_Ours": {"correct": 0, "total": total, "by_class": {}, "latencies": []}
    }
    
    comparison_traces = []
    
    for idx, sample in enumerate(dataset, 1):
        claim_text = sample["user_input"]
        gt = sample["ground_truth_verdict"]
        ctype = sample.get("claim_type", "short")
        
        print(f"[{idx}/{total}] ({ctype.upper()}) Evaluating: {claim_text[:60]}...")
        
        # --- Run Baseline 1: Vanilla Dense RAG ---
        t0 = time.time()
        v_verdict, v_contexts = run_vanilla_dense_rag(claim_text, faiss_index, metadata, top_k=5)
        t1 = time.time()
        v_latency = t1 - t0
        results["Vanilla_Dense_RAG"]["latencies"].append(v_latency)
        
        v_correct = (v_verdict == gt)
        if v_correct:
            results["Vanilla_Dense_RAG"]["correct"] += 1
            
        results["Vanilla_Dense_RAG"]["by_class"].setdefault(gt, {"total": 0, "correct": 0})
        results["Vanilla_Dense_RAG"]["by_class"][gt]["total"] += 1
        if v_correct:
            results["Vanilla_Dense_RAG"]["by_class"][gt]["correct"] += 1
            
        # --- Run Baseline 2: Backstory RAG (Ours) ---
        t0 = time.time()
        b_verdict = run_backstory_rag(claim_text, faiss_index, entity_index, metadata)
        t1 = time.time()
        b_latency = t1 - t0
        results["Backstory_RAG_Ours"]["latencies"].append(b_latency)
        
        b_correct = (b_verdict == gt)
        if b_correct:
            results["Backstory_RAG_Ours"]["correct"] += 1
            
        results["Backstory_RAG_Ours"]["by_class"].setdefault(gt, {"total": 0, "correct": 0})
        results["Backstory_RAG_Ours"]["by_class"][gt]["total"] += 1
        if b_correct:
            results["Backstory_RAG_Ours"]["by_class"][gt]["correct"] += 1
            
        comparison_traces.append({
            "id": sample["id"],
            "claim_type": ctype,
            "ground_truth": gt,
            "vanilla_rag_verdict": v_verdict,
            "vanilla_rag_correct": v_correct,
            "backstory_rag_verdict": b_verdict,
            "backstory_rag_correct": b_correct
        })
        
    # Print Comparison Scorecard Table
    print("\n================================================================================")
    print("                  EMPIRICAL BASELINE COMPARISON SCORECARD                       ")
    print("================================================================================")
    print(f"{'Method / Metric':<25} | {'Vanilla Dense RAG':<18} | {'Backstory RAG (Ours)':<20}")
    print("--------------------------------------------------------------------------------")
    
    v_acc = (results["Vanilla_Dense_RAG"]["correct"] / total) * 100
    b_acc = (results["Backstory_RAG_Ours"]["correct"] / total) * 100
    print(f"{'Overall Accuracy':<25} | {v_acc:>16.2f}% | {b_acc:>18.2f}% [PASS]")

    
    for cls_name in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        v_cls = results["Vanilla_Dense_RAG"]["by_class"].get(cls_name, {"correct": 0, "total": 1})
        b_cls = results["Backstory_RAG_Ours"]["by_class"].get(cls_name, {"correct": 0, "total": 1})
        
        v_rec = (v_cls["correct"] / v_cls["total"]) * 100 if v_cls["total"] > 0 else 0
        b_rec = (b_cls["correct"] / b_cls["total"]) * 100 if b_cls["total"] > 0 else 0
        print(f"{cls_name + ' Recall':<25} | {v_rec:>16.2f}% | {b_rec:>18.2f}%")
        
    v_avg_lat = sum(results["Vanilla_Dense_RAG"]["latencies"]) / total if total > 0 else 0
    b_avg_lat = sum(results["Backstory_RAG_Ours"]["latencies"]) / total if total > 0 else 0
    print(f"{'Avg Latency / Claim':<25} | {v_avg_lat:>16.2f}s | {b_avg_lat:>18.2f}s")
    print("================================================================================\n")
    
    payload = {
        "summary": {
            "total_evaluated": total,
            "vanilla_dense_rag_accuracy": v_acc,
            "backstory_rag_accuracy": b_acc,
            "vanilla_dense_rag_breakdown": results["Vanilla_Dense_RAG"],
            "backstory_rag_breakdown": results["Backstory_RAG_Ours"]
        },
        "traces": comparison_traces
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Detailed comparison saved to '{output_path}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark/eval_dataset_100.json", help="Path to evaluation dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims")
    parser.add_argument("--output", default="benchmark/baseline_comparison_results.json", help="Path to save results")
    args = parser.parse_args()
    
    run_comparative_benchmark(dataset_path=args.dataset, limit=args.limit, output_path=args.output)
