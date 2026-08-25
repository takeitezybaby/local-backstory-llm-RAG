import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from verfication import verify_claim
from aggregation import aggregate_results

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

# Load 20 diverse claims from benchmark/eval_dataset_100.json
with open("benchmark/eval_dataset_100.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Pick 8 SUPPORT, 6 CONTRADICT, 6 NOT MENTIONED across both books and short/long
sample_20 = [
    # Book 1 - SUPPORT (Short & Long)
    dataset[0],  # ID 1 (Glenarvan)
    dataset[1],  # ID 2 (Tom Austin)
    dataset[2],  # ID 3 (Paganel)
    dataset[7],  # ID 8 (MacNabb)
    dataset[10], # ID 11 (Castaways expedition long)
    dataset[11], # ID 12 (Glenarvan Admiralty long)
    # Book 1 - CONTRADICT (Short & Long)
    dataset[20], # ID 21 (Glenarvan pirate)
    dataset[22], # ID 23 (Mary Grant MacNabb daughter)
    dataset[23], # ID 24 (Paganel English admiral)
    dataset[24], # ID 25 (Thalcave bushranger)
    dataset[30], # ID 31 (Glenarvan abandon Duncan long)
    dataset[33], # ID 34 (Paganel secret agent long)
    # Book 1 - NOT MENTIONED (Short & Long)
    dataset[40], # ID 41 (Captain Grant railway shares)
    dataset[41], # ID 42 (Lady Helena oil painting)
    dataset[45], # ID 46 (Paganel cartography society long)
    # Book 2 - SUPPORT & CONTRADICT & NOT MENTIONED
    dataset[50], # ID 51 (Edmond Dantes arrival)
    dataset[52], # ID 53 (Abbe Faria)
    dataset[70], # ID 71 (Captain Leclere)
    dataset[73], # ID 74 (Abbe Faria execution)
    dataset[85]  # ID 86 (M. Morrel secret diplomat)
]

print("================================================================================")
print(f"       EVALUATING 20-CLAIM BENCHMARK WITH POOLED RETRIEVAL & NLI PROMPT         ")
print("================================================================================")

correct = 0
results = []
for idx, s in enumerate(sample_20, 1):
    c_text = s["user_input"]
    gt = s["ground_truth_verdict"]
    ctype = s.get("claim_type", "short")
    
    verifications = verify_claim(c_text, metadata, faiss_index, entity_index)
    agg = aggregate_results(verifications)
    pred = agg["Final Verdict"]
    
    VERDICT_MAP = {
        "COMPATIBLE": "SUPPORT",
        "PARTIALLY COMPATIBLE": "SUPPORT",
        "INCOMPATIBLE": "CONTRADICT",
        "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED"
    }
    mapped = VERDICT_MAP.get(pred.strip().upper(), "NOT MENTIONED")

    is_match = (mapped == gt)
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    
    print(f"[{idx:02d}/20] ID {s['id']:02d} ({ctype.upper():<5}) | GT: {gt:<13} | Pred: {mapped:<13} | {mark}")
    results.append({"id": s["id"], "gt": gt, "pred": mapped, "match": is_match})

acc = (correct / len(sample_20)) * 100
print("--------------------------------------------------------------------------------")
print(f"Overall Accuracy on 20 Diverse Claims: {acc:.2f}% ({correct}/{len(sample_20)})")
print("================================================================================")
