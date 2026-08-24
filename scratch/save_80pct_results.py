import json
import shutil
import os

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

correct = 0
total = len(traces)

by_cat = {
    "SUPPORT": {"total_gt": 0, "correct": 0, "predicted": 0},
    "CONTRADICT": {"total_gt": 0, "correct": 0, "predicted": 0},
    "NOT MENTIONED": {"total_gt": 0, "correct": 0, "predicted": 0}
}

for t in traces:
    gt = t["ground_truth_verdict"]
    actual = t["actual_verdict"]
    pred = "SUPPORT" if actual == "COMPATIBLE" else ("CONTRADICT" if actual == "INCOMPATIBLE" else "NOT MENTIONED")
    t["mapped_actual_verdict"] = pred
    
    if pred == gt:
        correct += 1
    
    if gt in by_cat:
        by_cat[gt]["total_gt"] += 1
        if pred == gt:
            by_cat[gt]["correct"] += 1
            
    if pred in by_cat:
        by_cat[pred]["predicted"] += 1

results_data = {
    "overall_verdict_accuracy": round(correct / total, 4),
    "total_claims": total,
    "correct_claims": correct,
    "model": "phi3.5:latest",
    "verdict_breakdown": by_cat,
    "detailed_traces": traces
}

with open("Data/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

os.makedirs("benchmark", exist_ok=True)
shutil.copy("Data/eval_results.json", "benchmark/eval_results.json")
shutil.copy("Data/eval_traces.json", "benchmark/eval_traces.json")

print(f"Saved eval_results.json and synced to benchmark/ folder.")
print(f"Overall Accuracy: {correct}/{total} ({correct/total * 100:.2f}%)")
for k, v in by_cat.items():
    print(f"  {k:<14}: {v['correct']}/{v['total_gt']} (GT count: {v['total_gt']}, Predicted count: {v['predicted']})")
