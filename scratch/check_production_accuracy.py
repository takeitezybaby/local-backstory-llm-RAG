import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

print(f"{'ID':<3} | {'GT':<14} | {'Actual Verdict':<16} | {'Score':<6} | {'Match?':<8}")
print("-" * 60)

correct = 0
total = len(traces)

by_cat = {"SUPPORT": {"total": 0, "correct": 0}, "CONTRADICT": {"total": 0, "correct": 0}, "NOT MENTIONED": {"total": 0, "correct": 0}}

for t in traces:
    gt = t["ground_truth_verdict"]
    actual = t["actual_verdict"]
    # Map COMPATIBLE -> SUPPORT, INCOMPATIBLE -> CONTRADICT
    pred = "SUPPORT" if actual == "COMPATIBLE" else ("CONTRADICT" if actual == "INCOMPATIBLE" else "NOT MENTIONED")
    match = (pred == gt)
    if match:
        correct += 1
    
    if gt in by_cat:
        by_cat[gt]["total"] += 1
        if match:
            by_cat[gt]["correct"] += 1
            
    print(f"{t['id']:<3} | {gt:<14} | {pred:<16} | {t['normalized_score']:<6.2f} | {'PASS' if match else 'FAIL':<8}")

print("-" * 60)
print(f"OVERALL PRODUCTION ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%\n")
for cat, stats in by_cat.items():
    pct = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"  {cat:<14}: {stats['correct']}/{stats['total']} = {pct:.2f}%")
