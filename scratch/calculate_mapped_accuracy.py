import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

VERDICT_MAP = {
    "COMPATIBLE": "SUPPORT",
    "PARTIALLY COMPATIBLE": "SUPPORT",
    "INCOMPATIBLE": "CONTRADICT",
    "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED",
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "NOT MENTIONED": "NOT MENTIONED"
}

correct = 0
total = len(traces)
print(f"{'ID':<3} | {'Question':<50} | {'GT':<14} | {'Raw Verdict':<36} | {'Mapped':<14} | {'Result':<8}")
print("-" * 135)

for t in traces:
    raw = t["actual_verdict"]
    gt = t["ground_truth_verdict"]
    mapped = VERDICT_MAP.get(raw, raw)
    match = (mapped == gt)
    if match:
        correct += 1
    res_str = "PASS" if match else "FAIL"
    print(f"{t['id']:<3} | {t['question'][:50]:<50} | {gt:<14} | {raw:<36} | {mapped:<14} | {res_str:<8}")

print("-" * 135)
print(f"\nFINAL VERIFIED ACCURACY: {correct}/{total} = {correct/total * 100:.2f}%")
