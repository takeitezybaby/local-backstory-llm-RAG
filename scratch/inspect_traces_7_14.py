import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

for cid in [7, 8, 9, 10, 11, 12, 13, 14]:
    t = [x for x in traces if x["id"] == cid][0]
    print(f"\n================ CLAIM {cid} ================")
    print("Question:", t["question"])
    print("GT:", t["ground_truth_verdict"])
    print("Actual:", t["actual_verdict"])
    print("System Response:", t["response"])
    print("Context count:", len(t["contexts"]))
    print("Top Context:")
    for c in t["contexts"][:2]:
        print(f"  - {c[:120]}...")
