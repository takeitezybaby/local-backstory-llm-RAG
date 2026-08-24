import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

for t in traces:
    if t["id"] in [9, 10, 11, 12, 13, 14]:
        print(f"=== ID {t['id']} ===")
        print(f"Claim: {t['question']}")
        print(f"Response: {t['response']}")
        print()
