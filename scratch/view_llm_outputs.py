import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

for t in traces:
    print(f"\n================ CLAIM {t['id']} ================")
    print("Question:", t["question"])
    print("GT:", t["ground_truth_verdict"])
    print("Actual:", t["actual_verdict"])
    print("Response Summary:", t["response"])
    print("Context Count:", len(t["contexts"]))
    # Let's inspect the top 2 contexts
    for i, c in enumerate(t["contexts"][:2]):
        print(f"  Context {i+1}: {c[:100]}...")
