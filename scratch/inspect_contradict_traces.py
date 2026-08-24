import json

traces = json.load(open("Data/eval_traces.json", encoding="utf-8"))
for t in traces:
    if t["ground_truth_verdict"] == "CONTRADICT":
        print(f"=== CLAIM {t['id']} ===")
        print("Claim:", t["user_input"])
        print("GT:", t["ground_truth_verdict"])
        print("Actual Verdict:", t["actual_verdict"])
        print("\nResponse / Trace:")
        print(t.get("response", ""))
        print("=" * 60)
