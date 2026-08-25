import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

long_fails = []
for t in traces:
    gt = t["ground_truth_verdict"].strip().upper()
    act = t.get("mapped_actual_verdict", t.get("actual_verdict", "")).strip().upper()
    ctype = t.get("claim_type", "short")
    if gt != act:
        long_fails.append(t)

print(f"Total Failures Across All 110 Claims: {len(long_fails)}/110")
print("================================================================================")
print("                   BREAKDOWN OF ALL 40 FAILED CLAIMS                            ")
print("================================================================================")

for idx, t in enumerate(long_fails, 1):
    cid = t["id"]
    gt = t["ground_truth_verdict"]
    act = t.get("mapped_actual_verdict", t.get("actual_verdict", ""))
    ctype = t.get("claim_type", "short")
    txt = t["user_input"]
    resp = t.get("response", "")
    print(f"[{idx:02d}] ID {cid:02d} ({ctype.upper():<14}) | GT: {gt:<13} | Pred: {act:<13}")
    print(f"     Claim: {txt}")
    print(f"     System Response: {resp[:140]}")
    print("-" * 80)
