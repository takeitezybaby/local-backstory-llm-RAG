import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

contradict_traces = [t for t in traces if t.get("ground_truth_verdict") == "CONTRADICT"]
print(f"Total CONTRADICT claims in dataset: {len(contradict_traces)}")

retrieval_failures = 0
reasoning_failures = 0
correct_contradicts = 0

print("=" * 80)
print("       DETAILED DIAGNOSTIC AUDIT OF ALL CONTRADICT CLAIMS                    ")
print("=" * 80)

for idx, t in enumerate(contradict_traces, 1):
    cid = t["id"]
    claim = t["user_input"]
    pred = t.get("mapped_actual_verdict", "NOT MENTIONED")
    contexts = t.get("contexts", [])
    ref = t.get("ground_truth", "")
    
    is_correct = (pred == "CONTRADICT")
    if is_correct:
        correct_contradicts += 1
        status = "[CORRECT CONTRADICT]"
    elif pred == "SUPPORT":
        status = "[HALLUCINATED SUPPORT (DANGEROUS)]"
    else:
        status = "[OVER-CAUTIOUS ABSTENTION (NOT MENTIONED)]"
        
    print(f"[{idx:02d}] ID {cid:03d} | Pred: {pred:<13} | {status}")
    print(f"     Claim: {claim[:90]}...")
    print(f"     Ground Truth Reference: {ref}")
    print("-" * 80)

print(f"\nSummary across {len(contradict_traces)} CONTRADICT claims:")
print(f"  - Correctly Identified (CONTRADICT): {correct_contradicts} ({correct_contradicts/len(contradict_traces)*100:.1f}%)")
print(f"  - Over-Cautious Abstentions (NOT MENTIONED): {sum(1 for t in contradict_traces if t.get('mapped_actual_verdict') == 'NOT MENTIONED')}")
print(f"  - Dangerous Hallucinated Supports (SUPPORT): {sum(1 for t in contradict_traces if t.get('mapped_actual_verdict') == 'SUPPORT')}")
