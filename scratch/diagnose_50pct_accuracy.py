import json
from collections import Counter

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

print("================================================================================")
print("              DEEP ERROR ANALYSIS & CONFUSION MATRIX (110 CLAIMS)               ")
print("================================================================================")

confusion = {}
failure_examples = {"CONTRADICT_as_NOT_MENTIONED": [], "SUPPORT_as_NOT_MENTIONED": [], "OTHER": []}

for t in traces:
    gt = t["ground_truth_verdict"].strip().upper()
    act = t.get("mapped_actual_verdict", t.get("actual_verdict", "")).strip().upper()
    
    if act in ["COMPATIBLE", "PARTIALLY COMPATIBLE"]:
        act = "SUPPORT"
    elif act in ["INCOMPATIBLE"]:
        act = "CONTRADICT"
    elif "NOT SUPPORTED" in act or "NOT MENTIONED" in act:
        act = "NOT MENTIONED"
        
    confusion.setdefault(gt, Counter())
    confusion[gt][act] += 1
    
    if gt != act:
        category = f"{gt}_as_{act}"
        if category in failure_examples:
            failure_examples[category].append(t)
        else:
            failure_examples["OTHER"].append(t)

print("CONFUSION MATRIX:")
print(f"{'Ground Truth':<15} | {'Pred: SUPPORT':<15} | {'Pred: CONTRADICT':<17} | {'Pred: NOT MENTIONED':<20}")
print("-" * 75)
for gt_cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
    c = confusion.get(gt_cls, Counter())
    print(f"{gt_cls:<15} | {c.get('SUPPORT', 0):>13} | {c.get('CONTRADICT', 0):>15} | {c.get('NOT MENTIONED', 0):>18}")
print("-" * 75)

print("\n--------------------------------------------------------------------------------")
print("SAMPLE FAILURES: CONTRADICT CLASSIFIED AS NOT MENTIONED")
print("--------------------------------------------------------------------------------")
for t in failure_examples.get("CONTRADICT_as_NOT_MENTIONED", [])[:5]:
    print(f"ID {t['id']} ({t.get('claim_type', 'short').upper()}): {t['user_input']}")
    print(f"   Ground Truth Ref: {t.get('ground_truth', '')[:100]}...")
    print(f"   Raw System Response: {t.get('response', '')}")
    print(f"   Contexts retrieved ({len(t.get('contexts', []))} chunks):")
    for i, ctx in enumerate(t.get("contexts", [])[:2]):
        print(f"      [{i+1}] {ctx[:120]}...")
    print()

print("--------------------------------------------------------------------------------")
print("SAMPLE FAILURES: SUPPORT CLASSIFIED AS NOT MENTIONED")
print("--------------------------------------------------------------------------------")
for t in failure_examples.get("SUPPORT_as_NOT_MENTIONED", [])[:5]:
    print(f"ID {t['id']} ({t.get('claim_type', 'short').upper()}): {t['user_input']}")
    print(f"   Ground Truth Ref: {t.get('ground_truth', '')[:100]}...")
    print(f"   Raw System Response: {t.get('response', '')}")
    print(f"   Contexts retrieved ({len(t.get('contexts', []))} chunks):")
    for i, ctx in enumerate(t.get("contexts", [])[:2]):
        print(f"      [{i+1}] {ctx[:120]}...")
    print()
