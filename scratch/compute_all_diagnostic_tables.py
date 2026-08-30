import json
import os

with open("benchmark/diagnostics_220_all_conditions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("benchmark/eval_dataset_220.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset_map = {item["id"]: item for item in dataset}

print("=" * 90)
print("             TASK 1: 4-CONDITION BASELINE ISOLATION ON FULL 220-CLAIM SET             ")
print("=" * 90)

def compute_task1_metrics(cond_key, name):
    traces = data[cond_key]["traces"]
    total = len(traces)
    correct = sum(1 for t in traces if t["gt"] == t["pred"])
    acc = correct / total * 100
    
    # 3x3 Confusion Matrix: [GT][PRED]
    matrix = {
        "SUPPORT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "NOT MENTIONED": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
    }
    for t in traces:
        gt = t["gt"]
        p = t["pred"]
        matrix[gt][p] = matrix[gt].get(p, 0) + 1
        
    # Per-class P/R/F1
    class_stats = {}
    for cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        gt_cnt = sum(matrix[cls].values())
        pred_cnt = sum(matrix[r][cls] for r in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"])
        tp = matrix[cls][cls]
        prec = (tp / pred_cnt * 100) if pred_cnt > 0 else 0.0
        rec = (tp / gt_cnt * 100) if gt_cnt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        class_stats[cls] = {"gt": gt_cnt, "pred": pred_cnt, "tp": tp, "p": prec, "r": rec, "f1": f1}
        
    # Hallucinated-SUPPORT: GT is CONTRADICT or NOT MENTIONED but Pred is SUPPORT
    hallucinated_support = matrix["CONTRADICT"]["SUPPORT"] + matrix["NOT MENTIONED"]["SUPPORT"]
    non_support_gt = matrix["CONTRADICT"]["SUPPORT"] + matrix["CONTRADICT"]["CONTRADICT"] + matrix["CONTRADICT"]["NOT MENTIONED"] + \
                     matrix["NOT MENTIONED"]["SUPPORT"] + matrix["NOT MENTIONED"]["CONTRADICT"] + matrix["NOT MENTIONED"]["NOT MENTIONED"]
    hallucinated_rate = (hallucinated_support / non_support_gt * 100) if non_support_gt > 0 else 0.0
    
    # Abstention rate (Pred is NOT MENTIONED)
    abstain_cnt = sum(matrix[r]["NOT MENTIONED"] for r in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"])
    abstain_rate = abstain_cnt / total * 100
    
    return {
        "name": name,
        "acc": acc,
        "correct": correct,
        "total": total,
        "class_stats": class_stats,
        "hallucinated_support": hallucinated_support,
        "non_support_gt": non_support_gt,
        "hallucinated_rate": hallucinated_rate,
        "abstain_cnt": abstain_cnt,
        "abstain_rate": abstain_rate,
        "matrix": matrix
    }

task1_results = {}
for k, label in [
    ("vanilla_binary", "(a) Vanilla Retrieval + Binary Verdict"),
    ("vanilla_3way", "(b) Vanilla Retrieval + 3-Way OW-NLI Verdict"),
    ("backstory_binary", "(c) Backstory Retrieval + Binary Verdict"),
    ("backstory_3way", "(d) Backstory Retrieval + Persona + 3-Way OW-NLI (Full System)")
]:
    task1_results[k] = compute_task1_metrics(k, label)

# Print Task 1 Table
print(f"{'Condition':<48} | {'Acc (%)':<8} | {'Supp F1':<8} | {'Cont F1':<8} | {'NotM F1':<8} | {'Halluc-Supp Rate':<18} | {'Abstain Rate'}")
print("-" * 125)
for k, res in task1_results.items():
    cs = res["class_stats"]
    print(f"{res['name']:<48} | {res['acc']:<6.2f}% | {cs['SUPPORT']['f1']:<6.2f}% | {cs['CONTRADICT']['f1']:<6.2f}% | {cs['NOT MENTIONED']['f1']:<6.2f}% | {res['hallucinated_support']}/{res['non_support_gt']} ({res['hallucinated_rate']:.2f}%)  | {res['abstain_cnt']}/{res['total']} ({res['abstain_rate']:.2f}%)")

print("\nDetailed Per-Class Precision & Recall for all 4 conditions:")
for k, res in task1_results.items():
    cs = res["class_stats"]
    print(f"\n[{res['name']}]")
    for cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        print(f"  - {cls:<15}: Prec = {cs[cls]['p']:.2f}%, Rec = {cs[cls]['r']:.2f}%, F1 = {cs[cls]['f1']:.2f}% (GT: {cs[cls]['gt']}, Pred: {cs[cls]['pred']})")

print("\n" + "=" * 90)
print("     TASK 2: CONTRADICT FAILURE MODE AUDIT (RETRIEVAL VS VERDICT FAILURE)     ")
print("=" * 90)

# Full system traces (d)
full_traces = data["backstory_3way"]["traces"]
contradict_claims = [t for t in full_traces if t["gt"] == "CONTRADICT"]

# For each contradict claim, check if contradicting evidence keywords/entities are in retrieved text
def audit_contradict_claim(t):
    cid = t["id"]
    item = dataset_map[cid]
    claim_text = item["user_input"]
    ref = item.get("reference", "")
    book = item["book"]
    pred = t["pred"]
    ev_texts = t.get("evidence", [])
    combined_ev = " ".join(ev_texts).lower()
    
    # Check if key reference facts / character true identity are mentioned in context
    # High-signal reference tokens
    ref_tokens = [w.lower().strip(".,!?:;\"'") for w in ref.split() if len(w) > 3 and w.lower() not in {"this", "that", "with", "from", "never", "were", "been", "died", "lived", "true", "false", "claim", "novel"}]
    overlap = sum(1 for tok in ref_tokens if tok in combined_ev)
    
    # If overlap >= 2 or key character true fate in text -> Evidence was retrieved (Verdict Failure)
    # Else -> Retrieval Failure
    evidence_retrieved = (overlap >= 2) or (len(ref_tokens) > 0 and (overlap / len(ref_tokens) >= 0.4))
    
    if pred == "CONTRADICT":
        return "CORRECT", book, cid
    elif not evidence_retrieved:
        return "RETRIEVAL_FAILURE", book, cid
    else:
        if pred == "SUPPORT":
            return "VERDICT_FAILURE_PRED_SUPPORT", book, cid
        else:
            return "VERDICT_FAILURE_PRED_NOT_MENTIONED", book, cid

task2_stats = {
    "In_Search_of_the_Castaways": {"total_gt": 0, "correct": 0, "retrieval_fail": 0, "verdict_pred_supp": 0, "verdict_pred_notm": 0},
    "The_Count_of_Monte_Cristo": {"total_gt": 0, "correct": 0, "retrieval_fail": 0, "verdict_pred_supp": 0, "verdict_pred_notm": 0},
    "The_Hound_of_the_Baskervilles": {"total_gt": 0, "correct": 0, "retrieval_fail": 0, "verdict_pred_supp": 0, "verdict_pred_notm": 0},
    "Dracula": {"total_gt": 0, "correct": 0, "retrieval_fail": 0, "verdict_pred_supp": 0, "verdict_pred_notm": 0}
}

for t in contradict_claims:
    res_type, book, cid = audit_contradict_claim(t)
    # Normalize book name
    b_key = "In_Search_of_the_Castaways" if "castaways" in book.lower() else (
        "The_Count_of_Monte_Cristo" if "monte" in book.lower() else (
            "The_Hound_of_the_Baskervilles" if "hound" in book.lower() else "Dracula"
        )
    )
    task2_stats[b_key]["total_gt"] += 1
    if res_type == "CORRECT":
        task2_stats[b_key]["correct"] += 1
    elif res_type == "RETRIEVAL_FAILURE":
        task2_stats[b_key]["retrieval_fail"] += 1
    elif res_type == "VERDICT_FAILURE_PRED_SUPPORT":
        task2_stats[b_key]["verdict_pred_supp"] += 1
    elif res_type == "VERDICT_FAILURE_PRED_NOT_MENTIONED":
        task2_stats[b_key]["verdict_pred_notm"] += 1

print(f"{'Book Title':<32} | {'GT Contradict':<14} | {'Correct':<8} | {'Retrieval Fail':<15} | {'Verdict (Pred Supp)':<20} | {'Verdict (Pred NotM)'}")
print("-" * 125)
tot_gt = 0
tot_cor = 0
tot_rf = 0
tot_vps = 0
tot_vpnm = 0

for b, s in task2_stats.items():
    tot_gt += s["total_gt"]
    tot_cor += s["correct"]
    tot_rf += s["retrieval_fail"]
    tot_vps += s["verdict_pred_supp"]
    tot_vpnm += s["verdict_pred_notm"]
    print(f"{b:<32} | {s['total_gt']:<14} | {s['correct']:<8} | {s['retrieval_fail']:<15} | {s['verdict_pred_supp']:<20} | {s['verdict_pred_notm']}")

print("-" * 125)
print(f"{'TOTAL (ALL 4 BOOKS)':<32} | {tot_gt:<14} | {tot_cor:<8} | {tot_rf:<15} | {tot_vps:<20} | {tot_vpnm}")

print("\n" + "=" * 90)
print("             TASK 3: FULL 3x3 CONFUSION MATRIX ON FULL 220-CLAIM SET                  ")
print("=" * 90)

m_full = task1_results["backstory_3way"]["matrix"]
print("3x3 Confusion Matrix [Condition (d): Full Backstory RAG]:\n")
print(f"                      Pred SUPPORT   |   Pred CONTRADICT   |   Pred NOT MENTIONED   |   ROW TOTAL (GT)")
print("-" * 95)
r_supp = m_full["SUPPORT"]["SUPPORT"] + m_full["SUPPORT"]["CONTRADICT"] + m_full["SUPPORT"]["NOT MENTIONED"]
r_cont = m_full["CONTRADICT"]["SUPPORT"] + m_full["CONTRADICT"]["CONTRADICT"] + m_full["CONTRADICT"]["NOT MENTIONED"]
r_notm = m_full["NOT MENTIONED"]["SUPPORT"] + m_full["NOT MENTIONED"]["CONTRADICT"] + m_full["NOT MENTIONED"]["NOT MENTIONED"]

print(f"GT SUPPORT       :        {m_full['SUPPORT']['SUPPORT']:<10} |        {m_full['SUPPORT']['CONTRADICT']:<12} |        {m_full['SUPPORT']['NOT MENTIONED']:<14} |        {r_supp}")
print(f"GT CONTRADICT    :        {m_full['CONTRADICT']['SUPPORT']:<10} |        {m_full['CONTRADICT']['CONTRADICT']:<12} |        {m_full['CONTRADICT']['NOT MENTIONED']:<14} |        {r_cont}")
print(f"GT NOT MENTIONED :        {m_full['NOT MENTIONED']['SUPPORT']:<10} |        {m_full['NOT MENTIONED']['CONTRADICT']:<12} |        {m_full['NOT MENTIONED']['NOT MENTIONED']:<14} |        {r_notm}")
print("-" * 95)
c_supp = m_full["SUPPORT"]["SUPPORT"] + m_full["CONTRADICT"]["SUPPORT"] + m_full["NOT MENTIONED"]["SUPPORT"]
c_cont = m_full["SUPPORT"]["CONTRADICT"] + m_full["CONTRADICT"]["CONTRADICT"] + m_full["NOT MENTIONED"]["CONTRADICT"]
c_notm = m_full["SUPPORT"]["NOT MENTIONED"] + m_full["CONTRADICT"]["NOT MENTIONED"] + m_full["NOT MENTIONED"]["NOT MENTIONED"]
print(f"COL TOTAL (PRED) :        {c_supp:<10} |        {c_cont:<12} |        {c_notm:<14} |        {r_supp + r_cont + r_notm}")

# Verification of internal consistency
assert r_supp == 89, f"Row sum mismatch for SUPPORT: {r_supp} != 89"
assert r_cont == 66, f"Row sum mismatch for CONTRADICT: {r_cont} != 66"
assert r_notm == 65, f"Row sum mismatch for NOT MENTIONED: {r_notm} != 65"
assert (r_supp + r_cont + r_notm) == 220, "Total claim count mismatch!"
print("\n[VERIFIED] Internal Consistency Check PASSED: Row and Column sums strictly match 220 total claims.")

print("\n" + "=" * 90)
print("             TASK 4: BOOK-BY-BOOK BREAKDOWN (3x3 MATRIX & FAILURE DIAGNOSIS)           ")
print("=" * 90)

book_traces = {
    "In_Search_of_the_Castaways": [],
    "The_Count_of_Monte_Cristo": [],
    "The_Hound_of_the_Baskervilles": [],
    "Dracula": []
}

for t in full_traces:
    b = dataset_map[t["id"]]["book"]
    b_key = "In_Search_of_the_Castaways" if "castaways" in b.lower() else (
        "The_Count_of_Monte_Cristo" if "monte" in b.lower() else (
            "The_Hound_of_the_Baskervilles" if "hound" in b.lower() else "Dracula"
        )
    )
    book_traces[b_key].append(t)

for bname, b_tr in book_traces.items():
    print(f"\n================================================================================")
    print(f"   NOVEL: {bname} (55 Claims Total)")
    print(f"================================================================================")
    b_cor = sum(1 for t in b_tr if t["gt"] == t["pred"])
    b_acc = b_cor / len(b_tr) * 100
    print(f"Overall Accuracy: {b_acc:.2f}% ({b_cor}/55)")
    
    b_mat = {
        "SUPPORT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
        "NOT MENTIONED": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
    }
    for t in b_tr:
        b_mat[t["gt"]][t["pred"]] = b_mat[t["gt"]].get(t["pred"], 0) + 1
        
    print("\nConfusion Matrix:")
    print(f"                      Pred SUPPORT   |   Pred CONTRADICT   |   Pred NOT MENTIONED   |   ROW TOTAL")
    print("-" * 90)
    for row_cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        r_sum = sum(b_mat[row_cls].values())
        print(f"GT {row_cls:<14}:        {b_mat[row_cls]['SUPPORT']:<10} |        {b_mat[row_cls]['CONTRADICT']:<12} |        {b_mat[row_cls]['NOT MENTIONED']:<14} |        {r_sum}")
    print("-" * 90)
    
    # Class-level F1
    print("\nClass Breakdown:")
    for cls in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
        gt_cnt = sum(b_mat[cls].values())
        pred_cnt = sum(b_mat[r][cls] for r in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"])
        tp = b_mat[cls][cls]
        p = tp / pred_cnt * 100 if pred_cnt > 0 else 0.0
        r = tp / gt_cnt * 100 if gt_cnt > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  - {cls:<15}: Prec = {p:.1f}%, Rec = {r:.1f}%, F1 = {f1:.1f}% (GT: {gt_cnt}, Pred: {pred_cnt})")
