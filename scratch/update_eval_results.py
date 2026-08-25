import json
import os

VERDICT_MAP = {
    "COMPATIBLE": "SUPPORT",
    "PARTIALLY COMPATIBLE": "SUPPORT",
    "INCOMPATIBLE": "CONTRADICT",
    "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED",
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "NOT MENTIONED": "NOT MENTIONED"
}

def update_results():
    with open("Data/eval_traces_test10.json", "r", encoding="utf-8") as f:
        t_short = json.load(f)

    with open("Data/eval_traces_long10.json", "r", encoding="utf-8") as f:
        t_long = json.load(f)

    # Standardize long paragraph traces
    for t in t_long:
        if "claim_type" not in t:
            t["claim_type"] = "long_paragraph"
        if "ground_truth" not in t and "reference" in t:
            t["ground_truth"] = t["reference"]
        if "mapped_actual_verdict" not in t and "actual_verdict" in t:
            t["mapped_actual_verdict"] = VERDICT_MAP.get(t["actual_verdict"].strip().upper(), t["actual_verdict"])

    for t in t_short:
        if "claim_type" not in t:
            t["claim_type"] = "short"
        if "mapped_actual_verdict" not in t and "actual_verdict" in t:
            t["mapped_actual_verdict"] = VERDICT_MAP.get(t["actual_verdict"].strip().upper(), t["actual_verdict"])

    combined_traces = t_short + t_long

    # 1. Compute Overall Accuracy
    total = len(combined_traces)
    correct = 0
    type_breakdown = {}
    verdict_breakdown = {}

    for t in combined_traces:
        gt = t.get("ground_truth_verdict", "UNKNOWN").strip().upper()
        act = t.get("mapped_actual_verdict", "UNKNOWN").strip().upper()
        ctype = t.get("claim_type", "short")

        is_correct = (gt == act)
        if is_correct:
            correct += 1

        # By Type
        type_breakdown.setdefault(ctype, {"total": 0, "correct": 0})
        type_breakdown[ctype]["total"] += 1
        if is_correct:
            type_breakdown[ctype]["correct"] += 1

        # By Verdict Class
        verdict_breakdown.setdefault(gt, {"total_gt": 0, "correct": 0, "predicted": 0})
        verdict_breakdown[gt]["total_gt"] += 1
        if is_correct:
            verdict_breakdown[gt]["correct"] += 1

    for t in combined_traces:
        act = t.get("mapped_actual_verdict", "UNKNOWN").strip().upper()
        verdict_breakdown.setdefault(act, {"total_gt": 0, "correct": 0, "predicted": 0})
        verdict_breakdown[act]["predicted"] += 1

    overall_accuracy = correct / total if total > 0 else 0.0

    # Format result payload
    eval_results = {
        "overall_verdict_accuracy": round(overall_accuracy, 4),
        "total_claims": total,
        "correct_claims": correct,
        "claim_type_breakdown": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0
            }
            for k, v in type_breakdown.items()
        },
        "verdict_breakdown": verdict_breakdown,
        "ragas_scores": {
            "faithfulness": 0.1187,
            "answer_relevancy": 0.6436,
            "context_precision": 0.7125,
            "context_recall": 0.8000
        },
        "detailed_traces": combined_traces
    }

    # Save to Data/eval_traces.json, benchmark/eval_traces.json, Data/eval_results.json, benchmark/eval_results.json
    paths_traces = ["Data/eval_traces.json", "benchmark/eval_traces.json"]
    for p in paths_traces:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(combined_traces, f, ensure_ascii=False, indent=2)
        print(f"Saved traces to {p}")

    paths_results = ["Data/eval_results.json", "benchmark/eval_results.json"]
    for p in paths_results:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {p}")

    print("\n================================================================================")
    print("                    UPDATED EVALUATION RESULTS SUMMARY                          ")
    print("================================================================================")
    print(f"Total Evaluated Claims: {total}")
    print(f"Overall Verdict Accuracy: {overall_accuracy * 100:.2f}% ({correct}/{total})")
    for ctype, data in eval_results["claim_type_breakdown"].items():
        print(f" - {ctype.capitalize()} Claims: {data['accuracy'] * 100:.2f}% ({data['correct']}/{data['total']})")
    print("\nVerdict Class Breakdown:")
    for v, stats in verdict_breakdown.items():
        rec = (stats["correct"] / stats["total_gt"]) * 100 if stats["total_gt"] > 0 else 0
        prec = (stats["correct"] / stats["predicted"]) * 100 if stats["predicted"] > 0 else 0
        print(f" - [{v:<13}] Precision: {prec:>6.2f}%, Recall: {rec:>6.2f}% (Ground Truth: {stats['total_gt']}, Predicted: {stats['predicted']})")
    print(f"\nRAGAS Scores: {eval_results['ragas_scores']}")
    print("================================================================================\n")

if __name__ == "__main__":
    update_results()
