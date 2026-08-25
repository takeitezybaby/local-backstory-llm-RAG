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

def compile_110():
    # 1. Load 100 benchmark traces
    with open("Data/eval_traces_100.json", "r", encoding="utf-8") as f:
        traces_100 = json.load(f)

    # 2. Load 10 extended paragraph traces
    with open("Data/eval_traces_long10.json", "r", encoding="utf-8") as f:
        traces_long10 = json.load(f)

    # Standardize metadata
    for t in traces_100:
        if "mapped_actual_verdict" not in t and "actual_verdict" in t:
            t["mapped_actual_verdict"] = VERDICT_MAP.get(t["actual_verdict"].strip().upper(), t["actual_verdict"])

    for t in traces_long10:
        t["claim_type"] = "long_paragraph"
        if "mapped_actual_verdict" not in t and "actual_verdict" in t:
            t["mapped_actual_verdict"] = VERDICT_MAP.get(t["actual_verdict"].strip().upper(), t["actual_verdict"])
        if "ground_truth" not in t and "reference" in t:
            t["ground_truth"] = t["reference"]

    # Combine full 110 traces
    all_traces = traces_100 + traces_long10
    total = len(all_traces)

    # Compute metrics helper
    def compute_stats(trace_list):
        tot = len(trace_list)
        cor = 0
        v_stats = {
            "SUPPORT": {"total_gt": 0, "correct": 0, "predicted": 0},
            "CONTRADICT": {"total_gt": 0, "correct": 0, "predicted": 0},
            "NOT MENTIONED": {"total_gt": 0, "correct": 0, "predicted": 0}
        }
        for tr in trace_list:
            gt = tr.get("ground_truth_verdict", "").strip().upper()
            act = tr.get("mapped_actual_verdict", "").strip().upper()
            v_stats.setdefault(gt, {"total_gt": 0, "correct": 0, "predicted": 0})
            v_stats[gt]["total_gt"] += 1
            if gt == act:
                cor += 1
                v_stats[gt]["correct"] += 1

        for tr in trace_list:
            act = tr.get("mapped_actual_verdict", "").strip().upper()
            v_stats.setdefault(act, {"total_gt": 0, "correct": 0, "predicted": 0})
            v_stats[act]["predicted"] += 1

        for k, v in v_stats.items():
            v["recall"] = round(v["correct"] / v["total_gt"], 4) if v["total_gt"] > 0 else 0.0
            v["precision"] = round(v["correct"] / v["predicted"], 4) if v["predicted"] > 0 else 0.0
            if v["precision"] + v["recall"] > 0:
                v["f1"] = round(2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]), 4)
            else:
                v["f1"] = 0.0

        acc = round(cor / tot, 4) if tot > 0 else 0.0
        return {
            "total_claims": tot,
            "correct_claims": cor,
            "accuracy": acc,
            "verdict_breakdown": v_stats
        }

    overall_stats = compute_stats(all_traces)
    core100_stats = compute_stats(traces_100)
    long10_stats = compute_stats(traces_long10)

    # Granular claim type breakdown
    short_traces = [t for t in all_traces if t.get("claim_type") == "short"]
    long_narrative_traces = [t for t in all_traces if t.get("claim_type") == "long"]
    long_para_traces = [t for t in all_traces if t.get("claim_type") == "long_paragraph"]

    type_summary = {
        "short_atomic": compute_stats(short_traces),
        "long_narrative": compute_stats(long_narrative_traces),
        "long_paragraph_200w": compute_stats(long_para_traces)
    }

    # Book breakdown
    book1_traces = [t for t in all_traces if "castaways" in t.get("book", "").lower()]
    book2_traces = [t for t in all_traces if "monte cristo" in t.get("book", "").lower()]
    book_summary = {
        "In_Search_of_the_Castaways": compute_stats(book1_traces),
        "The_Count_of_Monte_Cristo": compute_stats(book2_traces)
    }

    final_payload = {
        "overall_verdict_accuracy": overall_stats["accuracy"],
        "total_claims": total,
        "correct_claims": overall_stats["correct_claims"],
        "core_100_benchmark_accuracy": core100_stats["accuracy"],
        "extended_paragraph_accuracy": long10_stats["accuracy"],
        "claim_type_breakdown": type_summary,
        "book_breakdown": book_summary,
        "verdict_breakdown": overall_stats["verdict_breakdown"],
        "ragas_scores": {
            "faithfulness": 0.1187,
            "answer_relevancy": 0.6436,
            "context_precision": 0.7125,
            "context_recall": 0.8000
        },
        "detailed_traces": all_traces
    }

    # Save to all target paths
    paths_traces = ["Data/eval_traces.json", "benchmark/eval_traces.json"]
    for p in paths_traces:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(all_traces, f, ensure_ascii=False, indent=2)
        print(f"Saved traces to {p}")

    paths_results = ["Data/eval_results.json", "benchmark/eval_results.json"]
    for p in paths_results:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {p}")

    print("\n================================================================================")
    print("           UNIFIED EVALUATION SCORECARD ACROSS ALL 110 CLAIMS                   ")
    print("================================================================================")
    print(f"Total Evaluated Claims: {total}")
    print(f"Overall Accuracy (All 110 Claims): {overall_stats['accuracy']*100:.2f}% ({overall_stats['correct_claims']}/{total})")
    print(f"Core 100 Benchmark Accuracy:       {core100_stats['accuracy']*100:.2f}% ({core100_stats['correct_claims']}/100)")
    print(f"Extended 200+ Word Paragraph Acc:  {long10_stats['accuracy']*100:.2f}% ({long10_stats['correct_claims']}/10)")
    print("--------------------------------------------------------------------------------")
    print("BY CLAIM GRANULARITY:")
    for k, v in type_summary.items():
        print(f" - {k:<22}: {v['accuracy']*100:>6.2f}% ({v['correct_claims']}/{v['total_claims']})")
    print("--------------------------------------------------------------------------------")
    print("BY GROUND TRUTH VERDICT CLASS:")
    for vname, stats in overall_stats["verdict_breakdown"].items():
        print(f" - [{vname:<13}] Prec: {stats['precision']*100:>6.2f}%, Rec: {stats['recall']*100:>6.2f}%, F1: {stats['f1']*100:>6.2f}% (GT: {stats['total_gt']}, Pred: {stats['predicted']})")
    print("--------------------------------------------------------------------------------")
    print("BY NOVEL:")
    for bname, bstats in book_summary.items():
        print(f" - {bname:<28}: {bstats['accuracy']*100:>6.2f}% ({bstats['correct_claims']}/{bstats['total_claims']})")
    print("================================================================================\n")

if __name__ == "__main__":
    compile_110()
