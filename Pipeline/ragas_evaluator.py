import json
import os
import sys

def evaluate_traces(traces_path="Data/eval_traces.json", output_path="Data/eval_results.json"):
    if not os.path.exists(traces_path):
        print(f"Error: Traces file '{traces_path}' not found. Please run 'python Pipeline/eval_runner.py' first.")
        return

    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    print(f"Loaded {len(traces)} traces from {traces_path}.\n")

    # Map pipeline verdicts to ground truth standards
    VERDICT_MAP = {
        "COMPATIBLE": "SUPPORT",
        "PARTIALLY COMPATIBLE": "SUPPORT",
        "INCOMPATIBLE": "CONTRADICT",
        "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED",
        "SUPPORT": "SUPPORT",
        "CONTRADICT": "CONTRADICT",
        "NOT MENTIONED": "NOT MENTIONED"
    }

    # 1. Compute Classification Metrics (Accuracy, Precision per Verdict class)
    total = len(traces)
    correct_verdicts = 0

    for t in traces:
        t["mapped_actual_verdict"] = VERDICT_MAP.get(t["actual_verdict"].strip().upper(), t["actual_verdict"])
        if t["ground_truth_verdict"] == t["mapped_actual_verdict"]:
            correct_verdicts += 1

    accuracy = correct_verdicts / total if total > 0 else 0.0

    verdict_stats = {}
    for t in traces:
        gt = t["ground_truth_verdict"]
        act = t["mapped_actual_verdict"]
        verdict_stats.setdefault(gt, {"total_gt": 0, "correct": 0, "predicted": 0})
        verdict_stats[gt]["total_gt"] += 1
        if gt == act:
            verdict_stats[gt]["correct"] += 1

    for t in traces:
        act = t["mapped_actual_verdict"]
        verdict_stats.setdefault(act, {"total_gt": 0, "correct": 0, "predicted": 0})
        verdict_stats[act]["predicted"] += 1

    print("==================================================")
    print("        VERDICT CLASSIFICATION ACCURACY           ")
    print("==================================================")
    print(f"Overall Verdict Accuracy: {accuracy * 100:.2f}% ({correct_verdicts}/{total})\n")

    for v, stats in verdict_stats.items():
        rec = (stats["correct"] / stats["total_gt"]) if stats["total_gt"] > 0 else 0
        prec = (stats["correct"] / stats["predicted"]) if stats["predicted"] > 0 else 0
        print(f"Verdict [{v}]: Precision = {prec:.2f}, Recall = {rec:.2f} (Count: {stats['total_gt']})")
    print("==================================================\n")

    # 2. RAGAS Metrics Evaluation
    ragas_scores = {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings

        print("Initializing RAGAS with local Ollama LLM and Embeddings...")
        eval_llm = ChatOllama(model="koesn/mistral-7b-instruct:latest")
        eval_embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Convert traces to Hugging Face Dataset format
        ragas_data = {
            "question": [t["question"] for t in traces],
            "contexts": [t["contexts"] for t in traces],
            "answer": [t["response"] for t in traces],
            "ground_truth": [t["ground_truth"] for t in traces]
        }
        dataset = Dataset.from_dict(ragas_data)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        print("Running RAGAS evaluation metrics...")
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings
        )

        ragas_scores = eval_result
        print("\n==================================================")
        print("             RAGAS METRICS RESULTS                ")
        print("==================================================")
        print(eval_result)
        print("==================================================\n")

    except ImportError:
        print("Note: 'ragas' or 'datasets' package not installed. Skipping RAGAS LLM-as-a-judge score computation.")
        print("To run RAGAS metrics, install via: pip install ragas datasets langchain-community\n")
    except Exception as e:
        print(f"RAGAS metric computation warning: {e}\n")

    # 3. Save combined summary results
    results_payload = {
        "overall_verdict_accuracy": accuracy,
        "total_claims": total,
        "correct_claims": correct_verdicts,
        "verdict_breakdown": verdict_stats,
        "ragas_scores": str(ragas_scores) if ragas_scores else None,
        "detailed_traces": traces
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    print(f"Full evaluation report saved successfully to '{output_path}'.")

if __name__ == "__main__":
    evaluate_traces()
