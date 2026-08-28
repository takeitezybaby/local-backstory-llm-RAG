import json
import os
import sys
import time
import requests
import faiss
import numpy as np

# Ensure Pipeline imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Pipeline"))

from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks, get_canonical_profile
from claimExtraction import extract_atomic_claims
from reranker import rerank_candidates
from aggregation import aggregate_results

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

def query_llm(prompt, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120}
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            time.sleep(2)
    return ""

def build_prompt(claim, evidence_list, entity="", use_canonical=True):
    evidence_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list[:10])])
    profile_section = ""
    if use_canonical and entity:
        prof = get_canonical_profile(entity)
        if prof:
            profile_section = f"Canonical Knowledge about {entity}:\n{prof}\n\n"
            
    prompt = f"""<|user|>
You are a precise literary fact-checker. Evaluate the Claim against the Canonical Knowledge and Novel Excerpts.

Claim: "{claim}"
Character: "{entity}"

{profile_section}Source Excerpts:
{evidence_text}

CLASSIFICATION RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with the character's canonical identity, parentage, role, allegiance, or fate (e.g. wrong parent, claiming they are a pirate/traitor/convict when they are noble/loyal, claiming they died when they lived or were executed instead of dying of illness).
2. SUPPORT: The claim is directly confirmed true by the excerpts or canonical facts.
3. NOT MENTIONED: The claim describes an unmentioned private past, investment, hobby, or background detail (e.g. investing in railway shares, painting landscapes, learning harp in Vienna, writing a personal memoir, or past job prior to the novel) that is simply absent from the text without creating an impossible contradiction.

End on the final line with exactly:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
    return prompt

class AblationRunner:
    def __init__(self, dataset_path="benchmark/eval_dataset.json"):
        self.dataset_path = dataset_path
        self.atomic_chunks = loadChunks(os.path.join("Data", "atomicChunks.json"))
        self.entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
        self.faiss_index = faiss.read_index(os.path.join("Data", "atomic.index"))
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def run_configuration(self, config_name, use_canonical=True, use_reranker=True, use_pooling=True, is_vanilla=False, limit=None):
        data = self.dataset[:limit] if limit else self.dataset
        print(f"\n================================================================================")
        print(f"   RUNNING CONFIGURATION: {config_name} (Total: {len(data)} claims)            ")
        print(f"================================================================================")
        
        traces = []
        t_start = time.time()
        
        for idx, s in enumerate(data, 1):
            claim_text = s["user_input"]
            claim_type = s.get("claim_type", "short")
            gt = s["ground_truth_verdict"].strip().upper()
            t0 = time.time()
            
            if is_vanilla:
                # Vanilla Dense RAG: No decomposition, simple global search
                query_embed = normalize(createEmbeddings(claim_text))
                scores, indices = self.faiss_index.search(query_embed, 5)
                evidence = [self.atomic_chunks[i] for i in indices[0] if i < len(self.atomic_chunks)]
                prompt = build_prompt(claim_text, evidence, "", use_canonical=False)
                resp = query_llm(prompt)
                sub_verifications = [{"Claim": claim_text, "Evidence": evidence, "Verification_result": resp}]
            else:
                # Decomposed verification
                sub_claims = extract_atomic_claims(claim_text)
                sub_verifications = []
                for sub in sub_claims:
                    ent = extract_entity(sub, self.entity_index) if use_pooling else ""
                    pooled_cids = get_pooled_entity_chunks(ent, self.entity_index) if (use_pooling and ent) else []
                    
                    target_book = None
                    if pooled_cids and pooled_cids[0] < len(self.atomic_chunks):
                        target_book = self.atomic_chunks[pooled_cids[0]].get("Book")
                        
                    global_res = global_search(sub, self.faiss_index, self.atomic_chunks, target_book=target_book, top_k=25)
                    entity_res = subset_search(sub, pooled_cids, self.faiss_index, self.atomic_chunks, top_k=25) if pooled_cids else []
                    
                    seen = set()
                    cand = []
                    for r in global_res + entity_res:
                        txt = r["text"].strip()
                        if txt not in seen:
                            seen.add(txt)
                            cand.append(r)
                            
                    if use_reranker:
                        evidence = rerank_candidates(sub, cand, top_k=8)
                    else:
                        evidence = cand[:8]
                        
                    prompt = build_prompt(sub, evidence, ent, use_canonical=use_canonical)
                    resp = query_llm(prompt)
                    sub_verifications.append({"Claim": sub, "Evidence": evidence, "Verification_result": resp})
                    
            agg = aggregate_results(sub_verifications)
            pred_raw = agg["Final Verdict"]
            mapped = "SUPPORT" if "COMPATIBLE" in pred_raw else ("CONTRADICT" if "INCOMPATIBLE" in pred_raw else "NOT MENTIONED")
            is_match = (mapped == gt)
            elapsed = time.time() - t0
            
            mark = "[PASS]" if is_match else "[FAIL]"
            print(f"[{idx:02d}/{len(data)}] ({claim_type[:5].upper()}) GT: {gt:<13} | Pred: {mapped:<13} | {mark} ({elapsed:.1f}s)")
            
            traces.append({
                "id": s["id"],
                "claim_type": claim_type,
                "ground_truth": gt,
                "predicted": mapped,
                "correct": is_match,
                "latency": elapsed
            })
            
        total_time = time.time() - t_start
        stats = self.compute_metrics(traces, total_time)
        return stats

    def compute_metrics(self, traces, total_time):
        total = len(traces)
        correct = sum(1 for t in traces if t["correct"])
        acc = round(correct / total, 4) if total > 0 else 0.0
        
        # Granular types
        types = {}
        for ct in ["short", "long", "long_paragraph"]:
            subset = [t for t in traces if t["claim_type"] == ct]
            c_cnt = sum(1 for t in subset if t["correct"])
            types[ct] = {
                "total": len(subset),
                "correct": c_cnt,
                "accuracy": round(c_cnt / len(subset), 4) if subset else 0.0
            }
            
        # Class stats
        classes = {}
        macro_f1_list = []
        for cls_name in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            gt_cnt = sum(1 for t in traces if t["ground_truth"] == cls_name)
            pred_cnt = sum(1 for t in traces if t["predicted"] == cls_name)
            tp = sum(1 for t in traces if t["ground_truth"] == cls_name and t["predicted"] == cls_name)
            
            prec = round(tp / pred_cnt, 4) if pred_cnt > 0 else 0.0
            rec = round(tp / gt_cnt, 4) if gt_cnt > 0 else 0.0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
            macro_f1_list.append(f1)
            
            classes[cls_name] = {
                "gt_count": gt_cnt,
                "pred_count": pred_cnt,
                "true_positives": tp,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }
            
        macro_f1 = round(sum(macro_f1_list) / len(macro_f1_list), 4) if macro_f1_list else 0.0
        avg_latency = round(total_time / total, 2) if total > 0 else 0.0
        
        return {
            "total_claims": total,
            "correct_claims": correct,
            "overall_accuracy": acc,
            "macro_f1": macro_f1,
            "avg_latency_sec": avg_latency,
            "claim_type_breakdown": types,
            "class_breakdown": classes
        }

def export_latex_table(results_dict, output_tex="benchmark/ablation_results.tex"):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{System Configuration} & \textbf{Overall Acc (\%)} & \textbf{Macro F1 (\%)} & \textbf{Short Acc (\%)} & \textbf{Narrative Acc (\%)} & \textbf{Paragraph Acc (\%)} & \textbf{Latency (s)} \\",
        r"\midrule"
    ]
    for config_name, m in results_dict.items():
        o_acc = m["overall_accuracy"] * 100
        m_f1 = m["macro_f1"] * 100
        s_acc = m["claim_type_breakdown"].get("short", {}).get("accuracy", 0.0) * 100
        n_acc = m["claim_type_breakdown"].get("long", {}).get("accuracy", 0.0) * 100
        p_acc = m["claim_type_breakdown"].get("long_paragraph", {}).get("accuracy", 0.0) * 100
        lat = m["avg_latency_sec"]
        
        bold_pre = r"\textbf{" if "Ours" in config_name or "Full" in config_name else ""
        bold_post = "}" if "Ours" in config_name or "Full" in config_name else ""
        
        line = f"{bold_pre}{config_name}{bold_post} & {bold_pre}{o_acc:.1f}{bold_post} & {bold_pre}{m_f1:.1f}{bold_post} & {s_acc:.1f} & {n_acc:.1f} & {p_acc:.1f} & {lat:.1f} \\\\"
        lines.append(line)
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Ablation Study across 110 Benchmark Claims on Local Phi-3.5 (3.8B). Demonstrates the contribution of automated canonical knowledge grounding, cross-encoder neural reranking, and deep entity pooling.}",
        r"\label{tab:ablation_results}",
        r"\end{table*}"
    ])
    
    tex_content = "\n".join(lines)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"\nLaTeX table exported successfully to '{output_tex}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims for quick test")
    args = parser.parse_args()
    
    runner = AblationRunner()
    
    ablation_results = {}
    
    # 1. Full Backstory RAG (Ours)
    ablation_results["Backstory RAG (Full System - Ours)"] = runner.run_configuration(
        "Backstory RAG (Full System)", use_canonical=True, use_reranker=True, use_pooling=True, limit=args.limit
    )
    
    # 2. w/o Canonical Grounding
    ablation_results["w/o Canonical Knowledge Grounding"] = runner.run_configuration(
        "w/o Canonical Grounding", use_canonical=False, use_reranker=True, use_pooling=True, limit=args.limit
    )
    
    # 3. w/o Cross-Encoder Reranker
    ablation_results["w/o Cross-Encoder Reranker"] = runner.run_configuration(
        "w/o Cross-Encoder Reranker", use_canonical=True, use_reranker=False, use_pooling=True, limit=args.limit
    )
    
    # 4. w/o Entity Pooling
    ablation_results["w/o Entity Pooling (Global Search Only)"] = runner.run_configuration(
        "w/o Entity Pooling", use_canonical=True, use_reranker=True, use_pooling=False, limit=args.limit
    )
    
    # 5. Vanilla Dense RAG
    ablation_results["Vanilla Dense RAG (Baseline)"] = runner.run_configuration(
        "Vanilla Dense RAG Baseline", is_vanilla=True, limit=args.limit
    )
    
    # Save JSON results
    with open("benchmark/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        
    # Export LaTeX Table
    export_latex_table(ablation_results)
