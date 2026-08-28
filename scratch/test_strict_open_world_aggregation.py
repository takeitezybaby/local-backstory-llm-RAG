import json
import re

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

def parse_breakdown(resp_str):
    m = re.search(r"Breakdown:\s*(\{.*?\})", resp_str)
    if m:
        try:
            b_str = m.group(1).replace("'", '"')
            return json.loads(b_str)
        except Exception:
            pass
    return None

def test_aggregation_rules():
    print("=" * 80)
    print("   EVALUATING STRICT OPEN-WORLD AGGREGATION ON ALL 220 TRACES           ")
    print("=" * 80)
    
    rules = [
        ("Current Optimistic Aggregation (Any Support -> SUPPORT)",
         lambda s, c, nm, tot: "CONTRADICT" if ((c >= 1 and s == 0) or c >= 2) else ("SUPPORT" if s >= 1 else "NOT MENTIONED")),
        
        ("Strict Verification (SUPPORT only if ALL clauses supported, no unconfirmed details)",
         lambda s, c, nm, tot: "CONTRADICT" if c >= 1 else ("SUPPORT" if (s >= 1 and nm == 0) else "NOT MENTIONED")),
        
        ("Balanced OWA (SUPPORT if s >= 1 and nm <= 1 and c == 0, CONTRADICT if c >= 1)",
         lambda s, c, nm, tot: "CONTRADICT" if c >= 1 else ("SUPPORT" if (s >= 2 and nm == 0) or (s == 1 and nm == 0 and tot == 1) else "NOT MENTIONED"))
    ]
    
    for name, rule in rules:
        cor = 0
        matrix = {
            "SUPPORT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
            "CONTRADICT": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0},
            "NOT MENTIONED": {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
        }
        
        for t in traces:
            gt = t["ground_truth_verdict"]
            b = parse_breakdown(t.get("response", ""))
            if b:
                s = b.get("Supporting claims", 0)
                c = b.get("Contradicting claims", 0)
                nm = b.get("Not Mentioned claims", 0)
                tot = b.get("Total claims", 1)
                pred = rule(s, c, nm, tot)
            else:
                pred = t.get("mapped_actual_verdict", "NOT MENTIONED")
                
            matrix[gt][pred] = matrix[gt].get(pred, 0) + 1
            if pred == gt:
                cor += 1
                
        acc = cor / len(traces) * 100
        print(f"\n--- Rule: {name} ---")
        print(f"Overall Accuracy: {acc:.2f}% ({cor}/{len(traces)})")
        print(f"Hallucinated-SUPPORT on CONTRADICT: {matrix['CONTRADICT']['SUPPORT']} (Lower is safer!)")
        print(f"Hallucinated-SUPPORT on NOT MENTIONED: {matrix['NOT MENTIONED']['SUPPORT']} (Lower is safer!)")
        print("Confusion Matrix [Row=Ground Truth, Col=Predicted]:")
        print(f"               Pred SUPPORT | Pred CONTRADICT | Pred NOT MENTIONED")
        print(f"GT SUPPORT    :    {matrix['SUPPORT']['SUPPORT']:<8} |    {matrix['SUPPORT']['CONTRADICT']:<10} |    {matrix['SUPPORT']['NOT MENTIONED']:<10}")
        print(f"GT CONTRADICT :    {matrix['CONTRADICT']['SUPPORT']:<8} |    {matrix['CONTRADICT']['CONTRADICT']:<10} |    {matrix['CONTRADICT']['NOT MENTIONED']:<10}")
        print(f"GT NOT MENTION:    {matrix['NOT MENTIONED']['SUPPORT']:<8} |    {matrix['NOT MENTIONED']['CONTRADICT']:<10} |    {matrix['NOT MENTIONED']['NOT MENTIONED']:<10}")

test_aggregation_rules()
