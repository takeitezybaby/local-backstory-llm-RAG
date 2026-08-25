import json
import re

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

def parse_breakdown(resp_str):
    # Breakdown: {'Supporting claims': 1, 'Contradicting claims': 0, 'Not Mentioned claims': 1, 'Total claims': 2}
    m = re.search(r"Breakdown:\s*(\{.*?\})", resp_str)
    if m:
        try:
            # Clean string to valid json
            b_str = m.group(1).replace("'", '"')
            return json.loads(b_str)
        except Exception:
            pass
    return None

def test_decision_rules():
    print("Testing different aggregation thresholds on all 110 traces...")
    
    thresholds = [
        ("Strict Contradict First (Baseline)", lambda s, c, nm, tot: "CONTRADICT" if c > 0 else ("SUPPORT" if s > 0 else "NOT MENTIONED")),
        ("Majority Vote (s vs c)", lambda s, c, nm, tot: "CONTRADICT" if c > s else ("SUPPORT" if s > c else "NOT MENTIONED")),
        ("Confidence-Weighted: Contradict if c >= 1 and s == 0, or c >= 2", lambda s, c, nm, tot: "CONTRADICT" if (c >= 1 and s == 0) or c >= 2 else ("SUPPORT" if s >= 1 else "NOT MENTIONED")),
        ("Balanced Threshold: Support if s >= 1 and c == 0; Contradict if c >= 1 and s == 0", lambda s, c, nm, tot: "SUPPORT" if s >= 1 and c == 0 else ("CONTRADICT" if c >= 1 and s == 0 else ("CONTRADICT" if c > s else ("SUPPORT" if s > c else "NOT MENTIONED"))))
    ]
    
    for name, rule in thresholds:
        cor = 0
        v_breakdown = {"SUPPORT": 0, "CONTRADICT": 0, "NOT MENTIONED": 0}
        type_cor = {"short": 0, "long": 0, "long_paragraph": 0}
        type_tot = {"short": 0, "long": 0, "long_paragraph": 0}
        
        for t in traces:
            gt = t["ground_truth_verdict"].strip().upper()
            ctype = t.get("claim_type", "short")
            type_tot[ctype] += 1
            
            b = parse_breakdown(t["response"])
            if b:
                s = b.get("Supporting claims", 0)
                c = b.get("Contradicting claims", 0)
                nm = b.get("Not Mentioned claims", 0)
                tot = b.get("Total claims", 1)
                pred = rule(s, c, nm, tot)
            else:
                pred = t.get("mapped_actual_verdict", "NOT MENTIONED")
                
            if pred == gt:
                cor += 1
                type_cor[ctype] += 1
                
        acc = cor / len(traces) * 100
        print(f"\n--- {name} ---")
        print(f"Overall Accuracy: {acc:.2f}% ({cor}/{len(traces)})")
        for ct in ["short", "long", "long_paragraph"]:
            ct_acc = type_cor[ct] / type_tot[ct] * 100 if type_tot[ct] > 0 else 0
            print(f"  {ct:<15}: {ct_acc:.2f}% ({type_cor[ct]}/{type_tot[ct]})")

test_decision_rules()
