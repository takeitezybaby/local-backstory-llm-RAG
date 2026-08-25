import json
import re

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

def clean_extract_verdict(raw_text):
    if not raw_text:
        return "NOT MENTIONED"
    
    # 1. Search for Verdict: at the end of the text
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        line_up = line.upper()
        if "VERDICT:" in line_up:
            v_part = line_up.split("VERDICT:")[1].strip()
            if "CONTRADICT" in v_part or "INCOMPATIBLE" in v_part:
                return "CONTRADICT"
            elif "SUPPORT" in v_part and "NOT" not in v_part:
                return "SUPPORT"
            elif "NOT MENTIONED" in v_part or "UNMENTIONED" in v_part:
                return "NOT MENTIONED"
        if line_up.startswith("VERDICT"):
            if "CONTRADICT" in line_up:
                return "CONTRADICT"
            elif "SUPPORT" in line_up and "NOT" not in line_up:
                return "SUPPORT"
            elif "NOT MENTIONED" in line_up:
                return "NOT MENTIONED"
                
    # 2. Check the very last line directly
    if lines:
        last = lines[-1].upper()
        if "CONTRADICT" in last:
            return "CONTRADICT"
        if "SUPPORT" in last and "NOT" not in last:
            return "SUPPORT"
        if "NOT MENTIONED" in last:
            return "NOT MENTIONED"
            
    # 3. Overall keyword frequency
    raw_up = raw_text.upper()
    if "CONTRADICT" in raw_up:
        return "CONTRADICT"
    if "SUPPORT" in raw_up and "NOT MENTIONED" not in raw_up:
        return "SUPPORT"
    return "NOT MENTIONED"

print("Auditing clean extraction vs ground truth on all 110 traces...")
cor = 0
for t in traces:
    gt = t["ground_truth_verdict"].strip().upper()
    resp = t.get("response", "")
    pred = clean_extract_verdict(resp)
    if pred == gt:
        cor += 1

print(f"Direct Response Accuracy on all 110 claims: {cor}/110 ({cor/110*100:.2f}%)")
