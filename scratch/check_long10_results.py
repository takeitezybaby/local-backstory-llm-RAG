import json

with open('Data/eval_traces_long10.json', 'r', encoding='utf-8') as f:
    traces = json.load(f)

print('================================================================================')
print('                 10 EXTENDED PARAGRAPH CLAIMS (>= 200 WORDS)                   ')
print('================================================================================')

correct_cnt = 0
by_class = {"SUPPORT": {"total": 0, "correct": 0}, "CONTRADICT": {"total": 0, "correct": 0}, "NOT MENTIONED": {"total": 0, "correct": 0}}

for idx, t in enumerate(traces, 1):
    cid = t.get('id', idx)
    gt = t.get('ground_truth_verdict', 'UNKNOWN').strip().upper()
    v = t.get('actual_verdict', 'UNKNOWN').strip().upper()
    
    # Standard mapping
    v_mapped = v
    if v in ['COMPATIBLE', 'PARTIALLY COMPATIBLE']:
        v_mapped = 'SUPPORT'
    elif v in ['INCOMPATIBLE']:
        v_mapped = 'CONTRADICT'
    elif 'NOT SUPPORTED' in v or 'NOT MENTIONED' in v:
        v_mapped = 'NOT MENTIONED'
        
    is_match = (v_mapped == gt)
    if is_match:
        correct_cnt += 1
        
    if gt in by_class:
        by_class[gt]["total"] += 1
        if is_match:
            by_class[gt]["correct"] += 1
            
    mark = '[PASS]' if is_match else '[FAIL]'
    
    print(f"[{idx:02d}] ID {cid:03d} | GT: {gt:<13} | Pred: {v_mapped:<13} | {mark}")
    print(f"     Entity: {t.get('entity', 'N/A')}")
    print(f"     Claim excerpt: {t.get('user_input', '')[:85]}...")
    print(f"     Raw SLM output verdict: {v}")
    print('-' * 80)

acc = (correct_cnt / len(traces)) * 100
print(f"Overall Paragraph-Length Claim Accuracy: {acc:.2f}% ({correct_cnt}/{len(traces)})")
for cname, cdata in by_class.items():
    crec = (cdata['correct'] / cdata['total']) * 100 if cdata['total'] > 0 else 0
    print(f" - {cname:<14} Recall: {crec:.1f}% ({cdata['correct']}/{cdata['total']})")
print('================================================================================')
