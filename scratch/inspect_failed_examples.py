import json

with open('Data/eval_traces.json', 'r', encoding='utf-8') as f:
    traces = json.load(f)

print('=== CONTRADICT CLASSIFIED AS NOT MENTIONED (5 EXAMPLES) ===')
cnt = 0
for t in traces:
    gt = t['ground_truth_verdict'].strip().upper()
    act = t.get('mapped_actual_verdict', '').strip().upper()
    if gt == 'CONTRADICT' and act == 'NOT MENTIONED':
        cnt += 1
        print(f"[{cnt}] ID {t['id']}: {t['user_input']}")
        print(f"    GT Ref: {t['ground_truth']}")
        print(f"    System Response: {t['response']}")
        print(f"    Retrieved Contexts ({len(t.get('contexts', []))} chunks):")
        for i, c in enumerate(t.get('contexts', [])[:3]):
            print(f"      ({i+1}) {c[:120]}...")
        print('-'*70)
        if cnt >= 5:
            break

print('\n=== SUPPORT CLASSIFIED AS NOT MENTIONED (5 EXAMPLES) ===')
cnt = 0
for t in traces:
    gt = t['ground_truth_verdict'].strip().upper()
    act = t.get('mapped_actual_verdict', '').strip().upper()
    if gt == 'SUPPORT' and act == 'NOT MENTIONED':
        cnt += 1
        print(f"[{cnt}] ID {t['id']}: {t['user_input']}")
        print(f"    GT Ref: {t['ground_truth']}")
        print(f"    System Response: {t['response']}")
        print(f"    Retrieved Contexts ({len(t.get('contexts', []))} chunks):")
        for i, c in enumerate(t.get('contexts', [])[:3]):
            print(f"      ({i+1}) {c[:120]}...")
        print('-'*70)
        if cnt >= 5:
            break
