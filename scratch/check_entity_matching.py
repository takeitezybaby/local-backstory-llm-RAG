import sys
import os
import json

sys.path.append("Pipeline")
from querySearch import loadEntityIndex, extract_entity

entity_index = loadEntityIndex("Data/entity.json")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"{'ID':<3} | {'Extracted Entity':<25} | {'In Entity Index?':<18} | {'Search Type Used':<26} | {'GT Verdict':<15}")
print("-" * 92)

for s in dataset:
    claim_text = s["user_input"]
    ent = extract_entity(claim_text)
    in_idx = ent in entity_index if ent else False
    search_type = "Entity-restricted-search" if in_idx else "Global-search"
    print(f"{s['id']:<3} | {str(ent):<25} | {str(in_idx):<18} | {search_type:<26} | {s['ground_truth_verdict']:<15}")
