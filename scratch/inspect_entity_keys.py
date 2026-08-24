import json

with open("Data/entity.json", "r", encoding="utf-8") as f:
    entity_index = json.load(f)

print("Total entities in index:", len(entity_index))
for target in ["fernand", "mondego", "leclere", "leclère", "mercedes", "mercédès", "dantes", "dantès", "ayrton", "macnabb", "mangles"]:
    matches = [k for k in entity_index.keys() if target in k]
    print(f"Target '{target}' matches: {matches}")
