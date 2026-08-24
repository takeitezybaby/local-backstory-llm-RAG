import json
import re
import unicodedata

with open("Data/entity.json", "r", encoding="utf-8") as f:
    entity_index = json.load(f)

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def find_entity_in_index(raw_entity, index):
    if not raw_entity:
        return None
    # 1. Clean possessive
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    
    # 2. Exact match
    if cleaned in index:
        return cleaned
    
    # 3. Unaccented match
    unaccented = strip_accents(cleaned)
    if unaccented in index:
        return unaccented
    
    # 4. Check tokens or partial matches
    for key in index.keys():
        key_unacc = strip_accents(key)
        if cleaned == key or unaccented == key_unacc:
            return key
        if cleaned in key or key in cleaned:
            return key
        if unaccented in key_unacc or key_unacc in unaccented:
            return key
            
    return None

test_entities = [
    "Edward Glenarvan", "Tom Austin", "Jacques Paganel", "Captain Harry Grant",
    "Edmond Dantès", "Gérard de Villefort", "Abbé Faria", "Danglars",
    "Major MacNabb", "Ayrton", "Mary Grant", "Captain Leclère",
    "Fernand Mondego", "Mercédès", "Lord Glenarvan", "Lady Helena",
    "Thalcave", "M. Morrel", "Villefort", "Haydée"
]

print(f"{'Raw Entity':<25} | {'Matched Key in entity.json':<30} | {'Status':<10}")
print("-" * 70)
for ent in test_entities:
    matched = find_entity_in_index(ent, entity_index)
    status = "SUCCESS" if matched else "MISS"
    print(f"{ent:<25} | {str(matched):<30} | {status:<10}")
