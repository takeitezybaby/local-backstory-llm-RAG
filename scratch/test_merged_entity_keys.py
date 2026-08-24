import json
import re
import unicodedata

with open("Data/entity.json", "r", encoding="utf-8") as f:
    entity_index = json.load(f)

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def get_all_entity_chunks(raw_entity, index):
    if not raw_entity:
        return []
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    unacc = strip_accents(cleaned)
    
    # Collect all chunks from all matching keys in index
    all_chunks = set()
    for key, chunk_list in index.items():
        key_unacc = strip_accents(key)
        # Check if the entity is in the key or key is in entity (for tokens > 3 chars)
        if len(unacc) > 3 and (unacc in key_unacc or key_unacc in unacc):
            all_chunks.update(chunk_list)
            
    return sorted(list(all_chunks))

for name in ["Danglars", "Leclere", "Abbé Faria", "Fernand Mondego", "MacNabb", "Mary Grant", "Ayrton"]:
    chunks_found = get_all_entity_chunks(name, entity_index)
    print(f"Entity '{name}': {len(chunks_found)} total chunks pooled.")
