import sys, os, json
sys.path.append("Pipeline")
from querySearch import loadEntityIndex

entity_index = loadEntityIndex("Data/entity.json")

def get_pooled_entity_chunks(entity_name, entity_index):
    if not entity_name:
        return []
    
    name_tokens = [t.lower() for t in entity_name.split() if len(t) > 2 and t.lower() not in {"lord", "lady", "captain", "major", "abbé", "baron", "count", "monsieur"}]
    pooled_cids = set()
    
    # 1. Exact match
    low_name = entity_name.lower().strip()
    if low_name in entity_index:
        pooled_cids.update(entity_index[low_name])
        
    # 2. Token / alias matching across all entity keys
    for key, cids in entity_index.items():
        key_low = key.lower()
        if any(t in key_low for t in name_tokens):
            pooled_cids.update(cids)
            
    return sorted(list(pooled_cids))

test_names = [
    "Gérard de Villefort",
    "Lord Edward Glenarvan",
    "Jacques Paganel",
    "Edmond Dantès",
    "Major MacNabb",
    "Mary Grant",
    "Albert de Morcerf"
]

print("================================================================================")
print("              ENTITY INDEX CHUNK POOLING BEFORE vs. AFTER                       ")
print("================================================================================")
for name in test_names:
    exact = entity_index.get(name.lower(), [])
    pooled = get_pooled_entity_chunks(name, entity_index)
    print(f"Entity: {name:<25} | Exact Match: {len(exact):>3} chunks | Pooled Aliases: {len(pooled):>4} chunks [OK]")

