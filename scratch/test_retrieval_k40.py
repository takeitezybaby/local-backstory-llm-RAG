import sys
import numpy as np
import faiss
import json

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, normalize, strip_accents
from claimExtraction import extract_atomic_claims

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def get_entity_book(matched_key, entity_index, metadata):
    if not matched_key or matched_key not in entity_index:
        return None
    cids = entity_index[matched_key]
    if not cids:
        return None
    first_cid = cids[0]
    if first_cid < len(metadata):
        return metadata[first_cid].get("Book")
    return None

def deep_global_search(query, faiss_index, metadata, target_book=None, top_n=15):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    k_search = 100 if target_book else 20
    scores, indices = faiss_index.search(query_embed, k_search)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        Parentdata = metadata[idx]
        if target_book and Parentdata.get("Book") != target_book:
            continue
        results.append({
            "Score": float(score),
            "text": Parentdata["text"],
            "Book": Parentdata["Book"],
            "Chapter": Parentdata["Chapter"],
            "Parent Chunk id": Parentdata["Parent Chunk id"],
            "Atomic id": Parentdata["Atomic id"]
        })
        if len(results) >= top_n:
            break
    return results

claim3 = "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India."
claim_ent = extract_entity(claim3, entity_index)
matched_k = find_entity_in_index(claim_ent, entity_index)
book = get_entity_book(matched_k, entity_index, chunks)
print(f"Claim: {claim3}")
print(f"Entity: {claim_ent} -> Matched: {matched_k} -> Book: {book}")

res = deep_global_search(claim3, faiss_index, chunks, target_book=book, top_n=10)
print(f"Retrieved {len(res)} chunks:")
for i, r in enumerate(res[:5]):
    print(f"\n--- Result {i+1} (Ch: {r['Chapter']}) ---")
    print(r['text'])
