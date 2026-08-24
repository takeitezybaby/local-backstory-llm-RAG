import sys
import json
import faiss
import numpy as np
import spacy
import re
import requests

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks, createEmbeddings, normalize
from querySearch import loadEntityIndex, find_entity_in_index, extract_entity
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def compute_keyword_score(query, text):
    # Extract salient non-stopword terms
    doc = nlp(query.lower())
    query_terms = [t.text for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2]
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for term in query_terms if term in text_lower)
    return matches / len(query_terms)

def hybrid_keyword_dense_search(query, faiss_index, metadata, entity_key=None, k=15):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    # Reconstruct embeddings
    embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    
    if entity_key and entity_key in entity_index:
        candidate_ids = entity_index[entity_key]
    else:
        candidate_ids = list(range(len(metadata)))
        
    candidate_embeddings = embeddings[candidate_ids]
    dense_scores = np.dot(candidate_embeddings, query_embed.T).flatten()
    
    scored_results = []
    for i, idx in enumerate(candidate_ids):
        item = metadata[idx]
        kw_score = compute_keyword_score(query, item["text"])
        combined_score = float(dense_scores[i]) + 0.4 * kw_score
        scored_results.append((combined_score, item))
        
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored_results[:k]]

test_claims = [
    (7, "Abbé Faria was an Italian priest imprisoned in the Château d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dantès."),
    (8, "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon.")
]

for cid, text in test_claims:
    print(f"\n================ CLAIM {cid} ================")
    ent = extract_entity(text, entity_index)
    results = hybrid_keyword_dense_search(text, faiss_index, chunks, entity_key=ent, k=3)
    print("Entity:", ent)
    for r in results:
        print("  * Top snippet:", r["text"][:140])
