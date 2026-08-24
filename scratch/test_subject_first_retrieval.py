import spacy
import re
import unicodedata
import json
import sys
import requests
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, global_search, subset_search, find_entity_in_index
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def extract_primary_entity(query, index):
    doc = nlp(query)
    
    # 1. First check if subject (nsubj) is a recognized entity in index
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            # check subtree or token
            subj_text = " ".join([t.text for t in token.subtree if not t.is_punct])
            k = find_entity_in_index(subj_text, index)
            if k:
                return k
            k = find_entity_in_index(token.text, index)
            if k:
                return k
                
    # 2. Check entities in order of appearance (first named person)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            k = find_entity_in_index(ent.text, index)
            if k:
                return k
                
    # 3. Check any named entity
    for ent in doc.ents:
        k = find_entity_in_index(ent.text, index)
        if k:
            return k
            
    return None

print(f"{'ID':<3} | {'Claim Prefix':<40} | {'Primary Entity':<20}")
print("-" * 70)
for s in dataset:
    ent = extract_primary_entity(s["user_input"], entity_index)
    print(f"{s['id']:<3} | {s['user_input'][:40]:<40} | {str(ent):<20}")
