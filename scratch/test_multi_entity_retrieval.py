import sys
import json
import faiss
import spacy
import re
import unicodedata

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, global_search, subset_search, find_entity_in_index
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results

nlp = spacy.load("en_core_web_sm")
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def extract_all_entities_from_claim(claim, index):
    doc = nlp(claim)
    found_keys = []
    
    # 1. Named entities
    for ent in doc.ents:
        k = find_entity_in_index(ent.text, index)
        if k and k not in found_keys:
            found_keys.append(k)
            
    # 2. Noun chunks & subjects
    for chunk in doc.noun_chunks:
        k = find_entity_in_index(chunk.text, index)
        if k and k not in found_keys:
            found_keys.append(k)
            
    # 3. Individual capitalized tokens
    for token in doc:
        if token.text[0].isupper() and len(token.text) > 2:
            k = find_entity_in_index(token.text, index)
            if k and k not in found_keys:
                found_keys.append(k)
                
    return found_keys

test_cases = [
    (8, "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain."),
    (9, "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."),
    (11, "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel."),
    (12, "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."),
    (13, "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon.")
]

for cid, text in test_cases:
    print(f"\n================ CLAIM {cid} ================")
    print("Claim:", text)
    ents = extract_all_entities_from_claim(text, entity_index)
    print("Found Entities in index:", ents)
