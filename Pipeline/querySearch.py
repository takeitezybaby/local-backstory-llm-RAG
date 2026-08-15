import faiss
import json
import numpy as np
import spacy
from embeddingsGeneration import createEmbeddings, normalize, loadChunks
import os

k =15
nlp = spacy.load("en_core_web_sm")


import re
import unicodedata

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def find_entity_in_index(raw_entity, index):
    if not raw_entity or not index:
        return None
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    if cleaned in index:
        return cleaned
    unaccented = strip_accents(cleaned)
    if unaccented in index:
        return unaccented
        
    sorted_keys = sorted(index.keys(), key=lambda x: len(x), reverse=True)
    for key in sorted_keys:
        if len(key) <= 3:
            continue
        key_unacc = strip_accents(key)
        if cleaned == key or unaccented == key_unacc:
            return key
        if cleaned in key or key in cleaned:
            return key
        if unaccented in key_unacc or key_unacc in unaccented:
            return key
    return None

#entity extraction from query prioritizing subject
def extract_entity(query, index=None) :
      doc = nlp(query)
      if index:
            for token in doc:
                  if token.dep_ in {"nsubj", "nsubjpass"}:
                        subj_text = " ".join([t.text for t in token.subtree if not t.is_punct])
                        k = find_entity_in_index(subj_text, index)
                        if k:
                              return k
                        k = find_entity_in_index(token.text, index)
                        if k:
                              return k
            for ent in doc.ents:
                  if ent.label_ == "PERSON":
                        k = find_entity_in_index(ent.text, index)
                        if k:
                              return k
            for ent in doc.ents:
                  k = find_entity_in_index(ent.text, index)
                  if k:
                              return k

      for ent in doc.ents :
            if ent.label_ == "PERSON" :
                  return re.sub(r"['’]s?\b", "", ent.text).strip(" .,!?:;\"'").lower()
            
      for token in doc :
            if token.dep_ in {"nsubj", "nsubjpass"} :
                  return re.sub(r"['’]s?\b", "", token.text).strip(" .,!?:;\"'").lower()
      return None


#load entity index
def loadEntityIndex (jsonpath) :
      with open(jsonpath, "r", encoding="utf-8") as f :
            return json.load(f)
      

#global search if entity not found (supports optional book filtering)
def global_search(query, faiss_index, metadata, target_book=None, top_k=12):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    k_search = 120 if target_book else 20
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
        if len(results) >= top_k:
            break

    return results
#subsetting using entity grounded embeddings
def subset_search (query, entity_index,  faiss_index, metadata) :
      query_embeddings = createEmbeddings(query)
      query_embeddings = normalize(query_embeddings)
      
      embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
      
      candidates = embeddings [entity_index]
      
      scores = np.dot(candidates, query_embeddings.T).flatten()

      topIndex = np.argsort(scores)[::-1][:k]

      results = []

      for i in topIndex :
            Parentdata = metadata[entity_index[i]]
            results.append({
                  "Score" : float(scores[i]),
                  "text" : Parentdata["text"],
                  "Book" : Parentdata["Book"],
                  "Chapter" : Parentdata["Chapter"],
                  "Parent Chunk id" : Parentdata["Parent Chunk id"],
                  "Atomic id" : Parentdata["Atomic id"]
            })
            

      return results



if __name__ == '__main__' :
      index = faiss.read_index(os.path.join("Data", "atomic.index"))
      atomicChunk = loadChunks(os.path.join("Data", "atomicChunks.json"))
      entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
      while(True) :
            query =  input("Enter a backstory claim (e or E to exit) :\n")
            if query in "eE" :
                  break
            query_entity = extract_entity(query)
            if query_entity and query_entity in entity_index :
                  results = subset_search(query,entity_index[query_entity], index, atomicChunk)
            else :
                  results = global_search(query,index,atomicChunk)
            print("\nTop matches:\n")
            for i, res in enumerate(results, 1):
                  print(f"{i}. Score: {res['Score']:.3f}")
                  print(f"   Text: {res['text']}")
                  print(f"   Chapter: {res['Chapter']}")
                  print(f"   Atomic ChunkID: {res['Atomic id']}")
                  print(f"   Book: {res['Book']}")
                  print("-" * 60)