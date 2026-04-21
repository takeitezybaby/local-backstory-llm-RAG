import faiss
import json
import numpy as np
import spacy
from embeddingsGeneration import createEmbeddings, normalize, loadChunks
import os

k =15
nlp = spacy.load("en_core_web_sm")


#entity extraction from query
def extract_entity(query) :
      doc = nlp(query)
      for ent in doc.ents :
            if ent.label_ == "PERSON" :
                  return ent.text.lower()
            
      #Adding subject based entity detection to avoid unnecessary global serach
      for token in doc :
            if token.dep_ in {"nsubj", "nsubjpass"} :
                  return token.text.lower()
      return None


#load entity index
def loadEntityIndex (jsonpath) :
      with open(jsonpath, "r", encoding="utf-8") as f :
            return json.load(f)
      

#global search if entity not found
def global_search(query, faiss_index, metadata):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    scores, indices = faiss_index.search(query_embed, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        Parentdata = metadata[idx]
        results.append({
            "Score": float(score),
            "text": Parentdata["text"],
            "Book": Parentdata["Book"],
            "Chapter": Parentdata["Chapter"],
            "Parent Chunk id": Parentdata["Parent Chunk id"],
            "Atomic id": Parentdata["Atomic id"]
        })

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