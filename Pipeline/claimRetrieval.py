import json
import faiss
from embeddingsGeneration import createEmbeddings, loadChunks, normalize
from querySearch import extract_entity, loadEntityIndex, global_search, subset_search, find_entity_in_index
from claimExtraction import extract_atomic_claims
import os

k = 10


#Post filtering to improve retrieval
def filterByEntity(results, entity):
      filtered = []
      entity_tokens = [t for t in entity.lower().split() if len(t) > 2]
      for r in results :
            text_lower = r["text"].lower()
            if any(token in text_lower for token in entity_tokens):
                  filtered.append(r)
      return filtered if filtered else results


#claim retrieval pipeline
def claim_retrieval(backstory, metadata, faiss_index, entity_index) :
      claims = extract_atomic_claims(backstory)
      retrievals = []
      for claim in claims :
            claim_entity = extract_entity(claim)
            matched_key = find_entity_in_index(claim_entity, entity_index)
            global_results = global_search(claim, faiss_index, metadata)
            
            if matched_key and matched_key in entity_index :
                  entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata)
                  
                  # Interleave entity-focused and global evidence without duplicates
                  seen_texts = set()
                  combined = []
                  for r in entity_results + global_results :
                        t = r["text"].strip()
                        if t not in seen_texts :
                              seen_texts.add(t)
                              combined.append(r)
                  result = combined[:15]
                  search_type = "Hybrid (Entity + Global)"
            else :
                  result = global_results[:15]
                  search_type = "Global-search"

            retrievals.append ({
                  "Claim" : claim,
                  "Entity" : claim_entity,
                  "Search_type" : search_type,
                  "Evidence" : result
            })
      return retrievals

if __name__ == "__main__" :
      chunks = loadChunks(os.path.join("Data", "atomicChunks.json"))
      faiss_index = faiss.read_index(os.path.join("Data", "atomic.index"))
      entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
      while True :
            query = input("Enter backstory (e or E to exit) :")
            if query in "eE" :
                  print("exiting...")
                  break
            retrievals = claim_retrieval(query,chunks,faiss_index,entity_index)

            for i,ret in enumerate(retrievals,1) :
                  print(f"{i} Claim : {ret['Claim']}")
                  print(f"    Entity : {ret['Entity']}")
                  print(f"    Search : {ret['Search_type']}")
                  print(f"    Top evidence :\n")
                  for evid in ret["Evidence"] :
                        print(f"    Score : {evid['Score']:.3f}")
                        print(f"    Text: {evid['text']}")
                        print(f"    Chapter: {evid['Chapter']}")
                        print(f"    Atomic ChunkID: {evid['Atomic id']}")
                        print(f"    Book: {evid['Book']}")
                        print("-" * 60)