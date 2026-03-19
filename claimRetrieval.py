import json
import faiss
from embeddingsGeneration import createEmbeddings, loadChunks, normalize
from querySearch import extract_entity, loadEntityIndex, global_search, subset_search
from claimExtraction import extract_atomic_claims

k = 10

def claim_retrieval(backstory, metadata, faiss_index, entity_index) :
      claims = extract_atomic_claims(backstory)
      retrievals = []
      for claim in claims :
            search_type = None
            claim_entity = extract_entity(claim)
            if claim_entity and claim_entity in entity_index :
                  result = subset_search(claim,entity_index[claim_entity],faiss_index, metadata)
                  search_type = "Entity-restricted-seacrh"
            else :
                  result = global_search(claim,faiss_index, metadata)
                  search_type = "Global-search`"
            retrievals.append ({
                  "Claim" : claim,
                  "Entity" : claim_entity,
                  "Search_type" : search_type,
                  "Evidence" : result
            })
      return retrievals

if __name__ == "__main__" :
      chunks = loadChunks("atomicChunks.json")
      faiss_index = faiss.read_index("atomic.index")
      entity_index = loadEntityIndex("entity.json")
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