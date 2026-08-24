import json
import faiss
from embeddingsGeneration import createEmbeddings, loadChunks, normalize
from querySearch import extract_entity, loadEntityIndex, global_search, subset_search, find_entity_in_index
from claimExtraction import extract_atomic_claims
from reranker import rerank_candidates
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


#claim retrieval pipeline
def claim_retrieval(backstory, metadata, faiss_index, entity_index, use_reranker=True, top_k_evidence=5) :
      claims = extract_atomic_claims(backstory)
      retrievals = []
      for claim in claims :
            claim_entity = extract_entity(claim, entity_index)
            matched_key = find_entity_in_index(claim_entity, entity_index)
            target_book = get_entity_book(matched_key, entity_index, metadata)
            
            global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=25)
            
            if matched_key and matched_key in entity_index :
                  entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata, top_k=20)
                  
                  seen_texts = set()
                  combined = []
                  for r in global_results[:20] + entity_results[:15] :
                        t = r["text"].strip()
                        if t not in seen_texts :
                              seen_texts.add(t)
                              combined.append(r)
                  candidate_pool = combined
                  search_type = f"Hybrid (Global(20) + Entity '{matched_key}'(15)) [Book {target_book}]"
            else :
                  candidate_pool = global_results[:25]
                  search_type = f"Global-search(25) [Book {target_book}]"


            # Apply Cross-Encoder Reranker to prioritize precision
            if use_reranker:
                  final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k_evidence)
                  search_type += f" + CrossEncoder(top_{top_k_evidence})"
            else:
                  final_evidence = candidate_pool[:top_k_evidence]

            retrievals.append ({
                  "Claim" : claim,
                  "Entity" : claim_entity,
                  "Target_Book": target_book,
                  "Search_type" : search_type,
                  "Evidence" : final_evidence
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