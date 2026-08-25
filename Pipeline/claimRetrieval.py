import json
import faiss
from embeddingsGeneration import createEmbeddings, loadChunks, normalize
from querySearch import extract_entity, loadEntityIndex, global_search, subset_search, find_entity_in_index, get_pooled_entity_chunks
from claimExtraction import extract_atomic_claims
from reranker import rerank_candidates
import os

#claim retrieval pipeline
def claim_retrieval(backstory, metadata, faiss_index, entity_index, use_reranker=True, top_k_evidence=8) :
      claims = extract_atomic_claims(backstory)
      retrievals = []
      for claim in claims :
            claim_entity = extract_entity(claim, entity_index)
            pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
            
            target_book = None
            if pooled_cids and pooled_cids[0] < len(metadata):
                  target_book = metadata[pooled_cids[0]].get("Book")
            
            global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=30)
            
            if pooled_cids:
                  entity_results = subset_search(claim, pooled_cids, faiss_index, metadata, top_k=30)
                  
                  seen_texts = set()
                  combined = []
                  for r in global_results[:25] + entity_results[:25] :
                        t = r["text"].strip()
                        if t not in seen_texts :
                              seen_texts.add(t)
                              combined.append(r)
                  candidate_pool = combined
                  search_type = f"Hybrid (Global(25) + Pooled Entity '{claim_entity}'({len(pooled_cids)} chunks)) [Book {target_book}]"
            else :
                  candidate_pool = global_results[:30]
                  search_type = f"Global-search(30) [Book {target_book}]"

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