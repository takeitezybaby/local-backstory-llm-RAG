import os
import sys
from flashrank import Ranker, RerankRequest

# Initialize lightweight local cross-encoder ranker (singleton)
_RANKER_INSTANCE = None

def get_ranker(model_name="ms-marco-MiniLM-L-12-v2"):
    global _RANKER_INSTANCE
    if _RANKER_INSTANCE is None:
        try:
            _RANKER_INSTANCE = Ranker(model_name=model_name)
        except Exception:
            # Fallback to ultra-lightweight TinyBERT if needed
            _RANKER_INSTANCE = Ranker()
    return _RANKER_INSTANCE

def rerank_candidates(query, candidate_chunks, top_k=5):
    """
    Rerank retrieved candidate chunks using a local Cross-Encoder (FlashRank).
    
    Args:
        query (str): The atomic claim or search query text.
        candidate_chunks (list): List of dicts representing candidate chunks with 'text' field.
        top_k (int): Number of top reranked chunks to return.
        
    Returns:
        list: Top-k reranked candidate chunk dicts with an added 'rerank_score' field.
    """
    if not candidate_chunks:
        return []
        
    if len(candidate_chunks) <= top_k:
        return candidate_chunks

    ranker = get_ranker()
    
    # Prepare passages for FlashRank
    passages = []
    for idx, chunk in enumerate(candidate_chunks):
        passages.append({
            "id": idx,
            "text": chunk.get("text", "").strip(),
            "meta": chunk
        })
        
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)
    
    top_results = []
    for item in reranked_results[:top_k]:
        original_chunk = item["meta"].copy()
        original_chunk["rerank_score"] = float(item["score"])
        top_results.append(original_chunk)
        
    return top_results

if __name__ == "__main__":
    test_query = "Lord Glenarvan is a Scottish peer and owner of the yacht Duncan."
    test_passages = [
        {"id": 1, "text": "The yacht was called the Duncan, and belonged to Lord Glenarvan, one of the sixteen Scottish peers."},
        {"id": 2, "text": "Mercédès was a Catalan girl living in the village of Catalans near Marseilles."},
        {"id": 3, "text": "Lord Glenarvan was married to Lady Helena, daughter of traveler William Duff."},
        {"id": 4, "text": "The weather was stormy in the Pacific Ocean."}
    ]
    
    print("Testing Cross-Encoder Reranker...")
    reranked = rerank_candidates(test_query, test_passages, top_k=2)
    for i, r in enumerate(reranked, 1):
        print(f"Top {i} (Score: {r['rerank_score']:.4f}): {r['text']}")
