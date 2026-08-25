import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from reranker import rerank_candidates
from verfication import verify_claim, generate_response, prompt_generation
from aggregation import aggregate_results

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

# Test 5 failed SUPPORT claims from earlier
test_claims = [
    {"id": 2, "user_input": "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark.", "gt": "SUPPORT"},
    {"id": 3, "user_input": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "user_input": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"},
    {"id": 9, "user_input": "Ayrton was the former quartermaster of the Britannia who abandoned Captain Grant after an attempted mutiny.", "gt": "SUPPORT"},
    {"id": 23, "user_input": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"}
]

print("================================================================================")
print("             TESTING EXPANDED RETRIEVAL (TOP-10 RERANKED EVIDENCE)              ")
print("================================================================================")

for tc in test_claims:
    print(f"\nEvaluating ID {tc['id']}: {tc['user_input']}")
    print(f"Ground Truth: {tc['gt']}")
    
    # Retrieve top 10 reranked chunks
    retrievals = claim_retrieval(tc['user_input'], metadata, faiss_index, entity_index, use_reranker=True, top_k_evidence=10)
    
    for ret in retrievals:
        prompt = prompt_generation(ret["Claim"], ret["Evidence"], ret["Entity"])
        resp = generate_response(prompt)
        print("  -> LLM Result:", resp.strip().replace("\n", " ")[:150] + "...")
        ret["Verification_result"] = resp
        
    agg = aggregate_results(retrievals)
    print(f"  ==> Final Verdict: {agg['Final Verdict']} (GT was {tc['gt']})")
