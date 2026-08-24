import requests
import json
import sys
import faiss

sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from verfication import prompt_generation, generate_response
from aggregation import extract_single_verdict

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

claim_test = "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and the owner of the steam yacht Duncan."
print("Testing Claim:", claim_test)
retrievals = claim_retrieval(claim_test, chunks, faiss_index, entity_index)
for ret in retrievals:
    print("\n--- Atomic Claim:", ret["Claim"])
    print("Entity:", ret["Entity"])
    print("Evidence Count:", len(ret["Evidence"]))
    prompt = prompt_generation(ret["Claim"], ret["Evidence"][:5], ret["Entity"])
    raw_resp = generate_response(prompt)
    print("\n--- RAW LLM RESPONSE ---")
    print(raw_resp)
    print("\n--- EXTRACTED VERDICT ---")
    print(extract_single_verdict(raw_resp))
