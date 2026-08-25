import sys, os, json, requests, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def get_evidence(claim, top_k=8):
    claim_entity = extract_entity(claim, entity_index)
    pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
    
    target_book = None
    if pooled_cids and pooled_cids[0] < len(metadata):
        target_book = metadata[pooled_cids[0]].get("Book")
        
    global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=25)
    entity_results = []
    if pooled_cids:
        entity_results = subset_search(claim, pooled_cids, faiss_index, metadata, top_k=25)
        
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            candidate_pool.append(r)
            
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k)
    return claim_entity, final_evidence

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3.5:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120}
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        return r.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

def parse_verdict_smart(resp):
    resp_up = resp.upper()
    # Check explicitly for contradiction / conflict
    if "CONTRADICT" in resp_up or "CONFLICT" in resp_up or "INCOMPATIBLE" in resp_up or "FALSE" in resp_up:
        return "CONTRADICT"
    # Check for support / match / confirmed
    if "SUPPORT" in resp_up or "MATCHES" in resp_up or "CONFIRMED" in resp_up or "COMPATIBLE" in resp_up:
        if "NOT SUPPORTED" not in resp_up and "DOES NOT SUPPORT" not in resp_up:
            return "SUPPORT"
    return "NOT MENTIONED"

def two_stage_verify(claim, entity, evidence_list):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    
    prompt = f"""<|user|>
You are an expert fact-checker. Compare the Claim against the Evidence.

Claim: "{claim}"
Target Entity: "{entity}"

Evidence Excerpts:
{ev_text}

Answer these two questions concisely:
1. What does the Evidence establish about "{entity}" regarding this topic?
2. Does the Claim MATCH the evidence (SUPPORT), CONFLICT with what the evidence establishes (CONTRADICT), or is it completely UNMENTIONED?

Conclude on the final line with:
Verdict: SUPPORT or CONTRADICT or NOT MENTIONED<|end|>
<|assistant|>"""

    resp = query_ollama(prompt)
    v_pred = parse_verdict_smart(resp)
    return v_pred, resp


test_cases_12 = [
    # SUPPORT
    {"id": 1, "claim": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.", "gt": "SUPPORT"},
    {"id": 3, "claim": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "claim": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"},
    {"id": 51, "claim": "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the merchant ship Pharaon.", "gt": "SUPPORT"},
    
    # CONTRADICT
    {"id": 21, "claim": "Lord Edward Glenarvan is a notorious pirate captain operating out of Glasgow.", "gt": "CONTRADICT"},
    {"id": 23, "claim": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 24, "claim": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.", "gt": "CONTRADICT"},
    {"id": 25, "claim": "Thalcave is an Australian bushranger who stole the yacht Duncan and sailed it to Twofold Bay.", "gt": "CONTRADICT"},
    
    # NOT MENTIONED
    {"id": 41, "claim": "Captain Harry Grant secretly invested in commercial railway shares in London during the early 1850s.", "gt": "NOT MENTIONED"},
    {"id": 42, "claim": "Lady Helena Glenarvan was an accomplished amateur painter who created landscape portraits of Loch Lomond.", "gt": "NOT MENTIONED"},
    {"id": 87, "claim": "Gérard de Villefort wrote a published memoir detailing his early prosecutorial career under the Bourbon Restoration.", "gt": "NOT MENTIONED"},
    {"id": 88, "claim": "Haydée attended school in Vienna where she learned classical harp and fluent German before moving to Paris.", "gt": "NOT MENTIONED"}
]

print("================================================================================")
print("             TESTING 2-STAGE QUESTION FACT-CHECKER (12 BALANCED CASES)          ")
print("================================================================================")

correct = 0
for tc in test_cases_12:
    ent, ev = get_evidence(tc["claim"], top_k=8)
    pred, raw = two_stage_verify(tc["claim"], ent, ev)
    is_match = (pred == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {pred:<13} | {mark}")

acc = (correct / len(test_cases_12)) * 100
print("--------------------------------------------------------------------------------")
print(f"2-Stage Verifier Accuracy: {acc:.1f}% ({correct}/{len(test_cases_12)})")
print("================================================================================")
