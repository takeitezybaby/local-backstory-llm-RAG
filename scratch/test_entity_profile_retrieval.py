import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates
from verfication import generate_response

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def profile_enhanced_retrieval(claim, top_k=8):
    claim_entity = extract_entity(claim, entity_index)
    pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
    
    target_book = None
    if pooled_cids and pooled_cids[0] < len(metadata):
        target_book = metadata[pooled_cids[0]].get("Book")
        
    # 1. Global semantic search
    global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=25)
    
    # 2. Entity subset search (dense on claim query)
    entity_results = []
    if pooled_cids:
        entity_results = subset_search(claim, pooled_cids, faiss_index, metadata, top_k=25)
        
    # 3. Core Entity Anchor Chunks (the top foundational chunks introducing this character)
    anchor_chunks = []
    if pooled_cids:
        # First 5 chronological chunks for this entity (their introduction/origin)
        for cid in pooled_cids[:5]:
            if cid < len(metadata):
                anchor_chunks.append(metadata[cid])
                
    # 4. Combine and deduplicate
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results + anchor_chunks:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            candidate_pool.append(r)
            
    # 5. Cross-Encoder rerank
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k)
    return claim_entity, target_book, final_evidence

def profile_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    return f"""<|user|>
You are an expert literary verifier. Determine if the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED by the Source Excerpts.

Claim: "{claim}"
Character: "{entity}"

Source Excerpts:
{evidence_text}

Rules:
- SUPPORT: The excerpts confirm the claim is true.
- CONTRADICT: The excerpts show the claim is false (e.g. character has a different profession, nationality, parent, fate, or actions).
- NOT MENTIONED: The excerpts have no information about the claim.

Conclude with exactly one line:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

test_suite = [
    {"id": 1, "claim": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.", "gt": "SUPPORT"},
    {"id": 2, "claim": "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark.", "gt": "SUPPORT"},
    {"id": 3, "claim": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "claim": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"},
    {"id": 21, "claim": "Lord Edward Glenarvan is a notorious pirate captain operating out of Glasgow.", "gt": "CONTRADICT"},
    {"id": 23, "claim": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 24, "claim": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.", "gt": "CONTRADICT"},
    {"id": 25, "claim": "Thalcave is an Australian bushranger who stole the yacht Duncan and sailed it to Twofold Bay.", "gt": "CONTRADICT"},
    {"id": 41, "claim": "Captain Harry Grant secretly invested in commercial railway shares in London during the early 1850s.", "gt": "NOT MENTIONED"},
    {"id": 42, "claim": "Lady Helena Glenarvan was an accomplished amateur painter who created landscape portraits of Loch Lomond.", "gt": "NOT MENTIONED"},
    {"id": 51, "claim": "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the merchant ship Pharaon.", "gt": "SUPPORT"},
    {"id": 53, "claim": "Abbé Faria was an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the Monte Cristo treasure.", "gt": "SUPPORT"},
    {"id": 74, "claim": "Abbé Faria was executed by the guillotine in the courtyard of the Château d'If in 1820.", "gt": "CONTRADICT"},
    {"id": 86, "claim": "M. Morrel previously served as a secret diplomat for the King of Spain before establishing his shipping business in Marseilles.", "gt": "NOT MENTIONED"}
]

print("================================================================================")
print("             TESTING ANCHOR-ENHANCED RETRIEVAL (14 CLAIMS)                      ")
print("================================================================================")

correct = 0
for tc in test_suite:
    ent, book, ev = profile_enhanced_retrieval(tc["claim"], top_k=8)
    p = profile_prompt(tc["claim"], ev, ent)
    resp = generate_response(p)
    
    v_pred = "NOT MENTIONED"
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    for l in reversed(lines):
        up = l.upper()
        if "SUPPORT" in up and "NOT" not in up:
            v_pred = "SUPPORT"
            break
        elif "CONTRADICT" in up:
            v_pred = "CONTRADICT"
            break
        elif "NOT MENTIONED" in up:
            v_pred = "NOT MENTIONED"
            break

    is_match = (v_pred == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {v_pred:<13} | {mark}")

acc = (correct / len(test_suite)) * 100
print("--------------------------------------------------------------------------------")
print(f"Accuracy with Anchor-Enhanced Retrieval: {acc:.2f}% ({correct}/{len(test_suite)})")
print("================================================================================")
