import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates
from verfication import generate_response

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def test_retrieval(claim):
    claim_entity = extract_entity(claim, entity_index)
    pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
    
    target_book = None
    if pooled_cids and pooled_cids[0] < len(metadata):
        target_book = metadata[pooled_cids[0]].get("Book")
        
    global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=30)
    entity_results = []
    if pooled_cids:
        entity_results = subset_search(claim, pooled_cids, faiss_index, metadata, top_k=30)
        
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            candidate_pool.append(r)
            
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=8)
    return claim_entity, target_book, final_evidence

def structured_2step_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    return f"""<|user|>
Evaluate this Claim against the provided Source Excerpts.

Claim: "{claim}"
Target Entity: "{entity}"

Source Excerpts:
{evidence_text}

Task:
1. Identify if the excerpts state facts that AGREE with the claim or CONFLICT with the claim.
2. If the excerpts confirm the character's role/actions, output SUPPORT.
3. If the excerpts state a different/conflicting character, role, parent, or fate, output CONTRADICT.
4. If the excerpts simply do not contain the claim's events, output NOT MENTIONED.

Provide your response in this exact format:
Reasoning: <1 sentence explaining agreement or conflict>
Verdict: <SUPPORT or CONTRADICT or NOT MENTIONED><|end|>
<|assistant|>"""

test_cases = [
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
print("             TESTING STRUCTURED 2-STEP PROMPT (14 TEST CASES)                   ")
print("================================================================================")

correct = 0
for tc in test_cases:
    ent, book, ev = test_retrieval(tc["claim"])
    p = structured_2step_prompt(tc["claim"], ev, ent)
    resp = generate_response(p)
    
    # Extract verdict from last line
    v_pred = "NOT MENTIONED"
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    for l in reversed(lines):
        up = l.upper()
        if "VERDICT:" in up:
            v_val = up.split("VERDICT:")[1].strip()
            if "SUPPORT" in v_val and "NOT" not in v_val:
                v_pred = "SUPPORT"
            elif "CONTRADICT" in v_val:
                v_pred = "CONTRADICT"
            elif "NOT MENTIONED" in v_val:
                v_pred = "NOT MENTIONED"
            break
        elif up in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            v_pred = up
            break

    is_match = (v_pred == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {v_pred:<13} | {mark}")
    print(f"   Resp: {resp.strip().replace(chr(10), ' ')}")

acc = (correct / len(test_cases)) * 100
print("--------------------------------------------------------------------------------")
print(f"Accuracy on 14 Test Cases: {acc:.2f}% ({correct}/{len(test_cases)})")
print("================================================================================")
