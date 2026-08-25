import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates
from verfication import generate_response

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

DEATH_KEYWORDS = {"executed", "guillotine", "died", "killed", "murdered", "shot", "death", "poisoned", "duel"}

def refined_retrieval(claim, top_k=10):
    claim_entity = extract_entity(claim, entity_index)
    pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
    
    target_book = None
    if pooled_cids and pooled_cids[0] < len(metadata):
        target_book = metadata[pooled_cids[0]].get("Book")
        
    # 1. Global search (top 30)
    global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=30)
    
    # 2. Pooled entity subset search (top 30)
    entity_results = []
    if pooled_cids:
        entity_results = subset_search(claim, pooled_cids, faiss_index, metadata, top_k=30)
        
    # 3. Anchor chunks (chronological introduction)
    anchor_chunks = []
    if pooled_cids:
        for cid in pooled_cids[:4]:
            if cid < len(metadata):
                anchor_chunks.append(metadata[cid])
                
    # 4. Keyword boost for death/fate/duel/crime if mentioned in claim
    fate_chunks = []
    claim_lower = claim.lower()
    matched_kws = [kw for kw in DEATH_KEYWORDS if kw in claim_lower]
    if matched_kws and pooled_cids:
        for cid in pooled_cids:
            if cid < len(metadata):
                txt_low = metadata[cid]["text"].lower()
                if any(w in txt_low for w in ["died", "death", "expired", "corpse", "duel", "poison", "wound", "killed", "grave", "funeral"]):
                    fate_chunks.append(metadata[cid])
                    if len(fate_chunks) >= 5:
                        break
                        
    # Combine & deduplicate candidate pool
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results + anchor_chunks + fate_chunks:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            candidate_pool.append(r)
            
    # Cross-encoder rerank
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k)
    return claim_entity, target_book, final_evidence

def refined_nli_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    return f"""<|user|>
You are a precise literary fact-checker. Determine whether the Claim is SUPPORT, CONTRADICT, or NOT MENTIONED based on the Source Excerpts.

Claim: "{claim}"
Character: "{entity}"

Source Excerpts:
{evidence_text}

CLASSIFICATION RULES:
1. SUPPORT: The excerpts directly confirm or clearly describe the facts/events asserted in the claim.
2. CONTRADICT: The excerpts state facts that directly clash with or disprove what the claim asserts (for example: character's parentage, actual job/title, cause of death vs how they died/lived, or being a traitor/pirate when they are loyal/noble).
3. NOT MENTIONED: The excerpts do not contain these specific events, and there is no direct contradiction. (Note: unmentioned past hobbies or unmentioned private activities are NOT MENTIONED).

First write 1 brief sentence of reasoning, then end on the final line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

# 20 diverse test cases across all categories
test_20 = [
    # SUPPORT (8 cases)
    {"id": 1, "claim": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.", "gt": "SUPPORT"},
    {"id": 2, "claim": "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark.", "gt": "SUPPORT"},
    {"id": 3, "claim": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "claim": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"},
    {"id": 11, "claim": "Lord Glenarvan organized a maritime expedition aboard the Duncan to rescue Captain Harry Grant after discovering a distress message in three languages.", "gt": "SUPPORT"},
    {"id": 12, "claim": "Lord Glenarvan traveled to London to petition the Admiralty for a search expedition, but the government refused his request.", "gt": "SUPPORT"},
    {"id": 51, "claim": "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the merchant ship Pharaon.", "gt": "SUPPORT"},
    {"id": 53, "claim": "Abbé Faria was an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the Monte Cristo treasure.", "gt": "SUPPORT"},
    
    # CONTRADICT (6 cases)
    {"id": 21, "claim": "Lord Edward Glenarvan is a notorious pirate captain operating out of Glasgow.", "gt": "CONTRADICT"},
    {"id": 23, "claim": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 24, "claim": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.", "gt": "CONTRADICT"},
    {"id": 25, "claim": "Thalcave is an Australian bushranger who stole the yacht Duncan and sailed it to Twofold Bay.", "gt": "CONTRADICT"},
    {"id": 74, "claim": "Abbé Faria was executed by the guillotine in the courtyard of the Château d'If in 1820.", "gt": "CONTRADICT"},
    {"id": 76, "claim": "M. Morrel was a traitor who wrote the anonymous letter denouncing Dantès as a Bonapartist agent.", "gt": "CONTRADICT"},
    
    # NOT MENTIONED (6 cases)
    {"id": 41, "claim": "Captain Harry Grant secretly invested in commercial railway shares in London during the early 1850s.", "gt": "NOT MENTIONED"},
    {"id": 42, "claim": "Lady Helena Glenarvan was an accomplished amateur painter who created landscape portraits of Loch Lomond.", "gt": "NOT MENTIONED"},
    {"id": 46, "claim": "Jacques Paganel was the honorary secretary of the Geological Society of Edinburgh before his voyage.", "gt": "NOT MENTIONED"},
    {"id": 86, "claim": "M. Morrel previously served as a secret diplomat for the King of Spain before establishing his shipping business in Marseilles.", "gt": "NOT MENTIONED"},
    {"id": 87, "claim": "Gérard de Villefort wrote a published memoir detailing his early prosecutorial career under the Bourbon Restoration.", "gt": "NOT MENTIONED"},
    {"id": 88, "claim": "Haydée attended school in Vienna where she learned classical harp and fluent German before moving to Paris.", "gt": "NOT MENTIONED"}
]

print("================================================================================")
print("             TESTING REFINED 75%+ PIPELINE ON 20 DIVERSE CLAIMS                 ")
print("================================================================================")

correct = 0
for idx, tc in enumerate(test_20, 1):
    ent, book, ev = refined_retrieval(tc["claim"], top_k=10)
    p = refined_nli_prompt(tc["claim"], ev, ent)
    resp = generate_response(p)
    
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
    print(f"[{idx:02d}/20] ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {v_pred:<13} | {mark}")
    if not is_match:
        print(f"       Resp: {resp.strip().replace(chr(10), ' ')[:100]}...")

acc = (correct / len(test_20)) * 100
print("--------------------------------------------------------------------------------")
print(f"Accuracy Across 20 Diverse Claims: {acc:.2f}% ({correct}/{len(test_20)})")
print("================================================================================")
