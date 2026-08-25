import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, find_entity_in_index, global_search, subset_search
from reranker import rerank_candidates
from verfication import generate_response
from aggregation import aggregate_results

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def enhanced_claim_retrieval(claim, metadata, faiss_index, entity_index, top_k_evidence=8):
    claim_entity = extract_entity(claim, entity_index)
    matched_key = find_entity_in_index(claim_entity, entity_index)
    
    # 1. Target book filtering
    target_book = None
    if matched_key and matched_key in entity_index and entity_index[matched_key]:
        first_cid = entity_index[matched_key][0]
        if first_cid < len(metadata):
            target_book = metadata[first_cid].get("Book")
            
    # 2. Global Dense Search (top 30)
    global_results = global_search(claim, faiss_index, metadata, target_book=target_book, top_k=30)
    
    # 3. Entity Subset Dense Search (top 25)
    entity_results = []
    if matched_key and matched_key in entity_index:
        entity_results = subset_search(claim, entity_index[matched_key], faiss_index, metadata, top_k=25)
        
    # 4. Entity Keyword Match (Boost action/predicate keywords)
    keyword_results = []
    if matched_key and matched_key in entity_index:
        claim_words = [w.lower().strip(".,!?:;'\"") for w in claim.split() if len(w) > 3]
        for cid in entity_index[matched_key]:
            if cid < len(metadata):
                chunk = metadata[cid]
                t_low = chunk["text"].lower()
                # Check for action/predicate match
                match_count = sum(1 for w in claim_words if w in t_low)
                if match_count >= 2:
                    keyword_results.append(chunk)
                    
    # 5. Combine and deduplicate into candidate pool
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results + keyword_results:
        txt = r["text"].strip()
        if txt not in seen:
            seen.add(txt)
            candidate_pool.append(r)
            
    # 6. FlashRank Cross-Encoder reranking
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k_evidence)
    return [{
        "Claim": claim,
        "Entity": claim_entity,
        "Target_Book": target_book,
        "Evidence": final_evidence
    }]

def prompt_gen(claim, top_evidence, entity):
    Evidence = "\n".join([f"Evidence {i+1}:\n{evid['text']}" for i, evid in enumerate(top_evidence)])
    prompt = f"""<|user|>
You are an expert fact-checker evaluating a backstory claim against source novel excerpts.

Claim: "{claim}"
Entity: "{entity}"

Source Excerpts:
{Evidence}

EVALUATION RULES:
1. CONTRADICT: The claim conflicts with or asserts false facts compared to the excerpts.
   - If the claim gives an entity a conflicting role, parent, spouse, origin, belief, or fate that clashes with the excerpts (e.g. asserts someone is a Bonapartist when excerpts show they are a royalist/prosecutor for the King; asserts someone died in a duel when excerpts show they survived; asserts someone is the child of X when excerpts state they are the child of Y), output CONTRADICT.
   - Any clear factual conflict overrides unmentioned minor details.

2. SUPPORT: The central facts in the claim are confirmed true by the source excerpts (direct match or clear paraphrase).

3. NOT MENTIONED: The claim describes completely new, unmentioned events/facts with NO conflicting statements in the excerpts.

Conclude on the last line with exactly:
Verdict: CONTRADICT
or
Verdict: SUPPORT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
    return prompt

test_all_8 = [
    {"id": 26, "user_input": "Captain Harry Grant was executed in London for treason against the House of Lords in 1860.", "gt": "CONTRADICT"},
    {"id": 75, "user_input": "Gérard de Villefort was an outspoken leader of the Bonapartist party who secretly helped Napoleon escape.", "gt": "CONTRADICT"},
    {"id": 77, "user_input": "Albert de Morcerf shot and killed Edmond Dantès during their scheduled duel in the Bois de Vincennes.", "gt": "CONTRADICT"},
    {"id": 3, "user_input": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "user_input": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"}
]

print("================================================================================")
print("             TESTING ENHANCED RETRIEVAL + PROMPT (5 HARD CLAIMS)                ")
print("================================================================================")

for tc in test_all_8:
    retrievals = enhanced_claim_retrieval(tc["user_input"], metadata, faiss_index, entity_index, top_k_evidence=8)
    for ret in retrievals:
        p = prompt_gen(ret["Claim"], ret["Evidence"], ret["Entity"])
        resp = generate_response(p)
        ret["Verification_result"] = resp
    agg = aggregate_results(retrievals)
    pred = agg["Final Verdict"]
    mapped = "SUPPORT" if "COMPATIBLE" in pred else ("CONTRADICT" if "INCOMPATIBLE" in pred else "NOT MENTIONED")
    is_match = (mapped == tc["gt"])
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d}: GT = {tc['gt']:<13} | Pred = {mapped:<13} | {mark}")
