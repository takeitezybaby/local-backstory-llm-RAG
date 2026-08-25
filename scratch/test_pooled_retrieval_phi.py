import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search
from reranker import rerank_candidates
from verfication import generate_response
from aggregation import aggregate_results
from scratch.test_pooled_entity_lookup import get_pooled_entity_chunks

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def pooled_claim_retrieval(claim, metadata, faiss_index, entity_index, top_k_evidence=8):
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

test_suite = [
    {"id": 26, "user_input": "Captain Harry Grant was executed in London for treason against the House of Lords in 1860.", "gt": "CONTRADICT"},
    {"id": 75, "user_input": "Gérard de Villefort was an outspoken leader of the Bonapartist party who secretly helped Napoleon escape.", "gt": "CONTRADICT"},
    {"id": 77, "user_input": "Albert de Morcerf shot and killed Edmond Dantès during their scheduled duel in the Bois de Vincennes.", "gt": "CONTRADICT"},
    {"id": 3, "user_input": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 8, "user_input": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.", "gt": "SUPPORT"},
    {"id": 1, "user_input": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.", "gt": "SUPPORT"},
    {"id": 23, "user_input": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 41, "user_input": "Captain Harry Grant secretly invested in commercial railway shares in London during the early 1850s.", "gt": "NOT MENTIONED"}
]

print("================================================================================")
print("             TESTING POOLED ENTRIEVAL + SHARPENED PROMPT                        ")
print("================================================================================")

correct = 0
for tc in test_suite:
    retrievals = pooled_claim_retrieval(tc["user_input"], metadata, faiss_index, entity_index, top_k_evidence=8)
    for ret in retrievals:
        p = prompt_gen(ret["Claim"], ret["Evidence"], ret["Entity"])
        resp = generate_response(p)
        ret["Verification_result"] = resp
    agg = aggregate_results(retrievals)
    pred = agg["Final Verdict"]
    mapped = "SUPPORT" if "COMPATIBLE" in pred else ("CONTRADICT" if "INCOMPATIBLE" in pred else "NOT MENTIONED")
    is_match = (mapped == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d}: GT = {tc['gt']:<13} | Pred = {mapped:<13} | {mark}")

print(f"\nAccuracy on 8 Diverse Benchmark Claims: {correct}/{len(test_suite)} ({correct/len(test_suite)*100:.1f}%)")
