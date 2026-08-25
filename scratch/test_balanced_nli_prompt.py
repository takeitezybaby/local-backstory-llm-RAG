import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates
from verfication import generate_response
from aggregation import aggregate_results

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def test_pooled_retrieval(claim, top_k=8):
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
            
    final_evidence = rerank_candidates(claim, candidate_pool, top_k=top_k)
    return claim_entity, target_book, final_evidence

def balanced_prompt(claim, evidence_list, entity):
    evidence_text = "\n".join([f"Excerpt {i+1}:\n{e['text']}" for i, e in enumerate(evidence_list)])
    return f"""<|user|>
You are a precise literary fact-checker. Determine whether the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on the Source Excerpts.

Claim: "{claim}"
Subject Entity: "{entity}"

Source Excerpts:
{evidence_text}

CLASSIFICATION RULES:
- SUPPORT: The source excerpts state or clearly confirm the facts asserted in the claim.
- CONTRADICT: The source excerpts state facts that directly conflict with or disprove what the claim asserts (for example: character's parentage, profession, loyalty, death vs survival, or actions are the opposite/different in the excerpts).
- NOT MENTIONED: The source excerpts simply do not contain information about the claim, with no contradicting statements.

First briefly state your reasoning in 1-2 sentences, then conclude on the final line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

with open("benchmark/eval_dataset_100.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 20 diverse claims
test_ids = [1, 2, 3, 8, 11, 12, 21, 23, 24, 25, 31, 34, 41, 42, 46, 51, 53, 71, 74, 86]
sample_claims = [c for c in dataset if c["id"] in test_ids]

VERDICT_MAP = {
    "COMPATIBLE": "SUPPORT",
    "PARTIALLY COMPATIBLE": "SUPPORT",
    "INCOMPATIBLE": "CONTRADICT",
    "NO CONTRADICTION, BUT NOT SUPPORTED": "NOT MENTIONED"
}

print("================================================================================")
print("             TESTING BALANCED NLI PROMPT ON 20 DIVERSE CLAIMS                   ")
print("================================================================================")

correct = 0
for idx, s in enumerate(sample_claims, 1):
    c_text = s["user_input"]
    gt = s["ground_truth_verdict"]
    ctype = s.get("claim_type", "short")
    
    ent, book, evidence = test_pooled_retrieval(c_text, top_k=8)
    p = balanced_prompt(c_text, evidence, ent)
    resp = generate_response(p)
    
    # Check verdict from single verification
    # Parse verdict cleanly
    v_pred = "NOT MENTIONED"
    lines = [l.strip().upper() for l in resp.split("\n") if l.strip()]
    for l in reversed(lines):
        if "SUPPORT" in l and "NOT" not in l:
            v_pred = "SUPPORT"
            break
        elif "CONTRADICT" in l:
            v_pred = "CONTRADICT"
            break
        elif "NOT MENTIONED" in l:
            v_pred = "NOT MENTIONED"
            break
            
    is_match = (v_pred == gt)
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"[{idx:02d}/20] ID {s['id']:02d} ({ctype.upper():<5}) | GT: {gt:<13} | Pred: {v_pred:<13} | {mark}")
    if not is_match:
        print(f"     Reasoning: {resp[:120].replace(chr(10), ' ')}...")

acc = (correct / len(sample_claims)) * 100
print("--------------------------------------------------------------------------------")
print(f"Total Correct: {correct}/{len(sample_claims)} | Accuracy: {acc:.2f}%")
print("================================================================================")
