import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks, get_canonical_profile
from reranker import rerank_candidates
from verfication import generate_response

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset_100.json", "r", encoding="utf-8") as f:
    dataset_100 = json.load(f)

with open("benchmark/eval_dataset_long10.json", "r", encoding="utf-8") as f:
    dataset_long10 = json.load(f)

all_110 = dataset_100 + dataset_long10

def retrieve_narrative_evidence(claim_text, top_k=10):
    claim_entity = extract_entity(claim_text, entity_index)
    pooled_cids = get_pooled_entity_chunks(claim_entity, entity_index)
    
    target_book = None
    if pooled_cids and pooled_cids[0] < len(metadata):
        target_book = metadata[pooled_cids[0]].get("Book")
        
    global_results = global_search(claim_text, faiss_index, metadata, target_book=target_book, top_k=30)
    entity_results = []
    if pooled_cids:
        entity_results = subset_search(claim_text, pooled_cids, faiss_index, metadata, top_k=30)
        
    anchor_chunks = []
    if pooled_cids:
        for cid in pooled_cids[:4]:
            if cid < len(metadata):
                anchor_chunks.append(metadata[cid])
                
    seen = set()
    candidate_pool = []
    for r in global_results + entity_results + anchor_chunks:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            candidate_pool.append(r)
            
    final_evidence = rerank_candidates(claim_text, candidate_pool, top_k=top_k)
    return claim_entity, target_book, final_evidence

def verify_narrative_claim(claim_text):
    entity, book, evidence = retrieve_narrative_evidence(claim_text, top_k=10)
    profile = get_canonical_profile(entity)
    profile_section = f"Canonical Knowledge about {entity}:\n{profile}\n\n" if profile else ""
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence)])
    
    prompt = f"""<|user|>
You are an expert literary fact-checker. Evaluate whether the Backstory Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on Canonical Knowledge and Novel Excerpts.

Backstory Claim:
"{claim_text}"

Character Focus: "{entity}"

{profile_section}Source Novel Excerpts:
{ev_text}

DECISION RULES:
1. CONTRADICT: The backstory asserts false facts that directly conflict with the character's canonical identity, parentage, role, allegiance, historical actions, or fate (e.g. wrong parent, claiming they are a pirate/traitor/convict when they are noble/loyal, claiming they died when they lived or were executed instead of dying of illness).
2. SUPPORT: The central events and assertions in the backstory are confirmed true by the novel excerpts or canonical facts.
3. NOT MENTIONED: The backstory describes completely unmentioned private history, hobbies, investments, or background details that do NOT contradict canonical facts.

First state 1 brief sentence of reasoning, then conclude on the last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

    resp = generate_response(prompt)
    
    # Parse verdict cleanly
    v_pred = "NOT MENTIONED"
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    for l in reversed(lines):
        up = l.upper()
        if "VERDICT:" in up:
            val = up.split("VERDICT:")[1].strip()
            if "CONTRADICT" in val:
                v_pred = "CONTRADICT"
            elif "SUPPORT" in val and "NOT" not in val:
                v_pred = "SUPPORT"
            elif "NOT MENTIONED" in val:
                v_pred = "NOT MENTIONED"
            break
        elif up in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            v_pred = up
            break
            
    return v_pred, resp, [e["text"] for e in evidence]

# Test on 30 diverse claims across short, long narrative, and extended paragraph
test_sample = all_110[::3][:30]  # Every 3rd claim, total 30
print(f"Testing cohesive narrative verification on 30 balanced claims...")

correct = 0
results = []
for idx, s in enumerate(test_sample, 1):
    c_text = s["user_input"]
    gt = s["ground_truth_verdict"]
    ctype = s.get("claim_type", "short")
    
    pred, raw_resp, contexts = verify_narrative_claim(c_text)
    is_match = (pred == gt)
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"[{idx:02d}/30] ID {s['id']:02d} ({ctype.upper():<14}) | GT: {gt:<13} | Pred: {pred:<13} | {mark}")
    results.append({"id": s["id"], "gt": gt, "pred": pred, "match": is_match, "resp": raw_resp})

acc = (correct / len(test_sample)) * 100
print("--------------------------------------------------------------------------------")
print(f"Accuracy Across 30 Sample Claims: {acc:.2f}% ({correct}/{len(test_sample)})")
print("================================================================================")
