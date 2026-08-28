import os
import sys
import json
import time

sys.path.append("Pipeline")
from querySearch import extract_entity, get_pooled_entity_chunks, get_canonical_profile, global_search, subset_search, loadEntityIndex
from embeddingsGeneration import loadChunks
from reranker import rerank_candidates
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results
from verfication import generate_response
import faiss

# Load indexed assets
chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

with open("benchmark/eval_dataset.json", "r", encoding="utf-8") as f:
    all_claims = json.load(f)

new_claims = [c for c in all_claims if c["id"] >= 111]
print(f"Testing Backstory RAG on {len(new_claims)} claims from new books: The Hound of the Baskervilles & Dracula...\n")

def verify_single_claim(claim_item):
    claim_text = claim_item["user_input"]
    sub_claims = extract_atomic_claims(claim_text)
    
    sub_verifications = []
    for sub in sub_claims:
        ent = extract_entity(sub, entity_index)
        pooled_cids = get_pooled_entity_chunks(ent, entity_index) if ent else []
        
        target_book = None
        if pooled_cids and pooled_cids[0] < len(chunks):
            target_book = chunks[pooled_cids[0]].get("Book")
            
        global_res = global_search(sub, faiss_index, chunks, target_book=target_book, top_k=25)
        entity_res = subset_search(sub, pooled_cids, faiss_index, chunks, top_k=25) if pooled_cids else []
        
        seen = set()
        cand = []
        for r in global_res + entity_res:
            txt = r["text"].strip()
            if txt not in seen:
                seen.add(txt)
                cand.append(r)
                
        evidence = rerank_candidates(sub, cand, top_k=8)
        
        prof = get_canonical_profile(ent) if ent else ""
        prof_section = f"Canonical Knowledge about {ent}:\n{prof}\n\n" if prof else ""
        ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence)])
        
        prompt = f"""<|user|>
You are a precise literary fact-checker. Evaluate the Claim against the Canonical Knowledge and Novel Excerpts.

Claim: "{sub}"
Character: "{ent}"

{prof_section}Source Excerpts:
{ev_text}

CLASSIFICATION RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with the character's canonical identity, parentage, role, allegiance, or fate (e.g. wrong parent, claiming they are a pirate/traitor/convict when they are noble/loyal, claiming they died when they lived or were executed instead of dying of illness).
2. SUPPORT: The claim is directly confirmed true by the excerpts or canonical facts.
3. NOT MENTIONED: The claim describes an unmentioned private past, investment, hobby, or background detail that is simply absent from the text without creating an impossible contradiction.

End on the final line with exactly:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

        resp = generate_response(prompt)
        sub_verifications.append({"Claim": sub, "Evidence": evidence, "Verification_result": resp})
        
    agg = aggregate_results(sub_verifications)
    pred_raw = agg["Final Verdict"]
    mapped = "SUPPORT" if "COMPATIBLE" in pred_raw else ("CONTRADICT" if "INCOMPATIBLE" in pred_raw else "NOT MENTIONED")
    return mapped, agg

cor = 0
book_stats = {}
type_stats = {}

for idx, c in enumerate(new_claims, 1):
    gt = c["ground_truth_verdict"]
    bname = c["book"]
    ctype = c.get("claim_type", "short")
    t0 = time.time()
    
    pred, agg_res = verify_single_claim(c)
    elapsed = time.time() - t0
    is_match = (pred == gt)
    if is_match:
        cor += 1
        
    book_stats.setdefault(bname, {"total": 0, "correct": 0})
    book_stats[bname]["total"] += 1
    if is_match:
        book_stats[bname]["correct"] += 1
        
    type_stats.setdefault(ctype, {"total": 0, "correct": 0})
    type_stats[ctype]["total"] += 1
    if is_match:
        type_stats[ctype]["correct"] += 1
        
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"[{idx:02d}/{len(new_claims)}] ID {c['id']:03d} ({bname[:12]}) ({ctype[:5].upper()}) | GT: {gt:<13} | Pred: {pred:<13} | {mark} ({elapsed:.1f}s)")

acc = cor / len(new_claims) * 100
print("\n" + "=" * 80)
print(f"Accuracy Across {len(new_claims)} New Novel Claims: {acc:.2f}% ({cor}/{len(new_claims)})")
print("-" * 80)
print("By Book:")
for b, s in book_stats.items():
    b_acc = s["correct"] / s["total"] * 100
    print(f"  - {b}: {b_acc:.2f}% ({s['correct']}/{s['total']})")
print("\nBy Granularity:")
for ct, s in type_stats.items():
    ct_acc = s["correct"] / s["total"] * 100
    print(f"  - {ct}: {ct_acc:.2f}% ({s['correct']}/{s['total']})")
print("=" * 80)
