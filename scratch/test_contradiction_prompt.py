import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex
from claimRetrieval import claim_retrieval
from reranker import rerank_candidates
from verfication import generate_response
from aggregation import aggregate_results

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def test_prompt_generation(claim, top_evidence, entity):
    Evidence = "\n".join([
        f"Evidence {i+1}:\n{evid['text']}" for i, evid in enumerate(top_evidence)
    ])
    prompt = f"""<|user|>
You are an expert fact-checker evaluating a backstory claim against source novel excerpts.

Claim: "{claim}"
Entity: "{entity}"

Source Excerpts:
{Evidence}

EVALUATION RULES:
1. CONTRADICT: The claim conflicts with or asserts false facts compared to the excerpts.
   - If the claim gives an entity a conflicting role, parent, spouse, origin, or fate that clashes with the excerpts (e.g. asserts someone is the daughter of X when excerpts state they are the daughter of Y; asserts someone was a pirate/merchant when excerpts state they were a passenger/fisherman; asserts someone died when they lived), output CONTRADICT.
   - Any clear factual conflict overrides unmentioned details.

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

test_contradictions = [
    {"id": 23, "user_input": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 24, "user_input": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.", "gt": "CONTRADICT"},
    {"id": 25, "user_input": "Thalcave is an Australian bushranger who stole the yacht Duncan and sailed it to Twofold Bay.", "gt": "CONTRADICT"},
    {"id": 26, "user_input": "Captain Harry Grant was executed in London for treason against the House of Lords in 1860.", "gt": "CONTRADICT"},
    {"id": 74, "user_input": "Abbé Faria was executed by the guillotine in the courtyard of the Château d'If in 1820.", "gt": "CONTRADICT"},
    {"id": 75, "user_input": "Gérard de Villefort was an outspoken leader of the Bonapartist party who secretly helped Napoleon escape.", "gt": "CONTRADICT"},
    {"id": 76, "user_input": "M. Morrel was a traitor who wrote the anonymous letter denouncing Dantès as a Bonapartist agent.", "gt": "CONTRADICT"},
    {"id": 77, "user_input": "Albert de Morcerf shot and killed Edmond Dantès during their scheduled duel in the Bois de Vincennes.", "gt": "CONTRADICT"}
]

print("================================================================================")
print("             TESTING SHARPENED CONTRADICTION PROMPT (8 CLAIMS)                  ")
print("================================================================================")

correct = 0
for tc in test_contradictions:
    retrievals = claim_retrieval(tc['user_input'], metadata, faiss_index, entity_index, use_reranker=True, top_k_evidence=8)
    for ret in retrievals:
        prompt = test_prompt_generation(ret["Claim"], ret["Evidence"], ret["Entity"])
        resp = generate_response(prompt)
        ret["Verification_result"] = resp
    
    agg = aggregate_results(retrievals)
    pred = agg["Final Verdict"]
    is_match = (pred == "INCOMPATIBLE")
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']}: GT = {tc['gt']} | Pred = {pred:<13} | {mark}")

print(f"\nContradiction Recall on Hard Test Cases: {correct}/{len(test_contradictions)} ({correct/len(test_contradictions)*100:.1f}%)")
