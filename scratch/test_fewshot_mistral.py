import sys, os, json, faiss, requests
sys.path.append("Pipeline")

from embeddingsGeneration import loadChunks

from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

def get_evidence_for_claim(claim, top_k=8):
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

def query_ollama(prompt, model_name):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 150}
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        return r.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

test_claims = [
    {"id": 1, "claim": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.", "gt": "SUPPORT"},
    {"id": 3, "claim": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.", "gt": "SUPPORT"},
    {"id": 51, "claim": "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the merchant ship Pharaon.", "gt": "SUPPORT"},
    {"id": 53, "claim": "Abbé Faria was an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the Monte Cristo treasure.", "gt": "SUPPORT"},
    {"id": 21, "claim": "Lord Edward Glenarvan is a notorious pirate captain operating out of Glasgow.", "gt": "CONTRADICT"},
    {"id": 23, "claim": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.", "gt": "CONTRADICT"},
    {"id": 24, "claim": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.", "gt": "CONTRADICT"},
    {"id": 41, "claim": "Captain Harry Grant secretly invested in commercial railway shares in London during the early 1850s.", "gt": "NOT MENTIONED"},
    {"id": 42, "claim": "Lady Helena Glenarvan was an accomplished amateur painter who created landscape portraits of Loch Lomond.", "gt": "NOT MENTIONED"},
    {"id": 87, "claim": "Gérard de Villefort wrote a published memoir detailing his early prosecutorial career under the Bourbon Restoration.", "gt": "NOT MENTIONED"}
]


def fewshot_prompt(claim, entity, evidence_list):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    return f"""You are an expert literary fact-checker evaluating claims about characters against novel excerpts.

### EXAMPLES:
[Example 1]
Claim: "Captain Harry Grant is an English bishop from London."
Excerpts: "[1] Captain Harry Grant was a brave Scottish navigator from Dundee who commanded the Britannia."
Reasoning: The text states he is a Scottish sea captain, which directly contradicts being an English bishop.
Verdict: CONTRADICT

[Example 2]
Claim: "Captain Harry Grant enjoyed drinking chamomile tea in the evenings."
Excerpts: "[1] Captain Harry Grant was a brave Scottish navigator who commanded the Britannia."
Reasoning: The text does not mention his evening beverage preferences, and there is no contradiction.
Verdict: NOT MENTIONED

[Example 3]
Claim: "Captain Harry Grant was a Scottish navigator who commanded the vessel Britannia."
Excerpts: "[1] Captain Harry Grant was a brave Scottish navigator from Dundee who commanded the Britannia."
Reasoning: The text directly confirms his Scottish nationality and command of the Britannia.
Verdict: SUPPORT

---

### YOUR TASK:
Claim: "{claim}"
Character: "{entity}"

Source Excerpts:
{ev_text}

Provide:
Reasoning: <1 sentence>
Verdict: <SUPPORT or CONTRADICT or NOT MENTIONED>"""

def parse_verdict(resp):
    up = resp.upper()
    lines = [l.strip() for l in up.split("\n") if l.strip()]
    for l in reversed(lines):
        if "VERDICT:" in l:
            v_val = l.split("VERDICT:")[1].strip()
            if "SUPPORT" in v_val and "NOT" not in v_val:
                return "SUPPORT"
            elif "CONTRADICT" in v_val:
                return "CONTRADICT"
            elif "NOT MENTIONED" in v_val:
                return "NOT MENTIONED"
        elif l in ["SUPPORT", "CONTRADICT", "NOT MENTIONED"]:
            return l
    if "CONTRADICT" in up:
        return "CONTRADICT"
    elif "SUPPORT" in up and "NOT MENTIONED" not in up:
        return "SUPPORT"
    return "NOT MENTIONED"

print("================================================================================")
print("             TESTING FEW-SHOT IN-CONTEXT LEARNING WITH MISTRAL-7B               ")
print("================================================================================")

correct = 0
for tc in test_claims:
    ent, ev = get_evidence_for_claim(tc["claim"])
    p = fewshot_prompt(tc["claim"], ent, ev)
    resp = query_ollama(p, "mistral:7b")
    pred = parse_verdict(resp)
    is_match = (pred == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {pred:<13} | {mark}")
    if not is_match:
        print(f"   Reasoning: {resp[:100].replace(chr(10), ' ')}...")

acc = (correct / len(test_claims)) * 100
print("--------------------------------------------------------------------------------")
print(f"Few-Shot Mistral Accuracy: {acc:.1f}% ({correct}/{len(test_claims)})")
print("================================================================================")
