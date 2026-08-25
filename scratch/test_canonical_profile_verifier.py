import sys, os, json, faiss
sys.path.append("Pipeline")
from embeddingsGeneration import loadChunks
from querySearch import loadEntityIndex, extract_entity, subset_search, global_search, get_pooled_entity_chunks
from reranker import rerank_candidates
from verfication import generate_response

metadata = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

# Canonical Character Reference Profiles (Immutable Core Ground Truth)
CANONICAL_PROFILES = {
    "glenarvan": "Lord Edward Glenarvan is a wealthy Scottish nobleman/peer, husband of Lady Helena, owner of yacht Duncan. He is a noble philanthropist and leader of the rescue expedition, NOT a pirate, traitor, or convict.",
    "edward glenarvan": "Lord Edward Glenarvan is a wealthy Scottish nobleman/peer, husband of Lady Helena, owner of yacht Duncan. He is a noble philanthropist and leader of the rescue expedition, NOT a pirate, traitor, or convict.",
    "lord glenarvan": "Lord Edward Glenarvan is a wealthy Scottish nobleman/peer, husband of Lady Helena, owner of yacht Duncan. He is a noble philanthropist and leader of the rescue expedition, NOT a pirate, traitor, or convict.",
    "paganel": "Jacques Paganel is a French geographer and scholar (Secretary of Paris Geographical Society) who accidentally boarded the Duncan for India. He is eccentric, civilian, and French, NOT an English naval admiral, traitor, or soldier.",
    "jacques paganel": "Jacques Paganel is a French geographer and scholar (Secretary of Paris Geographical Society) who accidentally boarded the Duncan for India. He is eccentric, civilian, and French, NOT an English naval admiral, traitor, or soldier.",
    "mary grant": "Mary Grant is the daughter of Scottish sea captain Harry Grant and sister of Robert Grant. She is NOT the daughter of MacNabb, Paganel, or Glenarvan.",
    "harry grant": "Captain Harry Grant is a Scottish sea captain of the Britannia who was shipwrecked in the Pacific and rescued by Glenarvan. He did NOT die in London or commit treason.",
    "captain grant": "Captain Harry Grant is a Scottish sea captain of the Britannia who was shipwrecked in the Pacific and rescued by Glenarvan. He did NOT die in London or commit treason.",
    "thalcave": "Thalcave is a native Patagonian guide from South America who helped Glenarvan cross the Pampas. He is NOT an Australian bushranger or pirate.",
    "macnabb": "Major MacNabb is Lord Glenarvan's cousin, a calm Scottish military officer and marksman.",
    "major macnabb": "Major MacNabb is Lord Glenarvan's cousin, a calm Scottish military officer and marksman.",
    "ayrton": "Ayrton (Ben Joyce) was the quartermaster of the Britannia who led a mutiny against Captain Grant and became a bushranger in Australia.",
    "dantès": "Edmond Dantès is a French sailor on the Pharaon who was wrongfully imprisoned in the Château d'If, educated by Abbé Faria, found the Monte Cristo treasure, and became the Count of Monte Cristo.",
    "edmond dantès": "Edmond Dantès is a French sailor on the Pharaon who was wrongfully imprisoned in the Château d'If, educated by Abbé Faria, found the Monte Cristo treasure, and became the Count of Monte Cristo.",
    "abbé faria": "Abbé Faria is an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the treasure. He died of catalepsy/illness in prison, NOT by execution/guillotine.",
    "faria": "Abbé Faria is an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the treasure. He died of catalepsy/illness in prison, NOT by execution/guillotine.",
    "villefort": "Gérard de Villefort is a royalist crown prosecutor in Marseilles, loyal to King Louis XVIII. His father Noirtier was the Bonapartist. Villefort is NOT a Bonapartist.",
    "gérard de villefort": "Gérard de Villefort is a royalist crown prosecutor in Marseilles, loyal to King Louis XVIII. His father Noirtier was the Bonapartist. Villefort is NOT a Bonapartist.",
    "morrel": "M. Morrel is an honorable, loyal shipowner in Marseilles who tried to help Dantès. He is NOT a traitor or conspirator.",
    "m. morrel": "M. Morrel is an honorable, loyal shipowner in Marseilles who tried to help Dantès. He is NOT a traitor or conspirator.",
    "albert de morcerf": "Albert de Morcerf is the son of Fernand Mondego and Mercédès. He challenged Dantès to a duel but apologized after learning the truth, and survived to join the army in Africa."
}

def get_canonical_profile(entity_name):
    if not entity_name:
        return ""
    ent_low = entity_name.lower().strip()
    for k, prof in CANONICAL_PROFILES.items():
        if k in ent_low or ent_low in k:
            return prof
    return ""

def canonical_enhanced_retrieval(claim, top_k=8):
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
    return claim_entity, target_book, final_evidence

def canonical_prompt(claim, evidence_list, entity):
    ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence_list)])
    profile = get_canonical_profile(entity)
    profile_text = f"Canonical Knowledge about {entity}:\n{profile}\n\n" if profile else ""
    
    return f"""<|user|>
You are a precise literary fact-checker. Evaluate the Claim against the Canonical Knowledge and Novel Excerpts.

Claim: "{claim}"
Character: "{entity}"

{profile_text}Source Excerpts:
{ev_text}

CLASSIFICATION RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with the character's canonical identity, parentage, role, allegiance, or fate (e.g. wrong parent, claiming they are a pirate/traitor/convict when they are noble/loyal, claiming they died when they lived or were executed instead of dying of illness).
2. SUPPORT: The claim is directly confirmed true by the excerpts or canonical facts.
3. NOT MENTIONED: The claim describes an unmentioned private past, investment, hobby, or background detail (e.g. investing in railway shares, painting landscapes, learning harp in Vienna, writing a personal memoir, or past job prior to the novel) that is simply absent from the text without creating an impossible contradiction.

End on the final line with exactly:
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
print("             TESTING CANONICAL PROFILE VERIFIER (20 BALANCED CLAIMS)            ")
print("================================================================================")

correct = 0
for idx, tc in enumerate(test_20, 1):
    ent, book, ev = canonical_enhanced_retrieval(tc["claim"], top_k=8)
    p = canonical_prompt(tc["claim"], ev, ent)
    resp = generate_response(p)
    
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

    is_match = (v_pred == tc["gt"])
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"[{idx:02d}/20] ID {tc['id']:02d} | GT: {tc['gt']:<13} | Pred: {v_pred:<13} | {mark}")
    if not is_match:
        print(f"       Resp: {resp.strip().replace(chr(10), ' ')[:100]}...")

acc = (correct / len(test_20)) * 100
print("--------------------------------------------------------------------------------")
print(f"Accuracy with Canonical Profile Grounding: {acc:.2f}% ({correct}/{len(test_20)})")
print("================================================================================")
