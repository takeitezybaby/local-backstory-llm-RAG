import json
import os
import sys
import time

sys.path.append("Pipeline")
from querySearch import extract_entity, get_pooled_entity_chunks, get_canonical_profile, global_search, subset_search, loadEntityIndex
from embeddingsGeneration import loadChunks
from reranker import rerank_candidates
from claimExtraction import extract_atomic_claims
from aggregation import aggregate_results
from verfication import generate_response
import faiss

ADVERSARIAL_CLAIMS = [
    # --- Book 1: In Search of the Castaways ---
    {
        "id": 1,
        "book": "In_Search_of_the_Castaways",
        "category": "Entity_Role_Conflation",
        "claim": "Major MacNabb was the absent-minded French secretary of the Paris Geographical Society who accidentally boarded the Duncan for Calcutta.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates Scottish Major MacNabb with French geographer Jacques Paganel."
    },
    {
        "id": 2,
        "book": "In_Search_of_the_Castaways",
        "category": "Temporal_Transposition",
        "claim": "Lord Glenarvan discovered the message in the shark's belly in 1875 after returning from his completed Antarctic expedition.",
        "ground_truth": "CONTRADICT",
        "explanation": "Wrong date and timing; the message was found in 1864 before any expedition."
    },
    {
        "id": 3,
        "book": "In_Search_of_the_Castaways",
        "category": "Admissible_Open_World",
        "claim": "John Mangles owned a silver pocket barometer engraved with his family crest from Dundee.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private item absent from text."
    },
    {
        "id": 4,
        "book": "In_Search_of_the_Castaways",
        "category": "Near_Miss_Alias",
        "claim": "Robert Glenarvan, the young boy on the Duncan, was the son of Captain Harry Grant.",
        "ground_truth": "CONTRADICT",
        "explanation": "Near-miss name confusion: Robert Grant was Harry Grant's son, not 'Robert Glenarvan'."
    },
    {
        "id": 5,
        "book": "In_Search_of_the_Castaways",
        "category": "Admissible_Open_World",
        "claim": "Thalcave practiced medicinal root carving during winter months in the northern foothills of the Andes.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible cultural craft absent from the novel."
    },

    # --- Book 2: The Count of Monte Cristo ---
    {
        "id": 6,
        "book": "The_Count_of_Monte_Cristo",
        "category": "Entity_Role_Conflation",
        "claim": "Fernand Mondego was the ambitious royalist crown prosecutor in Marseilles who burned the Bonapartist letter.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates Catalan fisherman Fernand with royalist prosecutor Villefort."
    },
    {
        "id": 7,
        "book": "The_Count_of_Monte_Cristo",
        "category": "Temporal_Transposition",
        "claim": "Edmond Dantès purchased the island of Monte Cristo before his arrest at the Reserve tavern in 1815.",
        "ground_truth": "CONTRADICT",
        "explanation": "Temporal impossibility: Dantès learned of the treasure years later from Faria in prison."
    },
    {
        "id": 8,
        "book": "The_Count_of_Monte_Cristo",
        "category": "Admissible_Open_World",
        "claim": "Abbé Faria studied classical Greek architecture during his early clerical novitiate in Rome.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible intellectual study absent from the text."
    },
    {
        "id": 9,
        "book": "The_Count_of_Monte_Cristo",
        "category": "Near_Miss_Alias",
        "claim": "Noirtier de Morcerf was the Bonapartist president of the Club of the Rue Saint-Jacques in Paris.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates Noirtier de Villefort with the surname Morcerf."
    },
    {
        "id": 10,
        "book": "The_Count_of_Monte_Cristo",
        "category": "Admissible_Open_World",
        "claim": "Maximilian Morrel purchased an English saddle from a saddler in Lyons before receiving his lieutenant commission.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private purchase absent from text."
    },

    # --- Book 3: The Hound of the Baskervilles ---
    {
        "id": 11,
        "book": "The_Hound_of_the_Baskervilles",
        "category": "Entity_Role_Conflation",
        "claim": "Dr. John Watson lived at Merripit House as an entomologist and trained a phosphorus-coated bloodhound on Dartmoor.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates Dr. Watson with Jack Stapleton."
    },
    {
        "id": 12,
        "book": "The_Hound_of_the_Baskervilles",
        "category": "Temporal_Transposition",
        "claim": "Sir Charles Baskerville was killed by the hound in the yew alley after Sherlock Holmes shot Jack Stapleton in London.",
        "ground_truth": "CONTRADICT",
        "explanation": "Chronological reversal: Sir Charles died months before Holmes began the investigation."
    },
    {
        "id": 13,
        "book": "The_Hound_of_the_Baskervilles",
        "category": "Near_Miss_Alias",
        "claim": "Dr. Arthur Mortimer was the country medical practitioner who brought the 1742 manuscript to Baker Street.",
        "ground_truth": "CONTRADICT",
        "explanation": "Name distortion: Dr. James Mortimer, not 'Arthur Mortimer'."
    },
    {
        "id": 14,
        "book": "The_Hound_of_the_Baskervilles",
        "category": "Admissible_Open_World",
        "claim": "Sir Henry Baskerville subscribed to a Toronto agricultural journal while managing his farm in Canada.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private subscription absent from text."
    },
    {
        "id": 15,
        "book": "The_Hound_of_the_Baskervilles",
        "category": "Admissible_Open_World",
        "claim": "Inspector Lestrade bought a silver pocket watch chain from a jeweler on Fleet Street in 1887.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private purchase absent from text."
    },

    # --- Book 4: Dracula ---
    {
        "id": 16,
        "book": "Dracula",
        "category": "Entity_Role_Conflation",
        "claim": "Dr. John Seward was the Dutch specialist in obscure diseases who placed communion wafers in Lucy's tomb.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates English asylum doctor Seward with Dutch professor Van Helsing."
    },
    {
        "id": 17,
        "book": "Dracula",
        "category": "Temporal_Transposition",
        "claim": "Jonathan Harker slit Count Dracula's throat in Transylvania before traveling to Munich for his real estate clerkship.",
        "ground_truth": "CONTRADICT",
        "explanation": "Temporal reversal: the final battle occurs at the end of the novel."
    },
    {
        "id": 18,
        "book": "Dracula",
        "category": "Near_Miss_Alias",
        "claim": "Arthur Murray was the young nobleman who proposed marriage to Lucy Westenra.",
        "ground_truth": "CONTRADICT",
        "explanation": "Conflates Arthur Holmwood with Mina Murray's surname."
    },
    {
        "id": 19,
        "book": "Dracula",
        "category": "Admissible_Open_World",
        "claim": "Quincey Morris owned a collection of pre-Columbian pottery shards from an expedition to New Mexico in 1889.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private collection absent from text."
    },
    {
        "id": 20,
        "book": "Dracula",
        "category": "Admissible_Open_World",
        "claim": "Mrs. Westenra kept an English translation of Dante's Inferno on her nightstand in Whitby.",
        "ground_truth": "NOT MENTIONED",
        "explanation": "Plausible private book absent from text."
    }
]

with open("benchmark/adversarial_near_miss.json", "w", encoding="utf-8") as f:
    json.dump(ADVERSARIAL_CLAIMS, f, ensure_ascii=False, indent=2)

print("Constructed 20 Adversarial Near-Miss Claims across 4 distinct error taxonomies!")
print("Saved to 'benchmark/adversarial_near_miss.json'.\n")

print("=" * 80)
print("       EVALUATING BACKSTORY RAG ON ADVERSARIAL NEAR-MISS BENCHMARK           ")
print("=" * 80)

chunks = loadChunks("Data/atomicChunks.json")
entity_index = loadEntityIndex("Data/entity.json")
faiss_index = faiss.read_index("Data/atomic.index")

cor = 0
category_stats = {}

for idx, item in enumerate(ADVERSARIAL_CLAIMS, 1):
    claim_text = item["claim"]
    gt = item["ground_truth"]
    cat = item["category"]
    bname = item["book"]
    
    sub_claims = extract_atomic_claims(claim_text)
    sub_verifs = []
    for sub in sub_claims:
        ent = extract_entity(sub, entity_index)
        pooled_cids = get_pooled_entity_chunks(ent, entity_index) if ent else []
        target_book = chunks[pooled_cids[0]].get("Book") if (pooled_cids and pooled_cids[0] < len(chunks)) else None
        
        g_res = global_search(sub, faiss_index, chunks, target_book=target_book, top_k=25)
        e_res = subset_search(sub, pooled_cids, faiss_index, chunks, top_k=25) if pooled_cids else []
        
        seen = set()
        cand = []
        for r in g_res + e_res:
            txt = r["text"].strip()
            if txt not in seen:
                seen.add(txt)
                cand.append(r)
                
        evidence = rerank_candidates(sub, cand, top_k=8)
        prof = get_canonical_profile(ent) if ent else ""
        prof_sec = f"Canonical Knowledge about {ent}:\n{prof}\n\n" if prof else ""
        ev_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence)])
        
        prompt = f"""<|user|>
Evaluate whether the Claim is SUPPORTED, CONTRADICTED, or NOT MENTIONED based on Canonical Knowledge and Novel Excerpts.

Claim: "{sub}"
Character: "{ent}"

{prof_sec}Source Excerpts:
{ev_text}

RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with canonical facts, identity, parentage, role, or fate in the text.
2. SUPPORT: The claim is confirmed true by the excerpts.
3. NOT MENTIONED: The claim describes unmentioned private history, hobbies, or details absent from the text without direct contradiction.

End on the last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

        resp = generate_response(prompt)
        sub_verifs.append({"Claim": sub, "Evidence": evidence, "Verification_result": resp})
        
    agg = aggregate_results(sub_verifs)
    pred_raw = agg["Final Verdict"]
    pred = "CONTRADICT" if "INCOMPATIBLE" in pred_raw else ("SUPPORT" if "COMPATIBLE" in pred_raw else "NOT MENTIONED")
    
    is_match = (pred == gt)
    if is_match:
        cor += 1
        
    category_stats.setdefault(cat, {"total": 0, "correct": 0})
    category_stats[cat]["total"] += 1
    if is_match:
        category_stats[cat]["correct"] += 1
        
    mark = "[PASS]" if is_match else "[FAIL]"
    print(f"[{idx:02d}/20] ({cat:<25}) | GT: {gt:<13} | Pred: {pred:<13} | {mark}")

acc = cor / len(ADVERSARIAL_CLAIMS) * 100
print("\n" + "=" * 80)
print(f"Adversarial Near-Miss Accuracy: {acc:.2f}% ({cor}/{len(ADVERSARIAL_CLAIMS)})")
print("-" * 80)
print("By Adversarial Error Category:")
for c, s in category_stats.items():
    print(f"  - {c:<25}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
print("=" * 80)

