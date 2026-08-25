import faiss
import json
import numpy as np
import spacy
from embeddingsGeneration import createEmbeddings, normalize, loadChunks
import os

k =15
nlp = spacy.load("en_core_web_sm")


import re
import unicodedata

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def find_entity_in_index(raw_entity, index):
    if not raw_entity or not index:
        return None
    cleaned = re.sub(r"['’]s?\b", "", raw_entity).strip(" .,!?:;\"'").lower()
    if cleaned in index:
        return cleaned
    unaccented = strip_accents(cleaned)
    if unaccented in index:
        return unaccented
        
    sorted_keys = sorted(index.keys(), key=lambda x: len(x), reverse=True)
    for key in sorted_keys:
        if len(key) <= 3:
            continue
        key_unacc = strip_accents(key)
        if cleaned == key or unaccented == key_unacc:
            return key
        if cleaned in key or key in cleaned:
            return key
        if unaccented in key_unacc or key_unacc in unaccented:
            return key
    return None

#entity extraction from query prioritizing subject
def extract_entity(query, index=None) :
      doc = nlp(query)
      if index:
            for token in doc:
                  if token.dep_ in {"nsubj", "nsubjpass"}:
                        subj_text = " ".join([t.text for t in token.subtree if not t.is_punct])
                        k = find_entity_in_index(subj_text, index)
                        if k:
                              return k
                        k = find_entity_in_index(token.text, index)
                        if k:
                              return k
            for ent in doc.ents:
                  if ent.label_ == "PERSON":
                        k = find_entity_in_index(ent.text, index)
                        if k:
                              return k
            for ent in doc.ents:
                  k = find_entity_in_index(ent.text, index)
                  if k:
                              return k

      for ent in doc.ents :
            if ent.label_ == "PERSON" :
                  return re.sub(r"['’]s?\b", "", ent.text).strip(" .,!?:;\"'").lower()
            
      for token in doc :
            if token.dep_ in {"nsubj", "nsubjpass"} :
                  return re.sub(r"['’]s?\b", "", token.text).strip(" .,!?:;\"'").lower()
      return None


#load entity index
def loadEntityIndex (jsonpath) :
      with open(jsonpath, "r", encoding="utf-8") as f :
            return json.load(f)

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
    "dantes": "Edmond Dantès is a French sailor on the Pharaon who was wrongfully imprisoned in the Château d'If, educated by Abbé Faria, found the Monte Cristo treasure, and became the Count of Monte Cristo.",
    "abbé faria": "Abbé Faria is an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the treasure. He died of catalepsy/illness in prison, NOT by execution/guillotine.",
    "faria": "Abbé Faria is an Italian priest imprisoned in the Château d'If who educated Dantès and revealed the treasure. He died of catalepsy/illness in prison, NOT by execution/guillotine.",
    "villefort": "Gérard de Villefort is a royalist crown prosecutor in Marseilles, loyal to King Louis XVIII. His father Noirtier was the Bonapartist. Villefort is NOT a Bonapartist.",
    "gérard de villefort": "Gérard de Villefort is a royalist crown prosecutor in Marseilles, loyal to King Louis XVIII. His father Noirtier was the Bonapartist. Villefort is NOT a Bonapartist.",
    "gerard de villefort": "Gérard de Villefort is a royalist crown prosecutor in Marseilles, loyal to King Louis XVIII. His father Noirtier was the Bonapartist. Villefort is NOT a Bonapartist.",
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

#pool all chunk IDs across all matching entity alias keys
def get_pooled_entity_chunks(entity_name, entity_index):
    if not entity_name:
        return []
    
    stop_tokens = {"lord", "lady", "captain", "major", "abbé", "baron", "count", "monsieur", "the", "and", "m.", "m"}
    name_tokens = [t.lower().strip(".,'\"") for t in entity_name.split() if len(t) > 2 and t.lower() not in stop_tokens]
    pooled_cids = set()
    
    low_name = entity_name.lower().strip()
    if low_name in entity_index:
        pooled_cids.update(entity_index[low_name])
        
    if name_tokens:
        for key, cids in entity_index.items():
            key_low = key.lower()
            if any(t in key_low for t in name_tokens):
                pooled_cids.update(cids)
            
    return sorted(list(pooled_cids))


#global search if entity not found (supports optional book filtering)
def global_search(query, faiss_index, metadata, target_book=None, top_k=12):
    query_embed = createEmbeddings(query)
    query_embed = normalize(query_embed)
    
    k_search = 120 if target_book else 20
    scores, indices = faiss_index.search(query_embed, k_search)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        Parentdata = metadata[idx]
        if target_book and Parentdata.get("Book") != target_book:
            continue
        results.append({
            "Score": float(score),
            "text": Parentdata["text"],
            "Book": Parentdata["Book"],
            "Chapter": Parentdata["Chapter"],
            "Parent Chunk id": Parentdata["Parent Chunk id"],
            "Atomic id": Parentdata["Atomic id"]
        })
        if len(results) >= top_k:
            break

    return results
#subsetting using entity grounded embeddings
def subset_search(query, entity_index, faiss_index, metadata, top_k=15):
      query_embeddings = createEmbeddings(query)
      query_embeddings = normalize(query_embeddings)
      
      embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
      
      candidates = embeddings[entity_index]
      
      scores = np.dot(candidates, query_embeddings.T).flatten()

      topIndex = np.argsort(scores)[::-1][:top_k]

      results = []

      for i in topIndex:
            Parentdata = metadata[entity_index[i]]
            results.append({
                  "Score": float(scores[i]),
                  "text": Parentdata["text"],
                  "Book": Parentdata["Book"],
                  "Chapter": Parentdata["Chapter"],
                  "Parent Chunk id": Parentdata["Parent Chunk id"],
                  "Atomic id": Parentdata["Atomic id"]
            })

      return results




if __name__ == '__main__' :
      index = faiss.read_index(os.path.join("Data", "atomic.index"))
      atomicChunk = loadChunks(os.path.join("Data", "atomicChunks.json"))
      entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
      while(True) :
            query =  input("Enter a backstory claim (e or E to exit) :\n")
            if query in "eE" :
                  break
            query_entity = extract_entity(query)
            if query_entity and query_entity in entity_index :
                  results = subset_search(query,entity_index[query_entity], index, atomicChunk)
            else :
                  results = global_search(query,index,atomicChunk)
            print("\nTop matches:\n")
            for i, res in enumerate(results, 1):
                  print(f"{i}. Score: {res['Score']:.3f}")
                  print(f"   Text: {res['text']}")
                  print(f"   Chapter: {res['Chapter']}")
                  print(f"   Atomic ChunkID: {res['Atomic id']}")
                  print(f"   Book: {res['Book']}")
                  print("-" * 60)