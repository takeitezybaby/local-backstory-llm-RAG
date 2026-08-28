import json
import os
import requests
import time
import sys

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

def query_llm(prompt, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 180
        }
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error querying LLM: {e}")
                return ""
            time.sleep(2)
    return ""

def induce_character_profile(character_name, chunk_texts, book_id):
    evidence_text = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(chunk_texts)])
    
    prompt = f"""<|user|>
You are an expert literary researcher. Read the following novel excerpts regarding the character "{character_name}".

Source Excerpts:
{evidence_text}

Task: Write a concise 2-3 sentence Canonical Invariant Profile for "{character_name}".
Include:
1. Canonical identity, nationality, and primary profession/role.
2. Key family lineage, parentage, or core relationships.
3. Fundamental allegiances and true fate/resolution in the novel.

Focus ONLY on immutable factual truths established in the excerpts. Do NOT hallucinate.

Profile for {character_name}:<|end|>
<|assistant|>"""

    profile_text = query_llm(prompt)
    # Clean output
    clean_profile = profile_text.replace("<|end|>", "").strip()
    return clean_profile

def run_induction(
    chunks_path="Data/atomicChunks.json",
    entity_path="Data/entity.json",
    output_path="Data/canonical_profiles.json",
    min_chunks=15
):
    print("Loading corpus chunks and entity index for automated profile induction...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(entity_path, "r", encoding="utf-8") as f:
        entity_index = json.load(f)

    # Filter major characters with significant presence
    major_characters = {}
    for ent_key, cids in entity_index.items():
        # Clean entity key
        k_clean = ent_key.lower().strip(" .,!?:;'\"")
        if len(k_clean) > 2 and len(cids) >= min_chunks:
            # Avoid generic titles alone
            if k_clean not in {"lord", "lady", "captain", "major", "abbé", "baron", "count", "monsieur", "the", "and"}:
                if k_clean not in major_characters or len(cids) > len(major_characters[k_clean]):
                    major_characters[k_clean] = cids

    print(f"Identified {len(major_characters)} major characters for canonical induction.\n")

    induced_profiles = {}
    
    for idx, (char_name, cids) in enumerate(major_characters.items(), 1):
        print(f"[{idx}/{len(major_characters)}] Inducing canonical profile for '{char_name}' ({len(cids)} chunks)...")
        
        # Sample anchor chunks: beginning (intro), middle (action), end (climax/resolution)
        sampled_cids = []
        if len(cids) <= 10:
            sampled_cids = cids
        else:
            # First 4 (introduction)
            sampled_cids.extend(cids[:4])
            # Middle 3 (development)
            mid = len(cids) // 2
            sampled_cids.extend(cids[mid-1:mid+2])
            # Last 3 (fate/resolution)
            sampled_cids.extend(cids[-3:])
            
        sampled_texts = []
        book_id = 1
        for cid in sampled_cids:
            if cid < len(chunks):
                sampled_texts.append(chunks[cid]["text"])
                book_id = chunks[cid].get("Book", 1)
                
        profile = induce_character_profile(char_name.title(), sampled_texts, book_id)
        print(f"   -> Result: {profile[:100]}...\n")
        
        induced_profiles[char_name] = {
            "name": char_name.title(),
            "book": book_id,
            "chunk_count": len(cids),
            "canonical_profile": profile
        }

    # Save to json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(induced_profiles, f, ensure_ascii=False, indent=2)
        
    print(f"Automated induction complete! Profiles saved to '{output_path}'.")
    return induced_profiles

if __name__ == "__main__":
    run_induction()
