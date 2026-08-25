import sys, os, json, spacy
sys.path.append("Pipeline")
from querySearch import loadEntityIndex



nlp = spacy.load("en_core_web_sm")
entity_index = loadEntityIndex("Data/entity.json")

with open("benchmark/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def extract_main_character(text, entity_index):
    # 1. Check known entity names
    t_low = text.lower()
    for ent_name in sorted(entity_index.keys(), key=lambda x: len(x), reverse=True):
        if len(ent_name) > 3 and ent_name in t_low:
            return ent_name.title()
            
    # 2. Check spaCy PERSON entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

missed_spacy = 0
found_enhanced = 0

for s in dataset:
    doc = nlp(s["user_input"])
    spacy_ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    enhanced_ent = extract_main_character(s["user_input"], entity_index)
    
    if not spacy_ents:
        missed_spacy += 1
        print(f"ID {s['id']:02d}: spaCy missed entity! Enhanced found: '{enhanced_ent}' | Claim: {s['user_input'][:70]}...")

print("-" * 80)
print(f"Total claims where spaCy NER failed: {missed_spacy}/110")
