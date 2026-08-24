import sys
import json

sys.path.append("Pipeline")
import spacy
from atomicChunking import split_sentences
from claimExtraction import compound_clauses

nlp = spacy.load("en_core_web_sm")

def clean_resolver(claims):
    resolved = []
    main_subject = None
    
    for claim in claims:
        doc = nlp(claim)
        person_ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if person_ents:
            main_subject = person_ents[0]
            
        tokens = []
        for token in doc:
            if token.text.lower() in {"he", "she", "they"} and token.dep_ in {"nsubj", "nsubjpass"} and main_subject:
                tokens.append(main_subject)
            else:
                tokens.append(token.text)
                
        resolved_claim = " ".join(tokens)
        resolved_claim = (
            resolved_claim.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" '", "'")
            .replace(" ?", "?")
            .replace(" !", "!")
        )
        resolved.append(resolved_claim.strip())
    return resolved

def extract_clean_atomic_claims(query):
    sentences = split_sentences(query)
    atomic_claims = []
    for sent in sentences:
        atomic_claims.extend(compound_clauses(sent))
    atomic_claims = clean_resolver(atomic_claims)
    return atomic_claims

with open("Data/eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for s in dataset:
    print(f"[{s['id']}] Input: {s['user_input']}")
    extracted = extract_clean_atomic_claims(s["user_input"])
    print(f"     Extracted: {extracted}")
