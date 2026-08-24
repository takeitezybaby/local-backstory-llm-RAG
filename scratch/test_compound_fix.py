import spacy
nlp = spacy.load("en_core_web_sm")

def compound_clauses(sentence):
    doc = nlp(sentence)
    has_compound = False
    for token in doc:
        if token.dep_ == "cc" and token.text.lower() in {"and", "but"}:
            right_tokens = list(doc[token.i+1:token.i+6])
            if any(t.dep_ in {"nsubj", "nsubjpass"} for t in right_tokens):
                has_compound = True
                
    if not has_compound:
        return [sentence]
        
    clauses = []
    current = []
    for token in doc:
        if token.dep_ == "cc" and token.text.lower() in {"and", "but"} and any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc[token.i+1:token.i+6]):
            if current:
                clauses.append(" ".join(current).strip())
                current = []
        else:
            current.append(token.text)
    if current:
        clauses.append(" ".join(current).strip())
    return clauses if clauses else [sentence]

test_sents = [
    "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel.",
    "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.",
    "He escaped from prison and he drowned in the sea."
]

for s in test_sents:
    print("\nInput:", s)
    print("Output:", compound_clauses(s))
