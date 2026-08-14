import spacy
import json
from atomicChunking import split_sentences

nlp = spacy.load("en_core_web_sm")


#extracting clauses (he escaped and he drowned -> he escaped, he drowned)
def compound_clauses (sentence) :
      doc = nlp(sentence)
      has_compound = False
      for token in doc :
            if token.dep_ == "cc" and token.text.lower() in {"and", "but"} :
                  right_tokens = list(doc[token.i+1:token.i+6])
                  if any(t.dep_ in {"nsubj", "nsubjpass"} for t in right_tokens) :
                        has_compound = True
                        
      if not has_compound :
            return [sentence]
            
      clauses = []
      current = []
      for token in doc :
            if token.dep_ == "cc" and token.text.lower() in {"and", "but"} and any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc[token.i+1:token.i+6]) :
                  if current :
                        clauses.append(" ".join(current).strip())
                        current = []
            else :
                  current.append(token.text)
      if current :
            clauses.append(" ".join(current).strip())
      return clauses if clauses else [sentence]


#pronoun resolver (handles orphan claims too)
def resolver(claims) :
      resolved = []
      main_subject = None
      
      for claim in claims :
            doc = nlp(claim)
            person_ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if person_ents:
                main_subject = person_ents[0]
                
            has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
            
            tokens = []
            for token in doc :
                  # Only resolve cross-clause personal pronouns (he, she, they)
                  if token.text.lower() in {"he", "she", "they"} and token.dep_ in {"nsubj", "nsubjpass"} and main_subject :
                        tokens.append(main_subject)
                  else :
                        tokens.append(token.text)
            
            resolved_claim = " ".join(tokens)
            if not has_subject and main_subject:
                has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)
                resolved_claim = f"{main_subject} is {resolved_claim}" if not has_verb else f"{main_subject} {resolved_claim}"

            resolved_claim = (
                resolved_claim.replace(" ,", ",")
                .replace(" .", ".")
                .replace(" '", "'")
                .replace(" ?", "?")
                .replace(" !", "!")
            )
            resolved.append(resolved_claim.strip())
      return resolved


#final claim verifier
def is_valid_claim(claim):
      doc = nlp(claim)
      has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)
      return len(doc) >= 3 and has_verb



#final extractor            
def extract_atomic_claims(query) :
      sentences = split_sentences(query)

      atomic_claims = []
      for sent in sentences :
            atomic_claims.extend(compound_clauses(sent))

      atomic_claims = resolver(atomic_claims)
      atomic_claims = [c.strip() for c in atomic_claims if is_valid_claim(c)]
      return atomic_claims

if __name__ == "__main__" :
      while True :
            query = input("Enter backstory (e or E to exit) :\n")
            if query in "eE" :
                  break
            claims = extract_atomic_claims(query)
            print("\nExtracted Claims:")
            for i, claim in enumerate(claims, 1):
                  print(f"{i}. {claim}")

