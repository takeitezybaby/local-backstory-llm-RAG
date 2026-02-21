import spacy
import json
from atomicChunking import split_sentences

nlp = spacy.load("en_core_web_sm")


#extracting clauses (he escaped and drowned -> he escaped, drowned)
def compound_clauses (sentence) :
      doc = nlp(sentence)
      verbs = [ token for token in doc if token.pos_ in {"VERB", "AUX"} and token.dep_ == "ROOT" or token.dep_ == "conj"]
      if len(verbs)<2:
            return sentence
      
      clauses = []
      current = []
      for token in doc :
            if token.dep_ == "cc" and token.text.lower() in {"and", "but"} :
                  clauses.append(" ".join([text for text in current]).strip())
                  current = []
            else :
                  current.append(token.text)

      if current :
            clauses.append(" ".join([text for text in current]).strip())

      return clauses


#pronoun resolver (handles orphan claims too)
def resolver(claims) :
      resolved = []
      current_entity = None
      
      for claim in claims :
            doc = nlp(claim)
            has_subject = False
            tokens = []

            for token in doc :
                  if token.dep_ in {"nsubj", "nsubjpass"} :
                        has_subject = True
            for ent in doc.ents :
                  if ent.label_ == "PERSON" :
                        current_entity = ent.text
            for token in doc :
                  if token.pos_ == "PRON" and token.dep_ in {"nsubj", "nsubjpass"} and current_entity :
                        tokens.append(current_entity)
                  else :
                        tokens.append(token.text)
            resolved_claim = " ".join(tokens)

            if not has_subject and current_entity :
                  resolved_claim = current_entity + " " + resolved_claim

            resolved.append(resolved_claim.strip())
      return resolved



#final claim verifier
def is_valid_claim(claim):
      doc = nlp(claim)
      has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
      has_verb = any(token.pos_ in {"VERB", "AUX"} and token.dep_ in {"ROOT", 'conj'} for token in doc)
      return has_subject and has_verb



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

