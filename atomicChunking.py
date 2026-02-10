import json
import os
import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner"])
#seperating sentences
def split_sentences (text) :
      doc = nlp(text)
      return [sent.text.strip() for sent in doc.sents if len(sent.text.strip())>20]

#Resolving sentences using character memory
def resolveSent(sentences) :
      resolvedSent = []
      currentChar = []
      for sent in sentences :
            doc = nlp(sent)
            for ent in doc.ents :
                  if ent.label_ == "PERSON" :
                        currentChar = ent.text
            tokens = []

            for token in doc :
                  if (
                        token.pos_ == "PRON"
                        and token.dep_ in {"nsubj", "nsubjpass"}
                        and currentChar
                  ):
                        tokens.append(currentChar)
                  else :
                        tokens.append(token.text)
            resolvedSent.append(" ".join(tokens))
      return resolvedSent



#Checking if a sentence has some factual information
def is_factual (text) :
      doc = nlp(text)
      has_root = False
      has_subject = False

      for token in doc :
            if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"} :
                  has_root=True
            if token.dep_ in {"nsubj", "nsubjpass"} :
                  has_subject=True
      return has_root and has_subject

#Making atomic chunks
def atomic_chunking(text, min_words = 40, max_words = 100) :
      sentences = split_sentences(text)
      sentences = resolveSent(sentences)

      atomic_chunks = []
      current = []
      word_count = 0

      for s in sentences :
            w = len(s.strip())

            if w + word_count <= max_words :
                  current.append(s)
                  word_count += w
            else :
                  if word_count > min_words :
                        atomic_chunks.append(" ".join(current))
                        current=[s]
                        word_count = w
                  else :
                        current.append(s)
                        word_count += w
      
      if current :
            atomic_chunks.append(" ".join(current))

      return atomic_chunks


#Full pipeline
def atomic_pipeline (input_path, output_path):
      with open(input_path, "r", encoding="utf-8") as f :
            scenic_text = json.load(f)
      output_chunks = []
      for scene in scenic_text :
            text = scene["text"]

            atomic_chunks = atomic_chunking(text)
            atomic_chunks = [
                  a for a in atomic_chunks
                  if is_factual(a)
            ]

            for idx, atomic_text in enumerate(atomic_chunks, 1) :
                  output_chunks.append({
                        "Book" : scene["Book"],
                        "Chapter" : scene["Chapter"],
                        "Chapter Number" : scene["Chapter Number"],
                        "Parent Chunk id" : scene["Chunk id"],
                        "Atomic id" : idx,
                        "Word count" : len(atomic_text.split()),
                        "text" : atomic_text
                  })
            
      with open(output_path, "w", encoding="utf-8") as f :
            json.dump(output_chunks,f)
      
      return output_chunks

if __name__ == '__main__' :
      output_chunks = atomic_pipeline("chunks.json", "atomicChunks.json")
      print("Total atomic chunks:", len(output_chunks))
      print("\nSample atomic chunk:\n")
      print(output_chunks[0]["text"])


      
      

