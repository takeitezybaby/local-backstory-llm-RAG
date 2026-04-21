import requests
import json
import numpy as np
import faiss
import spacy

k=10
nlp = spacy.load("en_core_web_sm")

#loading chunks
def loadChunks(path):
      with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


#creating embeddings using nomic embed model
def createEmbeddings(texts) :
      embeddings = []
      
      response = requests.post (
            "http://localhost:11434/api/embed",
                  json= {
                        "model" : "nomic-embed-text",
                        "input" : texts
                  }
            )
      response.raise_for_status()
      embedding = response.json()["embeddings"]
      embeddings.extend(embedding)
      return embeddings

#creating entity map 
def build_entity_map(chunks) :
      entity_map={}
      
      for idx, chunk in enumerate(chunks) :
            doc = nlp(chunk["text"])
            for ent in doc.ents :
                  if ent.label_ == "PERSON" :
                        name = ent.text.lower()
                        entity_map.setdefault(name, []).append(idx)
      return entity_map

#normalizing vectors for cosine similarity
def normalize(vectors) :
      norm = np.linalg.norm(vectors, axis=1,keepdims=True)
      return vectors/norm

#Indexing vectors and inner product
def vectorIndex (embeddings) :
      dim = embeddings.shape[1]
      index = faiss.IndexFlatIP(dim)
      index.add(embeddings)
      return index


if __name__ == '__main__' :
      print("Loading Chunks")
      atomicChunk = loadChunks("atomicChunks.json")
      print("Chunk loading successfull")
      texts = [metadata["text"] for metadata in atomicChunk]
      print(f"Total Chunks : {len(atomicChunk)}\n")
      print("Generating embeddings")
      embeddings = createEmbeddings(texts)
      print("Embedding generation successful\n\n")
      embeddings = np.array(embeddings, dtype="float32")
      print(f"Embedding shape : {embeddings.shape}")
      embeddings = normalize(embeddings)
      index = vectorIndex(embeddings)
      faiss.write_index(index, "atomic.index")
      print("index saved successfully.")
      print("Building entity map")
      entity_map = build_entity_map(atomicChunk)
      print("Saving entity map")
      with open("entity.json", "w", encoding="utf-8") as f :
            json.dump(entity_map, f, ensure_ascii=False)
      print("Saved successfully")