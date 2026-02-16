import requests
import json
import numpy as np
import faiss

k=10

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

