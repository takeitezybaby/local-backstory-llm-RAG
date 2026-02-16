import faiss
import json
from embeddingsGeneration import createEmbeddings, normalize, loadChunks
k =10
#searching top results
def searchIndex(query,faiss_index,metadata, top = k) :
      query_embed = createEmbeddings(query)
      query_embed = normalize(query_embed)

      scores, indicies = faiss_index.search(query_embed, top)

      results = []

      for score,idx in zip(scores[0], indicies[0]) :
            Parentdata= metadata[idx]

            results.append({
                  "Score" : float(score),
                  "text" : Parentdata["text"],
                  "Book" : Parentdata["Book"],
                  "Chapter" : Parentdata["Chapter"],
                  "Parent Chunk id" : Parentdata["Parent Chunk id"],
                  "Atomic id" : Parentdata["Atomic id"]
            })
      
      return results


if __name__ == '__main__' :
      index = faiss.read_index("atomic.index")
      atomicChunk = loadChunks("atomicChunks.json")
      while(True) :
            query =  input("Enter a backstory claim (e or E to exit) :\n")
            if query in "eE" :
                  break
            results = searchIndex(query, index,atomicChunk)
            print("\nTop matches:\n")
            for i, res in enumerate(results, 1):
                  print(f"{i}. Score: {res['Score']:.3f}")
                  print(f"   Text: {res['text']}")
                  print(f"   Chapter: {res['Chapter']}")
                  print(f"   Book: {res['Book']}")
                  print("-" * 60)