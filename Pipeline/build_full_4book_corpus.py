import os
import sys
import json
import time
import numpy as np
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__)))

from ingestion import read_text
from chunking import full_pipeline
from atomicChunking import atomic_pipeline
from embeddingsGeneration import loadChunks, createEmbeddings, normalize, vectorIndex, build_entity_map
from characterProfileInducer import run_induction

def main():
    print("=" * 80)
    print("      BUILDING MULTI-NOVEL 4-BOOK CORPUS INDEX FOR BACKSTORY RAG        ")
    print("=" * 80)
    
    # 1. Ingestion
    books_dir = os.path.join("Data", "Books")
    book_files = sorted([f for f in os.listdir(books_dir) if f.endswith(".txt")])
    print(f"\n[Step 1/5] Ingesting {len(book_files)} books from {books_dir}...")
    
    text_data = []
    for idx, bname in enumerate(book_files, 1):
        bpath = os.path.join(books_dir, bname)
        content = read_text(bpath)
        w_cnt = len(content.split())
        print(f"  ({idx}) Book {idx}: {bname} ({w_cnt:,} words)")
        text_data.append({"Book Number": idx, "Title": bname.replace(".txt", ""), "Content": content})
        
    text_json_path = os.path.join("Data", "text.json")
    with open(text_json_path, "w", encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False)
    print(f"Saved text data to {text_json_path}\n")
    
    # 2. Scene Chunking
    print("[Step 2/5] Building scene-level chapter chunks...")
    chunks_path = os.path.join("Data", "chunks.json")
    scene_chunks = full_pipeline(text_json_path)
    print(f"Generated {len(scene_chunks):,} scene chunks.")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(scene_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved to {chunks_path}\n")
    
    # 3. Atomic Fact Chunking
    print("[Step 3/5] Generating atomic fact-level chunks with coreference resolution...")
    atomic_path = os.path.join("Data", "atomicChunks.json")
    atomic_chunks = atomic_pipeline(chunks_path, atomic_path)
    print(f"Generated {len(atomic_chunks):,} atomic chunks.")
    print(f"Saved to {atomic_path}\n")
    
    # 4. Dense Embeddings & Entity Indexing
    print(f"[Step 4/5] Generating dense nomic embeddings & entity inverted index for {len(atomic_chunks):,} chunks...")
    texts = [m["text"] for m in atomic_chunks]
    
    t0 = time.time()
    embeddings = createEmbeddings(texts, batch_size=128)
    print(f"Generated embeddings in {time.time() - t0:.1f}s")
    
    embeddings = np.array(embeddings, dtype="float32")
    norm_embeddings = normalize(embeddings)
    
    index = vectorIndex(norm_embeddings)
    index_path = os.path.join("Data", "atomic.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    print("Extracting entity inverted index...")
    entity_map = build_entity_map(atomic_chunks)
    entity_path = os.path.join("Data", "entity.json")
    with open(entity_path, "w", encoding="utf-8") as f:
        json.dump(entity_map, f, ensure_ascii=False)
    print(f"Saved entity index ({len(entity_map):,} entities) to {entity_path}\n")
    
    # 5. Automated Canonical Persona Induction
    print("[Step 5/5] Running automated canonical character persona induction...")
    profiles_path = os.path.join("Data", "canonical_profiles.json")
    run_induction(
        chunks_path=atomic_path,
        entity_path=entity_path,
        output_path=profiles_path,
        min_chunks=15
    )
    
    print("\n" + "=" * 80)
    print("   MULTI-NOVEL CORPUS INGESTION & INDEXING COMPLETED SUCCESSFULLY!       ")
    print("=" * 80)

if __name__ == "__main__":
    main()
