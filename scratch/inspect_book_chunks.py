import json

with open("Data/atomicChunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))
for i in range(15):
    c = chunks[i]
    print(f"\nChunk {i} | Book {c.get('Book')} | Chapter {c.get('Chapter')}:")
    print(c["text"][:200])
