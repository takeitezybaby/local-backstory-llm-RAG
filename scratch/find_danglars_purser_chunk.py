import json

with open("Data/atomicChunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Searching for purser and Danglars in atomicChunks.json...")
found = []
for i, c in enumerate(chunks):
    t = c["text"].lower()
    if "danglar" in t and "purser" in t:
        found.append((i, c))

print(f"Found {len(found)} chunks matching 'danglar' and 'purser':")
for idx, c in found:
    print(f"\nChunk ID: {idx} | Chapter: {c['Chapter']} | Book: {c['Book']}")
    print(c["text"])
