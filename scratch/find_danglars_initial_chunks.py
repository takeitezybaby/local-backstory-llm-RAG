import json

with open("Data/atomicChunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Searching for Danglars in Chapter 1 & 2 of Monte Cristo...")
for i, c in enumerate(chunks):
    if c["Book"] == "Count_of_Monte_Cristo" and int(c.get("Chapter", 999)) <= 3:
        if "danglar" in c["text"].lower():
            print(f"\nChunk ID: {i} | Chapter {c['Chapter']}")
            print(c["text"][:300])
