import os
import requests
import re

BOOKS = [
    {
        "title": "The Hound of the Baskervilles",
        "url": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
        "filename": "The Hound of the Baskervilles.txt"
    },
    {
        "title": "Dracula",
        "url": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
        "filename": "Dracula.txt"
    }
]

os.makedirs(os.path.join("Data", "Books"), exist_ok=True)

for b in BOOKS:
    out_path = os.path.join("Data", "Books", b["filename"])
    print(f"Downloading '{b['title']}' from {b['url']}...")
    try:
        r = requests.get(b["url"], timeout=30)
        r.raise_for_status()
        text = r.text
        
        # Strip Gutenberg header and footer
        start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]
            
        end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
        if end_match:
            text = text[:end_match.start()]
            
        text = text.strip()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        word_cnt = len(text.split())
        print(f"Saved '{b['filename']}' ({word_cnt} words) to {out_path}\n")
    except Exception as e:
        print(f"Failed to download {b['title']}: {e}\n")
