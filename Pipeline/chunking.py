import os
import json
import re
import unicodedata
import ftfy


Min_word = 200
Max_word =400

#basic cleaner
def cleanup(text) :
      #fixing unicode issues
      text = ftfy.fix_text(text)
      #Normalising unicode
      text = unicodedata.normalize("NFKC", text)

      #fixing structural issue
      text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
      text = re.sub(r"\n{3,}", "\n\n", text)
      text = re.sub(r"[ \t]+", " ", text) 
            
      return text.strip()

#splitting chapters
def splitChapters(book_text, min_text_length = 50) :
      pattern = r"(chapter\s+\d+|chapter\s+[ivxlcdm]+\.?)" #capturing paranthesis ensure that chapter number is also in parts
      parts = re.split(pattern, book_text, flags=re.IGNORECASE)
      
      
      chapter_title = None
      chapter = []

      for part in parts : 
            part = part.strip()
            if re.match(pattern, part, flags=re.IGNORECASE) :
                  chapter_title=part
            elif chapter_title :
                  word_count = len(part.split())
                  if (word_count>min_text_length):
                        chapter.append((chapter_title,part))
                  chapter_title=None
      
      return chapter
            
def build_chunks(paragraph, Min_word = Min_word, Max_word = Max_word) :
      chunks = []
      word_count = 0
      current = []

      for p in paragraph :
            words = p.split()
            w = len(words)

            if word_count + w <= Max_word :
                  current.append(p)
                  word_count += w
            else :
                  if word_count>=Min_word :
                        chunks.append(" ".join(current))
                        current = [p]
                        word_count = w
                  else :
                        current.append(p)
                        word_count += w
      if current :
            chunks.append(" ".join(current))

      return chunks


#fill pipeline
def full_pipeline(json_path) :
      with open(json_path, "r", encoding="utf-8") as f :
            books=json.load(f)  
      final_chunks = []
      for book in books :
            book_num = book["Book Number"]
            text = cleanup(book["Content"])

            chapters = splitChapters(text)
            print(f"Book {book_num} chapters found:", len(chapters))
            for chapter_id, (chapter_title, text) in enumerate(chapters,1) :
                  paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                  chunks = build_chunks(paragraphs)
                  for i,chunk in enumerate(chunks,1) :
                        final_chunks.append({
                              "Book" : book_num,
                              "Chapter" : chapter_title,
                              "Chapter Number" : chapter_id,
                              "Chunk id" : i,
                              "Word_count" : len(chunk.split()),
                              "text" : chunk
                              })
 
      return final_chunks

if __name__ == "__main__" :
      chunks = full_pipeline(os.path.join("Data", "text.json"))
      # print("Total chunks:", len(chunks))
      # print("\nSample chunk:\n")
      # print(chunks[0]["Chapter"])
      # print(chunks[0]["text"][:1000]) 
      with open(os.path.join("Data", "chunks.json"), "w", encoding="utf-8") as f :
            json.dump(chunks,f, ensure_ascii=False, indent=2)