import json
import requests 
import os 

def read_text(file_path) :
      with open(file_path, "r", encoding="utf-8") as f :
            text = f.read()
      return text

if __name__ == '__main__' :
      Books = os.listdir("Books")
      text = [] 
      for i,book in enumerate(Books) :
            path = os.path.join("Books", book)
            Book_Number = i+1
            text.append({"Book Number" : Book_Number, "Content" : (read_text(path))})
      with open("text.json", "w") as f :
            json.dump(text,f)