import json
import requests
import faiss
import os
from claimRetrieval import claim_retrieval, loadEntityIndex, loadChunks

API = "http://localhost:11434/api/generate"
Tokens = 2048


import time

#function to get response from llm
def generate_response (prompt, max_retries=3) :
      for attempt in range(max_retries) :
            try :
                  response = requests.post(API, json={
                        "model" : "phi3.5:latest",
                        "prompt" : prompt,
                        "stream" : False,
                        "options" : {
                               "num_ctx" : Tokens,
                               "temperature" : 0.0
                        }
                  }, timeout=90)
                  response.raise_for_status()
                  return response.json()["response"]
            except Exception as e :
                  if attempt == max_retries - 1 :
                        raise e
                  time.sleep(2)
      return ""


#Generating prompt
def prompt_generation (claim, evidence_list, entity) :
      top_evidence = evidence_list[:12]
      Evidence = "\n".join(
            [
                  f"Evidence {i+1}:\n{evid['text']}" for i,evid in enumerate(top_evidence)
            ]
      )
      prompt  = f"""<|user|>
You are an expert fact-checker evaluating a backstory claim against source novel excerpts.

Claim: "{claim}"
Entity: "{entity}"

Source Excerpts:
{Evidence}

EVALUATION CRITERIA:
1. SUPPORT: The claim is confirmed true by the source excerpts.
2. CONTRADICT: The claim contradicts or conflicts with facts in the source excerpts (e.g. asserts someone was captain when the excerpts show someone else was captain; asserts someone was a merchant when excerpts show they were a fisherman; asserts someone died vs arrived safely; asserts a character has different parents).
3. NOT MENTIONED: The asserted event/fact is completely unmentioned in the source excerpts.

Evaluate concisely, then conclude on the last line with exactly:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""
      return prompt


#verifying final claim through llm
def verify_claim (backstory, metadata, faiss_index, entity_index) :
      Verification = []
      retrievals = claim_retrieval(backstory,metadata,faiss_index,entity_index)
      for retrieval in retrievals :
            prompt = prompt_generation(
                  retrieval["Claim"],
                  retrieval["Evidence"],
                  retrieval["Entity"]
            )
            result = generate_response(prompt)
            Verification.append(
                  {
                        "Claim" : retrieval["Claim"],
                        "Evidence" : retrieval["Evidence"],
                        "Verification_result" : result
                  }
            )
      return Verification

if __name__ == '__main__' :
      metadata = loadChunks(os.path.join("Data", "atomicChunks.json"))
      entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
      faiss_index =  faiss.read_index(os.path.join("Data", "atomic.index"))
      while (True) :
            query = input("Enter Backstory (e or E to exit) :\n")
            if query in "eE" :
                  print("Exiting.....")
                  break
            Verification = verify_claim(query,metadata,faiss_index,entity_index)
            print(Verification)