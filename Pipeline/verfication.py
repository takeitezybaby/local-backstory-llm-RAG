import json
import requests
import faiss
import os
from claimRetrieval import claim_retrieval, loadEntityIndex, loadChunks

API = "http://localhost:11434/api/generate"
Tokens = 8192


#function to get response from llm
def generate_response (prompt) :
      response = requests.post(API, json={
            "model" : "koesn/mistral-7b-instruct:latest",
            "prompt" : prompt,
            "stream" : False,
            "options" : {
                   "num_ctx" : Tokens,
                   "temperature" : 0.2
            }
      })
      response = response.json()["response"]
      return response


#Generating prompt
def prompt_generation (claim, evidence_list, entity) :
      Evidence = "\n".join(
            [
                  f"Evidence {1+1} :\n {evid["text"]}" for i,evid in enumerate(evidence_list)
            ]
      )
      prompt  = f"""
      You're verifying a factual claim against evidence from  a novel, Verify the claim strictly
      Claim : {claim}
      Entity : {entity}
      Evidence : {Evidence}

      INSTRUCTIONS:
      1. The evidence must explicitly mention both :
        - The SAME character/entity
        - The SAME action or fact as the claim
      2. If character matches but the action is not supported, ANSWER "NOT MENTIONED"
      3. Do not assume
      4. Do not guess based on the similar situations
      
      Does the evidence SUPPORT, CONTRADICT or NOT MENTIONED in the claim?
      ANSWER IN ONE WORD ONLY : SUPPORT OR CONTRADICT OR NOT MENTIONED
      DO NOT EXPLAIN
      ONLY USE ONE WORD ANSWER
      """
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