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
                        "model" : "koesn/mistral-7b-instruct:latest",
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
      top_evidence = evidence_list[:10]
      Evidence = "\n".join(
            [
                  f"Evidence {i+1} :\n {evid['text']}" for i,evid in enumerate(top_evidence)
            ]
      )
      prompt  = f"""[INST] You are an expert fact-checker evaluating a backstory claim against novel evidence excerpts.

Claim: "{claim}"
Entity: "{entity}"

Evidence Excerpts:
{Evidence}

EVALUATION RULES:
1. SUPPORT: The evidence explicitly confirms the claim's core facts (the character, role, and actions/events).
2. CONTRADICT: The claim asserts facts that conflict with or contradict the source evidence (e.g. asserts an entity has a different role/title, wrong parent, wrong job, or claims an event succeeded when evidence shows they died or were betrayed).
3. NOT MENTIONED: The key asserted fact/action is completely unmentioned in the evidence excerpts (e.g. mentions Lady Helena, but says nothing about being a military nurse in the Crimean War).

CRITICAL RULE: If the evidence shows the character in an entirely different role, relation, or state than asserted in the claim (e.g. Major MacNabb is Lord Glenarvan's cousin rather than captain; Fernand is a fisherman rather than a wealthy Parisian merchant; Ayrton is a convict/mutineer rather than loyal mate; Leclère died at sea rather than safely landing), you MUST classify as "Verdict: CONTRADICT".

Briefly verify the claim, then conclude your answer on the last line with exactly:
"Verdict: SUPPORT", "Verdict: CONTRADICT", or "Verdict: NOT MENTIONED". [/INST]"""
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