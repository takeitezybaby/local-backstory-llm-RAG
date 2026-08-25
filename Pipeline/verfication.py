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


from querySearch import get_canonical_profile

#Generating prompt
def prompt_generation (claim, evidence_list, entity) :
      top_evidence = evidence_list[:12]
      Evidence = "\n".join(
            [
                  f"Evidence {i+1}:\n{evid['text']}" for i,evid in enumerate(top_evidence)
            ]
      )
      profile = get_canonical_profile(entity)
      profile_section = f"Canonical Knowledge about {entity}:\n{profile}\n\n" if profile else ""
      
      prompt  = f"""<|user|>
You are a precise literary fact-checker. Evaluate the Claim against the Canonical Knowledge and Novel Excerpts.

Claim: "{claim}"
Character: "{entity}"

{profile_section}Source Excerpts:
{Evidence}

CLASSIFICATION RULES:
1. CONTRADICT: The claim asserts false facts that directly clash with the character's canonical identity, parentage, role, allegiance, or fate (e.g. wrong parent, claiming they are a pirate/traitor/convict when they are noble/loyal, claiming they died when they lived or were executed instead of dying of illness).
2. SUPPORT: The claim is directly confirmed true by the excerpts or canonical facts.
3. NOT MENTIONED: The claim describes an unmentioned private past, investment, hobby, or background detail (e.g. investing in railway shares, painting landscapes, learning harp in Vienna, writing a personal memoir, or past job prior to the novel) that is simply absent from the text without creating an impossible contradiction.

End on the final line with exactly:
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