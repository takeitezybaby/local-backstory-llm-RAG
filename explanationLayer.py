from claimRetrieval import claim_retrieval, loadEntityIndex, loadChunks
from aggregation import aggregate_results
from verfication import verify_claim
import requests
import faiss

API = "http://localhost:11434/api/generate"
tokens = 8092

#function to structure the verification results to feed into llm prompt
def structure_claims(results) :
      structured = ""
      for i,r in enumerate(results,1):
            evidence_text = ""
            if r["Evidence"] :
                  for evid in r["Evidence"][:4]:
                        evidence_text += f"-{evid["text"]}\n"
            else :
                  evidence_text += "-No valid evidence found"
            structured += f"""
{i}. Claim : {r["Claim"]}
   Verification Result : {r["Verification_result"]}
   Evidence : {evidence_text}
"""
      print(f"This is debugging\n{structured.strip()}\n\n")
      return structured.strip()

#function to generate sttict and grounded prompt for the llm
def prompt_generation(llm_result, aggregated_result) :
      llm_structured = structure_claims(llm_result)
      prompt =f"""
You are explaining the evaluation of a fictional backstory against source material.

Final Verdict: {aggregated_result['Final Verdict']}
Score: {round(aggregated_result['Normalized Score'], 2)}

Breakdown:
SUPPORT: {aggregated_result['Breakdown']['Supporting claims']}
CONTRADICT: {aggregated_result['Breakdown']['Contradicting claims']}
NOT_MENTIONED: {aggregated_result['Breakdown']['Not Mentioned claims']}
TOTAL: {aggregated_result['Breakdown']['Total claims']}

Claims:
{llm_structured}

Instructions:
- Explain the final verdict clearly for each claim
- For each claim, explain WHY it was SUPPORT, CONTRADICT, or NOT_MENTIONED
- Use ONLY the provided evidence
- Do NOT add new facts
- Do NOT hallucinate
- If a claim is SUPPORTED, clearly explain the supporting statements
- If a claim is NOT_MENTIONED, explicitly say no evidence supports it
- If a claim is CONTRADICT, clearly explain the contradiction
- Keep explanation concise and structured
- Suggest how the backstory can be improved

Output:
A clear and structured explanation for the user.

"""
      return prompt.strip()


#function to call out local LLM
def llm_call(prompt) :
      result =  requests.post(API,json={
            "model":"mistral:7b",
            "prompt":prompt,
            "stream":False,
            "options":{
                  "temperature":0.2,
                  "num_ctx":tokens
            }
      })
      return result.json()["response"]


if __name__ == '__main__' :
      atomicChunks = loadChunks("atomicChunks.json")
      entity_index = loadEntityIndex("entity.json")
      faiss_index = faiss.read_index("atomic.index")
      while(True) :
            backstory = input("Enter backstory (e or E to exit) :\n")
            if backstory in "eE" :
                  print("Exiting...")
                  break
            llm_verification = verify_claim(backstory, atomicChunks,faiss_index, entity_index)
            aggregated = aggregate_results(llm_verification)
            prompt = prompt_generation(llm_verification,aggregated)
            response = llm_call(prompt)
            print(response)