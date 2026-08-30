from verfication import verify_claim
from claimRetrieval import claim_retrieval, loadEntityIndex, loadChunks
import faiss
import os

"""
Weights being used : 
      1. SUPPORTS = +1
      2. NOT MENTIONED = 0
      3. CONTRADICTION = -2
"""

import re

def extract_single_verdict(raw_str):
    if not isinstance(raw_str, str):
        return "NOT MENTIONED"
    
    # 1. Look for explicit "Verdict: <LABEL>"
    matches = re.findall(r'verdict\s*:\s*["\']?\s*(SUPPORT|CONTRADICT|NOT MENTIONED|INCOMPATIBLE|COMPATIBLE)', raw_str, re.IGNORECASE)
    if matches:
        last_match = matches[-1].upper()
        if last_match in ["CONTRADICT", "INCOMPATIBLE"]:
            return "CONTRADICT"
        if last_match in ["SUPPORT", "COMPATIBLE"]:
            return "SUPPORT"
        if last_match in ["NOT MENTIONED"]:
            return "NOT MENTIONED"

    # 2. Strict exact line matching from bottom up
    lines = [l.strip() for l in raw_str.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        up = line.upper().strip(" '\"`.:;,")
        if up in ["SUPPORT", "VERDICT: SUPPORT", "VERDICT:SUPPORT"]:
            return "SUPPORT"
        if up in ["CONTRADICT", "VERDICT: CONTRADICT", "VERDICT:CONTRADICT"]:
            return "CONTRADICT"
        if up in ["NOT MENTIONED", "VERDICT: NOT MENTIONED", "VERDICT:NOT MENTIONED"]:
            return "NOT MENTIONED"

    # 3. Fallback regex detection without negation
    if re.search(r'\b(not\s+mentioned|unmentioned|not\s+supported)\b', raw_str, re.IGNORECASE):
        return "NOT MENTIONED"
    if re.search(r'\b(contradict|contradicts|contradiction|incompatible)\b', raw_str, re.IGNORECASE):
        return "CONTRADICT"
    if re.search(r'\b(support|supports|supported|compatible)\b', raw_str, re.IGNORECASE):
        return "SUPPORT"

    return "NOT MENTIONED"

def aggregate_results (llm_response) :
      support = 0 
      contradict = 0
      not_mentioned = 0
      for response in llm_response :
            verdict = extract_single_verdict(response.get("Verification_result", ""))
            if verdict == 'CONTRADICT':
                  contradict += 1
            elif verdict == 'SUPPORT':
                  support += 1
            else:
                  not_mentioned += 1
      total_length = len(llm_response)
      score = (1*support) + (-2 * contradict)
      normalized_score = score/total_length if total_length > 0 else 0

      # DECISION LOGIC (Conjunctive / Pessimistic on Contradiction - Canonical Config [2])
      if contradict >= 1:
            verdict = "INCOMPATIBLE"
      elif support >= 1 and not_mentioned == 0:
            verdict = "COMPATIBLE"
      else:
            verdict = "NO CONTRADICTION, BUT NOT SUPPORTED"

      
      return {
            "Final Verdict" : verdict,
            "Normalized Score" : normalized_score,
            "Breakdown" : {
                  "Supporting claims" : support,
                  "Contradicting claims" : contradict,
                  "Not Mentioned claims" :not_mentioned,
                  "Total claims" : total_length
            }
      }


if __name__ == '__main__' :
      atomicChunks = loadChunks(os.path.join("Data", "atomicChunks.json"))
      entity_index = loadEntityIndex(os.path.join("Data", "entity.json"))
      faiss_index = faiss.read_index(os.path.join("Data", "atomic.index"))
      while(True) :
            backstory = input("Enter backstory (e or E to exit) :\n")
            if backstory in "eE" :
                  print("Exiting...")
                  break
            llm_verification = verify_claim(backstory, atomicChunks,faiss_index, entity_index)
            aggregated = aggregate_results(llm_verification)
            print("-"*10+" LLM RESPONSE "+"-"*10)
            for response in llm_verification :
                  print(f"Claim:{response["Claim"]}\nResult:{response["Verification_result"]}")
                  print("-"*40)
            print("-"*10+" AGGREGATE RESULT "+"-"*10)
            print(f"Final Verdict:{aggregated["Final Verdict"]}\nFinal Score:{aggregated["Normalized Score"]}\n{aggregated["Breakdown"]}")