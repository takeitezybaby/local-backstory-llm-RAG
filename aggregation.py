from verfication import verify_claim
from claimRetrieval import claim_retrieval, loadEntityIndex, loadChunks
import faiss

"""
Weights being used : 
      1. SUPPORTS = +1
      2. NOT MENTIONED = 0
      3. CONTRADICTION = -2
"""

def aggregate_results (llm_response) :
      support = 0 
      contradict = 0
      not_mentioned = 0
      for response in llm_response :
            result = response["Verification_result"].strip().upper()
            if result == 'SUPPORT' :
                  support +=1
            elif result == 'NOT MENTIONED' :
                  not_mentioned += 1
            elif result == 'CONTRADICT' : 
                  contradict +=1
      total_length = len(llm_response)
      score = (1*support) + (-2 * contradict)
      normalized_score = score/total_length if total_length > 0 else 0

      #DECISION LOGIC 
      if contradict > 0 : 
            verdict = "INCOMPATIBLE"
      else :
            if normalized_score>0.75 :
                  verdict = "COMPATIBLE"
            elif 0.3<normalized_score<=0.75 :
                  verdict = "PARTIALLY COMPATIBLE"
            else :
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
            print("-"*10+" LLM RESPONSE "+"-"*10)
            for response in llm_verification :
                  print(f"Claim:{response["Claim"]}\nResult:{response["Verification_result"]}")
                  print("-"*40)
            print("-"*10+" AGGREGATE RESULT "+"-"*10)
            print(aggregated)