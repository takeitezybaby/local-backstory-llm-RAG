import requests
import json

c12 = "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars."
ev = """Evidence 1:
" We have lost our poor Captain Leclère . " " And the cargo ? " " Is all safe , your honor ; but poor Captain Leclère— " " What is the matter with him ? " " He is dead . "

Evidence 2:
" He died of brain fever in dreadful agony , off the Isle of Monte Cristo . " " What a misfortune ! "

Evidence 3:
Dantès was the mate who brought the Pharaon safely into the harbor of Marseilles after the captain's death ."""

prompt = f"""<|user|>
You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{c12}"

Evidence Excerpts:
{ev}

EVALUATION CRITERIA:
1. SUPPORT: The claim is explicitly confirmed as true by the evidence excerpts.
2. CONTRADICT: The claim contradicts facts in the evidence excerpts. Specifically, if the claim attributes a role, title, or deed to a person (e.g. "Captain Leclère commanded the ship into harbor"), but the evidence shows that person died before the event or someone else performed the deed, this is an explicit conflict. You must classify any such conflict as CONTRADICT.
3. NOT MENTIONED: Used ONLY when the topic or character is completely absent and no conflicting facts exist in the excerpts.

Evaluate concisely, then conclude on the very last line with:
Verdict: SUPPORT
or
Verdict: CONTRADICT
or
Verdict: NOT MENTIONED<|end|>
<|assistant|>"""

resp = requests.post("http://localhost:11434/api/generate", json={
    "model": "phi3.5:latest",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0}
}).json()["response"]

print("Phi Response for Claim 12:")
print(resp)
