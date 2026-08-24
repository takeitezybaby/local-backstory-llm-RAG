import requests
import json

c9 = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."
ev = """Evidence 1:
Lord Edward Glenarvan was on board with his young wife , Lady Helena , and one of his cousins , Major MacNabb . The yacht was called the Duncan .

Evidence 2:
The captain , John Mangles , at once informed Lord Glenarvan of the fact , who mounted on deck with Major MacNabb , and asked the captain what he thought of the animal .

Evidence 3:
Tom Austin , the mate , caught sight of the shark . John Mangles was the captain of the Duncan ."""

prompt = f"""<|user|>
You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{c9}"

Evidence Excerpts:
{ev}

EVALUATION CRITERIA:
1. SUPPORT: The claim is explicitly confirmed as true by the evidence excerpts.
2. CONTRADICT: The claim contradicts facts in the evidence excerpts. Specifically, if the claim attributes a role, title, or deed to a person (e.g. "Major MacNabb is the captain"), but the evidence shows SOMEONE ELSE held that role or deed (e.g. "John Mangles was the captain"), this is an explicit conflict. You must classify any such conflict as CONTRADICT.
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

print("Phi Response for Claim 9 with Explicit Role Conflict Rule:")
print(resp)
