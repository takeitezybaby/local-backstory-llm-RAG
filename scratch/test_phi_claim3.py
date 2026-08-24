import requests
import json

c3 = "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India."
ev = """Evidence 1:
Jacques Paganel was the secretary of the Geographical Society of Paris .

Evidence 2:
But to mistake the vessel , and be sailing to Chili , when he wished to go to India , was the height of absent - mindedness .

Evidence 3:
He had come on board the Duncan by mistake , having taken her for the Scotia which was to sail for Calcutta ."""

prompt = f"""<|user|>
You are an expert fact-checker evaluating a claim against novel evidence excerpts.

Claim: "{c3}"

Evidence:
{ev}

Evaluate if the evidence supports, contradicts, or does not mention the claim.
Conclude on the very last line with:
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

print("Phi Response:")
print(resp)
