import json

with open("Data/eval_traces.json", "r", encoding="utf-8") as f:
    traces = json.load(f)

for idx in [2, 3, 5, 9, 16]:
    t = [x for x in traces if x["id"] == idx][0]
    print(f"\n================ CLAIM {idx} ================")
    print("Question:", t["question"])
    print("Response:", t["response"])
    print("Context count:", len(t["contexts"]))
    print("Ground Truth:", t["ground_truth"])
