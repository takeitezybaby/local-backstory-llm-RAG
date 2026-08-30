import time
import requests
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3.5:latest"

def test_query(i):
    t0 = time.time()
    resp = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": f"Write a single sentence about character number {i}.",
        "stream": False,
        "options": {"num_predict": 30, "temperature": 0.0}
    }, timeout=60)
    return i, time.time() - t0, resp.json().get("response", "")[:30]

print("Testing 4 concurrent requests to Ollama...")
t_start = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(test_query, range(4)))

print(f"Total time for 4 concurrent queries: {time.time() - t_start:.2f}s")
for idx, el, txt in results:
    print(f"Query {idx}: {el:.2f}s -> {txt}...")
