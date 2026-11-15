"""Authenticated tf500 TP4 readiness and one short generation request."""

import json
import time
import urllib.request
from pathlib import Path

base = "http://<TF500_IP>:30001"
key = Path("/data/sglang-api-key").read_text().strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def request(path, data=None, timeout=30):
    raw = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(base + path, raw, headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

models = request("/v1/models")
print("models:", [item["id"] for item in models["data"]], flush=True)
start = time.monotonic()
result = request(
    "/v1/chat/completions",
    {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "请只回复：测试成功"}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=180,
)
print("seconds:", round(time.monotonic() - start, 3), flush=True)
print("usage:", result.get("usage"), flush=True)
print("answer:", result["choices"][0]["message"].get("content"), flush=True)
