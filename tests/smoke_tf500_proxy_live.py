"""One small inference through the deployed proxy; never print credentials or text."""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import tf500_host  # noqa: E402

key = Path(os.environ.get("TF500_PROXY_KEY_FILE", "/data/xtier-86kd-console/proxy-api-key")).read_text().strip()
base = f"http://{tf500_host.resolve_host()}:8000/v1"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
headers = {"Authorization": "Bearer " + key}
started = time.monotonic()
models = urllib.request.Request(base + "/models", headers=headers)
with opener.open(models, timeout=15) as response:
    listed = json.load(response)
assert any(row.get("id") == "DeepSeek-V4.1-Flash-1017" for row in listed.get("data", []))
body = json.dumps({"model": "DeepSeek-V4.1-Flash-1017",
                   "messages": [{"role": "user", "content": "请只回复：收到"}],
                   "max_tokens": 8, "temperature": 0, "stream": False}).encode()
request = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={**headers, "Content-Type": "application/json"})
with opener.open(request, timeout=120) as response:
    result = json.load(response)
assert result.get("choices"), result.get("error", "missing choices")
print(json.dumps({"models_http": 200, "chat_http": 200,
                  "elapsed_s": round(time.monotonic() - started, 2),
                  "usage": result.get("usage", {})}, ensure_ascii=False))
