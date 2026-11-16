"""Read-only smoke check for the deployed four-page WebUI."""

import json
import time
from urllib.request import urlopen


BASE = "http://127.0.0.1:8100"


def get(path):
    with urlopen(BASE + path, timeout=15) as response:
        assert response.status == 200, (path, response.status)
        return response.read().decode("utf-8")


pages = {path: get(path) for path in ("/", "/benchmark", "/proxy", "/server")}
for path, html in pages.items():
    for route in ('href="/"', 'href="/benchmark"', 'href="/proxy"', 'href="/server"'):
        assert route in html, (path, route)
    assert html.count('aria-current="page"') == 1, path
    assert "192.0.2.1" not in html and "/home/example" not in html, path
assert "bench-form" not in pages["/"] and "new-key-value" not in pages["/"]
assert "bench-form" in pages["/benchmark"] and "new-key-value" not in pages["/benchmark"]
assert "new-key-value" in pages["/proxy"] and "bench-form" not in pages["/proxy"]
assert "gpu-grid" in pages["/server"] and "bench-form" not in pages["/server"]

first = json.loads(get("/api/server/status"))
time.sleep(1)
second = json.loads(get("/api/server/status"))
assert second["gpu_count"] == 8 and len(second["gpus"]) == 8, second.get("gpu_error")
assert all(gpu["memory_total_mib"] > 0 for gpu in second["gpus"])
assert second["cpu"]["logical_cores"] > 0 and second["memory"]["total_bytes"] > 0
assert second["disk"]["total_bytes"] > 0
assert second["cpu"]["util_pct"] is not None
print(json.dumps({"pages": list(pages), "gpu_count": second["gpu_count"],
                  "gpu_power_w": second["gpu"]["power_w"],
                  "cpu_util_pct": second["cpu"]["util_pct"],
                  "model_state": second["model_state"]}, ensure_ascii=False))
