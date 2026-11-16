"""Read-only check that the proxy and benchmark select the live backend."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import tf500_api_proxy as proxy
import tf500_benchmark as benchmark


target = proxy.backend_target()
assert target["online"] and target["port"] == 30000, target
capacity = benchmark._server_info()
assert capacity["backend_port"] == target["port"], capacity
assert capacity["context"] > 0 and capacity["kv"] > 0 and capacity["max_running"] > 0
print(json.dumps({"backend_port": target["port"], "backend_label": target["label"],
                  "context": capacity["context"], "kv": capacity["kv"],
                  "max_running": capacity["max_running"]}, ensure_ascii=False))
