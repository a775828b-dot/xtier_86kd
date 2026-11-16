"""Live smoke check for the loopback admin's master-key copy action."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import tf500_proxy_keys as keys

BASE = "http://127.0.0.1:8100"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
expected = keys.MASTER_FILE.read_text(encoding="utf-8").strip()
assert expected

with opener.open(BASE + "/proxy", timeout=5) as response:
    html = response.read().decode("utf-8")
assert "复制系统主 Key" in html
assert expected not in html

with opener.open(BASE + "/api/proxy/admin", timeout=5) as response:
    admin = response.read().decode("utf-8")
assert expected not in admin


def copy(headers):
    req = urllib.request.Request(BASE + "/api/proxy/master-key/copy", b"copy=1", headers=headers)
    try:
        with opener.open(req, timeout=5) as response:
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.load(exc)


status, headers, payload = copy({"X-Requested-With": "fetch", "Origin": BASE})
assert status == 200 and payload["key"] == expected
assert headers["Cache-Control"] == "no-store"
assert copy({})[0] == 403
assert copy({"X-Requested-With": "fetch", "Origin": "http://evil.invalid"})[0] == 403
print("copy button, masked page, copy endpoint and request checks: OK")
