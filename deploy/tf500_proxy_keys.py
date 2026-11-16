"""Local API proxy keys and metadata-only usage accounting."""

import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path("/data/xtier-86kd-console")
MASTER_FILE = ROOT / "proxy-api-key"
KEYS_FILE = ROOT / "proxy-keys.json"
ACCESS_LOG = ROOT / "proxy-access.jsonl"
KEY_LOCK = threading.Lock()
USAGE_LOCK = threading.Lock()
USAGE_CACHE = {"path": None, "inode": None, "offset": 0, "tail": b"",
               "by_key": {}, "days": {}, "recent": []}
VALID_DAYS = (0, 1, 7, 30, 90)
MAX_REQUESTS = 100_000_000


def _read_keys():
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        return data.get("keys", []) if isinstance(data.get("keys"), list) else []
    except (OSError, ValueError):
        return []


def _write_keys(entries):
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="proxy-keys.", suffix=".tmp", dir=KEYS_FILE.parent)
    path = Path(name)
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "keys": entries}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        path.replace(KEYS_FILE)
    finally:
        path.unlink(missing_ok=True)


def entries():
    master = MASTER_FILE.read_text(encoding="utf-8").strip()
    if not master:
        raise RuntimeError("代理主密钥为空")
    return [{"id": "master", "name": "系统主 Key", "key": master,
             "created_at": 0, "expires_at": 0, "max_requests": 0, "master": True}] + [
        {**item, "master": False} for item in _read_keys()]


def create(name, days=0, max_requests=0):
    name = " ".join(str(name or "").split())
    if not name or len(name) > 40:
        raise ValueError("名称长度应为 1–40 字")
    try:
        days = int(days)
        max_requests = int(max_requests)
    except (TypeError, ValueError):
        raise ValueError("有效期或请求额度无效") from None
    if days not in VALID_DAYS:
        raise ValueError("有效期只支持永久、1、7、30、90 天")
    if not 0 <= max_requests <= MAX_REQUESTS:
        raise ValueError("请求额度应在 0–1 亿之间，0 为不限")
    with KEY_LOCK:
        items = _read_keys()
        if name == "系统主 Key" or any(item.get("name") == name for item in items):
            raise ValueError("已有同名 Key")
        now = int(time.time())
        item = {"id": "k" + secrets.token_hex(6), "name": name,
                "key": "xt86_" + secrets.token_urlsafe(48),
                "created_at": now, "expires_at": now + days * 86400 if days else 0,
                "max_requests": max_requests}
        _write_keys(items + [item])
        return dict(item)


def delete(key_id):
    if key_id == "master":
        raise ValueError("系统主 Key 不能删除")
    with KEY_LOCK:
        items = _read_keys()
        remaining = [item for item in items if item.get("id") != key_id]
        if len(remaining) == len(items):
            raise ValueError("没有这个 Key")
        _write_keys(remaining)


def usage():
    with USAGE_LOCK:
        try:
            stat = ACCESS_LOG.stat()
        except OSError:
            stat = None
        path = str(ACCESS_LOG)
        inode = (stat.st_dev, stat.st_ino) if stat else None
        if (USAGE_CACHE["path"] != path or USAGE_CACHE["inode"] != inode or
                (stat and stat.st_size < USAGE_CACHE["offset"])):
            USAGE_CACHE.update(path=path, inode=inode, offset=0, tail=b"",
                               by_key={}, days={}, recent=[])
        if stat and stat.st_size > USAGE_CACHE["offset"]:
            with ACCESS_LOG.open("rb") as stream:
                stream.seek(USAGE_CACHE["offset"])
                chunk = USAGE_CACHE["tail"] + stream.read(stat.st_size - USAGE_CACHE["offset"])
            lines = chunk.split(b"\n")
            USAGE_CACHE["tail"] = lines.pop()
            USAGE_CACHE["offset"] = stat.st_size
            for line in lines:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(item, dict):
                    continue
                kid = str(item.get("key_id") or "legacy")
                stamp = int(item.get("time") or 0)
                day = time.strftime("%Y-%m-%d", time.localtime(stamp))
                by_key = USAGE_CACHE["by_key"].setdefault(kid, {"requests": 0, "errors": 0,
                    "input": 0, "output": 0, "last": 0})
                by_day = USAGE_CACHE["days"].setdefault(day, {"requests": 0, "errors": 0,
                    "input": 0, "output": 0})
                for target in (by_key, by_day):
                    target["requests"] += 1
                    target["errors"] += int(int(item.get("status") or 0) >= 400)
                    target["input"] += int(item.get("input") or 0)
                    target["output"] += int(item.get("output") or 0)
                by_key["last"] = max(by_key["last"], stamp)
                USAGE_CACHE["recent"].append(item)
                if len(USAGE_CACHE["recent"]) > 100:
                    del USAGE_CACHE["recent"][:-100]
        return {"by_key": json.loads(json.dumps(USAGE_CACHE["by_key"])),
                "days": json.loads(json.dumps(USAGE_CACHE["days"])),
                "recent": list(USAGE_CACHE["recent"])}


def authenticate(value):
    if not value:
        return None, 401
    match = None
    for entry in entries():
        if hmac.compare_digest(value, entry["key"]):
            match = entry
    if not match or match.get("expires_at", 0) and time.time() >= match["expires_at"]:
        return None, 401
    limit = int(match.get("max_requests") or 0)
    if limit and usage()["by_key"].get(match["id"], {}).get("requests", 0) >= limit:
        return None, 402
    return match, 200


def public_entries():
    stats = usage()["by_key"]
    result = []
    for entry in entries():
        key = entry["key"]
        used = stats.get(entry["id"], {})
        result.append({"id": entry["id"], "name": entry["name"],
                       "masked_key": key[:9] + "…" + key[-4:],
                       "created_at": entry.get("created_at", 0),
                       "expires_at": entry.get("expires_at", 0),
                       "max_requests": entry.get("max_requests", 0),
                       "master": entry["master"], "usage": used})
    return result
