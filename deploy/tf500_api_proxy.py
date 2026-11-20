"""Authenticated OpenAI API gateway for the local SGLang service.

Only /v1/ is forwarded. Request bodies and response bodies are never logged.
Supports prefix-sticky DP routing (TF500_PROXY_STICKY_DP=1) and metadata-only
observability (cached_tokens, dp_rank, temperature, stream, ttft_s, decode_tps).
"""

import http.client
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import tf500_dp_router as dp_router
import tf500_host
import tf500_proxy_keys as keys

HOST = tf500_host.resolve_host()
PORT = 8000
BACKEND_HOST = os.environ.get("TF500_BACKEND_HOST", "").strip() or HOST
BACKEND_PORT = 30000
ROOT = Path("/data/xtier-86kd-console")
PROXY_KEY_FILE = ROOT / "proxy-api-key"
BACKEND_KEY_FILE = Path("/data/sglang-api-key")
ACCESS_LOG = ROOT / "proxy-access.jsonl"
MAX_BODY = 64 * 1024 * 1024
_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"\\]{1,120})"')
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailer", "transfer-encoding", "upgrade", "content-length"}
REPLACED = HOP | {"host", "authorization", "x-api-key", "accept-encoding"}
LOG_LOCK = threading.Lock()
BACKEND_LOCK = threading.Lock()
BACKEND_CACHE = {"at": 0.0, "host": None, "default_port": None, "value": None}
DP_ROUTER = dp_router.PrefixStickyDPRouter()


def _key(path):
    return path.read_text(encoding="utf-8").strip()


def ensure_proxy_key():
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(PROXY_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("xt86_" + secrets.token_urlsafe(48) + "\n")
    if not _key(PROXY_KEY_FILE) or not _key(BACKEND_KEY_FILE):
        raise RuntimeError("API 密钥文件为空")


def _authenticate(headers):
    authorization = headers.get("Authorization", "")
    supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else headers.get("x-api-key", "")
    return keys.authenticate(supplied)


def _probe_backend(port):
    try:
        request = urllib.request.Request(
            f"http://{BACKEND_HOST}:{port}/v1/models",
            headers={"Authorization": "Bearer " + _key(BACKEND_KEY_FILE)},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def backend_target():
    """Probe the TP8 backend on BACKEND_PORT; keep one target per forwarded request."""
    now = time.monotonic()
    with BACKEND_LOCK:
        if (BACKEND_CACHE["value"] is not None and now - BACKEND_CACHE["at"] < 3
                and BACKEND_CACHE["host"] == BACKEND_HOST
                and BACKEND_CACHE["default_port"] == BACKEND_PORT):
            return dict(BACKEND_CACHE["value"])
        online = _probe_backend(BACKEND_PORT)
        value = {"host": BACKEND_HOST, "port": BACKEND_PORT, "online": online,
                 "label": "TP8 · 86KD"}
        BACKEND_CACHE.update(at=time.monotonic(), host=BACKEND_HOST,
                             default_port=BACKEND_PORT, value=value)
        return dict(value)


def _backend_online():
    return backend_target()["online"]


def status():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{HOST}:{PORT}/health", timeout=2) as response:
            data = json.load(response)
        return {"proxy": "online", "backend": "online" if data.get("backend_online") else "offline",
                "backend_port": data.get("backend_port"), "backend_label": data.get("backend_label"),
                "dp_router": data.get("dp_router"),
                "base_url": f"http://{HOST}:{PORT}/v1"}
    except (OSError, urllib.error.URLError, ValueError):
        return {"proxy": "offline", "backend": "unknown", "base_url": f"http://{HOST}:{PORT}/v1"}


def _record(method, path, status_code, elapsed_s, bytes_out, key_id, model, tokens, extra=None):
    item = {"time": int(time.time()), "method": method, "path": path,
            "status": status_code, "elapsed_s": round(elapsed_s, 3), "bytes_out": bytes_out,
            "key_id": key_id, "model": model, "input": tokens[0], "output": tokens[1]}
    if isinstance(extra, dict):
        item.update(extra)
    try:
        with LOG_LOCK:
            ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ACCESS_LOG.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _has_token_delta(value):
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            delta = c0.get("delta")
            if isinstance(delta, dict):
                if (delta.get("content") or delta.get("reasoning_content")
                        or delta.get("reasoning") or delta.get("tool_calls")):
                    return True
            if c0.get("text"):
                return True
    evt_type = value.get("type")
    if isinstance(evt_type, str) and evt_type.endswith(".delta") and value.get("delta"):
        return True
    return False


class UsageCapture:
    def __init__(self, content_type, started=None):
        self.sse = "text/event-stream" in content_type
        self.started = started
        self.buffer = b""
        self.tokens = (0, 0)
        self.cached_tokens = 0
        self.first_token_at = None
        self.last_token_at = None

    def _extract_usage(self, usage):
        try:
            prompt_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            comp_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            self.tokens = (prompt_tok, comp_tok)
        except (TypeError, ValueError):
            pass
        cached = 0
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            try:
                cached = int(details.get("cached_tokens") or 0)
            except (TypeError, ValueError):
                cached = 0
        elif usage.get("cached_tokens") is not None:
            try:
                cached = int(usage.get("cached_tokens") or 0)
            except (TypeError, ValueError):
                cached = 0
        if cached >= 0:
            self.cached_tokens = cached

    def _event(self, data, now=None):
        try:
            value = json.loads(data)
        except ValueError:
            return
        if not isinstance(value, dict):
            return
        if self.sse and now is not None and _has_token_delta(value):
            if self.first_token_at is None:
                self.first_token_at = now
            self.last_token_at = now
        usage = value.get("usage")
        if isinstance(usage, dict) and usage:
            self._extract_usage(usage)

    def feed(self, chunk, now=None):
        if self.sse:
            ts = time.monotonic() if now is None else now
            self.buffer += chunk
            if len(self.buffer) > 2 * 1024 * 1024:
                self.buffer = self.buffer[-1024 * 1024:]
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]" or not payload:
                    continue
                has_usage = b'"usage"' in payload
                if self.first_token_at is None or has_usage:
                    self._event(payload, now=ts)
                elif ((b'"delta"' in payload or b'"text"' in payload)
                        and b'"choices":[]' not in payload
                        and b'"choices": []' not in payload):
                    self.last_token_at = ts
        elif len(self.buffer) + len(chunk) <= 2 * 1024 * 1024:
            self.buffer += chunk

    def result(self):
        if not self.sse:
            self._event(self.buffer)
        return self.tokens

    def metrics(self):
        prompt_tok, comp_tok = self.result()
        ttft_s = None
        decode_tps = None
        if self.sse and self.first_token_at is not None and self.started is not None:
            ttft_val = self.first_token_at - self.started
            if ttft_val >= 0:
                ttft_s = round(ttft_val, 4)
            if self.last_token_at is not None and comp_tok >= 2:
                decode_s = self.last_token_at - self.first_token_at
                if decode_s > 0.001:
                    decode_tps = round((comp_tok - 1) / decode_s, 2)
        return {
            "cached_tokens": self.cached_tokens,
            "ttft_s": ttft_s,
            "decode_tps": decode_tps,
        }


def _extract_model(body):
    if not body:
        return ""
    if len(body) <= 65536:
        try:
            payload = json.loads(body)
            return str(payload.get("model") or "")[:120] if isinstance(payload, dict) else ""
        except ValueError:
            return ""
    match = _MODEL_RE.search(body[:16384]) or _MODEL_RE.search(body[-8192:])
    return match.group(1).decode("utf-8", "replace")[:120] if match else ""


def _parse_request_meta(body, path):
    """Parse JSON body once for model, observability metadata, and DP routing."""
    if not body:
        return None, "", None, False
    payload = None
    if path in dp_router.ROUTABLE_PATHS or len(body) <= 65536:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                payload = parsed
        except ValueError:
            payload = None
    if payload is not None:
        model = str(payload.get("model") or "")[:120]
        raw_temp = payload.get("temperature")
        temperature = None
        if isinstance(raw_temp, (int, float)) and not isinstance(raw_temp, bool):
            temperature = round(float(raw_temp), 4)
        stream = bool(payload.get("stream", False))
        return payload, model, temperature, stream
    return None, _extract_model(body), None, False


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        server_version = "XTier86KDProxy/1.1"
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args):
            pass

        def _json(self, status_code, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def do_DELETE(self):
            self._handle()

        def do_PUT(self):
            self._handle()

        def _handle(self):
            self.headers_sent = False
            parsed = urlsplit(self.path)
            if self.command == "GET" and parsed.path == "/health":
                target = backend_target()
                return self._json(200, {
                    "proxy": "ok",
                    "backend_online": target["online"],
                    "backend_port": target["port"],
                    "backend_label": target["label"],
                    "dp_router": DP_ROUTER.stats(),
                })
            decoded_path = unquote(unquote(parsed.path))
            if (not decoded_path.startswith("/v1/") or
                    any(part in (".", "..") for part in decoded_path.split("/")) or
                    "\\" in decoded_path):
                return self._json(404, {"error": "only /v1/ API paths are available"})
            entry, auth_status = _authenticate(self.headers)
            if not entry:
                return self._json(auth_status, {"error": "request quota exhausted" if auth_status == 402
                                                 else "invalid or expired API key"})
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) > 1:
                return self._json(400, {"error": "chunked request bodies are unsupported"})
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else 0
            except ValueError:
                return self._json(400, {"error": "invalid Content-Length"})
            if length < 0 or length > MAX_BODY:
                return self._json(413, {"error": "request body too large"})
            body = self.rfile.read(length) if length else None
            payload, model, temperature, stream = _parse_request_meta(body, decoded_path)
            backend = backend_target()
            headers = {name: value for name, value in self.headers.items()
                       if name.lower() not in REPLACED}
            headers.update({"Host": f"{backend['host']}:{backend['port']}",
                            "Authorization": "Bearer " + _key(BACKEND_KEY_FILE),
                            "Accept-Encoding": "identity"})
            lease = DP_ROUTER.acquire(decoded_path, headers, payload, raw_len=len(body) if body else 0)
            if lease.inject_headers and lease.rank is not None:
                headers[dp_router.HEADER_DP_RANK] = str(lease.rank)
                headers[dp_router.HEADER_OVERRIDE_DP_RANK] = str(lease.rank)

            started = time.monotonic()
            status_code = 502
            bytes_out = 0
            tokens = (0, 0)
            obs = {
                "cached_tokens": 0,
                "dp_rank": lease.rank,
                "dp_route": lease.reason,
                "temperature": temperature,
                "stream": stream,
                "ttft_s": None,
                "decode_tps": None,
            }
            connection = http.client.HTTPConnection(backend["host"], backend["port"], timeout=30)
            target = parsed.path + ("?" + parsed.query if parsed.query else "")
            try:
                connection.connect()
                connection.sock.settimeout(1800)
                connection.request(self.command, target, body=body, headers=headers)
                response = connection.getresponse()
                status_code = response.status
                self.send_response(response.status, response.reason)
                content_type = response.getheader("Content-Type", "")
                capture = UsageCapture(content_type, started=started)
                for name, value in response.getheaders():
                    if name.lower() not in HOP:
                        self.send_header(name, value)
                if lease.rank is not None:
                    self.send_header("X-TF500-DP-Rank", str(lease.rank))
                    self.send_header("X-TF500-DP-Route", lease.reason)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    bytes_out += len(chunk)
                    capture.feed(chunk)
                    if "text/event-stream" in content_type:
                        self.wfile.flush()
                tokens = capture.result()
                obs.update(capture.metrics())
            except (OSError, http.client.HTTPException):
                if not self.wfile.closed and not self.headers_sent:
                    self._json(502, {"error": "inference backend unavailable"})
                status_code = 502
            finally:
                DP_ROUTER.release(lease, ok=(200 <= status_code < 300))
                connection.close()
                _record(self.command, parsed.path, status_code,
                        time.monotonic() - started, bytes_out, entry["id"], model, tokens, extra=obs)

    # BaseHTTPRequestHandler does not expose this flag, so track whether response
    # headers have already been sent before an upstream stream fails.
    original = Handler.end_headers

    def end_headers(self):
        self.headers_sent = True
        return original(self)

    Handler.end_headers = end_headers
    return Handler


CARD_TMPL = """
<section class="xt-proxy"><div><small>API PROXY</small><h2>推理 API 代理</h2>
<p>本机客户端使用此入口访问当前模型；支持独立 Key、有效期、请求额度和用量统计。</p><a href="/proxy">打开代理配置页 →</a></div>
<div><code>http://__PROXY_HOST__:__PROXY_PORT__/v1</code><span id="xt-proxy-state">检测中</span></div></section>
"""
CSS = """
.xt-proxy{display:flex;justify-content:space-between;gap:20px;align-items:center;margin:18px 0 0;padding:20px 22px;border:1px solid #345578;border-radius:18px;background:#081225;color:#dcecff;font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
.xt-proxy small{font:700 11px ui-monospace,monospace;letter-spacing:.24em;color:#3ee6ff}.xt-proxy h2{margin:3px 0 4px;font-size:20px}.xt-proxy p{margin:0;color:#8fa3be;font-size:12px}.xt-proxy>div:last-child{text-align:right}.xt-proxy code{display:block;white-space:nowrap;color:#9cf06b}.xt-proxy span{display:block;margin-top:7px;color:#8fa3be;font-size:12px}@media(max-width:700px){.xt-proxy{display:block}.xt-proxy>div:last-child{text-align:left;margin-top:12px}}
.xt-proxy a{display:inline-block;margin-top:8px;color:#3ee6ff;text-decoration:none}.xt-proxy a:hover{text-decoration:underline}
"""
JS = """
(()=>{async function poll(){try{const r=await fetch('/api/proxy/status',{cache:'no-store'}),d=await r.json();document.getElementById('xt-proxy-state').textContent=d.proxy==='online'?'代理在线 · 后端'+(d.backend==='online'?'在线':'离线'):'代理离线'}catch(e){document.getElementById('xt-proxy-state').textContent='代理状态未知'}finally{setTimeout(poll,5000)}}poll()})();
"""


def decorate(page):
    card = CARD_TMPL.replace("__PROXY_HOST__", str(HOST)).replace("__PROXY_PORT__", str(PORT))
    return (page.replace("</style></head>", CSS + "</style></head>", 1)
            .replace("</main><script>", card + "</main><script>", 1)
            .replace("</body></html>", "<script>" + JS + "</script></body></html>", 1))


class ProxyHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def main():
    ensure_proxy_key()
    server = ProxyHTTPServer((HOST, PORT), make_handler())
    print(f"XTier 86KD API proxy on http://{HOST}:{PORT}/v1 (sticky_dp={DP_ROUTER.is_enabled()}, dp_size={DP_ROUTER.dp_size})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()