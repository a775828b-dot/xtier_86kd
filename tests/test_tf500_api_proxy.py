"""Local fake backend tests; never contact the model service."""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import tf500_api_proxy as proxy
import tf500_proxy_keys as keys


class FakeBackend(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = b'{"data":[{"id":"test-model"}]}'
            self.send_response(200)
        else:
            body = b'{"error":"unexpected"}'
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if json.loads(body).get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.2)
            self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
            return
        result = json.dumps({"auth": self.headers.get("Authorization"),
                             "body": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)


class ProxyTests(unittest.TestCase):
    def test_backend_selection_follows_ready_service(self):
        with patch.object(proxy, "BACKEND_PORT", 30000), \
             patch.object(proxy, "_probe_backend", return_value=False):
            proxy.BACKEND_CACHE["at"] = 0
            target = proxy.backend_target()
            self.assertEqual((target["port"], target["online"], target["label"]), (30000, False, "TP8 · 86KD"))
        with patch.object(proxy, "BACKEND_PORT", 30000), \
             patch.object(proxy, "_probe_backend", side_effect=lambda port: port == 30000):
            proxy.BACKEND_CACHE["at"] = 0
            target = proxy.backend_target()
            self.assertEqual((target["port"], target["online"], target["label"]), (30000, True, "TP8 · 86KD"))

    def test_auth_forwarding_and_path_restriction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "proxy-api-key").write_text("public-test\n", encoding="utf-8")
            (root / "backend-api-key").write_text("backend-test\n", encoding="utf-8")
            backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
            with patch.object(proxy, "BACKEND_HOST", "127.0.0.1"), \
                 patch.object(proxy, "BACKEND_PORT", backend.server_port), \
                 patch.object(proxy, "PROXY_KEY_FILE", root / "proxy-api-key"), \
                 patch.object(proxy, "BACKEND_KEY_FILE", root / "backend-api-key"), \
                 patch.object(proxy, "ACCESS_LOG", root / "access.jsonl"), \
                 patch.object(keys, "MASTER_FILE", root / "proxy-api-key"), \
                 patch.object(keys, "KEYS_FILE", root / "keys.json"), \
                 patch.object(keys, "ACCESS_LOG", root / "access.jsonl"):
                server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.make_handler())
                threads = [threading.Thread(target=item.serve_forever, daemon=True)
                           for item in (backend, server)]
                for thread in threads:
                    thread.start()
                try:
                    def call(method, path, body=None, headers=None):
                        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                        conn.request(method, path, body=body, headers=headers or {})
                        response = conn.getresponse()
                        result = response.status, json.loads(response.read())
                        conn.close()
                        return result

                    self.assertEqual(call("GET", "/health")[0], 200)
                    self.assertEqual(call("GET", "/v1/models")[0], 401)
                    auth = {"Authorization": "Bearer public-test"}
                    self.assertEqual(call("GET", "/v1/models", headers=auth)[1]["data"][0]["id"], "test-model")
                    status, result = call("POST", "/v1/chat/completions",
                                          body=b'{"model":"test-model"}', headers=auth)
                    self.assertEqual(status, 200)
                    self.assertEqual(result["auth"], "Bearer backend-test")
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    conn.request("POST", "/v1/chat/completions", body=b'{"stream":true}', headers=auth)
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'data: [DONE]', response.read())
                    conn.close()
                    client = keys.create("test client", 0, 1)
                    client_auth = {"x-api-key": client["key"]}
                    self.assertEqual(call("GET", "/v1/models", headers=client_auth)[0], 200)
                    self.assertEqual(call("GET", "/v1/models", headers=client_auth)[0], 402)
                    self.assertEqual(call("POST", "/flush_cache", headers=auth)[0], 404)
                    self.assertEqual(call("POST", "/v1/%2e%2e/flush_cache", headers=auth)[0], 404)
                    log = (root / "access.jsonl").read_text(encoding="utf-8")
                    self.assertNotIn("public-test", log)
                    self.assertNotIn("backend-test", log)
                    self.assertNotIn(client["key"], log)
                    self.assertIn('"input": 3', log)
                finally:
                    server.shutdown()
                    backend.shutdown()
                    server.server_close()
                    backend.server_close()


if __name__ == "__main__":
    unittest.main()
