"""Comprehensive unit & local mock E2E tests for the Agent benchmark & UI.

Never contacts or modifies any real SGLang inference backend.
"""

from __future__ import annotations

import contextlib
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
import tf500_benchmark as bench
import tf500_nav as nav
import tf500_proxy_keys as keys


class FastServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def _make_mock_sglang_handler(slow_event: threading.Event | None = None):
    class MockSGLang(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args):
            pass

        def _send_json(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                return self._send_json(200, {"data": [{"id": "DeepSeek-V4.1-Flash-1017"}]})
            if self.path in ("/server_info", "/get_server_info"):
                return self._send_json(
                    200,
                    {
                        "context_length": 524288,
                        "max_running_requests": 64,
                        "page_size": 256,
                        "chunked_prefill_size": 32768,
                        "speculative_algorithm": "DSPARK",
                        "speculative_num_draft_tokens": 6,
                        "internal_states": [
                            {
                                "effective_max_running_requests_per_dp": 64,
                                "memory_usage": {"token_capacity": 20_000_000},
                            }
                        ],
                    },
                )
            return self._send_json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/v1/tokenize":
                text = json.dumps(body.get("messages") or [], ensure_ascii=False)
                est = max(600, len(text.encode("utf-8")) // 3)
                return self._send_json(200, {"count": est})

            if self.path == "/v1/chat/completions":
                msgs = body.get("messages") or []
                prompt_tok = max(600, len(json.dumps(msgs, ensure_ascii=False).encode("utf-8")) // 3)
                tool_choice = body.get("tool_choice", "auto")
                max_tok = int(body.get("max_tokens") or 512)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()

                if slow_event is not None and slow_event.is_set():
                    time.sleep(10.0)
                    return

                is_delivery = tool_choice == "none" or (
                    msgs and "代码交付阶段" in str(msgs[-1].get("content") or "")
                )
                if not is_delivery:
                    # Wave 0: emit a realistic write_file + check_python tool call
                    tc_args = json.dumps(
                        {"path": "bench/test_sandbox_suite.py", "content": "def test_ok():\n    assert 1 + 1 == 2\n"}
                    )
                    ev1 = {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_w0",
                                            "type": "function",
                                            "function": {"name": "write_file", "arguments": tc_args},
                                        },
                                        {
                                            "index": 1,
                                            "id": "call_c0",
                                            "type": "function",
                                            "function": {
                                                "name": "check_python",
                                                "arguments": '{"path":"bench/test_sandbox_suite.py"}',
                                            },
                                        },
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": 24,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                    ev2 = {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        "usage": {
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": 36,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                        "sglext": {"spec_tokens_details": {"spec_verify_ct": 10}},
                    }
                    self.wfile.write(f"data: {json.dumps(ev1)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.02)
                    self.wfile.write(f"data: {json.dumps(ev2)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                # Long code delivery wave: stream 6 timed chunks over ~120ms
                cached = int(prompt_tok * 0.92) if len(msgs) > 4 else 0
                total_out = min(max_tok, 360)
                steps = 6
                for step_i in range(1, steps + 1):
                    cum = (total_out * step_i) // steps
                    chunk_ev = {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"def test_case_{step_i}(): assert True\n"},
                                "finish_reason": "stop" if step_i == steps else None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": cum,
                            "prompt_tokens_details": {"cached_tokens": cached},
                        },
                        "sglext": {"spec_tokens_details": {"spec_verify_ct": max(1, cum // 4)}},
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_ev)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if step_i < steps:
                        time.sleep(0.025)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            return self._send_json(404, {"error": "unknown endpoint"})

    return MockSGLang


class BenchmarkUnitTests(unittest.TestCase):
    def test_concurrency_tiers_512k_and_dp_planning(self):
        # All 7 required concurrencies work in single-point context mode
        for c in (1, 4, 8, 16, 32, 64, 100):
            parsed = bench._parse({"mode": "context", "concurrency": str(c), "maximum": "524288", "points": "1"})
            self.assertEqual(parsed["concurrency"], c)
            self.assertEqual(parsed["maximum"], 524288)

        # Invalid concurrencies rejected
        for bad in (0, 2, 3, 24, 50, 128):
            with self.assertRaises(ValueError):
                bench._parse({"mode": "context", "concurrency": str(bad)})

        # Multi-concurrency sweep parses all 7 tiers in sorted order
        sweep = bench._parse(
            {"mode": "concurrency", "concurrencies": "100,1,32,4,64,8,16", "maximum": "524288", "output": "1024"}
        )
        self.assertEqual(sweep["concurrencies"], [1, 4, 8, 16, 32, 64, 100])
        self.assertEqual(sweep["points"], 1)

        # DP=8 capacity math: 8 ranks * 4.5M single_kv = 36M total KV -> 64 concurrency has 8 reqs/rank -> full 512K fits!
        dp8_cap = {
            "model": "DeepSeek-V4.1-Flash-1017",
            "backend_host": "<TF500_IP>",
            "backend_port": 30000,
            "backend_label": "TP8 · 86KD",
            "context": 1048576,
            "dp_ranks": 8,
            "single_kv": 4_500_000,
            "kv": 36_000_000,
            "max_running": 64,
            "page_size": 256,
            "fingerprint": "fp-dp8",
        }
        plan_dp8 = bench.build_plan(sweep, cap=dp8_cap)
        self.assertTrue(plan_dp8["online"])
        step_by_c = {s["concurrency"]: s for s in plan_dp8["steps"]}
        self.assertEqual(step_by_c[64]["target"], 524288)
        self.assertEqual(step_by_c[64]["queued"], 0)
        # C=100 is allowed (not hard-blocked) and reports 36 queued behind max_running=64
        self.assertEqual(step_by_c[100]["queued"], 36)
        self.assertTrue(any("max_running_requests=64" in n for n in plan_dp8["notes"]))

        # Offline preview works cleanly when backend is down
        with patch.object(bench, "_server_info", side_effect=ValueError("后端离线")):
            offline_plan = bench.Benchmark().plan_preview({"mode": "concurrency", "maximum": "524288"})
            self.assertFalse(offline_plan["online"])
            self.assertEqual(offline_plan["concurrencies"], [1, 4, 8, 16, 32, 64, 100])

    def test_overlay_sandbox_isolation_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            sample = base_dir / "sample.py"
            original_code = "def add(a, b):\n    return a + b\n"
            sample.write_text(original_code, encoding="utf-8")
            vfiles = {"pkg/sample.py": sample}

            sb1 = bench.OverlaySandbox(vfiles)
            sb2 = bench.OverlaySandbox(vfiles)
            try:
                self.assertIn("pkg/", sb1.call("list_dir", {"path": "."}))
                self.assertIn("return a + b", sb1.call("read_file", {"path": "pkg/sample.py"}))
                self.assertIn("pkg/sample.py:1:", sb1.call("grep", {"pattern": r"def add", "path": "pkg"}))

                # Unique edit succeeds in sb1 without altering disk or sb2
                res_edit = sb1.call(
                    "edit_file",
                    {"path": "pkg/sample.py", "old_string": "return a + b", "new_string": "return a + b + 1"},
                )
                self.assertIn("已修改", res_edit)
                self.assertIn("return a + b + 1", sb1.call("read_file", {"path": "pkg/sample.py"}))
                self.assertTrue(sb2.call("read_file", {"path": "pkg/sample.py"}).endswith("return a + b"))
                self.assertEqual(sample.read_text(encoding="utf-8"), original_code)

                # Write + syntax check
                sb1.call("write_file", {"path": "pkg/bad.py", "content": "def broken(:\n    pass\n"})
                self.assertIn("语法错误", sb1.call("check_python", {"path": "pkg/bad.py"}))
                sb1.call("write_file", {"path": "pkg/good.py", "content": "def good():\n    return 42\n"})
                self.assertEqual(sb1.call("check_python", {"path": "pkg/good.py"}), "语法检查通过")

                # Path traversal blocked
                self.assertIn("超出沙盒", sb1.call("read_file", {"path": "../../etc/passwd"}))
            finally:
                sb1.close()
                sb2.close()

    def test_aggregate_point_100_sessions_total_and_per_session_math(self):
        rows = []
        # Simulate 100 concurrent sessions: Cold prefill at t=0..2.0s, parallel decode at t=2.0..4.0s
        for lane in range(1, 101):
            rows.append(
                {
                    "ok": True,
                    "lane": lane,
                    "turn": 0,
                    "prompt_tokens": 32768,
                    "cached_tokens": 0,
                    "uncached_tokens": 32768,
                    "output_tokens": 512,
                    "requested_output": 512,
                    "measured_tokens": 500,
                    "ttft_s": 2.0,
                    "decode_s": 0.5,  # Fast 8-GPU decode (<1.0s) must qualify!
                    "wall_s": 2.5,
                    "ind_prefill_tps": 16384.0,
                    "decode_tps": 1000.0 / 20.0,  # 50 tok/s per session
                    "verify_steps": 128,
                    "started": 10.0,
                    "first": 12.0,
                    "last": 12.5,
                    "ended": 12.5,
                    "short_output": False,
                    "tool_calls": ["write_file"],
                }
            )
        agg = bench.aggregate_point(rows, concurrency=100)
        # Total cold uncached = 100 * 32768 = 3,276,800 over 2.0s => 1,638,400 tok/s total; 16,384 tok/s per session
        self.assertEqual(agg["prefill_total_tps"], 1638400.0)
        self.assertEqual(agg["prefill_per_session_tps"], 16384.0)
        # Total measured decode = 100 * 500 = 50,000 over union [12.0, 12.5] = 0.5s => 100,000 tok/s total; 1,000 tok/s per session
        self.assertEqual(agg["decode_total_tps"], 100000.0)
        self.assertEqual(agg["decode_per_session_tps"], 1000.0)
        self.assertEqual(len(agg["lanes"]), 100)
        self.assertEqual(agg["accept_len"], 4.0)

    def test_ui_page_navigation_and_zero_reference_leaks(self):
        raw_html = bench.page_html()
        decorated = nav.decorate(raw_html, "benchmark")
        self.assertIn("bench-form", decorated)
        self.assertIn("xt-bench", decorated)
        for c in (1, 4, 8, 16, 32, 64, 100):
            self.assertIn(f'data-conc="{c}"', decorated)
        self.assertIn('data-max="524288"', decorated)
        self.assertIn("Prefill 总速度", decorated)
        self.assertIn("Prefill 每会话速度", decorated)
        self.assertIn("Decode 总速度", decorated)
        self.assertIn("Decode 每会话速度", decorated)
        self.assertNotIn("new-key-value", decorated)
        self.assertEqual(decorated.count('aria-current="page"'), 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "worker process-group lifecycle uses Linux /proc")
    def test_e2e_high_concurrency_agent_worker_stop_and_clear(self):
        slow_flag = threading.Event()
        sglang_srv = FastServer(("127.0.0.1", 0), _make_mock_sglang_handler(slow_flag))
        t_sg = threading.Thread(target=sglang_srv.serve_forever, daemon=True)
        t_sg.start()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proxy_key = root / "proxy-api-key"
            backend_key = root / "backend-api-key"
            proxy_key.write_text("xt86_unit_proxy_key\n", encoding="utf-8")
            backend_key.write_text("unit_backend_key\n", encoding="utf-8")
            mini_repo = root / "mini_repo"
            (mini_repo / "bench").mkdir(parents=True)
            (mini_repo / "bench" / "agent_bench.py").write_text(
                "\n".join(f"def fn_{i}(x): return x + {i}" for i in range(400)) + "\n",
                encoding="utf-8",
            )

            with contextlib.ExitStack() as stack:
                for target_mod, attr, val in (
                    (proxy, "HOST", "127.0.0.1"),
                    (proxy, "BACKEND_HOST", "127.0.0.1"),
                    (proxy, "BACKEND_PORT", sglang_srv.server_port),
                    (proxy, "PROXY_KEY_FILE", proxy_key),
                    (proxy, "BACKEND_KEY_FILE", backend_key),
                    (proxy, "ACCESS_LOG", root / "access.jsonl"),
                    (keys, "MASTER_FILE", proxy_key),
                    (keys, "KEYS_FILE", root / "keys.json"),
                    (keys, "ACCESS_LOG", root / "access.jsonl"),
                ):
                    stack.enter_context(patch.object(target_mod, attr, val))

                proxy.BACKEND_CACHE["at"] = 0.0
                proxy_srv = proxy.ProxyHTTPServer(("127.0.0.1", 0), proxy.make_handler())
                t_px = threading.Thread(target=proxy_srv.serve_forever, daemon=True)
                t_px.start()
                stack.callback(proxy_srv.server_close)
                stack.callback(proxy_srv.shutdown)

                for attr, val in (
                    ("HOST", "127.0.0.1"),
                    ("PORT", proxy_srv.server_port),
                    ("BACKEND_PORTS", (sglang_srv.server_port,)),
                    ("KEY_FILE", backend_key),
                    ("PROXY_KEY_FILE", proxy_key),
                    ("ROOT", root / "bench_state"),
                    ("XTIER_ROOT", mini_repo),
                    ("REPO_ROOT", mini_repo),
                    ("SGLANG_ROOT", mini_repo),
                    ("TOKENIZER_FILE", root / "no_tok.json"),
                ):
                    stack.enter_context(patch.object(bench, attr, val))

                opts = bench._parse(
                    {
                        "mode": "concurrency",
                        "concurrencies": "1,64,100",
                        "maximum": "4096",
                        "output": "512",
                        "turns": "1",
                        "strategy": "agent_auto",
                    }
                )
                cap = bench._server_info()
                spec = bench.build_plan(opts, cap=cap)
                state_path = root / "bench_state" / "state.json"
                spec_path = root / "bench_state" / "spec.json"
                cancel_path = root / "bench_state" / "cancel"
                res_path = root / "bench_state" / "res.json"
                bench._write(
                    spec_path,
                    {
                        **spec,
                        "id": "test-run-100",
                        "proxy_host": "127.0.0.1",
                        "proxy_port": proxy_srv.server_port,
                        "proxy_key_file": str(proxy_key),
                        "result_file": str(res_path),
                    },
                )
                bench._write(
                    state_path,
                    {
                        "id": "test-run-100",
                        "state": "preparing",
                        "options": opts,
                        "started_at": time.time(),
                        "results": [],
                    },
                )
                bench.run_worker(spec_path, state_path, cancel_path)

                ctrl = bench.Benchmark()
                st = ctrl.status()
                self.assertEqual(st["state"], "done", st.get("message"))
                self.assertEqual(len(st["results"]), 3)
                p1, p64, p100 = st["results"]
                self.assertEqual(p1["concurrency"], 1)
                self.assertEqual(p64["concurrency"], 64)
                self.assertEqual(p100["concurrency"], 100)
                self.assertEqual(len(p100["lanes"]), 100)
                self.assertEqual(p100["success"], 100)
                self.assertIsNotNone(p100["prefill_total_tps"])
                self.assertIsNotNone(p100["prefill_per_session_tps"])
                self.assertIsNotNone(p100["decode_total_tps"])
                self.assertIsNotNone(p100["decode_per_session_tps"])
                self.assertGreater(p100["cache_hit_pct"], 50.0)
                self.assertIn("write_file", p100["lanes"][0]["tools"])

                cleared = ctrl.clear("test-run-100")
                self.assertEqual(cleared["state"], "idle")
                self.assertEqual(cleared["results"], [])

                slow_flag.set()
                cancel_path.unlink(missing_ok=True)
                bench._write(
                    state_path,
                    {
                        "id": "test-stop",
                        "state": "preparing",
                        "options": opts,
                        "started_at": time.time(),
                        "results": [],
                    },
                )
                worker_t = threading.Thread(
                    target=bench.run_worker, args=(spec_path, state_path, cancel_path), daemon=True
                )
                worker_t.start()
                time.sleep(0.35)
                ctrl.stop("test-stop")
                worker_t.join(timeout=5.0)
                st_stopped = ctrl.status()
                self.assertEqual(st_stopped["state"], "stopped")
        sglang_srv.shutdown()
        sglang_srv.server_close()


if __name__ == "__main__":
    unittest.main()
