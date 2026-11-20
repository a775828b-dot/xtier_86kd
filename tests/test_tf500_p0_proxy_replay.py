"""Offline unit and end-to-end tests for P0 (N1 sticky DP router + N2 observability & replay)."""

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "deploy"))
sys.path.insert(0, str(ROOT_DIR / "bench"))

import agent_replay
import tf500_api_proxy as proxy
import tf500_dp_router as dp_router
import tf500_proxy_keys as keys


class FakeDPRadixBackend(BaseHTTPRequestHandler):
    """Simulates an 8-rank DP SGLang server with isolated per-rank prefix caches."""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = b'{"data":[{"id":"DeepSeek-V4.1-Flash-1017"}]}'
            self.send_response(200)
        else:
            body = b'{"error":"not found"}'
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw)
        hdr_a = self.headers.get("X-Data-Parallel-Rank")
        hdr_b = self.headers.get("x-override-routed-dp-rank")
        state = self.server.dp_state
        with state["lock"]:
            state["seen_headers"].append((hdr_a, hdr_b))
            if hdr_a is not None and hdr_a.isdigit():
                rank = int(hdr_a)
            elif hdr_b is not None and hdr_b.isdigit():
                rank = int(hdr_b)
            else:
                rank = state["rr"] % 8
                state["rr"] += 1

            messages = payload.get("messages") or []
            # Compute character-based pseudo-tokens & simulate per-rank RadixCache prefix match
            msg_keys = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages]
            rank_cache = state["caches"][rank]
            total_tok = max(16, sum(len(s) for s in msg_keys) // 4)
            cached_tok = 0
            prefix_acc = []
            for s in msg_keys:
                prefix_acc.append(s)
                pk = tuple(prefix_acc)
                if pk in rank_cache:
                    cached_tok = max(cached_tok, sum(len(x) for x in prefix_acc) // 4)
                rank_cache.add(pk)
            cached_tok = min(cached_tok, total_tok)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # First empty role chunk (must NOT count as TTFT)
            self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.02)
            # First real content chunk
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Plan step 1: "}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.04)
            # Subsequent chunks
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"optimize routing."}}]}\n\n')
            self.wfile.flush()
            usage_evt = {
                "choices": [],
                "usage": {
                    "prompt_tokens": total_tok,
                    "completion_tokens": 11,
                    "prompt_tokens_details": {"cached_tokens": cached_tok},
                },
            }
            self.wfile.write(("data: " + json.dumps(usage_evt) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        out = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": total_tok,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": cached_tok},
            },
            "rank": rank,
            "hdr_a": hdr_a,
            "hdr_b": hdr_b,
        }
        body = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StickyRouterUnitTests(unittest.TestCase):
    def test_anti_system_hotspot_and_multi_turn_tool_stickiness(self):
        router = dp_router.PrefixStickyDPRouter(dp_size=8, enabled=True)
        tools = agent_replay.SHARED_TOOLS
        sys_msg = {"role": "system", "content": agent_replay.SHARED_SYSTEM_PROMPT}

        # 16 concurrent new sessions sharing identical system + tools
        lane_ranks = {}
        turn0_leases = []
        for lane in range(16):
            payload = {
                "model": "ds41",
                "tools": tools,
                "messages": [sys_msg, {"role": "user", "content": f"unique prompt for lane {lane}"}],
            }
            lease = router.acquire("/v1/chat/completions", {}, payload, raw_len=40000)
            self.assertEqual(lease.reason, "least_loaded")
            self.assertTrue(lease.inject_headers)
            lane_ranks[lane] = lease.rank
            turn0_leases.append(lease)

        # Must spread evenly across all 8 ranks (2 per rank), zero hotspot!
        counts = [sum(1 for r in lane_ranks.values() if r == rk) for rk in range(8)]
        self.assertEqual(counts, [2] * 8)
        for lease in turn0_leases:
            router.release(lease, ok=True)

        # Turn 1 with tool_calls (content: None) + role=tool must stick 100% to Turn 0 rank
        for lane in range(16):
            payload = {
                "model": "ds41",
                "tools": tools,
                "messages": [
                    sys_msg,
                    {"role": "user", "content": f"unique prompt for lane {lane}"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{lane}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                        }],
                    },
                    {"role": "tool", "tool_call_id": f"call_{lane}", "name": "read_file", "content": "file body"},
                ],
            }
            lease = router.acquire("/v1/chat/completions", {}, payload, raw_len=52000)
            self.assertEqual(lease.reason, "sticky_hit")
            self.assertEqual(lease.rank, lane_ranks[lane])
            router.release(lease, ok=True)

        # Turn 2 continuing after tool_calls must also stick 100%
        for lane in range(16):
            payload = {
                "model": "ds41",
                "tools": tools,
                "messages": [
                    sys_msg,
                    {"role": "user", "content": f"unique prompt for lane {lane}"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{lane}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                        }],
                    },
                    {"role": "tool", "tool_call_id": f"call_{lane}", "name": "read_file", "content": "file body"},
                    {"role": "assistant", "content": "Next analysis"},
                    {"role": "user", "content": "Proceed"},
                ],
            }
            lease = router.acquire("/v1/chat/completions", {}, payload, raw_len=60000)
            self.assertEqual(lease.reason, "sticky_hit")
            self.assertEqual(lease.rank, lane_ranks[lane])
            router.release(lease, ok=True)

    def test_overload_fallback_skew_and_hard_cap(self):
        router = dp_router.PrefixStickyDPRouter(dp_size=4, max_inflight=7, max_skew=4, enabled=True)
        p0 = {"messages": [{"role": "user", "content": "session-alpha"}]}
        l0 = router.acquire("/v1/chat/completions", {}, p0, raw_len=1000)
        target_rank = l0.rank
        router.release(l0, ok=True)

        # Pin 4 inflight requests on target_rank via client_override -> skew becomes 4 - 0 = 4
        pinned = []
        for i in range(4):
            pinned.append(
                router.acquire(
                    "/v1/chat/completions",
                    {"X-Data-Parallel-Rank": str(target_rank)},
                    {"messages": [{"role": "user", "content": f"busy-{i}"}]},
                    raw_len=1000,
                )
            )
        p1 = {"messages": [{"role": "user", "content": "session-alpha"}, {"role": "assistant", "content": "hi"}]}
        l1 = router.acquire("/v1/chat/completions", {}, p1, raw_len=1200)
        self.assertEqual(l1.reason, "overload_fallback")
        self.assertNotEqual(l1.rank, target_rank)
        router.release(l1, ok=True)
        for p in pinned:
            router.release(p, ok=True)

    def test_ttl_expiry_and_lru_eviction(self):
        now = [100.0]
        router = dp_router.PrefixStickyDPRouter(
            dp_size=4, max_entries=16, ttl_s=10.0, enabled=True, clock=lambda: now[0]
        )
        p = {"messages": [{"role": "user", "content": "ttl-test"}]}
        l0 = router.acquire("/v1/chat/completions", {}, p, raw_len=500)
        router.release(l0, ok=True)

        # Before expiry -> sticky hit
        now[0] = 105.0
        l1 = router.acquire("/v1/chat/completions", {}, p, raw_len=500)
        self.assertEqual(l1.reason, "sticky_hit")
        router.release(l1, ok=True)

        # After expiry -> miss
        now[0] = 120.0
        l2 = router.acquire("/v1/chat/completions", {}, p, raw_len=500)
        self.assertEqual(l2.reason, "least_loaded")
        router.release(l2, ok=True)

        # LRU cap check
        for i in range(40):
            li = router.acquire("/v1/chat/completions", {}, {"messages": [{"role": "user", "content": f"u-{i}"}]}, 400)
            router.release(li, ok=True)
        self.assertLessEqual(router.stats()["lru_size"], 16)

    def test_client_override_and_env_disable(self):
        with patch.dict(os.environ, {"TF500_PROXY_STICKY_DP": "0"}):
            router = dp_router.PrefixStickyDPRouter(dp_size=8)
            self.assertFalse(router.is_enabled())
            l_off = router.acquire("/v1/chat/completions", {}, {"messages": [{"role": "user", "content": "a"}]}, 200)
            self.assertEqual(l_off.reason, "disabled")
            self.assertIsNone(l_off.rank)
            self.assertFalse(l_off.inject_headers)

            # Client override still respected even when sticky is off
            l_ovr = router.acquire(
                "/v1/chat/completions",
                {"x-override-routed-dp-rank": "5"},
                {"messages": [{"role": "user", "content": "a"}]},
                200,
            )
            self.assertEqual(l_ovr.reason, "client_override")
            self.assertEqual(l_ovr.rank, 5)
            self.assertFalse(l_ovr.inject_headers)
            router.release(l_ovr, ok=True)


class ProxyAndReplayE2ETests(unittest.TestCase):
    def test_e2e_sticky_proxy_observability_and_agent_replay(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "proxy-api-key").write_text("secret-proxy-key\n", encoding="utf-8")
            (root / "backend-api-key").write_text("secret-backend-key\n", encoding="utf-8")

            backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeDPRadixBackend)
            backend.dp_state = {
                "lock": threading.Lock(),
                "rr": 0,
                "caches": [set() for _ in range(8)],
                "seen_headers": [],
            }
            fresh_router = dp_router.PrefixStickyDPRouter(dp_size=8, enabled=True)

            with patch.object(proxy, "BACKEND_HOST", "127.0.0.1"), \
                 patch.object(proxy, "BACKEND_PORT", backend.server_port), \
                 patch.object(proxy, "PROXY_KEY_FILE", root / "proxy-api-key"), \
                 patch.object(proxy, "BACKEND_KEY_FILE", root / "backend-api-key"), \
                 patch.object(proxy, "ACCESS_LOG", root / "access.jsonl"), \
                 patch.object(proxy, "DP_ROUTER", fresh_router), \
                 patch.object(keys, "MASTER_FILE", root / "proxy-api-key"), \
                 patch.object(keys, "KEYS_FILE", root / "keys.json"), \
                 patch.object(keys, "ACCESS_LOG", root / "access.jsonl"):
                proxy.BACKEND_CACHE["at"] = 0
                server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.make_handler())
                for s in (backend, server):
                    threading.Thread(target=s.serve_forever, daemon=True).start()
                try:
                    # 1. Run 8-concurrency 3-turn Agent replay through proxy
                    res = agent_replay.run_replay(
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        api_key="secret-proxy-key",
                        model="DeepSeek-V4.1-Flash-1017",
                        concurrency=8,
                        init_tokens=512,
                        turn_append_tokens=128,
                        turns=3,
                        max_tokens=64,
                        temperature=0.0,
                        think_min_s=0.0,
                        think_max_s=0.0,
                        stagger_s=0.0,
                        timeout_s=10.0,
                        use_tool_role=True,
                        run_salt="e2e_test",
                    )
                    summary = res["summary"]
                    self.assertEqual(summary["failed_requests"], 0)
                    self.assertEqual(summary["ok_requests"], 24)
                    # Every lane sticks to its Turn 0 rank across Turn 1 and Turn 2
                    self.assertEqual(summary["lane_sticky_consistency_pct"], 100.0)
                    # All 8 DP ranks used (anti-system hotspot verified!)
                    self.assertEqual(len(summary["dp_rank_counts"]), 8)
                    # Warm multi-turn prefix cache hit rate high
                    self.assertGreaterEqual(summary["warm_cache_hit_pct"], 75.0)
                    # Both routing headers sent to backend
                    for hdr_a, hdr_b in backend.dp_state["seen_headers"]:
                        self.assertIsNotNone(hdr_a)
                        self.assertEqual(hdr_a, hdr_b)

                    # 2. Check /health includes dp_router stats
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    conn.request("GET", "/health")
                    health = json.loads(conn.getresponse().read())
                    conn.close()
                    self.assertTrue(health["dp_router"]["enabled"])
                    self.assertEqual(health["dp_router"]["hits"], 16)
                    self.assertEqual(health["dp_router"]["misses"], 8)
                    self.assertEqual(health["dp_router"]["inflight_reqs"], [0] * 8)

                    # 3. Verify proxy-access.jsonl observability fields & privacy
                    raw_log = (root / "access.jsonl").read_text(encoding="utf-8")
                    self.assertNotIn("secret-proxy-key", raw_log)
                    self.assertNotIn("secret-backend-key", raw_log)
                    self.assertNotIn("context_block=", raw_log)

                    log_report = agent_replay.analyze_proxy_log(str(root / "access.jsonl"))
                    self.assertEqual(log_report["ok_requests"], 24)
                    self.assertEqual(log_report["rows_with_p0_obs"], 24)
                    self.assertGreater(log_report["total_cached_tokens"], 0)
                    self.assertIsNotNone(log_report["ttft_p50_s"])
                    self.assertIsNotNone(log_report["decode_tps_p50"])
                    self.assertEqual(log_report["temperature_zero_pct"], 100.0)
                    self.assertEqual(log_report["dp_route_counts"].get("sticky_hit"), 16)
                    self.assertEqual(log_report["dp_route_counts"].get("least_loaded"), 8)
                finally:
                    server.shutdown()
                    backend.shutdown()
                    server.server_close()
                    backend.server_close()


if __name__ == "__main__":
    unittest.main()
