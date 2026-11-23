#!/usr/bin/env python3
"""Plan C LocalMaxxing Head-to-Head Benchmark Runner for TF500 (<= 64 concurrency).

Measures:
1. Suite 1: All 9 Single-Stream Cold Prefill tiers from LocalMaxxing:
   8K, 16K, 32K, 64K, 128K, 256K, 512K, 1019K (1,043,456), and 1,048,512.
2. Suite 2: Full 8 Contexts (8K..1019K) x 7 Concurrencies (1, 2, 4, 8, 16, 32, 64)
   = 56 Sustained Independent-Prefix Decode cells + Concurrent Cold Prefill speeds.

Saves desensitized incremental results atomically to /data/bench/localmaxxing_c/results_live.json
after every single measurement.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

OUT_DIR = Path("/data/bench/localmaxxing_c")
RESULTS_FILE = OUT_DIR / "results_live.json"
LOG_FILE = OUT_DIR / "bench.log"
GPU_LOG_FILE = OUT_DIR / "gpu_samples.jsonl"
MANIFEST_FILE = OUT_DIR / "launch_manifest.json"
CORPUS_PATH = Path("/data/bench/sglang-corpus.txt")
TOKENIZER_PATH = "/data/models/DeepSeek-V4.1-Flash-1017"
MODEL_NAME = "DeepSeek-V4.1-Flash-1017"

PREFILL_TIERS = [
    ("8K", 8192, 2),
    ("16K", 16384, 2),
    ("32K", 32768, 2),
    ("64K", 65536, 2),
    ("128K", 131072, 2),
    ("256K", 262144, 1),
    ("512K", 524288, 1),
    ("1019K", 1043456, 1),
    ("1,048,512", 1048512, 1),
]

# (label, target_tokens, warmup_batch_size)
DECODE_CONTEXT_TIERS = [
    ("8K", 8192, 64),
    ("16K", 16384, 64),
    ("32K", 32768, 64),
    ("64K", 65536, 64),
    ("128K", 131072, 64),
    ("256K", 262144, 64),
    ("512K", 524288, 32),
    ("1019K", 1043456, 16),
]

MAX_TOKENS_BY_CONC = {
    1: 4096,
    2: 4096,
    4: 3600,
    8: 3200,
    16: 2400,
    32: 1800,
    64: 1600,
}

SUFFIX_INSTRUCTION = (
    "\n\nBased on the codebase above, implement a complete production Python module "
    "with full classes, type annotations, docstrings, edge-case error handling, and "
    "an exhaustive unittest test suite without omitting any implementation details."
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[idx]


class GpuMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.min_free_mib: float | None = None
        self.max_used_mib: float = 0.0
        self.max_power_w: float = 0.0
        self.samples_count = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def snapshot_And_reset_phase(self) -> dict:
        with self.lock:
            return {
                "global_min_free_mib": self.min_free_mib,
                "global_min_free_gib": round(self.min_free_mib / 1024.0, 4) if self.min_free_mib is not None else None,
                "global_max_used_mib": self.max_used_mib,
                "global_max_power_w": round(self.max_power_w, 1),
                "samples": self.samples_count,
            }

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                rows = []
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        rows.append({
                            "gpu": int(parts[0]),
                            "used_mib": float(parts[1]),
                            "free_mib": float(parts[2]),
                            "util_pct": float(parts[3]),
                            "power_w": float(parts[4]),
                        })
                if rows:
                    min_f = min(r["free_mib"] for r in rows)
                    max_u = max(r["used_mib"] for r in rows)
                    tot_p = sum(r["power_w"] for r in rows)
                    with self.lock:
                        if self.min_free_mib is None or min_f < self.min_free_mib:
                            self.min_free_mib = min_f
                        if max_u > self.max_used_mib:
                            self.max_used_mib = max_u
                        if tot_p > self.max_power_w:
                            self.max_power_w = tot_p
                        self.samples_count += 1
                    with open(GPU_LOG_FILE, "a", encoding="utf-8") as gf:
                        gf.write(json.dumps({"ts": round(time.time(), 2), "min_free_mib": min_f, "max_used_mib": max_u, "total_power_w": round(tot_p, 1)}) + "\n")
            except Exception:
                pass
            self.stop_event.wait(3.0)


class BenchRunner:
    def __init__(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        self.host = manifest["env"]["HOST"]
        self.port = int(manifest["env"]["PORT"])
        key_file = manifest["env"].get("API_KEY_FILE") or "/data/sglang-api-key"
        self.api_key = Path(key_file).read_text(encoding="utf-8").strip() if Path(key_file).is_file() else ""
        self.base_url = f"http://{self.host}:{self.port}"
        self.run_tag = uuid.uuid4().hex[:8]
        self.gpu_mon = GpuMonitor()
        self.state = {
            "benchmark": "LocalMaxxing Speed Test Matrix Comparison (Plan C, concurrency <= 64)",
            "reference_url": "https://www.localmaxxing.com/es/inference-benchmarks/cmw41hn3g0c5goa01z8a4ylt4",
            "model": MODEL_NAME,
            "host": "<TF500_IP>",
            "port": self.port,
            "api_key_file": "<API_KEY_FILE>",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server_info": {},
            "prefill_single": [],
            "concurrent_cold_prefill": [],
            "sustained_decode": [],
            "gpu_telemetry": {},
            "status": "starting",
        }
        self._prior_prefill: dict[str, dict] = {}
        if RESULTS_FILE.is_file():
            try:
                prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
                for row in prev.get("prefill_single") or []:
                    if row.get("status") == "Valid" and row.get("context") != "8K":
                        self._prior_prefill[row["context"]] = row
                if prev.get("status") in ("running", "completed", "failed"):
                    log(f"Found prior results_live.json (reusing {len(self._prior_prefill)} clean Suite 1 rows); backing up.")
                    RESULTS_FILE.rename(OUT_DIR / f"results_backup_{int(time.time())}.json")
            except Exception:
                pass
        self._save()

    def _headers(self, dp_rank: int | None = None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if dp_rank is not None:
            h["X-Data-Parallel-Rank"] = str(dp_rank)
            h["x-override-routed-dp-rank"] = str(dp_rank)
        return h

    def _save(self) -> None:
        self.state["gpu_telemetry"] = self.gpu_mon.snapshot_And_reset_phase()
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = RESULTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(RESULTS_FILE)

    def wait_for_server(self, timeout_s: int = 1500) -> None:
        log(f"Waiting for SGLang server on port {self.port} to become ready...")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(f"{self.base_url}/v1/models", headers=self._headers())
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        log("Server /v1/models returned 200 OK!")
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"Server did not become ready within {timeout_s}s")

        # Fetch server_info KV capacities
        for ep in ("/get_server_info", "/server_info"):
            try:
                req = urllib.request.Request(f"{self.base_url}{ep}", headers=self._headers())
                with urllib.request.urlopen(req, timeout=10) as resp:
                    info = json.loads(resp.read().decode("utf-8"))
                states = info.get("internal_states") or []
                mem_states = [(st.get("memory_usage") or st.get("memory") or st) for st in states if isinstance(st, dict)]
                tok_caps = [int(ms.get("token_capacity") or 0) for ms in mem_states if ms.get("token_capacity")]
                swa_caps = [int(ms.get("token_capacity_swa") or 0) for ms in mem_states if ms.get("token_capacity_swa")]
                self.state["server_info"] = {
                    "dp_ranks": len(states),
                    "per_dp_token_capacity": tok_caps[0] if tok_caps else None,
                    "global_kv_tokens": sum(tok_caps) if tok_caps else None,
                    "per_dp_swa_capacity": swa_caps[0] if swa_caps else None,
                    "weight_gib_per_gpu": mem_states[0].get("weight") if mem_states else None,
                    "startup_available_gib": mem_states[0].get("startup_available") if mem_states else None,
                }
                log(f"Server KV info: {json.dumps(self.state['server_info'])}")
                if tok_caps:
                    break
            except Exception as exc:
                log(f"Warning reading {ep}: {exc}")

    def flush_cache(self) -> None:
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/flush_cache",
                    headers=self._headers(),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp.read()
                time.sleep(0.5)
                return
            except Exception as exc:
                log(f"flush_cache attempt {attempt+1} wait: {exc}")
                time.sleep(2.0)

    def verify_calls(self) -> float | None:
        try:
            req = urllib.request.Request(f"{self.base_url}/metrics", headers=self._headers())
            txt = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "replace")
            vals = re.findall(r'^sglang:spec_verify_calls_total\{[^}]*\}\s+([0-9.e+]+)$', txt, re.M)
            return sum(map(float, vals)) if vals else None
        except Exception:
            return None

    def load_tokenizer_and_corpus(self) -> None:
        log("Loading tokenizer and /data/bench/sglang-corpus.txt...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        raw_text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
        base_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
        log(f"Base corpus tokens: {len(base_ids):,}")
        while len(base_ids) < 1_200_000:
            base_ids = base_ids + base_ids
        self.corpus_ids = base_ids[:1_200_000]
        self._body_cache: dict[int, str] = {}
        self._body_lock = threading.Lock()

        # Measure exact server chat-template + prefix + suffix token overhead via 1 live call!
        sample_prefix = self._slot_prefix("CAL000", 0)
        hello_str = "calibration_probe_word " * 16
        hello_ids = self.tokenizer.encode(hello_str, add_special_tokens=False)
        cal_res = self.stream_chat_one(sample_prefix + hello_str + SUFFIX_INSTRUCTION, max_tokens=1, dp_rank=0)
        if not cal_res["ok"]:
            raise RuntimeError(f"Calibration request failed: {cal_res}")
        self.fixed_overhead_tokens = cal_res["prompt_tokens"] - len(hello_ids)
        log(f"Measured exact server chat-template + prefix + suffix overhead: {self.fixed_overhead_tokens} tokens")
        self.flush_cache()

    def _slot_prefix(self, tier_key: str, slot_id: int) -> str:
        return f"[Run {self.run_tag} | Tier {tier_key:<10s} | Slot {slot_id:03d} | Nonce {100000 + slot_id}]\n"

    def get_prompt_for_slot(self, tier_key: str, target_tokens: int, slot_id: int) -> str:
        with self._body_lock:
            if target_tokens not in self._body_cache:
                # Keep only current tier's large string in cache to keep RAM tiny
                self._body_cache.clear()
                # Leave 16 tokens safety margin so 1,048,512 + max_tokens=1 never exceeds 1,048,576!
                margin = 16 if target_tokens >= 1040000 else 4
                body_budget = max(64, target_tokens - self.fixed_overhead_tokens - margin)
                body_text = self.tokenizer.decode(self.corpus_ids[:body_budget])
                # Verify actual token count of decoded/re-encoded string to eliminate BPE boundary drift
                re_ids = self.tokenizer.encode(body_text, add_special_tokens=False)
                drift = len(re_ids) - body_budget
                if abs(drift) > 2:
                    body_budget = max(64, body_budget - drift)
                    body_text = self.tokenizer.decode(self.corpus_ids[:body_budget])
                self._body_cache[target_tokens] = body_text
                log(f"Prepared body for {tier_key.strip()} (target={target_tokens}, body_budget={body_budget})")
            body_str = self._body_cache[target_tokens]
        return self._slot_prefix(tier_key, slot_id) + body_str + SUFFIX_INSTRUCTION

    def stream_chat_one(
        self,
        content: str,
        max_tokens: int,
        dp_rank: int | None = None,
        record_events: bool = False,
        timeout_s: int = 2400,
    ) -> dict:
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_spec_tokens_details": True,
            "return_cached_tokens_details": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if dp_rank is not None:
            body["routed_dp_rank"] = dp_rank
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(dp_rank=dp_rank),
            method="POST",
        )
        started = time.perf_counter()
        first = None
        last = None
        usage = {}
        spec_details = {}
        events: list[tuple[float, str]] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if ev.get("usage"):
                        usage = ev["usage"]
                    sd = (ev.get("sglext") or {}).get("spec_tokens_details")
                    if sd:
                        spec_details = sd
                    for ch in ev.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            last = now
                            if record_events:
                                events.append((now, piece))
            finished = time.perf_counter()
            comp_tokens = int(usage.get("completion_tokens") or 0)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            if first is None or comp_tokens <= 0:
                raise ValueError(f"Empty stream (first={first}, comp_tokens={comp_tokens})")
            return {
                "ok": True,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "output_tokens": comp_tokens,
                "verify_steps": int(spec_details.get("spec_verify_ct") or 0),
                "started": started,
                "first": first,
                "last": last,
                "finished": finished,
                "ttft_s": first - started,
                "tpot_ms": 1000.0 * (last - first) / (comp_tokens - 1) if comp_tokens > 1 else 0.0,
                "events": events,
            }
        except Exception as exc:
            return {
                "ok": False,
                "started": started,
                "finished": time.perf_counter(),
                "error": str(exc)[:400],
            }

    def run_suite1_single_prefill(self) -> None:
        log("=== SUITE 1: Single-Stream Cold Prefill Ladder (9 Tiers) ===")
        for label, target_toks, passes in PREFILL_TIERS:
            if label in self._prior_prefill:
                cached_entry = self._prior_prefill[label]
                log(
                    f"  [REUSED] Prefill {label}: prompt={cached_entry['prompt_tokens']:,} "
                    f"ttft={cached_entry['ttft_ms']:.1f}ms speed={cached_entry['prefill_tok_s']:,.1f} tok/s"
                )
                self.state["prefill_single"].append(cached_entry)
                self._save()
                continue
            pass_rows = []
            for p_idx in range(passes):
                self.flush_cache()
                prompt = self.get_prompt_for_slot(f"P1-{label}-{p_idx}", target_toks, p_idx)
                res = self.stream_chat_one(prompt, max_tokens=1, dp_rank=0, record_events=False)
                if not res["ok"]:
                    log(f"  [WARN] Prefill {label} pass {p_idx} error: {res['error']}")
                    continue
                tps = res["prompt_tokens"] / res["ttft_s"]
                log(
                    f"  Prefill {label} pass {p_idx+1}/{passes}: "
                    f"prompt={res['prompt_tokens']:,} cached={res['cached_tokens']} "
                    f"ttft={res['ttft_s']*1000:.1f}ms speed={tps:,.1f} tok/s"
                )
                pass_rows.append(res)

            if pass_rows:
                med_ttft_s = statistics.median(r["ttft_s"] for r in pass_rows)
                med_prompt = int(statistics.median(r["prompt_tokens"] for r in pass_rows))
                med_tps = round(med_prompt / med_ttft_s, 1)
                entry = {
                    "context": label,
                    "target_tokens": target_toks,
                    "prompt_tokens": med_prompt,
                    "concurrency": 1,
                    "status": "Valid",
                    "prefill_tok_s": med_tps,
                    "ttft_ms": round(med_ttft_s * 1000.0, 1),
                    "completed_requests": len(pass_rows),
                    "cached_tokens": max(r["cached_tokens"] for r in pass_rows),
                }
            else:
                entry = {
                    "context": label,
                    "target_tokens": target_toks,
                    "concurrency": 1,
                    "status": "Error",
                }
            self.state["prefill_single"].append(entry)
            self._save()

    def _warm_slot_range(self, tier_label: str, target_toks: int, start_slot: int, end_slot: int) -> dict:
        slots = list(range(start_slot, end_slot))
        conc = len(slots)
        log(f"  Warming {tier_label} slots [{start_slot}..{end_slot-1}] (batch_conc={conc})...")
        t_batch_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [
                pool.submit(
                    self.stream_chat_one,
                    self.get_prompt_for_slot(f"D-{tier_label}", target_toks, s),
                    1,
                    s % 8,
                    False,
                    2400,
                )
                for s in slots
            ]
            results = [f.result() for f in futs]
        good = [r for r in results if r["ok"]]
        errs = [r["error"] for r in results if not r["ok"]]
        if not good:
            raise RuntimeError(f"Warmup failed for {tier_label} [{start_slot}..{end_slot-1}]: {errs[:2]}")
        wall_s = max(r["first"] for r in good) - min(r["started"] for r in results)
        tot_prompt = sum(r["prompt_tokens"] for r in good)
        tot_cached = sum(r["cached_tokens"] for r in good)
        tps = round(tot_prompt / wall_s, 1)
        p50_ttft = round(statistics.median(r["ttft_s"] for r in good), 2)
        log(
            f"  Warmed {len(good)}/{conc} slots for {tier_label}: "
            f"total_prompt={tot_prompt:,} cached={tot_cached:,} "
            f"cold_prefill_tps={tps:,.1f} tok/s p50_ttft={p50_ttft}s (wall={time.perf_counter()-t_batch_start:.1f}s)"
        )
        return {
            "context": tier_label,
            "batch_concurrency": conc,
            "slots": [start_slot, end_slot - 1],
            "ok_count": len(good),
            "total_prompt_tokens": tot_prompt,
            "total_cached_tokens": tot_cached,
            "cold_prefill_tok_s": tps,
            "p50_cold_ttft_s": p50_ttft,
            "wall_s": round(wall_s, 2),
            "errors": errs[:3],
        }

    def _measure_sustained_decode_cell(self, tier_label: str, target_toks: int, conc: int) -> dict:
        max_toks = MAX_TOKENS_BY_CONC[conc]
        slots = list(range(conc))
        steps_before = self.verify_calls()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [
                pool.submit(
                    self.stream_chat_one,
                    self.get_prompt_for_slot(f"D-{tier_label}", target_toks, s),
                    max_toks,
                    s % 8,
                    True,
                    1800,
                )
                for s in slots
            ]
            results = [f.result() for f in futs]
        steps_after = self.verify_calls()
        good = [r for r in results if r["ok"]]
        errs = [r["error"] for r in results if not r["ok"]]
        if len(good) < conc:
            log(f"  [WARN] {tier_label} x {conc}: {len(good)}/{conc} ok, errs={errs[:2]}")
        if not good:
            return {
                "context": tier_label,
                "target_tokens": target_toks,
                "concurrency": conc,
                "status": "Error",
                "errors": errs[:3],
            }

        tot_out = sum(r["output_tokens"] for r in good)
        tot_in = sum(r["prompt_tokens"] for r in good)
        tot_cached = sum(r["cached_tokens"] for r in good)
        cache_hit_pct = round(100.0 * tot_cached / tot_in, 2) if tot_in else 0.0

        min_first = min(r["first"] for r in good)
        max_last = max(r["last"] for r in good)
        decode_window_s = max(0.001, max_last - min_first)
        decode_window_tps = round(tot_out / decode_window_s, 1)

        e2e_s = max(0.001, max(r["finished"] for r in results) - min(r["started"] for r in results))
        e2e_tps = round(tot_out / e2e_s, 1)

        # Strict 100%-active overlap window [max(first), min(last)]
        overlap_start = max(r["first"] for r in good)
        overlap_end = min(r["last"] for r in good)
        overlap_s = overlap_end - overlap_start
        if len(good) > 1 and overlap_s >= 3.0:
            overlap_toks = 0
            for r in good:
                text_in_ov = "".join(piece for ts, piece in r["events"] if overlap_start < ts <= overlap_end)
                if text_in_ov:
                    overlap_toks += len(self.tokenizer.encode(text_in_ov, add_special_tokens=False))
            sustained_overlap_tps = round(overlap_toks / overlap_s, 1)
        else:
            overlap_s = decode_window_s
            sustained_overlap_tps = decode_window_tps

        # Primary headline total_tok_s: decode_window_tps (consistent with bench_serving.py & steady window)
        per_req_rates = [
            (r["output_tokens"] - 1) / (r["last"] - r["first"])
            for r in good
            if r["last"] > r["first"] and r["output_tokens"] > 1
        ]
        med_per_req_tps = round(statistics.median(per_req_rates), 1) if per_req_rates else 0.0
        med_tbt_ms = round(statistics.median(r["tpot_ms"] for r in good), 2)
        p95_tbt_ms = round(percentile([r["tpot_ms"] for r in good], 0.95), 2)
        med_ttft_ms = round(statistics.median(r["ttft_s"] * 1000.0 for r in good), 1)

        verify_steps = sum(r["verify_steps"] for r in good)
        if not verify_steps and steps_before is not None and steps_after is not None:
            verify_steps = max(0.0, steps_after - steps_before)
        accept_len = round(tot_out / verify_steps, 3) if verify_steps else None

        # Cleanup heavy event lists immediately
        for r in results:
            r.pop("events", None)

        row = {
            "context": tier_label,
            "target_tokens": target_toks,
            "concurrency": conc,
            "status": "Valid" if len(good) == conc else f"Partial ({len(good)}/{conc})",
            "total_tok_s": decode_window_tps,
            "sustained_overlap_tok_s": sustained_overlap_tps,
            "e2e_output_tok_s": e2e_tps,
            "per_request_tok_s": med_per_req_tps,
            "ttft_ms": med_ttft_ms,
            "tbt_ms": med_tbt_ms,
            "p95_tbt_ms": p95_tbt_ms,
            "cache_hit_pct": cache_hit_pct,
            "output_tokens": tot_out,
            "measured_for_s": round(decode_window_s, 1),
            "overlap_window_s": round(overlap_s, 1),
            "completed_requests": len(good),
            "accept_tokens_per_step": accept_len,
        }
        log(
            f"  [DECODE] {tier_label:>5s} x {conc:<2d} | "
            f"total={row['total_tok_s']:>7,.1f} tok/s (overlap={sustained_overlap_tps:>7,.1f}) | "
            f"per_req={med_per_req_tps:>5.1f} | TBT={med_tbt_ms:>5.2f}ms | "
            f"TTFT={med_ttft_ms:>6.1f}ms | cache={cache_hit_pct:>5.1f}% | win={decode_window_s:.1f}s"
        )
        return row

    def run_suite2_sustained_matrix(self) -> None:
        log("=== SUITE 2: Sustained Independent Decode Matrix (8 Contexts x [1,2,4,8,16,32,64]) ===")
        for tier_label, target_toks, warm_batch in DECODE_CONTEXT_TIERS:
            log(f"--- Context Tier: {tier_label} ({target_toks:,} tokens) ---")
            self.flush_cache()

            if warm_batch == 64:
                warm_info = self._warm_slot_range(tier_label, target_toks, 0, 64)
                self.state["concurrent_cold_prefill"].append(warm_info)
                self._save()
                # Run c=64 FIRST on pristine freshly warmed 64 prefixes, then 32..1
                for conc in [64, 32, 16, 8, 4, 2, 1]:
                    cell = self._measure_sustained_decode_cell(tier_label, target_toks, conc)
                    self.state["sustained_decode"].append(cell)
                    self._save()
            elif warm_batch == 32:
                # 512K: warm 0..31 -> test 32,16,8,4,2,1 -> warm 32..63 -> test 64!
                w1 = self._warm_slot_range(tier_label, target_toks, 0, 32)
                self.state["concurrent_cold_prefill"].append(w1)
                self._save()
                for conc in [32, 16, 8, 4, 2, 1]:
                    cell = self._measure_sustained_decode_cell(tier_label, target_toks, conc)
                    self.state["sustained_decode"].append(cell)
                    self._save()
                w2 = self._warm_slot_range(tier_label, target_toks, 32, 64)
                self.state["concurrent_cold_prefill"].append(w2)
                self._save()
                cell64 = self._measure_sustained_decode_cell(tier_label, target_toks, 64)
                self.state["sustained_decode"].append(cell64)
                self._save()
            elif warm_batch == 16:
                # 1019K: warm 0..15 -> test 16,8,4,2,1 -> warm 16..31 -> test 32 -> warm 32..47 & 48..63 -> test 64!
                w1 = self._warm_slot_range(tier_label, target_toks, 0, 16)
                self.state["concurrent_cold_prefill"].append(w1)
                self._save()
                for conc in [16, 8, 4, 2, 1]:
                    cell = self._measure_sustained_decode_cell(tier_label, target_toks, conc)
                    self.state["sustained_decode"].append(cell)
                    self._save()

                w2 = self._warm_slot_range(tier_label, target_toks, 16, 32)
                self.state["concurrent_cold_prefill"].append(w2)
                self._save()
                cell32 = self._measure_sustained_decode_cell(tier_label, target_toks, 32)
                self.state["sustained_decode"].append(cell32)
                self._save()

                w3 = self._warm_slot_range(tier_label, target_toks, 32, 48)
                self.state["concurrent_cold_prefill"].append(w3)
                self._save()
                w4 = self._warm_slot_range(tier_label, target_toks, 48, 64)
                self.state["concurrent_cold_prefill"].append(w4)
                self._save()
                cell64 = self._measure_sustained_decode_cell(tier_label, target_toks, 64)
                self.state["sustained_decode"].append(cell64)
                self._save()

    def run_all(self) -> None:
        self.gpu_mon.start()
        try:
            self.wait_for_server()
            self.load_tokenizer_and_corpus()
            self.state["status"] = "running"
            self._save()
            self.run_suite1_single_prefill()
            self.run_suite2_sustained_matrix()
            self.state["status"] = "completed"
            self.state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._save()
            log("=== ALL PLAN C BENCHMARKS COMPLETED SUCCESSFULLY! ===")
        except Exception as exc:
            self.state["status"] = "failed"
            self.state["fatal_error"] = repr(exc)
            self._save()
            log(f"FATAL ERROR: {exc!r}")
            raise
        finally:
            self.gpu_mon.stop()


if __name__ == "__main__":
    BenchRunner().run_all()
