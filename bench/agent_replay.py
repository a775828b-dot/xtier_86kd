#!/usr/bin/env python3
"""Multi-turn Agent replay benchmark & proxy access-log analyzer (P0-N2).

Examples:
  # 1) 16-concurrency multi-turn Agent replay against local TF500 API proxy (:8000)
  python bench/agent_replay.py --concurrency 16 --init-tokens 40960 --turn-append-tokens 4096 --turns 4 --max-tokens 512

  # 2) Quick 4-concurrency smoke replay
  python bench/agent_replay.py --concurrency 4 --init-tokens 4096 --turn-append-tokens 1024 --turns 3 --max-tokens 128 --think-min-s 0 --think-max-s 0

  # 3) Analyze real production traffic from proxy-access.jsonl
  python bench/agent_replay.py --analyze-log /data/xtier-86kd-console/proxy-access.jsonl --since-minutes 60
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import http.client
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from urllib.parse import urlsplit

DEFAULT_BASE_URL = os.environ.get("TF500_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"
DEFAULT_KEY_FILE = "/data/xtier-86kd-console/proxy-api-key"

SHARED_SYSTEM_PROMPT = (
    "You are a senior systems performance engineer working inside a large "
    "production codebase. Analyze file excerpts, tool outputs, and trace logs "
    "carefully, reason step by step, and produce concise, concrete code or "
    "diagnosis without repeating large raw inputs verbatim."
)

SHARED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of lines from a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def build_synthetic_text(target_tokens: int, seed_tag: str) -> str:
    """Generate deterministic code/log-like prose approximating target_tokens (~3.8 chars/token)."""
    if target_tokens <= 0:
        return ""
    target_chars = max(64, int(target_tokens * 3.8))
    rng = random.Random(seed_tag)
    vocab = [
        "scheduler", "radix_cache", "prefill_chunk", "decode_step", "kv_pool",
        "numa_node", "dp_rank", "all_gather", "cuda_graph", "engram_table",
        "tensor_core", "latency_ms", "throughput", "batch_size", "token_budget",
        "attention_head", "sliding_window", "speculative_draft", "verify_step",
        "memory_fraction", "prefix_hash", "inflight_queue", "router_lease",
        "tpot_p95", "ttft_cold", "cache_hit", "worker_slot", "stream_delta",
    ]
    lines = [f"# context_block={seed_tag}"]
    total = len(lines[0]) + 1
    idx = 0
    while total < target_chars:
        w = [rng.choice(vocab) for _ in range(10)]
        line = (
            f"def fn_{seed_tag}_{idx}(x_{idx}): "
            f"return {w[0]}({w[1]}, {w[2]}, {w[3]}) + {w[4]}({w[5]}, {w[6]}) "
            f"# {w[7]} {w[8]} {w[9]} idx={idx}"
        )
        lines.append(line)
        total += len(line) + 1
        idx += 1
    return "\n".join(lines)[:target_chars]


def _extract_delta_text(obj: dict) -> str:
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            delta = c0.get("delta")
            if isinstance(delta, dict):
                parts = []
                for key in ("content", "reasoning_content", "reasoning"):
                    v = delta.get(key)
                    if isinstance(v, str) and v:
                        parts.append(v)
                if parts:
                    return "".join(parts)
                if delta.get("tool_calls"):
                    return "[tool_call]"
            txt = c0.get("text")
            if isinstance(txt, str) and txt:
                return txt
    return ""


def send_streaming_chat(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout_s: float = 600.0,
    extra_headers: dict | None = None,
) -> dict:
    parsed = urlsplit(base_url.rstrip("/") + "/chat/completions")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout_s)

    started = time.monotonic()
    first_token_at = None
    last_token_at = None
    chunks_text = []
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    status = 0
    dp_rank_hdr = None
    dp_route_hdr = None
    error = None

    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        dp_rank_hdr = resp.getheader("X-TF500-DP-Rank")
        dp_route_hdr = resp.getheader("X-TF500-DP-Route")
        if status != 200:
            raw_err = resp.read(2048).decode("utf-8", "replace")
            error = f"HTTP {status}: {raw_err[:300]}"
        else:
            ctype = resp.getheader("Content-Type", "")
            if "text/event-stream" in ctype:
                buf = b""
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    now = time.monotonic()
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.rstrip(b"\r")
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:]
                        if not data or data == b"[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        txt = _extract_delta_text(obj)
                        if txt:
                            if first_token_at is None:
                                first_token_at = now
                            last_token_at = now
                            if len(chunks_text) < 256:
                                chunks_text.append(txt)
                        usage = obj.get("usage")
                        if isinstance(usage, dict) and usage:
                            prompt_tokens = int(usage.get("prompt_tokens") or 0)
                            completion_tokens = int(usage.get("completion_tokens") or 0)
                            details = usage.get("prompt_tokens_details") or {}
                            if isinstance(details, dict) and details.get("cached_tokens") is not None:
                                cached_tokens = int(details.get("cached_tokens") or 0)
                            elif usage.get("cached_tokens") is not None:
                                cached_tokens = int(usage.get("cached_tokens") or 0)
            else:
                raw = resp.read()
                now = time.monotonic()
                first_token_at = now
                last_token_at = now
                obj = json.loads(raw)
                usage = obj.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                details = usage.get("prompt_tokens_details") or {}
                if isinstance(details, dict) and details.get("cached_tokens") is not None:
                    cached_tokens = int(details.get("cached_tokens") or 0)
                choices = obj.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    if isinstance(msg.get("content"), str):
                        chunks_text.append(msg["content"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()

    ended = time.monotonic()
    wall_s = ended - started
    ttft_s = (first_token_at - started) if first_token_at is not None else None
    decode_s = (
        (last_token_at - first_token_at)
        if (first_token_at is not None and last_token_at is not None and last_token_at > first_token_at)
        else None
    )
    decode_tps = ((completion_tokens - 1) / decode_s) if (decode_s and decode_s > 0.001 and completion_tokens >= 2) else None
    tpot_ms = (1000.0 * decode_s / (completion_tokens - 1)) if (decode_s and completion_tokens >= 2) else None
    dp_rank = int(dp_rank_hdr) if (dp_rank_hdr is not None and dp_rank_hdr.isdigit()) else None

    return {
        "ok": error is None and status == 200,
        "status": status,
        "error": error,
        "started": started,
        "first_token_at": first_token_at,
        "last_token_at": last_token_at,
        "ended": ended,
        "wall_s": round(wall_s, 4),
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "decode_s": round(decode_s, 4) if decode_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "decode_tps": round(decode_tps, 2) if decode_tps is not None else None,
        "tpot_ms": round(tpot_ms, 2) if tpot_ms is not None else None,
        "dp_rank": dp_rank,
        "dp_route": dp_route_hdr,
        "reply_preview": "".join(chunks_text)[:1200],
    }


def run_lane(
    lane_id: int,
    run_salt: str,
    base_url: str,
    api_key: str,
    model: str,
    init_tokens: int,
    turn_append_tokens: int,
    turns: int,
    max_tokens: int,
    temperature: float,
    think_min_s: float,
    think_max_s: float,
    stagger_s: float,
    timeout_s: float,
    use_tool_role: bool,
) -> list[dict]:
    if stagger_s > 0 and lane_id > 0:
        time.sleep(stagger_s * lane_id)

    rng = random.Random(f"{run_salt}-lane-{lane_id}")
    init_ctx = build_synthetic_text(init_tokens, f"{run_salt}_L{lane_id}_T0")
    messages = [
        {"role": "system", "content": SHARED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"[Session {run_salt}-L{lane_id}] Inspect the repository module below "
                f"and provide a structured optimization plan with concrete code steps:\n\n{init_ctx}"
            ),
        },
    ]

    rows = []
    for turn in range(turns):
        payload = {
            "model": model,
            "messages": messages,
            "tools": SHARED_TOOLS,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        res = send_streaming_chat(base_url, api_key, payload, timeout_s=timeout_s)
        res["lane"] = lane_id
        res["turn"] = turn
        rows.append(res)
        if not res["ok"]:
            break

        if turn + 1 < turns:
            if think_max_s > 0:
                delay = rng.uniform(min(think_min_s, think_max_s), max(think_min_s, think_max_s))
                if delay > 0:
                    time.sleep(delay)
            tool_out = build_synthetic_text(turn_append_tokens, f"{run_salt}_L{lane_id}_T{turn + 1}")
            reply_text = res["reply_preview"] or f"Analyzed turn {turn} for lane {lane_id}."
            if use_tool_role:
                call_id = f"call_{lane_id}_{turn}"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": f"src/mod_{lane_id}_{turn}.py"}),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "read_file",
                    "content": tool_out,
                })
            else:
                messages.append({"role": "assistant", "content": reply_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Tool output turn {turn + 1}] Continue diagnosing based on "
                        f"this new trace/code output and write the next detailed step:\n\n{tool_out}"
                    ),
                })
    return rows


def summarize_replay(rows: list[dict], concurrency: int) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    err_rows = [r for r in rows if not r.get("ok")]
    cold_rows = [r for r in ok_rows if r.get("turn") == 0]
    warm_rows = [r for r in ok_rows if (r.get("turn") or 0) >= 1]

    def _hit_pct(subset):
        tot_in = sum(int(r.get("prompt_tokens") or 0) for r in subset)
        tot_cached = sum(int(r.get("cached_tokens") or 0) for r in subset)
        return round(100.0 * tot_cached / tot_in, 2) if tot_in > 0 else 0.0

    dec_vals = [r["decode_tps"] for r in ok_rows if r.get("decode_tps") is not None]
    warm_dec_vals = [r["decode_tps"] for r in warm_rows if r.get("decode_tps") is not None]
    tpot_vals = [r["tpot_ms"] for r in ok_rows if r.get("tpot_ms") is not None]
    ttft_all = [r["ttft_s"] for r in ok_rows if r.get("ttft_s") is not None]
    ttft_cold = [r["ttft_s"] for r in cold_rows if r.get("ttft_s") is not None]
    ttft_warm = [r["ttft_s"] for r in warm_rows if r.get("ttft_s") is not None]

    firsts = [r["first_token_at"] for r in ok_rows if r.get("first_token_at") is not None]
    lasts = [r["last_token_at"] for r in ok_rows if r.get("last_token_at") is not None]
    agg_decode_tps = None
    if firsts and lasts and max(lasts) > min(firsts):
        tot_out = sum(max(0, int(r.get("completion_tokens") or 0) - 1) for r in ok_rows)
        agg_decode_tps = round(tot_out / (max(lasts) - min(firsts)), 2)

    dp_rank_counts: dict[str, int] = {}
    dp_route_counts: dict[str, int] = {}
    lane_t0_rank: dict[int, int] = {}
    for r in ok_rows:
        rk = r.get("dp_rank")
        if rk is not None:
            dp_rank_counts[str(rk)] = dp_rank_counts.get(str(rk), 0) + 1
            if r.get("turn") == 0:
                lane_t0_rank[r["lane"]] = rk
        rt = r.get("dp_route")
        if rt:
            dp_route_counts[rt] = dp_route_counts.get(rt, 0) + 1

    sticky_checks = [r for r in warm_rows if r.get("dp_rank") is not None and r["lane"] in lane_t0_rank]
    sticky_same = sum(1 for r in sticky_checks if r["dp_rank"] == lane_t0_rank[r["lane"]])
    sticky_consistency = round(100.0 * sticky_same / len(sticky_checks), 2) if sticky_checks else None

    warm_hit = _hit_pct(warm_rows)
    dec_p50 = round(percentile(dec_vals, 50), 2) if dec_vals else None
    dec_mean = round(statistics.fmean(dec_vals), 2) if dec_vals else None

    m1_pass = (
        len(err_rows) == 0
        and warm_hit >= 75.0
        and (dec_mean is not None and dec_mean >= 85.0)
    )

    return {
        "concurrency": concurrency,
        "total_requests": len(rows),
        "ok_requests": len(ok_rows),
        "failed_requests": len(err_rows),
        "total_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in ok_rows),
        "total_cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in ok_rows),
        "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in ok_rows),
        "overall_cache_hit_pct": _hit_pct(ok_rows),
        "warm_cache_hit_pct": warm_hit,
        "decode_tps_p10": round(percentile(dec_vals, 10), 2) if dec_vals else None,
        "decode_tps_p50": dec_p50,
        "decode_tps_mean": dec_mean,
        "decode_tps_p90": round(percentile(dec_vals, 90), 2) if dec_vals else None,
        "warm_decode_tps_mean": round(statistics.fmean(warm_dec_vals), 2) if warm_dec_vals else None,
        "agg_decode_tps": agg_decode_tps,
        "ttft_cold_p50_s": round(percentile(ttft_cold, 50), 3) if ttft_cold else None,
        "ttft_cold_p95_s": round(percentile(ttft_cold, 95), 3) if ttft_cold else None,
        "ttft_warm_p50_s": round(percentile(ttft_warm, 50), 3) if ttft_warm else None,
        "ttft_warm_p95_s": round(percentile(ttft_warm, 95), 3) if ttft_warm else None,
        "ttft_all_p95_s": round(percentile(ttft_all, 95), 3) if ttft_all else None,
        "tpot_p50_ms": round(percentile(tpot_vals, 50), 2) if tpot_vals else None,
        "tpot_p95_ms": round(percentile(tpot_vals, 95), 2) if tpot_vals else None,
        "dp_rank_counts": dp_rank_counts,
        "dp_route_counts": dp_route_counts,
        "lane_sticky_consistency_pct": sticky_consistency,
        "m1_gate_pass": m1_pass,
        "sample_errors": [r["error"] for r in err_rows[:5]],
    }


def run_replay(
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    init_tokens: int,
    turn_append_tokens: int,
    turns: int,
    max_tokens: int,
    temperature: float,
    think_min_s: float,
    think_max_s: float,
    stagger_s: float,
    timeout_s: float,
    use_tool_role: bool,
    run_salt: str | None = None,
) -> dict:
    salt = run_salt or f"replay_{int(time.time())}"
    t0 = time.monotonic()
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(
                run_lane,
                lane_id=i,
                run_salt=salt,
                base_url=base_url,
                api_key=api_key,
                model=model,
                init_tokens=init_tokens,
                turn_append_tokens=turn_append_tokens,
                turns=turns,
                max_tokens=max_tokens,
                temperature=temperature,
                think_min_s=think_min_s,
                think_max_s=think_max_s,
                stagger_s=stagger_s,
                timeout_s=timeout_s,
                use_tool_role=use_tool_role,
            )
            for i in range(concurrency)
        ]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
    all_rows.sort(key=lambda r: (r["lane"], r["turn"]))
    summary = summarize_replay(all_rows, concurrency=concurrency)
    summary["wall_total_s"] = round(time.monotonic() - t0, 2)
    summary["run_salt"] = salt
    return {"summary": summary, "rows": all_rows}


def analyze_proxy_log(log_path: str, last_n: int = 0, since_minutes: float = 0.0) -> dict:
    p = Path(log_path)
    if not p.is_file():
        raise FileNotFoundError(f"proxy access log not found: {log_path}")
    now_ts = time.time()
    cutoff = (now_ts - since_minutes * 60.0) if since_minutes > 0 else 0.0
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if not isinstance(item, dict):
                continue
            if cutoff and float(item.get("time") or 0) < cutoff:
                continue
            rows.append(item)
    if last_n > 0:
        rows = rows[-last_n:]

    inf_rows = [r for r in rows if str(r.get("path") or "").startswith("/v1/") and r.get("path") != "/v1/models"]
    ok_rows = [r for r in inf_rows if int(r.get("status") or 0) == 200]
    err_rows = [r for r in inf_rows if int(r.get("status") or 0) >= 400]

    events = []
    for r in inf_rows:
        t_end = float(r.get("time") or 0)
        t_start = t_end - float(r.get("elapsed_s") or 0)
        events.append((t_start, 1))
        events.append((t_end, -1))
    events.sort()
    cur = max_c = 0
    for _, delta in events:
        cur += delta
        if cur > max_c:
            max_c = cur

    tot_in = sum(int(r.get("input") or 0) for r in ok_rows)
    tot_cached = sum(int(r.get("cached_tokens") or 0) for r in ok_rows)
    tot_out = sum(int(r.get("output") or 0) for r in ok_rows)
    has_cached_field = sum(1 for r in ok_rows if "cached_tokens" in r)

    dec_vals = [float(r["decode_tps"]) for r in ok_rows if isinstance(r.get("decode_tps"), (int, float))]
    ttft_vals = [float(r["ttft_s"]) for r in ok_rows if isinstance(r.get("ttft_s"), (int, float))]
    in_vals = [float(r.get("input") or 0) for r in ok_rows if int(r.get("input") or 0) > 0]
    out_vals = [float(r.get("output") or 0) for r in ok_rows if int(r.get("output") or 0) > 0]

    temps = [float(r["temperature"]) for r in ok_rows if isinstance(r.get("temperature"), (int, float))]
    temp_zero = sum(1 for t in temps if t == 0.0)

    dp_ranks: dict[str, int] = {}
    dp_routes: dict[str, int] = {}
    for r in ok_rows:
        if r.get("dp_rank") is not None:
            k = str(r["dp_rank"])
            dp_ranks[k] = dp_ranks.get(k, 0) + 1
        if r.get("dp_route"):
            rt = str(r["dp_route"])
            dp_routes[rt] = dp_routes.get(rt, 0) + 1

    return {
        "log_path": str(p),
        "inference_requests": len(inf_rows),
        "ok_requests": len(ok_rows),
        "error_requests": len(err_rows),
        "max_concurrency": max_c,
        "input_p50": round(percentile(in_vals, 50), 1) if in_vals else None,
        "input_p90": round(percentile(in_vals, 90), 1) if in_vals else None,
        "output_p50": round(percentile(out_vals, 50), 1) if out_vals else None,
        "total_input_tokens": tot_in,
        "total_cached_tokens": tot_cached,
        "total_output_tokens": tot_out,
        "cache_hit_pct": round(100.0 * tot_cached / tot_in, 2) if (tot_in > 0 and has_cached_field > 0) else None,
        "rows_with_p0_obs": has_cached_field,
        "decode_tps_p10": round(percentile(dec_vals, 10), 2) if dec_vals else None,
        "decode_tps_p50": round(percentile(dec_vals, 50), 2) if dec_vals else None,
        "decode_tps_mean": round(statistics.fmean(dec_vals), 2) if dec_vals else None,
        "decode_tps_p90": round(percentile(dec_vals, 90), 2) if dec_vals else None,
        "ttft_p50_s": round(percentile(ttft_vals, 50), 3) if ttft_vals else None,
        "ttft_p95_s": round(percentile(ttft_vals, 95), 3) if ttft_vals else None,
        "temperature_samples": len(temps),
        "temperature_zero_pct": round(100.0 * temp_zero / len(temps), 1) if temps else None,
        "dp_rank_counts": dp_ranks,
        "dp_route_counts": dp_routes,
    }


def _resolve_api_key(api_key: str, api_key_file: str) -> str:
    if api_key:
        return api_key.strip()
    env_key = os.environ.get("TF500_API_KEY", "").strip()
    if env_key:
        return env_key
    p = Path(api_key_file)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return ""


def _print_summary(title: str, data: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in data.items():
        print(f"  {k:30s}: {v}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TF500 P0 Multi-turn Agent Replay & Proxy Log Analyzer")
    parser.add_argument("--analyze-log", default="", help="Analyze existing proxy-access.jsonl instead of running replay")
    parser.add_argument("--last", type=int, default=0, help="Only analyze last N lines of log")
    parser.add_argument("--since-minutes", type=float, default=0.0, help="Only analyze log entries from last M minutes")

    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-file", default=DEFAULT_KEY_FILE)
    parser.add_argument("--concurrency", default="16", help="Single int or comma-separated list, e.g. 16,32")
    parser.add_argument("--init-tokens", type=int, default=40960)
    parser.add_argument("--turn-append-tokens", type=int, default=4096)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-min-s", type=float, default=2.0)
    parser.add_argument("--think-max-s", type=float, default=6.0)
    parser.add_argument("--stagger-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--use-tool-role", action="store_true", help="Use assistant.tool_calls + role=tool messages")
    parser.add_argument("--output", default="", help="Optional JSON file path to save full results")
    args = parser.parse_args(argv)

    if args.analyze_log:
        report = analyze_proxy_log(args.analyze_log, last_n=args.last, since_minutes=args.since_minutes)
        _print_summary("Proxy Access Log Summary", report)
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    api_key = _resolve_api_key(args.api_key, args.api_key_file)
    tiers = [int(x.strip()) for x in str(args.concurrency).split(",") if x.strip()]
    results = []
    for c in tiers:
        res = run_replay(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            concurrency=c,
            init_tokens=args.init_tokens,
            turn_append_tokens=args.turn_append_tokens,
            turns=args.turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            think_min_s=args.think_min_s,
            think_max_s=args.think_max_s,
            stagger_s=args.stagger_s,
            timeout_s=args.timeout_s,
            use_tool_role=args.use_tool_role,
        )
        results.append(res)
        _print_summary(f"Agent Replay C={c}", res["summary"])

    if args.output:
        Path(args.output).write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved full replay report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
