"""Agent benchmark engine and controller for the XTier 86KD WebUI console.

Supports concurrency tiers 1 / 4 / 8 / 16 / 32 / 64 / 100, input lengths up to
512K (and 1M when the online model allows), real coding-agent tool schemas and
isolated copy-on-write sandboxes, and exact total + per-session Prefill and
Decode speed telemetry.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import http.client
import json
import math
import os
import py_compile
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import tf500_benchmark_ui

HOST = "<TF500_IP>"
PORT = 8000
BACKEND_PORT = 30000
BACKEND_PORTS = (30000,)
MODEL = "DeepSeek-V4.1-Flash-1017"
KEY_FILE = Path("/data/sglang-api-key")
PROXY_KEY_FILE = Path("/data/xtier-86kd-console/proxy-api-key")
TOKENIZER_FILE = Path("/data/models/DeepSeek-V4.1-Flash-1017/tokenizer.json")
XTIER_ROOT = Path("/data/src/xtier")
REPO_ROOT = Path("/data/src/xtier_86kd")
SGLANG_ROOT = Path("/data/src/sglang-xtier")
PYTHON = Path("/data/xtier-env/bin/python")
ROOT = Path("/data/xtier-86kd-console/benchmark")

CONCURRENCIES = (1, 4, 8, 16, 32, 64, 100)
INPUT_TIERS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 393216, 524288, 1048576)
PROMPTS = INPUT_TIERS
OUTPUTS = (512, 1024, 2048, 4096)
TURNS = (1, 2, 3)
MODES = ("context", "concurrency")
STRATEGIES = ("agent_auto", "agent_fast")
ACTIVE = ("preparing", "running", "stopping")

SYSTEM = (
    "你是一个在终端里工作的编码代理（Agent）。工作目录是一个 Python 推理加速项目，"
    "包含 xtier（混合专家调度、KV 缓存池、DeepSeek V4.1 算子适配）、bench（性能基准与工具脚本）"
    "以及 sglang/srt 的核心源码。\n"
    "规则：先用工具确认代码细节再修改；修改使用 edit_file（old_string 必须唯一且逐字匹配）"
    "或 write_file；修改后用 check_python 做语法检查；回答使用中文、保持专业简洁。"
    "上文历史已读取的文件不要重复读取；完成修改和检查后给出总结，不再调用工具。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容（带行号）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作目录的路径"},
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
            "name": "list_dir",
            "description": "列出目录内容。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "在文件或目录中按正则表达式搜索，返回 路径:行号:内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "将文件中唯一出现的 old_string 替换为 new_string。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入或覆盖文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_python",
            "description": "对 Python 文件执行 py_compile 语法检查（不运行代码）。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

AGENT_TASK = (
    "为 bench/agent_bench.py 中的沙盒文件工具编写一份完整的 pytest 单元测试模块 bench/test_sandbox_suite.py，"
    "覆盖 read_file（含行区间截断）、list_dir、grep、edit_file（唯一命中、零命中、重复命中报错）、"
    "write_file、check_python（合法语法与语法错误）以及目录越界拦截，至少包含 14 个独立测试用例。"
    "相关核心源码已在上文历史中读取，请先用 write_file 写出完整测试模块，再用 check_python 校验语法。"
    "仅在当前沙盒路径内操作。"
)

LONG_DELIVERY_PROMPT = (
    "请基于上文已阅读的项目源码，直接在本次回答中交付完整的 pytest 单元测试与并发指标验证代码模块"
    "（至少 28 个独立测试函数，覆盖沙盒文件读写、行号分页、正则检索、唯一字符串替换、路径越界拦截、"
    "语法编译检查、流式区间并集吞吐计算、分位数统计与异常分支，包含完整辅助桩类、夹具与详尽断言）。"
    "当前是 Agent 代码交付阶段：不要调用任何工具、不要写摘要、不要省略任何实现代码，直接完整输出全部 Python 源码。"
)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _round_or_none(value: float | None, digits: int = 1) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _safe_worker(pid: int) -> bool:
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != os.getuid():
            return False
        stat_text = (proc / "stat").read_text(encoding="utf-8", errors="replace")
        if stat_text.rsplit(")", 1)[-1].strip().startswith("Z"):
            return False
        command = (proc / "cmdline").read_bytes()
        return b"tf500_benchmark.py" in command and b"--worker" in command
    except OSError:
        return False


def _log_tail(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-4000:].splitlines()[-8:]
    except OSError:
        return []


def _server_info() -> dict:
    """Query proxy and active SGLang backend for live model & DP-aware KV capacity."""
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{HOST}:{PORT}/health", timeout=4) as response:
            proxy_state = json.load(response)
        backend_port = proxy_state.get("backend_port")
        backend_host = proxy_state.get("backend_host") or HOST
        backend_label = proxy_state.get("backend_label") or f":{backend_port}"
        if not proxy_state.get("backend_online") or backend_port not in BACKEND_PORTS:
            raise ValueError("代理当前未检测到在线推理服务（:30000）")

        def get_json(endpoint: str) -> dict:
            req = urllib.request.Request(
                f"http://{backend_host}:{backend_port}{endpoint}",
                headers={"Authorization": f"Bearer {key}"},
            )
            with opener.open(req, timeout=8) as resp:
                return json.load(resp)

        models = get_json("/v1/models").get("data") or []
        served_ids = [str(item.get("id") or "") for item in models if isinstance(item, dict)]
        if not served_ids:
            raise ValueError("在线后端未返回可用模型列表")
        served = MODEL if MODEL in served_ids else served_ids[0]
        try:
            info = get_json("/server_info")
        except (OSError, urllib.error.URLError, ValueError):
            info = get_json("/get_server_info")
    except ValueError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError("推理后端未在线或无法读取容量，请先确认启动台与代理状态") from exc

    states = info.get("internal_states") or []
    caps = [
        state.get("memory_usage", {}).get("token_capacity")
        for state in states
        if isinstance(state, dict)
    ]
    caps = [val for val in caps if isinstance(val, int) and val > 0]
    context = info.get("context_length")
    if not caps or not isinstance(context, int) or context < 4096:
        raise ValueError("后端未返回有效的 context_length 或共享 KV 容量")

    dp_ranks = len(caps)
    single_kv = min(caps)
    total_kv = sum(caps)

    per_dp_running = [
        state.get("effective_max_running_requests_per_dp")
        for state in states
        if isinstance(state, dict)
        and isinstance(state.get("effective_max_running_requests_per_dp"), int)
        and state.get("effective_max_running_requests_per_dp") > 0
    ]
    raw_max_running = info.get("max_running_requests")
    if per_dp_running and len(per_dp_running) == dp_ranks:
        max_running = sum(per_dp_running)
    elif isinstance(raw_max_running, int) and raw_max_running > 0:
        max_running = raw_max_running
    else:
        max_running = 64

    page_size = info.get("page_size")
    page = page_size if isinstance(page_size, int) and page_size > 0 else 256
    spec_algo = info.get("speculative_algorithm") or ""
    chunked = info.get("chunked_prefill_size")

    signature = {
        "served": served,
        "backend_port": backend_port,
        "context": context,
        "dp_ranks": dp_ranks,
        "single_kv": single_kv,
        "total_kv": total_kv,
        "max_running": max_running,
        "chunked_prefill_size": chunked,
        "speculative_algorithm": spec_algo,
        "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "model": served,
        "backend_host": backend_host,
        "backend_port": backend_port,
        "backend_label": backend_label,
        "context": context,
        "dp_ranks": dp_ranks,
        "single_kv": single_kv,
        "kv": total_kv,
        "max_running": max_running,
        "page_size": page,
        "chunked_prefill_size": chunked,
        "spec_algo": spec_algo,
        "fingerprint": fingerprint,
    }


def _first_val(form: dict, name: str, default: str) -> str:
    raw = form.get(name, default)
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else default
    return str(raw)


def _parse_concurrencies(form: dict, mode: str) -> list[int]:
    if mode == "concurrency":
        raw_list = form.get("concurrencies") or form.get("concurrency") or []
        if isinstance(raw_list, (str, int)):
            parts = [p.strip() for p in str(raw_list).split(",") if p.strip()]
        else:
            parts = []
            for item in raw_list:
                parts.extend(p.strip() for p in str(item).split(",") if p.strip())
        if not parts:
            parts = [str(c) for c in CONCURRENCIES]
        selected = []
        for item in parts:
            try:
                val = int(item)
            except (TypeError, ValueError):
                raise ValueError("并发档位取值无效") from None
            if val not in CONCURRENCIES:
                raise ValueError(f"并发档位仅支持 {', '.join(map(str, CONCURRENCIES))}")
            if val not in selected:
                selected.append(val)
        return sorted(selected)

    raw = _first_val(form, "concurrency", "1")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError("concurrency 取值无效") from None
    if val not in CONCURRENCIES:
        raise ValueError(f"concurrency 只支持 {', '.join(map(str, CONCURRENCIES))}")
    return [val]


def _parse(form: dict) -> dict:
    mode = _first_val(form, "mode", "context").strip()
    if mode not in MODES:
        raise ValueError("测速模式无效")

    concurrencies = _parse_concurrencies(form, mode)

    max_raw = form.get("maximum") if "maximum" in form else form.get("prompt_tokens", "524288")
    if isinstance(max_raw, (list, tuple)):
        max_raw = max_raw[0] if max_raw else "524288"
    try:
        maximum = int(max_raw)
    except (TypeError, ValueError):
        raise ValueError("输入长度取值无效") from None
    if maximum not in INPUT_TIERS:
        raise ValueError("请选择支持的输入长度档位（4K–512K / 1M）")

    out_raw = form.get("output") if "output" in form else form.get("output_tokens", "1024")
    if isinstance(out_raw, (list, tuple)):
        out_raw = out_raw[0] if out_raw else "1024"
    try:
        output = int(out_raw)
    except (TypeError, ValueError):
        raise ValueError("输出上限取值无效") from None
    if output not in OUTPUTS:
        raise ValueError(f"单次输出上限只支持 {', '.join(map(str, OUTPUTS))}")

    turns_raw = form.get("turns") if "turns" in form else form.get("rounds", "1")
    if isinstance(turns_raw, (list, tuple)):
        turns_raw = turns_raw[0] if turns_raw else "1"
    try:
        turns = int(turns_raw)
    except (TypeError, ValueError):
        raise ValueError("采样轮数取值无效") from None
    if turns not in TURNS:
        raise ValueError("采样轮数只支持 1、2、3")

    points_raw = _first_val(form, "points", "8" if mode == "context" and maximum >= 32768 else "1")
    try:
        points = int(points_raw)
    except (TypeError, ValueError):
        raise ValueError("测试点数取值无效") from None
    if not 1 <= points <= 16:
        raise ValueError("测试点数须在 1–16 之间")

    strategy = _first_val(form, "strategy", "agent_auto").strip()
    if strategy not in STRATEGIES:
        raise ValueError("Agent 采样模式无效")

    return {
        "mode": mode,
        "concurrency": concurrencies[0],
        "concurrencies": concurrencies,
        "maximum": maximum,
        "prompt_tokens": maximum,
        "points": 1 if mode == "concurrency" else points,
        "output": output,
        "output_tokens": output,
        "turns": turns,
        "rounds": turns,
        "strategy": strategy,
    }


def _lane_limits(concurrency: int, output: int, cap: dict) -> dict:
    dp_ranks = max(1, int(cap.get("dp_ranks") or 1))
    single_kv = int(cap.get("single_kv") or cap["kv"])
    context = int(cap["context"])
    max_running = max(1, int(cap.get("max_running") or concurrency))
    page = max(1, int(cap.get("page_size") or 256))
    reserve = max(page, 256)

    reqs_per_dp = max(1, math.ceil(concurrency / dp_ranks))
    active_per_dp = max(1, math.ceil(min(concurrency, max_running) / dp_ranks))

    resident_lane = min(context, (single_kv // reqs_per_dp // page) * page)
    active_lane = min(context, (single_kv // active_per_dp // page) * page)

    eviction_possible = False
    if resident_lane - output - reserve >= 2048:
        per_lane = resident_lane
    elif active_lane - output - reserve >= 2048:
        per_lane = active_lane
        eviction_possible = True
    else:
        raise ValueError(
            f"并发 {concurrency} 下共享 KV 池不足以分配至少 2K 输入 + {output} 输出，请降低并发或输出上限"
        )

    safe_input = min(context, per_lane) - output - reserve
    if safe_input < 2048:
        raise ValueError("扣除输出与调度页预留后，单会话可用输入不足 2K tokens")

    return {
        "per_lane": min(context, per_lane),
        "safe_input": safe_input,
        "reserve": reserve,
        "queued": max(0, concurrency - max_running),
        "eviction_possible": eviction_possible,
    }


def build_plan(options: dict, cap: dict | None = None) -> dict:
    """Create a validated execution plan (or offline preview when cap is None)."""
    mode = options["mode"]
    output = options["output"]
    maximum = options["maximum"]

    if cap is None:
        if mode == "concurrency":
            steps = [
                {
                    "index": idx + 1,
                    "concurrency": c,
                    "target": maximum,
                    "requested_target": maximum,
                    "per_lane": maximum + output + 256,
                    "reserve": 256,
                    "queued": 0,
                    "adjusted": False,
                }
                for idx, c in enumerate(options["concurrencies"])
            ]
        else:
            pts = options["points"]
            c = options["concurrency"]
            targets = (
                [maximum]
                if pts == 1
                else [maximum * (i + 1) // pts for i in range(pts)]
            )
            steps = [
                {
                    "index": idx + 1,
                    "concurrency": c,
                    "target": t,
                    "requested_target": t,
                    "per_lane": maximum + output + 256,
                    "reserve": 256,
                    "queued": 0,
                    "adjusted": False,
                }
                for idx, t in enumerate(targets)
            ]
        return {
            "online": False,
            "mode": mode,
            "options": options,
            "points": len(steps),
            "steps": steps,
            "targets": [s["target"] for s in steps],
            "concurrencies": [s["concurrency"] for s in steps],
            "effective_maximum": maximum,
            "adjusted": False,
            "notes": ["等待推理后端上线（:30000）；上线后自动校验实时 KV 容量并解锁测速。"],
            "method": "xtier-86kd-agent-v2",
        }

    notes: list[str] = []
    steps: list[dict] = []
    any_adjusted = False

    if mode == "concurrency":
        queued_tiers = []
        evict_tiers = []
        for idx, c in enumerate(options["concurrencies"]):
            lim = _lane_limits(c, output, cap)
            eff = min(maximum, lim["safe_input"])
            adjusted = eff < maximum
            any_adjusted = any_adjusted or adjusted
            if lim["queued"] > 0:
                queued_tiers.append(f"{c}并发(排队{lim['queued']})")
            if lim["eviction_possible"]:
                evict_tiers.append(str(c))
            steps.append(
                {
                    "index": idx + 1,
                    "concurrency": c,
                    "target": eff,
                    "requested_target": maximum,
                    "per_lane": lim["per_lane"],
                    "reserve": lim["reserve"],
                    "queued": lim["queued"],
                    "adjusted": adjusted,
                }
            )
        if any_adjusted:
            notes.append("部分高并发档位受共享 KV 或模型上下文限制，已自动收敛单会话输入长度以防溢出。")
        if queued_tiers:
            notes.append(
                f"当前后端 max_running_requests={cap['max_running']}；{', '.join(queued_tiers)} 超出部分将在服务端队列等待。"
            )
        if evict_tiers:
            notes.append(f"并发 {'/'.join(evict_tiers)} 总历史超出静态常驻 KV，多轮间可能触发 RadixCache 换出。")
        eff_max = max(s["target"] for s in steps)
    else:
        c = options["concurrency"]
        pts = options["points"]
        lim = _lane_limits(c, output, cap)
        eff_max = min(maximum, lim["safe_input"])
        any_adjusted = eff_max < maximum
        if pts == 1:
            targets = [eff_max]
        else:
            span_top = min(maximum, lim["per_lane"])
            targets = [min(span_top * (i + 1) // pts, eff_max) for i in range(pts)]
            if targets[0] < 2048 or len(set(targets)) != pts:
                raise ValueError("当前可用输入上限较小，无法均分所选测试点数，请减少点数或降低并发")
        for idx, t in enumerate(targets):
            steps.append(
                {
                    "index": idx + 1,
                    "concurrency": c,
                    "target": t,
                    "requested_target": maximum if pts == 1 else (maximum * (idx + 1) // pts),
                    "per_lane": lim["per_lane"],
                    "reserve": lim["reserve"],
                    "queued": lim["queued"],
                    "adjusted": any_adjusted and idx == pts - 1,
                }
            )
        if any_adjusted:
            notes.append(
                f"容量保护：并发 {c} 下单会话安全输入上限调整为 {eff_max:,} tokens（预留 {output:,} 输出 + {lim['reserve']} 调度 tokens）。"
            )
        if lim["queued"] > 0:
            notes.append(
                f"当前后端 max_running_requests={cap['max_running']}，{c} 并发中有 {lim['queued']} 路将在服务端队列等待调度。"
            )
        if lim["eviction_possible"]:
            notes.append("高并发总历史超过共享 KV 池静态容量，多轮 Agent 采样可能触发部分前缀换出。")

    return {
        "online": True,
        "mode": mode,
        "options": options,
        "capacity": cap,
        "points": len(steps),
        "steps": steps,
        "targets": [s["target"] for s in steps],
        "concurrencies": [s["concurrency"] for s in steps],
        "effective_maximum": eff_max,
        "adjusted": any_adjusted,
        "notes": notes,
        "method": "xtier-86kd-agent-v2",
    }


def _collect_virtual_files() -> dict[str, Path]:
    """Build read-only mapping of sandbox relative paths to real Python source files."""
    mapping: dict[str, Path] = {}
    trees = [
        (XTIER_ROOT / "xtier", "xtier", "*.py"),
        (REPO_ROOT / "src" / "xtier_86kd", "xtier_86kd", "*.py"),
        (REPO_ROOT / "bench", "bench", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "layers", "sglang/srt/layers", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "managers", "sglang/srt/managers", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "models", "sglang/srt/models", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "model_executor", "sglang/srt/model_executor", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "speculative", "sglang/srt/speculative", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "mem_cache", "sglang/srt/mem_cache", "*.py"),
        (SGLANG_ROOT / "python" / "sglang" / "srt" / "entrypoints", "sglang/srt/entrypoints", "*.py"),
        (SGLANG_ROOT / "python" / "sglang", "sglang", "*.py"),
    ]
    for src_dir, dst_prefix, pattern in trees:
        if not src_dir.is_dir():
            continue
        for file_path in sorted(src_dir.rglob(pattern)):
            if "__pycache__" in file_path.parts or not file_path.is_file():
                continue
            try:
                if file_path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            rel = f"{dst_prefix}/{file_path.relative_to(src_dir).as_posix()}"
            if rel not in mapping:
                mapping[rel] = file_path

    if not mapping:
        for file_path in sorted(REPO_ROOT.rglob("*.py")):
            if "__pycache__" not in file_path.parts and file_path.is_file():
                rel = file_path.relative_to(REPO_ROOT).as_posix()
                mapping[rel] = file_path
    return mapping


def _format_numbered(text: str, start_line: int | None = None, end_line: int | None = None) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    a = max(1, int(start_line or 1))
    b = min(len(lines), int(end_line or len(lines)), a + 1999)
    if a > len(lines):
        return f"... (共 {len(lines)} 行)"
    body = "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(a, b + 1))
    if b < len(lines):
        body += f"\n... (共 {len(lines)} 行，可用 start_line/end_line 继续读取)"
    return body


class OverlaySandbox:
    """Zero-copy isolated sandbox per Agent session; writes stay in memory / lazy temp file."""

    def __init__(self, base_files: dict[str, Path], text_cache: dict[str, str] | None = None):
        self._base = base_files
        self._cache = text_cache if text_cache is not None else {}
        self._writes: dict[str, str] = {}
        self._tmpdir: Path | None = None

    def _norm(self, path: str) -> str:
        raw = str(path or "").strip().replace("\\", "/").lstrip("/")
        if not raw or raw == ".":
            return ""
        parts = []
        for part in raw.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise ValueError("路径超出沙盒工作目录")
            parts.append(part)
        return "/".join(parts)

    def _read_raw(self, rel: str) -> str:
        if rel in self._writes:
            return self._writes[rel]
        if rel in self._cache:
            return self._cache[rel]
        src = self._base.get(rel)
        if src is None or not src.is_file():
            raise FileNotFoundError(f"文件不存在：{rel}")
        text = src.read_text(encoding="utf-8", errors="replace")
        self._cache[rel] = text
        return text

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return f"未知工具：{name}"
        try:
            return fn(**args)
        except TypeError as exc:
            return f"参数错误：{exc}"
        except Exception as exc:
            return f"错误：{type(exc).__name__}: {exc}"

    def t_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        rel = self._norm(path)
        if not rel:
            raise ValueError("请提供具体文件路径")
        return _format_numbered(self._read_raw(rel), start_line, end_line)

    def t_list_dir(self, path: str = ".") -> str:
        prefix = self._norm(path)
        entries: set[str] = set()
        all_keys = set(self._base) | set(self._writes)
        lead = f"{prefix}/" if prefix else ""
        for key in all_keys:
            if prefix and not key.startswith(lead):
                continue
            rem = key[len(lead):] if prefix else key
            if not rem:
                continue
            head, sep, _ = rem.partition("/")
            entries.add(head + ("/" if sep else ""))
        if not entries and prefix and prefix not in all_keys:
            raise FileNotFoundError(f"目录不存在：{path}")
        return "\n".join(sorted(entries)[:300])

    def t_grep(self, pattern: str, path: str = ".") -> str:
        rx = re.compile(pattern)
        prefix = self._norm(path)
        all_keys = sorted(set(self._base) | set(self._writes))
        if prefix in all_keys:
            targets = [prefix]
        else:
            lead = f"{prefix}/" if prefix else ""
            targets = [k for k in all_keys if not prefix or k.startswith(lead)]
        hits: list[str] = []
        for key in targets:
            try:
                text = self._read_raw(key)
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{key}:{lineno}:{line[:200]}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n...（仅显示前 200 条）"
        return "\n".join(hits) or "无匹配"

    def t_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        rel = self._norm(path)
        if not rel:
            raise ValueError("无效路径")
        current = self._read_raw(rel)
        occurrences = current.count(old_string)
        if occurrences != 1:
            return f"错误：old_string 出现 {occurrences} 次（必须恰好 1 次）"
        self._writes[rel] = current.replace(old_string, new_string, 1)
        return f"已修改 {rel}"

    def t_write_file(self, path: str, content: str) -> str:
        rel = self._norm(path)
        if not rel:
            raise ValueError("无效路径")
        text = str(content)
        self._writes[rel] = text
        return f"已写入 {rel}（{len(text.splitlines())} 行）"

    def t_check_python(self, path: str) -> str:
        rel = self._norm(path)
        if not rel:
            raise ValueError("无效路径")
        source = self._read_raw(rel)
        try:
            compile(source, rel, "exec")
            return "语法检查通过"
        except SyntaxError as exc:
            return f"语法错误：第 {exc.lineno} 行: {exc.msg}"

    def close(self) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


class CorpusCache:
    """Pre-formatted code corpus with fast token prefix table up to >1M tokens."""

    def __init__(self, virtual_files: dict[str, Path]):
        self.virtual_files = virtual_files
        self.raw_cache: dict[str, str] = {}
        self.tokenizer = None
        if TOKENIZER_FILE.is_file():
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
            except Exception:
                self.tokenizer = None

        priority = [
            "bench/agent_bench.py",
            "xtier/plugin.py",
            "xtier/platform.py",
            "xtier/config.py",
        ]
        ordered_keys = [k for k in priority if k in virtual_files]
        ordered_keys.extend(k for k in sorted(virtual_files) if k not in set(ordered_keys))

        self.entries: list[tuple[str, str, int, int]] = []
        total_tokens = 0
        total_chars = 0
        batch: list[tuple[str, str]] = []

        def flush_batch() -> None:
            nonlocal total_tokens, total_chars
            if not batch:
                return
            texts = [item[1] for item in batch]
            if self.tokenizer is not None:
                counts = [len(enc.ids) for enc in self.tokenizer.encode_batch(texts)]
            else:
                counts = [max(1, len(t.encode("utf-8")) // 3) for t in texts]
            for (k, numbered), tok_raw in zip(batch, counts):
                tok_est = tok_raw + 22
                self.entries.append((k, numbered, len(numbered), tok_est))
                total_chars += len(numbered)
                total_tokens += tok_est
            batch.clear()

        for key in ordered_keys:
            try:
                raw = virtual_files[key].read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not raw.strip():
                continue
            self.raw_cache[key] = raw
            numbered = _format_numbered(raw)
            batch.append((key, numbered))
            if len(batch) >= 48:
                flush_batch()
                if total_tokens >= 1_250_000 and total_chars >= 4_600_000:
                    break
        flush_batch()

        if total_tokens < 1_200_000:
            synth = "\n".join(
                f"{i:6d}\tdef xtier_dispatch_kernel_{i}(row: int, expert: int) -> int: return ((row * 131 + expert * 17 + {i}) ^ 0x5A5A3C3C)"
                for i in range(1, 1601)
            )
            synth_tok = self.count_text(synth) + 22
            synth_idx = 0
            while total_tokens < 1_200_000:
                self.entries.append((f"bench/synthetic_corpus_{synth_idx}.py", synth, len(synth), synth_tok))
                total_chars += len(synth)
                total_tokens += synth_tok
                synth_idx += 1

        self.total_chars = sum(item[2] for item in self.entries)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return max(1, len(text.encode("utf-8")) // 3)

    def estimate_char_budget(self, needed_tokens: int) -> int:
        if needed_tokens <= 0:
            return 0
        acc_tok = 0
        acc_chars = 0
        for _name, _text, n_chars, n_tok in self.entries:
            if acc_tok + n_tok >= needed_tokens:
                frac = max(0.0, min(1.0, (needed_tokens - acc_tok) / max(1, n_tok)))
                return acc_chars + max(1, int(n_chars * frac))
            acc_tok += n_tok
            acc_chars += n_chars
        return self.total_chars

    def build_history_tail(self, char_budget: int) -> list[dict]:
        msgs: list[dict] = []
        remaining = max(0, int(char_budget))
        for idx, (name, numbered, n_chars, _n_tok) in enumerate(self.entries):
            if remaining <= 0:
                break
            snippet = numbered if remaining >= n_chars else numbered[:remaining]
            cid = f"history_read_{idx}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": name}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": cid, "content": snippet})
            remaining -= n_chars
        return msgs


class Transport:
    """Thread-safe HTTP client through the local API proxy with instant socket abort."""

    def __init__(self, host: str, port: int, api_key: str, cancel: threading.Event | None = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.cancel = cancel or threading.Event()
        self.connections: set[http.client.HTTPConnection] = set()
        self.lock = threading.Lock()

    def check(self) -> None:
        if self.cancel.is_set():
            raise InterruptedError("已停止测速")

    def close_all(self) -> None:
        self.cancel.set()
        with self.lock:
            conns = list(self.connections)
        for conn in conns:
            for attr in ("_stream_socket", "sock"):
                sock = getattr(conn, attr, None)
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            try:
                conn.close()
            except OSError:
                pass

    def open_raw(
        self,
        method: str,
        path: str,
        body_bytes: bytes | None = None,
        timeout: int = 60,
        extra_headers: dict[str, str] | None = None,
    ):
        self.check()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        with self.lock:
            self.connections.add(conn)
        try:
            conn.connect()
            if conn.sock is not None:
                try:
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                conn.sock.settimeout(timeout)
                conn._stream_socket = conn.sock.dup()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept-Encoding": "identity",
            }
            if extra_headers:
                headers.update(extra_headers)
            if body_bytes is not None:
                headers["Content-Type"] = "application/json"
                headers["Content-Length"] = str(len(body_bytes))
            conn.request(method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read(600).decode("utf-8", errors="replace")
                if resp.status in (502, 503):
                    raise ValueError("推理后端未在线或正在加载，请检查启动台状态")
                raise RuntimeError(f"接口返回 HTTP {resp.status}: {detail}")
            return conn, resp
        except Exception:
            self.close(conn)
            raise

    def close(self, conn: http.client.HTTPConnection) -> None:
        try:
            conn.close()
        except OSError:
            pass
        dup_sock = getattr(conn, "_stream_socket", None)
        if dup_sock is not None:
            try:
                dup_sock.close()
            except OSError:
                pass
        with self.lock:
            self.connections.discard(conn)

    def post_json(self, path: str, payload: dict, timeout: int = 120) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        conn, resp = self.open_raw("POST", path, encoded, timeout=timeout)
        try:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
        finally:
            self.close(conn)


def count_chat_tokens(
    transport: Transport,
    cache: CorpusCache,
    model: str,
    messages: list[dict],
    tool_choice: str = "auto",
) -> int:
    """Count exact chat-template + tools tokens via /v1/tokenize with local fallback."""
    try:
        data = transport.post_json(
            "/v1/tokenize",
            {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": tool_choice,
            },
            timeout=180,
        )
        n = data.get("count")
        if isinstance(n, int) and n > 0:
            return n
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
    serialized = json.dumps(TOOLS, ensure_ascii=False) + "\n" + "\n".join(
        str(m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False)
        for m in messages
    )
    return cache.count_text(serialized)


def fit_point_sessions(
    transport: Transport,
    cache: CorpusCache,
    model: str,
    target: int,
    run_id: str,
    point_idx: int,
    concurrency: int,
    strategy: str,
) -> tuple[list[dict], int]:
    """Construct `concurrency` independent cold-prefix Agent sessions in O(1) tokenize calls."""
    user_prompt = LONG_DELIVERY_PROMPT if strategy == "agent_fast" else AGENT_TASK
    tool_choice = "auto"

    def make_msgs(lane: int, tail: list[dict]) -> list[dict]:
        marker = f"{run_id}-P{point_idx:02d}-S{lane:03d}"
        return [
            {"role": "system", "content": f"（Agent 测速会话 {marker}）\n{SYSTEM}"},
            {"role": "user", "content": user_prompt},
            *tail,
        ]

    # Leave a 4-token cushion so 3-digit lane markers S000..S099 never exceed `target`.
    aim = max(512, target - 4)
    base_count = count_chat_tokens(transport, cache, model, make_msgs(0, []), tool_choice=tool_choice)
    if base_count > aim:
        raise ValueError(f"目标输入档位 {target} 不足以容纳 Agent 系统提示与工具定义（基础开销 {base_count} tokens）")

    needed = aim - base_count
    low = 0
    high = cache.total_chars
    best_tail: list[dict] = []
    best_count = base_count
    budget = min(high, max(1, cache.estimate_char_budget(needed)))

    tolerance = min(128, max(24, aim // 400))
    for _ in range(12):
        transport.check()
        tail = cache.build_history_tail(budget)
        cnt = count_chat_tokens(transport, cache, model, make_msgs(0, tail), tool_choice=tool_choice)
        if cnt <= aim:
            best_tail, best_count, low = tail, cnt, budget
            if aim - cnt <= tolerance:
                break
        else:
            high = budget
        if high - low <= 2:
            break
        ratio = max(1, aim - base_count) / max(1, cnt - base_count)
        proposed = int(budget * ratio)
        if proposed <= low or proposed >= high:
            proposed = (low + high) // 2
        budget = max(low + 1, min(high - 1, proposed))

    if aim - best_count > max(384, aim // 50):
        raise ValueError(
            f"真实代码语料不足以构造 {target:,} tokens 档位（当前最大可得 {best_count:,} tokens）"
        )

    sessions = []
    for lane in range(concurrency):
        msgs = make_msgs(lane, best_tail)
        sessions.append(
            {
                "lane": lane + 1,
                "messages": msgs,
                "prev_tokens": best_count + 4,
                "prev_msg_len": len(msgs),
                "active": strategy == "agent_auto",
                "trimmed": False,
            }
        )
    return sessions, best_count + 4


def bound_session_history_fast(
    transport: Transport,
    cache: CorpusCache,
    model: str,
    session: dict,
    limit: int,
    tool_choice: str,
) -> tuple[int, bool]:
    """O(1) local delta bound check for high-concurrency Wave 1+; trims seeded history only if needed."""
    messages = session["messages"]
    added_msgs = messages[session.get("prev_msg_len", len(messages)):]
    delta_tokens = sum(
        cache.count_text(str(m.get("content") or "") + str(m.get("reasoning_content") or "")) + 24
        for m in added_msgs
    )
    est = int(session.get("prev_tokens") or 0) + delta_tokens
    if est <= limit - 192:
        session["prev_tokens"] = est
        session["prev_msg_len"] = len(messages)
        return est, False

    # Copy seeded tool messages before mutating so shared Turn-0 strings stay intact for sibling lanes.
    for idx, msg in enumerate(messages):
        if msg.get("role") == "tool" and str(msg.get("tool_call_id", "")).startswith("history_read_"):
            messages[idx] = dict(msg)

    trimmed = False
    for _ in range(10):
        if est <= limit - 64:
            break
        candidates = [
            m
            for m in messages
            if m.get("role") == "tool"
            and str(m.get("tool_call_id", "")).startswith("history_read_")
            and len(m.get("content", "")) > 64
        ]
        if not candidates:
            break
        need_remove_tok = max(128, est - (limit - 128))
        remove_chars = need_remove_tok * 4
        freed_tok = 0
        for msg in reversed(candidates):
            cut = min(remove_chars, len(msg["content"]) - 64)
            if cut > 0:
                removed_slice = msg["content"][-cut:]
                msg["content"] = msg["content"][:-cut]
                freed_tok += max(1, cache.count_text(removed_slice))
                remove_chars -= cut
                trimmed = True
            if remove_chars <= 0:
                break
        est = max(512, est - freed_tok)

    if est > limit:
        exact = count_chat_tokens(transport, cache, model, messages, tool_choice=tool_choice)
        if exact > limit:
            raise ValueError("Agent 历史已占满单路安全容量，无法继续追加轮次")
        est = exact

    session["prev_tokens"] = est
    session["prev_msg_len"] = len(messages)
    return est, trimmed


def qualify_decode(row: dict) -> dict:
    if "raw_stream_tps" not in row:
        row["raw_stream_tps"] = row.get("decode_tps")
    req_out = int(row.get("requested_output") or 1024)
    min_out = min(256, max(64, req_out // 2))
    min_measured = min(128, max(32, min_out // 2))
    reason = None
    if not row.get("ok", True):
        reason = row.get("error") or "请求失败"
    elif (row.get("output_tokens") or 0) < min_out:
        reason = f"Agent 短工具调用（输出 {row.get('output_tokens', 0)} < {min_out} tokens），仅计入首字与工具轮次"
    elif (row.get("decode_s") or 0.0) < 0.08:
        reason = "流式解码窗口不足 80ms，不作为持续解码速度样本"
    elif (row.get("measured_tokens") or 0) < min_measured:
        reason = f"首包后可计时 token 不足 {min_measured}，不计入持续解码速度"

    row["decode_note"] = reason
    row["decode_valid"] = reason is None and row.get("raw_stream_tps") is not None
    row["decode_tps"] = row["raw_stream_tps"] if row["decode_valid"] else None
    return row


def stream_agent_request(
    transport: Transport,
    model: str,
    messages: list[dict],
    output: int,
    barrier: threading.Barrier,
    expected_limit: int,
    tool_choice: str = "auto",
    on_first_token=None,
    on_chunk_tokens=None,
    dp_rank: int | None = None,
) -> tuple[dict, str, str, dict]:
    """Pre-serialize payload before the barrier, then stream and time SSE tokens accurately."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "max_tokens": output,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_cached_tokens_details": True,
        "return_spec_tokens_details": True,
    }
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    extra_hdrs = {"X-Data-Parallel-Rank": str(dp_rank)} if dp_rank is not None else None

    barrier.wait(timeout=180)
    started = time.perf_counter()
    conn, resp = transport.open_raw(
        "POST", "/v1/chat/completions", body_bytes, timeout=1800, extra_headers=extra_hdrs
    )

    first = last = None
    first_tokens = last_tokens = None
    chunk_count = 0
    usage: dict = {}
    spec_details: dict = {}
    calls: dict[int, dict] = {}
    texts: list[str] = []
    reasoning: list[str] = []
    finish = None
    done = False

    try:
        for raw in resp:
            transport.check()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if isinstance(event.get("usage"), dict) and event["usage"]:
                usage = event["usage"]
            sglext = event.get("sglext")
            if isinstance(sglext, dict) and isinstance(sglext.get("spec_tokens_details"), dict):
                spec_details = sglext["spec_tokens_details"]

            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                emitted = False
                if delta.get("content"):
                    texts.append(delta["content"])
                    emitted = True
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                    emitted = True
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                    emitted = True
                if emitted:
                    now = time.perf_counter()
                    chunk_count += 1
                    cum_tok = (event.get("usage") or {}).get("completion_tokens")
                    if first is None:
                        first = last = now
                        first_tokens = last_tokens = cum_tok if isinstance(cum_tok, int) else None
                        if on_first_token is not None:
                            on_first_token(now)
                    else:
                        if isinstance(cum_tok, int) and first_tokens is not None:
                            if cum_tok > last_tokens:
                                last = now
                                last_tokens = cum_tok
                        else:
                            last = now
                    if on_chunk_tokens is not None:
                        on_chunk_tokens(cum_tok if isinstance(cum_tok, int) else chunk_count)
        ended = time.perf_counter()
    finally:
        transport.close(conn)

    comp_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if not done or first is None or not isinstance(comp_tokens, int) or comp_tokens < 1 or not isinstance(prompt_tokens, int):
        raise RuntimeError("流式响应中断或未返回完整 token usage")
    if prompt_tokens > expected_limit + 128:
        raise RuntimeError(f"实际输入 {prompt_tokens} 超出安全规划上限 {expected_limit}")

    decode_s = max(0.0, (last - first) if (first is not None and last is not None) else 0.0)
    if first_tokens is not None and last_tokens is not None and last_tokens >= first_tokens:
        measured_tokens = last_tokens - first_tokens
        timing_method = "continuous-usage-v2"
    else:
        measured_tokens = max(0, comp_tokens - 1)
        timing_method = "stream-span-v1"

    decode_tps = (measured_tokens / decode_s) if (decode_s > 0 and measured_tokens > 0) else None
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else 0
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        cached_tokens = 0
    cached_tokens = min(cached_tokens, prompt_tokens)
    uncached_tokens = prompt_tokens - cached_tokens
    ttft_s = max(1e-6, first - started)
    verify_steps = int(spec_details.get("spec_verify_ct") or 0)

    row = {
        "ok": True,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "output_tokens": comp_tokens,
        "requested_output": output,
        "measured_tokens": measured_tokens,
        "timing_method": timing_method,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "wall_s": ended - started,
        "ind_prefill_tps": uncached_tokens / ttft_s if uncached_tokens > 0 else None,
        "decode_tps": decode_tps,
        "verify_steps": verify_steps,
        "started": started,
        "first": first,
        "last": last,
        "ended": ended,
        "finish_reason": finish,
        "short_output": comp_tokens < output,
        "tool_calls": [c["name"] for c in calls.values() if c.get("name")],
    }
    qualify_decode(row)
    return row, "".join(texts), "".join(reasoning), calls


def advance_agent_session(
    sandbox: OverlaySandbox,
    messages: list[dict],
    text: str,
    reasoning: str,
    calls: dict,
    turn: int,
) -> bool:
    if not calls:
        messages.append({"role": "assistant", "content": text, "reasoning_content": reasoning})
        return False

    allowed = {item["function"]["name"] for item in TOOLS}
    tool_calls_out = []
    tool_results_out = []
    for idx, call in sorted(calls.items()):
        cid = call.get("id") or f"call_{turn}_{idx}"
        name = call.get("name") or ""
        raw_args = call.get("args") or "{}"
        try:
            parsed = json.loads(raw_args)
            if not isinstance(parsed, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            result = sandbox.call(name, parsed) if name in allowed else f"未知工具：{name}"
            safe_args = json.dumps(parsed, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            result = f"工具参数解析错误：{exc}"
            safe_args = "{}"
        tool_calls_out.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": safe_args},
            }
        )
        tool_results_out.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": str(result)[:60000],
            }
        )

    messages.append(
        {
            "role": "assistant",
            "content": text,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls_out,
        }
    )
    messages.extend(tool_results_out)
    return True


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    cleaned = sorted((a, b) for a, b in intervals if b > a)
    if not cleaned:
        return 0.0
    total = 0.0
    lo, hi = cleaned[0]
    for start, end in cleaned[1:]:
        if start > hi:
            total += hi - lo
            lo, hi = start, end
        elif end > hi:
            hi = end
    return total + (hi - lo)


def aggregate_point(rows: list[dict], concurrency: int) -> dict:
    """Compute exact Total and Per-Session Prefill & Decode speeds across all concurrent sessions."""
    for row in rows:
        qualify_decode(row)

    ok_rows = [r for r in rows if r.get("ok", True)]
    cold_rows = [r for r in ok_rows if r.get("turn", 0) == 0]
    later_rows = [r for r in ok_rows if r.get("turn", 0) > 0]
    valid_decode = [r for r in ok_rows if r.get("decode_valid") and r.get("decode_tps") is not None]

    # --- 1. Prefill Total Speed & Per-Session Speed (Turn 0 Cold Prefill) ---
    prefill_total_tps = None
    prefill_per_session_tps = None
    prefill_session_mean_tps = None
    group_ttft_s = None
    ttft_mean_s = None
    ttft_p50_s = None
    ttft_p95_s = None
    actual_input = None

    if cold_rows:
        earliest_start = min(r["started"] for r in cold_rows)
        latest_first = max(r["first"] for r in cold_rows)
        group_ttft_s = max(0.0, latest_first - earliest_start)
        total_uncached = sum(
            max(0, int(r["prompt_tokens"]) - int(r.get("cached_tokens") or 0))
            for r in cold_rows
        )
        if group_ttft_s > 0 and total_uncached > 0:
            prefill_total_tps = total_uncached / group_ttft_s
            prefill_per_session_tps = prefill_total_tps / max(1, len(cold_rows))

        ind_rates = [r["ind_prefill_tps"] for r in cold_rows if r.get("ind_prefill_tps") is not None]
        if ind_rates:
            prefill_session_mean_tps = statistics.mean(ind_rates)

        ttfts = [r["ttft_s"] for r in cold_rows]
        ttft_mean_s = statistics.mean(ttfts)
        ttft_p50_s = _percentile(ttfts, 0.50)
        ttft_p95_s = _percentile(ttfts, 0.95)
        actual_input = round(statistics.mean(r["prompt_tokens"] for r in cold_rows))

    cached_ttft_p50_s = None
    cache_hit_pct = None
    if later_rows:
        cached_ttft_p50_s = _percentile([r["ttft_s"] for r in later_rows], 0.50)
        later_prompt = sum(r["prompt_tokens"] for r in later_rows)
        later_cached = sum(r.get("cached_tokens") or 0 for r in later_rows)
        if later_prompt > 0:
            cache_hit_pct = 100.0 * later_cached / later_prompt

    # --- 2. Decode Total Speed & Per-Session Speed ---
    turns_present = sorted({r.get("turn", 0) for r in valid_decode})
    by_turn = {t: [r for r in valid_decode if r.get("turn", 0) == t] for t in turns_present}

    full_turns = [t for t, grp in by_turn.items() if len(grp) == concurrency]
    majority_threshold = max(1, math.ceil(concurrency * 0.8))
    majority_turns = [t for t, grp in by_turn.items() if len(grp) >= majority_threshold]

    selected_wave_turns = full_turns or majority_turns or turns_present
    wave_tokens = 0
    wave_union_s = 0.0
    overlap_sum_s = 0.0
    for t in selected_wave_turns:
        grp = by_turn[t]
        u = _interval_union([(r["first"], r["last"]) for r in grp])
        if u > 0:
            wave_tokens += sum(r["measured_tokens"] for r in grp)
            wave_union_s += u
            overlap_sum_s += sum(r["decode_s"] for r in grp)

    decode_total_tps = (wave_tokens / wave_union_s) if wave_union_s > 0 else None
    decode_per_session_tps = (decode_total_tps / max(1, concurrency)) if decode_total_tps is not None else None
    decode_overlap = (overlap_sum_s / wave_union_s) if wave_union_s > 0 else None

    # --- 3. Per-Session Breakdown (Lanes 1..C) ---
    lanes = []
    lane_rates: list[float] = []
    for lane_id in range(1, concurrency + 1):
        lane_all = [r for r in ok_rows if r.get("lane") == lane_id]
        lane_cold = next((r for r in cold_rows if r.get("lane") == lane_id), None)
        lane_good = [r for r in valid_decode if r.get("lane") == lane_id]
        dec_sec = sum(r["decode_s"] for r in lane_good)
        dec_tok = sum(r["measured_tokens"] for r in lane_good)
        lane_dec_tps = (dec_tok / dec_sec) if dec_sec > 0 else None
        if lane_dec_tps is not None:
            lane_rates.append(lane_dec_tps)

        v_steps = sum(r.get("verify_steps") or 0 for r in lane_all)
        out_sum = sum(r.get("output_tokens") or 0 for r in lane_all)
        tools_used = []
        for r in lane_all:
            tools_used.extend(r.get("tool_calls") or [])

        lanes.append(
            {
                "lane": lane_id,
                "ok": bool(lane_all),
                "prompt_tokens": lane_cold["prompt_tokens"] if lane_cold else None,
                "ttft_s": _round_or_none(lane_cold["ttft_s"], 2) if lane_cold else None,
                "prefill_alloc_tps": _round_or_none(prefill_per_session_tps, 1),
                "prefill_ind_tps": _round_or_none(lane_cold.get("ind_prefill_tps"), 1) if lane_cold else None,
                "decode_tps": _round_or_none(lane_dec_tps, 1),
                "decode_alloc_tps": _round_or_none(decode_per_session_tps, 1),
                "output_tokens": out_sum,
                "turns": len(lane_all),
                "accept_len": _round_or_none(out_sum / v_steps, 2) if v_steps > 0 else None,
                "tools": tools_used[:6],
            }
        )

    decode_lane_sum_tps = sum(lane_rates) if lane_rates else None
    decode_lane_mean_tps = statistics.mean(lane_rates) if lane_rates else None
    decode_lane_p10_tps = _percentile(lane_rates, 0.10)
    decode_lane_p50_tps = _percentile(lane_rates, 0.50)
    decode_lane_p90_tps = _percentile(lane_rates, 0.90)
    decode_lane_min_tps = min(lane_rates) if lane_rates else None
    decode_lane_max_tps = max(lane_rates) if lane_rates else None

    # --- 4. End-to-End & DSpark Speculative Metrics ---
    turn_walls = []
    for t in sorted({r.get("turn", 0) for r in ok_rows}):
        grp = [r for r in ok_rows if r.get("turn", 0) == t]
        if grp:
            turn_walls.append(max(r["ended"] for r in grp) - min(r["started"] for r in grp))
    total_wall = sum(turn_walls)
    total_out_tokens = sum(r.get("output_tokens") or 0 for r in ok_rows)
    e2e_total_tps = (total_out_tokens / total_wall) if total_wall > 0 else None
    e2e_per_session_tps = (e2e_total_tps / max(1, concurrency)) if e2e_total_tps is not None else None

    spec_rows = [r for r in valid_decode if (r.get("verify_steps") or 0) > 0]
    total_verify = sum(r["verify_steps"] for r in spec_rows)
    accept_len = (sum(r["output_tokens"] for r in spec_rows) / total_verify) if total_verify > 0 else None
    step_ms = (1000.0 * sum(r["decode_s"] for r in spec_rows) / total_verify) if total_verify > 0 else None

    notes = list(dict.fromkeys(r["decode_note"] for r in rows if r.get("decode_note")))

    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "success": len(cold_rows),
        "failed_requests": len(rows) - len(ok_rows),
        "actual_input": actual_input,
        "prefill_total_tps": _round_or_none(prefill_total_tps, 1),
        "prefill_per_session_tps": _round_or_none(prefill_per_session_tps, 1),
        "prefill_session_mean_tps": _round_or_none(prefill_session_mean_tps, 1),
        "decode_total_tps": _round_or_none(decode_total_tps, 1),
        "decode_per_session_tps": _round_or_none(decode_per_session_tps, 1),
        "decode_lane_sum_tps": _round_or_none(decode_lane_sum_tps, 1),
        "decode_lane_mean_tps": _round_or_none(decode_lane_mean_tps, 1),
        "decode_lane_p10_tps": _round_or_none(decode_lane_p10_tps, 1),
        "decode_lane_p50_tps": _round_or_none(decode_lane_p50_tps, 1),
        "decode_lane_p90_tps": _round_or_none(decode_lane_p90_tps, 1),
        "decode_lane_min_tps": _round_or_none(decode_lane_min_tps, 1),
        "decode_lane_max_tps": _round_or_none(decode_lane_max_tps, 1),
        "decode_overlap": _round_or_none(decode_overlap, 2),
        "e2e_total_tps": _round_or_none(e2e_total_tps, 1),
        "e2e_per_session_tps": _round_or_none(e2e_per_session_tps, 1),
        "group_ttft_s": _round_or_none(group_ttft_s, 2),
        "ttft_mean_s": _round_or_none(ttft_mean_s, 2),
        "ttft_p50_s": _round_or_none(ttft_p50_s, 2),
        "ttft_p95_s": _round_or_none(ttft_p95_s, 2),
        "cached_ttft_p50_s": _round_or_none(cached_ttft_p50_s, 2),
        "cache_hit_pct": _round_or_none(cache_hit_pct, 1),
        "accept_len": _round_or_none(accept_len, 2),
        "step_ms": _round_or_none(step_ms, 2),
        "valid_decode_requests": len(valid_decode),
        "excluded_decode_requests": len(ok_rows) - len(valid_decode),
        "qualified_parallel_rounds": len(majority_turns),
        "short_outputs": sum(1 for r in ok_rows if r.get("short_output")),
        "total_output_tokens": total_out_tokens,
        "decode_note": "；".join(notes[:3]),
        "lanes": lanes,
    }


def run_worker(spec_path: Path, state_path: Path, cancel_path: Path) -> None:
    spec = _read(spec_path)
    run_id = spec["id"]
    options = spec["options"]
    steps = spec["steps"]
    total_points = len(steps)
    proxy_key = Path(spec["proxy_key_file"]).read_text(encoding="utf-8").strip()
    transport = Transport(spec.get("proxy_host", HOST), int(spec.get("proxy_port", PORT)), proxy_key)

    state = _read(state_path)
    state.update(pid=os.getpid(), state="preparing", phase="prepare")
    state_lock = threading.Lock()
    last_publish = [0.0]

    def publish(force: bool = True, **changes) -> None:
        now = time.monotonic()
        with state_lock:
            state.update(changes)
            if not force and now - last_publish[0] < 0.35:
                return
            last_publish[0] = now
            state["updated_at"] = time.time()
            _write(state_path, state)

    def watch_cancel() -> None:
        while not transport.cancel.wait(0.2):
            if cancel_path.exists():
                transport.close_all()

    threading.Thread(target=watch_cancel, daemon=True).start()
    publish(message="正在索引真实项目代码与分词缓存…")

    raw_archive: list[dict] = []
    try:
        virtual_files = _collect_virtual_files()
        corpus_cache = CorpusCache(virtual_files)

        for idx, step in enumerate(steps):
            transport.check()
            live_cap = _server_info()
            if live_cap["fingerprint"] != spec["capacity"]["fingerprint"]:
                raise ValueError("测速期间推理后端模型或容量发生变化，已安全终止以免混入异构数据")

            concurrency = int(step["concurrency"])
            target = int(step["target"])
            output = int(options["output"])
            desired_turns = int(options["turns"])
            strategy = options["strategy"]
            per_lane_safe = int(step["per_lane"]) - output - int(step["reserve"])

            publish(
                state="preparing",
                phase="prepare",
                current_point=idx + 1,
                total_points=total_points,
                live_progress={"turn": 0, "prefilled": 0, "done": 0, "total": concurrency, "streamed_tokens": 0},
                message=f"第 {idx + 1}/{total_points} 点（{concurrency} 并发 · 目标 {target // 1024}K）· 构造独立 Agent 会话…",
            )

            sandboxes = [OverlaySandbox(virtual_files, corpus_cache.raw_cache) for _ in range(concurrency)]
            point_rows: list[dict] = []
            partial_summary: dict = {"qualified_parallel_rounds": 0}
            try:
                sessions, est_tokens = fit_point_sessions(
                    transport, corpus_cache, live_cap["model"], target, run_id, idx + 1, concurrency, strategy
                )

                max_waves = desired_turns if strategy == "agent_fast" else (desired_turns + 1)
                for wave in range(max_waves):
                    transport.check()
                    if _server_info()["fingerprint"] != spec["capacity"]["fingerprint"]:
                        raise ValueError("测速期间后端容量签名变化，已停止")

                    dp_ranks = max(1, int(live_cap.get("dp_ranks") or 1))
                    prepared_wave = []
                    for lane_idx, sess in enumerate(sessions):
                        use_delivery = (
                            strategy == "agent_fast"
                            or wave >= 1
                            or not sess["active"]
                        )
                        # Keep tool_choice="auto" across all waves so SGLang renders identical tool schemas
                        # at the top of the prompt and preserves 95%+ RadixCache prefix hit in Wave 1+.
                        choice = "auto"
                        if wave >= 1 and use_delivery and (
                            not sess["messages"]
                            or sess["messages"][-1].get("content") != LONG_DELIVERY_PROMPT
                        ):
                            sess["messages"].append({"role": "user", "content": LONG_DELIVERY_PROMPT})

                        if wave == 0:
                            tok_cnt = est_tokens
                        else:
                            tok_cnt, trimmed = bound_session_history_fast(
                                transport,
                                corpus_cache,
                                live_cap["model"],
                                sess,
                                min(target + 16384, per_lane_safe),
                                choice,
                            )
                            sess["trimmed"] = trimmed
                        dp_target = (lane_idx % dp_ranks) if dp_ranks > 1 else None
                        prepared_wave.append((lane_idx, sess, choice, tok_cnt, use_delivery, dp_target))

                    wave_desc = (
                        "冷预填充 + Agent 工具轮"
                        if (wave == 0 and strategy == "agent_auto")
                        else ("冷预填充 + Agent 长代码交付" if wave == 0 else "缓存前缀 + Agent 长代码并发解码")
                    )

                    live_lock = threading.Lock()
                    prefilled_flags = [False] * concurrency
                    first_decode_ts: list[float] = []
                    lane_streamed = [0] * concurrency
                    done_in_wave = [0]
                    wave_active = threading.Event()
                    wave_active.set()

                    def make_first_cb(l_idx: int):
                        def _cb(ts: float) -> None:
                            with live_lock:
                                if not prefilled_flags[l_idx]:
                                    prefilled_flags[l_idx] = True
                                    first_decode_ts.append(ts)
                        return _cb

                    def make_chunk_cb(l_idx: int):
                        def _cb(tok_n: int) -> None:
                            lane_streamed[l_idx] = tok_n
                        return _cb

                    def heartbeat_loop(w_idx: int, desc: str) -> None:
                        while wave_active.is_set() and not transport.cancel.is_set():
                            time.sleep(0.55)
                            if not wave_active.is_set():
                                break
                            with live_lock:
                                pf_cnt = sum(prefilled_flags)
                                earliest_f = min(first_decode_ts) if first_decode_ts else None
                            tok_sum = sum(lane_streamed)
                            done_cnt = done_in_wave[0]
                            live_tps = None
                            if earliest_f is not None and tok_sum > pf_cnt:
                                dt = time.perf_counter() - earliest_f
                                if dt > 0.2:
                                    live_tps = round(max(1, tok_sum - pf_cnt) / dt, 1)
                            cur_phase = "decode" if pf_cnt == concurrency else ("prefill" if w_idx == 0 else "decode")
                            tps_part = f" · 实时 {live_tps:,.1f} tok/s" if live_tps else ""
                            publish(
                                force=False,
                                state="running",
                                phase=cur_phase,
                                live_progress={
                                    "turn": w_idx + 1,
                                    "prefilled": pf_cnt,
                                    "done": done_cnt,
                                    "total": concurrency,
                                    "streamed_tokens": tok_sum,
                                    "live_tps": live_tps,
                                },
                                message=(
                                    f"第 {idx + 1}/{total_points} 点（{concurrency} 并发 · {target // 1024}K）· "
                                    f"第 {w_idx + 1} 波（{desc}）· 首字 {pf_cnt}/{concurrency} · "
                                    f"已完成 {done_cnt}/{concurrency} · 已输出 {tok_sum:,} tok{tps_part}"
                                ),
                            )

                    publish(
                        state="running",
                        phase="prefill" if wave == 0 else "decode",
                        live_progress={
                            "turn": wave + 1,
                            "prefilled": 0,
                            "done": 0,
                            "total": concurrency,
                            "streamed_tokens": 0,
                            "live_tps": None,
                        },
                        message=f"第 {idx + 1}/{total_points} 点 · 第 {wave + 1} 波（{wave_desc}）· 已并发发出 {concurrency} 会话",
                    )

                    hb_thread = threading.Thread(target=heartbeat_loop, args=(wave, wave_desc), daemon=True)
                    hb_thread.start()

                    barrier = threading.Barrier(len(prepared_wave))
                    wave_errors: list[str] = []

                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared_wave)) as pool:
                            future_map = {
                                pool.submit(
                                    stream_agent_request,
                                    transport,
                                    live_cap["model"],
                                    sess["messages"],
                                    output,
                                    barrier,
                                    max(tok_cnt, target),
                                    choice,
                                    make_first_cb(lane_idx),
                                    make_chunk_cb(lane_idx),
                                    dp_target,
                                ): (lane_idx, sess, use_delivery)
                                for lane_idx, sess, choice, tok_cnt, use_delivery, dp_target in prepared_wave
                            }
                            try:
                                for fut in concurrent.futures.as_completed(future_map):
                                    lane_idx, sess, use_delivery = future_map[fut]
                                    try:
                                        row, text, reasoning_text, calls = fut.result()
                                        row.update(
                                            lane=lane_idx + 1,
                                            turn=wave,
                                            sample_phase="code_delivery" if use_delivery else "agent_tools",
                                            history_trimmed=sess["trimmed"],
                                        )
                                        point_rows.append(row)
                                        sess["prev_tokens"] = row["prompt_tokens"] + row["output_tokens"]
                                        sess["prev_msg_len"] = len(sess["messages"]) + 1
                                        if not use_delivery or calls:
                                            sess["active"] = advance_agent_session(
                                                sandboxes[lane_idx],
                                                sess["messages"],
                                                text,
                                                reasoning_text,
                                                calls,
                                                wave,
                                            )
                                            if use_delivery:
                                                sess["active"] = False
                                        else:
                                            sess["messages"].append(
                                                {"role": "assistant", "content": text, "reasoning_content": reasoning_text}
                                            )
                                            sess["active"] = False
                                    except InterruptedError:
                                        raise
                                    except ValueError:
                                        raise
                                    except Exception as exc:
                                        err_text = str(exc)[:240]
                                        wave_errors.append(err_text)
                                        point_rows.append(
                                            {
                                                "ok": False,
                                                "lane": lane_idx + 1,
                                                "turn": wave,
                                                "error": err_text,
                                                "output_tokens": 0,
                                                "requested_output": output,
                                            }
                                        )
                                    done_in_wave[0] += 1
                            except (InterruptedError, ValueError):
                                barrier.abort()
                                transport.close_all()
                                raise
                    finally:
                        wave_active.clear()

                    if done_in_wave[0] == len(wave_errors):
                        raise RuntimeError(wave_errors[0] if wave_errors else "当前波次全部请求失败")

                    partial_summary = {
                        "point": idx + 1,
                        "target": target,
                        "requested_target": step["requested_target"],
                        "complete": False,
                        **aggregate_point(point_rows, concurrency),
                    }
                    publish(partial=partial_summary)

                    if partial_summary["qualified_parallel_rounds"] >= desired_turns:
                        break

                final_point = {
                    "point": idx + 1,
                    "target": target,
                    "requested_target": step["requested_target"],
                    "complete": True,
                    "sampling_complete": partial_summary["qualified_parallel_rounds"] >= desired_turns,
                    **aggregate_point(point_rows, concurrency),
                }
                raw_archive.append({"point": final_point, "rows": point_rows})
                with state_lock:
                    results = list(state.get("results") or [])
                    results.append(final_point)
                publish(results=results, partial=None)
            finally:
                for sb in sandboxes:
                    sb.close()

        result_file = Path(spec["result_file"])
        _write(result_file, {"id": run_id, "spec": spec, "points": raw_archive, "finished_at": time.time()})
        incomplete = sum(1 for item in state.get("results", []) if not item.get("sampling_complete"))
        done_msg = (
            f"Agent 测速完成（共 {len(state.get('results', []))} 点；其中 {incomplete} 点因短回复未满目标长解码轮数）"
            if incomplete
            else f"Agent 测速全部完成（共 {len(state.get('results', []))} 个测试点），可查看各会话明细或导出 JSON/CSV"
        )
        publish(
            state="done",
            phase="complete",
            finished_at=time.time(),
            live_progress=None,
            message=done_msg,
        )
    except Exception as exc:
        cancelled = cancel_path.exists() or isinstance(exc, InterruptedError)
        publish(
            state="stopped" if cancelled else "failed",
            phase="stopped" if cancelled else "failed",
            finished_at=time.time(),
            partial=None,
            live_progress=None,
            message="测速已手动停止；已完成的测试点已保留" if cancelled else f"测速中止：{exc}",
        )
    finally:
        transport.close_all()


class Benchmark:
    def __init__(self):
        self.lock = threading.Lock()
        self.state_file = ROOT / "state.json"
        self.cancel_file = ROOT / "cancel"
        self.proc: subprocess.Popen | None = None

    def _owned_worker(self, pid: int) -> bool:
        return bool(self.proc and self.proc.pid == pid and self.proc.poll() is None) or _safe_worker(pid)

    def plan_preview(self, form: dict) -> dict:
        options = _parse(form)
        try:
            cap = _server_info()
        except ValueError as exc:
            preview = build_plan(options, cap=None)
            preview["offline_reason"] = str(exc)
            return preview
        return build_plan(options, cap=cap)

    def status(self) -> dict:
        data = _read(self.state_file)
        if not data:
            return {
                "state": "idle",
                "phase": "idle",
                "message": "配置并发档位与最高 512K 输入长度，开始 Agent 测速",
                "results": [],
            }
        if data.get("state") in ACTIVE:
            pid = int(data.get("pid") or 0)
            alive = pid > 0 and self._owned_worker(pid)
            if not alive and time.time() - float(data.get("started_at") or 0) > 3.0:
                if data.get("state") == "stopping" or self.cancel_file.exists():
                    data.update(
                        state="stopped",
                        phase="stopped",
                        partial=None,
                        live_progress=None,
                        finished_at=data.get("finished_at") or time.time(),
                        message="测速已停止；已完成测试点已保留",
                    )
                else:
                    log_path = Path(data.get("log_file") or "")
                    tail = _log_tail(log_path) if log_path.name else []
                    data.update(
                        state="interrupted",
                        phase="failed",
                        partial=None,
                        live_progress=None,
                        finished_at=data.get("finished_at") or time.time(),
                        message=tail[-1][:240] if tail else "测速工作进程意外退出，已完成数据已保留",
                    )
                _write(self.state_file, data)

        out = copy.deepcopy(data)
        if out.get("started_at"):
            end_ref = out.get("finished_at") or time.time()
            out["elapsed_s"] = max(0, round(end_ref - out["started_at"]))
        for internal_key in ("result_file", "log_file", "spec_file", "pid"):
            out.pop(internal_key, None)
        return out

    def start(self, form: dict) -> dict:
        with self.lock:
            current = self.status()
            if current["state"] in ACTIVE:
                raise ValueError("已有测速任务正在进行，请先停止或等待完成")

            options = _parse(form)
            cap = _server_info()
            confirmed_fp = _first_val(form, "fingerprint", "").strip()
            if confirmed_fp and confirmed_fp != cap["fingerprint"]:
                raise ValueError("在线模型或 KV 容量刚刚发生变化，已自动刷新容量，请重新点击开始")

            spec = build_plan(options, cap=cap)
            if not PYTHON.is_file():
                raise ValueError("服务器缺少 Python 运行环境")
            try:
                if not PROXY_KEY_FILE.read_text(encoding="utf-8").strip():
                    raise ValueError("系统代理密钥为空")
            except OSError as exc:
                raise ValueError("API 代理密钥未就绪，请先检查代理服务") from exc

            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
            spec_dir = ROOT / "runs"
            log_dir = ROOT / "logs"
            res_dir = ROOT / "results"
            for directory in (spec_dir, log_dir, res_dir):
                directory.mkdir(parents=True, exist_ok=True)

            self.cancel_file.unlink(missing_ok=True)
            spec_file = spec_dir / f"{run_id}.spec.json"
            result_file = res_dir / f"{run_id}.json"
            log_file = log_dir / f"{run_id}.log"

            worker_spec = {
                **spec,
                "id": run_id,
                "proxy_host": HOST,
                "proxy_port": PORT,
                "proxy_key_file": str(PROXY_KEY_FILE),
                "result_file": str(result_file),
            }
            _write(spec_file, worker_spec)

            argv = [
                str(PYTHON),
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--spec",
                str(spec_file),
                "--state",
                str(self.state_file),
                "--cancel",
                str(self.cancel_file),
            ]
            env = dict(os.environ)
            for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env.pop(proxy_var, None)
            env["NO_PROXY"] = env["no_proxy"] = ",".join((HOST, "127.0.0.1", "localhost"))

            initial_state = {
                "id": run_id,
                "state": "preparing",
                "phase": "prepare",
                "message": "正在启动 Agent 测速引擎并加载分词器…",
                "options": options,
                "spec": {k: v for k, v in spec.items() if k != "options"},
                "capacity": cap,
                "started_at": time.time(),
                "current_point": 1,
                "total_points": spec["points"],
                "results": [],
                "partial": None,
                "live_progress": None,
                "spec_file": str(spec_file),
                "result_file": str(result_file),
                "log_file": str(log_file),
            }
            _write(self.state_file, initial_state)

            with log_file.open("wb") as log_stream:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT if REPO_ROOT.is_dir() else Path(__file__).resolve().parent),
                    env=env,
                    start_new_session=True,
                )
            self.proc = proc
            initial_state["pid"] = proc.pid
            _write(self.state_file, initial_state)
            threading.Thread(target=self._reap, args=(proc, run_id), daemon=True).start()
            return self.status()

    def _reap(self, proc: subprocess.Popen, run_id: str) -> None:
        code = proc.wait()
        with self.lock:
            data = _read(self.state_file)
            if data.get("id") != run_id or data.get("state") not in ACTIVE:
                return
            if self.cancel_file.exists() or data.get("state") == "stopping":
                data.update(
                    state="stopped",
                    phase="stopped",
                    partial=None,
                    live_progress=None,
                    message="测速已停止；已完成测试点已保留",
                )
            else:
                tail = _log_tail(Path(data.get("log_file") or ""))
                data.update(
                    state="failed",
                    phase="failed",
                    partial=None,
                    live_progress=None,
                    message=tail[-1][:240] if tail else f"测速进程异常退出（code {code}）",
                )
            data["finished_at"] = time.time()
            _write(self.state_file, data)

    def stop(self, expected_id: str | None = None) -> dict:
        with self.lock:
            data = _read(self.state_file)
            if expected_id and data.get("id") and data.get("id") != expected_id:
                raise ValueError("测速任务已变化，请刷新页面")
            if data.get("state") not in ACTIVE:
                return self.status()
            self.cancel_file.parent.mkdir(parents=True, exist_ok=True)
            self.cancel_file.touch()
            data.update(state="stopping", message="正在中断流式请求并保留已完成测试点…")
            _write(self.state_file, data)

            pid = data.get("pid")
            if isinstance(pid, int) and pid > 0 and self._owned_worker(pid):
                def _escalate_kill(target_pid: int, target_id: str) -> None:
                    time.sleep(3.0)
                    with self.lock:
                        cur = _read(self.state_file)
                        if cur.get("id") == target_id and cur.get("state") in ACTIVE and self._owned_worker(target_pid):
                            try:
                                os.killpg(target_pid, signal.SIGTERM)
                            except OSError:
                                pass

                threading.Thread(target=_escalate_kill, args=(pid, data.get("id", "")), daemon=True).start()
            return self.status()

    def clear(self, expected_id: str | None = None) -> dict:
        with self.lock:
            current = self.status()
            if current["state"] in ACTIVE:
                raise ValueError("测速进行中，请先停止测速再清除结果")
            if expected_id and current.get("id") and current.get("id") != expected_id:
                raise ValueError("本轮数据已更新，请刷新页面后重试")
            self.cancel_file.unlink(missing_ok=True)
            empty = {
                "state": "idle",
                "phase": "idle",
                "message": "已清除本轮曲线与明细，可重新配置并开始测速",
                "results": [],
            }
            _write(self.state_file, empty)
            return empty


def page_html() -> str:
    return tf500_benchmark_ui.page_html()


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--cancel", required=True)
    args = parser.parse_args()
    if args.worker:
        run_worker(Path(args.spec), Path(args.state), Path(args.cancel))


if __name__ == "__main__":
    _cli_main()
