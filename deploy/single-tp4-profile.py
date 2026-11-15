"""One sequential 32K/1K TP4 probe, with a short CUDA profile of decode."""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from transformers import AutoTokenizer

BASE = "http://<TF500_IP>:30001"
MODEL = "DeepSeek-V4.1-Flash-1017"
parser = argparse.ArgumentParser()
parser.add_argument("--no-profile", action="store_true")
options = parser.parse_args()
OUT = Path("/data/bench/xtier-tp4-single-plain.json" if options.no_profile
           else "/data/bench/xtier-tp4-single.json")
PROFILE = "/data/bench/xtier-tp4-single-profile"
key = Path("/data/sglang-api-key").read_text().strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def request(path, data=None, method=None, timeout=60):
    raw = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, raw, headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def verify_metric():
    text = request("/metrics")
    return sum(float(x) for x in re.findall(
        r"^sglang:spec_verify_calls_total\{[^}]*\}\s+([0-9.e+]+)$", text, re.M
    ))


def generation(prompt, count):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": count,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_spec_tokens_details": True,
        "return_cached_tokens_details": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started_ns = time.time_ns()
    started = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), headers)
    first = last = None
    usage = details = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            details = (event.get("sglext") or {}).get("spec_tokens_details") or details
            if any((c.get("delta") or {}).get("content") or
                   (c.get("delta") or {}).get("reasoning_content")
                   for c in event.get("choices") or []):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    finished = time.perf_counter()
    tokens = int(usage.get("completion_tokens") or 0)
    return {
        "start_epoch_ns": started_ns,
        "end_epoch_ns": time.time_ns(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "output_tokens": tokens,
        "verify_steps": details.get("spec_verify_ct"),
        "first_token_s": round(first - started, 4) if first else None,
        "decode_s": round(last - first, 4) if first and last else None,
        "decode_tps": round((tokens - 1) / (last - first), 2)
        if first and last and last > first and tokens > 1 else None,
        "ms_per_verify_step": round(1000 * (last - first) / int(details["spec_verify_ct"]), 3)
        if first and last and details.get("spec_verify_ct") else None,
        "stream_chunks": chunks,
        "wall_s": round(finished - started, 4),
    }


@contextmanager
def gpu_sampler():
    path = OUT.with_suffix(".gpu.csv")
    with path.open("w") as stream:
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=stream, stderr=subprocess.DEVNULL,
        )
        try:
            yield
        finally:
            proc.terminate()
            proc.wait(timeout=10)


OUT.parent.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("/data/models/DeepSeek-V4.1-Flash-1017", trust_remote_code=True)
ids = tok.encode(Path("/data/bench/sglang-corpus.txt").read_text(errors="ignore")[:1_000_000],
                 add_special_tokens=False)
assert len(ids) >= 32768, f"corpus has only {len(ids)} tokens"
prompt = "TP4 single diagnostic: " + tok.decode(ids[:32768])
with gpu_sampler():
    request("/flush_cache", method="POST")
    cold = generation(prompt, 1)
    print("cold:", json.dumps({k: cold[k] for k in ("prompt_tokens", "cached_tokens", "first_token_s")}), flush=True)
    before = verify_metric()
    if not options.no_profile:
        request("/start_profile", {
            "output_dir": PROFILE,
            "num_steps": 24,
            "activities": ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": False,
        })
    single = generation(prompt, 1024)
    after = verify_metric()
single["verify_steps_metric_delta_all_ranks"] = after - before
result = {"cold": cold, "single": single, "profile_dir": None if options.no_profile else PROFILE}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
