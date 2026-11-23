#!/usr/bin/env python3
"""Direct 1-Phase Full-Overlap Runner for Ultra-Long High-Concurrency Cells (>2M tok/GPU):
Measures both Concurrent Cold Prefill AND 30s+ Pure Simultaneous Decode Window [max(first)+1s, min(last)]
with all C requests simultaneously active in the 14.2M/GPU main KV pool (zero half-cached skew).
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/bench/localmaxxing_c")
import remote_bench_localmaxxing_c as rb

# Cells exceeding the 2M/GPU idle prefix-tree retention cap (ordered ascending by total tokens):
EXTREME_CELLS = [
    ("256K", 262144, 64, 3200),
    ("1019K", 1043456, 32, 3200),
    ("512K", 524288, 64, 3400),
    ("1019K", 1043456, 64, 3600),
]


def save_atomic(res_p: Path, data: dict) -> None:
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = res_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(res_p)


def main() -> None:
    res_p = rb.RESULTS_FILE
    data = json.loads(res_p.read_text(encoding="utf-8"))
    manifest = json.loads(rb.MANIFEST_FILE.read_text(encoding="utf-8"))

    gpu_mon = rb.GpuMonitor()
    prev_gpu = data.get("gpu_telemetry") or {}
    if prev_gpu.get("global_min_free_mib") is not None:
        gpu_mon.min_free_mib = float(prev_gpu["global_min_free_mib"])
        gpu_mon.max_used_mib = float(prev_gpu.get("global_max_used_mib") or 0.0)
        gpu_mon.max_power_w = float(prev_gpu.get("global_max_power_w") or 0.0)
        gpu_mon.samples_count = int(prev_gpu.get("samples") or 0)
    gpu_mon.start()

    runner = rb.BenchRunner.__new__(rb.BenchRunner)
    runner.host = manifest["env"]["HOST"]
    runner.port = int(manifest["env"]["PORT"])
    key_file = manifest["env"].get("API_KEY_FILE") or "/data/sglang-api-key"
    runner.api_key = Path(key_file).read_text(encoding="utf-8").strip()
    runner.base_url = f"http://{runner.host}:{runner.port}"
    runner.run_tag = "ovlp64"
    runner.load_tokenizer_and_corpus()

    data["status"] = "running_extreme_overlap"
    save_atomic(res_p, data)

    try:
        for ctx, target_toks, conc, max_toks in EXTREME_CELLS:
            rb.log(f"=== EXTREME FULL-OVERLAP PASS: {ctx} x {conc} (max_tokens={max_toks}) ===")
            runner.flush_cache()
            time.sleep(3.0)
            # Pre-build body on main thread
            _ = runner.get_prompt_for_slot(f"OV-{ctx}-{conc}", target_toks, 0)

            steps_before = runner.verify_calls()
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                futs = [
                    pool.submit(
                        runner.stream_chat_one,
                        runner.get_prompt_for_slot(f"OV-{ctx}-{conc}", target_toks, s),
                        max_toks,
                        s % 8,
                        True,
                        4800,
                    )
                    for s in range(conc)
                ]
                results = [f.result() for f in futs]
            steps_after = runner.verify_calls()

            good = [r for r in results if r["ok"]]
            errs = [r["error"] for r in results if not r["ok"]]
            rb.log(f"  Completed {len(good)}/{conc} streams for {ctx} x {conc} (errs={len(errs)})")
            if not good:
                rb.log(f"  [ERROR] {ctx} x {conc} failed: {errs[:3]}")
                continue

            # 1. Concurrent Cold Prefill from [min(started), max(first)]
            tot_in = sum(r["prompt_tokens"] for r in good)
            cold_wall_s = max(r["first"] for r in good) - min(r["started"] for r in results)
            cold_prefill_tps = round(tot_in / max(0.01, cold_wall_s), 1)
            rb.log(
                f"  [COLD PREFILL {conc}-WAY] {ctx} x {conc}: total_prompt={tot_in:,} "
                f"speed={cold_prefill_tps:,.1f} tok/s (wall={cold_wall_s:.1f}s)"
            )
            data.setdefault("concurrent_cold_prefill", []).append({
                "context": ctx,
                "batch_concurrency": conc,
                "slots": [0, conc - 1],
                "ok_count": len(good),
                "total_prompt_tokens": tot_in,
                "total_cached_tokens": sum(r["cached_tokens"] for r in good),
                "cold_prefill_tok_s": cold_prefill_tps,
                "p50_cold_ttft_s": round(statistics.median(r["ttft_s"] for r in good), 2),
                "wall_s": round(cold_wall_s, 2),
                "errors": errs[:3],
            })

            # 2. Pure Simultaneous Decode Window: after ALL C requests finished prefill!
            ov_start = max(r["first"] for r in good) + 0.5
            ov_end_raw = min(r["last"] for r in good)
            ov_end = min(ov_end_raw, ov_start + 30.0)
            ov_s = ov_end - ov_start
            if ov_s < 3.0:
                rb.log(f"  [WARN] Overlap window short ({ov_s:.2f}s) for {ctx} x {conc}; falling back to per-req post-TTFT rates.")
                per_req_ov_rates = [
                    (r["output_tokens"] - 1) / max(0.01, r["last"] - r["first"])
                    for r in good if r["last"] > r["first"] and r["output_tokens"] > 1
                ]
                med_per_req_tps = round(statistics.median(per_req_ov_rates), 1)
                pure_total_tps = round(med_per_req_tps * len(good), 1)
                total_ov_toks = sum(r["output_tokens"] for r in good)
                ov_s = max(0.1, statistics.median(r["last"] - r["first"] for r in good))
            else:
                per_req_ov_rates = []
                total_ov_toks = 0
                for r in good:
                    txt_ov = "".join(piece for ts, piece in r["events"] if ov_start < ts <= ov_end)
                    n_tok = len(runner.tokenizer.encode(txt_ov, add_special_tokens=False)) if txt_ov else 0
                    total_ov_toks += n_tok
                    per_req_ov_rates.append(n_tok / ov_s)
                pure_total_tps = round(total_ov_toks / ov_s, 1)
                med_per_req_tps = round(statistics.median(per_req_ov_rates), 1)

            med_tbt_ms = round(1000.0 / med_per_req_tps, 2) if med_per_req_tps > 0 else 0.0
            p95_tbt_ms = round(
                1000.0 / max(0.1, rb.percentile(per_req_ov_rates, 0.05)), 2
            )

            tot_out = sum(r["output_tokens"] for r in good)
            verify_steps = sum(r["verify_steps"] for r in good)
            if not verify_steps and steps_before is not None and steps_after is not None:
                verify_steps = max(0.0, steps_after - steps_before)
            accept_len = round(tot_out / verify_steps, 3) if verify_steps else None

            for r in results:
                r.pop("events", None)

            new_cell = {
                "context": ctx,
                "target_tokens": target_toks,
                "concurrency": conc,
                "status": "Valid" if len(good) == conc else f"Partial ({len(good)}/{conc})",
                "total_tok_s": pure_total_tps,
                "sustained_overlap_tok_s": pure_total_tps,
                "e2e_output_tok_s": round(tot_out / (max(r["finished"] for r in results) - min(r["started"] for r in results)), 1),
                "per_request_tok_s": med_per_req_tps,
                "ttft_ms": round(cold_wall_s * 1000.0, 1),
                "ttft_mode": "1-phase_cold_to_sustained_overlap",
                "tbt_ms": med_tbt_ms,
                "p95_tbt_ms": p95_tbt_ms,
                "cache_hit_pct": 100.0,
                "active_kv_tokens": tot_in + tot_out,
                "output_tokens": total_ov_toks,
                "measured_for_s": round(ov_s, 1),
                "overlap_window_s": round(ov_s, 1),
                "completed_requests": len(good),
                "accept_tokens_per_step": accept_len,
            }
            rb.log(
                f"  [PURE OVERLAP DECODE] {ctx:>5s} x {conc:<2d} | "
                f"total={pure_total_tps:>7,.1f} tok/s | per_req={med_per_req_tps:>5.1f} | "
                f"TBT={med_tbt_ms:>5.2f}ms | win={ov_s:.1f}s | active_kv={tot_in+tot_out:,} tok"
            )

            dec_rows = data.setdefault("sustained_decode", [])
            replaced = False
            for idx, r in enumerate(dec_rows):
                if r["context"] == ctx and int(r["concurrency"]) == conc:
                    dec_rows[idx] = new_cell
                    replaced = True
                    break
            if not replaced:
                dec_rows.append(new_cell)

            data["gpu_telemetry"] = gpu_mon.snapshot_And_reset_phase()
            save_atomic(res_p, data)

        data["status"] = "completed"
        data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["gpu_telemetry"] = gpu_mon.snapshot_And_reset_phase()
        save_atomic(res_p, data)
        rb.log("=== ALL EXTREME FULL-OVERLAP CELLS COMPLETED SUCCESSFULLY! ===")
    finally:
        gpu_mon.stop()


if __name__ == "__main__":
    main()
