"""Summarize XTier CPU callbacks during the single decode request."""

import argparse
import json
import statistics
import struct
from pathlib import Path

ROOT = Path("/data")
parser = argparse.ArgumentParser()
parser.add_argument('--json', default='/data/bench/xtier-tp4-single.json')
options = parser.parse_args()
run = json.loads(Path(options.json).read_text())
window = run["single"]
head = struct.Struct("<QIIIHBBhBBI")
results = []

for path in sorted(ROOT.glob("xtier-tp4-route.bin.*")):
    data = path.read_bytes()
    events = []
    pos = 0
    while pos + head.size <= len(data):
        t, wall, gate, down, layer, m, k, groups, active, _, seq = head.unpack_from(data, pos)
        pos += head.size + m * k * 4
        if window["start_epoch_ns"] <= t <= window["end_epoch_ns"] and layer < 40:
            events.append((t, wall, gate, down, layer, m, active, seq))
    steps = []
    current = []
    for event in events:
        if event[4] == 0 and current:
            steps.append(current)
            current = []
        current.append(event)
    if current:
        steps.append(current)
    complete = [s for s in steps if len(s) == 40 and [e[4] for e in s] == list(range(40))]
    if len(complete) < 5:
        continue
    # First and last groups can be partial; use complete sequential groups only.
    # TraceHead stores wall/gate/down in nanoseconds.
    cpu_ms = [sum(e[1] for e in s) / 1e6 for s in complete]
    cpu_call_ms = [e[1] / 1e6 for s in complete for e in s if e[6] > 0]
    gate_ms = [sum(e[2] for e in s) / 1e6 for s in complete]
    down_ms = [sum(e[3] for e in s) / 1e6 for s in complete]
    spans_ms = [(s[-1][0] + s[-1][1] - s[0][0]) / 1e6 for s in complete]
    starts = [s[0][0] for s in complete]
    periods_ms = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
    percentile = lambda values, fraction: sorted(values)[int((len(values) - 1) * fraction)]
    results.append({
        "file": path.name,
        "complete_steps": len(complete),
        "rows_per_cpu_call": sorted(set(e[5] for s in complete for e in s)),
        "active_rows_per_cpu_call": sorted(set(e[6] for s in complete for e in s)),
        "cpu_sum_ms_median": round(statistics.median(cpu_ms), 3),
        "cpu_call_ms_median": round(statistics.median(cpu_call_ms), 3),
        "cpu_sum_ms_mean": round(statistics.mean(cpu_ms), 3),
        "cpu_gate_ms_median": round(statistics.median(gate_ms), 3),
        "cpu_down_ms_median": round(statistics.median(down_ms), 3),
        "first_to_last_cpu_span_ms_median": round(statistics.median(spans_ms), 3),
        "between_cpu_calls_ms_median": round(statistics.median([span - cpu for span, cpu in zip(spans_ms, cpu_ms)]), 3),
        "verify_start_period_ms_median": round(statistics.median(periods_ms), 3),
        "verify_start_period_ms_mean": round(statistics.mean(periods_ms), 3),
        "verify_start_period_ms_p90": round(percentile(periods_ms, .9), 3),
        "verify_start_period_ms_p99": round(percentile(periods_ms, .99), 3),
        "periods_over_100ms": sum(p > 100 for p in periods_ms),
        "cpu_fraction_of_period_median": round(statistics.median(
            a / b for a, b in zip(cpu_ms, periods_ms) if b > 0), 3),
    })

print(json.dumps(results, ensure_ascii=False, indent=2))
