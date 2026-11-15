#!/usr/bin/env python3
"""Render the tf500 32K/1K development A/B sweep from raw JSON files."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = (1, 4, 8, 16, 32)
SOURCES = {
    "原版静态": ("serving-32k1k-baseline-shared-c1-c16.json",
                 "serving-32k1k-baseline-c32-shared.json"),
    "86KD 静态": ("serving-32k1k-86kd-static.json",
                  "serving-32k1k-86kd-static-c1-repeat.json"),
    "86KD 变长": ("serving-32k1k-86kd-compact.json",
                  "serving-32k1k-86kd-compact-c1-repeat.json"),
}
KV_POOL = {"原版静态": 5_247_744, "86KD 静态": 5_189_888,
           "86KD 变长": 5_189_888}


def read_mode(paths):
    rows = {}
    for filename in paths:
        record = json.loads((HERE / filename).read_text(encoding="utf-8"))
        for item in record["rounds"]:
            summary = item["summary"]
            rows[summary["concurrency"]] = summary
    if set(rows) != set(CASES):
        raise ValueError(f"missing concurrency in {paths}: {set(CASES) - set(rows)}")
    return rows


data = {name: read_mode(paths) for name, paths in SOURCES.items()}
lines = [
    "# tf500 DeepSeek V4.1 Flash 1017：32K/1K 开发测速",
    "",
    "测试日期：2026-11-15。RTX PRO 6000D ×8、SGLang TP8、Humming MoE、DSpark。",
    "XTIER 混合推理保持关闭；86KD 仅安装独立加速钩子。",
    "",
    "第一阶段各请求独立冷 32K 提示，只输出 1 token；第二阶段清缓存、种入公共 32K 前缀，",
    "各请求共享前缀输出 1K。P 是第一阶段总输入除以首发到末首字的窗口，",
    "D 是第二阶段总输出除以首字到末输出的窗口。步长是每个 DSpark 验证步接受的 token；",
    "ms/步是每请求活跃解码时长之和除以验证步总数，包含调度间隔，不等于单个 GPU 核耗时。",
    "单次开发扫档，提示含各次运行标识，故模式间的生成文本与接受率不完全相同；数值用于筛选，",
    "并非固定提示的统计显著性结论。单发在 86KD 两组使用加载稳定后的复测值。",
    "",
    "| 模式 | 并发 | 冷 P tok/s | 缓存 D tok/s | token/步 | ms/步 | 冷 TTFT p95 s | 缓存% | 成功 | full KV token |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name, summaries in data.items():
    for concurrency in CASES:
        row = summaries[concurrency]
        if row["success"] != concurrency or row["prefill_success"] != concurrency:
            raise ValueError(f"unsuccessful run: {name} c{concurrency}")
        lines.append(
            f"| {name} | {concurrency} | {row['prefill_window_tps_est']:,.0f} | "
            f"{row['decode_window_tps']:,.0f} | {row['accept_tokens_per_step_est']:.2f} | "
            f"{row['mean_request_ms_per_step_est']:.2f} | {row['p95_cold_ttft_s']:.2f} | "
            f"{row['decode_cached_pct']:.2f} | {row['success']}/{concurrency} | "
            f"{KV_POOL[name]:,} |"
        )

lines.extend(("", "| 并发 | 86KD 静态 D 对原版 | 86KD 变长 D 对原版 | 变长 D 对 86KD 静态 |",
              "| ---: | ---: | ---: | ---: |"))
for concurrency in CASES:
    baseline = data["原版静态"][concurrency]["decode_window_tps"]
    static = data["86KD 静态"][concurrency]["decode_window_tps"]
    compact = data["86KD 变长"][concurrency]["decode_window_tps"]
    lines.append(f"| {concurrency} | {(static / baseline - 1) * 100:+.1f}% | "
                 f"{(compact / baseline - 1) * 100:+.1f}% | "
                 f"{(compact / static - 1) * 100:+.1f}% |")

lines.extend((
    "", "冷 P 的缓存 token 均为 0；缓存 D 的公共前缀命中约 99.15%。",
    "full KV 为 SGLang 的 `max_total_num_tokens` 指标，不代表每条独立长前缀都能留在 radix 缓存。",
    "86KD 的 full KV 池比原版少 57,856 token（约 1.10%）。三组正式扫档无请求失败，服务无自动重启。",
    "首次加载后的 86KD 静态/变长单发冷 P 分别为 4,212/4,325 tok/s；",
    "热机复测分别为表中的 12,425/12,386 tok/s。",
    "", "原始 JSON 与本报告同目录；SPS 标定表为 `86kd-sps-6000d.json`。",
))
(HERE / "32k1k-ab.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(HERE / "32k1k-ab.md")

