# xtier_86kd

DeepSeek V4.1 Flash 1017 acceleration for SGLang tensor parallelism 8 on eight RTX PRO 6000D GPUs. This project ports the DeepSeek pack from [XTier](https://github.com/a775828b-dot/xtier) without enabling XTier's hybrid MoE engine. SGLang's Humming MoE runner remains in use.

The complete DeepSeek pack is in `src/xtier_86kd/packs/deepseek_v4`. Its applicable hooks cover DSpark verification, paged decode indexing, FP8 GEMM tuning, FlashInfer MXFP8 small batch, attention `wo_a`, and draft FP8 LM head. Current SGLang already selects dense-block candidates on SM120. The legacy candidate override is retained behind `XTIER_86KD_LEGACY_INDEXER=1` for older commits and must stay off on the tested commit. Legacy GPU hooks and the optional Engram NVMe row store are retained. The row store's C++ source is in `csrc/tables`; it is inactive with the default host Engram table.

## Install on the server

```bash
/data/xtier-env/bin/pip install -e /data/src/xtier_86kd --no-deps
```

Set `XTIER=0` and `XTIER_86KD=1` in the existing SGLang startup environment. Existing `XTIER_DS_*`, `XTIER_MXFP8_B12X_ROWS`, `XTIER_DRAFT_LMHEAD_FP8`, and related pack switches remain supported. Use the same SGLang commit and serving parameters as the baseline for an A/B benchmark. `XTIER_86KD=0` removes all hooks without reinstalling.

The FP8 tuned config JSONs were measured for RTX PRO 5000; they remain in the pack but do not match the 6000D device name. New 6000D tuning must be measured before adding configs.

The optional NVMe Engram row store requires a small native library. Build it only when using `XTIER_ENGRAM_ROWS=nvme`:

```bash
cmake -S csrc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
mkdir -p src/xtier_86kd/lib
cp build/libxtier_tables.so src/xtier_86kd/lib/
```

For this 1 TiB RAM host, leave `XTIER_ENGRAM_ROWS` unset. SGLang stores Engram on the CPU and the other model weights on the eight GPUs.

## Benchmarks

`bench/bench_serving.py` measures concurrent streaming requests. For the development sweep, use `--two-phase --shared-decode-prefix --prompt-tokens 32768 --output-tokens 1024 --concurrencies 1,4,8,16,32 --rounds 1` with the same model, server parameters, and corpus. Each round gets a unique prefix to prevent reuse of earlier KV cache entries. The first phase uses independent cold prompts and one output token to measure prefill. The second phase seeds one 32K corpus and has the requests share it while producing 1024 tokens; this is needed because the tested server retains only about 20 independent 32K prefixes even with a reported 5,247,744-token full KV pool. `prefill_window_tps_est` is total cold input tokens divided by the interval from the first request start to the last first token. `decode_window_tps` is total output tokens divided by the interval from the first to last streamed output token. `sum_request_decode_tps_est` sums each request's own active decode rate. The script reads each response's `spec_verify_ct`, falling back to the `spec_verify_calls_total` metric, to calculate accepted tokens per step and `mean_request_ms_per_step_est` from the sum of each request's decode duration. That elapsed-time step estimate includes scheduling gaps, not just the GPU kernel. Report success counts and cached token counts with all rates.

`bench/agent_bench.py` is adapted from XTier's coding-agent benchmark. It supports `--api-key-file` for this server and stores only the file path, never the secret. Supply `--repo /data/src/xtier --lsglang /data/src/sglang-xtier --tokenizer /data/models/DeepSeek-V4.1-Flash-1017`. It reports per-request cold prefill, cached TTFT, decode throughput, and DSpark verify step time. For compact verification, build an SPS cost table on this machine with `bench/sps_profile.py` before enabling its table path; a table from another GPU is not valid calibration.

The measured development sweep and its raw JSON files are in [`bench/32k1k-ab.md`](bench/32k1k-ab.md). The report includes prefill and cached decode throughput, accepted tokens per verification step, per-request milliseconds per step, and full KV pool size for 1, 4, 8, 16, and 32 requests. The machine-specific SPS table is [`bench/86kd-sps-6000d.json`](bench/86kd-sps-6000d.json). On tf500, setting `SGLANG_RAGGED_VERIFY_MODE=compact` and `SPS_TABLE=/data/bench/86kd-sps-6000d.json` enables compact verification with the measured table; the existing launch script maps these to the SGLang flags. The baseline and acceleration sweep used one round per concurrency, so repeat with fixed prompts before treating small differences as robust.

Source license: Apache-2.0; see `LICENSE`.

