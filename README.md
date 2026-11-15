# xtier_86kd

## tf500 TP4 混合推理

`deploy/start-xtier-tp4.sh` 在 GPU 0–3 与 CPU/NUMA 0 上运行原版 XTier 混合专家引擎，独立监听 `30001`。它不启用本仓库的 TP8 加速插件；原 TP8 启动台卡片和服务配置保留。配置、单 NUMA TP4 补丁及运行检查见 [`deploy/README-tf500-tp4.md`](deploy/README-tf500-tp4.md)，单发验收见 [`deploy/tf500-tp4-single.md`](deploy/tf500-tp4-single.md)。

DeepSeek V4.1 Flash 1017 acceleration for SGLang tensor parallelism 8 on eight RTX PRO 6000D GPUs. This project ports the DeepSeek pack from [XTier](https://github.com/a775828b-dot/xtier) without enabling XTier's hybrid MoE engine. SGLang's Humming MoE runner remains in use.

The complete DeepSeek pack is in `src/xtier_86kd/packs/deepseek_v4`. Its applicable hooks cover DSpark verification, paged decode indexing, FP8 GEMM tuning, FlashInfer MXFP8 small batch, attention `wo_a`, and draft FP8 LM head. Current SGLang already selects dense-block candidates on SM120. The legacy candidate override is retained behind `XTIER_86KD_LEGACY_INDEXER=1` for older commits and must stay off on the tested commit. Legacy GPU hooks and the optional Engram NVMe row store are retained. The row store's C++ source is in `csrc/tables`; it is inactive with the default host Engram table.

## Install on the server

```bash
/data/xtier-env/bin/pip install -e /data/src/xtier_86kd --no-deps
```

Set `XTIER=0` and `XTIER_86KD=1` in the existing SGLang startup environment. Existing `XTIER_DS_*`, `XTIER_MXFP8_B12X_ROWS`, `XTIER_DRAFT_LMHEAD_FP8`, and related pack switches remain supported. Use the same SGLang commit and serving parameters as the baseline for an A/B benchmark. `XTIER_86KD=0` removes all hooks without reinstalling.

The FP8 tuned config JSONs were measured for RTX PRO 5000; they remain in the pack but do not match the 6000D device name. New 6000D tuning must be measured before adding configs.

## tf500 WebUI

`deploy/tf500_console.py` reuses the XTier console page and controls this host's TP8 SGLang launch script. It detects the existing `sglang-ds41.service` without restarting the model. The page listens only on `127.0.0.1:8100`; connect from a workstation with `ssh -L 8100:127.0.0.1:8100 <user>@<TF500_IP>`, then open `http://127.0.0.1:8100/`.

The page shows the current system service, GPU status and recent decode logs. Its defaults reproduce the present 86KD + DSpark compact configuration. Start first terminates the current instance, waits for GPU memory to release, then starts `/data/src/start-sglang-plan.sh` with the selected settings. Stop terminates the current instance. The API key is read from `/data/sglang-api-key` only by the launch script and authenticated health probe; the WebUI does not return it. A console-started model runs in an independent user session, while the original system service still starts with its default settings after a host reboot. A console-started model does not inherit the system service's `Restart=on-failure` policy.

The installed launch entry is `$HOME/start-xtier-86kd-console.sh`, registered in the `<user>` crontab with `@reboot`. Console logs and state live under `/data/xtier-86kd-console/`. No external WebUI dependencies are required beyond the existing XTier installation.

The WebUI has four separate pages selected from navigation cards: the launch console `/`, API benchmark `/benchmark`, API proxy settings `/proxy`, and read-only server information `/server`. The server page refreshes every five seconds and shows per-GPU utilization, VRAM, power, temperature and clock, plus CPU, memory, disk, network, uptime and model state. The benchmark supports 1, 4, 8, 16, and 32 concurrent requests. It uses 4K, 8K, or 32K input, 256, 512, or 1024 output tokens, and one or two rounds. The capacity check reads the backend directly; inference requests use the API proxy. The two-phase run clears the cache between cold prefill and shared-prefix decode, so use it when no other inference work is active.

The separate API proxy listens on `<TF500_IP>:8000`. Clients use `http://<TF500_IP>:8000/v1` and a Bearer key from `/data/xtier-86kd-console/proxy-api-key` (created with mode 0600 on first start). The proxy forwards only `/v1/` paths, accepts `Authorization: Bearer` or `x-api-key`, rewrites authentication to the backend key, streams responses, and records request metadata without prompts, responses, keys, or client addresses. `GET /health` reports whether the backend is reachable. The proxy starts on reboot via `deploy/start-api-proxy.sh`; the backend keeps its own lifecycle.

For an OpenAI-compatible client, set Base URL to `http://<TF500_IP>:8000/v1`, model to `DeepSeek-V4.1-Flash-1017`, and API key to the contents of `/data/xtier-86kd-console/proxy-api-key` on this host. The proxy key is separate from `/data/sglang-api-key`. Retrieve it over SSH with `ssh <user>@<TF500_IP> 'cat /data/xtier-86kd-console/proxy-api-key'`; keep it out of shared logs and screenshots.

The WebUI links to `/proxy`, a loopback-only API proxy configuration page. It shows the endpoint, model, proxy/backend status, self-test, per-key and recent request usage. Create client keys there with a name, expiry (permanent, 1, 7, 30, or 90 days), and a request-count quota (0 means unlimited). The new key is shown once; save it then. Deleting a client key invalidates it immediately. The original `proxy-api-key` remains a permanent system key for internal benchmarking and is never shown in full on the page. Client keys are stored in `/data/xtier-86kd-console/proxy-keys.json` with mode 0600; the 8000 proxy reads changes on each request, without restarting. Quotas are measured in request counts; no currency pricing is applied.

The model-independent SM120 FlashInfer MXFP8 prefill tile hook is enabled by default in 86KD. It selects tactic 4 only for fallback CUTLASS MXFP8 calls with at least 256 rows and the expected six-tactic runner. SGLang pins DeepSeek's MXFP8 calls, so the pinned exception is limited to TP8's (K=288, N=5120) expert down projection. On this RTX 6000D, tactic 4 and the pinned fallback were bitwise equal at M=256/512/1024/2048; M=2048's isolated GEMM time was 0.0345 vs 0.0356 ms (one short run, not a serving speed claim). Set `XTIER_86KD_MXFP8_PREFILL_TILE=0` to turn it off. The existing 86KD compact 32K/1K results remain the development baseline; enabling this hook does not change those recorded measurements.

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
