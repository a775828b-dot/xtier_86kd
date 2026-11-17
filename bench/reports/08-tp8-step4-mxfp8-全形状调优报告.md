# 08. TF500 TP8 Step 4（SM120 全形状 MXFP8 最优路由、草稿 MoE `MXMMA` 与显存水位加固）实测报告

- **日期**：2026-11-17
- **测试平台**：`tf500`（`8 × NVIDIA RTX PRO 6000D` SM120，Attention DP8 + TP8 + DSpark `block=5, draft=6` `compact`）
- **前置里程碑**：`sd-popt4b-s123`（Step 1–3，里程碑 #3）
- **结构化原始证据**：[`step4-mxfp8-graph-sweep.json`](data/step4-mxfp8-graph-sweep.json)、[`sd-popt5-analysis.json`](data/sd-popt5-analysis.json)
- **修改模块**：[`src/xtier_86kd/mxfp8_prefill.py`](../../src/xtier_86kd/mxfp8_prefill.py)、[`src/xtier_86kd/packs/deepseek_v4/__init__.py`](../../src/xtier_86kd/packs/deepseek_v4/__init__.py)、[`deploy/prefill-opt/prefill_opt_hook.py`](../../deploy/prefill-opt/prefill_opt_hook.py)、[`bench/sps_profile.py`](../../bench/sps_profile.py)

---

## 一、Step 4 核心发现与改造项

### 1.1 打破 SGLang `_b12x_or_cutlass_mxfp8_linear` 强制锁死 `backend="b12x"` 的瓶颈（Step 4A）
- **根因定位**：排查 `fp8_utils.py` 发现 SGLang 对所有满足 `K % 128 == 0` 的 MXFP8 线性层无条件传入 `backend="b12x", pin_tactic=True`，导致此前 `mxfp8_prefill.py` 的 `CutlassMxfp8GemmRunner` 钩子从未命中 `wqkv_a`、`wq_b`、`shared_experts.w13`、`engram.wkv` 等核心投影，使 `sd-popt4b-s123` 的 32K Prefill 中仍有 `318` 次调用落入小矩阵核 `b12xDenseGemmKernel`（累计耗时 **`427.46 ms`**）；同时 `K=288`（`shared_experts.w2`）在 `M < 256` 解码阶段默认落入 `tactic=-1`（`9.5–11.1 us`），比 `tactic=1`（`3.51–3.71 us`）慢 **`2.55×–3.15×`**。
- **全形状 CUDA Graph 扫描与逐比特一致性验证**（详见 [`step4-mxfp8-graph-sweep.json`](data/step4-mxfp8-graph-sweep.json)）：
  - 在 RTX PRO 6000D 上对整模全部 10 个 TP8/DP8 `(K, N)` 形状遍历 `b12x` 与 CUTLASS `tactic = -1, 0..5`，**验证所有输出 100.0% 逐比特完全一致（`rel_rms = 0.0`）**；
  - 关键加速点：
    - `attn.wqkv_a` `(5120, 1792)`：`M=5120` 下 `cutlass tactic=5` 为 **`258.0 us`** vs `b12x` `431.9 us`（**提速 `1.67×`**）；
    - `shared_experts.w13` `(5120, 576)`：64 并发全局解码 `M=320/384` 下 `cutlass tactic=-1/0` 为 **`15.4–23.4 us`** vs `b12x` `33.4–38.9 us`（**提速 `1.43×–2.53×`**）；`M=5120` 下 `tactic=3` 提速 **`1.45×`**；`M=16384` 提速 **`1.25×`**；`M=40960` 下 `tactic=2` 提速 **`1.13×`**；
    - `shared_experts.w2` `(288, 5120)`：`M <= 64` 解码切入 `tactic=1`（**提速 `2.55×–3.15×`**），`M=5120/16384` 切入 `tactic=3/2`（提速 `1.20×`）；
    - `attn.wq_b` `(1280, 32768)`：`M=40/48`（DP8 64 路解码）与 `M=5120` 切入 `tactic=3`（`24.0 us` vs `28.6 us`；`1490 us` vs `1565 us`），`M=16384` 切入 `tactic=2`（`4807 us` vs `4977 us`）。
  - 在 `select_sm120_mxfp8_plan` 与 `_mxfp8_linear` 中通过线程局部 `forced_cutlass_tactic` 精确下发最优 `(backend, tactic)`。

### 1.2 DSpark 草稿模型 128 专家 MoE 同步接入 SM120 原生 `MXMMA float8e3m4`（Step 4B）
- 草稿模型 `ds41-draft` 含 3 层 `128` 专家路由 MoE（TP8 下 `gate_up=(512, 5120)`、`down=(5120, 256)`，天然对齐 128 无需零填充）。通过 `XTIER_DRAFT_MOE_MXMMA=e3m4` 将其一并切入 `float8e3m4 + e8m0 group-32` 原生块缩放指令，8-rank Decode Trace 证实 `humming<... 5120, 384 ... float8e3m4>` 调用数由 `200` 增至 **`215`**（40 层目标 + 3 层草稿 × 5 步全覆盖），同时将草稿 Prefill MoE 激活工作区减半。

### 1.3 `CHUNK=40960` 全程显存底线加固（Step 4C）与 `sps_profile.py` Attention-DP8 修复
- 将多卡并发分块设为 `CHUNK=40960`（`5,120/rank`，单发仍维持 `16,384` 续块），**包含 8-rank CUPTI 硬件 Profiler 挂载在内的全部 `1,968` 次显存采样最低空闲显存升至 `2,186 MiB`（`2.1348 GiB`）**，较 `sd-popt4b` 采样瞬间提升 `+944 MiB`，全程严守 `>= 1.26 GiB` 安全线。
- 同时定位并修复上游 `dspark_sps_profiler` 在 `--attn-dp-size 8` 下误判 `dp_size=1` 及负载未按 DP Rank 路由对齐的两个缺陷，合入 [`bench/sps_profile.py`](../../bench/sps_profile.py) 并新增 [`tests/test_sps_profile_dp.py`](../../tests/test_sps_profile_dp.py) 单元测试。

---

## 二、整模实测与硬件 Profiler 对照（`sd-popt5-s1234`）

| 指标维度 | 基线 `sd-dev3h-r1-p1` | Step 1–3 `sd-popt4b-s123` (`CHUNK=49152`) | Step 1–4 `sd-popt5-s1234` (`CHUNK=40960`) | Step 4 关键变化与结论 |
|---|---:|---:|---:|---|
| **全局主 KV 容量 (`tokens`)** | `67,450,880` | `113,569,792` | **`113,958,912`** | **`+68.95%`**（较 49K 分块再增 `+38.9 万` token） |
| **含 CUPTI 全程最低空闲显存** | `4.744 GiB`（无 CUPTI） | `2.629 GiB`（CUPTI 瞬间 `1.213`） | **`2.1348 GiB` (`2,186 MiB`)** | **含 8 卡 Profiler 全程稳居 `> 2.13 GiB`** |
| **3 段代码 PPL** | `2.0692` (`p0: 2.0678`) | `2.0745` (`+0.256%`) | **`2.0678` (`-0.067%`)** | **与原生 `p0` 基线 `2.0678` 完全一致** |
| **`1×32K` 冷 Prefill (`tok/s`)** | `2,315` | **`10,886.1`** | **`10,579.8`**（TTFT `3.10s`） | **`4.57×`**（单发维持 `16K` 续块高速档） |
| **`64×32K` 冷 Prefill (`tok/s`)** | `20,543` | **`26,106.2`**（6 块/请求） | **`25,547.8`**（7 块/请求） | **`+24.36%`**（`5120/rank` 多切 1 块换取 `+0.92 GiB` CUPTI 余量） |
| **`64×32K` 共享前缀 Decode (`tok/s`)** | `4,214` | `4,671.7`（TPOT `9.30ms`） | **`4,620.1`**（窗口 `4,693.3`，**P50 TPOT `9.245 ms`**） | **单请求流式吞吐和 `6,682.8 tok/s` 与 TPOT 创最佳** |
| **`64` 路短提示 `real-short` Decode (`tok/s`)** | `2,922` | `3,247.0`（窗口 `3,275.3`） | **`3,285.5`（窗口 `3,318.6`）** | **`+12.44%` 刷新全阶段最高纪录** |
| **`64×128K` 单步硬件耗时 (`ms/step`)** | `53.67` | `51.83` | **`49.75`（首次跌破 `50 ms/步`）** | `b12x` 调用由 `318 -> 72`（Prefill）、`935 -> 470`（Decode） |
| **混合负载 `64 D + 2×300K P`** | — | `78.4s` / D `2,445.6` | **`85.2s` / 并发 D `2,515.7 tok/s`** | 流式 P50 TPOT 降至 **`11.24 ms`**，全程 0 错 |
