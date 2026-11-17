# 07. TF500 TP8 Step 3（Prefill + Decode 双引擎算子与分块深度提速）实测报告

- **日期**：2026-11-17
- **测试平台**：`tf500`（`8 × NVIDIA RTX PRO 6000D` SM120，Attention DP8 + TP8 + DSpark `block=5, draft=6` `compact`）
- **前置里程碑**：`sd-popt3b-s12`（Step 1 + Step 2，里程碑 #2）
- **结构化原始证据**：[`step3-unit-summary.json`](data/step3-unit-summary.json)、[`sd-popt4-analysis.json`](data/sd-popt4-analysis.json)
- **修改模块**：[`src/xtier_86kd/mxfp8_prefill.py`](../../src/xtier_86kd/mxfp8_prefill.py)、[`src/xtier_86kd/packs/deepseek_v4/wo_a.py`](../../src/xtier_86kd/packs/deepseek_v4/wo_a.py)、[`src/xtier_86kd/packs/deepseek_v4/__init__.py`](../../src/xtier_86kd/packs/deepseek_v4/__init__.py)、[`deploy/prefill-opt/prefill_opt_hook.py`](../../deploy/prefill-opt/prefill_opt_hook.py)

---

## 一、Step 3 四项核心优化与微基准依据

基于 `sd-popt3b-s12` 的 8-rank 硬件 Profiler 剩余热点与 [`step3-unit-summary.json`](data/step3-unit-summary.json) 算子实测数据，实施以下四项改造：

1. **Step 3A — 扩展 `mxfp8_prefill.py` 100% 逐比特一致 `LARGE_TILE = 4` 白名单**：
   - 算子实测证明，在 SM120 上将 `(1280, 32768)`（`attn.wq_b`）、`(5120, 1280)`（`attn.wq_a`）、`(6144, 25600)`（`engram.wkv`）、`(288, 5120)`（`shared_experts.w2`）及大 `M` 下的 `(8192, 5120)`（`attn.wo_b`）、`(5120, 512)`（`attn.wkv`）切入 `128×128×128` CUTLASS TMA Tile（`tactic=4`），输出与默认路径 **100.0% 逐比特完全相等（`rel_rms = 0.0`）**，单算子提速 **`1.05×–1.38×`**；同时剔除小输出维 `(1280, 4096)` 等默认更快形状，杜绝负优化。
2. **Step 3B — `wo_a` 解码阈值扩容（`MAX_T: 16 -> 56`）与 Prefill 无损 `bmm` / 可选原位 MXFP8**：
   - **Decode 侧**：DP8 64 并发下每卡承载 `local_bs=8`（草稿 `T=40`，目标验证 `T=48`），原阈值 `MAX_T=16` 导致全 40 层回退触发每 5 步 `200` 次 `_dequant_kernel`（`11.53 ms`）+ `215` 次 BF16 GEMM（`6.45 ms`）。上调 `MAX_T=56` 后，**8-rank Profiler 证实 `_dequant_kernel` 调用降至 `0` 次**，由融合 FP8 `_einsum_kernel`（`215` 次仅 **`9.17 ms`**）接管，`wo_a` 解码总耗时直降 **`49.0%`**。
   - **Prefill 侧 A/B 发现**：我们同时实现了基于 `block_scale_interleave` 的零额外权重显存 MXFP8 分组 `wo_a`（`sd-popt4-s123`）与零误差跨步 `torch.bmm`（`sd-popt4b-s123`，默认）。对照发现：`PREFILL_MXFP8=1` 在 `64×128K` 纯 Decode（**`3,800.3 tok/s`**）与 `real-short`（**`3,252.0 tok/s`**）达到最高值，但在 `32K/1K` 共享前缀场景因种子 KV 存在 `2.67%` 激活量化扰动使草稿接受长度由 `5.39–5.53` 微降至 `5.21`；默认采用无损 `_bmm_token_major`（`XTIER_DS_WO_A_PREFILL_MXFP8=0`）可同时兼顾 **`26,106+ tok/s` Prefill** 与共享前缀高接受率（`4,671.7 tok/s`，窗口 `4,745.6 tok/s`）。
3. **Step 3C — 按输出宽度 `N` 分流 `b12x` 解码小矩阵核**：
   - 算子微基准揭示：宽输出维 `N >= 16384`（`attn.wq_b` `N=32768`、`engram.wkv` `N=25600`）在 `M=40/48`（64 并发）下 `b12x` 比 CUTLASS 快 **`1.19×–1.24×`**（100% 逐比特一致）；而窄维 `N < 16384` 在 `M >= 24` 时 CUTLASS 反而快 `20%`。因此改为宽维 `1-64`、窄维 `1-16` 双区间路由。
4. **Step 3D — 隔离 `DecoderReplayTail` 计数覆盖 + `CHUNK=49152`（`6,144/rank`）**：
   - 修复尾层桶图重放 `[rows]*8` 将 `_LAST_GLOBAL_ACTIVE_RANKS` 覆盖回 `8` 的隐藏问题（`_IN_REPLAY_TAIL_DEPTH` 作用域守卫），使单发 `1×32K` 在首块确认单卡活跃后立即升至 **`16,384` 分块**（全程仅 3 块完成 `32K` Prefill，吞吐跃升至 **`10,886.1 tok/s`**）；同时在 `MXMMA e3m4` 节省 `4.98 GiB/卡` 基础上将多卡并发 `CHUNK` 提至 `49,152`（`6,144/rank`）。

---

## 二、整模实测演进对比表（基线 vs Step 1+2 vs Step 1+2+3）

| 指标维度 | 基线 `sd-dev3h-r1-p1` | Step 1+2 `sd-popt3b-s12` | Step 1+2+3 (`wo_a` MXFP8) `sd-popt4-s123` | Step 1+2+3 (`wo_a` 无损 BMM) `sd-popt4b-s123` | 最佳相对基线增益 |
|---|---:|---:|---:|---:|---:|
| **全局主 KV 容量 (`tokens`)** | `67,450,880` | **`114,348,032`** | `113,569,792` | `113,569,792` | **`+68.4%–69.5%`** |
| **业务测速最低空闲显存 (`GiB`)** | `4.744` | `4.432` | **`2.629` (`2692 MiB`)** | **`2.629`**（挂载 CUPTI 采样瞬间 `1.213`） | 业务全程 `> 2.6 GiB` |
| **3 段代码 PPL** | `2.0692` | `2.0714` (`+0.107%`) | **`2.0666` (`-0.125%`)** | **`2.0745` (`+0.256%`)** | 均远优于 `±0.5%` |
| **`1×32K` 冷 Prefill (`tok/s`)** | `2,315`（热 `~8,350`） | `8,632.1` | `10,496.8` | **`10,886.1`**（TTFT **`3.01s`**） | **`+370.2%`（`4.70×`）** |
| **`1×32K` 缓存 Decode 窗口 (`tok/s`)** | `275.2` | `296.6` (`19.50 ms/步`) | **`299.9`** (`19.51 ms/步`) | `207.1`（硬件 **`19.09 ms/步`**） | 单步提速至 **`19.09 ms`** |
| **`64×32K` 冷 Prefill (`tok/s`)** | `20,543` | `24,850.5` | `25,639.2` | **`26,106.2`**（P50 TTFT **`45.37s`**） | **`+27.08%`** |
| **`64×32K` 共享前缀 Decode (`tok/s`)** | `4,214` | **`4,740.5`**（窗口 `4,822`） | `4,467.4`（`51.37 ms/步`） | **`4,671.7`**（窗口 `4,745.6`，P50 TPOT **`9.30 ms`**） | **`+10.9%–12.5%`** |
| **`64×128K` 冷 Prefill (`tok/s`)** | `20,660` | `25,158.1` | `25,993.5` | **`26,128.8`** | **`+26.47%`** |
| **`64×128K` 纯 Decode 窗口 (`tok/s`)** | `3,567` | `3,671.6` | **`3,800.3`**（输出 `3,678.5`） | **`3,705.9`**（输出 `3,588.6`） | **`+6.54%`** |
| **`64` 路短提示 `real-short` Decode (`tok/s`)** | `2,922` | `3,095.4` | **`3,252.0`**（窗口 **`3,283.7`**） | **`3,247.0`**（窗口 `3,275.3`） | **`+11.29%`**（`48.3 ms/步`） |
| **混合负载 `2×300K` TTFT (`s`)** | — | `95.20 / 95.23` | `78.95 / 82.12` | **`78.38 / 79.38`**（并发 D `2,445.6`） | 较 Step 2 再快 **`-17.1%`** |

---

## 三、新暴露关键线索（Step 4 攻关方向）

1. **显存余量加固**：在 `CHUNK=49152` 下业务测速全程空闲 `>= 2.629 GiB`，但在额外挂载 `torch.profiler` CUPTI 缓冲瞬间曾触及 `1.213 GiB`（略低于 `1.26 GiB` 目标线 `48 MiB`）。下一阶段将验证 `CHUNK=40960`（`5,120/rank`），确保即便叠加 CUPTI 采样亦稳居 `2.0 GiB+`。
2. **Prefill `b12xDenseGemmKernel` 拦截盲区修复**：`sd-popt4b-s123` 的 Prefill Trace 显示仍有 `318` 次调用落入 `b12xDenseGemmKernel`（耗时 **`427.46 ms`**，占整段 32K Prefill 近 30%！），原因是 FlashInfer 在 `with autotune(False, skip_ops={"mxfp8_gemm"}):` 下绕过了 `AutoTuner.choose_one` 回调。直接在 `fp8_utils.flashinfer_mxfp8_blockscaled_linear` 钩子层对 `M >= 256` 的安全白名单形状取消 `pin_tactic=True`（或透传 `LARGE_TILE`），有望再释放显著 Prefill 潜能。
3. **DP8 专属 `SPS_TABLE` 重标定**：当前 `bench/86kd-sps-6000d.json` 仅标定至 `bs=32, M=192`，将在 Step 4 对 DP8 `bs=1..64`（`M` 至 `384`）重新运行 `sps_profile.py` 实测标定。
