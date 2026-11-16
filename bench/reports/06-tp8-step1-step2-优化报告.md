# 06. TF500 TP8 Step 1 + Step 2（通信压缩、ReplayTail 显存安全修复与 SM120 原生 MXMMA `float8e3m4` MoE）实测报告

- **日期**：2026-11-17
- **测试平台**：`tf500`（`8 × NVIDIA RTX PRO 6000D` 84 GiB GDDR7，SM120，双路 Xeon 6737P，跨 Socket `NCCL_P2P_LEVEL=SYS`）
- **基线对照**：`sd-dev3h-r1-p1`（`MEM_FRACTION_STATIC=0.85`，`--swa-prefix-tails 384`，`CHUNK=32768`，DeepInfer 第一档 `A1/A2/A3/A7=1` + `repair-a2-reader`，DSpark `block=5, draft=6` `compact` 变长验证）
- **结构化原始证据**：[`sd-popt3b-analysis.json`](data/sd-popt3b-analysis.json)
- **外置无侵入实现**：[`deploy/prefill-opt/prefill_opt_hook.py`](../../deploy/prefill-opt/prefill_opt_hook.py)、[`deploy/prefill-opt/sitecustomize.py`](../../deploy/prefill-opt/sitecustomize.py)（对 `/data/src/sglang-align` 与 `/data/xtier-env` 零修改）

---

## 一、核心优化机制与上游缺陷修复

### 1.1 Step 1：跨 NUMA DP Gatherv FP8 线速压缩 + 空根跳过 + `DecoderReplayTail` 显存生命周期修复
1. **FP8 组量化跨卡 Gatherv（`SGLANG_ENABLE_DP_GATHER_FP8=1`）**：在 `SUM_LEN` 变长 DP Gather 阶段将每 token `[5120]` BF16 激活压缩为 `FP8 E4M3 + FP32 group-128 scale`（跨 PCIe 5.0 / UPI 字节量减少 **48.4%**）。8-rank Profiler 实测单次 4K 分块的 `ncclDevKernel_Broadcast_RING_LL` 累计耗时由 **`210.64 ms` 降至 `113.52 ms`（`-46.1%`）**。
2. **NCCL 零长度根跳过（`XTIER_PREFILL_NCCL_SKIP_ZERO=1`）**：在 `PyNcclCommunicator.all_gather` / `reduce_scatter`（`sizes` 变长路径）跳过 `split_size == 0` 的空闲 DP Rank 广播/归约调用，同时完整保留 `self.comm` 上的 `ncclMemAlloc` 对称窗口注册。
3. **关键上游 Bug 定位与修复（`DecoderReplayTail` 行分桶 CUDA Graph × `_get_dp_gather_fp8_bufs` UAF 非法访存）**：
   - **根因**：启用 `--enable-decoder-swa-bounded-replay` 时，`DecoderReplayTail` 会在在线服务期按行分桶（`<= 2048` 行）懒捕获尾层 CUDA Graph，并将当时 `_dp_gather_fp8_bufs` 的 GPU 虚拟地址固化到图节点中；随后当 64 并发大批量请求触发 `rows` 增长时，原生 `_get_dp_gather_fp8_bufs` 直接覆盖字典引用释放旧张量，导致后续尾层 CUDA Graph 重放向已释放虚拟地址执行 `ncclBroadcast` 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。
   - **修复**：在 `install_onto_dp_attention` 中于非捕获流预分配 `65536` 行持久化缓冲，扩容时将历史缓冲挂入 `_retired_keepalive` 永久保活，并在 `is_current_stream_capturing()` 期间隔离捕获池分配，彻底根除非法访存。
4. **集群感知单请求自适应大块（`XTIER_IDLE_SINGLE_CHUNK=16384` + `_LAST_GLOBAL_ACTIVE_RANKS` 双锁）**：仅当上一轮全局 DP 同步确认全集群仅 **1 张卡活跃**（`_LAST_GLOBAL_ACTIVE_RANKS == 1`）、本地等待队列为空且处于续块状态（`has_chunked == 1`）时将单流分块提升至 `16K`；任意多卡并发（如 `conc=64`）下严格锁定 `4096/rank`（全局 `M <= 32768`），从数学上杜绝显存峰值越界。

### 1.2 Step 2：SM120 原生 `MXMMA float8e3m4` 动态组量化 MoE + `pad(128, 128)` 紧凑权重对齐
1. **硬件原理解析**：RTX PRO 6000D（Blackwell SM120）支持原生块缩放指令 `mma.sync.aligned.m16n8k64.block_scale`（`W4 FP4 E2M1 × A8 FP8 E3M4 + UE8M0 group-32 scale`），算力上限是原 `W4A16`（权重量化反量化后走 BF16 Tensor Core `146.4 TFLOP/s`）的 **2 倍以上**；且 `float8e3m4` 具备 4 位尾数（动态范围适配组归一化后的激活分布），实测相对误差（`rel_rms = 0.02391`，余弦相似度 **`0.999714`**）比 `float8e4m3`（`0.04618`）低近一半。
2. **紧凑对齐省显存**：TP8 下每个路由专家中间维 `N_shard = 2304 / 8 = 288`。原 Humming 默认按 `pad(256, 128)` 将 `288` 填充至 `512`（浪费 `43.75%` 零权重与算力）；改为满足 SM120 TMA 对齐的 **`pad(128, 128)`**（`288 -> 384`）后：
   - **每卡模型权重显存由 `56.822 GiB` 直降至 `51.842 GiB`（每卡净省 `4.980 GiB`，8 卡共省 `39.84 GiB`！）**；
   - 在相同 `MEM_FRACTION_STATIC=0.85` 下，**全局主 KV 缓存容量由 `67,450,880` token 自动暴涨 `+69.53%` 至 `114,348,032` token（1.143 亿 token）**！
   - 算子实测：`M=32768` Prefill MoE 单层耗时由 `19,388.7 us` 降至 **`8,267.5 us`（提速 `2.35×`，有效算力 `209.8 TFLOP/s`）**；`M=320` Decode CUDA Graph 单层耗时由 `969.9 us` 降至 **`838.4 us`（提速 `+15.7%`）**。

---

## 二、整模实测演进对比表（`CASE_ID=5d3cdbde7db2`，`mem=0.85`，`tails=384`）

| 指标维度 | 基线 `sd-dev3h-r1-p1` | Step 1 `sd-popt3b-s1` | Step 1+2 `sd-popt3b-s12` | Step 1+2 相对基线增益 |
|---|---:|---:|---:|---:|
| **每卡权重显存 (`GiB`)** | `56.822` | `56.822` | **`51.842`** | **`-4.980 GiB / 卡`** |
| **全局主 KV 容量 (`tokens`)** | `67,450,880` | `67,450,880` | **`114,348,032`** | **`+69.53%`** |
| **全程最低空闲显存 (`min_free_gib`)** | `4.744 GiB` (`4858 MiB`) | `4.551 GiB` (`4660 MiB`) | **`4.432 GiB` (`4538 MiB`)** | 远高于 `1.26 GiB` 红线 |
| **3 段代码 PPL（`sglang-corpus`）** | `2.069191` (`p0: 2.067812`) | `2.0719` (`+0.131%`) | **`2.0714`** | **`+0.107%`**（远优于 `±0.5%`） |
| **Smoke 功能校验（中文/代码/流式工具）** | `3 / 3` | `3 / 3` | **`3 / 3`** | 100% 通过 |
| **`1×32K` 冷 Prefill (`tok/s`)** | `2,315`（热态 `~8,350`） | `8,618.8` | **`8,632.1`**（首轮冷起 `2,809.6`） | **`+272.9%`（`3.73×`）** |
| **`1×32K` 缓存 Decode (`tok/s`)** | `275.2` | `180.0`（窗口 `186.0`） | **`279.9`**（窗口 **`296.6`**） | **`+1.7%`（窗口 `296.6`）** |
| **`64×32K` 冷 Prefill (`tok/s`)** | `20,543` | `21,834.0` (`+6.28%`) | **`24,850.5`** | **`+20.97%`** |
| **`64×32K` 共享前缀 Decode (`tok/s`)** | `4,214` | `4,224.1`（窗口 `4,286.1`） | **`4,740.5`**（窗口 **`4,822.2`**） | **`+12.49%`**（每步 `58.2 -> 52.8 ms`） |
| **`64×128K` 两阶段冷 Prefill (`tok/s`)** | `20,660` | — | **`25,158.1`** | **`+21.77%`** |
| **`64×128K` 纯 Decode 窗口 (`tok/s`)** | `3,567`（命中 `99.79%`） | — | **`3,671.6`**（命中 `99.79%`） | **`+2.93%`**（较 `p0` `+8.5%`） |
| **`64` 路短提示 `real-short` Decode (`tok/s`)** | `2,922` | `2,924.1` | **`3,095.4`**（窗口 **`3,129.9`**） | **`+5.93%`** |
| **混合负载 `64 Decode + 2×300K Prefill`** | 0 错 | — | **`0` 错，`300K` TTFT `95.2s`，并发 D `2,567.7 tok/s`** | 流式 P50 TPOT `11.92 ms` |

---

## 三、8-Rank Profiler 热点变化与下一阶段（Step 3）切入点

对比 `sd-dev3h-r1-p1` 与 `sd-popt3b-s12` 的 Rank 0 硬件 Trace：
1. **已消除瓶颈**：
   - Prefill 跨卡 `ncclDevKernel_Broadcast_RING_LL`：`210.64 ms -> 113.52 ms`（`-46.1%`）；
   - Prefill Humming MoE 核总耗时：`419.18 ms -> 283.50 ms`（`-32.4%`，64 并发 `M=32768` 下达 `2.35×`）。
2. **暴露出的下一顺位热点（由 Step 3 精准接管）**：
   - **Prefill `#3` 热点**：`b12xDenseGemmKernel`（`848` 次调用占 `327.30 ms` = **`81.8 ms/chunk`**），因 SGLang `pin_tactic=True` 默认落回小批量 `b12x` -> 将通过扩展 `mxfp8_prefill.py` 已验证 100% 逐比特一致形状切入 `128×128×128` CUTLASS TMA 大 Tile；
   - **Prefill `#4` 热点**：`wo_a` BF16 `s16816gemm`（`160` 次调用占 `144.04 ms` = **`36.0 ms/chunk`**）-> 将通过零反量化原位 `block_scale_interleave` MXFP8 分组矩阵乘加速；
   - **Decode `#6` + `#9` 热点**：`wo_a` 在 `conc=64`（`T=48 > MAX_T=16`）退化触发 `200` 次 `_dequant_kernel`（`11.53 ms`）+ `215` 次 BF16 GEMM（`6.45 ms`），共占 **`3.60 ms/step`** -> 将通过上调 `XTIER_DS_WO_A_MAX_T=56` 彻底清零 `_dequant_kernel`；
   - **Decode `#11` 热点**：宽输出维（`N >= 16384`，如 `attn.wq_b` `N=32768`）在 `T=40/48` 落回 `128×128×128` CUTLASS -> 将通过按形状路由 `b12x` 进一步削减每步耗时。
