# xtier_86kd：TF500（8 × RTX PRO 6000D）DeepSeek-V4.1-Flash TP8 双引擎加速包、对齐补丁与实测归档

本仓库专用于 **`tf500`（8 × NVIDIA RTX PRO 6000D，Blackwell SM120，双路 Xeon 6737P + PCIe 5.0）** 上的 **`DeepSeek-V4.1-Flash-1017` + `DSpark` 全显存 TP8 / Attention DP8 高性能推理部署**（旧 TP4 混合方案已彻底废弃移除）。

所有优化均采用 **插件化（`src/xtier_86kd`）+ 外置 `sitecustomize` 钩子（`deploy/prefill-opt`、`deploy/repair-a2-reader`）** 架构实现，对底层 `/data/src/sglang-align` 与 `/data/xtier-env` 保持 **零侵入修改**，支持通过 WebUI 启动台（`:8100`）或环境变量一键开启与无损回退。

---

## 一、8 小时自主开发总结汇报（2026-11-17 双引擎提速终章）

围绕 **“64 并发与单发 Prefill 速度、Decode 速度同步大幅提升，同时严守显存余量与生成精度”** 的目标，本轮完成了 **Step 1 → Step 5 五个递进阶段** 的底层算子、跨 NUMA 通信、SM120 原生块缩放指令（MXMMA）与 Attention-DP8 投机解码成本表重构，每阶段均经过 8 卡整模基准、3 段 PPL 与硬件 Profiler 闭环验证并逐次推送 GitHub。

### 1.1 优化前后核心指标总览（同口径 `CASE_ID=5d3cdbde7db2`，`mem=0.85`，`tails=384`，`A1/A2/A3/A7=1`）

| 核心评估维度 | 优化前基线 (`sd-dev3h-r1-p1`) | 优化后默认巅峰栈 (`sd-popt6-dp8sps`, `CHUNK=49152`) | 优化后高余量/短输入栈 (`sd-popt5-s1234`, `CHUNK=40960`) | 净增益 / 关键突破 |
|---|---:|---:|---:|---|
| **每卡模型权重显存** | `56.822 GiB` | **`51.842 GiB`** | **`51.842 GiB`** | **每卡净省 `-4.980 GiB`（8 卡省 `39.84 GiB`）** |
| **全局主 KV 缓存容量** | `67,450,880` token | **`113,569,792` token** | **`113,958,912` token**（32K 分块达 `114.35M`） | **容量暴涨 `+68.4%` ~ `+69.5%`（破 1.14 亿）** |
| **`1×32K` 单发冷 Prefill** | `2,315 tok/s`（热 `~8,350`） | **`10,674.1 tok/s`**（TTFT `3.07s`） | **`10,579.8 tok/s`**（Step 3 峰值 **`10,886.1`**） | **`4.61×` ~ `4.70×`（TTFT 压至 `3.01s`）** |
| **`64×32K` 并发冷 Prefill** | `20,543 tok/s` | **`26,042.3 tok/s`**（P50 TTFT `45.5s`） | `25,547.8 tok/s`（Step 3 峰值 **`26,106.2`**） | **`+26.8%` ~ `+27.1%`** |
| **`64×128K` 两阶段冷 Prefill** | `20,660 tok/s` | **`26,478.5 tok/s`**（P50 TTFT `178.4s`） | `25,755.5 tok/s` | **`+28.16%`（全阶段最高纪录）** |
| **`64×32K` 共享前缀 Decode** | `4,214 tok/s` | **`4,862.6 tok/s`（窗口 `4,967.2`）** | `4,620.1 tok/s`（窗口 `4,693.3`） | **`+15.39%`（流式和 `7,115 tok/s`）** |
| **`64×32K` Decode TPOT (P50 / P95)** | `10.5+ ms` / `14+ ms` | **`8.885 ms` / `9.404 ms`** | `9.245 ms` / `12.401 ms` | **64 并发 32K 上下文双双进入 `< 9.5 ms`** |
| **`64×128K` 纯 Decode 窗口（命中 `99.79%`）** | `3,567 tok/s`（`53.7 ms/步`） | **`3,894.5 tok/s`（输出 `3,760.3`，`47.19 ms/步`）** | `3,665.1 tok/s`（`49.75 ms/步`） | **`+9.18%`（较 `p0` 基线 `+15.1%`）** |
| **`64` 路短提示 `real-short` Decode** | `2,922 tok/s` | `3,183.4 tok/s`（**P50 TPOT `14.71 ms`**） | **`3,285.5 tok/s`（窗口 `3,318.6`）** | **`+8.94%` ~ `+12.44%`** |
| **混合负载 (`64 D + 2×300K P`)** | 0 错 | **`300K` TTFT `79.27s`，并发 D `2,482.7`** | `300K` TTFT `85.23s`，并发 D `2,515.7` | **流式 P50/P95 TPOT `11.12 / 85.93 ms` 最佳，0 错** |
| **3 段代码 PPL (`sglang-corpus`)** | `2.0692`（`p0: 2.0678`） | **`2.0708`（`+0.078%`）** | **`2.0678`（`-0.067%`，同 `p0`）** | **精度零退化（远优于 `±0.5%` 阈值）** |
| **全程最低空闲显存 (`min_free_gib`)** | `4.744 GiB`（32K 块） | **`1.9492 GiB` (`1,996 MiB`)** | **`2.1348 GiB` (`2,186 MiB`，含 8 卡 CUPTI)** | **全程严守 `>= 1.26 GiB` 红线之上** |

---

## 二、五步核心技术拆解与上游 Bug 修复

### Step 1：跨 NUMA 变长 DP Gatherv FP8 线速压缩 + 空根跳过 + `DecoderReplayTail` UAF 修复
- **代码位置**：[`deploy/prefill-opt/prefill_opt_hook.py`](deploy/prefill-opt/prefill_opt_hook.py)（`install_onto_dp_attention`、`install_onto_pynccl`）
- **跨 Socket 带宽减半**：`tf500` 为双路 Xeon 6737P（GPU 0–3 与 GPU 4–7 跨 UPI 互联，`NCCL_P2P_LEVEL=SYS`）。开启 `SGLANG_ENABLE_DP_GATHER_FP8=1` 将每层进入共享专家/MoE 前的 `[M, 5120]` BF16 激活在线压缩为 `FP8 E4M3 + FP32 group-128 scale`（通信字节量减少 **48.4%**），配合 `XTIER_PREFILL_NCCL_SKIP_ZERO=1` 跳过 `sizes` 中长度为 0 的空闲 DP Rank 广播/归约。8-rank Profiler 实测单块 `ncclDevKernel_Broadcast_RING_LL` 耗时由 **`210.64 ms` 降至 `113.52 ms`（`-46.1%`）**。
- **关键上游 Bug 根因定位与修复（CUDA Graph 悬垂指针非法访存）**：
  - *现象*：原版 SGLang 同时开启 `--enable-decoder-swa-bounded-replay` 与 `SGLANG_ENABLE_DP_GATHER_FP8=1` 时，64 并发必现 `CUDA_ERROR_ILLEGAL_ADDRESS`。
  - *根因*：`DecoderReplayTail` 在服务期按行分桶（`<= 2048` 行）懒捕获尾层 CUDA Graph，将当时 `_dp_gather_fp8_bufs` 的 GPU 虚拟地址固化入图；当后续并发请求触发 `_get_dp_gather_fp8_bufs` 动态扩容时，原实现直接覆盖字典释放旧张量，导致尾层 CUDA Graph 重放时向已释放显存写入。
  - *修复*：在非捕获流预分配 `65,536` 行持久化 FP8 Gather 池，扩容时将退役缓冲推入 `_retired_keepalive` 永久保活，并在流捕获态隔离临时池分配，彻底根除非法访存。

### Step 2：SM120 原生 `MXMMA float8e3m4` 动态组量化 MoE + `pad(128, 128)` 紧凑权重对齐
- **代码位置**：[`deploy/prefill-opt/prefill_opt_hook.py`](deploy/prefill-opt/prefill_opt_hook.py)（`install_onto_humming_utils`、`install_onto_humming_layer`）
- **释放 Blackwell 原生块缩放算力**：将主模型（40 层 × 384 专家）与 DSpark 草稿模型（3 层 × 128 专家）的 Humming MoE 输入模式由 `W4A16`（反量化走 BF16 Tensor Core）切换为 SM120 原生硬件指令 `mma.sync.aligned.m16n8k64.block_scale`（**`W4 FP4 E2M1 × A8 FP8 E3M4 + UE8M0 group-32 scale`**）。`float8e3m4` 的 4 位尾数对组归一化激活具有比 `e4m3` 高近一倍的精度（单层余弦相似度 **`0.999714`**），`M=32768` Prefill MoE 单层耗时由 `19.39 ms` 直降至 **`8.27 ms`（提速 `2.35×`，有效算力 `209.8 TFLOP/s`）**。
- **消除 `43.75%` 零填充显存浪费**：TP8 下路由专家中间维 `N_shard = 2304 / 8 = 288`。Humming 默认 `pad(256, 128)` 将其填充至 `512`；改为满足 SM120 TMA 对齐的 **`pad(128, 128)`**（`288 -> 384`）后，**每卡模型权重直降 `4.980 GiB`（`56.822 -> 51.842 GiB`），使全局主 KV 容量一口气净增 `+4,689 万` token（`+69.5%`）**！

### Step 3：`attn.wo_a` 解码反量化清零 + 单请求自适应 `16K` 续块守卫
- **代码位置**：[`src/xtier_86kd/packs/deepseek_v4/wo_a.py`](src/xtier_86kd/packs/deepseek_v4/wo_a.py)、[`deploy/prefill-opt/prefill_opt_hook.py`](deploy/prefill-opt/prefill_opt_hook.py)（`install_onto_replay_tail`、`install_onto_scheduler`）
- **清零 64 并发 Decode 反量化核**：DP8 64 并发下每卡承载 `local_bs=8`（草稿 `T=40`、目标验证 `T=48`），超过了旧版 `XTIER_DS_WO_A_MAX_T=16`，导致全 40 层回退触发每 5 步 `200` 次 `_dequant_kernel`（`11.53 ms`）+ BF16 GEMM（`6.45 ms`）。将默认上限提至 **`MAX_T=56`** 后，**Decode `_dequant_kernel` 调用降至 `0` 次**，由融合 FP8 `_einsum_kernel` 接管，`wo_a` 解码核总耗时直降 **`49.0%`（`17.97 -> 9.17 ms / 5步`）**；Prefill 侧采用零误差跨步 `_bmm_token_major`（并保留零额外显存的 `XTIER_DS_WO_A_PREFILL_MXFP8` 实验开关）。
- **激活真正的单发 `16K` 续块**：修复 `DecoderReplayTail` 重放合成 `[rows]*8` 覆盖 `_LAST_GLOBAL_ACTIVE_RANKS` 计数的问题（新增 `_IN_REPLAY_TAIL_DEPTH` 作用域守卫），使单请求 `1×32K` 在首块确认全集群仅单卡活跃后，后续分块立即跃升至 **`16,384` token**（仅需 3 块完成 32K），**单发 32K 冷 Prefill 由 `2,315` 飙升至 `10,886.1 tok/s`（`4.70×`）**；而任意多卡并发下严格锁定安全分块，杜绝显存越界。

### Step 4：全形状 100% 逐比特一致 SM120 MXFP8 `(backend, tactic)` 最优表
- **代码位置**：[`src/xtier_86kd/mxfp8_prefill.py`](src/xtier_86kd/mxfp8_prefill.py)、[`src/xtier_86kd/packs/deepseek_v4/__init__.py`](src/xtier_86kd/packs/deepseek_v4/__init__.py)
- **打破 SGLang 强制锁死 `b12x` 盲区**：排查发现 `fp8_utils._b12x_or_cutlass_mxfp8_linear` 对所有 `K % 128 == 0` 投影无条件传入 `backend="b12x"`，导致 `CutlassMxfp8GemmRunner` 的大矩阵 TMA Tile 在 Prefill 与 64 路全局 MLP 解码（`M=320/384`）下从未生效；同时 `K=288`（`shared_experts.w2`）在小 `M` 下误走 `tactic=-1`。
- **真机 CUDA Graph 全扫描查表**（[`step4-mxfp8-graph-sweep.json`](bench/reports/data/step4-mxfp8-graph-sweep.json)）：对整模全部 10 个 TP8/DP8 `(K, N)` 形状遍历验证 **`b12x` 与 CUTLASS `tactic=-1, 0..5` 输出 100.0% 逐比特完全一致（`rel_rms = 0.0`）**，通过线程局部 `forced_cutlass_tactic` 精确路由：`wqkv_a` `M=5120` 切 `tactic=5`（**`1.67×`**）、`shared.w13` `M=320/384` 切 `tactic=-1`（**`1.43×–2.53×`**）、`shared.w2` `M<=64` 切 `tactic=1`（**`2.55×–3.15×`**）、`wq_b` `M=40..16384` 切 `tactic=3/2`，使 Prefill `b12xDenseGemmKernel` 调用由 `318` 次骤降至 `72` 次，`64` 路 `real-short` Decode 冲上 **`3,285.5 tok/s`（窗口 `3,318.6`）**。

### Step 5：Attention-DP8 原生 `SPS_TABLE` 均衡路由标定与 `CHUNK=49152` 巅峰融合
- **代码位置**：[`bench/sps_profile.py`](bench/sps_profile.py)、[`bench/86kd-sps-6000d-dp8.json`](bench/86kd-sps-6000d-dp8.json)
- **修复上游 DP8 标定器缺陷**：修复 `dspark_sps_profiler` 在 `--attn-dp-size 8` 下误读 `dp_size=1` 以及负载未按 DP Rank 路由导致 300s 超时报错的两个问题，在 `bench/sps_profile.py` 中自动按 `len(internal_states)=8` 修正并发并通过 `x-override-routed-dp-rank` 8 卡同步均衡施压，**30 轮稳态标定跨卡步对齐率全部达到 `match_fraction = 1.00`**。
- **消除投机预算欠拟合**：新表准确刻画 DP8 真实固定步长 `bias_seconds = 25.43 ms`（旧纯 TP8 表仅 `11.35 ms`），使 `HostConfidenceBudgetPlanner` 释放中置信度草稿验证潜力：`64×32K` 平均每步接受长度由 `5.326` 跃升至 **`5.577`**，单步耗时反降至 **`50.28 ms`**，**共享前缀 Decode 冲上 `4,862.6 tok/s`（窗口 `4,967.2 tok/s`），`64×128K` 纯 Decode 窗口冲上 `3,894.5 tok/s`（`47.19 ms/步`）**！

---

## 三、全阶段演进与实测报告归档总表（01–09）

| 阶段与报告入口 | GitHub 里程碑 / 配置要点 | 单发 32K 冷 P (`tok/s`) | 64 路冷 P (`tok/s`) | 64 路缓存 / 短提示 D (`tok/s`) | 关键结论 |
|---|---|---:|---:|---:|---|
| **01. 装机检查与纯 TP8 初测**<br>[`01-tf500-前置检查与测速.md`](bench/reports/01-tf500-前置检查与测速.md) | 纯 TP8（无 DP）、FP8 KV、`mem=0.78` | `12,386`（热） | 32 路：`14,105` | 32 路：`2,678`（`+154.8%`） | 8 卡硬件/P2P/NVMe 达标；纯 TP8 全局 KV 仅 `5.19M` |
| **02. 86KD 32K/1K 专项对比**<br>[`02-xtier_86kd-32k1k-对比.md`](bench/reports/02-xtier_86kd-32k1k-对比.md) | 原版静态 vs 86KD 静态 vs 86KD 变长 | `12,386`（冷起 `4,325`） | 32 路：`14,105` | 8 路：`1,589`；32 路：`2,678` | 确立 `XTIER_86KD=1` + `compact` 变长验证路线 |
| **03. DP8 + Gatherv + RS 优化**<br>[`03-tp8-dp8-fp8kv-阶段报告.md`](bench/reports/03-tp8-dp8-fp8kv-阶段报告.md) | `attn-dp=8` + `DP LM head` + `gatherv` + `RS` | `5,405` | 32K：`20,421–20,700` | 32K：`4,036–4,074`<br>短提示：`2,919` | `gatherv` 提速 64 路冷 P `+26.4%`；FP8 KV `20.89M` |
| **04. DeepInfer 第一档与 U5 修复**<br>[`04-deepinfer-tier1-单元验证报告.md`](bench/reports/04-deepinfer-tier1-单元验证报告.md)<br>[`repair-fp32mid-r3-summary.json`](bench/reports/data/repair-fp32mid-r3-summary.json) | `A1/A2/A3/A7` + 外置 `repair-a2-reader`（FP32 `mid_out`） | — | — | Reader CUDA Graph 加速 **`10.2%–18.0%`** | 消除 Split-K 二次 BF16 截断误差（RMS `-25%~-29%`），全 6 形状通过 U5 |
| **05. FP4 KV + 3h 配对基线落盘**<br>[`05-dev3h-配对开发验证报告-20261116.md`](bench/reports/05-dev3h-配对开发验证报告-20261116.md)<br>[`sd-dev3h-r1-analysis.json`](bench/reports/data/sd-dev3h-r1-analysis.json) | Commit `2934cd4`：`mem=0.85, tails=384`，主 KV `67.45M` | `2,315`（热 `~8,350`） | 32K：`20,543`<br>128K：`20,660` | 32K：`4,214`<br>128K 纯 D：`3,567`<br>短提示：`2,922` | `64×128K` 命中 `99.79%`，PPL `2.0692`（`+0.067%`），清理全部旧 TP4 残留 |
| **06. Step 1 + Step 2 通信与 MXMMA MoE**<br>[`06-tp8-step1-step2-优化报告.md`](bench/reports/06-tp8-step1-step2-优化报告.md)<br>[`sd-popt3b-analysis.json`](bench/reports/data/sd-popt3b-analysis.json) | Commit `cbb364c`：FP8 Gatherv + UAF 修复 + 目标 MoE `MXMMA e3m4 pad(128,128)` | **`8,632`**（`3.73×`） | 32K：**`24,851` (`+21.0%`)**<br>128K：**`25,158` (`+21.8%`)** | 32K：**`4,740.5` (`+12.5%`)**<br>128K 纯 D：`3,671.6`<br>短提示：`3,095` | 每卡省权重 **`4.98 GiB`**，主 KV 暴涨 **`+69.5%`** 至 **`1.143 亿`**；PPL `2.0714` |
| **07. Step 3 双引擎算子与自适应分块**<br>[`07-tp8-step3-双引擎深度优化报告.md`](bench/reports/07-tp8-step3-双引擎深度优化报告.md)<br>[`sd-popt4-analysis.json`](bench/reports/data/sd-popt4-analysis.json) | Commit `6bba7f4`：`wo_a MAX_T=56` + 尾层深度守卫激活单发 `16K` 续块 + `CHUNK=49152` | **`10,886`（`4.70×`，TTFT `3.01s`）** | 32K：**`26,106` (`+27.1%`)**<br>128K：**`26,129` (`+26.5%`)** | 32K：`4,672`（窗口 `4,746`）<br>128K 纯 D：`3,706`–`3,800`<br>短提示：**`3,252`** | Decode `_dequant_kernel` 清零（`wo_a` 耗时 `-49%`）；`2×300K` TTFT 缩至 **`78.4s`** |
| **08. Step 4 全形状 MXFP8 表与显存加固**<br>[`08-tp8-step4-mxfp8-全形状调优报告.md`](bench/reports/08-tp8-step4-mxfp8-全形状调优报告.md)<br>[`sd-popt5-analysis.json`](bench/reports/data/sd-popt5-analysis.json) | Commit `d248b46`：10 形状 MXFP8 最优表 + 草稿 `MXMMA e3m4` + DP8 `sps_profile` 修复 + `CHUNK=40960` | **`10,580`**（`4.57×`） | 32K：`25,548`<br>128K：`25,756` | 短提示：**`3,285.5`（窗口 `3,318.6`）**<br>32K P50 TPOT：**`9.245 ms`**<br>128K：**`49.75 ms/步`** | 含 8 卡 CUPTI Profiler 全程最低空闲 **`2.135 GiB`**，PPL **`2.0678`**（同 `p0`） |
| **09. Step 5 DP8 原生 SPS 表巅峰融合**<br>[`09-tp8-step5-dp8-sps-巅峰融合报告.md`](bench/reports/09-tp8-step5-dp8-sps-巅峰融合报告.md)<br>[`sd-popt6-analysis.json`](bench/reports/data/sd-popt6-analysis.json) | Commit `960e102`：实测 `86kd-sps-6000d-dp8.json`（30 轮 `match=1.0`）+ `CHUNK=49152` | **`10,674`**（`4.61×`） | 32K：**`26,042` (`+26.8%`)**<br>128K：**`26,478.5` (`+28.2%`)** | 32K：**`4,862.6`（窗口 `4,967.2`，`+15.4%`）**<br>128K 纯 D：**`3,894.5`（`47.19 ms/步`）** | **P50/P95 TPOT `8.885 / 9.404 ms`**，PPL `2.0708`（`+0.078%`），最低空闲 **`1.949 GiB`** |

---

## 四、8-Rank 硬件 Profiler 热点消除量化证据

基于 Rank 0 真实 `torch.profiler` 硬件核轨迹（Prefill 为 `1×32K` 整段请求，Decode 为 `conc=64` 稳态连续 `5` 个验证步 = `200` 层目标 + `15` 层草稿）：

| GPU 硬件算子 / 通信核类别 | 优化前 (`sd-dev3h-r1-p1`) | Step 1+2 (`sd-popt3b-s12`) | Step 3 (`sd-popt4b-s123`) | Step 4 (`sd-popt5-s1234`) | 累计优化机制与降幅 |
|---|---:|---:|---:|---:|---|
| **Prefill 跨卡 `ncclBroadcast_RING_LL`（单块）** | `210.64 ms` | **`113.52 ms`** | `131.29 ms`（大块） | `131.35 ms` | FP8 DP Gatherv 字节减半（单块 **`-46.1%`**，总块数 `8 -> 3`） |
| **Prefill Humming MoE 核总耗时（32K 全段）** | `419.18 ms`（W4A16） | `283.50 ms` | `169.72 ms` | **`139.71 ms`** | 目标 + 草稿 MoE 切入 SM120 原生 `MXMMA float8e3m4`（**`-66.7%`**） |
| **Prefill `b12xDenseGemmKernel` 调用与耗时** | `848` 次 / 高占比 | `848` 次 / `327.30 ms` | `318` 次 / `427.46 ms` | **`72` 次 / `218.57 ms`** | Step 3D 减块 + Step 4A 切入 CUTLASS TMA `tactic=2/3/5`（调用 **`-91.5%`**） |
| **Decode `wo_a` `_dequant_kernel`（64 路 5 步）** | `200` 次 / `11.53 ms` | `200` 次 / `11.53 ms` | **`0` 次 / `0.00 ms`** | **`0` 次 / `0.00 ms`** | `XTIER_DS_WO_A_MAX_T=56` 融合 FP8 `_einsum_kernel`（**`-100%` 清零**） |
| **Decode `wo_a` 投影总耗时（64 路 5 步）** | `17.98 ms` | `17.98 ms` | **`9.17 ms`** | **`9.15 ms`** | 消除反量化 + BF16 GEMM（**`-49.1%`**） |
| **Decode `b12x` 小矩阵核调用数（64 路 5 步）** | `> 1,100` 次 | `1,150` 次 | `935` 次（`30.81 ms`） | **`470` 次（`14.37 ms`）** | `shared.w13` (`M=320/384`) 与 `wq_b` (`M=40/48`) 分流最优 CUTLASS Tile |

---

## 五、仓库目录结构

```text
xtier_86kd/
├── src/xtier_86kd/                  # 86KD TP8 独立加速插件源码
│   ├── plugin.py                    # SGLang 插件入口（XTIER_86KD=1 自动挂载）
│   ├── mxfp8_prefill.py             # SM120 10 形状 100% 逐比特一致 (backend, tactic) 选择器与 TMA Tile 钩子
│   ├── wo_a.py                      # 通用 FP8 wo_a 分组 einsum 核
│   └── packs/deepseek_v4/           # DeepSeek-V4.1 专属加速包（MAX_T=56 wo_a、变长验证、分页索引器、FP8 草稿头）
├── deploy/
│   ├── prefill-opt/                 # 外置零侵入 Step 1/2/3D/4B 钩子（FP8 Gatherv UAF 修复、MXMMA e3m4 MoE、16K 单流续块）
│   ├── repair-a2-reader/            # 外置零侵入 DeepInfer A2 Split-K FP32-mid CUDA 内核与自动挂载钩子
│   ├── tf500_console.py             # :8100 启动台后端（默认搭载 Step 1–5 巅峰配置，支持下拉一键回退基线）
│   ├── tf500_api_proxy.py           # :8000 OpenAI 兼容多租户鉴权网关与配额计费（密钥存主机状态目录，绝不入库）
│   ├── tf500_benchmark.py           # /benchmark 在线多并发与长上下文 Agent 测速面板
│   ├── tf500_server_info.py         # /server 8 卡实时功耗、频率、显存、温度监控面板
│   └── start-tp8-fp4-card.sh        # 生产卡片对应的 TP8 SGLang 启动脚本
├── patches/
│   ├── sm120-official-fp4-kv.patch  # SM120 官方混合 V4.1 FP4 主 KV + FP8 SWA 补丁
│   └── deepinfer-tier1/             # DeepInfer 第一档（A1/A2/A3/A7）分项 diff 与完整补丁快照
├── bench/
│   ├── sps_profile.py               # 支持 --attn-dp-size 8 均衡路由的 DSpark 变长验证成本标定工具
│   ├── 86kd-sps-6000d-dp8.json      # 实测 8 卡 Attention-DP8 专属 SPS 成本表（默认推荐）
│   ├── 86kd-sps-6000d.json          # 旧版 SPS 成本表（超短无缓存输入备选）
│   ├── bench_serving.py             # 多并发两阶段冷 Prefill / 共享前缀 Decode 测速工具
│   └── reports/                     # 01–09 全阶段中文实测报告与 data/*.json 原始结构化证据
└── tests/                           # 17 项本地零依赖单元测试与在线只读巡检脚本
```

---

## 六、部署、启动台使用与一键回退指南

### 1. 安装插件与日常启动（WebUI `:8100` 推荐）

```bash
/data/xtier-env/bin/pip install -e /data/src/xtier_86kd --no-deps
```

- **打开 TF500 启动台**：在本地执行 `ssh -L 8100:127.0.0.1:8100 <user>@<TF500_IP>`，浏览器访问 `http://127.0.0.1:8100/`：
  - **默认配置（已同步最新最优栈）**：`双引擎深度加速 = 开`、`DSpark 验证成本表 = 变长 · DP8 实测表`、`Prefill 分块 = 48K（49152）`、`静态显存比例 = 0.85`、`SWA 尾部预算 = 384`、`DeepInfer 第一档对齐 = 开`（点击「启动」按钮后才加载模型，控制台本身不占任何显存）。
  - **场景微调与一键回退**：
    - 若需挂载重度 GPU 诊断工具（如 8 卡 CUPTI Profiler），可在下拉框将 `Prefill 分块` 切为 `40K · 高余量稳健（40960）`；
    - 若业务全部为 `< 64` token 无缓存极短请求，可将成本表切为 `变长 · 旧版成本表`；
    - 若需回退至优化前 `sd-dev3h-r1` 历史基线做 A/B 对照，仅需将 `双引擎深度加速` 切为 `关`、`Prefill 分块` 切为 `32K`。
- **OpenAI 兼容 API 入口**：
  - Base URL：`http://<TF500_IP>:8000/v1`，Model：`DeepSeek-V4.1-Flash-1017`（在 `/proxy` 页面管理客户端子密钥）。

### 2. 核心环境变量速查表（CLI 手动启动用）

| 环境变量 | 推荐生产值 | 作用说明（回退设法） |
|---|---|---|
| `XTIER_86KD` | `1` | 启用 `xtier_86kd` 插件钩子（设 `0` 完全停用插件） |
| `XTIER_PREFILL_OPT` | `1` | 启用 `deploy/prefill-opt` 外置加速钩子（需配合 `PYTHONPATH` 首项指向 `deploy/prefill-opt`；设 `0` 停用） |
| `SGLANG_ENABLE_DP_GATHER_FP8` | `1` | 变长 DP Gatherv FP8 组量化跨卡压缩 + 持久化安全缓冲池（设 `0` 恢复 BF16 Gatherv） |
| `XTIER_PREFILL_NCCL_SKIP_ZERO` | `1` | 跳过变长 `all_gather` / `reduce_scatter` 中长度为 0 的空闲 Rank |
| `XTIER_PREFILL_MOE_MXMMA` | `e3m4` | 主模型 384 专家 MoE 启用 SM120 原生 `MXMMA float8e3m4 + e8m0`（设空字符串回退 W4A16） |
| `XTIER_DRAFT_MOE_MXMMA` | `e3m4` | DSpark 草稿 128 专家 MoE 启用 `MXMMA float8e3m4`（设 `off` 单独关闭草稿 MXMMA） |
| `XTIER_MOE_PAD_N` / `PAD_K` | `128` / `128` | Humming 专家权重按 `128` 紧凑对齐，每卡省 `4.98 GiB` 显存 |
| `XTIER_IDLE_SINGLE_CHUNK` | `16384` | 全集群仅单卡活跃且无排队时单流续块自动提至 `16K`（设 `0` 关闭） |
| `XTIER_86KD_MXFP8_TABLE` | `1` | 启用 SM120 10 形状 100% 逐比特一致 `(backend, tactic)` 最优表（设 `0` 回退） |
| `XTIER_DS_WO_A_MAX_T` | `56`（默认） | 覆盖 DP8 64 并发 `T=40/48` 清零解码反量化（设 `16` 回退旧阈值） |
| `SPS_TABLE` | `.../bench/86kd-sps-6000d-dp8.json` | Attention-DP8 实测变长验证成本表 |

### 3. 运行单元测试

全部 17 项单元测试均基于纯内存 Mock，不依赖真实 GPU 或在线推理服务：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 七、后续二期开发计划（`Stage 1 → Stage 4` 定版路线图）

在一期 Step 1–5 完成落盘的基础上，结合 [`sd-popt5-analysis.json`](bench/reports/data/sd-popt5-analysis.json) 的 8-Rank 硬件 Profiler 剩余热点与 `tf500` 双路 Xeon 6737P（128 线程 + `597 GB/s` DDR5 + 单卡 `56.5 GB/s` PCIe 5.0）硬件拓扑，制定二期四步递进提速计划（完整技术蓝图与门禁细则见 [`10-tp8-二期提速规划-stage1-4.md`](bench/reports/10-tp8-二期提速规划-stage1-4.md)）：

### 1. **`Stage 1`：Decode CUDA Graph 图内 FP8 `dp_gather` 跨槽压缩 + 双路 NUMA 物理绑核**
- **剩余瓶颈**：一期 `SGLANG_ENABLE_DP_GATHER_FP8=1` 仅覆盖变长 Prefill（`all_gatherv`）；64 并发 Decode 图内的定长 `_dp_gather` 仍在跨双路 CPU UPI 总线跑 BF16 `AllGather`（每 5 步耗时 `34.02 ms`，叠加 `ReduceScatter` 共占单步总耗时 **`27.2%`**），且 8 个进程尚未锁死物理核亲和性。
- **实施方案**：① 在 `deploy/prefill-opt` 中为 Decode 图预分配固定虚拟地址持久缓冲，将每 token `[5120B FP8 + 160B FP32 group-128 scale]` 打包为连续 `5280B` 单张量执行 **单次 `tp_group.all_gather` + Triton 融合解包反量化**（跨 UPI 流量减少 **`48.4%`**，捕获期零动态分配）；② 按拓扑将 `TP Rank 0–3`（`GPU 0–3`）严格绑至 `NUMA 0` 物理核、`TP Rank 4–7`（`GPU 4–7`）绑至 `NUMA 1` 物理核，并将 HTTP / 分词 / 流式工具解析隔离至预留核组。
- **目标收益**：64 路 Decode **`+5% ~ +8%`**（`64×32K` 冲击 **`5,150–5,250 tok/s`**），P95 TPOT 毛刺进一步收敛；独立开关 `XTIER_DECODE_DP_GATHER_FP8=1`、`XTIER_NUMA_PIN=1`。

### 2. **`Stage 2`：融合 `mHC`（超连接残差混合）算子链（替换旧 SM80 BF16 核）**
- **剩余瓶颈**：32K 冷 Prefill Trace 中，`mhc_pre`（仍调用旧版 SM80 `s16816gemm` BF16 核，`211.16 ms`）+ `_hc_mix_stats_partial`（`115.08 ms`）+ `mhc_post_tilelang`（`88.04 ms`）+ `_hc_combine_norm`（`54.49 ms`）四项合计耗时 **`468.77 ms`（占整段 Prefill `~24%`，第二大耗时模块）**，因 `[M, 20480]` 高维残差流反复往返显存（`M=49152` 单次往返达 `4.03 GB`）。
- **实施方案**：编写 SM120 原生无损融合算子（`src/xtier_86kd/packs/deepseek_v4/fused_mhc.py`），前段合并统计量归约与 `mhc_pre` 投影，后段将 `mhc_post` 加权合并与下层入口 RMSNorm 融合为单遍显存读写（100% 数学等价，与 MoE 走 TP8 或 EP8 完全正交）。
- **目标收益**：冷 Prefill **`+7% ~ +11%`**（`64×128K` 冲击 **`28,500–29,300 tok/s`**），Decode **`+2% ~ +3%`**；独立开关 `XTIER_86KD_FUSED_MHC=1`。

### 3. **`Stage 3`：CPU 128 核异步提前执行 `Engram` 哈希、DDR5 本地聚集与双缓冲 PCIe 5.0 预取**
- **剩余瓶颈**：第 1 层与第 14 层共 `189.13 GiB` 的 FP8 `Engram` 表驻留主机内存，现有路径在 GPU 执行到层内时触发跨槽同步读取（Decode 每 5 步 `ncclSendRecv` 占 `9.32 ms`），而双路 Xeon 128 线程与 `597 GB/s` DDR5 带宽在 GPU 计算前置层时处于闲置。
- **实施方案**：复用 `Stage 1` 绑核线程池与 `csrc/` C++/OpenMP 扩展，在调度器确定本步 `input_ids`（第 0 层启动前）即由本地 NUMA CPU 并行完成 24 列 N-gram 哈希、DDR5 行聚集与原地 BF16 反量化（64 路 Decode 每步仅 `2.43 MB`，耗时 `< 15 us`），经独立 `copy_stream` 走 PCIe 5.0（`56.5 GB/s`）写入固定地址 Ping-Pong 双缓冲，第 1/14 层在 CUDA Graph 内仅做 `wait_event` 零等待消费。
- **目标收益**：Decode **`+3% ~ +5%`**，冷 Prefill **`+2% ~ +4%`**；独立开关 `XTIER_CPU_ENGRAM_PREFETCH=1`。

### 4. **`Stage 4`：MoE `EP8`（专家并行）+ 共享专家本地双流重叠（带 15 分钟微基准硬门禁）**
- **架构潜力与风险**：路由专家切 `EP8`（每卡完整持有 `48` 个全宽专家，消除 `288 -> 384` 填充再省约 `1.8 GiB/卡` 显存）并将仅 `0.94 GiB` 的共享专家留在 DP 本地副流与 **FP8 `AllToAll`** 完全重叠，可省去一次全局同步屏障；但 `tf500` 无 NVSwitch，跨双路 UPI 的小包 `AllToAll` 存在尾延迟风险。
- **硬门禁执行准则**：先跑 15 分钟 8 卡 CUDA Graph 微基准对比 `[EP8 FP8 AllToAll + 本地共享专家重叠]` vs `[Stage 1–3 最优 TP8]`——**仅当 `M_global=384` 端到端净胜 `>= 3.0%` 且 Prefill 不回退时才推进整模合入；否则立即熔断跳过并归档微基准数据**。
- **目标收益**：若跨槽延迟压住，64 路 Decode 潜在 **`+8% ~ +15%`**。

### 5. **冻结观察（暂不处理）：`Stage 5` — 主机内存 `HiCache` 二级 KV 缓存池**
- **冻结原因**：利用闲置 `512–640 GiB` DDR5 构建二级 KV 池属于业务层多轮缓存扩容优化。由于 `DeepSeek-V4.1-Flash` 同时包含 **FP4 压缩主池 + FP8 SWA 滑动窗口池 + `--swa-prefix-tails 384` + `DecoderReplayTail` 有界重放**，双异构池在跨池驱逐、部分命中换入与引用计数回收交界处极易出现隐蔽一致性边界问题；故在 `Stage 1–4` 全部收敛后视最终情况再议，**默认严格不动**。（此外，鉴于 8 卡 GDDR7 带宽 `10.24 TB/s` 为双路 DDR5 `0.60 TB/s` 的 17 倍且 64 并发每步命中 `99.97%` 专家，CPU MoE 算力卸载方案已明确永久排除。）

---

## 八、许可证

Apache-2.0，详见 [`LICENSE`](LICENSE)。
