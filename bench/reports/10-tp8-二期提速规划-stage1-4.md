# 10. TF500 TP8 二期深度提速规划（Stage 1–4 实施方案与冻结边界说明）

- **制定日期**：2026-11-17
- **基线版本**：一期 Step 1–5 收官版（一期终版，对应实测标签 `sd-popt6-dp8sps` / `sd-popt5-s1234`）
- **定位依据**：[`sd-popt5-analysis.json`](data/sd-popt5-analysis.json)（8-Rank 硬件 Profiler 剩余热点）与 [`01-tf500-前置检查与测速.md`](01-tf500-前置检查与测速.md)（双路 Xeon 6737P / DDR5 / PCIe 5.0 实测带宽账本）
- **工程铁律**：继续严守对 `/data/src/sglang-align` 与 `/data/xtier-env` **零侵入修改**；每完成一个 Stage 均须通过 `Smoke 3/3`、`3-seg PPL`（相对基线波动 `<= ±0.5%`）、全程 `min_free_gib >= 1.26 GiB` 与单元测试后独立推送 GitHub。

---

## 一、一期优化后的硬件剩余瓶颈全景（为什么要做这四步？）

经过一期 Step 1–5 优化后，`wo_a` 解码反量化已归零、Humming 目标/草稿 MoE 已切入 SM120 原生 `MXMMA float8e3m4`、10 个 MXFP8 投影已按真机最优 Tile 分流。此时 [`sd-popt5-analysis.json`](data/sd-popt5-analysis.json) 的 Rank 0 硬件 Trace 清晰揭示出剩余耗时集中的四大板块：

1. **64 并发 Decode 跨插槽集合通信（占单步总耗时 `~27.2%`）**：
   - 连续 5 个验证步中，`ncclDevKernel_AllGather_RING_LL`（`230` 次，`34.02 ms`）+ `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`（`215` 次，`34.44 ms`）合计 **`68.46 ms`（`13.69 ms/步`）**。
   - 原因：一期 `SGLANG_ENABLE_DP_GATHER_FP8=1` 仅作用于变长 Prefill（`all_gatherv`），而 Decode CUDA Graph 内的定长 `_dp_gather` 仍在跨双路 CPU 的 UPI 总线传输 BF16 张量（每步 43 层约 `167 MB` BF16 数据，正处于跨插槽 1–4MB `14.8–22 GB/s` 带宽凹陷区）；同时 8 个工作进程尚未做严格 NUMA 物理核锁死。
2. **`mHC` 超连接残差流混合算子链（占 32K 冷 Prefill 总耗时 `~24.0%`）**：
   - `1×32K` Prefill 中，`mhc_pre`（仍在调用旧版 SM80 `cutlass_80_tensorop_bf16_s16816gemm`，`60` 次 `211.16 ms`）+ `_hc_mix_stats_partial_kernel`（`240` 次 `115.08 ms`）+ `mhc_post_tilelang_kernel`（`120` 次 `88.04 ms`）+ `_hc_combine_norm_prefill`（`120` 次 `54.49 ms`）四项合计 **`468.77 ms`**（仅次于稀疏 MLA 主核的第二大耗时源），Decode 每 5 步亦占 `~11 ms`。
   - 原因：`20,480` 维（`4 × 5120`）高维超连接残差张量在多个小核之间反复落盘读写 GDDR7（`M=49152` 单次往返即产生 `4.03 GB` 显存流量）。
3. **第 1 / 14 层 `Engram` 主机表跨槽同步与层内等待**：
   - 两层共 `189.13 GiB` 的 FP8 记忆嵌入表驻留主机 DDR5 内存；当前由 GPU 执行到对应层时发起同步与读取（Decode 每 5 步 `ncclDevKernel_SendRecv` 占 **`9.32 ms` = `1.86 ms/步`**，Prefill 占 `18.92 ms`），而此时双路 Xeon 6737P 的 **128 个逻辑线程与 `597 GB/s` DDR5 带宽几乎完全闲置**。
4. **MoE TP8 全局同步与共享专家串行等待**：
   - 当前路由专家采用 TP8，每层前后需各做一次 8 卡全局集合通信，且权重极小的共享专家（每层 FP8 仅 `23.6 MB`，40 层共 `0.94 GiB`）被迫串行排在 `AllGather` 之后。

---

## 二、定版执行路线图（`Stage 1 → Stage 4`）详细设计

### Stage 1：Decode CUDA Graph 图内 FP8 `dp_gather` 跨槽压缩 + 双路 NUMA 物理绑核

- **改动模块**：`deploy/prefill-opt/prefill_opt_hook.py`（`install_onto_dp_attention`、新增 `install_numa_affinity`）、`deploy/start-tp8-fp4-card.sh`
- **详细实施方案**：
  1. **图捕获安全定长 FP8 `dp_gather`**：
     - 在非变长 Decode 分支（`forward_batch.is_extend_in_batch == False` 且启用 `XTIER_DECODE_DP_GATHER_FP8=1`）中，于 CUDA Graph 捕获前按最大解码图批量（`max_bs_per_rank * verify_tokens * 8 = 384` 行，`hidden = 5120`）在对称/持久池预分配固定虚拟地址的量化发送缓冲 `q_local [48, 5120] uint8`、`s_local [48, 40] float32` 与全局接收缓冲 `gq [384, 5120] uint8`、`gs [384, 40] float32`；
     - 为避免两次小包 `all_gather` 带来的双倍 NCCL 启动开销，将每行 `5120B FP8 + 160B FP32 scale` 融合打包为单行连续 **`5280 Bytes`（对齐 16B）** 的单一张量执行 **单次 `tp_group.all_gather`**，随后由单次融合 Triton 核完成解包 + `group-128` 反量化写入 `global_tokens`；
     - 跨双路 UPI 字节量由 `10,240 B/token` 降至 `5,280 B/token`（**净减 `48.4%`**），且单层仅发起 **1 次** NCCL Kernel。
  2. **双路 NUMA 物理核与本地内存亲和性锁死**：
     - `tf500` 拓扑为 `GPU 0–3 ∈ NUMA 0`、`GPU 4–7 ∈ NUMA 1`；
     - 在 Worker 进程初始化早期按 `tp_rank` 自动读取 `/sys/devices/system/node/node{0,1}/cpulist`，将 `Rank 0–3` 严格绑定至 `NUMA 0` 物理核（每 Rank 独占 7–8 个物理核）、`Rank 4–7` 严格绑定至 `NUMA 1` 物理核，并将主进程 HTTP / Tokenizer / Detokenizer / Harmony 流式工具解析隔离至各插槽预留核组，消除跨插槽页迁移与调度抖动。
- **预期量化收益**：64 路 Decode 吞吐 **`+5% ~ +8%`**（`64×32K` 冲击 **`5,150–5,250 tok/s`**，`64×128K` 单步压向 `44.5–45.5 ms`），P95 TPOT 毛刺显著收敛。
- **独立回退开关**：`XTIER_DECODE_DP_GATHER_FP8=1`、`XTIER_NUMA_PIN=1`（设 `0` 完全恢复原路径）。

---

### Stage 2：融合 `mHC`（超连接残差混合）算子链（替换旧 SM80 BF16 核）

- **改动模块**：新增 `src/xtier_86kd/packs/deepseek_v4/fused_mhc.py` 并通过 `HookRegistry` 挂载
- **详细实施方案**：
  1. **消除 `mhc_pre` 前序显存往返与旧架构核**：
     - 针对 `[M, 20480]` 多流残差输入，将 `_hc_mix_stats_partial_kernel` 的均值/方差/范数统计与 `mhc_pre` 线性投影合并，或针对 SM120 定制高带宽 Split-K / TMA BF16 核（FP32 累加），消除旧版 `cutlass_80_tensorop_bf16_s16816gemm` 在 Blackwell SM120 上寄存器与共享内存利用率不足的问题。
  2. **融合 `mhc_post + _hc_combine_norm`（单遍读写）**：
     - 将 Sinkhorn 混合系数应用后的多流残差写回与下一子层入口 RMSNorm 融合为单次 Triton/CUDA Kernel（Single-Pass Read/Write），直接省去每层一次 `[M, 20480]`（`M=49152` 时高达 `2.01 GB` 写 + `2.01 GB` 读）的 GDDR7 往返流量。
- **验证门禁与预期收益**：
  - **100% 数学等价无损**：单元测试在 `M = 6, 48, 384, 6144, 16384, 49152` 全尺寸下核对 BF16 输出逐元素误差（`max_abs_diff <= 1 ULP`，3 段 PPL 零漂移）；
  - **预期收益**：冷 Prefill **`+7% ~ +11%`**（`64×128K` 冷 Prefill 冲击 **`28,500–29,300 tok/s`**，`1×32K` 冲击 **`11,600+ tok/s`**），Decode **`+2% ~ +3%`**。
- **独立回退开关**：`XTIER_86KD_FUSED_MHC=1`（设 `0` 回退 SGLang 原生算子链）。

---

### Stage 3：CPU 128 核异步提前执行 `Engram` 哈希、DDR5 本地聚集与双缓冲 PCIe 5.0 预取

- **改动模块**：扩展 `csrc/` C++/OpenMP 聚集扩展与 `src/xtier_86kd/packs/deepseek_v4/__init__.py`
- **详细实施方案**：
  1. **提前至第 0 层启动前的 CPU 侧全流水线**：
     - 复用 `Stage 1` 建立的双路 NUMA 线程隔离，在 `NUMA 0` 与 `NUMA 1` 各驻留一组无锁 C++ OpenMP 工作线程；
     - 当本步 `input_ids` 在调度器/ModelRunner 入口确定时（**早于第 0 层 Attention/MoE 上卡执行**），CPU 立即对全序列计算第 `1` 层与第 `14` 层的 24 列 N-gram 哈希 ID，直接从本地 NUMA DDR5（合计 `597 GB/s`）并行拉取 `[256B FP8 + 8B E8M0]` 行并利用 AVX-512 / AMX 原地反量化为紧凑 BF16 激活写入固定页锁定内存（Pinned Memory）。
     - *物理耗时核算*：64 并发 Decode 每步仅 `384 × 24 = 9,216` 行（`2.43 MB`），CPU 本地内存聚集仅需 **`< 15 us`**，PCIe 5.0（单卡 `56.5 GB/s`）异步 DMA 入卡仅需 **`~45 us`**，完全被第 `0` 层（`~1.2 ms`）与第 `2–13` 层（`~14 ms`）100% 隐藏！
  2. **CUDA Graph 安全的双缓冲（Ping-Pong）+ `cudaEvent` 同步**：
     - 为每个解码图批量档位预分配固定 GPU 地址接收槽，独立 `copy_stream` 负责 H2D 传输并记录 `done_event`；主计算流在进入第 `1` 层和第 `14` 层 `engram.wkv` 前仅做 `main_stream.wait_event(done_event)`，彻底消除跨插槽 `ncclSendRecv` 阻塞。
- **预期量化收益**：Decode **`+3% ~ +5%`**，冷 Prefill **`+2% ~ +4%`**。
- **独立回退开关**：`XTIER_CPU_ENGRAM_PREFETCH=1`（设 `0` 回退现有 GPU 侧主机表查找）。

---

### Stage 4：MoE `EP8`（专家并行）+ 共享专家本地双流重叠（带 15 分钟微基准硬门禁）

- **改动模块**：先执行独立微基准探针，通过门禁后再接入 `deploy/prefill-opt` 与启动参数 `--ep-size 8`
- **详细实施方案与两段式门禁**：
  1. **架构收益点**：
     - 路由专家改用 `EP8` 后，每张卡完整持有 `384 / 8 = 48` 个专家的全宽权重（`N_w13 = 4608`、`K_w2 = 2304`），彻底消除 TP8 下 `288 -> 384` 的零填充算力与显存开销（每卡可再省约 **`1.8 GiB`** 权重显存）；
     - 共享专家（`shared_experts`，40 层总重仅 `0.94 GiB`）直接在每张卡的 Attention DP 本地激活（`M_local = 48`）上运行于副流 `shared_stream`，与主流的 **FP8 `AllToAll` 令牌分发 + 48 专家 MXMMA 矩阵乘 + `AllToAll` 结果归并** 完全并行重叠，省掉一次全局同步屏障。
  2. **15 分钟微基准硬门禁（防跨 UPI 尾延迟踩坑）**：
     - 鉴于 `tf500` 无 NVSwitch（GPU 0–3 与 4–7 跨双路 Xeon UPI 总线通信），小批量全对全 `AllToAll` 可能产生队列争用；
     - 在动整模代码前，先运行 8 卡实机微基准，严格对比 CUDA Graph 内的 **`[EP8 FP8 AllToAll + 48 专家全宽 MXMMA + 本地共享专家重叠]`** 与 **`[Stage 1–3 最优 TP8]`** 在 `M_global = 8, 48, 320, 384, 49152` 下的端到端耗时：
       - **准入线**：`M_global = 384`（64 并发 Decode）单层端到端净胜 **`>= 3.0%`** 且 `M_global = 49152`（Prefill）不劣化 → 进入整模 `EP8` 适配与全套验收；
       - **熔断线**：若净胜 `< 3.0%` 或变慢 → **15 分钟内立即熔断终止 `Stage 4`**，将实测对比数据归档入库，保留最优 TP8 架构作为终态。

---

## 三、冻结观察项与明确排除项说明

### 1. 冻结观察（暂不处理）：`Stage 5` — 1 TiB 主机内存 `HiCache` 二级 KV 缓存池
- **方案定位**：利用主机闲置的 `512–640 GiB` DDR5 内存构建二级 KV 缓存池，属于面向多轮对话 / Agent 复用的**业务层缓存扩容优化**，而非底层单步算力加速。
- **暂不处理的工程原因**：`DeepSeek-V4.1-Flash` 当前同时启用 **FP4 E2M1 压缩主 KV 池（`288B/token`）+ FP8 E4M3 SWA 滑动窗口池（`528B/token`）+ `--swa-prefix-tails 384` + `DecoderReplayTail` 尾层有界重放**。在双异构池并存条件下开启 `HiCache` 主机换入换出，极易在部分前缀命中、尾层窗口截断与 RadixTree 引用计数回收交界处触发隐蔽的数据一致性或显存越界问题。因此在 `Stage 1–4` 全部落地稳定前 **默认冻结不碰**，后续视实际业务多轮缓存淘汰率再决定是否单独开分支攻关。

### 2. 永久排除项（已从计划剔除）
- **CPU 混合专家计算卸载（XTier / KTransformers MoE Offload）**：`tf500` 8 卡 GDDR7 显存总带宽为 `10.24 TB/s`，是双路 Xeon DDR5 总带宽（`0.60 TB/s`）的 **17 倍**；64 并发下每步激活 `99.97%` 专家，将任何比例的专家算力卸回 CPU 都会导致 Decode 与 Prefill 吞吐严重倒退。
- **Prompt-Lookup 后缀草稿 / 手写 SM120 FP8-Q 稀疏 MLA Prefill 核 / 主板 BIOS QAT 压缩**：分别因对非复现通用跑分无增益、FP8 Query 易破坏 DeepInfer U5 精度对齐且工时过重、以及需改动宿主机 BIOS 重启而明确剔除。
