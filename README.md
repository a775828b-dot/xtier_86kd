# xtier_86kd：TF500（8 × RTX PRO 6000D）DeepSeek-V4.1-Flash TP8 加速包、对齐补丁与实测归档

本仓库专用于 **`tf500`（8 × NVIDIA RTX PRO 6000D，Blackwell SM120，双路 Xeon 6737P + PCIe 5.0）** 上的 **`DeepSeek-V4.1-Flash-1017` 全显存 TP8 / Attention DP8 高性能推理部署**。

仓库集中收录了：
1. **`xtier_86kd` 独立加速插件**（无需启用 XTier CPU/GPU 混合专家引擎，直接挂载在 SGLang + Humming MoE 之上；旧 TP4 混合方案已废弃移除）；
2. **SM120 官方混合 KV 缓存补丁**（主压缩 KV 为 `FP4 E2M1 + E4M3 scale` 288B/条，SWA 滑动窗口为 `FP8 E4M3 + UE8M0 scale` 528B/条）；
3. **DeepInfer 第一档精度对齐补丁（A1 / A2 / A3 / A7）** 与 **外置无侵入 `repair-a2-reader`（SM120 Split-K FP32 中间累加修复）**；
4. **TF500 启动台 WebUI（`:8100`）、在线测速面板与鉴权 API 网关（`:8000`）**；
5. **从装机前置检查到最新 3 小时精简配对验证的全阶段实测报告与结构化 JSON 数据**。

---

## 一、当前基线架构与核心实测指标（`sd-dev3h-r1-p1`）

### 1.1 运行配置栈

| 维度 | 当前验证配置 |
|---|---|
| 硬件平台 | `8 × NVIDIA RTX PRO 6000D`（每卡 84 GiB GDDR7，SM120，双路 NUMA，跨 CPU Socket `NCCL_P2P_LEVEL=SYS`） |
| 并行策略 | `--tp-size 8 --attn-dp-size 8 --enable-dp-lm-head` + `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1` |
| MoE / GEMM | `--moe-runner-backend humming` + `--fp8-gemm-backend flashinfer_cutlass` + `XTIER_86KD=1`（Engram 表驻留主机 1 TiB 内存） |
| 投机解码 | `--speculative-algorithm DSPARK`（`block=5, draft=6`，`compact` 变长验证 + 本机标定表 `bench/86kd-sps-6000d.json`） |
| KV 缓存与容量 | `MEM_FRACTION_STATIC=0.85`、`--swa-prefix-tails 384`、上下文上限 `1,048,576`；**实测全局主 KV `67,450,880` token**（每 DP `8,431,360`），每 DP SWA 池 `165,376` token |
| DeepInfer 对齐 | `SGLANG_DI_ALIGN_A1=1 A2=1 A3=1 A7=1`（`SGLANG_DI_ALIGN=0`，`A4/A5/A8=0`）+ `deploy/repair-a2-reader` |

### 1.2 全阶段演进与关键实测成绩总览

> 注：不同阶段使用了不同 KV 位宽、`CASE_ID` 或预热策略，跨阶段横向引用须结合对应报告中的测试条件；最新 `sd-dev3h-r1` 为单轮限时开发配对验证（非 3 轮中位数或 1M 正式全量验收）。

| 阶段与报告入口 | 配置要点 | 并发 1 冷 P (`tok/s`) | 并发 32/64 冷 P (`tok/s`) | 缓存 D (`tok/s`) | 关键结论 |
|---|---|---:|---:|---:|---|
| **01. 装机检查与纯 TP8 初测**<br>[`01-tf500-前置检查与测速.md`](bench/reports/01-tf500-前置检查与测速.md) | 纯 TP8（无 DP）、FP8 KV、`mem=0.78` | `11,626`（原版）<br>`12,386`（86KD 变长热态） | 32 路：`13,976` → `14,105` | 32 路共享前缀：`1,051` → **`2,678`（`+154.8%`）** | 8 卡硬件/P2P/NVMe 全部达标；86KD 变长验证使 32 路 Decode 翻倍以上，但纯 TP8 全局 KV 仅 `5.19M` |
| **02. 86KD 32K/1K 专项对比**<br>[`02-xtier_86kd-32k1k-对比.md`](bench/reports/02-xtier_86kd-32k1k-对比.md) | 原版静态 vs 86KD 静态 vs 86KD 变长（1–32 并发） | `12,386`（首轮冷启动 `4,325`） | 16 路：`14,084`<br>32 路：`14,105` | 8 路：`1,589`（`+145.1%`）<br>32 路：`2,678` | 确立 `XTIER_86KD=1` + `compact` 变长验证 + 本机 SPS 表为基线 |
| **03. DP8 + Gatherv + RS 优化**<br>[`03-tp8-dp8-fp8kv-阶段报告.md`](bench/reports/03-tp8-dp8-fp8kv-阶段报告.md) | `attn-dp=8` + `DP LM head` + `CHUNK=32K` + `gatherv` + `RS` + `min-free-slots-delay=1` | `5,405`（DP8 单卡承载） | **64 路：`20,421–20,700`** | **64 路共享前缀：`4,036–4,074`**<br>64 路短提示：`2,919` | `gatherv` 将 64 路冷 P 由 `16.2K` 提升 `+26.4%` 至 `20.5K+`；全局 KV 扩至 `20.89M` |
| **04. DeepInfer 第一档单元与 U5 修复**<br>[`04-deepinfer-tier1-单元验证报告.md`](bench/reports/04-deepinfer-tier1-单元验证报告.md)<br>[`repair-fp32mid-r3-summary.json`](bench/reports/data/repair-fp32mid-r3-summary.json) | `A1/A2/A3/A7` 逐字节对齐 + 外置 `repair-a2-reader`（FP32 `mid_out`） | — | — | Reader CUDA Graph 加速 **`10.2%–18.0%`**（最大回退 `0.00%`） | 解决原生 FlashInfer Split-K 二次 BF16 截断误差：`8×64` RMS `0.002128 -> 0.001586`（`-25.4%`），`48×64` `0.002480 -> 0.001754`（`-29.3%`），全 6 形状通过 U5 |
| **05. FP4 KV + 缓存修复 + 3h 配对整模验证**<br>[`05-dev3h-配对开发验证报告-20261116.md`](bench/reports/05-dev3h-配对开发验证报告-20261116.md)<br>[`sd-dev3h-r1-analysis.json`](bench/reports/data/sd-dev3h-r1-analysis.json) | `mem=0.85`、`tails=384`（主 KV **`67.45M`**），`p0`（全关）vs `p1`（`A1/2/3/7` + `repair-a2-reader`） | `2,131`（`p0`）→ **`2,315`**（`p1`，冷首轮含 JIT；热态 `~8,350`） | **64 路 32K：`20,543`**<br>**64×128K：`20,660`** | **64 路 32K：`4,214`**（`p0: 4,359`）<br>**64×128K 纯 D：`3,567`（`+5.43%`）**<br>64 路短提示：`2,922` | `64×128K` 命中率 **`99.79%`**（双侧 14.4s+ 64 路共同窗口），最低空闲显存 **`4,858 MiB`（`4.744 GiB`）**，3 段 PPL 变化仅 **`+0.067%`**，全程 0 错 |
| **06. Step 1 + Step 2 通信压缩与 MXMMA `e3m4` MoE 提速**<br>[`06-tp8-step1-step2-优化报告.md`](bench/reports/06-tp8-step1-step2-优化报告.md)<br>[`sd-popt3b-analysis.json`](bench/reports/data/sd-popt3b-analysis.json) | 外置 `deploy/prefill-opt`：FP8 DP Gatherv + ReplayTail UAF 修复 + 零根跳过 + 目标/草稿 MoE `MXMMA float8e3m4 pad(128,128)` | **`8,632`**（`3.73×`；冷首轮 `2,810`） | **64 路 32K：`24,850`（`+21.0%`）**<br>**64×128K：`25,158`（`+21.8%`）** | **64 路 32K：`4,740.5`（`+12.5%`，窗口 `4,822`）**<br>**64×128K 纯 D：`3,671.6`**<br>64 路短提示：**`3,095`（`+5.9%`）** | 每卡省权重 **`4.98 GiB`**，全局主 KV 暴涨 **`+69.5%`** 至 **`1.143 亿`**（`114,348,032`）；最低空闲 **`4.432 GiB`**，3 段 PPL `2.0714`（`+0.107%`），混合 `2×300K` 0 错 |
| **07. Step 3 双引擎算子与自适应分块深度提速**<br>[`07-tp8-step3-双引擎深度优化报告.md`](bench/reports/07-tp8-step3-双引擎深度优化报告.md)<br>[`sd-popt4-analysis.json`](bench/reports/data/sd-popt4-analysis.json) | `wo_a` `MAX_T=56` 清零解码反量化 + 宽维 `b12x 1-64` + 尾层深度守卫激活单发 `16K` 续块 + `CHUNK=49152` | **`10,886`（`4.70×`，TTFT `3.01s`）** | **64 路 32K：`26,106`（`+27.1%`）**<br>**64×128K：`26,129`（`+26.5%`）** | **64 路 32K：`4,672`（窗口 `4,746`，TPOT `9.30ms`）**<br>**64×128K 纯 D：`3,706`–`3,800`**<br>64 路短提示：**`3,252`（`+11.3%`）** | Decode `_dequant_kernel` 归零（`wo_a` 耗时 `-49%`，短提示单步降至 **`48.3 ms`**）；混合 `2×300K` TTFT 缩至 **`78.4s`**，PPL `2.0666`–`2.0745` |
| **08. Step 4 全形状 MXFP8 最优表 + 草稿 MXMMA + 显存加固**<br>[`08-tp8-step4-mxfp8-全形状调优报告.md`](bench/reports/08-tp8-step4-mxfp8-全形状调优报告.md)<br>[`sd-popt5-analysis.json`](bench/reports/data/sd-popt5-analysis.json) | 打破 `K%128==0` 锁死 `b12x` 下发全形状最优 `(backend, tactic)` + 草稿 128 专家 `MXMMA e3m4` + `CHUNK=40960` + 修复 `sps_profile.py` DP8 标定 | **`10,580`（`4.57×`）** | **64 路 32K：`25,548`**<br>**64×128K：`25,756`** | **64 路短提示：`3,285.5`（`+12.4%`，窗口 `3,318.6`）**<br>**64 路 32K P50 TPOT：`9.245 ms`**<br>**64×128K 单步：`49.75 ms`** | 含 8 卡 CUPTI Profiler 全程最低空闲 **`2.135 GiB`（`2,186 MiB`）**，PPL **`2.0678`**（与 `p0` 完全一致），主 KV **`1.140 亿`** |

---

## 二、仓库目录结构

```text
xtier_86kd/
├── src/xtier_86kd/                  # 86KD TP8 独立加速插件源码（DSpark 变长验证、分页索引器、FP8 wo_a、MXFP8 tile4 等）
├── csrc/                            # 可选 NVMe Engram 行存原生扩展源码（1TiB 主机内存下默认驻留 RAM 无需启用）
├── patches/
│   ├── sm120-official-fp4-kv.patch  # SM120 官方混合 V4.1 FP4 主 KV + FP8 SWA 补丁
│   └── deepinfer-tier1/             # DeepInfer 第一档（A1/A2/A3/A7）分项 diff 与 sglang-align 完整补丁快照
├── deploy/
│   ├── tf500_console.py             # :8100 启动台后端（仅保留纯 TP8 模型卡片，支持一键切换 DeepInfer 对齐开/关）
│   ├── tf500_console_ui.py          # 启动台前端渲染模块
│   ├── tf500_api_proxy.py           # :8000 OpenAI 兼容鉴权网关（自动探活并转发 :30000 TP8 服务，脱敏计费统计）
│   ├── tf500_proxy_admin.py         # /proxy 密钥管理页与多租户配额管理（tf500_proxy_keys.py）
│   ├── tf500_benchmark.py           # /benchmark 在线并发与 Agent 测速后端及 UI（tf500_benchmark_ui.py）
│   ├── tf500_server_info.py         # /server 八卡实时功耗、频率、显存与系统遥测页
│   ├── start-tp8-fp4-card.sh        # 生产卡片对应的 TP8 SGLang 真实启动脚本
│   ├── start-console.sh             # :8100 控制台守护启动脚本
│   ├── start-api-proxy.sh           # :8000 API 网关启动脚本
│   └── repair-a2-reader/            # 外置零侵入 A2 Split-K FP32-mid CUDA 内核与 sitecustomize 自动挂载钩子
├── bench/
│   ├── bench_serving.py             # 多并发流式冷 Prefill / 共享前缀 Decode 两阶段测速工具
│   ├── agent_bench.py               # 多会话长上下文工具调用 Agent 基准
│   ├── sps_profile.py               # 本机 DSpark 变长验证成本标定工具与实测表 86kd-sps-6000d.json
│   ├── 32k1k-ab.md                  # 早期 86KD 静态/变长 32K/1K 对比与原始 JSON
│   └── reports/                     # 全阶段中文实测报告归档（01–05）与 data/ 核心 JSON 证据（含 SD10/SD13 硬件峰值）
└── tests/                           # 代理网关、密钥配额、测速面板与系统监控本地单元测试
```

---

## 三、安装、部署与日常使用

### 1. 安装 `xtier_86kd` 插件到环境

```bash
/data/xtier-env/bin/pip install -e /data/src/xtier_86kd --no-deps
```

启动 SGLang 时保持 `XTIER=0`（不接管混合专家）并设置 `XTIER_86KD=1` 启用本仓库 TP8 加速钩子；如需回退对比，设 `XTIER_86KD=0` 即可零侵入关闭全部钩子。

### 2. WebUI 启动台（`127.0.0.1:8100`）与 API 网关（`:8000`）

- **访问启动台**：在本地开发机执行 SSH 端口转发 `ssh -L 8100:127.0.0.1:8100 <user>@<TF500_IP>`，浏览器打开 `http://127.0.0.1:8100/`。
  - `/`：**启动台**（展示 8 卡实时状态、参数下拉选择与一键启停；默认 `mem=0.85, tails=384`，支持切换 `DeepInfer 第一档对齐：开 / 关`）；
  - `/benchmark`：**在线测速**（支持 1–64 并发流式测速与长上下文 Agent 测速）；
  - `/proxy`：**API 网关与子密钥管理**（支持创建带有效期和请求次数配额的客户端密钥，密钥密文存储于 `/data/xtier-86kd-console/`，绝不入库）；
  - `/server`：**服务器硬件监控**（每 5 秒刷新单卡功耗、频率、显存、温度与主机负载）。
- **OpenAI 兼容调用入口**：
  - Base URL：`http://<TF500_IP>:8000/v1`
  - Model：`DeepSeek-V4.1-Flash-1017`
  - API Key：在 `/proxy` 页面生成客户端密钥，或在服务器读取 `/data/xtier-86kd-console/proxy-api-key`（请勿将密钥写入任何仓库文件或公开日志）。

### 3. 运行本地单元测试

本地单元测试全部使用内存级模拟服务，不会触碰在线 `:30000` 推理进程或占用 GPU：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 四、许可证

Apache-2.0，详见 [`LICENSE`](LICENSE)。
