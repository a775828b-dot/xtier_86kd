# tf500 XTier TP4 混合推理实例

本目录只管理推理服务。启动台卡片与 API 代理由用户维护；TP8 基准卡片和原 `sglang-ds41.service` 保留。

## 实例

| 项目 | 配置 |
| --- | --- |
| 服务 | `xtier-tp4.service`，开机启动，失败自动重启 |
| 地址 | `http://<TF500_IP>:30001/v1`，API key 仍从 `/data/sglang-api-key` 读取 |
| 硬件 | GPU 0–3，CPU/NUMA 0；GPU 4–7 留给后续第二实例 |
| 模型 | `/data/models/DeepSeek-V4.1-Flash-1017`，DSpark draft `/data/models/ds41-draft` |
| KV / 调度 | 20,000,000 token 上限；48K prefill chunk；两次 prefill 间插入最多 72 步 decode；最多 32 并发 |
| 内存 | Engram 位于主机内存；XTier 自适应热专家缓存，弹性部分保留 2 GiB/GPU |

启动脚本是 `start-xtier-tp4.sh`。`xtier-tp4-numa.patch` 修补 XTier 的单 NUMA TP4 放置逻辑；每个 TP rank 使用 CPU0 内互不重叠的 8 个物理核心组，其中 6 个物理核心及其超线程用于专家计算。`check-tp4-placement.py` 可单独核对 worker、dispatcher 与 scheduler CPU 绑定。

`xtier-capture-slot.patch` 让 CUDA Graph 捕获复用同形状 eager 槽；`xtier-capture-prewarm.patch` 在捕获前为每个非全缓存层预分配 1–16 token 的 mailbox 槽；否则 SGLang 首次捕获 32 路、每路 6 token 的验证图时会抛出 `Prepare this stream/shape before graph capture`。`xtier-trace-per-rank.patch` 为临时诊断生成独立的 CPU 路由记录文件。三个补丁应用于 `/data/src/xtier`，其中 C++ 变更已重编原生库。

TP4 的 FP4 专家中间维度为 576，不满足 FlashInfer MXFP4 CUTLASS 后端的 128 对齐要求，因此显式指定 `--moe-runner-backend triton`；XTier 的混合专家计算仍由其 CPU/GPU 路径负责。`xtier-fp4-prefill-k576.patch` 让该维度的冷 prefill 改走解包 BF16 的 GPU 分块重叠路径，避免 E8M0 scale 从 18 补到 20 后 reshape 失败。共享专家的 MXFP8 GEMM 也会出现 K=576，`xtier-mxfp8-k576.patch` 限制 b12x 仅用于 K 为 128 倍数的形状，其余保留 SGLang 的 CUTLASS 回退。`check-fp4-prefill-shape.py` 已验证形状选择，`check-mxfp8-backend.py` 已验证 K=576 选 CUTLASS、K=512 选 b12x，`check-mxfp8-k576-gemm.py` 已在 GPU0 实际执行 K=576 的 CUTLASS GEMM。

原 TP8 服务 `sglang-ds41.service` 已停止并取消开机自启，以防重启时与 TP4 争用 GPU。其 unit 和启动台卡片都保留，后续可在关闭 TP4 后手动启动，用于基准复测。

验收以实际 `/v1/models`、推理请求、KV 分配日志和 `nvidia-smi` 为准。已用 `single-tp4-profile.py` 完成顺序单发：32K 冷 prefill，然后同前缀 1K decode，并记录 CPU 路由调用和短时 CUDA 时间线；并发测速暂不执行。生产启动脚本不设置 `XTIER_ROUTE_TRACE`，需要复诊时可临时提供该环境变量，避免跟踪文件持续增长。

首次单发结果与 CPU/GPU 等待归因见 [`tf500-tp4-single.md`](tf500-tp4-single.md)。
