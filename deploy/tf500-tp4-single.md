# tf500 XTier TP4 单发验收与验证步归因

2026-11-15，北京时间。DeepSeek-V4.1-Flash-1017，GPU 0–3 + CPU/NUMA 0，DSpark draft 6 token，静态变长验证。只运行单个请求；32 并发仅为服务配置上限，未测速。启动脚本见 `start-xtier-tp4.sh`。

## 运行态

| 项目 | 实测 |
| --- | ---: |
| 服务 | `xtier-tp4.service`，30001，短请求正常，0 次自动重启（测试时） |
| KV 池 | 日志 `full=20000000` token；`swa=185088` |
| GPU | 0–3 工作，4–7 空闲 |
| 显存余量 | 0.5 秒采样最低 2142 MiB（GPU 1、2）；超过 2 GiB 约 94 MiB |
| Engram | RAM，`replication=none`，每张表约 24,170 MiB |
| 热专家缓存 | 40 个目标层自适应，另 3 个 draft 层全缓存；每工作卡初始约 7.7 GiB，弹性预留 2 GiB |
| 主机内存 | 本轮 systemd MemoryCurrent 约 690 GiB，峰值约 778 GiB，swap 约 4 GiB（测试后观察） |

## 顺序单发

同一 32,777 token 语料，冷请求输出 1 token；随后同前缀输出 1,024 token，KV 命中 32,512 token。两轮间自适应热专家缓存可能已变热；第一轮还开启了前 24 步的 PyTorch CPU/GPU profiler，因此两轮不能当作严格 A/B。

| 指标 | 首轮：有 profiler | 次轮：无 profiler |
| --- | ---: | ---: |
| 冷 prefill 首字 | 21.507 秒 | 14.094 秒 |
| 冷 prefill 估算输入速度 | 1,524 token/s | 2,326 token/s |
| 缓存后首字 | 3.650 秒 | 2.611 秒 |
| 1K 解码耗时 | 15.766 秒 | 8.138 秒 |
| 解码速度 | 64.89 token/s | 125.71 token/s |
| 验证步数 | 290 | 308 |
| 接受 token/步 | 3.53 | 3.32 |
| 解码平均 ms/步 | 54.37 | 26.42 |

无 profiler 轮的 XTier TP0 路由记录含 308 个完整验证步。CPU 专家计算每步中位 11.343 ms、每层有效调用中位 0.340 ms，gate 与 down 每步中位分别 7.380 / 4.839 ms。层间 CPU 无专家工作空档中位 9.354 ms；相邻验证步起点间隔中位 22.739 ms、均值 26.502 ms、P90 34.865 ms。TP1–3 结果接近。

首轮 GPU 时间线中，热专家 `_gate_up` 与 `_down` 内核中位分别 0.0499 / 0.0123 ms，远短于无 profiler 轮 0.340 ms 的 CPU 每层调用；`_unpack` 在 CPU 回调完成后执行。Profiler 轮的 GPU 在 CPU 回调末尾空闲中位约 1.97 ms，但 profiler 放大 CPU 回调，**只用于确认等待方向，不作为生产等待时长**。结论：每层 GPU 热专家先完成，GPU mailbox 主要等待 CPU 专家结果；CPU 在层间也等待 GPU 前向/通信与调度，单发总体两侧均有可见耗时。

原始数据在服务器 `/data/bench/xtier-tp4-single*.json`、`/data/bench/xtier-tp4-single*.gpu.csv`、`/data/bench/xtier-tp4-single-profile/`。`analyze-tp4-trace.py` 读取临时路由记录。生产启动脚本已取消默认 `XTIER_ROUTE_TRACE`，避免跟踪文件持续增长。以上显存余量为采样值，不能证明采样间隔内的瞬时最低值。

## 生产重启复核

去掉临时路由跟踪后，服务重新加载成功，`XTIER_ROUTE_TRACE` 未设置。`/v1/models` 返回正确模型；鉴权短请求 0.443 秒返回“测试成功”。`xtier-tp4.service` 运行中、0 次重启；日志再次确认 KV `full=20000000`。GPU 4–7 空闲，原 TP8 服务停用但保留。此处只复核可用性，不把短请求当作 32K/1K 性能结果。
