# Tier1 单元验收汇总（不代替整模报告）

结论：**all_passed=false**。U5 的“不劣于 off”严格门槛未通过；未修改 attention 或放宽容差。源码已冻结，GPU 已交还整模验收。机器可读入口：`/data/bench/align/unit-summary.json`；内含来源 SHA256。

| 门槛 | 证据与结果 |
|---|---|
| U1/U2 A1/A2 写缓存 | 六个真实生产入口，各约百万输入 token；DeepInfer oracle 全字节相同；17,024,482 个 FP4 精确中点；关闭后与 dev 的六路 SHA 全同 |
| U3 A3 索引 K/Q | 四路各 1,048,576 行，261 组逐字节检查通过；off/dev 四路 SHA 全同 |
| U4 A7 激活 FP8 | 36 形状、32,640 个 BF16 最大值枚举、18 GEMM；编码与尺度零差异；最大 GEMM relative RMS 0.0000604331；off/dev fused 全同 |
| U5 reader | 六形状绝对 RMS 均 <2.5%；8 行/64 头混合 KV 不满足不劣于 off，严格失败，A2-only 复现 |
| U6 Graph | 各团队真实生产入口更换输入重放通过；捕获 shape 固定，未宣称动态 shape |
| U7 A1/A2 | eager 与 graph 各18档，3轮×100观测；最大回退分别 +1.56775% / +0.00856%；首轮三个轻微超限结果保留 |
| U7 A3 | graph 12档、1轮×100观测均≤3%；没有 eager 或三轮证据 |
| U7 A7 | graph 29档，最大 +1.72836%；eager 曾达 +58.11169%，原摘要归因为 host launch gaps，不能据 graph 成绩宣布实际 eager 已过 |
| U8 | off 原 suite 与 on 对齐 oracle 各7通过、1 ROCm 跳过；生产调用/断言/容差不变 |

## U5 失败的原始统计与单位

相对 RMS 定义为 sqrt(sum((actual-reference)^2)/sum(reference^2))，统计全程 FP64。

| 8行×64头混合缓存，131072元素 | off | A2-only / A1+A2 |
|---|---:|---:|
| relative RMS 比例 | 0.0021278410241240683 | 0.002129337399466778 |
| relative RMS 百分数 | 0.21278410241240683% | 0.2129337399466778% |
| squared error sum | 0.836476606803744 | 0.837627430144650 |
| reference squared sum | 184746.170298003 | 184740.420067868 |

差值为 0.00014963753427097 个百分点。绝对限值是比例0.025，即2.5%；绝对门槛通过不抵消“不劣于 off”失败。

参考采用**各配置自身解量化缓存**计算 FP64 attention，不是相对同一原始 BF16 输入的总量化误差。A1-only 六例均等于 off；A2-only 与 A1+A2 一致。4096 行输出有限检查完整，但 FP64 比较采样4行，其他形状亦按脚本 sample 行验收。

缓存变化已逐字节分类：A1 SWA 的512个 payload bytes、32个 scales 全在旧下限114对应组内；A2 的128/256页分别125/252个 payload bytes变化，对应125/253个 FP4元素，全部精确中点、非中点变化0、尺度变化0。

## 证据入口

- `/data/bench/align/tests/A1A2-final-gates.json`、`A1A2-followup-summary.json`、`A1A2-summary.json`（保留首轮）。
- `/data/bench/align/tests/u5-u8-audit64-{off,a1,a2,on}_result.json`；同前缀 attention/cache `.pt` 保存输入、输出、缓存、FP64参考；`cache-diff-*.json` 保存逐字节分类。
- `/data/bench/align/tests/u7-{eager,graph}-{1,2,3}-{off,on}.json` 保留全部原始计时样本。
- `/data/bench/align/tests/a3_summary.json`、`a3_{dev,off,on}_result.json`、`/data/bench/align/REPORT-A3.md`。
- `/data/bench/align/unit-summary-A7.json`、`VERIFY-A7.md`。
- `/data/bench/align/logs/u8-existing-off.log`、`u8-existing-on-aligned-oracle.log`。
- `/data/bench/align/patches/A1A2.diff`、`A3.diff`；A1/A2详情 `A1A2-REPORT.md`，fast-math证据 `VERIFY-A1A2.md`。

限制：本机验证 SM120；ROCm 与 SM100 专用分支没有设备验收。单元内核耗时不能外推整模收益。整模 F1–F10、三轮真实负载与速度采集由协调者另行报告，本文件不覆盖 `REPORT-tier1.md`。
