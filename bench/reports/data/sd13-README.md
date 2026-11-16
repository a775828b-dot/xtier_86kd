# SD13 卡间带宽与延迟实测

GPU0–3属于NUMA0，GPU4–7属于NUMA1；同侧NODE，跨侧SYS。512MiB，5次采样，保留nvbandwidth默认校验。

| 测试 | 单位 | 同CPU配对中位数 | 跨CPU配对中位数 |
|---|---|---:|---:|
| device_to_device_memcpy_write_ce | GB/s | 40.905 | 37.910 |
| device_to_device_memcpy_read_ce | GB/s | 40.842 | 37.999 |
| device_to_device_memcpy_write_sm | GB/s | 31.129 | 29.760 |
| device_to_device_memcpy_read_sm | GB/s | 42.055 | 41.497 |
| device_to_device_latency_sm | ns | 1951.080 | 1976.465 |

配对中位数来自24个同侧和32个跨侧有向配对；完整8×8矩阵、范围与SHA见原始JSON和socket-summary.json。
这些是独立卡对测量，不能代替TP8同时通信、CPU内存到4卡传输或STREAM带宽。未运行模型，不改变对齐验收失败结论。

原始数据目录：`sd-peer-0730/`。旧占位文件已备份。
