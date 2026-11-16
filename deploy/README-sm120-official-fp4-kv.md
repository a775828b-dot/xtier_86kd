# SM120 官方 V4.1 FP4 KV 适配

状态：开发验证中，未切换正式服务。当前通过底层读写/注意力/CUDA Graph测试及TP8整模冒烟、首轮64并发32K/1K；完整长上下文与精度验收待完成。

## 实现

`patches/sm120-official-fp4-kv.patch` 适用于本机现有SGLang源码快照，目标为独立 `/data/src/sglang-dev`。
共享 `/data/src/sglang-xtier` 和 `/data/xtier-env` 不修改。

- SWA窗口：`V41_NEW`，528字节/条，512个FP8 E4M3值与16个UE8M0尺度。
- C1/C2压缩主KV：`V41_FP4_NEW`，288字节/条，512个FP4 E2M1值与32个E4M3尺度。
- 索引K：继续使用已有FP4索引器。
- 注意力计算：已安装FlashInfer公开 `SparseMLASm120Wrapper`，`compute_precision="bf16"`，在内核中读取压缩缓存；没有另建完整FP8/BF16 KV副本。

本机依赖快照：FlashInfer 0.7.4、PyTorch 2.14.1+cu130、Humming 0.1.18、Triton 3.8.0。
版本与FlashInfer SM120 Python源码SHA256记录在`/data/bench/dev/fp4-dependencies.json`；不能只凭同名接口假定其他版本兼容。

官方定义：[DeepInfer cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/cache_common.py)。
主KV默认FP4配置：[DeepInfer config.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/hugging_face/config.py)。

## 开发启动环境

```bash
export PYTHONPATH=/data/src/sglang-dev/python
export SGLANG_SM120_DSV41_KV=1
export SGLANG_DSV4_KV_LAYOUT=v41_new
export SGLANG_DSV4_COMPRESSED_KV_LAYOUT=auto
```

保持原CLI `kv_cache_dtype=fp8_e4m3`；真实混合布局由上面的选项控制，不能仅凭该CLI标签判定压缩主KV位宽。
auto对本模型C1/C2使用FP4。新增适配入口只允许SM120、v41_new及FlashMLA后端组合，并检查FlashInfer API能力。
完整tf500开发启动脚本：`/data/bench/dev/run-fp4.sh`；仍绑定127.0.0.1:30010，不自动部署正式服务。
报告二按用户指定显式传入`MEM_FRACTION_STATIC=0.90`、64并发；泛用启动器的默认显存比例仍为0.78，复现时应使用报告内完整环境。

## 测试与报告

- `/data/bench/dev/test_sm120_fp4_kv.py`：独立FlashInfer写入逐字节校验，参考注意力、分页、无效索引、变长、空行与图捕获。
- 首次默认精度路径在含大值样例出现5.315%相对RMS误差，保留失败记录；显式BF16路径已测样例相对RMS误差最大0.32536%，通过2.5%门槛。这是内核结果，不代替PPL。
- `/data/bench/dev/report-v1-fp8/REPORT.md`：旧版封存报告，包含未测项。
- `/data/bench/dev/fp4-kv-source/manifest.json`：补丁前后文件SHA256。每次新实验另存TAG.source.json/.patch。
- 报告二将在完整复测后生成；新旧配置同时改变了KV存储和注意力计算路径，性能变化不可仅归因于位宽。

## 回退

停止指定开发实例后，原正式源码和原启动配置仍可使用。不得在运行中的进程下热切换KV布局；需要重启并重新建立缓存与CUDA Graph。
