#!/usr/bin/env bash
# tf500: one XTier hybrid instance on GPU 0-3 and CPU/NUMA 0.
# GPU 4-7 / NUMA 1 and port 30002 are reserved for a future second instance.
set -euo pipefail

VENV=${VENV:-/data/xtier-env}
CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
NCCL_LIB=$VENV/lib/python3.12/site-packages/nvidia/nccl/lib
# FlashInfer's JIT linker asks for libnvrtc.so; the cu13 wheel ships only .so.13.
if [[ ! -e "$CUDA_HOME/lib/libnvrtc.so" ]]; then
  test -f "$CUDA_HOME/lib/libnvrtc.so.13"
  ln -s libnvrtc.so.13 "$CUDA_HOME/lib/libnvrtc.so"
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$NCCL_LIB:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CUDA_ARCH_LIST=12.0 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export SGLANG_AUTO_NUMA_BIND=0
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
export NCCL_P2P_LEVEL=SYS

# TP4 ranks share CPU0. The XTier NUMA patch gives each rank its own 8 cores.
export XTIER=1 XTIER_TP_NODES='0|0|0|0'
unset XTIER_86KD SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE
export XTIER_SPLIT_NODES=4 XTIER_TP_CORE_GROUPS='0|1|2|3'
export XTIER_THREADS=12 XTIER_NODE_CORES=6
export XTIER_MAIN_CPUS='6-7,70|14-15,78|22-23,86|30-31,94'
export XTIER_PROBE_DISPATCH_CPU='71|79|87|95'
export XTIER_SPIN_COUNT=1048576 XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1
export XTIER_DOWN_GRAIN=24 XTIER_THP=0 XTIER_DRAFT=1

# Engram lives in host RAM. No XTIER_ENGRAM_ROWS=nvme on this host.
unset XTIER_ENGRAM_ROWS XTIER_ENGRAM_MODEL XTIER_ENGRAM_CACHE_GIB
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-35}
export XTIER_CACHE_EVERY=2 XTIER_CACHE_SWAPS=32 XTIER_CACHE_LOG=500
export XTIER_CACHE_ELASTIC=1 XTIER_CACHE_ELASTIC_MAX_SLOTS=384
export XTIER_CACHE_ELASTIC_RESERVE_GIB=2
export XTIER_PREFILL_FP4=1 XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_OVERLAP=1 XTIER_PREFILL_PREFETCH_CHUNKS=6
export XTIER_PREFILL_EXPERT_CHUNK=8 XTIER_PREFILL_ROUTE_ACCUM=1
export XTIER_CPU_PREFILL_BATCH=2048 XTIER_GPU_PREFILL_MIN=1024
export XTIER_HYBRID_PREFILL_MAX=1024 XTIER_SPLIT_PREFILL_MAX=6144
export XTIER_PREFILL_CPU_SHARE=1
export XTIER_DOWN_PRECISION=tf32 XTIER_PREFILL_TILE_M=64
export XTIER_GATE_BN=128 XTIER_GATE_LAUNCH=8,3
export XTIER_DOWN_BN=128 XTIER_DOWN_BK=64 XTIER_DOWN_LAUNCH=8,3

MODEL=/data/models/DeepSeek-V4.1-Flash-1017
DRAFT=/data/models/ds41-draft
API_KEY_FILE=${API_KEY_FILE:-/data/sglang-api-key}
HOST=${HOST:-<TF500_IP>}
PORT=${PORT:-30001}
CONTEXT=${CONTEXT:-524288}
KV_POOL=${KV_POOL:-20000000}
BATCH=${BATCH:-49152}
MAX_RUNNING=${MAX_RUNNING:-32}
INTERLEAVE=${INTERLEAVE:-72}
MEM=${MEM:-0.85}

test -r "$MODEL/model.safetensors.index.json"
test -r "$DRAFT/model.safetensors.index.json"
test "$(cat /data/models/model-verification.exit 2>/dev/null)" = 0
test -r "$API_KEY_FILE"

args=(
  --model-path "$MODEL" --served-model-name DeepSeek-V4.1-Flash-1017
  --host "$HOST" --port "$PORT" --tp-size 4 --trust-remote-code
  --engram-config '{"placement":"cpu","replication":"none","lookup":"auto"}'
  --context-length "$CONTEXT" --max-total-tokens "$KV_POOL"
  --max-running-requests "$MAX_RUNNING"
  --chunked-prefill-size "$BATCH" --max-prefill-tokens "$BATCH"
  --prefill-decode-interval "$INTERLEAVE" --mem-fraction-static "$MEM"
  --disable-shared-experts-fusion
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full
  --cuda-graph-max-bs-decode 32 --fp8-gemm-backend flashinfer_cutlass
  --moe-runner-backend triton
  --enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report
  --weight-loader-drop-cache-after-load
  --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT"
  --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
  --api-key "$(cat "$API_KEY_FILE")"
)

exec numactl --cpunodebind=0 --preferred=0 "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
