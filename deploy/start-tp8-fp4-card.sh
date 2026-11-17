#!/usr/bin/env bash
set -euo pipefail
unset ALIGN_READER_CPB ALIGN_READER_MIN_TOKENS SGLANG_ALIGN_FINISH_SWA_TRIM

VENV=${VENV:-/data/xtier-env}
CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
NCCL_LIB=$VENV/lib/python3.12/site-packages/nvidia/nccl/lib
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$NCCL_LIB:${LD_LIBRARY_PATH:-}"
export FLASHINFER_CUDA_ARCH_LIST=12.0
export TVM_FFI_GPU_BACKEND=cuda
export TVM_FFI_CUDA_ARCH_LIST=12.0
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=SYS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XTIER=0

MODEL=/data/models/DeepSeek-V4.1-Flash-1017
DRAFT=/data/models/ds41-draft
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
MAX_RUNNING=${MAX_RUNNING:-4}
USE_DSPARK=${USE_DSPARK:-1}
MOE_BACKEND=${MOE_BACKEND:-humming}
CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-32}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.78}
API_KEY_FILE=${API_KEY_FILE:-}

[ -f "$MODEL/model.safetensors.index.json" ] || { echo 'model index missing' >&2; exit 1; }
[ "$(cat /data/models/model-verification.exit 2>/dev/null)" = 0 ] || {
  echo 'model SHA256 verification incomplete' >&2; exit 1;
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "${XTIER_PREFILL_OPT:-0}" = 1 ]; then
  if [ -r "$SCRIPT_DIR/prefill-opt/sitecustomize.py" ]; then
    true
  elif [ -r /data/bench/align/prefill-opt-v1/sitecustomize.py ]; then
    true
  else
    echo 'Prefill/MoE opt hook unreadable in deploy/prefill-opt or /data/bench/align/prefill-opt-v1' >&2; exit 1;
  fi
fi
if [ "${SGLANG_DI_ALIGN_A2:-0}" = 1 ] && [ "${SGLANG_REPAIR_A2_FP32MID:-0}" = 1 ]; then
  if [ -r "$SCRIPT_DIR/repair-a2-reader/sitecustomize.py" ]; then
    true
  elif [ -r /data/bench/align/repair-a2-reader/sitecustomize.py ]; then
    true
  else
    echo 'A2 FP32-mid reader hook unreadable in deploy/repair-a2-reader or /data/bench/align/repair-a2-reader' >&2; exit 1;
  fi
fi

args=(
  --model-path "$MODEL" --served-model-name DeepSeek-V4.1-Flash-1017
  --tp-size 8 --trust-remote-code --host "$HOST" --port "$PORT"
  --context-length "$CONTEXT_LENGTH" --max-running-requests "$MAX_RUNNING"
  --chunked-prefill-size "${CHUNK:-2048}" --max-prefill-tokens "${CHUNK:-2048}"
  --prefill-decode-interval "${INTERVAL:-10}" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --disable-shared-experts-fusion
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
  --fp8-gemm-backend flashinfer_cutlass
  --moe-runner-backend "$MOE_BACKEND"
  --enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
  --enable-metrics --enable-cache-report
  --weight-loader-drop-cache-after-load --watchdog-timeout 1200
)

if [ "$USE_DSPARK" = 1 ]; then
  [ -f "$DRAFT/model.safetensors.index.json" ] || { echo 'draft index missing' >&2; exit 1; }
  args+=(
    --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT"
    --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
  )
  if [ "$SGLANG_RAGGED_VERIFY_MODE" = compact ]; then
    args+=(--speculative-dspark-align-verify-tokens-to-graph-tier)
    if [ -n "${SPS_TABLE:-}" ]; then
      [ -r "$SPS_TABLE" ] || { echo "SPS cost table unreadable: $SPS_TABLE" >&2; exit 1; }
      args+=(--speculative-dspark-sps-table-path "$SPS_TABLE")
    fi
  fi
fi
if [ -n "$API_KEY_FILE" ]; then
  [ -r "$API_KEY_FILE" ] || { echo 'API key file unreadable' >&2; exit 1; }
  args+=(--api-key "$(cat "$API_KEY_FILE")")
fi

args+=(${EXTRA_ARGS:-})
exec "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
