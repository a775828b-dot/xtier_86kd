"""SM120 FlashInfer MXFP8 per-shape (backend, tactic) selector and prefill tile hook.

SGLang's ``_b12x_or_cutlass_mxfp8_linear`` unconditionally passes ``backend="b12x"``
whenever ``K % 128 == 0``, preventing FlashInfer's 6-tactic ``CutlassMxfp8GemmRunner``
from running on prefill or medium-batch decode shapes where CUTLASS TMA tiles are up to
2.53x faster, while ``K = 288`` (``shared_experts.w2``) defaults to CUTLASS ``tactic=-1``
at ``M < 256`` instead of ``tactic=1`` (3.15x faster in CUDA Graph).

Every ``b12x`` and CUTLASS tactic ``-1, 0..5`` is 100.0% bitwise identical on SM120
(``rel_rms == 0.0``). ``select_sm120_mxfp8_plan`` encodes the measured CUDA-Graph and
stream-timed winners on RTX PRO 6000D across all DeepSeek-V4.1-Flash-1017 TP8/DP8 shapes.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import logging
import os
import threading

logger = logging.getLogger(__name__)
AUTOTUNER = "flashinfer.autotuner.autotuner.AutoTuner.choose_one"
LARGE_TILE = 4  # 128 x 128 x 128, without A/B swap
LARGE_ROWS = 256

PINNED_SAFE_SHAPES = {
    (288, 5120),
    (1280, 32768),
    (5120, 1280),
    (6144, 25600),
    (4096, 1024),
}
PINNED_SAFE_SHAPES_LARGE_M = {
    (8192, 5120): 4096,
    (5120, 512): 6144,
}

_tls = threading.local()
_logged = False


@contextmanager
def forced_cutlass_tactic(tactic: int | None):
    prev = getattr(_tls, "tactic", None)
    _tls.tactic = tactic
    try:
        yield
    finally:
        _tls.tactic = prev


def select_sm120_mxfp8_plan(m: int, k: int, n: int) -> tuple[str, int | None] | None:
    """Return ``(backend, cutlass_tactic)`` for measured DSv4.1 TP8/DP8 shapes on SM120.

    Returns ``None`` for unmeasured shapes so callers fall back to generic heuristics.
    """
    if m <= 0:
        return None
    kn = (k, n)

    if kn == (288, 5120):  # shared_experts.w2 (K % 128 != 0 -> cutlass only)
        if m <= 64:
            return ("cutlass", 1)  # 3.51-3.71 us vs 9.45-11.06 us (2.55x-3.15x)
        if m <= 1024:
            return ("cutlass", 4)  # 6.77 us vs 10.61-11.74 us
        if m <= 8192:
            return ("cutlass", 3)  # 58.48 us vs 70.88 us
        if m <= 24576:
            return ("cutlass", 2)  # 180.16 us vs 214.91 us
        return ("cutlass", 4)      # 495.84 us at M=40960

    if kn == (5120, 1792):  # target attn.wqkv_a (40 layers)
        if m <= 1024:
            return ("b12x", None)
        return ("cutlass", 5)      # 258.0 us vs 431.9 us at M=5120 (1.67x); 1065 vs 1107 at 16K

    if kn == (5120, 1280):  # draft/unfused attn.wq_a
        if m <= 64:
            return ("b12x", None)
        if m <= 1024:
            return ("cutlass", 1)  # 21.83 us vs 37.50 us (1.72x)
        return ("cutlass", 5)      # 223.4 us vs 264.8 us at M=5120 (1.19x)

    if kn == (5120, 512):  # draft/unfused attn.wkv
        if m <= 42:
            return ("cutlass", 1)  # 17.49-21.91 us vs 23.59-24.80 us
        if m <= 64:
            return ("b12x", None)  # 12.48 us at M=48
        if m <= 1024:
            return ("cutlass", -1)  # 23.70 us vs 44.92 us (1.90x)
        if m <= 8192:
            return ("cutlass", 3)  # 94.73 us vs 232.04 us (2.45x)
        return ("b12x", None)      # 218.4 us at M=16384

    if kn == (1280, 32768):  # attn.wq_b (43 layers)
        if m <= 24:
            return ("b12x", None)  # 12.3-14.4 us
        if m <= 64:
            return ("cutlass", 3)  # 24.00-24.28 us vs 25.27-28.60 us (conc=64 DP8)
        if m <= 1024:
            return ("cutlass", 4)  # 118.7 us vs 122.5 us
        if m <= 8192:
            return ("cutlass", 3)  # 1490.3 us vs 1565.1 us
        return ("cutlass", 2)      # 4807.4 us vs 4977.1 us at M=16384

    if kn == (8192, 5120):  # attn.wo_b (43 layers)
        return ("b12x", None)

    if kn == (1280, 4096):  # indexer.wq_b (8 layers)
        if 128 <= m <= 1024:
            return ("cutlass", 1)  # 18.69 us vs 21.27 us
        return ("b12x", None)

    if kn == (5120, 576):  # shared_experts.w13 (43 layers; global M=320/384 in conc=64 decode)
        if m <= 64:
            return ("b12x", None)
        if m <= 1024:
            return ("cutlass", -1)  # 15.36-24.26 us vs 33.40-38.89 us (1.4x-2.53x)
        if m <= 8192:
            return ("cutlass", 3)  # 92.36 us vs 133.86 us (1.45x)
        if m <= 24576:
            return ("cutlass", -1)  # 361.03 us vs 451.08 us at M=16384 (1.25x)
        return ("cutlass", 2)      # 886.03 us vs 1003.17 us at M=40960 (1.13x)

    if kn == (6144, 25600):  # engram.wkv (2 layers)
        if m <= 128 or m > 24576:
            return ("b12x", None)  # b12x wins at small M and M=40960 (45.3 ms)
        if m <= 1024:
            return ("cutlass", 4)
        if m <= 8192:
            return ("cutlass", 3)  # 5684 us vs 5797 us
        return ("cutlass", 5)      # 17962 us vs 18231 us at M=16384

    if kn == (15360, 5120):  # mtp.main_proj (DSpark draft)
        if m <= 128:
            return ("b12x", None)
        if m <= 1024:
            return ("cutlass", 0)
        return ("cutlass", 2)      # 2927 us vs 2969 us; 9170 us vs 9298 us

    return None


@lru_cache
def _sm120_cutlass_mxfp8(device_index, runner_type, tactic_count):
    import torch

    return (
        runner_type == "CutlassMxfp8GemmRunner"
        and tactic_count == 6
        and torch.cuda.get_device_capability(device_index)[0] == 12
    )


def _pinned_shape_allowed(m: int, k: int, n: int) -> bool:
    kn = (k, n)
    if kn in PINNED_SAFE_SHAPES:
        return True
    min_m = PINNED_SAFE_SHAPES_LARGE_M.get(kn)
    return min_m is not None and m >= min_m


def _choose_one(original, self, custom_op, runners, tuning_config, inputs, *args, **kwargs):
    global _logged
    runner, tactic = original(self, custom_op, runners, tuning_config, inputs, *args, **kwargs)
    if (
        custom_op != "mxfp8_gemm"
        or self.is_tuning_mode
        or not inputs
        or len(inputs) < 2
        or inputs[0].dim() != 2
        or not inputs[0].is_cuda
    ):
        return runner, tactic

    forced = getattr(_tls, "tactic", None)
    if forced is not None and _sm120_cutlass_mxfp8(
        inputs[0].device.index,
        type(runner).__name__,
        len(runner.get_valid_tactics(inputs, None)),
    ):
        if not _logged:
            logger.info(
                "xtier_86kd: SM120 MXFP8 table tactic %d active at M=%d K=%d",
                forced, inputs[0].shape[0], inputs[0].shape[1],
            )
            _logged = True
        return runner, forced

    pinned = custom_op in getattr(self, "_effective_skip_ops", ())
    if (
        tactic == -1
        and inputs[1].dim() == 2
        and inputs[0].shape[0] >= LARGE_ROWS
        and (not pinned or _pinned_shape_allowed(inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[1]))
        and _sm120_cutlass_mxfp8(
            inputs[0].device.index,
            type(runner).__name__,
            len(runner.get_valid_tactics(inputs, None)),
        )
    ):
        if not _logged:
            logger.info(
                "xtier_86kd: SM120 MXFP8 prefill tile %d active at M=%d K=%d N=%d pinned=%s",
                LARGE_TILE, inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[1], pinned,
            )
            _logged = True
        return runner, LARGE_TILE
    return runner, tactic


def install(registry, hook_type):
    if os.environ.get("XTIER_86KD_MXFP8_PREFILL_TILE", "1") != "0":
        registry.register(AUTOTUNER, _choose_one, hook_type.AROUND)
