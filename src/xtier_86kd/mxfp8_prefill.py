"""Use XTier's larger FlashInfer MXFP8 tile for untuned SM120 prefill calls.

Only FlashInfer's six-tactic CUTLASS runner is changed. Explicitly tuned calls,
other runner types, small batches, and other GPU architectures keep their
original tactic. Ported from XTier's model-independent platform hook. SGLang
pins DeepSeek's block-FP8-as-MXFP8 linears, so their fallback is overridden
only for a TP8 expert shape checked bitwise on this host.
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)
AUTOTUNER = "flashinfer.autotuner.autotuner.AutoTuner.choose_one"
LARGE_TILE = 4  # 128 x 128 x 128, without A/B swap
LARGE_ROWS = 256
PINNED_SAFE_SHAPES = {(288, 5120)}  # (K, N), TP8 expert down projection
_logged = False


@lru_cache
def _sm120_cutlass_mxfp8(device_index, runner_type, tactic_count):
    import torch

    return (
        runner_type == "CutlassMxfp8GemmRunner"
        and tactic_count == 6
        and torch.cuda.get_device_capability(device_index)[0] == 12
    )


def _choose_one(original, self, custom_op, runners, tuning_config, inputs, *args, **kwargs):
    global _logged
    runner, tactic = original(self, custom_op, runners, tuning_config, inputs, *args, **kwargs)
    pinned = custom_op in getattr(self, "_effective_skip_ops", ())
    if (
        tactic == -1
        and custom_op == "mxfp8_gemm"
        and not self.is_tuning_mode
        and inputs
        and len(inputs) >= 2
        and inputs[0].dim() == 2
        and inputs[1].dim() == 2
        and inputs[0].shape[0] >= LARGE_ROWS
        and inputs[0].is_cuda
        and (not pinned or (inputs[0].shape[1], inputs[1].shape[1]) in PINNED_SAFE_SHAPES)
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

