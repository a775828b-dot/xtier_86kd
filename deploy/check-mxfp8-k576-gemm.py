"""Execute the TP4 MXFP8 shared-expert shape with FlashInfer CUTLASS."""

import torch
from sglang.srt.layers.quantization.fp8_utils import (
    flashinfer_mxfp8_blockscaled_linear,
    flashinfer_mxfp8_quantize,
)

x = torch.randn((6, 576), device="cuda", dtype=torch.bfloat16)
w = torch.randn((5120, 576), device="cuda", dtype=torch.bfloat16)
qw, sw = flashinfer_mxfp8_quantize(w, is_sf_swizzled_layout=True, alignment=32)
y = flashinfer_mxfp8_blockscaled_linear(x, qw, sw, backend="cutlass", pin_tactic=True)
torch.cuda.synchronize()
assert y.shape == (6, 5120) and torch.isfinite(y).all()
print("MXFP8 CUTLASS K=576 GEMM passed")
