"""Check XTier's MXFP8 backend selection for TP4 shared experts."""

import torch
from xtier.packs.deepseek_v4 import _mxfp8_linear

def selected(m, k):
    x = torch.empty((m, k))
    w = torch.empty((5120, k))
    return _mxfp8_linear(lambda _x, _w, **kw: kw["backend"], x, w, backend="cutlass")

assert selected(6, 576) == "cutlass"
assert selected(6, 512) == "b12x"
print("MXFP8 backend: K=576 cutlass; K=512 b12x")
