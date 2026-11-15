"""Check TP4's unsupported K=576 direct FP4 prefill falls back safely."""

from types import SimpleNamespace

import torch
from xtier.backend.prefill import Prefill

def eligible(inter):
    plan = SimpleNamespace(
        inter=inter,
        mxfp4=True,
        experts=384,
        chunk=8,
        weights=[torch.ones(1), torch.ones(1)],
    )
    return Prefill._fp4_ready(plan, torch.bfloat16)

assert not eligible(576)
assert eligible(512)
print("FP4 direct prefill: I=576 fallback; I=512 eligible")
