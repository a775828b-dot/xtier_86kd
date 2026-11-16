"""Unified zero-overhead MetaPathFinder sitecustomize for A2 FP32-mid reader + Prefill/MoE Opt."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent

for _p in list(sys.path):
    try:
        if Path(_p).resolve() == _THIS_DIR:
            continue
    except OSError:
        continue
    _cand = os.path.join(_p, "sitecustomize.py")
    if os.path.isfile(_cand):
        with open(_cand, "rb") as _f:
            exec(compile(_f.read(), _cand, "exec"), globals())
        break

_HOOK_MODULES = {}


def _resolve_hook_file(filename: str) -> Path:
    direct = _THIS_DIR / filename
    if direct.is_file():
        return direct
    sibling = _THIS_DIR.parent / "repair-a2-reader" / filename
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(f"Hook module {filename} not found in {_THIS_DIR} or {sibling.parent}")


def _load_local_module(mod_key: str, filename: str):
    if mod_key not in _HOOK_MODULES:
        spec = importlib.util.spec_from_file_location(mod_key, _resolve_hook_file(filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HOOK_MODULES[mod_key] = mod
    return _HOOK_MODULES[mod_key]


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, original, callback):
        self.original = original
        self.callback = callback

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        self.callback(module)


class _UnifiedFinder(importlib.abc.MetaPathFinder):
    _xtier_unified_finder = True

    def __init__(self):
        self.targets = {}
        if (
            os.environ.get("SGLANG_DI_ALIGN_A2", "0") == "1"
            and os.environ.get("SGLANG_REPAIR_A2_FP32MID", "0") == "1"
        ):
            self.targets["sglang.kernels.ops.attention.dsv41_sm120"] = (
                lambda m: _load_local_module("_a2_fp32mid_isolated", "a2_fp32mid_reader.py").install_onto(m)
            )
        if os.environ.get("XTIER_PREFILL_OPT", "0") == "1":
            self.targets["sglang.srt.distributed.device_communicators.pynccl"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_pynccl(m)
            )
            self.targets["sglang.srt.layers.dp_attention"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_dp_attention(m)
            )
            self.targets["humming.layer"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_humming_layer(m)
            )
            self.targets["sglang.srt.layers.quantization.humming_utils"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_humming_utils(m)
            )
            self.targets["sglang.srt.managers.scheduler"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_scheduler(m)
            )
            self.targets["sglang.srt.models.deepseek_v4_replay_tail"] = (
                lambda m: _load_local_module("_prefill_opt_isolated", "prefill_opt_hook.py").install_onto_replay_tail(m)
            )

    def find_spec(self, fullname, path=None, target_module=None):
        cb = self.targets.get(fullname)
        if cb is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None and not isinstance(spec.loader, _PatchingLoader):
            spec.loader = _PatchingLoader(spec.loader, cb)
        return spec


if not any(getattr(f, "_xtier_unified_finder", False) for f in sys.meta_path):
    _finder = _UnifiedFinder()
    if _finder.targets:
        sys.meta_path.insert(0, _finder)
