"""External PYTHONPATH hook installing isolated A2 FP32-mid SM120 reader."""
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

_TARGET = "sglang.kernels.ops.attention.dsv41_sm120"
_READER_FILE = Path(__file__).resolve().parent / "a2_fp32mid_reader.py"
_reader_mod = None


def _load_reader():
    global _reader_mod
    if _reader_mod is None:
        spec = importlib.util.spec_from_file_location("_a2_fp32mid_reader_isolated", _READER_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _reader_mod = mod
    return _reader_mod


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        _load_reader().install_onto(module)


class _Finder(importlib.abc.MetaPathFinder):
    _a2_fp32mid = True

    def find_spec(self, fullname, path=None, target_module=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None and not isinstance(spec.loader, _Loader):
            spec.loader = _Loader(spec.loader)
        return spec


if not any(getattr(f, "_a2_fp32mid", False) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
