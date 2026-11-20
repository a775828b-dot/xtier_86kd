"""Offline unit tests for P1: N3 decode-protected dynamic chunking & S1b NUMA pinning."""

import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

HOOK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "prefill-opt" / "prefill_opt_hook.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("_test_prefill_opt_hook", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeBatch:
    def __init__(self, empty: bool):
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class _FakeTensor:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        _, col = key
        return _FakeCol([r[col] for r in self._rows])


class _FakeCol:
    def __init__(self, vals):
        self._vals = vals

    def __gt__(self, other):
        return _FakeBoolCol([v > other for v in self._vals])

    def __eq__(self, other):
        return _FakeBoolCol([v == other for v in self._vals])


class _FakeBoolCol:
    def __init__(self, vals):
        self._vals = vals

    def __and__(self, other):
        return _FakeBoolCol([a and b for a, b in zip(self._vals, other._vals)])

    def sum(self):
        return _FakeScalar(sum(1 for v in self._vals if v))


class _FakeScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class TestP1SchedulerAndNuma(unittest.TestCase):
    def test_compute_numa_cpu_set_disjoint_and_reserved(self):
        hook = _load_hook()
        fake_nodes = {
            0: list(range(0, 32)) + list(range(64, 96)),
            1: list(range(32, 64)) + list(range(96, 128)),
        }
        seen = set()
        for gpu in range(8):
            cpus = hook.compute_numa_cpu_set(gpu, gpus_per_node=4, reserve_phys_cores=4, numa_nodes_cpus=fake_nodes)
            self.assertEqual(len(cpus), 14)  # 7 physical + 7 HT
            self.assertFalse(seen & cpus, f"Overlap on gpu {gpu}")
            seen |= cpus
            if gpu < 4:
                self.assertTrue(cpus.issubset(set(fake_nodes[0])))
                # Reserved physical cores 28..31 and HT 92..95 untouched
                self.assertTrue(cpus.isdisjoint(set(range(28, 32)) | set(range(92, 96))))
            else:
                self.assertTrue(cpus.issubset(set(fake_nodes[1])))
                self.assertTrue(cpus.isdisjoint(set(range(60, 64)) | set(range(124, 128))))

    def test_decode_protect_chunk_local_and_global_and_idle_boost(self):
        hook = _load_hook()

        class BaseAdder:
            def __init__(self, a0, a1, a2, running_batch, a4, max_prefill, chunk_sz, **kwargs):
                self.chunk_sz = chunk_sz

        class FakeScheduler:
            def __init__(self):
                self.chunked_req = object()

            def get_prefill_adder_class(self):
                return BaseAdder

        class FakeMLPSync:
            def __init__(self, rows, health=False):
                self._rows = rows
                self.any_health_check = health
                self.tp0_info_cpu = None

            def all_gather(self):
                self.tp0_info_cpu = _FakeTensor(self._rows)

        dp_attn_mod = types.SimpleNamespace(MLPSyncBatchInfo=FakeMLPSync)
        sched_mod = types.SimpleNamespace(
            Scheduler=FakeScheduler,
            get_device=lambda: types.SimpleNamespace(gpu_id=5),
        )

        with patch.dict(
            os.environ,
            {
                "XTIER_IDLE_SINGLE_CHUNK": "16384",
                "XTIER_DECODE_PROTECT_CHUNK": "2048",
                "XTIER_NUMA_PIN": "1",
            },
        ), patch.object(hook, "_apply_numa_pin_for_gpu", return_value={39}) as mock_pin:
            hook.install_onto_dp_attn_sync(dp_attn_mod)
            hook.install_onto_scheduler(sched_mod)

            sched = FakeScheduler()
            mock_pin.assert_called_once_with(5)
            adder_cls = sched.get_prefill_adder_class()

            # 1) Pure cold prefill across all ranks -> decode_ranks=0, chunk stays 6144
            sync_prefill = FakeMLPSync([[6144, 0, 0, 1], [6144, 0, 0, 1]])
            sync_prefill.all_gather()
            self.assertEqual(hook._LAST_GLOBAL_DECODE_RANKS, 0)
            a1 = adder_cls(None, None, None, _FakeBatch(empty=True), None, 49152, 6144, waiting_queue_len=1)
            self.assertEqual(a1.chunk_sz, 6144)

            # 2) Single active rank idle prefill -> boosts to 16384
            hook._LAST_GLOBAL_ACTIVE_RANKS = 1
            a2 = adder_cls(None, None, None, _FakeBatch(empty=True), None, 49152, 6144, waiting_queue_len=0)
            self.assertEqual(a2.chunk_sz, 16384)

            # 3) Peer rank is decoding (col0>0, col3==0), even when local running_batch is empty -> clamps to 2048!
            sync_decode = FakeMLPSync([[0, 0, 1, 0], [12, 0, 1, 0]])
            sync_decode.all_gather()
            self.assertEqual(hook._LAST_GLOBAL_DECODE_RANKS, 1)
            a3 = adder_cls(None, None, None, _FakeBatch(empty=True), None, 49152, 6144, waiting_queue_len=0)
            self.assertEqual(a3.chunk_sz, 2048)

            # 4) Local rank has running decode batch even if global tracker is 0 -> clamps to 2048!
            hook._LAST_GLOBAL_DECODE_RANKS = 0
            a4 = adder_cls(None, None, None, _FakeBatch(empty=False), None, 49152, 6144, waiting_queue_len=0)
            self.assertEqual(a4.chunk_sz, 2048)


if __name__ == "__main__":
    unittest.main()
