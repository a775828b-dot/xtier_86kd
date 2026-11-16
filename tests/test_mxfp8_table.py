import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtier_86kd import mxfp8_prefill as mp  # noqa: E402


class TestSm120Mxfp8Table(unittest.TestCase):
    def test_unaligned_k288_never_selects_b12x(self):
        for m in (1, 5, 6, 8, 40, 48, 320, 384, 5120, 16384, 40960):
            plan = mp.select_sm120_mxfp8_plan(m, 288, 5120)
            self.assertIsNotNone(plan)
            self.assertEqual(plan[0], "cutlass")
        self.assertEqual(mp.select_sm120_mxfp8_plan(6, 288, 5120), ("cutlass", 1))
        self.assertEqual(mp.select_sm120_mxfp8_plan(48, 288, 5120), ("cutlass", 1))
        self.assertEqual(mp.select_sm120_mxfp8_plan(384, 288, 5120), ("cutlass", 4))
        self.assertEqual(mp.select_sm120_mxfp8_plan(5120, 288, 5120), ("cutlass", 3))
        self.assertEqual(mp.select_sm120_mxfp8_plan(16384, 288, 5120), ("cutlass", 2))
        self.assertEqual(mp.select_sm120_mxfp8_plan(40960, 288, 5120), ("cutlass", 4))

    def test_prefill_and_decode_winners_for_key_shapes(self):
        # Fused wqkv_a: b12x in decode, cutlass tactic 5 in prefill
        self.assertEqual(mp.select_sm120_mxfp8_plan(48, 5120, 1792), ("b12x", None))
        self.assertEqual(mp.select_sm120_mxfp8_plan(5120, 5120, 1792), ("cutlass", 5))
        self.assertEqual(mp.select_sm120_mxfp8_plan(16384, 5120, 1792), ("cutlass", 5))

        # Shared experts gate_up: cutlass wins at conc=64 decode M=320/384 and prefill
        self.assertEqual(mp.select_sm120_mxfp8_plan(6, 5120, 576), ("b12x", None))
        self.assertEqual(mp.select_sm120_mxfp8_plan(320, 5120, 576), ("cutlass", -1))
        self.assertEqual(mp.select_sm120_mxfp8_plan(384, 5120, 576), ("cutlass", -1))
        self.assertEqual(mp.select_sm120_mxfp8_plan(5120, 5120, 576), ("cutlass", 3))
        self.assertEqual(mp.select_sm120_mxfp8_plan(16384, 5120, 576), ("cutlass", -1))
        self.assertEqual(mp.select_sm120_mxfp8_plan(40960, 5120, 576), ("cutlass", 2))

        # Engram wkv keeps b12x at M=40960 to avoid CUTLASS large-M slowdown
        self.assertEqual(mp.select_sm120_mxfp8_plan(16384, 6144, 25600), ("cutlass", 5))
        self.assertEqual(mp.select_sm120_mxfp8_plan(40960, 6144, 25600), ("b12x", None))

        # Unmeasured shapes return None
        self.assertIsNone(mp.select_sm120_mxfp8_plan(64, 1024, 1024))

    def test_forced_cutlass_tactic_scoping(self):
        self.assertIsNone(getattr(mp._tls, "tactic", None))
        with mp.forced_cutlass_tactic(5):
            self.assertEqual(mp._tls.tactic, 5)
            with mp.forced_cutlass_tactic(1):
                self.assertEqual(mp._tls.tactic, 1)
            self.assertEqual(mp._tls.tactic, 5)
        self.assertIsNone(getattr(mp._tls, "tactic", None))


if __name__ == "__main__":
    unittest.main()
