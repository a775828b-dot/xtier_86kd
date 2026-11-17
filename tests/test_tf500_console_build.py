import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))


def _import_tf500_console():
    fake_core = types.ModuleType("xtier.console.core")
    fake_core.Console = object
    fake_main = types.ModuleType("xtier.console.__main__")
    fake_main.Jobs = object
    fake_main.make_handler = lambda *a, **kw: object
    fake_pkg = types.ModuleType("xtier")
    fake_console_pkg = types.ModuleType("xtier.console")
    fake_console_pkg.core = fake_core
    with patch.dict(
        sys.modules,
        {
            "xtier": fake_pkg,
            "xtier.console": fake_console_pkg,
            "xtier.console.core": fake_core,
            "xtier.console.__main__": fake_main,
        },
    ):
        if "tf500_console" in sys.modules:
            del sys.modules["tf500_console"]
        return importlib.import_module("tf500_console")


class TestTf500ConsoleBuild(unittest.TestCase):
    def test_default_optimized_build_wires_popt_and_dp8_sps(self):
        tc = _import_tf500_console()
        defaults = {opt["key"]: opt["default"] for opt in tc.MODEL["options"]}
        self.assertEqual(defaults["popt"], "on")
        self.assertEqual(defaults["chunk"], "49152")
        self.assertEqual(defaults["kvpool"], "113569792")
        self.assertEqual(defaults["align"], "on")

        env, cmd = tc.build(defaults, 30000)
        self.assertEqual(cmd, ["/bin/bash", "/data/src/xtier_86kd/deploy/start-tp8-fp4-card.sh"])
        self.assertEqual(env["XTIER_PREFILL_OPT"], "1")
        self.assertEqual(env["SGLANG_ENABLE_DP_GATHER_FP8"], "1")
        self.assertEqual(env["XTIER_PREFILL_MOE_MXMMA"], "e3m4")
        self.assertEqual(env["XTIER_DRAFT_MOE_MXMMA"], "e3m4")
        self.assertEqual(env["XTIER_MOE_PAD_N"], "128")
        self.assertEqual(env["XTIER_86KD_MXFP8_TABLE"], "1")
        self.assertEqual(env["CHUNK"], "49152")
        self.assertEqual(env["SGLANG_RAGGED_VERIFY_MODE"], "compact")
        self.assertTrue(env["SPS_TABLE"].endswith("86kd-sps-6000d-dp8.json"))
        self.assertIn("prefill-opt", env["PYTHONPATH"])
        self.assertEqual(env["SGLANG_DI_ALIGN_A2"], "1")
        self.assertEqual(env["SGLANG_REPAIR_A2_FP32MID"], "1")
        self.assertIn("MXMMA·双引擎加速", tc.tags(defaults))

    def test_rollback_popt_off_restores_baseline_env(self):
        tc = _import_tf500_console()
        opts = {opt["key"]: opt["default"] for opt in tc.MODEL["options"]}
        opts["popt"] = "off"
        opts["chunk"] = "32768"
        env, _ = tc.build(opts, 30000)
        self.assertEqual(env["XTIER_PREFILL_OPT"], "0")
        self.assertEqual(env["XTIER_86KD_MXFP8_TABLE"], "0")
        self.assertEqual(env["XTIER_DS_WO_A_MAX_T"], "16")
        self.assertEqual(env["SPS_TABLE"], "/data/bench/86kd-sps-6000d.json")
        self.assertIn("repair-a2-reader", env["PYTHONPATH"])
        self.assertNotIn("MXMMA·双引擎加速", tc.tags(opts))


if __name__ == "__main__":
    unittest.main()
