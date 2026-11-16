from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sps_profile", ROOT / "bench" / "sps_profile.py")
sps_profile = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sps_profile)


@dataclass(frozen=True)
class _DummyCtx:
    dp_size: int
    skip_max_running_requests_threshold: float


class TestSpsProfileAttnDp(unittest.TestCase):
    def test_fetch_context_upgrades_dp_size_from_internal_states(self):
        fake_profiler = types.SimpleNamespace(
            DEFAULT_TIMEOUT=10,
            RANDOM_TOKEN_LOW=1000,
            RANDOM_TOKEN_HIGH_MARGIN=1000,
            fetch_server_context=lambda **kw: _DummyCtx(dp_size=1, skip_max_running_requests_threshold=8.0),
            start_load=lambda **kw: None,
        )
        sps_profile._install_attn_dp_patches(fake_profiler)

        fake_info = {
            "internal_states": [
                {"effective_max_running_requests_per_dp": 8} for _ in range(8)
            ]
        }
        with patch.object(sps_profile, "_get_json", return_value=fake_info):
            ctx = fake_profiler.fetch_server_context(base_url="http://127.0.0.1:30010")
        self.assertEqual(ctx.dp_size, 8)
        self.assertEqual(ctx.skip_max_running_requests_threshold, 64.0)
        self.assertEqual(fake_profiler._xtier_detected_dp_size, 8)

    def test_start_load_routes_evenly_across_dp_ranks(self):
        calls = []
        fake_profiler = types.SimpleNamespace(
            DEFAULT_TIMEOUT=10,
            RANDOM_TOKEN_LOW=1000,
            RANDOM_TOKEN_HIGH_MARGIN=1000,
            fetch_server_context=lambda **kw: _DummyCtx(dp_size=8, skip_max_running_requests_threshold=64.0),
            start_load=lambda **kw: None,
            _xtier_detected_dp_size=8,
        )
        sps_profile._install_attn_dp_patches(fake_profiler)

        def _fake_post(url, payload, headers, timeout):
            calls.append((url, len(payload["input_ids"]), headers["x-override-routed-dp-rank"]))

        with patch.object(sps_profile, "_post_json", side_effect=_fake_post):
            t = fake_profiler.start_load(
                base_url="http://127.0.0.1:30010",
                num_requests=24,
                input_len=16,
                max_new_tokens=32,
                temperature=1.0,
                vocab_size=10000,
                rng=random.Random(0),
            )
            t.join(timeout=5)

        self.assertEqual(len(calls), 8)
        self.assertEqual(sorted(int(c[2]) for c in calls), list(range(8)))
        self.assertTrue(all(c[1] == 3 for c in calls))


if __name__ == "__main__":
    unittest.main()
