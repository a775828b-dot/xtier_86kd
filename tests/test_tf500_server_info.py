"""Pure checks for telemetry parsing and the four page navigation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import tf500_benchmark
import tf500_nav
import tf500_server_info


class ServerInfoTest(unittest.TestCase):
    def test_gpu_csv_and_missing_optional_field(self):
        sample = ("1, NVIDIA RTX PRO 6000D, 23, 10, 2048, 85651, 240.5, 600, 51, [N/A], 1200\n"
                  "0, NVIDIA RTX PRO 6000D, 0, 0, 1024, 85651, 90, 600, 39, [N/A], 300\n")
        gpus = tf500_server_info.parse_gpus(sample)
        self.assertEqual([gpu["index"] for gpu in gpus], [0, 1])
        self.assertEqual(gpus[1]["power_w"], 240.5)
        self.assertIsNone(gpus[1]["fan_pct"])

    def test_pages_are_separate_and_navigation_has_four_links(self):
        page = tf500_nav.decorate(tf500_benchmark.page_html(), "benchmark")
        self.assertEqual(page.count('aria-current="page"'), 1)
        for route in ('href="/"', 'href="/benchmark"', 'href="/proxy"', 'href="/server"'):
            self.assertIn(route, page)
        self.assertIn("xt-bench", page)
        self.assertNotIn("new-key-value", page)
        server = tf500_nav.decorate(tf500_server_info.page_html(), "server")
        self.assertIn("gpu-grid", server)
        self.assertNotIn("bench-form", server)


if __name__ == "__main__":
    unittest.main()
