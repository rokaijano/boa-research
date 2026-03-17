from __future__ import annotations

import unittest

from boaresearch.init_banner import render_banner


class InitBannerTests(unittest.TestCase):
    def test_unicode_banner_wide_snapshot(self) -> None:
        banner = render_banner(width=160, height=24, allow_unicode=True)
        self.assertIn("██████╗  ██████╗  █████╗", banner)
        self.assertIn("⚗️  Bayesian Optimized Agents  ⚗️", banner)
        self.assertIn("🐍  code patch → descriptor → trial → promotion  🐍", banner)
        lines = banner.splitlines()
        self.assertTrue(lines[0].startswith(" "))

    def test_banner_collapses_when_narrow(self) -> None:
        banner = render_banner(width=40, height=24, allow_unicode=True)
        self.assertEqual(banner, "BOA Research | boa init")

    def test_ascii_banner_snapshot(self) -> None:
        banner = render_banner(width=100, height=24, allow_unicode=False)
        self.assertIn("BOA RESEARCH", banner)
        self.assertIn("code patch -> descriptor -> trial -> promotion", banner)


if __name__ == "__main__":
    unittest.main()
