from __future__ import annotations

import re
import unittest

from boaresearch.init import render_banner


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class InitBannerTests(unittest.TestCase):
    def test_unicode_banner_wide_snapshot(self) -> None:
        banner = render_banner(width=160, height=24, allow_unicode=True)
        plain = ANSI_ESCAPE_RE.sub("", banner)
        self.assertIn("██████    ██████    █████", plain)
        self.assertIn("██████   ███████  ███████  ███████", plain)
        self.assertIn("⚗️  Bayesian Optimized Agents  ⚗️", plain)
        self.assertIn("🐍  code patch → descriptor → trial → promotion  🐍", plain)
        self.assertIn("\033[38;5;", banner)
        lines = plain.splitlines()
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
