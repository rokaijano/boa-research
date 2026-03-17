from __future__ import annotations


UNICODE_BANNER = [
    "██████╗  ██████╗  █████╗     ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗███████╗██████╗ ",
    "██╔══██╗██╔═══██╗██╔══██╗    ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗",
    "██████╔╝██║   ██║███████║    ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║█████╗  ██████╔╝",
    "██╔══██╗██║   ██║██╔══██║    ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║██╔══╝  ██╔══██╗",
    "██████╔╝╚██████╔╝██║  ██║    ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║███████╗██║  ██║",
    "╚═════╝  ╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝",
    "",
    "⚗️  Bayesian Optimized Agents  ⚗️",
    "🐍  code patch → descriptor → trial → promotion  🐍",
]

ASCII_BANNER = [
    "BOA RESEARCHER",
    "",
    "Bayesian Optimized Agents",
    "code patch -> descriptor -> trial -> promotion",
]


def _center_lines(lines: list[str], width: int) -> list[str]:
    centered: list[str] = []
    for line in lines:
        if not line.strip():
            centered.append("")
            continue
        padding = max(0, (width - len(line)) // 2)
        centered.append(" " * padding + line)
    return centered


def render_banner(*, width: int, height: int, allow_unicode: bool = True) -> str:
    if width < 60 or height < 18:
        return "BOA Researcher | boa init"
    lines = UNICODE_BANNER if allow_unicode else ASCII_BANNER
    return "\n".join(_center_lines(lines, width))
