from __future__ import annotations

GREEN_SHADES = (28, 34, 40, 46)
YELLOW_SHADES = (220, 226)

UNICODE_BANNER = [
    "██████    ██████    █████",
    "██   ██  ██    ██  ██   ██",
    "██████   ██    ██  ███████",
    "██   ██  ██    ██  ██   ██",
    "██████    ██████   ██   ██",
    "",
    "██████   ███████  ███████  ███████   █████   ██████   ██████  ██   ██",
    "██   ██  ██       ██       ██       ██   ██  ██   ██  ██      ██   ██",
    "██████   █████    ███████  █████    ███████  ██████   ██      ███████",
    "██   ██  ██            ██  ██       ██   ██  ██   ██  ██      ██   ██",
    "██   ██  ███████  ███████  ███████  ██   ██  ██   ██  ██████  ██   ██",
    "",
    "⚗️  Bayesian Optimized Agents  ⚗️",
    "🐍  code patch → descriptor → trial → promotion  🐍",
]

ASCII_BANNER = [
    "BOA RESEARCH",
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


def _colorize_banner_line(line: str, *, line_index: int) -> str:
    stripped = line.strip()
    if not stripped:
        return line
    if "⚗" in stripped:
        shade = YELLOW_SHADES[0]
    elif "🐍" in stripped:
        shade = GREEN_SHADES[-1]
    else:
        shade = GREEN_SHADES[line_index % len(GREEN_SHADES)]
    return f"\033[38;5;{shade}m{line}\033[0m"


def render_banner(*, width: int, height: int, allow_unicode: bool = True) -> str:
    if width < 60 or height < 18:
        return "BOA Research | boa init"
    lines = UNICODE_BANNER if allow_unicode else ASCII_BANNER
    centered = _center_lines(lines, width)
    if not allow_unicode:
        return "\n".join(centered)
    return "\n".join(_colorize_banner_line(line, line_index=index) for index, line in enumerate(centered))
