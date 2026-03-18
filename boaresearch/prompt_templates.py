from __future__ import annotations

from pathlib import Path


PROMPTS_ROOT = (Path(__file__).resolve().parent / "prompts").resolve()


def read_prompt_template(*parts: str) -> str:
    path = PROMPTS_ROOT.joinpath(*parts)
    return path.read_text(encoding="utf-8").strip()


def render_prompt_template(*parts: str, **values: str) -> str:
    template = read_prompt_template(*parts)
    return template.format_map({key: str(value) for key, value in values.items()}).strip()
