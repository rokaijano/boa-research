from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from typing import Iterable


CODEX_MODELS_URL = "https://developers.openai.com/codex/models"
MODEL_SNIPPET_PATTERN = re.compile(r"codex\s+-m\s+([A-Za-z0-9][A-Za-z0-9._:-]*)")


def fetch_codex_models_page(url: str = CODEX_MODELS_URL, *, timeout_seconds: int = 10) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "boa-research-codex-model-helper/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_codex_models(page_text: str) -> list[str]:
    decoded = html.unescape(str(page_text or ""))
    models = MODEL_SNIPPET_PATTERN.findall(decoded)
    return list(dict.fromkeys(models))


def format_models(models: Iterable[str], *, as_json: bool) -> str:
    items = list(models)
    if as_json:
        return json.dumps({"source": CODEX_MODELS_URL, "models": items}, indent=2)
    return "\n".join(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Codex model names from the OpenAI Codex models page.")
    parser.add_argument("--url", default=CODEX_MODELS_URL, help="Override the source page URL.")
    parser.add_argument("--json", action="store_true", help="Print models as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        page_text = fetch_codex_models_page(args.url)
    except Exception as exc:
        print(f"Failed to fetch Codex models page: {exc}", file=sys.stderr)
        return 1
    models = extract_codex_models(page_text)
    if not models:
        print("No Codex models were found on the page.", file=sys.stderr)
        return 2
    print(format_models(models, as_json=bool(args.json)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())