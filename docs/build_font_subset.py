"""Build the subset of DejaVu Sans Mono the documentation ships.

The report renderers draw with block, box-drawing and arrow characters, every one of which
is East Asian *Ambiguous*: a renderer may give it one column or two, and a font that lacks
the glyph falls back to one whose advance width is its own.  Either way a row shifts, which
is why the same table looks different in a terminal, on Read the Docs and on GitLab Pages.

Pinning one webfont that actually holds every glyph removes the choice.  DejaVu Sans Mono
covers all of them; subsetting keeps the download small enough to ship in the repo.

Run from the repository root with the dev environment active::

    .venv/bin/python docs/build_font_subset.py
"""

from __future__ import annotations

import pathlib
import sys

from fontTools import subset

SOURCE = pathlib.Path("/usr/share/fonts/truetype/dejavu")
OUT = pathlib.Path(__file__).parent / "source" / "_static" / "fonts"

#: Everything the report renderers can emit, plus the block each character sits in so a new
#: glyph from the same block needs no rebuild.  `src/dataeval_flow/workflow/_text_report.py`
#: and the triage report are the writers; keep this in step with them.
RANGES = ",".join(  # noqa: FLY002
    [
        "U+0020-007E",  # ASCII
        "U+00A0-00FF",  # Latin-1: · ° × ² and the currency signs
        "U+0391-03C9",  # Greek: μ and the statistical symbols beside it
        "U+2000-206F",  # General punctuation: – — …
        "U+20A0-20BF",  # Currency symbols: €
        "U+2190-21FF",  # Arrows: ← → ↔ ↕
        "U+2500-257F",  # Box drawing: ─ │ ├ ┤
        "U+2580-259F",  # Block elements: the sparkline and bar glyphs
        "U+25A0-25FF",  # Geometric shapes: ● ◐ ▶ ▼
        "U+2713-2718",  # Check marks: ✓ ✗
    ]
)

WEIGHTS = {"DejaVuSansMono.ttf": "DataEvalMono-Regular", "DejaVuSansMono-Bold.ttf": "DataEvalMono-Bold"}


def main() -> int:
    if not SOURCE.is_dir():
        print(f"DejaVu source fonts not found at {SOURCE}", file=sys.stderr)
        print("Install them with: apt-get install fonts-dejavu-core", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name, stem in WEIGHTS.items():
        for flavor in ("woff", None):
            args = [
                str(SOURCE / source_name),
                f"--unicodes={RANGES}",
                f"--output-file={OUT / (stem + ('.woff' if flavor else '.ttf'))}",
                "--layout-features=*",
            ]
            if flavor:
                args.append(f"--flavor={flavor}")
            subset.main(args)
    for built in sorted(OUT.glob("DataEvalMono-*")):
        print(f"  {built.name:<28} {built.stat().st_size / 1024:6.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
