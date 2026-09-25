"""Show which drawing characters a renderer gives an unexpected width.

Every non-ASCII character the reports draw with is East Asian *Ambiguous*: the standard
permits one column or two, so a terminal, a browser and a PDF engine may each choose
differently.  A font that lacks the glyph is a second way to lose: the renderer falls back
to a font whose advance width is its own, and every cell after it shifts.

This prints one row per character group, each ending with a ``|``.  Where every ``|`` lands
in a single column, that renderer draws the reports the way they are written.  Any ``|``
that sits left or right of the ASCII baseline's is a character to stop using there.

Run it in each place the docs are read -- a terminal, and by pasting the output into a page
built by Sphinx -- and compare::

    python docs/check_glyph_widths.py

It imports nothing outside the standard library, so it runs under any Python 3.
"""

from __future__ import annotations

import sys
import unicodedata

# Drawn this many times per row.  Wide enough that a half-cell error is unmistakable.
RUN = 24

GROUPS: list[tuple[str, str]] = [
    ("ASCII baseline", "#"),
    ("vertical eighths (sparklines)", "▁▂▃▄▅▆▇█"),
    ("left eighths (bar fractions)", "▏▎▍▌▋▊▉█"),
    ("light shade (ratio bars)", "░"),
    ("box drawing (box plots)", "─│├┤"),
    ("punctuation (table columns)", "–—→·"),
]


def _widths(chars: str) -> str:
    """Name the East Asian width class each character in the group carries."""
    classes = {unicodedata.east_asian_width(ch) for ch in chars}
    names = {"Na": "narrow", "N": "neutral", "A": "ambiguous", "W": "wide", "F": "full", "H": "half"}
    return "/".join(sorted(names.get(c, c) for c in classes))


def main() -> int:
    label_w = max(len(label) for label, _ in GROUPS)
    ruler = "".join(str((i // 10) % 10) if i % 10 == 0 else "." for i in range(RUN))
    print(f"{'':<{label_w}}  {ruler}|  (every | below should land in this column)\n")
    for label, chars in GROUPS:
        row = "".join(chars[i % len(chars)] for i in range(RUN))
        print(f"{label:<{label_w}}  {row}|  {_widths(chars)}")
    print(
        "\nEvery | in one column: this renderer agrees with the reports.\n"
        "A | pushed right: that group is being drawn wider than one cell, or the font\n"
        "lacks the glyph and a fallback with its own advance width is drawing it.\n"
        "\nThe documentation pins DataEval Mono (see docs/build_font_subset.py) so the HTML\n"
        "does not depend on installed fonts.  A terminal uses whichever font it is set to,\n"
        "so fix a misaligned terminal by choosing a font with full Block Elements coverage\n"
        "-- DejaVu Sans Mono, JetBrains Mono and Cascadia Code all qualify."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
