"""Golden SVG snapshots, one per archetype (design spec §9).

The geometry gate proves a figure is *legal* -- nothing overflows, nothing
overlaps, every connector lands. It cannot prove a figure still looks the way it
looked yesterday: a change that moves every node ten pixels left, swaps the
palette, or quietly drops a sublabel passes the gate and changes every slide.

These snapshots close that gap. They are the compiler's exact output for the
worked example of each archetype, so any style or layout drift shows up as a
diff on a reviewable text file rather than as a surprise in a rendered deck.

To update after an intentional change::

    .venv/bin/python tests/test_figure_golden.py --update

Read the diff before committing it. A one-line change is usually the change you
meant; 35 changed files usually means a token or metric moved under you.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.figures.compiler import compile_figure  # noqa: E402
from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "figure_golden"


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.svg"


@pytest.mark.parametrize("archetype", sorted(EXAMPLES))
def test_archetype_matches_its_golden_svg(archetype: str) -> None:
    path = _golden_path(archetype)
    assert path.is_file(), (
        f"no golden snapshot for '{archetype}'. Run "
        "`.venv/bin/python tests/test_figure_golden.py --update` to create it."
    )
    produced = compile_figure(EXAMPLES[archetype]).svg
    expected = path.read_text(encoding="utf-8")
    if produced != expected:
        pytest.fail(
            f"'{archetype}' no longer compiles to its golden snapshot "
            f"({len(expected)} bytes recorded, {len(produced)} produced). If the change "
            "was intended, review the diff and re-record with "
            "`.venv/bin/python tests/test_figure_golden.py --update`."
        )


def test_every_archetype_has_a_golden_snapshot() -> None:
    """A new archetype must arrive with a snapshot, not slip in unwatched."""

    recorded = {path.stem for path in GOLDEN_DIR.glob("*.svg")}
    assert recorded == set(EXAMPLES), {
        "missing_snapshots": sorted(set(EXAMPLES) - recorded),
        "orphan_snapshots": sorted(recorded - set(EXAMPLES)),
    }


def _update() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GOLDEN_DIR.glob("*.svg"):
        if stale.stem not in EXAMPLES:
            stale.unlink()
    changed = 0
    for name in sorted(EXAMPLES):
        path = _golden_path(name)
        produced = compile_figure(EXAMPLES[name]).svg
        if not path.is_file() or path.read_text(encoding="utf-8") != produced:
            path.write_text(produced, encoding="utf-8")
            changed += 1
            print(f"updated {path.relative_to(REPO_ROOT)}")
    print(f"{changed} snapshot(s) updated, {len(EXAMPLES)} total")
    return 0


if __name__ == "__main__":
    if "--update" in sys.argv:
        raise SystemExit(_update())
    raise SystemExit("pass --update to re-record the golden snapshots")
