"""Can a figure built to its published budget actually be drawn?

The authoring prompt hands the model two hard numbers per archetype: how many
primary items it may use and how long each label may be. Nothing checked that
a spec obeying both could be compiled, and for sixteen of thirty-five
archetypes it could not. Two of the CVPR 2026 canary deck's four refused
figures sat *exactly* on their stated maximum -- six frames in a strip whose
budget said six, five regions in a landscape whose budget said five -- so the
author was told what to write, wrote it, and the compiler refused after the
paid call was already spent.

The budgets in `ARCHETYPES` are now measured rather than guessed. These tests
keep them measured: raising one without teaching the layout to hold it fails
here, and so does a layout change that quietly loses capacity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))

from figure_budget_cases import at_budget  # noqa: E402

from conference_pipeline.figures.compiler import compile_figure  # noqa: E402
from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402
from conference_pipeline.figures.spec import (  # noqa: E402
    ARCHETYPES,
    load_spec,
    validate_spec,
)

GROWABLE = sorted(name for name in ARCHETYPES if at_budget(name) is not None)
FIXED = sorted(set(ARCHETYPES) - set(GROWABLE))


@pytest.mark.parametrize("archetype", GROWABLE)
def test_an_archetype_draws_at_its_published_budget(archetype: str) -> None:
    """max_primary items, each carrying a max_label_chars label, must compile."""

    arch = ARCHETYPES[archetype]
    payload = at_budget(archetype)
    errors = validate_spec(load_spec(payload, strict=False))
    assert errors == [], f"{archetype} at its own budget does not validate: {errors[:2]}"
    figure = compile_figure(payload)
    # A ladder rung trimming edge labels or annotations is the designed remedy
    # and stays legal; a drop for any other reason is already a gate error, so
    # it would have refused above rather than reach here.
    assert [drop for drop in figure.manifest.labels_dropped
            if not str(drop["reason"]).startswith("ladder:")] == []
    assert figure.manifest.labels_ignored == []
    assert figure.svg.startswith("<svg"), (
        f"{archetype} publishes {arch.max_primary} items x {arch.max_label_chars} chars "
        "and cannot draw it"
    )


@pytest.mark.parametrize("archetype", GROWABLE)
def test_every_count_up_to_the_budget_draws(archetype: str) -> None:
    """A budget is a promise about every count below it, not only the top one.

    A ceiling that holds at eight items and fails at seven is worse than a
    ceiling of six: the author has no way to know which counts are real.
    Counts the structure lint refuses on their own terms -- a two-stage loop,
    a one-branch tree -- are not capacity limits and are skipped.
    """

    arch = ARCHETYPES[archetype]
    unusable: list[int] = []
    for count in range(1, arch.max_primary + 1):
        # At each count the contract promises that count's own allowance.
        payload = at_budget(archetype, count=count, chars=arch.label_chars(count))
        if validate_spec(load_spec(payload, strict=False)):
            continue  # below this archetype's own structural minimum
        try:
            compile_figure(payload)
        except Exception:  # noqa: BLE001 - any refusal is the failure here
            unusable.append(count)
    assert unusable == [], (
        f"{archetype} publishes a budget of {arch.max_primary} but refuses at {unusable}"
    )


@pytest.mark.parametrize("archetype", FIXED)
def test_a_fixed_archetype_draws_its_worked_example(archetype: str) -> None:
    """A tug of war has two sides; there is no list to grow, so the budget holds."""

    compile_figure(EXAMPLES[archetype])


def test_every_archetype_is_covered_by_one_of_the_two_cases() -> None:
    assert set(GROWABLE) | set(FIXED) == set(ARCHETYPES)
    assert len(GROWABLE) >= 30, sorted(FIXED)
