#!/usr/bin/env python3
"""Re-measure blueprint content-sufficiency numbers per deck, at $0.

Phase 10.1-10 measured this once with a throwaway, uncommitted script
(``/private/tmp/.../scratchpad/measure_thresholds.py``) that was deleted
after its own record was written -- that measurement is not reproducible.
This script is the committed replacement, reused here to re-measure the same
two numbers against the post-Phase-10.1.1 ledgers.

Method (identical to Phase 10.1-10's own precedent, see
``10.1-THRESHOLD-REEVALUATION.md``'s Method section): load
``scripts/presentations/generate_editorial_blueprint.py`` as a module via
``importlib.util.spec_from_file_location`` and call its own
``build_source_catalog(ledger)`` -- the exact function ``main()`` calls
before the first content-sufficiency gate fires, which happens before any
Opus client is ever constructed. No provider call is made anywhere in this
script.

Two numbers are reported per deck, both derived from the live catalog:

- ``eligible_source_count`` -- ``len(build_source_catalog(ledger))``,
  directly comparable to the live-module-read minimum eligible-source
  threshold.
- ``named_people_ceiling`` -- the size of the union of every catalog entry's
  ``people`` list (each entry capped at a handful of names by
  ``build_source_catalog`` itself). This is reported alongside a structured,
  always-present qualifier recording that it is an UPPER BOUND, not a pass:
  the real named-people gate (``coverage_stats``, in the same generator
  module) counts unique people actually cited in Opus-authored slide
  mentions, which requires a real (paid) Opus call this script deliberately
  never makes. Clearing the ceiling does not guarantee the real gate passes;
  failing the ceiling DOES prove the real gate cannot pass, because the
  ceiling is the most generous possible count.

The live minimum-eligible-source and minimum-named-people threshold values
are read from the loaded generator module at measurement time and are never
restated as integer literals anywhere in this script, in the receipt, or in
any report generated from it -- restating a threshold as a literal is
precisely how a stale expectation can silently survive a real threshold
change without anyone noticing.

$0 / no provider calls: refuses to run if ``anthropic``/``openai`` is already
imported, imports no provider SDK, and the receipt records
``provider_calls_made: false`` / ``live_provider_calls_made: false``.

Usage::

    .venv/bin/python scripts/measure_blueprint_content_sufficiency.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Refuse, at import time, to have any provider SDK anywhere on the path this
# script pulls in -- if a future edit accidentally reintroduces a provider
# call, this trips immediately rather than silently spending money. Mirrors
# scripts/measure_ledger_author_coverage.py and
# scripts/measure_source_pack_family_volumes.py verbatim in shape.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the $0 blueprint content-sufficiency measurement path alongside a live "
            "provider SDK."
        )

from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402

BLUEPRINT_GENERATOR_PATH = REPO_ROOT / "scripts" / "presentations" / "generate_editorial_blueprint.py"

# The structured qualifier attached to every receipt (top level) and every
# per-deck entry's named-people figure. Carried as data, not only as prose in
# a report file, so it survives being quoted elsewhere without being dropped.
NAMED_PEOPLE_CEILING_QUALIFIER = {
    "is_upper_bound": True,
    "not_a_pass": True,
    "explanation": (
        "named_people_ceiling is the size of the union of every eligible "
        "catalog entry's people list -- the most generous count a maximally "
        "effective editorial board could possibly reach if it cited every "
        "eligible source and every listed person on it. The real named-people "
        "gate counts unique people actually cited in Opus-authored slide "
        "mentions, which requires a real (paid) Opus call this measurement "
        "deliberately never makes. Clearing this ceiling does not guarantee "
        "the real gate passes. Failing this ceiling DOES prove the real gate "
        "cannot pass, because no selection could ever cite more named people "
        "than are listed on the eligible sources in the first place."
    ),
}


def _load_blueprint_module_from_path(path: Path, *, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_blueprint_module(path: Path | None = None) -> Any:
    """Load ``generate_editorial_blueprint.py`` as a module, live, read-only.

    Never writes to, and never re-derives, anything in the loaded module --
    this only imports it so its own ``build_source_catalog`` and its own
    live threshold values can be read directly, exactly as Phase 10.1-10's
    now-deleted throwaway script did.
    """

    return _load_blueprint_module_from_path(
        path or BLUEPRINT_GENERATOR_PATH,
        module_name="blueprint_gen_for_content_sufficiency_measurement",
    )


def measure_deck(ledger: dict[str, Any], module: Any) -> dict[str, Any]:
    """Measure one deck's ledger against the live generator module's own catalog.

    Reads the live minimum-eligible-source and minimum-named-people
    threshold values off ``module`` at call time -- never a literal copy --
    so a future threshold edit is reflected immediately, not silently stale.
    """

    catalog = module.build_source_catalog(ledger)
    eligible_source_count = len(catalog)
    named_people_ceiling = len({person for entry in catalog for person in entry.get("people") or []})

    min_sources = module.MIN_SOURCES
    min_people = module.MIN_PEOPLE

    return {
        "eligible_source_count": eligible_source_count,
        "min_sources": min_sources,
        "eligible_source_count_pass": eligible_source_count >= min_sources,
        "named_people_ceiling": named_people_ceiling,
        "min_people": min_people,
        "named_people_ceiling_clears_min_people": named_people_ceiling >= min_people,
        "named_people_ceiling_qualifier": dict(NAMED_PEOPLE_CEILING_QUALIFIER),
    }


def _measured_input_paths(workspace: Path) -> list[str]:
    decks_dir = workspace / "decks"
    if not decks_dir.is_dir():
        return []
    return sorted(
        platform_corpus_vector_store.workspace_relpath(workspace, path)
        for path in decks_dir.glob("*/speaker_author_work_ledger.json")
    )


def measure_workspace(workspace: Path, module: Any) -> dict[str, Any]:
    """Walk every ``decks/<id>/speaker_author_work_ledger.json`` under a workspace.

    Tolerates a missing or unreadable ledger by recording it in
    ``skipped_decks`` rather than raising -- a deck with no ledger at all
    (e.g. an executive-summary deck with no narrative source pack) still
    gets an explicit, reported entry rather than being silently absent from
    the receipt.
    """

    workspace = workspace.expanduser().resolve()
    decks_dir = workspace / "decks"
    per_deck: dict[str, dict[str, Any]] = {}
    skipped_decks: list[dict[str, str]] = []

    deck_dirs = sorted(child for child in decks_dir.iterdir() if child.is_dir()) if decks_dir.is_dir() else []
    for deck_dir in deck_dirs:
        ledger_path = deck_dir / "speaker_author_work_ledger.json"
        if not ledger_path.is_file():
            skipped_decks.append({"deck_id": deck_dir.name, "reason": "ledger_missing"})
            continue
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped_decks.append({"deck_id": deck_dir.name, "reason": f"ledger_unreadable:{exc}"})
            continue
        if not isinstance(ledger, dict):
            skipped_decks.append({"deck_id": deck_dir.name, "reason": "ledger_not_a_dict"})
            continue
        per_deck[deck_dir.name] = measure_deck(ledger, module)

    return {
        "decks": per_deck,
        "skipped_decks": skipped_decks,
    }


def write_content_sufficiency_receipt(
    path: Path,
    *,
    workspace: Path,
    run_id: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Write an atomic JSON content-sufficiency receipt.

    Never includes author names, titles, or any other corpus free text --
    counts, booleans, and paths/digests only.
    """

    workspace = workspace.expanduser().resolve()
    inputs = {
        relpath: platform_corpus_vector_store.sha256_file(workspace / relpath)
        for relpath in _measured_input_paths(workspace)
    }
    receipt = {
        "schema_version": 1,
        "generated_at": platform_corpus_vector_store.utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "named_people_ceiling_qualifier": dict(NAMED_PEOPLE_CEILING_QUALIFIER),
        "decks": measurement.get("decks", {}),
        "skipped_decks": measurement.get("skipped_decks", []),
        "inputs": inputs,
    }
    platform_corpus_vector_store.atomic_write_json(path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=None,
        help=(
            "Receipt path (default: storage/workflow-runs/<run-id>/"
            "blueprint_content_sufficiency.json under the workspace)"
        ),
    )
    parser.add_argument(
        "--blueprint-generator",
        type=Path,
        default=None,
        help="Override path to generate_editorial_blueprint.py (default: the in-repo copy)",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    module = load_blueprint_module(args.blueprint_generator)
    measurement = measure_workspace(workspace, module)

    output_receipt = args.output_receipt
    if output_receipt is None:
        output_receipt = workspace / "storage" / "workflow-runs" / args.run_id / "blueprint_content_sufficiency.json"
    elif not output_receipt.is_absolute():
        output_receipt = workspace / output_receipt

    receipt = write_content_sufficiency_receipt(
        output_receipt,
        workspace=workspace,
        run_id=args.run_id,
        measurement=measurement,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    else:
        print(f"Wrote blueprint content-sufficiency receipt for {len(measurement['decks'])} deck(s) to {output_receipt}")
        for deck_id, deck_result in measurement["decks"].items():
            print(
                f"  {deck_id}: eligible_sources={deck_result['eligible_source_count']}"
                f" (min={deck_result['min_sources']}, pass={deck_result['eligible_source_count_pass']})"
                f"  named_people_ceiling={deck_result['named_people_ceiling']}"
                f" (min={deck_result['min_people']},"
                f" clears={deck_result['named_people_ceiling_clears_min_people']}, upper bound only)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
