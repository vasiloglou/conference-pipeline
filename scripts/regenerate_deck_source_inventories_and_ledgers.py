#!/usr/bin/env python3
"""Regenerate every deck's source-corpus inventory + speaker/author/work
ledger at $0 (Phase 10.1.1 plan 05, task 1).

``scripts/regenerate_narrative_ledgers_without_synthesis.py`` rebuilds the
narrative source packs and ``analysis/topic_reference_ledgers/*.json`` -- but
each deck's own ``decks/<id>/source_corpus_inventory.json`` pins a sha256
digest for every input it consumed, including that same topic reference
ledger file. Regenerating the ledger without rebuilding the inventory leaves
the inventory's pin stale (verified directly against the live
``02_generative_visual_media`` workspace: the pinned digest is
``553161e4...`` while the live ledger file's digest is ``e108f4df...``), which
hands the next phase a broken provenance chain. ``build_speaker_author_work_
ledger`` then reads the inventory for source PATHS and re-reads each file
live, so the *ledger content* itself is unaffected by the stale pin -- but the
hash-bound inventory contract itself is broken until this script runs.

This script is the missing third regeneration step: for every deck directory
under the workspace's ``decks/``, it rebuilds the source-corpus inventory
(replicating ``scripts/presentations/run_topic_presentation_build.py``'s
``stage_corpus_inventory`` candidate-input construction exactly, so the
inventory pins the same input set the canonical topic-build pipeline would),
then rebuilds the speaker/author/work ledger from that fresh inventory,
invoking the same two documented owner scripts
(``scripts/presentations/build_presentation_source_corpus_inventory.py`` and
``scripts/presentations/build_speaker_author_work_ledger.py``) as subprocess
calls so the stage-owner-audit's documented owner commands remain the actual
executed path.

Neither owner script makes a provider call. This script asserts that
invariant is honored the same way its precedent
(``regenerate_narrative_ledgers_without_synthesis.py``) does: it refuses to
run if ``anthropic``/``openai`` is already imported, and imports no provider
SDK itself.

Usage::

    .venv/bin/python scripts/regenerate_deck_source_inventories_and_ledgers.py \\
        --workspace conferences/cvpr2026 --json

Add ``--dry-run`` to print the exact commands and planned writes without
executing or writing anything. Add repeatable ``--deck <deck_id>`` to
regenerate a subset; omitted, every deck directory found under
``<workspace>/decks/`` is regenerated.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
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
# scripts/regenerate_narrative_ledgers_without_synthesis.py verbatim in shape.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the $0 deck inventory/ledger regeneration path alongside a live provider SDK."
        )

PRESENTATIONS_SCRIPTS = REPO_ROOT / "scripts" / "presentations"
BUILD_INVENTORY_SCRIPT = PRESENTATIONS_SCRIPTS / "build_presentation_source_corpus_inventory.py"
BUILD_LEDGER_SCRIPT = PRESENTATIONS_SCRIPTS / "build_speaker_author_work_ledger.py"

# Verbatim from run_topic_presentation_build.stage_corpus_inventory's own
# glob-candidate list (scripts/presentations/run_topic_presentation_build.py).
# Replicated, not reinvented -- a different input set would silently produce a
# different, non-comparable inventory (this plan's own "Verified prerequisite"
# section).
_CORPUS_INVENTORY_GLOB_PATTERNS = (
    "analysis/deep_dive_topics/{deck}__dd0*__global.md",
    "analysis/deep_dive_topics/{deck}__dd0*__heuristic.md",
    "analysis/exhaustive_summaries/{deck}__papers.json",
    "analysis/exhaustive_summaries/{deck}__papers.md",
    "analysis/exhaustive_summaries/{deck}__transcripts.json",
    "analysis/exhaustive_summaries/{deck}__transcripts.md",
    "analysis/narrative_source_packs/{deck}.json",
    "analysis/provenance/{deck}.json",
    "analysis/provenance/{deck}.md",
    "analysis/provenance/query_plans/{deck}.json",
    "analysis/topic_reference_candidate_resolution/{deck}*.*",
    "analysis/topic_reference_ledgers/{deck}*.json",
    "analysis/topic_reference_repairs/{deck}*.*",
    "analysis/topic_reference_validation/{deck}*.*",
    "decks/{deck}/extracted_content.json",
    "decks/{deck}/ranked_papers.json",
    "decks/{deck}/source_attributions.json",
    "decks/{deck}/source_pack_ranked_papers.json",
    "decks/{deck}/source_pack_receipt.json",
)


def deck_inventory_source_inputs(workspace: Path, deck_id: str) -> list[str]:
    """Replicate ``stage_corpus_inventory``'s candidate-input construction.

    Starts with the deck's ``analysis/<deck>.md``, then globs each documented
    pattern under the workspace (sorted per-pattern, converted to
    workspace-relative paths), keeping only paths that resolve to an existing
    file. Raises ``RuntimeError`` with the same message
    ``run_topic_presentation_build.stage_corpus_inventory`` raises when a deck
    yields no inputs at all.
    """

    candidates: list[str] = [f"analysis/{deck_id}.md"]
    for pattern in _CORPUS_INVENTORY_GLOB_PATTERNS:
        resolved_pattern = pattern.format(deck=deck_id)
        for match in sorted(glob.glob(str(workspace / resolved_pattern))):
            candidates.append(str(Path(match).relative_to(workspace)))

    inputs = [item for item in candidates if (workspace / item).is_file()]
    if not inputs:
        raise RuntimeError("corpus inventory: no analysis-foundation source inputs found")
    return inputs


def _printable_cmd(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run_owner_script(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def regenerate_deck(workspace: Path, deck_id: str, *, dry_run: bool) -> dict[str, Any]:
    """Rebuild one deck's source-corpus inventory, then its speaker/author/
    work ledger, in that order, by invoking the two documented owner scripts
    as subprocesses under this process's own ``sys.executable``."""

    deck_rel = f"decks/{deck_id}"
    inputs = deck_inventory_source_inputs(workspace, deck_id)

    inventory_output = f"{deck_rel}/source_corpus_inventory.json"
    inventory_receipt = f"{deck_rel}/source_corpus_inventory_receipt.json"
    inventory_cmd = [str(BUILD_INVENTORY_SCRIPT), "--workspace", str(workspace), "--topic-id", deck_id, "--deck-dir", deck_rel]
    for item in inputs:
        inventory_cmd += ["--source-input", item]
    inventory_cmd += ["--output-json", inventory_output, "--receipt-path", inventory_receipt, "--summary"]

    ledger_output = f"{deck_rel}/speaker_author_work_ledger.json"
    ledger_receipt = f"{deck_rel}/speaker_author_work_ledger_receipt.json"
    ledger_cmd = [
        str(BUILD_LEDGER_SCRIPT),
        "--workspace", str(workspace),
        "--inventory-json", inventory_output,
        "--output-json", ledger_output,
        "--receipt-path", ledger_receipt,
        "--summary",
    ]

    if dry_run:
        return {
            "deck_id": deck_id,
            "dry_run": True,
            "source_input_count": len(inputs),
            "commands": {
                "corpus_inventory": _printable_cmd(inventory_cmd),
                "work_ledger": _printable_cmd(ledger_cmd),
            },
            "planned_writes": [inventory_output, inventory_receipt, ledger_output, ledger_receipt],
        }

    inventory_result = _run_owner_script(inventory_cmd)
    result: dict[str, Any] = {
        "deck_id": deck_id,
        "corpus_inventory": {
            "command": _printable_cmd(inventory_cmd),
            "exit_code": inventory_result.returncode,
            "receipt_path": inventory_receipt,
            "stdout_tail": _tail(inventory_result.stdout),
            "stderr_tail": _tail(inventory_result.stderr),
        },
    }
    if inventory_result.returncode != 0:
        result["work_ledger"] = {"skipped": True, "reason": "corpus_inventory_failed"}
        return result

    ledger_result = _run_owner_script(ledger_cmd)
    result["work_ledger"] = {
        "command": _printable_cmd(ledger_cmd),
        "exit_code": ledger_result.returncode,
        "receipt_path": ledger_receipt,
        "stdout_tail": _tail(ledger_result.stdout),
        "stderr_tail": _tail(ledger_result.stderr),
    }
    return result


def _default_deck_ids(workspace: Path) -> list[str]:
    decks_dir = workspace / "decks"
    if not decks_dir.is_dir():
        return []
    return sorted(child.name for child in decks_dir.iterdir() if child.is_dir())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--deck",
        action="append",
        default=[],
        help="deck id to regenerate (repeatable); default is every deck directory found",
    )
    parser.add_argument("--dry-run", action="store_true", help="print planned commands/writes; execute nothing")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    deck_ids = list(args.deck) if args.deck else _default_deck_ids(workspace)

    decks: list[dict[str, Any]] = []
    exit_code = 0
    for deck_id in deck_ids:
        try:
            deck_result = regenerate_deck(workspace, deck_id, dry_run=args.dry_run)
        except RuntimeError as exc:
            decks.append({"deck_id": deck_id, "error": str(exc)})
            exit_code = 2
            continue
        decks.append(deck_result)
        if not args.dry_run:
            inventory_exit = deck_result.get("corpus_inventory", {}).get("exit_code")
            ledger_exit = deck_result.get("work_ledger", {}).get("exit_code")
            if inventory_exit not in (0,) or (not deck_result.get("work_ledger", {}).get("skipped") and ledger_exit not in (0,)):
                exit_code = max(exit_code, 1)

    summary = {
        "status": "dry_run" if args.dry_run else ("success" if exit_code == 0 else "failed"),
        "workspace": str(workspace),
        "deck_count": len(deck_ids),
        "deck_ids": deck_ids,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "decks": decks,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(f"{summary['status']}: {len(deck_ids)} deck(s) processed")
        for deck_result in decks:
            if "error" in deck_result:
                print(f"  {deck_result['deck_id']}: ERROR {deck_result['error']}")
            elif args.dry_run:
                print(f"  {deck_result['deck_id']}: would rebuild inventory + ledger ({deck_result['source_input_count']} inputs)")
            else:
                inv_exit = deck_result["corpus_inventory"]["exit_code"]
                led = deck_result["work_ledger"]
                led_exit = led.get("exit_code", "skipped")
                print(f"  {deck_result['deck_id']}: inventory exit={inv_exit} ledger exit={led_exit}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
