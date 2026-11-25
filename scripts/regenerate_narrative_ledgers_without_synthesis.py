#!/usr/bin/env python3
"""Regenerate narrative source packs + topic reference ledgers WITHOUT
invoking any narrative-synthesis provider call (Phase 10.1-10, Finding C).

Why this script exists (do not "just re-run --stage analysis" instead):

``scripts/run_cvpr2026_production.py``'s ``run_analysis`` unconditionally
reaches its deep-narrative call site once the Stage 6 receipt and the 07H/
07B/07C upstream receipts are present and succeeded (verified live this
session -- ``resume`` is accepted as a parameter but never referenced
anywhere in the function body, so ``--resume`` changes nothing about this).
Two facts about that call site make it unsafe to invoke here:

1. Its own call to ``render_narrative_outputs_from_source_packs`` hardcodes
   ``reuse_existing_pass_reports=False`` (a literal, not a CLI flag) --
   every invocation always attempts fresh synthesis for every topic, never
   reuses an existing passing report.
2. If ``--provider-cost-approved`` is passed, that fresh synthesis is a REAL
   PAID Anthropic/OpenAI call for every topic -- explicitly NOT approved by
   the operator's 2026-08-03 decision (``10.1-BACKFILL-PLAN.md``:
   "reuse_existing_pass_reports stays at its default" / "Forced paid
   narrative-markdown resynthesis... is explicitly NOT approved").
3. If ``--provider-cost-approved`` is NOT passed, ``llm_client`` is ``None``
   and ``allow_deterministic_fallback=False``, so
   ``render_narrative_topic_report`` returns a "blocked" PLACEHOLDER string
   for every topic -- which then gets written straight over the EXISTING,
   good ``analysis/{topic}.md`` files from the prior successful run. That is
   a silent, destructive regression, not a repair.

Neither outcome is acceptable, so this script does the narrow, deterministic
$0 subset of Stage 7 that the corpus-metadata repair actually needs --
confirmed by direct code trace this session that
``build_reference_ledger``/``run_topic_reference_validation`` (the functions
that write ``topic_reference_ledgers/*.json``, which is what
``speaker_author_work_ledger.json`` actually reads) never read
``analysis/{topic}.md`` at all:

1. ``build_analysis_source_packs`` -- rebuilds
   ``analysis/narrative_source_packs/*.json`` from the (freshly re-embedded)
   ``papers``/``artifacts`` vector-store sidecars, through this runner's own
   (Finding-A-fixed) ``_analysis_compact_chunk``. Deterministic keyword scan,
   $0, no provider call -- the SAME function ``run_analysis`` itself calls.
2. ``run_topic_reference_validation`` -- rebuilds
   ``analysis/topic_reference_ledgers/*.json`` from those source packs plus
   the EXISTING, untouched ``analysis/topic_reconciliations/{topic}.md``
   files (read, never regenerated, never required to be regenerated --
   ``build_reference_ledger`` does not consume reconciliation markdown
   content for its entries at all). Pure Python/file I/O, $0.

Neither step imports or constructs any provider client. This script asserts
that invariant is honored by refusing to import ``anthropic``/``openai``/
``OpenAIDeepNarrativeClient`` at all.

Usage::

    .venv/bin/python scripts/regenerate_narrative_ledgers_without_synthesis.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production \\
        --json

Add ``--dry-run`` to report what would be rebuilt without writing anything.
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
# call, this trips immediately rather than silently spending money.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the zero-paid ledger regeneration path alongside a live provider SDK."
        )

from conference_pipeline.analysis.topic_reference_validation import (  # noqa: E402
    run_topic_reference_validation,
)

RUNNER_PATH = REPO_ROOT / "scripts" / "run_cvpr2026_production.py"


def _load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cvpr2026_runner_for_ledger_regeneration", RUNNER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="report what would be rebuilt; write nothing")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = _load_runner_module()
    workspace = args.workspace.expanduser().resolve()
    run_id = runner.validate_run_id(args.run_id)
    target = runner.run_dir(workspace, run_id)

    # --- Mirror run_analysis's own Stage 6 admission gate (read-only) -------
    stage6 = runner.STAGES["corpus-inventory-index"]
    stage6_receipt_path = target / stage6.receipt_filename
    stage6_receipt, stage6_error = runner._analysis_load_json(stage6_receipt_path)
    if stage6_error or not isinstance(stage6_receipt, dict):
        print(f"ERROR: stage6 receipt missing or malformed at {stage6_receipt_path}", file=sys.stderr)
        return 2
    stage6_checks = stage6_receipt.get("checks") if isinstance(stage6_receipt.get("checks"), dict) else {}
    if (
        stage6_receipt.get("status") != "succeeded"
        or not stage6_checks.get("vector_embeddings_written")
        or stage6_checks.get("chunk_counts") != stage6_checks.get("embedded_chunk_counts")
    ):
        print("ERROR: stage6 vector evidence required (status/vector_embeddings_written/chunk_counts mismatch)", file=sys.stderr)
        return 2

    index_digest_path = workspace / runner.VECTOR_STORE_DIR / run_id / "index_digest.json"
    index_digest, index_digest_error = runner._analysis_load_json(index_digest_path)
    if (
        index_digest_error
        or not isinstance(index_digest, dict)
        or runner._file_sha256(index_digest_path) != str(stage6_checks.get("index_digest_sha256") or "")
    ):
        print(f"ERROR: stage6 index_digest stale or missing at {index_digest_path}", file=sys.stderr)
        return 2

    try:
        topic_specs = runner.load_analysis_topic_specs(workspace, run_id=run_id)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        summary = {
            "status": "dry_run",
            "workspace": str(workspace),
            "run_id": run_id,
            "topic_ids": [str(spec["id"]) for spec in topic_specs],
            "stage6_status": stage6_receipt.get("status"),
            "action": "would call build_analysis_source_packs then run_topic_reference_validation; no provider client constructed",
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Dry-run: would regenerate source packs + topic_reference_ledgers for topics:")
            for topic_id in summary["topic_ids"]:
                print(f"  {topic_id}")
        return 0

    # --- Step 1: deterministic source packs (Finding-A-fixed compaction) ---
    source_pack_result = runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    # --- Step 2: topic reference ledgers (pure file I/O, no provider) ------
    ledger_result = run_topic_reference_validation(
        workspace=workspace,
        run_id=run_id,
        topics=[str(spec["id"]) for spec in topic_specs],
    )

    summary = {
        "status": "success",
        "workspace": str(workspace),
        "run_id": run_id,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "source_pack_result": {
            "insufficient_topics": source_pack_result.get("insufficient_topics"),
            "source_pack_count": source_pack_result.get("source_pack_count"),
            "source_pack_paths": source_pack_result.get("source_pack_paths"),
        },
        "topic_reference_validation_result": ledger_result,
    }
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(
            f"Regenerated source packs + topic_reference_ledgers for "
            f"{len(topic_specs)} topics. insufficient_topics="
            f"{source_pack_result.get('insufficient_topics')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
