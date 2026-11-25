#!/usr/bin/env python3
"""Refresh stale ``analysis_manifest.json`` topic markdown digests (Phase 8 repair).

Root cause (found while executing Phase 8 Plan 08-04's live Stage 8
``qa-gap-fill-synthesis`` run): Plan 07-05's post-synthesis fact-check
remediation edited 8 of the 12 CVPR2026 topic markdown files (fixing
fabricated figures and misattributed citations, all independently reviewed
and operator-approved -- see ``07-05-SUMMARY.md``) but never regenerated
``analysis/analysis_manifest.json``'s per-topic ``markdown_sha256`` fields
afterward. Both the manifest and the corrected ``.md`` files landed in the
same commit (``829e708f6``), so the manifest has been silently stale
relative to its own approved content since Phase 7 closed. Stage 8's
``qa-gap-fill-synthesis`` traceability gate
(``stage8_topic_file_digest_mismatch``) was the first stage to ever read
``markdown_sha256``, and correctly caught the drift by blocking with zero
provider spend rather than fabricating a pass.

This is a pure digest-refresh, not a manifest rebuild:

- Only per-topic ``markdown_sha256`` and the manifest's own top-level
  ``content_sha256`` are recomputed, using the exact ``sha256_file`` /
  ``_compute_content_sha256`` helpers ``emit_analysis_manifest`` itself
  already uses -- no bespoke hashing.
- No other manifest field (``deep_validation``, ``narrative_model``,
  ``citation_counts``, ``analysis_profile``, ``slide_brief_path``,
  provenance paths, ``topic_reconciliation``, ...) is touched. This is
  deliberately *not* ``scripts/backfill_analysis_manifest.py`` -- that
  script calls ``emit_analysis_manifest`` with no
  ``citations_by_topic``/``provenance_by_topic``/
  ``narrative_metadata_by_topic`` arguments and would silently wipe all of
  the above for every topic.
- ``stage-07-analysis.json``'s ``checks.analysis_manifest_content_sha256``
  is updated to match the freshly recomputed manifest digest, and
  ``verify_stage_contract(stage_id="analysis", ...)`` is re-run against the
  updated receipt (reconstructing the exact provisional-receipt shape
  ``run_analysis``'s own call site builds) to confirm its
  ``stage_contract.status`` is still ``"pass"`` after the update -- Phase
  7's already-``07-VERIFICATION.md``-approved contract must not silently
  regress. If it would regress, this script aborts without writing
  anything.
- The receipt's recorded ``lineage`` is carried through (never re-bound)
  and its ``output_digest`` is recomputed, so downstream stages that bound
  the old analysis output read stale; the prior receipt is preserved as a
  ``.superseded-*`` copy. A receipt without lineage is refused -- backfill
  the run first.

Usage::

    .venv/bin/python scripts/refresh_analysis_manifest_digests.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production

Add ``--dry-run`` to print the before/after digest diff without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.manifest import _compute_content_sha256
from conference_pipeline.receipt_lineage import finalize_receipt, preserve_prior_receipt
from conference_pipeline.stage_contracts import verify_stage_contract
from conference_pipeline.workflow_runs import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Path to the conference workspace (e.g., conferences/cvpr2026).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Production run id whose stage-07-analysis.json receipt should be reconciled.",
    )
    parser.add_argument(
        "--stage07-receipt",
        type=Path,
        default=None,
        help=(
            "Override path to the analysis-stage receipt "
            "(default: <workspace>/storage/workflow-runs/<run-id>/stage-07-analysis.json)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace.resolve()
    manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
    stage07_path = args.stage07_receipt or (
        workspace_root / "storage" / "workflow-runs" / args.run_id / "stage-07-analysis.json"
    )

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not stage07_path.is_file():
        print(f"ERROR: stage-07 receipt not found: {stage07_path}", file=sys.stderr)
        return 2

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_manifest_content_sha256 = str(manifest_data.get("content_sha256") or "")

    topics = manifest_data.get("topics")
    if not isinstance(topics, list):
        print("ERROR: manifest has no 'topics' list", file=sys.stderr)
        return 2

    topic_digest_changes: list[dict[str, str]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = str(topic.get("id") or "")
        markdown_path_rel = topic.get("markdown_path")
        if not markdown_path_rel:
            continue
        markdown_path = (workspace_root / str(markdown_path_rel)).resolve()
        if not markdown_path.is_file():
            print(
                f"WARNING: markdown file missing for topic {topic_id}: {markdown_path}",
                file=sys.stderr,
            )
            continue
        current_sha256 = sha256_file(markdown_path)
        recorded_sha256 = str(topic.get("markdown_sha256") or "")
        if current_sha256 != recorded_sha256:
            topic_digest_changes.append(
                {
                    "topic_id": topic_id,
                    "path": str(markdown_path_rel),
                    "old_markdown_sha256": recorded_sha256,
                    "new_markdown_sha256": current_sha256,
                }
            )
            topic["markdown_sha256"] = current_sha256

    new_manifest_content_sha256 = _compute_content_sha256(manifest_data)
    manifest_data["content_sha256"] = new_manifest_content_sha256

    print("=== Per-topic markdown_sha256 changes ===")
    if topic_digest_changes:
        for change in topic_digest_changes:
            print(f"  {change['topic_id']} ({change['path']}):")
            print(f"    old: {change['old_markdown_sha256']}")
            print(f"    new: {change['new_markdown_sha256']}")
    else:
        print("  (none)")

    print("=== Manifest content_sha256 ===")
    print(f"  old: {old_manifest_content_sha256}")
    print(f"  new: {new_manifest_content_sha256}")

    stage07_data = json.loads(stage07_path.read_text(encoding="utf-8"))
    checks = stage07_data.get("checks")
    if not isinstance(checks, dict):
        print("ERROR: stage-07 receipt has no 'checks' object", file=sys.stderr)
        return 2
    old_receipt_content_sha256 = str(checks.get("analysis_manifest_content_sha256") or "")
    checks["analysis_manifest_content_sha256"] = new_manifest_content_sha256

    print("=== stage-07-analysis.json checks.analysis_manifest_content_sha256 ===")
    print(f"  old: {old_receipt_content_sha256}")
    print(f"  new: {new_manifest_content_sha256}")

    if not topic_digest_changes and old_manifest_content_sha256 == new_manifest_content_sha256:
        print("Nothing to refresh; manifest and receipt are already consistent.")
        return 0

    # Reconstruct the provisional receipt exactly as run_analysis's own
    # verify_stage_contract call site does (scripts/run_cvpr2026_production.py,
    # run_analysis, ~L9919-9933), so re-verification exercises the identical
    # contract path Phase 7's own live run used -- not a bespoke one.
    #
    # Receipt lineage (v5.1): the refresh changes no input, so the receipt's
    # *recorded* lineage is carried through unchanged (never re-bound -- a
    # stale receipt must stay stale). Its output digest is recomputed from the
    # refreshed checks, so every stage that bound the old analysis output
    # reads stale and re-runs.
    recorded_lineage = stage07_data.get("lineage")
    if not isinstance(recorded_lineage, dict):
        print(
            "ERROR: stage-07 receipt has no lineage; run `conference-pipeline workflow "
            "bind-lineage --backfill` for this run first. Nothing written.",
            file=sys.stderr,
        )
        return 1
    provisional_receipt = finalize_receipt(
        {
            "mode": stage07_data.get("mode"),
            "script_registry_sha256": stage07_data.get("script_registry_sha256"),
            "required_script_ids": stage07_data.get("required_script_ids"),
            "owner_execution_evidence": stage07_data.get("owner_execution_evidence"),
            "checks": checks,
        },
        workspace=workspace_root,
        run_id=args.run_id,
        stage_id="analysis",
        run_dir=stage07_path.parent,
        lineage=recorded_lineage,
        bind=False,
    )
    new_contract = verify_stage_contract(
        workspace=workspace_root,
        run_id=args.run_id,
        stage_id="analysis",
        receipt=provisional_receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()

    print("=== Re-verified stage_contract (stage_id=analysis) ===")
    print(f"  status: {new_contract['status']}")
    if new_contract["status"] != "pass":
        print(
            "ERROR: stage_contract no longer passes after the digest refresh; "
            "aborting WITHOUT writing anything.",
            file=sys.stderr,
        )
        print(json.dumps(new_contract, indent=2), file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry-run: not writing any file.")
        return 0

    stage07_data["stage_contract"] = new_contract
    for key in ("output_artifacts", "output_digest", "lineage"):
        stage07_data[key] = provisional_receipt[key]

    write_json_atomic(manifest_path, manifest_data)
    archived = preserve_prior_receipt(stage07_path, new_status=str(stage07_data.get("status") or ""))
    write_json_atomic(stage07_path, stage07_data)
    if archived is not None:
        print(f"Preserved prior receipt as {archived}")

    print(f"Wrote {manifest_path}")
    print(f"Wrote {stage07_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
