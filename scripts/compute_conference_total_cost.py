#!/usr/bin/env python3
"""Compute the TOTAL cost of a conference: canonical workspace receipts plus
attributable misplaced repo-root receipts (see docs/reference/stage-owner-audit.md).

Background
----------
Some provider-calling scripts historically ran without a workspace-rooted
``provider_receipt_context`` (see D-QUICK-260715-rz1 / Task 2 of this plan),
so their receipts fell back to ``Path.cwd()`` and landed under
``<repo-root>/storage/provider-calls/<segment>/`` instead of the owning
conference workspace's ``storage/``. Those legacy receipts are real spend
that belongs to a specific conference, but ``conference_pipeline.costs
actual`` only scans the workspace it's pointed at and never sees them.

This command reports the TRUE total cost of a conference by combining:
  1. Canonical receipts: everything under ``<workspace>/storage/**/provider-calls/**``
     (via ``scan_provider_call_receipts(workspace)`` — covers both the flat
     ``storage/provider-calls/`` layout and the per-run
     ``storage/workflow-runs/<run-id>/provider-calls/`` layout).
  2. Misplaced repo-root receipts that the operator explicitly attributes to
     this conference via ``--attribute-segment`` and/or ``--since``/``--until``.
     Receipts that cannot be confidently attributed are reported in a
     separate ``misplaced_unattributed`` bucket — never silently dropped and
     never folded into ``total_usd``.

This script is READ-ONLY on the budget cap guardrail: it reports breach
status against the existing ``GLOBAL_PROVIDER_BUDGET_CAP_USD`` constant but
never writes/raises it (halt-and-approve guardrail, see threat model T-rz1-02).

It does not move, delete, or rewrite any existing receipt file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.provider_receipts import (  # noqa: E402
    aggregate_provider_call_receipts,
    scan_provider_call_receipts,
)


def _segment_for_receipt(receipt: dict[str, Any]) -> str:
    """Return the receipt's owning segment (directory name under provider-calls/).

    Receipts written without an explicit ``stage_id`` (the exact bug this
    plan's Task 2 fixes going forward) store ``stage_id: null`` even though
    the file landed under a caller-derived directory name
    (``provider-calls/<segment>/...``, see ``_call_receipt_path`` /
    ``safe_segment`` in provider_receipts.py). Prefer the recorded
    ``stage_id`` when present; otherwise fall back to the receipt's own path.
    """

    stage_id = receipt.get("stage_id")
    if stage_id:
        return str(stage_id)
    receipt_path = receipt.get("receipt_path")
    if receipt_path:
        return Path(receipt_path).parent.name
    return "unknown"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _receipt_started_at(receipt: dict[str, Any]) -> datetime | None:
    raw = receipt.get("started_at")
    if not raw:
        return None
    try:
        return _parse_iso(str(raw))
    except ValueError:
        return None


def _bucket_breakdown(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a by-model / by-provider / by-stage breakdown for one bucket.

    Reuses ``aggregate_provider_call_receipts`` (provider/model totals,
    token totals, unknown/failed counts) rather than reimplementing pricing
    math, and adds a by-stage grouping (plain summation of already-priced
    ``estimated_cost_usd`` values — no new pricing logic).
    """

    aggregate = aggregate_provider_call_receipts(receipts)
    by_stage: dict[str, float] = {}
    for receipt in receipts:
        cost = receipt.get("estimated_cost_usd")
        if not isinstance(cost, (int, float)):
            continue
        segment = _segment_for_receipt(receipt)
        by_stage[segment] = by_stage.get(segment, 0.0) + float(cost)

    return {
        "total_usd": aggregate["canonical_total_estimated_cost_usd"],
        "receipt_count": aggregate["canonical_receipt_files_scanned"],
        "unknown_cost_receipts": aggregate["unknown_cost_receipts"],
        "failed_receipts": aggregate["failed_receipts"],
        "by_provider": aggregate["canonical_totals_by_provider"],
        "by_model": aggregate["canonical_totals_by_model"],
        "by_stage": {key: round(value, 6) for key, value in sorted(by_stage.items())},
    }


def _attribute_misplaced_receipts(
    receipts: list[dict[str, Any]],
    *,
    attribute_segments: set[str],
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split repo-root receipts into (attributed, unattributed).

    A receipt is attributed only when the operator gave at least one
    attribution filter (``--attribute-segment`` and/or ``--since``/``--until``)
    AND the receipt satisfies every filter that was given. With no filters at
    all, everything is reported unattributed (safe default — never silently
    claim all repo-root spend for one conference).
    """

    if not attribute_segments and since is None and until is None:
        return [], list(receipts)

    attributed: list[dict[str, Any]] = []
    unattributed: list[dict[str, Any]] = []
    for receipt in receipts:
        ok = True
        if attribute_segments and _segment_for_receipt(receipt) not in attribute_segments:
            ok = False
        if ok and since is not None:
            started_at = _receipt_started_at(receipt)
            if started_at is None or started_at < since:
                ok = False
        if ok and until is not None:
            started_at = _receipt_started_at(receipt)
            if started_at is None or started_at > until:
                ok = False
        (attributed if ok else unattributed).append(receipt)
    return attributed, unattributed


def _budget_cap_status(total_usd: float) -> dict[str, Any]:
    """Report (never mutate) the global provider budget cap guardrail.

    Deferred import mirrors ``budget_enforcement._global_cap()`` — breaks a
    potential import cycle while keeping GLOBAL_PROVIDER_BUDGET_CAP_USD as
    the single source of truth (D-01). This function only reads the
    constant; it has no code path that writes/raises it (T-rz1-02).
    """

    from conference_pipeline.presentation.blueprint_generation import (
        GLOBAL_PROVIDER_BUDGET_CAP_USD,
    )

    cap = float(GLOBAL_PROVIDER_BUDGET_CAP_USD)
    return {
        "global_cap_usd": cap,
        "breach": total_usd > cap,
        "note": "Read-only guardrail; this command never raises or modifies the cap.",
    }


def compute_conference_total_cost(
    workspace: Path,
    *,
    attribute_segments: set[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace)
    attribute_segments = attribute_segments or set()
    since_dt = _parse_iso(since)
    until_dt = _parse_iso(until)

    canonical_receipts = scan_provider_call_receipts(workspace)
    repo_root_receipts = scan_provider_call_receipts(REPO_ROOT)

    attributed, unattributed = _attribute_misplaced_receipts(
        repo_root_receipts,
        attribute_segments=attribute_segments,
        since=since_dt,
        until=until_dt,
    )

    canonical = _bucket_breakdown(canonical_receipts)
    misplaced_attributed = _bucket_breakdown(attributed)
    misplaced_unattributed = _bucket_breakdown(unattributed)

    total_usd = round(canonical["total_usd"] + misplaced_attributed["total_usd"], 6)

    return {
        "workspace": str(workspace),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attribution_filters": {
            "attribute_segments": sorted(attribute_segments),
            "since": since,
            "until": until,
        },
        "total_usd": total_usd,
        "canonical": canonical,
        "misplaced_attributed": misplaced_attributed,
        "misplaced_unattributed": misplaced_unattributed,
        "budget_cap": _budget_cap_status(total_usd),
        "notes": [
            "misplaced_unattributed receipts are reported for visibility only and "
            "are NOT included in total_usd — they were not confidently attributed "
            "to this conference.",
            "This command never moves, deletes, or rewrites any receipt file.",
        ],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Total conference cost: {report['workspace']}",
        f"Generated: {report['generated_at']}",
        f"TOTAL (canonical + attributed misplaced): ${report['total_usd']:,.4f}",
        "",
        f"Canonical: ${report['canonical']['total_usd']:,.4f} "
        f"({report['canonical']['receipt_count']} receipts)",
        f"Misplaced (attributed): ${report['misplaced_attributed']['total_usd']:,.4f} "
        f"({report['misplaced_attributed']['receipt_count']} receipts)",
        f"Misplaced (unattributed, excluded from total): "
        f"${report['misplaced_unattributed']['total_usd']:,.4f} "
        f"({report['misplaced_unattributed']['receipt_count']} receipts)",
        "",
        "Budget cap: "
        f"${report['budget_cap']['global_cap_usd']:,.2f} "
        f"({'BREACHED' if report['budget_cap']['breach'] else 'within cap'})",
    ]
    if report["misplaced_unattributed"]["receipt_count"] and not (
        report["attribution_filters"]["attribute_segments"]
        or report["attribution_filters"]["since"]
        or report["attribution_filters"]["until"]
    ):
        lines.append("")
        lines.append(
            "No --attribute-segment/--since/--until given — all repo-root "
            "receipts are reported unattributed. Pass one or more of those "
            "flags to claim the subset that belongs to this conference."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path, help="Conference workspace root.")
    parser.add_argument(
        "--attribute-segment",
        action="append",
        default=[],
        dest="attribute_segments",
        metavar="SEGMENT",
        help=(
            "Repo-root provider-calls/<segment>/ directory name to attribute to this "
            "conference (repeatable), e.g. --attribute-segment run_icml2026_production."
        ),
    )
    parser.add_argument(
        "--since",
        help="Only attribute repo-root receipts with started_at >= this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--until",
        help="Only attribute repo-root receipts with started_at <= this ISO-8601 timestamp.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = compute_conference_total_cost(
        args.workspace,
        attribute_segments=set(args.attribute_segments),
        since=args.since,
        until=args.until,
    )

    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
