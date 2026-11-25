"""Which of a deck's figure findings still await the operator.

The adversarial figure critic stops after three rounds; what is open then is
decided by the operator through ``scripts/presentations/patch_figure_spec.py``
(accept, patch, replace), which writes ``operator_resolution`` into the
figure's critic sidecar. A deck is not accepted while any finding is
undecided (docs/operations/figure-fidelity-policy.md, item 9). This module is
the one place that question is answered, for the acceptance receipt and for
the runner alike.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEMANTIC_CODES = frozenset({
    "unsupported_number", "unsupported_label", "misattribution", "overclaim", "claim_not_encoded",
    "depiction_mismatch", "sequence_implies_causality", "alternatives_as_sequence", "incommensurable_axes",
})


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def undecided_figure_findings(deck_dir: Path) -> list[dict[str, Any]]:
    """Open major critic findings with no recorded operator outcome.

    Reads the spec receipt's rows and, for each, the critic sidecar beside the
    spec: the sidecar carries the outcome written after the receipt, so a
    resolution recorded since the last run still counts.
    """

    deck_dir = Path(deck_dir)
    receipt = _load(deck_dir / "figure_spec_receipt.json")
    open_findings: list[dict[str, Any]] = []
    for row in receipt.get("specs") or []:
        if not isinstance(row, dict):
            continue
        critique = row.get("critic") if isinstance(row.get("critic"), dict) else {}
        sidecar = _load(deck_dir / "figure_specs" / f"{row.get('figure_id')}.critic.json")
        if sidecar:
            critique = sidecar
        issues = [i for i in critique.get("unresolved_major") or [] if isinstance(i, dict)]
        if not issues or critique.get("operator_resolution"):
            continue
        open_findings.append({
            "slide_id": row.get("slide_id"),
            "figure_id": row.get("figure_id"),
            "semantic": any(i.get("code") in SEMANTIC_CODES for i in issues),
            "codes": [i.get("code") for i in issues],
        })
    return open_findings


def figure_findings_summary(deck_dir: Path) -> dict[str, Any]:
    """Counts an acceptance receipt records as evidence of the figure gate."""

    deck_dir = Path(deck_dir)
    receipt = _load(deck_dir / "figure_spec_receipt.json")
    rows = [r for r in receipt.get("specs") or [] if isinstance(r, dict)]
    resolved = 0
    for row in rows:
        sidecar = _load(deck_dir / "figure_specs" / f"{row.get('figure_id')}.critic.json")
        if sidecar.get("operator_resolution"):
            resolved += 1
    undecided = undecided_figure_findings(deck_dir)
    return {
        "figures": len(rows),
        "critic_rubric": (receipt.get("critic") or {}).get("rubric_version"),
        "operator_resolved": resolved,
        "undecided": len(undecided),
        "undecided_semantic": sum(1 for f in undecided if f["semantic"]),
        "undecided_slide_ids": [f["slide_id"] for f in undecided],
    }
