#!/usr/bin/env python3
"""Offline calibration analysis CLI for the v2.4 figure gate stack (CALIB-06).

Pure functions over receipt JSON — no provider imports, no scoring imports,
fully unit-testable offline with fixture receipts. Subcommands:

- ``assemble-dataset``: one labeled row per batch attempt plus the six pooled
  trial-v2 finals from the replay receipt (D-12 label_source rules, D-16
  final_render_failed markers, D-04 grounding-skip evidence, Pitfall 5
  null-tuple-fields-vs-unlabeled semantics, sha-chained inputs).
- ``build-review``: blind review manifest (NO gate-score keys anywhere in
  its tree, D-07), gate-reveal sidecar (separate file), and a blind HTML
  page; item list per D-08 (every final render plus the last gate-rejected
  attempt per redraw slide), deterministically shuffled by the batch
  receipt's sha (Pitfall 6).
- ``merge-verdicts``: enum-validated operator verdict capture (D-06) with
  exact one-verdict-per-manifest-item coverage.
- ``analyze``: added by plan 02-04 — the subparser registry below stays open.

Exit codes (shared vocabulary with run_figure_trial.py /
replay_figure_trial.py): 0 success, 1 missing input / contract failure,
2 validation error or refusal.

Usage:
  .venv/bin/python scripts/presentations/calibrate_figure_thresholds.py \
    assemble-dataset --batch-receipt <receipt> --replay-receipt <receipt> \
    [--verdicts <verdicts>] --out <dataset>
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.workflow_runs import write_json_atomic

PRODUCED_BY = "scripts/presentations/calibrate_figure_thresholds.py"
SCHEMA_VERSION = 1

# D-12 locked trial-v2 labels, keyed by the 5-char slide_id prefix. These six
# figures were adjudicated by the operator during trial v2 and are NEVER
# re-adjudicated in the blind review session (T-02-08) — the re-application
# done gate presumes these labels.
TRIAL_V2_LOCKED_LABELS = {
    "sl_05": "pass",
    "sl_06": "pass",
    "sl_10": "pass",
    "sl_11": "fail",
    "sl_19": "fail",
    "sl_23": "pass",
}
TRIAL_V2_LABEL_SOURCE = "trial_v2_operator"

# D-06 verdict shape: binary verdict plus failure-mode tags from a fixed
# vocabulary on fail. ROC uses the binary; Phase 03's failure-mode coverage
# table reads off the tags.
VERDICT_VALUES = ("pass", "fail")
FAILURE_MODE_TAGS = (
    "wrong-source",
    "missing-components",
    "invented-components",
    "legibility",
    "style-drift",
    "other",
)

# D-16: gate-exhausted slides keep their highest-coverage attempt as the
# final render with an explicit failed marker and this production note.
PRODUCTION_NOTE = "production behavior would be synthetic fallback for this slide"

# Label provenance vocabulary. ``human`` is the spot-check default (D-05..D-08);
# ``ai_panel`` records a blind multi-model AI-panel adjudication when the
# operator approves that protocol in lieu of a single human sitting — its
# per-verdict label_source is propagated verbatim so the dataset never
# misrepresents an AI-panel label as human ground truth.
LABEL_SOURCES = ("human", "ai_panel", "critique", TRIAL_V2_LABEL_SOURCE)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path | None) -> dict[str, Any] | None:
    """Sha-chained input reference ({path, sha256}); None for absent inputs."""

    if path is None:
        return None
    return {
        "path": str(path),
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _load_json(path: Path, label: str) -> dict[str, Any] | None:
    """Read a JSON object; print a contract error and return None on failure."""

    if not path.is_file():
        print(f"[calibrate] {label} not found: {path}", file=sys.stderr)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[calibrate] {label} is not valid JSON: {path} ({exc})", file=sys.stderr)
        return None
    if not isinstance(payload, (dict, list)):
        print(f"[calibrate] {label} must be a JSON object or list: {path}", file=sys.stderr)
        return None
    return payload


def derive_margin_sign(margin: Any) -> bool | None:
    """Row-level margin sign (CALIB-03 evidence, never a tunable threshold).

    Trial replay margins nest per-metric blocks ({"dinov2": {"margin", ...}});
    when present, the sign is derived as dinov2 margin > 0. Batch attempt
    margins are flat margin_sign_test output — the recorded margin_sign is
    copied through. Null when the margin was never computed (ungrounded
    tiers, degraded scoring).
    """

    if not isinstance(margin, dict):
        return None
    nested = margin.get("dinov2")
    if isinstance(nested, dict):
        value = nested.get("margin")
        if isinstance(value, (int, float)):
            return value > 0
        return nested.get("margin_sign")
    return margin.get("margin_sign")


def _final_attempt_number(outcome: dict[str, Any]) -> int | None:
    """Attempt number of the outcome's surviving final render.

    Matches the outcome's rendered_png against attempt records (redraw
    tiers); falls back to parsing the ``_attempt{n}.png`` suffix for tiers
    whose attempt records carry no rendered_png (d2, html_css).
    """

    final_png = outcome.get("rendered_png")
    if not final_png:
        return None
    for record in outcome.get("attempts", []) or []:
        if record.get("rendered_png") and record["rendered_png"] == final_png:
            return record.get("attempt")
    match = re.search(r"_attempt(\d+)\.png$", str(final_png))
    return int(match.group(1)) if match else None


def last_rejected_attempt(outcome: dict[str, Any]) -> dict[str, Any] | None:
    """D-08 selector: the last gate-rejected attempt that is not the final.

    Among rendered attempts with gate_decision.accept == False, take the
    highest attempt number that is not the surviving final render. Returns
    None for ungrounded tiers (their attempt records carry no rendered_png
    or gate_decision) and for slides whose every non-final attempt passed.
    """

    final_png = outcome.get("rendered_png")
    rejected = [
        a
        for a in outcome.get("attempts", []) or []
        if a.get("rendered_png")
        and not (a.get("gate_decision") or {}).get("accept")
        and a.get("rendered_png") != final_png
    ]
    return max(rejected, key=lambda a: a["attempt"]) if rejected else None


def locked_trial_label(slide_id: str) -> str:
    """Locked trial-v2 label for a replay figure, keyed by slide_id prefix."""

    prefix = str(slide_id)[:5]
    label = TRIAL_V2_LOCKED_LABELS.get(prefix)
    if label is None:
        raise ValueError(
            f"replay figure {slide_id!r} has no locked trial-v2 label "
            f"(known prefixes: {sorted(TRIAL_V2_LOCKED_LABELS)})"
        )
    return label


# ---------------------------------------------------------------------------
# assemble-dataset (Task 1): per-attempt rows + trial-v2 pooling
# ---------------------------------------------------------------------------


def _verdict_lookup(verdicts: dict[str, Any] | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (slide_id, attempt) -> verdict item from a merge-verdicts file.

    Review ids encode slide_id + kind + attempt, but the items also carry
    those fields explicitly — the join never parses review_id strings.
    Trial-v2 rows are never joinable: no review items exist for them by
    construction (T-02-08).
    """

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    if not verdicts:
        return lookup
    for item in verdicts.get("items", []) or []:
        slide_id = item.get("slide_id")
        attempt = item.get("attempt")
        if slide_id is None or attempt is None:
            continue
        lookup[(str(slide_id), int(attempt))] = item
    return lookup


def _resolve_label(
    *,
    critique_verdict: Any,
    human_verdict: str | None,
    verdict_label_source: str | None = None,
) -> tuple[str | None, str | None]:
    """D-12 label resolution for a batch row: (label, label_source).

    An adjudicated verdict overrides on every row that has one; remaining
    rows use the critique verdict (pass -> pass, any other recorded verdict
    -> fail). Rows with neither stay unlabeled (label null) — Pitfall 5's
    unlabeled_row_count counts exactly these.

    The adjudicated verdict's provenance is recorded HONESTLY: when the
    verdict item carries an explicit ``label_source`` (e.g. ``ai_panel`` for
    a blind AI-panel adjudication), that source is propagated verbatim;
    verdict items WITHOUT a ``label_source`` resolve to ``"human"`` for
    backward compatibility with the human-spot-check flow (D-05..D-08).
    """

    if human_verdict is not None:
        return human_verdict, (verdict_label_source or "human")
    if critique_verdict is None:
        return None, None
    return ("pass" if critique_verdict == "pass" else "fail"), "critique"


def build_attempt_rows(
    result: dict[str, Any],
    verdict_lookup: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dataset rows for one batch result entry (one row per attempt).

    Skipped slides contribute no attempt rows (their evidence lives in
    grounding_skips). The surviving final of a best_effort outcome carries
    the D-16 markers (final_render_failed=true + production_note).
    """

    outcome = result.get("outcome") or {}
    status = outcome.get("status")
    if status not in ("generated", "best_effort", "failed"):
        return []
    slide_id = str(result.get("slide_id"))
    tier = result.get("tier") or outcome.get("tier")
    grounding = outcome.get("grounding") or {}
    claim_match = grounding.get("claim_match")
    final_png = outcome.get("rendered_png")
    final_attempt = _final_attempt_number(outcome)
    rows: list[dict[str, Any]] = []
    for record in outcome.get("attempts", []) or []:
        attempt = record.get("attempt")
        if record.get("rendered_png") and final_png:
            is_final = record["rendered_png"] == final_png
        else:
            is_final = final_attempt is not None and attempt == final_attempt
        final_render_failed = bool(is_final and status == "best_effort")
        verdict_item = verdict_lookup.get((slide_id, attempt)) if attempt is not None else None
        human_verdict = verdict_item.get("verdict") if verdict_item else None
        # Honor a per-verdict label_source when the verdict carries one
        # (e.g. ``ai_panel``); absent it, the verdict resolves to ``human``.
        verdict_label_source = verdict_item.get("label_source") if verdict_item else None
        label, label_source = _resolve_label(
            critique_verdict=record.get("critique_verdict"),
            human_verdict=human_verdict,
            verdict_label_source=verdict_label_source,
        )
        rows.append(
            {
                "row_id": f"batch:{slide_id}:a{attempt}",
                "origin": "calibration_batch",
                "slide_id": slide_id,
                "tier": tier,
                "attempt": attempt,
                "is_final_render": bool(is_final),
                "final_render_failed": final_render_failed,
                "production_note": PRODUCTION_NOTE if final_render_failed else None,
                "coverage": record.get("coverage"),
                "dinov2_similarity": record.get("dinov2_similarity"),
                "clip_similarity": record.get("clip_similarity"),
                "margin": record.get("margin"),
                "margin_sign": derive_margin_sign(record.get("margin")),
                "critique_verdict": record.get("critique_verdict"),
                "claim_match": claim_match,
                "estimator_findings": record.get("estimator_findings"),
                "human_verdict": human_verdict,
                "failure_modes": list(verdict_item.get("failure_modes") or []) if verdict_item else [],
                "label": label,
                "label_source": label_source,
            }
        )
    return rows


def build_trial_rows(replay_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """The six pooled trial-v2 finals from the replay receipt (D-12 locked).

    Full tuples exist for FINALS only (trial v2's per-attempt records lack
    DINOv2/coverage), so exactly the figures[] entries pool into the ROC,
    with label_source "trial_v2_operator" and the locked labels.
    """

    rows: list[dict[str, Any]] = []
    for figure in replay_receipt.get("figures", []) or []:
        slide_id = str(figure.get("slide_id"))
        rows.append(
            {
                "row_id": f"trial:{slide_id}",
                "origin": "trial_v2_replay",
                "slide_id": slide_id,
                "tier": figure.get("tier"),
                "attempt": None,
                "is_final_render": True,
                "final_render_failed": False,
                "production_note": None,
                "coverage": figure.get("coverage"),
                "dinov2_similarity": figure.get("dinov2_similarity"),
                "clip_similarity": figure.get("clip_similarity"),
                "margin": figure.get("margin"),
                "margin_sign": derive_margin_sign(figure.get("margin")),
                "critique_verdict": (figure.get("critique") or {}).get("verdict"),
                "claim_match": None,
                "estimator_findings": figure.get("estimator_findings"),
                "human_verdict": None,
                "failure_modes": [],
                "label": locked_trial_label(slide_id),
                "label_source": TRIAL_V2_LABEL_SOURCE,
            }
        )
    return rows


def build_grounding_skips(batch_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """D-04 evidence rows for slides whose redraw tier was skipped at run
    time (claim_match below confidence / ranker rejected all candidates)."""

    skips: list[dict[str, Any]] = []
    for result in batch_receipt.get("results", []) or []:
        outcome = result.get("outcome") or {}
        if outcome.get("status") != "skipped":
            continue
        skip_info = outcome.get("grounding_skip") or {}
        skips.append(
            {
                "slide_id": result.get("slide_id"),
                "tier": result.get("tier") or outcome.get("tier"),
                "skip_reason_text": outcome.get("reason"),
                "reason": skip_info.get("reason"),
                "claim_match": skip_info.get("claim_match"),
                "claim_match_threshold": skip_info.get("claim_match_threshold"),
                "pooled_notes": skip_info.get("pooled_notes"),
                "pool_candidate_count": skip_info.get("pool_candidate_count"),
                "ranker_rationale": skip_info.get("ranker_rationale"),
                "ranker_receipt_path": skip_info.get("ranker_receipt_path"),
            }
        )
    return skips


def assemble_dataset(
    batch_receipt: dict[str, Any],
    replay_receipt: dict[str, Any],
    verdicts: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pure dataset assembly (RESEARCH Pattern 2 contract, minus file refs).

    The CLI wrapper attaches the sha-chained inputs block; everything here
    is a deterministic function of the three parsed payloads.
    """

    verdict_lookup = _verdict_lookup(verdicts)
    rows: list[dict[str, Any]] = []
    for result in batch_receipt.get("results", []) or []:
        rows.extend(build_attempt_rows(result, verdict_lookup))
    rows.extend(build_trial_rows(replay_receipt))
    return {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": _utc_now(),
        "dinov2": batch_receipt.get("dinov2"),
        "coverage_threshold": batch_receipt.get("coverage_threshold"),
        "rows": rows,
        "grounding_skips": build_grounding_skips(batch_receipt),
        "unlabeled_row_count": sum(1 for row in rows if row["label"] is None),
    }


def _cmd_assemble_dataset(args: argparse.Namespace) -> int:
    batch_receipt = _load_json(args.batch_receipt, "batch receipt")
    if batch_receipt is None:
        return 1
    replay_receipt = _load_json(args.replay_receipt, "replay receipt")
    if replay_receipt is None:
        return 1
    verdicts: dict[str, Any] | None = None
    if args.verdicts is not None:
        verdicts = _load_json(args.verdicts, "verdicts file")
        if verdicts is None:
            return 1
    try:
        dataset = assemble_dataset(batch_receipt, replay_receipt, verdicts)
    except ValueError as exc:
        print(f"[calibrate] dataset contract failure: {exc}", file=sys.stderr)
        return 1
    dataset["inputs"] = {
        "batch_receipt": _file_ref(args.batch_receipt),
        "replay_receipt": _file_ref(args.replay_receipt),
        "verdicts": _file_ref(args.verdicts),
    }
    write_json_atomic(args.out, dataset)
    print(
        f"[calibrate] dataset: {len(dataset['rows'])} rows "
        f"({dataset['unlabeled_row_count']} unlabeled), "
        f"{len(dataset['grounding_skips'])} grounding skips -> {args.out}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# build-review (Task 2): blind manifest + gate-reveal sidecar + blind HTML
# (D-05..D-08). The manifest carries NO gate-score keys anywhere in its tree
# — scores live only in the separate gate_reveal sidecar (T-02-05).
# ---------------------------------------------------------------------------


def claims_by_slide(deck_contract: dict[str, Any]) -> dict[str, str]:
    """slide_id -> claim text from presentation_deck.json (claims[] joined;
    title fallback for slides without recorded claims)."""

    claims: dict[str, str] = {}
    for slide in deck_contract.get("slides", []) or []:
        slide_id = slide.get("slide_id")
        if not slide_id:
            continue
        slide_claims = [str(claim) for claim in (slide.get("claims") or []) if claim]
        claims[str(slide_id)] = (
            " ".join(slide_claims) if slide_claims else str(slide.get("title") or "")
        )
    return claims


def build_review_items(
    batch_receipt: dict[str, Any], claims: dict[str, str]
) -> list[dict[str, Any]]:
    """D-08 review item list, blind fields only.

    Every final render (generated AND best_effort — D-16 failed finals get a
    human verdict as negative-class datapoints) plus the last gate-rejected
    attempt per redraw slide. D-04 skipped slides have no render and
    contribute NO review item (their skip evidence is recapped in the
    gate-reveal sidecar). Attempt numbers ride in render paths — a minor,
    disclosed leak; renames would break receipt traceability (Pitfall 6).
    """

    items: list[dict[str, Any]] = []
    for result in batch_receipt.get("results", []) or []:
        outcome = result.get("outcome") or {}
        if outcome.get("status") not in ("generated", "best_effort"):
            continue
        slide_id = str(result.get("slide_id"))
        claim = claims.get(slide_id, "")
        crop_path = (outcome.get("grounding") or {}).get("crop_path")
        final_attempt = _final_attempt_number(outcome)
        items.append(
            {
                "review_id": f"rev_{slide_id}_final_{(final_attempt or 0):02d}",
                "slide_id": slide_id,
                "attempt": final_attempt,
                "kind": "final",
                "render_path": outcome.get("rendered_png"),
                "crop_path": crop_path,
                "claim": claim,
            }
        )
        rejected = last_rejected_attempt(outcome)
        if rejected is not None:
            items.append(
                {
                    "review_id": f"rev_{slide_id}_last_rejected_{rejected['attempt']:02d}",
                    "slide_id": slide_id,
                    "attempt": rejected.get("attempt"),
                    "kind": "last_rejected",
                    "render_path": rejected.get("rendered_png"),
                    "crop_path": crop_path,
                    "claim": claim,
                }
            )
    return items


def deterministic_shuffle(
    items: list[dict[str, Any]], receipt_bytes: bytes
) -> list[dict[str, Any]]:
    """Shuffled-but-deterministic review order, seeded by the batch
    receipt's raw bytes (Pitfall 6: a sequential order mirroring the receipt
    would let the operator infer gate outcomes)."""

    seed = int(hashlib.sha256(receipt_bytes).hexdigest(), 16)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _reveal_tuple_from_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    status = outcome.get("status")
    return {
        "coverage": outcome.get("coverage"),
        "dinov2_similarity": outcome.get("dinov2_similarity"),
        "clip_similarity": outcome.get("clip_similarity"),
        "margin": outcome.get("margin"),
        "margin_sign": derive_margin_sign(outcome.get("margin")),
        "critique_verdict": (outcome.get("critique") or {}).get("verdict"),
        "critique_feedback": (outcome.get("critique") or {}).get("feedback"),
        "claim_match": (outcome.get("grounding") or {}).get("claim_match"),
        "estimator_findings": outcome.get("estimator_findings"),
        "gate_decision": outcome.get("gate_decision"),
        "status": status,
        "final_render_failed": status == "best_effort",
    }


def _reveal_tuple_from_attempt(
    record: dict[str, Any], grounding: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "coverage": record.get("coverage"),
        "dinov2_similarity": record.get("dinov2_similarity"),
        "clip_similarity": record.get("clip_similarity"),
        "margin": record.get("margin"),
        "margin_sign": derive_margin_sign(record.get("margin")),
        "critique_verdict": record.get("critique_verdict"),
        "critique_feedback": record.get("critique_feedback"),
        "claim_match": (grounding or {}).get("claim_match"),
        "estimator_findings": record.get("estimator_findings"),
        "gate_decision": record.get("gate_decision"),
        "status": None,
        "final_render_failed": False,
    }


def build_gate_reveal(
    batch_receipt: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """review_id -> full gate tuple, opened only AFTER each verdict (D-07)."""

    outcomes = {
        str(result.get("slide_id")): (result.get("outcome") or {})
        for result in batch_receipt.get("results", []) or []
    }
    reveal: dict[str, dict[str, Any]] = {}
    for item in items:
        outcome = outcomes.get(item["slide_id"], {})
        if item["kind"] == "final":
            reveal[item["review_id"]] = _reveal_tuple_from_outcome(outcome)
        else:
            record = next(
                (
                    a
                    for a in outcome.get("attempts", []) or []
                    if a.get("attempt") == item["attempt"]
                ),
                {},
            )
            reveal[item["review_id"]] = _reveal_tuple_from_attempt(
                record, outcome.get("grounding")
            )
    return reveal


def _deck_relative(path_value: Any, deck_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        return str(path.relative_to(deck_dir))
    except ValueError:
        return str(path)


def blind_review_page(items: list[dict[str, Any]], deck_dir: Path) -> str:
    """Self-contained blind review HTML: render + source crop + claim ONLY.

    Mirrors run_figure_trial._comparison_page's section/pair structure with
    the score-bearing meta line REPLACED by claim-only text (D-07). The CSS
    deliberately avoids every banned vocabulary word (no box-offset
    shorthand whose property name collides with the margin key).
    """

    sections = []
    for item in items:
        render_rel = _deck_relative(item.get("render_path"), deck_dir) or ""
        crop_rel = _deck_relative(item.get("crop_path"), deck_dir)
        crop_html = (
            f'<img src="{html_module.escape(crop_rel)}"/>'
            if crop_rel
            else "<p>claim-only item (no grounded source for this tier)</p>"
        )
        sections.append(
            f"""
  <section class="slide">
    <h2>{html_module.escape(item["review_id"])}</h2>
    <p class="claim">{html_module.escape(str(item.get("claim") or ""))}</p>
    <div class="pair">
      <figure><figcaption>Render</figcaption><img src="{html_module.escape(render_rel)}"/></figure>
      <figure><figcaption>Source</figcaption>{crop_html}</figure>
    </div>
  </section>"""
        )
    return (
        """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Calibration blind review</title>
<style>
 body { background:#0f0f1d; color:#eee; font-family: Helvetica, Arial, sans-serif; padding: 24px; }
 h1 { color:#a8d8ea; } h2 { color:#ffd43b; font-size:18px; }
 .claim { color:#aaa; font-size:14px; }
 .pair { display:flex; gap:16px; }
 figure { flex:1; padding:0; } figcaption { color:#a8d8ea; font-size:13px; padding-bottom:4px; }
 img { width:100%; border:1px solid #333355; background:#1a1a2e; }
 section { padding-bottom:18px; border-bottom:1px solid #333355; }
</style></head><body>
<h1>Calibration blind review</h1>
<p class="claim">Record a verdict for each item BEFORE opening the gate-reveal sidecar.</p>
"""
        + "\n".join(sections)
        + "\n</body></html>\n"
    )


def _cmd_build_review(args: argparse.Namespace) -> int:
    if args.out_manifest.resolve() == args.out_reveal.resolve():
        print(
            "[calibrate] refusing: the gate-reveal sidecar must be a separate "
            "file from the blind manifest (D-07)",
            file=sys.stderr,
        )
        return 2
    if not args.batch_receipt.is_file():
        print(f"[calibrate] batch receipt not found: {args.batch_receipt}", file=sys.stderr)
        return 1
    receipt_bytes = args.batch_receipt.read_bytes()
    try:
        batch_receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        print(
            f"[calibrate] batch receipt is not valid JSON: {args.batch_receipt} ({exc})",
            file=sys.stderr,
        )
        return 1
    deck_dir = args.deck_dir.resolve()
    deck_contract = _load_json(deck_dir / "presentation_deck.json", "deck contract")
    if deck_contract is None:
        return 1
    items = deterministic_shuffle(
        build_review_items(batch_receipt, claims_by_slide(deck_contract)),
        receipt_bytes,
    )
    receipt_ref = _file_ref(args.batch_receipt)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": _utc_now(),
        "inputs": {"batch_receipt": receipt_ref},
        "items": items,
    }
    reveal = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": _utc_now(),
        "inputs": {"batch_receipt": receipt_ref},
        "gate_reveal": build_gate_reveal(batch_receipt, items),
        "grounding_skips": build_grounding_skips(batch_receipt),
    }
    write_json_atomic(args.out_manifest, manifest)
    write_json_atomic(args.out_reveal, reveal)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(blind_review_page(items, deck_dir), encoding="utf-8")
    print(
        f"[calibrate] review: {len(items)} blind items -> {args.out_manifest}; "
        f"gate reveal -> {args.out_reveal}; page -> {args.out_html}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# merge-verdicts (Task 3): enum-validated operator verdict capture (D-06).
# Operator verdict strings are untrusted input (T-02-06 / ASVS V5): verdict
# enum, fixed tag vocabulary, fail->tags non-empty, pass->tags empty, and an
# exact one-verdict-per-manifest-item bijection — exit 2 on any violation.
# ---------------------------------------------------------------------------


def validate_draft_verdicts(
    draft: Any, manifest_items: list[dict[str, Any]]
) -> list[str]:
    """Every validation error in the raw in-session draft, by review_id."""

    if not isinstance(draft, list):
        return ["draft verdicts must be a JSON list of {review_id, verdict, failure_modes, notes?}"]
    errors: list[str] = []
    manifest_ids = [str(item.get("review_id")) for item in manifest_items]
    manifest_id_set = set(manifest_ids)
    seen: set[str] = set()
    for index, entry in enumerate(draft):
        if not isinstance(entry, dict):
            errors.append(f"draft[{index}]: entry is not an object")
            continue
        review_id = entry.get("review_id")
        label = str(review_id) if review_id else f"draft[{index}]"
        if not review_id:
            errors.append(f"draft[{index}]: missing review_id")
        elif review_id in seen:
            errors.append(f"{review_id}: duplicate verdict entry")
        else:
            seen.add(str(review_id))
            if review_id not in manifest_id_set:
                errors.append(f"{review_id}: not a review item in the manifest")
        verdict = entry.get("verdict")
        if verdict not in VERDICT_VALUES:
            errors.append(
                f"{label}: verdict {verdict!r} is not in {{{', '.join(VERDICT_VALUES)}}}"
            )
        tags = entry.get("failure_modes")
        if not isinstance(tags, list):
            errors.append(f"{label}: failure_modes must be a list")
            tags = []
        unknown = [tag for tag in tags if tag not in FAILURE_MODE_TAGS]
        if unknown:
            errors.append(
                f"{label}: unknown failure_modes {unknown}; "
                f"allowed: {', '.join(FAILURE_MODE_TAGS)}"
            )
        if verdict == "fail" and not tags:
            errors.append(f"{label}: verdict=fail requires at least one failure-mode tag")
        if verdict == "pass" and tags:
            errors.append(f"{label}: verdict=pass must carry no failure-mode tags")
    for review_id in manifest_ids:
        if review_id not in seen:
            errors.append(f"{review_id}: manifest item has no verdict in the draft")
    return errors


def merge_verdicts(
    manifest: dict[str, Any], draft: list[dict[str, Any]], *, reviewer: str
) -> dict[str, Any]:
    """Validated verdicts payload, in manifest (session) item order.

    Items carry slide_id/kind/attempt copied from the manifest item — the
    assemble-dataset join is deterministic on (slide_id, attempt) and never
    parses review_id strings.
    """

    by_id = {entry["review_id"]: entry for entry in draft}
    items: list[dict[str, Any]] = []
    for manifest_item in manifest.get("items", []) or []:
        entry = by_id[manifest_item["review_id"]]
        item: dict[str, Any] = {
            "review_id": manifest_item["review_id"],
            "slide_id": manifest_item["slide_id"],
            "kind": manifest_item["kind"],
            "attempt": manifest_item["attempt"],
            "verdict": entry["verdict"],
            "failure_modes": [str(tag) for tag in entry.get("failure_modes") or []],
        }
        if entry.get("notes"):
            item["notes"] = str(entry["notes"])
        items.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "reviewer": reviewer,
        "recorded_at": _utc_now(),
        "items": items,
    }


def _cmd_merge_verdicts(args: argparse.Namespace) -> int:
    manifest = _load_json(args.manifest, "review manifest")
    if manifest is None:
        return 1
    draft = _load_json(args.draft, "draft verdicts")
    if draft is None:
        return 1
    errors = validate_draft_verdicts(draft, manifest.get("items", []) or [])
    if errors:
        for error in errors:
            print(f"[calibrate] verdict validation: {error}", file=sys.stderr)
        print(
            f"[calibrate] refusing to merge: {len(errors)} validation error(s)",
            file=sys.stderr,
        )
        return 2
    payload = merge_verdicts(manifest, draft, reviewer=args.reviewer)
    payload["inputs"] = {
        "manifest": _file_ref(args.manifest),
        "draft": _file_ref(args.draft),
    }
    write_json_atomic(args.out, payload)
    print(
        f"[calibrate] verdicts: {len(payload['items'])} recorded -> {args.out}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# analyze (plan 02-04): coverage sweep (D-09), degeneracy triggers (D-10),
# sensitivity populations (D-12), stack-level floor selection (Pattern 5),
# candidate CLIP floor (D-11), rank-AUC separation, re-application check.
#
# Every function below is pure over parsed rows/tuples — no provider, no
# scoring, no file I/O. 'fail' is the positive (must-block) class everywhere;
# degenerate results NEVER carry an adoptable threshold (no 0.8 fallback).
# ---------------------------------------------------------------------------

# D-10 trigger (c) tolerance: the zero-false-pass operating point's false-fail
# rate (false_fail.count / labeled-pass count) above which the data is treated
# as degenerate. Recorded in every result and presented — never silently
# applied as an accept/reject decision.
HALT_TOLERANCE_FALSE_FAIL = 0.25

# D-10 trigger (b): a sweep over fewer than this many labeled fails cannot
# support a defensible zero-false-pass operating point.
MIN_LABELED_FAILS = 3


def _coverage_classes(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Sorted (fails, passes) coverages over rows with non-null coverage."""

    fails = sorted(
        r["coverage"]
        for r in rows
        if r.get("label") == "fail" and r.get("coverage") is not None
    )
    passes = sorted(
        r["coverage"]
        for r in rows
        if r.get("label") == "pass" and r.get("coverage") is not None
    )
    return fails, passes


def _sweep_curve(fails: list[float], passes: list[float]) -> list[dict[str, Any]]:
    """Full curve-as-data: candidate thresholds at every fail/pass midpoint
    plus boundary points, each with per-point false_pass/false_fail counts.

    Candidates are the midpoints between consecutive sorted distinct
    coverage values across BOTH classes, plus a below-min and above-max
    boundary, so D-10's halt-and-surface presents real operating points.
    """

    values = sorted(set(fails) | set(passes))
    if not values:
        return []
    candidates: list[float] = [values[0] - 0.05]
    for lo, hi in zip(values, values[1:]):
        candidates.append((lo + hi) / 2)
    candidates.append(values[-1] + 0.05)
    curve: list[dict[str, Any]] = []
    for t in candidates:
        # A render passes the coverage gate iff coverage >= t. False-pass =
        # a labeled-fail render that passes; false-fail = a labeled-pass
        # render that is blocked.
        false_pass = sum(1 for c in fails if c >= t)
        false_fail = sum(1 for c in passes if c < t)
        curve.append(
            {
                "threshold": t,
                "false_pass": {"count": false_pass, "of": len(fails)},
                "false_fail": {"count": false_fail, "of": len(passes)},
            }
        )
    return curve


def select_coverage_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """D-09 closed-form coverage-threshold sweep, owning ONLY D-10 trigger (a).

    'fail' is the positive (must-block) class. Among thresholds passing NO
    labeled-fail render, the false-fail-minimizing point is the SMALLEST
    threshold strictly above every labeled-fail coverage (false-fails are
    monotone nondecreasing in t). Placement is the midpoint between the
    highest labeled-fail coverage and the lowest passing coverage above it.

    Trigger (a) (a missing label class, or no zero-false-pass threshold
    exists) lives here because the closed form cannot emit a point without
    it. Triggers (b)/(c) live in apply_degeneracy_triggers. Degenerate
    results carry status "degenerate", a machine-readable reason, the full
    curve, candidate operating points, and threshold=None — never a
    fallback value.
    """

    fails, passes = _coverage_classes(rows)
    curve = _sweep_curve(fails, passes)
    if not fails or not passes:
        return {
            "status": "degenerate",
            "reason": "missing a label class",
            "threshold": None,
            "false_pass": {"count": 0, "of": len(fails)},
            "false_fail": {"count": 0, "of": len(passes)},
            "labeled_fail_count": len(fails),
            "labeled_pass_count": len(passes),
            "halt_tolerance": HALT_TOLERANCE_FALSE_FAIL,
            "curve": curve,
            "candidates": curve,
        }
    max_fail = max(fails)
    above = [c for c in passes if c > max_fail]
    if not above:
        return {
            "status": "degenerate",
            "reason": "no zero-false-pass threshold exists",
            "threshold": None,
            "false_pass": {"count": 0, "of": len(fails)},
            "false_fail": {"count": 0, "of": len(passes)},
            "labeled_fail_count": len(fails),
            "labeled_pass_count": len(passes),
            "halt_tolerance": HALT_TOLERANCE_FALSE_FAIL,
            "curve": curve,
            "candidates": curve,
        }
    threshold = (max_fail + min(above)) / 2  # deterministic midpoint
    false_fail = sum(1 for c in passes if c < threshold)
    return {
        "status": "selected",
        "reason": None,
        "threshold": threshold,
        "false_pass": {"count": 0, "of": len(fails)},
        "false_fail": {"count": false_fail, "of": len(passes)},
        "labeled_fail_count": len(fails),
        "labeled_pass_count": len(passes),
        "halt_tolerance": HALT_TOLERANCE_FALSE_FAIL,
        "curve": curve,
        "candidates": curve,
    }


def apply_degeneracy_triggers(
    sweep_result: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """D-10 triggers (b) and (c) — the second layer of the frozen split.

    Applied per population by the analyze wrapper. A "selected" sweep is
    downgraded to "degenerate" when EITHER too few labeled fails exist
    (trigger b: count < MIN_LABELED_FAILS) OR the zero-false-pass point's
    false-fail rate exceeds HALT_TOLERANCE_FALSE_FAIL (trigger c). The
    false-fail rate is false_fail.count / labeled-pass count (the sweep
    result's false_fail.of denominator). A downgrade nulls the threshold and
    demotes the closed-form point into candidates; an already-degenerate
    result (trigger a) passes through with the tolerance echoed.
    """

    result = dict(sweep_result)
    result.setdefault("halt_tolerance", HALT_TOLERANCE_FALSE_FAIL)
    if result.get("status") != "selected":
        # Trigger (a) already fired in the closed form; nothing to add but
        # the echoed tolerance (set above).
        return result

    fail_count = result.get("labeled_fail_count")
    if fail_count is None:
        fails, _ = _coverage_classes(rows)
        fail_count = len(fails)
    pass_total = result["false_fail"]["of"]
    false_fail_count = result["false_fail"]["count"]
    false_fail_rate = (false_fail_count / pass_total) if pass_total else 0.0

    reasons: list[str] = []
    if fail_count < MIN_LABELED_FAILS:
        reasons.append(
            f"only {fail_count} labeled-fail rows (< {MIN_LABELED_FAILS} required)"
        )
    if false_fail_rate > HALT_TOLERANCE_FALSE_FAIL:
        reasons.append(
            f"false-fail rate {false_fail_rate:.3f} exceeds halt tolerance "
            f"{HALT_TOLERANCE_FALSE_FAIL}"
        )

    if not reasons:
        result["false_fail_rate"] = false_fail_rate
        return result

    # Downgrade: the closed-form point becomes a candidate, never an
    # adoptable threshold.
    closed_form_point = {
        "threshold": sweep_result.get("threshold"),
        "false_pass": dict(sweep_result.get("false_pass") or {}),
        "false_fail": dict(sweep_result.get("false_fail") or {}),
        "demoted_from": "closed_form_selection",
    }
    candidates = list(result.get("candidates") or [])
    if not any(
        c.get("threshold") == closed_form_point["threshold"] for c in candidates
    ):
        candidates = [closed_form_point, *candidates]
    result.update(
        {
            "status": "degenerate",
            "reason": "; ".join(reasons),
            "threshold": None,
            "false_fail_rate": false_fail_rate,
            "candidates": candidates,
        }
    )
    return result


# D-12 sensitivity populations (Open Question 1, settled by 02-04; population
# membership adapted in 02-06 for the operator-approved AI-panel labeling
# methodology): the P1 headline is the human-equivalent ADJUDICATED population.
# In 02-05 the operator substituted a blind multi-model AI panel for the single
# human sitting (D-05..D-08), so ``ai_panel`` labels are human-equivalent
# adjudicated verdicts and belong in P1 alongside ``human`` and
# ``trial_v2_operator``. This changes ONLY the headline population membership;
# no degeneracy trigger, floor-selection, or tie-breaking logic is touched.
# The thresholds calibrated against this population are AI-panel-grounded, NOT
# human ground truth — Phase 03's graduation contract must cite that. The other
# two populations (P2 all-labeled incl. critique, P3 finals-only) surface
# machine-label and attempt-level sensitivity without driving the halt.
_HEADLINE_LABEL_SOURCES = ("human", "ai_panel", TRIAL_V2_LABEL_SOURCE)


def _population_rows(
    rows: list[dict[str, Any]], predicate
) -> list[dict[str, Any]]:
    return [
        r for r in rows if r.get("coverage") is not None and predicate(r)
    ]


def build_populations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The three frozen D-12 populations, each with its definition and rows.

    P1 headline: label_source in {human, ai_panel, trial_v2_operator} and
    coverage non-null (the human-equivalent adjudicated operating population —
    Open Question 1; ai_panel added in 02-06 per the operator-approved
    AI-panel labeling methodology, AI-panel-grounded not human ground truth).
    P2 all-labeled: every labeled row incl. critique-labeled, coverage
    non-null. P3 finals-only: is_final_render true, both origins, coverage
    non-null. Rows must already carry a non-null ``label`` to count.
    """

    def labeled(r: dict[str, Any]) -> bool:
        return r.get("label") is not None

    p1 = _population_rows(
        rows,
        lambda r: labeled(r) and r.get("label_source") in _HEADLINE_LABEL_SOURCES,
    )
    p2 = _population_rows(rows, labeled)
    p3 = _population_rows(
        rows, lambda r: labeled(r) and bool(r.get("is_final_render"))
    )
    return [
        {
            "name": "P1_headline",
            "definition": (
                "label_source in {human, ai_panel, trial_v2_operator} and "
                "coverage non-null (human-equivalent adjudicated headline "
                "operating population; halt authority; AI-panel-grounded)"
            ),
            "rows": p1,
            "row_count": len(p1),
        },
        {
            "name": "P2_all_labeled",
            "definition": (
                "all labeled rows incl. critique-labeled, coverage non-null "
                "(machine-label sensitivity)"
            ),
            "rows": p2,
            "row_count": len(p2),
        },
        {
            "name": "P3_finals_only",
            "definition": (
                "is_final_render true, both origins, coverage non-null "
                "(finals-vs-attempts sensitivity)"
            ),
            "rows": p3,
            "row_count": len(p3),
        },
    ]


# ---------------------------------------------------------------------------
# Stack-level floor selection (Pattern 5), per-metric degeneracy evidence,
# rank-AUC separation record (Task 2). The deliverable floor is DINOv2-only;
# CLIP runs solely to produce the computed-not-adopted candidate (D-11).
# ---------------------------------------------------------------------------


def _metric_rows(
    rows: list[dict[str, Any]], metric_key: str
) -> tuple[list[dict[str, Any]], int]:
    """Rows with a non-null metric value, plus the excluded-row count."""

    kept = [r for r in rows if r.get(metric_key) is not None]
    return kept, len(rows) - len(kept)


def _stack_fail(
    row: dict[str, Any], t_cov: float, floor: float, metric_key: str
) -> tuple[bool, list[str]]:
    """Conjunctive gate-stack classification + which gate(s) fire.

    fail = (coverage < t_cov) OR (metric < floor) OR (margin_sign is False)
    OR (critique fail). Returns (is_fail, firing_gates). Pitfall 7: classify
    from the recorded tuple, never from any status field.
    """

    gates: list[str] = []
    coverage = row.get("coverage")
    if coverage is None or coverage < t_cov:
        gates.append("coverage")
    metric = row.get(metric_key)
    if metric is not None and metric < floor:
        gates.append("floor")
    if row.get("margin_sign") is False:
        gates.append("margin")
    if row.get("critique_verdict") not in (None, "pass"):
        gates.append("critique")
    return (bool(gates), gates)


def _per_metric_sweep(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, Any]:
    """The NAIVE per-metric zero-false-pass sweep (evidence only).

    Reuses the coverage closed form against the metric values: a metric floor
    that passes NO labeled-fail render. On the pooled trial labels this is
    degenerate by construction for DINOv2 (sl_19 0.7456 > sl_23 0.6913) —
    recorded, never a blocker.
    """

    proxy_rows = [
        {"coverage": r.get(metric_key), "label": r.get("label")}
        for r in rows
        if r.get(metric_key) is not None
    ]
    sweep = apply_degeneracy_triggers(
        select_coverage_threshold(proxy_rows), proxy_rows
    )
    return {
        "floor": sweep.get("threshold"),
        "status": sweep["status"],
        "reason": sweep.get("reason"),
        "false_fail": sweep.get("false_fail"),
    }


def _candidate_floors(values: list[float]) -> list[float]:
    """Floor candidates: midpoints between sorted distinct metric values plus
    a below-min baseline (floor below everything = floor gate never fires)."""

    distinct = sorted(set(values))
    if not distinct:
        return []
    candidates = [distinct[0] - 0.05]
    for lo, hi in zip(distinct, distinct[1:]):
        candidates.append((lo + hi) / 2)
    return candidates


def select_stack_floor(
    rows: list[dict[str, Any]], t_cov: float, metric_key: str
) -> dict[str, Any]:
    """Stack-level floor selection (Pattern 5) — the deliverable path.

    Sweeps candidate floors over the metric; a floor is FEASIBLE when, with
    the full conjunctive gate stack, no labeled-fail render passes all gates
    (stack-level zero false-pass) AND the stack-level false-fail count is
    minimal. Picks the LARGEST feasible floor in the minimal-false-fail set
    (the strongest sanity floor that does not over-block), and records the
    feasible interval, per-row gate attribution, and the naive per-metric
    sweep as evidence. The per-metric sweep's degeneracy never makes this
    fail (Pitfall 1).
    """

    kept, excluded = _metric_rows(rows, metric_key)
    metric_values = [r[metric_key] for r in kept]
    candidates = _candidate_floors(metric_values)
    per_metric = {"dinov2" if metric_key.startswith("dinov2") else "clip": _per_metric_sweep(rows, metric_key)}

    fails = [r for r in kept if r.get("label") == "fail"]
    passes = [r for r in kept if r.get("label") == "pass"]

    feasible: list[tuple[float, int]] = []  # (floor, false_fail_count)
    for floor in candidates:
        false_pass = sum(1 for r in fails if not _stack_fail(r, t_cov, floor, metric_key)[0])
        if false_pass:
            continue
        false_fail = sum(1 for r in passes if _stack_fail(r, t_cov, floor, metric_key)[0])
        feasible.append((floor, false_fail))

    base = {
        "metric_key": metric_key,
        "t_cov": t_cov,
        "per_metric": per_metric,
        "excluded_row_count": excluded,
        "labeled_fail_count": len(fails),
        "labeled_pass_count": len(passes),
        "candidates": candidates,
    }

    if not feasible:
        base.update(
            {
                "status": "degenerate",
                "reason": "no floor achieves stack-level zero false-pass",
                "floor": None,
                "feasible_interval": None,
                "gate_attribution": [],
            }
        )
        return base

    min_false_fail = min(ff for _, ff in feasible)
    minimal = [floor for floor, ff in feasible if ff == min_false_fail]
    chosen = max(minimal)
    feasible_interval = [min(minimal), max(minimal)]

    attribution = []
    for r in fails:
        is_fail, gates = _stack_fail(r, t_cov, chosen, metric_key)
        attribution.append(
            {"slide_id": r.get("slide_id"), "is_fail": is_fail, "gates": gates}
        )

    base.update(
        {
            "status": "selected",
            "reason": None,
            "floor": chosen,
            "feasible_interval": feasible_interval,
            "false_fail_count": min_false_fail,
            "gate_attribution": attribution,
        }
    )
    return base


def clip_candidate_floor(
    rows: list[dict[str, Any]], t_cov: float
) -> dict[str, Any]:
    """D-11: the CLIP floor computed from the same labels but NOT adopted.

    Runs the identical stack-level procedure with metric_key="clip_similarity"
    and stamps adopted=false so it can never be merged into the deliverable
    DINOv2 floor — it is input to Phase 03's keep-or-retire decision only.
    """

    result = select_stack_floor(rows, t_cov, "clip_similarity")
    result["adopted"] = False
    return result


def rank_auc(
    pass_scores: list[float], fail_scores: list[float]
) -> float | None:
    """P(random pass-render scores higher than a random fail-render); ties 0.5.

    Equivalent to the Mann-Whitney U / (n_pos * n_neg) — cross-checkable with
    scipy.stats.mannwhitneyu. Returns None when either class is empty.
    """

    if not pass_scores or not fail_scores:
        return None
    wins = sum(
        (p > f) + 0.5 * (p == f) for p in pass_scores for f in fail_scores
    )
    return wins / (len(pass_scores) * len(fail_scores))


def _interval_overlap(
    pass_scores: list[float], fail_scores: list[float]
) -> list[float] | None:
    """The [lo, hi] overlap of the two score ranges, or None when disjoint."""

    lo = max(min(pass_scores), min(fail_scores))
    hi = min(max(pass_scores), max(fail_scores))
    return [lo, hi] if lo <= hi else None


def _best_single_threshold_accuracy(
    pass_scores: list[float], fail_scores: list[float]
) -> float:
    """Best accuracy of any single >= threshold rule (pass iff score >= t)."""

    total = len(pass_scores) + len(fail_scores)
    if total == 0:
        return 0.0
    thresholds = sorted(set(pass_scores) | set(fail_scores))
    candidates = [thresholds[0] - 0.05]
    for lo, hi in zip(thresholds, thresholds[1:]):
        candidates.append((lo + hi) / 2)
    candidates.append(thresholds[-1] + 0.05)
    best = 0.0
    for t in candidates:
        correct = sum(1 for s in pass_scores if s >= t) + sum(
            1 for s in fail_scores if s < t
        )
        best = max(best, correct / total)
    return best


def _separation_block(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, Any]:
    pass_scores = [
        r[metric_key]
        for r in rows
        if r.get("label") == "pass" and r.get(metric_key) is not None
    ]
    fail_scores = [
        r[metric_key]
        for r in rows
        if r.get("label") == "fail" and r.get(metric_key) is not None
    ]
    block: dict[str, Any] = {
        "metric_key": metric_key,
        "auc": rank_auc(pass_scores, fail_scores),
        "n_pass": len(pass_scores),
        "n_fail": len(fail_scores),
        "score_ranges": {
            "pass": [min(pass_scores), max(pass_scores)] if pass_scores else None,
            "fail": [min(fail_scores), max(fail_scores)] if fail_scores else None,
        },
        "overlap_interval": (
            _interval_overlap(pass_scores, fail_scores)
            if pass_scores and fail_scores
            else None
        ),
        "best_single_threshold_accuracy": (
            _best_single_threshold_accuracy(pass_scores, fail_scores)
            if pass_scores and fail_scores
            else None
        ),
    }
    return block


def separation_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """DINOv2-vs-CLIP separation on pooled labels (D-11 input to Phase 03).

    Computes the rank-AUC and range-overlap record per metric over rows where
    the metric is present. Pooled on ALL labeled rows (Pitfall 2 sensitivity
    is the population builder's job; this record is the raw separation).
    """

    return {
        "dinov2": _separation_block(rows, "dinov2_similarity"),
        "clip": _separation_block(rows, "clip_similarity"),
    }


# ---------------------------------------------------------------------------
# Re-application check (Pattern 6) + analyze assembly (Task 3). The
# re-application classifications are RECOMPUTED from the recorded replay
# tuples; the analysis never reads any status/classification field (Pitfall
# 7 / Anti-pattern "Letting the analysis read status to classify").
# ---------------------------------------------------------------------------

# The six trial-v2 figures and which gate must own each labeled fail at the
# chosen operating point — mirrors replay_figure_trial.EXPECTED_CLASSIFICATIONS
# style. sl_11 is the wrong-figure (floor) class; sl_19 the dropped-components
# (coverage) class.
REAPPLY_EXPECTED_FAILS = ("sl_11", "sl_19")


def _replay_margin_sign(margin: Any) -> bool | None:
    """Margin sign from a replay-receipt figure margin block (nested dinov2)."""

    return derive_margin_sign(margin)


def classify_reapply_tuple(
    *,
    coverage: float | None,
    dinov2_similarity: float | None,
    margin_sign: bool | None,
    critique_verdict: Any,
    t_cov: float,
    floor: float,
) -> tuple[str, list[str]]:
    """Pure stack-rule classification of one figure tuple (Pattern 5/6).

    fail = (coverage < t_cov) OR (dinov2 < floor) OR (margin_sign is False)
    OR (critique not pass). Returns (classification, firing_gates).
    Deliberately takes ONLY tuple values + thresholds so status fields cannot
    influence the result.
    """

    gates: list[str] = []
    if coverage is None or coverage < t_cov:
        gates.append("coverage")
    if dinov2_similarity is not None and dinov2_similarity < floor:
        gates.append("floor")
    if margin_sign is False:
        gates.append("margin")
    if critique_verdict not in (None, "pass"):
        gates.append("critique")
    return ("fail" if gates else "pass", gates)


def reapply_thresholds(
    replay_figures: list[dict[str, Any]], t_cov: float, floor: float
) -> dict[str, Any]:
    """Re-apply (t_cov, floor) to the six trial-v2 replay tuples (done gate).

    Recomputes each figure's stack classification from its recorded tuple
    fields and asserts sl_11 and sl_19 classify fail. ``ok`` is True iff both
    expected fails are blocked. Any classification/status fields on the input
    figures are ignored by construction (classify_reapply_tuple takes only
    tuple values).
    """

    classifications: list[dict[str, Any]] = []
    for figure in replay_figures:
        slide_id = str(figure.get("slide_id"))
        classification, gates = classify_reapply_tuple(
            coverage=figure.get("coverage"),
            dinov2_similarity=figure.get("dinov2_similarity"),
            margin_sign=_replay_margin_sign(figure.get("margin")),
            critique_verdict=(figure.get("critique") or {}).get("verdict"),
            t_cov=t_cov,
            floor=floor,
        )
        classifications.append(
            {"slide_id": slide_id, "classification": classification, "gates": gates}
        )
    by_prefix = {c["slide_id"][:5]: c for c in classifications}
    expected_fail_ok = all(
        by_prefix.get(prefix, {}).get("classification") == "fail"
        for prefix in REAPPLY_EXPECTED_FAILS
    )
    return {
        "ok": bool(expected_fail_ok),
        "t_cov": t_cov,
        "floor": floor,
        "expected_fails": list(REAPPLY_EXPECTED_FAILS),
        "classifications": classifications,
    }


def build_analysis(
    dataset: dict[str, Any],
    replay_receipt: dict[str, Any],
    *,
    operator_threshold: float | None = None,
    operator_floor: float | None = None,
    operator_note: str | None = None,
) -> dict[str, Any]:
    """Assemble the calibration_threshold_analysis payload (Task 3 contract).

    Runs the three D-12 population sweeps with apply_degeneracy_triggers per
    population (frozen layering; only P1 drives the halt), stack-level floor
    selection over the headline rows, the CLIP candidate (D-11), the
    separation record, and the re-application check at the chosen operating
    point. Operator values, when supplied, are recorded verbatim with
    recomputed error rates and a re-applied done-gate check — they never
    replace the recorded degenerate evidence.
    """

    rows = dataset.get("rows", []) or []
    populations = build_populations(rows)

    population_results: list[dict[str, Any]] = []
    headline_result: dict[str, Any] | None = None
    for pop in populations:
        sweep = apply_degeneracy_triggers(
            select_coverage_threshold(pop["rows"]), pop["rows"]
        )
        entry = {
            "name": pop["name"],
            "definition": pop["definition"],
            "row_count": pop["row_count"],
            "result": sweep,
        }
        population_results.append(entry)
        if pop["name"] == "P1_headline":
            headline_result = sweep

    assert headline_result is not None  # build_populations always emits P1
    headline_rows = next(
        p["rows"] for p in populations if p["name"] == "P1_headline"
    )

    # Floor selection uses the headline coverage threshold when one exists;
    # otherwise the operator threshold (D-10 resolution) or None.
    t_cov = headline_result.get("threshold")
    if t_cov is None and operator_threshold is not None:
        t_cov = operator_threshold

    floor_selection: dict[str, Any] | None = None
    clip_candidate: dict[str, Any] | None = None
    if t_cov is not None:
        floor_selection = select_stack_floor(headline_rows, t_cov, "dinov2_similarity")
        clip_candidate = clip_candidate_floor(headline_rows, t_cov)

    separation = separation_record(rows)

    per_metric_curves = {
        "dinov2": _per_metric_sweep(headline_rows, "dinov2_similarity"),
        "clip": _per_metric_sweep(headline_rows, "clip_similarity"),
    }

    # Operating point for the re-application done gate. Operator values win
    # (D-10 resolution); else the selected headline threshold + selected floor.
    op_threshold = operator_threshold
    op_floor = operator_floor
    selected_floor = (
        floor_selection.get("floor") if floor_selection else None
    )
    reapply_threshold = op_threshold if op_threshold is not None else t_cov
    reapply_floor = op_floor if op_floor is not None else selected_floor

    replay_figures = replay_receipt.get("figures", []) or []
    if reapply_threshold is not None and reapply_floor is not None:
        reapplication = reapply_thresholds(
            replay_figures, reapply_threshold, reapply_floor
        )
    else:
        reapplication = {
            "ok": False,
            "reason": "no operating point available (degenerate headline, no operator decision)",
            "t_cov": reapply_threshold,
            "floor": reapply_floor,
            "classifications": [],
        }

    operator_decision: dict[str, Any] | None = None
    if operator_threshold is not None and operator_floor is not None:
        operator_decision = {
            "threshold": operator_threshold,
            "floor": operator_floor,
            "note": operator_note,
            "observed_error_rates": {
                "coverage_sweep_false_fail": headline_result.get("false_fail"),
                "floor_false_fail_count": (
                    floor_selection.get("false_fail_count") if floor_selection else None
                ),
            },
        }

    # Status precedence: operator decision is explicit; else degenerate
    # headline halts; else selected.
    if operator_decision is not None:
        status = "operator_decided"
    elif headline_result.get("status") == "degenerate":
        status = "degenerate"
    else:
        status = "selected"

    return {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": _utc_now(),
        "halt_tolerance": HALT_TOLERANCE_FALSE_FAIL,
        "status": status,
        "coverage_threshold_selection": {
            "headline": "P1_headline",
            "populations": population_results,
        },
        "floor_selection": floor_selection,
        "clip_candidate_floor": clip_candidate,
        "per_metric_curves": per_metric_curves,
        "separation": separation,
        "reapplication": reapplication,
        "operator_decision": operator_decision,
    }


def _cmd_analyze(args: argparse.Namespace) -> int:
    # All-or-nothing operator trio for the threshold+floor pair (note
    # optional): refuse a partial operator decision (config error, exit 2).
    has_threshold = args.operator_threshold is not None
    has_floor = args.operator_floor is not None
    if has_threshold != has_floor:
        print(
            "[calibrate] --operator-threshold and --operator-floor must be "
            "given together (the note is optional)",
            file=sys.stderr,
        )
        return 2

    dataset = _load_json(args.dataset, "calibration dataset")
    if dataset is None:
        return 1
    replay_receipt = _load_json(args.replay_receipt, "replay receipt")
    if replay_receipt is None:
        return 1

    analysis = build_analysis(
        dataset,
        replay_receipt,
        operator_threshold=args.operator_threshold,
        operator_floor=args.operator_floor,
        operator_note=args.operator_note,
    )
    analysis["inputs"] = {
        "dataset": _file_ref(args.dataset),
        "replay_receipt": _file_ref(args.replay_receipt),
    }
    # Halt-and-surface needs the artifact: write the receipt BEFORE returning
    # the degenerate exit code (mirrors the budget-halt receipt-then-exit-3
    # contract).
    write_json_atomic(args.out, analysis)

    status = analysis["status"]
    reapply_ok = bool((analysis.get("reapplication") or {}).get("ok"))
    print(
        f"[calibrate] analysis: status={status}, "
        f"reapplication ok={reapply_ok} -> {args.out}",
        file=sys.stderr,
    )

    if status == "operator_decided":
        # Exit 0 only if the operator point still holds the re-application
        # done gate; otherwise surface the contract failure (exit 1).
        return 0 if reapply_ok else 1
    if status == "degenerate":
        return 3  # D-10 halt-and-surface (receipt already written)
    return 0 if reapply_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_assemble = sub.add_parser(
        "assemble-dataset",
        help="Assemble the labeled calibration dataset from receipt JSON.",
    )
    p_assemble.add_argument("--batch-receipt", type=Path, required=True)
    p_assemble.add_argument("--replay-receipt", type=Path, required=True)
    p_assemble.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="Optional merge-verdicts output; pre-verdict assembly omits it.",
    )
    p_assemble.add_argument("--out", type=Path, required=True)
    p_assemble.set_defaults(handler=_cmd_assemble_dataset)

    p_review = sub.add_parser(
        "build-review",
        help="Build the blind review manifest, gate-reveal sidecar, and HTML.",
    )
    p_review.add_argument("--batch-receipt", type=Path, required=True)
    p_review.add_argument("--deck-dir", type=Path, required=True)
    p_review.add_argument("--out-manifest", type=Path, required=True)
    p_review.add_argument("--out-reveal", type=Path, required=True)
    p_review.add_argument("--out-html", type=Path, required=True)
    p_review.set_defaults(handler=_cmd_build_review)

    p_merge = sub.add_parser(
        "merge-verdicts",
        help="Validate and record operator verdicts against the manifest.",
    )
    p_merge.add_argument("--manifest", type=Path, required=True)
    p_merge.add_argument("--draft", type=Path, required=True)
    p_merge.add_argument("--out", type=Path, required=True)
    p_merge.add_argument("--reviewer", default="operator")
    p_merge.set_defaults(handler=_cmd_merge_verdicts)

    # plan 02-04: the `analyze` subcommand (D-09 coverage sweep with D-10
    # degeneracy + D-12 populations, stack-level DINOv2 floor, candidate CLIP
    # floor D-11, DINOv2-vs-CLIP separation record, sl_11/sl_19 re-application
    # check, operator-decision recording). Exit codes: 0 selected/operator-
    # decided, 3 degenerate halt-and-surface, 1 missing input, 2 config error.
    p_analyze = sub.add_parser(
        "analyze",
        help="Select coverage threshold + DINOv2 floor from the dataset; "
        "record separation and the sl_11/sl_19 re-application check.",
    )
    p_analyze.add_argument("--dataset", type=Path, required=True)
    p_analyze.add_argument("--replay-receipt", type=Path, required=True)
    p_analyze.add_argument("--out", type=Path, required=True)
    p_analyze.add_argument(
        "--operator-threshold",
        type=float,
        default=None,
        help="D-10 resolution: operator-chosen coverage threshold (with "
        "--operator-floor).",
    )
    p_analyze.add_argument(
        "--operator-floor",
        type=float,
        default=None,
        help="D-10 resolution: operator-chosen DINOv2 floor (with "
        "--operator-threshold).",
    )
    p_analyze.add_argument(
        "--operator-note",
        type=str,
        default=None,
        help="Optional rationale recorded verbatim with the operator decision.",
    )
    p_analyze.set_defaults(handler=_cmd_analyze)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
