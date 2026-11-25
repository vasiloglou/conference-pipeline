#!/usr/bin/env python3
"""Figure trial v2: vision-grounded redraw + D2 + HTML/CSS tiers on a slide subset.

Generates trial figures ALONGSIDE the existing phase16 assets (separate
trial-v2 directory, nothing overwritten), with per-call provider receipts, a
hard budget cap (halt-and-approve, never auto-raised), CLIP similarity scores
for grounded redraws, and a side-by-side comparison page for the human gate.

Usage:
  .venv/bin/python scripts/presentations/run_figure_trial.py \
    --workspace conferences/iclr2026 \
    --deck-dir conferences/iclr2026/decks/03_evaluation_benchmarks_data \
    --trial-config conferences/iclr2026/decks/03_evaluation_benchmarks_data/figure_trial_config.json \
    --budget-cap-usd 15 --live
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html as html_module
import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.figure_grounding import (
    GroundingResult,
    GroundingSkip,
    bare_openreview_id,
    ground_slide_figure,
)
from conference_pipeline.presentation.figure_similarity import (
    DEFAULT_SIMILARITY_THRESHOLD,
    clip_image_similarity,
    dinov2_image_similarity,
    dinov2_runtime_info,
    margin_sign_test,
    preflight_dinov2,
)
from conference_pipeline.presentation.renderer_registry import (
    validate_figure_design_brief_v2,
)
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    provider_receipt_context,
)
from conference_pipeline.workflow_runs import write_json_atomic


def _load_phase16_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_programmatic_figures",
        Path(__file__).resolve().parent / "generate_programmatic_figures.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PHASE16 = _load_phase16_module()

TRIAL_STAGE_ID = "figure_trial_v2"
OPUS_MODEL = "claude-opus-4-8"  # model-literal-ok: figure_trial_v2 harness compares fixed Opus vs Sonnet arms; pinned for trial reproducibility
SONNET_MODEL = "claude-sonnet-5"  # model-literal-ok: figure_trial_v2 harness compares fixed Opus vs Sonnet arms; pinned for trial reproducibility
TIERS = ("vision_redraw", "data_redraw", "d2", "html_css")
D2_BRAND_HEADER = PROJECT_ROOT / "assets" / "brand" / "relationalai" / "d2_brand_header.d2"
BRAND_CSS = PROJECT_ROOT / "assets" / "brand" / "relationalai" / "brand_slide.css"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


class TrialBudgetExceeded(RuntimeError):
    """Raised when accumulated provider spend reaches the configured cap."""


class BudgetTracker:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0

    def record(self, result: Any) -> None:
        self.calls += 1
        receipt = getattr(result, "receipt", None)
        cost = getattr(receipt, "estimated_cost_usd", None)
        if cost is None:
            # WR-01 fail-closed: an unpriced call (model id missing from
            # pricing_catalog.json) would otherwise never count against the
            # cap, silently turning the halt-and-approve guardrail into a
            # no-op. Halt for operator approval instead — the cap is a
            # guardrail and is never auto-raised. A priced 0.0-cost call is
            # legitimate and distinguishable from an unpriced one.
            raise TrialBudgetExceeded(
                f"call {self.calls} returned no cost estimate "
                f"(cost_status={getattr(receipt, 'cost_status', None)!r}); "
                "cannot enforce the budget cap — halting for operator approval"
            )
        self.spent_usd += float(cost)
        if self.spent_usd >= self.cap_usd:
            raise TrialBudgetExceeded(
                f"figure trial spend ${self.spent_usd:.2f} reached the budget cap "
                f"${self.cap_usd:.2f} after {self.calls} calls; halting for operator approval"
            )


class SvgRenderer:
    """Render SVG or HTML files to 1280x720 PNGs with one shared Chromium."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    def _page(self) -> Any:
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch()
        return self._browser.new_page(viewport={"width": 1280, "height": 720})

    def render(self, source_path: Path, png_path: Path) -> None:
        page = self._page()
        try:
            page.goto(source_path.resolve().as_uri())
            page.screenshot(path=str(png_path))
        finally:
            page.close()

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._playwright.stop()
            self._browser = None
            self._playwright = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--budget-cap-usd", type=float, default=15.0)
    parser.add_argument("--opus-model", default=OPUS_MODEL)
    parser.add_argument("--critique-model", default=SONNET_MODEL)
    parser.add_argument("--ranker-model", default=SONNET_MODEL)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Blocking coverage-gate prior (CALIB-02) until Phase 02 selects it empirically",
    )
    parser.add_argument(
        "--claim-match-threshold",
        type=float,
        default=0.7,
        help="Grounding provenance confidence below which the redraw tier is skipped (CALIB-04)",
    )
    parser.add_argument("--live", action="store_true", help="Make live provider calls")
    parser.add_argument("--anthropic-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--force-overwrite-receipt",
        action="store_true",
        help=(
            "Overwrite an existing figure_trial_receipt.json in the deck dir "
            "(destroys the prior provenance record — CR-01 guard override)"
        ),
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=None,
        help=(
            "run receipt path (default: <deck-dir>/figure_trial_receipt.json; "
            "relative paths resolve under --deck-dir). Protected targets "
            "(trial/replay receipts) are refused — see _protected_receipt_refusal"
        ),
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=None,
        help=(
            "comparison page path (default: derived from the resolved "
            "--receipt-out by replacing a trailing _receipt.json with "
            "_comparison.html; relative paths resolve next to the receipt)"
        ),
    )
    parser.add_argument(
        "--stage-id",
        default=TRIAL_STAGE_ID,
        help=(
            "provider receipt stage id under storage/provider-calls/ "
            f"(default: {TRIAL_STAGE_ID}; batch runs should use their own id)"
        ),
    )
    parser.add_argument(
        "--resume-from-receipt",
        type=Path,
        default=None,
        help=(
            "halted batch receipt to resume from (D-15): slides whose prior "
            "outcome status is terminal are skipped (no re-spend), prior "
            "results merge verbatim into the new receipt with resumed_from "
            "sha-chain provenance; relative paths resolve under --deck-dir"
        ),
    )
    parser.add_argument(
        "--adjudication-decisions",
        type=Path,
        default=None,
        help=(
            "Phase 03 Codex adjudication decisions JSON. When supplied, trial "
            "routing uses adjudication decisions before provider attempts."
        ),
    )
    parser.add_argument(
        "--budget-projection-id",
        default=None,
        help="Identifier for the pre-spend budget projection recorded in schema_version 3 receipts.",
    )
    return parser.parse_args(argv)


def _receipt_overwrite_refusal(receipt_path: Path, force: bool) -> str | None:
    """Refusal message when a run would overwrite an existing trial receipt (CR-01).

    The schema-1 trial receipt is the immutable provenance record the replay
    verifies by sha256 (``replay_of.sha256``) and the Phase 02 calibration
    baseline. Returns ``None`` when the path does not exist or the operator
    passed ``--force-overwrite-receipt``; otherwise a refusal message naming
    the existing receipt's schema_version, the path, and the override flag.
    Pure helper so the guard is unit-testable without ``--live``.
    """

    if force or not receipt_path.is_file():
        return None
    try:
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        schema_version: Any = existing.get("schema_version")
    except (OSError, ValueError, AttributeError):
        schema_version = None
    if schema_version is None:
        schema_version = "unknown"
    return (
        f"refusing to overwrite existing trial receipt "
        f"(schema_version {schema_version}) at {receipt_path}; "
        "pass --force-overwrite-receipt to override"
    )


def _resolve_receipt_out(receipt_out: Path | None, deck_dir: Path) -> Path:
    """Resolve ``--receipt-out`` to its on-disk target.

    ``None`` keeps the as-shipped trial default
    (``<deck-dir>/figure_trial_receipt.json``); relative paths land under
    ``deck_dir``; absolute paths pass through unchanged. Pure resolve-only
    helper (no I/O) so it is unit-testable without ``--live``.
    """

    if receipt_out is None:
        return deck_dir / "figure_trial_receipt.json"
    if receipt_out.is_absolute():
        return receipt_out
    return deck_dir / receipt_out


_RECEIPT_NAME_SUFFIX = "_receipt.json"
_COMPARISON_NAME_SUFFIX = "_comparison.html"


def _resolve_comparison_out(comparison_out: Path | None, receipt_out: Path) -> Path:
    """Resolve ``--comparison-out`` from the resolved receipt-out path.

    Derivation rule: replace a trailing ``_receipt.json`` with
    ``_comparison.html`` (``figure_trial_receipt.json`` ->
    ``figure_trial_comparison.html``, ``figure_calibration_receipt.json`` ->
    ``figure_calibration_comparison.html``); fallback
    ``{stem}_comparison.html``. An explicit ``--comparison-out`` overrides:
    absolute passes through, relative resolves next to the receipt. Pure
    helper, no I/O.
    """

    if comparison_out is not None:
        if comparison_out.is_absolute():
            return comparison_out
        return receipt_out.parent / comparison_out
    name = receipt_out.name
    if name.endswith(_RECEIPT_NAME_SUFFIX):
        derived = name[: -len(_RECEIPT_NAME_SUFFIX)] + _COMPARISON_NAME_SUFFIX
    else:
        derived = receipt_out.stem + _COMPARISON_NAME_SUFFIX
    return receipt_out.with_name(derived)


def _protected_receipt_refusal(
    receipt_out: Path,
    comparison_out: Path,
    deck_dir: Path,
    *,
    force: bool = False,
) -> str | None:
    """Refusal when batch outputs would clobber immutable trial provenance.

    T-02-01 mitigation (exit 2), pure resolve-and-compare — no I/O beyond
    ``Path.resolve``. Protected set:

    - ``figure_trial_receipt.json``: refused as a ``--receipt-out`` target
      only when ``--force-overwrite-receipt`` would otherwise allow the
      overwrite (without force the CR-01 exists-check guard owns this path,
      keeping the as-shipped trial flow byte-identical).
    - ``figure_trial_replay_receipt.json``: never a valid ``--receipt-out``
      target, under any flag combination.
    - ``figure_trial_comparison.html``: never a resolved ``--comparison-out``
      target unless the resolved receipt-out IS the back-compat trial
      default — a batch run can never clobber the trial comparison page.
    """

    trial_receipt = (deck_dir / "figure_trial_receipt.json").resolve()
    replay_receipt = (deck_dir / "figure_trial_replay_receipt.json").resolve()
    trial_comparison = (deck_dir / "figure_trial_comparison.html").resolve()
    receipt_resolved = receipt_out.resolve()
    if receipt_resolved == replay_receipt:
        return (
            f"refusing --receipt-out at the protected replay receipt {receipt_out}; "
            "figure_trial_replay_receipt.json is an immutable provenance record"
        )
    if force and receipt_resolved == trial_receipt:
        return (
            "refusing --force-overwrite-receipt on the protected trial receipt "
            f"{receipt_out}; force can never destroy trial provenance — use "
            "--receipt-out with a fresh path instead"
        )
    if comparison_out.resolve() == trial_comparison and receipt_resolved != trial_receipt:
        return (
            f"refusing --comparison-out at the protected trial comparison page "
            f"{comparison_out}; batch runs must not clobber figure_trial_comparison.html"
        )
    return None


# D-15 resume: a prior outcome in any of these statuses is durable — its
# receipts/assets stand and the slide is never re-run (no re-spend). A slide
# ABSENT from the prior results (mid-attempt at the budget halt — the
# TrialBudgetExceeded propagates out of the tier executor before
# results.append) restarts from its own beginning, with its superseded
# per-call provider receipts kept in the stage dir.
RESUME_TERMINAL_STATUSES = ("generated", "best_effort", "skipped", "failed")


def _resume_skip_set(prior_receipt: dict[str, Any]) -> set[str]:
    """Slide ids whose prior outcome reached a terminal status (D-15).

    Pure helper over the prior receipt dict — no I/O — so the no-re-spend
    contract is unit-testable without ``--live``.
    """

    return {
        str(entry.get("slide_id"))
        for entry in prior_receipt.get("results", [])
        if (entry.get("outcome") or {}).get("status") in RESUME_TERMINAL_STATUSES
    }


def _merge_resumed_results(
    prior_results: list[dict[str, Any]], new_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge prior TERMINAL entries (verbatim) ahead of this run's results.

    Non-terminal prior entries are dropped — their slides re-ran and
    contribute fresh entries. A duplicate slide_id between the kept prior
    entries and the new results means a terminal slide was re-run, which
    the skip set forbids: refuse loudly rather than double-count (D-15).
    """

    merged = [
        entry
        for entry in prior_results
        if (entry.get("outcome") or {}).get("status") in RESUME_TERMINAL_STATUSES
    ]
    seen = {str(entry.get("slide_id")) for entry in merged}
    for entry in new_results:
        slide_id = str(entry.get("slide_id"))
        if slide_id in seen:
            raise ValueError(
                f"duplicate slide_id in resume merge: {slide_id}; "
                "a terminal slide must never be re-run (D-15 no re-spend)"
            )
        seen.add(slide_id)
        merged.append(entry)
    return merged


def _resumed_from_block(
    resume_path: Path, prior_receipt_bytes: bytes, prior_receipt: dict[str, Any]
) -> dict[str, Any]:
    """``resumed_from`` provenance block (sha-chain over the prior RAW bytes).

    Records the halt -> approval -> resume chain: where the halted receipt
    came from, its content hash, and the prior run's spend/call totals so
    the cumulative receipt numbers are auditable against the stage dir.
    """

    return {
        "path": str(resume_path),
        "sha256": hashlib.sha256(prior_receipt_bytes).hexdigest(),
        "prior_spend_usd": float(prior_receipt.get("estimated_spend_usd") or 0.0),
        "prior_call_count": int(prior_receipt.get("provider_call_count") or 0),
        "prior_budget_halt": prior_receipt.get("budget_halt"),
    }


def _cumulative_spend(
    prior_receipt: dict[str, Any] | None, run_spend_usd: float, run_calls: int
) -> tuple[float, int]:
    """Cumulative spend/calls = prior receipt totals + this run's tracker.

    The fresh ``--budget-cap-usd`` applies to THIS run's tracker only; the
    cap value is never modified in code (halt-and-approve guardrail — the
    operator chooses the resume cap at approval time).
    """

    if not prior_receipt:
        return round(run_spend_usd, 4), run_calls
    prior_spend = float(prior_receipt.get("estimated_spend_usd") or 0.0)
    prior_calls = int(prior_receipt.get("provider_call_count") or 0)
    return round(prior_spend + run_spend_usd, 4), prior_calls + run_calls


def _archive_halted_receipt(receipt_path: Path) -> Path:
    """Archive-before-overwrite (01-07 pattern) for an in-place resume.

    Moves the halted receipt to ``{stem}.halted-{UTC timestamp}.json`` in
    the same directory so the original survives byte-identical (its sha is
    recorded in ``resumed_from``) and the CR-01 exists-check below never
    fires on a receipt the operator is deliberately superseding.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = receipt_path.with_name(f"{receipt_path.stem}.halted-{stamp}.json")
    counter = 1
    while archive.exists():
        archive = receipt_path.with_name(
            f"{receipt_path.stem}.halted-{stamp}-{counter}.json"
        )
        counter += 1
    receipt_path.rename(archive)
    return archive


def _anthropic_text(result: Any) -> str:
    return "".join(
        block.text for block in result.response.content if getattr(block, "type", "") == "text"
    )


def _image_block(image_path: Path, *, max_dim: int = 1280) -> dict[str, Any]:
    import io

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DRAWSVG_HARD_RULES = """Hard requirements:
- Return Python code only, no markdown fences.
- Define exactly: build_svg(output_path: str) -> None
- Use variable name d for the draw.Drawing object; canvas exactly 1280x720.
- The final statement inside build_svg must be exactly: d.save_svg(output_path)
- Use only: import drawsvg as draw, optional import math. No other imports; never touch files, network, os, sys, subprocess, pathlib.
- Fonts LARGE: every text element >= 18px, node/box titles >= 24px. Avoid pure black.
- Use only these colors: #1a1a2e, #a8d8ea, #ff6b6b, #51cf66, #ffd43b, #9775fa, #4ecdc4, #ffffff, #e0e0e0, #cccccc, #333344, #2a2a4a, #16213e. Background must be #1a1a2e.
- At most 26 visible text labels. Every label is a COMPLETE phrase; never end a label on a connector word (and, or, of, the, to, for, with, in) or a hyphen.
- Size every box FROM its text: box_width = max(len(label) for label in box_labels) * font_size * 0.62 + 28; box_height = line_count * font_size * 1.5 + 20. Compute these as variables before drawing.
- Draw ALL boxes first, then route thin connectors (stroke-width 2-3, arrowheads <= 10px) between box BORDERS along clear lanes; never start, end, or route a line under or through any text.
- Never place two text elements at the same coordinates; keep all text inside the canvas; leave clear padding around every label.
- Never show raw provenance ids (no 'paper:', 'SRC-', 'transcript:' strings), file paths, stage names, or internal scaffold labels.
- Keep every Python string literal under 100 characters; plain ASCII quotes only."""


def _redraw_prompt(row: dict[str, Any], grounding: GroundingResult, *, tier: str, feedback: str) -> str:
    payload = {
        "slide_id": row["slide_id"],
        "title_short": shorten(row["title"], width=90, placeholder="..."),
        "claim_short": shorten(row["claim"], width=200, placeholder="..."),
        "figure_brief": shorten(" ".join(str(row.get("visual_concept") or "").split()), width=460, placeholder=""),
        "source_paper": grounding.paper_title,
        "source_figure_caption": grounding.caption,
        "attribution_line": grounding.attribution_line(),
        "required_entities": row["required_entities"][:10],
        "forbidden_patterns": row["forbidden_patterns"],
    }
    data_note = ""
    if tier == "data_redraw":
        data_note = (
            "\n- This is a RESULTS/CHART redraw: take the chart TYPE, axes, and series structure from the "
            "attached source figure, but take every NUMBER from figure_brief / claim_short — never read "
            "data values off the source pixels. If a value is not given in the brief, depict the "
            "qualitative relationship without inventing precise numbers."
        )
    repair_note = f"\nPrevious attempt feedback (fix every point):\n{feedback}\n" if feedback else ""
    return f"""You are redrawing a REAL figure from a research paper as a branded editorial SVG.

The attached image is the source figure from the paper "{grounding.paper_title}".
Stage 1 — analysis (write as a Python comment block at the top of the file): list the source figure's
actual components, stages, connections, and flow direction as you read them from the image. Use the
paper's own component names. If a label in the image is unreadable, do NOT guess letter-by-letter;
use a faithful short paraphrase from the caption or claim.
Stage 2 — code: reproduce the SAME structure (same components, same connections, same ordering) as a
clean 1280x720 drawsvg figure in the brand style. Translate the layout; do not invent components that
are not in the source figure and do not drop its core stages.{data_note}

{DRAWSVG_HARD_RULES}
- Render the attribution_line EXACTLY ONCE as an 18px italic caption in color #cccccc, placed directly
  under the main heading area (top of the canvas), never at the bottom edge.
- Do NOT copy the source figure's colors or fonts; use only the brand palette above.

Figure contract JSON:
{json.dumps(payload, indent=2, sort_keys=True)}
{repair_note}"""


def _d2_prompt(row: dict[str, Any], *, feedback: str) -> str:
    payload = {
        "slide_id": row["slide_id"],
        "title_short": shorten(row["title"], width=90, placeholder="..."),
        "claim_short": shorten(row["claim"], width=200, placeholder="..."),
        "figure_brief": shorten(" ".join(str(row.get("visual_concept") or "").split()), width=460, placeholder=""),
        "required_entities": row["required_entities"][:10],
        "required_relationships": row["required_relationships"][:8],
        "source_chip_bindings": row.get("source_chip_bindings") or [],
    }
    repair_note = f"\nPrevious attempt failed to compile or review; fix every point:\n{feedback}\n" if feedback else ""
    return f"""Write the BODY of a D2 diagram (the brand header with classes is prepended by the pipeline).

Depict the figure_brief as a real mechanism/architecture/flow diagram — never a bullet list or a
grid of disconnected boxes. Use the ACTUAL named works and concepts from required_entities as node
labels and draw required_relationships as labeled edges between them.

Syntax — follow this shape EXACTLY (simple lowercase ids; label and class inside the block):
leaderboard: "Static leaderboard average" {{ class: highlight }}
contract: "Measurement contract" {{ class: panel }}
failures: "Hidden reliability failures" {{
  class: group
  tail: "Tail coverage" {{ class: stage }}
}}
leaderboard -> contract: "exposes axes" {{ class: edge }}
contract -> failures.tail: {{ class: edge-accent }}

Rules:
- Output D2 source only, no markdown fences, no explanations.
- Do NOT define vars, classes, or styles — only nodes, containers, and edges that use the existing
  classes: panel, stage, highlight, good, note, group (nodes) and edge, edge-accent (connections).
- Every node MUST set a quoted label and one class exactly as in the example; never append a class
  to a node id or label (no 'name.good' nodes) and ALWAYS double-quote labels.
- Node labels: short complete phrases (<= 5 words), never ending on a connector word; ids are short
  lowercase identifiers without spaces, dots, or special characters.
- 6 to 12 nodes; at most 2 container groups; every node connected — no orphans; prefer a COMPACT
  layout (two ranks of nodes, not one long chain) so labels stay large when scaled to a slide.
- Edge labels optional, <= 4 words, always double-quoted.
- Never show raw provenance ids ('paper:', 'SRC-', 'transcript:'), file paths, or scaffold terms.
- When a work is named, attribute it inline as 'Short Name (Lead Author et al.)' using
  source_chip_bindings; never attach one work's authors to a different work.

Figure contract JSON:
{json.dumps(payload, indent=2, sort_keys=True)}
{repair_note}"""


def _html_prompt(row: dict[str, Any], *, feedback: str) -> str:
    payload = {
        "slide_id": row["slide_id"],
        "title_short": shorten(row["title"], width=90, placeholder="..."),
        "claim_short": shorten(row["claim"], width=200, placeholder="..."),
        "figure_brief": shorten(" ".join(str(row.get("visual_concept") or "").split()), width=520, placeholder=""),
        "required_entities": row["required_entities"][:10],
        "required_relationships": row["required_relationships"][:8],
        "source_chip_bindings": row.get("source_chip_bindings") or [],
    }
    css_excerpt = BRAND_CSS.read_text(encoding="utf-8")
    repair_note = f"\nPrevious attempt feedback (fix every point):\n{feedback}\n" if feedback else ""
    return f"""Write the BODY content of a 1280x720 HTML figure (the brand stylesheet below is injected
by the pipeline as the first <style> block; html/body sizing is already handled).

Build a dense, evidence-rich composite: panels, flows, metric callouts, and bars that depict the
figure_brief's components and quantitative data. Use ONLY the classes and CSS custom properties from
the brand kit; no inline colors, no external resources, no JavaScript, no <img>.

Rules:
- Output one <div class="figure-root">...</div> only, no markdown fences, no <html>/<head>/<style>.
- All content must fit 1280x720 with nothing clipped (the kit uses overflow hidden — overflow is a defect).
- Use real named entities from required_entities; depict required_relationships as visual flow
  (.row + .arrow) or grouped panels, never as a sentence list.
- Numbers shown must come from figure_brief or claim_short; never invent precise values.
- Never show raw provenance ids ('paper:', 'SRC-', 'transcript:'), file paths, or scaffold terms.
- Attribute works inline as 'Short Name (Lead Author et al.)' from source_chip_bindings.

Brand kit (for reference, do not repeat it):
{css_excerpt}

Figure contract JSON:
{json.dumps(payload, indent=2, sort_keys=True)}
{repair_note}"""


def _critique_prompt(*, tier: str, has_source: bool) -> str:
    flag_part = (
        "Flag: overlapping or clipped text, labels ending mid-phrase, text crossing lines, "
        "bullet-box grids with no relationships, raw provenance ids (SRC-, paper:), unreadable small "
        "text, garbled or hallucinated words. "
    )
    if has_source:
        # CALIB-02 structured critique: the SAME single call returns component
        # lists (enumerate-then-match) so coverage is computed deterministically
        # downstream — zero extra provider calls. Gap-closure amendment
        # (01-05-SUMMARY.md, operator-directed): deliberate redraw-policy
        # deltas are excluded from missing/invented counts INSIDE the prompt —
        # the coverage arithmetic stays pure, no post-hoc list filtering.
        return (
            "Image 1 is the SOURCE figure from the paper; image 2 is our branded redraw. "
            "Judge whether the redraw preserves the source's structure: same components, same "
            "connections/flow direction, no invented components, no dropped core stages. Then judge "
            "legibility of the redraw itself. "
            + flag_part
            + "The redraw follows a deliberate restyling policy. The following four categories are "
            "intentional redraw-policy deltas and MUST NOT be listed in missing_components or "
            "invented_components: (a) attribution or licensing text, such as an 'Adapted from "
            "... CC BY 4.0' subtitle; (b) slide titles or headlines added by the redraw; "
            "(c) brand theme, background, or color restyling — a restyled component is NOT "
            "missing or distorted solely because its colors or styling changed; (d) annotation "
            "callouts or highlight borders added by the redraw. Purely stylistic source "
            "attributes (color scheme, background, fonts) are not components and must not be "
            "enumerated in the source inventory. "
            "These exclusions do NOT relax structural fidelity: dropped or distorted core "
            "stages/boxes/axes/data series, wrong or missing connections/flow direction, and "
            "garbled or altered data labels (e.g., a Greek letter rendered as a Latin letter in "
            "an axis label) must still be counted as missing, and redraw-only structural "
            "elements (new boxes, stages, arrows, data series) must still be counted as "
            "invented. The verdict semantics are unchanged. "
            "Work in three steps. FIRST, list every STRUCTURAL component you can read in the "
            "SOURCE figure (stages, boxes, axes, data series, labels, arrows/connections). "
            "SECOND, classify each source component as matched (faithfully present in the "
            "redraw) or missing (absent or distorted in the redraw); matched_components plus "
            "missing_components together must cover the FULL source component inventory. THIRD, "
            "list redraw components that do not exist in the source as invented_components, "
            "excluding the intentional redraw-policy deltas above. Respond with JSON only: "
            '{"verdict": "pass" | "fail", "matched_components": ["<short component name>"], '
            '"missing_components": ["..."], "invented_components": ["..."], '
            '"structural_issues": ["..."], "legibility_issues": ["..."], '
            '"feedback": "<concise repair instructions for the next attempt>"}'
        )
    return (
        "The image is a generated presentation figure. Judge its legibility and information density. "
        + flag_part
        + "Respond with JSON only: "
        '{"verdict": "pass" | "fail", "structural_issues": ["..."], "legibility_issues": ["..."], '
        '"feedback": "<concise repair instructions for the next attempt>"}'
    )


CRITIQUE_COMPONENT_KEYS = (
    "matched_components",
    "missing_components",
    "invented_components",
)


def _validate_critique_payload(payload: dict) -> tuple[dict, str]:
    """Validate the structured-critique component lists (CALIB-02, T-01-01).

    Untrusted model JSON becomes the blocking-gate input, so enforce types
    before any gate math: each of the three component keys must be a list of
    strings. Non-string items are coerced via ``str()``; a missing or
    non-list key is replaced with ``[]`` and flags ``parse_status`` as
    ``"parse_failed"`` so Phase 02 can distinguish "model said fail" from
    "parse failed". Returns ``(normalized_payload, parse_status)``.
    """

    normalized = dict(payload)
    parse_status = "ok"
    for key in CRITIQUE_COMPONENT_KEYS:
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = []
            parse_status = "parse_failed"
            continue
        normalized[key] = [
            item if isinstance(item, str) else str(item) for item in value
        ]
    return normalized, parse_status


def component_coverage(
    matched: list[str], missing: list[str], invented: list[str]
) -> float | None:
    """Deterministic coverage = matched/(matched+missing+invented) (CALIB-02).

    Normalization before counting: whitespace-collapse, casefold, drop empty
    items, dedupe per list. All-empty lists yield ``None`` — an unlabeled
    gate cannot pass, so ``None`` BLOCKS acceptance (locked CALIB-02
    semantics). Pure function: no provider calls, no model coupling.
    """

    def _norm(items: list[str]) -> set[str]:
        return {
            " ".join(str(item).split()).casefold()
            for item in items
            if item and str(item).strip()
        }

    m, mi, inv = _norm(matched), _norm(missing), _norm(invented)
    total = len(m) + len(mi) + len(inv)
    if total == 0:
        return None
    return len(m) / total


def _margin_distractor_pool(
    source_ids: list[str], chosen_crop: Path, grounding_root: Path
) -> dict[str, Any]:
    """Distractor pool under the locked bound_papers_v1 protocol (CALIB-03).

    Pool = candidate crops already on disk from ALL the slide's bound papers
    (``grounding_root/crops/<note>/*.png``), excluding the chosen crop by
    resolved-path comparison. On-disk only: this helper never triggers crop
    extraction or network fetches. The protocol id is load-bearing — it is
    the replay/calibration pool definition Phase 02 re-derives from receipts.
    """

    notes: list[str] = []
    distractors: list[Path] = []
    chosen_resolved = chosen_crop.resolve()
    for source_id in source_ids:
        note_id = bare_openreview_id(source_id)
        if not note_id or note_id in notes:
            continue
        crop_paths = sorted((grounding_root / "crops" / note_id).glob("*.png"))
        if not crop_paths:
            continue
        notes.append(note_id)
        for crop in crop_paths:
            if crop.resolve() == chosen_resolved:
                continue
            distractors.append(crop)
    return {"distractors": distractors, "notes": notes, "protocol": "bound_papers_v1"}


def _attempt_gate_decision(
    *,
    critique_verdict: Any,
    coverage: float | None,
    coverage_threshold: float,
    margin_sign: bool | None,
) -> dict[str, Any]:
    """Per-attempt acceptance over the blocking gates (CALIB-02/CALIB-03).

    accept = critique pass AND coverage labeled AND coverage >= threshold AND
    margin_sign is not False. ``margin_sign`` None (no distractors on disk or
    degraded scorer) does NOT block — only an actual negative sign fails the
    attempt (locked CALIB-03: sign test, no tunable threshold). ``coverage``
    None blocks: an unlabeled blocking gate cannot pass (locked CALIB-02).
    """

    blocking_reasons: list[str] = []
    if critique_verdict != "pass":
        blocking_reasons.append("critique verdict fail")
    if coverage is None:
        blocking_reasons.append("coverage undefined (empty component lists)")
    elif coverage < coverage_threshold:
        blocking_reasons.append(
            f"coverage {coverage:.2f} below threshold {coverage_threshold:.2f}"
        )
    if margin_sign is False:
        blocking_reasons.append(
            "negative margin: render closer to a distractor crop than its claimed source"
        )
    return {"accept": not blocking_reasons, "blocking_reasons": blocking_reasons}


def _repair_feedback(
    critique: dict,
    estimator_findings: dict[str, list[str]],
    margin_note: str | None,
) -> str:
    """Repair feedback for the NEXT attempt (CALIB-05 advisory-with-retry).

    Joins critique issues, the first missing/invented components, critique
    feedback, flattened phase16 estimator findings, and the margin note —
    "; "-separated, capped at 1200 chars (existing cap). Estimator findings
    inform the next attempt but never block an otherwise-passing figure
    (mirrors phase16's own advisory pattern).
    """

    parts: list[str] = [
        *map(str, critique.get("structural_issues") or []),
        *map(str, critique.get("legibility_issues") or []),
    ]
    missing = [str(item) for item in (critique.get("missing_components") or [])][:6]
    if missing:
        parts.append("missing components: " + ", ".join(missing))
    invented = [str(item) for item in (critique.get("invented_components") or [])][:6]
    if invented:
        parts.append("invented components: " + ", ".join(invented))
    feedback = str(critique.get("feedback") or "")
    if feedback:
        parts.append(feedback)
    for findings in (estimator_findings or {}).values():
        parts.extend(str(finding) for finding in findings)
    if margin_note:
        parts.append(margin_note)
    return "; ".join(part for part in parts if part)[:1200]


ADJUDICATION_DECISIONS = {
    "accept",
    "accept_with_notes",
    "revise",
    "redraw",
    "needs_human",
    "benchmark_mismatch",
}


def _adjudication_receipt_block(adjudication: dict[str, Any] | None) -> dict[str, Any] | None:
    if adjudication is None:
        return None
    return {
        "id": adjudication.get("id") or adjudication.get("slide_id"),
        "decision": adjudication.get("decision"),
        "confidence": adjudication.get("confidence"),
        "failure_modes": list(adjudication.get("failure_modes") or []),
        "rationale": adjudication.get("rationale"),
        "next_action": adjudication.get("next_action"),
        "judge_votes": dict(adjudication.get("judge_votes") or {}),
    }


def _adjudication_repair_feedback(adjudication: dict[str, Any], *, max_chars: int = 1200) -> str:
    """Bounded repair feedback derived from one Codex adjudication record."""

    parts: list[str] = []
    decision = str(adjudication.get("decision") or "")
    if decision:
        parts.append(f"Codex adjudication decision: {decision}")
    failure_modes = [str(mode) for mode in adjudication.get("failure_modes") or []]
    if failure_modes:
        parts.append("Failure modes: " + ", ".join(failure_modes[:8]))
    next_action = str(adjudication.get("next_action") or "")
    if next_action:
        parts.append(f"Next action: {next_action}")
    rationale = str(adjudication.get("rationale") or "")
    if rationale:
        parts.append(f"Rationale: {rationale[:420]}")
    judge_rationales = adjudication.get("judge_rationales") or {}
    for judge, text in list(judge_rationales.items())[:3]:
        if text:
            parts.append(f"{judge} rationale: {str(text)[:240]}")
    estimator_findings = adjudication.get("estimator_findings") or {}
    for name, findings in estimator_findings.items():
        if findings:
            joined = ", ".join(str(item) for item in list(findings)[:4])
            parts.append(f"{name}: {joined}")
    return "; ".join(part for part in parts if part)[:max_chars]


def _route_adjudication_decision(adjudication: dict[str, Any]) -> dict[str, Any]:
    """Map one Codex adjudication decision to trial-control behavior."""

    decision = adjudication.get("decision")
    if decision not in ADJUDICATION_DECISIONS:
        raise ValueError(f"unknown adjudication decision: {decision!r}")
    stop_reason_by_decision = {
        "accept": "accepted",
        "accept_with_notes": "accepted_with_notes",
        "revise": "revision_required",
        "redraw": "redraw_required",
        "needs_human": "needs_human",
        "benchmark_mismatch": "benchmark_mismatch",
    }
    requires_provider_attempt = decision in {"revise", "redraw"}
    automatic_success = decision in {"accept", "accept_with_notes"}
    return {
        "decision": decision,
        "stop_reason": stop_reason_by_decision[decision],
        "requires_provider_attempt": requires_provider_attempt,
        "automatic_success": automatic_success,
        "excluded_from_automatic_success": decision in {"needs_human", "benchmark_mismatch"},
        "repair_feedback": (
            _adjudication_repair_feedback(adjudication)
            if requires_provider_attempt
            else ""
        ),
        "adjudication": _adjudication_receipt_block(adjudication),
    }


def _attempt_quality_score(attempt: dict[str, Any]) -> float:
    for key in ("coverage", "dinov2_similarity", "clip_similarity"):
        value = attempt.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return -1.0


def _adjudication_loop_controls(
    *,
    attempts: list[dict[str, Any]],
    attempt_cap: int,
    budget_projection_id: str | None,
) -> dict[str, Any]:
    scores = [_attempt_quality_score(attempt) for attempt in attempts]
    best_attempt = None
    if attempts:
        best_index, _ = max(enumerate(scores), key=lambda item: item[1])
        best_attempt = attempts[best_index].get("attempt")
    return {
        "attempt_cap": max(1, attempt_cap),
        "best_attempt": best_attempt,
        "regression_detected": any(
            later < earlier for earlier, later in zip(scores, scores[1:], strict=False)
        ),
        "convergence_guard": {
            "status": "max_attempts_reached" if len(attempts) >= max(1, attempt_cap) else "not_triggered",
            "attempt_count": len(attempts),
        },
        "budget_projection_id": budget_projection_id,
    }


def _adjudication_terminal_outcome(
    *,
    adjudication: dict[str, Any],
    tier: str,
    budget_projection_id: str | None,
) -> dict[str, Any]:
    route = _route_adjudication_decision(adjudication)
    status = "accepted" if route["automatic_success"] else "held"
    return {
        "status": status,
        "tier": tier,
        "schema_version": 3,
        "stop_reason": route["stop_reason"],
        "automatic_success": route["automatic_success"],
        "excluded_from_automatic_success": route["excluded_from_automatic_success"],
        "adjudication": route["adjudication"],
        "attempts": [],
        **_adjudication_loop_controls(
            attempts=[],
            attempt_cap=0,
            budget_projection_id=budget_projection_id,
        ),
    }


def _targeted_vs_blind_ab_result(
    *,
    blind_outcome: dict[str, Any],
    targeted_outcome: dict[str, Any],
    failure_modes: list[str],
) -> dict[str, Any]:
    """Dry-run scaffold for comparing blind and adjudication-targeted redraws."""

    blind_score = _attempt_quality_score(blind_outcome)
    targeted_score = _attempt_quality_score(targeted_outcome)
    return {
        "failure_modes": failure_modes,
        "blind_status": blind_outcome.get("status"),
        "targeted_status": targeted_outcome.get("status"),
        "blind_score": blind_score,
        "targeted_score": targeted_score,
        "targeted_improved": targeted_score > blind_score,
    }


def _load_adjudication_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", [])
    lookup: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        slide_id = decision.get("slide_id")
        if slide_id:
            lookup[str(slide_id)] = decision
    return lookup


# ---------------------------------------------------------------------------
# Tier executors
# ---------------------------------------------------------------------------


def _run_critique(
    *,
    client: Any,
    args: argparse.Namespace,
    budget: BudgetTracker,
    rendered_png: Path,
    source_crop: Path | None,
    tier: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if source_crop is not None:
        content.append(_image_block(source_crop))
    content.append(_image_block(rendered_png))
    content.append({"type": "text", "text": _critique_prompt(tier=tier, has_source=source_crop is not None)})
    result = anthropic_messages_create(
        client,
        caller=__file__,
        model=args.critique_model,
        # Component lists need headroom: truncation yields invalid JSON and a
        # silent fail-fallback (RESEARCH Pitfall 5), so 1500 not 700.
        max_tokens=1500,
        messages=[{"role": "user", "content": content}],
    )
    budget.record(result)
    text = _anthropic_text(result)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    verdict.setdefault("verdict", "fail")
    verdict.setdefault("feedback", text[:400])
    if source_crop is not None:
        # Grounded critiques carry the CALIB-02 component lists; validate the
        # untrusted JSON and record the parse status. Malformed JSON keeps the
        # degrade path above (verdict "fail" + feedback excerpt) and is marked
        # critique_parse_status "parse_failed".
        verdict, parse_status = _validate_critique_payload(verdict)
        verdict["critique_parse_status"] = parse_status
    verdict["receipt_path"] = getattr(result.receipt, "receipt_path", None)
    return verdict


def _generate_redraw_figure(
    *,
    row: dict[str, Any],
    tier: str,
    args: argparse.Namespace,
    client: Any,
    budget: BudgetTracker,
    renderer: SvgRenderer,
    grounding_root: Path,
    render_dir: Path,
    adjudication_record: dict[str, Any] | None = None,
    initial_feedback: str = "",
    budget_projection_id: str | None = None,
) -> dict[str, Any]:
    source_ids = [str(r.get("source_id") or "") for r in row.get("source_refs", [])]
    grounding = ground_slide_figure(
        client=client,
        caller=__file__,
        ranker_model=args.ranker_model,
        source_ids=source_ids,
        slide_claim=row["claim"],
        slide_title=row["title"],
        grounding_root=grounding_root,
        claim_match_threshold=args.claim_match_threshold,
        on_call=budget.record,
    )
    if isinstance(grounding, GroundingSkip):
        # CALIB-04: grounding too uncertain to redraw — skip the redraw tier
        # instead of redrawing the wrong paper's figure (the sl_11 failure mode).
        if grounding.reason == "claim_match_below_threshold":
            claim_match = grounding.claim_match if grounding.claim_match is not None else 0.0
            reason = (
                f"claim_match {claim_match:.2f} below threshold "
                f"{grounding.claim_match_threshold:.2f}; redraw tier skipped"
            )
        else:
            reason = (
                f"ranker rejected all {grounding.pool_candidate_count} pooled "
                "candidates; redraw tier skipped"
            )
        return {
            "status": "skipped",
            "reason": reason,
            "tier": tier,
            "grounding_skip": grounding.to_dict(),
        }
    if grounding is None:
        return {"status": "skipped", "reason": "no usable grounded source figure", "tier": tier}
    grounding.grounding_mode = tier
    if not grounding.license_adaptable:
        return {
            "status": "skipped",
            "reason": f"license {grounding.license!r} is not adaptable; figure not redrawn",
            "tier": tier,
            "grounding": grounding.to_dict(),
        }

    crop_path = Path(grounding.crop_path)
    # Margin distractor pool (CALIB-03): computed once per figure under the
    # locked bound_papers_v1 protocol — on-disk crops only, chosen excluded.
    margin_pool = _margin_distractor_pool(source_ids, crop_path, grounding_root)
    adjudication_block = _adjudication_receipt_block(adjudication_record)
    feedback = initial_feedback
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for attempt in range(1, max(1, args.max_attempts) + 1):
        prompt = _redraw_prompt(row, grounding, tier=tier, feedback=feedback)
        result = anthropic_messages_create(
            client,
            caller=__file__,
            model=args.opus_model,
            max_tokens=6000,
            system="You write safe, bounded Python code for editorial SVG figures. Return code only.",
            messages=[
                {
                    "role": "user",
                    "content": [_image_block(crop_path), {"type": "text", "text": prompt}],
                }
            ],
        )
        budget.record(result)
        code = PHASE16._normalize_phase16_generated_code(
            PHASE16._extract_python_code(_anthropic_text(result))
        )
        # Redraws take their content contract from the SOURCE figure, not the
        # binding entity list; safety/scaffold/raw-id checks still apply, while
        # the vision critique + similarity gate verify the semantics.
        review_row = {**row, "required_entities": [], "required_relationships": []}
        review = PHASE16._review_phase16_generated_code(code, row=review_row)
        record: dict[str, Any] = {
            "attempt": attempt,
            "provider_receipt_path": getattr(result.receipt, "receipt_path", None),
            "code_review_status": review["status"],
        }
        if adjudication_block is not None:
            record["adjudication"] = adjudication_block
        if review["status"] != "pass":
            record["findings"] = review["findings"]
            attempts.append(record)
            feedback = "; ".join(str(f.get("message")) for f in review["findings"][:6])
            continue
        row["code_path"].write_text(code.rstrip() + "\n", encoding="utf-8")
        execution_error = PHASE16._execute_phase16_generated_code(row["code_path"], row["asset_path"])
        if execution_error:
            record["execution_error"] = execution_error
            attempts.append(record)
            feedback = f"code failed to execute: {execution_error}"
            continue
        # Phase16 estimators (CALIB-05): run on the attempt SVG (exists before
        # rendering). The four named estimators — NOT _phase16_advisory_issues,
        # which includes a fifth check. Advisory-with-retry: findings feed the
        # NEXT attempt's repair feedback, never block a passing figure.
        estimator_findings = {
            "oversized_arrowhead": PHASE16._phase16_oversized_arrowhead_issues(row["asset_path"]),
            "line_over_text": PHASE16._phase16_line_over_text_issues(row["asset_path"]),
            "text_overlap": PHASE16._phase16_text_overlap_issues(row["asset_path"]),
            "text_exceeds_canvas": PHASE16._phase16_text_exceeds_canvas_issues(row["asset_path"]),
        }
        record["estimator_findings"] = estimator_findings
        rendered_png = render_dir / f"{row['asset_path'].stem}_attempt{attempt}.png"
        renderer.render(row["asset_path"], rendered_png)
        critique = _run_critique(
            client=client, args=args, budget=budget,
            rendered_png=rendered_png, source_crop=crop_path, tier=tier,
        )
        similarity = clip_image_similarity(rendered_png, crop_path)
        dinov2_warnings: list[str] = []
        dinov2_similarity = dinov2_image_similarity(
            rendered_png, crop_path, warnings=dinov2_warnings
        )
        # Margin sign test (CALIB-03) over the bound_papers_v1 pool. A degraded
        # chosen-crop score records the all-None margin plus a warning instead
        # of failing the attempt (post-preflight degrade contract, D-02).
        if dinov2_similarity is None:
            margin = {"margin": None, "margin_sign": None, "distractor_count": 0}
            dinov2_warnings.append(
                "margin not computed: dinov2 similarity for the chosen crop degraded to None"
            )
        else:
            distractor_sims = []
            for distractor in margin_pool["distractors"]:
                sim = dinov2_image_similarity(
                    rendered_png, distractor, warnings=dinov2_warnings
                )
                if sim is not None:
                    distractor_sims.append(sim)
            margin = margin_sign_test(dinov2_similarity, distractor_sims)
        margin = {
            **margin,
            "pool_notes": margin_pool["notes"],
            "protocol": margin_pool["protocol"],
        }
        # Coverage (CALIB-02): deterministic from the grounded critique's lists.
        coverage = component_coverage(
            critique.get("matched_components") or [],
            critique.get("missing_components") or [],
            critique.get("invented_components") or [],
        )
        decision = _attempt_gate_decision(
            critique_verdict=critique.get("verdict"),
            coverage=coverage,
            coverage_threshold=args.coverage_threshold,
            margin_sign=margin.get("margin_sign"),
        )
        record.update(
            {
                "critique_verdict": critique.get("verdict"),
                "critique_feedback": critique.get("feedback"),
                "clip_similarity": similarity,
                "dinov2_similarity": dinov2_similarity,
                "coverage": coverage,
                "margin": margin,
                "gate_decision": decision,
                "rendered_png": str(rendered_png),
            }
        )
        if dinov2_warnings:
            record["dinov2_warnings"] = dinov2_warnings
        attempts.append(record)
        candidate = {
            "code": code,
            "critique": critique,
            "similarity": similarity,
            "dinov2_similarity": dinov2_similarity,
            "coverage": coverage,
            "margin": margin,
            "estimator_findings": estimator_findings,
            "gate_decision": decision,
            "rendered_png": rendered_png,
            "attempt": attempt,
        }
        candidates.append(candidate)
        if decision["accept"]:
            best = candidate
            break
        margin_note = (
            "negative margin: render closer to a distractor crop than its claimed source"
            if margin.get("margin_sign") is False
            else None
        )
        feedback = _repair_feedback(critique, estimator_findings, margin_note)

    if best is None and candidates:
        # No attempt passed the gates: best effort = highest coverage
        # (None treated as -1.0); max() keeps the EARLIEST attempt on ties.
        best = max(
            candidates,
            key=lambda c: -1.0 if c["coverage"] is None else c["coverage"],
        )
    if best is None:
        return {
            "status": "failed",
            "reason": "no attempt produced a renderable figure",
            "tier": tier,
            "grounding": grounding.to_dict(),
            "attempts": attempts,
            "schema_version": 3 if adjudication_block is not None else 2,
            "adjudication": adjudication_block,
            "stop_reason": "max_attempts_exhausted",
            **_adjudication_loop_controls(
                attempts=attempts,
                attempt_cap=args.max_attempts,
                budget_projection_id=budget_projection_id,
            ),
        }
    # Restore the best attempt's code/SVG (a later failed attempt may have overwritten them).
    row["code_path"].write_text(best["code"].rstrip() + "\n", encoding="utf-8")
    PHASE16._execute_phase16_generated_code(row["code_path"], row["asset_path"])
    similarity = best["similarity"]
    status = "generated" if best["gate_decision"]["accept"] else "best_effort"
    return {
        "status": status,
        "tier": tier,
        "schema_version": 3 if adjudication_block is not None else 2,
        "adjudication": adjudication_block,
        "stop_reason": "accepted" if status == "generated" else "max_attempts_exhausted",
        "renderer_family": "vision_redraw_svg",
        "asset_path": str(row["asset_path"]),
        "code_path": str(row["code_path"]),
        "svg_sha256": hashlib.sha256(row["asset_path"].read_bytes()).hexdigest(),
        "grounding": grounding.to_dict(),
        "attribution_line": grounding.attribution_line(),
        "clip_similarity": similarity,
        "dinov2_similarity": best["dinov2_similarity"],
        "coverage": best["coverage"],
        "coverage_threshold": args.coverage_threshold,
        "margin": best["margin"],
        "estimator_findings": best["estimator_findings"],
        "gate_decision": best["gate_decision"],
        "similarity_threshold": args.similarity_threshold,
        "similarity_pass": (similarity is not None and similarity >= args.similarity_threshold),
        "critique": best["critique"],
        "attempts": attempts,
        "rendered_png": str(best["rendered_png"]),
        **_adjudication_loop_controls(
            attempts=attempts,
            attempt_cap=args.max_attempts,
            budget_projection_id=budget_projection_id,
        ),
    }


def _wrap_d2_svg(compiled_svg: Path, output_svg: Path, *, heading: str) -> None:
    """Letterbox the compiled D2 SVG into a 1280x720 brand canvas."""

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.parse(compiled_svg)
    inner = tree.getroot()
    view_box = inner.get("viewBox")
    if view_box:
        _, _, vw, vh = (float(v) for v in view_box.split())
    else:
        vw = float(re.sub(r"[a-z]+", "", inner.get("width") or "1280"))
        vh = float(re.sub(r"[a-z]+", "", inner.get("height") or "720"))
    margin_x, top, bottom = 40.0, 64.0, 24.0
    avail_w, avail_h = 1280 - 2 * margin_x, 720 - top - bottom
    scale = min(avail_w / vw, avail_h / vh, 1.5)
    width, height = vw * scale, vh * scale
    inner.set("x", f"{(1280 - width) / 2:.1f}")
    inner.set("y", f"{top + (avail_h - height) / 2:.1f}")
    inner.set("width", f"{width:.1f}")
    inner.set("height", f"{height:.1f}")

    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": "1280", "height": "720", "viewBox": "0 0 1280 720",
    })
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "width": "1280", "height": "720", "fill": "#1a1a2e",
    })
    title = ET.SubElement(root, f"{{{SVG_NS}}}text", {
        "x": "40", "y": "42", "fill": "#a8d8ea", "font-size": "26",
        "font-family": "Helvetica, Arial, sans-serif", "font-weight": "bold",
    })
    title.text = heading
    root.append(inner)
    ET.ElementTree(root).write(output_svg, encoding="unicode", xml_declaration=False)


def _generate_d2_figure(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
    budget: BudgetTracker,
    renderer: SvgRenderer,
    render_dir: Path,
) -> dict[str, Any]:
    header = D2_BRAND_HEADER.read_text(encoding="utf-8")
    d2_path = row["code_path"].with_suffix(".d2")
    compiled_svg = render_dir / f"{row['asset_path'].stem}_d2_raw.svg"
    feedback = ""
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for attempt in range(1, max(1, args.max_attempts) + 1):
        prompt = _d2_prompt(row, feedback=feedback)
        result = anthropic_messages_create(
            client,
            caller=__file__,
            model=args.opus_model,
            max_tokens=3000,
            system="You write D2 diagram source for editorial figures. Return D2 source only.",
            messages=[{"role": "user", "content": prompt}],
        )
        budget.record(result)
        body = _anthropic_text(result).strip()
        body = re.sub(r"^```[a-z0-9]*\n|\n```$", "", body.strip(), flags=re.MULTILINE)
        d2_path.write_text(header + "\n\n" + body + "\n", encoding="utf-8")
        compile_proc = subprocess.run(
            ["d2", "--layout=elk", str(d2_path), str(compiled_svg)],
            capture_output=True, text=True, timeout=120,
        )
        record: dict[str, Any] = {
            "attempt": attempt,
            "provider_receipt_path": getattr(result.receipt, "receipt_path", None),
            "compile_status": "pass" if compile_proc.returncode == 0 else "fail",
        }
        if compile_proc.returncode != 0:
            stderr = (compile_proc.stderr or compile_proc.stdout)[-1200:]
            record["compile_error"] = stderr
            attempts.append(record)
            feedback = f"d2 compile failed:\n{stderr}"
            continue
        _wrap_d2_svg(compiled_svg, row["asset_path"], heading=shorten(row["title"], width=80, placeholder="..."))
        rendered_png = render_dir / f"{row['asset_path'].stem}_attempt{attempt}.png"
        renderer.render(row["asset_path"], rendered_png)
        critique = _run_critique(
            client=client, args=args, budget=budget,
            rendered_png=rendered_png, source_crop=None, tier="d2",
        )
        record["critique_verdict"] = critique.get("verdict")
        attempts.append(record)
        best = {
            "d2_source": d2_path.read_text(encoding="utf-8"),
            "critique": critique,
            "rendered_png": rendered_png,
        }
        if critique.get("verdict") == "pass":
            break
        feedback = str(critique.get("feedback") or "")[:1200]
    if best is None:
        return {"status": "failed", "reason": "d2 compile failed on every attempt", "tier": "d2", "attempts": attempts}
    # Restore the best compiled attempt (a later failed compile may have left
    # stale .d2 source while the asset still holds the last good render).
    d2_path.write_text(best["d2_source"], encoding="utf-8")
    subprocess.run(["d2", "--layout=elk", str(d2_path), str(compiled_svg)], capture_output=True, text=True, timeout=120)
    _wrap_d2_svg(compiled_svg, row["asset_path"], heading=shorten(row["title"], width=80, placeholder="..."))
    return {
        "status": "generated" if best["critique"].get("verdict") == "pass" else "best_effort",
        "tier": "d2",
        "renderer_family": "d2",
        "asset_path": str(row["asset_path"]),
        "code_path": str(d2_path),
        "svg_sha256": hashlib.sha256(row["asset_path"].read_bytes()).hexdigest(),
        "critique": best["critique"],
        "attempts": attempts,
        "rendered_png": str(best["rendered_png"]),
    }


def _wrap_png_in_svg(png_path: Path, output_svg: Path) -> None:
    data = base64.standard_b64encode(png_path.read_bytes()).decode("ascii")
    output_svg.write_text(
        f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" width="1280" height="720" '
        f'viewBox="0 0 1280 720"><image width="1280" height="720" '
        f'xlink:href="data:image/png;base64,{data}"/></svg>\n',
        encoding="utf-8",
    )


def _generate_html_figure(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
    budget: BudgetTracker,
    renderer: SvgRenderer,
    render_dir: Path,
) -> dict[str, Any]:
    css = BRAND_CSS.read_text(encoding="utf-8")
    html_path = row["code_path"].with_suffix(".html")
    feedback = ""
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, args.max_attempts) + 1):
        prompt = _html_prompt(row, feedback=feedback)
        result = anthropic_messages_create(
            client,
            caller=__file__,
            model=args.opus_model,
            max_tokens=4000,
            system="You write HTML for branded 1280x720 editorial figures. Return HTML only.",
            messages=[{"role": "user", "content": prompt}],
        )
        budget.record(result)
        body = _anthropic_text(result).strip()
        body = re.sub(r"^```[a-z0-9]*\n|\n```$", "", body.strip(), flags=re.MULTILINE)
        if "<script" in body.lower() or "<img" in body.lower():
            feedback = "remove all <script> and <img> tags; only brand-kit styled markup is allowed"
            attempts.append({"attempt": attempt, "error": "disallowed tags"})
            continue
        html_path.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{css}</style></head><body>{body}</body></html>",
            encoding="utf-8",
        )
        rendered_png = render_dir / f"{row['asset_path'].stem}_attempt{attempt}.png"
        renderer.render(html_path, rendered_png)
        critique = _run_critique(
            client=client, args=args, budget=budget,
            rendered_png=rendered_png, source_crop=None, tier="html_css",
        )
        attempts.append({
            "attempt": attempt,
            "provider_receipt_path": getattr(result.receipt, "receipt_path", None),
            "critique_verdict": critique.get("verdict"),
        })
        if critique.get("verdict") == "pass" or attempt == args.max_attempts:
            _wrap_png_in_svg(rendered_png, row["asset_path"])
            return {
                "status": "generated" if critique.get("verdict") == "pass" else "best_effort",
                "tier": "html_css",
                "renderer_family": "html_css",
                "asset_path": str(row["asset_path"]),
                "code_path": str(html_path),
                "svg_sha256": hashlib.sha256(row["asset_path"].read_bytes()).hexdigest(),
                "critique": critique,
                "attempts": attempts,
                "rendered_png": str(rendered_png),
            }
        feedback = str(critique.get("feedback") or "")[:1200]
    return {"status": "failed", "reason": "html figure failed every attempt", "tier": "html_css", "attempts": attempts}


# ---------------------------------------------------------------------------
# Comparison page + main
# ---------------------------------------------------------------------------


TIER_RENDERER_FAMILY = {
    "vision_redraw": "vision_redraw_svg",
    "data_redraw": "vision_redraw_svg",
    "d2": "d2",
    "html_css": "html_css",
}


def _design_brief_v2(row: dict[str, Any], outcome: dict[str, Any], *, blueprint_id: str) -> dict[str, Any]:
    tier = str(outcome.get("tier"))
    brief = {
        "blueprint_id": blueprint_id,
        "slide_id": row["slide_id"],
        "cluster_id": (row.get("cluster_ids") or [""])[0],
        "source_labels": row["source_labels"],
        "claim_to_visualize": row["claim"],
        "depiction_type": tier,
        "renderer_family": str(outcome.get("renderer_family") or TIER_RENDERER_FAMILY.get(tier, "")),
        "required_labels": row["required_entities"],
        "forbidden_patterns": row["forbidden_patterns"],
        "narration_hook": row["narration_hook"],
    }
    if outcome.get("grounding"):
        grounding = dict(outcome["grounding"])
        grounding["crop_published"] = False
        brief["grounding"] = grounding
        brief["grounding_mode"] = grounding.get("grounding_mode")
    return brief


def _comparison_page(results: list[dict[str, Any]], deck_dir: Path, output: Path) -> None:
    rows_html = []
    for item in results:
        outcome = item["outcome"]
        old_rel = item.get("existing_asset") or ""
        new_path = outcome.get("asset_path")
        new_rel = str(Path(new_path).relative_to(deck_dir)) if new_path else ""
        similarity = outcome.get("clip_similarity")
        sim_text = f"{similarity:.3f}" if isinstance(similarity, (int, float)) else "n/a"
        grounding = outcome.get("grounding") or {}
        rows_html.append(f"""
  <section class="slide">
    <h2>{html_module.escape(item['slide_id'])} — tier {html_module.escape(str(outcome.get('tier')))}
        — status {html_module.escape(str(outcome.get('status')))}</h2>
    <p class="meta">CLIP similarity: {sim_text} | critique: {html_module.escape(str((outcome.get('critique') or {}).get('verdict', 'n/a')))}
       | source: {html_module.escape(str(grounding.get('paper_title', '')))}</p>
    <div class="pair">
      <figure><figcaption>Current (phase16)</figcaption>{f'<img src="{html_module.escape(old_rel)}"/>' if old_rel else '<p>none</p>'}</figure>
      <figure><figcaption>Trial v2</figcaption>{f'<img src="{html_module.escape(new_rel)}"/>' if new_rel else f"<p>{html_module.escape(str(outcome.get('reason', 'not generated')))}</p>"}</figure>
    </div>
  </section>""")
    output.write_text(
        """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Figure trial v2 comparison</title>
<style>
 body { background:#0f0f1d; color:#eee; font-family: Helvetica, Arial, sans-serif; margin: 24px; }
 h1 { color:#a8d8ea; } h2 { color:#ffd43b; font-size:18px; }
 .meta { color:#aaa; font-size:14px; }
 .pair { display:flex; gap:16px; }
 figure { flex:1; margin:0; } figcaption { color:#a8d8ea; font-size:13px; margin-bottom:4px; }
 img { width:100%; border:1px solid #333355; background:#1a1a2e; }
 section { margin-bottom:36px; border-bottom:1px solid #333355; padding-bottom:18px; }
</style></head><body>
<h1>Figure trial v2 — side-by-side comparison</h1>
"""
        + "\n".join(rows_html)
        + "\n</body></html>\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.live:
        print("refusing to run without --live (this trial makes paid provider calls)", file=sys.stderr)
        return 2

    deck_dir = args.deck_dir.resolve()
    workspace = args.workspace.resolve()

    receipt_out = _resolve_receipt_out(args.receipt_out, deck_dir)
    comparison_out = _resolve_comparison_out(args.comparison_out, receipt_out)

    # Protected-path refusal (T-02-01): the immutable trial/replay receipts
    # and the trial comparison page can never be batch output targets, even
    # under --force-overwrite-receipt. Runs in the pre-spend window.
    refusal = _protected_receipt_refusal(
        receipt_out, comparison_out, deck_dir, force=args.force_overwrite_receipt
    )
    if refusal is not None:
        print(f"[trial] {refusal}", file=sys.stderr)
        return 2

    # D-15 resume validation, inside the pre-spend window (exit 1 on any
    # failure): the prior receipt must exist, parse, and carry a results
    # list. When the resume source IS the receipt-out target, archive the
    # halted receipt (rename, original preserved with its sha recorded in
    # resumed_from) BEFORE the CR-01 exists-check below — the only path
    # that may bypass CR-01, and only because nothing is destroyed.
    prior_receipt: dict[str, Any] | None = None
    prior_receipt_bytes: bytes | None = None
    resume_path: Path | None = None
    if args.resume_from_receipt is not None:
        resume_path = _resolve_receipt_out(args.resume_from_receipt, deck_dir)
        if not resume_path.is_file():
            print(f"[trial] resume receipt not found: {resume_path}", file=sys.stderr)
            return 1
        prior_receipt_bytes = resume_path.read_bytes()
        try:
            prior_receipt = json.loads(prior_receipt_bytes.decode("utf-8"))
        except ValueError as exc:
            print(
                f"[trial] resume receipt is not valid JSON: {resume_path} ({exc})",
                file=sys.stderr,
            )
            return 1
        if not isinstance(prior_receipt, dict) or not isinstance(
            prior_receipt.get("results"), list
        ):
            print(
                f"[trial] resume receipt has no results list: {resume_path}",
                file=sys.stderr,
            )
            return 1
        if resume_path.resolve() == receipt_out.resolve():
            archived = _archive_halted_receipt(resume_path)
            print(f"[trial] archived halted receipt to {archived}", file=sys.stderr)

    # CR-01: refuse to overwrite an existing receipt at the RESOLVED
    # --receipt-out path BEFORE the DINOv2 preflight and `import anthropic` —
    # i.e., before any provider spend. Guarding at write time (the
    # 01-REVIEW.md suggestion) would discard a fully paid run's receipt;
    # guarding before spend preserves both the provenance record and the
    # budget — and a re-run cannot silently clobber a paid batch receipt
    # either. Exit 5 is distinct from the existing codes (2=config error or
    # protected-path refusal, 3=budget halt, 4=DINOv2 preflight failure).
    refusal = _receipt_overwrite_refusal(receipt_out, args.force_overwrite_receipt)
    if refusal is not None:
        print(f"[trial] {refusal}", file=sys.stderr)
        return 5

    deck_contract = json.loads((deck_dir / "presentation_deck.json").read_text(encoding="utf-8"))
    source_binding = json.loads((deck_dir / "source_binding_report.json").read_text(encoding="utf-8"))
    visual_plan = json.loads((deck_dir / "visual_plan.json").read_text(encoding="utf-8"))
    trial_config = json.loads(args.trial_config.read_text(encoding="utf-8"))
    tier_by_slide = {
        str(entry["slide_id"]): str(entry["tier"])
        for entry in trial_config.get("slides", [])
        if str(entry.get("tier")) in TIERS
    }
    if not tier_by_slide:
        print("trial config selected no valid slides", file=sys.stderr)
        return 2

    trial_dir = deck_dir / "assets" / "generated" / "trial-v2"
    code_dir = trial_dir / "code"
    render_dir = trial_dir / "renders"
    for path in (trial_dir, code_dir, render_dir):
        path.mkdir(parents=True, exist_ok=True)
    grounding_root = workspace / "cache" / "figure-grounding"

    row_args = SimpleNamespace(generated_dir=trial_dir, figure_code_dir=code_dir, workspace=workspace)
    rows = PHASE16._phase16_figure_rows(
        deck_contract=deck_contract, source_binding=source_binding,
        visual_plan=visual_plan, args=row_args,
    )
    rows = [row for row in rows if row["slide_id"] in tier_by_slide]
    adjudication_by_slide = _load_adjudication_decisions(args.adjudication_decisions)
    missing = sorted(set(tier_by_slide) - {row["slide_id"] for row in rows})
    if missing:
        print(f"warning: trial slides not found in visual plan: {missing}", file=sys.stderr)
    if prior_receipt is not None:
        # D-15: skip exactly the terminal-status slides from the halted run
        # (same filter shape as the config filter above) — no re-spend on a
        # completed slide; a mid-attempt slide is absent from the prior
        # results and restarts from its own beginning.
        done = _resume_skip_set(prior_receipt)
        skipped_done = sorted({row["slide_id"] for row in rows} & done)
        rows = [row for row in rows if row["slide_id"] not in done]
        print(
            f"[trial] resume: skipping {len(skipped_done)} completed slides; "
            f"{len(rows)} remaining",
            file=sys.stderr,
        )

    # DINOv2 preflight (D-02): hard-fail BEFORE the Anthropic client exists —
    # no provider spend on a run whose primary metric cannot be computed.
    try:
        dinov2_info = preflight_dinov2()
    except RuntimeError as exc:
        print(f"[trial] {exc}", file=sys.stderr)
        return 4
    print(
        f"[trial] dinov2 preflight ok: device={dinov2_info['device']} "
        f"revision={dinov2_info['revision']}",
        file=sys.stderr,
    )

    import anthropic

    client = anthropic.Anthropic(timeout=args.anthropic_timeout_seconds)
    budget = BudgetTracker(args.budget_cap_usd)
    renderer = SvgRenderer()
    blueprint_id = str(deck_contract.get("blueprint_id") or deck_contract.get("presentation_id") or "")

    results: list[dict[str, Any]] = []
    halted: str | None = None
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        with provider_receipt_context(
            workspace_root=PROJECT_ROOT, stage_id=args.stage_id, caller=__file__,
        ):
            for row in rows:
                tier = tier_by_slide[row["slide_id"]]
                adjudication_record = adjudication_by_slide.get(row["slide_id"])
                adjudication_route = (
                    _route_adjudication_decision(adjudication_record)
                    if adjudication_record is not None
                    else None
                )
                print(f"[trial] {row['slide_id']}: tier={tier}", file=sys.stderr)
                try:
                    if (
                        adjudication_record is not None
                        and adjudication_route is not None
                        and not adjudication_route["requires_provider_attempt"]
                    ):
                        outcome = _adjudication_terminal_outcome(
                            adjudication=adjudication_record,
                            tier=tier,
                            budget_projection_id=args.budget_projection_id,
                        )
                    elif tier in ("vision_redraw", "data_redraw"):
                        outcome = _generate_redraw_figure(
                            row=row, tier=tier, args=args, client=client, budget=budget,
                            renderer=renderer, grounding_root=grounding_root, render_dir=render_dir,
                            adjudication_record=adjudication_record,
                            initial_feedback=(
                                adjudication_route["repair_feedback"]
                                if adjudication_route is not None
                                else ""
                            ),
                            budget_projection_id=args.budget_projection_id,
                        )
                    elif tier == "d2":
                        outcome = _generate_d2_figure(
                            row=row, args=args, client=client, budget=budget,
                            renderer=renderer, render_dir=render_dir,
                        )
                    else:
                        outcome = _generate_html_figure(
                            row=row, args=args, client=client, budget=budget,
                            renderer=renderer, render_dir=render_dir,
                        )
                except TrialBudgetExceeded:
                    raise
                except Exception as exc:  # noqa: BLE001 — one slide must not sink the trial
                    outcome = {"status": "failed", "reason": str(exc)[:500], "tier": tier}
                if adjudication_route is not None:
                    outcome.setdefault("schema_version", 3)
                    outcome.setdefault("adjudication", adjudication_route["adjudication"])
                    outcome.setdefault("stop_reason", adjudication_route["stop_reason"])
                    outcome.setdefault("automatic_success", adjudication_route["automatic_success"])
                    outcome.setdefault(
                        "excluded_from_automatic_success",
                        adjudication_route["excluded_from_automatic_success"],
                    )
                brief = _design_brief_v2(row, outcome, blueprint_id=blueprint_id)
                brief_errors: tuple = ()
                if outcome.get("status") in ("generated", "best_effort"):
                    brief_errors = validate_figure_design_brief_v2(brief)
                old_asset = None
                for path in (deck_dir / "assets" / "generated" / "phase16").glob(f"{row['slide_id']}*.svg"):
                    old_asset = str(path.relative_to(deck_dir))
                    break
                results.append(
                    {
                        "slide_id": row["slide_id"],
                        "tier": tier,
                        "outcome": outcome,
                        "figure_design_brief": brief,
                        "brief_validation_errors": list(brief_errors),
                        "existing_asset": old_asset,
                    }
                )
    except TrialBudgetExceeded as exc:
        halted = str(exc)
        print(f"[trial] HALTED: {exc}", file=sys.stderr)
    finally:
        renderer.close()

    if prior_receipt is not None:
        # D-15 merge: prior terminal entries verbatim, ahead of this run's
        # results; cumulative totals span both runs.
        results = _merge_resumed_results(prior_receipt.get("results", []), results)
    spend_usd, call_count = _cumulative_spend(prior_receipt, budget.spent_usd, budget.calls)

    receipt = {
        # schema_version 2: outcome entries gained required gate-tuple fields
        # (coverage, dinov2_similarity, margin, estimator_findings,
        # gate_decision). The schema-1 trial-v2 receipt on disk is NEVER
        # rewritten — it stays the immutable provenance record.
        "schema_version": 3 if adjudication_by_slide else 2,
        "stage_id": args.stage_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "deck_dir": str(deck_dir),
        "budget_cap_usd": args.budget_cap_usd,
        "estimated_spend_usd": spend_usd,
        "provider_call_count": call_count,
        "budget_halt": halted,
        "models": {
            "codegen": args.opus_model,
            "critique": args.critique_model,
            "ranker": args.ranker_model,
        },
        "dinov2": dinov2_runtime_info(),
        "similarity_threshold": args.similarity_threshold,
        "coverage_threshold": args.coverage_threshold,
        "claim_match_threshold": args.claim_match_threshold,
        "margin_pool_protocol": "bound_papers_v1",
        "results": results,
    }
    if adjudication_by_slide:
        receipt["adjudication_policy"] = {
            "decisions_path": str(args.adjudication_decisions),
            "budget_projection_id": args.budget_projection_id,
            "decision_count": len(adjudication_by_slide),
        }
    if prior_receipt is not None and prior_receipt_bytes is not None and resume_path is not None:
        receipt["resumed_from"] = _resumed_from_block(
            resume_path, prior_receipt_bytes, prior_receipt
        )
    receipt_path = receipt_out
    write_json_atomic(receipt_path, receipt)
    comparison_path = comparison_out
    _comparison_page(results, deck_dir, comparison_path)

    generated = sum(1 for item in results if item["outcome"].get("status") in ("generated", "best_effort"))
    print(
        f"[trial] done: {generated}/{len(results)} figures, "
        f"${budget.spent_usd:.2f} of ${args.budget_cap_usd:.2f} spent, "
        f"receipt: {receipt_path}, comparison: {comparison_path}",
        file=sys.stderr,
    )
    return 0 if halted is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
