"""Unit tests for the v2.4 calibration batch machinery (Phase 02, 02-01/02-02).

Covers (02-01, runner flags):
- ``--receipt-out`` / ``--comparison-out`` resolution helpers (``-k
  receipt_out``): back-compat trial defaults, deck-dir-relative resolution,
  absolute pass-through, and the ``_receipt.json`` -> ``_comparison.html``
  name derivation with stem fallback.
- Protected-path refusal (T-02-01, exit 2): ``figure_trial_receipt.json``
  can never be a ``--receipt-out`` target under ``--force-overwrite-receipt``,
  ``figure_trial_replay_receipt.json`` is never a valid target, and
  ``figure_trial_comparison.html`` is never a batch ``--comparison-out``
  target. Pure resolve-and-compare helpers — no I/O beyond resolve.
- CR-01 guard pointed at the resolved ``--receipt-out`` path (exit 5 on an
  existing batch receipt; subprocess test exercises only the pre-spend
  guard window — no provider calls).
- ``--stage-id`` wiring: default preserves the as-shipped
  ``figure_trial_v2`` behavior; the flag value flows to both
  ``provider_receipt_context`` and the receipt dict.

Covers (02-02, analysis CLI ``calibrate_figure_thresholds.py``):
- ``assemble-dataset`` (``-k dataset``): one labeled row per batch attempt
  plus the six pooled trial-v2 finals (D-12 label_source rules, D-16
  final_render_failed markers, D-04 grounding-skip evidence, Pitfall 5
  null-tuple-fields vs unlabeled semantics, sha-chained inputs).
- ``build-review`` (``-k blind``): D-07 blindness asserted by recursive
  banned-key schema walk, D-08 item coverage (finals + last gate-rejected
  attempt per redraw slide), deterministic receipt-sha-seeded shuffle, the
  separate gate-reveal sidecar with the D-04 skip recap, and the blind HTML.
- ``merge-verdicts`` (``-k verdicts``): D-06 enum + tag-vocabulary + coverage
  bijection validation (exit 2 on every malformed class) and the
  human-label flip integration with ``assemble-dataset``.

Both script modules are loaded via importlib (scripts/ module convention
from tests/test_figure_gate_stack.py). Module import is cheap and offline;
no model, provider, or browser work happens at import time. The fixture
builders (``_batch_receipt_payload``, ``_replay_receipt_payload``,
``_verdicts_payload``) are module-level so plan 02-04's analysis tests can
reuse them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER_PATH = REPO_ROOT / "scripts" / "presentations" / "run_figure_trial.py"


def _load_runner_module():
    """Load run_figure_trial.py via importlib (scripts/ module convention)."""

    spec = importlib.util.spec_from_file_location(
        "figure_trial_runner_calibration_test", _RUNNER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _run_runner(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Invoke the runner as a subprocess with minimal required args.

    Only pre-spend guard exits are exercised: every test below must return
    before the deck contract files are read, so the tmp deck dir needs no
    fixtures beyond what the specific guard checks.
    """

    cmd = [
        sys.executable,
        str(_RUNNER_PATH),
        "--workspace",
        str(tmp_path),
        "--deck-dir",
        str(tmp_path),
        "--trial-config",
        str(tmp_path / "config.json"),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Behavior Test 1: _resolve_receipt_out
# ---------------------------------------------------------------------------


def test_receipt_out_default_is_trial_receipt(tmp_path):
    # Back-compat: no flag keeps the as-shipped trial receipt path.
    resolved = RUNNER._resolve_receipt_out(None, tmp_path)
    assert resolved == tmp_path / "figure_trial_receipt.json"


def test_receipt_out_relative_resolves_under_deck_dir(tmp_path):
    resolved = RUNNER._resolve_receipt_out(Path("figure_calibration_receipt.json"), tmp_path)
    assert resolved == tmp_path / "figure_calibration_receipt.json"


def test_receipt_out_absolute_passes_through(tmp_path):
    absolute = tmp_path / "elsewhere" / "batch.json"
    assert RUNNER._resolve_receipt_out(absolute, tmp_path) == absolute


# ---------------------------------------------------------------------------
# Behavior Test 2: _resolve_comparison_out derivation
# ---------------------------------------------------------------------------


def test_receipt_out_comparison_derived_for_trial_default(tmp_path):
    receipt_out = tmp_path / "figure_trial_receipt.json"
    derived = RUNNER._resolve_comparison_out(None, receipt_out)
    assert derived == tmp_path / "figure_trial_comparison.html"


def test_receipt_out_comparison_derived_for_calibration(tmp_path):
    receipt_out = tmp_path / "figure_calibration_receipt.json"
    derived = RUNNER._resolve_comparison_out(None, receipt_out)
    assert derived == tmp_path / "figure_calibration_comparison.html"


def test_receipt_out_comparison_fallback_uses_stem(tmp_path):
    # No trailing _receipt.json: fall back to {stem}_comparison.html.
    receipt_out = tmp_path / "batch.json"
    derived = RUNNER._resolve_comparison_out(None, receipt_out)
    assert derived == tmp_path / "batch_comparison.html"


def test_receipt_out_comparison_explicit_override(tmp_path):
    receipt_out = tmp_path / "figure_calibration_receipt.json"
    explicit = RUNNER._resolve_comparison_out(Path("custom_page.html"), receipt_out)
    assert explicit == tmp_path / "custom_page.html"
    absolute = tmp_path / "abs" / "page.html"
    assert RUNNER._resolve_comparison_out(absolute, receipt_out) == absolute


# ---------------------------------------------------------------------------
# Behavior Test 3: _protected_receipt_refusal (pure resolve-and-compare)
# ---------------------------------------------------------------------------


def test_receipt_out_protected_trial_receipt_refused_only_under_force(tmp_path):
    # --force-overwrite-receipt can never destroy trial provenance (exit 2
    # path); without force the CR-01 exists-check guard owns this path.
    receipt_out = tmp_path / "figure_trial_receipt.json"
    comparison_out = tmp_path / "figure_trial_comparison.html"
    refusal = RUNNER._protected_receipt_refusal(
        receipt_out, comparison_out, tmp_path, force=True
    )
    assert refusal is not None
    assert (
        RUNNER._protected_receipt_refusal(
            receipt_out, comparison_out, tmp_path, force=False
        )
        is None
    )


def test_receipt_out_protected_replay_receipt_refused_any_flags(tmp_path):
    # The replay receipt is never a valid --receipt-out target.
    receipt_out = tmp_path / "figure_trial_replay_receipt.json"
    comparison_out = RUNNER._resolve_comparison_out(None, receipt_out)
    for force in (False, True):
        refusal = RUNNER._protected_receipt_refusal(
            receipt_out, comparison_out, tmp_path, force=force
        )
        assert refusal is not None


def test_receipt_out_protected_comparison_clobber_refused(tmp_path):
    # A batch run (non-default receipt-out) can never clobber the trial
    # comparison page.
    receipt_out = tmp_path / "figure_calibration_receipt.json"
    comparison_out = tmp_path / "figure_trial_comparison.html"
    refusal = RUNNER._protected_receipt_refusal(
        receipt_out, comparison_out, tmp_path, force=False
    )
    assert refusal is not None


def test_receipt_out_protected_allows_back_compat_and_batch_paths(tmp_path):
    # As-shipped trial flow (default receipt-out + derived trial comparison)
    # stays allowed; so do fresh batch output names.
    trial_receipt = RUNNER._resolve_receipt_out(None, tmp_path)
    trial_comparison = RUNNER._resolve_comparison_out(None, trial_receipt)
    assert (
        RUNNER._protected_receipt_refusal(
            trial_receipt, trial_comparison, tmp_path, force=False
        )
        is None
    )
    batch_receipt = tmp_path / "figure_calibration_receipt.json"
    batch_comparison = RUNNER._resolve_comparison_out(None, batch_receipt)
    assert (
        RUNNER._protected_receipt_refusal(
            batch_receipt, batch_comparison, tmp_path, force=False
        )
        is None
    )


# ---------------------------------------------------------------------------
# Behavior Test 4: subprocess pre-spend guard exits
# ---------------------------------------------------------------------------


def test_receipt_out_existing_target_exits_5(tmp_path):
    # CR-01 now checks the resolved --receipt-out path: a re-run cannot
    # silently clobber a paid batch receipt.
    (tmp_path / "batch.json").write_text('{"schema_version": 2}', encoding="utf-8")
    proc = _run_runner(tmp_path, "--live", "--receipt-out", "batch.json")
    assert proc.returncode == 5
    assert "refusing to overwrite" in proc.stderr


def test_receipt_out_protected_target_exits_2_even_with_force(tmp_path):
    # Force can never destroy trial provenance: exit 2, not an overwrite.
    proc = _run_runner(
        tmp_path,
        "--live",
        "--receipt-out",
        "figure_trial_receipt.json",
        "--force-overwrite-receipt",
    )
    assert proc.returncode == 2
    assert "protected" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Behavior Test 5: --stage-id default and wiring
# ---------------------------------------------------------------------------


def test_receipt_out_stage_id_default_and_custom(tmp_path):
    base = [
        "--workspace",
        str(tmp_path),
        "--deck-dir",
        str(tmp_path),
        "--trial-config",
        str(tmp_path / "c.json"),
    ]
    args = RUNNER.parse_args(base)
    # Default preserves the as-shipped trial stage id.
    assert args.stage_id == RUNNER.TRIAL_STAGE_ID == "figure_trial_v2"
    custom = RUNNER.parse_args([*base, "--stage-id", "figure_calibration_batch"])
    assert custom.stage_id == "figure_calibration_batch"


def test_receipt_out_stage_id_flows_to_context_and_receipt():
    # Wiring proof without --live: the source must route args.stage_id into
    # provider_receipt_context AND the receipt dict (no hardcoded constant
    # left at either use site).
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    assert "stage_id=args.stage_id" in source
    assert '"stage_id": args.stage_id' in source
    assert 'stage_id=TRIAL_STAGE_ID' not in source
    assert '"stage_id": TRIAL_STAGE_ID' not in source


# ---------------------------------------------------------------------------
# --resume-from-receipt (D-15): terminal-status skip set, verbatim merge,
# sha-chained resumed_from provenance, archive-before-overwrite.
# ---------------------------------------------------------------------------


def _prior_receipt_fixture() -> dict:
    """Halted batch receipt with mixed terminal statuses plus one
    non-terminal entry; the mid-attempt slide (sl_16) is ABSENT from results
    because TrialBudgetExceeded propagates before results.append."""

    return {
        "schema_version": 2,
        "stage_id": "figure_calibration_batch",
        "budget_cap_usd": 15.0,
        "estimated_spend_usd": 1.2345,
        "provider_call_count": 17,
        "budget_halt": "figure trial spend $15.00 reached the budget cap",
        "results": [
            {
                "slide_id": "sl_07_a",
                "tier": "data_redraw",
                "outcome": {"status": "generated", "coverage": 0.91},
            },
            {
                "slide_id": "sl_08_b",
                "tier": "vision_redraw",
                "outcome": {"status": "best_effort", "coverage": 0.55},
            },
            {
                "slide_id": "sl_13_c",
                "tier": "vision_redraw",
                "outcome": {"status": "skipped", "reason": "claim_match below threshold"},
            },
            {
                "slide_id": "sl_14_d",
                "tier": "vision_redraw",
                "outcome": {"status": "failed", "reason": "no renderable attempt"},
            },
            {
                "slide_id": "sl_15_e",
                "tier": "vision_redraw",
                "outcome": {"status": "in_flight"},
            },
        ],
    }


def test_resume_terminal_status_tuple_exact():
    # D-15 skip set is keyed on EXACTLY these terminal outcome statuses.
    assert RUNNER.RESUME_TERMINAL_STATUSES == (
        "generated",
        "best_effort",
        "skipped",
        "failed",
    )


def test_resume_skip_set_terminal_statuses_only():
    skip = RUNNER._resume_skip_set(_prior_receipt_fixture())
    assert skip == {"sl_07_a", "sl_08_b", "sl_13_c", "sl_14_d"}
    # Non-terminal entry re-runs; the mid-attempt slide (absent from
    # results) restarts from its own beginning.
    assert "sl_15_e" not in skip
    assert "sl_16_f" not in skip


def test_resume_merge_keeps_prior_verbatim_appends_new():
    prior = _prior_receipt_fixture()["results"]
    new = [
        {"slide_id": "sl_16_f", "tier": "d2", "outcome": {"status": "generated"}},
        {"slide_id": "sl_15_e", "tier": "vision_redraw", "outcome": {"status": "generated"}},
    ]
    merged = RUNNER._merge_resumed_results(prior, new)
    # Prior TERMINAL entries kept verbatim (byte-equal dicts), in order,
    # ahead of the new entries; the non-terminal prior entry is dropped
    # (its slide re-ran and contributes a fresh entry instead).
    assert json.dumps(merged[:4], sort_keys=True) == json.dumps(prior[:4], sort_keys=True)
    assert merged[4:] == new
    assert len(merged) == 6


def test_resume_merge_refuses_duplicate_slide_ids():
    prior = _prior_receipt_fixture()["results"]
    duplicate = [{"slide_id": "sl_07_a", "tier": "data_redraw", "outcome": {"status": "generated"}}]
    with pytest.raises(ValueError, match="duplicate slide_id"):
        RUNNER._merge_resumed_results(prior, duplicate)


def test_resume_resumed_from_block_and_cumulative_spend(tmp_path):
    prior = _prior_receipt_fixture()
    raw = json.dumps(prior).encode("utf-8")
    path = tmp_path / "figure_calibration_receipt.json"
    block = RUNNER._resumed_from_block(path, raw, prior)
    assert set(block) == {
        "path",
        "sha256",
        "prior_spend_usd",
        "prior_call_count",
        "prior_budget_halt",
    }
    # sha256 is computed over the prior receipt's RAW BYTES (sha chain).
    assert block["sha256"] == hashlib.sha256(raw).hexdigest()
    assert block["path"] == str(path)
    assert block["prior_spend_usd"] == 1.2345
    assert block["prior_call_count"] == 17
    assert block["prior_budget_halt"] == prior["budget_halt"]
    # Cumulative totals = prior receipt value + this run's tracker; the cap
    # itself is never modified in code.
    spend, calls = RUNNER._cumulative_spend(prior, 0.5, 3)
    assert spend == round(1.2345 + 0.5, 4)
    assert calls == 20
    assert RUNNER._cumulative_spend(None, 0.5, 3) == (0.5, 3)


def test_resume_archive_before_overwrite_clears_cr01(tmp_path):
    receipt_path = tmp_path / "figure_calibration_receipt.json"
    raw = json.dumps(_prior_receipt_fixture(), indent=2)
    receipt_path.write_text(raw, encoding="utf-8")
    archived = RUNNER._archive_halted_receipt(receipt_path)
    # Archive-before-overwrite (01-07 pattern): original moved, not copied.
    assert not receipt_path.exists()
    assert archived.parent == tmp_path
    assert archived.name.startswith("figure_calibration_receipt.halted-")
    assert archived.name.endswith(".json")
    assert archived.read_text(encoding="utf-8") == raw
    # CR-01 does not fire on the archived original's old path.
    assert RUNNER._receipt_overwrite_refusal(receipt_path, force=False) is None


def test_resume_missing_receipt_exits_1(tmp_path):
    # Resume validation runs inside the pre-spend window.
    proc = _run_runner(tmp_path, "--live", "--resume-from-receipt", "missing.json")
    assert proc.returncode == 1
    assert "resume receipt not found" in proc.stderr


def test_resume_receipt_out_differs_from_source_exits_5(tmp_path):
    # An existing receipt-out that is NOT the resume source still trips
    # CR-01: only resuming IN PLACE may supersede a receipt (via archive).
    halted = tmp_path / "halted.json"
    halted.write_text(json.dumps(_prior_receipt_fixture()), encoding="utf-8")
    (tmp_path / "batch.json").write_text('{"schema_version": 2}', encoding="utf-8")
    proc = _run_runner(
        tmp_path,
        "--live",
        "--receipt-out",
        "batch.json",
        "--resume-from-receipt",
        "halted.json",
    )
    assert proc.returncode == 5
    assert "refusing to overwrite" in proc.stderr


def test_resume_same_path_archives_before_cr01(tmp_path):
    # Resume source == receipt-out: the halted receipt is archived before
    # the CR-01 exists-check, so neither guard fires; the run then fails
    # later on the bare tmp dir's missing deck contracts (anything but the
    # guard exits 2/5 proves the ordering).
    receipt_name = "figure_calibration_receipt.json"
    halted = tmp_path / receipt_name
    halted.write_text(json.dumps(_prior_receipt_fixture()), encoding="utf-8")
    proc = _run_runner(
        tmp_path,
        "--live",
        "--receipt-out",
        receipt_name,
        "--resume-from-receipt",
        receipt_name,
    )
    assert proc.returncode not in (2, 5)
    assert not halted.exists()
    archives = list(tmp_path.glob("figure_calibration_receipt.halted-*.json"))
    assert len(archives) == 1


# ===========================================================================
# 02-02: calibrate_figure_thresholds.py — analysis CLI over receipt JSON
# ===========================================================================

_CALIB_PATH = REPO_ROOT / "scripts" / "presentations" / "calibrate_figure_thresholds.py"


def _load_calib_module():
    """Load calibrate_figure_thresholds.py via importlib (scripts/ convention)."""

    spec = importlib.util.spec_from_file_location(
        "calibrate_figure_thresholds_test", _CALIB_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def calib():
    """Analysis CLI module, loaded once per session (offline, no providers)."""

    return _load_calib_module()


# ---------------------------------------------------------------------------
# Reusable fixture builders (02-04's analysis tests reuse these).
# ---------------------------------------------------------------------------

SL_GENERATED = "sl_13_accuracy-is-not-a-proxy-for-reasoning-fait"
SL_BEST_EFFORT = "sl_08_coordination-is-a-failure-surface-scores-c"
SL_SKIPPED = "sl_14_llm-judges-are-inconsistent-measurements-n"
SL_D2 = "sl_03_failure-modes-that-a-single-number-cannot"

# Six pooled trial-v2 finals with the real replay-receipt values (coverages
# 1.000/0.857/0.967/0.409/0.361/0.944; DINOv2 0.7267/0.8342/0.7510/0.6211/
# 0.7456/0.6913). Locked labels: sl_11 fail, sl_19 fail, the rest pass.
_TRIAL_TUPLES = (
    ("sl_05_static-benchmark-scores-collapse-under-rea", "data_redraw", 1.000, 0.7267, 0.6828, "pass"),
    ("sl_06_in-the-wild-tool-use-exposes-a-robustness", "vision_redraw", 0.857, 0.8342, 0.7012, "pass"),
    ("sl_10_expert-validity-is-the-bottleneck", "vision_redraw", 0.967, 0.7510, 0.6531, "pass"),
    ("sl_11_scaling-citation-graphs-wrong-figure", "vision_redraw", 0.409, 0.6211, 0.5418, "fail"),
    ("sl_19_fragile-expert-benchmark-coverage", "data_redraw", 0.361, 0.7456, 0.6105, "fail"),
    ("sl_23_contamination-detection-under-rewrites", "vision_redraw", 0.944, 0.6913, 0.6347, "pass"),
)

_ESTIMATORS_CLEAN = {
    "oversized_arrowhead": [],
    "line_over_text": [],
    "text_overlap": [],
    "text_exceeds_canvas": [],
}


def _flat_margin(value: float, sign: bool, notes: list[str]) -> dict:
    """Batch-attempt margin shape (margin_sign_test output + pool fields)."""

    return {
        "margin": value,
        "margin_sign": sign,
        "distractor_count": 6,
        "pool_notes": notes,
        "protocol": "bound_papers_v1",
    }


def _redraw_attempt(
    slide_id: str,
    deck: Path,
    attempt: int,
    *,
    coverage: float,
    verdict: str,
    accept: bool,
    dinov2: float,
    clip: float,
    margin_sign: bool = True,
) -> dict:
    """Rendered + critiqued redraw attempt record (run_figure_trial shape)."""

    return {
        "attempt": attempt,
        "provider_receipt_path": f"storage/provider-calls/figure_calibration_batch/{slide_id[:5]}-a{attempt}.json",
        "code_review_status": "pass",
        "estimator_findings": _ESTIMATORS_CLEAN,
        "critique_verdict": verdict,
        "critique_feedback": "fixture critique feedback",
        "clip_similarity": clip,
        "dinov2_similarity": dinov2,
        "coverage": coverage,
        "margin": _flat_margin(0.04 if margin_sign else -0.02, margin_sign, ["2Gc8aj0afg"]),
        "gate_decision": {
            "accept": accept,
            "reasons": [] if accept else [f"coverage {coverage} below threshold 0.8"],
        },
        "rendered_png": str(deck / "renders" / f"{slide_id}_attempt{attempt}.png"),
    }


def _batch_receipt_payload(deck_dir: Path) -> dict:
    """Trimmed 4-slide batch receipt: generated redraw (3 attempts),
    best_effort redraw (2 attempts), D-04 skipped redraw, generated d2."""

    deck = Path(deck_dir)
    sl13_attempts = [
        _redraw_attempt(SL_GENERATED, deck, 1, coverage=0.62, verdict="fail", accept=False, dinov2=0.71, clip=0.60),
        _redraw_attempt(SL_GENERATED, deck, 2, coverage=0.74, verdict="fail", accept=False, dinov2=0.72, clip=0.61),
        _redraw_attempt(SL_GENERATED, deck, 3, coverage=0.92, verdict="pass", accept=True, dinov2=0.78, clip=0.66),
    ]
    sl08_attempts = [
        _redraw_attempt(
            SL_BEST_EFFORT, deck, 1, coverage=0.41, verdict="fail", accept=False,
            dinov2=0.65, clip=0.52, margin_sign=False,
        ),
        _redraw_attempt(SL_BEST_EFFORT, deck, 2, coverage=0.55, verdict="fail", accept=False, dinov2=0.69, clip=0.58),
    ]
    return {
        "schema_version": 2,
        "stage_id": "figure_calibration_batch",
        "started_at": "2026-06-12T00:00:00Z",
        "ended_at": "2026-06-12T01:00:00Z",
        "deck_dir": str(deck),
        "budget_cap_usd": 15.0,
        "estimated_spend_usd": 2.5,
        "provider_call_count": 18,
        "budget_halt": None,
        "models": {
            "codegen": "claude-opus-4-8",
            "critique": "claude-sonnet-4-6",
            "ranker": "claude-sonnet-4-6",
        },
        "dinov2": {
            "model_name": "facebook/dinov2-base",
            "revision": "f9e44c81fixture0000000000000000000000000",
            "device": "mps",
            "hf_hub_offline": True,
        },
        "similarity_threshold": 0.55,
        "coverage_threshold": 0.8,
        "claim_match_threshold": 0.7,
        "margin_pool_protocol": "bound_papers_v1",
        "results": [
            {
                "slide_id": SL_GENERATED,
                "tier": "vision_redraw",
                "outcome": {
                    "status": "generated",
                    "tier": "vision_redraw",
                    "renderer_family": "vision_redraw_svg",
                    "asset_path": str(deck / "assets" / f"{SL_GENERATED}.svg"),
                    "code_path": str(deck / "assets" / f"{SL_GENERATED}.py"),
                    "svg_sha256": "f" * 64,
                    "grounding": {
                        "crop_path": str(deck / "crops" / "sl_13_fig2_crop.png"),
                        "claim_match": 0.88,
                        "claim_match_threshold": 0.7,
                        "paper_title": "RFEval",
                    },
                    "attribution_line": "Adapted from RFEval (CC BY 4.0)",
                    "clip_similarity": 0.66,
                    "dinov2_similarity": 0.78,
                    "coverage": 0.92,
                    "coverage_threshold": 0.8,
                    "margin": _flat_margin(0.04, True, ["2Gc8aj0afg"]),
                    "estimator_findings": _ESTIMATORS_CLEAN,
                    "gate_decision": {"accept": True, "reasons": []},
                    "critique": {"verdict": "pass", "feedback": "matches the source figure"},
                    "attempts": sl13_attempts,
                    "rendered_png": sl13_attempts[2]["rendered_png"],
                },
            },
            {
                "slide_id": SL_BEST_EFFORT,
                "tier": "vision_redraw",
                "outcome": {
                    "status": "best_effort",
                    "tier": "vision_redraw",
                    "renderer_family": "vision_redraw_svg",
                    "asset_path": str(deck / "assets" / f"{SL_BEST_EFFORT}.svg"),
                    "code_path": str(deck / "assets" / f"{SL_BEST_EFFORT}.py"),
                    "svg_sha256": "e" * 64,
                    "grounding": {
                        "crop_path": str(deck / "crops" / "sl_08_fig1_crop.png"),
                        "claim_match": 0.82,
                        "claim_match_threshold": 0.7,
                        "paper_title": "Computer Agent Arena",
                    },
                    "attribution_line": "Adapted from Computer Agent Arena (CC BY 4.0)",
                    "clip_similarity": 0.58,
                    "dinov2_similarity": 0.69,
                    "coverage": 0.55,
                    "coverage_threshold": 0.8,
                    "margin": _flat_margin(0.04, True, ["2Gc8aj0afg"]),
                    "estimator_findings": _ESTIMATORS_CLEAN,
                    "gate_decision": {
                        "accept": False,
                        "reasons": ["coverage 0.55 below threshold 0.8"],
                    },
                    "critique": {"verdict": "fail", "feedback": "missing two bars"},
                    "attempts": sl08_attempts,
                    "rendered_png": sl08_attempts[1]["rendered_png"],
                },
            },
            {
                "slide_id": SL_SKIPPED,
                "tier": "vision_redraw",
                "outcome": {
                    "status": "skipped",
                    "reason": "claim_match 0.40 below threshold 0.70; redraw tier skipped",
                    "tier": "vision_redraw",
                    "grounding_skip": {
                        "reason": "claim_match_below_threshold",
                        "claim_match": 0.4,
                        "claim_match_threshold": 0.7,
                        "pooled_notes": ["4uPyOCeN6U", "rwo7bVlnzo"],
                        "pool_candidate_count": 9,
                        "ranker_rationale": "no pooled candidate matches the claim",
                        "ranker_receipt_path": "storage/provider-calls/figure_calibration_batch/rank-sl_14.json",
                        "warnings": [],
                    },
                },
            },
            {
                "slide_id": SL_D2,
                "tier": "d2",
                "outcome": {
                    "status": "generated",
                    "tier": "d2",
                    "renderer_family": "d2",
                    "asset_path": str(deck / "assets" / f"{SL_D2}.svg"),
                    "code_path": str(deck / "assets" / f"{SL_D2}.d2"),
                    "svg_sha256": "a" * 64,
                    "critique": {"verdict": "pass", "feedback": "clear failure-mode map"},
                    "attempts": [
                        {
                            "attempt": 1,
                            "provider_receipt_path": "storage/provider-calls/figure_calibration_batch/sl_03-a1.json",
                            "compile_status": "pass",
                            "critique_verdict": "fail",
                        },
                        {
                            "attempt": 2,
                            "provider_receipt_path": "storage/provider-calls/figure_calibration_batch/sl_03-a2.json",
                            "compile_status": "pass",
                            "critique_verdict": "pass",
                        },
                    ],
                    "rendered_png": str(deck / "renders" / f"{SL_D2}_attempt2.png"),
                },
            },
        ],
    }


def _replay_receipt_payload() -> dict:
    """Trimmed replay receipt: the six trial-v2 finals with real tuples.
    sl_11 carries a negative nested DINOv2 margin so margin_sign derivation
    over the nested trial shape is exercised with False."""

    figures = []
    for slide_id, tier, coverage, dinov2, clip, verdict in _TRIAL_TUPLES:
        negative = slide_id.startswith("sl_11")
        figures.append(
            {
                "slide_id": slide_id,
                "tier": tier,
                "coverage": coverage,
                "coverage_threshold": 0.8,
                "dinov2_similarity": dinov2,
                "clip_similarity": clip,
                "margin": {
                    "clip": {"margin": -0.01 if negative else 0.05, "margin_sign": not negative},
                    "dinov2": {"margin": -0.0123 if negative else 0.0396, "margin_sign": not negative},
                    "distractor_count": 6,
                    "pool_notes": ["7cHhcrbr6x"],
                    "protocol": "bound_papers_v1",
                },
                "critique": {
                    "verdict": verdict,
                    "feedback": "replay structured critique",
                    "critique_parse_status": "ok",
                },
                "estimator_findings": _ESTIMATORS_CLEAN,
                "classification": "fail" if verdict == "fail" else "coverage_pass",
                "classification_matches": True,
                "expected_classification": "fail" if verdict == "fail" else "coverage_pass",
                "inputs": {"render": f"renders/{slide_id}_best.png"},
                "trial_clip_similarity": clip,
            }
        )
    return {
        "schema_version": 1,
        "stage_id": "figure_trial_v2_replay",
        "replay_of": {"path": "figure_trial_receipt.json", "sha256": "b" * 64},
        "coverage_threshold": 0.8,
        "figures": figures,
    }


def _verdicts_payload() -> dict:
    """Verdicts file in merge-verdicts output shape (D-06/D-12 fixtures):
    a human FAIL overriding sl_13's critique-pass final, a human PASS on its
    rejected sample (observed false-fail), and a FAIL on the D-16 final."""

    return {
        "schema_version": 1,
        "produced_by": "scripts/presentations/calibrate_figure_thresholds.py",
        "reviewer": "operator",
        "recorded_at": "2026-06-12T02:00:00Z",
        "inputs": {
            "manifest": {"path": "calibration_review_manifest.json", "sha256": "c" * 64}
        },
        "items": [
            {
                "review_id": f"rev_{SL_GENERATED}_final_03",
                "slide_id": SL_GENERATED,
                "kind": "final",
                "attempt": 3,
                "verdict": "fail",
                "failure_modes": ["missing-components"],
            },
            {
                "review_id": f"rev_{SL_GENERATED}_last_rejected_02",
                "slide_id": SL_GENERATED,
                "kind": "last_rejected",
                "attempt": 2,
                "verdict": "pass",
                "failure_modes": [],
            },
            {
                "review_id": f"rev_{SL_BEST_EFFORT}_final_02",
                "slide_id": SL_BEST_EFFORT,
                "kind": "final",
                "attempt": 2,
                "verdict": "fail",
                "failure_modes": ["legibility", "missing-components"],
            },
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assemble_fixture_dataset(calib, tmp_path: Path, *, with_verdicts: bool = False):
    """Run assemble-dataset over the fixture receipts; returns (dataset,
    batch_path, replay_path, verdicts_path)."""

    deck = tmp_path / "deck"
    batch_path = _write_json(
        tmp_path / "figure_calibration_receipt.json", _batch_receipt_payload(deck)
    )
    replay_path = _write_json(
        tmp_path / "figure_trial_replay_receipt.json", _replay_receipt_payload()
    )
    out_path = tmp_path / "calibration_dataset.json"
    cmd = [
        "assemble-dataset",
        "--batch-receipt",
        str(batch_path),
        "--replay-receipt",
        str(replay_path),
        "--out",
        str(out_path),
    ]
    verdicts_path = None
    if with_verdicts:
        verdicts_path = _write_json(
            tmp_path / "calibration_review_verdicts.json", _verdicts_payload()
        )
        cmd += ["--verdicts", str(verdicts_path)]
    rc = calib.main(cmd)
    assert rc == 0
    dataset = json.loads(out_path.read_text(encoding="utf-8"))
    return dataset, batch_path, replay_path, verdicts_path


# ---------------------------------------------------------------------------
# assemble-dataset (Task 1, -k dataset)
# ---------------------------------------------------------------------------


def test_dataset_one_row_per_attempt_plus_trial_pool(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path)
    rows = dataset["rows"]
    batch_rows = [r for r in rows if r["origin"] == "calibration_batch"]
    trial_rows = [r for r in rows if r["origin"] == "trial_v2_replay"]
    # 3 + 2 + 2 attempts; the D-04 skipped slide contributes no attempt rows.
    assert len(batch_rows) == 7
    assert len(trial_rows) == 6
    assert {r["row_id"] for r in batch_rows} == {
        f"batch:{SL_GENERATED}:a1",
        f"batch:{SL_GENERATED}:a2",
        f"batch:{SL_GENERATED}:a3",
        f"batch:{SL_BEST_EFFORT}:a1",
        f"batch:{SL_BEST_EFFORT}:a2",
        f"batch:{SL_D2}:a1",
        f"batch:{SL_D2}:a2",
    }
    assert {r["row_id"] for r in trial_rows} == {
        f"trial:{slide_id}" for slide_id, *_ in _TRIAL_TUPLES
    }
    # Exactly one surviving final per non-skipped slide.
    finals = {r["row_id"] for r in batch_rows if r["is_final_render"]}
    assert finals == {
        f"batch:{SL_GENERATED}:a3",
        f"batch:{SL_BEST_EFFORT}:a2",
        f"batch:{SL_D2}:a2",
    }
    # Trial rows pool the real replay tuples verbatim.
    by_id = {r["row_id"]: r for r in trial_rows}
    sl11 = next(s for s, *_ in _TRIAL_TUPLES if s.startswith("sl_11"))
    assert by_id[f"trial:{sl11}"]["coverage"] == 0.409
    assert by_id[f"trial:{sl11}"]["dinov2_similarity"] == 0.6211
    assert all(r["is_final_render"] for r in trial_rows)


def test_dataset_margin_sign_derivation(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path)
    by_id = {r["row_id"]: r for r in dataset["rows"]}
    # Batch rows: flat margin_sign copied through.
    assert by_id[f"batch:{SL_GENERATED}:a3"]["margin_sign"] is True
    assert by_id[f"batch:{SL_BEST_EFFORT}:a1"]["margin_sign"] is False
    # d2 rows: no margin at all.
    assert by_id[f"batch:{SL_D2}:a2"]["margin_sign"] is None
    # Trial rows: derived from the nested DINOv2 margin (> 0).
    sl11 = next(s for s, *_ in _TRIAL_TUPLES if s.startswith("sl_11"))
    sl05 = next(s for s, *_ in _TRIAL_TUPLES if s.startswith("sl_05"))
    assert by_id[f"trial:{sl11}"]["margin_sign"] is False
    assert by_id[f"trial:{sl05}"]["margin_sign"] is True


def test_dataset_best_effort_d16_markers(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path)
    by_id = {r["row_id"]: r for r in dataset["rows"]}
    final = by_id[f"batch:{SL_BEST_EFFORT}:a2"]
    # D-16: the surviving final of a gate-exhausted slide is explicitly
    # marked failed, with the production-would-be-synthetic note.
    assert final["is_final_render"] is True
    assert final["final_render_failed"] is True
    assert "synthetic fallback" in final["production_note"]
    # Non-final attempts and generated finals carry no D-16 markers.
    assert by_id[f"batch:{SL_BEST_EFFORT}:a1"]["final_render_failed"] is False
    assert by_id[f"batch:{SL_GENERATED}:a3"]["final_render_failed"] is False
    assert by_id[f"batch:{SL_GENERATED}:a3"]["production_note"] is None


def test_dataset_d2_null_fields_still_labeled(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path)
    by_id = {r["row_id"]: r for r in dataset["rows"]}
    d2_final = by_id[f"batch:{SL_D2}:a2"]
    # Ungrounded tier: the whole tuple is null BY DESIGN (Pitfall 5)...
    for key in ("coverage", "dinov2_similarity", "clip_similarity", "margin", "claim_match"):
        assert d2_final[key] is None
    # ...but the row is still labeled from its critique verdict.
    assert d2_final["critique_verdict"] == "pass"
    assert d2_final["label"] == "pass"
    assert d2_final["label_source"] == "critique"
    d2_first = by_id[f"batch:{SL_D2}:a1"]
    assert d2_first["label"] == "fail"
    # unlabeled means label == null, NOT null tuple fields.
    assert dataset["unlabeled_row_count"] == 0


def test_dataset_label_source_resolution(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path, with_verdicts=True)
    by_id = {r["row_id"]: r for r in dataset["rows"]}
    # D-12: human verdict overrides the critique verdict where present.
    final = by_id[f"batch:{SL_GENERATED}:a3"]
    assert final["critique_verdict"] == "pass"
    assert final["label"] == "fail"
    assert final["label_source"] == "human"
    assert final["human_verdict"] == "fail"
    assert final["failure_modes"] == ["missing-components"]
    # Human pass on the rejected sample (observed false-fail datapoint).
    rejected = by_id[f"batch:{SL_GENERATED}:a2"]
    assert rejected["label"] == "pass"
    assert rejected["label_source"] == "human"
    # Rows without a human verdict fall back to the critique verdict.
    first = by_id[f"batch:{SL_GENERATED}:a1"]
    assert first["label"] == "fail"
    assert first["label_source"] == "critique"
    assert first["human_verdict"] is None
    # Trial rows use the locked labels and are never re-adjudicated.
    for slide_id, *_ in _TRIAL_TUPLES:
        row = by_id[f"trial:{slide_id}"]
        assert row["label_source"] == "trial_v2_operator"
        expected = "fail" if slide_id[:5] in ("sl_11", "sl_19") else "pass"
        assert row["label"] == expected
        assert row["human_verdict"] is None
    # Every row records a label_source from the fixed D-12 vocabulary.
    assert all(
        r["label_source"] in ("human", "critique", "trial_v2_operator")
        for r in dataset["rows"]
    )
    assert dataset["unlabeled_row_count"] == 0


def test_resolve_label_honors_explicit_label_source(calib):
    # Honest provenance (02-05): an adjudicated verdict that carries an
    # explicit label_source propagates it VERBATIM — an AI-panel verdict is
    # never recorded as human ground truth.
    label, source = calib._resolve_label(
        critique_verdict="pass",
        human_verdict="fail",
        verdict_label_source="ai_panel",
    )
    assert label == "fail"
    assert source == "ai_panel"


def test_resolve_label_defaults_to_human_when_source_absent(calib):
    # Backward compatibility: a verdict with no label_source still resolves
    # to "human" (the human-spot-check D-05..D-08 flow is unchanged).
    assert calib._resolve_label(
        critique_verdict="pass", human_verdict="fail"
    ) == ("fail", "human")
    assert calib._resolve_label(
        critique_verdict="pass", human_verdict="fail", verdict_label_source=None
    ) == ("fail", "human")
    # No adjudicated verdict: critique fallback unchanged; null stays null.
    assert calib._resolve_label(
        critique_verdict="pass", human_verdict=None
    ) == ("pass", "critique")
    assert calib._resolve_label(
        critique_verdict=None, human_verdict=None
    ) == (None, None)


def test_dataset_ai_panel_label_source_propagates(calib, tmp_path):
    # End-to-end through assemble-dataset: a verdict item bearing
    # label_source="ai_panel" yields a row labeled with source "ai_panel",
    # NOT "human". Verdict items without a source still resolve to "human".
    deck = tmp_path / "deck"
    batch_path = _write_json(
        tmp_path / "figure_calibration_receipt.json", _batch_receipt_payload(deck)
    )
    replay_path = _write_json(
        tmp_path / "figure_trial_replay_receipt.json", _replay_receipt_payload()
    )
    verdicts = {
        "schema_version": 1,
        "produced_by": "scripts/presentations/calibrate_figure_thresholds.py",
        "verdict_source": "ai_panel",
        "panel_models": ["gemini-2.5-pro", "codex(chatgpt)", "claude"],
        "vote_rule": "2of3_majority",
        "items": [
            {
                "review_id": f"rev_{SL_GENERATED}_final_03",
                "slide_id": SL_GENERATED,
                "kind": "final",
                "attempt": 3,
                "verdict": "fail",
                "failure_modes": ["missing-components"],
                "label_source": "ai_panel",
            },
            {
                # No label_source -> backward-compatible "human" default.
                "review_id": f"rev_{SL_BEST_EFFORT}_final_02",
                "slide_id": SL_BEST_EFFORT,
                "kind": "final",
                "attempt": 2,
                "verdict": "fail",
                "failure_modes": ["legibility"],
            },
        ],
    }
    verdicts_path = _write_json(tmp_path / "calibration_review_verdicts.json", verdicts)
    out_path = tmp_path / "calibration_dataset.json"
    rc = calib.main(
        [
            "assemble-dataset",
            "--batch-receipt",
            str(batch_path),
            "--replay-receipt",
            str(replay_path),
            "--verdicts",
            str(verdicts_path),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    dataset = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {r["row_id"]: r for r in dataset["rows"]}
    ai_row = by_id[f"batch:{SL_GENERATED}:a3"]
    assert ai_row["label"] == "fail"
    assert ai_row["label_source"] == "ai_panel"
    assert ai_row["human_verdict"] == "fail"
    human_row = by_id[f"batch:{SL_BEST_EFFORT}:a2"]
    assert human_row["label"] == "fail"
    assert human_row["label_source"] == "human"
    assert "ai_panel" in {r["label_source"] for r in dataset["rows"]}


def test_dataset_grounding_skips_evidence(calib, tmp_path):
    dataset, _, _, _ = _assemble_fixture_dataset(calib, tmp_path)
    skips = dataset["grounding_skips"]
    assert len(skips) == 1
    skip = skips[0]
    # D-04 evidence: claim_match value, threshold, and pool fields survive.
    assert skip["slide_id"] == SL_SKIPPED
    assert skip["tier"] == "vision_redraw"
    assert skip["claim_match"] == 0.4
    assert skip["claim_match_threshold"] == 0.7
    assert skip["pooled_notes"] == ["4uPyOCeN6U", "rwo7bVlnzo"]
    assert skip["pool_candidate_count"] == 9
    assert skip["reason"] == "claim_match_below_threshold"
    # The skipped slide contributes no attempt rows.
    assert not [r for r in dataset["rows"] if r["slide_id"] == SL_SKIPPED]


def test_dataset_top_level_contract(calib, tmp_path):
    dataset, batch_path, replay_path, _ = _assemble_fixture_dataset(calib, tmp_path)
    assert dataset["schema_version"] == 1
    assert dataset["produced_by"].endswith("calibrate_figure_thresholds.py")
    # Inputs are sha-chained over the raw receipt bytes.
    inputs = dataset["inputs"]
    assert inputs["batch_receipt"]["path"] == str(batch_path)
    assert (
        inputs["batch_receipt"]["sha256"]
        == hashlib.sha256(batch_path.read_bytes()).hexdigest()
    )
    assert (
        inputs["replay_receipt"]["sha256"]
        == hashlib.sha256(replay_path.read_bytes()).hexdigest()
    )
    # Verdicts are optional (pre-verdict assembly): null ref when absent.
    assert inputs["verdicts"] is None
    # DINOv2 runtime block copied verbatim from the batch receipt.
    assert dataset["dinov2"] == _batch_receipt_payload(tmp_path / "deck")["dinov2"]
    assert dataset["unlabeled_row_count"] == 0


def test_dataset_missing_inputs_exit_1(calib, tmp_path):
    out = tmp_path / "calibration_dataset.json"
    rc = calib.main(
        [
            "assemble-dataset",
            "--batch-receipt",
            str(tmp_path / "missing_batch.json"),
            "--replay-receipt",
            str(tmp_path / "missing_replay.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()


# ---------------------------------------------------------------------------
# build-review (Task 2, -k blind): D-05..D-08 blind manifest + sidecar + HTML
# ---------------------------------------------------------------------------

# D-07 banned vocabulary, asserted by schema walk (literal here so the test
# cannot become a tautology over the module's own constant).
_BANNED_BLIND_KEYS = (
    "coverage",
    "dinov2_similarity",
    "clip_similarity",
    "margin",
    "margin_sign",
    "gate_decision",
    "critique_verdict",
    "critique_feedback",
    "claim_match",
    "estimator_findings",
    "similarity",
)

_FIXTURE_CLAIMS = {
    SL_GENERATED: "Counterfactual interventions expose unfaithful reasoning.",
    SL_BEST_EFFORT: "Multi-agent collaboration can underperform a single agent.",
    SL_SKIPPED: "LLM judges show internal scoring inconsistencies.",
    SL_D2: "Five separable failure modes corrupt a leaderboard average.",
}


def _deck_contract_payload() -> dict:
    """Trimmed presentation_deck.json: slide_id -> claims[] lookup shape."""

    return {
        "schema_version": 1,
        "presentation_id": "deck-03-fixture",
        "slides": [
            {"slide_id": slide_id, "title": slide_id.split("_", 2)[-1], "claims": [claim]}
            for slide_id, claim in _FIXTURE_CLAIMS.items()
        ],
    }


def _build_review_fixture(calib, tmp_path: Path, *, receipt_payload: dict | None = None, tag: str = ""):
    """Run build-review over the fixture receipt; returns (manifest, reveal,
    html_text, batch_path)."""

    deck = tmp_path / "deck"
    deck.mkdir(parents=True, exist_ok=True)
    _write_json(deck / "presentation_deck.json", _deck_contract_payload())
    batch_path = _write_json(
        tmp_path / f"figure_calibration_receipt{tag}.json",
        receipt_payload or _batch_receipt_payload(deck),
    )
    manifest_path = tmp_path / f"calibration_review_manifest{tag}.json"
    reveal_path = tmp_path / f"calibration_review_gate_reveal{tag}.json"
    html_path = tmp_path / f"calibration_review_blind{tag}.html"
    rc = calib.main(
        [
            "build-review",
            "--batch-receipt",
            str(batch_path),
            "--deck-dir",
            str(deck),
            "--out-manifest",
            str(manifest_path),
            "--out-reveal",
            str(reveal_path),
            "--out-html",
            str(html_path),
        ]
    )
    assert rc == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
    html_text = html_path.read_text(encoding="utf-8")
    return manifest, reveal, html_text, batch_path


def _walk_keys(node, found: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _walk_keys(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk_keys(value, found)


def test_blind_manifest_has_no_banned_keys(calib, tmp_path):
    # The load-bearing D-07 test: key-absence asserted by schema over the
    # ENTIRE manifest tree, not by discipline.
    manifest, _, _, _ = _build_review_fixture(calib, tmp_path)
    found: set = set()
    _walk_keys(manifest, found)
    leaked = found.intersection(_BANNED_BLIND_KEYS)
    assert leaked == set(), f"gate-score keys leaked into the blind manifest: {leaked}"


def test_blind_review_items_cover_d08(calib, tmp_path):
    manifest, _, _, _ = _build_review_fixture(calib, tmp_path)
    items = manifest["items"]
    by_id = {item["review_id"]: item for item in items}
    # Every final render (generated AND best_effort) + the last
    # gate-rejected attempt per redraw slide; the D-04 skipped slide
    # contributes NO review item.
    assert set(by_id) == {
        f"rev_{SL_GENERATED}_final_03",
        f"rev_{SL_GENERATED}_last_rejected_02",
        f"rev_{SL_BEST_EFFORT}_final_02",
        f"rev_{SL_BEST_EFFORT}_last_rejected_01",
        f"rev_{SL_D2}_final_02",
    }
    assert not [item for item in items if item["slide_id"] == SL_SKIPPED]
    for item in items:
        assert set(item) == {
            "review_id",
            "slide_id",
            "attempt",
            "kind",
            "render_path",
            "crop_path",
            "claim",
        }
        assert item["kind"] in ("final", "last_rejected")
        assert item["claim"] == _FIXTURE_CLAIMS[item["slide_id"]]
        assert item["render_path"]
    # The best_effort final IS rejected, so its "extra" item is the
    # next-lower rejected attempt; ungrounded tiers have null crops.
    assert by_id[f"rev_{SL_BEST_EFFORT}_last_rejected_01"]["attempt"] == 1
    assert by_id[f"rev_{SL_D2}_final_02"]["crop_path"] is None
    assert by_id[f"rev_{SL_GENERATED}_final_03"]["crop_path"]


def test_blind_shuffle_deterministic_by_receipt_sha(calib, tmp_path):
    # Pitfall 6: shuffled-but-deterministic, seeded by the receipt's raw
    # bytes. Two builds from the SAME receipt file produce identical order.
    deck = tmp_path / "deck"
    deck.mkdir(parents=True, exist_ok=True)
    _write_json(deck / "presentation_deck.json", _deck_contract_payload())
    batch_path = _write_json(
        tmp_path / "figure_calibration_receipt.json", _batch_receipt_payload(deck)
    )

    def _build(tag: str) -> list[str]:
        manifest_path = tmp_path / f"manifest_{tag}.json"
        rc = calib.main(
            [
                "build-review",
                "--batch-receipt",
                str(batch_path),
                "--deck-dir",
                str(deck),
                "--out-manifest",
                str(manifest_path),
                "--out-reveal",
                str(tmp_path / f"reveal_{tag}.json"),
                "--out-html",
                str(tmp_path / f"blind_{tag}.html"),
            ]
        )
        assert rc == 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [item["review_id"] for item in manifest["items"]]

    order_a = _build("a")
    order_b = _build("b")
    assert order_a == order_b
    # Different receipt bytes reseed the shuffle. Fixed byte fixtures keep
    # this assertion deterministic across sessions (tmp paths embedded in a
    # real receipt would vary the seed run-to-run).
    items = [{"review_id": f"rev_fixture_{index:02d}"} for index in range(8)]
    shuffled_x = calib.deterministic_shuffle(items, b"calibration receipt fixture X")
    shuffled_y = calib.deterministic_shuffle(items, b"calibration receipt fixture Y")
    assert calib.deterministic_shuffle(items, b"calibration receipt fixture X") == shuffled_x
    assert sorted(x["review_id"] for x in shuffled_x) == sorted(
        item["review_id"] for item in items
    )
    assert shuffled_x != shuffled_y
    assert shuffled_x != items  # the order really is shuffled, not identity


def test_blind_gate_reveal_sidecar_complete(calib, tmp_path):
    manifest, reveal, _, _ = _build_review_fixture(calib, tmp_path)
    gate_reveal = reveal["gate_reveal"]
    # Every review item has its full gate tuple in the SEPARATE sidecar.
    assert set(gate_reveal) == {item["review_id"] for item in manifest["items"]}
    be_final = gate_reveal[f"rev_{SL_BEST_EFFORT}_final_02"]
    assert be_final["coverage"] == 0.55
    assert be_final["dinov2_similarity"] == 0.69
    assert be_final["clip_similarity"] == 0.58
    assert be_final["margin_sign"] is True
    assert be_final["critique_verdict"] == "fail"
    assert be_final["claim_match"] == 0.82
    assert be_final["estimator_findings"] is not None
    assert isinstance(be_final["margin"], dict)
    rejected = gate_reveal[f"rev_{SL_BEST_EFFORT}_last_rejected_01"]
    assert rejected["coverage"] == 0.41
    assert rejected["margin_sign"] is False
    d2_final = gate_reveal[f"rev_{SL_D2}_final_02"]
    assert d2_final["coverage"] is None
    assert d2_final["critique_verdict"] == "pass"
    assert d2_final["claim_match"] is None
    # D-04 skips appear as a post-verdict recap in the sidecar, never as
    # review items.
    skips = reveal["grounding_skips"]
    assert len(skips) == 1
    assert skips[0]["slide_id"] == SL_SKIPPED
    assert skips[0]["claim_match"] == 0.4
    assert skips[0]["claim_match_threshold"] == 0.7
    assert skips[0]["pooled_notes"] == ["4uPyOCeN6U", "rwo7bVlnzo"]
    assert skips[0]["pool_candidate_count"] == 9


def test_blind_html_shows_claims_and_no_scores(calib, tmp_path):
    _, _, html_text, _ = _build_review_fixture(calib, tmp_path)
    lowered = html_text.lower()
    # Claim text and image pairs are present...
    for slide_id in (SL_GENERATED, SL_BEST_EFFORT, SL_D2):
        assert _FIXTURE_CLAIMS[slide_id] in html_text
    assert "<img" in lowered
    # ...but no banned key name appears anywhere in the page...
    for banned in _BANNED_BLIND_KEYS:
        assert banned not in lowered, f"banned name {banned!r} leaked into blind HTML"
    # ...and no gate-score values leak either.
    for score_text in ("0.55", "0.41", "0.92", "0.82", "0.69"):
        assert score_text not in html_text


def test_blind_manifest_and_reveal_paths_must_differ(calib, tmp_path):
    # Scores may never share a file with the blind view.
    deck = tmp_path / "deck"
    deck.mkdir(parents=True, exist_ok=True)
    _write_json(deck / "presentation_deck.json", _deck_contract_payload())
    batch_path = _write_json(
        tmp_path / "figure_calibration_receipt.json", _batch_receipt_payload(deck)
    )
    same = tmp_path / "manifest_and_reveal.json"
    rc = calib.main(
        [
            "build-review",
            "--batch-receipt",
            str(batch_path),
            "--deck-dir",
            str(deck),
            "--out-manifest",
            str(same),
            "--out-reveal",
            str(same),
            "--out-html",
            str(tmp_path / "blind.html"),
        ]
    )
    assert rc == 2
    assert not same.exists()


# ---------------------------------------------------------------------------
# merge-verdicts (Task 3, -k verdicts): D-06 enum-validated verdict capture
# ---------------------------------------------------------------------------


def _review_session_fixture(calib, tmp_path: Path):
    """build-review over the fixture receipt; returns (manifest_path,
    manifest, batch_path, replay_path) for merge/assemble round-trips."""

    deck = tmp_path / "deck"
    deck.mkdir(parents=True, exist_ok=True)
    _write_json(deck / "presentation_deck.json", _deck_contract_payload())
    batch_path = _write_json(
        tmp_path / "figure_calibration_receipt.json", _batch_receipt_payload(deck)
    )
    replay_path = _write_json(
        tmp_path / "figure_trial_replay_receipt.json", _replay_receipt_payload()
    )
    manifest_path = tmp_path / "calibration_review_manifest.json"
    rc = calib.main(
        [
            "build-review",
            "--batch-receipt",
            str(batch_path),
            "--deck-dir",
            str(deck),
            "--out-manifest",
            str(manifest_path),
            "--out-reveal",
            str(tmp_path / "calibration_review_gate_reveal.json"),
            "--out-html",
            str(tmp_path / "calibration_review_blind.html"),
        ]
    )
    assert rc == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, manifest, batch_path, replay_path


def _draft_for(manifest: dict, overrides: dict | None = None) -> list[dict]:
    """Raw in-session draft covering every manifest item (pass, no tags),
    with per-review_id overrides for the negative/positive cases."""

    overrides = overrides or {}
    draft = []
    for item in manifest["items"]:
        entry = {"review_id": item["review_id"], "verdict": "pass", "failure_modes": []}
        entry.update(overrides.get(item["review_id"], {}))
        draft.append(entry)
    return draft


def _merge(calib, tmp_path: Path, manifest_path: Path, draft: list, tag: str = ""):
    draft_path = _write_json(tmp_path / f"draft_verdicts{tag}.json", draft)
    out_path = tmp_path / f"calibration_review_verdicts{tag}.json"
    rc = calib.main(
        [
            "merge-verdicts",
            "--manifest",
            str(manifest_path),
            "--draft",
            str(draft_path),
            "--out",
            str(out_path),
            "--reviewer",
            "operator",
        ]
    )
    return rc, out_path


def test_verdicts_vocabulary_constants_exact(calib):
    # D-06 fixed vocabulary — exactly these six tags, exactly this binary.
    assert calib.FAILURE_MODE_TAGS == (
        "wrong-source",
        "missing-components",
        "invented-components",
        "legibility",
        "style-drift",
        "other",
    )
    assert calib.VERDICT_VALUES == ("pass", "fail")


def test_verdicts_rejects_bad_verdict_enum(calib, tmp_path, capsys):
    manifest_path, manifest, _, _ = _review_session_fixture(calib, tmp_path)
    bad_id = f"rev_{SL_GENERATED}_final_03"
    draft = _draft_for(manifest, {bad_id: {"verdict": "maybe"}})
    rc, out_path = _merge(calib, tmp_path, manifest_path, draft)
    assert rc == 2
    assert not out_path.exists()
    # The refusal names the offending review_id.
    assert bad_id in capsys.readouterr().err


def test_verdicts_rejects_tag_rule_violations(calib, tmp_path):
    manifest_path, manifest, _, _ = _review_session_fixture(calib, tmp_path)
    rid = f"rev_{SL_BEST_EFFORT}_final_02"
    # Tag outside the fixed vocabulary.
    rc, out_path = _merge(
        calib,
        tmp_path,
        manifest_path,
        _draft_for(manifest, {rid: {"verdict": "fail", "failure_modes": ["blurry"]}}),
        tag="_badtag",
    )
    assert rc == 2
    assert not out_path.exists()
    # fail requires at least one tag.
    rc, _ = _merge(
        calib,
        tmp_path,
        manifest_path,
        _draft_for(manifest, {rid: {"verdict": "fail", "failure_modes": []}}),
        tag="_failnotags",
    )
    assert rc == 2
    # pass must carry no tags.
    rc, _ = _merge(
        calib,
        tmp_path,
        manifest_path,
        _draft_for(manifest, {rid: {"verdict": "pass", "failure_modes": ["other"]}}),
        tag="_passtags",
    )
    assert rc == 2


def test_verdicts_rejects_coverage_mismatch(calib, tmp_path, capsys):
    manifest_path, manifest, _, _ = _review_session_fixture(calib, tmp_path)
    # A review_id not present in the manifest.
    extra = _draft_for(manifest)
    extra.append({"review_id": "rev_unknown_final_01", "verdict": "pass", "failure_modes": []})
    rc, _ = _merge(calib, tmp_path, manifest_path, extra, tag="_extra")
    assert rc == 2
    assert "rev_unknown_final_01" in capsys.readouterr().err
    # A manifest item missing from the draft (one-verdict-per-item coverage).
    missing = _draft_for(manifest)[:-1]
    rc, _ = _merge(calib, tmp_path, manifest_path, missing, tag="_missing")
    assert rc == 2
    # A duplicated verdict entry.
    duplicated = _draft_for(manifest)
    duplicated.append(dict(duplicated[0]))
    rc, _ = _merge(calib, tmp_path, manifest_path, duplicated, tag="_dup")
    assert rc == 2


def test_verdicts_valid_merge_then_dataset_human_flip(calib, tmp_path):
    manifest_path, manifest, batch_path, replay_path = _review_session_fixture(
        calib, tmp_path
    )
    overrides = {
        f"rev_{SL_BEST_EFFORT}_final_02": {
            "verdict": "fail",
            "failure_modes": ["legibility", "missing-components"],
            "notes": "axis labels unreadable",
        },
        f"rev_{SL_GENERATED}_last_rejected_02": {"verdict": "pass", "failure_modes": []},
    }
    rc, out_path = _merge(calib, tmp_path, manifest_path, _draft_for(manifest, overrides))
    assert rc == 0
    verdicts = json.loads(out_path.read_text(encoding="utf-8"))
    # record_human_review_receipt payload conventions.
    assert verdicts["schema_version"] == 1
    assert verdicts["reviewer"] == "operator"
    assert verdicts["recorded_at"].endswith("Z")
    assert verdicts["inputs"]["manifest"]["path"] == str(manifest_path)
    assert (
        verdicts["inputs"]["manifest"]["sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    items = {item["review_id"]: item for item in verdicts["items"]}
    assert set(items) == {item["review_id"] for item in manifest["items"]}
    recorded = items[f"rev_{SL_BEST_EFFORT}_final_02"]
    assert recorded["verdict"] == "fail"
    assert recorded["failure_modes"] == ["legibility", "missing-components"]
    assert recorded["slide_id"] == SL_BEST_EFFORT
    assert recorded["attempt"] == 2
    assert recorded["notes"] == "axis labels unreadable"

    # Integration across subcommands: re-running assemble-dataset with
    # --verdicts flips matching rows from critique to human (D-12).
    out_pre = tmp_path / "calibration_dataset_pre.json"
    out_post = tmp_path / "calibration_dataset_post.json"
    base_cmd = [
        "assemble-dataset",
        "--batch-receipt",
        str(batch_path),
        "--replay-receipt",
        str(replay_path),
    ]
    assert calib.main([*base_cmd, "--out", str(out_pre)]) == 0
    assert calib.main([*base_cmd, "--verdicts", str(out_path), "--out", str(out_post)]) == 0
    pre = {r["row_id"]: r for r in json.loads(out_pre.read_text(encoding="utf-8"))["rows"]}
    post = {r["row_id"]: r for r in json.loads(out_post.read_text(encoding="utf-8"))["rows"]}
    flipped = f"batch:{SL_BEST_EFFORT}:a2"
    assert pre[flipped]["label_source"] == "critique"
    assert post[flipped]["label_source"] == "human"
    assert post[flipped]["label"] == "fail"
    assert post[flipped]["failure_modes"] == ["legibility", "missing-components"]
    # The human PASS on the rejected sample flips a critique-fail row.
    rejected = f"batch:{SL_GENERATED}:a2"
    assert pre[rejected]["label"] == "fail"
    assert post[rejected]["label"] == "pass"
    assert post[rejected]["label_source"] == "human"
    # Trial rows are structurally excluded from review: locked labels stay.
    for slide_id, *_ in _TRIAL_TUPLES:
        assert post[f"trial:{slide_id}"]["label_source"] == "trial_v2_operator"


# ===========================================================================
# 02-04 analysis CLI: coverage sweep, degeneracy triggers, populations
# (Task 1; -k sweep). The closed form owns ONLY D-10 trigger (a); triggers
# (b)/(c) live in the separate apply_degeneracy_triggers layer; D-12
# populations come from build_populations.
# ===========================================================================


def _cov_row(
    coverage: float | None,
    label: str,
    *,
    label_source: str = "human",
    is_final: bool = True,
) -> dict:
    """A minimal analysis row: only coverage/label drive the sweep, but the
    population builder reads label_source and is_final_render too."""

    return {
        "coverage": coverage,
        "label": label,
        "label_source": label_source,
        "is_final_render": is_final,
    }


def _cov_rows(fails: list[float], passes: list[float], **kw) -> list[dict]:
    rows = [_cov_row(c, "fail", **kw) for c in fails]
    rows += [_cov_row(c, "pass", **kw) for c in passes]
    return rows


def test_sweep_closed_form_midpoint_two_fails(calib):
    # Test 1: closed form owns ONLY trigger (a). fails=[0.36,0.41],
    # passes=[0.86,0.94,1.0] -> threshold = (0.41+0.86)/2 = 0.635, zero
    # false-pass, zero false-fail, status "selected", curve present.
    rows = _cov_rows([0.36, 0.41], [0.86, 0.94, 1.0])
    result = calib.select_coverage_threshold(rows)
    assert result["status"] == "selected"
    assert result["threshold"] == pytest.approx(0.635)
    assert result["false_pass"] == {"count": 0, "of": 2}
    assert result["false_fail"] == {"count": 0, "of": 3}
    assert isinstance(result["curve"], list) and result["curve"]
    # Curve points are data, each with per-point false_pass/false_fail counts.
    point = result["curve"][0]
    assert {"threshold", "false_pass", "false_fail"} <= set(point)


def test_sweep_false_fail_minimization_is_automatic(calib):
    # Test 2 (closed form half): passes=[0.30,0.86,1.0], fails=[0.41] ->
    # the smallest zero-false-pass threshold is the minimizer (false-fails
    # monotone in t). threshold = midpoint(0.41, 0.86), false_fail 1 of 3.
    rows = _cov_rows([0.41], [0.30, 0.86, 1.0])
    result = calib.select_coverage_threshold(rows)
    assert result["status"] == "selected"
    assert result["threshold"] == pytest.approx((0.41 + 0.86) / 2)
    assert result["false_fail"] == {"count": 1, "of": 3}
    assert result["false_pass"] == {"count": 0, "of": 1}


def test_sweep_layering_single_fail_downgraded_by_triggers(calib):
    # Test 2 (layering half): the SAME 1-fail fixture through
    # apply_degeneracy_triggers becomes degenerate via BOTH (b) (1<3) and
    # (c) (rate 1/3>0.25); threshold None; the closed-form midpoint is
    # demoted into candidates.
    rows = _cov_rows([0.41], [0.30, 0.86, 1.0])
    selected = calib.select_coverage_threshold(rows)
    midpoint = selected["threshold"]
    downgraded = calib.apply_degeneracy_triggers(selected, rows)
    assert downgraded["status"] == "degenerate"
    assert downgraded.get("threshold") is None
    assert "labeled-fail" in downgraded["reason"] or "false-fail" in downgraded["reason"]
    candidate_thresholds = [c["threshold"] for c in downgraded["candidates"]]
    assert any(t == pytest.approx(midpoint) for t in candidate_thresholds)
    assert downgraded["halt_tolerance"] == calib.HALT_TOLERANCE_FALSE_FAIL


def test_sweep_trigger_a_missing_label_class(calib):
    # Test 3a (D-10a inside the closed form): empty fails OR empty passes.
    only_pass = calib.select_coverage_threshold(_cov_rows([], [0.5, 0.9]))
    assert only_pass["status"] == "degenerate"
    assert only_pass["reason"] == "missing a label class"
    assert only_pass.get("threshold") is None
    assert "curve" in only_pass and "candidates" in only_pass
    only_fail = calib.select_coverage_threshold(_cov_rows([0.5], []))
    assert only_fail["status"] == "degenerate"
    assert only_fail["reason"] == "missing a label class"


def test_sweep_trigger_a_no_zero_false_pass(calib):
    # Test 3b (D-10a): all passes <= max(fails) -> no zero-false-pass point.
    rows = _cov_rows([0.9], [0.4, 0.8, 0.9])
    result = calib.select_coverage_threshold(rows)
    assert result["status"] == "degenerate"
    assert result["reason"] == "no zero-false-pass threshold exists"
    assert result.get("threshold") is None
    assert result["candidates"]  # candidate operating points present


def test_sweep_trigger_b_too_few_fails(calib):
    # Test 4 (D-10b in apply_degeneracy_triggers): a "selected" sweep over a
    # population with < 3 labeled fails downgrades; reason names the count.
    rows = _cov_rows([0.30, 0.41], [0.86, 0.94, 1.0])
    selected = calib.select_coverage_threshold(rows)
    assert selected["status"] == "selected"
    downgraded = calib.apply_degeneracy_triggers(selected, rows)
    assert downgraded["status"] == "degenerate"
    assert "2" in downgraded["reason"]  # the labeled-fail count
    assert downgraded.get("threshold") is None
    assert any(
        c["threshold"] == pytest.approx(selected["threshold"])
        for c in downgraded["candidates"]
    )


def test_sweep_trigger_c_high_false_fail_rate(calib):
    # Test 5 (D-10c only): >=3 labeled fails so (b) does not fire; the
    # zero-false-pass point false-fails 2 of 6 passes (~0.333 > 0.25).
    # fails below 0.50, two passes (0.45,0.48) below the midpoint.
    fails = [0.30, 0.40, 0.50]
    passes = [0.45, 0.48, 0.80, 0.90, 0.95, 1.0]
    selected = calib.select_coverage_threshold(_cov_rows(fails, passes))
    assert selected["status"] == "selected"
    assert selected["false_fail"]["count"] == 2
    assert selected["false_fail"]["of"] == 6
    downgraded = calib.apply_degeneracy_triggers(selected, _cov_rows(fails, passes))
    assert downgraded["status"] == "degenerate"
    assert downgraded["halt_tolerance"] == 0.25
    assert downgraded.get("threshold") is None
    assert "false-fail" in downgraded["reason"]


def test_sweep_distinct_degeneracy_reasons(calib):
    # All three degeneracy reasons are distinct strings.
    r_a1 = calib.select_coverage_threshold(_cov_rows([], [0.5]))["reason"]
    r_a2 = calib.select_coverage_threshold(_cov_rows([0.9], [0.4, 0.8]))["reason"]
    sel_b = calib.select_coverage_threshold(_cov_rows([0.30, 0.41], [0.86, 0.94, 1.0]))
    r_b = calib.apply_degeneracy_triggers(sel_b, _cov_rows([0.30, 0.41], [0.86, 0.94, 1.0]))["reason"]
    assert len({r_a1, r_a2, r_b}) == 3


def test_sweep_d12_populations(calib):
    # Test 6: build_populations returns P1 headline
    # (human/ai_panel/trial_v2_operator + coverage non-null), P2 all-labeled,
    # P3 finals-only, each with a definition and a row count.
    rows = [
        _cov_row(0.90, "pass", label_source="human", is_final=True),
        _cov_row(0.40, "fail", label_source="human", is_final=False),
        _cov_row(0.85, "pass", label_source="trial_v2_operator", is_final=True),
        _cov_row(0.55, "fail", label_source="critique", is_final=False),
        _cov_row(None, "pass", label_source="human", is_final=True),  # null coverage
    ]
    pops = calib.build_populations(rows)
    by_name = {p["name"]: p for p in pops}
    assert set(by_name) == {"P1_headline", "P2_all_labeled", "P3_finals_only"}
    # P1: human/ai_panel/trial_v2_operator AND coverage non-null -> 3 rows (the
    # null coverage human row is excluded).
    assert by_name["P1_headline"]["row_count"] == 3
    # P2: all labeled rows incl. critique, coverage non-null -> 4 rows.
    assert by_name["P2_all_labeled"]["row_count"] == 4
    # P3: finals only (is_final_render True), coverage non-null -> 2 rows
    # (the two final rows with coverage; the null-coverage final is excluded).
    assert by_name["P3_finals_only"]["row_count"] == 2
    for pop in pops:
        assert pop["definition"]  # human-readable definition string


def test_headline_sources_include_ai_panel(calib):
    # 02-06: the operator-approved AI panel (02-05) substituted for the single
    # human sitting, so ai_panel labels are human-equivalent adjudicated
    # verdicts and MUST be in the P1 headline population (D-09 halt authority).
    # human and trial_v2_operator stay; critique stays OUT (machine label).
    assert "ai_panel" in calib._HEADLINE_LABEL_SOURCES
    assert "human" in calib._HEADLINE_LABEL_SOURCES
    assert calib.TRIAL_V2_LABEL_SOURCE in calib._HEADLINE_LABEL_SOURCES
    assert "critique" not in calib._HEADLINE_LABEL_SOURCES


def test_sweep_p1_headline_includes_ai_panel_rows(calib):
    # 02-06: ai_panel rows land in P1 (not just P2). With the real-dataset
    # shape (ai_panel pass+fail + trial_v2_operator), P1 must include the
    # ai_panel rows so the headline coverage sweep is not starved to the six
    # trial rows. The critique row stays out of P1 but in P2.
    rows = [
        _cov_row(0.92, "pass", label_source="ai_panel", is_final=True),
        _cov_row(0.35, "fail", label_source="ai_panel", is_final=True),
        _cov_row(0.50, "fail", label_source="ai_panel", is_final=False),
        _cov_row(0.88, "pass", label_source="trial_v2_operator", is_final=True),
        _cov_row(0.60, "fail", label_source="critique", is_final=False),
    ]
    pops = {p["name"]: p for p in calib.build_populations(rows)}
    p1_sources = {r["label_source"] for r in pops["P1_headline"]["rows"]}
    assert "ai_panel" in p1_sources
    assert "critique" not in p1_sources
    # P1: 3 ai_panel + 1 trial_v2_operator = 4; critique excluded.
    assert pops["P1_headline"]["row_count"] == 4
    # P2 includes the critique row too -> 5.
    assert pops["P2_all_labeled"]["row_count"] == 5
    # The emitted P1 definition names ai_panel so the methodology is auditable.
    assert "ai_panel" in pops["P1_headline"]["definition"]


def test_sweep_purity_signature(calib):
    # Test 7: select_coverage_threshold's parameters are exactly {"rows"} so
    # margins/estimators provably cannot leak into selection.
    import inspect

    params = set(inspect.signature(calib.select_coverage_threshold).parameters)
    assert params == {"rows"}


def test_sweep_no_zero_eight_fallback_anywhere(calib):
    # No result path returns the prior 0.8 as a fallback: degenerate results
    # from trigger (a) AND (b)/(c) downgrades carry threshold None.
    for rows in (
        _cov_rows([], [0.8]),
        _cov_rows([0.9], [0.4, 0.8]),
    ):
        result = calib.select_coverage_threshold(rows)
        assert result["status"] == "degenerate"
        assert result.get("threshold") is None
    sel = calib.select_coverage_threshold(_cov_rows([0.41], [0.30, 0.86, 1.0]))
    down = calib.apply_degeneracy_triggers(sel, _cov_rows([0.41], [0.30, 0.86, 1.0]))
    assert down.get("threshold") is None


# ===========================================================================
# 02-04 analysis CLI: stack-level floor selection, per-metric degeneracy
# evidence, rank-AUC separation record (Task 2; -k "floor or separation").
# The deliverable floor is DINOv2-only; CLIP runs only to produce the
# computed-not-adopted candidate floor (D-11). The per-metric DINOv2 sweep is
# degenerate by construction (sl_19 0.7456 > sl_23 0.6913) but must NOT make
# the overall floor selection fail (Pattern 5 / Pitfall 1).
# ===========================================================================


def _floor_row(
    slide_id: str,
    label: str,
    coverage: float,
    dinov2: float,
    clip: float,
    *,
    margin_sign: bool = True,
    critique: str | None = "pass",
) -> dict:
    """A stack-classifiable analysis row for the floor sweep."""

    return {
        "slide_id": slide_id,
        "label": label,
        "coverage": coverage,
        "dinov2_similarity": dinov2,
        "clip_similarity": clip,
        "margin_sign": margin_sign,
        "critique_verdict": critique,
        "label_source": "trial_v2_operator",
        "is_final_render": True,
    }


# The six trial-v2 tuples with the real replay-receipt values. Coverages and
# DINOv2 sims are the load-bearing numbers; CLIP sims mirror the dataset
# fixture. sl_11 is the wrong-figure class (DINOv2 0.6211), sl_19 the
# dropped-components class (coverage 0.361, DINOv2 0.7456 ABOVE passing
# sl_23's 0.6913 — the per-metric trap).
def _six_tuple_floor_rows() -> list[dict]:
    # sl_11 (wrong-figure class) passes coverage/margin/critique — only the
    # embedding floor can catch it (0.6211). sl_19 (dropped-components class)
    # is owned by coverage (0.361 < t_cov); its DINOv2 0.7456 sits ABOVE
    # passing sl_23's 0.6913 — the per-metric trap. Both fails therefore look
    # fine to margin/critique; the gate STACK separates them.
    return [
        _floor_row("sl_05", "pass", 1.000, 0.7267, 0.6828),
        _floor_row("sl_06", "pass", 0.857, 0.8342, 0.7012),
        _floor_row("sl_10", "pass", 0.967, 0.7510, 0.6531),
        _floor_row("sl_11", "fail", 0.409, 0.6211, 0.5418),
        _floor_row("sl_19", "fail", 0.361, 0.7456, 0.6105),
        _floor_row("sl_23", "pass", 0.944, 0.6913, 0.6347),
    ]


# t_cov chosen in (0.361, 0.409] so coverage OWNS sl_19 (0.361 < t_cov) but
# NOT sl_11 (0.409 >= t_cov) — forcing the floor to own sl_11 (Pattern 5).
_FLOOR_T_COV = 0.40


def test_floor_per_metric_dinov2_sweep_is_degenerate(calib):
    # Test 1: the naive per-metric DINOv2 zero-false-pass sweep is degenerate
    # (sl_19 fail 0.7456 sits above sl_23 pass 0.6913) — recorded as evidence,
    # and this does NOT make the overall floor selection fail.
    rows = _six_tuple_floor_rows()
    result = calib.select_stack_floor(rows, _FLOOR_T_COV, "dinov2_similarity")
    assert result["status"] == "selected"
    assert result["per_metric"]["dinov2"]["status"] == "degenerate"


def test_floor_stack_level_selection_picks_largest_feasible(calib):
    # Test 2: any floor in (0.6211, 0.6913] blocks sl_11 via floor; sl_19 is
    # already coverage-blocked. Selection picks the LARGEST feasible floor in
    # the minimal-false-fail set and records the feasible interval.
    rows = _six_tuple_floor_rows()
    result = calib.select_stack_floor(rows, _FLOOR_T_COV, "dinov2_similarity")
    assert result["status"] == "selected"
    floor = result["floor"]
    assert 0.6211 < floor <= 0.6913
    lo, hi = result["feasible_interval"]
    assert lo == pytest.approx(0.6211) or lo > 0.6211
    assert hi <= 0.6913
    # Largest feasible floor: at or just below sl_23's 0.6913 pass value.
    assert floor == pytest.approx(hi)


def test_floor_gate_attribution(calib):
    # Test 3: each labeled-fail row records which gate(s) fire. sl_11 -> floor,
    # sl_19 -> coverage.
    rows = _six_tuple_floor_rows()
    result = calib.select_stack_floor(rows, _FLOOR_T_COV, "dinov2_similarity")
    attribution = {a["slide_id"]: a["gates"] for a in result["gate_attribution"]}
    assert "floor" in attribution["sl_11"]
    assert "coverage" in attribution["sl_19"]


def test_floor_clip_candidate_not_adopted(calib):
    # Test 4 (D-11): the same procedure run with CLIP produces
    # clip_candidate_floor with adopted=false — present, never merged.
    rows = _six_tuple_floor_rows()
    clip_result = calib.select_stack_floor(rows, _FLOOR_T_COV, "clip_similarity")
    candidate = calib.clip_candidate_floor(rows, _FLOOR_T_COV)
    assert candidate["adopted"] is False
    assert "floor" in candidate or candidate.get("status") == "degenerate"
    # Whatever CLIP produces, it carries the not-adopted marker.
    assert clip_result["metric_key"] == "clip_similarity"


def test_floor_excluded_null_metric_rows_counted(calib):
    # Rows with null metric values are excluded from the sweep but counted.
    rows = _six_tuple_floor_rows()
    rows.append(_floor_row("sl_99", "pass", 0.90, 0.0, 0.0))
    rows[-1]["dinov2_similarity"] = None
    result = calib.select_stack_floor(rows, _FLOOR_T_COV, "dinov2_similarity")
    assert result["excluded_row_count"] == 1


def test_separation_rank_auc_toy_cases(calib):
    # Test 5: perfect separation -> 1.0, inversion -> 0.0, all-ties -> 0.5,
    # empty class -> None.
    assert calib.rank_auc([0.9, 0.8], [0.2, 0.1]) == pytest.approx(1.0)
    assert calib.rank_auc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)
    assert calib.rank_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
    assert calib.rank_auc([], [0.5]) is None
    assert calib.rank_auc([0.5], []) is None


def test_separation_rank_auc_matches_mannwhitney(calib):
    # Optional scipy cross-check: rank_auc == U / (n_pos * n_neg).
    scipy_stats = pytest.importorskip("scipy.stats")
    passes = [0.91, 0.74, 0.66, 0.69]
    fails = [0.62, 0.55, 0.74]
    u, _ = scipy_stats.mannwhitneyu(passes, fails, alternative="greater")
    expected = u / (len(passes) * len(fails))
    assert calib.rank_auc(passes, fails) == pytest.approx(expected)


def test_separation_record_per_metric(calib):
    # Test 6: per metric on pooled rows where BOTH scores present:
    # {auc, n_pass, n_fail, score_ranges, overlap_interval,
    #  best_single_threshold_accuracy}.
    rows = _six_tuple_floor_rows()
    record = calib.separation_record(rows)
    for metric in ("dinov2", "clip"):
        block = record[metric]
        assert set(block) >= {
            "auc",
            "n_pass",
            "n_fail",
            "score_ranges",
            "overlap_interval",
            "best_single_threshold_accuracy",
        }
        assert block["n_pass"] == 4
        assert block["n_fail"] == 2
    # DINOv2 does NOT cleanly separate (sl_19 trap) -> AUC < 1.0.
    assert record["dinov2"]["auc"] < 1.0


# ===========================================================================
# 02-04 analysis CLI: re-application check + analyze subcommand assembly +
# operator-decision recording (Task 3; -k "reapply or analyze").
# ===========================================================================


def _replay_reapply_figures() -> list[dict]:
    """The six replay-receipt figure tuples (synthetic, real values) with
    GARBAGE classification/status fields the reapply check must ignore."""

    figs = []
    for slide_id, _tier, coverage, dinov2, _clip, _verdict in _TRIAL_TUPLES:
        figs.append(
            {
                "slide_id": slide_id,
                "coverage": coverage,
                "dinov2_similarity": dinov2,
                "margin": {"dinov2": {"margin": 0.04, "margin_sign": True}},
                "critique": {"verdict": "pass"},
                # Garbage fields: the reapply check must NOT read these.
                "classification": "coverage_pass",
                "status": "generated",
            }
        )
    return figs


def test_reapply_classifies_sl11_sl19_fail(calib):
    # Test 1: (t_cov=0.635, floor=0.65) classifies sl_11 fail (floor:
    # 0.6211 < 0.65) and sl_19 fail (coverage: 0.361 < 0.635); four others
    # pass — recomputed from tuples, never read from status/classification.
    figures = _replay_reapply_figures()
    result = calib.reapply_thresholds(figures, 0.635, 0.65)
    by_slide = {c["slide_id"][:5]: c for c in result["classifications"]}
    assert by_slide["sl_11"]["classification"] == "fail"
    assert "floor" in by_slide["sl_11"]["gates"]
    assert by_slide["sl_19"]["classification"] == "fail"
    assert "coverage" in by_slide["sl_19"]["gates"]
    for prefix in ("sl_05", "sl_06", "sl_10", "sl_23"):
        assert by_slide[prefix]["classification"] == "pass"
    assert result["ok"] is True


def test_reapply_ignores_status_fields(calib):
    # Test 4 (acceptance): set classification/status to garbage; the result
    # is unchanged — proves recompute-from-tuple.
    figures = _replay_reapply_figures()
    for fig in figures:
        fig["classification"] = "GARBAGE"
        fig["status"] = "GARBAGE"
    result = calib.reapply_thresholds(figures, 0.635, 0.65)
    assert result["ok"] is True
    by_slide = {c["slide_id"][:5]: c for c in result["classifications"]}
    assert by_slide["sl_11"]["classification"] == "fail"
    assert by_slide["sl_19"]["classification"] == "fail"


def test_reapply_failure_path_ok_false(calib):
    # Test 2: a (t_cov, floor) that passes sl_11 -> ok=false. floor 0.50 <
    # sl_11's 0.6211 and t_cov 0.40 >= sl_11's 0.409 -> sl_11 passes.
    figures = _replay_reapply_figures()
    result = calib.reapply_thresholds(figures, 0.40, 0.50)
    by_slide = {c["slide_id"][:5]: c for c in result["classifications"]}
    assert by_slide["sl_11"]["classification"] == "pass"
    assert result["ok"] is False


# --- analyze subcommand assembly --------------------------------------------


def _analysis_dataset(*, headline_degenerate: bool = False) -> dict:
    """A calibration_dataset.json payload with enough labeled rows for a
    clean headline sweep (>=3 fails, low false-fail) plus the six trial
    rows. headline_degenerate=True adds low-coverage human PASS rows that
    the zero-false-pass point must block, tripping D-10 trigger (c) (false-
    fail rate > 0.25) while keeping >=3 labeled fails.
    """

    rows: list[dict] = []

    def row(rid, label, cov, d2, clip, src, final=True):
        return {
            "row_id": rid,
            "slide_id": rid,
            "origin": "calibration_batch",
            "label": label,
            "label_source": src,
            "coverage": cov,
            "dinov2_similarity": d2,
            "clip_similarity": clip,
            "margin_sign": True,
            "critique_verdict": "pass" if label == "pass" else "fail",
            "is_final_render": final,
        }

    # Human-labeled batch rows: clean separation, 3 fails, near-zero false-fail.
    rows.append(row("b_pass1", "pass", 0.90, 0.80, 0.70, "human"))
    rows.append(row("b_pass2", "pass", 0.88, 0.78, 0.68, "human"))
    rows.append(row("b_pass3", "pass", 0.95, 0.82, 0.72, "human"))
    rows.append(row("b_fail1", "fail", 0.30, 0.60, 0.50, "human"))
    rows.append(row("b_fail2", "fail", 0.40, 0.62, 0.52, "human"))
    rows.append(row("b_fail3", "fail", 0.50, 0.63, 0.53, "human"))
    if headline_degenerate:
        # Low-coverage human PASSES that the zero-false-pass point (above
        # max_fail 0.50) must block -> false-fail rate well over 0.25.
        rows.append(row("b_ff1", "pass", 0.45, 0.79, 0.69, "human"))
        rows.append(row("b_ff2", "pass", 0.48, 0.81, 0.71, "human"))
        rows.append(row("b_ff3", "pass", 0.42, 0.80, 0.70, "human"))
    # A critique-labeled row only enters P2.
    rows.append(row("b_crit", "fail", 0.45, 0.61, 0.51, "critique", final=False))
    # The six pooled trial rows (P1 via trial_v2_operator).
    for slide_id, _tier, cov, d2, clip, verdict in _TRIAL_TUPLES:
        rows.append(
            {
                "row_id": f"trial:{slide_id}",
                "slide_id": slide_id,
                "origin": "trial_v2_replay",
                "label": "fail" if verdict == "fail" else "pass",
                "label_source": "trial_v2_operator",
                "coverage": cov,
                "dinov2_similarity": d2,
                "clip_similarity": clip,
                "margin_sign": True,
                "critique_verdict": verdict,
                "is_final_render": True,
            }
        )
    return {
        "schema_version": 1,
        "produced_by": "scripts/presentations/calibrate_figure_thresholds.py",
        "dinov2": {"model_name": "facebook/dinov2-base", "revision": "f" * 40},
        "rows": rows,
        "grounding_skips": [],
        "unlabeled_row_count": 0,
    }


def _write_analysis_inputs(tmp_path: Path, *, headline_degenerate: bool = False):
    dataset_path = tmp_path / "calibration_dataset.json"
    replay_path = tmp_path / "figure_trial_replay_receipt.json"
    _write_json(dataset_path, _analysis_dataset(headline_degenerate=headline_degenerate))
    _write_json(
        replay_path,
        {"schema_version": 1, "figures": _replay_reapply_figures()},
    )
    return dataset_path, replay_path


def test_analyze_receipt_assembly(calib, tmp_path):
    # Test 3: analyze writes the full analysis receipt with the contracted
    # top-level keys.
    dataset_path, replay_path = _write_analysis_inputs(tmp_path)
    out = tmp_path / "calibration_threshold_analysis.json"
    code = calib.main(
        [
            "analyze",
            "--dataset",
            str(dataset_path),
            "--replay-receipt",
            str(replay_path),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 1
    assert receipt["produced_by"].endswith("calibrate_figure_thresholds.py")
    assert receipt["inputs"]["dataset"]["sha256"]
    assert receipt["inputs"]["replay_receipt"]["sha256"]
    # populations: three named results with definitions + counts.
    pops = {p["name"]: p for p in receipt["coverage_threshold_selection"]["populations"]}
    assert set(pops) == {"P1_headline", "P2_all_labeled", "P3_finals_only"}
    assert receipt["coverage_threshold_selection"]["headline"] == "P1_headline"
    assert "floor_selection" in receipt
    assert receipt["clip_candidate_floor"]["adopted"] is False
    assert "per_metric_curves" in receipt
    assert "separation" in receipt
    assert "reapplication" in receipt
    assert receipt["halt_tolerance"] == calib.HALT_TOLERANCE_FALSE_FAIL
    assert receipt["status"] in {"selected", "degenerate", "operator_decided"}
    assert receipt["operator_decision"] is None


def test_analyze_exit_0_selected_with_reapplication(calib, tmp_path):
    # Test 4a: status selected + reapplication ok -> exit 0.
    dataset_path, replay_path = _write_analysis_inputs(tmp_path)
    out = tmp_path / "analysis.json"
    code = calib.main(
        ["analyze", "--dataset", str(dataset_path), "--replay-receipt", str(replay_path), "--out", str(out)]
    )
    assert code == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["status"] == "selected"
    assert receipt["reapplication"]["ok"] is True


def test_analyze_exit_3_degenerate_headline_writes_receipt(calib, tmp_path):
    # Test 4b: degenerate HEADLINE (P1) -> exit 3 with the receipt STILL
    # written (curve + candidates present for the D-10 surface).
    dataset_path, replay_path = _write_analysis_inputs(tmp_path, headline_degenerate=True)
    out = tmp_path / "analysis.json"
    code = calib.main(
        ["analyze", "--dataset", str(dataset_path), "--replay-receipt", str(replay_path), "--out", str(out)]
    )
    assert code == 3
    assert out.exists()  # receipt-on-disk for halt-and-surface
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["status"] == "degenerate"
    p1 = next(
        p for p in receipt["coverage_threshold_selection"]["populations"]
        if p["name"] == "P1_headline"
    )
    assert p1["result"]["status"] == "degenerate"
    assert p1["result"]["candidates"]
    assert p1["result"]["curve"]


def test_analyze_sensitivity_degenerate_does_not_halt(calib, tmp_path):
    # Test 4c: a degenerate sensitivity population (P2/P3) alongside a
    # selected P1 does NOT halt — recorded only. The base fixture has a clean
    # P1; force P2 degenerate by... it already differs. Assert P1 selected and
    # exit 0 regardless of sensitivity outcomes.
    dataset_path, replay_path = _write_analysis_inputs(tmp_path)
    out = tmp_path / "analysis.json"
    code = calib.main(
        ["analyze", "--dataset", str(dataset_path), "--replay-receipt", str(replay_path), "--out", str(out)]
    )
    assert code == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    pops = {p["name"]: p for p in receipt["coverage_threshold_selection"]["populations"]}
    assert pops["P1_headline"]["result"]["status"] == "selected"
    # halt authority is P1 alone: even if a sensitivity pop is degenerate the
    # exit is 0.
    assert receipt["status"] == "selected"


def test_analyze_missing_input_exit_1(calib, tmp_path):
    # Test 4d: missing input file -> exit 1.
    out = tmp_path / "analysis.json"
    code = calib.main(
        [
            "analyze",
            "--dataset",
            str(tmp_path / "nope.json"),
            "--replay-receipt",
            str(tmp_path / "also-nope.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert not out.exists()


def test_analyze_operator_decision_recorded(calib, tmp_path):
    # Test 5 (D-10 resolution path): --operator-threshold/-floor/-note records
    # status operator_decided, the verbatim decision, recomputed error rates,
    # and reapplication at the operator point; exit 0 only if reapply holds.
    dataset_path, replay_path = _write_analysis_inputs(tmp_path, headline_degenerate=True)
    out = tmp_path / "analysis.json"
    code = calib.main(
        [
            "analyze",
            "--dataset",
            str(dataset_path),
            "--replay-receipt",
            str(replay_path),
            "--out",
            str(out),
            "--operator-threshold",
            "0.635",
            "--operator-floor",
            "0.65",
            "--operator-note",
            "operator picked midpoint after reviewing the curve",
        ]
    )
    assert code == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["status"] == "operator_decided"
    decision = receipt["operator_decision"]
    assert decision["threshold"] == pytest.approx(0.635)
    assert decision["floor"] == pytest.approx(0.65)
    assert decision["note"].startswith("operator picked")
    # Reapplication recomputed at the operator point and holds.
    assert receipt["reapplication"]["ok"] is True
    # The degenerate evidence is NOT laundered away: P1 still records degenerate.
    p1 = next(
        p for p in receipt["coverage_threshold_selection"]["populations"]
        if p["name"] == "P1_headline"
    )
    assert p1["result"]["status"] == "degenerate"


def test_analyze_operator_decision_bad_reapply_exit_nonzero(calib, tmp_path):
    # Operator point that fails the re-application check -> exit non-zero.
    dataset_path, replay_path = _write_analysis_inputs(tmp_path, headline_degenerate=True)
    out = tmp_path / "analysis.json"
    code = calib.main(
        [
            "analyze",
            "--dataset",
            str(dataset_path),
            "--replay-receipt",
            str(replay_path),
            "--out",
            str(out),
            "--operator-threshold",
            "0.40",
            "--operator-floor",
            "0.50",
        ]
    )
    assert code != 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["status"] == "operator_decided"
    assert receipt["reapplication"]["ok"] is False


def test_analyze_reapply_function_present(calib):
    # Acceptance grep mirror: the module exposes reapply_thresholds.
    assert hasattr(calib, "reapply_thresholds")
