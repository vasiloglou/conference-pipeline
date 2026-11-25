"""Unit tests for the v2.4 figure gate stack (Phase 01).

Covers:
- CALIB-01 (`-k dinov2`): pinned-revision constants, D-02 preflight hard-fail
  with prefetch remediation, post-preflight degrade-to-warning, runtime-info
  receipt fields, and one cache-gated real-load self-similarity check.
- CALIB-02 (`-k coverage`): deterministic component-coverage arithmetic
  (normalization, dedupe, empty-lists -> None blocking semantics), the
  structured-critique payload validation with parse status, and the
  per-attempt gate decision's coverage paths. Pure functions loaded from the
  trial runner via importlib — no provider calls.
- CALIB-03 (`-k margin`): pure margin-vs-distractors sign test — positive,
  negative, empty-pool, and strict-inequality tie paths, plus the
  bound_papers_v1 distractor-pool helper and the gate decision's
  negative-sign blocking path. No ML dependency.
- CALIB-04 (`-k claim_match`): pooled provenance ranking — claim_match
  parse/clamp/coercion of untrusted model JSON, pool truncation flag,
  collect-all-then-rank-once pooling, per-paper cap, and the GroundingSkip
  redraw-tier skip signal. No network, no PyMuPDF: every external step is
  monkeypatched.
- CALIB-05 (`-k estimator`): fixture SVGs proving the exact phase16 estimator
  wiring the trial uses catches the attempt-1-class oversized-arrowhead
  failure mode, and that findings reach the next attempt's repair feedback.

No unmarked test downloads a model: monkeypatched stubs stand in for the
DINOv2 bundle, and the single real-load test skips unless the pinned
snapshot is already in the local HF cache.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from conference_pipeline.presentation import figure_grounding, figure_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER_PATH = REPO_ROOT / "scripts" / "presentations" / "run_figure_trial.py"


def _load_runner_module():
    """Load run_figure_trial.py via importlib (scripts/ module convention).

    Module-scope PHASE16 load inside the runner is cheap and offline; no
    model, provider, or browser work happens at import time.
    """

    spec = importlib.util.spec_from_file_location("figure_trial_runner_test", _RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


_DINOV2_SNAPSHOT_DIR = (
    Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-base"
)


def _dinov2_real_load_available() -> bool:
    return (
        _DINOV2_SNAPSHOT_DIR.exists()
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
        and importlib.util.find_spec("transformers") is not None
    )


@pytest.fixture(autouse=True)
def _clear_dinov2_cache():
    """Pop the dinov2 bundle around every test so stubs never leak."""

    figure_similarity._MODEL_CACHE.pop("dinov2", None)
    yield
    figure_similarity._MODEL_CACHE.pop("dinov2", None)


class _RaisingProcessor:
    def __call__(self, *args, **kwargs):
        raise ValueError("processor exploded")


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


def _write_png(path: Path, size: int = 32) -> Path:
    from PIL import Image

    Image.new("RGB", (size, size), color=(21, 129, 88)).save(path)
    return path


def test_dinov2_revision_pinned():
    assert (
        figure_similarity.DINOV2_REVISION
        == "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    )
    assert figure_similarity.DINOV2_MODEL_NAME == "facebook/dinov2-base"


def test_dinov2_preflight_raises_with_prefetch_instructions(monkeypatch):
    def _raise_oserror():
        raise OSError("weights not found in cache")

    monkeypatch.setattr(figure_similarity, "_load_dinov2", _raise_oserror)
    with pytest.raises(RuntimeError) as excinfo:
        figure_similarity.preflight_dinov2()
    message = str(excinfo.value)
    assert "prefetch" in message.lower()
    assert "f9e44c814b77203eaa57a6bdbbd535f21ede1415" in message


def test_dinov2_similarity_degrades_to_none_with_warning(monkeypatch, tmp_path):
    stub_bundle = {
        "processor": _RaisingProcessor(),
        "model": object(),
        "torch": _FakeTorch(),
        "device": "cpu",
    }
    monkeypatch.setattr(
        figure_similarity, "_load_dinov2", lambda: stub_bundle
    )
    png = _write_png(tmp_path / "render.png")
    warnings: list[str] = []
    result = figure_similarity.dinov2_image_similarity(png, png, warnings)
    assert result is None
    assert len(warnings) == 1
    assert str(png) in warnings[0]


def test_dinov2_runtime_info_fields(monkeypatch):
    monkeypatch.setitem(
        figure_similarity._MODEL_CACHE,
        "dinov2",
        {
            "processor": object(),
            "model": object(),
            "torch": _FakeTorch(),
            "device": "cpu",
        },
    )
    info = figure_similarity.dinov2_runtime_info()
    assert set(info) == {"model_name", "revision", "device", "hf_hub_offline"}
    assert info["revision"] == "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    assert info["model_name"] == "facebook/dinov2-base"
    assert info["device"] == "cpu"
    assert isinstance(info["hf_hub_offline"], bool)


def test_hf_hub_offline_field_respects_falsy_overrides(monkeypatch):
    # WR-03 (01-06): the receipt field follows huggingface_hub's own env-var
    # semantics — "0"/"false" mean offline-disabled. monkeypatch is required
    # here: importing the replay module sets HF_HUB_OFFLINE=1 process-wide
    # (01-REVIEW.md IN-06), so this test must control the variable locally.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert figure_similarity.dinov2_runtime_info()["hf_hub_offline"] is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "false")
    assert figure_similarity.dinov2_runtime_info()["hf_hub_offline"] is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert figure_similarity.dinov2_runtime_info()["hf_hub_offline"] is True
    monkeypatch.setenv("HF_HUB_OFFLINE", "TRUE")
    assert figure_similarity.dinov2_runtime_info()["hf_hub_offline"] is True
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert figure_similarity.dinov2_runtime_info()["hf_hub_offline"] is False


@pytest.mark.skipif(
    not _dinov2_real_load_available(),
    reason="pinned DINOv2 snapshot or optional ML runtime is unavailable; "
    "run prefetch_dinov2_weights() and install torch/torchvision/transformers first",
)
def test_dinov2_real_load_self_similarity(tmp_path):
    png = _write_png(tmp_path / "fixture.png")
    similarity = figure_similarity.dinov2_image_similarity(png, png)
    assert similarity is not None
    assert abs(similarity - 1.0) < 1e-3
    info = figure_similarity.preflight_dinov2()
    assert info["device"] in {"mps", "cpu"}


def test_margin_sign_positive():
    result = figure_similarity.margin_sign_test(0.75, [0.64, 0.61])
    assert result["margin"] == pytest.approx(0.11)
    assert result["margin_sign"] is True
    assert result["distractor_count"] == 2


def test_margin_sign_negative():
    result = figure_similarity.margin_sign_test(0.60, [0.70])
    assert result["margin"] == pytest.approx(-0.10)
    assert result["margin_sign"] is False


def test_margin_sign_empty_pool():
    result = figure_similarity.margin_sign_test(0.5, [])
    assert result == {
        "margin": None,
        "margin_sign": None,
        "distractor_count": 0,
    }


def test_margin_sign_tie_not_positive():
    result = figure_similarity.margin_sign_test(0.5, [0.5])
    assert result["margin"] == 0.0
    assert result["margin_sign"] is False


# --------------------------------------------------------------------------
# CALIB-04: pooled provenance ranking + claim_match (`-k claim_match`)
# --------------------------------------------------------------------------


class _RankerTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _RankerStubReceipt:
    def __init__(self, receipt_path: str) -> None:
        self.receipt_path = receipt_path


class _RankerStubResponse:
    def __init__(self, text: str) -> None:
        self.content = [_RankerTextBlock(text)]


class _RankerStubResult:
    def __init__(self, text: str) -> None:
        self.response = _RankerStubResponse(text)
        self.receipt = _RankerStubReceipt("/fake/receipts/ranker.json")


class _RankerCallStub:
    """Stands in for anthropic_messages_create; records every call."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, client, **kwargs):
        self.calls.append(kwargs)
        return _RankerStubResult(self.text)


def _fake_pooled_candidates(
    count: int, *, paper_title: str = "Paper A", note_id: str = "noteAAAA01"
) -> list[dict]:
    return [
        {
            "page": index + 1,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "image_path": f"/fake/crops/{note_id}/crop_{index}.png",
            "caption": f"Figure {index + 1}",
            "paper_title": paper_title,
            "note_id": note_id,
        }
        for index in range(count)
    ]


def _run_rank_candidates(monkeypatch, *, response_text: str, candidates: list[dict]):
    stub = _RankerCallStub(response_text)
    monkeypatch.setattr(figure_grounding, "anthropic_messages_create", stub)
    monkeypatch.setattr(
        figure_grounding,
        "_encode_image_for_ranker",
        lambda path: {"type": "text", "text": f"[image {path.name}]"},
    )
    verdict = figure_grounding.rank_candidates(
        client=object(),
        caller="test",
        model="stub-model",
        candidates=candidates,
        slide_claim="Proof Flow improves verified pass rates",
        slide_title="Results",
    )
    return verdict, stub


def test_claim_match_parsed_and_clamped(monkeypatch):
    verdict, _ = _run_rank_candidates(
        monkeypatch,
        response_text=(
            '{"chosen_index": 0, "claim_match": 1.7, '
            '"figure_kind": "results_chart", "rationale": "Paper A reports this."}'
        ),
        candidates=_fake_pooled_candidates(2),
    )
    assert verdict["claim_match"] == 1.0
    assert "claim_match_parse_warning" not in verdict
    assert "pool_truncated" not in verdict

    verdict, _ = _run_rank_candidates(
        monkeypatch,
        response_text=(
            '{"chosen_index": 0, "claim_match": "high", '
            '"figure_kind": "results_chart", "rationale": "Paper A reports this."}'
        ),
        candidates=_fake_pooled_candidates(2),
    )
    assert verdict["claim_match"] == 0.0
    assert "claim_match_parse_warning" in verdict


def test_claim_match_missing_defaults_zero(monkeypatch):
    verdict, _ = _run_rank_candidates(
        monkeypatch,
        response_text=(
            '{"chosen_index": 1, "figure_kind": "architecture", '
            '"rationale": "Paper A architecture."}'
        ),
        candidates=_fake_pooled_candidates(2),
    )
    assert verdict["claim_match"] == 0.0
    assert "claim_match_parse_warning" in verdict


def test_claim_match_pool_truncation_flag(monkeypatch):
    verdict, stub = _run_rank_candidates(
        monkeypatch,
        response_text=(
            '{"chosen_index": 0, "claim_match": 0.9, '
            '"figure_kind": "results_chart", "rationale": "Paper A."}'
        ),
        candidates=_fake_pooled_candidates(17),
    )
    assert verdict["pool_truncated"] is True
    assert len(stub.calls) == 1
    content = stub.calls[0]["messages"][0]["content"]
    headers = [
        block
        for block in content
        if block.get("type") == "text" and block["text"].startswith("Candidate ")
    ]
    assert len(headers) == figure_grounding.RANKER_MAX_POOLED_CANDIDATES == 16


_NOTE_A = "noteAAAA01"
_NOTE_B = "noteBBBB02"


def _stub_grounding_pipeline(monkeypatch, tmp_path, *, response_text: str):
    """Stub every external step of ground_slide_figure: two fake notes with
    2 and 12 fake candidates; no network, no PyMuPDF, no PIL."""

    notes = {
        _NOTE_A: {
            "openreview_id": _NOTE_A,
            "title": "Paper A",
            "authors": ["Ada Author"],
            "license": "CC BY 4.0",
            "venue": "ICLR 2026",
            "forum_url": f"https://openreview.net/forum?id={_NOTE_A}",
        },
        _NOTE_B: {
            "openreview_id": _NOTE_B,
            "title": "Paper B",
            "authors": ["Bo Builder", "Cy Coder"],
            "license": "CC BY 4.0",
            "venue": "ICLR 2026",
            "forum_url": f"https://openreview.net/forum?id={_NOTE_B}",
        },
    }
    candidate_counts = {_NOTE_A: 2, _NOTE_B: 12}

    def fake_resolve(note_id, **kwargs):
        return notes[note_id]

    def fake_ensure(note_id, cache_dir, **kwargs):
        return tmp_path / f"{note_id}.pdf"

    def fake_extract(pdf_path, output_dir):
        note_id = pdf_path.stem
        candidates = []
        for index in range(candidate_counts[note_id]):
            crop = tmp_path / f"{note_id}_crop_{index}.png"
            crop.write_bytes(b"fake-crop-bytes-%d" % index)
            candidates.append(
                {
                    "page": index + 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "image_path": str(crop),
                    "caption": f"Figure {index + 1}",
                }
            )
        return candidates

    stub = _RankerCallStub(response_text)
    monkeypatch.setattr(figure_grounding, "resolve_openreview_note", fake_resolve)
    monkeypatch.setattr(figure_grounding, "ensure_paper_pdf", fake_ensure)
    monkeypatch.setattr(figure_grounding, "extract_candidate_crops", fake_extract)
    monkeypatch.setattr(figure_grounding, "anthropic_messages_create", stub)
    monkeypatch.setattr(
        figure_grounding,
        "_encode_image_for_ranker",
        lambda path: {"type": "text", "text": f"[image {path.name}]"},
    )
    return stub


def _call_ground_slide_figure(tmp_path):
    return figure_grounding.ground_slide_figure(
        client=object(),
        caller="test",
        ranker_model="stub-model",
        source_ids=[f"paper:{_NOTE_A}", f"paper_abstract:{_NOTE_B}"],
        slide_claim="Proof Flow improves verified pass rates",
        slide_title="Results",
        grounding_root=tmp_path / "grounding",
    )


def test_claim_match_below_threshold_returns_skip(monkeypatch, tmp_path):
    _stub_grounding_pipeline(
        monkeypatch,
        tmp_path,
        response_text=(
            '{"chosen_index": 0, "claim_match": 0.4, "figure_kind": "results_chart", '
            '"rationale": "Paper A axis labels overlap the claim."}'
        ),
    )
    skip = _call_ground_slide_figure(tmp_path)
    assert isinstance(skip, figure_grounding.GroundingSkip)
    assert skip.reason == "claim_match_below_threshold"
    assert skip.claim_match == 0.4
    assert skip.claim_match_threshold == 0.7
    payload = skip.to_dict()
    assert payload == {
        "reason": "claim_match_below_threshold",
        "claim_match": 0.4,
        "claim_match_threshold": 0.7,
        "pooled_notes": [_NOTE_A, _NOTE_B],
        "pool_candidate_count": 10,
        "ranker_rationale": "Paper A axis labels overlap the claim.",
        "ranker_receipt_path": "/fake/receipts/ranker.json",
        "warnings": skip.warnings,
    }


def test_claim_match_above_threshold_returns_pooled_result(monkeypatch, tmp_path):
    stub = _stub_grounding_pipeline(
        monkeypatch,
        tmp_path,
        response_text=(
            '{"chosen_index": 0, "claim_match": 0.9, "figure_kind": "results_chart", '
            '"rationale": "Paper A reports exactly this result."}'
        ),
    )
    result = _call_ground_slide_figure(tmp_path)
    assert isinstance(result, figure_grounding.GroundingResult)
    assert result.claim_match == 0.9
    assert result.claim_match_threshold == 0.7
    assert result.pooled_notes == [_NOTE_A, _NOTE_B]
    assert result.pool_candidate_count == 10  # 2 + min(12, 8)
    assert result.pool_truncated is False
    assert result.openreview_id == _NOTE_A
    assert result.license_adaptable is True
    # ONE pooled ranking call — first-usable-source-wins is gone.
    assert len(stub.calls) == 1
    payload = result.to_dict()
    for key in (
        "claim_match",
        "claim_match_threshold",
        "pooled_notes",
        "pool_candidate_count",
        "pool_truncated",
    ):
        assert key in payload


def test_claim_match_per_paper_cap_applied(monkeypatch, tmp_path):
    stub = _stub_grounding_pipeline(
        monkeypatch,
        tmp_path,
        response_text=(
            '{"chosen_index": 0, "claim_match": 0.9, "figure_kind": "results_chart", '
            '"rationale": "Paper A."}'
        ),
    )
    result = _call_ground_slide_figure(tmp_path)
    assert isinstance(result, figure_grounding.GroundingResult)
    assert any(_NOTE_B in warning for warning in result.warnings)
    content = stub.calls[0]["messages"][0]["content"]
    paper_b_headers = [
        block
        for block in content
        if block.get("type") == "text"
        and block["text"].startswith("Candidate ")
        and "Paper B" in block["text"]
    ]
    assert len(paper_b_headers) == figure_grounding.RANKER_MAX_CANDIDATES == 8


def test_claim_match_ranker_rejected_all_returns_skip(monkeypatch, tmp_path):
    _stub_grounding_pipeline(
        monkeypatch,
        tmp_path,
        response_text=(
            '{"chosen_index": -1, "claim_match": 0.9, "figure_kind": "other", '
            '"rationale": "No candidate paper makes this claim."}'
        ),
    )
    skip = _call_ground_slide_figure(tmp_path)
    assert isinstance(skip, figure_grounding.GroundingSkip)
    assert skip.reason == "ranker_rejected_all"
    assert skip.pooled_notes == [_NOTE_A, _NOTE_B]
    assert skip.pool_candidate_count == 10
    assert skip.to_dict()["reason"] == "ranker_rejected_all"


# --------------------------------------------------------------------------
# CALIB-02: deterministic component coverage (`-k coverage`)
# --------------------------------------------------------------------------


def test_coverage_basic_fraction():
    matched = [f"component {i}" for i in range(8)]
    coverage = RUNNER.component_coverage(matched, ["legend"], ["extra box"])
    assert coverage == pytest.approx(0.8)


def test_coverage_normalization_dedupes():
    # "X Axis" and "x  axis" normalize identically: whitespace-collapse +
    # casefold + per-list set dedupe -> matched counts as 1.
    coverage = RUNNER.component_coverage(["X Axis", "x  axis"], ["Legend"], [])
    assert coverage == pytest.approx(0.5)


def test_coverage_empty_lists_returns_none():
    # None means the BLOCKING gate fails: an unlabeled gate cannot pass.
    assert RUNNER.component_coverage([], [], []) is None


def test_coverage_drops_blank_items():
    # Empty and whitespace-only items are dropped: only "node" counts.
    coverage = RUNNER.component_coverage(["", "  ", "node"], ["axis"], [])
    assert coverage == pytest.approx(0.5)
    assert RUNNER.component_coverage(["", "  ", "node"], [], []) == pytest.approx(1.0)


def test_coverage_validate_critique_payload_parse_status():
    complete = {
        "verdict": "pass",
        "matched_components": ["box a", 7],
        "missing_components": [],
        "invented_components": ["ghost arrow"],
    }
    normalized, status = RUNNER._validate_critique_payload(complete)
    assert status == "ok"
    # Non-string items are coerced via str().
    assert normalized["matched_components"] == ["box a", "7"]

    incomplete = {
        "verdict": "fail",
        "matched_components": ["box a"],
        "missing_components": ["box b"],
    }
    normalized, status = RUNNER._validate_critique_payload(incomplete)
    assert status == "parse_failed"
    assert normalized["invented_components"] == []


# --------------------------------------------------------------------------
# CALIB-02 gap closure (01-06): grounded critique prompt excludes deliberate
# redraw-policy deltas from component counts (operator-directed amendment,
# 01-05-SUMMARY.md). Prompt-only fix: gate math stays pure and untouched.
# --------------------------------------------------------------------------


def test_critique_prompt_grounded_excludes_redraw_policy_deltas():
    prompt = RUNNER._critique_prompt(tier="vision_redraw", has_source=True)
    lowered = prompt.lower()
    # All four operator-named exclusion categories are named in the prompt.
    assert "attribution" in lowered
    assert "title" in lowered or "headline" in lowered
    assert "brand" in lowered
    assert "callout" in lowered
    # The exclusion targets both component lists explicitly.
    assert "must not be listed in missing_components or invented_components" in lowered
    # JSON contract unchanged: all three component-list keys still present.
    assert "matched_components" in prompt
    assert "missing_components" in prompt
    assert "invented_components" in prompt
    # Source-inventory coverage rule retained verbatim in spirit.
    assert (
        "matched_components plus missing_components together must cover the "
        "full source component inventory" in lowered
    )


def test_critique_prompt_grounded_keeps_structural_fidelity_language():
    prompt = RUNNER._critique_prompt(tier="vision_redraw", has_source=True)
    lowered = prompt.lower()
    # Structural-fidelity counting language is intact: dropped/distorted
    # structural components still count as missing, redraw-only structural
    # elements still count as invented.
    assert "missing" in lowered
    assert "invented" in lowered
    assert "structural" in lowered or "structure" in lowered
    assert "do not relax structural fidelity" in lowered
    assert "must still be counted as missing" in lowered
    assert "must still be counted as invented" in lowered


def test_critique_prompt_ungrounded_branch_unchanged():
    # The has_source=False branch carries no component-list contract.
    prompt = RUNNER._critique_prompt(tier="d2", has_source=False)
    assert "matched_components" not in prompt


# --------------------------------------------------------------------------
# Gate decision + margin pool + repair feedback (Task 2 pure helpers)
# --------------------------------------------------------------------------


def test_coverage_gate_decision_blocks_low_coverage():
    decision = RUNNER._attempt_gate_decision(
        critique_verdict="pass",
        coverage=0.6,
        coverage_threshold=0.8,
        margin_sign=True,
    )
    assert decision["accept"] is False
    assert any("0.60" in r and "0.80" in r for r in decision["blocking_reasons"])


def test_coverage_gate_decision_blocks_none_coverage():
    # An unlabeled blocking gate cannot pass (locked CALIB-02 semantics).
    decision = RUNNER._attempt_gate_decision(
        critique_verdict="pass",
        coverage=None,
        coverage_threshold=0.8,
        margin_sign=True,
    )
    assert decision["accept"] is False
    assert any("undefined" in r for r in decision["blocking_reasons"])


def test_margin_gate_decision_blocks_negative_sign():
    decision = RUNNER._attempt_gate_decision(
        critique_verdict="pass",
        coverage=0.9,
        coverage_threshold=0.8,
        margin_sign=False,
    )
    assert decision["accept"] is False
    assert any("negative margin" in r for r in decision["blocking_reasons"])

    # margin_sign None (no distractors / degraded scorer) does NOT block when
    # the other gates pass (locked CALIB-03: only an actual negative sign fails).
    decision = RUNNER._attempt_gate_decision(
        critique_verdict="pass",
        coverage=0.9,
        coverage_threshold=0.8,
        margin_sign=None,
    )
    assert decision["accept"] is True
    assert decision["blocking_reasons"] == []


def test_estimator_findings_join_repair_feedback():
    critique = {
        "structural_issues": ["dropped the encoder stage"],
        "legibility_issues": [],
        "missing_components": ["encoder", "decoder"],
        "invented_components": ["ghost box"],
        "feedback": "restore the encoder stage",
    }
    findings = {
        "oversized_arrowhead": [
            "oversized_arrowhead: a filled shape spans 50x50px; arrowheads must stay ~10px per side"
        ],
        "line_over_text": ["line_over_text: a connector line crosses the label 'encoder'"],
        "text_overlap": [],
        "text_exceeds_canvas": [],
    }
    margin_note = "negative margin: render closer to a distractor crop than its claimed source"
    feedback = RUNNER._repair_feedback(critique, findings, margin_note)
    assert "oversized_arrowhead: a filled shape spans 50x50px" in feedback
    assert "line_over_text: a connector line crosses" in feedback
    assert "missing components: encoder, decoder" in feedback
    assert "invented components: ghost box" in feedback
    assert "restore the encoder stage" in feedback
    assert margin_note in feedback
    assert len(feedback) <= 1200


def test_margin_distractor_pool_excludes_chosen(tmp_path):
    grounding_root = tmp_path / "grounding"
    crops_dir = grounding_root / "crops" / _NOTE_A
    crops_dir.mkdir(parents=True)
    crop_paths = []
    for index in range(3):
        crop = crops_dir / f"crop_{index}.png"
        crop.touch()
        crop_paths.append(crop)
    chosen = crop_paths[1]
    pool = RUNNER._margin_distractor_pool(
        # speech_transcript ids are unresolvable -> contribute nothing.
        [f"paper:{_NOTE_A}", "speech_transcript:proof-flow-talk"],
        chosen,
        grounding_root,
    )
    assert pool["protocol"] == "bound_papers_v1"
    assert pool["notes"] == [_NOTE_A]
    assert len(pool["distractors"]) == 2
    assert chosen.resolve() not in {p.resolve() for p in pool["distractors"]}


# --------------------------------------------------------------------------
# CALIB-05: estimator-wiring fixture SVGs (`-k estimator`)
#
# The fixtures reproduce attempt-1-class oversized-arrowhead geometry (large
# filled triangles over text) through the EXACT phase16 functions the trial
# wires via the runner's PHASE16 constant — no mocks, no stubs. The trigger
# (generate_programmatic_figures.py) is min(width, height) > 16 and
# max(width, height) > 20 on a polygon or closed M..L..L..Z path.
# --------------------------------------------------------------------------

_TRIP_TRIANGLE = "100,100 150,100 125,150"  # 50x50 px: attempt-1-class geometry
_CLEAN_TRIANGLE = "100,100 110,100 105,110"  # 10x10 px: within the style cap


def _write_estimator_fixture_svg(path: Path, *, triangle_points: str) -> Path:
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" '
        'viewBox="0 0 1280 720">'
        '<rect width="1280" height="720" fill="#1a1a2e"/>'
        f'<polygon points="{triangle_points}" fill="#a8d8ea"/>'
        '<text x="100" y="130" font-size="20" fill="#ffffff">Stage label</text>'
        "</svg>\n",
        encoding="utf-8",
    )
    return path


def test_estimator_fixture_triangle_trips_oversized_arrowhead(tmp_path):
    svg = _write_estimator_fixture_svg(
        tmp_path / "trip.svg", triangle_points=_TRIP_TRIANGLE
    )
    findings = RUNNER.PHASE16._phase16_oversized_arrowhead_issues(svg)
    assert findings  # non-empty: the real estimator catches the failure mode
    assert any("oversized_arrowhead" in finding for finding in findings)


def test_estimator_clean_fixture_no_findings(tmp_path):
    svg = _write_estimator_fixture_svg(
        tmp_path / "clean.svg", triangle_points=_CLEAN_TRIANGLE
    )
    assert RUNNER.PHASE16._phase16_oversized_arrowhead_issues(svg) == []


def test_estimator_all_four_callable_on_fixture(tmp_path):
    # Wiring smoke test for the exact four CALIB-05 estimator functions.
    svg = _write_estimator_fixture_svg(
        tmp_path / "clean.svg", triangle_points=_CLEAN_TRIANGLE
    )
    for estimator_name in (
        "_phase16_oversized_arrowhead_issues",
        "_phase16_line_over_text_issues",
        "_phase16_text_overlap_issues",
        "_phase16_text_exceeds_canvas_issues",
    ):
        findings = getattr(RUNNER.PHASE16, estimator_name)(svg)
        assert isinstance(findings, list)


def test_estimator_findings_feed_feedback_payload(tmp_path):
    # The live-loop half of the amended gate: real estimator findings reach
    # the next attempt's repair feedback via _repair_feedback.
    svg = _write_estimator_fixture_svg(
        tmp_path / "trip.svg", triangle_points=_TRIP_TRIANGLE
    )
    triangle_findings = RUNNER.PHASE16._phase16_oversized_arrowhead_issues(svg)
    assert triangle_findings
    findings = {
        "oversized_arrowhead": triangle_findings,
        "line_over_text": [],
        "text_overlap": [],
        "text_exceeds_canvas": [],
    }
    feedback = RUNNER._repair_feedback({"verdict": "fail"}, findings, None)
    assert triangle_findings[0] in feedback


# --------------------------------------------------------------------------
# Replay harness (`-k replay`): classification purity, expected table,
# refusal gate, stage-id separation. No --live, no network, no model loads —
# classification and shape logic only (live execution is plan 01-05).
# --------------------------------------------------------------------------

_REPLAY_PATH = REPO_ROOT / "scripts" / "presentations" / "replay_figure_trial.py"


def _load_replay_module():
    """Load replay_figure_trial.py via importlib (same scripts/ convention).

    Module import sets HF_HUB_OFFLINE via setdefault and re-execs the trial
    runner module under a private name — both cheap and offline; no model,
    provider, or browser work happens at import time.
    """

    spec = importlib.util.spec_from_file_location(
        "figure_trial_replay_test", _REPLAY_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY = _load_replay_module()


def test_replay_classification_pure():
    # fail iff verdict != pass OR coverage is None OR coverage < threshold
    # (amended done gates); otherwise coverage_pass.
    assert (
        REPLAY.classify_replay_figure(
            critique_verdict="pass", coverage=0.9, coverage_threshold=0.8
        )
        == "coverage_pass"
    )
    assert (
        REPLAY.classify_replay_figure(
            critique_verdict="fail", coverage=0.9, coverage_threshold=0.8
        )
        == "fail"
    )
    assert (
        REPLAY.classify_replay_figure(
            critique_verdict="pass", coverage=None, coverage_threshold=0.8
        )
        == "fail"
    )
    # Strict threshold: 0.79 < 0.8 fails.
    assert (
        REPLAY.classify_replay_figure(
            critique_verdict="pass", coverage=0.79, coverage_threshold=0.8
        )
        == "fail"
    )


def test_replay_expected_classifications_table():
    assert REPLAY.EXPECTED_CLASSIFICATIONS == {
        "sl_05": "coverage_pass",
        "sl_06": "coverage_pass",
        "sl_10": "coverage_pass",
        "sl_11": "fail",
        "sl_19": "fail",
        "sl_23": "coverage_pass",
    }


def test_replay_classification_ignores_margin_and_estimators():
    # Margins, estimator findings, and claim_match are recorded evidence
    # ONLY (amended done gates + Pitfall 7): the classification function's
    # signature proves they cannot influence the result.
    import inspect

    parameters = set(inspect.signature(REPLAY.classify_replay_figure).parameters)
    assert parameters == {"critique_verdict", "coverage", "coverage_threshold"}


def test_replay_refuses_without_live(tmp_path):
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(_REPLAY_PATH), "--deck-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "refusing" in proc.stderr.lower()


def test_replay_stage_id_distinct_from_trial():
    # The replay receipts its calls under its own stage dir; the trial
    # receipt directory stays immutable provenance.
    assert REPLAY.REPLAY_STAGE_ID == "figure_trial_v2_replay"
    assert REPLAY.REPLAY_STAGE_ID != RUNNER.TRIAL_STAGE_ID


# --------------------------------------------------------------------------
# Receipt-integrity guards (01-06 gap closure: CR-01 runner refuse-overwrite,
# WR-02 replay --out clobber guard). Pure helpers — no --live, no provider.
# --------------------------------------------------------------------------


def test_runner_refuses_receipt_overwrite_without_force(tmp_path):
    receipt_path = tmp_path / "figure_trial_receipt.json"
    receipt_path.write_text('{"schema_version": 1}', encoding="utf-8")
    refusal = RUNNER._receipt_overwrite_refusal(receipt_path, force=False)
    assert refusal is not None
    assert "schema_version" in refusal
    assert "--force-overwrite-receipt" in refusal
    # force=True overrides the guard; a non-existent path needs no guard.
    assert RUNNER._receipt_overwrite_refusal(receipt_path, force=True) is None
    assert (
        RUNNER._receipt_overwrite_refusal(tmp_path / "missing.json", force=False)
        is None
    )
    # Unparseable receipt still refuses, reporting schema_version unknown.
    receipt_path.write_text("not json {", encoding="utf-8")
    refusal = RUNNER._receipt_overwrite_refusal(receipt_path, force=False)
    assert refusal is not None
    assert "unknown" in refusal


def test_replay_out_guard_detects_trial_receipt_path(tmp_path):
    trial_receipt = tmp_path / "figure_trial_receipt.json"
    trial_receipt.write_text("{}", encoding="utf-8")
    # Same file, expressed once absolute and once via a ".." traversal.
    assert REPLAY._out_clobbers_trial_receipt(trial_receipt, trial_receipt)
    traversal = tmp_path / "subdir" / ".." / "figure_trial_receipt.json"
    assert REPLAY._out_clobbers_trial_receipt(traversal, trial_receipt)
    # The default replay receipt path is NOT the trial receipt.
    assert not REPLAY._out_clobbers_trial_receipt(
        tmp_path / "figure_trial_replay_receipt.json", trial_receipt
    )


# --------------------------------------------------------------------------
# Budget tracker fail-closed (01-06 gap closure, WR-01): an unpriced call
# halts for operator approval instead of silently disabling the cap.
# --------------------------------------------------------------------------


def test_budget_tracker_fails_closed_on_unpriced_call():
    import types

    tracker = RUNNER.BudgetTracker(2.0)
    unpriced = types.SimpleNamespace(
        receipt=types.SimpleNamespace(estimated_cost_usd=None, cost_status="unknown")
    )
    with pytest.raises(RUNNER.TrialBudgetExceeded) as excinfo:
        tracker.record(unpriced)
    assert "cost" in str(excinfo.value)

    # A priced zero-cost call is legitimate: distinguishable from unpriced.
    fresh = RUNNER.BudgetTracker(2.0)
    fresh.record(
        types.SimpleNamespace(
            receipt=types.SimpleNamespace(estimated_cost_usd=0.0, cost_status="ok")
        )
    )
    assert fresh.spent_usd == 0.0

    # Priced calls accumulate and the existing >= cap halt is preserved.
    capped = RUNNER.BudgetTracker(2.0)
    capped.record(
        types.SimpleNamespace(
            receipt=types.SimpleNamespace(estimated_cost_usd=1.5, cost_status="ok")
        )
    )
    assert capped.spent_usd == pytest.approx(1.5)
    with pytest.raises(RUNNER.TrialBudgetExceeded) as excinfo:
        capped.record(
            types.SimpleNamespace(
                receipt=types.SimpleNamespace(estimated_cost_usd=0.6, cost_status="ok")
            )
        )
    assert "budget cap" in str(excinfo.value)
