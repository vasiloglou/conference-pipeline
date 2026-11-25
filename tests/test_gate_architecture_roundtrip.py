"""Round-trip + config-shape tests for the v2.4 graduation-contract gate config
(Phase 03, 03-01).

This phase is a NEGATIVE-RESULT documentation/config phase. The two artifacts
under test are:

  * ``figure_gate_architecture_config.json`` (phase dir, tracked) - an
    architecture-only gate spec with ``status: do_not_adopt_metric_gate`` and NO
    numeric thresholds/floors (D-01 honesty contract).
  * the round-trip itself (D-02) - replaying the config over the Phase 02 labeled
    dataset reproduces the DEGENERATE classification: no metric gate separates
    pass/fail; figures route to the (future) panel.

The round-trip REUSES the existing, already-unit-tested
``classify_reapply_tuple`` from ``scripts/presentations/calibrate_figure_thresholds.py``
via the same importlib module-load fixture used by ``tests/test_calibration_analysis.py``.
That production/script function is NEVER modified (D-08).

CRITICAL honesty mechanism (T-02-23 / T-03-10): the metric gates (coverage,
embedding floor) are retired by EXCLUDING their gate names from the firing-gate
projection AFTER calling ``classify_reapply_tuple`` - they are NOT silenced by any
``t_cov`` / ``floor`` value. The coverage gate fires on the 7 null-coverage rows
regardless of ``t_cov`` (the ``coverage is None`` branch always fires), and the
floor gate is forced to fire on every non-null-DINOv2 row by passing
``floor=float("inf")`` (every finite similarity is ``< inf``). So for BOTH metric
gates, projection-by-exclusion is the only thing that retires them: the gate names
are dropped from the firing-gate list AFTER the call, never suppressed by the
threshold value. (Passing ``floor=float("-inf")`` would silence the floor gate by
value - nothing is ``< -inf`` - which is exactly the mechanism this contract swears
is NOT used and would make the ``"floor"`` exclusion a no-op; ``+inf`` makes the
exclusion genuinely load-bearing.) The test NEVER invokes the ``analyze`` CLI
operator path (that would fabricate a passing operating point). All pinned counts
were independently re-verified against the real (gitignored) dataset.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_CALIB_PATH = REPO_ROOT / "scripts" / "presentations" / "calibrate_figure_thresholds.py"

_CONFIG_PATH = (
    REPO_ROOT
    / ".planning"
    / "workstreams"
    / "v2-4"
    / "phases"
    / "03-graduation-contract-for-v2.3-adoption"
    / "figure_gate_architecture_config.json"
)

# The Phase 02 labeled dataset lives in the gitignored deck dir; it IS present on
# disk on the main tree (that is why the round-trip runs here, not in a worktree).
_DATASET_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "decks"
    / "03_evaluation_benchmarks_data"
    / "calibration_dataset.json"
)


def _load_calib_module():
    """Load calibrate_figure_thresholds.py via importlib (scripts/ convention).

    Verbatim copy of the loader in tests/test_calibration_analysis.py so the test
    imports classify_reapply_tuple WITHOUT modifying that file (D-08).
    """

    spec = importlib.util.spec_from_file_location(
        "calibrate_figure_thresholds_gatearch_test", _CALIB_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def calib():
    """Round-trip machinery module, loaded once per session (offline, no providers)."""

    return _load_calib_module()


@pytest.fixture(scope="session")
def gate_config() -> dict:
    """The architecture-only gate config under test (Task 1 artifact)."""

    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    assert isinstance(config, dict)
    return config


@pytest.fixture(scope="session")
def labeled_rows() -> list[dict]:
    """The Phase 02 labeled dataset rows (V5 input validation: shape-checked).

    Skips only if the gitignored evidence is genuinely absent (it is present on
    disk on the main tree). The dataset is a trusted, frozen Phase 02 contract.
    """

    if not _DATASET_PATH.exists():
        pytest.skip(f"gitignored Phase 02 evidence absent: {_DATASET_PATH}")
    try:
        with _DATASET_PATH.open(encoding="utf-8") as handle:
            dataset = json.load(handle)
    except PermissionError as exc:
        # The deck file is a symlink into the Google Drive backup leaf; a
        # sandboxed run (macOS TCC on ~/Library/CloudStorage) cannot read it.
        pytest.skip(f"gitignored Phase 02 evidence unreadable in this environment ({exc.strerror}): {_DATASET_PATH}")
    assert isinstance(dataset, dict), "dataset must be a JSON object"
    rows = dataset.get("rows")
    assert isinstance(rows, list), "dataset['rows'] must be a list"
    assert rows, "dataset['rows'] must be non-empty"
    for row in rows:
        assert isinstance(row, dict), "each row must be an object"
        assert "slide_id" in row, "each row must carry slide_id"
        assert "label" in row, "each row must carry label"
    return rows


# ---------------------------------------------------------------------------
# Expected gate roles/status (derived from the operator_resolution single source
# of truth; pinned so a silently-weakened config cannot pass).
# ---------------------------------------------------------------------------
_EXPECTED_GATE_IDS = [
    "claim_match_grounding_skip",
    "margin_sign_test",
    "dinov2_reject_floor",
    "coverage",
    "clip_similarity",
    "llm_panel",
]

_EXPECTED_ROLE_STATUS = {
    "claim_match_grounding_skip": ("structural", "keep"),
    "margin_sign_test": ("structural_content_sanity", "keep"),
    "dinov2_reject_floor": ("reject_floor", "uncalibrated_direction_only"),
    "coverage": ("metric_gate", "retired"),
    "clip_similarity": ("metric_gate", "retired"),
    "llm_panel": ("accept_gate", "recommended_future_milestone"),
}


@pytest.mark.repo_data(".planning/workstreams/v2-4/phases/03-graduation-contract-for-v2.3-adoption/figure_gate_architecture_config.json")
def test_config_shape(gate_config: dict) -> None:
    """Assert the full Task 1 config contract (D-01 honesty + roles/status)."""

    # Top-level honest marker + schema + description referencing the negative result.
    assert gate_config["status"] == "do_not_adopt_metric_gate"
    assert gate_config["schema_version"] == 1
    description = gate_config["description"]
    assert isinstance(description, str) and description
    assert "negative" in description.lower()

    gates = gate_config["gates"]
    assert [g["id"] for g in gates] == _EXPECTED_GATE_IDS

    # D-01 honesty: every gate has threshold == null and any floor field == null.
    for gate in gates:
        assert gate.get("threshold") is None, f"{gate['id']} carries a threshold"
        assert gate.get("floor") is None, f"{gate['id']} carries a floor"

    # Roles/status match the operator_resolution-derived expectations.
    by_id = {g["id"]: g for g in gates}
    for gate_id, (role, status) in _EXPECTED_ROLE_STATUS.items():
        assert by_id[gate_id]["role"] == role, gate_id
        assert by_id[gate_id]["status"] == status, gate_id

    # CALIB-07: CLIP retired (encodes the D-04 CLIP-retire decision).
    assert by_id["clip_similarity"]["status"] == "retired"
    assert by_id["clip_similarity"]["req"] == "CALIB-07"

    # DINOv2 reject-floor is direction-only with NO cutoff encoded (D-01).
    dinov2 = by_id["dinov2_reject_floor"]
    assert dinov2["status"] == "uncalibrated_direction_only"
    assert "cutoff" in dinov2["note"].lower()

    # Evidence block carries PATH pointers (not numbers).
    evidence = gate_config["evidence"]
    for key in ("analysis_receipt", "labeling_sensitivity", "dataset"):
        assert evidence[key].endswith(".json")
        assert "03_evaluation_benchmarks_data" in evidence[key]

    # round_trip block: metric gates excluded from the projection (not silenced by
    # a threshold) and the existing machinery is named.
    round_trip = gate_config["round_trip"]
    assert set(round_trip["metric_gates_excluded"]) == {"coverage", "floor"}
    assert "classify_reapply_tuple" in round_trip["machinery"]


def _classify_residual(calib, config: dict, rows: list[dict]):
    """Replay each labeled row through the EXISTING classify_reapply_tuple, then
    PROJECT out the excluded metric gates.

    t_cov=-inf and floor=+inf are chosen so BOTH metric gates FIRE, never so they
    are silenced by value:
      * t_cov=-inf: coverage still fires on the 7 null-coverage rows via the
        ``coverage is None`` branch (the ``< t_cov`` half is moot).
      * floor=+inf: the floor gate fires on every non-null-DINOv2 row because
        every finite similarity is ``< inf``.
    Retirement of both gates is then expressed SOLELY by excluding their gate names
    from the firing-gate list AFTER the call, so the ``"floor"`` exclusion is
    genuinely load-bearing (a ``floor=-inf`` would silence the gate by value and
    make the exclusion a no-op - the honesty defect WR-01 fixed).
    """

    excluded = set(config["round_trip"]["metric_gates_excluded"])
    assert excluded == {"coverage", "floor"}

    records = []
    for row in rows:
        raw_classification, raw_gates = calib.classify_reapply_tuple(
            coverage=row.get("coverage"),
            dinov2_similarity=row.get("dinov2_similarity"),
            margin_sign=row.get("margin_sign"),
            critique_verdict=row.get("critique_verdict"),
            t_cov=float("-inf"),
            floor=float("inf"),
        )
        residual_gates = [g for g in raw_gates if g not in excluded]
        residual_classification = "fail" if residual_gates else "pass"
        records.append(
            {
                "slide_id": row["slide_id"],
                "prefix": row["slide_id"][:5],
                "label": row["label"],
                "raw_classification": raw_classification,
                "raw_gates": raw_gates,
                "residual_gates": residual_gates,
                "residual_classification": residual_classification,
            }
        )
    return records


@pytest.mark.repo_data(".planning/workstreams/v2-4/phases/03-graduation-contract-for-v2.3-adoption/figure_gate_architecture_config.json")
def test_roundtrip_reproduces_degenerate(
    calib, gate_config: dict, labeled_rows: list[dict]
) -> None:
    """D-02 degenerate round-trip with metric gates excluded from the projection.

    All counts are pinned against the independently re-verified real dataset so a
    degenerate "always pass" test or silent dataset-narrowing cannot satisfy it
    (T-02-23 / T-03-10).
    """

    records = _classify_residual(calib, gate_config, labeled_rows)

    # --- Assertion 0: corpus-size + class-balance anchor (WR-03) ----------------
    # Pin the full Phase 02 corpus size and the labeled-class balance so a silently
    # narrowed/regenerated dataset fails LOUDLY here rather than slipping past the
    # downstream count assertions. The residual split must also sum to the corpus.
    assert len(records) == 35, len(records)  # full Phase 02 corpus
    label_counts = Counter(r["label"] for r in records)
    assert label_counts["pass"] == 18, label_counts  # known 18 labeled-pass
    assert label_counts["fail"] == 17, label_counts  # known 17 labeled-fail
    assert label_counts["pass"] > 0 and label_counts["fail"] > 0, label_counts

    # --- Assertion A: no metric gate is the SOLE reason any row fails -----------
    # Rows whose RAW firing gates are non-empty but become EMPTY after dropping
    # the excluded metric gates flip raw-fail -> residual-pass. Because BOTH metric
    # gates are forced to fire (coverage on the 7 null-coverage rows, floor on every
    # non-null-DINOv2 row via floor=+inf), EVERY residual-pass row is a metric-only
    # flip - there are no "clean" raw-pass rows. So the flip count is pinned to the
    # residual-pass count (12), which is the load-bearing proof that the name-
    # exclusion projection (not a silencing value) is what retires the metric gates.
    metric_only_flips = [
        r
        for r in records
        if r["raw_gates"] and not r["residual_gates"]
    ]
    assert len(metric_only_flips) == 12, [r["slide_id"] for r in metric_only_flips]
    for r in metric_only_flips:
        assert r["raw_classification"] == "fail"
        assert r["residual_classification"] == "pass"
        # Every flip's raw firing set is metric-only (a subset of the excluded set);
        # no margin/critique gate is hiding behind the projection.
        assert set(r["raw_gates"]) <= {"coverage", "floor"}, r["slide_id"]
    # The coverage-driven labeled-pass flips are still exactly sl_01, sl_12 (the 2
    # null-coverage rows retired by EXCLUDING "coverage"); this sub-invariant is the
    # honest coverage half that held before WR-01 and must be preserved.
    coverage_flip_prefixes = sorted(
        r["prefix"] for r in metric_only_flips if "coverage" in r["raw_gates"]
    )
    assert coverage_flip_prefixes == ["sl_01", "sl_12"], coverage_flip_prefixes
    for r in metric_only_flips:
        if "coverage" in r["raw_gates"]:
            assert r["label"] == "pass"

    # --- Assertion B: no labeled-pass row is blocked SOLELY by a metric gate ----
    # Asserting "metric gate not in residual_gates" would be vacuous (the projection
    # removes coverage/floor from residual_gates by construction, for any dataset,
    # including an empty one). Instead constrain the RAW (pre-projection) firing set,
    # which CAN contain metric gates: for every labeled-pass row whose RAW gates fired
    # ONLY metric gates (coverage/floor), the projection must clear it to residual-pass
    # - i.e. the row was held back by nothing but a retired metric gate, and exclusion
    # genuinely releases it. We also confirm at least one such row exists, so this
    # assertion cannot pass vacuously on an empty or narrowed dataset.
    metric_only_raw_pass_rows = [
        r
        for r in records
        if r["label"] == "pass"
        and r["raw_gates"]
        and set(r["raw_gates"]) <= {"coverage", "floor"}
    ]
    assert metric_only_raw_pass_rows, "expected >=1 labeled-pass row gated only by a metric gate"
    for r in metric_only_raw_pass_rows:
        # raw-fail blocked only by metric gate(s) -> projection clears it to pass.
        assert r["raw_classification"] == "fail", r["slide_id"]
        assert r["residual_classification"] == "pass", r["slide_id"]
        assert not r["residual_gates"], r["slide_id"]
    # And no labeled-pass row may keep a metric gate after projection (the projection
    # must drop coverage/floor whenever they fired raw) - checked against raw, so it
    # constrains real firings rather than restating the projection's definition.
    for r in records:
        if r["label"] == "pass":
            for metric_gate in ("coverage", "floor"):
                if metric_gate in r["raw_gates"]:
                    assert metric_gate not in r["residual_gates"], (r["slide_id"], metric_gate)

    # --- Assertion C: degenerate residual (pinned) ------------------------------
    # The union of all residual firing gates is exactly {margin, critique} (no
    # metric gate survives the projection); residual split exactly 23 fail / 12
    # pass; residual fails decompose margin-only 10 / critique-only 9 / both 4.
    residual_union: set[str] = set()
    for r in records:
        residual_union.update(r["residual_gates"])
    assert residual_union == {"margin", "critique"}, residual_union

    residual_split = Counter(r["residual_classification"] for r in records)
    assert residual_split["fail"] == 23, residual_split
    assert residual_split["pass"] == 12, residual_split
    # The split must account for every row (WR-03 narrowing guard).
    assert residual_split["fail"] + residual_split["pass"] == len(records), residual_split

    fail_gate_sets = [set(r["residual_gates"]) for r in records if r["residual_gates"]]
    margin_only = sum(1 for s in fail_gate_sets if s == {"margin"})
    critique_only = sum(1 for s in fail_gate_sets if s == {"critique"})
    both = sum(1 for s in fail_gate_sets if s == {"margin", "critique"})
    assert margin_only == 10, margin_only
    assert critique_only == 9, critique_only
    assert both == 4, both

    # --- Assertion D: kept margin gate fires as EXPECTED ------------------------
    # Exactly 6 LABEL == "pass" rows remain residual-fail with "margin" in their
    # residual gates: sl_13 (x2), sl_14, sl_22, sl_25 (x2). This is the KEPT
    # margin sign test firing on genuinely-negative-margin rows - EXPECTED
    # surviving-gate behavior, NOT a defect.
    margin_fired_labeled_pass = [
        r
        for r in records
        if r["label"] == "pass" and "margin" in r["residual_gates"]
    ]
    assert len(margin_fired_labeled_pass) == 6, [
        r["slide_id"] for r in margin_fired_labeled_pass
    ]
    prefix_counts = Counter(r["prefix"] for r in margin_fired_labeled_pass)
    assert set(prefix_counts) == {"sl_13", "sl_14", "sl_22", "sl_25"}, prefix_counts
    assert prefix_counts["sl_13"] == 2, prefix_counts
    assert prefix_counts["sl_25"] == 2, prefix_counts
    assert prefix_counts["sl_14"] == 1, prefix_counts
    assert prefix_counts["sl_22"] == 1, prefix_counts
