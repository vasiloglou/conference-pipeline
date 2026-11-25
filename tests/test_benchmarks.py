"""Tests for the benchmark harness, receipt schema, and CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.benchmarks.harness import BenchmarkHarness, RunSpec
from conference_pipeline.benchmarks.receipt_schema import (
    BenchmarkReceipt,
    receipt_from_dict,
    receipt_to_dict,
    validate_receipt,
)
from conference_pipeline.model_governance import ModelGovernanceError, load_registry
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile


def _make_registry():
    return load_registry()


def _make_profile(**overrides) -> LocalRuntimeProfile:
    defaults = dict(
        profile_id="mlx_light",
        runtime_kind="mlx",
        base_url="http://127.0.0.1:8080/v1",
        manage_server=True,
        optional_deps=[],
        startup_timeout_s=1.0,
        request_timeout_s=2.0,
        max_concurrency=1,
        max_batch_size=1,
        lock_scope="heavy",
        hardware_fit_status="verified",
        approval_status="benchmark_pending",
    )
    defaults.update(overrides)
    return LocalRuntimeProfile(**defaults)


class _FakeLock:
    def __init__(self, _path=None):
        self.acquired = []
        self.released = 0

    def acquire(self, metadata):
        self.acquired.append(metadata)

    def release(self):
        self.released += 1

    def is_held(self):
        return False

    def owner_metadata(self):
        return None


class _FakeServer:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.started = []
        self.stopped = 0

    def start(self, profile, model_id, *, lock=None, task_id="managed_mlx_server"):
        self.events.append("server.start")
        self.started.append((profile, model_id, lock, task_id))

    def stop(self):
        self.events.append("server.stop")
        self.stopped += 1


class _FakeClient:
    def __init__(self, output="source evidence from paper", events=None):
        self.output = output
        self.events = events if events is not None else []
        self.complete_calls = []
        self.health_checks = 0

    def complete(self, prompt, *, model, max_tokens=512, **kwargs):
        self.events.append("client.complete")
        self.complete_calls.append((prompt, model, max_tokens, kwargs))
        return self.output

    def health_check(self):
        self.health_checks += 1
        return True


def test_benchmark_harness_dry_run(tmp_path):
    registry = _make_registry()
    harness = BenchmarkHarness(registry, tmp_path)
    receipts = harness.run_all(mode="dry-run")
    assert all(isinstance(r, BenchmarkReceipt) for r in receipts)
    assert len(receipts) > 0
    json_files = list(tmp_path.rglob("*.json"))
    assert len(json_files) >= 1


def test_benchmark_receipt_json_roundtrip():
    r = BenchmarkReceipt(
        workspace="test-workspace",
        task_id="ocr.slide_image_parse",
        task_family="ocr",
        stage="transcription-ocr",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-04-27T00:00:00+00:00",
        execution_mode="dry-run",
        provider="openai",
        model_id="gpt-5.4-mini",
        runtime_id="mock",
        hardware_profile="mock",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://ocr",
        output_ref=None,
        raw_output_snippet="abc123",
        rubric_scores={"quality": 0.9, "schema_ok": 1.0},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=0.0,
        latency_ms=0.0,
        cost_usd=0.0,
        budget_violations=[],
        selection_decision="skipped",
        failure_reason=None,
    )
    assert receipt_from_dict(receipt_to_dict(r)) == r


def test_validate_receipt_rejects_invalid_decision():
    r = BenchmarkReceipt(
        workspace="test-workspace",
        task_id="ocr.slide_image_parse",
        task_family="ocr",
        stage="transcription-ocr",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-04-27T00:00:00+00:00",
        execution_mode="dry-run",
        provider="openai",
        model_id="gpt-5.4-mini",
        runtime_id="mock",
        hardware_profile="mock",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://ocr",
        output_ref=None,
        raw_output_snippet="abc123",
        rubric_scores={},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=None,
        latency_ms=None,
        cost_usd=None,
        budget_violations=[],
        selection_decision="maybe",
        failure_reason=None,
    )
    with pytest.raises(ValueError, match="invalid selection_decision"):
        validate_receipt(r)


def test_run_matrix_includes_authority_and_local(tmp_path):
    registry = _make_registry()
    harness = BenchmarkHarness(registry, tmp_path)

    # Find any task that has benchmark_pending local candidates
    task_id = None
    for tid, policy in registry.tasks.items():
        if any(c.status == "benchmark_pending" for c in policy.local_candidates):
            task_id = tid
            break

    assert task_id is not None, "registry has no tasks with benchmark_pending local candidates"

    specs = harness.build_run_matrix(task_id)
    roles = {s.role for s in specs}
    assert "authority" in roles, f"no authority spec in matrix for {task_id}: {specs}"
    assert "cloud_candidate" in roles, f"no cloud_candidate spec in matrix for {task_id}: {specs}"
    assert "local_candidate" in roles, f"no local_candidate spec in matrix for {task_id}: {specs}"


def test_live_local_calls_mlx_client_complete(tmp_path):
    registry = _make_registry()
    profile = _make_profile()
    client = _FakeClient()
    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=_FakeServer,
        client_factory=lambda base_url, timeout_s: client,
    )
    spec = RunSpec(
        task_id="analysis.source_reading",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )
    input_sample = {"prompt": "summarize this source"}

    receipt = harness.run_task("analysis.source_reading", spec, input_sample, mode="live")

    assert client.complete_calls
    assert client.complete_calls[0][0] == json.dumps(input_sample, sort_keys=True)
    assert client.complete_calls[0][1] == spec.model_id
    assert receipt.selection_decision == "passed"
    assert receipt.hardware_profile != "mock"
    assert receipt.output_ref is not None
    assert Path(receipt.output_ref).exists()


def test_live_local_preflight_failure_fails_closed(tmp_path):
    registry = _make_registry()
    profile = _make_profile()
    client = _FakeClient()

    def fail_preflight(profile, model_id, lock):
        raise ModelGovernanceError("preflight failed")

    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=fail_preflight,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=_FakeServer,
        client_factory=lambda base_url, timeout_s: client,
    )
    spec = RunSpec(
        task_id="analysis.source_reading",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task("analysis.source_reading", spec, {"prompt": "x"}, mode="live")

    assert client.complete_calls == []
    assert receipt.selection_decision == "failed"
    assert "preflight failed" in (receipt.failure_reason or "")
    persisted = json.loads(next((tmp_path / "analysis.source_reading").glob("*.json")).read_text())
    assert persisted["selection_decision"] == "failed"
    assert "preflight failed" in persisted["failure_reason"]


def test_live_local_acquires_and_releases_heavy_lock(tmp_path):
    registry = _make_registry()
    profile = _make_profile(profile_id="mlx_heavy", manage_server=False, lock_scope="heavy")
    locks: list[_FakeLock] = []

    def lock_factory(path):
        lock = _FakeLock(path)
        locks.append(lock)
        return lock

    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_heavy",
        local_profiles={"mlx_heavy": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=lock_factory,
        client_factory=lambda base_url, timeout_s: _FakeClient(),
    )
    spec = RunSpec(
        task_id="analysis.source_reading",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task("analysis.source_reading", spec, {"prompt": "x"}, mode="live")

    assert receipt.selection_decision == "passed"
    assert len(locks) == 1
    lock = locks[0]
    assert len(lock.acquired) == 1
    assert lock.released == 1
    assert set(lock.acquired[0]) == {"task_id", "profile_id", "model_id", "pid", "started_at"}
    for key in lock.acquired[0]:
        assert not any(bad in key.lower() for bad in ("key", "token", "secret", "password"))


def test_live_local_uses_managed_mlx_server(tmp_path):
    registry = _make_registry()
    profile = _make_profile(manage_server=True)
    events: list[str] = []
    server = _FakeServer(events)
    client = _FakeClient(events=events)

    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=lambda: server,
        client_factory=lambda base_url, timeout_s: client,
    )
    spec = RunSpec(
        task_id="analysis.source_reading",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task("analysis.source_reading", spec, {"prompt": "x"}, mode="live")

    assert receipt.selection_decision == "passed"
    assert server.started
    assert server.started[0][0] == profile
    assert server.started[0][1] == spec.model_id
    assert server.started[0][2] is not None
    assert server.started[0][3] == "analysis.source_reading"
    assert server.stopped == 1
    assert events.index("server.start") < events.index("client.complete")


def test_live_local_tts_audio_runtime_still_fails_closed(tmp_path):
    """v3.0 Phase 03-01: tts and audio task families remain blocked.
    Transcription is lifted (tested separately). We verify the guard still fires
    for 'tts' tasks by constructing a registry with a tts task."""
    import yaml as _yaml
    registry_yaml = """\
schema_version: 1
tasks:
  podcast.tts:
    task_family: tts
    stage: podcast
    model_ref: gpt4o_mini_tts
    current_research:
      required: true
      status: pending
    benchmark:
      required: true
      receipt_required: true
      candidates: [gpt-4o-mini-tts]
    quality_threshold: any
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_receipts
    selection_status: planning_default
    allowed_selections:
      - {provider: openai, model: gpt-4o-mini-tts, role: default}
    local_candidates:
      - model_id: "kokoro-local-tts"
        checked_date: "2026-06-25"
        runtime_support: mlx_community_tested
        status: benchmark_pending
"""
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(registry_yaml)
        reg_path = pathlib.Path(f.name)
    from conference_pipeline.model_governance.registry import load_registry as _lr
    registry = _lr(reg_path)
    profile = _make_profile()
    client = _FakeClient()
    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=_FakeServer,
        client_factory=lambda base_url, timeout_s: client,
    )
    spec = RunSpec(
        task_id="podcast.tts",
        provider="local",
        model_id="kokoro-local-tts",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task("podcast.tts", spec, {"text": "fixture"}, mode="live")

    assert receipt.selection_decision == "failed"
    assert "unsupported_local_runtime_shape" in (receipt.failure_reason or "")
    assert client.complete_calls == []


def test_live_local_vision_runtime_no_longer_blocked(tmp_path):
    """v1.61 Phase 67: vision.figure_caption with Qwen3-VL must NO LONGER
    fail closed at the runtime-shape gate. With the guard lifted, the
    fake server/client are exercised end-to-end and the receipt records
    the passing decision (Phase 69 separately gates promotion-to-approved
    on actual benchmark thresholds)."""
    registry = _make_registry()
    profile = _make_profile()
    events: list[str] = []
    server = _FakeServer(events)
    client = _FakeClient(
        output=(
            "salient claim with citation source and evidence from the paper "
            "and openreview url; figure caption preserved"
        ),
        events=events,
    )
    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=lambda: server,
        client_factory=lambda base_url, timeout_s: client,
    )
    spec = RunSpec(
        task_id="vision.figure_caption",
        provider="local",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task(
        "vision.figure_caption", spec, {"image": "fixture"}, mode="live"
    )

    # Guard was lifted: harness must call the (fake) server/client.
    assert "unsupported_local_runtime_shape" not in (receipt.failure_reason or "")
    assert receipt.provider == "local"
    assert receipt.model_id == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert client.complete_calls, "harness must reach the local client now"
    assert events.index("server.start") < events.index("client.complete")


def test_validate_live_local_pass_requires_real_output_artifact():
    receipt = BenchmarkReceipt(
        workspace="test-workspace",
        task_id="analysis.source_reading",
        task_family="analysis",
        stage="analysis",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-04-27T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        hardware_profile="mock",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://analysis",
        output_ref=None,
        raw_output_snippet="",
        rubric_scores={"quality": 0.0},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=None,
        latency_ms=0.0,
        cost_usd=0.0,
        budget_violations=[],
        selection_decision="passed",
        failure_reason=None,
    )
    with pytest.raises(ValueError, match="live local passed receipt"):
        validate_receipt(receipt)


def test_live_local_blank_output_cannot_pass(tmp_path):
    registry = _make_registry()
    profile = _make_profile()
    harness = BenchmarkHarness(
        registry,
        tmp_path,
        local_runtime_profile_id="mlx_light",
        local_profiles={"mlx_light": profile},
        preflight_runner=lambda profile, model_id, lock: profile,
        runtime_lock_factory=_FakeLock,
        managed_server_factory=_FakeServer,
        client_factory=lambda base_url, timeout_s: _FakeClient(output="   "),
    )
    spec = RunSpec(
        task_id="analysis.source_reading",
        provider="local",
        model_id="mlx-community/Qwen3-8B-4bit",
        runtime_id="mlx",
        role="local_candidate",
    )

    receipt = harness.run_task("analysis.source_reading", spec, {"prompt": "x"}, mode="live")

    assert receipt.selection_decision == "failed"
    assert "missing_local_output" in (receipt.failure_reason or "")
    assert receipt.output_ref is None
    persisted = json.loads(next((tmp_path / "analysis.source_reading").glob("*.json")).read_text())
    assert persisted["selection_decision"] == "failed"
    assert "missing_local_output" in persisted["failure_reason"]


def test_run_benchmarks_script_dry_run(tmp_path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_benchmarks.py"
    result = subprocess.run(
        [sys.executable, str(script), "--mode", "dry-run", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"script failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "receipts_written" in result.stdout


# ---------------------------------------------------------------------------
# Task 1 (03-01): Transcription benchmark scoring tests
# ---------------------------------------------------------------------------


def test_transcription_candidate_is_benchmarkable():
    """transcription.session_speech must appear in the local run matrix (D-02).

    With 'transcription' lifted from _UNSUPPORTED_LOCAL_TASK_FAMILIES, the harness
    must include a local_candidate RunSpec for this task.
    """
    registry = _make_registry()
    harness = BenchmarkHarness(registry, Path("/tmp/unused"))

    specs = harness.build_run_matrix("transcription.session_speech")
    local_specs = [s for s in specs if s.role == "local_candidate"]
    assert local_specs, (
        "transcription.session_speech must have at least one local_candidate RunSpec; "
        "lift 'transcription' from _UNSUPPORTED_LOCAL_TASK_FAMILIES"
    )
    model_ids = {s.model_id for s in local_specs}
    assert any("whisper" in mid.lower() for mid in model_ids), (
        f"expected a whisper local candidate in transcription.session_speech matrix, got: {model_ids}"
    )


def test_transcription_scoring_receipt_has_wer_cer(tmp_path):
    """A dry-run fixture benchmark for transcription.session_speech must write a
    BenchmarkReceipt whose rubric_scores contain 'wer', 'cer', and 'quality'.
    """
    from conference_pipeline.benchmarks.harness import RunSpec

    registry = _make_registry()
    harness = BenchmarkHarness(registry, tmp_path)

    spec = RunSpec(
        task_id="transcription.session_speech",
        provider="local",
        model_id="mlx-community/whisper-large-v3-mlx",
        runtime_id="mlx",
        role="local_candidate",
    )
    input_sample = {
        "audio_url": "fixture://mock-session-audio.wav",
        "reference_text": "hello world this is a test",
        "language": "en",
    }

    receipt = harness.run_task("transcription.session_speech", spec, input_sample, mode="dry-run")

    rubric = receipt.rubric_scores
    assert "wer" in rubric, f"rubric_scores missing 'wer': {rubric}"
    assert "cer" in rubric, f"rubric_scores missing 'cer': {rubric}"
    assert "quality" in rubric, f"rubric_scores missing 'quality': {rubric}"
    # Verify the receipt JSON file was written
    receipt_files = list(tmp_path.rglob("*.json"))
    assert receipt_files, "no receipt JSON file was written to output_dir"
    persisted = json.loads(receipt_files[0].read_text())
    assert "wer" in persisted.get("rubric_scores", {}), "wer not in persisted receipt"
    assert "cer" in persisted.get("rubric_scores", {}), "cer not in persisted receipt"


def test_run_benchmarks_live_local_failure_exits_nonzero(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """schema_version: 1
tasks:
  test.task:
    task_family: test
    stage: test-stage
    model_ref: gpt54_mini_legacy
    current_research:
      required: true
      status: pending
    benchmark:
      required: true
      receipt_required: true
      candidates: [gpt-5.4-mini]
    quality_threshold: any
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_receipts
    selection_status: planning_default
    allowed_selections:
      - {provider: openai, model: gpt-5.4-mini, role: default}
    local_candidates:
      - model_id: mlx-community/TestModel-4bit
        checked_date: "2026-04-27"
        runtime_support: mlx_community_tested
        status: benchmark_pending
""",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_benchmarks.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "live",
            "--output-dir",
            str(tmp_path / "receipts"),
            "--registry-path",
            str(registry_path),
            "--task-ids",
            "test.task",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert {"receipts_written", "mode", "live_local_total", "live_local_failed", "live_local_failures"} <= set(payload)
    assert payload["mode"] == "live"
    assert payload["live_local_failed"] >= 1
    failures = payload["live_local_failures"]
    assert failures
    for entry in failures:
        assert set(entry) == {"task_id", "model_id", "selection_decision", "failure_reason"}
        assert not any(
            bad in json.dumps(entry).lower()
            for bad in ("token", "secret", "password", "api_key")
        )
    receipt_files = list((tmp_path / "receipts").rglob("*.json"))
    assert receipt_files
    decisions = [json.loads(path.read_text())["selection_decision"] for path in receipt_files]
    assert "failed" in decisions
