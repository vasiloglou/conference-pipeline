"""Focused tests for MLX client surface and import boundary (D-16)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conference_pipeline.model_governance.local_preflight import (
    check_optional_deps,
    check_overlapping_workload,
    check_platform,
    clamp_concurrency,
)
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile, load_profiles
import conference_pipeline.model_governance.mlx_client as mlx_client_module
from conference_pipeline.model_governance.mlx_client import ManagedMlxServer, MlxServerClient
from conference_pipeline.model_governance.registry import ModelGovernanceError
from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock


# ---------------------------------------------------------------------------
# Import boundary tests — prove that config and registry do NOT pull in mlx
# ---------------------------------------------------------------------------

def _run_import(module: str) -> subprocess.CompletedProcess:
    """Run a bare import in a subprocess with a fake mlx sentinel that fails."""
    env_script = (
        "import sys; "
        "import types; "
        # Block mlx and mlx_lm at the sys.modules level
        "sys.modules['mlx'] = None; "
        "sys.modules['mlx_lm'] = None; "
        f"import {module}; "
        "print('imported')"
    )
    return subprocess.run(
        [sys.executable, "-c", env_script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ":".join([
                str(Path(__file__).parent.parent / "src"),
                *sys.path,
            ])
        },
    )


def test_mlx_import_boundary_config():
    """conference_pipeline.config must import cleanly without mlx."""
    result = _run_import("conference_pipeline.config")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
    assert "mlx" not in result.stderr.lower()


def test_mlx_import_boundary_registry():
    """conference_pipeline.model_governance.registry must import cleanly without mlx."""
    result = _run_import("conference_pipeline.model_governance.registry")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_mlx_import_boundary_init():
    """conference_pipeline.model_governance (package init) must import cleanly without mlx."""
    result = _run_import("conference_pipeline.model_governance")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_mlx_import_boundary_benchmark_harness():
    """conference_pipeline.benchmarks.harness must import cleanly without mlx."""
    result = _run_import("conference_pipeline.benchmarks.harness")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


# ---------------------------------------------------------------------------
# Unit tests — no real MLX server required
# ---------------------------------------------------------------------------

def _make_profile(**overrides) -> LocalRuntimeProfile:
    defaults = dict(
        profile_id="test_profile",
        runtime_kind="mlx",
        base_url="http://127.0.0.1:8080/v1",
        manage_server=True,
        optional_deps=["mlx", "mlx_lm"],
        startup_timeout_s=5.0,
        request_timeout_s=30.0,
        max_concurrency=1,
        max_batch_size=1,
        lock_scope="heavy",
        hardware_fit_status="unknown",
        approval_status="deferred",
    )
    defaults.update(overrides)
    return LocalRuntimeProfile(**defaults)


def test_managed_server_rejects_non_loopback():
    """ManagedMlxServer.start() must fail with ModelGovernanceError for non-loopback URLs."""
    profile = _make_profile(base_url="http://192.168.1.1:8080/v1", manage_server=True)
    server = ManagedMlxServer()
    with pytest.raises(ModelGovernanceError, match="loopback"):
        server.start(profile, "any-model")


def test_managed_server_rejects_manage_false():
    """ManagedMlxServer.start() must fail when manage_server=False."""
    profile = _make_profile(base_url="http://127.0.0.1:8080/v1", manage_server=False)
    server = ManagedMlxServer()
    with pytest.raises(ModelGovernanceError):
        server.start(profile, "any-model")


def test_managed_server_status_absent_state(tmp_path):
    """ManagedMlxServer.status() returns all-None/False when state file is absent."""
    server = ManagedMlxServer(state_path=tmp_path / "absent.json")
    result = server.status()
    assert result["pid"] is None
    assert result["model_id"] is None
    assert result["base_url"] is None
    assert result["health"] is False
    assert result["started_at"] is None
    assert result["lock_owner"] is None


def test_mlx_client_health_check_returns_false_on_refused():
    """MlxServerClient.health_check() returns False when port is closed."""
    client = MlxServerClient("http://127.0.0.1:19999/v1")
    assert client.health_check() is False


def test_mlx_client_complete_raises_on_refused():
    """MlxServerClient.complete() raises ModelGovernanceError when server is unreachable."""
    client = MlxServerClient("http://127.0.0.1:19999/v1", timeout_s=2.0)
    with pytest.raises(ModelGovernanceError):
        client.complete("hello", model="test-model")


def test_managed_server_stop_noop_when_no_state(tmp_path):
    """ManagedMlxServer.stop() does nothing when state file is absent."""
    server = ManagedMlxServer(state_path=tmp_path / "absent.json")
    server.stop()  # must not raise


def test_managed_server_start_acquires_and_stop_releases_runtime_lock(tmp_path, monkeypatch):
    """Managed server lifecycle owns the heavy lock when a lock is provided."""

    class FakeProcess:
        pid = 999999

        def terminate(self):
            pass

    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setattr(mlx_client_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(mlx_client_module.MlxServerClient, "health_check", lambda self: True)

    state_path = tmp_path / "state.json"
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    profile = _make_profile()
    server = ManagedMlxServer(state_path=state_path)

    server.start(profile, "qwen-7b-instruct", lock=lock, task_id="analysis.source_reading")

    owner = lock.owner_metadata()
    assert owner is not None
    assert owner["task_id"] == "analysis.source_reading"
    assert owner["profile_id"] == profile.profile_id
    assert owner["model_id"] == "qwen-7b-instruct"
    assert server.status()["lock_owner"] == owner

    server.stop()
    assert lock.owner_metadata() is None
    assert not state_path.exists()


def test_mlx_client_init_stores_base_url():
    """MlxServerClient stores base_url and timeout_s on construction."""
    client = MlxServerClient("http://127.0.0.1:8080/v1", timeout_s=15.0)
    assert client.base_url == "http://127.0.0.1:8080/v1"
    assert client.timeout_s == 15.0


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------

def test_preflight_platform_check_fails_on_non_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ModelGovernanceError, match="macOS|darwin"):
        check_platform()


def test_preflight_optional_dep_check_fails_missing(monkeypatch):
    profile = _make_profile(optional_deps=["mlx"])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModelGovernanceError, match="mlx"):
        check_optional_deps(profile)


def test_preflight_clamp_concurrency_heavy_model():
    profile = _make_profile(max_concurrency=4, max_batch_size=4)
    result = clamp_concurrency(profile, "qwen-72b-instruct")
    assert result.max_concurrency == 1
    assert result.max_batch_size == 1


def test_preflight_clamp_concurrency_light_model_unchanged():
    profile = _make_profile(max_concurrency=4, max_batch_size=4)
    result = clamp_concurrency(profile, "qwen-7b-instruct")
    assert result.max_concurrency == 4


def test_preflight_overlapping_workload_raises_when_held(tmp_path):
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    lock.acquire(
        {
            "task_id": "test-task",
            "profile_id": "test_profile",
            "model_id": "qwen-72b-instruct",
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    profile = _make_profile(lock_scope="heavy")
    with pytest.raises(ModelGovernanceError, match="heavy"):
        check_overlapping_workload(profile, lock)
    lock.release()


# ---------------------------------------------------------------------------
# Lock tests
# ---------------------------------------------------------------------------

def _make_lock_metadata(**overrides) -> dict:
    base = {
        "task_id": "test-task",
        "profile_id": "test_profile",
        "model_id": "qwen-7b-instruct",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


def test_heavy_lock_acquire_release(tmp_path):
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    lock.acquire(_make_lock_metadata())
    assert (tmp_path / "lock.json").exists()
    assert lock.is_held() is True
    lock.release()
    assert lock.is_held() is False


def test_heavy_lock_conflict_raises(tmp_path):
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    lock.acquire(_make_lock_metadata())
    with pytest.raises(ModelGovernanceError, match="pid"):
        lock.acquire(_make_lock_metadata())
    lock.release()


def test_heavy_lock_metadata_contains_no_secrets(tmp_path):
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    lock.acquire(_make_lock_metadata())
    data = (tmp_path / "lock.json").read_text(encoding="utf-8")
    for bad in ("key", "token", "secret", "password"):
        assert bad not in data.lower() or f'"{bad}"' not in data.lower()
    import json as _json
    parsed = _json.loads(data)
    for k in parsed:
        assert not any(pat in k.lower() for pat in ("key", "token", "secret", "password"))
    lock.release()


def test_heavy_lock_release_idempotent(tmp_path):
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    lock.release()
    lock.release()
