"""Local-only regression coverage for the CVPR backup preflight observation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline import backup_preflight_observation as observation


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "run_cvpr2026_private_backup_observed.sh"


def _fail_if_child_dispatch(*_: object, **__: object) -> None:
    raise AssertionError("the observation boundary must not construct a child")


@pytest.fixture
def no_child_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail locally if the pure helper ever reaches an OS child-dispatch seam."""

    monkeypatch.setattr(observation.os, "system", _fail_if_child_dispatch, raising=False)
    monkeypatch.setattr(observation.os, "popen", _fail_if_child_dispatch, raising=False)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _simulate_sourced_root_presence(
    *, inherited: dict[str, str], sourced: dict[str, str]
) -> bool:
    """Model the wrapper's required unset-before-source boundary without a shell."""

    effective = dict(inherited)
    effective.pop(observation.REQUIRED_VARIABLE, None)
    effective.update(sourced)
    return bool(effective.get(observation.REQUIRED_VARIABLE, ""))


def test_wrapper_contract_clears_inherited_root_before_only_root_env_source() -> None:
    """The production shell has no alternate environment or backup dispatch surface."""

    source = WRAPPER.read_text(encoding="utf-8")

    unset = "unset GDRIVE_BACKUP_ROOT"
    root_source = '. "$REPO_ROOT/.env"'
    assert source.index(unset) < source.index(root_source)
    assert source.count(root_source) == 1
    assert "--env-file" not in source
    assert "sync_to_gdrive" not in source
    assert "remote-path" not in source
    assert "receipt-path" not in source
    assert "--delete" not in source
    assert "--write" not in source
    assert "--dry-run" not in source
    assert "subprocess" not in source
    assert "exec \"$REPO_ROOT/.venv/bin/python\"" in source

    inherited = {observation.REQUIRED_VARIABLE: "/inherited/must-not-survive"}
    assert not _simulate_sourced_root_presence(inherited=inherited, sourced={})
    assert not _simulate_sourced_root_presence(
        inherited=inherited,
        sourced={observation.REQUIRED_VARIABLE: ""},
    )
    assert _simulate_sourced_root_presence(
        inherited=inherited,
        sourced={observation.REQUIRED_VARIABLE: "configured"},
    )


def test_absent_root_atomically_replaces_observation_without_child(
    tmp_path: Path, no_child_sentinel: None
) -> None:
    """An empty sourced root records an absent-only, childless local observation."""

    output = tmp_path / "05-28-BACKUP-PREFLIGHT-OBSERVATION.json"
    output.write_text('{"stale": true}', encoding="utf-8")

    record = observation.write_observation(
        observation_path=output,
        backup_root_present=False,
        observed_at="2026-07-17T12:00:00Z",
    )

    assert record == _read_json(output)
    assert record["backup_root_present"] is False
    assert record["sync_invoked"] is False
    assert record["sync_invocation_started"] is False
    assert record["child_exit_status"] is None
    assert record["status"] == observation.STATUS_BLOCKED
    assert record["reason"] == observation.ABSENT_ROOT_REASON
    assert record["observation_only"] is True
    assert record["expect_absent"] is True
    assert record["historical_non_invocation_not_proven"] is True
    assert record["future_dry_run_authorized"] is False
    assert not list(tmp_path.glob(".*.tmp"))


def test_positive_root_hard_stops_with_sanitized_boolean_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_child_sentinel: None
) -> None:
    """A present root is an explicit authorization block, never a dispatch decision."""

    output = tmp_path / "05-28-BACKUP-PREFLIGHT-OBSERVATION.json"
    monkeypatch.setattr(observation, "DEFAULT_OBSERVATION_PATH", output)

    exit_code = observation.main(
        ["--observation-only", "--expect-absent", "--backup-root-present", "true"]
    )

    record = _read_json(output)
    assert exit_code == observation.ROOT_PRESENT_EXIT_CODE
    assert record["backup_root_present"] is True
    assert record["sync_invoked"] is False
    assert record["sync_invocation_started"] is False
    assert record["child_exit_status"] is None
    assert record["reason"] == observation.ROOT_PRESENT_REASON
    assert record["requires_human_authorization"] is True
    rendered = json.dumps(record, sort_keys=True)
    assert "/inherited/must-not-survive" not in rendered
    assert "remote_path" not in rendered
    assert "receipt_path" not in rendered
    assert "stdout" not in rendered
    assert "stderr" not in rendered


@pytest.mark.parametrize(
    "forbidden",
    [
        "--delete",
        "--write",
        "--dry-run",
        "--sync-mode",
        "--remote-path",
        "--receipt-path",
        "--child-command",
        "--retry",
        "--finalizer",
        "--stage-5",
    ],
)
def test_observation_cli_rejects_backup_capability_arguments(
    forbidden: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper exposes only the fixed observation operation and a boolean."""

    monkeypatch.setattr(
        observation,
        "DEFAULT_OBSERVATION_PATH",
        tmp_path / "05-28-BACKUP-PREFLIGHT-OBSERVATION.json",
    )

    with pytest.raises(SystemExit):
        observation.main(
            [
                "--observation-only",
                "--expect-absent",
                "--backup-root-present",
                "false",
                forbidden,
            ]
        )


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "backup_root_value",
        "remote_path",
        "receipt_path",
        "stdout",
        "stderr",
        "api_key",
        "secret_token",
    ],
)
def test_observation_schema_rejects_values_paths_and_secret_like_keys(
    tmp_path: Path, forbidden_key: str
) -> None:
    """Only the approved boolean/lifecycle schema may be persisted."""

    record = observation.build_observation(
        backup_root_present=False,
        observed_at="2026-07-17T12:00:00Z",
    )
    record[forbidden_key] = "must-not-persist"

    with pytest.raises(observation.ObservationError):
        observation.validate_observation(record)

    assert not (tmp_path / "05-28-BACKUP-PREFLIGHT-OBSERVATION.json").exists()


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_plan_observation_is_not_manifest_owned_backup_selection() -> None:
    """The plan-local observation cannot silently widen Drive/manifest scope."""

    manifest = json.loads(
        (
            REPO_ROOT
            / "conferences"
            / "cvpr2026"
            / "storage"
            / "storage_manifest.json"
        ).read_text(encoding="utf-8")
    )
    selected_paths = {
        str(family.get("path", ""))
        for family in manifest["artifact_families"]
    }
    assert observation.OBSERVATION_RELATIVE_PATH not in selected_paths
    assert all("BACKUP-PREFLIGHT-OBSERVATION" not in path for path in selected_paths)
