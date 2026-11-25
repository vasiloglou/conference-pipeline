from __future__ import annotations

import importlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).parent.parent

PUBLIC_RELEASE_FLAGS = {
    "--conference-dir",
    "--conference-slug",
    "--bundle-id",
    "--receipt-path",
    "--env-file",
    "--validate-manifest",
    "--dry-run",
    "--write-bundle",
    "--export-write-receipt-path",
    "--build-index",
    "--generate-site",
    "--stage-repo",
    "--package-dataset",
    "--preflight-live",
    "--check-liveness",
    "--check-metadata",
    "--live-github",
    "--live-r2",
    "--live-pages",
    "--live-hf",
    "--live-zenodo",
    "--reviewed-dry-run-receipt",
    "--public-repo-dry-run-receipt",
    "--ack-caution-family",
    "--ack-live-github",
    "--ack-live-r2",
    "--ack-live-pages",
    "--ack-live-hf",
    "--ack-live-zenodo",
    "--github-repo",
    "--r2-base-url",
    "--r2-bucket",
    "--r2-prefix",
    "--pages-project",
    "--hf-repo-id",
    "--zenodo-sandbox",
}

STAGE_SCRIPT_NAMES = {
    "scripts/export_for_public.py",
    "scripts/build_public_index.py",
    "scripts/generate_public_site.py",
    "scripts/push_public_repo.py",
    "scripts/upload_public_media.py",
    "scripts/deploy_public_pages.py",
    "scripts/package_public_dataset.py",
    "scripts/deposit_zenodo.py",
    "scripts/push_hf_dataset.py",
    "scripts/preflight_public_deployment.py",
    "scripts/check_public_urls.py",
    "scripts/check_public_release_metadata.py",
}


def load_module():
    return importlib.import_module("conference_pipeline.publishing.public_release")


def stage_id(command: Any) -> str:
    value = getattr(command.stage, "value", command.stage)
    return str(value)


def _clean_stub(paths: list[Path]) -> list[dict]:
    return []


def argv_text(command: Any) -> str:
    return " ".join(str(part) for part in command.argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_project(tmp_path: Path, *, slug: str = "sampleconf2027") -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    conference_dir = project_root / "conferences" / slug
    (conference_dir / "storage" / "workflow-runs").mkdir(parents=True)
    (conference_dir / "decks").mkdir(parents=True)
    (conference_dir / "decks" / "slides.md").write_text("public slides\n", encoding="utf-8")
    write_json(
        conference_dir / "storage" / "public_export_manifest.json",
        {
            "schema_version": "1.0.0",
            "conference": slug,
            "release": {"source_sha": ""},
            "families": [
                {
                    "id": "presentation_markdown",
                    "private_path": "decks",
                    "public_path": "exports/public-release/{bundle}/decks/",
                    "storage_manifest_id": "presentation_markdown",
                    "redistribution_status": "allowed",
                    "license": "CC-BY-SA-4.0",
                    "provenance": "pipeline-generated",
                }
            ],
        },
    )
    return project_root, conference_dir


def write_reviewed_export_receipt(path: Path, conference_dir: Path, *, bundle_id: str | None = None) -> None:
    export_module = load_export_module()
    bundle = bundle_id or f"{conference_dir.name}-v1.0.0"
    plan = export_module.build_export_plan(
        export_module.ExportConfig(
            conference_dir=conference_dir,
            export_manifest=conference_dir / "storage" / "public_export_manifest.json",
            output_dir=conference_dir / "exports" / "public-release",
            bundle_id=bundle,
            dry_run=True,
            receipt_path=path,
            reviewed_dry_run_receipt=None,
            ack_caution_families=(),
            source_sha=None,
        )
    )
    write_json(
        path,
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.public_export",
            "status": "succeeded",
            "exit_code": 0,
            "dry_run": True,
            "conference": plan.conference,
            "bundle_id": plan.bundle_id,
            "source_sha": plan.source_sha,
            "export_manifest_sha256": plan.manifest_sha256,
            "output_dir": str(plan.config.output_dir),
            "planned_artifacts": export_module.planned_artifacts(plan),
            "cdn_stubs": export_module.cdn_stub_artifacts(plan),
        },
    )


def write_runner_child_receipt(argv: list[str], conference_dir: Path) -> None:
    """Materialize the receipt a successful export subprocess would write."""
    if not any(str(part).endswith("scripts/export_for_public.py") for part in argv):
        return
    export_module = load_export_module()
    receipt_path = Path(argv[argv.index("--receipt-path") + 1])
    stage = argv[argv.index("--orchestrator-stage") + 1]
    dry_run = "--dry-run" in argv
    plan = export_module.build_export_plan(
        export_module.ExportConfig(
            conference_dir=conference_dir,
            export_manifest=conference_dir / "storage" / "public_export_manifest.json",
            output_dir=conference_dir / "exports" / "public-release",
            bundle_id=f"{conference_dir.name}-v1.0.0",
            dry_run=dry_run,
            receipt_path=receipt_path,
            reviewed_dry_run_receipt=None,
            ack_caution_families=(),
            source_sha=None,
            orchestrator_stage=stage,
        )
    )
    write_json(
        receipt_path,
        export_module.build_receipt_payload(
            plan,
            status="succeeded",
            exit_code=0,
        ),
    )


def ensure_public_bundle(conference_dir: Path) -> Path:
    bundle_id = f"{conference_dir.name}-v1.0.0"
    bundle_root = conference_dir / "exports" / "public-release" / bundle_id
    bundle_root.mkdir(parents=True, exist_ok=True)
    write_json(
        bundle_root / "RELEASE.json",
        {
            "bundle_id": bundle_id,
            "conference": conference_dir.name,
            "source_sha": "reviewed-source",
            "export_manifest_sha256": sha256_file(conference_dir / "storage" / "public_export_manifest.json"),
            "family_licenses": {"presentation_markdown": "MIT"},
        },
    )
    write_json(bundle_root / "CHECKSUMS.json", {"decks/slides.md": {"sha256": "abc123"}})
    (bundle_root / "decks").mkdir(exist_ok=True)
    (bundle_root / "decks" / "slides.md").write_text("public slides\n", encoding="utf-8")
    return bundle_root


def write_public_repo_receipt(path: Path, conference_dir: Path, *, repo_name: str | None = None) -> None:
    from conference_pipeline.publishing.public_repo import PublicRepoConfig, stage_public_repo

    bundle_root = ensure_public_bundle(conference_dir)
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle_root,
            target_dir=path.parent / "public-repo-dry-run",
            repo_name=repo_name or f"{conference_dir.name}-data",
            dry_run=True,
            receipt_path=path,
            secret_scanner=_clean_stub,
        )
    )


def load_export_module():
    path = REPO_ROOT / "scripts" / "export_for_public.py"
    spec = importlib.util.spec_from_file_location("export_for_public_for_public_release_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_help_exposes_public_release_flags():
    result = subprocess.run(
        [sys.executable, "scripts/publish_conference_public_release.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    missing = sorted(flag for flag in PUBLIC_RELEASE_FLAGS if flag not in result.stdout)
    assert not missing, f"missing CLI flags: {missing}"


def test_d01_d02_default_plan_validates_manifest_and_dry_runs_without_shell(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)

    commands = module.build_public_release_plan(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            dry_run=True,
        )
    )

    assert [stage_id(command) for command in commands] == [
        "validate_manifest",
        "export_dry_run",
    ]
    assert all(isinstance(command.argv, list) for command in commands)
    assert all(getattr(command, "shell", False) is False for command in commands)
    assert all(command.receipt_path.parent == conference_dir / "storage" / "workflow-runs" for command in commands)
    assert all(command.cwd == project_root for command in commands)
    assert any("scripts/export_for_public.py" in argv_text(command) for command in commands)


def test_d01_d02_slug_and_directory_inputs_derive_generic_bundle_ids(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path, slug="visionconf2027")

    from_slug = module.build_public_release_plan(
        module.PublicReleaseConfig(project_root=project_root, conference_slug="visionconf2027")
    )
    from_dir = module.build_public_release_plan(
        module.PublicReleaseConfig(project_root=project_root, conference_dir=conference_dir)
    )

    assert module.default_bundle_id(conference_dir) == "visionconf2027-v1.0.0"
    assert all("visionconf2027-v1.0.0" in argv_text(command) for command in from_slug)
    assert all("visionconf2027-v1.0.0" in argv_text(command) for command in from_dir)
    assert all("iclr2026-v1.0.0" not in argv_text(command) for command in [*from_slug, *from_dir])


def test_d03_selected_stages_delegate_to_existing_public_release_scripts(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_root = conference_dir / "storage" / "workflow-runs"
    reviewed = receipt_root / "dry-run.json"
    public_repo_reviewed = receipt_root / "public-repo-dry-run.json"
    write_json(reviewed, {})
    write_json(public_repo_reviewed, {})

    commands = module.build_public_release_plan(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            write_bundle=True,
            build_index=True,
            generate_site=True,
            stage_repo=True,
            package_dataset=True,
            preflight_live=True,
            check_liveness=True,
            check_metadata=True,
            live_github=True,
            live_r2=True,
            live_pages=True,
            live_hf=True,
            live_zenodo=True,
            reviewed_dry_run_receipt=reviewed,
            public_repo_dry_run_receipt=public_repo_reviewed,
            ack_live_github=True,
            ack_live_r2=True,
            ack_live_pages=True,
            ack_live_hf=True,
            ack_live_zenodo=True,
            github_repo="example/sampleconf2027-data",
            r2_base_url="https://cdn.example.test/conferences/sampleconf2027",
            r2_bucket="sampleconf-public",
            pages_project="sampleconf2027-data",
            hf_repo_id="example/sampleconf2027-data",
            zenodo_sandbox=True,
        )
    )

    planned_scripts = {
        Path(part).relative_to(REPO_ROOT.resolve()).as_posix()
        for command in commands
        for part in command.argv
        if str(part).endswith(".py") and "/scripts/" in str(part)
    }
    assert STAGE_SCRIPT_NAMES <= {str(path) for path in planned_scripts}


def test_6b_scripts_come_from_the_code_checkout_not_the_data_checkout(tmp_path: Path):
    """A data checkout on an older branch must not run its own scripts (CVPR 2026, 2026-11-19)."""
    module = load_module()
    project_root, _conference_dir = make_project(tmp_path)
    decoy = project_root / "scripts" / "export_for_public.py"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text("raise SystemExit('the data checkout ran its own script')\n", encoding="utf-8")

    commands = module.build_public_release_plan(
        module.PublicReleaseConfig(project_root=project_root, conference_slug="sampleconf2027", dry_run=True)
    )

    scripts = [Path(command.argv[1]) for command in commands if str(command.argv[1]).endswith(".py")]
    assert scripts
    for script in scripts:
        assert script.is_absolute() and script.is_file()
        assert script.resolve().is_relative_to(REPO_ROOT.resolve())
    export = module._load_export_script(project_root)
    assert Path(export.__file__).resolve() == (REPO_ROOT / "scripts" / "export_for_public.py").resolve()


def test_orchestrator_rejects_combined_dry_run_and_write_before_runner(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "combined.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        raise AssertionError("combined export modes must not run child commands")

    config = module.PublicReleaseConfig(
        project_root=project_root,
        conference_slug="sampleconf2027",
        validate_manifest=True,
        dry_run=True,
        write_bundle=True,
        receipt_path=receipt_path,
    )
    blocker = module.export_review_boundary_blocker(config)
    result = module.run_public_release(config, runner=runner)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert blocker == {
        "stage": "export_write",
        "reason": "export_review_boundary_required",
        "missing_approvals": [],
        "missing_receipts": [],
        "missing_targets": [],
    }
    assert runner_calls == []
    assert result.status == "operator_blocked"
    assert result.commands == ()
    assert receipt["blocked_stages"] == [blocker]
    assert receipt["secret_values_recorded"] is False


def test_orchestrator_write_receipt_override_retains_child_layers(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "reviewed-dry-run.json"
    write_reviewed_export_receipt(reviewed, conference_dir)
    child_write = conference_dir / "storage" / "workflow-runs" / "first-write-child.json"
    top_level = conference_dir / "storage" / "workflow-runs" / "first-write-orchestrator.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        write_runner_child_receipt(argv, conference_dir)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            write_bundle=True,
            reviewed_dry_run_receipt=reviewed,
            export_write_receipt_path=child_write,
            receipt_path=top_level,
        ),
        runner=runner,
    )
    receipt = json.loads(top_level.read_text(encoding="utf-8"))

    assert result.status == "succeeded"
    assert len(runner_calls) == 1
    assert runner_calls[0][runner_calls[0].index("--reviewed-dry-run-receipt") + 1] == str(reviewed)
    assert runner_calls[0][runner_calls[0].index("--receipt-path") + 1] == str(child_write)
    assert receipt["child_receipt_paths"] == [child_write.as_posix()]
    assert receipt["commands"][0]["shell"] is False


def test_public_release_journal_is_updated_after_each_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    expected_journal = receipt_path.with_name("orchestrator.journal.json")
    secure_writes: list[tuple[Path, dict[str, Any]]] = []
    original_secure_write = module.write_json_atomic_secure

    def secure_write_spy(path: Path, payload: dict[str, Any]) -> None:
        secure_writes.append((path, json.loads(json.dumps(payload))))
        original_secure_write(path, payload)

    observed_before_children: list[dict[str, Any]] = []

    def runner(argv: list[str], **kwargs: Any):
        observed_before_children.append(
            json.loads(expected_journal.read_text(encoding="utf-8"))
        )
        write_runner_child_receipt(argv, conference_dir)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "write_json_atomic_secure", secure_write_spy)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            dry_run=True,
            receipt_path=receipt_path,
        ),
        runner=runner,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    journal = json.loads(expected_journal.read_text(encoding="utf-8"))
    journal_transitions = [
        payload for path, payload in secure_writes if path == expected_journal
    ]
    assert result.status == "succeeded"
    assert result.journal_path == expected_journal
    assert module.public_release_journal_path(receipt_path) == expected_journal
    assert observed_before_children[0]["status"] == "running"
    assert observed_before_children[0]["command_results"] == []
    assert observed_before_children[1]["last_completed_stage"] == "validate_manifest"
    assert [row["stage"] for row in observed_before_children[1]["command_results"]] == [
        "validate_manifest"
    ]
    assert [transition["status"] for transition in journal_transitions] == [
        "running",
        "running",
        "running",
        "finalizing",
        "finalized",
    ]
    assert journal["status"] == "finalized"
    assert journal["last_completed_stage"] == "export_dry_run"
    assert [row["stage"] for row in journal["command_results"]] == [
        "validate_manifest",
        "export_dry_run",
    ]
    assert journal["secret_values_recorded"] is False
    assert receipt["journal_path"] == expected_journal.as_posix()
    assert receipt["child_receipt_paths"] == [
        command.receipt_path.as_posix() for command in result.commands
    ]


def test_top_level_receipt_failure_returns_durable_journal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    journal_path = receipt_path.with_name("orchestrator.journal.json")
    sentinels = (
        "TOKEN_SENTINEL=top-level-token",
        "PASSWORD_SENTINEL=top-level-password",
        "/Volumes/GoogleDrive/private/top-level.json",
        "Google Drive",
    )
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        write_runner_child_receipt(argv, conference_dir)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def fail_top_level_write(_path: Path, _payload: dict[str, Any]) -> Path:
        raise OSError(" | ".join(sentinels))

    monkeypatch.setattr(module, "write_public_release_receipt", fail_top_level_write)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            dry_run=True,
            receipt_path=receipt_path,
        ),
        runner=runner,
    )

    serialized = journal_path.read_text(encoding="utf-8")
    journal = json.loads(serialized)
    assert len(runner_calls) == 2
    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.receipt_path == receipt_path
    assert result.journal_path == journal_path
    assert not receipt_path.exists()
    assert journal["status"] == "failed"
    assert journal["failure_code"] == "top_level_receipt_persistence_failed"
    assert journal["exception_type"] == "OSError"
    assert [row["stage"] for row in journal["command_results"]] == [
        "validate_manifest",
        "export_dry_run",
    ]
    assert journal["last_completed_stage"] == "export_dry_run"
    assert journal["secret_values_recorded"] is False
    assert all(sentinel not in serialized for sentinel in sentinels)


def test_initial_journal_failure_runs_no_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def fail_journal_write(_path: Path, _payload: dict[str, Any]) -> Path:
        raise OSError("TOKEN_SENTINEL /Volumes/GoogleDrive/private/journal.json")

    monkeypatch.setattr(module, "write_public_release_journal", fail_journal_write, raising=False)
    with pytest.raises(
        module.PublicReleaseError,
        match="^public_release_journal_persistence_failed$",
    ) as raised:
        module.run_public_release(
            module.PublicReleaseConfig(
                project_root=project_root,
                conference_slug=conference_dir.name,
                dry_run=True,
                receipt_path=receipt_path,
            ),
            runner=runner,
        )

    assert runner_calls == []
    assert "TOKEN_SENTINEL" not in str(raised.value)
    assert "GoogleDrive" not in str(raised.value)


def test_d04_d05_live_provider_gates_block_without_reviewed_receipts_or_ack(tmp_path: Path):
    module = load_module()
    project_root, _conference_dir = make_project(tmp_path)
    receipt_path = project_root / "conferences" / "sampleconf2027" / "storage" / "workflow-runs" / "orchestrator.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        raise AssertionError("live provider commands must not run when gates are missing")

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=receipt_path,
        ),
        runner=runner,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert runner_calls == []
    assert result.status == "operator_blocked"
    assert result.exit_code == 2
    assert result.secret_values_recorded is False
    assert receipt["status"] == "operator_blocked"
    assert receipt["secret_values_recorded"] is False
    assert receipt["blocked_stages"][0]["stage"] == "live_github"
    assert "public_repo_dry_run_receipt" in receipt["blocked_stages"][0]["missing_receipts"]
    assert "ack_live_github" in receipt["blocked_stages"][0]["missing_approvals"]


def test_d04_d05_live_provider_gates_validate_public_repo_receipt_path(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = project_root / "conferences" / "sampleconf2027" / "storage" / "workflow-runs" / "orchestrator.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        raise AssertionError("live provider commands must not run with an unreadable receipt")

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_github=True,
            public_repo_dry_run_receipt=(
                conference_dir
                / "storage"
                / "workflow-runs"
                / "missing-public-repo-dry-run.json"
            ),
            ack_live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=receipt_path,
        ),
        runner=runner,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert runner_calls == []
    assert result.status == "operator_blocked"
    assert "public_repo_dry_run_receipt:missing_reviewed_receipt" in receipt["blocked_stages"][0]["missing_receipts"]


def test_live_github_blocks_public_repo_receipt_without_secret_gate(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "public-repo-dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_public_repo_receipt(reviewed, conference_dir)
    receipt = json.loads(reviewed.read_text(encoding="utf-8"))
    receipt.pop("secret_gate")
    write_json(reviewed, receipt)

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_github=True,
            public_repo_dry_run_receipt=reviewed,
            ack_live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=receipt_path,
            public_repo_secret_scanner=_clean_stub,
        )
    )

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    assert "public_repo_dry_run_receipt:secret_gate" in data["blocked_stages"][0]["missing_receipts"]


def test_live_github_runs_only_after_valid_reviewed_receipt_and_ack(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "public-repo-dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_public_repo_receipt(reviewed, conference_dir)
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="",
            stderr="live GitHub publication requires provider push proof",
        )

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_github=True,
            public_repo_dry_run_receipt=reviewed,
            ack_live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=receipt_path,
            public_repo_secret_scanner=_clean_stub,
        ),
        runner=runner,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    assert result.exit_code == 2
    assert receipt["status"] == "operator_blocked"
    assert receipt["command_results"][0]["exit_code"] == 2
    assert len(runner_calls) == 1
    assert "--reviewed-dry-run-receipt" in runner_calls[0]


def test_live_pages_blocks_stale_export_receipt(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    stale = conference_dir / "storage" / "workflow-runs" / "stale-export-dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_json(
        stale,
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.public_export",
            "status": "succeeded",
            "exit_code": 0,
            "dry_run": True,
            "conference": conference_dir.name,
            "bundle_id": f"{conference_dir.name}-v1.0.0",
            "source_sha": "stale",
            "export_manifest_sha256": sha256_file(conference_dir / "storage" / "public_export_manifest.json"),
            "output_dir": str(conference_dir / "exports" / "public-release"),
            "planned_artifacts": [],
            "cdn_stubs": [],
        },
    )

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_pages=True,
            reviewed_dry_run_receipt=stale,
            ack_live_pages=True,
            pages_project="sampleconf2027-data",
            receipt_path=receipt_path,
        )
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    missing = receipt["blocked_stages"][0]["missing_receipts"]
    assert "reviewed_dry_run_receipt:source_sha" in missing
    assert "reviewed_dry_run_receipt:planned_artifacts" in missing


def test_reviewed_export_gate_sanitizes_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_reviewed_export_receipt(reviewed, conference_dir)
    sentinels = (
        "TOKEN_SENTINEL=top-secret-token",
        "PASSWORD_SENTINEL=hunter2",
        "/Volumes/GoogleDrive/private/source.json",
        "Google Drive",
    )

    def fail_current_contract(_config: Any) -> dict[str, Any]:
        raise RuntimeError(" | ".join(sentinels))

    monkeypatch.setattr(module, "_current_export_contract", fail_current_contract)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            live_pages=True,
            reviewed_dry_run_receipt=reviewed,
            ack_live_pages=True,
            pages_project="sampleconf2027-data",
            receipt_path=receipt_path,
        )
    )

    serialized = receipt_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert result.status == "operator_blocked"
    assert payload["blocked_stages"][0]["missing_receipts"] == [
        "reviewed_dry_run_receipt:current_plan:exception:RuntimeError"
    ]
    assert payload["secret_values_recorded"] is False
    assert all(sentinel not in serialized for sentinel in sentinels)

    unsafe_type = type("Unsafe-Class/Name", (Exception,), {})
    assert module.stable_receipt_exception_code(
        "reviewed_dry_run_receipt",
        "current_plan",
        unsafe_type("TOKEN_SENTINEL"),
    ) == "reviewed_dry_run_receipt:current_plan:exception:Exception"


def test_public_repo_gate_sanitizes_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "public-repo-dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_public_repo_receipt(reviewed, conference_dir)
    sentinels = (
        "TOKEN_SENTINEL=repo-token",
        "PASSWORD_SENTINEL=repo-password",
        "/Volumes/GoogleDrive/private/repo.json",
        "Google Drive",
    )

    def fail_current_contract(_config: Any) -> dict[str, Any]:
        raise OSError(" | ".join(sentinels))

    monkeypatch.setattr(module, "_current_public_repo_contract", fail_current_contract)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            live_github=True,
            public_repo_dry_run_receipt=reviewed,
            ack_live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=receipt_path,
            public_repo_secret_scanner=_clean_stub,
        )
    )

    serialized = receipt_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert result.status == "operator_blocked"
    assert payload["blocked_stages"][0]["missing_receipts"] == [
        "public_repo_dry_run_receipt:current_stage:exception:OSError"
    ]
    assert payload["secret_values_recorded"] is False
    assert all(sentinel not in serialized for sentinel in sentinels)


def test_live_r2_blocks_without_its_ack_and_names_no_permanent_blocker(tmp_path: Path):
    # v6.0 Phase 7.1 removed the permanent verified_r2_live_upload_support blocker once
    # the upload was tested end to end (test_release_publish_paths.py); the ack and the
    # bucket still gate it.
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_reviewed_export_receipt(reviewed, conference_dir)

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            live_r2=True,
            reviewed_dry_run_receipt=reviewed,
            ack_live_r2=False,
            receipt_path=receipt_path,
        )
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    blocked = receipt["blocked_stages"][0]
    # The owner's named approval is required too (phase-7.1-r1 F3).
    assert blocked["missing_approvals"] == ["ack_live_r2", "owner_approval"]
    assert blocked["missing_targets"] == ["live_r2_target"]
    assert "verified_r2_live_upload_support" not in blocked["missing_receipts"]


def test_preflight_live_missing_inputs_blocks_structurally(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            preflight_live=True,
            receipt_path=receipt_path,
        )
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    assert receipt["blocked_stages"][0]["stage"] == "preflight_live"
    assert "github_repo" in receipt["blocked_stages"][0]["missing_targets"]
    assert "r2_base_url" in receipt["blocked_stages"][0]["missing_targets"]
    assert "public_repo_dry_run_receipt" in receipt["blocked_stages"][0]["missing_receipts"]


@pytest.mark.parametrize("path_owner", ["top_level", "child"])
@pytest.mark.parametrize(
    "target_kind",
    [
        "credentials",
        "manifest",
        "selected_source",
        "release_metadata",
        "checksums",
        "bundle_descendant",
        "unrelated_project_file",
    ],
)
def test_public_release_receipt_paths_are_confined_before_runner(
    tmp_path: Path,
    path_owner: str,
    target_kind: str,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    bundle_id = f"{conference_dir.name}-v1.0.0"
    bundle_root = conference_dir / "exports" / "public-release" / bundle_id
    targets = {
        "credentials": project_root / ".env",
        "manifest": conference_dir / "storage" / "public_export_manifest.json",
        "selected_source": conference_dir / "decks" / "slides.md",
        "release_metadata": bundle_root / "RELEASE.json",
        "checksums": bundle_root / "CHECKSUMS.json",
        "bundle_descendant": bundle_root / "decks" / "slides.md",
        "unrelated_project_file": project_root / "unrelated.json",
    }
    target = targets[target_kind]
    target.parent.mkdir(parents=True, exist_ok=True)
    reviewed = conference_dir / "storage" / "workflow-runs" / "reviewed.json"
    write_reviewed_export_receipt(reviewed, conference_dir, bundle_id=bundle_id)
    sentinel = f"sentinel:{path_owner}:{target_kind}\n".encode()
    target.write_bytes(sentinel)
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    config_kwargs: dict[str, Any] = {
        "project_root": project_root,
        "conference_slug": conference_dir.name,
    }
    if path_owner == "top_level":
        config_kwargs.update(dry_run=True, receipt_path=target)
    else:
        config_kwargs.update(
            write_bundle=True,
            reviewed_dry_run_receipt=reviewed,
            export_write_receipt_path=target,
        )

    result = module.run_public_release(
        module.PublicReleaseConfig(**config_kwargs),
        runner=runner,
    )

    assert result.status == "operator_blocked"
    assert result.exit_code == 2
    assert runner_calls == []
    assert target.read_bytes() == sentinel
    safe_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / f"public-release-{bundle_id}-receipt.json"
    )
    payload = json.loads(safe_receipt.read_text(encoding="utf-8"))
    blocker = payload["blocked_stages"][0]
    assert blocker["reason"] == "unsafe_receipt_path"
    assert blocker["reason_code"] == "outside_workflow_runs"
    assert blocker["label"] in {"receipt_path", "export_write_receipt_path"}
    assert payload["secret_values_recorded"] is False


@pytest.mark.parametrize(
    ("receipt_kind", "path_kind", "expected_code"),
    [
        ("export", "outside", "outside_workflow_runs"),
        ("export", "symlink", "symlinked_receipt_component"),
        ("public_repo", "outside", "outside_workflow_runs"),
        ("public_repo", "symlink", "symlinked_receipt_component"),
    ],
)
def test_public_release_reviewed_receipts_must_be_confined(
    tmp_path: Path,
    receipt_kind: str,
    path_kind: str,
    expected_code: str,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_root = conference_dir / "storage" / "workflow-runs"
    external = project_root / f"{receipt_kind}-external.json"
    external.write_text('{"sentinel": true}\n', encoding="utf-8")
    if path_kind == "outside":
        reviewed = external
    else:
        reviewed = receipt_root / f"{receipt_kind}-linked.json"
        try:
            reviewed.symlink_to(external)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable in this environment: {exc}")
    top_level = receipt_root / f"{receipt_kind}-blocked.json"
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    if receipt_kind == "export":
        config = module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            live_pages=True,
            reviewed_dry_run_receipt=reviewed,
            ack_live_pages=True,
            pages_project="sampleconf2027-data",
            receipt_path=top_level,
        )
        label = "reviewed_dry_run_receipt"
    else:
        config = module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            live_github=True,
            public_repo_dry_run_receipt=reviewed,
            ack_live_github=True,
            github_repo="example/sampleconf2027-data",
            receipt_path=top_level,
        )
        label = "public_repo_dry_run_receipt"

    result = module.run_public_release(config, runner=runner)

    payload = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    assert runner_calls == []
    assert f"{label}:{expected_code}" in payload["blocked_stages"][0]["missing_receipts"]
    assert external.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_public_release_receipt_path_rejects_symlinked_ancestor(tmp_path: Path):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_root = conference_dir / "storage" / "workflow-runs"
    external = project_root / "external-receipts"
    external.mkdir()
    linked = receipt_root / "linked"
    try:
        linked.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug=conference_dir.name,
            dry_run=True,
            receipt_path=linked / "receipt.json",
        ),
        runner=runner,
    )

    assert result.status == "operator_blocked"
    assert runner_calls == []
    assert not (external / "receipt.json").exists()


@pytest.mark.parametrize(
    "alias_kind",
    ["top_level", "reviewed", "journal", "selected_child"],
)
def test_public_release_receipt_paths_reject_aliases_before_runner(
    tmp_path: Path,
    alias_kind: str,
):
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    receipt_root = conference_dir / "storage" / "workflow-runs"
    reviewed = receipt_root / "reviewed.json"
    write_reviewed_export_receipt(reviewed, conference_dir)
    top_level = receipt_root / "top-level.json"
    child = receipt_root / "child.json"
    config_kwargs: dict[str, Any] = {
        "project_root": project_root,
        "conference_slug": conference_dir.name,
        "write_bundle": True,
        "reviewed_dry_run_receipt": reviewed,
        "receipt_path": top_level,
        "export_write_receipt_path": child,
    }
    if alias_kind == "top_level":
        config_kwargs["export_write_receipt_path"] = top_level
    elif alias_kind == "reviewed":
        config_kwargs["export_write_receipt_path"] = reviewed
    elif alias_kind == "journal":
        config_kwargs["export_write_receipt_path"] = top_level.with_name(
            "top-level.journal.json"
        )
    else:
        config_kwargs["build_index"] = True
        config_kwargs["export_write_receipt_path"] = (
            receipt_root
            / f"public-index-{conference_dir.name}-v1.0.0-receipt.json"
        )
    runner_calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any):
        runner_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = module.run_public_release(
        module.PublicReleaseConfig(**config_kwargs),
        runner=runner,
    )

    payload = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert result.status == "operator_blocked"
    assert runner_calls == []
    assert payload["blocked_stages"][0]["reason"] == "receipt_path_alias"


def test_public_release_receipt_temp_symlink_cannot_redirect_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    from conference_pipeline.publishing import receipt_paths

    receipt_dir = tmp_path / "workflow-runs"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "receipt.json"
    sentinel = tmp_path / "credential-sentinel.json"
    sentinel_bytes = b"credential sentinel bytes\n"
    sentinel.write_bytes(sentinel_bytes)
    first_token = str(os.getpid())
    collision = receipt_dir / f".receipt.json.{first_token}.tmp"
    try:
        collision.symlink_to(sentinel)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    tokens = iter([first_token, "fresh"])
    monkeypatch.setattr(receipt_paths.secrets, "token_hex", lambda _size: next(tokens))

    module.write_json(receipt_path, {"status": "operator_blocked"})

    assert sentinel.read_bytes() == sentinel_bytes
    assert collision.is_symlink()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "status": "operator_blocked"
    }
    assert sorted(path.name for path in receipt_dir.iterdir()) == [
        f".receipt.json.{first_token}.tmp",
        "receipt.json",
    ]


def test_package_dataset_runs_before_stage_repo(tmp_path: Path) -> None:
    """v6.0 Phase 6 D15 (Phase 3 open question 2, IF6): the staged repository copies
    metadata/, so the dataset package must exist before it is staged."""
    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "dry-run.json"
    write_json(reviewed, {})
    commands = module.build_public_release_plan(
        module.PublicReleaseConfig(
            project_root=project_root,
            conference_slug="sampleconf2027",
            write_bundle=True,
            stage_repo=True,
            package_dataset=True,
            reviewed_dry_run_receipt=reviewed,
        )
    )
    stages = [command.stage.value for command in commands]
    assert stages.index("package_dataset") < stages.index("stage_repo")
