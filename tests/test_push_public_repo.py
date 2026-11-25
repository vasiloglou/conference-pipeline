from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.publishing import public_repo as public_repo_module
from conference_pipeline.publishing.export_gate import GateReceipt
from conference_pipeline.publishing.public_repo import (
    PublicRepoConfig,
    PublicRepoError,
    stage_public_repo,
)


REPO_ROOT = Path(__file__).parent.parent


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_stub(paths: list[Path]) -> list[dict]:
    return []


def _dirty_stub(paths: list[Path]) -> list[dict]:
    return [{"RuleID": "test-secret", "File": paths[0].as_posix() if paths else "none"}]


def make_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(
        root / "RELEASE.json",
        {
            "bundle_id": "bundle-test",
            "conference": "neurips2025",
            "source_sha": "abcdef1234567890",
            "export_manifest_sha256": "manifestdigest",
            "included_families": ["presentation_markdown", "reader_catalog"],
            "family_licenses": {"presentation_markdown": "MIT", "reader_catalog": "CC-BY-4.0"},
        },
    )
    write_json(root / "CHECKSUMS.json", {"decks/a.md": {"sha256": "x"}})
    (root / "decks").mkdir()
    (root / "decks" / "a.md").write_text("Alpha public text\n", encoding="utf-8")
    return root


def test_stages_public_repo_with_license_notice_and_citation(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["public_repo_url"] = "https://github.com/vasiloglou/neurips2025-data"
    release["cdn_base_url"] = "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/conferences/neurips2025"
    write_json(bundle / "RELEASE.json", release)
    target = tmp_path / "staged"
    result = stage_public_repo(
        PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True, secret_scanner=_clean_stub)
    )

    assert (target / "README.md").exists()
    assert (target / "LICENSE").exists()
    assert (target / "NOTICE.md").exists()
    assert (target / "CITATION.cff").exists()
    assert (target / "public_repo_manifest.json").exists()
    assert release["public_repo_url"] in (target / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((target / "public_repo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_repo_url"] == release["public_repo_url"]
    assert manifest["cdn_base_url"] == release["cdn_base_url"]
    assert result.manifest_path == target / "public_repo_manifest.json"


def test_rejects_private_workspace_root(tmp_path: Path):
    workspace = tmp_path / "conferences" / "neurips2025"
    workspace.mkdir(parents=True)
    with pytest.raises(PublicRepoError):
        stage_public_repo(PublicRepoConfig(bundle_root=workspace, target_dir=tmp_path / "out"))


def test_staged_repo_contains_no_private_path_markers(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    stage_public_repo(
        PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True, secret_scanner=_clean_stub)
    )

    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in target.rglob("*")
        if path.is_file()
    )
    for marker in (".env", "file://", "/Volumes/", "Google Drive", "drive.google.com", "/Users/"):
        assert marker not in combined


def test_excluded_private_marker_paths_are_redacted_in_public_manifest(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    (bundle / "self-host").mkdir()
    (bundle / "self-host" / "gpt-researcher.env.example").write_text(
        "PUBLIC_PLACEHOLDER=1\n", encoding="utf-8"
    )
    target = tmp_path / "staged"

    stage_public_repo(
        PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True, secret_scanner=_clean_stub)
    )

    manifest_text = (target / "public_repo_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert ".env" not in manifest_text
    assert manifest["excluded_files"] == [
        {
            "path": "self-host/gpt-researcher<private-marker>.example",
            "path_redacted": True,
            "reason": "private_marker",
        }
    ]


def test_public_repo_manifest_records_read_only_operator_gate(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    receipt = tmp_path / "receipt.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=target,
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=receipt,
            secret_scanner=_clean_stub,
        )
    )

    manifest = json.loads((target / "public_repo_manifest.json").read_text(encoding="utf-8"))
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert manifest["read_only"] is True
    assert manifest["repo_name"] == "neurips2025-data"
    assert data["operator_live_step_required"] is True
    assert data["dry_run"] is True
    assert data["live"] is False


def test_non_dry_run_records_public_safety_without_scanning_source_history(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    receipt = tmp_path / "receipt.json"

    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=target,
            dry_run=False,
            receipt_path=receipt,
            source_repo_root=REPO_ROOT,
            secret_scanner=_clean_stub,
        )
    )

    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["public_safety"] == {
        "status": "clean",
        "scope": "staged_public_repo_tree",
        "private_marker_count": 0,
        "source_repo_history_published": False,
    }
    assert "gitleaks" not in data


@pytest.mark.parametrize("target_factory", [
    lambda bundle, tmp_path: bundle,
    lambda bundle, tmp_path: bundle.parent,
    lambda bundle, tmp_path: Path.cwd(),
])
def test_refuses_unsafe_public_repo_stage_targets(tmp_path: Path, target_factory):
    bundle = make_bundle(tmp_path)
    target = target_factory(bundle, tmp_path)

    with pytest.raises(PublicRepoError, match="unsafe public repo target"):
        stage_public_repo(PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True))


def test_refuses_to_overwrite_unmanaged_existing_target(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    target.mkdir()
    existing = target / "operator-notes.md"
    existing.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(PublicRepoError, match="unmanaged public repo target"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=target,
                dry_run=True,
                secret_scanner=_clean_stub,
            )
        )

    assert existing.read_text(encoding="utf-8") == "do not delete\n"


def test_staged_repo_secret_gate_blocks_content_findings(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    (bundle / "decks" / "a.md").write_text(
        "OPENAI_API_KEY=sk-test-public-repo-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(PublicRepoError, match="secret scan failed"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=tmp_path / "staged",
                dry_run=True,
                secret_scanner=_dirty_stub,
            )
        )


def test_staged_repo_receipt_records_secret_gate(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "receipt.json"

    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=tmp_path / "staged",
            dry_run=True,
            receipt_path=receipt,
            secret_scanner=_clean_stub,
        )
    )

    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["secret_gate"]["gate_id"] == "secrets"
    assert data["secret_gate"]["status"] == "passed"
    assert data["secret_gate"]["findings"] == []


def test_live_public_repo_rejects_stale_reviewed_source_sha(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    dry_run_target = tmp_path / "dry-run"
    dry_run_receipt = tmp_path / "dry-run.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=dry_run_target,
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=dry_run_receipt,
            secret_scanner=_clean_stub,
        )
    )
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["source_sha"] = "changed-source"
    write_json(bundle / "RELEASE.json", release)

    with pytest.raises(PublicRepoError, match="source_sha"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=tmp_path / "live",
                repo_name="neurips2025-data",
                dry_run=False,
                live=True,
                reviewed_dry_run_receipt=dry_run_receipt,
                secret_scanner=_clean_stub,
            )
        )


def test_live_public_repo_rejects_stale_reviewed_output_checksums(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    dry_run_target = tmp_path / "dry-run"
    dry_run_receipt = tmp_path / "dry-run.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=dry_run_target,
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=dry_run_receipt,
            secret_scanner=_clean_stub,
        )
    )
    (bundle / "decks" / "a.md").write_text("Changed public text\n", encoding="utf-8")

    with pytest.raises(PublicRepoError, match="output checksum mismatch"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=tmp_path / "live",
                repo_name="neurips2025-data",
                dry_run=False,
                live=True,
                reviewed_dry_run_receipt=dry_run_receipt,
                secret_scanner=_clean_stub,
            )
        )


def test_live_public_repo_rejects_extra_current_outputs_not_in_reviewed_receipt(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    dry_run_target = tmp_path / "dry-run"
    dry_run_receipt = tmp_path / "dry-run.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=dry_run_target,
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=dry_run_receipt,
            secret_scanner=_clean_stub,
        )
    )
    (bundle / "decks" / "new.md").write_text("New public text\n", encoding="utf-8")

    with pytest.raises(PublicRepoError, match="checksum set mismatch"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=tmp_path / "live",
                repo_name="neurips2025-data",
                dry_run=False,
                live=True,
                reviewed_dry_run_receipt=dry_run_receipt,
                secret_scanner=_clean_stub,
            )
        )


def test_live_public_repo_writes_operator_blocked_receipt_until_provider_push_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    dry_run_receipt = tmp_path / "dry-run.json"
    live_receipt = tmp_path / "live.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=tmp_path / "dry-run",
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=dry_run_receipt,
            secret_scanner=_clean_stub,
        )
    )

    monkeypatch.setattr(
        public_repo_module,
        "publish_github_repo",
        lambda *_args, **_kwargs: {
            "status": "operator_blocked",
            "exit_code": 2,
            "blocked_reason": "github_provider_command_failed",
            "provider_command_exit_code": 1,
            "public_url": None,
            "commit_sha": None,
        },
    )
    with pytest.raises(PublicRepoError, match="provider command failed"):
        stage_public_repo(
            PublicRepoConfig(
                bundle_root=bundle,
                target_dir=tmp_path / "live",
                repo_name="neurips2025-data",
                dry_run=False,
                live=True,
                receipt_path=live_receipt,
                reviewed_dry_run_receipt=dry_run_receipt,
                secret_scanner=_clean_stub,
            )
        )

    data = json.loads(live_receipt.read_text(encoding="utf-8"))
    assert data["status"] == "operator_blocked"
    assert data["exit_code"] == 2
    assert data["operator_live_step_required"] is True
    assert data["blocked_reason"] == "github_provider_command_failed"


def test_live_public_repo_cli_returns_two_when_provider_push_is_not_implemented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    bundle = make_bundle(tmp_path)
    dry_run_receipt = tmp_path / "dry-run.json"
    live_receipt = tmp_path / "live.json"
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=bundle,
            target_dir=tmp_path / "dry-run",
            repo_name="neurips2025-data",
            dry_run=True,
            receipt_path=dry_run_receipt,
            secret_scanner=_clean_stub,
        )
    )

    def clean_no_secrets(paths, *, receipt_path=None, _scanner=None):
        return GateReceipt(
            gate_id="secrets",
            status="passed",
            timestamp_utc="2026-07-09T00:00:00Z",
            findings=[],
        )

    monkeypatch.setattr(public_repo_module, "no_secrets", clean_no_secrets)
    monkeypatch.setattr(
        public_repo_module,
        "publish_github_repo",
        lambda *_args, **_kwargs: {
            "status": "operator_blocked",
            "exit_code": 2,
            "blocked_reason": "github_provider_command_failed",
            "provider_command_exit_code": 1,
            "public_url": None,
            "commit_sha": None,
        },
    )
    exit_code = public_repo_module.main(
        [
            "--bundle-root",
            str(bundle),
            "--target-dir",
            str(tmp_path / "live"),
            "--repo-name",
            "neurips2025-data",
            "--reviewed-dry-run-receipt",
            str(dry_run_receipt),
            "--live",
            "--receipt-path",
            str(live_receipt),
        ]
    )

    assert exit_code == 2
    assert "provider command failed" in capsys.readouterr().err
    assert json.loads(live_receipt.read_text(encoding="utf-8"))["status"] == "operator_blocked"


def test_scripts_help_lists_live_and_reviewed_receipt():
    result = subprocess.run(
        [sys.executable, "scripts/push_public_repo.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--live" in result.stdout
    assert "--reviewed-dry-run-receipt" in result.stdout


def test_staged_repo_includes_public_url_liveness_workflow(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    stage_public_repo(
        PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True, secret_scanner=_clean_stub)
    )

    workflow = target / ".github" / "workflows" / "public-url-liveness.yml"
    text = workflow.read_text(encoding="utf-8")
    manifest = json.loads((target / "public_repo_manifest.json").read_text(encoding="utf-8"))
    assert workflow.exists()
    assert "scripts/check_public_urls.py" not in text
    assert "urllib.request" in text
    assert "public-url-liveness-receipt.json" in text
    for marker in (".env", "GOOGLE", "OPENAI", "ANTHROPIC", "CLOUDFLARE"):
        assert marker not in text
    assert ".github/workflows/public-url-liveness.yml" in manifest["output_checksums"]


def test_f9_the_staged_repo_states_each_family_license_from_the_release(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    stage_public_repo(PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True, secret_scanner=_clean_stub))

    license_file = (target / "LICENSE").read_text(encoding="utf-8")
    assert "## CC-BY-4.0\n\nFamilies: reader_catalog." in license_file
    assert "## MIT\n\nFamilies: presentation_markdown." in license_file
    assert "Permission is hereby granted, free of charge" in license_file
    assert "CC-BY-SA" not in license_file
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "Licenses by family: CC-BY-4.0 (reader_catalog); MIT (presentation_markdown)." in readme
    manifest = json.loads((target / "public_repo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["license_families"] == {"presentation_markdown": "MIT", "reader_catalog": "CC-BY-4.0"}


def test_f9_a_release_without_licenses_is_not_staged(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    del release["family_licenses"]
    write_json(bundle / "RELEASE.json", release)
    with pytest.raises(PublicRepoError, match="family_licenses"):
        stage_public_repo(
            PublicRepoConfig(bundle_root=bundle, target_dir=tmp_path / "staged", dry_run=True, secret_scanner=_clean_stub)
        )
