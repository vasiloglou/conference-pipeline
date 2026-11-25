"""v6.0 Phase 7.1, review phase-7.1-r2: one regression test per finding (F1-F7)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conference_pipeline.publishing.dataset_package import DatasetPackageError, verified_hf_upload
from tests.test_phase71_review_r1 import _listed_bundle


# --- F1: R2 sidecars anywhere in the bundle are size-checked ---------------------------


def test_f1_a_sidecar_under_public_repo_is_size_checked(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_cli import surface_findings
    from conference_pipeline.publishing.surface_limits import limit_for

    bundle = tmp_path / "b"
    (bundle / "public-repo" / "video").mkdir(parents=True)
    (bundle / "public-repo" / "video" / "t.mp4.cdn.json").write_text(
        json.dumps({"size_bytes": limit_for("r2").max_file_bytes + 1}), encoding="utf-8"
    )
    assert [f["path"] for f in surface_findings(bundle, ["r2"])] == ["public-repo/video/t.mp4.cdn.json"]


# --- F2: the orchestrator refuses a dry run that also asks for a live stage -----------


def test_f2_dry_run_with_a_live_stage_is_blocked(tmp_path: Path) -> None:
    from tests.test_publish_conference_public_release import load_module, make_project, write_reviewed_export_receipt

    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_reviewed_export_receipt(reviewed, conference_dir)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root, conference_slug="sampleconf2027", dry_run=True, live_r2=True, ack_live_r2=True,
            r2_bucket="b", reviewed_dry_run_receipt=reviewed, receipt_path=receipt_path,
            owner_approval={"approved_by": "Owner"},
        )
    )
    assert result.status == "operator_blocked"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "live_r2" not in [row.get("stage") for row in receipt.get("command_results", [])]
    assert any("dry_run" in " ".join(stage.get("missing_approvals", [])) for stage in receipt["blocked_stages"])


# --- F3: a listing without a digest vouches for nothing --------------------------------


@pytest.mark.parametrize("entry", [{}, {"sha256": None}, {"sha256": ""}, "x"])
def test_f3_a_checksum_entry_without_a_digest_refuses(tmp_path: Path, entry) -> None:
    bundle = _listed_bundle(tmp_path)
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    checksums["decks/a.md"] = entry
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(DatasetPackageError, match="decks/a.md"):
        verified_hf_upload(bundle, secret_scanner=lambda paths: [])


# --- F4: a retry after a failed tag push still preserves the replaced version ----------


def test_f4_a_retry_pushes_the_old_tag_it_failed_to_push(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.public_repo import publish_github_repo
    from tests.test_release_publish_paths import _gh_runner, _git, _stage

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    _stage(seed, "x-v1.0.0", {"old.md": "v1"})
    for args in (("init", "-b", "main"), ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-m", "v1")):
        _git(*args, cwd=seed)
    _git("push", str(remote), "HEAD:main", cwd=seed)
    old_head = _git("rev-parse", "HEAD", cwd=seed)
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@example.invalid")
    real = _gh_runner(remote, exists=True)
    state = {"reject": True}

    def runner(command, **kwargs):
        if "push" in command and any("refs/tags/x-v1.0.0" in part for part in command) and state["reject"]:
            return subprocess.CompletedProcess(command, 1, "", "rejected")
        return real(command, **kwargs)

    target = tmp_path / "public-repo"
    _stage(target, "x-v1.1.0", {"new.md": "v2"})
    assert publish_github_repo(target, "o/x-data", runner=runner)["status"] == "operator_blocked"
    state["reject"] = False
    result = publish_github_repo(target, "o/x-data", runner=runner)  # same staged directory
    assert result["status"] == "succeeded", result
    assert _git("rev-parse", "x-v1.0.0^{commit}", cwd=remote) == old_head
    assert _git("rev-parse", "x-v1.1.0^{commit}", cwd=remote) == _git("rev-parse", "main", cwd=remote)


# --- F5: a rebuilt index's manifest does not refuse its own upload ---------------------


def test_f5_an_index_built_twice_still_verifies(tmp_path: Path) -> None:
    from conference_pipeline.publishing import public_index as pi
    from conference_pipeline.publishing.dataset_package import DatasetPackageConfig, package_public_dataset
    from tests.public_site_fixtures import make_alpha

    import hashlib

    bundle = make_alpha(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["included_families"] = release.get("included_families") or ["decks"]
    release["family_licenses"] = {family: "MIT" for family in release["included_families"]}
    (bundle / "RELEASE.json").write_text(json.dumps(release), encoding="utf-8")
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    checksums["RELEASE.json"] = {"sha256": hashlib.sha256((bundle / "RELEASE.json").read_bytes()).hexdigest()}
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums), encoding="utf-8")
    fn = lambda chunks: __import__("numpy").ones((len(chunks), 8), dtype="float32")  # noqa: E731
    for _ in range(2):
        pi.build_public_index(pi.PublicIndexConfig(bundle_root=bundle, embedding_dim=8, canonical_embed_fn=fn))
    package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "r.json"))
    manifest = json.loads((bundle / "index" / "index_manifest.json").read_text(encoding="utf-8"))
    assert "index_manifest.json" not in manifest["output_checksums"]
    verified_hf_upload(bundle, secret_scanner=lambda paths: [])


# --- F6: the dataset package's default receipt is not part of the upload ---------------


def test_f6_the_default_package_receipt_is_left_out_of_the_hf_tree(tmp_path: Path) -> None:
    from conference_pipeline.publishing.dataset_package import DatasetPackageConfig, hf_upload_relatives, package_public_dataset

    bundle = _listed_bundle(tmp_path)
    package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata"))
    assert (bundle / "metadata" / "dataset_package_receipt.json").is_file()
    assert "metadata/dataset_package_receipt.json" not in hf_upload_relatives(bundle)
    verified_hf_upload(bundle, secret_scanner=lambda paths: [])


# --- F7: HF dataset ids follow the Hub's own rules -------------------------------------


@pytest.mark.parametrize("repo", ["o/a..b", "o/a--b", "o/a.git", "o/" + "a" * 97, "o/-a", "o/a-"])
def test_f7_an_hf_id_the_hub_rejects_is_refused(repo: str) -> None:
    from conference_pipeline.publishing.release_layout import ReleaseTargetError, stamped_urls

    with pytest.raises(ReleaseTargetError):
        stamped_urls({"hf_repo_id": repo})


@pytest.mark.parametrize("repo", ["o/a.git", "o/a..b"])
def test_f7_a_github_name_github_rejects_is_refused(repo: str) -> None:
    from conference_pipeline.publishing.release_layout import ReleaseTargetError, stamped_urls

    with pytest.raises(ReleaseTargetError):
        stamped_urls({"github_repo": repo})


def test_f7_the_live_names_still_pass() -> None:
    from conference_pipeline.publishing.release_layout import stamped_urls

    urls = stamped_urls({"github_repo": "vasiloglou/iclr2026-data", "hf_repo_id": "vasiloglou/iclr2026-data", "pages_project": "iclr2026-data"})
    assert urls["hf_dataset_url"].endswith("/vasiloglou/iclr2026-data")
