"""v6.0 Phase 7.1, review phase-7.1-r1: one regression test per finding (F1-F9)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conference_pipeline.publishing.dataset_package import DatasetPackageError, publish_hf_dataset_live
from tests.test_hf_release_tree import FakeApi


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listed_bundle(tmp_path: Path, bundle_id: str = "x-v1.1.0") -> Path:
    """A bundle whose every HF file is listed by CHECKSUMS.json or the index manifest."""
    bundle = tmp_path / "conferences" / "x" / "exports" / "public-release" / bundle_id
    files = {
        "RELEASE.json": json.dumps({"bundle_id": bundle_id, "conference": "X", "family_licenses": {"decks": "MIT"}}),
        "decks/a.md": "# A",
        "catalog/papers.jsonl": '{"paper_id": "p1"}\n',
        "index/corpus_chunks.parquet": "pq",
        "metadata/datapackage.json": "{}",
    }
    for rel, text in files.items():
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_text(text, encoding="utf-8")
    checksums = {rel: {"sha256": _sha(bundle / rel)} for rel in ("RELEASE.json", "decks/a.md", "catalog/papers.jsonl")}
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums), encoding="utf-8")
    (bundle / "index" / "index_manifest.json").write_text(
        json.dumps({"output_checksums": {"corpus_chunks.parquet": _sha(bundle / "index" / "corpus_chunks.parquet")}}),
        encoding="utf-8",
    )
    return bundle


def _publish(bundle: Path, api: FakeApi, **kw):
    import inspect

    # The scanner stub is passed only where the function takes one, so these tests show the
    # pre-fix behaviour (no refusal) rather than a TypeError on the new parameter.
    if "secret_scanner" in inspect.signature(publish_hf_dataset_live).parameters:
        kw["secret_scanner"] = lambda paths: []
    return publish_hf_dataset_live(metadata_dir=bundle / "metadata", repo_id="o/x-data", token="t", hf_api_factory=api, **kw)


# --- F1: the HF upload is only what the bundle's listings name, re-scanned -------------


def test_f1_a_file_added_after_the_review_is_refused(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    (bundle / "extra-private.txt").write_text("not reviewed", encoding="utf-8")
    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="extra-private.txt"):
        _publish(bundle, api)
    assert not [c for c in api.calls if c[0] in ("create_commit", "create_tag")]


def test_f1_a_listed_file_whose_bytes_changed_is_refused(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    (bundle / "decks" / "a.md").write_text("# changed", encoding="utf-8")
    with pytest.raises(DatasetPackageError, match="decks/a.md"):
        _publish(bundle, FakeApi())


def test_f1_the_secrets_gate_runs_over_the_exact_upload(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    seen: list[list[str]] = []

    def scanner(paths):
        # The generated card and licence are staged outside the bundle (pr-49-r1 F1).
        seen.append(sorted(p.relative_to(bundle).as_posix() if p.is_relative_to(bundle) else p.name for p in paths))
        return [{"RuleID": "fake", "File": str(paths[0])}]

    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="secrets"):
        publish_hf_dataset_live(metadata_dir=bundle / "metadata", repo_id="o/x-data", token="t", hf_api_factory=api, secret_scanner=scanner)
    assert seen and "decks/a.md" in seen[0]
    assert not [c for c in api.calls if c[0] == "create_commit"]


def test_f1_a_clean_listed_bundle_still_publishes(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    result = _publish(bundle, FakeApi())
    assert result["tag"] == "x-v1.1.0"


# --- F2: R2 objects are checked against the single-PUT limit ---------------------------


def test_f2_an_r2_object_over_the_put_limit_is_a_finding(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_cli import surface_findings
    from conference_pipeline.publishing.surface_limits import limit_for

    bundle = tmp_path / "b"
    (bundle / "video").mkdir(parents=True)
    over = limit_for("r2").max_file_bytes + 1
    (bundle / "video" / "talk.mp4.cdn.json").write_text(json.dumps({"size_bytes": over, "cdn_url": "https://x.test/v"}), encoding="utf-8")
    (bundle / "video" / "ok.mp4.cdn.json").write_text(json.dumps({"size_bytes": 10, "cdn_url": "https://x.test/o"}), encoding="utf-8")
    findings = surface_findings(bundle, ["r2"])
    assert [f["path"] for f in findings] == ["video/talk.mp4.cdn.json"]


# --- F3: the orchestrator's R2 stage needs the owner's named approval ------------------


def test_f3_live_r2_without_a_named_owner_approval_is_blocked(tmp_path: Path) -> None:
    from tests.test_publish_conference_public_release import load_module, make_project, write_reviewed_export_receipt

    module = load_module()
    project_root, conference_dir = make_project(tmp_path)
    reviewed = conference_dir / "storage" / "workflow-runs" / "dry-run.json"
    receipt_path = conference_dir / "storage" / "workflow-runs" / "orchestrator.json"
    write_reviewed_export_receipt(reviewed, conference_dir)
    result = module.run_public_release(
        module.PublicReleaseConfig(
            project_root=project_root, conference_slug="sampleconf2027", live_r2=True, ack_live_r2=True,
            r2_bucket="b", reviewed_dry_run_receipt=reviewed, receipt_path=receipt_path, owner_approval=None,
        )
    )
    assert result.status == "operator_blocked"
    blocked = json.loads(receipt_path.read_text(encoding="utf-8"))["blocked_stages"][0]
    assert "owner_approval" in blocked["missing_approvals"]


# --- F4: live and dry-run together refuse ---------------------------------------------


def test_f4_live_with_dry_run_refuses_before_any_upload(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import cdn
    from tests.test_release_publish_paths import _media_bundle

    bundle, _media = _media_bundle(tmp_path)
    calls: list = []
    monkeypatch.setattr(cdn, "_r2_upload_item", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(cdn.PublicCdnError, match="dry"):
        cdn.plan_public_media_upload(
            cdn.CdnUploadConfig(bundle_root=bundle, receipt_path=tmp_path / "r.json", bundle_id="x-v1", bucket="b", dry_run=True, live=True)
        )
    assert calls == []


# --- F5, F6: HF version tags are never skipped or silently reused ----------------------


def test_f5_an_unreadable_remote_release_refuses_instead_of_skipping_the_tag(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)

    class Broken(FakeApi):
        def hf_hub_download(self, **kwargs):
            raise OSError("network")

    api = Broken(remote={"RELEASE.json": "{}"})
    with pytest.raises(DatasetPackageError, match="RELEASE.json"):
        _publish(bundle, api)
    assert not [c for c in api.calls if c[0] in ("create_commit", "create_tag")]


def test_f6_a_bundle_id_already_tagged_refuses(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path, "x-v2")
    api = FakeApi(tags=["x-v2"])
    with pytest.raises(DatasetPackageError, match="x-v2"):
        _publish(bundle, api)
    assert not [c for c in api.calls if c[0] in ("create_commit", "create_tag")]


# --- F7: GitHub pushes the old version's tag before it moves main ---------------------


def test_f7_a_rejected_tag_push_leaves_main_unmoved(tmp_path: Path, monkeypatch) -> None:
    import subprocess

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
    pushes: list[list[str]] = []

    def runner(command, **kwargs):
        if "push" in command:
            pushes.append(list(command))
            if any("tags" in part for part in command):
                return subprocess.CompletedProcess(command, 1, "", "tag push rejected")
        return real(command, **kwargs)

    target = tmp_path / "public-repo"
    _stage(target, "x-v1.1.0", {"new.md": "v2"})
    result = publish_github_repo(target, "o/x-data", runner=runner)
    assert result["status"] == "operator_blocked"
    assert _git("rev-parse", "main", cwd=remote) == old_head  # main never moved
    assert all("HEAD:main" not in " ".join(p) for p in pushes)


def test_f6_github_refuses_a_bundle_id_already_tagged_on_the_remote(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from conference_pipeline.publishing.public_repo import publish_github_repo
    from tests.test_release_publish_paths import _gh_runner, _git, _stage

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    _stage(seed, "x-v1.0.0", {"old.md": "v1"})
    for args in (("init", "-b", "main"), ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-m", "v1"), ("tag", "x-v1.1.0")):
        _git(*args, cwd=seed)
    _git("push", str(remote), "HEAD:main", "--tags", cwd=seed)
    old_head = _git("rev-parse", "HEAD", cwd=seed)
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@example.invalid")
    target = tmp_path / "public-repo"
    _stage(target, "x-v1.1.0", {"new.md": "v2"})
    result = publish_github_repo(target, "o/x-data", runner=_gh_runner(remote, exists=True))
    assert result["status"] == "operator_blocked" and "tag" in result["blocked_reason"]
    assert _git("rev-parse", "main", cwd=remote) == old_head


# --- F8: a cache whose matrix is not the one its metadata describes refuses ------------


def test_f8_a_matrix_replaced_under_old_metadata_is_refused(tmp_path: Path) -> None:
    from conference_pipeline.publishing.embedding_cache import EmbeddingCacheError, load_lane, text_key, write_lane

    write_lane(tmp_path, "canonical", "m", [text_key("a")], np.ones((1, 4), dtype=np.float32))
    np.save(tmp_path / "canonical.npy", np.full((1, 4), 2.0, dtype=np.float32))  # interrupted overwrite
    with pytest.raises(EmbeddingCacheError, match="digest"):
        load_lane(tmp_path, "canonical", "m", ["a"])


# --- F9: every target is validated, whatever else is set ------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        {"pages_project": "INVALID/PROJECT", "public_site_url": "https://site.example.test"},
        {"hf_repo_id": "o/.."},
        {"github_repo": "o/."},
        {"cdn_base_url": "https://."},
        {"public_site_url": "https://-bad-.example"},
    ],
)
def test_f9_a_malformed_target_is_refused(settings: dict) -> None:
    from conference_pipeline.publishing.release_layout import ReleaseTargetError, stamped_urls

    with pytest.raises(ReleaseTargetError):
        stamped_urls(settings)
