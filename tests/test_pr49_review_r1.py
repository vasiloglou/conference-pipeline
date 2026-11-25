"""v6.0 Phase 7.1, PR 49 review pr-49-r1: one regression test per finding (F1-F7)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conference_pipeline.publishing.dataset_package import DatasetPackageError, publish_hf_dataset_live
from tests.test_hf_release_tree import FakeApi
from tests.test_phase71_review_r1 import _listed_bundle, _sha

HOME_PATH = "/" + "Users" + "/review-fixture"


def _publish(bundle: Path, api: FakeApi, scanner=lambda paths: []):
    return publish_hf_dataset_live(
        metadata_dir=bundle / "metadata", repo_id="o/x-data", token="t", hf_api_factory=api, secret_scanner=scanner
    )


def _relist_release(bundle: Path) -> None:
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    checksums["RELEASE.json"] = {"sha256": _sha(bundle / "RELEASE.json")}
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums), encoding="utf-8")


def _mutations(api: FakeApi) -> list[tuple]:
    return [call for call in api.calls if call[0] in ("create_commit", "create_tag")]


# --- F1: the generated card and licence are scanned before anything is pushed ----------


def test_f1_a_marker_that_only_the_generated_card_shows_refuses(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["conference"] = HOME_PATH
    # Escaped in the JSON bytes, so only the decoded card holds the literal path.
    (bundle / "RELEASE.json").write_text(json.dumps(release).replace("/", "\\u002f"), encoding="utf-8")
    _relist_release(bundle)
    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="README.md"):
        _publish(bundle, api)
    assert _mutations(api) == []


def test_f1_the_secrets_gate_sees_the_generated_files(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    seen: list[str] = []

    def scanner(paths):
        seen.extend(Path(p).name for p in paths)
        return []

    _publish(bundle, FakeApi(), scanner=scanner)
    assert "README.md" in seen and "LICENSE" in seen


# --- F2: a listed file missing from the bundle refuses, never deletes the remote copy ---


def test_f2_a_listed_file_missing_on_disk_refuses_before_any_mutation(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    (bundle / "catalog" / "papers.jsonl").unlink()
    api = FakeApi(remote={"catalog/papers.jsonl": "x"})
    with pytest.raises(DatasetPackageError, match="catalog/papers.jsonl"):
        _publish(bundle, api)
    assert _mutations(api) == []


def test_f2_a_missing_index_file_refuses_too(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    (bundle / "index" / "corpus_chunks.parquet").unlink()
    with pytest.raises(DatasetPackageError, match="corpus_chunks.parquet"):
        _publish(bundle, FakeApi())


# --- F3: the direct publishers carry the guards, not only release publish ------------


def test_f3_a_direct_live_r2_upload_without_an_approver_refuses(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import cdn
    from tests.test_release_publish_paths import _media_bundle

    bundle, _media = _media_bundle(tmp_path)
    calls: list = []
    monkeypatch.setattr(cdn, "_r2_upload_item", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(cdn.PublicCdnError, match="approved"):
        cdn.upload_main(["--bundle-root", str(bundle), "--receipt-path", str(tmp_path / "r.json"), "--live", "--bucket", "b"])
    assert calls == []


def test_f3_a_direct_live_r2_upload_over_the_put_limit_refuses(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import cdn
    from conference_pipeline.publishing.surface_limits import limit_for
    from tests.test_release_publish_paths import _media_bundle

    bundle, _media = _media_bundle(tmp_path)
    sidecar = bundle / "audio" / "a.mp3.cdn.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["size_bytes"] = limit_for("r2").max_file_bytes + 1
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    calls: list = []
    monkeypatch.setattr(cdn, "_r2_upload_item", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(cdn.PublicCdnError, match="limit"):
        cdn.plan_public_media_upload(
            cdn.CdnUploadConfig(
                bundle_root=bundle, receipt_path=tmp_path / "r.json", bundle_id="x-v1", bucket="b",
                dry_run=False, live=True, approved_by="Owner",
            )
        )
    assert calls == []


def test_f3_the_orchestrator_passes_the_approver_to_the_r2_script(tmp_path: Path) -> None:
    from conference_pipeline.publishing.public_release import PublicReleaseConfig, PublicReleaseStage, build_public_release_plan
    from tests.test_publish_conference_public_release import make_project

    project_root, _conference_dir = make_project(tmp_path)
    commands = build_public_release_plan(
        PublicReleaseConfig(
            project_root=project_root, conference_slug="sampleconf2027", live_r2=True, ack_live_r2=True,
            r2_bucket="b", owner_approval={"approved_by": "Owner"},
        )
    )
    r2 = [c for c in commands if c.stage is PublicReleaseStage.LIVE_R2][0]
    argv = [str(a) for a in r2.argv]
    assert argv[argv.index("--approved-by") + 1] == "Owner"


def test_f3_github_publish_refuses_a_file_over_its_limit(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import public_repo

    target = tmp_path / "repo"
    target.mkdir()
    (target / "RELEASE.json").write_text(json.dumps({"bundle_id": "x-v1"}), encoding="utf-8")
    big = target / "big.bin"
    with big.open("wb") as handle:
        handle.truncate(100 * 1024 * 1024 + 1)
    calls: list = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    result = public_repo.publish_github_repo(target, "o/x-data", runner=runner)
    assert result["blocked_reason"] == "github_surface_limit" and calls == []


def test_f3_pages_deploy_refuses_a_file_over_its_limit(tmp_path: Path) -> None:
    from conference_pipeline.publishing.live_deployment import CloudflarePagesConfig, deploy_cloudflare_pages
    from tests.test_phase71_review_r3 import _hub

    hub = _hub(tmp_path)
    with (hub / "huge.bin").open("wb") as handle:
        handle.truncate(25 * 1024 * 1024 + 1)
    requests: list = []
    payload = deploy_cloudflare_pages(
        CloudflarePagesConfig(bundle_root=hub, site_dir=hub, receipt_path=tmp_path / "p.json", project_name="p", live=True),
        request_json=lambda *a, **k: requests.append(a) or {"success": True, "result": {}},
        environ={"CLOUDFLARE_API_TOKEN": "t", "CLOUDFLARE_ACCOUNT_ID": "a"},
    )
    assert payload["status"] != "succeeded" and "surface_limit" in payload["blocked_by"] and requests == []


# --- F4: the hub's marker scan uses the environment the publish loads -----------------


def _hub_project(tmp_path: Path, extra: str = "") -> Path:
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    (root / "config" / "public_hub.yaml").write_text(f"title: T\npages_project: p\n{extra}", encoding="utf-8")
    return root


def test_f4_hub_publish_applies_markers_from_its_env_file(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import release_hub
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "notes.txt").write_text("codename-zephyr-fixture\n", encoding="utf-8")
    env_file = tmp_path / "hub.env"
    env_file.write_text("CONFERENCE_PIPELINE_PRIVATE_MARKERS=codename-zephyr-fixture\n", encoding="utf-8")
    deployed: list = []
    monkeypatch.setattr(release_hub, "deploy_cloudflare_pages", lambda *a, **k: deployed.append(a) or {"status": "succeeded"})
    with pytest.raises(release_hub.HubRefused, match="private_marker"):
        release_hub.hub_publish(
            hub, approved_by="Owner", project_root=_hub_project(tmp_path), env_file=env_file, environ={},
            receipt_path=tmp_path / "r.json", secret_scanner=lambda paths: [], echo=lambda s: None,
        )
    assert deployed == []


# --- F5: the HF size check counts the files the upload adds ---------------------------


def test_f5_the_generated_files_count_towards_the_hf_folder_limit(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_cli import surface_findings
    from conference_pipeline.publishing.surface_limits import limit_for

    bundle = _listed_bundle(tmp_path)
    limit = limit_for("hf").max_entries_per_dir
    existing = len(list(bundle.iterdir()))
    for number in range(limit - existing):
        (bundle / f"f{number:05d}.txt").write_text("x", encoding="utf-8")
    assert len(list(bundle.iterdir())) == limit
    findings = surface_findings(bundle, ["hf"])
    assert any(f["path"] == "." and "folder limit" in f["detail"] for f in findings)


def test_f5_the_hf_publish_checks_the_final_tree_before_any_mutation(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import surface_limits

    bundle = _listed_bundle(tmp_path)
    monkeypatch.setattr(
        surface_limits, "LIMITS",
        tuple(row if row.surface != "hf" else surface_limits.SurfaceLimit("hf", row.max_file_bytes, None, 5, row.source) for row in surface_limits.LIMITS),
    )
    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="surface_limit"):
        _publish(bundle, api)
    assert _mutations(api) == []


# --- F6: hub build runs the hub's gates -----------------------------------------------


def test_f6_hub_build_fails_on_a_finding(tmp_path: Path) -> None:
    from conference_pipeline.publishing import release_hub
    from tests.public_site_fixtures import make_alpha

    root = _hub_project(tmp_path, "code_repo_url: http://example.invalid/pipeline\n")
    lines: list[str] = []
    code = release_hub.hub_build([make_alpha(tmp_path)], tmp_path / "hub", project_root=root, echo=lines.append, secret_scanner=lambda paths: [])
    assert code == 1 and any("example.invalid" in line for line in lines)


def test_f6_a_clean_hub_build_still_exits_0(tmp_path: Path) -> None:
    from conference_pipeline.publishing import release_hub
    from tests.public_site_fixtures import make_alpha

    code = release_hub.hub_build(
        [make_alpha(tmp_path)], tmp_path / "hub", project_root=_hub_project(tmp_path), echo=lambda s: None, secret_scanner=lambda paths: []
    )
    assert code == 0


# --- F7: bundle ids are tag names git accepts -----------------------------------------


@pytest.mark.parametrize("tag", ["x..v2", "x-v2.lock", "x-v2.", "x-v1.1.0", "neurips2025-v6.0.0", "x@{v2"])
def test_f7_the_tag_rule_matches_git(tag: str) -> None:
    from conference_pipeline.publishing.release_layout import valid_tag

    git_ok = subprocess.run(["git", "check-ref-format", f"refs/tags/{tag}"], capture_output=True).returncode == 0
    assert valid_tag(tag) is git_ok


def test_f7_hf_refuses_an_invalid_bundle_id_before_any_mutation(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path, bundle_id="x..v2")
    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="tag"):
        _publish(bundle, api)
    assert api.calls == []


def test_f7_github_reads_an_invalid_bundle_id_as_none() -> None:
    from conference_pipeline.publishing.public_repo import _release_bundle_id

    assert _release_bundle_id(json.dumps({"bundle_id": "x-v2.lock"})) is None
    assert _release_bundle_id(json.dumps({"bundle_id": "x-v2.0.0"})) == "x-v2.0.0"
