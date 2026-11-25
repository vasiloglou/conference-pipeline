"""v6.0 Phase 7.1: the publish paths, each against a fake provider with no network.

T4 (size guard before any provider call), T11 (R2 upload end to end against a fake S3
server), T12 (the hub's Pages deploy), T14 (GitHub keeps the earlier version's history).
"""

from __future__ import annotations

import hashlib
import http.server
import json
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from conference_pipeline.publishing.cdn import CdnUploadConfig, plan_public_media_upload
from conference_pipeline.publishing.public_repo import publish_github_repo
from conference_pipeline.publishing.surface_limits import MiB, check_paths, check_tree, limit_for


# --- T3: limits ------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["github", "pages", "hf", "r2"])
def test_t3_a_file_one_byte_over_the_limit_is_a_finding_and_at_the_limit_is_not(tmp_path: Path, surface: str) -> None:
    limit = limit_for(surface).max_file_bytes
    sizes = [(f"at.bin", limit), ("over.bin", limit + 1)]
    findings = __import__("conference_pipeline.publishing.surface_limits", fromlist=["check_files"]).check_files(
        surface, sizes, {".": 2}
    )
    assert [f["path"] for f in findings] == ["over.bin"]
    assert surface in findings[0]["detail"]


def test_t3_a_site_over_the_file_count_and_a_folder_over_its_entries(tmp_path: Path) -> None:
    from conference_pipeline.publishing.surface_limits import check_files

    files = [(f"f{n}", 1) for n in range(20_001)]
    assert [f["path"] for f in check_files("pages", files, {".": 1})] == ["."]
    assert check_files("pages", files[:20_000], {".": 1}) == []
    assert [f["path"] for f in check_files("hf", [("d/x", 1)], {"d": 10_001})] == ["d"]
    assert check_files("hf", [("d/x", 1)], {"d": 10_000}) == []


def test_t3_real_trees_are_walked(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "big").write_bytes(b"\0" * 10)
    assert check_tree("pages", tmp_path) == []
    assert check_paths("hf", tmp_path, ["a/big"]) == []


def test_t3_every_limit_names_its_provider_page() -> None:
    from conference_pipeline.publishing.surface_limits import LIMITS

    assert {row.surface for row in LIMITS} == {"github", "pages", "hf", "r2"}
    assert all(row.source.startswith("https://") for row in LIMITS)
    assert limit_for("pages").max_file_bytes == 25 * MiB and limit_for("github").max_file_bytes == 100 * MiB


# --- T4: release publish refuses an oversized surface before any provider call -------


def test_t4_publish_refuses_an_oversized_site_before_running_anything(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import release_cli
    from conference_pipeline.publishing.surface_limits import SurfaceLimit

    conference = tmp_path / "conferences" / "x"
    bundle = conference / "exports" / "public-release" / "x-v1"
    (bundle / "site").mkdir(parents=True)
    (bundle / "site" / "big.json").write_bytes(b"0" * 2048)
    monkeypatch.setattr(release_cli, "default_bundle_root", lambda conference_dir, bundle_id: bundle)
    import conference_pipeline.publishing.surface_limits as limits

    monkeypatch.setattr(limits, "LIMITS", (SurfaceLimit("pages", 1024, None, None, "https://example.test"),
                                           SurfaceLimit("github", 1 << 40, None, None, "https://example.test"),
                                           SurfaceLimit("hf", 1 << 40, None, None, "https://example.test")))
    calls: list = []
    errors: list[str] = []
    code = release_cli.release_publish(
        "x", approved_by="Owner", surfaces=["pages"], project_root=tmp_path, bundle_id="x-v1",
        runner=lambda *a, **k: calls.append(a), echo=lambda m: None, err=errors.append,
    )
    assert code == 2 and calls == []
    assert any("surface_limit" in e and "site/big.json" in e for e in errors)


# --- T11: R2 end to end against a fake S3 endpoint ------------------------------------


class _FakeS3(http.server.BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers["content-length"])
        body = self.rfile.read(length)
        type(self).received.append(
            {"path": self.path, "sha256": hashlib.sha256(body).hexdigest(), "auth": self.headers.get("authorization", "")}
        )
        self.send_response(200)
        self.send_header("ETag", '"etag-1"')
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


def _media_bundle(tmp_path: Path) -> tuple[Path, bytes]:
    conference = tmp_path / "conferences" / "x"
    bundle = conference / "exports" / "public-release" / "x-v1"
    media = b"ID3 fake audio bytes"
    (conference / "audio").mkdir(parents=True)
    (conference / "audio" / "a.mp3").write_bytes(media)
    (bundle / "audio").mkdir(parents=True)
    (bundle / "RELEASE.json").write_text(json.dumps({"bundle_id": "x-v1"}), encoding="utf-8")
    (bundle / "CHECKSUMS.json").write_text(json.dumps({"audio/a.mp3.cdn.json": {"sha256": "x"}}), encoding="utf-8")
    (bundle / "audio" / "a.mp3.cdn.json").write_text(
        json.dumps(
            {
                "cdn_url": "https://pub-x.r2.dev/conferences/x/audio/a.mp3",
                "sha256": hashlib.sha256(media).hexdigest(),
                "size_bytes": len(media),
                "mime_type": "audio/mpeg",
                "source_filename": "a.mp3",
            }
        ),
        encoding="utf-8",
    )
    return bundle, media


def test_t11_the_r2_upload_puts_each_planned_object_once_with_its_bytes(tmp_path: Path, monkeypatch) -> None:
    bundle, media = _media_bundle(tmp_path)
    _FakeS3.received = []
    with socketserver.TCPServer(("127.0.0.1", 0), _FakeS3) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setenv("CLOUDFLARE_R2_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "test-secret")
        receipt = tmp_path / "r2.json"
        plan_public_media_upload(
            CdnUploadConfig(
                bundle_root=bundle, receipt_path=receipt, bundle_id="x-v1", bucket="b", prefix="", dry_run=False, live=True,
                approved_by="Owner",  # a live upload names its approver (pr-49-r1 F3)
            )
        )
        server.shutdown()
    assert [r["path"] for r in _FakeS3.received] == ["/b/conferences/x/audio/a.mp3"]
    assert _FakeS3.received[0]["sha256"] == hashlib.sha256(media).hexdigest()
    assert _FakeS3.received[0]["auth"].startswith("AWS4-HMAC-SHA256")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["uploaded_count"] == 1 and "test-secret" not in receipt.read_text(encoding="utf-8")


def test_t11_a_dry_run_uploads_nothing(tmp_path: Path, monkeypatch) -> None:
    bundle, _media = _media_bundle(tmp_path)
    _FakeS3.received = []
    plan_public_media_upload(CdnUploadConfig(bundle_root=bundle, receipt_path=tmp_path / "r.json", bundle_id="x-v1", dry_run=True))
    assert _FakeS3.received == []


def test_t11_release_publish_refuses_r2_without_an_approver(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_cli import release_publish

    errors: list[str] = []
    code = release_publish("x", approved_by=" ", surfaces=["r2"], project_root=tmp_path, err=errors.append, echo=lambda m: None)
    assert code == 2 and "owner_approval_missing" in errors[0]


# --- T14: GitHub keeps history ---------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _gh_runner(remote: Path, exists: bool):
    """Runs git for real; answers gh with a local bare repository standing for GitHub."""

    def runner(command, **kwargs):
        if command[0] == "gh":
            if command[1:3] == ["repo", "view"]:
                if not exists:
                    return subprocess.CompletedProcess(command, 1, "", "not found")
                return subprocess.CompletedProcess(command, 0, json.dumps({"url": "https://github.com/o/x-data"}), "")
            return subprocess.CompletedProcess(command, 1, "", "unexpected gh call")
        fixed = [part.replace("https://github.com/o/x-data.git", str(remote)) for part in command]
        return subprocess.run(fixed, **kwargs)

    return runner


def _stage(target: Path, bundle_id: str, files: dict[str, str]) -> None:
    if target.exists():
        subprocess.run(["rm", "-rf", str(target)], check=True)
    target.mkdir(parents=True)
    (target / "RELEASE.json").write_text(json.dumps({"bundle_id": bundle_id}), encoding="utf-8")
    for rel, text in files.items():
        (target / rel).write_text(text, encoding="utf-8")


def test_t14_a_new_version_is_committed_on_the_old_history_and_both_are_tagged(tmp_path: Path, monkeypatch) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    _stage(seed, "x-v1.0.0", {"old.md": "v1"})
    for args in (("init", "-b", "main"), ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-m", "v1")):
        _git(*args, cwd=seed)
    _git("push", str(remote), "HEAD:main", cwd=seed)
    old_head = _git("rev-parse", "HEAD", cwd=seed)

    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.invalid")
    target = tmp_path / "public-repo"
    _stage(target, "x-v1.1.0", {"new.md": "v2"})
    result = publish_github_repo(target, "o/x-data", runner=_gh_runner(remote, exists=True))
    assert result["status"] == "succeeded", result
    head = _git("rev-parse", "main", cwd=remote)
    assert _git("rev-parse", f"{head}^", cwd=remote) == old_head  # the old head is the parent
    assert _git("rev-parse", "x-v1.0.0^{commit}", cwd=remote) == old_head
    assert _git("rev-parse", "x-v1.1.0^{commit}", cwd=remote) == head
    assert _git("ls-tree", "--name-only", "main", cwd=remote).split() == ["RELEASE.json", "new.md"]  # old.md dropped


def test_t14_a_new_repository_is_created_and_tagged(tmp_path: Path, monkeypatch) -> None:
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@example.invalid")
    calls: list[list[str]] = []
    target = tmp_path / "public-repo"
    _stage(target, "x-v1.0.0", {"a.md": "a"})

    def runner(command, **kwargs):
        calls.append(list(command))
        if command[0] == "gh" and command[1:3] == ["repo", "view"]:
            created = any(c[:3] == ["gh", "repo", "create"] for c in calls)
            return subprocess.CompletedProcess(command, 0 if created else 1, json.dumps({"url": "https://github.com/o/x-data"}), "")
        if command[0] == "gh" or command[3:5] == ["push", "origin"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(command, **kwargs)

    result = publish_github_repo(target, "o/x-data", runner=runner)
    assert result["status"] == "succeeded"
    assert any(c[:3] == ["gh", "repo", "create"] for c in calls)
    assert not any("--force-with-lease" in c or "--force" in c for c in calls)
    assert _git("tag", "-l", cwd=target) == "x-v1.0.0"


# --- T12: the hub's Pages deploy -------------------------------------------------------


def _hub(tmp_path: Path) -> tuple[Path, Path]:
    from conference_pipeline.publishing.hub_site import build_hub_site
    from tests.public_site_fixtures import make_alpha, make_beta

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "public_hub.yaml").write_text(
        "title: Fixture hub\npages_project: fixture-hub\n", encoding="utf-8"
    )
    hub = tmp_path / "hub"
    build_hub_site([make_alpha(tmp_path), make_beta(tmp_path)], hub, hub_title="Fixture hub")
    return project, hub


ENV = {"CLOUDFLARE_API_TOKEN": "tok-test", "CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_PAGES_PROJECT": "fixture-hub"}


def test_t12_the_hub_deploys_its_own_directory_through_the_pages_api(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_hub import hub_publish

    project, hub = _hub(tmp_path)
    calls: list[tuple[str, str, dict]] = []

    def request_json(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return {"success": True, "result": {"name": "fixture-hub"}}
        return {"success": True, "result": {"id": "d1", "url": "https://d1.fixture-hub.pages.dev"}}

    receipt = tmp_path / "hub-receipt.json"
    code = hub_publish(hub, approved_by="Owner", project_root=project, env_file=tmp_path / "none.env",
                       receipt_path=receipt, request_json=request_json, environ=ENV, echo=lambda m: None)
    assert code == 0
    assert [c[0] for c in calls] == ["GET", "POST"] and calls[0][1].endswith("/pages/projects/fixture-hub")
    manifest = json.loads(calls[1][2]["data"]["manifest"])
    assert "hub_manifest.json" in manifest and "search/meta.json" in manifest
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "hub" and payload["owner_approval"]["approved_by"] == "Owner"
    assert "tok-test" not in receipt.read_text(encoding="utf-8")


def test_t12_the_hub_publish_refusals(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_hub import HubRefused, hub_publish

    project, hub = _hub(tmp_path)
    with pytest.raises(HubRefused, match="owner_approval_missing"):
        hub_publish(hub, approved_by="", project_root=project, environ=ENV)
    (hub / "hub_manifest.json").unlink()
    with pytest.raises(HubRefused, match="hub_manifest.json is missing"):
        hub_publish(hub, approved_by="Owner", project_root=project, environ=ENV)


def test_t12_a_dry_run_deploys_nothing(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_hub import hub_publish

    project, hub = _hub(tmp_path)
    calls: list = []
    receipt = tmp_path / "r.json"
    code = hub_publish(hub, approved_by="Owner", project_root=project, env_file=tmp_path / "none.env", live=False,
                       receipt_path=receipt, request_json=lambda *a, **k: calls.append(a), environ=ENV, echo=lambda m: None)
    assert code == 0 and calls == []
    assert json.loads(receipt.read_text(encoding="utf-8"))["deployment_url"] is None
