from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.publishing.cdn import (
    CdnUploadConfig,
    PublicCdnError,
    check_public_urls,
    default_http_get,
    is_completed_live_upload_receipt,
    object_key_for,
    plan_public_media_upload,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path, *, cdn_url: str = "https://cdn.example.test/audio.mp3") -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(root / "RELEASE.json", {"bundle_id": "bundle-test", "conference": "sampleconf", "source_sha": "abcdef", "family_licenses": {"podcasts": "MIT"}})
    write_json(root / "CHECKSUMS.json", {"audio/a.mp3.cdn.json": {"sha256": "x"}})
    write_json(
        root / "audio" / "a.mp3.cdn.json",
        {
            "cdn_url": cdn_url,
            "sha256": "abc123",
            "size_bytes": 123,
            "mime_type": "audio/mpeg",
            "source_filename": "a.mp3",
        },
    )
    return root


def test_cdn_upload_dry_run_writes_plan_without_uploading_bytes(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "cdn-receipt.json"
    items = plan_public_media_upload(
        CdnUploadConfig(bundle_root=bundle, receipt_path=receipt, dry_run=True)
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert len(items) == 1
    assert data["receipt_type"] == "conference_pipeline.public_cdn_upload"
    assert data["dry_run"] is True
    assert data["planned_upload_count"] == 1
    assert data["planned_uploads"][0]["cdn_url"] == "https://cdn.example.test/audio.mp3"


def test_completed_live_upload_requires_exact_uploaded_object_proof():
    planned = {
        "object_key": "conferences/iclr2026/bundle/audio.mp3",
        "sha256": "abc123",
        "size_bytes": 123,
    }
    payload = {
        "receipt_type": "conference_pipeline.public_cdn_upload",
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": False,
        "live": True,
        "bundle_id": "iclr2026-v1.0.0",
        "planned_upload_count": 1,
        "uploaded_count": 1,
        "planned_uploads": [planned],
        "uploaded_objects": [dict(planned)],
    }

    assert is_completed_live_upload_receipt(payload, "iclr2026-v1.0.0") is True
    for field, value in (("dry_run", True), ("uploaded_objects", []), ("uploaded_count", 0)):
        broken = dict(payload)
        broken[field] = value
        assert is_completed_live_upload_receipt(broken, "iclr2026-v1.0.0") is False

    broken = dict(payload)
    broken["uploaded_objects"] = [{**planned, "sha256": "drift"}]
    assert is_completed_live_upload_receipt(broken, "iclr2026-v1.0.0") is False


def test_iclr_versioned_cdn_url_maps_to_r2_object_key():
    url = (
        "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/"
        "conferences/iclr2026/iclr2026-v1.0.0/"
        "decks/payloads/topic/presentation_video.mp4"
    )

    assert object_key_for(url, "") == (
        "conferences/iclr2026/iclr2026-v1.0.0/"
        "decks/payloads/topic/presentation_video.mp4"
    )
    assert object_key_for(url, "conferences/iclr2026") == (
        "conferences/iclr2026/iclr2026-v1.0.0/"
        "decks/payloads/topic/presentation_video.mp4"
    )


def test_cdn_sidecar_missing_required_fields_fails(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    write_json(bundle / "audio" / "a.mp3.cdn.json", {"cdn_url": "https://cdn.example.test/a.mp3"})
    with pytest.raises(PublicCdnError):
        plan_public_media_upload(
            CdnUploadConfig(bundle_root=bundle, receipt_path=tmp_path / "receipt.json")
        )


def test_url_liveness_mocked_200_succeeds(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "liveness.json"
    check_public_urls(bundle, receipt_path=receipt, http_get=lambda _url: 200)
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert data["status"] == "succeeded"
    assert data["checked_count"] == 1
    assert data["passed_count"] == 1
    assert data["failed_count"] == 0
    assert data["secret_values_recorded"] is False


def test_url_liveness_collects_extra_metadata_and_records_sources(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["cdn_base_url"] = "https://cdn.example.test"
    write_json(bundle / "RELEASE.json", release)
    metadata = bundle / "metadata" / "release-note.json"
    provider_receipt = tmp_path / "github-neurips2025-phase06-proof-live-receipt.json"
    blocked_receipt = tmp_path / "hf-neurips2025-phase06-proof-live-receipt.json"
    write_json(
        metadata,
        {
            "public_repo_url": "https://github.com/vasiloglou/neurips2025-data",
            "public_site_url": "https://neurips2025-data.pages.dev",
            "cdn_base_url": "https://cdn.example.test",
            "evidence_url": "https://docs.example.test/evidence.json",
        },
    )
    write_json(provider_receipt, {"public_url": "https://github.com/vasiloglou/neurips2025-data"})
    write_json(blocked_receipt, {"status": "operator_blocked", "provider": "huggingface", "dataset_url": None})
    receipt = tmp_path / "liveness-extra.json"
    checked: list[str] = []

    def http_get(url: str) -> int:
        checked.append(url)
        return 200

    check_public_urls(
        bundle,
        receipt_path=receipt,
        extra_metadata=[metadata, provider_receipt, blocked_receipt],
        http_get=http_get,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert data["status"] == "succeeded"
    assert data["checked_sources"] == [
        "audio/a.mp3.cdn.json",
        "github-neurips2025-phase06-proof-live-receipt.json",
        "hf-neurips2025-phase06-proof-live-receipt.json",
        "metadata/release-note.json",
    ]
    assert data["checked_count"] == 4
    assert data["checked_urls"] == sorted(checked)
    assert "https://cdn.example.test" not in checked
    assert "https://github.com/vasiloglou/neurips2025-data" in checked
    assert "https://neurips2025-data.pages.dev" in checked
    assert "https://docs.example.test/evidence.json" in checked


def test_url_liveness_labels_root_docs_without_expanding_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["cdn_base_url"] = "https://cdn.example.test"
    write_json(bundle / "RELEASE.json", release)
    docs_release = tmp_path / "docs" / "operations" / "public-release.md"
    provider_receipt = (
        tmp_path
        / "conferences"
        / "neurips2025"
        / "storage"
        / "workflow-runs"
        / "github-neurips2025-phase06-proof-post-restamp-receipt.json"
    )
    docs_release.parent.mkdir(parents=True, exist_ok=True)
    docs_release.write_text("Evidence: `https://docs.example.test/final-proof.json`\n", encoding="utf-8")
    write_json(provider_receipt, {"public_url": "https://github.com/vasiloglou/neurips2025-data"})

    receipt = tmp_path / "liveness-docs.json"
    check_public_urls(
        bundle,
        receipt_path=receipt,
        extra_metadata=[docs_release, provider_receipt],
        http_get=lambda _url: 200,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert "docs/operations/public-release.md" in data["checked_sources"]
    assert "github-neurips2025-phase06-proof-post-restamp-receipt.json" in data["checked_sources"]
    assert "conferences/neurips2025/storage/workflow-runs" not in "\n".join(data["checked_sources"])
    assert "https://docs.example.test/final-proof.json`" not in data["checked_urls"]


def test_url_liveness_mocked_404_fails_and_lists_url(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "liveness.json"
    with pytest.raises(PublicCdnError):
        check_public_urls(bundle, receipt_path=receipt, http_get=lambda _url: 404)
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert data["status"] == "failed"
    assert data["failed_count"] == 1
    assert data["results"][0]["url"] == "https://cdn.example.test/audio.mp3"


def test_url_liveness_transport_error_writes_failed_receipt(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "liveness.json"

    def http_get(_url: str) -> int:
        raise TimeoutError("timed out")

    with pytest.raises(PublicCdnError):
        check_public_urls(bundle, receipt_path=receipt, http_get=http_get)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["failed_count"] == 1
    assert data["results"][0]["http_status"] is None
    assert data["results"][0]["reason"] == "transport_error"


def test_default_liveness_request_uses_public_release_user_agent(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        captured["user_agent"] = request.headers.get("User-agent")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert default_http_get("https://cdn.example.test/audio.mp3") == 200
    assert captured["timeout"] == 15
    assert captured["user_agent"].startswith("conference-pipeline-public-liveness/")


def test_liveness_rejects_file_and_drive_paths_before_http(tmp_path: Path):
    for bad_url in ("file://tmp/x", "/Users/example/Drive/file.mp3"):
        bundle = make_bundle(tmp_path, cdn_url=bad_url)
        called = False

        def http_get(_url: str) -> int:
            nonlocal called
            called = True
            return 200

        with pytest.raises(PublicCdnError):
            check_public_urls(bundle, receipt_path=tmp_path / f"{abs(hash(bad_url))}.json", http_get=http_get)
        assert called is False
        # Recreate temp bundle path for the next invalid URL.
        tmp_path = tmp_path / "next"


def test_staged_repo_workflow_contains_liveness_script(tmp_path: Path):
    from conference_pipeline.publishing.public_repo import PublicRepoConfig, stage_public_repo

    bundle = make_bundle(tmp_path)
    target = tmp_path / "staged"
    stage_public_repo(PublicRepoConfig(bundle_root=bundle, target_dir=target, dry_run=True))
    workflow = (target / ".github" / "workflows" / "public-url-liveness.yml").read_text(encoding="utf-8")
    assert "scripts/check_public_urls.py" not in workflow
    assert "urllib.request" in workflow
    for marker in (".env", "GOOGLE", "OPENAI", "ANTHROPIC", "CLOUDFLARE"):
        assert marker not in workflow
