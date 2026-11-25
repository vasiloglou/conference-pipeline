from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conference_pipeline.publishing.live_deployment import (
    CloudflarePagesConfig,
    DeploymentPreflightConfig,
    LiveDeploymentError,
    MetadataAgreementConfig,
    check_metadata_agreement,
    deploy_cloudflare_pages,
    preflight_public_deployment,
    restamp_release_metadata,
    write_operator_blocked_receipt,
)


REPO_ROOT = Path(__file__).parent.parent
R2_BASE_URL = "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/conferences/neurips2025"
GITHUB_REPO = "vasiloglou/neurips2025-data"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path, *, public_repo_url: str = "https://github.com/vasiloglou/neurips2025-data") -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "neurips2025-phase06-proof"
    root.mkdir(parents=True)
    release = {
        "bundle_id": "neurips2025-phase06-proof",
        "conference": "neurips2025",
        "source_sha": "abcdef1234567890",
        "public_repo_url": public_repo_url,
        "public_site_url": "https://neurips2025-data.pages.dev",
        "cdn_base_url": R2_BASE_URL,
        "zenodo_doi": "10.5281/zenodo.1234567",
        "hf_dataset_url": "https://huggingface.co/datasets/vasiloglou/neurips2025-data",
    }
    write_json(root / "RELEASE.json", release)
    write_json(root / "CHECKSUMS.json", {"RELEASE.json": {"sha256": "old-release-digest"}})
    write_json(root / "audio" / "talk.mp3.cdn.json", {"cdn_url": f"{R2_BASE_URL}/audio/talk.mp3"})
    write_json(root / "site" / "site_manifest.json", {"public_site_url": release["public_site_url"]})
    write_json(
        root / "metadata" / "croissant.json",
        {
            "@type": "Dataset",
            "url": public_repo_url,
            "sameAs": [release["public_site_url"], release["hf_dataset_url"]],
            "identifier": release["zenodo_doi"],
            "distribution": [{"contentUrl": f"{R2_BASE_URL}/audio/talk.mp3"}],
        },
    )
    write_json(
        root / "metadata" / "datapackage.json",
        {
            "homepage": public_repo_url,
            "sources": [release["public_site_url"]],
            "resources": [{"path": f"{R2_BASE_URL}/audio/talk.mp3"}],
        },
    )
    write_json(
        root / "metadata" / "dataset_manifest.json",
        {
            "public_repo_url": public_repo_url,
            "public_site_url": release["public_site_url"],
            "cdn_base_url": R2_BASE_URL,
            "zenodo_doi": release["zenodo_doi"],
            "hf_dataset_url": release["hf_dataset_url"],
        },
    )
    (root / "metadata" / "CITATION.cff").write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                "title: NeurIPS 2025 data",
                f"url: {public_repo_url}",
                f"doi: {release['zenodo_doi']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "public-repo").mkdir()
    (root / "public-repo" / "README.md").write_text(
        f"# neurips2025-data\n\nRepo: {public_repo_url}\nSite: {release['public_site_url']}\nCDN: {R2_BASE_URL}\n",
        encoding="utf-8",
    )
    (root / "public-repo" / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nurl: {public_repo_url}\ndoi: {release['zenodo_doi']}\n",
        encoding="utf-8",
    )
    return root


def make_pages_site(bundle: Path) -> Path:
    site = bundle / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text("<!doctype html><title>NeurIPS 2025</title>\n", encoding="utf-8")
    write_json(site / "site_manifest.json", {"bundle_id": "neurips2025-phase06-proof"})
    write_json(site / "search-data.json", [{"chunk_id": "c1", "text": "public"}])
    return site


def pages_config(tmp_path: Path, bundle: Path, site: Path, *, live: bool = True) -> CloudflarePagesConfig:
    return CloudflarePagesConfig(
        bundle_root=bundle,
        site_dir=site,
        receipt_path=tmp_path / "cloudflare-pages.json",
        project_name="neurips2025-data",
        live=live,
    )


def write_success_receipts(tmp_path: Path) -> dict[str, Path]:
    receipts = {
        "public_repo_dry_run": tmp_path / "public-repo-receipt.json",
        "site": tmp_path / "site-receipt.json",
        "dataset": tmp_path / "dataset-receipt.json",
        "r2_upload": tmp_path / "r2-upload-receipt.json",
        "r2_liveness": tmp_path / "r2-liveness-receipt.json",
    }
    write_json(
        receipts["public_repo_dry_run"],
        {
            "receipt_type": "conference_pipeline.public_repo_push",
            "status": "succeeded",
            "dry_run": True,
            "repo_name": "neurips2025-data",
            "public_safety": {
                "status": "clean",
                "scope": "staged_public_repo_tree",
                "private_marker_count": 0,
                "source_repo_history_published": False,
            },
            "secret_gate": {
                "gate_id": "secrets",
                "status": "passed",
                "timestamp_utc": "2026-07-09T00:00:00Z",
                "findings": [],
            },
        },
    )
    write_json(receipts["site"], {"receipt_type": "conference_pipeline.public_site", "status": "succeeded"})
    write_json(
        receipts["dataset"],
        {"receipt_type": "conference_pipeline.public_dataset_package", "status": "succeeded"},
    )
    write_json(
        receipts["r2_upload"],
        {
            "receipt_type": "conference_pipeline.public_cdn_upload",
            "status": "succeeded",
            "live": True,
            "base_url": R2_BASE_URL,
            "planned_upload_count": 546,
            "uploaded_count": 546,
            "uploaded_objects": [{"object_key": "audio/talk.mp3", "etag": "etag-1"}],
        },
    )
    write_json(
        receipts["r2_liveness"],
        {
            "receipt_type": "conference_pipeline.public_url_liveness",
            "status": "succeeded",
            "checked_count": 546,
            "failed_count": 0,
        },
    )
    return receipts


def make_preflight_config(tmp_path: Path, bundle: Path, receipts: dict[str, Path]) -> DeploymentPreflightConfig:
    canonical_env = tmp_path / "canonical" / ".env"
    repo_env = tmp_path / "repo" / ".env"
    repo_env.parent.mkdir(parents=True)
    repo_env.write_text(
        "\n".join(
            [
                "CLOUDFLARE_API_TOKEN=cf_dummy_public_test_value",
                "ZENODO_TOKEN=zenodo_dummy_public_test_value",
                "HF_TOKEN=hf_dummy_public_test_value",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return DeploymentPreflightConfig(
        bundle_root=bundle,
        receipt_path=tmp_path / "preflight.json",
        github_repo=GITHUB_REPO,
        r2_base_url=R2_BASE_URL,
        public_repo_dry_run_receipt=receipts["public_repo_dry_run"],
        site_receipt=receipts["site"],
        dataset_receipt=receipts["dataset"],
        r2_upload_receipt=receipts["r2_upload"],
        r2_liveness_receipt=receipts["r2_liveness"],
        canonical_env_path=canonical_env,
        repo_env_path=repo_env,
    )


def test_preflight_records_reviewed_bundle_r2_and_github_target(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipts = write_success_receipts(tmp_path)
    config = make_preflight_config(tmp_path, bundle, receipts)

    preflight_public_deployment(config)
    data = json.loads(config.receipt_path.read_text(encoding="utf-8"))

    assert data["status"] == "succeeded"
    assert data["bundle_id"] == "neurips2025-phase06-proof"
    assert data["github"]["owner"] == "vasiloglou"
    assert data["github"]["repo"] == "vasiloglou/neurips2025-data"
    assert data["github"]["repo_name"] == "neurips2025-data"
    assert data["r2"]["base_url"] == R2_BASE_URL
    assert data["r2"]["live_upload"]["receipt_path"] == receipts["r2_upload"].as_posix()
    assert data["r2"]["live_upload"]["uploaded_count"] == 546
    assert data["r2"]["live_liveness"]["receipt_path"] == receipts["r2_liveness"].as_posix()
    assert data["r2"]["live_liveness"]["failed_count"] == 0
    assert data["public_repo_safety"]["status"] == "clean"
    assert data["public_repo_safety"]["source_repo_history_published"] is False
    assert data["secret_values_recorded"] is False


def test_preflight_rejects_missing_r2_uploaded_count(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipts = write_success_receipts(tmp_path)
    upload = json.loads(receipts["r2_upload"].read_text(encoding="utf-8"))
    upload.pop("uploaded_count")
    write_json(receipts["r2_upload"], upload)

    with pytest.raises(LiveDeploymentError, match="R2 upload receipt count mismatch"):
        preflight_public_deployment(make_preflight_config(tmp_path, bundle, receipts))


def test_preflight_rejects_missing_r2_uploaded_object_proof(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipts = write_success_receipts(tmp_path)
    upload = json.loads(receipts["r2_upload"].read_text(encoding="utf-8"))
    upload["uploaded_objects"] = []
    write_json(receipts["r2_upload"], upload)

    with pytest.raises(LiveDeploymentError, match="missing uploaded object proof"):
        preflight_public_deployment(make_preflight_config(tmp_path, bundle, receipts))


def test_preflight_rejects_public_repo_receipt_without_secret_gate(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipts = write_success_receipts(tmp_path)
    receipt = json.loads(receipts["public_repo_dry_run"].read_text(encoding="utf-8"))
    receipt.pop("secret_gate")
    write_json(receipts["public_repo_dry_run"], receipt)

    with pytest.raises(LiveDeploymentError, match="missing passed secret gate"):
        preflight_public_deployment(make_preflight_config(tmp_path, bundle, receipts))


def test_preflight_records_credential_names_without_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for name in (
        "CLOUDFLARE_API_TOKEN",
        "ZENODO_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    bundle = make_bundle(tmp_path)
    receipts = write_success_receipts(tmp_path)
    config = make_preflight_config(tmp_path, bundle, receipts)

    preflight_public_deployment(config)
    text = config.receipt_path.read_text(encoding="utf-8")
    data = json.loads(text)

    assert data["credentials"]["required_env_vars"] == [
        "CLOUDFLARE_API_TOKEN",
        "ZENODO_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
    ]
    assert data["credentials"]["canonical_env_path_exists"] is False
    assert data["credentials"]["repo_env_path_exists"] is True
    assert data["credentials"]["env_path_discrepancy"] is True
    assert data["credentials"]["present_env_vars"] == [
        "CLOUDFLARE_API_TOKEN",
        "HF_TOKEN",
        "ZENODO_TOKEN",
    ]
    assert data["credentials"]["missing_env_vars"] == ["HUGGINGFACE_API_KEY"]
    assert data["credentials"]["hf_token_alias_mismatch"] is True
    for dummy_value in (
        "cf_dummy_public_test_value",
        "zenodo_dummy_public_test_value",
        "hf_dummy_public_test_value",
    ):
        assert dummy_value not in text


def test_operator_blocked_receipt_records_blockers_and_redacts(tmp_path: Path):
    receipt = tmp_path / "blocked.json"
    write_operator_blocked_receipt(
        receipt,
        provider="zenodo",
        target="zenodo.org",
        blocked_by=["missing_token", "missing_ack"],
        missing_env_vars=["ZENODO_TOKEN"],
        missing_approvals=["ack_live_doi"],
    )
    text = receipt.read_text(encoding="utf-8")
    data = json.loads(text)

    assert data["status"] == "operator_blocked"
    assert data["exit_code"] == 2
    assert data["provider"] == "zenodo"
    assert data["target"] == "zenodo.org"
    assert data["blocked_by"] == ["missing_token", "missing_ack"]
    assert data["missing_env_vars"] == ["ZENODO_TOKEN"]
    assert data["missing_approvals"] == ["ack_live_doi"]
    assert data["secret_values_recorded"] is False
    assert "fake-secret-token" not in text


def test_metadata_agreement_fails_when_public_surfaces_disagree(tmp_path: Path):
    bundle = make_bundle(
        tmp_path,
        public_repo_url="https://github.com/vasiloglou/neurips2025-data",
    )
    (bundle / "public-repo" / "README.md").write_text(
        "Repo: https://github.com/someone-else/neurips2025-data\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "metadata-agreement.json"

    with pytest.raises(LiveDeploymentError):
        check_metadata_agreement(MetadataAgreementConfig(bundle_root=bundle, receipt_path=receipt))

    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["bundle_id"] == "neurips2025-phase06-proof"
    assert any(issue["field"] == "public_repo_url" for issue in data["issues"])
    assert data["secret_values_recorded"] is False


def test_restamp_release_metadata_sets_success_values_and_clears_blocked_values(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    write_json(bundle / "public-repo" / "public_repo_manifest.json", {"public_repo_url": "old"})

    result = restamp_release_metadata(
        bundle_root=bundle,
        public_repo_url="https://github.com/vasiloglou/neurips2025-data",
        public_site_url=None,
        cdn_base_url=R2_BASE_URL,
        zenodo_doi=None,
        zenodo_record_url=None,
        hf_dataset_url=None,
    )

    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    assert release["public_repo_url"] == "https://github.com/vasiloglou/neurips2025-data"
    assert release["public_site_url"] is None
    assert release["cdn_base_url"] == R2_BASE_URL
    assert release["zenodo_doi"] is None
    assert release["zenodo_record_url"] is None
    assert release["hf_dataset_url"] is None
    assert "RELEASE.json" in checksums
    assert "metadata/dataset_manifest.json" in checksums
    assert "public-repo/public_repo_manifest.json" in checksums
    assert "public_site_url" in result["updated_fields"]


def test_restamp_release_metadata_preserves_omitted_identifiers(tmp_path: Path):
    bundle = make_bundle(tmp_path)

    result = restamp_release_metadata(
        bundle_root=bundle,
        public_repo_url="https://github.com/vasiloglou/neurips2025-data-v2",
    )

    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    assert release["public_repo_url"] == "https://github.com/vasiloglou/neurips2025-data-v2"
    assert release["public_site_url"] == "https://neurips2025-data.pages.dev"
    assert release["cdn_base_url"] == R2_BASE_URL
    assert release["zenodo_doi"] == "10.5281/zenodo.1234567"
    assert release["hf_dataset_url"] == "https://huggingface.co/datasets/vasiloglou/neurips2025-data"
    assert result["updated_fields"] == ["public_repo_url"]


def test_metadata_agreement_checks_success_and_blocked_provider_receipts(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["public_site_url"] = None
    release["zenodo_doi"] = None
    release["zenodo_record_url"] = None
    release["hf_dataset_url"] = None
    write_json(bundle / "RELEASE.json", release)
    (bundle / "public-repo" / "README.md").write_text(
        f"# neurips2025-data\n\nRepo: {release['public_repo_url']}\nCDN: {R2_BASE_URL}\n",
        encoding="utf-8",
    )
    write_json(
        bundle / "metadata" / "croissant.json",
        {"url": release["public_repo_url"], "sameAs": [R2_BASE_URL]},
    )
    write_json(
        bundle / "metadata" / "datapackage.json",
        {"homepage": release["public_repo_url"], "cdn_base_url": R2_BASE_URL},
    )
    write_json(
        bundle / "metadata" / "dataset_manifest.json",
        {
            "public_repo_url": release["public_repo_url"],
            "public_site_url": None,
            "cdn_base_url": R2_BASE_URL,
            "zenodo_doi": None,
            "zenodo_record_url": None,
            "hf_dataset_url": None,
        },
    )
    write_json(bundle / "site" / "site_manifest.json", {"cdn_base_url": R2_BASE_URL, "public_site_url": None})
    write_json(
        bundle / "public-repo" / "public_repo_manifest.json",
        {"public_repo_url": release["public_repo_url"], "cdn_base_url": R2_BASE_URL},
    )
    receipts = write_success_receipts(tmp_path)
    write_json(
        tmp_path / "preflight.json",
        {
            "receipt_type": "conference_pipeline.public_deployment_preflight",
            "status": "succeeded",
            "r2": {"base_url": R2_BASE_URL},
        },
    )
    write_json(
        tmp_path / "github.json",
        {
            "status": "succeeded",
            "provider": "github",
            "public_url": release["public_repo_url"],
        },
    )
    for name, provider in (
        ("pages.json", "cloudflare_pages"),
        ("zenodo.json", "zenodo"),
        ("hf.json", "huggingface"),
    ):
        write_json(tmp_path / name, {"status": "operator_blocked", "provider": provider, "secret_values_recorded": False})

    receipt = tmp_path / "metadata-agreement.json"
    result = check_metadata_agreement(
        MetadataAgreementConfig(
            bundle_root=bundle,
            receipt_path=receipt,
            preflight_receipt=tmp_path / "preflight.json",
            github_receipt=tmp_path / "github.json",
            pages_receipt=tmp_path / "pages.json",
            zenodo_receipt=tmp_path / "zenodo.json",
            hf_receipt=tmp_path / "hf.json",
        )
    )

    assert result["status"] == "succeeded"
    assert "github.json" in result["checked_sources"]


def test_cloudflare_pages_refuses_non_site_source_under_public_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    make_pages_site(bundle)
    wrong_source = bundle / "public-repo"
    wrong_source.mkdir(exist_ok=True)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf_dummy_public_test_value")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")

    with pytest.raises(LiveDeploymentError, match="site/"):
        deploy_cloudflare_pages(pages_config(tmp_path, bundle, wrong_source), request_json=lambda *args, **kwargs: {})


def test_cloudflare_pages_mock_success_records_redacted_deployment_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    site = make_pages_site(bundle)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf_dummy_public_test_value")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    calls: list[tuple[str, str, dict]] = []

    def request_json(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return {
                "success": True,
                "result": {
                    "name": "neurips2025-data",
                    "subdomain": "neurips2025-data.pages.dev",
                },
            }
        return {
            "success": True,
            "result": {
                "id": "deployment-123",
                "url": "https://deployment-123.neurips2025-data.pages.dev",
            },
        }

    receipt = deploy_cloudflare_pages(pages_config(tmp_path, bundle, site), request_json=request_json)
    text = (tmp_path / "cloudflare-pages.json").read_text(encoding="utf-8")

    assert receipt["status"] == "succeeded"
    assert receipt["provider"] == "cloudflare_pages"
    assert receipt["project_name"] == "neurips2025-data"
    assert receipt["public_url"] == "https://neurips2025-data.pages.dev"
    assert receipt["deployment_url"] == "https://deployment-123.neurips2025-data.pages.dev"
    assert receipt["deployment_id"] == "deployment-123"
    assert receipt["source_dir"] == site.as_posix()
    assert receipt["live"] is True
    assert receipt["secret_values_recorded"] is False
    assert "cf_dummy_public_test_value" not in text
    assert calls[0][0] == "GET"
    assert calls[-1][0] == "POST"
    assert "data" in calls[-1][2]
    assert "manifest" in calls[-1][2]["data"]
    assert "json" not in calls[-1][2]


def test_cloudflare_pages_missing_env_writes_operator_blocked_without_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    site = make_pages_site(bundle)
    for name in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_PAGES_PROJECT"):
        monkeypatch.delenv(name, raising=False)

    receipt = deploy_cloudflare_pages(
        CloudflarePagesConfig(
            bundle_root=bundle,
            site_dir=site,
            receipt_path=tmp_path / "cloudflare-pages.json",
            live=True,
        )
    )

    assert receipt["status"] == "operator_blocked"
    assert receipt["provider"] == "cloudflare_pages"
    assert receipt["deployment_url"] is None
    assert receipt["secret_values_recorded"] is False
    assert sorted(receipt["missing_env_vars"]) == [
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_PAGES_PROJECT",
    ]


def test_cloudflare_pages_helper_does_not_require_wrangler_or_package_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    site = make_pages_site(bundle)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf_dummy_public_test_value")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("CLOUDFLARE_PAGES_PROJECT", "neurips2025-data")

    def request_json(method: str, url: str, **kwargs):
        if "wrangler" in json.dumps(kwargs).lower():
            raise AssertionError("wrangler should not appear in Pages API payloads")
        if method == "GET":
            return {"success": True, "result": {"name": "neurips2025-data"}}
        return {"success": True, "result": {"id": "deployment-123", "url": "https://pages.example.test"}}

    deploy_cloudflare_pages(
        CloudflarePagesConfig(
            bundle_root=bundle,
            site_dir=site,
            receipt_path=tmp_path / "cloudflare-pages.json",
            live=True,
        ),
        request_json=request_json,
    )

    assert not (REPO_ROOT / "package.json").exists()
    assert not (REPO_ROOT / "package-lock.json").exists()


def test_cloudflare_pages_live_uses_wrangler_deploy_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    site = make_pages_site(bundle)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf_dummy_public_test_value")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("CLOUDFLARE_PAGES_PROJECT", "neurips2025-data")
    calls: list[list[str]] = []

    def runner(cmd, capture_output=True, text=True, check=False, env=None):
        calls.append(list(cmd))
        assert env["CLOUDFLARE_API_TOKEN"] == "cf_dummy_public_test_value"
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Deployment complete: https://neurips2025-data.pages.dev\n",
            stderr="",
        )

    receipt = deploy_cloudflare_pages(
        CloudflarePagesConfig(
            bundle_root=bundle,
            site_dir=site,
            receipt_path=tmp_path / "cloudflare-pages.json",
            live=True,
        ),
        runner=runner,
    )

    assert calls == [
        [
            "npx",
            "--yes",
            "wrangler",
            "pages",
            "deploy",
            str(site.resolve()),
            "--project-name",
            "neurips2025-data",
            "--branch",
            "main",
        ]
    ]
    assert receipt["status"] == "succeeded"
    assert receipt["deploy_method"] == "wrangler_pages_deploy"
    assert receipt["public_url"] == "https://neurips2025-data.pages.dev"
    assert receipt["deployment_url"] == "https://neurips2025-data.pages.dev"
