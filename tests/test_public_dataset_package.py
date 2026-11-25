from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.publishing.dataset_package import (
    DatasetPackageConfig,
    DatasetPackageError,
    collect_zenodo_upload_files,
    deposit_zenodo,
    package_public_dataset,
    push_hf_dataset,
    zenodo_metadata_payload,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path, *, conference: str = "neurips2025") -> Path:
    root = tmp_path / "conferences" / conference / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(
        root / "RELEASE.json",
        {
            "bundle_id": "bundle-test",
            "conference": conference,
            "source_sha": "abcdef",
            "included_families": ["presentation_markdown"],
            "family_licenses": {"presentation_markdown": "MIT"},
        },
    )
    (root / "decks").mkdir()
    (root / "decks" / "a.md").write_text("Alpha public text\n", encoding="utf-8")
    # Real digests: the HF push re-verifies every file against its listing (Phase 7.1 r1 F1).
    import hashlib

    write_json(root / "CHECKSUMS.json", {"decks/a.md": {"sha256": hashlib.sha256(b"Alpha public text\n").hexdigest()}})
    write_json(root / "index" / "index_manifest.json", {"chunk_count": 1})
    return root


def test_dataset_metadata_records_bundle_resources_without_private_paths(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    metadata_dir = tmp_path / "metadata"
    package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=metadata_dir))

    assert (metadata_dir / "croissant.json").exists()
    assert (metadata_dir / "datapackage.json").exists()
    assert (metadata_dir / "CITATION.cff").exists()
    assert (metadata_dir / "dataset_manifest.json").exists()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in metadata_dir.rglob("*") if path.is_file())
    for marker in ("/Users/", "file://", "Google Drive", ".env"):
        assert marker not in combined
    assert "bundle-test" in (metadata_dir / "CITATION.cff").read_text(encoding="utf-8")
    assert "neurips2025" in (metadata_dir / "CITATION.cff").read_text(encoding="utf-8")


def test_dataset_metadata_echoes_public_identifiers_and_blocked_nulls(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release.update(
        {
            "public_repo_url": "https://github.com/vasiloglou/neurips2025-data",
            "public_site_url": None,
            "cdn_base_url": "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/conferences/neurips2025",
            "zenodo_doi": None,
            "zenodo_record_url": None,
            "hf_dataset_url": None,
        }
    )
    write_json(bundle / "RELEASE.json", release)

    metadata_dir = package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata"))

    croissant = json.loads((metadata_dir / "croissant.json").read_text(encoding="utf-8"))
    datapackage = json.loads((metadata_dir / "datapackage.json").read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert croissant["url"] == release["public_repo_url"]
    assert release["cdn_base_url"] in croissant["sameAs"]
    assert datapackage["homepage"] == release["public_repo_url"]
    assert datapackage["public_identifiers"]["public_site_url"] is None
    assert manifest["hf_dataset_url"] is None


def test_zenodo_metadata_payload_uses_manifest_conference_name(tmp_path: Path):
    bundle = make_bundle(tmp_path, conference="iclr2026")
    metadata_dir = package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata"))

    payload = zenodo_metadata_payload(metadata_dir)

    assert payload["metadata"]["title"] == "iclr2026 public data release bundle-test"
    assert "iclr2026 conference pipeline" in payload["metadata"]["description"]
    assert "NeurIPS 2025" not in json.dumps(payload)


def test_zenodo_dry_run_receipt_redacts_tokens(tmp_path: Path, monkeypatch):
    bundle = make_bundle(tmp_path)
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata")
    )
    receipt = tmp_path / "zenodo.json"
    monkeypatch.setenv("ZENODO_TOKEN", "fake-secret-token")
    deposit_zenodo(
        metadata_dir=metadata_dir,
        receipt_path=receipt,
        sandbox=True,
        live=False,
        dry_run=True,
        ack_live_doi=False,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.zenodo_deposit"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["dry_run"] is True
    assert data["live"] is False
    assert data["target"] == "sandbox.zenodo.org"
    assert data["secret_values_recorded"] is False
    assert "fake-secret-token" not in receipt.read_text(encoding="utf-8")


def test_zenodo_live_requires_ack_before_network(tmp_path: Path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    write_json(metadata_dir / "croissant.json", {})
    receipt = tmp_path / "receipt.json"
    network_calls: list[str] = []
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_secret_value")

    def fail_network(*args, **kwargs):
        network_calls.append("network")
        raise AssertionError("network must not be reached before ack")

    deposit_zenodo(
        metadata_dir=metadata_dir,
        receipt_path=receipt,
        sandbox=False,
        live=True,
        dry_run=False,
        ack_live_doi=False,
        request_json=fail_network,
        upload_bytes=fail_network,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert network_calls == []
    assert data["receipt_type"] == "conference_pipeline.zenodo_deposit"
    assert data["status"] == "operator_blocked"
    assert data["exit_code"] == 2
    assert data["live"] is True
    assert data["target"] == "zenodo.org"
    assert data["provider"] == "zenodo"
    assert data["blocker_code"] == "missing_live_ack"
    assert data["missing_approvals"] == ["ack_live_doi"]
    assert data["secret_values_recorded"] is False
    assert "zenodo_secret_value" not in receipt.read_text(encoding="utf-8")


def test_zenodo_live_requires_token_before_network(tmp_path: Path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    write_json(metadata_dir / "croissant.json", {})
    receipt = tmp_path / "receipt.json"
    network_calls: list[str] = []
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)

    def fail_network(*args, **kwargs):
        network_calls.append("network")
        raise AssertionError("network must not be reached before token")

    deposit_zenodo(
        metadata_dir=metadata_dir,
        receipt_path=receipt,
        sandbox=False,
        live=True,
        dry_run=False,
        ack_live_doi=True,
        request_json=fail_network,
        upload_bytes=fail_network,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert network_calls == []
    assert data["status"] == "operator_blocked"
    assert data["blocker_code"] == "missing_zenodo_token"
    assert data["missing_env_vars"] == ["ZENODO_TOKEN"]
    assert data["secret_values_recorded"] is False


def test_zenodo_live_success_records_public_identifiers_and_redacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    write_json(bundle / "audio" / "talk.mp3.cdn.json", {"cdn_url": "https://cdn.example/talk.mp3"})
    # The package receipt goes outside the bundle: the HF push uploads only listed files
    # and refuses an unlisted one, such as a receipt left in metadata/ (Phase 7.1 r1 F1).
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "package-receipt.json")
    )
    receipt = tmp_path / "zenodo-live.json"
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_live_secret")
    uploaded: list[str] = []

    def request_json(method: str, url: str, **kwargs):
        if method == "POST" and url.endswith("/api/deposit/depositions"):
            return {"id": 12345, "links": {"bucket": "https://zenodo.org/api/files/bucket-id"}}
        if method == "PUT" and url.endswith("/api/deposit/depositions/12345"):
            return {"id": 12345}
        if method == "POST" and url.endswith("/api/deposit/depositions/12345/actions/publish"):
            return {
                "id": 12345,
                "doi": "10.5281/zenodo.12345",
                "links": {"html": "https://zenodo.org/records/12345"},
            }
        raise AssertionError(f"unexpected Zenodo request {method} {url}")

    def upload_bytes(url: str, filename: str, data: bytes, **kwargs):
        assert url.startswith("https://zenodo.org/api/files/bucket-id/")
        assert "/" not in filename
        assert "%2F" not in url
        uploaded.append(filename)
        return {"filename": filename, "size": len(data)}

    deposit_zenodo(
        metadata_dir=metadata_dir,
        receipt_path=receipt,
        sandbox=False,
        live=True,
        dry_run=False,
        ack_live_doi=True,
        bundle_root=bundle,
        request_json=request_json,
        upload_bytes=upload_bytes,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.zenodo_deposit"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["live"] is True
    assert data["target"] == "zenodo.org"
    assert data["provider"] == "zenodo"
    assert data["deposition_id"] == "12345"
    assert data["doi"] == "10.5281/zenodo.12345"
    assert data["record_url"] == "https://zenodo.org/records/12345"
    assert data["uploaded_file_count"] == len(uploaded)
    assert data["uploaded_file_count"] >= 4
    assert "metadata__CITATION.cff" in uploaded
    assert "index__index_manifest.json" in uploaded
    assert "cdn-sidecars.zip" in uploaded
    assert data["sidecar_archive"]["source_file_count"] == 1
    assert data["token_env_var"] == "ZENODO_TOKEN"
    assert data["secret_values_recorded"] is False
    assert "zenodo_live_secret" not in receipt.read_text(encoding="utf-8")


def test_zenodo_upload_filenames_are_flat_provider_names(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    write_json(bundle / "audio" / "talk.mp3.cdn.json", {"cdn_url": "https://cdn.example/talk.mp3"})
    # The package receipt goes outside the bundle: the HF push uploads only listed files
    # and refuses an unlisted one, such as a receipt left in metadata/ (Phase 7.1 r1 F1).
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "package-receipt.json")
    )

    files = collect_zenodo_upload_files(bundle, metadata_dir)
    filenames = {item.filename for item in files}

    assert "metadata__croissant.json" in filenames
    assert "index__index_manifest.json" in filenames
    assert "audio__talk.mp3.cdn.json" not in filenames
    assert all("/" not in filename for filename in filenames)


def test_zenodo_api_failure_writes_operator_blocked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    # The package receipt goes outside the bundle: the HF push uploads only listed files
    # and refuses an unlisted one, such as a receipt left in metadata/ (Phase 7.1 r1 F1).
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "package-receipt.json")
    )
    receipt = tmp_path / "zenodo-failed.json"
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_live_secret")

    def request_json(*args, **kwargs):
        raise DatasetPackageError("Zenodo API HTTP 401: Unauthorized")

    deposit_zenodo(
        metadata_dir=metadata_dir,
        receipt_path=receipt,
        sandbox=False,
        live=True,
        dry_run=False,
        ack_live_doi=True,
        bundle_root=bundle,
        request_json=request_json,
        upload_bytes=lambda *args, **kwargs: {},
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["status"] == "operator_blocked"
    assert data["provider"] == "zenodo"
    assert data["blocker_code"] == "zenodo_api_error"
    assert data["doi"] is None
    assert data["record_url"] is None
    assert data["secret_values_recorded"] is False
    assert "zenodo_live_secret" not in receipt.read_text(encoding="utf-8")


def test_hf_dry_run_receipt_redacts_tokens(tmp_path: Path, monkeypatch):
    bundle = make_bundle(tmp_path)
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata")
    )
    receipt = tmp_path / "hf.json"
    monkeypatch.setenv("HF_TOKEN", "hf_fake_secret")
    push_hf_dataset(
        metadata_dir=metadata_dir,
        repo_id="org/neurips2025-data",
        receipt_path=receipt,
        dry_run=True,
        live=False,
        ack_live_push=False,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.hf_dataset_push"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["dry_run"] is True
    assert data["live"] is False
    assert data["target"] == "org/neurips2025-data"
    assert data["secret_values_recorded"] is False
    assert "hf_fake_secret" not in receipt.read_text(encoding="utf-8")


def test_hf_live_requires_ack_before_network(tmp_path: Path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    write_json(metadata_dir / "datapackage.json", {})
    receipt = tmp_path / "hf.json"
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    network_calls: list[str] = []

    def fail_api(*args, **kwargs):
        network_calls.append("network")
        raise AssertionError("network must not be reached before ack")

    push_hf_dataset(
        metadata_dir=metadata_dir,
        repo_id="org/neurips2025-data",
        receipt_path=receipt,
        dry_run=False,
        live=True,
        ack_live_push=False,
        hf_api_factory=fail_api,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert network_calls == []
    assert data["receipt_type"] == "conference_pipeline.hf_dataset_push"
    assert data["status"] == "operator_blocked"
    assert data["exit_code"] == 2
    assert data["live"] is True
    assert data["target"] == "org/neurips2025-data"
    assert data["provider"] == "huggingface"
    assert data["blocker_code"] == "missing_live_ack"
    assert data["missing_approvals"] == ["ack_live_push"]
    assert data["secret_values_recorded"] is False
    assert "hf_secret_value" not in receipt.read_text(encoding="utf-8")


def test_hf_live_requires_token_before_network(tmp_path: Path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    write_json(metadata_dir / "datapackage.json", {})
    receipt = tmp_path / "hf.json"
    network_calls: list[str] = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    def fail_api(*args, **kwargs):
        network_calls.append("network")
        raise AssertionError("network must not be reached before token")

    push_hf_dataset(
        metadata_dir=metadata_dir,
        repo_id="org/neurips2025-data",
        receipt_path=receipt,
        dry_run=False,
        live=True,
        ack_live_push=True,
        hf_api_factory=fail_api,
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert network_calls == []
    assert data["status"] == "operator_blocked"
    assert data["blocker_code"] == "missing_hf_token"
    assert data["missing_env_vars"] == ["HF_TOKEN", "HUGGINGFACE_API_KEY"]
    assert data["secret_values_recorded"] is False


def test_hf_live_accepts_huggingface_api_key_alias_and_redacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    # The package receipt goes outside the bundle: the HF push uploads only listed files
    # and refuses an unlisted one, such as a receipt left in metadata/ (Phase 7.1 r1 F1).
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "package-receipt.json")
    )
    receipt = tmp_path / "hf-live.json"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_alias_secret")

    class FakeHfApi:
        def __init__(self, token: str):
            assert token == "hf_alias_secret"

        def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool):
            assert repo_id == "vasiloglou/neurips2025-data"
            assert repo_type == "dataset"
            assert exist_ok is True
            return "https://huggingface.co/datasets/vasiloglou/neurips2025-data"

        # v6.0 Phase 7.1: a metadata directory inside a bundle publishes the bundle's HF
        # tree with create_commit, then tags it (test_hf_release_tree.py covers the tree).
        def list_repo_files(self, repo_id: str, *, repo_type: str):
            return []

        def list_repo_refs(self, repo_id: str, *, repo_type: str):
            return type("Refs", (), {"tags": []})()

        def create_commit(self, *, repo_id: str, repo_type: str, operations, commit_message: str):
            assert repo_id == "vasiloglou/neurips2025-data"
            assert repo_type == "dataset"
            return type(
                "CommitInfo",
                (),
                {
                    "oid": "hfabc123",
                    "commit_url": "https://huggingface.co/datasets/vasiloglou/neurips2025-data/commit/hfabc123",
                },
            )()

        def create_tag(self, repo_id: str, *, tag: str, revision: str, repo_type: str):
            assert revision == "hfabc123"

    push_hf_dataset(
        metadata_dir=metadata_dir,
        repo_id="vasiloglou/neurips2025-data",
        receipt_path=receipt,
        dry_run=False,
        live=True,
        ack_live_push=True,
        hf_api_factory=FakeHfApi,
        secret_scanner=lambda paths: [],
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.hf_dataset_push"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["provider"] == "huggingface"
    assert data["dataset_url"] == "https://huggingface.co/datasets/vasiloglou/neurips2025-data"
    assert data["commit_sha"] == "hfabc123"
    assert data["commit_url"].endswith("/commit/hfabc123")
    assert data["token_env_var"] == "HUGGINGFACE_API_KEY"
    assert data["secret_values_recorded"] is False
    assert "hf_alias_secret" not in receipt.read_text(encoding="utf-8")


def test_hf_namespace_scope_failure_writes_operator_blocked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = make_bundle(tmp_path)
    # The package receipt goes outside the bundle: the HF push uploads only listed files
    # and refuses an unlisted one, such as a receipt left in metadata/ (Phase 7.1 r1 F1).
    metadata_dir = package_public_dataset(
        DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "package-receipt.json")
    )
    receipt = tmp_path / "hf-failed.json"
    monkeypatch.setenv("HF_TOKEN", "hf_live_secret")
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    class FailingHfApi:
        def __init__(self, token: str):
            assert token == "hf_live_secret"

        def create_repo(self, *args, **kwargs):
            raise RuntimeError(
                "403 forbidden: namespace scope missing at "
                "https://huggingface.co/api/repos/create"
            )

    push_hf_dataset(
        metadata_dir=metadata_dir,
        repo_id="vasiloglou/neurips2025-data",
        receipt_path=receipt,
        dry_run=False,
        live=True,
        ack_live_push=True,
        hf_api_factory=FailingHfApi,
        secret_scanner=lambda paths: [],
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["status"] == "operator_blocked"
    assert data["provider"] == "huggingface"
    assert data["blocker_code"] == "hf_namespace_scope_missing"
    assert data["dataset_url"] is None
    assert data["provider_error"] == "403 forbidden: namespace scope missing at <provider-url>"
    assert data["secret_values_recorded"] is False
    assert "hf_live_secret" not in receipt.read_text(encoding="utf-8")
    assert "https://huggingface.co/api" not in receipt.read_text(encoding="utf-8")


def _mixed_licenses(bundle: Path) -> dict:
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["included_families"] = ["presentation_markdown", "reader_catalog", "transcripts"]
    release["family_licenses"] = {
        "presentation_markdown": "MIT",
        "reader_catalog": "CC-BY-4.0",
        "transcripts": "CC-BY-NC-ND-4.0",
    }
    write_json(bundle / "RELEASE.json", release)
    return release


def test_f9_metadata_declares_each_family_license_from_the_release(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    release = _mixed_licenses(bundle)
    metadata_dir = package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata"))

    croissant = json.loads((metadata_dir / "croissant.json").read_text(encoding="utf-8"))
    datapackage = json.loads((metadata_dir / "datapackage.json").read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert croissant["license"] == [
        "https://spdx.org/licenses/CC-BY-4.0.html",
        "https://spdx.org/licenses/CC-BY-NC-ND-4.0.html",
        "https://spdx.org/licenses/MIT.html",
    ]
    assert [(row["name"], row["families"]) for row in datapackage["licenses"]] == [
        ("CC-BY-4.0", ["reader_catalog"]),
        ("CC-BY-NC-ND-4.0", ["transcripts"]),
        ("MIT", ["presentation_markdown"]),
    ]
    assert manifest["family_licenses"] == release["family_licenses"]
    assert "CC-BY-SA-4.0" not in json.dumps([croissant, datapackage, manifest])


@pytest.mark.parametrize(
    "change",
    [
        lambda release: release.pop("family_licenses"),
        lambda release: release.update(family_licenses={}),
        lambda release: release.update(family_licenses={"presentation_markdown": ""}),
        lambda release: release.update(family_licenses={"other_family": "MIT"}),
    ],
    ids=["absent", "empty", "blank", "wrong-families"],
)
def test_f9_a_release_that_does_not_name_its_licenses_is_refused(tmp_path: Path, change):
    bundle = make_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    change(release)
    write_json(bundle / "RELEASE.json", release)
    with pytest.raises(DatasetPackageError, match="family_licenses"):
        package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata"))


def test_f9_zenodo_takes_the_single_license_and_refuses_a_mixed_release(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    metadata_dir = package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "metadata"))
    assert zenodo_metadata_payload(metadata_dir)["metadata"]["license"] == "mit"

    _mixed_licenses(bundle)
    mixed_dir = package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=tmp_path / "mixed"))
    with pytest.raises(DatasetPackageError, match="zenodo_license_mixed"):
        zenodo_metadata_payload(mixed_dir)
