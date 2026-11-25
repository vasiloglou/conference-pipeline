from __future__ import annotations
import pytest

import json
from pathlib import Path

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    build_material_discovery_receipt,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    PublicMaterialResponse,
    run_fixed_r6_material_acquisition,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"


class FakeTransport:
    def __init__(self, responses: dict[str, PublicMaterialResponse]) -> None:
        self.responses = responses
        self.requested_urls: list[str] = []

    def open(self, url: str) -> PublicMaterialResponse:
        self.requested_urls.append(url)
        return self.responses[url]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    for name in ("config.yaml", "r6_material_source_policy.yaml"):
        (workspace / name).write_bytes((CVPR_WORKSPACE / name).read_bytes())
    return workspace


def _discovery(workspace: Path) -> Path:
    path = workspace / R6_MATERIAL_DISCOVERY_RELATIVE_PATH
    receipt = build_material_discovery_receipt(
        workspace=workspace,
        observations=[
            {
                "record_id": "paper-001",
                "event_id": "cvpr2026:fixture",
                "source_page_url": "https://cvpr.thecvf.com/virtual/2026/fixture",
                "observed_at": "2026-07-17T00:00:00Z",
                "direct_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf",
                "link_kind": "direct_public",
                "content_class": "paper_pdf",
                "mime_type": "application/pdf",
                "byte_count": 9,
                "redirect_hops": [],
            }
        ],
        output_path=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_fixed_r6_acquisition_writes_only_declared_local_payloads_and_receipt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    discovery = _discovery(workspace)
    url = "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"
    transport = FakeTransport(
        {url: PublicMaterialResponse(200, {"content-type": "application/pdf"}, (b"%PDF-1.4\n",))}
    )

    receipt = run_fixed_r6_material_acquisition(
        workspace=workspace,
        transport=transport,
        artifact_contract=R6ArtifactContract.for_test_fixture(workspace),
    )

    expected_path = (
        workspace / "storage/transcription-ocr/cvpr2026/r6/material_acquisition_receipt.json"
    )
    assert expected_path.is_file()
    assert receipt["discovery_receipt_sha256"]
    assert receipt["payloads"][0]["local_payload_path"]
    assert receipt["payloads"][0]["policy_digest"] == receipt["policy_digest"]
    assert receipt["terminal_records"] == []
    assert transport.requested_urls == [url]
    assert discovery.is_file()


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_fixed_r6_acquisition_never_uses_url_flags_or_storage_scans(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _discovery(workspace)
    ignored = workspace / "storage/transcription-ocr/cvpr2026/r6/material-payloads/unlisted.pdf"
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_bytes(b"%PDF-1.4\n")
    transport = FakeTransport({})

    receipt = run_fixed_r6_material_acquisition(
        workspace=workspace,
        transport=transport,
        artifact_contract=R6ArtifactContract.for_test_fixture(workspace),
    )

    assert receipt["payloads"] == []
    assert receipt["terminal_records"][0]["terminal_reason"] == "transport_failure"
    assert transport.requested_urls == [
        "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"
    ]
