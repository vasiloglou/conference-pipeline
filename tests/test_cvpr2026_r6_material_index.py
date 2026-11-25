from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from conference_pipeline.acquisition.cvpr_r6_material_index import (
    R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    MaterialPolicyError,
    build_material_discovery_receipt,
    load_r6_material_source_policy,
    validate_material_discovery_receipt,
    write_material_discovery_receipt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    for name in ("config.yaml", "r6_material_source_policy.yaml"):
        (workspace / name).write_bytes((CVPR_WORKSPACE / name).read_bytes())
    return workspace


def _observation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "record_id": "event-001:paper",
        "event_id": "cvpr2026:workshop-001",
        "source_page_url": "https://cvpr.thecvf.com/virtual/2026/workshop/001",
        "observed_at": "2026-07-17T00:00:00Z",
        "direct_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/example.pdf",
        "link_kind": "direct_public",
        "content_class": "paper_pdf",
        "mime_type": "application/pdf",
        "byte_count": 1024,
        "redirect_hops": [],
    }
    payload.update(overrides)
    return payload


def _receipt(workspace: Path, *observations: dict[str, object]) -> dict[str, object]:
    return build_material_discovery_receipt(
        workspace=workspace,
        observations=list(observations),
        output_path=workspace / R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    )


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_r6_material_policy_is_workspace_selected_finite_and_canonically_digested(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    policy = load_r6_material_source_policy(workspace)

    assert policy.policy_path == workspace / "r6_material_source_policy.yaml"
    assert policy.policy_digest
    assert policy.source_page_hosts == ("cvpr.thecvf.com",)
    assert policy.direct_target_hosts == ("cvpr.thecvf.com", "openaccess.thecvf.com")
    assert policy.max_redirects == 2
    assert policy.max_bytes == 104857600
    assert set(policy.content_class_mime_types) == {
        "paper_pdf",
        "deck_pdf",
        "presentation_file",
        "slide_image",
    }

    policy_payload = yaml.safe_load(policy.policy_path.read_text(encoding="utf-8"))
    policy_payload["direct_target_hosts"].append("new.example")
    policy.policy_path.write_text(yaml.safe_dump(policy_payload, sort_keys=False), encoding="utf-8")
    changed = load_r6_material_source_policy(workspace)
    assert changed.policy_digest != policy.policy_digest


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
@pytest.mark.parametrize(
    ("content_class", "mime_type", "target_suffix"),
    [
        ("paper_pdf", "application/pdf", "paper.pdf"),
        ("deck_pdf", "application/pdf", "deck.pdf"),
        (
            "presentation_file",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "slides.pptx",
        ),
        ("slide_image", "image/png", "slide.png"),
    ],
)
def test_r6_material_index_admits_only_declared_direct_public_content_classes(
    tmp_path: Path, content_class: str, mime_type: str, target_suffix: str
) -> None:
    workspace = _workspace(tmp_path)
    receipt = _receipt(
        workspace,
        _observation(
            record_id=f"event-001:{content_class}",
            content_class=content_class,
            mime_type=mime_type,
            direct_target_url=f"https://openaccess.thecvf.com/CVPR2026/{target_suffix}",
        ),
    )

    record = receipt["records"][0]
    assert record["content_class"] == content_class
    assert record["host_class"] == "direct_public"
    assert record["eligibility"] == "eligible_direct_public"
    assert record["terminal_reason"] is None
    assert record["payload_sha256"] is None
    assert record["local_path"] is None
    assert record["source_page_provenance"] == record["source_page_url"]
    assert receipt["output_path"] == R6_MATERIAL_DISCOVERY_RELATIVE_PATH.as_posix()
    assert receipt["policy_digest"] == record["policy_digest"]
    validate_material_discovery_receipt(workspace=workspace, receipt=receipt)


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
@pytest.mark.parametrize(
    ("link_kind", "target", "content_class", "mime_type", "reason"),
    [
        (
            "vendor_embed",
            "https://slideslive.com/123",
            "deck_pdf",
            "application/pdf",
            "vendor_embed",
        ),
        (
            "static_image_page",
            "https://cvpr.thecvf.com/virtual/2026/image",
            "slide_image",
            "image/png",
            "static_image_page",
        ),
        (
            "direct_public",
            "https://new-host.example/file.pdf",
            "paper_pdf",
            "application/pdf",
            "target_host_not_allowed",
        ),
        (
            "authenticated_only",
            "https://cvpr.thecvf.com/virtual/2026/private.pdf",
            "paper_pdf",
            "application/pdf",
            "authenticated_only",
        ),
        ("absent", None, "paper_pdf", "application/pdf", "no_material_link"),
    ],
)
def test_r6_material_index_keeps_nonpublic_or_unknown_links_classification_only(
    tmp_path: Path,
    link_kind: str,
    target: str | None,
    content_class: str,
    mime_type: str,
    reason: str,
) -> None:
    workspace = _workspace(tmp_path)
    receipt = _receipt(
        workspace,
        _observation(
            link_kind=link_kind,
            direct_target_url=target,
            content_class=content_class,
            mime_type=mime_type,
        ),
    )

    record = receipt["records"][0]
    assert record["eligibility"] == "classification_only"
    assert record["terminal_reason"] == reason
    assert record["payload_sha256"] is None
    assert record["local_path"] is None


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"direct_target_url": "not a URL"}, "malformed_target_url"),
        (
            {"direct_target_url": "https://openaccess.thecvf.com/CVPR2026/paper.pdf?token=no"},
            "malformed_target_url",
        ),
        (
            {"direct_target_url": "https://openaccess.thecvf.com:invalid/file.pdf"},
            "malformed_target_url",
        ),
        ({"redirect_hops": ["https://new-host.example/hop"]}, "redirect_host_not_allowed"),
        (
            {
                "redirect_hops": [
                    "https://cvpr.thecvf.com/1",
                    "https://cvpr.thecvf.com/2",
                    "https://cvpr.thecvf.com/3",
                ]
            },
            "redirect_limit_exceeded",
        ),
        ({"mime_type": "image/png"}, "mime_type_not_allowed"),
        ({"byte_count": 104857601}, "byte_count_exceeds_policy"),
    ],
)
def test_r6_material_index_records_policy_failures_as_complete_terminal_rows(
    tmp_path: Path, overrides: dict[str, object], reason: str
) -> None:
    workspace = _workspace(tmp_path)
    receipt = _receipt(workspace, _observation(**overrides))

    record = receipt["records"][0]
    assert record["eligibility"] == "classification_only"
    assert record["terminal_reason"] == reason
    assert receipt["counts"] == {
        "declared": 1,
        "eligible_direct_public": 0,
        "classification_only": 1,
    }


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
@pytest.mark.parametrize(
    "overrides",
    [
        {"source_page_url": "https://spoofed.example/event"},
        {"directory_glob": "storage/**/*.pdf"},
        {"r5_caption": "caption sidecar"},
        {"inferred_from": "video"},
    ],
)
def test_r6_material_index_rejects_undeclared_discovery_authority(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(MaterialPolicyError):
        _receipt(workspace, _observation(**overrides))


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_r6_material_index_rejects_duplicate_targets_secrets_and_invalid_receipt_bindings(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    first = _observation(record_id="event-001:paper-a")
    duplicate = _observation(record_id="event-001:paper-b")
    with pytest.raises(MaterialPolicyError, match="duplicate"):
        _receipt(workspace, first, duplicate)

    with pytest.raises(MaterialPolicyError, match="secret"):
        _receipt(workspace, _observation(authorization="not-permitted"))

    receipt = _receipt(workspace, first)
    tampered = copy.deepcopy(receipt)
    tampered["policy_digest"] = "0" * 64
    with pytest.raises(MaterialPolicyError, match="policy digest"):
        validate_material_discovery_receipt(workspace=workspace, receipt=tampered)


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_r6_material_index_writer_is_explicit_atomic_and_keeps_history_as_context(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    receipt = _receipt(workspace, _observation())

    assert receipt["historical_context"] == {
        "workshop_pdf_receipt_backed": 517,
        "workshop_pdf_missing": 60,
        "main_cvpr_pdf_existing": 0,
        "presentation_or_deck_existing": 0,
    }
    assert not any("historical" in str(record) for record in receipt["records"])

    output = write_material_discovery_receipt(
        workspace=workspace,
        receipt=receipt,
        output_path=workspace / R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    )
    assert output == workspace / R6_MATERIAL_DISCOVERY_RELATIVE_PATH
    assert json.loads(output.read_text(encoding="utf-8"))["records"] == receipt["records"]

    with pytest.raises(MaterialPolicyError, match="declared r6 material-discovery"):
        write_material_discovery_receipt(
            workspace=workspace,
            receipt=receipt,
            output_path=workspace / "storage/transcription-ocr/cvpr2026/r6/../escape.json",
        )
