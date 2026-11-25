from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conference_pipeline.acquisition.cvpr_r6_material_index import (
    R6MaterialSourcePolicy,
    load_r6_material_source_policy,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    PublicMaterialResponse,
    acquire_declared_public_material,
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


def _policy(workspace: Path) -> R6MaterialSourcePolicy:
    return load_r6_material_source_policy(workspace)


def _record(policy: R6MaterialSourcePolicy, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "record_id": "paper-001",
        "event_id": "cvpr2026:fixture",
        "source_page_url": "https://cvpr.thecvf.com/virtual/2026/fixture",
        "source_page_provenance": "https://cvpr.thecvf.com/virtual/2026/fixture",
        "direct_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf",
        "content_class": "paper_pdf",
        "mime_type": "application/pdf",
        "byte_count": 9,
        "eligibility": "eligible_direct_public",
        "terminal_reason": None,
        "policy_digest": policy.policy_digest,
    }
    record.update(overrides)
    return record


def _response(
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    chunks: tuple[bytes, ...] = (b"%PDF-1.4\n",),
) -> PublicMaterialResponse:
    return PublicMaterialResponse(
        status_code=status_code,
        headers=headers or {"content-type": "application/pdf"},
        body=chunks,
    )


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
@pytest.mark.parametrize(
    ("content_class", "mime_type", "payload", "suffix"),
    [
        ("paper_pdf", "application/pdf", b"%PDF-1.4\n", ".pdf"),
        ("deck_pdf", "application/pdf", b"%PDF-1.4\n", ".pdf"),
        (
            "presentation_file",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            b"PK\x03\x04pptx",
            ".pptx",
        ),
        ("slide_image", "image/png", b"\x89PNG\r\n\x1a\nslide", ".png"),
    ],
)
def test_acquires_only_policy_validated_declared_materials_atomically(
    tmp_path: Path, content_class: str, mime_type: str, payload: bytes, suffix: str
) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    url = "https://openaccess.thecvf.com/CVPR2026/material"
    transport = FakeTransport(
        {url: _response(headers={"content-type": mime_type}, chunks=(payload,))}
    )

    result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=_record(
            policy,
            content_class=content_class,
            mime_type=mime_type,
            direct_target_url=url,
            byte_count=len(payload),
        ),
        transport=transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )

    assert result["acquisition_state"] == "acquired"
    assert result["policy_digest"] == policy.policy_digest
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["byte_count"] == len(payload)
    assert result["local_payload_path"].endswith(suffix)
    assert (workspace / str(result["local_payload_path"])).read_bytes() == payload
    assert transport.requested_urls == [url]


@pytest.mark.parametrize(
    ("record_overrides", "responses", "reason", "requests"),
    [
        (
            {"source_page_url": "https://foreign.example/source"},
            {},
            "source_page_host_not_allowed",
            [],
        ),
        (
            {"direct_target_url": "https://foreign.example/paper.pdf"},
            {},
            "initial_target_host_not_allowed",
            [],
        ),
        (
            {},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    status_code=302, headers={"location": "https://foreign.example/paper.pdf"}
                )
            },
            "redirect_host_not_allowed",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
        (
            {},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    status_code=302, headers={"location": "http://openaccess.thecvf.com/paper.pdf"}
                )
            },
            "redirect_not_public_https",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
        (
            {},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    status_code=302, headers={"location": "https://openaccess.thecvf.com/login"}
                )
            },
            "redirect_to_authentication",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
        (
            {},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    headers={"content-type": "image/png"}
                )
            },
            "mime_type_not_allowed",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
        (
            {},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    chunks=(b"not-a-pdf",)
                )
            },
            "content_signature_mismatch",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
        (
            {"byte_count": 4},
            {
                "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf": _response(
                    chunks=(b"%PDF", b"-1.4\n")
                )
            },
            "byte_count_mismatch",
            ["https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"],
        ),
    ],
)
@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_policy_failures_are_terminal_without_payload_declarations(
    tmp_path: Path,
    record_overrides: dict[str, object],
    responses: dict[str, PublicMaterialResponse],
    reason: str,
    requests: list[str],
) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    transport = FakeTransport(responses)

    result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=_record(policy, **record_overrides),
        transport=transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )

    assert result["acquisition_state"] == "terminal"
    assert result["terminal_reason"] == reason
    assert "local_payload_path" not in result
    assert transport.requested_urls == requests
    payload_root = workspace / "storage/transcription-ocr/cvpr2026/r6/material-payloads"
    assert not payload_root.exists() or not any(payload_root.iterdir())


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_redirect_bound_loop_and_streaming_policy_are_fail_closed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    initial = "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"
    hop_one = "https://openaccess.thecvf.com/one"
    hop_two = "https://openaccess.thecvf.com/two"
    hop_three = "https://openaccess.thecvf.com/three"
    transport = FakeTransport(
        {
            initial: _response(status_code=302, headers={"location": hop_one}),
            hop_one: _response(status_code=302, headers={"location": hop_two}),
            hop_two: _response(status_code=302, headers={"location": hop_three}),
        }
    )

    result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=_record(policy),
        transport=transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )

    assert result["terminal_reason"] == "redirect_limit_exceeded"
    assert "local_payload_path" not in result
    assert transport.requested_urls == [initial, hop_one, hop_two]


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_redirect_loop_and_partial_stream_write_are_terminal_without_payload(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    initial = "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"
    loop = "https://openaccess.thecvf.com/loop"
    loop_transport = FakeTransport(
        {
            initial: _response(status_code=302, headers={"location": loop}),
            loop: _response(status_code=302, headers={"location": initial}),
        }
    )
    loop_result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=_record(policy),
        transport=loop_transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )
    assert loop_result["terminal_reason"] == "redirect_loop"

    def broken_body():
        yield b"%PDF-"
        raise RuntimeError("fixture stream failure")

    stream_transport = FakeTransport(
        {
            initial: _response(chunks=broken_body()),
        }
    )
    stream_result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=_record(policy),
        transport=stream_transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )
    assert stream_result["terminal_reason"] == "payload_write_failed"
    assert "local_payload_path" not in stream_result
    assert not list(
        (workspace / "storage/transcription-ocr/cvpr2026/r6/material-payloads").glob("*.tmp")
    )


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_rejects_policy_mismatch_bad_records_and_output_escape_before_transport(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    transport = FakeTransport({})

    for record, root, reason in (
        (
            _record(policy, policy_digest="0" * 64),
            Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
            "policy_digest_mismatch",
        ),
        (
            _record(policy, eligibility="classification_only"),
            Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
            "record_not_eligible",
        ),
        (_record(policy), Path("../escape"), "payload_root_outside_workspace"),
    ):
        result = acquire_declared_public_material(
            workspace=workspace,
            policy=policy,
            record=record,
            transport=transport,
            payload_root=root,
        )
        assert result["terminal_reason"] == reason
        assert "local_payload_path" not in result
    assert transport.requested_urls == []


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_rejects_malformed_record_before_transport(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    policy = _policy(workspace)
    transport = FakeTransport({})
    malformed = _record(policy)
    del malformed["event_id"]

    result = acquire_declared_public_material(
        workspace=workspace,
        policy=policy,
        record=malformed,
        transport=transport,
        payload_root=Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
    )

    assert result["terminal_reason"] == "malformed_discovery_record"
    assert transport.requested_urls == []
