"""Local contract tests for the CVPR Stage-4 acquisition primitives."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from conference_pipeline.acquisition.cvpr_corpus_download import (
    build_initial_ledger,
    canonical_ledger_digest,
    derive_required_byte_reservation,
    execute_public_acquisition,
    is_valid_pdf,
    resume_compatible_ledger,
    resume_artifact_is_valid,
    stream_to_workspace,
)


def _selection(*, duplicate_url: bool = False, count: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        url = "https://fixtures.example/shared" if duplicate_url and index < 2 else f"https://fixtures.example/{index}"
        rows.append(
            {
                "program_item_id": f"cvpr2026:program-item:{index:05d}",
                "source_record_ids": [f"cvpr2026:source:program-schedule:{index:05d}"],
                "source_family_ids": ["program-schedule"],
                "selection_status": "included",
                "planned_acquisition": [
                    {
                        "source_record_id": f"cvpr2026:source:program-schedule:{index:05d}",
                        "source_family_id": "program-schedule",
                        "source_urls": [url],
                        "phase": "download-corpus",
                        "intent": "fixture acquisition",
                    }
                ],
                "unreachable_source_path": None,
            }
        )
    return rows


def _lineage() -> dict[str, str]:
    return {
        "run_id": "fixture-r3",
        "phase_02_inventory_sha256": "a" * 64,
        "phase_03_selection_sha256": "b" * 64,
        "storage_manifest_sha256": "c" * 64,
    }


def test_initial_ledger_has_one_result_per_item_without_losing_duplicate_requests() -> None:
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=_selection(duplicate_url=True),
        policy_exclusions=[{"exclusion_id": "paywalled_official_video"}],
        expected_program_item_count=3,
    )

    assert [row["program_item_id"] for row in ledger["program_item_results"]] == [
        "cvpr2026:program-item:00000",
        "cvpr2026:program-item:00001",
        "cvpr2026:program-item:00002",
    ]
    assert len(ledger["program_item_results"]) == 3
    assert len(ledger["request_results"]) == 2
    assert ledger["counts"]["program_item_result_count"] == 3
    assert ledger["policy_exclusions"] == [{"exclusion_id": "paywalled_official_video"}]
    assert all(row["request_key"] for row in ledger["program_item_results"])


def test_initial_ledger_preserves_long_real_filenames_without_redaction() -> None:
    """A long, human-authored filename (e.g. a CVF paper title) must survive
    into both the fetchable request url and the displayed acquisition_plan.

    sanitize_for_receipt's opaque-payload heuristic (80+ chars of
    [A-Za-z0-9+=_-]) once redacted this to a literal ".../REDACTED.pdf" --
    an unreachable url -- despite it containing no secret at all. This must
    never regress: the request actually fetched has to match what the
    receipt displays.
    """

    long_filename = (
        "Xiao_Generalizable_Structure-Aware_Keypoint_Correspondence_for_"
        "Category-Unified_3D_Single_Object_Tracking_CVPR_2026_paper.pdf"
    )
    assert len(long_filename) > 80
    url = f"https://openaccess.thecvf.com/content/CVPR2026/papers/{long_filename}"
    rows = [
        {
            "program_item_id": "cvpr2026:program-item:00000",
            "source_record_ids": ["cvpr2026:source:cvf-openaccess-day-2026-06-05:00000"],
            "source_family_ids": ["cvf-openaccess-day-2026-06-05"],
            "selection_status": "included",
            "planned_acquisition": [
                {
                    "source_record_id": "cvpr2026:source:cvf-openaccess-day-2026-06-05:00000",
                    "source_family_id": "cvf-openaccess-day-2026-06-05",
                    "source_urls": [url],
                    "phase": "download-corpus",
                    "intent": "fixture acquisition",
                }
            ],
            "unreachable_source_path": None,
        }
    ]

    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=rows,
        policy_exclusions=[],
        expected_program_item_count=1,
    )

    assert ledger["request_results"][0]["source_url"] == url
    assert "REDACTED" not in ledger["request_results"][0]["source_url"]
    assert ledger["program_item_results"][0]["acquisition_plan"]["source_urls"] == [url]
    assert "REDACTED" not in json.dumps(ledger["program_item_results"][0]["acquisition_plan"])


def test_ledger_digest_ignores_volatile_values_and_full_count_is_not_a_report_cap() -> None:
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=_selection(count=10_515),
        policy_exclusions=[],
        expected_program_item_count=10_515,
    )
    digest = canonical_ledger_digest(ledger)
    changed = json.loads(json.dumps(ledger))
    changed["generated_at"] = "2999-01-01T00:00:00Z"
    changed["manifest_sha256"] = "f" * 64

    assert ledger["counts"]["program_item_result_count"] == 10_515
    assert ledger["report_pagination"]["page_size"] < 10_515
    assert canonical_ledger_digest(changed) == digest


def test_required_byte_reservation_is_manifest_and_scope_bound() -> None:
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=_selection(),
        policy_exclusions=[],
        expected_program_item_count=3,
    )
    reservation = derive_required_byte_reservation(
        ledger=ledger,
        storage_manifest={"artifact_families": [{"id": "download_corpus_artifacts"}]},
        safety_margin_bytes=64,
    )

    assert reservation["required_bytes"] > 0
    assert reservation["scope_ledger_sha256"] == canonical_ledger_digest(ledger)
    assert reservation["manifest_sha256"]
    assert reservation["safety_margin_bytes"] == 64


def test_resume_rejects_missing_tampered_and_workspace_escape_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / "storage" / "download-corpus" / "artifacts" / "paper.pdf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"%PDF-1.7\nfixture\n")
    valid = {
        "local_path": "storage/download-corpus/artifacts/paper.pdf",
        "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
        "byte_count": artifact.stat().st_size,
        "artifact_kind": "paper_pdf",
        "lineage": {"phase_03_selection_sha256": "b" * 64},
    }

    assert resume_artifact_is_valid(
        workspace=workspace,
        artifact=valid,
        expected_lineage={"phase_03_selection_sha256": "b" * 64},
    )
    assert not resume_artifact_is_valid(
        workspace=workspace,
        artifact={**valid, "sha256": "0" * 64},
        expected_lineage={"phase_03_selection_sha256": "b" * 64},
    )
    assert not resume_artifact_is_valid(
        workspace=workspace,
        artifact={**valid, "local_path": "../escaped.pdf"},
        expected_lineage={"phase_03_selection_sha256": "b" * 64},
    )


def test_resume_ledger_reuses_only_verified_artifacts_and_preserves_terminal_outcomes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    fresh = build_initial_ledger(
        lineage=_lineage(), selection_rows=_selection(count=2), policy_exclusions=[], expected_program_item_count=2
    )
    artifact_path = workspace / "storage" / "download-corpus" / "artifacts" / "source_page" / "fixture.html"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"<main>fixture</main>")
    digest = __import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest()
    artifact = {
        "artifact_id": "artifact-1",
        "request_key": fresh["request_results"][0]["request_key"],
        "artifact_kind": "source_page",
        "local_path": "storage/download-corpus/artifacts/source_page/fixture.html",
        "mime": "text/html",
        "byte_count": artifact_path.stat().st_size,
        "sha256": digest,
        "lineage": dict(fresh["lineage"]),
    }
    prior = json.loads(json.dumps(fresh))
    prior["request_results"][0].update(
        {
            "status": "succeeded",
            "final_url": "https://fixtures.example/0",
            "http_status": 200,
            "content_type": "text/html",
            "local_path": artifact["local_path"],
            "byte_count": artifact["byte_count"],
            "sha256": artifact["sha256"],
            "attempts": [{"attempt": 1, "http_status": 200, "reason": "transient_fetch"}],
        }
    )
    prior["request_results"][1].update(
        {
            "status": "invalid_content",
            "http_status": 200,
            "content_type": "text/html",
            "attempts": [{"attempt": 1, "http_status": 200, "reason": "transient_fetch"}],
        }
    )
    prior["artifacts"] = [artifact]
    prior["manifest_sha256"] = canonical_ledger_digest(prior)
    prior_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    resumed, evidence = resume_compatible_ledger(workspace=workspace, fresh_ledger=fresh)

    assert evidence == {
        "status": "reused",
        "prior_ledger_sha256": prior["manifest_sha256"],
        "reused_artifact_count": 1,
        "retained_terminal_request_count": 1,
        "discarded_artifact_count": 0,
    }
    assert resumed["request_results"][0]["status"] == "succeeded"
    assert resumed["request_results"][1]["status"] == "invalid_content"
    assert len(resumed["artifacts"]) == 1
    assert resumed["program_item_results"][0]["artifact_id"] == "artifact-1"
    assert resumed["program_item_results"][1]["blocker"]["code"] == "invalid_content"


def test_resume_ledger_rejects_changed_lineage_before_reusing_any_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    fresh = build_initial_ledger(
        lineage=_lineage(), selection_rows=_selection(count=1), policy_exclusions=[], expected_program_item_count=1
    )
    prior = json.loads(json.dumps(fresh))
    prior["lineage"]["phase_03_selection_sha256"] = "d" * 64
    prior["manifest_sha256"] = canonical_ledger_digest(prior)
    path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(prior), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage or digest"):
        resume_compatible_ledger(workspace=workspace, fresh_ledger=fresh)


def test_streaming_artifact_write_is_atomic_and_pdf_validation_is_truthful(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    destination = workspace / "storage" / "download-corpus" / "artifacts" / "paper.pdf"

    class Response:
        headers = {"Content-Type": "application/pdf"}
        status = 200
        url = "https://fixtures.example/paper.pdf?token=do-not-copy"

        def read(self, size: int) -> bytes:
            if not hasattr(self, "sent"):
                self.sent = True
                return b"%PDF-1.7\nfixture\n"
            return b""

        def close(self) -> None:
            return None

    artifact = stream_to_workspace(
        workspace=workspace,
        destination=destination,
        opener=lambda _url: Response(),
        url="https://fixtures.example/paper.pdf?token=do-not-copy",
        artifact_kind="paper_pdf",
    )

    assert artifact["status"] == "succeeded"
    assert destination.exists()
    assert artifact["local_path"] == "storage/download-corpus/artifacts/paper.pdf"
    assert "do-not-copy" not in json.dumps(artifact)
    assert is_valid_pdf(b"%PDF-1.7", "application/pdf")
    assert not is_valid_pdf(b"<html>not a pdf</html>", "text/html")


def test_streaming_deadline_turns_a_non_yielding_response_into_a_sanitized_blocker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    destination = workspace / "storage" / "download-corpus" / "artifacts" / "stalled.html"

    class NonYieldingResponse:
        headers = {"Content-Type": "text/html"}
        status = 200
        url = "https://fixtures.example/stalled?token=must-not-appear"

        def read(self, _size: int) -> bytes:
            time.sleep(5)
            return b"never reached"

        def close(self) -> None:
            return None

    result = stream_to_workspace(
        workspace=workspace,
        destination=destination,
        opener=lambda _url: NonYieldingResponse(),
        url="https://fixtures.example/stalled?token=must-not-appear",
        artifact_kind="source_page",
        response_deadline_seconds=0.01,
    )

    assert result == {
        "status": "unreachable_fetch_error",
        "source_url": "https://fixtures.example/stalled",
        "artifact_kind": "source_page",
        "error_type": "response_deadline_exceeded",
        "response_deadline_seconds": 0.01,
        "retry_events": [],
    }
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("artifact_kind", ["paper_pdf", "media_asset"])
def test_stream_rejects_unsupported_artifact_paths_before_opening(tmp_path: Path, artifact_kind: str) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="workspace"):
        stream_to_workspace(
            workspace=tmp_path / "workspace",
            destination=tmp_path / "escaped.bin",
            opener=lambda url: calls.append(url),
            url="https://fixtures.example/file",
            artifact_kind=artifact_kind,
        )
    assert calls == []


def test_public_acquisition_retries_then_checkpoints_one_result_for_all_duplicate_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=_selection(duplicate_url=True, count=2),
        policy_exclusions=[],
        expected_program_item_count=2,
    )
    responses = [429, 200]
    checkpoints: list[str] = []
    delays: list[float] = []

    class Response:
        headers = {"Content-Type": "text/html"}
        url = "https://fixtures.example/shared?token=do-not-copy"

        def __init__(self, status: int) -> None:
            self.status = status
            self.sent = False

        def read(self, _size: int) -> bytes:
            if self.sent or self.status != 200:
                return b""
            self.sent = True
            return b"<main>fixture source page</main>"

        def close(self) -> None:
            return None

    def opener(_url: str) -> Response:
        return Response(responses.pop(0))

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=opener,
        checkpoint=lambda payload: checkpoints.append(str(payload["manifest_sha256"])),
        sleeper=delays.append,
    )

    request = result["request_results"][0]
    assert request["status"] == "succeeded"
    assert request["retry_events"] == [{"attempt": 1, "backoff_seconds": 1.0, "http_status": 429, "reason": "transient_http"}]
    assert delays == [1.0]
    assert len(checkpoints) == 1
    assert len(result["artifacts"]) == 1
    assert all(row["blocker"] is None for row in result["program_item_results"])
    assert "do-not-copy" not in json.dumps(result)


def test_public_acquisition_uses_one_sanitized_auth_recovery_and_rejects_html_pdf(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rows = _selection(count=1)
    rows[0]["planned_acquisition"] = [{
        "source_record_id": "cvpr2026:source:cvf-openaccess:00001",
        "source_family_id": "cvf-openaccess",
        "source_urls": ["https://fixtures.example/paper.pdf"],
    }]
    ledger = build_initial_ledger(
        lineage=_lineage(), selection_rows=rows, policy_exclusions=[], expected_program_item_count=1
    )
    calls: list[str] = []

    class Response:
        def __init__(self, status: int, body: bytes, content_type: str) -> None:
            self.status = status
            self.headers = {"Content-Type": content_type}
            self.url = "https://fixtures.example/paper.pdf"
            self.body = body
            self.sent = False

        def read(self, _size: int) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return self.body

        def close(self) -> None:
            return None

    def recovery(_url: str) -> dict[str, object]:
        calls.append("recovery")
        return {
            "opener": lambda _value: Response(200, b"<html>login</html>", "application/pdf"),
            "mode": "openreview-configured",
            "environment_variable_names": ["OPENREVIEW_PASSWORD", "OPENREVIEW_USERNAME"],
            "token": "must-not-appear",
        }

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda _url: Response(403, b"", "text/html"),
        auth_recovery=recovery,
        checkpoint=lambda _payload: None,
        sleeper=lambda _seconds: None,
    )

    request = result["request_results"][0]
    assert calls == ["recovery"]
    assert request["status"] == "invalid_content"
    assert request["auth_recovery"] == {
        "mode": "openreview-configured",
        "status": "attempted",
        "environment_variable_names": ["OPENREVIEW_PASSWORD", "OPENREVIEW_USERNAME"],
    }
    assert result["program_item_results"][0]["blocker"]["code"] == "invalid_content"
    assert "must-not-appear" not in json.dumps(result)


def test_public_acquisition_exhausts_bounded_transient_retries_with_an_actionable_blocker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ledger = build_initial_ledger(
        lineage=_lineage(), selection_rows=_selection(count=1), policy_exclusions=[], expected_program_item_count=1
    )
    delays: list[float] = []

    class Response:
        status = 429
        headers = {"Content-Type": "text/html"}
        url = "https://fixtures.example/rate-limited"

        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            return None

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda _url: Response(),
        checkpoint=lambda _payload: None,
        sleeper=delays.append,
    )

    request = result["request_results"][0]
    assert request["status"] == "unreachable_http_error"
    assert [event["backoff_seconds"] for event in request["retry_events"]] == [1.0, 2.0]
    assert delays == [1.0, 2.0]
    assert result["program_item_results"][0]["blocker"]["code"] == "unreachable_http_error"


def test_public_acquisition_retries_a_non_yielding_response_then_checkpoints_its_blocker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ledger = build_initial_ledger(
        lineage=_lineage(), selection_rows=_selection(count=1), policy_exclusions=[], expected_program_item_count=1
    )
    delays: list[float] = []
    checkpoints: list[str] = []

    class NonYieldingResponse:
        headers = {"Content-Type": "text/html"}
        status = 200
        url = "https://fixtures.example/never-yields"

        def read(self, _size: int) -> bytes:
            time.sleep(5)
            return b"never reached"

        def close(self) -> None:
            return None

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda _url: NonYieldingResponse(),
        checkpoint=lambda payload: checkpoints.append(str(payload["manifest_sha256"])),
        response_deadline_seconds=0.01,
        sleeper=delays.append,
    )

    request = result["request_results"][0]
    assert request["status"] == "unreachable_fetch_error"
    assert request["error_type"] == "response_deadline_exceeded"
    assert request["response_deadline_seconds"] == 0.01
    assert [event["backoff_seconds"] for event in request["retry_events"]] == [1.0, 2.0]
    assert delays == [1.0, 2.0]
    assert len(request["attempts"]) == 3
    assert len(checkpoints) == 1
    assert result["program_item_results"][0]["blocker"]["code"] == "unreachable_fetch_error"


def test_public_acquisition_uses_bounded_concurrency_with_deterministic_parent_checkpoints(tmp_path: Path) -> None:
    def run_once(workspace: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        ledger = build_initial_ledger(
            lineage=_lineage(), selection_rows=_selection(count=4), policy_exclusions=[], expected_program_item_count=4
        )
        checkpoints: list[dict[str, object]] = []

        class Response:
            headers = {"Content-Type": "text/html"}
            status = 200

            def __init__(self, url: str) -> None:
                self.url = url
                self.sent = False

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                time.sleep(0.03)
                return b"<main>concurrent fixture</main>"

            def close(self) -> None:
                return None

        result = execute_public_acquisition(
            workspace=workspace,
            ledger=ledger,
            opener=lambda url: Response(url),
            checkpoint=lambda payload: checkpoints.append(json.loads(json.dumps(payload))),
            worker_count=2,
        )
        return result, checkpoints

    first, first_checkpoints = run_once(tmp_path / "first")
    second, second_checkpoints = run_once(tmp_path / "second")

    assert first["public_transport"] == {
        "scheduler": "bounded_threads_v1",
        "worker_count": 2,
        "response_deadline_seconds": 45.0,
        "max_in_flight": 2,
    }
    assert len(first_checkpoints) == 4
    assert all(snapshot["manifest_sha256"] == canonical_ledger_digest(snapshot) for snapshot in first_checkpoints)
    assert first["counts"]["program_item_result_count"] == 4
    assert first["counts"]["artifact_count"] == 4
    assert all(request["status"] == "succeeded" for request in first["request_results"])
    assert canonical_ledger_digest(first) == canonical_ledger_digest(second)
    assert len(second_checkpoints) == 4


def test_concurrent_public_acquisition_batches_local_control_checkpoints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rows = _selection(count=3)
    for index, row in enumerate(rows[:2]):
        row["source_family_ids"] = ["virtual-paper-media-posture"]
        row["planned_acquisition"] = [{
            "source_record_id": f"cvpr2026:source:virtual-paper-media-posture:{index:05d}",
            "source_family_id": "virtual-paper-media-posture",
            "source_urls": [f"https://fixtures.example/registration-only-{index}"],
        }]
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=rows,
        policy_exclusions=[{
            "exclusion_id": "paywalled_official_video",
            "family_id": "virtual-paper-media-posture",
            "reason": "locked policy",
        }],
        expected_program_item_count=3,
    )
    checkpoints: list[dict[str, object]] = []

    class Response:
        headers = {"Content-Type": "text/html"}
        status = 200

        def __init__(self, url: str) -> None:
            self.url = url
            self.sent = False

        def read(self, _size: int) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return b"<main>fixture source page</main>"

        def close(self) -> None:
            return None

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda url: Response(url),
        checkpoint=lambda payload: checkpoints.append(json.loads(json.dumps(payload))),
        worker_count=2,
    )

    # One checkpoint contains both local control outcomes; one contains the
    # public worker outcome.  Every boundary remains canonical and resumable.
    assert len(checkpoints) == 2
    assert all(snapshot["manifest_sha256"] == canonical_ledger_digest(snapshot) for snapshot in checkpoints)
    assert [request["status"] for request in result["request_results"]].count("policy_excluded_official_video") == 2
    assert result["counts"]["artifact_count"] == 1


def test_public_acquisition_preserves_official_video_policy_without_opening_registration_url(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rows = _selection(count=1)
    rows[0]["source_family_ids"] = ["virtual-paper-media-posture"]
    rows[0]["planned_acquisition"] = [{
        "source_record_id": "cvpr2026:source:virtual-paper-media-posture:00001",
        "source_family_id": "virtual-paper-media-posture",
        "source_urls": ["https://fixtures.example/registration-only"],
    }]
    ledger = build_initial_ledger(
        lineage=_lineage(),
        selection_rows=rows,
        policy_exclusions=[{
            "exclusion_id": "paywalled_official_video",
            "family_id": "virtual-paper-media-posture",
            "reason": "locked policy",
        }],
        expected_program_item_count=1,
    )
    opened: list[str] = []

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda url: opened.append(url),
        checkpoint=lambda _payload: None,
    )

    assert opened == []
    assert result["request_results"][0]["status"] == "policy_excluded_official_video"
    assert result["program_item_results"][0]["blocker"]["code"] == "policy_excluded_official_video"


def test_public_acquisition_consumes_workshop_owner_summary_without_another_auth_client(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rows = _selection(count=1)
    rows[0]["source_family_ids"] = ["openreview-workshop"]
    rows[0]["planned_acquisition"] = [{
        "source_record_id": "cvpr2026:source:openreview-workshop:00001",
        "source_family_id": "openreview-workshop",
        "source_urls": ["https://api2.openreview.net/notes?invitation=fixture"],
    }]
    ledger = build_initial_ledger(
        lineage=_lineage(), selection_rows=rows, policy_exclusions=[], expected_program_item_count=1
    )
    owner_calls: list[str] = []

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda _url: pytest.fail("workshop adapter must own the request"),
        workshop_owner=lambda parent: owner_calls.append(parent) or {
            "status": "succeeded",
            "download_receipt_path": "storage/workshops/download_receipt.json",
            "download": {"totals": {"available": 1}},
        },
        workshop_parent_venue="thecvf.com/CVPR/2026/Workshop",
        checkpoint=lambda _payload: None,
    )

    request = result["request_results"][0]
    assert owner_calls == ["thecvf.com/CVPR/2026/Workshop"]
    assert request["status"] == "workshop_owner_result"
    assert request["workshop_owner"]["download_receipt_path"] == "storage/workshops/download_receipt.json"
    assert result["program_item_results"][0]["blocker"]["code"] == "no_admissible_artifact"


def test_public_acquisition_reuses_one_registered_workshop_owner_result_per_parent_venue(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rows = _selection(count=2)
    for index, row in enumerate(rows):
        row["source_family_ids"] = ["openreview-workshop"]
        row["planned_acquisition"] = [{
            "source_record_id": f"cvpr2026:source:openreview-workshop:{index:05d}",
            "source_family_id": "openreview-workshop",
            "source_urls": [f"https://api2.openreview.net/notes?invitation=fixture-{index}"],
        }]
    ledger = build_initial_ledger(
        lineage=_lineage(), selection_rows=rows, policy_exclusions=[], expected_program_item_count=2
    )
    owner_calls: list[str] = []

    result = execute_public_acquisition(
        workspace=workspace,
        ledger=ledger,
        opener=lambda _url: pytest.fail("workshop adapter must own the request"),
        workshop_owner=lambda parent: owner_calls.append(parent) or {
            "status": "succeeded",
            "download_receipt_path": "storage/workshops/download_receipt.json",
            "download": {"totals": {"available": 2}},
        },
        workshop_parent_venue="thecvf.com/CVPR/2026/Workshop",
        checkpoint=lambda _payload: None,
        worker_count=2,
    )

    assert owner_calls == ["thecvf.com/CVPR/2026/Workshop"]
    assert [request["status"] for request in result["request_results"]] == [
        "workshop_owner_result", "workshop_owner_result"
    ]
