"""Offline contract tests for the receipt-owned CVPR r6 execution lane."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import conference_pipeline.acquisition.cvpr_r6_artifact_contract as artifact_contract_module
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract,
    R6ArtifactContractError,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Registry:
    def get(self, task_id: str):
        caps = {
            "transcription.session_speech": 200.0,
            "ocr.slide_image_parse": 30.0,
        }
        providers = {
            "transcription.session_speech": "openai",
            "ocr.slide_image_parse": "openai",
        }
        return SimpleNamespace(budget_cap_usd=caps[task_id], provider=providers[task_id])


class _FakeProvider:
    def __init__(self, events: list[str], *, failure: str | None = None) -> None:
        self.events = events
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def _result(self, *, operation: str, item_id: str, actual: float | None = 1.0):
        from conference_pipeline.provider_receipts import current_provider_receipt_context

        self.events.append(f"service:{operation}:{item_id}")
        self.calls.append(
            {
                "operation": operation,
                "item_id": item_id,
                "task_id": current_provider_receipt_context().task_id,
            }
        )
        receipt = SimpleNamespace(
            provider="openai",
            model="governed-fake-model",
            status="failed" if self.failure == "failed_receipt" else "succeeded",
            task_id=None,
            estimated_cost_usd=actual,
            receipt_path=None,
            to_dict=lambda: {
                "provider": "openai",
                "model": "governed-fake-model",
                "status": "failed" if self.failure == "failed_receipt" else "succeeded",
                "estimated_cost_usd": actual,
                "secrets_copied": False,
            },
        )
        if self.failure == "raise_with_receipt":
            error = RuntimeError("fake provider failed")
            error.provider_receipt = receipt
            raise error
        if self.failure == "missing_receipt":
            return SimpleNamespace(response={"text": operation}, receipt=None)
        return SimpleNamespace(response={"text": operation}, receipt=receipt)

    def benchmark(self, *, task_id: str, model: str, representative: object | None = None):
        del representative
        assert model == "governed-fake-model"
        return self._result(operation="benchmark", item_id=task_id)

    def transcribe(self, path: Path, *, item_id: str, model: str):
        assert path.is_file()
        assert model == "governed-fake-model"
        return self._result(operation="asr", item_id=item_id)

    def ocr(self, path: Path, *, item_id: str, model: str):
        assert path.is_file()
        assert model == "governed-fake-model"
        return self._result(operation="ocr", item_id=item_id)


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object], str]:
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    policy_digest = "a" * 64
    root = workspace / "storage/transcription-ocr/cvpr2026/r6"
    audio = root / "raw-audio/talk-001.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio-bytes")
    paper = root / "material-payloads/paper-001.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4\nfixture")
    image = root / "material-payloads/slide-001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    audio_receipt: dict[str, object] = {
        "status": "succeeded",
        "policy_digest": policy_digest,
        "records": [
            {
                "item_id": "talk-001",
                "terminal_state": "acquired",
                "source_url": "https://www.youtube.com/watch?v=abcDEF12345",
                "local_path": audio.relative_to(workspace).as_posix(),
                "sha256": _sha256(audio),
                "mime_type": "audio/mpeg",
                "byte_count": audio.stat().st_size,
                "duration_seconds": 61.0,
                "provenance": "direct_public_youtube",
            }
        ],
    }
    material_receipt: dict[str, object] = {
        "status": "succeeded",
        "policy_digest": policy_digest,
        "payloads": [
            {
                "record_id": "paper-001",
                "acquisition_state": "acquired",
                "content_class": "paper_pdf",
                "local_payload_path": paper.relative_to(workspace).as_posix(),
                "sha256": _sha256(paper),
                "mime_type": "application/pdf",
                "byte_count": paper.stat().st_size,
                "policy_digest": policy_digest,
                "source_page_provenance": "https://cvpr.thecvf.com/virtual/2026/event/fixture",
            },
            {
                "record_id": "slide-001",
                "acquisition_state": "acquired",
                "content_class": "slide_image",
                "local_payload_path": image.relative_to(workspace).as_posix(),
                "sha256": _sha256(image),
                "mime_type": "image/png",
                "byte_count": image.stat().st_size,
                "policy_digest": policy_digest,
                "source_page_provenance": "https://cvpr.thecvf.com/virtual/2026/event/fixture",
            },
        ],
    }
    return workspace, audio_receipt, material_receipt, policy_digest


def test_r6_strict_queue_rejects_a_redirected_production_workspace_before_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No strict queue may select an external workspace through a production link."""

    from conference_pipeline import r6_transcription_ocr_execution as execution

    repository = tmp_path / "repo"
    workspace = repository / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    workspace.rmdir()
    workspace.symlink_to(external, target_is_directory=True)
    policy_digest = "a" * 64

    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        execution.build_r6_declared_local_queues(
            workspace=workspace,
            audio_receipt={"status": "succeeded", "policy_digest": policy_digest, "records": []},
            material_acquisition_receipt={
                "status": "succeeded",
                "policy_digest": policy_digest,
                "payloads": [],
            },
            policy_digest=policy_digest,
        )


def test_r6_execution_uses_only_declared_local_queues_and_settles_every_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conference_pipeline import r6_transcription_ocr_execution as execution

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    events: list[str] = []
    original_reserve = execution.budget_enforcement.reserve_r6_budget
    original_preflight = execution.budget_enforcement.check_preflight
    original_postcall = execution.budget_enforcement.check_postcall
    original_settle = execution.budget_enforcement.settle_r6_budget

    def reserve(*args, **kwargs):
        events.append("reserve")
        return original_reserve(*args, **kwargs)

    def preflight(*args, **kwargs):
        events.append("preflight")
        return original_preflight(*args, **kwargs)

    def postcall(*args, **kwargs):
        events.append("postcall")
        return original_postcall(*args, **kwargs)

    def settle(*args, **kwargs):
        events.append("settle")
        return original_settle(*args, **kwargs)

    monkeypatch.setattr(execution.budget_enforcement, "_existing_spend", lambda _: 0.0)
    monkeypatch.setattr(execution.budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(execution.budget_enforcement, "reserve_r6_budget", reserve)
    monkeypatch.setattr(execution.budget_enforcement, "check_preflight", preflight)
    monkeypatch.setattr(execution.budget_enforcement, "check_postcall", postcall)
    monkeypatch.setattr(execution.budget_enforcement, "settle_r6_budget", settle)
    monkeypatch.setattr(
        execution, "default_model_for_provider", lambda task_id, provider: "governed-fake-model"
    )

    provider = _FakeProvider(events)
    result = execution.execute_r6_declared_local(
        workspace=workspace,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_receipt,
        policy_digest=policy_digest,
        ledger_path=contract.path("budget_ledger"),
        provider_receipts_path=contract.path("provider_receipts"),
        inventory_path=contract.path("processing_inventory"),
        registry=_Registry(),
        provider=provider,
        pdf_text_extractor=lambda path: f"text:{path.name}",
        maximum_charge_usd_by_task={
            "transcription.session_speech": 1.0,
            "ocr.slide_image_parse": 1.0,
        },
        artifact_contract=contract,
    )

    assert result["status"] == "succeeded"
    assert result["counts"] == {"speech": 1, "paper_text": 1, "ocr": 1, "blockers": 0}
    assert events == [
        "reserve",
        "preflight",
        "service:asr:talk-001",
        "postcall",
        "settle",
        "reserve",
        "preflight",
        "service:ocr:slide-001",
        "postcall",
        "settle",
    ]
    assert all("source_url" not in call for call in provider.calls)
    assert {call["task_id"] for call in provider.calls} == {
        "transcription.session_speech",
        "ocr.slide_image_parse",
    }
    assert (
        json.loads(contract.path("budget_ledger").read_text())["reservations"][-1]["state"]
        == "settled"
    )
    assert json.loads(contract.path("processing_inventory").read_text())["status"] == "succeeded"


@pytest.mark.parametrize("poisoned", ("root", "leaf"))
def test_r6_strict_speech_queue_rejects_raw_audio_links_before_provider_factory(
    tmp_path: Path, poisoned: str
) -> None:
    from conference_pipeline import r6_transcription_ocr_execution as execution

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    record = audio_receipt["records"][0]
    assert isinstance(record, dict)
    record["local_media_path"] = record.pop("local_path")
    audio_receipt["audio_policy_digest"] = "b" * 64
    raw_audio = contract.audio_root / "talk-001.mp3"
    external = tmp_path / "external-sentinel"
    external.mkdir()
    sentinel = external / raw_audio.name
    sentinel.write_bytes(b"audio-bytes")
    raw_audio.unlink()
    if poisoned == "root":
        raw_audio.parent.rmdir()
        raw_audio.parent.symlink_to(external, target_is_directory=True)
    else:
        raw_audio.parent.mkdir(parents=True, exist_ok=True)
        raw_audio.symlink_to(sentinel)
    constructed: list[object] = []

    with pytest.raises(execution.R6ExecutionError, match="raw audio"):
        execution.build_r6_declared_local_queues(
            workspace=workspace,
            audio_receipt=audio_receipt,
            material_acquisition_receipt=material_receipt,
            policy_digest=policy_digest,
            audio_policy_digest="b" * 64,
            provider_factory=lambda: constructed.append(object()),
            artifact_contract=contract,
        )

    assert constructed == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda audio, materials: audio["records"][0].__setitem__("local_path", "../escape.mp3"),
            "workspace",
        ),
        (
            lambda audio, materials: audio["records"][0].__setitem__(
                "local_path", "storage/transcription-ocr/cvpr2026/r5/raw.mp3"
            ),
            "r6",
        ),
        (lambda audio, materials: audio["records"][0].__setitem__("local_path", "*.mp3"), "glob"),
        (lambda audio, materials: audio["records"][0].__setitem__("byte_count", 999), "byte"),
        (lambda audio, materials: audio["records"][0].__setitem__("mime_type", ""), "MIME"),
        (
            lambda audio, materials: materials["payloads"][0].__setitem__(
                "acquisition_state", "terminal"
            ),
            "acquired",
        ),
    ],
)
def test_r6_execution_rejects_undeclared_or_invalid_payloads_before_provider_construction(
    tmp_path: Path, mutate, message: str
) -> None:
    from conference_pipeline import r6_transcription_ocr_execution as execution

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    mutate(audio_receipt, material_receipt)
    constructed = False

    def factory():
        nonlocal constructed
        constructed = True
        return _FakeProvider([])

    with pytest.raises(execution.R6ExecutionError, match=message):
        execution.build_r6_declared_local_queues(
            workspace=workspace,
            audio_receipt=audio_receipt,
            material_acquisition_receipt=material_receipt,
            policy_digest=policy_digest,
            provider_factory=factory,
            artifact_contract=contract,
        )
    assert constructed is False


@pytest.mark.parametrize("failure", ["raise_with_receipt", "missing_receipt"])
def test_r6_execution_stops_on_failed_or_missing_receipt_without_next_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from conference_pipeline import r6_transcription_ocr_execution as execution

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    monkeypatch.setattr(execution.budget_enforcement, "_existing_spend", lambda _: 0.0)
    monkeypatch.setattr(execution.budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(
        execution, "default_model_for_provider", lambda task_id, provider: "governed-fake-model"
    )
    events: list[str] = []
    provider = _FakeProvider(events, failure=failure)

    result = execution.execute_r6_declared_local(
        workspace=workspace,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_receipt,
        policy_digest=policy_digest,
        ledger_path=contract.path("budget_ledger"),
        provider_receipts_path=contract.path("provider_receipts"),
        inventory_path=contract.path("processing_inventory"),
        registry=_Registry(),
        provider=provider,
        pdf_text_extractor=lambda path: "text",
        maximum_charge_usd_by_task={
            "transcription.session_speech": 1.0,
            "ocr.slide_image_parse": 1.0,
        },
        artifact_contract=contract,
    )

    assert result["status"] == "blocked"
    assert len(provider.calls) == 1
    assert result["blockers"][0]["code"] in {"provider_failed", "provider_receipt_missing"}
    assert json.loads(contract.path("processing_inventory").read_text())["status"] == "blocked"


def test_r6_execution_cli_has_no_source_or_discovery_options() -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    options = {action.dest for action in cli.build_parser()._actions}
    assert options >= {"mode"}
    forbidden = {
        "workspace",
        "url",
        "source_url",
        "glob",
        "input",
        "output",
        "host",
        "cookie",
        "model",
    }
    assert not options.intersection(forbidden)


def test_r6_full_execution_reuses_bootstrap_prefix_and_consumes_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixed full owner must not create a second speech benchmark."""

    from conference_pipeline import r6_transcription_ocr_execution as execution
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        build_r6_dispatch_schedule,
        canonical_r6_queue,
    )

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    queue, queue_digest = canonical_r6_queue(
        speech_ids=["talk-001"], ocr_ids=["slide-001"]
    )
    schedule = build_r6_dispatch_schedule(
        run_id=contract.run_id,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=1.0,
        ocr_total_usd=1.0,
    )
    monkeypatch.setattr(execution.budget_enforcement, "_existing_spend", lambda _: 0.0)
    monkeypatch.setattr(execution.budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(
        execution, "default_model_for_provider", lambda _task, _provider: "governed-fake-model"
    )
    events: list[str] = []
    provider = _FakeProvider(events)
    common = {
        "workspace": workspace,
        "audio_receipt": audio_receipt,
        "material_acquisition_receipt": material_receipt,
        "policy_digest": policy_digest,
        "ledger_path": contract.path("budget_ledger"),
        "provider_receipts_path": contract.path("provider_receipts"),
        "inventory_path": contract.path("processing_inventory"),
        "registry": _Registry(),
        "provider": provider,
        "pdf_text_extractor": lambda _path: "text",
        "maximum_charge_usd_by_task": {
            "transcription.session_speech": 1.0,
            "ocr.slide_image_parse": 1.0,
        },
        "dispatch_schedule": schedule,
        "artifact_contract": contract,
    }

    with pytest.raises(execution.R6ExecutionError, match="cannot auto-benchmark"):
        execution.execute_r6_declared_local(**common, mode="full")
    assert events == []

    speech_bootstrap = execution.execute_r6_declared_local(
        **common,
        mode="benchmark-only",
        benchmark_task="transcription.session_speech",
    )
    assert speech_bootstrap["status"] == "succeeded"
    ocr_bootstrap = execution.execute_r6_declared_local(
        **common,
        mode="benchmark-only",
        benchmark_task="ocr.slide_image_parse",
    )
    assert ocr_bootstrap["status"] == "succeeded"
    full = execution.execute_r6_declared_local(
        **common,
        mode="full",
        prior_benchmarked_tasks=frozenset(
            {"transcription.session_speech", "ocr.slide_image_parse"}
        ),
    )
    assert full["status"] == "succeeded"
    assert events.count("service:benchmark:transcription.session_speech") == 1
    assert events.count("service:benchmark:ocr.slide_image_parse") == 1
    ledger = json.loads(contract.path("budget_ledger").read_text(encoding="utf-8"))
    assert ledger["next_dispatch_ordinal"] == len(schedule["entries"])
    assert all(row["state"] == "settled" for row in ledger["reservations"])


def test_r6_ocr_only_bootstrap_selects_the_next_ocr_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OCR-only queue has an explicit OCR benchmark producer."""

    from conference_pipeline import r6_transcription_ocr_execution as execution
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        build_r6_dispatch_schedule,
        canonical_r6_queue,
    )

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    audio_receipt["records"][0]["terminal_state"] = "download_failed"
    material_receipt["payloads"] = [material_receipt["payloads"][1]]
    queue, queue_digest = canonical_r6_queue(speech_ids=[], ocr_ids=["slide-001"])
    schedule = build_r6_dispatch_schedule(
        run_id=contract.run_id,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=0.0,
        ocr_total_usd=1.0,
    )
    monkeypatch.setattr(execution.budget_enforcement, "_existing_spend", lambda _: 0.0)
    monkeypatch.setattr(execution.budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(
        execution, "default_model_for_provider", lambda _task, _provider: "governed-fake-model"
    )
    events: list[str] = []
    result = execution.execute_r6_declared_local(
        workspace=workspace,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_receipt,
        policy_digest=policy_digest,
        ledger_path=contract.path("budget_ledger"),
        provider_receipts_path=contract.path("provider_receipts"),
        inventory_path=contract.path("processing_inventory"),
        registry=_Registry(),
        provider=_FakeProvider(events),
        maximum_charge_usd_by_task={"ocr.slide_image_parse": 1.0},
        dispatch_schedule=schedule,
        mode="benchmark-only",
        benchmark_task="ocr.slide_image_parse",
        artifact_contract=contract,
    )

    assert result["status"] == "succeeded"
    assert events == ["service:benchmark:ocr.slide_image_parse"]


def test_r6_terminal_lane_uses_no_provider_and_only_declared_pdf_work(tmp_path: Path) -> None:
    """Terminal audio/OCR inputs never load a paid execution lane."""

    from conference_pipeline import r6_transcription_ocr_execution as execution
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import r6_audio_policy_digest

    workspace, audio_receipt, material_receipt, policy_digest = _workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    audio_receipt["audio_policy_digest"] = r6_audio_policy_digest()
    audio_receipt["records"][0]["terminal_state"] = "download_failed"
    material_receipt["payloads"] = [material_receipt["payloads"][0]]

    result = execution.execute_r6_no_paid_lane(
        workspace=workspace,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_receipt,
        policy_digest=policy_digest,
        audio_policy_digest=r6_audio_policy_digest(),
        inventory_path=contract.path("processing_inventory"),
        pdf_text_extractor=lambda _path: "local PDF only",
        artifact_contract=contract,
    )

    assert result["status"] == "succeeded"
    assert result["mode"] == "no-paid-lane"
    assert result["counts"] == {"speech": 0, "paper_text": 1, "ocr": 0, "blockers": 0}
    assert result["provider_receipts_path"] is None


def _write_fixed_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixed_cli_contract(tmp_path: Path) -> R6ArtifactContract:
    """Create only the named fixed receipt paths in an isolated workspace."""

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    receipt_payloads = {
        "audio_recovery_receipt": {"status": "succeeded", "secrets_copied": False},
        "material_acquisition_receipt": {"status": "succeeded", "secrets_copied": False},
        "cost_estimate_receipt": {
            "status": "succeeded",
            "maximum_charge_usd_by_task": {},
            "secrets_copied": False,
        },
        "benchmark_receipt": {"status": "succeeded", "secrets_copied": False},
        "provider_receipts": {
            "status": "succeeded",
            "receipts": [],
            "secrets_copied": False,
        },
    }
    for key, payload in receipt_payloads.items():
        _write_fixed_receipt(contract.path(key), payload)
    return contract


@pytest.mark.parametrize(
    "payload",
    [
        {"secrets_copied": False},
        {"nested": {"secrets_copied": False}},
        {"nested": [{"secrets_copied": False}]},
    ],
    ids=("top-level", "nested-map", "nested-list"),
)
def test_fixed_receipt_reader_accepts_only_the_canonical_false_attestation(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    receipt = tmp_path / "receipt.json"
    _write_fixed_receipt(receipt, payload)

    assert cli._read_sanitized_receipt(receipt, label="fixture") == payload


@pytest.mark.parametrize(
    "payload, prohibited",
    [
        ({"secrets_copied": True}, "secrets_copied"),
        ({"secrets_copied": 0}, "secrets_copied"),
        ({"secrets_copied": "false"}, "secrets_copied"),
        ({"Secrets_Copied": False}, "Secrets_Copied"),
        ({"nested": {"unrelated_secret": False}}, "unrelated_secret"),
        ({"nested": [{"token": False}]}, "token"),
    ],
    ids=("true", "false-like-int", "false-like-string", "lookalike-key", "nested-key", "list-key"),
)
def test_fixed_receipt_reader_rejects_noncanonical_or_nested_secret_keys_generically(
    tmp_path: Path, payload: dict[str, object], prohibited: str
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    receipt = tmp_path / "receipt.json"
    _write_fixed_receipt(receipt, payload)

    with pytest.raises(cli.R6ExecutionError) as raised:
        cli._read_sanitized_receipt(receipt, label="fixture")

    assert str(raised.value) == "fixture receipt is not sanitized"
    assert prohibited not in str(raised.value)


def _receipt_with_placed_string(*, placement: str, literal: str) -> dict[str, object]:
    """Place a neutral raw value at each recursive reader boundary."""

    if placement == "top-level":
        return {"value": literal}
    if placement == "nested-map":
        return {"nested": {"value": literal}}
    if placement == "nested-list":
        return {"nested": [{"value": literal}]}
    raise AssertionError(f"unknown placement fixture: {placement}")


def _receipt_with_placed_key(*, placement: str, key: str) -> dict[str, object]:
    """Place an inert child under an untrusted key at every map/list boundary."""

    if placement == "top-level":
        return {key: {"inert": "fixture-only"}}
    if placement == "nested-map":
        return {"nested": {key: {"inert": "fixture-only"}}}
    if placement == "nested-list":
        return {"nested": [{key: {"inert": "fixture-only"}}]}
    raise AssertionError(f"unknown placement fixture: {placement}")


_SANITIZER_PLACEMENTS = ("top-level", "nested-map", "nested-list")


@pytest.mark.parametrize("placement", _SANITIZER_PLACEMENTS)
@pytest.mark.parametrize(
    "literal",
    [
        "api_key=fixture-only",
        "api-key=fixture-only",
        "API Key: fixture-only",
        "x-api-key: fixture-only",
        "X_api key=fixture-only",
        "Authorization: Bearer fixture-only",
        "authorization=Bearer fixture-only",
        "cookie=fixture-only",
        "cookie: fixture-only",
        "password=fixture-only",
        "password: fixture-only",
        "token=fixture-only",
        "token: fixture-only",
        "secret=fixture-only",
        "secret: fixture-only",
        "CVPR2026_VIRTUAL_COOKIES_FILE=fixture-only",
    ],
    ids=(
        "api-underscore-equals",
        "api-hyphen-equals",
        "api-space-header",
        "x-api-hyphen-header",
        "x-api-mixed-equals",
        "authorization-header",
        "authorization-query",
        "cookie-equals",
        "cookie-header",
        "password-equals",
        "password-header",
        "token-equals",
        "token-header",
        "secret-equals",
        "secret-header",
        "cvpr-cookie-file",
    ),
)
def test_fixed_receipt_reader_rejects_structured_raw_credentials_without_echoing_them(
    tmp_path: Path, placement: str, literal: str
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    receipt = tmp_path / "receipt.json"
    _write_fixed_receipt(receipt, _receipt_with_placed_string(placement=placement, literal=literal))

    with pytest.raises(cli.R6ExecutionError) as raised:
        cli._read_sanitized_receipt(receipt, label="fixture")

    assert str(raised.value) == "fixture receipt is not sanitized"
    assert literal not in str(raised.value)


@pytest.mark.parametrize("placement", _SANITIZER_PLACEMENTS)
@pytest.mark.parametrize(
    "key",
    (
        "api_key",
        "api-key",
        "API Key",
        "x-api-key",
        "x_api_key",
        "x api key",
    ),
)
def test_fixed_receipt_reader_rejects_finite_api_key_mapping_labels_without_echoing_them(
    tmp_path: Path, placement: str, key: str
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    receipt = tmp_path / "receipt.json"
    _write_fixed_receipt(receipt, _receipt_with_placed_key(placement=placement, key=key))

    with pytest.raises(cli.R6ExecutionError) as raised:
        cli._read_sanitized_receipt(receipt, label="fixture")

    assert str(raised.value) == "fixture receipt is not sanitized"
    assert key not in str(raised.value)


@pytest.mark.parametrize(
    "literal",
    (
        "API key rotation is discussed",
        "the token bucket is empty",
        "secret sauce",
        "password hygiene",
        "bearer instrument",
        "api---key=fixture-only",
        "x__api key: fixture-only",
    ),
)
def test_fixed_receipt_reader_accepts_benign_prose_and_repeated_separator_near_misses(
    tmp_path: Path, literal: str
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    receipt = tmp_path / "receipt.json"
    payload = {"neutral": [{"value": literal}]}
    _write_fixed_receipt(receipt, payload)

    assert cli._read_sanitized_receipt(receipt, label="fixture") == payload


@pytest.mark.parametrize("literal", ("api-key=fixture-only", "x-api-key: fixture-only"))
def test_fixed_execution_rejects_hyphenated_api_key_receipts_before_any_execution_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    """Reported CR-04 forms must stop the actual façade before any live seam."""

    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    contract = _fixed_cli_contract(tmp_path)
    _write_fixed_receipt(
        contract.path("audio_recovery_receipt"),
        {"status": "succeeded", "secrets_copied": False, "neutral": literal},
    )
    admission = SimpleNamespace(
        mode="bootstrap",
        material_policy_digest="a" * 64,
        audio_policy_digest="b" * 64,
        schedule={"fixture": True},
        settled_benchmark_tasks=frozenset(),
        benchmark_task="transcription.session_speech",
        envelope={},
    )
    seam_calls: list[str] = []
    settings_module = ModuleType("conference_pipeline.config")

    def poison_settings() -> object:
        seam_calls.append("settings")
        raise AssertionError("sanitizer must stop before settings")

    def poison_provider(_settings: object) -> object:
        seam_calls.append("provider")
        raise AssertionError("sanitizer must stop before provider construction")

    def poison_executor(**_kwargs: object) -> dict[str, object]:
        seam_calls.append("executor")
        raise AssertionError("sanitizer must stop before execution")

    settings_module.load_settings = poison_settings  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "admit_fixed_r6_pre_provider", lambda **_kwargs: admission)
    monkeypatch.setitem(sys.modules, "conference_pipeline.config", settings_module)
    monkeypatch.setattr(cli, "build_live_r6_provider", poison_provider)
    monkeypatch.setattr(cli, "execute_r6_declared_local", poison_executor)

    with pytest.raises(cli.R6ExecutionError) as raised:
        cli.run_fixed_r6_execution(mode="speech-benchmark-only", contract=contract)

    assert str(raised.value) == "audio receipt is not sanitized"
    assert literal not in str(raised.value)
    assert seam_calls == []


@pytest.mark.parametrize(
    "mode, admitted_mode, benchmark_task",
    [
        ("speech-benchmark-only", "bootstrap", "transcription.session_speech"),
        ("ocr-benchmark-only", "bootstrap", "ocr.slide_image_parse"),
        ("full", "full", None),
    ],
)
def test_fixed_execution_reads_canonical_attestations_before_the_mocked_provider_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    admitted_mode: str,
    benchmark_task: str | None,
) -> None:
    """The actual fixed façade cannot load settings before each receipt is sanitized."""

    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    contract = _fixed_cli_contract(tmp_path)
    events: list[str] = []
    admission = SimpleNamespace(
        mode=admitted_mode,
        material_policy_digest="a" * 64,
        audio_policy_digest="b" * 64,
        schedule={"fixture": True},
        settled_benchmark_tasks=frozenset(),
        benchmark_task=benchmark_task,
        envelope={},
    )
    original_reader = cli._read_sanitized_receipt

    def recorded_reader(path: Path, *, label: str) -> dict[str, object]:
        events.append(f"read:{label}")
        return original_reader(path, label=label)

    settings_module = ModuleType("conference_pipeline.config")

    def fake_load_settings() -> object:
        events.append("settings")
        return object()

    def fake_provider(settings: object) -> object:
        assert settings is not None
        events.append("provider")
        return object()

    def fake_execute(**kwargs: object) -> dict[str, object]:
        assert events[:3] == ["read:audio", "read:material", "read:estimate"]
        assert kwargs["provider"] is not None
        events.append("execute")
        _write_fixed_receipt(contract.path("processing_inventory"), {"status": "succeeded"})
        return {"status": "succeeded", "mode": "fixture"}

    settings_module.load_settings = fake_load_settings  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "admit_fixed_r6_pre_provider", lambda **_kwargs: admission)
    monkeypatch.setattr(cli, "_read_sanitized_receipt", recorded_reader)
    monkeypatch.setitem(sys.modules, "conference_pipeline.config", settings_module)
    monkeypatch.setattr(cli, "build_live_r6_provider", fake_provider)
    monkeypatch.setattr(cli, "execute_r6_declared_local", fake_execute)
    monkeypatch.setattr(cli, "load_registry", lambda: object())

    result = cli.run_fixed_r6_execution(mode=mode, contract=contract)

    assert result["status"] == "succeeded"
    assert events[:5] == [
        "read:audio",
        "read:material",
        "read:estimate",
        "settings",
        "provider",
    ]
    assert events[5:] == ["execute", "read:provider"]


def test_fixed_no_paid_execution_accepts_canonical_attestation_without_settings_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_cvpr2026_r6_transcription_ocr as cli

    contract = _fixed_cli_contract(tmp_path)
    calls: list[str] = []
    admission = SimpleNamespace(
        mode="no-paid-lane",
        material_policy_digest="a" * 64,
        audio_policy_digest="b" * 64,
        schedule=None,
        settled_benchmark_tasks=frozenset(),
        benchmark_task=None,
        envelope={
            "queue_digest": "c" * 64,
            "material_policy_digest": "a" * 64,
            "audio_policy_digest": "b" * 64,
        },
    )
    settings_module = ModuleType("conference_pipeline.config")

    def fail_if_settings_loaded() -> object:
        raise AssertionError("no-paid lane must not load settings")

    def fail_if_provider_built(_settings: object) -> object:
        raise AssertionError("no-paid lane must not construct a provider")

    def fake_no_paid(**kwargs: object) -> dict[str, object]:
        calls.append("no-paid")
        _write_fixed_receipt(contract.path("processing_inventory"), {"status": "succeeded"})
        return {"status": "succeeded", "mode": "no-paid-lane"}

    settings_module.load_settings = fail_if_settings_loaded  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "admit_fixed_r6_pre_provider", lambda **_kwargs: admission)
    monkeypatch.setitem(sys.modules, "conference_pipeline.config", settings_module)
    monkeypatch.setattr(cli, "build_live_r6_provider", fail_if_provider_built)
    monkeypatch.setattr(cli, "execute_r6_no_paid_lane", fake_no_paid)

    result = cli.run_fixed_r6_execution(mode="full", contract=contract)

    assert calls == ["no-paid"]
    assert result["status"] == "succeeded"
    assert result["envelope"]["secrets_copied"] is False
    for key in ("cost_estimate_receipt", "benchmark_receipt", "provider_receipts", "budget_ledger"):
        payload = json.loads(contract.path(key).read_text(encoding="utf-8"))
        assert payload["secrets_copied"] is False


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_admission_receipt_loader_exempts_provider_receipts_from_the_run_id_check(
    tmp_path: Path,
) -> None:
    """``provider_receipts.json`` (label="provider") never carries its own
    ``run_id`` — see ``_write_provider_receipt``, which persists only
    ``receipt_type``/``status``/``receipts``/``secrets_copied``. Its run
    binding instead comes from the budget ledger's schedule/queue-digest
    cross-check in ``validate_r6_budget_ledger``. The generic contract
    ``run_id`` check that every *other* r6 receipt type must satisfy must not
    reject this one for a field its schema was never given."""

    from conference_pipeline.acquisition.cvpr_r6_stage_admission import (
        R6PreProviderAdmissionError,
        _load,
    )

    path = tmp_path / "provider_receipts.json"
    path.write_text(
        json.dumps(
            {
                "receipt_type": "conference_pipeline.r6_provider_receipts",
                "status": "succeeded",
                "receipts": [],
                "secrets_copied": False,
            }
        ),
        encoding="utf-8",
    )

    payload, digest = _load(path, label="provider", required=False, require_run_id=False)
    assert payload is not None and digest is not None

    with pytest.raises(R6PreProviderAdmissionError, match="run id drifted"):
        _load(path, label="provider", required=False)

    # Receipt types that DO carry run_id must still be rejected on a genuine
    # mismatch — this exemption is scoped to require_run_id=False callers only.
    other = tmp_path / "cost_estimate_receipt.json"
    other.write_text(json.dumps({"run_id": "some-other-run"}), encoding="utf-8")
    with pytest.raises(R6PreProviderAdmissionError, match="run id drifted"):
        _load(other, label="estimate")
