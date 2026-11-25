"""v1.61 Phase 66 — MlxVisionServerClient + ManagedMlxVisionServer + vision message helper."""

from __future__ import annotations

import base64
import json
import struct
import sys
import types
from pathlib import Path

import pytest

import conference_pipeline.model_governance.mlx_vision_client as mlx_vision_module
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile
from conference_pipeline.model_governance.mlx_vision_client import (
    ManagedMlxVisionServer,
    MlxVisionServerClient,
    VisionPayloadError,
    VisionRuntimeUnavailableError,
    build_vision_messages,
    build_vision_messages_multi,
    extract_assistant_content,
    vision_endpoint_label,
)
from conference_pipeline.model_governance.registry import ModelGovernanceError
from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock
from conference_pipeline.provider_receipts import (
    LOCAL_VISION_RECEIPT_MARKER,
    LOCAL_VISION_RECEIPT_SCHEMA_VERSION,
    LocalVisionCallReceipt,
    record_local_vision_call,
    scan_local_vision_receipts,
    write_local_vision_receipt,
)


def _make_profile(**overrides) -> LocalRuntimeProfile:
    defaults = dict(
        profile_id="mlx_vision_light",
        runtime_kind="mlx",
        base_url="http://127.0.0.1:8090/v1",
        manage_server=True,
        optional_deps=["mlx", "mlx_vlm"],
        startup_timeout_s=5.0,
        request_timeout_s=60.0,
        max_concurrency=2,
        max_batch_size=1,
        lock_scope="light",
        hardware_fit_status="unknown",
        approval_status="benchmark_pending",
    )
    defaults.update(overrides)
    return LocalRuntimeProfile(**defaults)


def _minimal_png(width: int, height: int) -> bytes:
    """Produce a byte string with a valid PNG header reporting given dimensions.

    The IDAT/IEND chunks are not full PNG-spec valid (no zlib stream), but the
    8-byte signature plus IHDR is enough for the dimension parser tested here.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_payload = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return sig + ihdr_len + ihdr_type + ihdr_payload + b"\x00\x00\x00\x00"


def _minimal_jpeg(width: int, height: int) -> bytes:
    """JPEG byte string with a SOF0 marker reporting given dimensions."""
    soi = b"\xff\xd8"
    sof0 = b"\xff\xc0"
    seg_len = struct.pack(">H", 17)  # arbitrary >= 9
    precision = b"\x08"
    h_w = struct.pack(">HH", height, width)
    components = b"\x03" + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return soi + sof0 + seg_len + precision + h_w + components + b"\xff\xd9"


def _minimal_webp_vp8(width: int, height: int) -> bytes:
    """WebP (VP8 simple lossy) byte string with given dimensions.

    Layout: RIFF(4) size(4) WEBP(4) VP8 (4) chunk_size(4) [3 frame-tag bytes,
    3 sync code bytes, width(2 LE), height(2 LE), filler]. Width/height start
    at file offset 26.
    """
    payload = b"\x00\x00\x00"  # 3 bytes frame tag (positions 0-2)
    payload += b"\x9d\x01\x2a"  # 3 bytes VP8 sync code (positions 3-5)
    payload += struct.pack("<H", width & 0x3FFF)  # positions 6-7
    payload += struct.pack("<H", height & 0x3FFF)  # positions 8-9
    payload += b"\x00" * 6
    chunk = b"VP8 " + struct.pack("<I", len(payload)) + payload
    riff_size = struct.pack("<I", len(chunk) + 4)
    return b"RIFF" + riff_size + b"WEBP" + chunk


# ---------------------------------------------------------------------------
# Vision message helper
# ---------------------------------------------------------------------------


def test_build_vision_messages_png_includes_data_uri_and_dims():
    image = _minimal_png(640, 480)
    messages, dims = build_vision_messages(
        image_bytes=image, media_type="image/png", prompt="describe"
    )
    assert dims == (640, 480)
    assert messages[-1]["role"] == "user"
    blocks = messages[-1]["content"]
    image_block = blocks[0]
    assert image_block["type"] == "image_url"
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    encoded_b64 = url.split(",", 1)[1]
    assert base64.b64decode(encoded_b64) == image
    text_block = blocks[1]
    assert text_block == {"type": "text", "text": "describe"}


def test_build_vision_messages_jpeg_dims():
    image = _minimal_jpeg(800, 600)
    _, dims = build_vision_messages(
        image_bytes=image, media_type="image/jpeg", prompt="describe"
    )
    assert dims == (800, 600)


def test_build_vision_messages_webp_dims():
    image = _minimal_webp_vp8(123, 456)
    _, dims = build_vision_messages(
        image_bytes=image, media_type="image/webp", prompt="describe"
    )
    assert dims == (123, 456)


def test_build_vision_messages_rejects_unsupported_media_type():
    with pytest.raises(VisionPayloadError, match="media_type"):
        build_vision_messages(
            image_bytes=b"x", media_type="image/gif", prompt="hi"
        )


def test_build_vision_messages_rejects_empty_image():
    with pytest.raises(VisionPayloadError, match="non-empty"):
        build_vision_messages(image_bytes=b"", media_type="image/png", prompt="hi")


def test_build_vision_messages_rejects_empty_prompt():
    with pytest.raises(VisionPayloadError, match="non-empty"):
        build_vision_messages(
            image_bytes=_minimal_png(2, 2), media_type="image/png", prompt="   "
        )


def test_build_vision_messages_includes_system_prompt_when_supplied():
    image = _minimal_png(2, 2)
    messages, _ = build_vision_messages(
        image_bytes=image,
        media_type="image/png",
        prompt="user prompt",
        system_prompt="system prompt",
    )
    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1]["role"] == "user"


def test_build_vision_messages_multi_packs_all_images():
    images = [
        (_minimal_png(10, 20), "image/png"),
        (_minimal_jpeg(100, 200), "image/jpeg"),
    ]
    messages, dims = build_vision_messages_multi(images=images, prompt="compare")
    assert dims == [(10, 20), (100, 200)]
    blocks = messages[-1]["content"]
    assert sum(1 for b in blocks if b["type"] == "image_url") == 2
    assert blocks[-1] == {"type": "text", "text": "compare"}


def test_build_vision_messages_multi_rejects_empty_image_list():
    with pytest.raises(VisionPayloadError, match="at least one image"):
        build_vision_messages_multi(images=[], prompt="x")


# ---------------------------------------------------------------------------
# Endpoint label
# ---------------------------------------------------------------------------


def test_vision_endpoint_label_is_redacted():
    label = vision_endpoint_label("mlx-community/Qwen3-VL-8B-Instruct-4bit")
    assert label == "mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert "://" in label
    assert "8090" not in label  # never a port number
    assert "127." not in label  # never a host


# ---------------------------------------------------------------------------
# MlxVisionServerClient
# ---------------------------------------------------------------------------


def test_client_rejects_non_loopback():
    with pytest.raises(ModelGovernanceError, match="loopback"):
        MlxVisionServerClient("http://192.168.1.1:8090/v1")


def test_client_health_check_returns_false_on_refused():
    client = MlxVisionServerClient("http://127.0.0.1:19999/v1")
    assert client.health_check() is False


def test_client_health_check_accepts_root_base_url_v1_models(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(url, *, timeout):
        calls.append(url)
        if url == "http://127.0.0.1:8090/v1/models":
            return FakeResponse()
        raise mlx_vision_module.URLError("not found")

    monkeypatch.setattr(mlx_vision_module, "urlopen", fake_urlopen)

    client = MlxVisionServerClient("http://127.0.0.1:8090")

    assert client.health_check() is True
    assert calls == ["http://127.0.0.1:8090/v1/models"]


def test_client_chat_complete_raises_on_refused():
    client = MlxVisionServerClient("http://127.0.0.1:19999/v1", timeout_s=2.0)
    messages, _ = build_vision_messages(
        image_bytes=_minimal_png(2, 2), media_type="image/png", prompt="x"
    )
    with pytest.raises(ModelGovernanceError):
        client.chat_complete(messages, model="fake-vision-model")


def test_extract_assistant_content_string():
    payload = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    assert extract_assistant_content(payload, base_url="http://127.0.0.1:0") == "hello"


def test_extract_assistant_content_list_blocks():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "line 1\n"},
                        {"type": "text", "text": "line 2"},
                    ]
                }
            }
        ]
    }
    assert extract_assistant_content(payload, base_url="http://127.0.0.1:0") == "line 1\nline 2"


def test_extract_assistant_content_raises_on_empty():
    with pytest.raises(ModelGovernanceError, match="malformed|no chat choices"):
        extract_assistant_content({"choices": []}, base_url="http://127.0.0.1:0")


# ---------------------------------------------------------------------------
# ManagedMlxVisionServer lifecycle
# ---------------------------------------------------------------------------


def test_managed_vision_server_rejects_non_loopback():
    profile = _make_profile(base_url="http://192.168.1.1:8090/v1")
    server = ManagedMlxVisionServer()
    with pytest.raises(ModelGovernanceError, match="loopback"):
        server.start(profile, "any-vision-model")


def test_managed_vision_server_rejects_manage_false():
    profile = _make_profile(manage_server=False)
    server = ManagedMlxVisionServer()
    with pytest.raises(ModelGovernanceError):
        server.start(profile, "any-vision-model")


def test_managed_vision_server_stop_noop_when_no_state(tmp_path):
    server = ManagedMlxVisionServer(state_path=tmp_path / "absent.json")
    server.stop()  # must not raise


def test_managed_vision_server_status_absent_state(tmp_path):
    server = ManagedMlxVisionServer(state_path=tmp_path / "absent.json")
    result = server.status()
    assert result["pid"] is None
    assert result["model_id"] is None
    assert result["base_url"] is None
    assert result["health"] is False


def test_managed_vision_server_raises_when_no_runtime(tmp_path, monkeypatch):
    profile = _make_profile()
    server = ManagedMlxVisionServer(
        state_path=tmp_path / "state.json",
        entrypoint_candidates=(("-m", "package_that_does_not_exist_xyz"),),
    )
    with pytest.raises(VisionRuntimeUnavailableError, match="mlx_vlm|vllm_mlx"):
        server.start(profile, "any-vision-model")


def test_managed_vision_server_start_then_stop_with_mocked_subprocess(tmp_path, monkeypatch):
    """Lifecycle round-trip: lazy-import + spawn + state file + stop releases lock."""

    class FakeProcess:
        pid = 999998

        def terminate(self):
            pass

    fake_module_name = "mlx_vlm"
    monkeypatch.setitem(sys.modules, fake_module_name, types.ModuleType(fake_module_name))
    # find_spec uses sys.modules lookup; provide a Spec sentinel.
    import importlib.machinery as machinery

    fake_spec = machinery.ModuleSpec(fake_module_name, loader=None)
    sys.modules[fake_module_name].__spec__ = fake_spec
    monkeypatch.setattr(mlx_vision_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(mlx_vision_module.MlxVisionServerClient, "health_check", lambda self: True)
    # _pid_alive considers our fake pid alive only briefly; force it to False so stop cleans up.
    monkeypatch.setattr(mlx_vision_module, "_pid_alive", lambda pid: False)

    state_path = tmp_path / "vision_state.json"
    lock = HeavyRuntimeLock(tmp_path / "lock.json")
    profile = _make_profile()
    server = ManagedMlxVisionServer(state_path=state_path)

    server.start(profile, "mlx-community/Qwen3-VL-8B-Instruct-4bit", lock=lock, task_id="vision.slide_analysis")

    assert state_path.exists()
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["pid"] == FakeProcess.pid
    assert state_payload["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert state_payload["base_url"] == profile.base_url
    assert state_payload["entrypoint"] == ["-m", "mlx_vlm.server"]

    status = server.status()
    assert status["pid"] == FakeProcess.pid
    assert status["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert status["lock_owner"]["task_id"] == "vision.slide_analysis"

    server.stop()
    assert not state_path.exists()
    assert lock.owner_metadata() is None


def test_managed_vision_server_uses_first_available_entrypoint(tmp_path, monkeypatch):
    """When mlx_vlm is missing but vllm_mlx exists, fall back to vllm_mlx."""

    class FakeProcess:
        pid = 999997

        def terminate(self):
            pass

    fake_name = "vllm_mlx"
    monkeypatch.setitem(sys.modules, fake_name, types.ModuleType(fake_name))
    import importlib.machinery as machinery

    sys.modules[fake_name].__spec__ = machinery.ModuleSpec(fake_name, loader=None)
    # Make sure mlx_vlm is NOT available.
    sys.modules.pop("mlx_vlm", None)
    monkeypatch.setattr(
        mlx_vision_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    monkeypatch.setattr(mlx_vision_module.MlxVisionServerClient, "health_check", lambda self: True)
    monkeypatch.setattr(mlx_vision_module, "_pid_alive", lambda pid: False)

    state_path = tmp_path / "vision_state.json"
    profile = _make_profile()
    server = ManagedMlxVisionServer(state_path=state_path)

    # Force find_spec to claim mlx_vlm absent and vllm_mlx present.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, package=None):
        if name == "mlx_vlm":
            return None
        if name == "vllm_mlx":
            return machinery.ModuleSpec("vllm_mlx", loader=None)
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)

    server.start(profile, "model-x")
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["entrypoint"] == ["-m", "vllm_mlx.server"]
    server.stop()


# ---------------------------------------------------------------------------
# LocalVisionCallReceipt
# ---------------------------------------------------------------------------


def _make_receipt(**overrides) -> LocalVisionCallReceipt:
    defaults = dict(
        schema_version=LOCAL_VISION_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_VISION_RECEIPT_MARKER,
        provider="local_vlm",
        runtime_kind="vllm_mlx",
        model="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        endpoint_label="mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit",
        vision_task="slide_analysis",
        image_count=1,
        image_dimensions=[[640, 480]],
        image_bytes_size=12345,
        prompt_token_estimate=42,
        completion_token_estimate=128,
        latency_ms=1234.5,
        cost_usd=0.0,
        estimated_paid_equivalent_usd=0.0123,
        timestamp="2026-05-03T12:34:56+00:00",
        run_id="run-abc",
        stage_id="presentation",
        receipt_path=None,
    )
    defaults.update(overrides)
    return LocalVisionCallReceipt(**defaults)


def test_receipt_round_trip_json():
    receipt = _make_receipt()
    payload = receipt.to_dict()
    assert payload["receipt_type"] == LOCAL_VISION_RECEIPT_MARKER
    assert payload["schema_version"] == LOCAL_VISION_RECEIPT_SCHEMA_VERSION
    assert payload["cost_usd"] == 0.0
    assert payload["image_dimensions"] == [[640, 480]]
    # JSON serializable
    json.dumps(payload)


def test_receipt_endpoint_label_is_redacted():
    receipt = _make_receipt(endpoint_label="mlx-vision://model-id")
    payload = receipt.to_dict()
    label = payload["endpoint_label"]
    assert label.startswith("mlx-vision://")
    assert "127." not in label
    assert "8090" not in label
    assert "v1" not in label.split("://", 1)[-1].split("/")[0]


def test_write_local_vision_receipt_with_run_id(tmp_path):
    workspace = tmp_path
    receipt = _make_receipt(run_id="run-001", stage_id="presentation")
    path = write_local_vision_receipt(receipt, workspace)
    assert path.exists()
    expected_dir = (
        workspace / "storage" / "workflow-runs" / "run-001"
        / "local-vision-receipts" / "presentation"
    )
    assert path.parent == expected_dir
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-001"


def test_write_local_vision_receipt_without_run_id(tmp_path):
    workspace = tmp_path
    receipt = _make_receipt(run_id=None, stage_id="presentation")
    path = write_local_vision_receipt(receipt, workspace)
    expected_dir = workspace / "storage" / "local-vision-receipts" / "presentation"
    assert path.parent == expected_dir


def test_record_local_vision_call_no_workspace_returns_none():
    result = record_local_vision_call(
        provider="local_vlm",
        runtime_kind="vllm_mlx",
        model="m",
        endpoint_label="mlx-vision://m",
        vision_task="slide_analysis",
        image_count=1,
        image_dimensions=[(2, 2)],
        image_bytes_size=128,
    )
    assert result is None


def test_record_local_vision_call_writes_receipt(tmp_path):
    path = record_local_vision_call(
        provider="local_vlm",
        runtime_kind="vllm_mlx",
        model="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        endpoint_label="mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit",
        vision_task="slide_analysis",
        image_count=1,
        image_dimensions=[(640, 480)],
        image_bytes_size=2048,
        prompt_token_estimate=40,
        completion_token_estimate=200,
        latency_ms=900.5,
        run_id="run-xyz",
        stage_id="vision-bench",
        workspace=tmp_path,
    )
    assert path is not None
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["receipt_type"] == LOCAL_VISION_RECEIPT_MARKER
    assert payload["cost_usd"] == 0.0
    assert payload["image_dimensions"] == [[640, 480]]


def test_scan_local_vision_receipts_finds_recent_writes(tmp_path):
    record_local_vision_call(
        provider="local_vlm",
        runtime_kind="vllm_mlx",
        model="m1",
        endpoint_label="mlx-vision://m1",
        vision_task="slide_analysis",
        image_count=1,
        image_dimensions=[(2, 2)],
        image_bytes_size=128,
        run_id="run-1",
        stage_id="stage-a",
        workspace=tmp_path,
    )
    record_local_vision_call(
        provider="local_fastvlm",
        runtime_kind="mlx_native_fastvlm",
        model="apple/FastVLM-7B-MLX",
        endpoint_label="mlx-fastvlm://apple/FastVLM-7B-MLX",
        vision_task="figure_caption",
        image_count=1,
        image_dimensions=[(2, 2)],
        image_bytes_size=128,
        run_id="run-2",
        stage_id="stage-b",
        workspace=tmp_path,
    )
    scanned = scan_local_vision_receipts(tmp_path)
    assert len(scanned) == 2
    assert {r["model"] for r in scanned} == {"m1", "apple/FastVLM-7B-MLX"}
    assert all(r["receipt_type"] == LOCAL_VISION_RECEIPT_MARKER for r in scanned)
