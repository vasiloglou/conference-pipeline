"""v1.61 Phase 66 — Loopback vision-server client + lifecycle for vllm-mlx / mlx-vlm.server.

Module-level imports: stdlib only + ``ModelGovernanceError`` and ``is_loopback_url``.
``mlx_vlm`` / ``mlx`` are imported lazily inside ``ManagedMlxVisionServer.start``
to preserve the MLX-free import contract.

Mirrors the v1.9 ``ManagedMlxServer`` shape but on a vision-specific port,
with vision-specific entrypoints (``mlx_vlm.server`` first, ``vllm_mlx``
fallback) and an OpenAI-compatible chat client that accepts ``image_url``
data-URI content blocks.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.model_governance.registry import ModelGovernanceError

if TYPE_CHECKING:
    from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock


_SUPPORTED_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")
_DEFAULT_VISION_ENTRYPOINT_CANDIDATES: tuple[tuple[str, ...], ...] = (
    # Preferred: mlx_vlm.server (community-maintained, OpenAI-compatible).
    ("-m", "mlx_vlm.server"),
    # Fallback: vllm_mlx (provides /v1/chat/completions with vision).
    ("-m", "vllm_mlx.server"),
)


class VisionPayloadError(ModelGovernanceError):
    """Raised when image bytes or media types are not in supported shape."""


class VisionRuntimeUnavailableError(ModelGovernanceError):
    """Raised when no vllm-mlx / mlx-vlm runtime is available locally."""


def vision_endpoint_label(model_id: str) -> str:
    """Return the redacted endpoint label for a vision call.

    ``mlx-vision://<model-id>`` for any vllm-mlx/mlx-vlm-served model.
    Receipts use this string as ``endpoint_label`` — never the full URL.
    """
    return f"mlx-vision://{model_id}"


def _detect_image_dimensions(image_bytes: bytes, media_type: str) -> tuple[int, int]:
    """Return (width, height) parsed from the image header. No PIL dependency.

    Supports PNG, JPEG, WebP minimally — enough to populate receipts. On parse
    failure returns (0, 0); callers must not depend on dimensions for security.
    """
    try:
        if media_type == "image/png":
            # PNG: 8-byte signature, then IHDR chunk: length(4) type(4) width(4) height(4)...
            if len(image_bytes) >= 24 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", image_bytes[16:24])
                return int(w), int(h)
        elif media_type == "image/jpeg":
            # JPEG: scan for SOF0..SOF15 markers (0xFFC0..0xFFCF, excluding 0xFFC4/8/C).
            i = 0
            n = len(image_bytes)
            if n < 4 or image_bytes[0:2] != b"\xff\xd8":
                return 0, 0
            i = 2
            while i + 9 < n:
                if image_bytes[i] != 0xFF:
                    i += 1
                    continue
                marker = image_bytes[i + 1]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", image_bytes[i + 5 : i + 9])
                    return int(w), int(h)
                if marker in (0xD8, 0xD9):
                    i += 2
                    continue
                seg_len = struct.unpack(">H", image_bytes[i + 2 : i + 4])[0]
                i += 2 + seg_len
        elif media_type == "image/webp":
            # WebP: RIFF....WEBPVP8 / VP8L / VP8X
            if len(image_bytes) >= 30 and image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
                chunk = image_bytes[12:16]
                if chunk == b"VP8 ":
                    w = struct.unpack("<H", image_bytes[26:28])[0] & 0x3FFF
                    h = struct.unpack("<H", image_bytes[28:30])[0] & 0x3FFF
                    return int(w), int(h)
                if chunk == b"VP8L":
                    b1, b2, b3, b4 = image_bytes[21:25]
                    w = 1 + (((b2 & 0x3F) << 8) | b1)
                    h = 1 + (((b4 & 0x0F) << 10) | (b3 << 2) | ((b2 & 0xC0) >> 6))
                    return int(w), int(h)
                if chunk == b"VP8X":
                    w = 1 + (image_bytes[24] | (image_bytes[25] << 8) | (image_bytes[26] << 16))
                    h = 1 + (image_bytes[27] | (image_bytes[28] << 8) | (image_bytes[29] << 16))
                    return int(w), int(h)
    except Exception:
        return 0, 0
    return 0, 0


def build_vision_messages(
    *,
    image_bytes: bytes,
    media_type: str,
    prompt: str,
    system_prompt: str | None = None,
) -> tuple[list[dict], tuple[int, int]]:
    """Build OpenAI-compatible chat ``messages`` with one image + text block.

    Returns ``(messages, (width, height))``. Image bytes are base64-encoded
    inline; they are never written to disk by this function and the caller
    must ensure the bytes themselves never enter logs. ``media_type`` must be
    one of ``image/png``, ``image/jpeg``, ``image/webp``.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise VisionPayloadError("image_bytes must be a non-empty bytes object")
    if media_type not in _SUPPORTED_MEDIA_TYPES:
        raise VisionPayloadError(
            f"media_type must be one of {_SUPPORTED_MEDIA_TYPES}, got {media_type!r}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise VisionPayloadError("prompt must be a non-empty string")

    encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }
    text_block = {"type": "text", "text": prompt}
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [image_block, text_block],
        }
    )
    dims = _detect_image_dimensions(bytes(image_bytes), media_type)
    return messages, dims


def build_vision_messages_multi(
    *,
    images: Iterable[tuple[bytes, str]],
    prompt: str,
    system_prompt: str | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Variant of ``build_vision_messages`` for multi-image calls.

    Provided for forward-compat — v1.61 single-image-per-call is the
    contract — but the helper is here so v1.7's multi-image extensions
    don't need to reinvent the structure.
    """
    image_blocks: list[dict] = []
    dims_list: list[tuple[int, int]] = []
    for image_bytes, media_type in images:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise VisionPayloadError("image_bytes must be a non-empty bytes object")
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise VisionPayloadError(
                f"media_type must be one of {_SUPPORTED_MEDIA_TYPES}, got {media_type!r}"
            )
        encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
        image_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        )
        dims_list.append(_detect_image_dimensions(bytes(image_bytes), media_type))
    if not image_blocks:
        raise VisionPayloadError("multi-image call requires at least one image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VisionPayloadError("prompt must be a non-empty string")
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [*image_blocks, {"type": "text", "text": prompt}],
        }
    )
    return messages, dims_list


class MlxVisionServerClient:
    """OpenAI-compatible vision chat client for a local mlx-vlm/vllm-mlx server.

    Only stdlib ``urllib`` is used at call time. Loopback enforcement is
    applied at construction so non-loopback ``base_url`` fails fast.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        if not is_loopback_url(base_url):
            raise ModelGovernanceError(
                f"MlxVisionServerClient requires a loopback base_url; got {base_url!r}. "
                "Non-loopback vision endpoints fail closed."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat_complete(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """POST to ``{base_url}/chat/completions`` with vision messages.

        Returns the parsed JSON response (assistant content access is
        delegated to the caller — slide-analysis vs OCR have different
        downstream parsing needs).
        """
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs,
        }
        payload = json.dumps(body).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                status = resp.status
                response_body = resp.read()
        except URLError as exc:
            raise ModelGovernanceError(
                f"vision server at {self.base_url} unreachable: {exc}"
            ) from exc
        if status != 200:
            raise ModelGovernanceError(
                f"vision server at {self.base_url} returned HTTP {status}"
            )
        try:
            return json.loads(response_body)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"vision server at {self.base_url} returned invalid JSON"
            ) from exc

    def health_check(self) -> bool:
        """GET /v1/models or /health; True on 200, False otherwise. Never raises."""
        paths = ["/models", "/health"]
        if not self.base_url.endswith("/v1"):
            paths.insert(0, "/v1/models")
        for path in paths:
            try:
                with urlopen(f"{self.base_url}{path}", timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        return False


def extract_assistant_content(payload: dict, *, base_url: str) -> str:
    """Pull the assistant message string out of an OpenAI-style chat payload."""
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelGovernanceError(
            f"vision server at {base_url} returned no chat choices"
        ) from exc
    message = choice.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            # Some servers return list-of-blocks; join text blocks.
            text_parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "".join(text_parts).strip()
            if joined:
                return joined
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise ModelGovernanceError(
        f"vision server at {base_url} returned a malformed chat completion payload"
    )


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8090
    return host, port


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ManagedMlxVisionServer:
    """Manages the lifecycle of a vision-server subprocess (mlx_vlm.server / vllm_mlx.server).

    Mirrors :class:`conference_pipeline.model_governance.mlx_client.ManagedMlxServer`
    but on a vision-specific port and with vision-specific entrypoints. State
    file is secret-safe: contains pid, model_id, base_url, started_at only.
    """

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        entrypoint_candidates: tuple[tuple[str, ...], ...] = _DEFAULT_VISION_ENTRYPOINT_CANDIDATES,
    ) -> None:
        self._state_path = state_path or (
            Path(tempfile.gettempdir()) / "conference_pipeline_mlx_vision_state.json"
        )
        self._entrypoint_candidates = entrypoint_candidates

    def start(
        self,
        profile,
        model_id: str,
        *,
        lock: "HeavyRuntimeLock | None" = None,
        task_id: str = "managed_mlx_vision_server",
    ) -> None:
        """Start a managed vision-server subprocess for the given profile and model.

        Validates: ``manage_server=True``, loopback ``base_url``, ``mlx_vlm`` or
        ``vllm_mlx`` available. Polls health_check until ``startup_timeout_s``,
        then raises ``ModelGovernanceError``.
        """
        if not profile.manage_server:
            raise ModelGovernanceError(
                "ManagedMlxVisionServer.start() requires manage_server=True in the profile"
            )
        if not is_loopback_url(profile.base_url):
            raise ModelGovernanceError(
                f"Managed vision server requires a loopback base_url; got {profile.base_url!r}. "
                "Non-loopback managed endpoints fail closed."
            )
        entrypoint = self._select_available_entrypoint()

        lock_owner: dict | None = None
        lock_path: str | None = None
        if lock is not None:
            lock_owner = {
                "task_id": task_id,
                "profile_id": getattr(profile, "profile_id", "unknown"),
                "model_id": model_id,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            lock.acquire(lock_owner)
            lock_path = str(getattr(lock, "_lock_path", ""))

        host, port = _parse_host_port(profile.base_url)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    *entrypoint,
                    "--model",
                    model_id,
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            if lock is not None:
                lock.release()
            raise
        client = MlxVisionServerClient(profile.base_url, timeout_s=2.0)
        deadline = time.monotonic() + profile.startup_timeout_s
        healthy = False
        while time.monotonic() < deadline:
            if client.health_check():
                healthy = True
                break
            time.sleep(0.5)
        if not healthy:
            try:
                process.terminate()
            except OSError:
                pass
            if lock is not None:
                lock.release()
            raise ModelGovernanceError(
                f"Timed out starting managed MLX vision server for model {model_id!r} "
                f"at {profile.base_url} after {profile.startup_timeout_s}s"
            )
        self._write_state(
            pid=process.pid,
            model_id=model_id,
            base_url=profile.base_url,
            entrypoint=list(entrypoint),
            lock_owner=lock_owner,
            lock_path=lock_path,
        )

    def stop(self) -> None:
        """SIGTERM/SIGKILL the managed PID; remove state file. Idempotent."""
        state = self._read_state()
        if state is None:
            return
        pid = state.get("pid")
        if pid and _pid_alive(int(pid)):
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if not _pid_alive(int(pid)):
                    break
                time.sleep(0.2)
            if _pid_alive(int(pid)):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
        lock_path = state.get("lock_path")
        if lock_path:
            from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock

            HeavyRuntimeLock(Path(str(lock_path))).release()
        self._state_path.unlink(missing_ok=True)

    def switch_model(
        self,
        model_id: str,
        profile,
        *,
        lock: "HeavyRuntimeLock | None" = None,
        task_id: str = "managed_mlx_vision_server",
    ) -> None:
        self.stop()
        self.start(profile, model_id, lock=lock, task_id=task_id)

    def status(self) -> dict:
        """Return secret-safe status dict: pid, model_id, base_url, health, started_at, lock_owner."""
        state = self._read_state()
        if state is None:
            return {
                "pid": None,
                "model_id": None,
                "base_url": None,
                "health": False,
                "started_at": None,
                "lock_owner": None,
            }
        base_url = state.get("base_url")
        health = False
        if base_url:
            try:
                health = MlxVisionServerClient(base_url, timeout_s=2.0).health_check()
            except ModelGovernanceError:
                health = False
        return {
            "pid": state.get("pid"),
            "model_id": state.get("model_id"),
            "base_url": base_url,
            "health": health,
            "started_at": state.get("started_at"),
            "lock_owner": state.get("lock_owner"),
        }

    # --------------------------------------------------------- private helpers

    def _select_available_entrypoint(self) -> tuple[str, ...]:
        """Return the first vision-server entrypoint whose package imports.

        Imports are deferred to call time so module import does not require
        ``mlx_vlm``. ``importlib.util.find_spec`` is used to avoid actually
        executing the package, in case import has heavy side effects.
        """
        import importlib.util

        for entrypoint in self._entrypoint_candidates:
            # entrypoint is like ("-m", "mlx_vlm.server")
            if len(entrypoint) >= 2 and entrypoint[0] == "-m":
                module_path = entrypoint[1]
                top = module_path.split(".", 1)[0]
                if importlib.util.find_spec(top) is not None:
                    return entrypoint
        raise VisionRuntimeUnavailableError(
            "neither mlx_vlm nor vllm_mlx is installed; "
            "install one (e.g., `pip install conference-pipeline[mlx]`) before "
            "starting a managed vision server"
        )

    def _write_state(
        self,
        *,
        pid: int,
        model_id: str,
        base_url: str,
        entrypoint: list[str],
        lock_owner: dict | None = None,
        lock_path: str | None = None,
    ) -> None:
        payload: dict = {
            "pid": pid,
            "model_id": model_id,
            "base_url": base_url,
            "entrypoint": entrypoint,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if lock_owner is not None:
            payload["lock_owner"] = lock_owner
        if lock_path:
            payload["lock_path"] = lock_path
        self._state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_state(self) -> dict | None:
        if not self._state_path.exists():
            return None
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
