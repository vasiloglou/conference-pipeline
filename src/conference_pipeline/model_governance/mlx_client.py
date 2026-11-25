"""OpenAI-compatible MLX client and managed loopback server lifecycle.

Module-level imports: stdlib only + ModelGovernanceError.
mlx / mlx_lm are imported lazily inside functions that execute MLX work.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.model_governance.registry import ModelGovernanceError

if TYPE_CHECKING:
    from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock


class MlxServerClient:
    """OpenAI-compatible completions client for a local MLX server.

    Uses urllib.request (stdlib) only — no openai package required at import time.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512, **kwargs) -> str:
        """POST to {base_url}/chat/completions; return assistant content string."""
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }).encode("utf-8")
        req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                status = resp.status
                body = resp.read()
        except URLError as exc:
            raise ModelGovernanceError(f"MLX server at {self.base_url} unreachable: {exc}") from exc
        if status != 200:
            raise ModelGovernanceError(
                f"MLX server at {self.base_url} returned HTTP {status}"
            )
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"MLX server at {self.base_url} returned invalid JSON"
            ) from exc
        return _extract_assistant_content(data, base_url=self.base_url)

    def health_check(self) -> bool:
        """GET /v1/models or /health; return True on 200, False otherwise. Never raises."""
        for path in ("/v1/models", "/health"):
            try:
                with urlopen(f"{self.base_url}{path}", timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        return False


def _extract_assistant_content(payload: dict, *, base_url: str) -> str:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelGovernanceError(
            f"MLX server at {base_url} returned no chat choices"
        ) from exc
    message = choice.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise ModelGovernanceError(
        f"MLX server at {base_url} returned a malformed chat completion payload"
    )


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    return host, port


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ManagedMlxServer:
    """Manages the lifecycle of a mlx_lm.server subprocess.

    State file is secret-safe: contains pid, model_id, base_url, started_at only.
    mlx / mlx_lm imports are deferred to start() to preserve the MLX-free import contract.
    """

    def __init__(self, *, state_path: Path | None = None) -> None:
        self._state_path = state_path or (
            Path(tempfile.gettempdir()) / "conference_pipeline_mlx_state.json"
        )

    def start(
        self,
        profile,
        model_id: str,
        *,
        lock: "HeavyRuntimeLock | None" = None,
        task_id: str = "managed_mlx_server",
    ) -> None:
        """Start a managed mlx_lm.server subprocess for the given profile and model.

        Validates: manage_server=True, loopback base_url, optional deps available.
        Polls health_check until startup_timeout_s, then raises ModelGovernanceError.
        """
        if not profile.manage_server:
            raise ModelGovernanceError(
                "ManagedMlxServer.start() requires manage_server=True in the profile"
            )
        if not is_loopback_url(profile.base_url):
            raise ModelGovernanceError(
                f"Managed server mode requires a loopback base_url; got {profile.base_url!r}. "
                "Non-loopback managed endpoints fail closed."
            )
        try:
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise ModelGovernanceError(
                "mlx_lm not installed; install optional dependency for local MLX runtime"
            ) from exc

        lock_owner = None
        lock_path = None
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
                    "-m",
                    "mlx_lm.server",
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
        client = MlxServerClient(profile.base_url, timeout_s=2.0)
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
                f"Timed out starting managed MLX server for model {model_id!r} "
                f"at {profile.base_url} after {profile.startup_timeout_s}s"
            )

        self._write_state(
            pid=process.pid,
            model_id=model_id,
            base_url=profile.base_url,
            lock_owner=lock_owner,
            lock_path=lock_path,
        )

    def stop(self) -> None:
        """SIGTERM/SIGKILL the managed PID; remove state file. Swallows if not running."""
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
        task_id: str = "managed_mlx_server",
    ) -> None:
        """Stop current server and start a new one with the given model_id."""
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
            health = MlxServerClient(base_url, timeout_s=2.0).health_check()
        return {
            "pid": state.get("pid"),
            "model_id": state.get("model_id"),
            "base_url": base_url,
            "health": health,
            "started_at": state.get("started_at"),
            "lock_owner": state.get("lock_owner"),
        }

    # ------------------------------------------------------------------ private

    def _write_state(
        self,
        *,
        pid: int,
        model_id: str,
        base_url: str,
        lock_owner: dict | None = None,
        lock_path: str | None = None,
    ) -> None:
        payload = {
            "pid": pid,
            "model_id": model_id,
            "base_url": base_url,
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
