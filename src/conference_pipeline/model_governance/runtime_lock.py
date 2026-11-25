"""Non-blocking file-based heavy runtime lock for serializing large model workloads.

Acquiring when the lock is already held raises ModelGovernanceError immediately
with owner metadata — no waiting, no queue.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from conference_pipeline.model_governance.registry import ModelGovernanceError

_SECRET_KEY_PATTERNS = {"key", "token", "secret", "password"}

_REQUIRED_METADATA_KEYS = {"task_id", "profile_id", "model_id", "pid", "started_at"}


def _has_secret_key(metadata: dict) -> bool:
    for key in metadata:
        lower = key.lower()
        if any(pat in lower for pat in _SECRET_KEY_PATTERNS):
            return True
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class HeavyRuntimeLock:
    """File-based non-blocking lock for heavy local-model jobs.

    Metadata written to the lock file is audit-friendly and secret-safe.
    """

    def __init__(self, lock_path: Path | None = None) -> None:
        self._lock_path = lock_path or (
            Path(tempfile.gettempdir()) / "conference_pipeline_heavy_lock.json"
        )

    def acquire(self, metadata: dict) -> None:
        missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing:
            raise ValueError(f"lock metadata missing required keys: {missing}")
        if _has_secret_key(metadata):
            raise ValueError(
                "lock metadata must not contain secret-like keys "
                "(key, token, secret, password)"
            )

        if self._lock_path.exists():
            try:
                existing = json.loads(self._lock_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            existing_pid = existing.get("pid")
            if existing_pid and _pid_alive(int(existing_pid)):
                raise ModelGovernanceError(
                    f"heavy runtime lock held by task {existing.get('task_id')} "
                    f"/ pid {existing_pid} since {existing.get('started_at')}; "
                    "use release() when done"
                )

        tmp = self._lock_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        tmp.rename(self._lock_path)

    def is_held(self) -> bool:
        if not self._lock_path.exists():
            return False
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        pid = data.get("pid")
        return bool(pid and _pid_alive(int(pid)))

    def release(self) -> None:
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def owner_metadata(self) -> dict | None:
        if not self._lock_path.exists():
            return None
        try:
            return json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
