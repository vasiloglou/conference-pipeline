"""Typed local runtime profile contract for conference-pipeline local LLM execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LocalRuntimeProfile:
    """Typed profile describing a local LLM runtime endpoint and execution policy.

    All fields required. No MLX or runtime imports here — this module is pure
    dataclass construction from YAML, safe to import in any environment.
    """

    profile_id: str
    runtime_kind: str          # "mlx" | "ollama" | "custom"
    base_url: str              # e.g. "http://127.0.0.1:8080/v1"
    manage_server: bool        # opt-in managed server lifecycle; must be false unless loopback
    optional_deps: list[str]   # e.g. ["mlx", "mlx_lm"]
    startup_timeout_s: float   # seconds to wait for server health check
    request_timeout_s: float
    max_concurrency: int
    max_batch_size: int
    lock_scope: str            # "heavy" | "light" | "none"
    hardware_fit_status: str   # "unknown" | "pending" | "verified" | "rejected"
    approval_status: str       # "deferred" | "benchmark_pending" | "approved" | "blocked"
    model_id: str | None = None


_ALLOWED_RUNTIME_KINDS = {"mlx", "ollama", "custom"}
_ALLOWED_LOCK_SCOPES = {"heavy", "light", "none"}
_ALLOWED_HARDWARE_FIT_STATUSES = {"unknown", "pending", "verified", "rejected"}
_ALLOWED_APPROVAL_STATUSES = {"deferred", "benchmark_pending", "approved", "blocked"}


def _default_profiles_path() -> Path:
    return Path(str(resources.files(__package__).joinpath("local_runtime_profiles.yaml")))


def _parse_profile(profile_id: str, raw: dict) -> LocalRuntimeProfile:
    required_fields = [
        "runtime_kind", "base_url", "manage_server", "optional_deps",
        "startup_timeout_s", "request_timeout_s", "max_concurrency",
        "max_batch_size", "lock_scope", "hardware_fit_status", "approval_status",
    ]
    missing = [f for f in required_fields if f not in raw]
    if missing:
        raise ValueError(f"profile {profile_id!r}: missing required fields: {missing}")

    runtime_kind = str(raw["runtime_kind"])
    if runtime_kind not in _ALLOWED_RUNTIME_KINDS:
        raise ValueError(
            f"profile {profile_id!r}: runtime_kind must be one of {_ALLOWED_RUNTIME_KINDS}, "
            f"got {runtime_kind!r}"
        )
    lock_scope = str(raw["lock_scope"])
    if lock_scope not in _ALLOWED_LOCK_SCOPES:
        raise ValueError(
            f"profile {profile_id!r}: lock_scope must be one of {_ALLOWED_LOCK_SCOPES}, "
            f"got {lock_scope!r}"
        )
    hardware_fit_status = str(raw["hardware_fit_status"])
    if hardware_fit_status not in _ALLOWED_HARDWARE_FIT_STATUSES:
        raise ValueError(
            f"profile {profile_id!r}: hardware_fit_status must be one of "
            f"{_ALLOWED_HARDWARE_FIT_STATUSES}, got {hardware_fit_status!r}"
        )
    approval_status = str(raw["approval_status"])
    if approval_status not in _ALLOWED_APPROVAL_STATUSES:
        raise ValueError(
            f"profile {profile_id!r}: approval_status must be one of "
            f"{_ALLOWED_APPROVAL_STATUSES}, got {approval_status!r}"
        )

    manage_server = bool(raw["manage_server"])
    base_url = str(raw["base_url"])
    if manage_server and not base_url.startswith(("http://127.", "http://localhost")):
        raise ValueError(
            f"profile {profile_id!r}: manage_server=true requires a loopback base_url, "
            f"got {base_url!r}"
        )

    return LocalRuntimeProfile(
        profile_id=profile_id,
        runtime_kind=runtime_kind,
        base_url=base_url,
        manage_server=manage_server,
        optional_deps=list(raw["optional_deps"]),
        startup_timeout_s=float(raw["startup_timeout_s"]),
        request_timeout_s=float(raw["request_timeout_s"]),
        max_concurrency=int(raw["max_concurrency"]),
        max_batch_size=int(raw["max_batch_size"]),
        lock_scope=lock_scope,
        hardware_fit_status=hardware_fit_status,
        approval_status=approval_status,
        model_id=_optional_str(raw.get("model_id")),
    )


@lru_cache(maxsize=4)
def load_profiles(path: str | Path | None = None) -> dict[str, LocalRuntimeProfile]:
    """Load local runtime profiles from YAML.

    Returns a dict keyed by profile_id. Cached after first load.
    """
    profiles_path = Path(path) if path else _default_profiles_path()
    raw = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"local_runtime_profiles is not a mapping: {profiles_path}")
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("local_runtime_profiles must define a non-empty 'profiles' mapping")
    return {
        profile_id: _parse_profile(profile_id, profile_data)
        for profile_id, profile_data in profiles_raw.items()
    }


def _optional_str(value: object) -> str | None:
    if value in (None, "", "none"):
        return None
    return str(value)
