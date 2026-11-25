"""Hardware preflight checks, concurrency clamping, and overlapping-workload guard.

No mlx/mlx_lm module-level imports. HeavyRuntimeLock imported only under TYPE_CHECKING
to avoid circular imports with runtime_lock.py.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile
from conference_pipeline.model_governance.registry import ModelGovernanceError

if TYPE_CHECKING:
    from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock

_HEAVY_MODEL_KEYWORDS = {"70b", "72b", "65b", "32b"}

_DEFAULT_CACHE_ROOTS = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "mlx",
]


def check_platform() -> None:
    if sys.platform != "darwin":
        raise ModelGovernanceError(
            f"MLX runtime requires macOS/Apple Silicon; current platform: {sys.platform}"
        )


def check_optional_deps(profile: LocalRuntimeProfile) -> None:
    for dep in profile.optional_deps:
        if importlib.util.find_spec(dep) is None:
            raise ModelGovernanceError(
                f"optional dependency '{dep}' not installed; "
                "install it to use local MLX runtime"
            )


def check_model_assets(
    model_id: str,
    *,
    search_paths: list[Path] | None = None,
) -> Path:
    candidates = list(_DEFAULT_CACHE_ROOTS)
    if search_paths:
        candidates.extend(search_paths)

    model_leaf = model_id.split("/")[-1] if "/" in model_id else model_id

    for root in candidates:
        if not root.exists():
            continue
        # exact match on root itself
        if root.name == model_id or root.name == model_leaf:
            return root
        # check direct children
        for child in root.iterdir():
            if child.name == model_id or child.name == model_leaf:
                return child
            # huggingface hub stores as models--owner--name
            slug = "models--" + model_id.replace("/", "--")
            if child.name == slug:
                return child

    raise ModelGovernanceError(
        f"model assets for '{model_id}' not found in local cache; "
        "download the model first"
    )


def _model_size_tier(model_id: str) -> str:
    lower = model_id.lower()
    for keyword in _HEAVY_MODEL_KEYWORDS:
        if keyword in lower:
            return "heavy"
    return "light"


def clamp_concurrency(
    profile: LocalRuntimeProfile, model_id: str
) -> LocalRuntimeProfile:
    if _model_size_tier(model_id) == "heavy" and profile.max_concurrency > 1:
        from dataclasses import replace
        return replace(profile, max_concurrency=1, max_batch_size=1)
    return profile


def check_overlapping_workload(
    profile: LocalRuntimeProfile,
    lock: "HeavyRuntimeLock",
) -> None:
    if profile.lock_scope == "heavy" and lock.is_held():
        raise ModelGovernanceError(
            f"heavy local runtime lock is held by {lock.owner_metadata()}; "
            "cannot start a competing heavy local job"
        )


def run_all(
    profile: LocalRuntimeProfile,
    model_id: str,
    lock: "HeavyRuntimeLock",
) -> LocalRuntimeProfile:
    check_platform()
    check_optional_deps(profile)
    check_model_assets(model_id)
    check_overlapping_workload(profile, lock)
    return clamp_concurrency(profile, model_id)
