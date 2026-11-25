"""Provider call receipts for paid language-model boundaries.

This module is the only place in the repository that should call paid LLM SDK
methods directly.  Callers get a ``ProviderCallResult`` that behaves like the
underlying SDK response while exposing the durable ``receipt`` for audit.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from conference_pipeline.costs.estimator import load_pricing_catalog


SCHEMA_VERSION = 1
RECEIPT_MARKER = "conference_pipeline.provider_call_receipt"
LOCAL_LLM_RECEIPT_MARKER = "conference_pipeline.local_llm_receipt"
LOCAL_LLM_RECEIPT_SCHEMA_VERSION = 2
ESCALATION_RECEIPT_MARKER = "conference_pipeline.escalation_receipt"
ESCALATION_RECEIPT_SCHEMA_VERSION = 1
BLOCKER_RECEIPT_MARKER = "conference_pipeline.blocker_receipt"
BLOCKER_RECEIPT_SCHEMA_VERSION = 1
UNBLOCK_RECEIPT_MARKER = "conference_pipeline.unblock_receipt"
UNBLOCK_RECEIPT_SCHEMA_VERSION = 1
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SECRET_RE = re.compile(
    r"(?i)(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|"
    r"TAVILY_API_KEY|api[_-]?key|authorization|cookie|access[_-]?token|"
    r"refresh[_-]?token|bearer|\bsecret\b|secret[_-]?key)"
)


@dataclass(frozen=True)
class ProviderReceiptContext:
    """Runtime metadata used to place and label provider receipts."""

    workspace_root: Path | None = None
    run_id: str | None = None
    stage_id: str | None = None
    caller: str | None = None
    task_id: str | None = None  # D-08: task identity for cost rollup attribution


@dataclass(frozen=True)
class ProviderCallReceipt:
    """Serializable receipt for one provider SDK call."""

    schema_version: int
    receipt_type: str
    call_id: str
    provider: str
    model: str
    operation: str
    caller: str
    workspace: str
    run_id: str | None
    stage_id: str | None
    status: str
    started_at: str
    ended_at: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    audio_minutes: float | None = None
    tts_minutes: float | None = None
    credits: float | None = None
    estimated_cost_usd: float | None = None
    cost_status: str = "unknown"
    pricing_refreshed_at: str | None = None
    pricing_source_urls: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    response_id: str | None = None
    request_sha256: str | None = None
    request_summary: dict[str, Any] = field(default_factory=dict)
    receipt_path: str | None = None
    secrets_copied: bool = False
    task_id: str | None = None  # D-08: None on old receipts -> "unattributed" at rollup time

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pricing_source_urls"] = list(self.pricing_source_urls)
        payload["warnings"] = list(self.warnings)
        return sanitize_for_receipt(payload)


T = TypeVar("T")


@dataclass(frozen=True)
class ProviderCallResult(Generic[T]):
    """SDK response plus the receipt written for that response.

    Attribute access and iteration delegate to ``response`` so most existing
    call sites can keep treating the result like the raw SDK object while still
    being able to read ``.receipt``.
    """

    response: T
    receipt: ProviderCallReceipt

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.response)  # type: ignore[arg-type]

    def __getitem__(self, key: Any) -> Any:
        return self.response[key]  # type: ignore[index]

    def __bool__(self) -> bool:
        return bool(self.response)

    def __str__(self) -> str:
        return str(self.response)

    def __len__(self) -> int:
        return len(self.response)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LocalLLMReceipt:
    """Receipt for a local LLM extraction run (preflight, quality sample, or full run).

    The endpoint URL is never stored. Only ``endpoint_label`` is recorded.
    v1.8 allows backend-shaped labels such as ``mlx://<model-id>`` or a
    redacted loopback HTTP label like ``http://localhost:11434/v1``.
    ``paid_cloud_cost`` is always 0.0 — local inference has no cloud billing.
    """

    schema_version: int
    receipt_type: str  # LOCAL_LLM_RECEIPT_MARKER
    endpoint_label: str  # from EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL — never a URL
    model: str
    request_count: int
    token_count: int | None  # None when the local server does not report token usage
    confidence_path: str  # "heuristic_only" | "local_llm" | "cloud_escalation"
    skipped_record_count: int
    paid_cloud_cost: float  # always 0.0
    timestamp: str  # ISO-8601 UTC
    run_id: str | None
    stage_id: str | None
    preflight_mode: bool
    quality_sample_mode: bool
    receipt_path: str | None = None
    effective_thresholds: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


@dataclass(frozen=True)
class EscalationReceipt:
    """Receipt wrapping a cloud ProviderCallReceipt with escalation context.

    ``wrapped_receipt.to_dict()`` is called BEFORE the outer ``sanitize_for_receipt``
    pass so that the inner receipt's secrets are redacted first.
    """

    schema_version: int
    receipt_type: str  # ESCALATION_RECEIPT_MARKER
    wrapped_receipt: ProviderCallReceipt
    escalation_reason: str
    affected_record_ids: tuple[str, ...]  # frozen; convert to list in to_dict
    fail_closed_state: bool
    timestamp: str  # ISO-8601 UTC

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Convert tuple to list for JSON serialization
        payload["affected_record_ids"] = list(self.affected_record_ids)
        # Serialize wrapped_receipt with its own sanitize pass BEFORE the outer pass
        payload["wrapped_receipt"] = self.wrapped_receipt.to_dict()
        return sanitize_for_receipt(payload)


class BlockerClass:
    OCR_DEFERRED_BLOCKED = "ocr_deferred_blocked"
    LOCAL_LLM_NOT_READY = "local_llm_not_ready"
    MISSING_RECEIPT = "missing_receipt"
    THRESHOLD_VIOLATION = "threshold_violation"
    STALE_RECEIPT = "stale_receipt"
    STAGE_BLOCKED = "stage_blocked"
    STAGE_UNAVAILABLE = "stage_unavailable"
    # COST-04: budget cap enforcement — the cap is NEVER auto-raised; a breach halts
    # and requires explicit operator approval via UnblockReceipt(manual_ack, operator_id).
    BUDGET_CAP_EXCEEDED = "budget_cap_exceeded"
    ALL = (
        OCR_DEFERRED_BLOCKED,
        LOCAL_LLM_NOT_READY,
        MISSING_RECEIPT,
        THRESHOLD_VIOLATION,
        STALE_RECEIPT,
        STAGE_BLOCKED,
        STAGE_UNAVAILABLE,
        BUDGET_CAP_EXCEEDED,
    )


class MissingRequiredReceiptError(RuntimeError):
    """Raised when a stage requires a receipt that is not present."""

    def __init__(self, stage_id: str, expected_receipt_kind: str) -> None:
        super().__init__(
            f"stage {stage_id!r} requires {expected_receipt_kind} receipt but none found"
        )
        self.stage_id = stage_id
        self.expected_receipt_kind = expected_receipt_kind


class StaleReceiptError(RuntimeError):
    """Raised when a required receipt exists but is too old to trust."""


@dataclass(frozen=True)
class BlockerReceipt:
    schema_version: int
    receipt_type: str
    blocker_class: str
    stage_id: str
    reason: str
    remediation_path: str
    automated_unblock_available: bool
    endpoint_label: str | None
    timestamp: str
    run_id: str | None
    receipt_path: str | None = None

    def __post_init__(self) -> None:
        if self.blocker_class not in BlockerClass.ALL:
            raise ValueError(
                f"unknown blocker_class {self.blocker_class!r}; expected one of {BlockerClass.ALL}"
            )

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_receipt(asdict(self))


@dataclass(frozen=True)
class UnblockReceipt:
    schema_version: int
    receipt_type: str
    blocker_receipt_path: str
    unblock_method: str
    operator_id: str | None
    note: str | None
    timestamp: str
    run_id: str | None
    receipt_path: str | None = None

    def __post_init__(self) -> None:
        if self.unblock_method not in {"automated", "manual_ack"}:
            raise ValueError(
                "unblock_method must be 'automated' or 'manual_ack'"
            )
        if self.unblock_method == "manual_ack" and not self.operator_id:
            raise ValueError("manual_ack unblock receipts require operator_id")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_receipt(asdict(self))


_CURRENT_CONTEXT: contextvars.ContextVar[ProviderReceiptContext] = contextvars.ContextVar(
    "provider_receipt_context",
    default=ProviderReceiptContext(),
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_segment(value: str | None, *, default: str = "unknown") -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", str(value or "").strip()).strip("._")
    return cleaned[:120] or default


def sanitize_for_receipt(value: Any, *, _key: str | None = None) -> Any:
    """Redact genuine secrets before writing receipt JSON.

    Redaction is deliberately limited to *precise secret patterns* -- a dict
    key whose name is a known credential (``_SECRET_RE``), a ``Bearer <token>``
    header, or an ``api_key=/authorization=/cookie=/token=/secret=`` query
    value.  It intentionally does NOT redact by length: an earlier blanket
    "any 80+ alphanumeric run" rule silently rewrote legitimate long
    identifiers -- most damagingly CVPR paper filenames encoding the full
    title (``.../Xiao_Generalizable_..._CVPR_2026_paper.pdf``) -- into an
    unreachable ``.../REDACTED.pdf``.  Real credentials in this pipeline are
    always named or prefixed and are caught by the precise patterns; a long
    human-readable string is data, not a secret, and must round-trip intact.
    """

    if isinstance(value, dict):
        return {
            str(key): (
                "REDACTED"
                if _SECRET_RE.search(str(key))
                else sanitize_for_receipt(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_receipt(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        redacted = re.sub(
            r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+",
            r"\1REDACTED",
            value,
        )
        redacted = re.sub(
            r"(?i)((?:api[_-]?key|authorization|cookie|token|secret)=)[^\s&]+",
            r"\1REDACTED",
            redacted,
        )
        return redacted
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _env_context() -> ProviderReceiptContext:
    workspace = (
        os.environ.get("PROVIDER_RECEIPT_WORKSPACE")
        or os.environ.get("CONFERENCE_WORKSPACE")
        or os.environ.get("WORKSPACE_ROOT")
    )
    return ProviderReceiptContext(
        workspace_root=Path(workspace) if workspace else None,
        run_id=os.environ.get("PROVIDER_RECEIPT_RUN_ID") or os.environ.get("RUN_ID"),
        stage_id=os.environ.get("PROVIDER_RECEIPT_STAGE_ID"),
        caller=os.environ.get("PROVIDER_RECEIPT_CALLER"),
        task_id=os.environ.get("PROVIDER_RECEIPT_TASK_ID"),  # D-08: subprocess inheritance
    )


def current_provider_receipt_context() -> ProviderReceiptContext:
    local = _CURRENT_CONTEXT.get()
    env = _env_context()
    return ProviderReceiptContext(
        workspace_root=local.workspace_root or env.workspace_root,
        run_id=local.run_id or env.run_id,
        stage_id=local.stage_id or env.stage_id,
        caller=local.caller or env.caller,
        task_id=local.task_id or env.task_id,  # D-08: local context takes precedence
    )


# The receipt context is mirrored into os.environ so that subprocesses AND worker
# threads (which do not inherit a ContextVar) attribute their calls to the right
# stage. os.environ is process-global, so overlapping contexts cannot each "save the
# previous value and restore it": two threads that exit out of order leave the
# second one restoring the FIRST one's value, and a task id stays stuck for every
# later call (found on 2026-10-03: concurrent figure-spec critic calls left
# PROVIDER_RECEIPT_TASK_ID=presentation.figure_spec_critic behind). So the active
# contexts are tracked under a lock and the environment is recomputed from the
# values it had before the first one entered plus whatever is still active, in the
# order they entered. The result does not depend on the order contexts exit.
_RECEIPT_ENV_KEYS = (
    "PROVIDER_RECEIPT_WORKSPACE",
    "PROVIDER_RECEIPT_RUN_ID",
    "PROVIDER_RECEIPT_STAGE_ID",
    "PROVIDER_RECEIPT_CALLER",
    "PROVIDER_RECEIPT_TASK_ID",
)
_ENV_LOCK = threading.Lock()
_ACTIVE_ENV_OVERRIDES: list[tuple[object, dict[str, str]]] = []
_ENV_BASELINE: dict[str, str | None] = {}


def _context_env_overrides(ctx: ProviderReceiptContext) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if ctx.workspace_root is not None:
        overrides["PROVIDER_RECEIPT_WORKSPACE"] = str(ctx.workspace_root)
    if ctx.run_id:
        overrides["PROVIDER_RECEIPT_RUN_ID"] = ctx.run_id
    if ctx.stage_id:
        overrides["PROVIDER_RECEIPT_STAGE_ID"] = ctx.stage_id
    if ctx.caller:
        overrides["PROVIDER_RECEIPT_CALLER"] = ctx.caller
    if ctx.task_id:
        overrides["PROVIDER_RECEIPT_TASK_ID"] = ctx.task_id
    return overrides


def _apply_receipt_env_locked() -> None:
    """Set os.environ to the baseline plus every active context's overrides (lock held)."""

    values = dict(_ENV_BASELINE)
    for _owner, overrides in _ACTIVE_ENV_OVERRIDES:
        values.update(overrides)
    for key in _RECEIPT_ENV_KEYS:
        value = values.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextlib.contextmanager
def provider_receipt_context(
    *,
    workspace_root: Path | str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    caller: str | None = None,
    task_id: str | None = None,  # D-08: task identity for cost rollup attribution
) -> Iterator[None]:
    """Set receipt metadata for in-process calls and inherited subprocesses."""

    ctx = ProviderReceiptContext(
        workspace_root=Path(workspace_root).resolve() if workspace_root else None,
        run_id=run_id,
        stage_id=stage_id,
        caller=caller,
        task_id=task_id,
    )
    owner = object()
    token = _CURRENT_CONTEXT.set(ctx)
    with _ENV_LOCK:
        if not _ACTIVE_ENV_OVERRIDES:
            _ENV_BASELINE.clear()
            _ENV_BASELINE.update({key: os.environ.get(key) for key in _RECEIPT_ENV_KEYS})
        _ACTIVE_ENV_OVERRIDES.append((owner, _context_env_overrides(ctx)))
        _apply_receipt_env_locked()
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)
        with _ENV_LOCK:
            _ACTIVE_ENV_OVERRIDES[:] = [item for item in _ACTIVE_ENV_OVERRIDES if item[0] is not owner]
            _apply_receipt_env_locked()
            if not _ACTIVE_ENV_OVERRIDES:
                _ENV_BASELINE.clear()


def _workspace_from_context(context: ProviderReceiptContext) -> Path:
    return (context.workspace_root or Path.cwd()).resolve()


def infer_conference_workspace(start: Path | str) -> Path | None:
    """Best-effort discovery of a conference workspace root from any path under it.

    Standalone scripts that receive a positional file path (rather than an
    explicit ``--workspace`` flag) can use this to derive a workspace-rooted
    ``provider_receipt_context`` instead of silently defaulting to
    ``Path.cwd()`` (which is how receipts end up misplaced at the repo root;
    see docs/reference/stage-owner-audit.md). Returns ``None`` when no
    ``conferences/<slug>`` ancestor can be identified.
    """

    path = Path(start).resolve()
    candidates = [path] if path.is_dir() else []
    candidates.extend(path.parents)
    for candidate in candidates:
        if candidate.name and candidate.parent.name == "conferences":
            return candidate
        if (candidate / "config.yaml").is_file() and (candidate / "storage").is_dir():
            return candidate
    return None


def _call_receipt_path(
    *,
    context: ProviderReceiptContext,
    caller: str,
    call_id: str,
    started_at: str,
) -> Path:
    workspace = _workspace_from_context(context)
    segment = safe_segment(context.stage_id or Path(caller).stem or caller)
    filename = f"{started_at.replace(':', '').replace('-', '')}-{call_id}.json"
    if context.run_id:
        return (
            workspace
            / "storage"
            / "workflow-runs"
            / safe_segment(context.run_id)
            / "provider-calls"
            / segment
            / filename
        )
    return workspace / "storage" / "provider-calls" / segment / filename


def _request_fingerprint(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            str(key): ("REDACTED" if _SECRET_RE.search(str(key)) else _request_fingerprint(item, depth=depth + 1))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_request_fingerprint(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return {
            "type": "str",
            "length": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
        }
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr_sha256": hashlib.sha256(repr(value).encode()).hexdigest()}


def _request_digest(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    summary = _request_fingerprint(kwargs)
    encoded = json.dumps(summary, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), summary


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def gemini_output_tokens(usage: Any) -> int | None:
    """Billed Gemini output: visible tokens plus thinking tokens.

    Gemini reports thinking as ``thoughts_token_count``, apart from ``candidates_token_count``,
    and bills both as output (found 2026-11-18 by the v6.1 gold-labelling canary, where a check
    call reported 317 output tokens after thinking through an 8,000-token budget).
    """
    visible = _first_int(_get_attr_or_key(usage, "candidates_token_count"))
    thoughts = _first_int(_get_attr_or_key(usage, "thoughts_token_count"))
    if visible is None and thoughts is None:
        return None
    return int(visible or 0) + int(thoughts or 0)


def _extract_usage(
    response: Any,
    *,
    provider: str,
    audio_minutes: float | None = None,
    tts_minutes: float | None = None,
    credits: float | None = None,
) -> dict[str, Any]:
    target = response[-1] if isinstance(response, list) and response else response
    usage = _get_attr_or_key(target, "usage")
    if provider == "gemini":
        usage = _get_attr_or_key(target, "usage_metadata", usage)

    input_tokens = _first_int(
        _get_attr_or_key(usage, "input_tokens"),
        _get_attr_or_key(usage, "prompt_tokens"),
        _get_attr_or_key(usage, "prompt_token_count"),
    )
    output_tokens = _first_int(
        _get_attr_or_key(usage, "output_tokens"),
        _get_attr_or_key(usage, "completion_tokens"),
        gemini_output_tokens(usage),
    )
    total_tokens = _first_int(
        _get_attr_or_key(usage, "total_tokens"),
        _get_attr_or_key(usage, "total_token_count"),
    )
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "audio_minutes": audio_minutes,
        "tts_minutes": tts_minutes,
        "credits": credits,
    }


def _response_id(response: Any) -> str | None:
    target = response[-1] if isinstance(response, list) and response else response
    value = _get_attr_or_key(target, "id")
    return str(value) if value else None


def _estimate_cost(
    *,
    provider: str,
    model: str,
    operation: str,
    usage: dict[str, Any],
    wall_seconds: float | None = None,
) -> tuple[float | None, str, str | None, tuple[str, ...], tuple[str, ...]]:
    if provider.strip().lower() in {"local-openai-compatible", "mlx-local"}:
        return 0.0, "local_no_paid_provider", None, (), ()

    catalog = load_pricing_catalog()
    warnings: list[str] = []
    try:
        resolved_provider, resolved_model, model_data = catalog.resolve_model(model, provider)
    except KeyError as exc:
        return None, "unknown", catalog.refreshed_at, (), (str(exc),)

    pricing_unit = str(model_data.get("pricing_unit") or "")
    source_urls: list[str] = []
    try:
        if pricing_unit == "none":
            return (
                round(float(model_data.get("fixed_usd", 0.0)), 6),
                "estimated",
                catalog.refreshed_at,
                tuple(catalog.source_urls_for(model_data)),
                (),
            )

        if operation in {"audio_transcription", "transcription"} or pricing_unit == "audio_minute":
            minutes = usage.get("audio_minutes")
            if minutes:
                cost, source_urls, warnings = catalog.estimate_audio_minutes(
                    resolved_provider, resolved_model, float(minutes)
                )
                return round(cost, 6), "estimated", catalog.refreshed_at, tuple(source_urls), tuple(warnings)
            return None, "unknown", catalog.refreshed_at, tuple(catalog.source_urls_for(model_data)), (
                "Provider response did not include billable audio duration.",
            )

        if pricing_unit == "gpu_second":  # self-hosted (Modal): the request's wall time is the billed GPU time
            if wall_seconds is None:
                return None, "unknown", catalog.refreshed_at, tuple(catalog.source_urls_for(model_data)), (
                    "No wall time to price GPU seconds.",
                )
            cost, source_urls, warnings = catalog.estimate_gpu_cost(resolved_provider, resolved_model, wall_seconds)
            return round(cost, 6), "estimated", catalog.refreshed_at, tuple(source_urls), tuple(warnings)

        if pricing_unit == "credits":
            credits = usage.get("credits")
            if credits:
                cost, source_urls, warnings = catalog.estimate_credits(
                    resolved_provider, resolved_model, float(credits)
                )
                return round(cost, 6), "estimated", catalog.refreshed_at, tuple(source_urls), tuple(warnings)
            return None, "unknown", catalog.refreshed_at, tuple(catalog.source_urls_for(model_data)), (
                "Provider response did not include credit usage.",
            )

        if pricing_unit == "tts_minute_estimate" and usage.get("tts_minutes"):
            cost, source_urls, warnings = catalog.estimate_tts_minutes(
                resolved_provider, resolved_model, float(usage["tts_minutes"])
            )
            return round(cost, 6), "estimated", catalog.refreshed_at, tuple(source_urls), tuple(warnings)

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is not None or output_tokens is not None:
            cost, source_urls, warnings = catalog.estimate_tokens(
                resolved_provider,
                resolved_model,
                int(input_tokens or 0),
                int(output_tokens or 0),
            )
            return round(cost, 6), "estimated", catalog.refreshed_at, tuple(source_urls), tuple(warnings)
    except Exception as exc:  # noqa: BLE001 - receipts must not hide the provider response.
        return None, "unknown", catalog.refreshed_at, tuple(source_urls), (sanitize_for_receipt(str(exc)),)

    return None, "unknown", catalog.refreshed_at, tuple(catalog.source_urls_for(model_data)), (
        "Provider response did not include usage fields needed for pricing.",
    )


def _finalize_receipt(
    *,
    provider: str,
    model: str,
    operation: str,
    caller: str | None,
    status: str,
    started_at: str,
    started_perf: float,
    response: Any = None,
    error: BaseException | None = None,
    context: ProviderReceiptContext | None = None,
    request_kwargs: dict[str, Any] | None = None,
    audio_minutes: float | None = None,
    tts_minutes: float | None = None,
    credits: float | None = None,
) -> ProviderCallReceipt:
    effective_context = context or current_provider_receipt_context()
    effective_caller = caller or effective_context.caller or "unknown"
    ended_at = utc_now()
    request_sha256, request_summary = _request_digest(request_kwargs or {})
    usage = _extract_usage(
        response,
        provider=provider,
        audio_minutes=audio_minutes,
        tts_minutes=tts_minutes,
        credits=credits,
    )
    latency_seconds = round(time.perf_counter() - started_perf, 6)
    estimated_cost, cost_status, refreshed_at, source_urls, warnings = _estimate_cost(
        provider=provider,
        model=model,
        operation=operation,
        usage=usage,
        wall_seconds=latency_seconds,
    )
    call_id = uuid.uuid4().hex[:16]
    receipt_path = _call_receipt_path(
        context=effective_context,
        caller=effective_caller,
        call_id=call_id,
        started_at=started_at,
    )
    workspace = _workspace_from_context(effective_context)
    receipt = ProviderCallReceipt(
        schema_version=SCHEMA_VERSION,
        receipt_type=RECEIPT_MARKER,
        call_id=call_id,
        provider=provider,
        model=model,
        operation=operation,
        caller=effective_caller,
        workspace=str(workspace),
        run_id=effective_context.run_id,
        stage_id=effective_context.stage_id,
        task_id=effective_context.task_id,  # D-08: threaded from context
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        latency_seconds=latency_seconds,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        audio_minutes=usage.get("audio_minutes"),
        tts_minutes=usage.get("tts_minutes"),
        credits=usage.get("credits"),
        estimated_cost_usd=estimated_cost,
        cost_status=cost_status,
        pricing_refreshed_at=refreshed_at,
        pricing_source_urls=source_urls,
        warnings=warnings,
        error=sanitize_for_receipt(f"{type(error).__name__}: {error}") if error else None,
        response_id=_response_id(response),
        request_sha256=request_sha256,
        request_summary=request_summary,
        receipt_path=str(receipt_path),
        secrets_copied=False,
    )
    _atomic_write_json(receipt_path, receipt.to_dict())
    return receipt


def record_provider_call(
    invoke: Callable[[], T],
    *,
    provider: str,
    model: str,
    operation: str,
    caller: str | None = None,
    context: ProviderReceiptContext | None = None,
    request_kwargs: dict[str, Any] | None = None,
    audio_minutes: float | None = None,
    tts_minutes: float | None = None,
    credits: float | None = None,
) -> ProviderCallResult[T]:
    started_at = utc_now()
    started_perf = time.perf_counter()
    try:
        response = invoke()
    except Exception as exc:
        receipt = _finalize_receipt(
            provider=provider,
            model=model,
            operation=operation,
            caller=caller,
            context=context,
            status="failed",
            started_at=started_at,
            started_perf=started_perf,
            error=exc,
            request_kwargs=request_kwargs,
            audio_minutes=audio_minutes,
            tts_minutes=tts_minutes,
            credits=credits,
        )
        setattr(exc, "provider_receipt", receipt)
        raise
    receipt = _finalize_receipt(
        provider=provider,
        model=model,
        operation=operation,
        caller=caller,
        context=context,
        status="succeeded",
        started_at=started_at,
        started_perf=started_perf,
        response=response,
        request_kwargs=request_kwargs,
        audio_minutes=audio_minutes,
        tts_minutes=tts_minutes,
        credits=credits,
    )
    return ProviderCallResult(response=response, receipt=receipt)


async def record_async_provider_call(
    invoke: Callable[[], Awaitable[T]],
    *,
    provider: str,
    model: str,
    operation: str,
    caller: str | None = None,
    context: ProviderReceiptContext | None = None,
    request_kwargs: dict[str, Any] | None = None,
    audio_minutes: float | None = None,
    tts_minutes: float | None = None,
    credits: float | None = None,
) -> ProviderCallResult[T]:
    started_at = utc_now()
    started_perf = time.perf_counter()
    try:
        response = await invoke()
    except Exception as exc:
        receipt = _finalize_receipt(
            provider=provider,
            model=model,
            operation=operation,
            caller=caller,
            context=context,
            status="failed",
            started_at=started_at,
            started_perf=started_perf,
            error=exc,
            request_kwargs=request_kwargs,
            audio_minutes=audio_minutes,
            tts_minutes=tts_minutes,
            credits=credits,
        )
        setattr(exc, "provider_receipt", receipt)
        raise
    receipt = _finalize_receipt(
        provider=provider,
        model=model,
        operation=operation,
        caller=caller,
        context=context,
        status="succeeded",
        started_at=started_at,
        started_perf=started_perf,
        response=response,
        request_kwargs=request_kwargs,
        audio_minutes=audio_minutes,
        tts_minutes=tts_minutes,
        credits=credits,
    )
    return ProviderCallResult(response=response, receipt=receipt)


def _registry_reasoning_effort_for_model(model: str) -> str | None:
    """Return the registry-configured reasoning effort for an OpenAI model.

    A model opts into elevated reasoning effort when any governed task in
    ``model_governance/registry.yaml`` sets ``reasoning_effort`` next to a
    matching default ``model`` (e.g. GPT-5.6 Luna must run with high
    reasoning effort). Models without a configured entry keep current
    behavior: no reasoning parameter is injected.
    """
    if not model:
        return None
    try:
        from conference_pipeline.model_governance.registry import load_registry

        registry = load_registry()
    except Exception:  # pragma: no cover - registry unavailable in minimal harnesses.
        return None
    for policy in registry.tasks.values():
        if policy.reasoning_effort and policy.model == model:
            return policy.reasoning_effort
    return None


def _with_openai_reasoning_effort(kwargs: dict[str, Any], *, responses_api: bool = False) -> dict[str, Any]:
    """Inject the registry-configured reasoning effort into OpenAI kwargs.

    Explicit caller-provided ``reasoning_effort``/``reasoning`` values win;
    models without a registry reasoning contract are passed through untouched.
    """
    if "reasoning_effort" in kwargs or "reasoning" in kwargs:
        return kwargs
    effort = _registry_reasoning_effort_for_model(str(kwargs.get("model") or ""))
    if not effort:
        return kwargs
    updated = dict(kwargs)
    if responses_api:
        updated["reasoning"] = {"effort": effort}
    else:
        updated["reasoning_effort"] = effort
    return updated


def openai_chat_completions_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    kwargs = _with_openai_reasoning_effort(kwargs)
    return record_provider_call(
        lambda: client.chat.completions.create(**kwargs),
        provider="openai",
        model=str(kwargs.get("model") or "unknown"),
        operation="chat.completions",
        caller=caller,
        request_kwargs=kwargs,
    )


def openrouter_chat_completions_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    """Chat completions at OpenRouter's OpenAI-compatible endpoint.

    ``model`` is the catalog name (``openrouter/<vendor>/<slug>``) and is what
    the receipt records; the API itself is sent the bare ``<vendor>/<slug>``.
    """

    model = str(kwargs.get("model") or "unknown")
    api_kwargs = {**kwargs, "model": model.removeprefix("openrouter/")}
    return record_provider_call(
        lambda: client.chat.completions.create(**api_kwargs),
        provider="openrouter",
        model=model,
        operation="chat.completions",
        caller=caller,
        request_kwargs=kwargs,
    )


def local_openai_chat_completions_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    return record_provider_call(
        lambda: client.chat.completions.create(**kwargs),
        provider="local-openai-compatible",
        model=str(kwargs.get("model") or "unknown"),
        operation="chat.completions",
        caller=caller,
        request_kwargs=kwargs,
    )


def modal_chat_completions_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    """A self-hosted OpenAI-compatible endpoint on Modal: priced by GPU seconds (wall time), not tokens."""
    return record_provider_call(
        lambda: client.chat.completions.create(**kwargs),
        provider="modal",
        model=str(kwargs.pop("_priced_model", None) or kwargs.get("model") or "unknown"),
        operation="chat.completions",
        caller=caller,
        request_kwargs=kwargs,
    )


async def openai_chat_completions_create_async(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    kwargs = _with_openai_reasoning_effort(kwargs)
    return await record_async_provider_call(
        lambda: client.chat.completions.create(**kwargs),
        provider="openai",
        model=str(kwargs.get("model") or "unknown"),
        operation="chat.completions",
        caller=caller,
        request_kwargs=kwargs,
    )


def openai_responses_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    kwargs = _with_openai_reasoning_effort(kwargs, responses_api=True)
    return record_provider_call(
        lambda: client.responses.create(**kwargs),
        provider="openai",
        model=str(kwargs.get("model") or "unknown"),
        operation="responses",
        caller=caller,
        request_kwargs=kwargs,
    )


def openai_embeddings_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    return record_provider_call(
        lambda: client.embeddings.create(**kwargs),
        provider="openai",
        model=str(kwargs.get("model") or "unknown"),
        operation="embeddings",
        caller=caller,
        request_kwargs=kwargs,
    )


async def openai_audio_transcriptions_create_async(
    client: Any,
    *,
    caller: str | None = None,
    audio_minutes: float | None = None,
    **kwargs: Any,
) -> ProviderCallResult[Any]:
    return await record_async_provider_call(
        lambda: client.audio.transcriptions.create(**kwargs),
        provider="openai",
        model=str(kwargs.get("model") or "unknown"),
        operation="audio_transcription",
        caller=caller,
        request_kwargs=kwargs,
        audio_minutes=audio_minutes,
    )


def anthropic_messages_create(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    return record_provider_call(
        lambda: client.messages.create(**kwargs),
        provider="anthropic",
        model=str(kwargs.get("model") or "unknown"),
        operation="messages",
        caller=caller,
        request_kwargs=kwargs,
    )


def anthropic_messages_stream_create(
    client: Any,
    *,
    caller: str | None = None,
    **kwargs: Any,
) -> ProviderCallResult[Any]:
    def invoke() -> Any:
        with client.messages.stream(**kwargs) as stream:
            stream.until_done()
            return stream.get_final_message()

    return record_provider_call(
        invoke,
        provider="anthropic",
        model=str(kwargs.get("model") or "unknown"),
        operation="messages.stream",
        caller=caller,
        request_kwargs=kwargs,
    )


async def anthropic_messages_create_async(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    return await record_async_provider_call(
        lambda: client.messages.create(**kwargs),
        provider="anthropic",
        model=str(kwargs.get("model") or "unknown"),
        operation="messages",
        caller=caller,
        request_kwargs=kwargs,
    )


def gemini_generate_content(
    client: Any,
    *,
    caller: str | None = None,
    tts_minutes: float | None = None,
    operation: str = "generate_content",
    receipt_request_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ProviderCallResult[Any]:
    # D-12: tts_minutes extracted from kwargs so it is NOT forwarded to the SDK call
    # but IS passed to record_provider_call for cost estimation.
    # ``operation`` and ``receipt_request_kwargs`` let a caller name its own
    # operation and record a sanitized request summary (e.g. a media URL with
    # its query string stripped) instead of the raw SDK kwargs.
    return record_provider_call(
        lambda: client.models.generate_content(**kwargs),
        provider="gemini",
        model=str(kwargs.get("model") or "unknown"),
        operation=operation,
        caller=caller,
        tts_minutes=tts_minutes,
        request_kwargs=kwargs if receipt_request_kwargs is None else receipt_request_kwargs,
    )


def gemini_interactions_create(client: Any, *, caller: str | None = None, tts_minutes: float | None = None, **kwargs: Any) -> ProviderCallResult[Any]:
    """Gemini Interactions API call (``client.interactions.create``).

    Multi-speaker TTS on the governed ``gemini_tts`` catalog model is only
    accepted through this API (each turn annotated with its speaker);
    generate_content rejects it.
    """
    return record_provider_call(
        lambda: client.interactions.create(**kwargs),
        provider="gemini",
        model=str(kwargs.get("model") or "unknown"),
        operation="interactions_create",
        caller=caller,
        tts_minutes=tts_minutes,
        request_kwargs=kwargs,
    )


def gemini_generate_content_stream(client: Any, *, caller: str | None = None, **kwargs: Any) -> ProviderCallResult[list[Any]]:
    return record_provider_call(
        lambda: list(client.models.generate_content_stream(**kwargs)),
        provider="gemini",
        model=str(kwargs.get("model") or "unknown"),
        operation="generate_content_stream",
        caller=caller,
        request_kwargs=kwargs,
    )


def legacy_gemini_generate_content(
    model_client: Any,
    *args: Any,
    caller: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> ProviderCallResult[Any]:
    request_kwargs = {"args": args, **kwargs}
    model_name = model or str(getattr(model_client, "model_name", None) or getattr(model_client, "model", None) or "unknown")
    return record_provider_call(
        lambda: model_client.generate_content(*args, **kwargs),
        provider="gemini",
        model=model_name,
        operation="generate_content",
        caller=caller,
        request_kwargs=request_kwargs,
    )


def _provider_from_litellm_model(model: str) -> str:
    if model.startswith("openrouter/"):
        return "openrouter"
    prefix = model.split("/", 1)[0].split(":", 1)[0].lower()
    if prefix in {"anthropic", "claude"}:
        return "anthropic"
    if prefix in {"gemini", "google", "google_genai"}:
        return "gemini"
    if prefix in {"openai", "gpt", "o1", "o3"} or model.startswith(("gpt-", "o1", "o3")):
        return "openai"
    return prefix or "unknown"


def litellm_completion(
    completion_func: Callable[..., Any],
    *,
    caller: str | None = None,
    **kwargs: Any,
) -> ProviderCallResult[Any]:
    model = str(kwargs.get("model") or "unknown")
    return record_provider_call(
        lambda: completion_func(**kwargs),
        provider=_provider_from_litellm_model(model),
        model=model,
        operation="litellm.completion",
        caller=caller,
        request_kwargs=kwargs,
    )


VECTOR_STORE_SETUP_RECEIPT_MARKER = "vector-store-setup-receipt"
VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VectorStoreSetupReceipt:
    """v1.9 Phase 45: receipt produced when ``step_vector_store_setup`` runs.

    Records the active embedding provider/model/dim, the derived
    Chroma collection name, chunk + request counts, and timing —
    enough for an audit pass to verify which lane (cloud vs local)
    produced a given vector store directory without consulting Chroma
    metadata directly.

    The endpoint URL is never stored; only ``endpoint_label`` (typically
    redacted to ``http://localhost:<port>`` or ``mlx://<model>``).
    ``cost_usd`` is the aggregated cost from the embedding client's
    ``usage()`` (always 0.0 for MLX/Ollama; non-zero for OpenAI).
    """

    schema_version: int
    receipt_type: str  # VECTOR_STORE_SETUP_RECEIPT_MARKER
    provider: str  # "openai" | "mlx" | "ollama" | "fake"
    model: str
    embedding_dim: int
    collection_name: str
    chunk_count: int
    request_count: int
    cost_usd: float
    endpoint_label: str
    duration_s: float
    outcome: str  # "ok" | "failed" | "skipped"
    timestamp: str  # ISO-8601 UTC
    run_id: str | None
    stage_id: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


LOCAL_EMBEDDING_RECEIPT_MARKER = "local-embedding-receipt"
LOCAL_EMBEDDING_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LocalEmbeddingReceipt:
    """v1.9 Phase 45 Plan 04: per-call receipt for local embedding clients.

    Closes the BL-01 audit-trail gap surfaced in the Phase 44 review.
    Mirrors :class:`LocalLLMReceipt` shape — endpoint URL never
    persisted; only ``endpoint_label`` (mlx://<model> or
    http://localhost:<port>). ``cost_usd`` is always 0.0 — local
    inference has no cloud billing.

    v3.0 03-03 extension (D-10/LOCAL-03): optional paid-embedding fields
    appended at the end (additive, safe for existing positional callers).
    These allow recording OpenAI embedding calls with ``provider="openai"``
    and a ``cost_status`` of ``"estimated"`` or ``"unknown"`` while reusing
    the same receipt family (no new dataclass needed).

    T-03-03 mitigation: API keys are never persisted — only provider/model/
    dim/cost_status appear in the receipt.
    """

    schema_version: int
    receipt_type: str  # LOCAL_EMBEDDING_RECEIPT_MARKER
    provider: str  # "mlx" | "ollama" | "openai" | "fake"
    model: str
    embedding_dim: int
    endpoint_label: str
    input_count: int  # number of texts passed to embed_texts()
    total_input_chars: int  # sum of len(text) — proxy for tokens
    request_count: int  # number of HTTP / model invocations
    cost_usd: float  # 0.0 for local providers; actual cost for openai
    timestamp: str  # ISO-8601 UTC
    run_id: str | None
    stage_id: str | None
    # Additive paid-embedding fields (D-10 / LOCAL-03) — None on old receipts.
    cost_status: str | None = None  # "free" | "estimated" | "unknown" for paid
    task_id: str | None = None       # cost rollup attribution (D-08 threading)
    paid_opt_in_ref: str | None = None  # env var / flag that approved paid use
    # v3.0 03-04: cost-avoided fields (LOCAL-04 / D-13..D-16).
    # Additive; all None on old receipts — backward-compatible.
    benchmark_receipt_ref: str | None = None   # path/ref to benchmark receipt (T-03-04)
    cost_avoided_usd: float | None = None      # None = unknown (D-16 — never $0)
    cost_avoided_status: str | None = None     # "priced" | "unknown"
    paid_counterfactual_provider: str | None = None
    paid_counterfactual_model: str | None = None
    needs_pricing_backfill: bool | None = None  # True when cost_avoided_status=="unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


def write_local_embedding_receipt(
    receipt: LocalEmbeddingReceipt, workspace: Path
) -> Path:
    """Write a LocalEmbeddingReceipt atomically and return the path.

    Path convention (parallel to the v1.7 LocalLLMReceipt layout):
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/local-embedding-receipts/<stage>/<ts>-embed.json
      Without run_id: <workspace>/storage/local-embedding-receipts/<stage>/<ts>-embed.json
    """
    stage = safe_segment(receipt.stage_id or "embedding")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-embed.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "local-embedding-receipts" / stage / filename
        )
    else:
        path = (
            Path(workspace) / "storage" / "local-embedding-receipts"
            / stage / filename
        )
    _atomic_write_json(path, receipt.to_dict())
    return path


def record_local_embedding_call(
    *,
    provider: str,
    model: str,
    embedding_dim: int,
    endpoint_label: str,
    input_count: int,
    total_input_chars: int,
    request_count: int,
    cost_usd: float = 0.0,
    timestamp: str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    workspace: Path | None = None,
    # D-10 / LOCAL-03: paid-embedding additive fields.
    cost_status: str | None = None,   # "free" | "estimated" | "unknown"
    task_id: str | None = None,        # cost rollup attribution
    paid_opt_in_ref: str | None = None,  # approval reference
    # v3.0 03-04: cost-avoided additive fields (LOCAL-04 / D-13..D-16).
    benchmark_receipt_ref: str | None = None,   # path/ref to approving benchmark
    cost_avoided_usd: float | None = None,      # None = unknown (D-16)
    cost_avoided_status: str | None = None,     # "priced" | "unknown"
    paid_counterfactual_provider: str | None = None,
    paid_counterfactual_model: str | None = None,
    needs_pricing_backfill: bool | None = None,
) -> Path | None:
    """Construct and persist a LocalEmbeddingReceipt; return the path.

    No-op (returns None) when ``workspace`` is ``None`` — callers that
    don't have a workspace context (unit tests, ad-hoc embed calls)
    bypass the receipt write cleanly.

    D-10 extension: ``cost_status``, ``task_id``, and ``paid_opt_in_ref``
    are optional fields for recording paid (OpenAI) embedding calls with a
    ``provider="openai"`` and an explicit cost status. Local providers omit
    these fields (they remain ``None``). T-03-03: API keys are never
    written to the receipt.

    v3.0 03-04 (LOCAL-04): ``benchmark_receipt_ref``, ``cost_avoided_usd``,
    ``cost_avoided_status``, ``paid_counterfactual_provider``,
    ``paid_counterfactual_model``, and ``needs_pricing_backfill`` are additive
    cost-avoided fields stamped at substitution time (D-14). When
    ``cost_avoided_status`` is ``"unknown"``, ``cost_avoided_usd`` MUST be
    ``None`` — never ``$0`` (D-16 / T-03-09).
    """
    if workspace is None:
        return None
    from datetime import datetime, timezone
    receipt = LocalEmbeddingReceipt(
        schema_version=LOCAL_EMBEDDING_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_EMBEDDING_RECEIPT_MARKER,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        endpoint_label=endpoint_label,
        input_count=input_count,
        total_input_chars=total_input_chars,
        request_count=request_count,
        cost_usd=cost_usd,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        stage_id=stage_id,
        cost_status=cost_status,
        task_id=task_id,
        paid_opt_in_ref=paid_opt_in_ref,
        benchmark_receipt_ref=benchmark_receipt_ref,
        cost_avoided_usd=cost_avoided_usd,
        cost_avoided_status=cost_avoided_status,
        paid_counterfactual_provider=paid_counterfactual_provider,
        paid_counterfactual_model=paid_counterfactual_model,
        needs_pricing_backfill=needs_pricing_backfill,
    )
    return write_local_embedding_receipt(receipt, workspace)


LOCAL_VISION_RECEIPT_MARKER = "local-vision-receipt"
LOCAL_VISION_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LocalVisionCallReceipt:
    """v1.61: per-call receipt for local vision/OCR model calls.

    Sibling of :class:`LocalLLMReceipt` and :class:`LocalEmbeddingReceipt`.
    Endpoint URL is never stored — only ``endpoint_label`` (e.g.
    ``mlx-vision://<model-id>`` for managed servers,
    ``mlx-fastvlm://<model-id>`` for native FastVLM).
    Image bytes are never persisted; only their dimensions and byte size
    are recorded for cost/audit accounting. ``cost_usd`` is always 0.0;
    ``estimated_paid_equivalent_usd`` records what the same call would
    have cost on the paid lane (None when unknown).
    """

    schema_version: int
    receipt_type: str  # LOCAL_VISION_RECEIPT_MARKER
    provider: str  # "local_vlm" | "local_fastvlm" | "openai" | "anthropic"
    runtime_kind: str  # "vllm_mlx" | "mlx_vlm" | "mlx_native_fastvlm" | "remote"
    model: str
    endpoint_label: str  # mlx-vision://<model> or mlx-fastvlm://<model>
    vision_task: str  # "slide_analysis" | "figure_caption" | "chart_summary" | "ocr.slide_image_parse"
    image_count: int
    image_dimensions: list[list[int]]  # [[w, h], ...] per image
    image_bytes_size: int  # total bytes across all images in the call
    prompt_token_estimate: int | None
    completion_token_estimate: int | None
    latency_ms: float
    cost_usd: float  # 0.0 for local
    estimated_paid_equivalent_usd: float | None
    timestamp: str  # ISO-8601 UTC
    run_id: str | None
    stage_id: str | None
    receipt_path: str | None = None
    # v3.0 03-04: cost-avoided fields (LOCAL-04 / D-13..D-16).
    # Additive; all None on old receipts — backward-compatible.
    benchmark_receipt_ref: str | None = None   # path/ref to benchmark receipt (T-03-04)
    cost_avoided_usd: float | None = None      # None = unknown (D-16 — never $0)
    cost_avoided_status: str | None = None     # "priced" | "unknown"
    paid_counterfactual_provider: str | None = None
    paid_counterfactual_model: str | None = None
    needs_pricing_backfill: bool | None = None  # True when cost_avoided_status=="unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


def write_local_vision_receipt(
    receipt: LocalVisionCallReceipt, workspace: Path
) -> Path:
    """Write a LocalVisionCallReceipt atomically and return the path.

    Path convention (parallel to ``write_local_embedding_receipt``):
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/local-vision-receipts/<stage>/<ts>-vision.json
      Without run_id: <workspace>/storage/local-vision-receipts/<stage>/<ts>-vision.json
    """
    stage = safe_segment(receipt.stage_id or "vision")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-vision.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "local-vision-receipts" / stage / filename
        )
    else:
        path = (
            Path(workspace) / "storage" / "local-vision-receipts"
            / stage / filename
        )
    _atomic_write_json(path, receipt.to_dict())
    return path


def record_local_vision_call(
    *,
    provider: str,
    runtime_kind: str,
    model: str,
    endpoint_label: str,
    vision_task: str,
    image_count: int,
    image_dimensions: list[tuple[int, int]] | list[list[int]],
    image_bytes_size: int,
    prompt_token_estimate: int | None = None,
    completion_token_estimate: int | None = None,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    estimated_paid_equivalent_usd: float | None = None,
    timestamp: str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    workspace: Path | None = None,
    # v3.0 03-04: cost-avoided additive fields (LOCAL-04 / D-13..D-16).
    benchmark_receipt_ref: str | None = None,   # path/ref to approving benchmark
    cost_avoided_usd: float | None = None,      # None = unknown (D-16)
    cost_avoided_status: str | None = None,     # "priced" | "unknown"
    paid_counterfactual_provider: str | None = None,
    paid_counterfactual_model: str | None = None,
    needs_pricing_backfill: bool | None = None,
) -> Path | None:
    """Construct and persist a LocalVisionCallReceipt; return the path.

    No-op (returns None) when ``workspace`` is ``None`` — callers that
    don't have a workspace context (unit tests, ad-hoc vision calls)
    bypass the receipt write cleanly.

    v3.0 03-04 (LOCAL-04): ``benchmark_receipt_ref``, ``cost_avoided_usd``,
    ``cost_avoided_status``, ``paid_counterfactual_provider``,
    ``paid_counterfactual_model``, and ``needs_pricing_backfill`` are additive
    cost-avoided fields stamped at substitution time (D-14). When
    ``cost_avoided_status`` is ``"unknown"``, ``cost_avoided_usd`` MUST be
    ``None`` — never ``$0`` (D-16 / T-03-09).
    """
    if workspace is None:
        return None
    normalized_dims = [list(dim) for dim in image_dimensions]
    receipt = LocalVisionCallReceipt(
        schema_version=LOCAL_VISION_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_VISION_RECEIPT_MARKER,
        provider=provider,
        runtime_kind=runtime_kind,
        model=model,
        endpoint_label=endpoint_label,
        vision_task=vision_task,
        image_count=image_count,
        image_dimensions=normalized_dims,
        image_bytes_size=image_bytes_size,
        prompt_token_estimate=prompt_token_estimate,
        completion_token_estimate=completion_token_estimate,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        estimated_paid_equivalent_usd=estimated_paid_equivalent_usd,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        run_id=run_id,
        stage_id=stage_id,
        benchmark_receipt_ref=benchmark_receipt_ref,
        cost_avoided_usd=cost_avoided_usd,
        cost_avoided_status=cost_avoided_status,
        paid_counterfactual_provider=paid_counterfactual_provider,
        paid_counterfactual_model=paid_counterfactual_model,
        needs_pricing_backfill=needs_pricing_backfill,
    )
    return write_local_vision_receipt(receipt, workspace)


def scan_local_vision_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all LocalVisionCallReceipt dicts found under <workspace>/storage/."""
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("local-vision-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != LOCAL_VISION_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


VECTOR_STORE_REBUILD_RECEIPT_MARKER = "vector-store-rebuild-receipt"
VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VectorStoreRebuildReceipt:
    """v1.9 Phase 45: receipt produced by ``conference-pipeline vector-store rebuild``.

    Records the source-manifest hash for idempotency: a second
    invocation with unchanged source is a no-op (outcome=
    ``no-op-idempotent``). Operator id is recorded when available so
    audits can identify who ran the rebuild.
    """

    schema_version: int
    receipt_type: str  # VECTOR_STORE_REBUILD_RECEIPT_MARKER
    provider: str
    model: str
    embedding_dim: int
    collection_name: str
    source_manifest_hash: str
    source_manifest_path: str
    chunk_count: int
    duration_s: float
    outcome: str  # "rebuilt" | "no-op-idempotent" | "dry-run" | "failed"
    timestamp: str  # ISO-8601 UTC
    operator_id: str | None = None
    workspace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


def write_vector_store_rebuild_receipt(
    receipt: VectorStoreRebuildReceipt, workspace: Path
) -> Path:
    """Write a VectorStoreRebuildReceipt atomically and return the path.

    Path convention:
      <workspace>/storage/vector-store-receipts/rebuilds/<ts>-<provider>-<outcome>.json
    """
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-{safe_segment(receipt.provider)}-{safe_segment(receipt.outcome)}.json"
    path = (
        Path(workspace) / "storage" / "vector-store-receipts" / "rebuilds" / filename
    )
    _atomic_write_json(path, receipt.to_dict())
    return path


def write_vector_store_setup_receipt(
    receipt: VectorStoreSetupReceipt, workspace: Path
) -> Path:
    """Write a VectorStoreSetupReceipt atomically and return the path.

    Path convention (parallel to the v1.7 LocalLLMReceipt layout):
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/vector-store-receipts/<stage>/<ts>-setup.json
      Without run_id: <workspace>/storage/vector-store-receipts/<stage>/<ts>-setup.json
    """
    stage = safe_segment(receipt.stage_id or "vector-store-setup")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-setup.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "vector-store-receipts" / stage / filename
        )
    else:
        path = Path(workspace) / "storage" / "vector-store-receipts" / stage / filename
    _atomic_write_json(path, receipt.to_dict())
    return path


def write_local_llm_receipt(receipt: LocalLLMReceipt, workspace: Path) -> Path:
    """Write a LocalLLMReceipt to disk atomically and return the written path.

    Path convention (parallel to provider-calls):
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/local-llm-receipts/<stage>/<ts>-<mode>.json
      Without run_id: <workspace>/storage/local-llm-receipts/<stage>/<ts>-<mode>.json
    """
    stage = safe_segment(receipt.stage_id or "extraction")
    mode = (
        "preflight" if receipt.preflight_mode
        else "quality-sample" if receipt.quality_sample_mode
        else "extraction-run"
    )
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-{mode}.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "local-llm-receipts" / stage / filename
        )
    else:
        path = Path(workspace) / "storage" / "local-llm-receipts" / stage / filename
    _atomic_write_json(path, receipt.to_dict())
    return path


def write_escalation_receipt(receipt: EscalationReceipt, workspace: Path) -> Path:
    """Write an EscalationReceipt to disk atomically and return the written path.

    Path convention:
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/escalation-receipts/<stage>/<ts>.json
      Without run_id: <workspace>/storage/escalation-receipts/<stage>/<ts>.json
    """
    stage_id = receipt.wrapped_receipt.stage_id
    stage = safe_segment(stage_id or "escalation")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-escalation.json"
    run_id = receipt.wrapped_receipt.run_id
    if run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(run_id)
            / "escalation-receipts" / stage / filename
        )
    else:
        path = Path(workspace) / "storage" / "escalation-receipts" / stage / filename
    _atomic_write_json(path, receipt.to_dict())
    return path


def write_blocker_receipt(receipt: BlockerReceipt, workspace: Path) -> Path:
    stage = safe_segment(receipt.stage_id or "blocked-stage")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-{safe_segment(receipt.blocker_class)}.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "blocker-receipts" / stage / filename
        )
    else:
        path = Path(workspace) / "storage" / "blocker-receipts" / stage / filename
    _atomic_write_json(path, receipt.to_dict())
    return path


def write_unblock_receipt(receipt: UnblockReceipt, workspace: Path) -> Path:
    stage = "unblocked"
    blocker_path = _normalize_receipt_path(Path(workspace), receipt.blocker_receipt_path)
    if blocker_path is not None:
        stage = safe_segment(blocker_path.parent.name or stage)
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-{safe_segment(receipt.unblock_method)}.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "unblock-receipts" / stage / filename
        )
    else:
        path = Path(workspace) / "storage" / "unblock-receipts" / stage / filename
    _atomic_write_json(path, receipt.to_dict())
    return path


def scan_provider_call_receipts(workspace: Path) -> list[dict[str, Any]]:
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("provider-calls/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != RECEIPT_MARKER:
            continue
        payload.setdefault("receipt_path", str(path))
        receipts.append(payload)
    return receipts


def scan_local_llm_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all LocalLLMReceipt dicts found under <workspace>/storage/."""
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("local-llm-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != LOCAL_LLM_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


def scan_escalation_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all EscalationReceipt dicts found under <workspace>/storage/."""
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("escalation-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != ESCALATION_RECEIPT_MARKER:
            continue
        payload.setdefault("receipt_path", str(path))
        receipts.append(payload)
    return receipts


def scan_blocker_receipts(workspace: Path) -> list[dict[str, Any]]:
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("blocker-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != BLOCKER_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


def scan_unblock_receipts(workspace: Path) -> list[dict[str, Any]]:
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("unblock-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != UNBLOCK_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


# v1.9 review HIGH-6: scanner helpers for the new receipt types so audit
# pipelines can enumerate them via the canonical API instead of hand-
# rolling rglob calls. Mirrors the existing ``scan_local_llm_receipts``
# / ``scan_provider_call_receipts`` shape: filter by ``receipt_type``
# marker, attach ``receipt_path`` if missing.


def scan_local_embedding_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all LocalEmbeddingReceipt dicts found under <workspace>/storage/."""
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("local-embedding-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != LOCAL_EMBEDDING_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


def scan_vector_store_setup_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all VectorStoreSetupReceipt dicts found under <workspace>/storage/.

    Filters by the setup marker so this does not match
    VectorStoreRebuildReceipt entries, which share the
    ``vector-store-receipts/`` directory tree.
    """
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("vector-store-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != VECTOR_STORE_SETUP_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


def scan_vector_store_rebuild_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all VectorStoreRebuildReceipt dicts found under <workspace>/storage/.

    Rebuild receipts live under ``storage/vector-store-receipts/rebuilds/``;
    we still walk the broader tree and filter by marker so any future
    layout change continues to be caught.
    """
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("vector-store-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != VECTOR_STORE_REBUILD_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


def active_blockers(workspace: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    blockers = scan_blocker_receipts(workspace)
    if run_id is not None:
        blockers = [receipt for receipt in blockers if receipt.get("run_id") == run_id]
    blocker_paths = {
        _normalize_receipt_path(Path(workspace), str(receipt.get("receipt_path") or ""))
        for receipt in blockers
    }
    unblocked_paths = {
        _normalize_receipt_path(
            Path(workspace),
            str(receipt.get("blocker_receipt_path") or ""),
        )
        for receipt in scan_unblock_receipts(workspace)
        if run_id is None or receipt.get("run_id") == run_id
    }
    return [
        receipt
        for receipt in blockers
        if _normalize_receipt_path(Path(workspace), str(receipt.get("receipt_path") or ""))
        in blocker_paths - unblocked_paths
    ]


def is_receipt_stale(receipt: dict[str, Any] | LocalLLMReceipt, max_age_seconds: int) -> bool:
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    timestamp = (
        receipt.get("timestamp")
        if isinstance(receipt, dict)
        else receipt.timestamp
    )
    if not isinstance(timestamp, str):
        return True
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    age_seconds = (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()
    return age_seconds > max_age_seconds


def _normalize_receipt_path(workspace: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        direct_candidate = candidate
        try:
            if direct_candidate.exists():
                return direct_candidate.resolve()
        except OSError:
            pass
        candidate = Path(workspace) / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def aggregate_provider_call_receipts(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    totals_by_provider: dict[str, float] = {}
    totals_by_model: dict[str, float] = {}
    totals_by_operation: dict[str, float] = {}
    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    unknown_cost_receipts = 0
    failed_receipts = 0

    for receipt in receipts:
        provider = str(receipt.get("provider") or "unknown")
        model = str(receipt.get("model") or "unknown")
        operation = str(receipt.get("operation") or "unknown")
        failed = receipt.get("status") == "failed"
        cost = receipt.get("estimated_cost_usd")
        # A failed call's ``estimated_cost_usd`` is a duration/token projection
        # of what a SUCCESSFUL call would have cost — but the call did not
        # succeed, so no charge was incurred (a rejected request never
        # processes the input). Counting it inflates canonical spend, which
        # both misreports cost and can push cumulative-spend cap checks toward
        # a false halt. Attribute cost only for non-failed receipts; failed
        # ones are still tallied under ``failed_receipts`` for audit.
        if failed:
            failed_receipts += 1
        elif isinstance(cost, (int, float)):
            totals_by_provider[provider] = totals_by_provider.get(provider, 0.0) + float(cost)
            totals_by_model[f"{provider}:{model}"] = totals_by_model.get(f"{provider}:{model}", 0.0) + float(cost)
            totals_by_operation[operation] = totals_by_operation.get(operation, 0.0) + float(cost)
        else:
            unknown_cost_receipts += 1
        for key in token_totals:
            value = receipt.get(key)
            if isinstance(value, int):
                token_totals[key] += value

    return {
        "canonical_receipt_files_scanned": len(receipts),
        "canonical_total_estimated_cost_usd": round(sum(totals_by_provider.values()), 6),
        "canonical_totals_by_provider": {
            key: round(value, 6) for key, value in sorted(totals_by_provider.items())
        },
        "canonical_totals_by_model": {
            key: round(value, 6) for key, value in sorted(totals_by_model.items())
        },
        "canonical_totals_by_operation": {
            key: round(value, 6) for key, value in sorted(totals_by_operation.items())
        },
        "canonical_token_totals": {key: value for key, value in token_totals.items() if value},
        "unknown_cost_receipts": unknown_cost_receipts,
        "failed_receipts": failed_receipts,
        "receipt_paths": sorted(str(receipt.get("receipt_path")) for receipt in receipts if receipt.get("receipt_path")),
    }


def provider_receipt_summary(workspace: Path, run_id: str | None = None, stage_id: str | None = None) -> dict[str, Any]:
    receipts = scan_provider_call_receipts(workspace)
    if run_id is not None:
        receipts = [receipt for receipt in receipts if receipt.get("run_id") == run_id]
    if stage_id is not None:
        receipts = [receipt for receipt in receipts if receipt.get("stage_id") == stage_id]
    return aggregate_provider_call_receipts(receipts)


# ---------------------------------------------------------------------------
# Local-fallback receipt family (LOCAL-02 / T-03-02 / T-03-05)
# ---------------------------------------------------------------------------
# Written by resolve_local_first_selection when the local-first resolver falls
# back to the paid provider.  Kept intentionally small — it is an audit trail
# artifact, not a cost receipt.  No endpoint URL, no API key, no model
# weights are written.  sanitize_for_receipt() scrubs any accidental leaks.

LOCAL_FALLBACK_RECEIPT_MARKER = "local-fallback-receipt"
LOCAL_FALLBACK_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LocalFallbackReceipt:
    """Receipt written when the local-first resolver falls back to a paid provider.

    Schema version 1.  Marker ``local-fallback-receipt``.

    Fields:
        task_id           -- model-governance task id that triggered the fallback
        stage_id          -- pipeline stage id (may be None for non-staged callers)
        run_id            -- workflow-run id (may be None)
        requested_provider -- the local provider that was attempted ("local")
        requested_model   -- the local model that was requested (may be "" if unknown)
        fallback_provider -- the paid provider that will handle the call
        fallback_model    -- the paid model that will handle the call
        fallback_reason   -- one of: local_first_disabled, task_forced_paid,
                             approved_local_unavailable, local_readiness_failed
        benchmark_receipt_ref -- path/ref to the benchmark receipt for the local
                                 model (None when no approved local exists)
        cost_cap_checked  -- always True; the Phase-02 budget cap still guards spend
        timestamp         -- ISO-8601 UTC
        receipt_path      -- set by write_local_fallback_receipt; None before write
    """

    schema_version: int
    receipt_type: str   # LOCAL_FALLBACK_RECEIPT_MARKER
    task_id: str
    stage_id: str | None
    run_id: str | None
    requested_provider: str
    requested_model: str
    fallback_provider: str
    fallback_model: str
    fallback_reason: str
    benchmark_receipt_ref: str | None
    cost_cap_checked: bool
    timestamp: str
    receipt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return sanitize_for_receipt(payload)


def write_local_fallback_receipt(
    receipt: LocalFallbackReceipt, workspace: Path
) -> Path:
    """Write a LocalFallbackReceipt atomically and return the path.

    Path convention:
      With run_id:    <workspace>/storage/workflow-runs/<run_id>/local-fallback-receipts/<stage>/<ts>-fallback.json
      Without run_id: <workspace>/storage/local-fallback-receipts/<stage>/<ts>-fallback.json
    """
    stage = safe_segment(receipt.stage_id or "fallback")
    ts = receipt.timestamp.replace(":", "").replace("-", "")
    filename = f"{ts}-fallback.json"
    if receipt.run_id:
        path = (
            Path(workspace) / "storage" / "workflow-runs"
            / safe_segment(receipt.run_id)
            / "local-fallback-receipts" / stage / filename
        )
    else:
        path = (
            Path(workspace) / "storage" / "local-fallback-receipts"
            / stage / filename
        )
    _atomic_write_json(path, receipt.to_dict())
    return path


def record_local_fallback(
    *,
    task_id: str,
    stage_id: str | None = None,
    run_id: str | None = None,
    requested_provider: str = "local",
    requested_model: str = "",
    fallback_provider: str,
    fallback_model: str,
    fallback_reason: str,
    benchmark_receipt_ref: str | None = None,
    workspace: Path | None = None,
    timestamp: str | None = None,
) -> Path | None:
    """Construct and persist a LocalFallbackReceipt; return the written path.

    No-op (returns None) when ``workspace`` is None — keeps the
    resolve_local_first_selection resolver pure in test/dry-run contexts.

    T-03-02: always written when run context exists (audit trail).
    T-03-05: sanitize_for_receipt() called inside to_dict().
    cost_cap_checked is always True — the Phase-02 budget cap still guards spend.
    """
    if workspace is None:
        return None
    ts = timestamp or utc_now()
    receipt = LocalFallbackReceipt(
        schema_version=LOCAL_FALLBACK_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_FALLBACK_RECEIPT_MARKER,
        task_id=task_id,
        stage_id=stage_id,
        run_id=run_id,
        requested_provider=requested_provider,
        requested_model=requested_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        fallback_reason=fallback_reason,
        benchmark_receipt_ref=benchmark_receipt_ref,
        cost_cap_checked=True,
        timestamp=ts,
    )
    written_path = write_local_fallback_receipt(receipt, workspace)
    return written_path


def scan_local_fallback_receipts(workspace: Path) -> list[dict[str, Any]]:
    """Return all LocalFallbackReceipt dicts found under <workspace>/storage/.

    Filters by LOCAL_FALLBACK_RECEIPT_MARKER and attaches receipt_path when
    absent.  Mirrors the shape of scan_local_llm_receipts and
    scan_local_embedding_receipts.
    """
    storage = Path(workspace) / "storage"
    if not storage.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in storage.rglob("local-fallback-receipts/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("receipt_type") != LOCAL_FALLBACK_RECEIPT_MARKER:
            continue
        if not payload.get("receipt_path"):
            payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts


__all__ = [
    "BLOCKER_RECEIPT_MARKER",
    "BLOCKER_RECEIPT_SCHEMA_VERSION",
    "BlockerClass",
    "BlockerReceipt",
    "EscalationReceipt",
    "ESCALATION_RECEIPT_MARKER",
    "ESCALATION_RECEIPT_SCHEMA_VERSION",
    # v3.0 03-02: local-fallback receipt family (LOCAL-02, T-03-02, T-03-05)
    "LOCAL_FALLBACK_RECEIPT_MARKER",
    "LOCAL_FALLBACK_RECEIPT_SCHEMA_VERSION",
    "LocalFallbackReceipt",
    "record_local_fallback",
    "scan_local_fallback_receipts",
    "write_local_fallback_receipt",
    # v1.9 Phase 45: new receipt types — exported so audit consumers can
    # import them via the canonical API (review HIGH-6).
    "LOCAL_EMBEDDING_RECEIPT_MARKER",
    "LOCAL_EMBEDDING_RECEIPT_SCHEMA_VERSION",
    "LocalEmbeddingReceipt",
    "LocalLLMReceipt",
    "LOCAL_LLM_RECEIPT_MARKER",
    "LOCAL_LLM_RECEIPT_SCHEMA_VERSION",
    # Phase 02-01: local-vision symbols added to __all__ (Pitfall 4 fix)
    "LOCAL_VISION_RECEIPT_MARKER",
    "LOCAL_VISION_RECEIPT_SCHEMA_VERSION",
    "LocalVisionCallReceipt",
    "scan_local_vision_receipts",
    "write_local_vision_receipt",
    "MissingRequiredReceiptError",
    "ProviderCallReceipt",
    "ProviderCallResult",
    "ProviderReceiptContext",
    "StaleReceiptError",
    "UNBLOCK_RECEIPT_MARKER",
    "UNBLOCK_RECEIPT_SCHEMA_VERSION",
    "UnblockReceipt",
    "VECTOR_STORE_REBUILD_RECEIPT_MARKER",
    "VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION",
    "VECTOR_STORE_SETUP_RECEIPT_MARKER",
    "VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION",
    "VectorStoreRebuildReceipt",
    "VectorStoreSetupReceipt",
    "active_blockers",
    "aggregate_provider_call_receipts",
    "anthropic_messages_create",
    "anthropic_messages_create_async",
    "current_provider_receipt_context",
    "gemini_generate_content",
    "gemini_generate_content_stream",
    "legacy_gemini_generate_content",
    "litellm_completion",
    "local_openai_chat_completions_create",
    "openai_audio_transcriptions_create_async",
    "openai_chat_completions_create",
    "openai_chat_completions_create_async",
    "openai_embeddings_create",
    "openai_responses_create",
    "provider_receipt_context",
    "provider_receipt_summary",
    "record_async_provider_call",
    "record_local_embedding_call",
    "record_provider_call",
    "scan_blocker_receipts",
    "scan_escalation_receipts",
    "scan_local_embedding_receipts",
    "scan_local_llm_receipts",
    "scan_provider_call_receipts",
    "scan_unblock_receipts",
    "scan_vector_store_rebuild_receipts",
    "scan_vector_store_setup_receipts",
    "is_receipt_stale",
    "utc_now",
    "write_blocker_receipt",
    "write_escalation_receipt",
    "write_local_embedding_receipt",
    "write_local_llm_receipt",
    "write_unblock_receipt",
    "write_vector_store_rebuild_receipt",
    "write_vector_store_setup_receipt",
]
