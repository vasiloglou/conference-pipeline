"""Heuristic + local-only LLM-assist classifier for exhaustive extraction.

Default code path is ``classify_text_span`` (keyword-overlap over the v1.4
taxonomy). Optional OpenAI-compatible local LLM assist is gated behind
``EXTRACTION_LLM_ASSIST`` or ``EXTRACTION_LOCAL_LLM_ASSIST`` and requires a
loopback ``EXTRACTION_LOCAL_LLM_BASE_URL``. It is bounded by
``LLM_ASSIST_MAX_CALLS`` (default 1_000_000_000 — effectively unbounded for
local LLM calls; paid spend is protected separately by
``workflow.provider_cost.budget_cap_usd``).

Heuristic threshold ``tau`` (default 0.15) controls when an artifact is
considered confidently classified. Override per-run via
``EXTRACTION_HEURISTIC_TAU`` env var.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conference_pipeline.extractions.records import (
    ExtractionRecord,
    TopicMatch,
)
from conference_pipeline.model_governance import default_fallback_model
from conference_pipeline.iclr2025_live import (
    COMPLETED_MEDIA_PROCESSING_STATUSES,
    TOPIC_KEYWORDS,
    TopicFamily,
    sanitize_for_receipt,
    utc_now,
)
from conference_pipeline.provider_receipts import (
    ESCALATION_RECEIPT_MARKER,
    ESCALATION_RECEIPT_SCHEMA_VERSION,
    LOCAL_LLM_RECEIPT_MARKER,
    LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
    EscalationReceipt,
    LocalLLMReceipt,
    openai_chat_completions_create,
    provider_receipt_context,
    write_escalation_receipt,
    write_local_llm_receipt,
)

if TYPE_CHECKING:
    from conference_pipeline.extractions.walker import ExtractionInput


_DEFAULT_TAU = 0.15
_DEFAULT_MIN_CHARS = 200
_DEFAULT_LLM_MAX_CHARS = 12000
_DEFAULT_LLM_MAX_OUTPUT_TOKENS = 512
_DEFAULT_LLM_MAX_CALLS = 1_000_000_000
_DEFAULT_LOCAL_LLM_MODEL = "local-extraction-classifier"
_DEFAULT_LOCAL_LLM_ENDPOINT_LABEL = "local-openai-compatible"
_DEFAULT_CLOUD_ESCALATION_PROVIDER = "openai"
_DEFAULT_CLOUD_ESCALATION_MODEL = default_fallback_model("source_reading.categorization")
_DEFAULT_CLOUD_ESCALATION_RETRIES = 3
_DEFAULT_CLOUD_ESCALATION_RETRY_DELAY_SECONDS = 1.0
_TITLE_KEYWORD_STOPWORDS = frozenset(
    {
        "and",
        "for",
        "the",
        "with",
        "from",
        "into",
        "over",
        "using",
        "use",
        "systems",
        "models",
        "model",
        "learning",
        "ai",
    }
)

# Module-level call counter, protected by a lock for thread-safety.
_LLM_ASSIST_LOCK = threading.Lock()
_LLM_ASSIST_CALL_COUNT = 0

_API_KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"GOCSPX-[A-Za-z0-9_\-]{16,}"),
    re.compile(
        r"(?i)(api[_-]?key|secret|token|cookie|authorization|bearer)\s*[:=]\s*\S+"
    ),
)


class LocalLLMNotReadyError(RuntimeError):
    """Raised when check_local_llm_readiness() fails any check.

    Attributes:
        check_failed: Which check failed — "reachability", "model_not_listed",
            "inference", or "context_window".
        endpoint_label: The label of the endpoint (never the URL) from
            ``EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL``.
        fallback_options: Suggestions for the caller on how to recover.
    """

    def __init__(
        self,
        *,
        check_failed: str,
        endpoint_label: str,
        message: str,
        fallback_options: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.check_failed = check_failed
        self.endpoint_label = endpoint_label
        self.fallback_options = fallback_options or []


class CloudEscalationError(RuntimeError):
    """Raised when threshold-triggered cloud escalation cannot complete.

    This is intentionally not mapped to a BlockerReceipt. In workflow execution
    it should behave like an ordinary failed stage receipt with
    ``next_resume_stage`` pointing at the failed stage.
    """


class QualitySampleThresholdError(RuntimeError):
    """Raised when the quality sample remains below fail-closed thresholds."""

    def __init__(
        self,
        message: str,
        *,
        recommendation: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.recommendation = recommendation
        self.result = result


def _resolve_tau(tau: float | None) -> float:
    if tau is not None:
        return float(tau)
    raw = os.environ.get("EXTRACTION_HEURISTIC_TAU")
    if not raw:
        return _DEFAULT_TAU
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TAU


def _redact(message: str) -> str:
    """Strip API-key-shaped substrings from a free-form error message.

    Defense in depth: after the local prefix-anchored patterns, also pass the
    message through ``sanitize_for_receipt`` so long-opaque token-shaped runs
    without a recognizable prefix (signed cookies, refresh tokens, pre-auth
    URLs, etc.) are caught by ``LONG_OPAQUE_RE`` before truncation.
    """
    out = str(message)
    for pattern in _API_KEY_PATTERNS:
        out = pattern.sub("REDACTED", out)
    sanitized = sanitize_for_receipt(out)
    if isinstance(sanitized, str):
        out = sanitized
    return out[:500]


def _keywords_for_family(family: TopicFamily) -> tuple[str, ...]:
    keywords = TOPIC_KEYWORDS.get(family.id, ())
    if keywords:
        return keywords
    raw = " ".join([family.title, family.id.replace("_", " ")])
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", raw.lower())
        if token not in _TITLE_KEYWORD_STOPWORDS
    ]
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def classify_text_span(
    text: str,
    taxonomy: list[TopicFamily],
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
    tau: float | None = None,
) -> list[TopicMatch]:
    """Heuristic classifier — keyword overlap over the v1.4 taxonomy.

    Returns ``[]`` when the input text is empty/too short. Otherwise emits one
    ``TopicMatch`` per topic family whose confidence meets ``tau``.
    """
    if not text or len(text) < min_chars:
        return []

    tau_resolved = _resolve_tau(tau)
    lower = text.lower()
    matches: list[TopicMatch] = []
    for family in taxonomy:
        keywords = _keywords_for_family(family)
        if not keywords:
            continue
        score = 0
        for keyword in keywords:
            if not keyword:
                continue
            score += lower.count(keyword.lower())
        if score <= 0:
            continue
        confidence = min(1.0, score / max(1, len(keywords)))
        if confidence < tau_resolved:
            continue
        matches.append(
            TopicMatch(topic_id=family.id, score=float(score), confidence=confidence)
        )
    matches.sort(key=lambda m: (-m.confidence, -m.score, m.topic_id))
    return matches


def _llm_assist_call_limit() -> int:
    raw = os.environ.get("LLM_ASSIST_MAX_CALLS")
    if not raw:
        return _DEFAULT_LLM_MAX_CALLS
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_LLM_MAX_CALLS


def _llm_max_output_tokens() -> int:
    raw = (
        os.environ.get("EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS")
        or os.environ.get("EXTRACTION_LOCAL_LLM_MAX_OUTPUT_TOKENS")
    )
    if not raw:
        return _DEFAULT_LLM_MAX_OUTPUT_TOKENS
    try:
        return min(max(1, int(raw)), 2048)
    except ValueError:
        return _DEFAULT_LLM_MAX_OUTPUT_TOKENS


def _llm_assist_enabled() -> bool:
    raw = (
        os.environ.get("EXTRACTION_LOCAL_LLM_ASSIST")
        or os.environ.get("EXTRACTION_LLM_ASSIST")
        or ""
    )
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _local_llm_base_url() -> str:
    return (
        os.environ.get("EXTRACTION_LOCAL_LLM_BASE_URL")
        or os.environ.get("EXTRACTION_LLM_BASE_URL")
        or ""
    ).strip()


def _local_llm_model(default: str = _DEFAULT_LOCAL_LLM_MODEL) -> str:
    return (
        os.environ.get("EXTRACTION_LOCAL_LLM_MODEL")
        or os.environ.get("EXTRACTION_LLM_MODEL")
        or default
    ).strip() or default


def _local_llm_endpoint_label() -> str:
    return (
        os.environ.get("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL")
        or os.environ.get("EXTRACTION_LLM_ENDPOINT_LABEL")
        or _DEFAULT_LOCAL_LLM_ENDPOINT_LABEL
    ).strip() or _DEFAULT_LOCAL_LLM_ENDPOINT_LABEL


def _paid_cost_cap_usd() -> float:
    raw = (
        os.environ.get("EXTRACTION_LOCAL_LLM_MAX_PAID_COST_USD")
        or os.environ.get("EXTRACTION_LLM_MAX_PAID_COST_USD")
        or "0"
    )
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _is_loopback_base_url(base_url: str) -> bool:
    """Return True iff ``base_url`` is an http(s) URL pointing at loopback.

    v1.9 review HIGH-4: delegates host-classification to the shared
    strict helper :func:`conference_pipeline._security.loopback.is_loopback_url`
    so all three loopback boundaries in the project agree on what
    "loopback" means. The additional ``http``/``https`` scheme guard is
    preserved here because this site is the inbound check for a
    user-configured base URL that must be HTTP-shaped.
    """
    from urllib.parse import urlparse

    from conference_pipeline._security.loopback import is_loopback_url

    try:
        parsed = urlparse(base_url)
        parsed.port
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and is_loopback_url(base_url)


def _redacted_base_url_identity(base_url: str) -> str:
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base_url)
        port_value = parsed.port
    except ValueError:
        return "invalid://unknown"
    host = parsed.hostname or "unknown"
    port = f":{port_value}" if port_value else ""
    path = parsed.path.rstrip("/")
    scheme = parsed.scheme or "unknown"
    return f"{scheme}://{host}{port}{path}"


def _local_provider_meta(
    *,
    provider: str = "local-openai-compatible",
    model: str | None = None,
    endpoint_label: str | None = None,
    base_url: str | None = None,
    escalation_reason: str,
    status: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    request_id: str | None = None,
    receipt_path: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model or _local_llm_model(),
        "local_endpoint_label": endpoint_label or _local_llm_endpoint_label(),
        "redacted_base_url_identity": _redacted_base_url_identity(base_url)
        if base_url
        else None,
        "call_count": _LLM_ASSIST_CALL_COUNT,
        "max_call_count": _llm_assist_call_limit(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "confidence_path": "heuristic_none_then_local_llm_assist",
        "escalation_reason": escalation_reason,
        "estimated_paid_cost_usd": 0.0,
        "paid_cost_cap_usd": _paid_cost_cap_usd(),
        "cost_status": "local_no_paid_provider",
        "status": status,
        "request_id": request_id,
        "provider_receipt_path": receipt_path,
        **({"redacted_message": error} if error else {}),
    }


def reset_llm_assist_counter() -> None:
    """Test-only helper: zero the per-process LLM-assist counter."""
    global _LLM_ASSIST_CALL_COUNT  # noqa: PLW0603 - intentional shared state
    with _LLM_ASSIST_LOCK:
        _LLM_ASSIST_CALL_COUNT = 0


def check_local_llm_readiness(
    *,
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
) -> LocalLLMReceipt:
    """Run a local LLM readiness preflight and write a LocalLLMReceipt to disk.

    Checks in order:
    1. Base URL is set and is a loopback address.
    2. GET /v1/models returns 200 and lists the configured model.
    3. One sample inference call succeeds.

    Raises LocalLLMNotReadyError if any check fails — never silently routes to cloud.
    Returns LocalLLMReceipt(preflight_mode=True) on success.
    """
    from conference_pipeline.extractions.backends import select_local_llm_backend

    backend = select_local_llm_backend()
    backend.readiness()

    # All checks passed — write preflight receipt
    receipt = LocalLLMReceipt(
        schema_version=LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_LLM_RECEIPT_MARKER,
        endpoint_label=backend.endpoint_label,
        model=backend.model,
        request_count=1,
        token_count=None,
        confidence_path="local_llm",
        skipped_record_count=0,
        paid_cloud_cost=0.0,
        timestamp=utc_now(),
        run_id=run_id,
        stage_id=stage_id or "preflight",
        preflight_mode=True,
        quality_sample_mode=False,
    )
    if workspace is not None:
        write_local_llm_receipt(receipt, workspace)
    return receipt


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_token_values(*values: Any) -> int | None:
    tokens = [_coerce_int(value) for value in values]
    concrete = [value for value in tokens if value is not None]
    if not concrete:
        return None
    return sum(concrete)


def _local_llm_usage_from_provider_metadata(
    provider_metadata: dict[str, Any] | None,
) -> tuple[int, int | None]:
    if not provider_metadata or not provider_metadata.get("provider_receipt_path"):
        return 0, None
    return 1, _sum_token_values(
        provider_metadata.get("input_tokens"),
        provider_metadata.get("output_tokens"),
    )


def _local_llm_usage_from_provider_receipt(receipt: Any) -> tuple[int, int | None]:
    if receipt is None or not getattr(receipt, "receipt_path", None):
        return 0, None
    return 1, _sum_token_values(
        getattr(receipt, "input_tokens", None),
        getattr(receipt, "output_tokens", None),
    )


def _parse_llm_json_object(choice: str) -> dict[str, Any]:
    """Parse a JSON object from local model output.

    Some local instruction-tuned models ignore ``response_format`` hints and
    wrap the object in markdown fences or brief reasoning text. The caller still
    validates the parsed schema, so this helper only recovers the outer object.
    """
    import json as _json

    raw = (choice or "").strip()
    if not raw:
        return {}
    original_error: _json.JSONDecodeError | None = None
    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        original_error = exc
        parsed: Any = None
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            fenced = "\n".join(lines).strip()
            if fenced:
                try:
                    parsed = _json.loads(fenced)
                except _json.JSONDecodeError:
                    parsed = None
        if parsed is None:
            decoder = _json.JSONDecoder()
            candidates: list[dict[str, Any]] = []
            for index, char in enumerate(raw):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(raw[index:])
                except _json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    candidates.append(candidate)
            if not candidates:
                raise original_error
            parsed = candidates[-1]
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _cloud_escalation_provider() -> str:
    provider = (
        os.environ.get("EXTRACTION_CLOUD_ESCALATION_PROVIDER")
        or _DEFAULT_CLOUD_ESCALATION_PROVIDER
    )
    return provider.strip().lower()


def _cloud_escalation_model() -> str:
    return (
        os.environ.get("EXTRACTION_CLOUD_ESCALATION_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or _DEFAULT_CLOUD_ESCALATION_MODEL
    ).strip() or _DEFAULT_CLOUD_ESCALATION_MODEL


def _cloud_escalation_retries() -> int:
    raw = os.environ.get("EXTRACTION_CLOUD_ESCALATION_RETRIES")
    if not raw:
        return _DEFAULT_CLOUD_ESCALATION_RETRIES
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_CLOUD_ESCALATION_RETRIES


def _cloud_escalation_retry_delay_seconds() -> float:
    raw = os.environ.get("EXTRACTION_CLOUD_ESCALATION_RETRY_DELAY_SECONDS")
    if not raw:
        return _DEFAULT_CLOUD_ESCALATION_RETRY_DELAY_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_CLOUD_ESCALATION_RETRY_DELAY_SECONDS


def _quality_sample_record_id(record: Any, index: int) -> str:
    for attr in ("artifact_id", "source_artifact_id", "id"):
        value = getattr(record, attr, None)
        if value:
            return str(value)
    if isinstance(record, dict):
        for key in ("artifact_id", "source_artifact_id", "id"):
            value = record.get(key)
            if value:
                return str(value)
    return f"sample-{index + 1}"


def _quality_sample_escalation_prompt(
    *,
    affected_record_ids: tuple[str, ...],
    confidence_mean: float,
    confidence_p50: float,
    confidence_p90: float,
    failure_classes: dict[str, int],
    failure_class_rate: float,
    record_coverage: float,
    thresholds: dict[str, Any],
) -> str:
    return (
        "A local LLM quality sample fell below the configured cloud escalation "
        "trigger. Review these aggregate extraction-quality signals and return "
        "a compact JSON object with keys action, rationale, and next_checks. "
        "Do not include secrets or raw source content.\n\n"
        f"affected_record_ids: {list(affected_record_ids)}\n"
        f"confidence_mean: {confidence_mean:.4f}\n"
        f"confidence_p50: {confidence_p50:.4f}\n"
        f"confidence_p90: {confidence_p90:.4f}\n"
        f"failure_classes: {failure_classes}\n"
        f"failure_class_rate: {failure_class_rate:.4f}\n"
        f"record_coverage: {record_coverage:.4f}\n"
        f"effective_thresholds: {thresholds}\n"
    )


def _invoke_openai_cloud_escalation(
    *,
    model: str,
    prompt: str,
) -> Any:
    api_key = os.environ.get("EXTRACTION_CLOUD_ESCALATION_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise CloudEscalationError(
            "Cloud escalation requires OPENAI_API_KEY or "
            "EXTRACTION_CLOUD_ESCALATION_API_KEY."
        )

    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(api_key=api_key)
    return openai_chat_completions_create(
        client,
        caller=__file__,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an extraction quality reviewer. Return only JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=300,
    )


def _run_cloud_escalation(
    *,
    workspace: Path,
    run_id: str | None,
    stage_id: str | None,
    affected_record_ids: tuple[str, ...],
    fail_closed_state: bool,
    confidence_mean: float,
    confidence_p50: float,
    confidence_p90: float,
    failure_classes: dict[str, int],
    failure_class_rate: float,
    record_coverage: float,
    thresholds_payload: dict[str, Any],
) -> dict[str, Any]:
    provider = _cloud_escalation_provider()
    model = _cloud_escalation_model()
    if provider != "openai":
        raise CloudEscalationError(
            "Unsupported EXTRACTION_CLOUD_ESCALATION_PROVIDER="
            f"{provider!r}; supported values: openai."
        )

    prompt = _quality_sample_escalation_prompt(
        affected_record_ids=affected_record_ids,
        confidence_mean=confidence_mean,
        confidence_p50=confidence_p50,
        confidence_p90=confidence_p90,
        failure_classes=failure_classes,
        failure_class_rate=failure_class_rate,
        record_coverage=record_coverage,
        thresholds=thresholds_payload,
    )
    attempts = _cloud_escalation_retries()
    retry_delay = _cloud_escalation_retry_delay_seconds()
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with provider_receipt_context(
                workspace_root=workspace,
                run_id=run_id,
                stage_id=stage_id or "quality-sample",
                caller=__file__,
            ):
                result = _invoke_openai_cloud_escalation(
                    model=model,
                    prompt=prompt,
                )
        except Exception as exc:  # noqa: BLE001 - retry boundary
            last_exc = exc
            if attempt < attempts and retry_delay:
                time.sleep(retry_delay)
            continue

        receipt = EscalationReceipt(
            schema_version=ESCALATION_RECEIPT_SCHEMA_VERSION,
            receipt_type=ESCALATION_RECEIPT_MARKER,
            wrapped_receipt=result.receipt,
            escalation_reason="quality_sample_below_cloud_escalation_trigger",
            affected_record_ids=affected_record_ids,
            fail_closed_state=fail_closed_state,
            timestamp=utc_now(),
        )
        receipt_path = write_escalation_receipt(receipt, workspace)
        return {
            "status": "succeeded",
            "provider": provider,
            "model": model,
            "attempts": attempt,
            "escalation_receipt_path": str(receipt_path),
            "provider_receipt_path": result.receipt.receipt_path,
            "estimated_cost_usd": result.receipt.estimated_cost_usd,
        }

    error = CloudEscalationError(
        "Cloud escalation failed after "
        f"{attempts} attempt(s): {_redact(str(last_exc or 'unknown error'))}"
    )
    if last_exc is not None and hasattr(last_exc, "provider_receipt"):
        setattr(error, "provider_receipt", getattr(last_exc, "provider_receipt"))
    raise error from last_exc


def run_quality_sample(
    records: list[Any],
    taxonomy: list[Any],
    *,
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    sample_size: int | None = None,
    threshold_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a representative local LLM quality sample before full extraction.

    Processes ``sample_size`` records (default: EXTRACTION_QUALITY_SAMPLE_SIZE
    env var, or 20). Returns a dict with confidence stats and writes a
    LocalLLMReceipt(quality_sample_mode=True) to disk.

    Raises LocalLLMNotReadyError if the base URL is not loopback.
    """
    import statistics

    from conference_pipeline.extractions.backends import (
        BackendKind,
        select_local_llm_backend,
    )
    from conference_pipeline.extractions.thresholds import (
        default_quality_thresholds,
        resolve_quality_thresholds,
    )

    backend = select_local_llm_backend()
    if (
        backend.kind is BackendKind.HTTP
        and (
            not getattr(backend, "base_url", "")
            or not _is_loopback_base_url(str(getattr(backend, "base_url", "")))
        )
    ):
        raise LocalLLMNotReadyError(
            check_failed="reachability",
            endpoint_label=backend.endpoint_label,
            message=(
                "run_quality_sample() requires a loopback base URL. "
                "Call check_local_llm_readiness() first."
            ),
        )
    thresholds = resolve_quality_thresholds(workspace, threshold_overrides)

    effective_sample_size = sample_size
    if effective_sample_size is None:
        raw_env = os.environ.get("EXTRACTION_QUALITY_SAMPLE_SIZE")
        try:
            effective_sample_size = int(raw_env) if raw_env else 20
        except ValueError:
            effective_sample_size = 20

    sample_records = records[:effective_sample_size]

    confidences: list[float] = []
    failure_classes: dict[str, int] = {}
    request_count = 0
    token_count_total = 0
    saw_token_count = False
    affected_record_ids = tuple(
        _quality_sample_record_id(record, index)
        for index, record in enumerate(sample_records)
    )

    for record in sample_records:
        try:
            extraction_record = classify_artifact(record, taxonomy, run_id=run_id or "quality-sample")
            record_requests, record_tokens = _local_llm_usage_from_provider_metadata(
                extraction_record.provider_metadata,
            )
            request_count += record_requests
            if record_tokens is not None:
                token_count_total += record_tokens
                saw_token_count = True
            confidences.append(extraction_record.confidence)
        except Exception as exc:  # noqa: BLE001 - boundary catch per pattern
            record_requests, record_tokens = _local_llm_usage_from_provider_receipt(
                getattr(exc, "provider_receipt", None),
            )
            request_count += record_requests
            if record_tokens is not None:
                token_count_total += record_tokens
                saw_token_count = True
            cls = type(exc).__name__
            failure_classes[cls] = failure_classes.get(cls, 0) + 1

    if confidences:
        conf_mean = statistics.mean(confidences)
        conf_p50 = statistics.median(confidences)
        sorted_conf = sorted(confidences)
        p90_idx = max(0, int(len(sorted_conf) * 0.9) - 1)
        conf_p90 = sorted_conf[p90_idx]
    else:
        conf_mean = conf_p50 = conf_p90 = 0.0
    failure_rate = sum(failure_classes.values()) / max(1, len(sample_records))
    coverage = len(confidences) / max(1, len(sample_records))
    should_proceed = (
        conf_mean >= thresholds.min_mean_confidence
        and failure_rate <= thresholds.max_failure_class_rate
        and coverage >= thresholds.min_record_coverage
    )
    should_escalate = conf_mean < thresholds.cloud_escalation_trigger
    recommendation = {
        "should_proceed": should_proceed,
        "should_escalate": should_escalate,
        "reason": (
            "quality sample fell below one or more canonical thresholds"
            if not should_proceed
            else "quality sample meets canonical thresholds"
        ),
    }
    defaults = default_quality_thresholds()
    thresholds_payload = asdict(thresholds)
    default_threshold_payload = asdict(defaults)
    token_count = token_count_total if saw_token_count else None

    receipt = LocalLLMReceipt(
        schema_version=LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_LLM_RECEIPT_MARKER,
        endpoint_label=backend.endpoint_label,
        model=backend.model,
        request_count=request_count,
        token_count=token_count,
        confidence_path="local_llm",
        skipped_record_count=len(records) - len(sample_records),
        paid_cloud_cost=0.0,
        timestamp=utc_now(),
        run_id=run_id,
        stage_id=stage_id or "quality-sample",
        preflight_mode=False,
        quality_sample_mode=True,
        effective_thresholds=thresholds_payload,
    )
    if workspace is not None:
        write_local_llm_receipt(receipt, workspace)

    cloud_escalation: dict[str, Any] | None = None
    if should_escalate:
        if workspace is None:
            raise CloudEscalationError(
                "Cloud escalation requires a workspace so the EscalationReceipt "
                "can be written."
            )
        cloud_escalation = _run_cloud_escalation(
            workspace=Path(workspace),
            run_id=run_id,
            stage_id=stage_id or "quality-sample",
            affected_record_ids=affected_record_ids,
            fail_closed_state=not should_proceed,
            confidence_mean=conf_mean,
            confidence_p50=conf_p50,
            confidence_p90=conf_p90,
            failure_classes=failure_classes,
            failure_class_rate=failure_rate,
            record_coverage=coverage,
            thresholds_payload=thresholds_payload,
        )
        recommendation["cloud_escalation_status"] = cloud_escalation["status"]
        recommendation["cloud_escalation_receipt_path"] = cloud_escalation[
            "escalation_receipt_path"
        ]

    result = {
        "quality_sample_mode": True,
        "sample_size": len(sample_records),
        "confidence_mean": conf_mean,
        "confidence_p50": conf_p50,
        "confidence_p90": conf_p90,
        "failure_classes": failure_classes,
        "failure_class_rate": failure_rate,
        "record_coverage": coverage,
        "model": backend.model,
        "endpoint_label": backend.endpoint_label,
        "request_count": request_count,
        "token_count": token_count,
        "effective_thresholds": thresholds_payload,
        "recommendation": recommendation,
        "cloud_escalation_threshold_recommendation": (
            "increase"
            if thresholds_payload["cloud_escalation_trigger"]
            > default_threshold_payload["cloud_escalation_trigger"]
            else "review"
            if should_escalate
            else "current_threshold_adequate"
        ),
    }
    if cloud_escalation is not None:
        result["cloud_escalation"] = cloud_escalation
        result["escalation_receipt_path"] = cloud_escalation["escalation_receipt_path"]
    if not should_proceed:
        raise QualitySampleThresholdError(
            str(recommendation.get("reason") or "quality sample failed thresholds"),
            recommendation=recommendation,
            result=result,
        )
    return result


def classify_with_llm_assist(
    text: str,
    taxonomy: list[TopicFamily],
    *,
    max_chars: int = _DEFAULT_LLM_MAX_CHARS,
    model: str | None = None,
) -> tuple[list[TopicMatch], dict[str, Any]]:
    """Optional local LLM-assist escalation for ambiguous artifacts.

    Returns ``([], {"skipped": "llm_assist_disabled"})`` immediately unless
    ``EXTRACTION_LLM_ASSIST`` or ``EXTRACTION_LOCAL_LLM_ASSIST`` is truthy.
    A loopback OpenAI-compatible base URL is required; cloud OpenAI URLs are
    rejected here so exhaustive extraction does not silently become paid.

    All errors are caught and returned as
    ``([], {"error": "llm_assist_failed", "redacted_message": ...})``; this
    function never re-raises.
    """
    global _LLM_ASSIST_CALL_COUNT  # noqa: PLW0603 - intentional shared state

    if not _llm_assist_enabled():
        return [], {"skipped": "llm_assist_disabled"}

    from conference_pipeline.extractions.backends import (
        BackendKind,
        select_local_llm_backend,
    )

    resolved_model = _local_llm_model(model or _DEFAULT_LOCAL_LLM_MODEL)
    try:
        backend = select_local_llm_backend(resolved_model)
    except Exception as exc:  # noqa: BLE001 - backend resolution must not crash extraction
        return [], {
            "skipped": "backend_selection_failed",
            **_local_provider_meta(
                model=resolved_model,
                escalation_reason="heuristic_no_confident_match",
                status="failed",
                error=_redact(str(exc)),
            ),
        }

    base_url = getattr(backend, "base_url", None)
    if backend.kind is BackendKind.HTTP and not base_url:
        return [], {
            "skipped": "local_endpoint_missing",
            **_local_provider_meta(
                model=resolved_model,
                endpoint_label=backend.endpoint_label,
                escalation_reason="heuristic_no_confident_match",
                status="skipped",
            ),
        }
    if backend.kind is BackendKind.HTTP and not _is_loopback_base_url(str(base_url)):
        return [], {
            "skipped": "non_local_endpoint_rejected",
            **_local_provider_meta(
                model=resolved_model,
                endpoint_label=backend.endpoint_label,
                base_url=str(base_url),
                escalation_reason="heuristic_no_confident_match",
                status="skipped",
            ),
        }

    with _LLM_ASSIST_LOCK:
        if _LLM_ASSIST_CALL_COUNT >= _llm_assist_call_limit():
            return [], {
                "skipped": "max_calls_exceeded",
                **_local_provider_meta(
                    provider="mlx-local" if backend.kind is BackendKind.MLX else "local-openai-compatible",
                    model=resolved_model,
                    endpoint_label=backend.endpoint_label,
                    base_url=str(base_url) if base_url else None,
                    escalation_reason="heuristic_no_confident_match",
                    status="skipped",
                ),
            }
        _LLM_ASSIST_CALL_COUNT += 1

    bounded = (text or "")[:max_chars]
    provider_receipt: Any = None
    try:
        topic_ids_csv = ", ".join(family.id for family in taxonomy)
        system = (
            "You classify text into the conference's topic families. "
            "Return a JSON object with key 'topic_ids' (list of strings from: "
            f"{topic_ids_csv}). No prose."
        )
        user = bounded
        response = backend.chat_completion(
            caller=__file__,
            model=resolved_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=_llm_max_output_tokens(),
        )
        provider_receipt = getattr(response, "receipt", None)
        choice = response.choices[0].message.content if response.choices else "{}"
        parsed = _parse_llm_json_object(choice)
        raw_ids = parsed.get("topic_ids")
        valid_ids = {family.id for family in taxonomy}
        topic_ids = [
            str(item)
            for item in (raw_ids or [])
            if isinstance(item, str) and item in valid_ids
        ]
        matches = [
            TopicMatch(topic_id=topic_id, score=1.0, confidence=0.9)
            for topic_id in topic_ids
        ]
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(provider_receipt, "input_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(provider_receipt, "output_tokens", None)
        request_id = getattr(response, "id", None) or getattr(
            provider_receipt,
            "response_id",
            None,
        )
        receipt_path = getattr(provider_receipt, "receipt_path", None)
        provider_meta: dict[str, Any] = {
            **_local_provider_meta(
                provider="mlx-local" if backend.kind is BackendKind.MLX else "local-openai-compatible",
                model=resolved_model,
                endpoint_label=backend.endpoint_label,
                base_url=str(base_url) if base_url else None,
                escalation_reason="heuristic_no_confident_match",
                status="succeeded",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_id=request_id,
                receipt_path=receipt_path,
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": 0.0,
            "provider_receipt_path": receipt_path,
        }
        return matches, provider_meta
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        failed_receipt = getattr(exc, "provider_receipt", None) or provider_receipt
        return [], {
            "error": "llm_assist_failed",
            **_local_provider_meta(
                provider="mlx-local" if backend.kind is BackendKind.MLX else "local-openai-compatible",
                model=resolved_model,
                endpoint_label=backend.endpoint_label,
                base_url=str(base_url) if base_url else None,
                escalation_reason="heuristic_no_confident_match",
                status="failed",
                input_tokens=getattr(failed_receipt, "input_tokens", None),
                output_tokens=getattr(failed_receipt, "output_tokens", None),
                request_id=getattr(failed_receipt, "response_id", None),
                receipt_path=getattr(failed_receipt, "receipt_path", None),
                error=_redact(str(exc)),
            ),
        }


def classify_artifact(
    input_: "ExtractionInput",
    taxonomy: list[TopicFamily],
    *,
    run_id: str,
    min_chars: int = _DEFAULT_MIN_CHARS,
    tau: float | None = None,
) -> ExtractionRecord:
    """Drive heuristic-first classification for a single ``ExtractionInput``.

    Decision order:

    1. Upstream not in completed status → ``skipped_reason="upstream_blocked"``.
    2. Empty/short text → ``skipped_reason="empty_or_too_short"``.
    3. Heuristic match at confidence ≥ tau → ``heuristic`` record.
    4. ``EXTRACTION_LLM_ASSIST`` set → LLM escalation; if matches → ``llm_assisted``.
    5. Otherwise → ``unclassifiable`` (empty topic_ids, skipped_reason set).
    """

    common_quality: dict[str, Any] = {
        "input_kind": input_.kind,
        "text_length": len(input_.text_content or ""),
        "language": "unknown",
    }
    if input_.text_path:
        common_quality["source_text_path"] = input_.text_path
    for key, value in input_.metadata.items():
        if key not in common_quality:
            common_quality[key] = value
    base_evidence: dict[str, Any] = {
        "snippet": "",
        "span": {"start": 0, "end": 0},
    }

    if input_.status not in COMPLETED_MEDIA_PROCESSING_STATUSES:
        return ExtractionRecord(
            artifact_id=input_.artifact_id,
            topic_ids=(),
            source_artifact_id=input_.artifact_id,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest or "",
            evidence=base_evidence,
            extraction_method="heuristic",
            confidence=0.0,
            quality_metadata={**common_quality, "upstream_status": input_.status},
            skipped_reason="upstream_blocked",
            provider_metadata=None,
            run_id=run_id,
            generated_at=utc_now(),
        )

    text = input_.text_content or ""
    if input_.kind == "paper_metadata_fallback" and text:
        topic_ids = input_.metadata.get("topic_ids")
        if not isinstance(topic_ids, list) or not topic_ids:
            topic_ids = [family.id for family in taxonomy]
        topic_ids = [str(topic_id) for topic_id in topic_ids if str(topic_id).strip()]
        snippet_end = min(len(text), 240)
        evidence = {
            "snippet": text[:snippet_end],
            "span": {"start": 0, "end": snippet_end},
        }
        return ExtractionRecord(
            artifact_id=input_.artifact_id,
            topic_ids=tuple(topic_ids),
            source_artifact_id=input_.artifact_id,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest or "",
            evidence=evidence,
            extraction_method="heuristic",
            confidence=0.5,
            quality_metadata={
                **common_quality,
                "match_count": len(topic_ids),
                "confidence_path": "metadata_fallback_all_topics_for_semantic_retrieval",
            },
            skipped_reason=None,
            provider_metadata=None,
            run_id=run_id,
            generated_at=utc_now(),
        )

    if not text or len(text) < min_chars:
        return ExtractionRecord(
            artifact_id=input_.artifact_id,
            topic_ids=(),
            source_artifact_id=input_.artifact_id,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest or "",
            evidence=base_evidence,
            extraction_method="heuristic",
            confidence=0.0,
            quality_metadata=common_quality,
            skipped_reason="empty_or_too_short",
            provider_metadata=None,
            run_id=run_id,
            generated_at=utc_now(),
        )

    matches = classify_text_span(text, taxonomy, min_chars=min_chars, tau=tau)
    if matches:
        snippet_end = min(len(text), 240)
        evidence = {
            "snippet": text[:snippet_end],
            "span": {"start": 0, "end": snippet_end},
        }
        top_confidence = max(m.confidence for m in matches)
        return ExtractionRecord(
            artifact_id=input_.artifact_id,
            topic_ids=tuple(m.topic_id for m in matches),
            source_artifact_id=input_.artifact_id,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest or "",
            evidence=evidence,
            extraction_method="heuristic",
            confidence=top_confidence,
            quality_metadata={
                **common_quality,
                "match_count": len(matches),
                "topic_scores": {m.topic_id: m.score for m in matches},
            },
            skipped_reason=None,
            provider_metadata=None,
            run_id=run_id,
            generated_at=utc_now(),
        )

    # Heuristic produced nothing — try LLM-assist if enabled.
    llm_matches, provider_meta = classify_with_llm_assist(text, taxonomy)
    if llm_matches:
        snippet_end = min(len(text), 240)
        evidence = {
            "snippet": text[:snippet_end],
            "span": {"start": 0, "end": snippet_end},
        }
        top_confidence = max(m.confidence for m in llm_matches)
        return ExtractionRecord(
            artifact_id=input_.artifact_id,
            topic_ids=tuple(m.topic_id for m in llm_matches),
            source_artifact_id=input_.artifact_id,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest or "",
            evidence=evidence,
            extraction_method="llm_assisted",
            confidence=top_confidence,
            quality_metadata={
                **common_quality,
                "match_count": len(llm_matches),
            },
            skipped_reason=None,
            provider_metadata=provider_meta,
            run_id=run_id,
            generated_at=utc_now(),
        )

    return ExtractionRecord(
        artifact_id=input_.artifact_id,
        topic_ids=(),
        source_artifact_id=input_.artifact_id,
        source_artifact_path=input_.source_path,
        source_artifact_digest=input_.digest or "",
        evidence=base_evidence,
        extraction_method="heuristic",
        confidence=0.0,
        quality_metadata=common_quality,
        skipped_reason="unclassifiable",
        provider_metadata=provider_meta if provider_meta else None,
        run_id=run_id,
        generated_at=utc_now(),
    )


def get_extraction_status(workspace_root: Path) -> dict[str, Any]:
    """Return extraction posture from on-disk receipts. No live endpoint calls.

    Reads LocalLLMReceipt and EscalationReceipt files from workspace storage.
    Safe to run without a running LLM — no live endpoint calls are made.
    Endpoint URLs are never included in the output (endpoint_label only).

    Args:
        workspace_root: Path to the conference workspace root directory.

    Returns:
        dict with keys:
          extraction_mode: "local_llm" | "heuristic" | "unknown"
          local_llm_readiness: dict from most recent preflight receipt, or None
          cloud_escalation_count: int
          ocr_posture: "required" | "skipped_with_rationale" | "deferred_blocked" | "unknown"
          receipt_links: list[str]  — receipt_path values only (no URLs)
          blockers: list[str]
    """
    from conference_pipeline.provider_receipts import (  # noqa: PLC0415
        scan_escalation_receipts,
        scan_local_llm_receipts,
    )

    workspace = Path(workspace_root)
    llm_receipts = scan_local_llm_receipts(workspace)
    esc_receipts = scan_escalation_receipts(workspace)

    # Most recent preflight receipt (preflight_mode True, sorted by timestamp descending)
    preflight = next(
        (
            r
            for r in sorted(
                llm_receipts,
                key=lambda r: r.get("timestamp", ""),
                reverse=True,
            )
            if r.get("preflight_mode")
        ),
        None,
    )

    extraction_mode = "local_llm" if llm_receipts else "heuristic"

    blockers: list[str] = []
    if not llm_receipts:
        blockers.append(
            "No local LLM receipts found — check EXTRACTION_LOCAL_LLM_BASE_URL"
        )

    return {
        "extraction_mode": extraction_mode,
        "local_llm_readiness": preflight,
        "cloud_escalation_count": len(esc_receipts),
        "ocr_posture": "unknown",
        "receipt_links": [
            r.get("receipt_path")
            for r in llm_receipts
            if r.get("receipt_path")
        ],
        "blockers": blockers,
    }


__all__ = [
    "classify_artifact",
    "classify_text_span",
    "classify_with_llm_assist",
    "reset_llm_assist_counter",
    "LocalLLMNotReadyError",
    "CloudEscalationError",
    "QualitySampleThresholdError",
    "check_local_llm_readiness",
    "run_quality_sample",
    "get_extraction_status",
]
