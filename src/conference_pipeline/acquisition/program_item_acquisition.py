"""Per-program-item public acquisition: the Stage-4 ledger and its public lane.

The module owns the conference-neutral mechanics of a program-item
acquisition ledger: deterministic ledger construction, byte reservation,
resume, artifact containment/validation, atomic streamed writes and the
bounded public transport (:func:`execute_public_acquisition`), plus the
public lane hook (:func:`public_acquisition_lane`). Source lineage, capacity
and the stage receipt are owned by the shared Stage 4 frame
(:mod:`conference_pipeline.production.download_corpus_stage`).

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)`` (formerly ``acquisition/cvpr_corpus_download.py``;
that import path remains a compatibility alias).

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r3, the public
  lane: every frozen CVPR program item (papers on CVF Open Access, project
  pages, schedule pages) gets a public request result or a blocker, with the
  workshop requests routed once per parent venue to the registered workshop
  owner (:mod:`conference_pipeline.acquisition.cvf_proceedings`).
* **May also fit:** any conference whose Stage 3 freezes per-program-item
  rows with planned public acquisition URLs (ICCV, WACV, ECCV and other
  CVF-hosted proceedings in particular).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from conference_pipeline.provider_receipts import sanitize_for_receipt

RECOMMENDED_FOR = ("CVPR",)

ARTIFACT_KINDS = frozenset(
    {
        "source_page",
        "session_page",
        "paper_pdf",
        "source_archive",
        "media_asset",
        "youtube_caption",
        "slide_image",
        "policy_excluded_official_video",
    }
)
_SECRET_QUERY_MARKERS = ("api_key", "apikey", "authorization", "cookie", "token", "secret", "bearer")
_LEDGER_VOLATILE_FIELDS = frozenset({"generated_at", "updated_at", "manifest_sha256"})
_MINIMUM_REQUEST_RESERVATION_BYTES = 1_024
_MINIMUM_ITEM_RESERVATION_BYTES = 256
_PUBLIC_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_PUBLIC_BACKOFF_SECONDS = (1.0, 2.0)
_PUBLIC_ARTIFACT_ROOT = Path("storage/download-corpus/artifacts")
_CONTROL_RESULT_CHECKPOINT_BATCH_SIZE = 32
# The urllib connection timeout alone does not bound a peer that makes
# intermittent progress while never returning a response body.  Keep one
# wall-clock budget around the entire public opener/read sequence so every
# source-backed request can become a durable retry/blocker outcome.
_PUBLIC_RESPONSE_READ_DEADLINE_SECONDS = 45.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_payload(value: Any, *, exclude_volatile: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_payload(item, exclude_volatile=exclude_volatile)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (exclude_volatile and str(key) in _LEDGER_VOLATILE_FIELDS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_payload(item, exclude_volatile=exclude_volatile) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_ledger_digest(payload: Mapping[str, Any]) -> str:
    """Hash durable ledger state while excluding generated timestamps/digest fields."""

    canonical = _canonical_payload(payload, exclude_volatile=True)
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _digest_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical_payload(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _safe_url(value: Any) -> str:
    """Retain source provenance while stripping secret-like query values.

    Only the query string is touched (any param whose key matches
    ``_SECRET_QUERY_MARKERS``); the path is never passed through
    ``sanitize_for_receipt``.  That function's blanket 80+-char opaque-
    payload redaction is meant for things like base64 blobs or session
    tokens, but a real CVF paper filename is a long, human-authored title
    (e.g. ``Xiao_Generalizable_..._CVPR_2026_paper.pdf``, 100+ characters)
    that would be redacted to ``REDACTED.pdf`` -- an unreachable url --
    despite containing no secret at all.
    """

    if not isinstance(value, str):
        return ""
    parsed = urlsplit(value)
    filtered = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not any(marker in key.lower() for marker in _SECRET_QUERY_MARKERS)
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(filtered), ""))


def _request_key(*, source_url: str, artifact_kind: str) -> str:
    return sha256(f"{artifact_kind}\n{source_url}".encode("utf-8")).hexdigest()


def _workspace_path(workspace: Path, candidate: Path) -> Path:
    root = workspace.resolve()
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path must remain inside the workspace") from exc
    return resolved


def workspace_relative_path(workspace: Path, path: Path) -> str:
    """Return a verified workspace-relative artifact path."""

    return str(_workspace_path(workspace, path).relative_to(workspace.resolve()))


def is_valid_pdf(sample: bytes, content_type: str | None) -> bool:
    """Require real PDF magic bytes and reject explicitly non-PDF content."""

    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    return sample.startswith(b"%PDF-") and normalized not in {"text/html", "application/json", "text/plain"}


def _selection_plan(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    plans = row.get("planned_acquisition")
    if not isinstance(plans, list) or not plans:
        return None, None
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for raw_plan in plans:
        if not isinstance(raw_plan, Mapping):
            continue
        urls = raw_plan.get("source_urls")
        if not isinstance(urls, list):
            continue
        for raw_url in urls:
            url = _safe_url(raw_url)
            if url.startswith(("https://", "http://")):
                candidates.append((url, str(raw_plan.get("source_record_id") or ""), dict(raw_plan)))
    if not candidates:
        return None, None
    url, _source_record_id, plan = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return plan, url


def build_initial_ledger(
    *,
    lineage: Mapping[str, Any],
    selection_rows: Sequence[Mapping[str, Any]],
    policy_exclusions: Sequence[Mapping[str, Any]],
    expected_program_item_count: int,
    report_page_size: int = 50,
) -> dict[str, Any]:
    """Seed one durable result row per selected program item.

    A request may serve more than one item, but the result table never uses
    request deduplication as a scope filter.
    """

    if expected_program_item_count <= 0 or report_page_size <= 0:
        raise ValueError("program-item count and report page size must be positive")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in selection_rows:
        item_id = row.get("program_item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("selection row lacks a program_item_id")
        if item_id in by_id:
            raise ValueError("selection contains duplicate program_item_id")
        if row.get("selection_status") != "included":
            raise ValueError("initial download ledger accepts included rows only")
        by_id[item_id] = row
    if len(by_id) != expected_program_item_count:
        raise ValueError("selection scope is incomplete")

    policy_by_family = {
        str(policy["family_id"]): dict(policy)
        for policy in policy_exclusions
        if isinstance(policy.get("family_id"), str)
    }
    request_results: dict[str, dict[str, Any]] = {}
    program_item_results: list[dict[str, Any]] = []
    for item_id in sorted(by_id):
        row = by_id[item_id]
        plan, source_url = _selection_plan(row)
        source_ids = row.get("source_record_ids")
        family_ids = row.get("source_family_ids")
        source_record_id = str(plan.get("source_record_id") if plan else (source_ids[0] if isinstance(source_ids, list) and source_ids else ""))
        source_family_id = str(plan.get("source_family_id") if plan else (family_ids[0] if isinstance(family_ids, list) and family_ids else "unknown"))
        policy_boundary = policy_by_family.get(source_family_id)
        blocker: dict[str, str] | None = None
        request_key: str | None = None
        if source_url:
            request_key = _request_key(source_url=source_url, artifact_kind="source_page")
            request = request_results.setdefault(
                request_key,
                {
                    "request_key": request_key,
                    "source_url": source_url,
                    "final_url": None,
                    "artifact_kind": "source_page",
                    "status": "pending",
                    "http_status": None,
                    "content_type": None,
                    "local_path": None,
                    "byte_count": None,
                    "sha256": None,
                    "attempts": [],
                    "linked_program_item_ids": [],
                    "source_evidence": {"source_family_id": source_family_id, "source_record_id": source_record_id},
                },
            )
            request["linked_program_item_ids"].append(item_id)
        else:
            unreachable = row.get("unreachable_source_path")
            reason = "no_admissible_source_url"
            if isinstance(unreachable, Mapping) and isinstance(unreachable.get("reason"), str):
                reason = str(unreachable["reason"])
            blocker = {"code": "no_admissible_artifact", "reason": reason}
        program_item_results.append(
            {
                "program_item_id": item_id,
                "source_record_id": source_record_id,
                "source_family_id": source_family_id,
                "selection_status": "included",
                # Not sanitize_for_receipt(plan): plan's fields (source_record_id,
                # source_family_id, source_urls, the fixed phase/intent strings)
                # come entirely from public source-discovery evidence and never
                # carry a secret. Passing it through sanitize_for_receipt would
                # redact a long real paper filename to a misleading
                # ".../REDACTED.pdf" here, while request_results (the field
                # actually used for the fetch) shows the real url -- an
                # internally inconsistent, misleading receipt.
                "acquisition_plan": dict(plan) if plan else None,
                "request_key": request_key,
                "attempts": [],
                "blocker": blocker,
                "policy_boundary": sanitize_for_receipt(policy_boundary) if policy_boundary else None,
            }
        )

    for request in request_results.values():
        request["linked_program_item_ids"].sort()
    pending = sum(item["blocker"] is None for item in program_item_results)
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_download_corpus",
        "stage_id": "download-corpus",
        "status": "pre_request",
        "generated_at": _utc_now(),
        "lineage": sanitize_for_receipt(dict(lineage)),
        "program_item_results": program_item_results,
        "request_results": [request_results[key] for key in sorted(request_results)],
        "artifacts": [],
        "policy_exclusions": sanitize_for_receipt([dict(item) for item in policy_exclusions]),
        "blockers": [],
        "counts": {
            "program_item_result_count": len(program_item_results),
            "request_result_count": len(request_results),
            "pending_program_item_count": pending,
            "blocked_program_item_count": len(program_item_results) - pending,
            "artifact_count": 0,
        },
        "report_pagination": {
            "page_size": report_page_size,
            "visible_page_count": 1,
            "total_program_items": len(program_item_results),
        },
        "secrets_copied": False,
    }
    ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
    return ledger


def derive_required_byte_reservation(
    *,
    ledger: Mapping[str, Any],
    storage_manifest: Mapping[str, Any],
    safety_margin_bytes: int,
) -> dict[str, Any]:
    """Build a positive, deterministic capacity reservation before transport."""

    if not isinstance(safety_margin_bytes, int) or safety_margin_bytes < 0:
        raise ValueError("safety margin must be a nonnegative integer")
    counts = ledger.get("counts")
    request_count = int(counts.get("request_result_count", 0)) if isinstance(counts, Mapping) else 0
    item_count = int(counts.get("program_item_result_count", 0)) if isinstance(counts, Mapping) else 0
    if request_count <= 0 or item_count <= 0:
        raise ValueError("ledger must contain request and program-item results")
    required = (
        request_count * _MINIMUM_REQUEST_RESERVATION_BYTES
        + item_count * _MINIMUM_ITEM_RESERVATION_BYTES
        + safety_margin_bytes
    )
    return {
        "reservation_version": 1,
        "required_bytes": required,
        "request_reservation_bytes": request_count * _MINIMUM_REQUEST_RESERVATION_BYTES,
        "program_item_reservation_bytes": item_count * _MINIMUM_ITEM_RESERVATION_BYTES,
        "safety_margin_bytes": safety_margin_bytes,
        "scope_ledger_sha256": canonical_ledger_digest(ledger),
        "manifest_sha256": _digest_payload(storage_manifest),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_artifact_is_valid(
    *,
    workspace: Path,
    artifact: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
) -> bool:
    """Accept resume only for contained, nonempty, digest- and lineage-bound bytes."""

    local_path = artifact.get("local_path")
    artifact_kind = artifact.get("artifact_kind")
    if not isinstance(local_path, str) or not isinstance(artifact_kind, str) or artifact_kind not in ARTIFACT_KINDS:
        return False
    try:
        path = _workspace_path(workspace, workspace / local_path)
    except ValueError:
        return False
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    if artifact.get("byte_count") != path.stat().st_size or artifact.get("sha256") != _file_sha256(path):
        return False
    lineage = artifact.get("lineage")
    if not isinstance(lineage, Mapping):
        return False
    return all(lineage.get(key) == value for key, value in expected_lineage.items())


def resume_compatible_ledger(
    *,
    workspace: Path,
    fresh_ledger: Mapping[str, Any],
    ledger_path: Path = Path("storage/download-corpus/download_corpus.json"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only lineage- and digest-verified Stage-4 progress on resume.

    The frozen program-item/request set is rebuilt for every invocation.  A
    prior ledger may contribute only matching request outcomes and artifact
    bytes whose digest and full lineage still verify.  Any incompatible prior
    control state fails closed before a new transport is constructed.
    """

    if not isinstance(fresh_ledger, Mapping):
        raise ValueError("fresh download ledger must be an object")
    result = json.loads(json.dumps(_canonical_payload(fresh_ledger)))
    expected_lineage = result.get("lineage")
    expected_requests = result.get("request_results")
    expected_items = result.get("program_item_results")
    if (
        not isinstance(expected_lineage, Mapping)
        or not isinstance(expected_requests, list)
        or not isinstance(expected_items, list)
    ):
        raise ValueError("fresh download ledger is incomplete")
    target = _workspace_path(workspace, workspace / ledger_path)
    if not target.exists():
        return result, {
            "status": "not_found",
            "reused_artifact_count": 0,
            "retained_terminal_request_count": 0,
            "discarded_artifact_count": 0,
        }
    try:
        prior = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prior download ledger is unreadable") from exc
    if not isinstance(prior, Mapping):
        raise ValueError("prior download ledger must be an object")
    if (
        prior.get("receipt_type") != result.get("receipt_type")
        or prior.get("stage_id") != result.get("stage_id")
        or prior.get("lineage") != expected_lineage
        or prior.get("secrets_copied") is not False
        or prior.get("manifest_sha256") != canonical_ledger_digest(prior)
    ):
        raise ValueError("prior download ledger lineage or digest is incompatible")

    def indexed(rows: Any, *, key: str) -> dict[str, Mapping[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError("prior download ledger rows are incomplete")
        result_rows: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("prior download ledger row key is invalid")
            value = str(row[key])
            if value in result_rows:
                raise ValueError("prior download ledger row key is duplicated")
            result_rows[value] = row
        return result_rows

    expected_request_by_key = indexed(expected_requests, key="request_key")
    prior_request_by_key = indexed(prior.get("request_results"), key="request_key")
    expected_item_by_id = indexed(expected_items, key="program_item_id")
    prior_item_by_id = indexed(prior.get("program_item_results"), key="program_item_id")
    if set(expected_request_by_key) != set(prior_request_by_key) or set(expected_item_by_id) != set(prior_item_by_id):
        raise ValueError("prior download ledger scope differs from the frozen corpus")
    for key, expected in expected_request_by_key.items():
        prior_request = prior_request_by_key[key]
        if (
            prior_request.get("source_url") != expected.get("source_url")
            or sorted(prior_request.get("linked_program_item_ids") or [])
            != sorted(expected.get("linked_program_item_ids") or [])
        ):
            raise ValueError("prior download request provenance is incompatible")
    for item_id, expected in expected_item_by_id.items():
        prior_item = prior_item_by_id[item_id]
        if any(
            prior_item.get(field) != expected.get(field)
            for field in ("source_record_id", "source_family_id", "selection_status", "request_key")
        ):
            raise ValueError("prior download program-item provenance is incompatible")

    prior_artifacts = prior.get("artifacts")
    if not isinstance(prior_artifacts, list):
        raise ValueError("prior download ledger artifacts are incomplete")
    valid_artifacts: dict[str, Mapping[str, Any]] = {}
    discarded_artifact_count = 0
    for artifact in prior_artifacts:
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("request_key"), str):
            raise ValueError("prior download artifact is invalid")
        request_key = str(artifact["request_key"])
        if request_key not in expected_request_by_key or request_key in valid_artifacts:
            raise ValueError("prior download artifact scope is incompatible")
        if resume_artifact_is_valid(
            workspace=workspace,
            artifact=artifact,
            expected_lineage=expected_lineage,
        ):
            valid_artifacts[request_key] = artifact
        else:
            discarded_artifact_count += 1

    terminal_statuses = {
        "policy_excluded_official_video",
        "workshop_owner_result",
        "invalid_content",
        "no_admissible_artifact",
    }
    retained_request_fields = {
        "status",
        "final_url",
        "http_status",
        "content_type",
        "local_path",
        "byte_count",
        "sha256",
        "attempts",
        "retry_events",
        "auth_recovery",
        "policy_boundary",
        "workshop_owner",
        "error_type",
        "response_deadline_seconds",
    }
    result_artifacts: list[dict[str, Any]] = []
    retained_terminal_request_count = 0
    for request in expected_requests:
        request_key = str(request["request_key"])
        prior_request = prior_request_by_key[request_key]
        status = prior_request.get("status")
        if status == "succeeded" and request_key in valid_artifacts:
            request.update(
                sanitize_for_receipt(
                    {
                        field: prior_request[field]
                        for field in retained_request_fields
                        if field in prior_request
                    }
                )
            )
            result_artifacts.append(sanitize_for_receipt(dict(valid_artifacts[request_key])))
        elif status in terminal_statuses:
            request.update(
                sanitize_for_receipt(
                    {
                        field: prior_request[field]
                        for field in retained_request_fields
                        if field in prior_request
                    }
                )
            )
            retained_terminal_request_count += 1
    result["artifacts"] = result_artifacts
    _refresh_public_acquisition_accounting(result)
    result["resume"] = {
        "status": "reused",
        "prior_ledger_sha256": prior["manifest_sha256"],
        "reused_artifact_count": len(result_artifacts),
        "retained_terminal_request_count": retained_terminal_request_count,
        "discarded_artifact_count": discarded_artifact_count,
    }
    result["manifest_sha256"] = canonical_ledger_digest(result)
    return result, dict(result["resume"])


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    value = getattr(response, key, default)
    return value() if callable(value) and key in {"geturl", "getcode"} else value


@contextmanager
def _public_transport_deadline(seconds: float) -> Any:
    """Apply one wall-clock deadline to a public opener plus its response read.

    ``urlopen(timeout=...)`` limits idle socket I/O but can still be kept alive
    indefinitely by a hostile or slow peer.  Stage-4 runs in the main thread
    on supported POSIX hosts, where ``SIGALRM`` safely interrupts the blocking
    open/read.  Avoid changing another caller's signal timer; unsupported
    contexts retain the transport's ordinary socket-timeout behavior.
    """

    if seconds <= 0:
        raise ValueError("public transport deadline must be positive")
    if (
        not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        yield
        return

    def _raise_deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError("public transport response deadline exceeded")

    signal.signal(signal.SIGALRM, _raise_deadline)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def stream_to_workspace(
    *,
    workspace: Path,
    destination: Path,
    opener: Callable[[str], Any],
    url: str,
    artifact_kind: str,
    chunk_size: int = 64 * 1024,
    response_deadline_seconds: float = _PUBLIC_RESPONSE_READ_DEADLINE_SECONDS,
) -> dict[str, Any]:
    """Stream a public response into a temporary local file then atomically promote it.

    The opener is injected to keep this primitive independently testable.  It
    performs no retries itself; callers own policy-specific retry/backoff.
    """

    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError("unsupported artifact kind")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    safe_destination = _workspace_path(workspace, destination)
    safe_url = _safe_url(url)
    response: Any = None
    temporary_path: Path | None = None
    try:
        with _public_transport_deadline(response_deadline_seconds):
            response = opener(url)
            status_code = _response_value(response, "status", None)
            if status_code is None:
                status_code = _response_value(response, "getcode", None)
            if not isinstance(status_code, int):
                status_code = 200
            headers = getattr(response, "headers", {})
            content_type = ""
            if isinstance(headers, Mapping):
                content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
            elif hasattr(headers, "get_content_type"):
                content_type = str(headers.get_content_type())
            final_url = _safe_url(_response_value(response, "geturl", getattr(response, "url", safe_url)))
            if status_code < 200 or status_code >= 300:
                return {
                    "status": "unreachable_http_error",
                    "http_status": status_code,
                    "source_url": safe_url,
                    "final_url": final_url,
                    "artifact_kind": artifact_kind,
                    "retry_events": [],
                }
            safe_destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{safe_destination.name}.", suffix=".tmp", dir=safe_destination.parent)
            temporary_path = Path(temporary_name)
            digest = sha256()
            byte_count = 0
            sample = bytearray()
            with os.fdopen(descriptor, "wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise ValueError("response stream must yield bytes")
                    if len(sample) < 16:
                        sample.extend(chunk[: 16 - len(sample)])
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            if byte_count <= 0 or (artifact_kind == "paper_pdf" and not is_valid_pdf(bytes(sample), content_type)):
                temporary_path.unlink(missing_ok=True)
                return {
                    "status": "invalid_content",
                    "source_url": safe_url,
                    "final_url": final_url,
                    "artifact_kind": artifact_kind,
                    "http_status": status_code,
                    "content_type": content_type,
                    "byte_count": byte_count,
                    "retry_events": [],
                }
            os.replace(temporary_path, safe_destination)
            temporary_path = None
            return {
                "status": "succeeded",
                "source_url": safe_url,
                "final_url": final_url,
                "artifact_kind": artifact_kind,
                "http_status": status_code,
                "content_type": content_type,
                "local_path": workspace_relative_path(workspace, safe_destination),
                "byte_count": byte_count,
                "sha256": digest.hexdigest(),
                "retry_events": [],
            }
    except ValueError:
        raise
    except TimeoutError:
        return {
            "status": "unreachable_fetch_error",
            "source_url": safe_url,
            "artifact_kind": artifact_kind,
            "error_type": "response_deadline_exceeded",
            "response_deadline_seconds": response_deadline_seconds,
            "retry_events": [],
        }
    except Exception as exc:  # noqa: BLE001 - receipt data must not expose provider errors
        return {
            "status": "unreachable_fetch_error",
            "source_url": safe_url,
            "artifact_kind": artifact_kind,
            "error_type": type(exc).__name__,
            "retry_events": [],
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        close = getattr(response, "close", None)
        if callable(close):
            close()


def classify_public_artifact_kind(source_url: str) -> str:
    """Classify a source URL conservatively before an artifact is promoted.

    The classification is intentionally limited to directly observable URL
    evidence.  A response is still validated before it can become an
    artifact, so a ``.pdf`` URL returning login HTML remains an
    ``invalid_content`` blocker rather than a paper artifact.
    """

    path = urlsplit(source_url).path.lower()
    if path.endswith(".pdf"):
        return "paper_pdf"
    if path.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        return "source_archive"
    if path.endswith((".mp4", ".webm", ".mov", ".m4a", ".mp3", ".wav")):
        return "media_asset"
    if "/session" in path or "/schedule" in path:
        return "session_page"
    return "source_page"


def _artifact_destination(workspace: Path, request: Mapping[str, Any]) -> Path:
    """Build a deterministic, contained destination from an immutable key."""

    request_key = request.get("request_key")
    artifact_kind = request.get("artifact_kind")
    if not isinstance(request_key, str) or len(request_key) != 64:
        raise ValueError("request lacks an immutable request key")
    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError("request artifact kind is unsupported")
    extension = {
        "paper_pdf": ".pdf",
        "source_archive": ".archive",
        "media_asset": ".media",
        "source_page": ".html",
        "session_page": ".html",
    }.get(str(artifact_kind), ".bin")
    return _workspace_path(workspace, workspace / _PUBLIC_ARTIFACT_ROOT / str(artifact_kind) / f"{request_key}{extension}")


def _request_attempt(result: Mapping[str, Any], *, attempt: int, delay: float | None = None) -> dict[str, Any]:
    """Return sanitized, durable retry evidence without body or header data."""

    http_status = result.get("http_status")
    event: dict[str, Any] = {
        "attempt": attempt,
        "http_status": http_status if isinstance(http_status, int) else None,
        "reason": "transient_http" if isinstance(http_status, int) else "transient_fetch",
    }
    if delay is not None:
        event["backoff_seconds"] = delay
    return event


def _is_transient_public_result(result: Mapping[str, Any]) -> bool:
    """Identify only bounded retry outcomes for the public request lane."""

    status = result.get("status")
    http_status = result.get("http_status")
    return status == "unreachable_fetch_error" or (
        status == "unreachable_http_error" and isinstance(http_status, int) and http_status in _PUBLIC_TRANSIENT_HTTP_STATUSES
    )


def _safe_auth_recovery(result: Any) -> tuple[Callable[[str], Any] | None, dict[str, Any]]:
    """Accept a deliberately narrow recovery seam without serializing secrets.

    The configured auth mechanism remains responsible for credentials.  The
    ledger records only that one recovery was attempted, its named mode, and
    the environment-variable *names* that supplied credentials.
    """

    if not isinstance(result, Mapping):
        return None, {"mode": "unavailable", "status": "unavailable", "environment_variable_names": []}
    opener = result.get("opener")
    names = result.get("environment_variable_names")
    safe_names = sorted({str(name) for name in names if isinstance(name, str) and name}) if isinstance(names, list) else []
    receipt = {
        "mode": str(result.get("mode") or "configured"),
        "status": "attempted" if callable(opener) else "unavailable",
        "environment_variable_names": safe_names,
    }
    return opener if callable(opener) else None, receipt


def _policy_exclusion_for_request(
    ledger: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any] | None:
    evidence = request.get("source_evidence")
    family_id = evidence.get("source_family_id") if isinstance(evidence, Mapping) else None
    for exclusion in ledger.get("policy_exclusions", []):
        if not isinstance(exclusion, Mapping):
            continue
        if exclusion.get("exclusion_id") == "paywalled_official_video" and exclusion.get("family_id") == family_id:
            return sanitize_for_receipt(dict(exclusion))
    return None


def _refresh_public_acquisition_accounting(ledger: dict[str, Any]) -> None:
    """Fan each request result back out without collapsing program-item scope."""

    requests = {
        str(request.get("request_key")): request
        for request in ledger.get("request_results", [])
        if isinstance(request, dict) and isinstance(request.get("request_key"), str)
    }
    artifacts = ledger.get("artifacts")
    artifact_by_request = {
        str(artifact.get("request_key")): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and isinstance(artifact.get("request_key"), str)
    } if isinstance(artifacts, list) else {}
    blockers: list[dict[str, Any]] = []
    complete = True
    for item in ledger.get("program_item_results", []):
        if not isinstance(item, dict):
            complete = False
            continue
        request = requests.get(str(item.get("request_key"))) if item.get("request_key") else None
        if request is None:
            if not isinstance(item.get("blocker"), Mapping):
                item["blocker"] = {"code": "no_admissible_artifact", "reason": "missing_request_result"}
            blockers.append({"program_item_id": item.get("program_item_id"), **dict(item["blocker"])})
            continue
        item["attempts"] = list(request.get("attempts") or [])
        status = request.get("status")
        if status == "succeeded":
            artifact = artifact_by_request.get(str(request.get("request_key")))
            if artifact is None:
                item["blocker"] = {"code": "no_admissible_artifact", "reason": "successful_request_missing_artifact"}
                blockers.append({"program_item_id": item.get("program_item_id"), **dict(item["blocker"])})
            else:
                item["artifact_id"] = artifact.get("artifact_id")
                item["blocker"] = None
        elif status == "policy_excluded_official_video":
            item["policy_boundary"] = request.get("policy_boundary")
            item["blocker"] = {"code": "policy_excluded_official_video", "reason": "locked_official_video_policy"}
        elif status == "workshop_owner_result":
            item["workshop_owner"] = request.get("workshop_owner")
            item["blocker"] = {"code": "no_admissible_artifact", "reason": "workshop_owner_result_requires_per_paper_binding"}
            blockers.append({"program_item_id": item.get("program_item_id"), **dict(item["blocker"])})
        elif status in {"pending", None}:
            complete = False
            item["blocker"] = None
        else:
            item["blocker"] = {"code": str(status), "reason": "source_backed_request_did_not_yield_valid_artifact"}
            blockers.append({"program_item_id": item.get("program_item_id"), **dict(item["blocker"])})
    counts = ledger.setdefault("counts", {})
    if isinstance(counts, dict):
        rows = ledger.get("program_item_results")
        counts["program_item_result_count"] = len(rows) if isinstance(rows, list) else 0
        counts["artifact_count"] = len(artifacts) if isinstance(artifacts, list) else 0
        counts["blocked_program_item_count"] = sum(
            isinstance(item, Mapping) and isinstance(item.get("blocker"), Mapping)
            for item in rows if isinstance(rows, list)
        )
        counts["pending_program_item_count"] = sum(
            isinstance(item, Mapping) and item.get("blocker") is None and not item.get("artifact_id")
            for item in rows if isinstance(rows, list)
        )
    # Per-item blockers remain in ``program_item_results``.  Duplicating them
    # into the global stage blocker list on every atomic checkpoint would make
    # a resumed 10,515-row run grow without bound and obscure actual owner
    # blockers added by the runner.
    ledger["public_item_blocker_count"] = len(blockers)
    ledger["public_acquisition_complete"] = complete and all(
        isinstance(item, Mapping) and (isinstance(item.get("blocker"), Mapping) or item.get("artifact_id"))
        for item in ledger.get("program_item_results", [])
    )


def _apply_workshop_owner_result(
    *,
    request: dict[str, Any],
    workshop_owner: Callable[[str], Mapping[str, Any]] | None,
    workshop_parent_venue: str | None,
) -> None:
    """Bind the existing workshop owner output instead of recreating its client."""

    if workshop_owner is None or not workshop_parent_venue:
        request["status"] = "no_admissible_artifact"
        request["workshop_owner"] = {"status": "missing", "reason": "approved_workshop_owner_unavailable"}
        return
    try:
        owner_result = workshop_owner(workshop_parent_venue)
    except Exception as exc:  # noqa: BLE001 - preserve an actionable, secret-free blocker
        request["status"] = "no_admissible_artifact"
        request["workshop_owner"] = {"status": "blocked", "reason": type(exc).__name__}
        return
    if not isinstance(owner_result, Mapping):
        request["status"] = "no_admissible_artifact"
        request["workshop_owner"] = {"status": "invalid_result"}
        return
    download = owner_result.get("download")
    request["workshop_owner"] = sanitize_for_receipt(
        {
            "status": owner_result.get("status"),
            "download_receipt_path": owner_result.get("download_receipt_path"),
            "inventory_path": owner_result.get("inventory_path"),
            "download_totals": download.get("totals") if isinstance(download, Mapping) else None,
        }
    )
    if owner_result.get("status") in {"succeeded", "partial"} and isinstance(owner_result.get("download_receipt_path"), str):
        request["status"] = "workshop_owner_result"
    else:
        request["status"] = "no_admissible_artifact"


def _execute_public_acquisition_sequential(
    *,
    workspace: Path,
    ledger: dict[str, Any],
    opener: Callable[[str], Any],
    checkpoint: Callable[[dict[str, Any]], None],
    auth_recovery: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_parent_venue: str | None = None,
    max_attempts: int = 3,
    response_deadline_seconds: float = _PUBLIC_RESPONSE_READ_DEADLINE_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute public source requests with durable, per-request checkpoints.

    This adapter has no credentials and no OpenReview client.  It delegates a
    workshop family to the existing registered workshop owner and only records
    that owner's sanitized receipt/output boundary.  Every ordinary request
    is first attempted publicly; only an injected, configured recovery may
    follow a 401/403, and it can do so once.
    """

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if not callable(opener) or not callable(checkpoint):
        raise ValueError("public acquisition requires injected opener and checkpoint")
    resolved_workspace = workspace.resolve()
    artifacts = ledger.setdefault("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("ledger artifacts must be a list")
    artifact_keys = {str(item.get("request_key")) for item in artifacts if isinstance(item, Mapping)}
    lineage = ledger.get("lineage") if isinstance(ledger.get("lineage"), Mapping) else {}
    requests = ledger.get("request_results")
    if not isinstance(requests, list):
        raise ValueError("ledger request_results must be a list")

    for request in requests:
        if not isinstance(request, dict) or request.get("status") not in {"pending", "unreachable_fetch_error", "unreachable_http_error", "auth_required"}:
            continue
        source_url = request.get("source_url")
        if not isinstance(source_url, str) or not source_url:
            request["status"] = "no_admissible_artifact"
            _refresh_public_acquisition_accounting(ledger)
            ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
            checkpoint(ledger)
            continue
        request["artifact_kind"] = classify_public_artifact_kind(source_url)
        policy = _policy_exclusion_for_request(ledger, request)
        if policy is not None:
            request.update({"status": "policy_excluded_official_video", "policy_boundary": policy, "attempts": [], "retry_events": []})
            _refresh_public_acquisition_accounting(ledger)
            ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
            checkpoint(ledger)
            continue
        evidence = request.get("source_evidence")
        if isinstance(evidence, Mapping) and evidence.get("source_family_id") == "openreview-workshop":
            _apply_workshop_owner_result(
                request=request,
                workshop_owner=workshop_owner,
                workshop_parent_venue=workshop_parent_venue,
            )
            _refresh_public_acquisition_accounting(ledger)
            ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
            checkpoint(ledger)
            continue

        destination = _artifact_destination(resolved_workspace, request)
        retry_events: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        active_opener = opener
        used_auth_recovery = False
        for attempt in range(1, max_attempts + 1):
            result = stream_to_workspace(
                workspace=resolved_workspace,
                destination=destination,
                opener=active_opener,
                url=source_url,
                artifact_kind=str(request["artifact_kind"]),
                response_deadline_seconds=response_deadline_seconds,
            )
            attempts.append(_request_attempt(result, attempt=attempt))
            http_status = result.get("http_status")
            if result.get("status") == "unreachable_http_error" and http_status in {401, 403}:
                if used_auth_recovery or auth_recovery is None:
                    result["status"] = "auth_required"
                    break
                used_auth_recovery = True
                try:
                    recovery_result = auth_recovery(source_url)
                except Exception as exc:  # noqa: BLE001 - never expose auth configuration data
                    recovery_result = {"mode": "configured", "error_type": type(exc).__name__}
                recovered_opener, recovery_receipt = _safe_auth_recovery(recovery_result)
                request["auth_recovery"] = recovery_receipt
                if recovered_opener is None:
                    result["status"] = "auth_required"
                    break
                active_opener = recovered_opener
                continue
            if _is_transient_public_result(result) and attempt < max_attempts:
                delay = _PUBLIC_BACKOFF_SECONDS[min(attempt - 1, len(_PUBLIC_BACKOFF_SECONDS) - 1)]
                retry_events.append(_request_attempt(result, attempt=attempt, delay=delay))
                sleeper(delay)
                continue
            break
        if result is None:  # pragma: no cover - defensive for malformed injected seams
            result = {"status": "unreachable_fetch_error", "artifact_kind": request["artifact_kind"]}
        request.update(sanitize_for_receipt(result))
        request["attempts"] = attempts
        request["retry_events"] = retry_events
        if result.get("status") == "succeeded" and str(request.get("request_key")) not in artifact_keys:
            artifact = {
                "artifact_id": sha256(f"{request['request_key']}\n{request['sha256']}".encode("utf-8")).hexdigest(),
                "request_key": request["request_key"],
                "artifact_kind": request["artifact_kind"],
                "local_path": request["local_path"],
                "mime": request.get("content_type"),
                "byte_count": request["byte_count"],
                "sha256": request["sha256"],
                "lineage": sanitize_for_receipt(dict(lineage)),
            }
            artifacts.append(artifact)
            artifact_keys.add(str(request["request_key"]))
        _refresh_public_acquisition_accounting(ledger)
        ledger["status"] = "public_lane_complete" if ledger.get("public_acquisition_complete") else "public_lane_partial"
        ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
        checkpoint(ledger)
    return ledger


def _run_isolated_public_request(
    *,
    workspace: Path,
    request: Mapping[str, Any],
    lineage: Mapping[str, Any],
    policy_exclusions: Sequence[Mapping[str, Any]],
    opener: Callable[[str], Any],
    auth_recovery: Callable[[str], Mapping[str, Any]] | None,
    max_attempts: int,
    response_deadline_seconds: float,
) -> dict[str, Any]:
    """Run one ordinary public request against a private single-request ledger.

    Returns a ``{"request": ..., "artifacts": [...]}`` payload and mutates no
    shared state, so it is safe to run concurrently in a worker thread; the
    caller serializes every ledger checkpoint on the main thread.  A failure
    is reported as a sanitized error result rather than raised.
    """

    item_ids = request.get("linked_program_item_ids")
    item_id = str(item_ids[0]) if isinstance(item_ids, list) and item_ids else "request"
    mini_ledger: dict[str, Any] = {
        "lineage": dict(lineage),
        "request_results": [dict(request)],
        "program_item_results": [{"program_item_id": item_id, "request_key": request.get("request_key"), "blocker": None}],
        "artifacts": [],
        "policy_exclusions": [dict(row) for row in policy_exclusions],
    }
    try:
        completed = _execute_public_acquisition_sequential(
            workspace=workspace,
            ledger=mini_ledger,
            opener=opener,
            checkpoint=lambda _payload: None,
            auth_recovery=auth_recovery,
            max_attempts=max_attempts,
            response_deadline_seconds=response_deadline_seconds,
        )
        return sanitize_for_receipt(
            {
                "request": completed["request_results"][0],
                "artifacts": completed["artifacts"],
            }
        )
    except Exception as exc:  # noqa: BLE001 - return only an error class to the parent receipt.
        return {
            "request": {
                "request_key": request.get("request_key"),
                "status": "unreachable_fetch_error",
                "source_url": request.get("source_url"),
                "artifact_kind": request.get("artifact_kind"),
                "error_type": type(exc).__name__,
                "attempts": [],
                "retry_events": [],
            },
            "artifacts": [],
        }


def _concurrent_transport_available() -> bool:
    """Thread-pool transport is available on every supported host.

    Stage 4's public lane is pure network I/O, so it now fans out across a
    bounded ``ThreadPoolExecutor`` rather than forked processes.  Forking a
    process that has already loaded Foundation/Objective-C or an SSL context
    -- as this runner has -- crashes the child on macOS (``fork()`` is unsafe
    in a multi-threaded process there), which manifested as every download
    worker dying with no result.  Threads share the injected opener directly,
    need no pickling, and cannot hit that crash.
    """

    return True


def _checkpoint_public_request(
    *,
    ledger: dict[str, Any],
    request: dict[str, Any],
    completed_request: Mapping[str, Any],
    completed_artifacts: Sequence[Mapping[str, Any]],
    checkpoint: Callable[[dict[str, Any]], None],
) -> None:
    """Merge one isolated request result and atomically checkpoint in the parent."""

    request.update(sanitize_for_receipt(dict(completed_request)))
    artifacts = ledger.setdefault("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("ledger artifacts must be a list")
    known = {str(item.get("request_key")) for item in artifacts if isinstance(item, Mapping)}
    for artifact in completed_artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("concurrent artifact result is invalid")
        request_key = artifact.get("request_key")
        if not isinstance(request_key, str) or request_key != request.get("request_key"):
            raise ValueError("concurrent artifact request binding is invalid")
        if request_key not in known:
            artifacts.append(sanitize_for_receipt(dict(artifact)))
            known.add(request_key)
    artifacts.sort(key=lambda artifact: str(artifact.get("request_key") if isinstance(artifact, Mapping) else ""))
    _refresh_public_acquisition_accounting(ledger)
    ledger["status"] = "public_lane_complete" if ledger.get("public_acquisition_complete") else "public_lane_partial"
    ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
    checkpoint(ledger)


def _execute_public_acquisition_concurrent(
    *,
    workspace: Path,
    ledger: dict[str, Any],
    opener: Callable[[str], Any],
    checkpoint: Callable[[dict[str, Any]], None],
    auth_recovery: Callable[[str], Mapping[str, Any]] | None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None,
    workshop_parent_venue: str | None,
    max_attempts: int,
    response_deadline_seconds: float,
    worker_count: int,
) -> dict[str, Any]:
    """Run independent public URLs with fixed process concurrency and parent checkpoints."""

    if worker_count <= 1 or not _concurrent_transport_available():
        return _execute_public_acquisition_sequential(
            workspace=workspace,
            ledger=ledger,
            opener=opener,
            checkpoint=checkpoint,
            auth_recovery=auth_recovery,
            workshop_owner=workshop_owner,
            workshop_parent_venue=workshop_parent_venue,
            max_attempts=max_attempts,
            response_deadline_seconds=response_deadline_seconds,
        )
    requests = ledger.get("request_results")
    if not isinstance(requests, list):
        raise ValueError("ledger request_results must be a list")
    lineage = ledger.get("lineage") if isinstance(ledger.get("lineage"), Mapping) else {}
    policy_exclusions = ledger.get("policy_exclusions") if isinstance(ledger.get("policy_exclusions"), list) else []
    candidates: list[dict[str, Any]] = []
    uncheckpointed_control_results = 0

    def checkpoint_control_results(*, force: bool = False) -> None:
        """Persist local policy/owner outcomes in bounded atomic batches.

        A public request remains checkpointed as soon as its worker completes.
        The local control rows perform no public transfer themselves, so a
        small batch avoids re-hashing and rewriting a 10k-item ledger once per
        policy pointer while retaining a durable resume boundary.
        """

        nonlocal uncheckpointed_control_results
        if not uncheckpointed_control_results:
            return
        if not force and uncheckpointed_control_results < _CONTROL_RESULT_CHECKPOINT_BATCH_SIZE:
            return
        _refresh_public_acquisition_accounting(ledger)
        ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
        checkpoint(ledger)
        uncheckpointed_control_results = 0

    for request in requests:
        if not isinstance(request, dict) or request.get("status") not in {"pending", "unreachable_fetch_error", "unreachable_http_error", "auth_required"}:
            continue
        source_url = request.get("source_url")
        if not isinstance(source_url, str) or not source_url:
            request["status"] = "no_admissible_artifact"
            uncheckpointed_control_results += 1
            checkpoint_control_results()
            continue
        request["artifact_kind"] = classify_public_artifact_kind(source_url)
        policy = _policy_exclusion_for_request(ledger, request)
        if policy is not None:
            request.update({"status": "policy_excluded_official_video", "policy_boundary": policy, "attempts": [], "retry_events": []})
            uncheckpointed_control_results += 1
            checkpoint_control_results()
            continue
        evidence = request.get("source_evidence")
        if isinstance(evidence, Mapping) and evidence.get("source_family_id") == "openreview-workshop":
            _apply_workshop_owner_result(
                request=request,
                workshop_owner=workshop_owner,
                workshop_parent_venue=workshop_parent_venue,
            )
            uncheckpointed_control_results += 1
            checkpoint_control_results()
            continue
        candidates.append(request)
    checkpoint_control_results(force=True)
    candidates.sort(key=lambda request: str(request["request_key"]))
    ledger["public_transport"] = {
        "scheduler": "bounded_threads_v1",
        "worker_count": worker_count,
        "response_deadline_seconds": response_deadline_seconds,
        "max_in_flight": 0,
    }
    max_in_flight = min(worker_count, len(candidates))
    ledger["public_transport"]["max_in_flight"] = max_in_flight
    if candidates:
        # I/O-bound public downloads fan out across bounded threads.  The
        # workers only read the request and stream bytes to unique per-request
        # destinations; every ledger mutation and checkpoint stays on this
        # main thread, so the durable single-writer resume boundary is
        # preserved exactly as in the sequential path.
        shared_policy_exclusions = [dict(row) for row in policy_exclusions if isinstance(row, Mapping)]
        shared_lineage = dict(lineage)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_request = {
                executor.submit(
                    _run_isolated_public_request,
                    workspace=workspace,
                    request=dict(request),
                    lineage=shared_lineage,
                    policy_exclusions=shared_policy_exclusions,
                    opener=opener,
                    auth_recovery=auth_recovery,
                    max_attempts=max_attempts,
                    response_deadline_seconds=response_deadline_seconds,
                ): request
                for request in candidates
            }
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                payload = future.result()
                completed_request = payload.get("request") if isinstance(payload, Mapping) else None
                completed_artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
                if not isinstance(completed_request, Mapping) or not isinstance(completed_artifacts, list):
                    raise ValueError("concurrent public request result is invalid")
                _checkpoint_public_request(
                    ledger=ledger,
                    request=request,
                    completed_request=completed_request,
                    completed_artifacts=completed_artifacts,
                    checkpoint=checkpoint,
                )
    return ledger


def execute_public_acquisition(
    *,
    workspace: Path,
    ledger: dict[str, Any],
    opener: Callable[[str], Any],
    checkpoint: Callable[[dict[str, Any]], None],
    auth_recovery: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_parent_venue: str | None = None,
    max_attempts: int = 3,
    response_deadline_seconds: float = _PUBLIC_RESPONSE_READ_DEADLINE_SECONDS,
    worker_count: int = 1,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute public acquisition with bounded process concurrency when requested."""

    if worker_count < 1:
        raise ValueError("public acquisition worker count must be positive")
    # A workshop request is a pointer to its registry-owned pipeline, not an
    # independent OpenReview download.  Several source records can therefore
    # name the same parent venue.  Run that owner once per parent venue and
    # attach the same durable owner receipt to every such request.  Apart from
    # avoiding duplicated provider work, this keeps a slow workshop run from
    # serially delaying the public URL transport once for every pointer.
    if workshop_owner is not None:
        owner_results: dict[str, Mapping[str, Any]] = {}
        registered_owner = workshop_owner

        def workshop_owner(parent_venue: str) -> Mapping[str, Any]:
            if parent_venue not in owner_results:
                owner_results[parent_venue] = registered_owner(parent_venue)
            return owner_results[parent_venue]

    if worker_count == 1:
        return _execute_public_acquisition_sequential(
            workspace=workspace,
            ledger=ledger,
            opener=opener,
            checkpoint=checkpoint,
            auth_recovery=auth_recovery,
            workshop_owner=workshop_owner,
            workshop_parent_venue=workshop_parent_venue,
            max_attempts=max_attempts,
            response_deadline_seconds=response_deadline_seconds,
            sleeper=sleeper,
        )
    return _execute_public_acquisition_concurrent(
        workspace=workspace,
        ledger=ledger,
        opener=opener,
        checkpoint=checkpoint,
        auth_recovery=auth_recovery,
        workshop_owner=workshop_owner,
        workshop_parent_venue=workshop_parent_venue,
        max_attempts=max_attempts,
        response_deadline_seconds=response_deadline_seconds,
        worker_count=worker_count,
    )


# --------------------------------------------------------------------------
# Stage 4 lane on a program-item ledger
# --------------------------------------------------------------------------

PUBLIC_LANE_ID = "public-program-items"
PUBLIC_LANE_INCOMPLETE = "public_acquisition_lane_incomplete"


def public_acquisition_lane(
    *,
    enabled: bool,
    opener_factory: Callable[[], Any] | None,
    auth_recovery: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_parent_venue: str | None = None,
    worker_count: int = 1,
) -> Any:
    """The public program-item lane: build the transport, then acquire every item.

    The transport is created only here, after the frame has validated
    lineage and capacity. The lane completes when the ledger records a
    complete public acquisition.
    """

    from conference_pipeline.production.download_corpus_stage import stage4_blocker
    from conference_pipeline.production.hooks import AcquisitionLane, LaneResult

    def run(ctx: Any, _frozen: Any, _prior: Any) -> Any:
        state = ctx.state
        if not enabled:
            return LaneResult(completed=False)
        blockers: list[dict[str, Any]] = []
        fields: dict[str, Any] = {}
        if opener_factory is None:
            blockers.append(stage4_blocker("public_acquisition_transport_unavailable"))
        else:
            try:
                opener = opener_factory()
            except Exception as exc:  # noqa: BLE001 - do not leak transport configuration
                blockers.append(stage4_blocker("public_acquisition_transport_unavailable", reason=type(exc).__name__))
            else:
                if not callable(opener):
                    blockers.append(stage4_blocker("public_acquisition_transport_invalid"))
                else:
                    try:
                        ledger = execute_public_acquisition(
                            workspace=ctx.workspace,
                            ledger=state.ledger,
                            opener=opener,
                            checkpoint=state.checkpoint,
                            auth_recovery=auth_recovery,
                            workshop_owner=workshop_owner,
                            workshop_parent_venue=workshop_parent_venue,
                            worker_count=worker_count,
                        )
                        ledger["public_transport_executed"] = True
                    except (OSError, ValueError, TypeError) as exc:
                        blockers.append(stage4_blocker("public_acquisition_dispatch_failed", reason=type(exc).__name__))
                    else:
                        state.ledger = ledger
                        artifacts_valid = all(
                            resume_artifact_is_valid(
                                workspace=ctx.workspace,
                                artifact=artifact,
                                expected_lineage=ledger.get("lineage", {}),
                            )
                            for artifact in ledger.get("artifacts", [])
                            if isinstance(artifact, Mapping)
                        )
                        digests_verified = all(
                            resume_artifact_is_valid(
                                workspace=ctx.workspace,
                                artifact=artifact,
                                expected_lineage=ledger.get("lineage", {}),
                            )
                            for artifact in ledger.get("artifacts", [])
                            if isinstance(artifact, Mapping)
                        )
                        fields = {
                            "public_acquisition_lane_status": ledger.get("status"),
                            "complete_result_or_blocker_accounting": bool(ledger.get("public_acquisition_complete")),
                            "validated_local_artifacts": artifacts_valid,
                            "artifact_payload_digests_verified": digests_verified,
                        }
        return LaneResult(
            blockers=blockers,
            receipt_fields=fields,
            completed=bool(state.ledger.get("public_acquisition_complete")),
        )

    return AcquisitionLane(lane_id=PUBLIC_LANE_ID, run=run, incomplete_blocker=PUBLIC_LANE_INCOMPLETE)
