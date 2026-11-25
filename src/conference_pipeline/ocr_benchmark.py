"""Benchmark vision OCR models against an OpenAI authority.

The default mode is a dry-run fixture that uses local image metadata and mock
JSON. Pass ``--live`` to make provider calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from conference_pipeline.costs.estimator import (
    PricingCatalog,
    _discover_image_paths,
    _image_dimensions,
    estimate_anthropic_image_tokens,
    estimate_openai_image_tokens,
    load_pricing_catalog,
)
from conference_pipeline.slide_ocr import (
    AnthropicSlideOCRService,
    GeminiSlideOCRService,
    OpenAISlideOCRService,
    evaluate_ocr_confidence,
    parse_ocr_response,
)
from conference_pipeline.model_governance.registry import load_registry
from conference_pipeline.workflow_runs import write_json_atomic


def _authority_models() -> tuple[str, ...]:
    """Registry-backed OCR authority models (keeps literals out of code)."""
    benchmark = load_registry().get("ocr.slide_image_parse").benchmark
    return tuple(str(model) for model in benchmark.get("authority_models", []))


AUTHORITY_MODELS = _authority_models()


@dataclass(frozen=True)
class CandidateModel:
    provider: str
    model: str
    role: str = "candidate"


def _candidate_models() -> tuple[CandidateModel, ...]:
    """Registry-backed OCR candidates: each benchmark candidate paired with the
    provider its allowed selection registers it under."""
    policy = load_registry().get("ocr.slide_image_parse")
    provider_by_model = {item.model: item.provider for item in policy.allowed_selections}
    candidates = []
    for model in policy.benchmark.get("candidates", []):
        provider = provider_by_model.get(str(model))
        if provider is None:
            raise ValueError(
                f"ocr.slide_image_parse benchmark candidate {model!r} has no allowed selection"
            )
        candidates.append(CandidateModel(provider, str(model)))
    return tuple(candidates)


CANDIDATE_MODELS = _candidate_models()

SCHEMA_FIELDS = (
    ("slide_text",),
    ("slide_text", "title"),
    ("slide_text", "bullet_points"),
    ("slide_text", "other_text"),
    ("slide_text", "urls"),
    ("visual_elements",),
    ("key_concepts",),
    ("references",),
)

LOCATOR_RE = re.compile(
    r"(?i)(https?://\S+|doi:\s*10\.\d{4,9}/[-._;()/:a-z0-9]+|10\.\d{4,9}/[-._;()/:a-z0-9]+|arxiv:\s*\d{4}\.\d{4,5})"
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _collect_text(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_collect_text(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_collect_text(item))
        return values
    if value is None:
        return []
    return [str(value)]


def _flatten_response_text(response: str) -> str:
    parsed = parse_ocr_response(response)
    return " ".join(part.strip() for part in _collect_text(parsed) if part.strip())


def _schema_completeness(parsed: dict[str, object]) -> float:
    present = 0
    for path in SCHEMA_FIELDS:
        current: object = parsed
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current not in (None, "", []):
            present += 1
    return present / len(SCHEMA_FIELDS)


def _extract_locators(text: str) -> set[str]:
    return {match.group(0).lower().strip().rstrip(".,;") for match in LOCATOR_RE.finditer(text)}


def _extract_numbers(text: str) -> set[str]:
    return {match.group(0).lower() for match in NUMBER_RE.finditer(text)}


def _recall(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    return sorted_values[lower] * (upper - index) + sorted_values[upper] * (index - lower)


def _select_sample_images(workspace: Path, sample_size: int) -> tuple[list[Path], int]:
    images = sorted(_discover_image_paths(workspace))
    if sample_size <= 0 or len(images) <= sample_size:
        return images, len(images)
    if sample_size == 1:
        return [images[0]], len(images)
    indexes = {
        round(i * (len(images) - 1) / (sample_size - 1))
        for i in range(sample_size)
    }
    return [images[index] for index in sorted(indexes)], len(images)


def _image_input_tokens(provider: str, image_path: Path) -> int:
    dimensions = _image_dimensions(image_path)
    if dimensions is None:
        return 1105 if provider != "anthropic" else 1560
    width, height = dimensions
    if provider == "anthropic":
        return estimate_anthropic_image_tokens(width, height)
    return estimate_openai_image_tokens(width, height)


def _estimated_output_tokens(response_text: str | None = None, default: int = 1000) -> int:
    if not response_text:
        return default
    return max(1, math.ceil(len(response_text) / 4))


def _estimate_call_cost(
    catalog: PricingCatalog,
    provider: str,
    model: str,
    image_path: Path,
    response_text: str | None = None,
    *,
    default_output_tokens: int = 1000,
) -> dict[str, object]:
    input_tokens = _image_input_tokens(provider, image_path)
    output_tokens = _estimated_output_tokens(response_text, default_output_tokens)
    cost, source_urls, warnings = catalog.estimate_tokens(
        provider,
        model,
        input_tokens,
        output_tokens,
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "source_urls": source_urls,
        "warnings": warnings,
    }


def _score_response(authority_response: str, candidate_response: str) -> dict[str, object]:
    authority_text = _flatten_response_text(authority_response)
    candidate_text = _flatten_response_text(candidate_response)
    parsed_candidate = parse_ocr_response(candidate_response)
    confidence = evaluate_ocr_confidence(candidate_response)
    return {
        "json_valid": "parse_error" not in parsed_candidate,
        "confidence_accepted": confidence.accepted,
        "confidence_reason": confidence.reason,
        "similarity": round(SequenceMatcher(None, authority_text, candidate_text).ratio(), 6),
        "schema_completeness": round(_schema_completeness(parsed_candidate), 6),
        "locator_recall": round(_recall(_extract_locators(authority_text), _extract_locators(candidate_text)), 6),
        "numeric_recall": round(_recall(_extract_numbers(authority_text), _extract_numbers(candidate_text)), 6),
    }


def _mock_ocr_response(image_path: Path, title: str, model: str, *, authority: bool = False) -> str:
    dimensions = _image_dimensions(image_path)
    width, height = dimensions or (0, 0)
    stem = image_path.stem.replace("_", " ")
    bullet_points = [
        f"{title} sample slide: {stem}",
        f"Image dimensions: {width} x {height}",
        "Visible topic: multimodal representation learning",
    ]
    references: list[dict[str, str]] = [
        {
            "title": "Sample benchmark paper",
            "authors": "A. Researcher et al.",
            "venue": "ICLR",
            "year": "2025",
            "url": "10.1234/example.benchmark",
        }
    ]
    if "nano" in model:
        bullet_points = bullet_points[:2]
        references = []
    elif "flash-lite" in model:
        bullet_points.append("Low-cost parse preserves most visible text.")
    elif not authority:
        bullet_points.append(f"Parsed by {model}.")

    return json.dumps(
        {
            "slide_text": {
                "title": f"{stem} OCR",
                "bullet_points": bullet_points,
                "equations": [],
                "other_text": "Accuracy 92.5% on a held-out benchmark.",
                "urls": ["https://openreview.net/forum?id=example"],
            },
            "visual_elements": [
                {
                    "type": "figure",
                    "description": "A slide image used for OCR benchmark evaluation.",
                    "data": f"{width}x{height}",
                    "labels": ["accuracy", "benchmark"],
                }
            ],
            "key_concepts": [
                {
                    "concept": "OCR benchmark",
                    "explanation": "Compare candidate model extraction against an authority parse.",
                }
            ],
            "references": references,
        },
        sort_keys=True,
    )


def _api_key_for_provider(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "")
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return ""


def _service_for_candidate(candidate: CandidateModel, timeout_seconds: float):
    api_key = _api_key_for_provider(candidate.provider)
    if not api_key:
        return None
    if candidate.provider == "openai":
        return OpenAISlideOCRService(
            api_key=api_key,
            model=candidate.model,
            timeout_seconds=timeout_seconds,
        )
    if candidate.provider == "anthropic":
        return AnthropicSlideOCRService(
            api_key=api_key,
            model=candidate.model,
            timeout_seconds=timeout_seconds,
        )
    if candidate.provider == "gemini":
        return GeminiSlideOCRService(
            api_key=api_key,
            model=candidate.model,
            timeout_seconds=timeout_seconds,
        )
    return None


async def _call_service(service: object, image_path: Path, title: str) -> str:
    ocr_slide = getattr(service, "ocr_slide")
    return await ocr_slide(image_path, title, None)


async def _run_authority_call(
    image_path: Path,
    title: str,
    *,
    live: bool,
    catalog: PricingCatalog,
    per_call_timeout_seconds: float,
) -> dict[str, object]:
    if not live:
        model = AUTHORITY_MODELS[0]
        response = _mock_ocr_response(image_path, title, model, authority=True)
        cost = _estimate_call_cost(catalog, "openai", model, image_path, response)
        return {
            "image_path": str(image_path),
            "provider": "openai",
            "model": model,
            "status": "ok",
            "latency_seconds": 0.0,
            "response": response,
            **cost,
        }

    api_key = _api_key_for_provider("openai")
    if not api_key:
        return {
            "image_path": str(image_path),
            "provider": "openai",
            "model": AUTHORITY_MODELS[0],
            "status": "skipped_missing_api_key",
            "latency_seconds": 0.0,
            "response": "",
            "estimated_cost_usd": 0.0,
            "errors": ["OPENAI_API_KEY is required for live authority OCR."],
        }

    errors = []
    attempt_estimated_cost_usd = 0.0
    for model in AUTHORITY_MODELS:
        service = OpenAISlideOCRService(
            api_key=api_key,
            model=model,
            timeout_seconds=per_call_timeout_seconds,
        )
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                _call_service(service, image_path, title),
                timeout=per_call_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - report model availability/API failures.
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
            continue
        latency = time.perf_counter() - started
        cost = _estimate_call_cost(catalog, "openai", model, image_path, response)
        attempt_estimated_cost_usd += float(cost["estimated_cost_usd"])
        confidence = evaluate_ocr_confidence(response)
        if not confidence.accepted:
            errors.append(f"{model}: low_confidence_authority:{confidence.reason}")
            continue
        return {
            "image_path": str(image_path),
            "provider": "openai",
            "model": model,
            "status": "ok",
            "latency_seconds": round(latency, 6),
            "response": response,
            **cost,
            "attempt_estimated_cost_usd": round(attempt_estimated_cost_usd, 8),
            "errors": errors,
        }

    return {
        "image_path": str(image_path),
        "provider": "openai",
        "model": AUTHORITY_MODELS[-1],
        "status": "authority_failed",
        "latency_seconds": 0.0,
        "response": "",
        "estimated_cost_usd": 0.0,
        "attempt_estimated_cost_usd": round(attempt_estimated_cost_usd, 8),
        "errors": errors,
    }


def _candidate_summary(
    candidate: CandidateModel,
    records: list[dict[str, object]],
    *,
    total_image_count: int,
) -> dict[str, object]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    statuses = [str(record.get("status", "unknown")) for record in records]
    if ok_records and len(ok_records) == len(records):
        status = "ok"
    elif ok_records:
        status = "partial"
    elif "failed" in statuses:
        status = "failed"
    elif statuses:
        status = statuses[0]
    else:
        status = "no_records"
    costs = [float(record.get("estimated_cost_usd", 0.0)) for record in ok_records]
    latencies = [float(record.get("latency_seconds", 0.0)) for record in ok_records]
    score_keys = (
        "similarity",
        "schema_completeness",
        "locator_recall",
        "numeric_recall",
    )
    summary: dict[str, object] = {
        "provider": candidate.provider,
        "model": candidate.model,
        "status": status,
        "sample_calls": len(records),
        "successful_calls": len(ok_records),
        "failed_calls": len(records) - len(ok_records),
        "sample_cost_usd": round(sum(costs), 6),
        "avg_cost_per_image_usd": round(_safe_mean(costs), 8),
        "projected_full_corpus_cost_usd": round(_safe_mean(costs) * total_image_count, 6),
        "latency_p50_seconds": round(_percentile(latencies, 0.5), 6),
        "latency_p95_seconds": round(_percentile(latencies, 0.95), 6),
    }
    for key in score_keys:
        summary[f"avg_{key}"] = round(
            _safe_mean([float(record.get(key, 0.0)) for record in ok_records]),
            6,
        )
    summary["json_valid_rate"] = round(
        _safe_mean([1.0 if record.get("json_valid") else 0.0 for record in ok_records]),
        6,
    )
    summary["confidence_accept_rate"] = round(
        _safe_mean([1.0 if record.get("confidence_accepted") else 0.0 for record in ok_records]),
        6,
    )
    return summary


async def benchmark_ocr_models(
    workspace: Path,
    *,
    sample_size: int = 12,
    live: bool = False,
    budget_cap_usd: float = 30.0,
    output_dir: Path | None = None,
    title: str = "Conference OCR benchmark",
    include_raw: bool = False,
    pricing_catalog_path: Path | None = None,
    per_call_timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Run the OCR benchmark and write a JSON receipt."""

    load_dotenv()
    catalog = load_pricing_catalog(pricing_catalog_path)
    sample_images, total_image_count = _select_sample_images(workspace, sample_size)
    receipt_dir = output_dir or workspace / "storage" / "transcription-ocr" / "ocr-model-benchmark"
    receipt_path = receipt_dir / f"ocr_model_benchmark_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"

    preflight_cost = 0.0
    for image_path in sample_images:
        for model in AUTHORITY_MODELS[:1]:
            preflight_cost += float(
                _estimate_call_cost(catalog, "openai", model, image_path)["estimated_cost_usd"]
            )
        for candidate in CANDIDATE_MODELS:
            preflight_cost += float(
                _estimate_call_cost(catalog, candidate.provider, candidate.model, image_path)[
                    "estimated_cost_usd"
                ]
            )

    if live and preflight_cost > budget_cap_usd:
        raise RuntimeError(
            f"Preflight sample estimate ${preflight_cost:.2f} exceeds benchmark cap ${budget_cap_usd:.2f}."
        )

    authority_records: list[dict[str, object]] = []
    candidate_records: dict[str, list[dict[str, object]]] = {
        f"{candidate.provider}:{candidate.model}": [] for candidate in CANDIDATE_MODELS
    }
    spent_estimate = 0.0

    for image_path in sample_images:
        authority_record = await _run_authority_call(
            image_path,
            title,
            live=live,
            catalog=catalog,
            per_call_timeout_seconds=per_call_timeout_seconds,
        )
        spent_estimate += float(
            authority_record.get(
                "attempt_estimated_cost_usd",
                authority_record.get("estimated_cost_usd", 0.0),
            )
        )
        authority_records.append(authority_record)

    services = {
        f"{candidate.provider}:{candidate.model}": _service_for_candidate(
            candidate,
            per_call_timeout_seconds,
        )
        for candidate in CANDIDATE_MODELS
    } if live else {}

    for candidate in CANDIDATE_MODELS:
        key = f"{candidate.provider}:{candidate.model}"
        service = services.get(key)
        if live and service is None:
            candidate_records[key].append(
                {
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "status": "skipped_missing_api_key",
                    "estimated_cost_usd": 0.0,
                    "error": f"{candidate.provider.upper()} API key is not configured.",
                }
            )
            continue

        for image_path, authority_record in zip(sample_images, authority_records, strict=False):
            if authority_record.get("status") != "ok":
                candidate_records[key].append(
                    {
                        "image_path": str(image_path),
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "status": "skipped_no_authority",
                        "estimated_cost_usd": 0.0,
                    }
                )
                continue

            planned_cost = float(
                _estimate_call_cost(catalog, candidate.provider, candidate.model, image_path)[
                    "estimated_cost_usd"
                ]
            )
            if live and spent_estimate + planned_cost > budget_cap_usd:
                candidate_records[key].append(
                    {
                        "image_path": str(image_path),
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "status": "skipped_budget_cap",
                        "estimated_cost_usd": 0.0,
                    }
                )
                continue

            started = time.perf_counter()
            try:
                if live:
                    response = await asyncio.wait_for(
                        _call_service(service, image_path, title),
                        timeout=per_call_timeout_seconds,
                    )
                else:
                    response = _mock_ocr_response(image_path, title, candidate.model)
            except Exception as exc:  # noqa: BLE001 - individual model failures are benchmark data.
                candidate_records[key].append(
                    {
                        "image_path": str(image_path),
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "status": "failed",
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "estimated_cost_usd": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            latency = time.perf_counter() - started
            cost = _estimate_call_cost(catalog, candidate.provider, candidate.model, image_path, response)
            spent_estimate += float(cost["estimated_cost_usd"])
            scores = _score_response(str(authority_record.get("response", "")), response)
            record = {
                "image_path": str(image_path),
                "provider": candidate.provider,
                "model": candidate.model,
                "status": "ok",
                "latency_seconds": round(latency, 6),
                **cost,
                **scores,
            }
            if include_raw:
                record["response"] = response
                record["authority_response"] = authority_record.get("response", "")
            candidate_records[key].append(record)

    authority_costs = [
        float(record.get("estimated_cost_usd", 0.0))
        for record in authority_records
        if record.get("status") == "ok"
    ]
    authority_latencies = [
        float(record.get("latency_seconds", 0.0))
        for record in authority_records
        if record.get("status") == "ok"
    ]
    candidate_summaries = [
        _candidate_summary(candidate, candidate_records[f"{candidate.provider}:{candidate.model}"], total_image_count=total_image_count)
        for candidate in CANDIDATE_MODELS
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "workspace": str(workspace),
        "live": live,
        "budget_cap_usd": budget_cap_usd,
        "per_call_timeout_seconds": per_call_timeout_seconds,
        "spent_estimate_usd": round(spent_estimate, 6),
        "preflight_sample_estimate_usd": round(preflight_cost, 6),
        "pricing_refreshed_at": catalog.refreshed_at,
        "sample_size_requested": sample_size,
        "sample_image_count": len(sample_images),
        "total_image_count": total_image_count,
        "authority": {
            "provider": "openai",
            "models_tried": list(AUTHORITY_MODELS),
            "successful_calls": len(authority_costs),
            "sample_cost_usd": round(sum(authority_costs), 6),
            "avg_cost_per_image_usd": round(_safe_mean(authority_costs), 8),
            "projected_full_corpus_cost_usd": round(_safe_mean(authority_costs) * total_image_count, 6),
            "latency_p50_seconds": round(_percentile(authority_latencies, 0.5), 6),
            "latency_p95_seconds": round(_percentile(authority_latencies, 0.95), 6),
        },
        "candidate_summaries": candidate_summaries,
        "candidate_records": candidate_records,
        "authority_records": [
            {
                key: value
                for key, value in record.items()
                if include_raw or key not in {"response"}
            }
            for record in authority_records
        ],
    }
    payload["receipt_path"] = str(receipt_path)
    write_json_atomic(receipt_path, payload)
    return payload


def run_benchmark(
    workspace: Path,
    **kwargs: Any,
) -> dict[str, object]:
    return asyncio.run(benchmark_ocr_models(workspace, **kwargs))


def format_summary(payload: dict[str, object]) -> str:
    lines = [
        "OCR model benchmark",
        f"Workspace: {payload['workspace']}",
        f"Mode: {'live' if payload['live'] else 'dry-run'}",
        f"Sample: {payload['sample_image_count']} of {payload['total_image_count']} images",
        f"Spent estimate: ${float(payload['spent_estimate_usd']):,.2f} / ${float(payload['budget_cap_usd']):,.2f}",
        f"Receipt: {payload['receipt_path']}",
        "",
        "Candidate summary:",
    ]
    summaries = payload.get("candidate_summaries", [])
    if isinstance(summaries, list):
        for item in sorted(
            summaries,
            key=lambda row: (
                -float(row.get("avg_similarity", 0.0)),
                float(row.get("avg_cost_per_image_usd", 0.0)),
            ),
        ):
            lines.append(
                "  {provider}/{model}: similarity {similarity:.3f}, "
                "cost ${cost:.6f}/image, p50 {p50:.2f}s, status {status}".format(
                    provider=item.get("provider"),
                    model=item.get("model"),
                    similarity=float(item.get("avg_similarity", 0.0)),
                    cost=float(item.get("avg_cost_per_image_usd", 0.0)),
                    p50=float(item.get("latency_p50_seconds", 0.0)),
                    status=item.get("status"),
                )
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m conference_pipeline.ocr_benchmark",
        description="Benchmark slide/poster OCR models against an OpenAI authority.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workspace", required=True, type=Path)
    run_parser.add_argument("--sample-size", type=int, default=12)
    run_parser.add_argument("--budget-cap-usd", type=float, default=30.0)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--title", default="Conference OCR benchmark")
    run_parser.add_argument("--pricing-catalog", type=Path)
    run_parser.add_argument("--per-call-timeout-seconds", type=float, default=60.0)
    run_parser.add_argument("--live", action="store_true")
    run_parser.add_argument("--include-raw", action="store_true")
    run_parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"Unknown command: {args.command}")
    if args.json_output:
        logging.disable(logging.WARNING)
    try:
        payload = run_benchmark(
            args.workspace,
            sample_size=args.sample_size,
            live=args.live,
            budget_cap_usd=args.budget_cap_usd,
            output_dir=args.output_dir,
            title=args.title,
            include_raw=args.include_raw,
            pricing_catalog_path=args.pricing_catalog,
            per_call_timeout_seconds=args.per_call_timeout_seconds,
        )
    finally:
        if args.json_output:
            logging.disable(logging.NOTSET)
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
