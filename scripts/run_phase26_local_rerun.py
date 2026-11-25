#!/usr/bin/env python3
"""Run the Phase 26 two-topic local-Qwen extraction rerun.

This is a receipt-backed wrapper around the repository's exhaustive extraction
model boundary for ICLR 2025 storage transcripts. It intentionally writes a new
Phase 26 output lane instead of mutating the existing Phase 14A heuristic
rollups in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402
from exhaustive_extraction.models import ExtractionRunner, estimate_tokens  # noqa: E402
from exhaustive_extraction.quote_extractor import extract_quote_from_anchors  # noqa: E402
from run_phase26_local_sample import (  # noqa: E402
    TOPICS,
    artifact_id_for,
    build_prompt,
    clean_text,
    compact_spaces,
    media_source_path,
    normalize_extractions,
    select_window,
    sha256_text,
)


PHASE_ID = "26-two-topic-corrected-extraction-and-research-rerun"
DEFAULT_RUN_ID = "v15-iclr2025-production"
DEFAULT_OUTPUT_LANE = "phase26-local-rerun"
TOPIC_IDS = ("02_model_architectures", "03_training_optimization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local-Qwen Phase 26 extraction rerun for topics 02/03."
    )
    parser.add_argument("--workspace", default="conferences/iclr2025")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-lane", default=DEFAULT_OUTPUT_LANE)
    parser.add_argument(
        "--source-set",
        choices=("pilot-topic-seeds", "all-transcripts"),
        default="pilot-topic-seeds",
        help="pilot-topic-seeds uses current 02/03 by-topic media seeds; all-transcripts enumerates every transcript.",
    )
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument(
        "--model",
        default=os.getenv("EXHAUSTIVE_EXTRACTION_MODEL", "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"),
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        default=10,
        help="Abort after this many failed model calls in one run. Use 0 to disable.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_media_ids(workspace: Path, run_id: str) -> list[str]:
    ids: set[str] = set()
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "by-topic"
    for topic_id in TOPIC_IDS:
        path = by_topic_dir / f"{topic_id}.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for item in payload.get("items", []):
            artifact_id = str(item.get("artifact_id") or "")
            match = re.search(r"media:([^:]+):", artifact_id)
            if match:
                ids.add(match.group(1))
    return sorted(ids)


def all_transcript_media_ids(workspace: Path) -> list[str]:
    media_dir = workspace / "storage" / "transcription-ocr" / "media"
    return sorted(path.parent.name for path in media_dir.glob("*/transcript.txt"))


def source_media_ids(args: argparse.Namespace, workspace: Path) -> list[str]:
    if args.sample_ids:
        ids = list(dict.fromkeys(args.sample_ids))
    elif args.source_set == "all-transcripts":
        ids = all_transcript_media_ids(workspace)
    else:
        ids = seed_media_ids(workspace, args.run_id)
    if args.offset:
        ids = ids[args.offset :]
    if args.limit:
        ids = ids[: args.limit]
    return ids


def existing_seed_topics(workspace: Path, run_id: str, media_id: str) -> list[str]:
    artifact_id = artifact_id_for(media_id)
    out: list[str] = []
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "by-topic"
    for topic_id in TOPIC_IDS:
        path = by_topic_dir / f"{topic_id}.json"
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        if any(item.get("artifact_id") == artifact_id for item in payload.get("items", [])):
            out.append(topic_id)
    return out


def content_hash(media_id: str, extraction: dict[str, Any], quote: str | None) -> str:
    topic_id = str(extraction.get("topic_id") or "")
    summary = str(extraction.get("summary") or "")
    basis = f"{media_id}\n{topic_id}\n{quote or summary}"
    return "sha256:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _flexible_anchor_pattern(anchor: str) -> str:
    escaped = re.escape(anchor.strip())
    return re.sub(r"\\ +", r"\\s+", escaped)


def _sentence_bounds_around(text: str, start: int, end: int, *, padding: int = 240) -> tuple[int, int]:
    left_floor = max(0, start - padding)
    right_ceiling = min(len(text), end + padding)
    left = max(text.rfind(".", left_floor, start), text.rfind("\n", left_floor, start))
    if left == -1:
        left = left_floor
    else:
        left += 1
    stops = [pos for pos in (text.find(".", end, right_ceiling), text.find("\n", end, right_ceiling)) if pos != -1]
    right = min(stops) + 1 if stops else right_ceiling
    return left, right


def fallback_quote_from_anchors(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    *,
    max_quote_length: int = 1200,
) -> tuple[str | None, str | None]:
    """Conservative fallback for collapsed/overlapping anchors.

    The shared quote extractor expects start and end anchors to both appear in
    order. Models often return the same short phrase for both. In that case,
    pull the sentence around the exact phrase rather than discarding otherwise
    useful evidence.
    """

    if not source_text or not start_anchor:
        return None, "missing_start_anchor"
    start_pattern = _flexible_anchor_pattern(start_anchor)
    start_match = re.search(start_pattern, source_text, re.IGNORECASE | re.DOTALL)
    if not start_match:
        return None, "start_anchor_not_found"

    end_anchor = end_anchor or start_anchor
    anchors_overlap = (
        start_anchor.strip().lower() == end_anchor.strip().lower()
        or start_anchor.strip().lower() in end_anchor.strip().lower()
        or end_anchor.strip().lower() in start_anchor.strip().lower()
    )
    if anchors_overlap:
        left, right = _sentence_bounds_around(source_text, start_match.start(), start_match.end())
        quote = source_text[left:right].strip()
        if quote and len(quote) <= max_quote_length:
            return quote, None
        return None, "collapsed_anchor_sentence_too_long"

    end_pattern = _flexible_anchor_pattern(end_anchor)
    end_match = re.search(
        end_pattern,
        source_text[start_match.start() :],
        re.IGNORECASE | re.DOTALL,
    )
    if not end_match:
        return None, "end_anchor_not_found"
    absolute_end = start_match.start() + end_match.end()
    quote = source_text[start_match.start() : absolute_end].strip()
    if not quote:
        return None, "empty_quote"
    if len(quote) > max_quote_length:
        return None, "quote_too_long"
    return quote, None


def validate_extraction_quotes(
    media_id: str,
    sample_text: str,
    raw_extractions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, extraction in enumerate(raw_extractions, start=1):
        item = dict(extraction)
        topic_id = str(item.get("topic_id") or "")
        if topic_id not in TOPICS:
            item["qa"] = {
                "valid": False,
                "reason": f"topic_id_not_in_phase_scope:{topic_id}",
            }
            validated.append(item)
            continue

        start_anchor = str(item.get("quote_start_anchor") or "")
        end_anchor = str(item.get("quote_end_anchor") or "")
        quote = None
        qa: dict[str, Any]
        if start_anchor and end_anchor:
            quote_result = extract_quote_from_anchors(
                sample_text,
                start_anchor,
                end_anchor,
                max_quote_length=5000,
            )
            if quote_result.success:
                quote = quote_result.verbatim_quote
                qa = {
                    "valid": True,
                    "quote_method": quote_result.method,
                    "quote_chars": len(quote or ""),
                }
            else:
                fallback_quote, fallback_error = fallback_quote_from_anchors(
                    sample_text,
                    start_anchor,
                    end_anchor,
                )
                if fallback_quote:
                    quote = fallback_quote
                    qa = {
                        "valid": True,
                        "quote_method": "collapsed_anchor_sentence",
                        "quote_chars": len(quote),
                        "primary_error": quote_result.error,
                    }
                else:
                    qa = {
                        "valid": False,
                        "reason": fallback_error or quote_result.error or "anchor_match_failed",
                        "primary_error": quote_result.error,
                    }
        else:
            qa = {
                "valid": False,
                "reason": "missing_quote_anchors",
            }

        item["topic_name"] = item.get("topic_name") or TOPICS[topic_id]["title"]
        item["verbatim_quote"] = quote
        item["qa"] = qa
        item["content_hash"] = content_hash(media_id, item, quote)
        item["extraction_id"] = (
            f"{media_id}:{topic_id}:{item['content_hash'].split(':', 1)[1][:12]}:{index}"
        )
        validated.append(item)
    return validated


def record_path_for(by_artifact_dir: Path, media_id: str) -> Path:
    return by_artifact_dir / f"{safe_filename(artifact_id_for(media_id))}.json"


def can_resume(path: Path, transcript_digest: str, retry_failed: bool) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    if payload.get("transcript_digest") != transcript_digest:
        return False
    if retry_failed and payload.get("status") != "succeeded":
        return False
    return payload.get("status") in {"succeeded", "failed"}


def aggregate_by_topic(by_artifact_dir: Path, by_topic_dir: Path, extraction_root: Path) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = {topic_id: [] for topic_id in TOPIC_IDS}
    for path in sorted(by_artifact_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "succeeded":
            continue
        by_topic_counts: dict[str, int] = {}
        valid_extractions = [
            item
            for item in payload.get("extractions", [])
            if item.get("topic_id") in TOPIC_IDS and item.get("qa", {}).get("valid")
        ]
        for extraction in valid_extractions:
            topic_id = str(extraction.get("topic_id"))
            by_topic_counts[topic_id] = by_topic_counts.get(topic_id, 0) + 1
        for topic_id, count in by_topic_counts.items():
            topic_extractions = [
                item for item in valid_extractions if item.get("topic_id") == topic_id
            ]
            grouped[topic_id].append(
                {
                    "artifact_id": payload.get("artifact_id"),
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "source_artifact_path": payload.get("source_artifact_path"),
                    "source_artifact_digest": payload.get("source_artifact_digest"),
                    "transcript_path": payload.get("transcript_path"),
                    "transcript_digest": payload.get("transcript_digest"),
                    "record_path": rel(path, extraction_root),
                    "confidence": max(
                        float(item.get("relevance_score") or 0.0)
                        for item in topic_extractions
                    ),
                    "extraction_method": payload.get("extraction_method"),
                    "model_used": payload.get("model_used"),
                    "extraction_count": count,
                    "extractions": topic_extractions,
                }
            )
    counts: dict[str, int] = {}
    by_topic_dir.mkdir(parents=True, exist_ok=True)
    for topic_id, items in grouped.items():
        payload = {
            "schema_version": 1,
            "phase": PHASE_ID,
            "topic_id": topic_id,
            "topic_name": TOPICS[topic_id]["title"],
            "artifact_count": len(items),
            "extraction_count": sum(int(item["extraction_count"]) for item in items),
            "items": items,
        }
        write_json_atomic(by_topic_dir / f"{topic_id}.json", payload)
        counts[topic_id] = len(items)
    return counts


def write_review_markdown(path: Path, receipt: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines: list[str] = [
        "# Phase 26 Local Qwen Rerun",
        "",
        f"- Created: `{receipt['created_at']}`",
        f"- Run id: `{receipt['run_id']}`",
        f"- Output lane: `{receipt['output_lane']}`",
        f"- Model: `{receipt['model']}`",
        f"- Source set: `{receipt['source_set']}`",
        f"- Inputs selected: `{receipt['selected_count']}`",
        f"- Processed this run: `{receipt['totals']['processed_this_run']}`",
        f"- Resumed: `{receipt['totals']['resumed']}`",
        f"- Failed this run: `{receipt['totals']['failed_this_run']}`",
        f"- Valid extractions: `{receipt['totals']['valid_extractions']}`",
        f"- Wall seconds: `{receipt['totals']['wall_seconds']:.2f}`",
        "",
        "## Timing",
        "",
        "| Media | Status | Seconds | Input Tokens | Output Tokens | Valid Extractions | Seed Topics |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        seed_topics = ", ".join(record.get("existing_topic_seeds") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("media_id")),
                    str(record.get("status")),
                    f"{float(record.get('wall_seconds') or 0):.2f}",
                    str(record.get("input_tokens") or 0),
                    str(record.get("output_tokens") or 0),
                    str(record.get("valid_extraction_count") or 0),
                    seed_topics,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Extractions")
    lines.append("")
    for record in records:
        lines.append(f"### {record.get('media_id')}")
        lines.append("")
        lines.append(f"- Status: `{record.get('status')}`")
        lines.append(f"- Transcript: `{record.get('transcript_path')}`")
        lines.append(f"- Source artifact: `{record.get('source_artifact_path')}`")
        lines.append(f"- Window chars: `{record.get('char_start')}`-`{record.get('char_end')}`")
        lines.append("")
        if record.get("status") != "succeeded":
            lines.append(f"Error: `{record.get('error_type')}: {record.get('error')}`")
            lines.append("")
            continue
        valid_items = [
            item
            for item in record.get("extractions", [])
            if item.get("qa", {}).get("valid")
        ]
        if not valid_items:
            lines.append("_No valid Phase 26 extraction passed anchor QA._")
            lines.append("")
            continue
        for index, item in enumerate(valid_items, start=1):
            lines.append(f"{index}. `{item.get('topic_id')}` - {item.get('topic_name')}")
            lines.append(f"   - Score: `{item.get('relevance_score')}`")
            lines.append(f"   - Summary: {item.get('summary')}")
            quote = " ".join(str(item.get("verbatim_quote") or "").split())
            if quote:
                lines.append(f"   - Quote: \"{quote[:500]}\"")
            lines.append(f"   - Extraction id: `{item.get('extraction_id')}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = (REPO_ROOT / args.workspace).resolve()
    selected_ids = source_media_ids(args, workspace)

    extraction_root = (
        workspace
        / "storage"
        / "extractions"
        / args.run_id
        / args.output_lane
    )
    by_artifact_dir = extraction_root / "by-artifact"
    by_topic_dir = extraction_root / "by-topic"
    by_artifact_dir.mkdir(parents=True, exist_ok=True)
    by_topic_dir.mkdir(parents=True, exist_ok=True)

    workflow_dir = workspace / "storage" / "workflow-runs" / args.run_id / args.output_lane
    workflow_dir.mkdir(parents=True, exist_ok=True)
    progress_path = workflow_dir / "progress.json"

    os.environ["EXHAUSTIVE_EXTRACTION_PROVIDER"] = "local_mlx"
    os.environ["EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES"] = "0"
    os.environ["EXHAUSTIVE_EXTRACTION_MODEL"] = args.model
    os.environ["EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)

    runner = ExtractionRunner()
    created_at = utc_now()
    started = time.perf_counter()
    run_records: list[dict[str, Any]] = []
    failures_this_run = 0

    for index, media_id in enumerate(selected_ids, start=1):
        transcript_path = workspace / "storage" / "transcription-ocr" / "media" / media_id / "transcript.txt"
        if not transcript_path.exists():
            print(f"[{index}/{len(selected_ids)}] {media_id}: missing transcript", flush=True)
            continue

        full_text = clean_text(transcript_path.read_text(encoding="utf-8", errors="replace"))
        transcript_digest = sha256_text(full_text)
        out_path = record_path_for(by_artifact_dir, media_id)
        if can_resume(out_path, transcript_digest, args.retry_failed):
            previous = load_json(out_path)
            run_records.append(previous)
            print(f"[{index}/{len(selected_ids)}] {media_id}: resumed", flush=True)
            continue

        sample_text, char_start, char_end, window_policy, keyword_score = select_window(
            full_text,
            args.max_chars,
        )
        prompt = build_prompt(media_id, sample_text, char_start, char_end)
        estimated_prompt_tokens = estimate_tokens(prompt)
        print(
            f"[{index}/{len(selected_ids)}] {media_id}: local Qwen, chars={len(sample_text)}, est_tokens={estimated_prompt_tokens}",
            flush=True,
        )

        record: dict[str, Any] = {
            "schema_version": 1,
            "phase": PHASE_ID,
            "run_id": args.run_id,
            "output_lane": args.output_lane,
            "media_id": media_id,
            "artifact_id": artifact_id_for(media_id),
            "source_artifact_id": artifact_id_for(media_id),
            "source_artifact_path": media_source_path(media_id),
            "source_artifact_digest": transcript_digest,
            "transcript_path": rel(transcript_path, workspace),
            "transcript_digest": transcript_digest,
            "sample_digest": sha256_text(sample_text),
            "full_transcript_chars": len(full_text),
            "sample_chars": len(sample_text),
            "char_start": char_start,
            "char_end": char_end,
            "window_policy": window_policy,
            "window_keyword_score": keyword_score,
            "existing_topic_seeds": existing_seed_topics(workspace, args.run_id, media_id),
            "extraction_method": "local_qwen_mlx_windowed",
            "model_used": args.model,
            "provider": "local",
            "generated_at": utc_now(),
        }

        sample_started = time.perf_counter()
        try:
            with provider_receipt_context(
                workspace_root=workspace,
                run_id=args.run_id,
                stage_id=args.output_lane,
                caller=__file__,
            ):
                parsed, model_used, input_tokens, output_tokens = runner.run_extraction(
                    prompt,
                    estimated_tokens=estimated_prompt_tokens,
                )
            raw_extractions = normalize_extractions(parsed)
            extractions = validate_extraction_quotes(media_id, sample_text, raw_extractions)
            valid_count = sum(
                1
                for item in extractions
                if item.get("topic_id") in TOPIC_IDS and item.get("qa", {}).get("valid")
            )
            wall_seconds = time.perf_counter() - sample_started
            record.update(
                {
                    "status": "succeeded",
                    "model_used": model_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "wall_seconds": wall_seconds,
                    "raw_extraction_count": len(raw_extractions),
                    "valid_extraction_count": valid_count,
                    "topic_ids": sorted(
                        {
                            str(item.get("topic_id"))
                            for item in extractions
                            if item.get("topic_id") in TOPIC_IDS
                            and item.get("qa", {}).get("valid")
                        }
                    ),
                    "extractions": extractions,
                    "parsed": parsed,
                }
            )
            print(
                f"[{index}/{len(selected_ids)}] {media_id}: ok {wall_seconds:.2f}s valid={valid_count}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - phase receipt captures model failures
            failures_this_run += 1
            wall_seconds = time.perf_counter() - sample_started
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": compact_spaces(str(exc))[:1000],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "wall_seconds": wall_seconds,
                    "raw_extraction_count": 0,
                    "valid_extraction_count": 0,
                    "topic_ids": [],
                    "extractions": [],
                }
            )
            print(
                f"[{index}/{len(selected_ids)}] {media_id}: failed {type(exc).__name__}: {exc}",
                flush=True,
            )
        write_json_atomic(out_path, record)
        run_records.append(record)

        progress = {
            "schema_version": 1,
            "phase": PHASE_ID,
            "run_id": args.run_id,
            "output_lane": args.output_lane,
            "updated_at": utc_now(),
            "selected_count": len(selected_ids),
            "last_index": index,
            "last_media_id": media_id,
            "processed_records": len(list(by_artifact_dir.glob("*.json"))),
            "failures_this_run": failures_this_run,
        }
        write_json_atomic(progress_path, progress)
        if args.stop_after_failures and failures_this_run >= args.stop_after_failures:
            print(f"Stopping after {failures_this_run} failures", flush=True)
            break

    by_topic_counts = aggregate_by_topic(by_artifact_dir, by_topic_dir, extraction_root)
    all_records = []
    for path in sorted(by_artifact_dir.glob("*.json")):
        try:
            all_records.append(load_json(path))
        except json.JSONDecodeError:
            continue

    total_wall = time.perf_counter() - started
    succeeded = sum(1 for record in all_records if record.get("status") == "succeeded")
    failed = sum(1 for record in all_records if record.get("status") == "failed")
    resumed = sum(1 for record in run_records if record.get("status") in {"succeeded", "failed"}) - (
        len(run_records) - failures_this_run
    )
    receipt = {
        "schema_version": 1,
        "phase": PHASE_ID,
        "run_id": args.run_id,
        "output_lane": args.output_lane,
        "created_at": created_at,
        "completed_at": utc_now(),
        "workspace": rel(workspace, REPO_ROOT),
        "source_set": args.source_set,
        "selected_count": len(selected_ids),
        "model": args.model,
        "provider": "local",
        "provider_override": "local_mlx",
        "max_chars": args.max_chars,
        "max_output_tokens": args.max_output_tokens,
        "mutates_phase14a_outputs": False,
        "extraction_root": rel(extraction_root, workspace),
        "by_artifact_dir": rel(by_artifact_dir, workspace),
        "by_topic_dir": rel(by_topic_dir, workspace),
        "provider_receipts_dir": rel(
            workspace / "storage" / "workflow-runs" / args.run_id / "provider-calls" / args.output_lane,
            workspace,
        ),
        "by_topic_counts": by_topic_counts,
        "totals": {
            "records_total": len(all_records),
            "succeeded_total": succeeded,
            "failed_total": failed,
            "processed_this_run": sum(
                1 for record in run_records if record.get("generated_at", "") >= created_at
            ),
            "resumed": sum(1 for record in run_records if record.get("generated_at", "") < created_at),
            "failed_this_run": failures_this_run,
            "valid_extractions": sum(int(record.get("valid_extraction_count") or 0) for record in all_records),
            "input_tokens": sum(int(record.get("input_tokens") or 0) for record in all_records),
            "output_tokens": sum(int(record.get("output_tokens") or 0) for record in all_records),
            "wall_seconds": total_wall,
        },
    }
    receipt_path = workflow_dir / "phase26-local-rerun-receipt.json"
    md_path = workflow_dir / "phase26-local-rerun-review.md"
    write_json_atomic(receipt_path, receipt)
    write_review_markdown(md_path, receipt, all_records)
    print(f"Wrote {rel(receipt_path, REPO_ROOT)}", flush=True)
    print(f"Wrote {rel(md_path, REPO_ROOT)}", flush=True)
    return 0 if failures_this_run == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
