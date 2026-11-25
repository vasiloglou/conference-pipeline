#!/usr/bin/env python3
"""Run a bounded Phase 26 local-model extraction sample for ICLR 2025.

This is intentionally a pilot harness, not the production exhaustive runner. It
feeds selected transcript windows through the same local MLX model/parser path
used by the exhaustive extraction subsystem, then writes reviewable timing and
extraction evidence without mutating production by-topic/by-artifact files.
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

from exhaustive_extraction.models import ExtractionRunner, estimate_tokens  # noqa: E402
from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402


PHASE_ID = "26-two-topic-corrected-extraction-and-research-rerun"
DEFAULT_RUN_ID = "v15-iclr2025-production"
DEFAULT_SAMPLE_IDS = (
    "oral-31735",
    "poster-27711",
    "poster-28824",
    "workshop-23963",
)
TOPICS = {
    "02_model_architectures": {
        "title": "Model Architectures",
        "scope": "architecture trends and model families; transformers, attention, layers, graph models, diffusion, sparse or mixture models, and neural network design choices",
        "keywords": (
            "architecture",
            "architectures",
            "transformer",
            "attention",
            "network",
            "model",
            "models",
            "graph",
            "diffusion",
            "layer",
            "layers",
            "mixture",
            "sparse",
        ),
    },
    "03_training_optimization": {
        "title": "Training And Optimization",
        "scope": "training recipes, optimization behavior, scaling behavior, optimizers, learning rate, gradients, pretraining, fine-tuning, and regularization",
        "keywords": (
            "training",
            "train",
            "optimization",
            "optimizer",
            "learning rate",
            "gradient",
            "gradients",
            "pretraining",
            "pre-training",
            "fine-tuning",
            "finetuning",
            "regularization",
            "scaling",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small local MLX Phase 26 extraction sample."
    )
    parser.add_argument(
        "--workspace",
        default="conferences/iclr2025",
        help="Conference workspace path.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Workflow run id used for output placement.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Media ids under storage/transcription-ocr/media to sample.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum transcript characters sent per sample.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1500,
        help="Maximum local model output tokens per sample.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EXHAUSTIVE_EXTRACTION_MODEL", ""),
        help="Optional MLX model id. Defaults to exhaustive_extraction.models local default.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults under storage/workflow-runs/<run-id>/phase26-local-sample.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return "".join(
        char if char in "\n\r\t" or ord(char) >= 32 else " "
        for char in text
    )


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def media_source_path(media_id: str) -> str:
    return f"cache/acquisition-downloads/media/{media_id}/audio/raw.m4a"


def artifact_id_for(media_id: str) -> str:
    return f"media:{media_id}:REDACTED.m4a"


def existing_topic_seeds(workspace: Path, run_id: str, media_id: str) -> list[str]:
    artifact_id = artifact_id_for(media_id)
    result: list[str] = []
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "by-topic"
    for topic_id in TOPICS:
        path = by_topic_dir / f"{topic_id}.json"
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        for item in payload.get("items", []):
            if item.get("artifact_id") == artifact_id:
                result.append(topic_id)
                break
    return result


def score_window(text: str) -> int:
    lowered = text.lower()
    score = 0
    for topic in TOPICS.values():
        for keyword in topic["keywords"]:
            score += lowered.count(keyword)
    return score


def select_window(text: str, max_chars: int) -> tuple[str, int, int, str, int]:
    if len(text) <= max_chars:
        return text, 0, len(text), "full_transcript", score_window(text)

    stride = max(1, max_chars // 2)
    best_start = 0
    best_score = -1
    for start in range(0, len(text), stride):
        window = text[start : start + max_chars]
        if not window:
            continue
        score = score_window(window)
        if score > best_score:
            best_score = score
            best_start = start
        if start + max_chars >= len(text):
            break

    best_end = min(len(text), best_start + max_chars)
    window = text[best_start:best_end]
    return window, best_start, best_end, "keyword_scored_window", best_score


def build_prompt(media_id: str, sample_text: str, char_start: int, char_end: int) -> str:
    topic_lines = "\n".join(
        f"- {topic_id}: {topic['title']} - {topic['scope']}"
        for topic_id, topic in TOPICS.items()
    )
    return f"""Analyze this ICLR 2025 transcript excerpt for the Phase 26 pilot.

SOURCE:
- media_id: {media_id}
- artifact_id: {artifact_id_for(media_id)}
- source_artifact_path: {media_source_path(media_id)}
- transcript_window_chars: {char_start}-{char_end}

PHASE 26 PILOT TOPICS:
{topic_lines}

TRANSCRIPT EXCERPT:
{sample_text}

RULES:
1. Use only the transcript excerpt. Do not infer from outside knowledge.
2. Extract only evidence relevant to the two pilot topics above.
3. If there is no clear evidence, return {{"extractions": []}}.
4. For each extraction, provide quote_start_anchor and quote_end_anchor copied from the excerpt, 5 to 12 words each.
5. Keep summaries factual and specific. Prefer 1 to 4 extractions total.
6. Return only valid JSON.

Return JSON with this shape:
{{
  "extractions": [
    {{
      "topic_id": "02_model_architectures",
      "topic_name": "Model Architectures",
      "quote_start_anchor": "first words copied from transcript",
      "quote_end_anchor": "last words copied from transcript",
      "extraction_type": "verbatim_quote",
      "summary": "specific factual summary",
      "relevance_score": 0.0,
      "transcript_quality": "high|medium|low"
    }}
  ]
}}"""


def normalize_extractions(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("extractions", [])
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def markdown_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase 26 Local Model Sample")
    lines.append("")
    lines.append(f"- Created: `{payload['created_at']}`")
    lines.append(f"- Run id: `{payload['run_id']}`")
    lines.append(f"- Model requested: `{payload['model_requested']}`")
    lines.append(f"- Provider override: `{payload['provider_override']}`")
    lines.append(f"- Max transcript chars per sample: `{payload['sample_policy']['max_chars']}`")
    lines.append(f"- Max output tokens per sample: `{payload['sample_policy']['max_output_tokens']}`")
    lines.append("")
    lines.append("## Timing Summary")
    lines.append("")
    lines.append("| Sample | Status | Model | Transcript chars | Prompt tokens | Output tokens | Seconds | Extractions |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for sample in payload["samples"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(sample["media_id"]),
                    markdown_escape(sample["status"]),
                    markdown_escape(sample.get("model_used", "")),
                    str(sample.get("sample_chars", 0)),
                    str(sample.get("input_tokens", 0)),
                    str(sample.get("output_tokens", 0)),
                    f"{sample.get('wall_seconds', 0):.2f}",
                    str(sample.get("extraction_count", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    totals = payload["totals"]
    lines.append(
        f"Total wall time: `{totals['wall_seconds']:.2f}s`; "
        f"succeeded: `{totals['succeeded']}`; failed: `{totals['failed']}`; "
        f"extractions: `{totals['extractions']}`."
    )
    lines.append("")

    for sample in payload["samples"]:
        lines.append(f"## {sample['media_id']}")
        lines.append("")
        lines.append(f"- Status: `{sample['status']}`")
        lines.append(f"- Artifact id: `{sample['artifact_id']}`")
        lines.append(f"- Source artifact path: `{sample['source_artifact_path']}`")
        lines.append(f"- Transcript path: `{sample['transcript_path']}`")
        lines.append(f"- Transcript digest: `{sample['transcript_digest']}`")
        lines.append(f"- Window: chars `{sample['char_start']}`-`{sample['char_end']}` via `{sample['window_policy']}`")
        lines.append(f"- Existing heuristic pilot-topic seeds: `{', '.join(sample['existing_topic_seeds']) or 'none'}`")
        lines.append(f"- Window keyword score: `{sample['window_keyword_score']}`")
        lines.append(f"- Wall seconds: `{sample.get('wall_seconds', 0):.2f}`")
        if sample["status"] != "succeeded":
            lines.append(f"- Error: `{sample.get('error_type')}: {sample.get('error')}`")
            lines.append("")
            continue
        lines.append("")
        lines.append("### Local Model Extractions")
        lines.append("")
        if not sample["extractions"]:
            lines.append("_No pilot-topic extraction returned by the local model._")
        for index, extraction in enumerate(sample["extractions"], start=1):
            lines.append(f"{index}. `{extraction.get('topic_id', '')}` - {extraction.get('topic_name', '')}")
            lines.append(f"   - Score: `{extraction.get('relevance_score', '')}`")
            lines.append(f"   - Summary: {extraction.get('summary', '')}")
            lines.append(f"   - Start anchor: `{extraction.get('quote_start_anchor', '')}`")
            lines.append(f"   - End anchor: `{extraction.get('quote_end_anchor', '')}`")
        lines.append("")
        lines.append("### Transcript Window")
        lines.append("")
        lines.append("```text")
        lines.append(sample["sample_text"])
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = (REPO_ROOT / args.workspace).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace
        / "storage"
        / "workflow-runs"
        / args.run_id
        / "phase26-local-sample"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXHAUSTIVE_EXTRACTION_PROVIDER"] = "local_mlx"
    os.environ["EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES"] = "0"
    os.environ["EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    if args.model:
        os.environ["EXHAUSTIVE_EXTRACTION_MODEL"] = args.model

    runner = ExtractionRunner()
    created_at = utc_now()
    model_requested = os.getenv("EXHAUSTIVE_EXTRACTION_MODEL", "") or "mlx default"
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, media_id in enumerate(args.sample_ids, start=1):
        transcript_path = workspace / "storage" / "transcription-ocr" / "media" / media_id / "transcript.txt"
        print(f"[{index}/{len(args.sample_ids)}] {media_id}: reading {transcript_path.relative_to(REPO_ROOT)}", flush=True)
        sample_record: dict[str, Any] = {
            "media_id": media_id,
            "artifact_id": artifact_id_for(media_id),
            "source_artifact_path": media_source_path(media_id),
            "transcript_path": str(transcript_path.relative_to(workspace)),
            "existing_topic_seeds": existing_topic_seeds(workspace, args.run_id, media_id),
        }
        if not transcript_path.exists():
            sample_record.update(
                {
                    "status": "failed",
                    "error_type": "FileNotFoundError",
                    "error": str(transcript_path),
                    "wall_seconds": 0.0,
                    "sample_chars": 0,
                    "extraction_count": 0,
                }
            )
            samples.append(sample_record)
            continue

        full_text = clean_text(transcript_path.read_text(encoding="utf-8", errors="replace"))
        sample_text, char_start, char_end, window_policy, keyword_score = select_window(
            full_text,
            args.max_chars,
        )
        prompt = build_prompt(media_id, sample_text, char_start, char_end)
        estimated_prompt_tokens = estimate_tokens(prompt)
        sample_record.update(
            {
                "status": "running",
                "transcript_digest": sha256_text(full_text),
                "sample_digest": sha256_text(sample_text),
                "full_transcript_chars": len(full_text),
                "sample_chars": len(sample_text),
                "char_start": char_start,
                "char_end": char_end,
                "window_policy": window_policy,
                "window_keyword_score": keyword_score,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "sample_text": sample_text,
            }
        )

        print(
            f"[{index}/{len(args.sample_ids)}] {media_id}: local MLX prompt tokens est {estimated_prompt_tokens}, chars {len(sample_text)}",
            flush=True,
        )
        sample_started = time.perf_counter()
        try:
            with provider_receipt_context(
                workspace_root=workspace,
                run_id=args.run_id,
                stage_id="phase26-local-sample",
                caller=__file__,
            ):
                parsed, model_used, input_tokens, output_tokens = runner.run_extraction(
                    prompt,
                    estimated_tokens=estimated_prompt_tokens,
                )
            extractions = normalize_extractions(parsed)
            wall_seconds = time.perf_counter() - sample_started
            sample_record.update(
                {
                    "status": "succeeded",
                    "model_used": model_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "wall_seconds": wall_seconds,
                    "parsed": parsed,
                    "extractions": extractions,
                    "extraction_count": len(extractions),
                }
            )
            print(
                f"[{index}/{len(args.sample_ids)}] {media_id}: succeeded in {wall_seconds:.2f}s with {len(extractions)} extractions",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - pilot receipt should capture failures
            wall_seconds = time.perf_counter() - sample_started
            sample_record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": compact_spaces(str(exc))[:1000],
                    "wall_seconds": wall_seconds,
                    "model_used": os.getenv("EXHAUSTIVE_EXTRACTION_MODEL", model_requested),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "extractions": [],
                    "extraction_count": 0,
                }
            )
            print(
                f"[{index}/{len(args.sample_ids)}] {media_id}: failed in {wall_seconds:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
        samples.append(sample_record)

    total_wall = time.perf_counter() - started
    succeeded = sum(1 for sample in samples if sample.get("status") == "succeeded")
    failed = sum(1 for sample in samples if sample.get("status") == "failed")
    payload = {
        "schema_version": 1,
        "phase": PHASE_ID,
        "run_id": args.run_id,
        "created_at": created_at,
        "provider_override": "local_mlx",
        "model_requested": model_requested,
        "workspace": str(workspace.relative_to(REPO_ROOT)),
        "sample_policy": {
            "sample_ids": list(args.sample_ids),
            "max_chars": args.max_chars,
            "max_output_tokens": args.max_output_tokens,
            "window_selection": "full transcript when short; otherwise highest keyword-scored bounded window",
            "topics": TOPICS,
            "mutates_production_extractions": False,
        },
        "totals": {
            "samples": len(samples),
            "succeeded": succeeded,
            "failed": failed,
            "wall_seconds": total_wall,
            "extractions": sum(int(sample.get("extraction_count") or 0) for sample in samples),
            "input_tokens": sum(int(sample.get("input_tokens") or 0) for sample in samples),
            "output_tokens": sum(int(sample.get("output_tokens") or 0) for sample in samples),
        },
        "samples": samples,
    }

    json_path = output_dir / "phase26-local-sample-results.json"
    md_path = output_dir / "phase26-local-sample-results.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
