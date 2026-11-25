#!/usr/bin/env python3
"""Parallel bulk ASR for the r6 acquired public-YouTube audio.

Why this exists alongside ``run_cvpr2026_r6_transcription_ocr.py --mode full``:
the governed r6 owner serializes every paid provider call through an atomic
reservation ledger (``reserve_r6_*`` refuses a second reservation while any is
unsettled). That makes the cost cap unbreachable under concurrency, but forces
521 files through one-at-a-time dispatch — and it is fail-fast (a single
transient API error ``break``s the whole run). At ~80 s/file that is ~11.5 h of
wall time and fragile.

This driver keeps the governance that matters and drops the part that only
exists to make concurrency cap-safe:

  * Model is registry-governed (``transcription.session_speech`` -> the same
    model the estimate priced); never hardcoded.
  * The cap is enforced UP FRONT: existing canonical spend + the full declared
    estimate must fit under both the $200 speech task cap and the $800 all-in
    cap before a single call is made. Transcription cost is deterministic per
    audio-minute, so if the whole batch fits, every subset fits — the serial
    per-call reservation dance buys nothing here.
  * Every call auto-writes a per-call provider receipt into the canonical
    ``storage/provider-calls`` spend log (via one batch-wide
    ``provider_receipt_context``), so real spend stays fully accounted and
    visible to every downstream cap check.
  * Per-item error isolation + bounded retry/backoff + resume: one flaky item
    never aborts the batch, and re-running skips already-transcribed items.

It writes transcript sidecars to the governed r6 speech-text directory with
full provenance, and a summary receipt. It is honest about being a parallel
ASR driver rather than the serial ledger-bound owner.

Only the ``acquired`` (``direct_public_youtube``) rows of the succeeded r6
audio-recovery receipt are eligible; ``download_failed`` rows are accepted
coverage gaps and are never fabricated into transcripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.config import load_settings
from conference_pipeline.costs.budget_enforcement import _existing_spend
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.transcription import create_transcription_service

TASK_ID = "transcription.session_speech"
SPEECH_CAP_USD = 200.0
ALL_IN_CAP_USD = 800.0
STAGE_ID = "transcription-ocr-r6"

# gpt-4o-transcribe-diarize rejects audio longer than 1400 s. Split anything
# over the limit into safely-under-limit segments, transcribe each, and join.
MODEL_MAX_AUDIO_SECONDS = 1400.0
SEGMENT_SECONDS = 1200


async def _transcribe_long_audio(service: Any, audio_path: Path) -> str:
    """Split an over-limit audio into segments, transcribe each, join the text.

    Re-encodes to clean AAC/m4a segments (not stream-copy) so a fragmented
    source container never trips the upload, and so each segment carries an
    OpenAI-supported extension directly.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to split over-length audio but was not found")
    tmpdir = Path(tempfile.mkdtemp(prefix="r6-long-"))
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, "-nostdin", "-v", "error", "-i", str(audio_path),
            "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
            "-c:a", "aac", "-b:a", "128k",
            str(tmpdir / "seg_%03d.m4a"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg segment failed: {err.decode('utf-8', 'replace')[:200]}")
        segments = sorted(tmpdir.glob("seg_*.m4a"))
        if not segments:
            raise RuntimeError("ffmpeg produced no segments for over-length audio")
        parts: list[str] = []
        for seg in segments:
            chunk = await service.transcribe_single(seg)
            text = chunk.text or ""
            if text.startswith("[Transcription error"):
                raise RuntimeError(text)
            parts.append(text)
        return "\n\n".join(parts)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _safe_name(item_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in item_id)


def _load_acquired_records(contract: R6ArtifactContract) -> list[dict[str, Any]]:
    receipt = json.loads(contract.path("audio_recovery_receipt").read_bytes())
    if receipt.get("status") != "succeeded":
        raise SystemExit("audio recovery receipt is not succeeded; refusing to transcribe")
    rows = [
        row
        for row in receipt.get("records", [])
        if row.get("terminal_state") == "acquired" and row.get("local_media_path")
    ]
    if not rows:
        raise SystemExit("no acquired audio rows to transcribe")
    return rows


def _load_estimate_total(contract: R6ArtifactContract) -> float:
    est = json.loads(contract.path("cost_estimate_receipt").read_bytes())
    return float(est["estimate"]["estimated_cost_usd"])


def _existing_transcript(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("text")) and payload.get("transcript_source") == "asr_fallback"


async def _transcribe_one(
    *,
    service: Any,
    row: dict[str, Any],
    workspace: Path,
    out_dir: Path,
    model: str,
    run_id: str,
    sem: asyncio.Semaphore,
    max_attempts: int,
    progress: dict[str, int],
    total: int,
) -> dict[str, Any]:
    item_id = row["item_id"]
    out_path = out_dir / f"{_safe_name(item_id)}.json"
    if _existing_transcript(out_path):
        progress["skipped"] += 1
        return {"item_id": item_id, "status": "skipped"}

    audio_path = workspace / row["local_media_path"] if not Path(row["local_media_path"]).is_absolute() else Path(row["local_media_path"])
    duration_s = row.get("duration_seconds") or 0
    async with sem:
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                if duration_s > MODEL_MAX_AUDIO_SECONDS:
                    text = await _transcribe_long_audio(service, audio_path)
                else:
                    chunk = await service.transcribe_single(audio_path)
                    text = chunk.text or ""
                    if text.startswith("[Transcription error"):
                        raise RuntimeError(text)
                payload = {
                    "item_id": item_id,
                    "video_id": row.get("video_id"),
                    "source_url": row.get("source_url"),
                    "transcript_source": "asr_fallback",
                    "provider": "openai",
                    "model": model,
                    "audio_sha256": row.get("sha256"),
                    "audio_duration_seconds": row.get("duration_seconds"),
                    "char_count": len(text),
                    "text": text,
                    "run_id": run_id,
                }
                tmp = out_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(out_path)
                progress["done"] += 1
                n = progress["done"] + progress["skipped"] + progress["failed"]
                print(
                    f"[{n}/{total}] ok {item_id} ({len(text)} chars, attempt {attempt})",
                    flush=True,
                )
                return {"item_id": item_id, "status": "succeeded", "char_count": len(text)}
            except Exception as exc:  # per-item isolation; retry transient failures
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt < max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 30))
        progress["failed"] += 1
        n = progress["done"] + progress["skipped"] + progress["failed"]
        print(f"[{n}/{total}] FAIL {item_id} -> {last_error}", flush=True)
        return {"item_id": item_id, "status": "failed", "error": last_error}


async def _run(args: argparse.Namespace) -> int:
    contract = R6ArtifactContract.production()
    workspace = contract.workspace
    run_id = contract.run_id
    out_dir = workspace / contract.relative_path("speech_text")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_acquired_records(contract)
    if args.limit is not None:
        rows = rows[: args.limit]
    estimate_total = _load_estimate_total(contract)
    existing = _existing_spend(workspace)

    # Up-front cap enforcement (fail closed). Deterministic per-minute pricing
    # means the estimate is the spend, so a whole-batch fit guarantees every
    # subset fits — no per-call reservation needed.
    projected = existing + estimate_total
    if projected > SPEECH_CAP_USD or projected > ALL_IN_CAP_USD:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "estimate + existing spend would breach a cap",
                    "existing_usd": round(existing, 6),
                    "estimate_usd": round(estimate_total, 6),
                    "projected_usd": round(projected, 6),
                    "speech_cap_usd": SPEECH_CAP_USD,
                    "all_in_cap_usd": ALL_IN_CAP_USD,
                }
            )
        )
        return 2

    model = default_model_for_provider(TASK_ID, "openai")
    settings = load_settings()
    settings.openai_model = model  # pin the governed model the estimate priced
    service = create_transcription_service(settings)

    total = len(rows)
    progress = {"done": 0, "skipped": 0, "failed": 0}
    sem = asyncio.Semaphore(args.workers)
    started = time.time()

    print(
        f"bulk r6 ASR: {total} acquired items | model={model} | workers={args.workers} "
        f"| existing_spend=${existing:.4f} | estimate=${estimate_total:.4f} "
        f"| projected_total=${projected:.4f} (speech cap ${SPEECH_CAP_USD})",
        flush=True,
    )

    # One batch-wide receipt context: attribution is constant for all calls, so
    # the process-global env vars it sets never change mid-batch (concurrency
    # safe), while each call still writes its own uuid-named provider receipt
    # into the canonical spend log.
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=STAGE_ID,
        caller="transcription_ocr.cvpr2026.r6.bulk_parallel",
        task_id=TASK_ID,
    ):
        results = await asyncio.gather(
            *(
                _transcribe_one(
                    service=service,
                    row=row,
                    workspace=workspace,
                    out_dir=out_dir,
                    model=model,
                    run_id=run_id,
                    sem=sem,
                    max_attempts=args.max_attempts,
                    progress=progress,
                    total=total,
                )
                for row in rows
            )
        )

    elapsed = time.time() - started
    after_spend = _existing_spend(workspace)
    summary = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_bulk_transcription",
        "driver": "parallel_bulk_asr",
        "run_id": run_id,
        "model": model,
        "provider": "openai",
        "transcript_source": "asr_fallback",
        "workers": args.workers,
        "counts": {
            "acquired_total": total,
            "succeeded": progress["done"],
            "skipped_already_present": progress["skipped"],
            "failed": progress["failed"],
        },
        "spend_usd": {
            "existing_before": round(existing, 6),
            "canonical_after": round(after_spend, 6),
            "delta": round(after_spend - existing, 6),
            "estimate_total": round(estimate_total, 6),
            "speech_cap_usd": SPEECH_CAP_USD,
            "all_in_cap_usd": ALL_IN_CAP_USD,
        },
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": contract.relative_path("speech_text").as_posix(),
        "secrets_copied": False,
        "note": (
            "Parallel ASR driver, distinct from the serial ledger-bound r6 owner. "
            "Only acquired direct_public_youtube rows were transcribed; "
            "download_failed rows remain accepted coverage gaps."
        ),
        "failed_items": [r["item_id"] for r in results if r.get("status") == "failed"],
    }
    summary_path = workspace / "storage/transcription-ocr/cvpr2026/r6/bulk_transcription_receipt.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_items"}), flush=True)
    return 0 if progress["failed"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24, help="max concurrent transcriptions")
    parser.add_argument("--max-attempts", type=int, default=4, help="retry attempts per item")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N acquired items (smoke test)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
