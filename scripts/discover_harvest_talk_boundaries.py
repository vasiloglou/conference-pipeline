#!/usr/bin/env python3
"""Split long multi-talk recordings in the transcript corpus into talks.

Conference-neutral. Reads every admitted speech transcript from the corpus
inventory (base caption inventory and every evidence lane; segment JSON and
caption files alike), records an eligibility decision for each, and runs
windowed talk-boundary discovery with the governed model for
``transcription_ocr.workshop_talk_boundary_discovery`` on those at least
``--min-seconds`` long.

Three modes:

    --estimate                 cost estimate + eligibility only; no provider call
    --provider-cost-approved   run discovery (requires an estimate within the
                               task budget cap; halts before exceeding it)
    --report-only              print the talk maps already on disk

Outputs land next to the transcript: a lane's ``talks/`` directory, or
``storage/transcription-ocr/talk-maps/base/`` for base captions. Stage 6
attaches verified maps to their records; harvest lanes that bind maps in their
lane receipt must also be re-bound with
``scripts/build_youtube_harvest_evidence_lanes.py --attach-talk-maps``.
Provider credentials come from the repository ``.env`` (source it first).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.costs.budget_enforcement import check_preflight  # noqa: E402
from conference_pipeline.indexing import talk_maps  # noqa: E402
from conference_pipeline.indexing import transcript_evidence_lanes as lanes  # noqa: E402
from conference_pipeline.model_governance import default_model, default_provider  # noqa: E402
from conference_pipeline.model_governance.registry import load_registry  # noqa: E402
from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402
from conference_pipeline.talk_program_hints import load_program_hints  # noqa: E402
from conference_pipeline import workshop_talk_discovery_windowed as windowed  # noqa: E402
from conference_pipeline.workshop_talk_segmentation import (  # noqa: E402
    WORKSHOP_TALK_DISCOVERY_TASK_ID,
    _build_gemini_client,
    sha256_file,
    utc_now_iso,
    workspace_relpath,
)

ESTIMATE_NAME = "talk_discovery_cost_estimate.json"
BATCH_RECEIPT_NAME = "talk_discovery_batch_receipt.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


DEFAULT_CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")
SPEECH_SOURCE_FAMILY = "speech_transcript"


def _eligibility(record: dict[str, Any], transcript: Path, min_seconds: float) -> tuple[str, float, list[dict[str, Any]]]:
    """``(decision, duration_seconds, segments)`` for one admitted transcript."""

    segments = windowed.load_transcript_segments(transcript)
    if not segments:
        text = transcript.read_text(encoding="utf-8", errors="replace")
        words = len(text.split())
        estimate = words / windowed.WORDS_PER_MINUTE_ESTIMATE * 60.0
        duration = max(float(record.get("duration_seconds") or 0.0), estimate)
        decision = "long_but_untimed" if duration >= min_seconds else "too_short"
        return decision, duration, []
    duration = windowed.effective_duration(segments, record.get("duration_seconds"))
    if not windowed.qualifies_for_talk_splitting(duration, min_seconds=min_seconds):
        return "too_short", duration, segments
    return "eligible", duration, segments


def candidate_recordings(
    workspace: Path,
    *,
    lane_ids: list[str],
    video_ids: list[str],
    min_seconds: float,
    corpus_inventory: Path | None = None,
) -> tuple[list[windowed.RecordingInput], dict[str, Any]]:
    """Every admitted speech transcript is considered; returns ``(eligible, eligibility)``.

    The corpus inventory is the source (base captions and every evidence lane,
    caption files included), so no admitted recording is skipped silently:
    each gets a decision, and long recordings that cannot be split are listed.
    """

    inventory_path = corpus_inventory or (workspace / DEFAULT_CORPUS_INVENTORY)
    inventory = _read_json(inventory_path) or {}
    recordings: list[windowed.RecordingInput] = []
    decisions: dict[str, int] = {}
    excluded_long: list[dict[str, Any]] = []
    for record in inventory.get("artifact_records") or []:
        if not isinstance(record, dict) or record.get("source_family_id") != SPEECH_SOURCE_FAMILY:
            continue
        scope = talk_maps.talk_map_scope(record)
        video_id = str(record.get("video_id") or "")
        if lane_ids and scope not in lane_ids:
            continue
        if video_ids and video_id not in video_ids:
            continue
        transcript, error = lanes.resolve_workspace_member(workspace, record.get("transcript_path"))
        if error or transcript is None or not transcript.is_file():
            decisions["transcript_unreadable"] = decisions.get("transcript_unreadable", 0) + 1
            excluded_long.append({"video_id": video_id, "scope": scope, "decision": "transcript_unreadable"})
            continue
        decision, duration, _segments = _eligibility(record, transcript, min_seconds)
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision == "long_but_untimed":
            excluded_long.append(
                {"video_id": video_id, "scope": scope, "decision": decision, "duration_seconds": round(duration, 1)}
            )
        if decision != "eligible":
            continue
        urls = record.get("source_event_urls") or []
        recordings.append(
            windowed.RecordingInput(
                video_id=video_id,
                lane_id=scope,
                transcript_path=transcript,
                title=str(record.get("title") or record.get("source_title") or video_id),
                duration_seconds=duration,
                timing_kind=str(record.get("timing_kind") or "caption_cue"),
                hints=load_program_hints(
                    workspace,
                    source_url=urls[0] if urls else None,
                    title=record.get("title"),
                    channel_category=record.get("channel_category"),
                ),
            )
        )
    eligibility = {
        "corpus_inventory_path": workspace_relpath(workspace, inventory_path),
        "min_seconds": min_seconds,
        "decision_counts": dict(sorted(decisions.items())),
        "excluded_long_recordings": excluded_long,
    }
    return recordings, eligibility


def estimate(
    workspace: Path,
    recordings: list[windowed.RecordingInput],
    *,
    provider: str,
    model: str,
    cap: float | None,
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    total = 0.0
    total_tokens = 0
    for recording in recordings:
        plan = windowed.plan_recording(recording)
        tokens = sum(int(window["prompt_tokens"]) for window in plan["windows"])
        cost = sum(
            windowed.estimate_window_cost(int(window["prompt_tokens"]), model=model, provider=provider)
            for window in plan["windows"]
        )
        total += cost
        total_tokens += tokens
        rows.append(
            {
                "video_id": recording.video_id,
                "scope": recording.lane_id,
                "title": recording.title,
                "duration_seconds": round(float(plan["duration"]), 1),
                "timing_kind": recording.timing_kind,
                "window_count": len(plan["windows"]),
                "prompt_tokens": tokens,
                "estimated_cost_usd": round(cost, 4),
                "hint_coverage": recording.hints.get("hint_coverage"),
                "workshop_title": recording.hints.get("workshop_title"),
            }
        )
    payload = {
        "receipt_type": "conference_pipeline.talk_discovery_cost_estimate",
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
        "provider": provider,
        "model": model,
        "assumed_output_tokens_per_window": windowed.ASSUMED_OUTPUT_TOKENS_PER_WINDOW,
        "recording_count": len(rows),
        "window_count": sum(row["window_count"] for row in rows),
        "prompt_tokens": total_tokens,
        "estimated_cost_usd": round(total, 4),
        "task_budget_cap_usd": cap,
        "within_task_cap": cap is None or total <= cap,
        "recordings": rows,
        "eligibility": eligibility or {},
        "discovery_algorithm_version": windowed.DISCOVERY_ALGORITHM_VERSION,
        "provider_calls_made": False,
        "secrets_copied": False,
    }
    target = workspace / lanes.LANES_RELATIVE_DIR / ESTIMATE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["path"] = workspace_relpath(workspace, target)
    return payload


def _already_discovered(workspace: Path, recording: windowed.RecordingInput) -> dict[str, Any] | None:
    """A complete, current generation: same algorithm, full digest chain, receipt last."""

    files = talk_maps.talk_map_files(workspace, recording.lane_id, recording.video_id)
    talks = _read_json(files["talks"])
    receipt = _read_json(files["receipt"])
    if not talks or not receipt:
        return None
    if talks.get("discovery_method") != windowed.DISCOVERY_METHOD_WINDOWED:
        return None
    if talks.get("discovery_algorithm_version") != windowed.DISCOVERY_ALGORITHM_VERSION:
        return None
    _binding, code = talk_maps.verify_talk_map_chain(
        workspace,
        talks_path=files["talks"],
        annotated_path=files["annotated"],
        video_id=recording.video_id,
        transcript_sha256=sha256_file(recording.transcript_path),
    )
    if code is not None:
        return None
    if receipt.get("talks_json_sha256") != sha256_file(files["talks"]) or receipt.get(
        "annotated_transcript_json_sha256"
    ) != sha256_file(files["annotated"]):
        return None  # interrupted mid-write: the receipt is written last
    return talks


def run(
    workspace: Path,
    recordings: list[windowed.RecordingInput],
    *,
    provider: str,
    model: str,
    cap: float | None,
    run_id: str,
    resume: bool,
    workers: int = 1,
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_estimate = estimate(workspace, recordings, provider=provider, model=model, cap=cap, eligibility=eligibility)
    preflight = check_preflight(workspace, WORKSHOP_TALK_DISCOVERY_TASK_ID, plan_estimate["estimated_cost_usd"])
    batch: dict[str, Any] = {
        "receipt_type": "conference_pipeline.talk_discovery_batch",
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
        "provider": provider,
        "model": model,
        "task_budget_cap_usd": cap,
        "estimate_path": plan_estimate["path"],
        "estimated_cost_usd": plan_estimate["estimated_cost_usd"],
        "preflight": {"breach": preflight.breach, "reason": preflight.reason},
        "recordings": [],
        "secrets_copied": False,
    }
    if preflight.breach:
        batch["status"] = "blocked_budget_preflight"
        return batch
    state = {"spent": 0.0, "reserved": 0.0, "halted": False}
    lock = threading.Lock()
    estimates = {row["video_id"]: float(row["estimated_cost_usd"]) for row in plan_estimate["recordings"]}
    client = _build_gemini_client()

    def _one(recording: windowed.RecordingInput) -> dict[str, Any]:
        if resume and _already_discovered(workspace, recording):
            return {"video_id": recording.video_id, "status": "resume_skipped"}
        # Reserve this recording's estimate so concurrent workers never jointly cross the cap.
        with lock:
            reserve = estimates.get(recording.video_id, 0.0)
            remaining = None if cap is None else max(0.0, cap - state["spent"] - state["reserved"])
            forced = "budget_halt" if state["halted"] or (remaining is not None and reserve > remaining) else None
            if forced:
                state["halted"] = True
            else:
                state["reserved"] += reserve
        output_dir = talk_maps.talk_map_dir(workspace, recording.lane_id)
        receipt = windowed.discover_recording(
            workspace=workspace,
            recording=recording,
            output_dir=output_dir,
            client=client,
            provider=provider,
            model=model,
            remaining_budget_usd=None if remaining is None else remaining,
            call_window=windowed.gemini_call_window,
            fallback_reason=forced,
        )
        with lock:
            if not forced:
                state["reserved"] -= reserve
            state["spent"] += float(receipt.get("spent_usd") or 0.0)
            if receipt.get("fallback_reason") == "budget_halt_before_window":
                state["halted"] = True
        return {
            "video_id": recording.video_id,
            "scope": recording.lane_id,
            "discovery_method": receipt["discovery_method"],
            "fallback_reason": receipt.get("fallback_reason"),
            "talk_count": receipt["talk_count"],
            "window_count": receipt["window_count"],
            "spent_usd": round(float(receipt.get("spent_usd") or 0.0), 6),
            "validation_issues": receipt.get("validation_issues"),
        }

    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id="transcription-ocr",
        caller=__file__,
        task_id=WORKSHOP_TALK_DISCOVERY_TASK_ID,
    ):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            batch["recordings"] = list(pool.map(_one, recordings))
    spent = state["spent"]
    halted = state["halted"]
    batch["spent_usd"] = round(spent, 6)
    batch["halted_on_budget"] = halted
    batch["status"] = "halted_on_budget" if halted else "succeeded"
    target = workspace / lanes.LANES_RELATIVE_DIR / BATCH_RECEIPT_NAME
    target.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return batch


def report(workspace: Path, recordings: list[windowed.RecordingInput]) -> list[dict[str, Any]]:
    out = []
    for recording in recordings:
        talks = _read_json(talk_maps.talk_map_files(workspace, recording.lane_id, recording.video_id)["talks"])
        out.append(
            {
                "video_id": recording.video_id,
                "title": recording.title,
                "discovery_method": (talks or {}).get("discovery_method"),
                "fallback_reason": (talks or {}).get("fallback_reason"),
                "talks": [
                    f"{int(t['timestamp_start'])}-{int(t['timestamp_end'])}s [{t.get('talk_kind')}] "
                    f"{t.get('talk_title')} ({t.get('talk_speaker') or 'unknown'})"
                    for t in (talks or {}).get("talks") or []
                ],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--lane-id", action="append", default=[], help="limit to a lane id, or 'base' for the base caption inventory"
    )
    parser.add_argument("--corpus-inventory", type=Path, default=None, help="default: storage/corpus/corpus_inventory.json")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--min-seconds", type=float, default=windowed.SPLIT_MIN_SECONDS)
    parser.add_argument("--run-id", default="talk-boundary-discovery")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--provider-cost-approved", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument(
        "--remerge-only",
        action="store_true",
        help="re-apply current merge/validation rules to stored window candidates; no provider call",
    )
    parser.add_argument("--resume", action="store_true", help="skip recordings with a current model talk map")
    parser.add_argument("--workers", type=int, default=1, help="recordings discovered concurrently")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    recordings, eligibility = candidate_recordings(
        workspace,
        lane_ids=args.lane_id,
        video_ids=args.video_id,
        min_seconds=args.min_seconds,
        corpus_inventory=args.corpus_inventory.resolve() if args.corpus_inventory else None,
    )
    provider = default_provider(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    model = default_model(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    cap = load_registry().get(WORKSHOP_TALK_DISCOVERY_TASK_ID).budget_cap_usd
    if args.remerge_only:
        results = []
        for recording in recordings:
            receipt = windowed.remerge_recording(
                workspace=workspace,
                recording=recording,
                output_dir=talk_maps.talk_map_dir(workspace, recording.lane_id),
                provider=provider,
                model=model,
            )
            results.append(
                {
                    "video_id": recording.video_id,
                    "scope": recording.lane_id,
                    "remerged": receipt is not None,
                    "discovery_method": (receipt or {}).get("discovery_method"),
                    "talk_count": (receipt or {}).get("talk_count"),
                    "validation_issues": (receipt or {}).get("validation_issues"),
                }
            )
        summary = {
            "remerged": sum(1 for item in results if item["remerged"]),
            "fallbacks": sum(1 for item in results if item["discovery_method"] == windowed.DISCOVERY_METHOD_FALLBACK),
            "not_remergeable": [item["video_id"] for item in results if not item["remerged"]],
            "provider_calls_made": False,
        }
        print(json.dumps(summary, indent=2))
        return 0
    if args.report_only:
        print(json.dumps(report(workspace, recordings), indent=2))
        return 0
    if args.estimate:
        payload = estimate(workspace, recordings, provider=provider, model=model, cap=cap, eligibility=eligibility)
        payload.pop("recordings")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    batch = run(
        workspace,
        recordings,
        provider=provider,
        model=model,
        cap=cap,
        run_id=args.run_id,
        resume=args.resume,
        workers=args.workers,
        eligibility=eligibility,
    )
    summary = {key: value for key, value in batch.items() if key != "recordings"}
    summary["recordings"] = len(batch["recordings"])
    summary["fallbacks"] = sum(1 for item in batch["recordings"] if item.get("fallback_reason"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if batch["status"] in {"succeeded", "halted_on_budget"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
