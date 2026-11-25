#!/usr/bin/env python3
"""Split multi-talk event recordings (oral sessions, expo and affinity events,
workshops) into named talks, as talk maps Stage 6 verifies and attaches.

Conference-neutral. Recordings come from the SlidesLive recovery inventory,
selected by event kind (``--kind``) and/or item id (``--item-id``). When an
event page lists its talks (an oral session lists its orals), that program is
handed to the governed model for
``transcription_ocr.workshop_talk_boundary_discovery`` so every talk it finds
carries its scheduled title.

Modes:

    --estimate                 selection + rough cost; no provider call
    --provider-cost-approved   run discovery (refuses when the estimate is over
                               the task's registry budget cap)
    --resume (alone)           every selected map already current: re-stamp the
                               batch and lineage receipts, no provider call;
                               refuses if any recording still needs a split

Outputs: ``storage/transcription-ocr/talk-maps/base/<video_id>_{talks,
annotated,batches,discovery_receipt}.json`` -- only per-recording files, since
that directory is a registered corpus-inventory-index lineage input. The cost
estimate and batch summary go to ``storage/transcription-ocr/talk-maps/receipts/``;
with ``--run-id`` the batch summary is also the ``event-talk-splits`` lineage
sub-step receipt at ``storage/workflow-runs/<run-id>/event-talk-splits-batch-receipt.json``.
Provider credentials come from the repository ``.env`` (source it first).

    .venv/bin/python scripts/split_event_recordings.py --workspace conferences/<slug> \\
        --kind oral --kind affinity-event --item-id <id> --estimate
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.indexing import event_talk_splits as splits  # noqa: E402
from conference_pipeline.indexing import talk_maps  # noqa: E402
from conference_pipeline.model_governance import default_model, default_provider  # noqa: E402
from conference_pipeline.model_governance.registry import load_registry  # noqa: E402
from conference_pipeline.workshop_talk_discovery_windowed import estimate_window_cost  # noqa: E402
from conference_pipeline.workshop_talk_segmentation import WORKSHOP_TALK_DISCOVERY_TASK_ID  # noqa: E402

ESTIMATE_NAME = "event_split_cost_estimate.json"
BATCH_RECEIPT_NAME = "event_split_batch_receipt.json"
RECEIPTS_DIR_NAME = "receipts"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _talk_maps_on_disk(workspace: Path) -> dict[str, Any]:
    """What the talk-map directory holds now (the sub-step's actual output)."""

    directory = talk_maps.talk_map_dir(workspace, talk_maps.BASE_SCOPE)
    maps = sorted(directory.glob("*_talks.json")) if directory.is_dir() else []
    talks = 0
    for path in maps:
        try:
            talks += len(json.loads(path.read_text(encoding="utf-8")).get("talks") or [])
        except (OSError, json.JSONDecodeError):
            continue
    return {"recording_count": len(maps), "talk_count": talks}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--kind", action="append", default=[], help="event kind to split (repeatable)")
    parser.add_argument("--item-id", action="append", default=[], help="event item id to split (repeatable)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip recordings whose talk map is current; when every map is current this "
        "needs no --provider-cost-approved and only re-stamps the batch and lineage receipts",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, client: Any | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    if not args.kind and not args.item_id:
        print("choose recordings with --kind and/or --item-id", file=sys.stderr)
        return 2
    recordings, skipped = splits.load_event_recordings(workspace, kinds=args.kind, item_ids=args.item_id)
    if args.resume:
        current = [r for r in recordings if splits.talk_map_is_current(workspace, r)]
        skipped += [{"item_id": r.item_id, "video_id": r.video_id, "reason": "talk_map_current"} for r in current]
        recordings = [r for r in recordings if r not in current]
    provider = default_provider(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    model = default_model(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    cap = load_registry().get(WORKSHOP_TALK_DISCOVERY_TASK_ID).budget_cap_usd
    rows = []
    for recording in recordings:
        tokens = splits.estimate_prompt_tokens(recording)
        rows.append(
            {
                "item_id": recording.item_id,
                "video_id": recording.video_id,
                "kind": recording.kind,
                "title": recording.title,
                "scheduled_talk_count": len(recording.scheduled_talks),
                "prompt_tokens": tokens,
                "estimated_cost_usd": round(estimate_window_cost(tokens, model=model, provider=provider), 4),
            }
        )
    total = round(sum(row["estimated_cost_usd"] for row in rows), 4)
    out_dir = talk_maps.talk_map_dir(workspace, talk_maps.BASE_SCOPE).parent / RECEIPTS_DIR_NAME
    estimate = {
        "receipt_type": "conference_pipeline.event_split_cost_estimate",
        "schema_version": 1,
        "generated_at": _now(),
        "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
        "provider": provider,
        "model": model,
        "budget_cap_usd": cap,
        "selection": {"kinds": args.kind, "item_ids": args.item_id},
        "recording_count": len(rows),
        "estimated_total_usd": total,
        "recordings": rows,
        "skipped": skipped,
    }
    _write(out_dir / ESTIMATE_NAME, estimate)
    print(f"{len(rows)} recordings, {sum(r['scheduled_talk_count'] for r in rows)} scheduled talks, "
          f"~{sum(r['prompt_tokens'] for r in rows):,} prompt tokens, estimated ${total:.2f} "
          f"with {provider}/{model} (task cap ${cap})")
    for item in skipped:
        print(f"  skipped {item}")
    if args.estimate:
        return 0
    if rows and not args.provider_cost_approved:
        # Only a real split calls the provider; re-stamping current maps is free.
        print(
            f"refusing: {len(rows)} recordings need splitting, which calls the provider; "
            "pass --provider-cost-approved (or --estimate first)",
            file=sys.stderr,
        )
        return 1
    if cap is not None and total > cap:
        print(f"refusing: estimate ${total:.2f} is over the task budget cap ${cap}", file=sys.stderr)
        return 1

    def run(recording: splits.EventRecording) -> dict[str, Any]:
        try:
            return splits.split_recording(
                workspace, recording, provider_cost_approved=True, run_id=args.run_id, client=client
            )
        except Exception as error:  # noqa: BLE001 - one recording must not stop the batch
            return {"item_id": recording.item_id, "video_id": recording.video_id, "status": "failed",
                    "error": f"{type(error).__name__}: {str(error)[:300]}"}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, recordings))
    spent = round(sum(float(r.get("estimated_cost_usd") or 0) for r in results), 4)
    succeeded = [r for r in results if r.get("status") == "succeeded"]
    # Keep earlier runs' results (a --resume retry covers only what failed).
    prior_path = out_dir / BATCH_RECEIPT_NAME
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    this_run = {r.get("item_id") for r in results}
    history = [r for r in prior.get("results") or [] if r.get("item_id") not in this_run]
    all_results = history + results
    for result in results:
        matched = result.get("scheduled_titles_matched")
        program = f" program {matched}/{result.get('scheduled_talk_count')}" if matched is not None else ""
        print(f"{str(result.get('status')):>10}  {result.get('item_id')}  talks={result.get('talk_count')}{program}"
              + (f"  {result.get('blocker_code') or result.get('error')}" if result.get("status") != "succeeded" else ""))
    batch_receipt = {
            "receipt_type": "conference_pipeline.event_split_batch",
            "schema_version": 1,
            "generated_at": _now(),
            "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
            "provider": provider,
            "model": model,
            "selection": {"kinds": args.kind, "item_ids": args.item_id},
            "recording_count": len(all_results),
            "succeeded_count": sum(1 for r in all_results if r.get("status") == "succeeded"),
            "talk_count": sum(int(r.get("talk_count") or 0) for r in all_results if r.get("status") == "succeeded"),
            "estimated_cost_usd": round(float(prior.get("estimated_cost_usd") or 0) + spent, 4),
            "last_run": {"recording_count": len(results), "succeeded_count": len(succeeded), "estimated_cost_usd": spent},
            "results": all_results,
            "skipped": skipped,
            "secrets_copied": False,
            "status": "succeeded" if all(r.get("status") == "succeeded" for r in all_results) else "blocked",
            "mode": "live",
            "run_id": args.run_id,
            "talk_maps_path": str(talk_maps.talk_map_dir(workspace, talk_maps.BASE_SCOPE).relative_to(workspace)),
            "talk_maps_on_disk": _talk_maps_on_disk(workspace),
    }
    _write(out_dir / BATCH_RECEIPT_NAME, batch_receipt)
    if args.run_id:
        # The lineage sub-step: binds the talk maps to the transcription they
        # were split from; corpus-inventory-index reads the maps as an input.
        substep_path = splits.write_split_batch_receipt(workspace, dict(batch_receipt), run_id=args.run_id)
        print(f"lineage sub-step receipt: {substep_path}")
    print(f"{len(succeeded)}/{len(results)} recordings split into "
          f"{sum(int(r.get('talk_count') or 0) for r in succeeded)} talks; provider cost ${spent:.2f}")
    return 0 if len(succeeded) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
