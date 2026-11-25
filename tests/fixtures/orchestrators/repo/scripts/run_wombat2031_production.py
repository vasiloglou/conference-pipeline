"""Synthetic CVPR-shaped orchestrator (share audit fixture; never executed).

Shape: receipt-first runner with an extra ``--prepare-preflight-estimate``
command, a program-spine source discovery delegated to a conference-local
component, a download stage already lifted onto a shared stage module with an
organizer-channel acquisition lane from an annotated (``RECOMMENDED_FOR``)
shared acquisition module, a local transcription stage with a retry
authorization quirk, and an unreachable ``dispatch_stage`` fallback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from conference_pipeline.acquisition.wombat_program_spine import run_program_spine_discovery
from conference_pipeline.acquisition.youtube_channel_harvest import harvest_channel_talks
from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import download_corpus_stage as production_download_corpus_stage
from conference_pipeline.production import release_stage as production_release_stage
from conference_pipeline.production.hooks import conference_hook

RUNNER_ID = "wombat2031-production"
ORGANIZER_CHANNELS = ("workshop-a", "workshop-b")


@dataclass(frozen=True)
class ProductionStage:
    id: str
    order: int
    receipt_filename: str
    implemented: bool = True


STAGES: dict[str, ProductionStage] = {
    "runner-preflight": ProductionStage(id="runner-preflight", order=1, receipt_filename="stage-01-runner-preflight.json"),
    "source-discovery": ProductionStage(id="source-discovery", order=2, receipt_filename="stage-02-source-discovery.json"),
    "download-corpus": ProductionStage(id="download-corpus", order=4, receipt_filename="stage-04-download-corpus.json"),
    "transcription-ocr": ProductionStage(id="transcription-ocr", order=5, receipt_filename="stage-05-transcription-ocr.json"),
    "release": ProductionStage(id="release", order=11, receipt_filename="stage-11-release.json"),
}


def build_stage_receipt(workspace, run_id, stage: ProductionStage, status, checks):
    receipt = {
        "receipt_type": "wombat2031_production_stage_receipt",
        "title": "WOMBAT 2031 production stage",
        "stage_id": stage.id,
        "run_id": run_id,
        "status": status,
        "checks": dict(checks),
        "workspace": str(workspace),
    }
    return receipt


def run_runner_preflight(workspace, run_id):
    approval = Path(workspace) / "storage" / "workflow-runs" / run_id / "approval.json"
    approved = approval.is_file()
    checks = {"approval_present": approved, "estimate_scope": "prospective-calls"}
    receipt = build_stage_receipt(workspace, run_id, STAGES["runner-preflight"], "succeeded" if approved else "blocked", checks)
    return {"status": receipt["status"], "receipt": receipt, "summary": {"stage": "runner-preflight"}}


def prepare_preflight_estimate(workspace, run_id):
    rows = [{"task": "download-corpus.talks", "quantity": len(ORGANIZER_CHANNELS), "unit_usd": 0.0}]
    total = sum(row["quantity"] * row["unit_usd"] for row in rows)
    return {"run_id": run_id, "workspace": str(workspace), "rows": rows, "total_usd": total}


def production_context(workspace, run_id) -> ProductionRunnerContext:
    return ProductionRunnerContext(
        workspace=Path(workspace), run_id=run_id, runner_id=RUNNER_ID, receipt_type="stage_receipt"
    )


@conference_hook("download-corpus.acquisition_lane")
def organizer_channel_lane(ctx, frozen_corpus, prior_inventory):
    talks = harvest_channel_talks(ORGANIZER_CHANNELS, transport=lambda channel: [])
    known = {record.get("video_id") for record in prior_inventory}
    fresh = [talk for talk in talks if talk["video_id"] not in known]
    return {"artifacts": fresh, "blockers": [], "receipt_fields": {"channels": list(ORGANIZER_CHANNELS)}}


def run_download_corpus(workspace, run_id, resume=False):
    return production_download_corpus_stage.run_download_corpus_stage(
        production_context(workspace, run_id), resume=resume, hooks={"acquisition_lanes": (organizer_channel_lane,)}
    )


def run_transcription_ocr_stage(workspace, run_id, retry_authorization=None):
    captions = sorted(Path(workspace, "media", "captions").glob("*.json"))
    materialized = []
    for caption in captions:
        payload = json.loads(caption.read_text(encoding="utf-8"))
        if payload.get("status") == "failed" and not retry_authorization:
            continue
        materialized.append({"talk_id": payload.get("talk_id"), "chars": len(str(payload.get("text") or ""))})
    checks = {"caption_count": len(captions), "materialized": len(materialized)}
    receipt = build_stage_receipt(workspace, run_id, STAGES["transcription-ocr"], "succeeded", checks)
    return {"status": "succeeded", "receipt": receipt, "summary": {"stage": "transcription-ocr"}}


def run_release_stage(
    workspace,
    run_id,
    resume=True,
    publish_target=None,
    publish_approved_by=None,
):
    """Thin wrapper with a long signature: still an adapter by body lines."""

    return production_release_stage.run_release_stage(production_context(workspace, run_id), resume=resume)


def build_dispatch_fingerprint(workspace, run_id, stage_id):
    payload = {"workspace": str(workspace), "run_id": run_id, "stage_id": stage_id}
    encoded = json.dumps(payload, sort_keys=True)
    return {"fingerprint": encoded, "stage_id": stage_id}


def dispatch_stage(workspace, run_id, stage_id, static_fingerprint):
    checks = {"fingerprint": static_fingerprint, "dispatched": False}
    receipt = build_stage_receipt(workspace, run_id, STAGES[stage_id], "blocked", checks)
    return {"status": "blocked", "receipt": receipt, "summary": {"stage": stage_id}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="runner-preflight", choices=sorted(STAGES))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--retry-authorization", type=Path)
    parser.add_argument("--prepare-preflight-estimate", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.prepare_preflight_estimate:
        print(json.dumps(prepare_preflight_estimate(args.workspace, args.run_id), sort_keys=True))
        return 0
    stage = STAGES[args.stage]
    if stage.id == "runner-preflight":
        outcome = run_runner_preflight(args.workspace, args.run_id)
    elif stage.id == "source-discovery":
        outcome = run_program_spine_discovery(args.workspace, args.run_id)
    elif stage.id == "download-corpus":
        outcome = run_download_corpus(args.workspace, args.run_id, resume=args.resume)
    elif stage.id == "transcription-ocr":
        outcome = run_transcription_ocr_stage(
            args.workspace, args.run_id, retry_authorization=args.retry_authorization
        )
    elif stage.id == "release":
        outcome = run_release_stage(args.workspace, args.run_id, resume=args.resume)
    else:
        outcome = dispatch_stage(
            args.workspace,
            args.run_id,
            stage.id,
            build_dispatch_fingerprint(args.workspace, args.run_id, stage.id),
        )
    if args.summary:
        print(json.dumps(outcome["summary"], indent=2, sort_keys=True))
    return 0 if outcome["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
