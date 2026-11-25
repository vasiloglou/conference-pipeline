"""Synthetic ICML-shaped orchestrator (share audit fixture; never executed).

Shape: a fat runner with local Stages 1-7, declared quirk hooks for
authenticated downloads and a talk-recovery lane, a physics-first paper
ordering, forks of shared ``src`` utilities, dead legacy helpers, and
Stages 11-12 already running through shared ``conference_pipeline.production``
modules via ``run_shared_stage``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import proof_stage as production_proof_stage
from conference_pipeline.production import release_stage as production_release_stage
from conference_pipeline.production.hooks import conference_hook

RUNNER_ID = "quokka2031-production"
PHYSICS_TERMS = ("physics", "pde", "fluid")


@dataclass(frozen=True)
class ProductionStage:
    id: str
    order: int
    receipt_filename: str
    implemented: bool = True


STAGES: dict[str, ProductionStage] = {
    "runner-preflight": ProductionStage(id="runner-preflight", order=1, receipt_filename="stage-01-runner-preflight.json"),
    "download-corpus": ProductionStage(id="download-corpus", order=4, receipt_filename="stage-04-download-corpus.json"),
    "analysis": ProductionStage(id="analysis", order=7, receipt_filename="stage-07-analysis.json"),
    "release": ProductionStage(id="release", order=11, receipt_filename="stage-11-release.json"),
    "proof": ProductionStage(id="proof", order=12, receipt_filename="stage-12-proof.json"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_speech_lane(record):
    kind = str(record.get("kind") or "")
    if kind == "caption":
        return "caption-transcript"
    if kind == "audio":
        return "speech-transcription"
    if record.get("slides"):
        return "slide-ocr"
    return "unsupported-lane"


def build_stage_receipt(workspace, run_id, stage: ProductionStage, status, checks):
    receipt = {
        "receipt_type": "quokka2031_production_stage_receipt",
        "title": "QUOKKA 2031 production stage",
        "stage_id": stage.id,
        "run_id": run_id,
        "status": status,
        "checks": dict(checks),
        "workspace": str(workspace),
    }
    return receipt


def write_summary(workspace, run_id, receipt):
    summary_path = Path(workspace) / "storage" / "workflow-runs" / run_id / "summary.json"
    summary = {"run_id": run_id, "last_stage": receipt["stage_id"], "status": receipt["status"]}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_runner_preflight(args, stage: ProductionStage):
    checks = {"workspace_exists": Path(args.workspace).is_dir(), "started_at": utc_now()}
    status = "succeeded" if checks["workspace_exists"] else "blocked"
    receipt = build_stage_receipt(args.workspace, args.run_id, stage, status, checks)
    return (0 if status == "succeeded" else 1), write_summary(args.workspace, args.run_id, receipt)


@conference_hook("download-corpus.auth")
def openreview_bearer_auth(env):
    token_present = bool(env.get("OPENREVIEW_USERNAME")) and bool(env.get("OPENREVIEW_PASSWORD"))
    return {"mode": "bearer" if token_present else "public", "secrets_copied": False}


@conference_hook("download-corpus.acquisition_lane")
def slideslive_carry_forward_lane(prior_inventory):
    carried = []
    for record in prior_inventory:
        if record.get("slideslive_id") and record.get("media_path"):
            carried.append({**record, "lane": classify_speech_lane(record)})
    return {"artifacts": carried, "blockers": [], "receipt_fields": {"carried": len(carried)}}


def paper_priority_key(record):
    title = str(record.get("title") or "").lower()
    physics_first = 0 if any(term in title for term in PHYSICS_TERMS) else 1
    return (physics_first, title)


def run_download_corpus(args, stage: ProductionStage):
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    auth = openreview_bearer_auth({})
    papers = sorted(inventory.get("papers", []), key=paper_priority_key)
    artifacts = []
    for paper in papers:
        pdf = Path(args.workspace) / paper["path"]
        artifacts.append({"id": paper["id"], "path": paper["path"], "sha256": sha256_file(pdf)})
    lane = slideslive_carry_forward_lane(inventory.get("media", []))
    artifacts.extend(lane["artifacts"])
    checks = {"auth_mode": auth["mode"], "artifact_count": len(artifacts)}
    receipt = build_stage_receipt(args.workspace, args.run_id, stage, "succeeded", checks)
    receipt["artifacts"] = artifacts
    return 0, write_summary(args.workspace, args.run_id, receipt)


def run_analysis(args, stage: ProductionStage):
    topics = json.loads(Path(args.workspace, "analysis", "topics.json").read_text(encoding="utf-8"))
    steps = []
    for topic in topics:
        steps.append({"topic": topic["id"], "summary": "pending", "questions": []})
        steps.append({"topic": topic["id"], "deep_dive": "pending"})
        steps.append({"topic": topic["id"], "reconciliation": "pending"})
    checks = {"topic_count": len(topics), "step_count": len(steps)}
    receipt = build_stage_receipt(args.workspace, args.run_id, stage, "succeeded", checks)
    return 0, write_summary(args.workspace, args.run_id, receipt)


def production_context(workspace, run_id) -> ProductionRunnerContext:
    return ProductionRunnerContext(
        workspace=Path(workspace), run_id=run_id, runner_id=RUNNER_ID, receipt_type="stage_receipt"
    )


def run_shared_stage(args, stage: ProductionStage, run_stage):
    outcome = run_stage(production_context(args.workspace, args.run_id))
    summary = write_summary(args.workspace, args.run_id, outcome["receipt"] | {"stage_id": stage.id})
    return (0 if outcome["status"] == "succeeded" else 1), summary


def run_release(args, stage: ProductionStage):
    return run_shared_stage(
        args, stage, lambda ctx: production_release_stage.run_release_stage(ctx, resume=bool(args.resume))
    )


def run_proof(args, stage: ProductionStage):
    return run_shared_stage(
        args, stage, lambda ctx: production_proof_stage.run_proof_stage(ctx, resume=bool(args.resume))
    )


def run_unavailable_stage(args, stage: ProductionStage):
    checks = {"stage_implemented": False, "live_provider_calls_made": False}
    receipt = build_stage_receipt(args.workspace, args.run_id, stage, "blocked", checks)
    receipt["error"] = f"Production stage {stage.id} has no live implementation yet."
    return 1, write_summary(args.workspace, args.run_id, receipt)


def legacy_presentation_summary(workspace):
    decks = sorted(Path(workspace, "decks").glob("*/slides.md"))
    rows = []
    for deck in decks:
        rows.append({"deck": deck.parent.name, "bytes": deck.stat().st_size})
    return {"decks": rows, "count": len(rows)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage = STAGES[args.stage]
    if stage.id == "runner-preflight":
        exit_code, summary = run_runner_preflight(args, stage)
    elif stage.id == "download-corpus":
        exit_code, summary = run_download_corpus(args, stage)
    elif stage.id == "analysis":
        exit_code, summary = run_analysis(args, stage)
    elif stage.id == "release":
        exit_code, summary = run_release(args, stage)
    elif stage.id == "proof":
        exit_code, summary = run_proof(args, stage)
    elif not stage.implemented:
        exit_code, summary = run_unavailable_stage(args, stage)
    else:
        raise AssertionError(f"implemented stage has no dispatcher: {stage.id}")
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
