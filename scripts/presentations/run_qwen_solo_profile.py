#!/usr/bin/env python3
"""Solo-profile the local Qwen3-VL judge against existing Topic 3 panel labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.model_governance.mlx_vision_client import (
    ManagedMlxVisionServer,
    VisionRuntimeUnavailableError,
)
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile
from conference_pipeline.model_governance.registry import ModelGovernanceError
from scripts.presentations.figure_quality_panel import QWEN_MODEL, qwen_judge_adapter


DEFAULT_OUTPUT = Path(
    ".planning/workstreams/v2-5/phases/"
    "01-panel-core-judge-adapters-and-qwen-solo-vetting/01-qwen-solo-profile.json"
)
ANCHOR_PREFIXES = ("sl_11", "sl_19", "sl_23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck-dir",
        type=Path,
        default=Path("conferences/iclr2026/decks/03_evaluation_benchmarks_data"),
    )
    parser.add_argument(
        "--panel-json",
        type=Path,
        default=Path(
            "conferences/iclr2026/decks/03_evaluation_benchmarks_data/"
            "calibration_review_panel.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument(
        "--manage-server",
        action="store_true",
        help="Start and stop a managed mlx-vlm server for the profile run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for smoke tests; 0 profiles all matched slides.",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=900.0,
        help="Seconds to wait for a managed MLX vision server to become healthy.",
    )
    return parser.parse_args()


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"{path} did not contain an items list")
    return [item for item in items if isinstance(item, dict)]


def _render_index(deck_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in deck_dir.rglob("*.png"):
        if path.name == "render_contact_sheet.png":
            continue
        stem = path.stem
        if stem.startswith("slide_"):
            try:
                slide_num = int(stem.split("_", 1)[1])
            except ValueError:
                continue
            key = f"sl_{slide_num:02d}"
            index.setdefault(key, path)
            continue
        for part in stem.split("__"):
            if part.startswith("sl_"):
                index.setdefault(part[:5], path)
    return index


def _panel_vote(item: dict[str, Any], name: str) -> str:
    votes = item.get("votes")
    if isinstance(votes, dict):
        vote = str(votes.get(name, "")).lower()
        if vote in {"pass", "fail"}:
            return vote
    return "unknown"


def _known_vote(row: dict[str, Any], key: str) -> bool:
    return row.get(key) in {"pass", "fail"}


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _profile_render(
    *,
    slide_id: str,
    render: Path,
    base_url: str,
    baseline_item: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verdict = qwen_judge_adapter(render, None, base_url=base_url)
    gemini = _panel_vote(baseline_item or {}, "gemini")
    claude = _panel_vote(baseline_item or {}, "claude")
    row = {
        "slide_id": slide_id,
        "render_path": str(render),
        "baseline_present": baseline_item is not None,
        "qwen_verdict": verdict.verdict,
        "qwen_failure_modes": verdict.failure_modes,
        "qwen_status": verdict.status,
        "gemini_verdict": gemini,
        "claude_verdict": claude,
        "qwen_agrees_gemini": verdict.verdict == gemini if gemini != "unknown" else None,
        "qwen_agrees_claude": verdict.verdict == claude if claude != "unknown" else None,
    }
    anchor_result = {
        "qwen_verdict": verdict.verdict,
        "failure_modes": verdict.failure_modes,
        "defect_detected": verdict.verdict == "fail",
        "render_path": str(render),
        "baseline_present": baseline_item is not None,
    }
    return row, anchor_result


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    items = _load_items(args.panel_json)
    renders = _render_index(args.deck_dir)
    per_slide: list[dict[str, Any]] = []
    anchor_results: dict[str, dict[str, Any]] = {}
    for item in items:
        slide_id = str(item.get("slide_id") or "")
        slide_prefix = slide_id[:5]
        render = renders.get(slide_prefix)
        if render is None:
            continue
        row, anchor_result = _profile_render(
            slide_id=slide_id,
            render=render,
            base_url=args.base_url,
            baseline_item=item,
        )
        per_slide.append(row)
        if slide_prefix in ANCHOR_PREFIXES:
            anchor_results[slide_prefix] = anchor_result
        if args.limit and len(per_slide) >= args.limit:
            break
    if not args.limit:
        for slide_prefix in ANCHOR_PREFIXES:
            if slide_prefix in anchor_results:
                continue
            render = renders.get(slide_prefix)
            if render is None:
                continue
            row, anchor_result = _profile_render(
                slide_id=slide_prefix,
                render=render,
                base_url=args.base_url,
                baseline_item=None,
            )
            per_slide.append(row)
            anchor_results[slide_prefix] = anchor_result
    slide_count = len(per_slide)
    baseline_slide_count = sum(1 for row in per_slide if row.get("baseline_present"))
    anchor_slide_count = len(anchor_results)
    qwen_pass = sum(1 for row in per_slide if row["qwen_verdict"] == "pass")
    gemini_known = sum(1 for row in per_slide if _known_vote(row, "gemini_verdict"))
    claude_known = sum(1 for row in per_slide if _known_vote(row, "claude_verdict"))
    gemini_pass = sum(
        1 for row in per_slide if row["gemini_verdict"] == "pass"
    )
    claude_pass = sum(
        1 for row in per_slide if row["claude_verdict"] == "pass"
    )
    qwen_agrees_gemini_denominator = sum(
        1 for row in per_slide if row["qwen_agrees_gemini"] is not None
    )
    qwen_agrees_claude_denominator = sum(
        1 for row in per_slide if row["qwen_agrees_claude"] is not None
    )
    qwen_agrees_gemini = sum(
        1 for row in per_slide if row["qwen_agrees_gemini"] is True
    )
    qwen_agrees_claude = sum(
        1 for row in per_slide if row["qwen_agrees_claude"] is True
    )
    anchor_defect_detected = any(
        result.get("defect_detected") for result in anchor_results.values()
    )
    missing_anchor_prefixes = [
        prefix for prefix in ANCHOR_PREFIXES if prefix not in anchor_results
    ]
    return {
        "profiled_at": datetime.now(UTC).isoformat(),
        "model": QWEN_MODEL,
        "base_url": args.base_url,
        "slide_count": slide_count,
        "baseline_slide_count": baseline_slide_count,
        "anchor_slide_count": anchor_slide_count,
        "qwen_accept_rate": _pct(qwen_pass, slide_count),
        "gemini_accept_rate": _pct(gemini_pass, gemini_known),
        "claude_accept_rate": _pct(claude_pass, claude_known),
        "qwen_agrees_gemini_pct": _pct(
            qwen_agrees_gemini,
            qwen_agrees_gemini_denominator,
        ),
        "qwen_agrees_claude_pct": _pct(
            qwen_agrees_claude,
            qwen_agrees_claude_denominator,
        ),
        "anchor_results": anchor_results,
        "missing_anchor_prefixes": missing_anchor_prefixes,
        "anchor_defect_detected": anchor_defect_detected,
        "per_slide": per_slide,
        "panel11_precondition_satisfied": bool(
            anchor_defect_detected
            and not missing_anchor_prefixes
            and anchor_slide_count >= len(ANCHOR_PREFIXES)
            and slide_count >= 3
        ),
    }


def _managed_profile(base_url: str, *, startup_timeout_s: float) -> LocalRuntimeProfile:
    return LocalRuntimeProfile(
        profile_id="v2_5_qwen_vision_heavy",
        runtime_kind="mlx",
        base_url=base_url,
        manage_server=True,
        optional_deps=["mlx", "mlx_vlm"],
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=600.0,
        max_concurrency=1,
        max_batch_size=1,
        lock_scope="heavy",
        hardware_fit_status="pending",
        approval_status="benchmark_pending",
        model_id=QWEN_MODEL,
    )


def main() -> int:
    args = parse_args()
    manager = ManagedMlxVisionServer() if args.manage_server else None
    try:
        if manager is not None:
            manager.start(
                _managed_profile(args.base_url, startup_timeout_s=args.startup_timeout_s),
                QWEN_MODEL,
                task_id="qwen_solo_profile",
            )
        payload = _profile(args)
    except (VisionRuntimeUnavailableError, ModelGovernanceError) as exc:
        print(f"HALT: Qwen-VL server unavailable: {exc}", file=sys.stderr)
        return 1
    finally:
        if manager is not None:
            manager.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
