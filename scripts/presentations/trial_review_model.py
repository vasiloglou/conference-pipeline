#!/usr/bin/env python3
"""Trial a candidate model as a presentation reviewer, against recorded evidence.

Allowing a model for a governed task removes a code limit; it says nothing about
whether the model is any good at it. This measures that, cheaply and without
touching any production stage receipt:

``replay``   re-judge recorded decks' rendered slides with the candidate and compare
             with the baseline report already on disk (the production reviewer's).
``seeded``   plant known defects in slides the baseline passed, then ask each model
             (as visual-QC reviewer and as figure-panel judge) whether it notices.
             The ground truth is by construction, so it is not circular.

Every call is receipted under its own run id (``storage/workflow-runs/<run-id>/
provider-calls``), separate from the production run, and the trial halts once the
receipted spend reaches ``--budget-usd``.

    .venv/bin/python scripts/presentations/trial_review_model.py replay \\
        --workspace conferences/icml2026 --decks 01_ai_for_science_physics_discovery 02_agentic_ai \\
        --model openrouter/moonshotai/kimi-k3 --run-id k3-trial --out-dir storage/analysis/k3-trial
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "presentations"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from PIL import Image, ImageChops, ImageDraw  # noqa: E402

from conference_pipeline.model_governance.llm_client import resolve_selection  # noqa: E402
from conference_pipeline.presentation import vision_qc  # noqa: E402
from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402

BASELINE_REPORT = Path("assets/generated/visual_llm_qc_report.json")
SCREENSHOTS = Path("assets/generated/qc/screenshots")


# ---- plumbing ---------------------------------------------------------------

def load_env() -> None:
    """Load the repo-local .env (CLAUDE.md: provider keys live only there)."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def spent_usd(workspace: Path, run_id: str) -> float:
    total = 0.0
    root = workspace / "storage" / "workflow-runs" / run_id / "provider-calls"
    for path in root.rglob("*.json") if root.exists() else []:
        try:
            total += float(json.loads(path.read_text()).get("estimated_cost_usd") or 0.0)
        except (OSError, ValueError):
            continue
    return round(total, 4)


def check_budget(workspace: Path, run_id: str, budget: float, *, what: str) -> None:
    spent = spent_usd(workspace, run_id)
    if spent >= budget:
        raise SystemExit(f"halted before {what}: receipted spend ${spent:.2f} reached the ${budget:.2f} cap")
    print(f"  [{what}] spent so far ${spent:.2f} of ${budget:.2f}", file=sys.stderr)


def _slide_key(slide: dict[str, Any]) -> int:
    return int(slide["slide_number"])


# ---- replay -----------------------------------------------------------------

def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = {_slide_key(s): s for s in baseline["slides"]}
    cand = {_slide_key(s): s for s in candidate["slides"]}
    rows, binary_agree, exact_agree = [], 0, 0
    for number in sorted(set(base) & set(cand)):
        b, c = base[number], cand[number]
        b_flag, c_flag = b["status"] != "pass", c["status"] != "pass"
        binary_agree += b_flag == c_flag
        exact_agree += b["status"] == c["status"]
        rows.append({
            "slide": number,
            "baseline": b["status"], "candidate": c["status"],
            "baseline_issues": [(i["check_id"], i["severity"]) for i in b["issues"]],
            "candidate_issues": [(i["check_id"], i["severity"], i["description"][:160]) for i in c["issues"]],
            "agree_flagged_or_not": b_flag == c_flag,
        })
    n = len(rows)
    return {
        "slides": n,
        "binary_agreement": binary_agree, "exact_status_agreement": exact_agree,
        "candidate_only_flags": [r["slide"] for r in rows if r["candidate"] != "pass" and r["baseline"] == "pass"],
        "baseline_only_flags": [r["slide"] for r in rows if r["baseline"] != "pass" and r["candidate"] == "pass"],
        "candidate_provider_failures": [
            r["slide"] for r in rows
            if any(i[0] == "vision_llm_visual_qc_parseable" for i in r["candidate_issues"])
        ],
        "rows": rows,
    }


def cmd_replay(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    provider, model = resolve_selection(vision_qc.MODEL_TASK_ID, args.provider, args.model)
    out_dir = workspace / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"provider": provider, "model": model, "run_id": args.run_id, "decks": {}}
    for deck in args.decks:
        check_budget(workspace, args.run_id, args.budget_usd, what=f"replay {deck}")
        deck_dir = workspace / "decks" / deck
        report_path = out_dir / f"replay_{deck}.{model.rsplit('/', 1)[-1]}.json"
        started = time.monotonic()
        candidate = vision_qc.inspect_screenshots_with_llm(
            deck_dir / SCREENSHOTS, output_report=report_path, workspace=workspace,
            provider=provider, model=model, run_id=args.run_id, stage_id="visual_qc_trial",
            max_slides=args.max_slides,
        )
        baseline = json.loads((deck_dir / BASELINE_REPORT).read_text())
        comparison = compare_reports(baseline, candidate)
        comparison["wall_seconds"] = round(time.monotonic() - started, 1)
        comparison["candidate_report_status"] = candidate["status"]
        summary["decks"][deck] = comparison
        print(f"{deck}: {comparison['binary_agreement']}/{comparison['slides']} agree on flagged-or-not; "
              f"candidate-only flags {comparison['candidate_only_flags']}, "
              f"baseline-only flags {comparison['baseline_only_flags']}, "
              f"provider failures {comparison['candidate_provider_failures']}", file=sys.stderr)
    summary["spent_usd"] = spent_usd(workspace, args.run_id)
    write_json_atomic(out_dir / f"replay_summary.{model.rsplit('/', 1)[-1]}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "decks"}, indent=2))
    return 0


# ---- seeded defects -----------------------------------------------------------

def inject_oversized_arrow(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """A saturated block with an arrowhead dropped on the figure."""

    out = img.copy()
    w, h = out.size
    half = max(6, int(100 * strength))
    cx, cy = w // 2, h // 2
    draw = ImageDraw.Draw(out)
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(255, 0, 0))
    draw.polygon([(cx, cy - half - int(40 * strength)), (cx - half // 2, cy - half), (cx + half // 2, cy - half)],
                 fill=(255, 0, 0))
    return out


def inject_text_clip(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """A black bar over the bottom of the slide, hiding what was there."""

    out = img.copy()
    w, h = out.size
    ImageDraw.Draw(out).rectangle([0, int(h * (1 - 0.15 * strength)), w, h], fill=(0, 0, 0))
    return out


def inject_off_palette(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """A flat colour outside the brand palette."""

    out = img.copy()
    w, h = out.size
    rw, rh = int(w * 0.25 * strength), int(h * 0.25 * strength)
    ImageDraw.Draw(out).rectangle([w // 2 - rw, h // 2 - rh, w // 2 + rw, h // 2 + rh], fill=(0, 255, 128))
    return out


def inject_text_overlap(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Ghost the title band over itself, shifted: text drawn on top of text."""

    out = img.copy()
    w, h = out.size
    band = out.crop((0, 0, w, int(h * 0.24)))
    shifted = ImageChops.offset(band, max(1, int(11 * strength)), max(1, int(7 * strength)))
    out.paste(ImageChops.darker(band, shifted), (0, 0))
    return out


def inject_low_contrast(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Wash the middle of the slide out toward its own average colour."""

    out = img.copy()
    w, h = out.size
    box = (int(w * 0.12), int(h * 0.30), int(w * 0.88), int(h * 0.80))
    region = out.crop(box)
    flat = Image.new("RGB", region.size, tuple(int(v) for v in region.resize((1, 1)).getpixel((0, 0))))
    out.paste(Image.blend(region, flat, min(0.9, 0.3 + 0.55 * strength)), box)
    return out


def inject_edge_clip(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Cut the right edge off in the background colour: content clipped at the slide edge."""

    out = img.copy()
    w, h = out.size
    ImageDraw.Draw(out).rectangle([int(w * (1 - 0.14 * strength)), 0, w, h], fill=out.getpixel((3, 3)))
    return out


INJECTIONS = {
    # name: (function, class). "hard" defects are unambiguous blockers at full
    # strength; "subtle" ones are real but easier to dismiss.
    "oversized_arrow": (inject_oversized_arrow, "hard"),
    "text_clip_bar": (inject_text_clip, "hard"),
    "edge_clip": (inject_edge_clip, "hard"),
    "text_overlap": (inject_text_overlap, "subtle"),
    "low_contrast": (inject_low_contrast, "subtle"),
    "off_palette": (inject_off_palette, "subtle"),
}
STRENGTHS = {"blatant": ("seeded", 1.0), "mild": ("seeded_mild", 0.25)}


def pick_clean_slides(workspace: Path, decks: list[str], per_deck: int) -> list[tuple[str, int, Path]]:
    """Evenly spaced slides each deck's baseline passed (a defect there is the only defect)."""

    picks: list[tuple[str, int, Path]] = []
    for deck in decks:
        deck_dir = workspace / "decks" / deck
        baseline = json.loads((deck_dir / BASELINE_REPORT).read_text())
        clean = [s for s in baseline["slides"] if s["status"] == "pass"]
        step = max(1, len(clean) // per_deck)
        for slide in clean[step // 2 :: step][:per_deck]:
            picks.append((deck, int(slide["slide_number"]), workspace / slide["screenshot_path"]))
    return picks


def cmd_seed_build(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    set_name, strength = STRENGTHS[args.strength]
    out_dir = workspace / args.out_dir / set_name
    shots = out_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for deck, number, source in pick_clean_slides(workspace, args.decks, args.per_deck):
        image = Image.open(source).convert("RGB")
        original = f"{deck[:2]}_s{number:02d}__clean.png"
        image.save(shots / original)
        cases.append({"case_id": original[:-4], "deck": deck, "slide": number, "injection": None,
                      "class": "control", "expected_flagged": False, "file": original})
        for name, (function, severity) in INJECTIONS.items():
            filename = f"{deck[:2]}_s{number:02d}__{name}.png"
            function(image, strength).save(shots / filename)
            cases.append({"case_id": filename[:-4], "deck": deck, "slide": number, "injection": name,
                          "class": severity, "expected_flagged": True, "file": filename})
    write_json_atomic(out_dir / "manifest.json", {"cases": cases})
    print(f"built {len(cases)} cases ({sum(c['expected_flagged'] for c in cases)} seeded, "
          f"{sum(not c['expected_flagged'] for c in cases)} controls) in {shots}")
    return 0


def score_seeded(cases: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """verdicts: case_id -> {"flagged": bool, "blocker": bool, "failed": bool}."""

    def rate(selected: list[dict[str, Any]], key: str) -> str:
        hits = sum(bool(verdicts[c["case_id"]][key]) for c in selected)
        return f"{hits}/{len(selected)}"

    seeded = [c for c in cases if c["expected_flagged"]]
    controls = [c for c in cases if not c["expected_flagged"]]
    by_injection = {
        name: rate([c for c in seeded if c["injection"] == name], "flagged") for name in INJECTIONS
    }
    return {
        "detected_any_flag": rate(seeded, "flagged"),
        "detected_as_blocker": rate(seeded, "blocker"),
        "hard_defects_flagged": rate([c for c in seeded if c["class"] == "hard"], "flagged"),
        "subtle_defects_flagged": rate([c for c in seeded if c["class"] == "subtle"], "flagged"),
        "by_injection": by_injection,
        "controls_wrongly_flagged": rate(controls, "flagged"),
        "call_failures": sum(bool(v["failed"]) for v in verdicts.values()),
    }


def cmd_seed_run(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    provider, model = resolve_selection(vision_qc.MODEL_TASK_ID, args.provider, args.model)
    seeded_dir = workspace / args.out_dir / args.set
    cases = json.loads((seeded_dir / "manifest.json").read_text())["cases"]
    check_budget(workspace, args.run_id, args.budget_usd, what=f"seeded visual QC ({model}, {args.set})")
    slug = model.rsplit("/", 1)[-1]
    report = vision_qc.inspect_screenshots_with_llm(
        seeded_dir / "screenshots", output_report=seeded_dir / f"visual_qc.{slug}.json",
        workspace=workspace, provider=provider, model=model, run_id=args.run_id, stage_id="visual_qc_seeded_trial",
    )
    by_file = {Path(s["screenshot_path"]).name: s for s in report["slides"]}
    verdicts = {}
    for case in cases:
        slide = by_file[case["file"]]
        failed = any(i["check_id"] == "vision_llm_visual_qc_parseable" for i in slide["issues"])
        verdicts[case["case_id"]] = {
            "flagged": slide["status"] != "pass" and not failed,
            "blocker": slide["status"] == "fail" and not failed,
            "failed": failed,
            "issues": [(i["check_id"], i["severity"]) for i in slide["issues"]],
        }
    result = {"provider": provider, "model": model, "role": "visual_qc",
              "score": score_seeded(cases, verdicts), "verdicts": verdicts,
              "spent_usd_so_far": spent_usd(workspace, args.run_id)}
    write_json_atomic(seeded_dir / f"seeded_result.visual_qc.{slug}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "verdicts"}, indent=2))
    return 0


def cmd_seed_panel(args: argparse.Namespace) -> int:
    """The same seeded slides, judged as figures by the figure-panel judge."""

    import figure_quality_panel as panel

    workspace = args.workspace.resolve()
    seeded_dir = workspace / args.out_dir / args.set
    cases = json.loads((seeded_dir / "manifest.json").read_text())["cases"]
    provider, model = resolve_selection(panel.PANEL_TASK_ID, args.provider, args.model)
    check_budget(workspace, args.run_id, args.budget_usd, what=f"seeded panel judge ({model}, {args.set})")

    def judge(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        verdict = panel.api_judge_adapter(
            seeded_dir / "screenshots" / case["file"], None, model=model, provider=provider,
        )
        failed = verdict.status != "ok"
        return case["case_id"], {
            "flagged": verdict.verdict == "fail" and not failed,
            "blocker": verdict.verdict == "fail" and not failed,
            "failed": failed,
            "issues": verdict.failure_modes,
            "rationale": verdict.rationale[:240],
        }

    with provider_receipt_context(workspace_root=workspace, run_id=args.run_id,
                                  stage_id="figure_panel_seeded_trial", task_id=panel.PANEL_TASK_ID,
                                  caller=__file__):
        with ThreadPoolExecutor(max_workers=4) as pool:
            verdicts = dict(pool.map(judge, cases))
    result = {"provider": provider, "model": model, "role": "figure_panel_judge",
              "score": score_seeded(cases, verdicts), "verdicts": verdicts,
              "spent_usd_so_far": spent_usd(workspace, args.run_id)}
    write_json_atomic(seeded_dir / f"seeded_result.panel.{model.rsplit('/', 1)[-1]}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "verdicts"}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workspace", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, required=True, help="relative to the workspace")
        p.add_argument("--run-id", required=True, help="receipts go under storage/workflow-runs/<run-id>")
        p.add_argument("--budget-usd", type=float, default=5.0)
        p.add_argument("--provider")
        p.add_argument("--model")
        p.add_argument("--set", default="seeded", choices=[name for name, _ in STRENGTHS.values()],
                       help="which seeded set to judge")

    replay = sub.add_parser("replay", help="re-judge recorded decks, compare with the baseline report")
    common(replay)
    replay.add_argument("--decks", nargs="+", required=True)
    replay.add_argument("--max-slides", type=int)
    replay.set_defaults(func=cmd_replay)

    build = sub.add_parser("seed-build", help="plant known defects in slides the baseline passed")
    build.add_argument("--workspace", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--decks", nargs="+", required=True)
    build.add_argument("--per-deck", type=int, default=3)
    build.add_argument("--strength", choices=sorted(STRENGTHS), default="blatant")
    build.set_defaults(func=cmd_seed_build)

    run = sub.add_parser("seed-run", help="judge the seeded slides as visual QC")
    common(run)
    run.set_defaults(func=cmd_seed_run)

    pan = sub.add_parser("seed-panel", help="judge the seeded slides as the figure-panel judge")
    common(pan)
    pan.set_defaults(func=cmd_seed_panel)

    args = parser.parse_args(argv)
    load_env()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
