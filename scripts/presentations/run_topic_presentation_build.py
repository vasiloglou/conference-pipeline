#!/usr/bin/env python3
"""Resumable per-topic presentation build orchestrator (v2.3 Phase 02).

Drives the runbook stages (source pack -> ... -> video-sync) for one topic
deck with per-stage state, gate checks between stages, the render_spec
delivery bootstrap built in, and the workspace-join path conventions encoded
once (see `.planning/workstreams/v2-3/phases/01-topic-presentation-build-
runbook/01-RUNBOOK.md`).

State lives at `<deck_dir>/orchestrator_state.json`; a resumed run skips
stages recorded `done` unless `--force-from` rewinds them. Stage logs go to
`<deck_dir>/orchestrator_logs/<stage>.log`.

The produced cross-AI review and human approval are intentionally NOT part of
the default run; `--with-produced-review` opts in.

Example:
    PYTHONPATH=src .venv/bin/python scripts/presentations/run_topic_presentation_build.py \
      --workspace conferences/iclr2026 --deck-id 02_agentic_systems \
      --final-report-url "https://drive.google.com/file/d/<id>"
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.model_governance import model_for_alias  # noqa: E402


def _opus_model() -> str:
    """The Phase 16 figure gate pins the Opus tier; read it from the catalog."""
    return model_for_alias("opus")

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_SVG_STACK_RESEARCH = (
    ".planning/workstreams/v2-1/phases/16-brand-locked-editorial-topic-4-canary-rebuild/"
    "16-SVG-STACK-RESEARCH.md"
)
SCRIPTS = PROJECT_ROOT / "scripts" / "presentations"


def _config_workflow_run_id(workspace: Path) -> str:
    """Return ``workflow.run_id`` from the workspace ``config.yaml`` (else "")."""
    config_path = workspace / "config.yaml"
    if not config_path.is_file():
        return ""
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - best-effort default resolution
        return ""
    workflow = payload.get("workflow") if isinstance(payload, dict) else None
    if isinstance(workflow, dict):
        run_id = workflow.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
    return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class BuildContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workspace = Path(args.workspace)
        if not self.workspace.is_absolute():
            self.workspace = PROJECT_ROOT / self.workspace
        self.deck_id = args.deck_id
        self.deck_rel = f"decks/{self.deck_id}"
        self.deck_dir = self.workspace / "decks" / self.deck_id
        self.deck_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.deck_dir / "orchestrator_logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.state_path = self.deck_dir / "orchestrator_state.json"
        self.state = self._load_state()
        self.env = self._build_env(args.env_file)
        self.topic_title = args.topic_title or self._derive_topic_title()

    # -- state ------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("stages"), dict):
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        return {"schema_version": 1, "deck_id": self.deck_id, "stages": {}}

    def _save_state(self) -> None:
        self.state["updated_at"] = _now()
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def stage_status(self, name: str) -> str:
        return str((self.state["stages"].get(name) or {}).get("status") or "pending")

    def mark(self, name: str, status: str, detail: str = "") -> None:
        entry = self.state["stages"].setdefault(name, {})
        entry["status"] = status
        entry["detail"] = detail
        entry["at"] = _now()
        self._save_state()

    def rewind_after(self, name: str, order: list[str]) -> None:
        """When a stage re-runs, everything after it is stale."""
        if name not in order:
            return
        for later in order[order.index(name) + 1 :]:
            if self.stage_status(later) == "done":
                self.mark(later, "stale", f"upstream stage {name} re-ran")

    # -- env / subprocess --------------------------------------------------
    def _build_env(self, env_file: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        if env_file and env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                env.setdefault(key, value.strip().strip('"').strip("'"))
        env.setdefault("ENGAGING_NARRATION_HOST_A", self.args.host_a)
        env.setdefault("ENGAGING_NARRATION_HOST_B", self.args.host_b)
        return env

    def run(self, stage: str, cmd: list[str]) -> None:
        log_path = self.logs_dir / f"{stage}.log"
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable[:160]}{'...' if len(printable) > 160 else ''}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {_now()} ===\n{printable}\n")
            log.flush()
            result = subprocess.run(
                [sys.executable, *cmd] if cmd and str(cmd[0]).endswith(".py") else cmd,
                cwd=PROJECT_ROOT,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-15:])
            raise RuntimeError(
                f"stage {stage} command failed (exit {result.returncode}). Log tail:\n{tail}"
            )

    def run_parallel(self, stage: str, labelled_cmds: list[tuple[str, list[str]]]) -> None:
        """Run several independent commands concurrently (each to its own log).
        Raises if any one fails, after letting the others finish."""
        procs: list[tuple[str, Path, Any, Any]] = []
        for label, cmd in labelled_cmds:
            log_path = self.logs_dir / f"{label}.log"
            printable = " ".join(str(c) for c in cmd)
            print(f"  $[{label}] {printable[:150]}{'...' if len(printable) > 150 else ''}")
            log = log_path.open("a", encoding="utf-8")
            log.write(f"\n=== {_now()} ===\n{printable}\n")
            log.flush()
            argv = [sys.executable, *cmd] if cmd and str(cmd[0]).endswith(".py") else cmd
            proc = subprocess.Popen(
                argv, cwd=PROJECT_ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT
            )
            procs.append((label, log_path, proc, log))
        failures: list[str] = []
        for label, log_path, proc, log in procs:
            code = proc.wait()
            log.close()
            if code != 0:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-12:])
                failures.append(f"{label} (exit {code}). Log tail:\n{tail}")
        if failures:
            raise RuntimeError(f"stage {stage} had {len(failures)} failed command(s):\n" + "\n".join(failures))

    def _derive_topic_title(self) -> str:
        pack = self.workspace / "analysis" / "narrative_source_packs" / f"{self.deck_id}.json"
        if pack.is_file():
            try:
                payload = _load_json(pack)
                title = str(
                    payload.get("topic_title") or payload.get("title") or ""
                ).strip()
                if title:
                    return title
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        analysis = self.workspace / "analysis" / f"{self.deck_id}.md"
        if analysis.is_file():
            # The analysis report's H1 is the canonical topic name.
            for line in analysis.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
                if line.startswith("# "):
                    return line[2:].strip()
        words = self.deck_id.split("_")[1:] or [self.deck_id]
        return " ".join(words).replace("-", " ").strip().capitalize()

    # -- path helpers -------------------------------------------------------
    def rel(self, name: str) -> str:
        """Workspace-relative path (for scripts that JOIN --workspace)."""
        return f"{self.deck_rel}/{name}"

    def full(self, name: str) -> str:
        """Repo-root path (for scripts that do NOT join --workspace)."""
        return str(self.deck_dir / name)

    def ids(self) -> list[str]:
        a = self.args
        return [
            "--workspace", str(self.workspace),
            "--presentation-id", self.deck_id,
            "--presentation-type", a.presentation_type,
            "--topic", self.deck_id,
        ]


# --------------------------------------------------------------------------
# Stage implementations. Each takes ctx and raises on gate failure.
# --------------------------------------------------------------------------

def stage_source_pack(ctx: BuildContext) -> None:
    ctx.run("source_pack", [
        str(SCRIPTS / "build_deck_source_pack.py"), *ctx.ids(),
        "--extraction-run-id", ctx.args.extraction_run_id,
    ])


def stage_corpus_inventory(ctx: BuildContext) -> None:
    deck = ctx.deck_id
    candidates = [f"analysis/{deck}.md"]
    patterns = [
        f"analysis/deep_dive_topics/{deck}__dd0*__global.md",
        f"analysis/deep_dive_topics/{deck}__dd0*__heuristic.md",
        f"analysis/exhaustive_summaries/{deck}__papers.json",
        f"analysis/exhaustive_summaries/{deck}__papers.md",
        f"analysis/exhaustive_summaries/{deck}__transcripts.json",
        f"analysis/exhaustive_summaries/{deck}__transcripts.md",
        f"analysis/narrative_source_packs/{deck}.json",
        f"analysis/provenance/{deck}.json",
        f"analysis/provenance/{deck}.md",
        f"analysis/provenance/query_plans/{deck}.json",
        f"analysis/topic_reference_candidate_resolution/{deck}*.*",
        f"analysis/topic_reference_ledgers/{deck}*.json",
        f"analysis/topic_reference_repairs/{deck}*.*",
        f"analysis/topic_reference_validation/{deck}*.*",
        f"decks/{deck}/extracted_content.json",
        f"decks/{deck}/ranked_papers.json",
        f"decks/{deck}/source_attributions.json",
        f"decks/{deck}/source_pack_ranked_papers.json",
        f"decks/{deck}/source_pack_receipt.json",
    ]
    for pattern in patterns:
        for match in sorted(glob.glob(str(ctx.workspace / pattern))):
            candidates.append(str(Path(match).relative_to(ctx.workspace)))
    inputs: list[str] = []
    for item in candidates:
        if (ctx.workspace / item).is_file():
            inputs.extend(["--source-input", item])
    if not inputs:
        raise RuntimeError("corpus inventory: no analysis-foundation source inputs found")
    ctx.run("corpus_inventory", [
        str(SCRIPTS / "build_presentation_source_corpus_inventory.py"),
        "--workspace", str(ctx.workspace), "--topic-id", ctx.deck_id,
        "--deck-dir", ctx.deck_rel, *inputs,
        "--output-json", ctx.rel("source_corpus_inventory.json"),
        "--receipt-path", ctx.rel("source_corpus_inventory_receipt.json"), "--summary",
    ])


def stage_work_ledger(ctx: BuildContext) -> None:
    ctx.run("work_ledger", [
        str(SCRIPTS / "build_speaker_author_work_ledger.py"),
        "--workspace", str(ctx.workspace),
        "--inventory-json", ctx.rel("source_corpus_inventory.json"),
        "--output-json", ctx.rel("speaker_author_work_ledger.json"),
        "--receipt-path", ctx.rel("speaker_author_work_ledger_receipt.json"), "--summary",
    ])


def stage_source_repair(ctx: BuildContext) -> None:
    # A deck's reviewed repairs (an operator's recorded evidence that a
    # blocked source is what it says) must survive every rebuild.
    reviewed = ctx.deck_dir / "source_repair_reviewed.json"
    ctx.run("source_repair", [
        str(SCRIPTS / "repair_presentation_sources.py"),
        "--workspace", str(ctx.workspace),
        "--inventory-json", ctx.rel("source_corpus_inventory.json"),
        "--ledger-json", ctx.rel("speaker_author_work_ledger.json"),
        "--output-json", ctx.rel("source_repair_findings.json"),
        "--receipt-path", ctx.rel("source_repair_receipt.json"), "--summary",
        *(["--reviewed-repairs-json", ctx.rel("source_repair_reviewed.json")] if reviewed.is_file() else []),
    ])


def stage_cluster_plan(ctx: BuildContext) -> None:
    ctx.run("cluster_plan", [
        str(SCRIPTS / "build_research_cluster_plan.py"),
        "--workspace", str(ctx.workspace),
        "--inventory-json", ctx.rel("source_corpus_inventory.json"),
        "--ledger-json", ctx.rel("speaker_author_work_ledger.json"),
        "--repair-findings-json", ctx.rel("source_repair_findings.json"),
        "--output-json", ctx.rel("research_cluster_plan.json"),
        "--markdown-output", ctx.rel("research_cluster_plan.md"),
        "--receipt-path", ctx.rel("research_cluster_plan_receipt.json"), "--summary",
    ])


def stage_blueprint(ctx: BuildContext) -> None:
    if ctx.args.blueprint_mode != "editorial":
        raise RuntimeError("Phase 11 production builds require --blueprint-mode editorial")
    frame_path = ctx.deck_dir / "presentation_frame.json"
    if not frame_path.is_file():
        raise FileNotFoundError(f"presentation_frame.json not found: {frame_path}")
    ctx.run("blueprint", [
        str(SCRIPTS / "generate_editorial_blueprint.py"), *ctx.ids(),
        "--topic-title", ctx.topic_title,
        "--analysis-md", f"analysis/{ctx.deck_id}.md",
        "--exhaustive-papers", f"analysis/exhaustive_summaries/{ctx.deck_id}__papers.json",
        "--speaker-author-work-ledger", ctx.rel("speaker_author_work_ledger.json"),
        "--source-corpus-inventory", ctx.rel("source_corpus_inventory.json"),
        "--source-repair-findings", ctx.rel("source_repair_findings.json"),
        "--research-cluster-plan", ctx.rel("research_cluster_plan.json"),
        "--presentation-frame", ctx.rel("presentation_frame.json"),
        "--final-report-url", ctx.args.final_report_url,
        "--output-json", ctx.rel("presentation_blueprint.json"),
        "--output-md", ctx.rel("presentation_blueprint.md"),
        "--receipt-path", ctx.rel("presentation_blueprint_receipt.json"),
        "--provider-cost-approved",
    ])
    import hashlib

    receipt = _load_json(ctx.deck_dir / "presentation_blueprint_receipt.json")
    blueprint_sha = hashlib.sha256(
        (ctx.deck_dir / "presentation_blueprint.json").read_bytes()
    ).hexdigest()
    if receipt.get("blueprint_sha256") != blueprint_sha:
        raise RuntimeError("editorial blueprint receipt sha mismatch")
    coverage = receipt.get("coverage") or {}
    print(f"  editorial blueprint: {coverage}")


def stage_blueprint_review(ctx: BuildContext) -> None:
    # NOTE: review does NOT join --workspace; full paths are required.
    ctx.run("blueprint_review", [
        str(SCRIPTS / "review_presentation_blueprint.py"),
        "--presentation-id", ctx.deck_id,
        "--presentation-type", ctx.args.presentation_type,
        "--topic", ctx.deck_id,
        "--presentation-blueprint", ctx.full("presentation_blueprint.json"),
        "--blueprint-receipt", ctx.full("presentation_blueprint_receipt.json"),
        "--output-json", ctx.full("blueprint_review.json"),
        "--output-md", ctx.full("blueprint_review.md"),
        "--approve-for-draft-generation", "--approve-for-deck-contract-generation",
        "--approved-by", ctx.args.approved_by,
        "--llm-judge-approved",
        "--llm-judge-artifact", ctx.full("blueprint_llm_judge.md"),
    ])
    import hashlib

    review = _load_json(ctx.deck_dir / "blueprint_review.json")
    blueprint_sha = hashlib.sha256(
        (ctx.deck_dir / "presentation_blueprint.json").read_bytes()
    ).hexdigest()
    approval = (review.get("approval") or {}).get("status")
    if review.get("status") != "pass":
        raise RuntimeError(f"blueprint review status={review.get('status')}")
    if approval != "approved_for_deck_contract_generation":
        raise RuntimeError(f"blueprint review approval={approval}")
    if review.get("blueprint_sha256") != blueprint_sha:
        raise RuntimeError("blueprint review hash does not match blueprint on disk")


def stage_slides_draft(ctx: BuildContext) -> None:
    """Slides-agent draft (runbook 3a). The deck contract does not consume it,
    but the editorial storyboard derives its clusters from the drafted slides
    markdown."""
    blueprint = _load_json(ctx.deck_dir / "presentation_blueprint.json")
    display_title = str(blueprint.get("audience_title") or ctx.topic_title).strip()
    ctx.run("slides_draft", [
        str(SCRIPTS / "generate_slides_agent.py"), *ctx.ids(),
        "--from-blueprint", "--display-title", display_title,
        "--presentation-blueprint", ctx.rel("presentation_blueprint.json"),
        "--blueprint-receipt", ctx.rel("presentation_blueprint_receipt.json"),
        "--blueprint-review", ctx.rel("blueprint_review.json"),
        "--output-dir", ctx.deck_rel,
        "--output-slides-md", ctx.rel("slides.md"),
    ])
    if not (ctx.deck_dir / "slides.md").is_file():
        raise RuntimeError("slides_draft did not produce slides.md")


def stage_deck_contract(ctx: BuildContext) -> None:
    ctx.run("deck_contract", [
        str(SCRIPTS / "generate_deck_contract.py"), *ctx.ids(),
        "--presentation-blueprint", ctx.full("presentation_blueprint.json"),
        "--blueprint-receipt", ctx.full("presentation_blueprint_receipt.json"),
        "--blueprint-review", ctx.full("blueprint_review.json"),
        "--research-cluster-plan", ctx.full("research_cluster_plan.json"),
        "--speaker-author-work-ledger", ctx.full("speaker_author_work_ledger.json"),
        "--output", ctx.full("presentation_deck.json"),
        "--receipt-output", ctx.full("deck_contract_receipt.json"),
    ])


def stage_source_binding(ctx: BuildContext) -> None:
    ctx.run("source_binding", [
        str(SCRIPTS / "bind_presentation_sources.py"), *ctx.ids(),
        "--deck-contract", ctx.full("presentation_deck.json"),
        "--speaker-author-work-ledger", ctx.full("speaker_author_work_ledger.json"),
        "--research-cluster-plan", ctx.full("research_cluster_plan.json"),
        "--matched-papers",
        f"storage/paper-matching/{ctx.args.extraction_run_id}/{ctx.deck_id}/matched_papers.json",
        "--output", ctx.full("source_binding_report.json"),
    ])


def _ensure_clean_deck(ctx: BuildContext) -> None:
    """The bookends stage injects s-events/s-resources INTO the deck contract;
    any re-run of figures pass 1 or render_spec needs the bookend-free deck, so
    regenerate deck + binding (both deterministic) when bookends are present."""
    deck = _load_json(ctx.deck_dir / "presentation_deck.json")
    has_bookends = any(
        s.get("slide_kind") in ("events", "resources") for s in deck.get("slides") or []
    )
    if has_bookends:
        print("  deck contains injected bookends; regenerating clean deck + binding")
        stage_deck_contract(ctx)
        stage_source_binding(ctx)


def _figures_common(ctx: BuildContext) -> list[str]:
    return [
        *ctx.ids(),
        "--deck-contract", ctx.rel("presentation_deck.json"),
        "--source-binding-report", ctx.rel("source_binding_report.json"),
        "--visual-plan", ctx.rel("visual_plan.json"),
        "--asset-inventory", ctx.rel("asset_inventory.json"),
        "--generated-dir", ctx.rel("assets/generated/phase16"),
        "--figure-specs-output", ctx.rel("figure_specs.json"),
    ]


def _purge_stale_figures(ctx: BuildContext) -> None:
    """Remove figure SVG/code files from earlier blueprints whose slide ids no
    longer exist — they pollute validation sweeps and the richness gate."""
    deck = _load_json(ctx.deck_dir / "presentation_deck.json")
    valid = {str(s.get("slide_id")) for s in deck.get("slides") or []}
    removed = 0
    for pattern in ("assets/generated/phase16/*.svg", "assets/generated/phase16/code/*.py"):
        for path in ctx.deck_dir.glob(pattern):
            if not any(path.name.startswith(prefix) for prefix in valid):
                path.unlink()
                removed += 1
    if removed:
        print(f"  purged {removed} stale figure files")


def stage_figures_pass1(ctx: BuildContext) -> None:
    _ensure_clean_deck(ctx)
    _purge_stale_figures(ctx)
    ctx.run("figures_pass1", [
        str(SCRIPTS / "generate_programmatic_figures.py"), *_figures_common(ctx),
    ])


def stage_figures_pass2(ctx: BuildContext) -> None:
    cmd = [
        str(SCRIPTS / "generate_programmatic_figures.py"), *_figures_common(ctx),
        "--svg-stack-research", ctx.args.svg_stack_research,
    ]
    if (ctx.deck_dir / "qc_figure_feedback.json").is_file():
        cmd += ["--qc-feedback-json", ctx.rel("qc_figure_feedback.json")]
    if (ctx.deck_dir / "qc_figure_verdicts.json").is_file():
        cmd += ["--qc-verdicts-json", ctx.rel("qc_figure_verdicts.json")]
    cmd += [
        "--figure-code-dir", ctx.rel("assets/generated/phase16/code"),
        "--model-receipt-output", ctx.rel("figure_generation_model_receipt.json"),
        "--code-review-output", ctx.rel("figure_code_review_report.json"),
        "--provider", "anthropic", "--model", _opus_model(), "--provider-cost-approved",  # the Phase 16 gate pins the opus alias
        # Eight attempts: with the measured geometry gate a dense figure needs
        # more than five repair rounds to converge (each round fixes one defect
        # class; the cumulative feedback keeps earlier fixes in place).
        "--codegen-max-attempts", "8",
        "--allowed-renderer", "drawsvg", "--allowed-renderer", "graphviz",
        "--allowed-renderer", "altair-vl-convert", "--allowed-renderer", "pycairo",
        "--disallowed-default-renderer", "matplotlib",
        "--disallowed-default-renderer", "plotly",
        "--figure-workers", str(ctx.args.figure_workers),
    ]
    attempts = 0
    while True:
        attempts += 1
        ctx.run("figures_pass2", cmd)
        receipt = _load_json(ctx.deck_dir / "figure_generation_model_receipt.json")
        if receipt.get("status") == "pass":
            reused = receipt.get("reused_figure_count")
            print(f"  figures receipt pass (reused {reused} of {receipt.get('figure_count')})")
            return
        if attempts >= 2:
            raise RuntimeError(
                f"figure receipt status={receipt.get('status')} after {attempts} runs; "
                "inspect figure_generation_model_receipt.json figure_results"
            )
        # The per-figure reuse cache makes a retry cheap: accepted figures are
        # reused, only the best-effort ones regenerate -- seeded with the
        # measured geometry defects of the figure they are replacing so the
        # first attempt of the retry already knows what went wrong.
        seeded = write_geometry_figure_feedback(ctx, receipt)
        if seeded and "--qc-feedback-json" not in cmd:
            cmd += ["--qc-feedback-json", ctx.rel("qc_figure_feedback.json")]
        # Final seeded run: a figure that still will not converge is accepted
        # with a larger recorded residue rather than stalling the deck; the
        # pixel-level review (and its own regeneration loop) is the authority.
        if "--best-effort-max-weight" not in cmd:
            cmd += ["--best-effort-max-weight", "8"]
        print(
            f"  figures receipt blocked; retrying with geometry feedback for {seeded} "
            "figure(s) (accepted figures will be reused)"
        )


def stage_validate_svgs(ctx: BuildContext) -> None:
    ctx.run("validate_svgs", [
        str(SCRIPTS / "validate_svgs.py"), *ctx.ids(),
        "--dir", ctx.rel("assets/generated/phase16"), "--json",
        "--report-output", ctx.rel("figure_svg_validation_report.json"),
    ])
    report = _load_json(ctx.deck_dir / "figure_svg_validation_report.json")
    errors = (report.get("summary") or {}).get("error")
    if errors:
        raise RuntimeError(f"SVG validation errors={errors}")


def stage_semantic_review(ctx: BuildContext) -> None:
    ctx.run("semantic_review", [
        str(SCRIPTS / "review_programmatic_figures.py"), *ctx.ids(),
        "--dir", ctx.rel("assets/generated/phase16"),
        "--asset-inventory", ctx.rel("asset_inventory.json"),
        "--report-output", ctx.rel("figure_semantic_review.json"),
        "--markdown-output", ctx.rel("figure_semantic_review.md"),
    ])
    report = _load_json(ctx.deck_dir / "figure_semantic_review.json")
    if report.get("status") != "pass":
        raise RuntimeError(f"figure semantic review status={report.get('status')}")


def stage_delivery_bootstrap(ctx: BuildContext) -> None:
    """render_spec requires deck-aligned delivery contracts that only really
    exist after the presenter-script stage; bootstrap stubs per deck slide."""
    deck = _load_json(ctx.deck_dir / "presentation_deck.json")
    slides = deck.get("slides") or []
    if not slides:
        raise RuntimeError("delivery bootstrap: deck has no slides")
    template_segment = {
        "segment_id": "seg-01", "slide_id": "", "slide_number": 1, "speaker": "stub",
        "start_seconds": 0.0, "end_seconds": 12.0, "duration_seconds": 12.0,
        "narration": "(stub)", "source_chips": [], "source_ids": [],
        "source_labels": [], "visual_asset_refs": [], "slide_kind": "content",
    }
    for name, style in (
        ("delivery_single_speaker.json", "single_speaker"),
        ("delivery_notebooklm.json", "notebooklm"),
    ):
        path = ctx.deck_dir / name
        payload = _load_json(path) if path.is_file() else {
            "schema_version": 1, "presentation_id": ctx.deck_id, "style": style,
            "approve_for_tts": False, "segments": [],
        }
        base = (payload.get("segments") or [template_segment])[0]
        segments = []
        clock = 0.0
        for index, slide in enumerate(slides):
            segment = copy.deepcopy(base)
            segment.update({
                "segment_id": f"seg-{index + 1:02d}",
                "slide_id": slide["slide_id"],
                "slide_number": index + 1,
                "start_seconds": clock, "end_seconds": clock + 12.0,
                "duration_seconds": 12.0,
                "narration": "(bootstrap stub — regenerated at presenter-script stage)",
                "source_ids": [], "source_labels": [], "visual_asset_refs": [],
                "slide_kind": slide.get("slide_kind", "content"),
            })
            segments.append(segment)
            clock += 12.0
        payload["segments"] = segments
        payload["total_duration_seconds"] = clock
        if isinstance(payload.get("timing_map"), dict):
            payload["timing_map"] = {s["slide_id"]: s["duration_seconds"] for s in segments}
        if isinstance(payload.get("slide_state_map"), dict):
            payload["slide_state_map"] = {s["slide_id"]: "stub" for s in segments}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  bootstrapped {len(slides)}-segment stub deliveries")


def stage_render_spec(ctx: BuildContext) -> None:
    _ensure_clean_deck(ctx)
    stage_delivery_bootstrap(ctx)  # deck may have just been regenerated
    ctx.run("render_spec", [
        str(SCRIPTS / "generate_render_spec.py"), *ctx.ids(),
        "--presentation-blueprint", ctx.rel("presentation_blueprint.json"),
        "--deck-contract", ctx.rel("presentation_deck.json"),
        "--source-binding-report", ctx.rel("source_binding_report.json"),
        "--visual-plan", ctx.rel("visual_plan.json"),
        "--asset-inventory", ctx.rel("asset_inventory.json"),
        "--figure-specs", ctx.rel("figure_specs.json"),
        "--figure-generation-model-receipt", ctx.rel("figure_generation_model_receipt.json"),
        "--figure-code-review", ctx.rel("figure_code_review_report.json"),
        "--figure-svg-validation", ctx.rel("figure_svg_validation_report.json"),
        "--figure-semantic-review", ctx.rel("figure_semantic_review.json"),
        "--output", ctx.rel("render_spec.json"),
    ])


def stage_bookends(ctx: BuildContext) -> None:
    ctx.run("bookends", [
        str(SCRIPTS / "add_bookend_slides.py"), "--deck-dir", str(ctx.deck_dir),
    ])


def stage_render_html(ctx: BuildContext) -> None:
    ctx.run("render_html", [
        str(SCRIPTS / "md_to_revealjs.py"), *ctx.ids(),
        "--render-spec", ctx.rel("render_spec.json"),
        "--output-html", ctx.rel("presentation.html"),
        "--slide-mapping", ctx.rel("presentation_slide_mapping.json"),
        "--require-brand-assets",
    ])


def stage_export_pdf(ctx: BuildContext) -> None:
    ctx.run("export_pdf", [
        str(SCRIPTS / "export_revealjs_pdf.py"), *ctx.ids(),
        "--render-spec", ctx.rel("render_spec.json"),
        "--input-html", ctx.rel("presentation.html"),
        "--output-pdf", ctx.rel("presentation.pdf"),
        "--slide-mapping", ctx.rel("presentation_slide_mapping.json"),
        "--frame-manifest", ctx.rel("slide_frame_manifest.json"),
        "--frames-dir", ctx.rel("rendered_frames"),
        "--contact-sheet", ctx.rel("render_contact_sheet.png"),
        "--receipt-output", ctx.rel("render_export_receipt.json"),
    ])
    receipt = _load_json(ctx.deck_dir / "render_export_receipt.json")
    if receipt.get("status") != "pass":
        raise RuntimeError(f"PDF export status={receipt.get('status')}")


def stage_screenshots(ctx: BuildContext) -> None:
    try:
        ctx.run("screenshots", [
            str(SCRIPTS / "inspect_rendered_slides.py"), *ctx.ids(),
            "--input-html", ctx.rel("presentation.html"),
            "--report-output", ctx.rel("assets/generated/qc/inspect_report.json"),
            "--screenshots-dir", ctx.rel("assets/generated/qc/screenshots"),
        ])
    except RuntimeError:
        # The inspector exits non-zero on heuristic findings (e.g. "source
        # chips are missing" on bookend slides, which have none by design).
        # The authoritative visual verdict is the visual_qc stage; here the
        # gate is that every slide actually produced a screenshot.
        pass
    report = _load_json(ctx.deck_dir / "assets/generated/qc/inspect_report.json")
    render_spec = _load_json(ctx.deck_dir / "render_spec.json")
    expected = len(render_spec.get("slides") or [])
    shots = len(list((ctx.deck_dir / "assets/generated/qc/screenshots").glob("*.png")))
    if expected and shots != expected:
        raise RuntimeError(f"screenshots: captured {shots}, expected {expected}")
    bookend_numbers = {
        index + 1
        for index, s in enumerate(render_spec.get("slides") or [])
        if s.get("slide_kind") in ("events", "resources")
    }
    real_errors = [
        e
        for e in report.get("errors") or []
        if not (
            e.get("check_id") == "source chips"
            and e.get("slide_number") in bookend_numbers
        )
    ]
    if real_errors:
        raise RuntimeError(f"screenshots inspector errors: {real_errors[:3]}")
    print(f"  {shots}/{expected} screenshots captured")


QC_RETRY_STAGES = (
    "figures_pass2", "validate_svgs", "semantic_review", "delivery_bootstrap",
    "render_spec", "bookends", "render_html", "export_pdf", "screenshots", "visual_qc",
)


def write_qc_figure_verdicts(ctx: BuildContext) -> None:
    """Pin pixel-judge verdicts to exact figure bytes: a figure that PASSED
    visual QC must not be churned by stricter deterministic estimators on the
    next figures run."""
    import hashlib

    report = _load_json(ctx.deck_dir / "assets/generated/visual_llm_qc_report.json")
    render_spec = _load_json(ctx.deck_dir / "render_spec.json")
    slides = render_spec.get("slides") or []
    verdicts: dict[str, dict[str, str]] = {}
    for entry in report.get("slides") or []:
        number = int(entry.get("slide_number") or 0)
        if not (1 <= number <= len(slides)):
            continue
        slide_id = str(slides[number - 1].get("slide_id") or "")
        if not slide_id.startswith("sl_"):
            continue
        svgs = sorted((ctx.deck_dir / "assets/generated/phase16").glob(f"{slide_id}-*.svg"))
        if not svgs:
            continue
        verdicts[slide_id] = {
            "status": str(entry.get("status") or ""),
            "svg_sha256": hashlib.sha256(svgs[0].read_bytes()).hexdigest(),
        }
    (ctx.deck_dir / "qc_figure_verdicts.json").write_text(
        json.dumps(verdicts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_semantic_figure_feedback(ctx: BuildContext) -> int:
    """Seed ``qc_figure_feedback.json`` with the semantic review's findings
    (required labels missing from the rendered SVG) so the flagged figures
    regenerate with the exact terms to restore."""

    report = _load_json(ctx.deck_dir / "figure_semantic_review.json")
    path = ctx.deck_dir / "qc_figure_feedback.json"
    feedback: dict[str, str] = {}
    if path.is_file():
        try:
            feedback = {str(k): str(v) for k, v in _load_json(path).items()}
        except Exception:  # noqa: BLE001
            feedback = {}
    by_slide: dict[str, list[str]] = {}
    for finding in report.get("findings") or []:
        if finding.get("severity") != "error":
            continue
        slide_id = str(finding.get("slide_id") or "")
        if slide_id:
            by_slide.setdefault(slide_id, []).append(f"{finding.get('check_id')}: {finding.get('message')}")
    for slide_id, messages in by_slide.items():
        note = "The semantic review of your previous figure found: " + "; ".join(messages[:4]) + \
            ". Restore every required label exactly as given while keeping the layout clean."
        feedback[slide_id] = (feedback[slide_id] + " " + note) if slide_id in feedback else note
    if by_slide:
        path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(by_slide)


def write_visual_qc_waiver(ctx: BuildContext, *, approved_by: str) -> dict:
    """Record an explicit operator waiver of residual visual-QC findings.

    The waiver names who accepted, when, which slides still fail and why, so a
    later reviewer can see exactly what was shipped past the pixel judge."""

    from datetime import datetime, timezone

    report = _load_json(ctx.deck_dir / "assets/generated/visual_llm_qc_report.json")
    failing = []
    for entry in report.get("slides") or []:
        if entry.get("status") != "fail":
            continue
        failing.append(
            {
                "slide_number": entry.get("slide_number"),
                "issues": [
                    {"check_id": i.get("check_id"), "severity": i.get("severity"), "description": i.get("description")}
                    for i in entry.get("issues") or []
                    if i.get("severity") in ("blocker", "warn")
                ],
            }
        )
    waiver = {
        "schema_version": 1,
        "waiver_type": "conference_pipeline.presentation.visual_qc_residual",
        "deck_id": ctx.deck_id,
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "qc_report": "assets/generated/visual_llm_qc_report.json",
        "qc_status": report.get("status"),
        "failing_slide_count": len(failing),
        "failing_slides": failing,
        "note": "Residual figure-layout findings accepted after the regeneration loop was exhausted; "
        "the deck proceeds to narration and video with these findings on record.",
    }
    (ctx.deck_dir / "visual_qc_waiver.json").write_text(
        json.dumps(waiver, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return waiver


def write_geometry_figure_feedback(ctx: BuildContext, receipt: dict) -> int:
    """Seed ``qc_figure_feedback.json`` with the measured geometry defects of
    every best-effort figure in the receipt (merging with any pixel-review
    feedback already present) so the regeneration starts from the defects
    instead of rediscovering them."""

    from conference_pipeline.presentation.svg_geometry import geometry_findings

    path = ctx.deck_dir / "qc_figure_feedback.json"
    feedback: dict[str, str] = {}
    if path.is_file():
        try:
            feedback = {str(k): str(v) for k, v in _load_json(path).items()}
        except Exception:  # noqa: BLE001 - a corrupt seed file is not worth failing the retry
            feedback = {}
    generated_dir = ctx.deck_dir / "assets/generated/phase16"
    # Feedback keyed by a slide id that no longer exists (a previous blueprint's
    # slides) is stale; drop it so the seed file describes the current deck.
    current_ids = {str(r.get("slide_id") or "") for r in receipt.get("figure_results") or []}
    feedback = {k: v for k, v in feedback.items() if k in current_ids}
    seeded = 0
    for result in receipt.get("figure_results") or []:
        if result.get("status") != "best_effort":
            continue
        slide_id = str(result.get("slide_id") or "")
        visual_id = str(result.get("visual_id") or "")
        if not slide_id:
            continue
        svg_candidates = sorted(generated_dir.glob(f"{slide_id}-*.svg")) if slide_id else []
        if not svg_candidates and visual_id:
            svg_candidates = sorted(generated_dir.glob(f"*{visual_id}.svg"))
        if not svg_candidates:
            continue
        try:
            findings = geometry_findings(svg_candidates[0].read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        errors = [f"{f.check_id}: {f.message}" for f in findings if f.severity == "error"][:6]
        if not errors:
            continue
        note = (
            "Measured geometry defects in your previous figure (fix ALL of them together, "
            "keep the same content): " + "; ".join(errors)
        )
        feedback[slide_id] = (feedback[slide_id] + " " + note) if slide_id in feedback else note
        seeded += 1
    if seeded:
        path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seeded


def write_qc_figure_feedback(ctx: BuildContext) -> int:
    """Convert visual-QC blockers into per-slide feedback for figure
    regeneration (the pixel judge closes the loop the geometric estimators
    cannot)."""
    report = _load_json(ctx.deck_dir / "assets/generated/visual_llm_qc_report.json")
    render_spec = _load_json(ctx.deck_dir / "render_spec.json")
    slides = render_spec.get("slides") or []
    feedback: dict[str, str] = {}
    for entry in report.get("slides") or []:
        if entry.get("status") != "fail":
            continue
        number = int(entry.get("slide_number") or 0)
        if not (1 <= number <= len(slides)):
            continue
        slide_id = str(slides[number - 1].get("slide_id") or "")
        if not slide_id.startswith("sl_"):
            continue  # bookend/template slides are not figure problems
        issues = "; ".join(
            f"{i.get('check_id')}: {i.get('description')}"
            for i in entry.get("issues") or []
            if i.get("severity") in ("blocker", "warn")
            # slide-title defects come from the deck contract, not the figure;
            # regenerating the figure for them just churns
            and "slide title" not in str(i.get("description") or "").lower()
        )
        if issues:
            feedback[slide_id] = (
                "A pixel-level visual review of the RENDERED slide found these defects "
                f"in your previous figure: {issues[:700]}. Redraw with these fixed — "
                "more spacing, no text under arrows/lines, every label inside its shape."
            )
    path = ctx.deck_dir / "qc_figure_feedback.json"
    path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(feedback)


def stage_visual_qc(ctx: BuildContext) -> None:
    try:
        ctx.run("visual_qc", [
            str(SCRIPTS / "inspect_rendered_slides_llm.py"), *ctx.ids(),
            "--screenshots-dir", ctx.rel("assets/generated/qc/screenshots"),
            "--report-output", ctx.rel("assets/generated/visual_llm_qc_report.json"),
        ])
    except RuntimeError:
        pass  # the script exits non-zero on QC fail; the report is the verdict
    report = _load_json(ctx.deck_dir / "assets/generated/visual_llm_qc_report.json")
    write_qc_figure_verdicts(ctx)
    slides = report.get("slides") or []
    failing = [s.get("slide_number") for s in slides if s.get("status") == "fail"]
    if report.get("status") not in ("passed", "warnings") or failing:
        raise RuntimeError(
            f"visual LLM QC status={report.get('status')}; failing slides={failing}. "
            "Inspect assets/generated/visual_llm_qc_report.json, fix the figure or "
            "template defect, then resume (figures reuse makes regeneration cheap)."
        )
    feedback_path = ctx.deck_dir / "qc_figure_feedback.json"
    if feedback_path.is_file():
        feedback_path.unlink()
    print(f"  visual QC passed ({len(slides)}/{len(slides)} slides)")


def stage_visual_qc_advisory(ctx: BuildContext) -> None:
    """A second vision model's opinion, for the Gate A reviewers. Never blocks.

    Runs only when the conference configures ``presentation_review.advisory_visual_qc``
    (see ``conference_pipeline.presentation.review_settings``). It reports more than the
    primary judge and over-states severity, so its findings are hints: a summary is
    written beside the report for Gate A, and nothing here can fail the build.
    """

    from conference_pipeline.presentation.advisory_qc import (
        ADVISORY_REPORT,
        ADVISORY_SUMMARY,
        advisory_command,
        write_advisory_summary,
    )
    from conference_pipeline.presentation.review_settings import advisory_visual_qc_selection

    try:
        selection = advisory_visual_qc_selection(ctx.workspace)
    except Exception as exc:  # noqa: BLE001 - a misconfigured advisory pass must not stop a build
        print(f"  advisory visual QC skipped: {exc}")
        return
    if selection is None:
        print("  advisory visual QC not configured")
        return
    provider, model = selection
    report_path = ctx.deck_dir / ADVISORY_REPORT
    try:
        ctx.run("visual_qc_advisory", advisory_command(
            workspace=ctx.workspace, deck_id=ctx.deck_id, presentation_type=ctx.args.presentation_type,
            screenshots_dir=ctx.deck_dir / "assets/generated/qc/screenshots",
            report_output=report_path, provider=provider, model=model,
        )[1:])  # ctx.run prepends the interpreter
    except RuntimeError:
        pass  # the script exits non-zero when it records findings; the report is the answer
    summary = write_advisory_summary(
        report_path, ctx.deck_dir / ADVISORY_SUMMARY,
        primary_report_path=ctx.deck_dir / "assets/generated/visual_llm_qc_report.json",
    )
    print(f"  advisory visual QC ({model}): {summary.get('status')}, "
          f"{summary.get('flagged_slide_count', 0)} slide(s) flagged "
          f"({summary.get('flagged_only_by_advisory', 0)} not by the primary judge)")


def stage_presenter_script(ctx: BuildContext) -> None:
    ctx.run("presenter_script", [
        str(SCRIPTS / "generate_presenter_script.py"), *ctx.ids(),
        "--render-spec", ctx.rel("render_spec.json"),
        "--slide-mapping", ctx.rel("presentation_slide_mapping.json"),
        "--delivery-single-speaker", ctx.rel("delivery_single_speaker.json"),
        "--delivery-notebooklm", ctx.rel("delivery_notebooklm.json"),
        "--debug-script-md", ctx.rel("6_presenter_script.md"),
        "--debug-notebooklm-script-md", ctx.rel("6_presenter_script_notebooklm_2host.md"),
        "--conference-label", ctx.args.conference_label,
    ])


def stage_engaging_narration(ctx: BuildContext) -> None:
    ctx.run("engaging_narration", [
        str(SCRIPTS / "generate_engaging_narration.py"), "--deck-dir", str(ctx.deck_dir),
    ])


def stage_rebuild_delivery(ctx: BuildContext) -> None:
    ctx.run("rebuild_delivery", [
        str(SCRIPTS / "rebuild_delivery_engaging.py"), "--deck-dir", str(ctx.deck_dir),
    ])


def stage_reconcile_hashes(ctx: BuildContext) -> None:
    ctx.run("reconcile_hashes", [
        str(SCRIPTS / "reconcile_contract_hashes.py"), "--deck-dir", str(ctx.deck_dir),
    ])


def _narration_lint_review(ctx: BuildContext) -> None:
    ctx.run("narration_gates", [
        str(SCRIPTS / "lint_narration_script.py"),
        "--script-md", ctx.full("6_presenter_script_notebooklm_2host.ENGAGING.md"),
        "--deck-kind", "topic", "--topic-id", ctx.deck_id,
        "--conference-label", ctx.args.conference_label,
        "--report-out", ctx.full("narration_script_lint.json"),
    ])
    ctx.run("narration_gates", [
        str(SCRIPTS / "review_narration_script.py"),
        "--script-md", ctx.full("6_presenter_script_notebooklm_2host.ENGAGING.md"),
        "--deck-kind", "topic", "--topic-id", ctx.deck_id,
        "--conference-label", ctx.args.conference_label,
        "--lint-report", ctx.full("narration_script_lint.json"),
        "--json-out", ctx.full("narration_script_review.json"),
    ])


def stage_editorial_storyboard(ctx: BuildContext) -> None:
    """The deterministic full storyboard (with retained/merged/dropped
    rationale) that the editorial gate requires for topic decks; the partial
    one emitted by the presenter script is not sufficient."""
    ctx.run("editorial_storyboard", [
        str(SCRIPTS / "build_editorial_storyboard.py"),
        "--deck-dir", str(ctx.deck_dir),
        "--slides-md", ctx.full("slides.md"),
    ])


def stage_narration_gates(ctx: BuildContext) -> None:
    try:
        _narration_lint_review(ctx)
    except RuntimeError:
        # The runbook remedy for a blocked narration review is the deterministic
        # fixer followed by a delivery rebuild and one re-review.
        print("  narration review blocked; running fix_narration_script + re-review")
        # Fix the raw JSON (the TTS source of truth), then re-render the
        # markdown from it: a markdown-only fix passed review on text that
        # was never spoken.
        ctx.run("narration_gates", [
            str(SCRIPTS / "fix_narration_script.py"),
            "--script-md", ctx.full("6_presenter_script_notebooklm_2host.ENGAGING.md"),
            "--script-json", ctx.full("engaging_narration_raw.json"),
            "--deck-kind", "topic",
            "--conference-label", ctx.args.conference_label,
            "--report-out", ctx.full("narration_script_fix_report.json"),
        ])
        ctx.run("narration_gates", [
            str(SCRIPTS / "generate_engaging_narration.py"),
            "--deck-dir", str(ctx.deck_dir), "--render-from-raw",
        ])
        stage_rebuild_delivery(ctx)
        stage_reconcile_hashes(ctx)
        _narration_lint_review(ctx)
    ctx.run("narration_gates", [
        str(SCRIPTS / "presentation_editorial_gate.py"),
        "--deck-dir", str(ctx.deck_dir), "--deck-kind", "topic",
        "--report-out", ctx.full("presentation_editorial_gate_report.json"),
    ])
    editorial = _load_json(ctx.deck_dir / "presentation_editorial_gate_report.json")
    if editorial.get("status") not in ("passed", "pass"):
        raise RuntimeError(f"editorial gate status={editorial.get('status')}")
    for name in ("delivery_single_speaker.json", "delivery_notebooklm.json"):
        delivery = _load_json(ctx.deck_dir / name)
        if delivery.get("approve_for_tts") is not True:
            raise RuntimeError(f"{name} approve_for_tts is not true")


def stage_coverage_gate(ctx: BuildContext) -> None:
    """Fail fast on content coverage BEFORE the expensive figure/narration
    stages: a deck without canary-level sources/people/works cannot ship."""
    ctx.run("coverage_gate", [
        str(SCRIPTS / "verify_presentation_content.py"),
        "--deck-dir", str(ctx.deck_dir),
        "--skip", "duration", "--skip", "figure", "--skip", "template",
        "--report-out", ctx.full("content_verification_coverage.json"),
    ])


def stage_content_verifiers(ctx: BuildContext) -> None:
    """Full content gates (coverage + duration + figure richness + template
    leaks), calibrated against the approved Topic 4 canary."""
    ctx.run("content_verifiers", [
        str(SCRIPTS / "verify_presentation_content.py"),
        "--deck-dir", str(ctx.deck_dir),
    ])


def stage_build_videos(ctx: BuildContext) -> None:
    """Single-speaker and NotebookLM videos are independent (separate delivery
    contracts, build dirs, and outputs) — build them CONCURRENTLY by default."""
    single = [
        str(SCRIPTS / "build_presenter_video.py"), *ctx.ids(),
        "--slides-html", ctx.rel("presentation.html"),
        "--slide-mapping", ctx.rel("presentation_slide_mapping.json"),
        "--delivery-contract", ctx.rel("delivery_single_speaker.json"),
        "--blueprint", ctx.rel("presentation_blueprint.json"),
        "--blueprint-review", ctx.rel("blueprint_review.json"),
        "--build-dir", ctx.rel("video_build"),
        "--output", ctx.rel("presentation_video.mp4"),
    ]
    notebooklm = [
        str(SCRIPTS / "build_notebooklm_style_presenter_video.py"), *ctx.ids(),
        "--slides-html", ctx.rel("presentation.html"),
        "--slide-mapping", ctx.rel("presentation_slide_mapping.json"),
        "--delivery-contract", ctx.rel("delivery_notebooklm.json"),
        "--blueprint", ctx.rel("presentation_blueprint.json"),
        "--blueprint-review", ctx.rel("blueprint_review.json"),
        "--build-dir", ctx.rel("video_build_notebooklm"),
        "--output", ctx.rel("presentation_video_notebooklm.mp4"),
        "--podcast-script-out", ctx.rel("video_build_notebooklm/script.json"),
        "--host-a", ctx.args.host_a, "--host-b", ctx.args.host_b,
        "--voice-a", ctx.args.host_a, "--voice-b", ctx.args.host_b,
        "--conference-label", ctx.args.conference_label,
    ]
    if ctx.args.sequential_videos:
        ctx.run("video_single", single)
        ctx.run("video_notebooklm", notebooklm)
    else:
        ctx.run_parallel(
            "build_videos",
            [("video_single", single), ("video_notebooklm", notebooklm)],
        )


def stage_video_sync(ctx: BuildContext) -> None:
    # NOTE: validate_video_sync does NOT join --workspace; full paths required.
    ctx.run("video_sync", [
        str(SCRIPTS / "validate_video_sync.py"), *ctx.ids(),
        "--render-spec", ctx.full("render_spec.json"),
        "--delivery-single-speaker", ctx.full("delivery_single_speaker.json"),
        "--delivery-notebooklm", ctx.full("delivery_notebooklm.json"),
        "--single-speaker-video", ctx.full("presentation_video.mp4"),
        "--notebooklm-video", ctx.full("presentation_video_notebooklm.mp4"),
        "--single-speaker-report", ctx.full("video_sync_validation_single_speaker.json"),
        "--notebooklm-report", ctx.full("video_sync_validation_notebooklm.json"),
        "--enforce-truth-thresholds",
    ])
    for name in (
        "video_sync_validation_single_speaker.json",
        "video_sync_validation_notebooklm.json",
    ):
        report = _load_json(ctx.deck_dir / name)
        if report.get("status") != "pass":
            raise RuntimeError(f"{name} status={report.get('status')}")


def stage_produced_review(ctx: BuildContext) -> None:
    ctx.run("produced_review", [
        str(SCRIPTS / "review_presentation.py"), *ctx.ids(),
        "--requirements", ctx.rel("presentation_blueprint.json"),
        "--narration-script", ctx.rel("6_presenter_script.ENGAGING.md"),
        "--screenshots-dir", ctx.rel("assets/generated/qc/screenshots"),
        "--blueprint", ctx.rel("presentation_blueprint.json"),
        "--blueprint-review", ctx.rel("blueprint_review.json"),
        "--deck-contract", ctx.rel("presentation_deck.json"),
        "--render-spec", ctx.rel("render_spec.json"),
        "--source-binding", ctx.rel("source_binding_report.json"),
        "--delivery-contract", ctx.rel("delivery_notebooklm.json"),
        "--video-receipt", ctx.rel("video_sync_validation_notebooklm.json"),
        "--visual-qc-report", ctx.rel("assets/generated/visual_llm_qc_report.json"),
        "--contact-sheet", ctx.rel("render_contact_sheet.png"),
        *(["--visual-qc-advisory", ctx.rel("assets/generated/visual_llm_qc_advisory_summary.json")]
          if (ctx.deck_dir / "assets/generated/visual_llm_qc_advisory_summary.json").is_file() else []),
        "--report-output", ctx.rel("presentation_review.json"),
        "--markdown-output", ctx.rel("presentation_review.md"),
        "--prompt-output", ctx.rel("presentation_review.prompt.md"),
    ])


STAGES: list[tuple[str, Callable[[BuildContext], None]]] = [
    ("source_pack", stage_source_pack),
    ("corpus_inventory", stage_corpus_inventory),
    ("work_ledger", stage_work_ledger),
    ("source_repair", stage_source_repair),
    ("cluster_plan", stage_cluster_plan),
    ("blueprint", stage_blueprint),
    ("blueprint_review", stage_blueprint_review),
    ("slides_draft", stage_slides_draft),
    ("deck_contract", stage_deck_contract),
    ("source_binding", stage_source_binding),
    ("coverage_gate", stage_coverage_gate),
    ("figures_pass1", stage_figures_pass1),
    ("figures_pass2", stage_figures_pass2),
    ("validate_svgs", stage_validate_svgs),
    ("semantic_review", stage_semantic_review),
    ("delivery_bootstrap", stage_delivery_bootstrap),
    ("render_spec", stage_render_spec),
    ("bookends", stage_bookends),
    ("render_html", stage_render_html),
    ("export_pdf", stage_export_pdf),
    ("screenshots", stage_screenshots),
    ("visual_qc", stage_visual_qc),
    ("visual_qc_advisory", stage_visual_qc_advisory),
    ("presenter_script", stage_presenter_script),
    ("engaging_narration", stage_engaging_narration),
    ("rebuild_delivery", stage_rebuild_delivery),
    ("reconcile_hashes", stage_reconcile_hashes),
    ("editorial_storyboard", stage_editorial_storyboard),
    ("narration_gates", stage_narration_gates),
    ("content_verifiers", stage_content_verifiers),
    ("build_videos", stage_build_videos),
    ("video_sync", stage_video_sync),
    ("produced_review", stage_produced_review),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--presentation-type", default="topic-deep-dive")
    parser.add_argument(
        "--conference-label",
        default=None,
        help="Conference display label; defaults to conference.name from the "
        "workspace config.yaml when omitted.",
    )
    parser.add_argument(
        "--extraction-run-id",
        default=None,
        help="Extraction run id; defaults to workflow.run_id from the workspace "
        "config.yaml when omitted.",
    )
    parser.add_argument("--topic-title", default="")
    parser.add_argument("--final-report-url", required=True)
    parser.add_argument(
        "--blueprint-mode",
        default="editorial",
        choices=["editorial"],
        help="Production topic builds only accept the framed editorial blueprint path.",
    )
    parser.add_argument("--approved-by", default="Nikolaos Vasiloglou")
    parser.add_argument("--host-a", default="Iapetus")
    parser.add_argument("--host-b", default="Erinome")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--svg-stack-research", default=DEFAULT_SVG_STACK_RESEARCH)
    parser.add_argument("--figure-workers", type=int, default=4,
                        help="Concurrent figures in pass-2 codegen (default 4; 1 = serial).")
    parser.add_argument("--sequential-videos", action="store_true",
                        help="Build the two videos one after another instead of concurrently.")
    parser.add_argument("--with-produced-review", action="store_true",
                        help="Also run the cross-AI produced review (operator opt-in).")
    parser.add_argument(
        "--accept-visual-qc-residual",
        action="store_true",
        help="After the visual-QC regeneration loop is exhausted, record an operator waiver "
        "(visual_qc_waiver.json: who, when, failing slides, findings) and continue to narration "
        "and video instead of stopping. Never automatic: an operator must pass this flag.",
    )
    parser.add_argument("--force-from", default="",
                        help="Re-run from this stage even if recorded done (later stages rewind).")
    parser.add_argument("--stop-after", default="", help="Stop after this stage.")
    parser.add_argument("--dry-run", action="store_true", help="List stages and their state.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    if args.conference_label is None:
        args.conference_label = conference_name(workspace, default="the conference")
    if args.extraction_run_id is None:
        args.extraction_run_id = _config_workflow_run_id(workspace)
    if not args.extraction_run_id and not args.dry_run:
        print(
            "no --extraction-run-id given and workflow.run_id is absent from "
            f"{workspace / 'config.yaml'}; pass --extraction-run-id or set "
            "workflow.run_id in the workspace config.",
            file=sys.stderr,
        )
        return 2
    ctx = BuildContext(args)
    order = [name for name, _ in STAGES if name != "produced_review" or args.with_produced_review]

    if args.force_from:
        if args.force_from not in order:
            print(f"unknown --force-from stage: {args.force_from}", file=sys.stderr)
            return 2
        for name in order[order.index(args.force_from):]:
            if ctx.stage_status(name) == "done":
                ctx.mark(name, "stale", "forced re-run")

    if args.dry_run:
        print(f"deck: {ctx.deck_id} | workspace: {ctx.workspace}")
        for name in order:
            print(f"  [{ctx.stage_status(name):>7}] {name}")
        return 0

    stage_fns = dict(STAGES)
    qc_retries = 0
    semantic_retries = 0
    while True:
        failed = None
        for name in order:
            if ctx.stage_status(name) == "done":
                print(f"[skip] {name} (done)")
                continue
            print(f"[run ] {name}")
            ctx.mark(name, "running")
            try:
                stage_fns[name](ctx)
            except Exception as exc:  # noqa: BLE001 - orchestrator boundary
                ctx.mark(name, "failed", str(exc)[:500])
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                failed = name
                break
            ctx.mark(name, "done")
            if args.stop_after and name == args.stop_after:
                print(f"[stop] requested stop after {name}")
                return 0
        if failed is None:
            print("[done] all stages complete")
            return 0
        if failed == "visual_qc" and qc_retries < 2:
            flagged = write_qc_figure_feedback(ctx)
            if flagged:
                qc_retries += 1
                print(
                    f"[loop] visual QC flagged {flagged} figures; regenerating with "
                    f"pixel-judge feedback (retry {qc_retries}/2)"
                )
                for stage in QC_RETRY_STAGES:
                    if ctx.stage_status(stage) in ("done", "failed"):
                        ctx.mark(stage, "stale", "visual_qc feedback retry")
                continue
        if failed == "semantic_review" and semantic_retries < 1:
            seeded = write_semantic_figure_feedback(ctx)
            if seeded:
                semantic_retries += 1
                print(f"[loop] semantic review flagged {seeded} figure(s); regenerating with the missing terms seeded")
                for stage in ("figures_pass2", "validate_svgs", "semantic_review"):
                    if ctx.stage_status(stage) in ("done", "failed"):
                        ctx.mark(stage, "stale", "semantic review retry")
                continue
        if failed == "visual_qc" and args.accept_visual_qc_residual:
            waiver = write_visual_qc_waiver(ctx, approved_by=args.approved_by)
            ctx.mark("visual_qc", "done", f"waived by {args.approved_by}: {waiver['failing_slide_count']} slides")
            print(
                f"[waive] visual QC residue on {waiver['failing_slide_count']} slide(s) accepted by "
                f"{args.approved_by}; recorded in visual_qc_waiver.json; continuing"
            )
            continue
        print(
            f"Resume with the same command after fixing; state: {ctx.state_path}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
