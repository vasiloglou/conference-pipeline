"""Whole-presentation review gate for rendered conference decks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from conference_pipeline.model_governance import model_for_alias
from conference_pipeline.model_governance import default_model, validate_selection
from conference_pipeline.model_governance.llm_client import complete, resolve_selection
from conference_pipeline.presentation.vision_qc import SUPPORTED_SCREENSHOT_SUFFIXES
from conference_pipeline.presentation.review_settings import configured_reviewer_tokens
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic


SCHEMA_VERSION = 2
MODEL_TASK_ID = "presentation.presentation_review"
DEFAULT_STAGE_ID = "presentation_review"
DEFAULT_GEMINI_MODEL = default_model(MODEL_TASK_ID)
BLOCKING_STATUSES = {"fail", "failed", "blocker", "blocked", "not_ready"}

REVIEW_CRITERIA: tuple[dict[str, str], ...] = (
    {"id": "source_fidelity", "question": "Do claims match bound source notes, without unsupported absences, composition or attribution?"},
    {
        "id": "guideline_adherence",
        "question": "Does this presentation follow the supplied guidelines and requirements?",
    },
    {
        "id": "technical_audience_pleasantness",
        "question": "Would this be a pleasant presentation for a technical audience?",
    },
    {
        "id": "production_readiness",
        "question": "Is the presentation production ready?",
    },
    {
        "id": "aesthetic_pleasantness",
        "question": "Is the presentation aesthetically pleasant?",
    },
    {
        "id": "clarity_and_congestion",
        "question": "Are the slides clear, or are they congested?",
    },
    {
        "id": "artifact_contract_adherence",
        "question": (
            "Do the produced slides, narration, source IDs, delivery contracts, "
            "video receipts, and review evidence match the approved contracts?"
        ),
    },
)


@dataclass(frozen=True)
class ReviewerSpec:
    """A reviewer's model route, and how it is reached.

    ``transport="cli"`` runs a command that reads the prompt on stdin and opens
    the screenshot files itself. ``transport="api"`` calls the governed model
    through ``llm_client`` and is sent the screenshots as images, so any
    provider the task allows can sit on the panel without needing a CLI.
    """

    reviewer_id: str
    display_name: str
    provider: str
    model: str
    command_env: str
    default_command: tuple[str, ...]
    transport: str = "cli"

    def command(self) -> tuple[str, ...]:
        if self.transport == "api":
            return ()
        override = os.getenv(self.command_env)
        if override and override.strip():
            return tuple(shlex.split(override))
        return self.default_command


_GEMINI_MODEL = os.getenv("PRESENTATION_REVIEW_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

DEFAULT_REVIEWERS: tuple[ReviewerSpec, ...] = (
    ReviewerSpec(
        # The ensemble's Claude member is the Opus TIER (a Claude Code CLI call,
        # not a registry-governed API task), so it follows the catalog alias.
        reviewer_id="claude_code_opus",
        display_name="Claude Code Opus",
        provider="anthropic",
        model=model_for_alias("opus"),
        command_env="PRESENTATION_REVIEW_CLAUDE_COMMAND",
        default_command=("claude", "--model", model_for_alias("opus"), "--print"),
    ),
    ReviewerSpec(
        # The Codex CLI member follows the catalog's `sol` alias, the review
        # task's registered authority model. A pinned "codex-5.5" stopped being
        # served to ChatGPT-account Codex, and the member silently dropped out
        # of every CVPR 2026 review until 2026-09-25. The id is a stable key.
        reviewer_id="codex_5_5",
        display_name="Codex CLI",
        provider="openai",
        model=model_for_alias("sol"),
        command_env="PRESENTATION_REVIEW_CODEX_COMMAND",
        default_command=("codex", "exec", "--model", model_for_alias("sol"), "-"),
    ),
    ReviewerSpec(
        reviewer_id="gemini",
        display_name="Gemini",
        provider="gemini",
        model=_GEMINI_MODEL,
        command_env="PRESENTATION_REVIEW_GEMINI_COMMAND",
        default_command=("gemini", "--model", _GEMINI_MODEL),
    ),
)

ReviewerRunner = Callable[[ReviewerSpec, str, Path, Path], str]


def api_reviewer(provider: str | None = None, model: str | None = None) -> ReviewerSpec:
    """An API-transport reviewer for a (provider, model) the task allows.

    Either half may be omitted (``resolve_selection`` fills it in); the pair is
    validated against the registry here, so a model the review task has not
    allowed cannot sit on the panel.
    """

    provider, model = resolve_selection(MODEL_TASK_ID, provider, model)
    validate_selection(MODEL_TASK_ID, provider=provider, model=model)
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return ReviewerSpec(
        reviewer_id=f"api_{slug}",
        display_name=f"{model} (API)",
        provider=provider,
        model=model,
        command_env="",
        default_command=(),
        transport="api",
    )


def configured_reviewers(
    spec: str | None = None, *, workspace: Path | str | None = None
) -> tuple[ReviewerSpec, ...]:
    """The review panel: ``DEFAULT_REVIEWERS`` unless something says otherwise.

    In order of precedence: an explicit ``spec`` (``--reviewers``), the
    ``PRESENTATION_REVIEW_REVIEWERS`` variable, then the conference's own
    ``presentation_review.reviewers`` (see ``review_settings``). Each is a
    comma-separated list. A token is either the id of a
    default member (``claude_code_opus``, ``codex_5_5``, ``gemini``) or a model
    the review task allows (``openrouter/moonshotai/kimi-k3``), which joins as
    an API reviewer. Order is kept; an unknown token is an error, never a
    silently smaller panel.
    """

    raw = os.environ.get("PRESENTATION_REVIEW_REVIEWERS") if spec is None else spec
    if (not raw or not raw.strip()) and spec is None:
        raw = configured_reviewer_tokens(workspace)
    if not raw or not raw.strip():
        return DEFAULT_REVIEWERS
    defaults = {reviewer.reviewer_id: reviewer for reviewer in DEFAULT_REVIEWERS}
    panel: list[ReviewerSpec] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        member = defaults.get(token) or api_reviewer(model=token)
        if member.reviewer_id in {existing.reviewer_id for existing in panel}:
            raise ValueError(f"reviewer {member.reviewer_id!r} is listed twice in PRESENTATION_REVIEW_REVIEWERS")
        panel.append(member)
    if not panel:
        raise ValueError("PRESENTATION_REVIEW_REVIEWERS names no reviewers")
    return tuple(panel)


def review_presentation(
    *,
    workspace: Path,
    presentation_id: str,
    presentation_type: str,
    requirements_path: Path,
    narration_script_path: Path,
    screenshots_dir: Path,
    report_output: Path,
    markdown_output: Path | None = None,
    prompt_output: Path | None = None,
    topic: str | None = None,
    reviewers: Sequence[ReviewerSpec] | None = None,
    live: bool = False,
    max_screenshots: int | None = None,
    runner: ReviewerRunner | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    claim_fidelity_report_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Review a rendered presentation with multiple independent LLM reviewers."""

    workspace = workspace.expanduser().resolve()
    reviewers = configured_reviewers(workspace=workspace) if reviewers is None else reviewers
    requirements_path = _resolve_inside(requirements_path, workspace, label="requirements_path")
    narration_script_path = _resolve_inside(
        narration_script_path,
        workspace,
        label="narration_script_path",
    )
    screenshots_dir = _resolve_inside(screenshots_dir, workspace, label="screenshots_dir")
    report_output = _resolve_inside(report_output, workspace, label="report_output")
    markdown_output = (
        _resolve_inside(markdown_output, workspace, label="markdown_output")
        if markdown_output is not None
        else None
    )
    prompt_output = (
        _resolve_inside(prompt_output, workspace, label="prompt_output")
        if prompt_output is not None
        else report_output.with_suffix(".prompt.md")
    )
    claim_fidelity_report_paths = tuple(
        _resolve_inside(path, workspace, label="claim_fidelity_report") for path in claim_fidelity_report_paths
    )
    artifact_paths = _resolve_artifact_paths({**(artifact_paths or {}), **{
        f"claim_fidelity_report_{index}": path for index, path in enumerate(claim_fidelity_report_paths, 1)
    }}, workspace)

    missing_inputs = _missing_inputs(requirements_path, narration_script_path, screenshots_dir)
    missing_inputs.extend(_missing_artifact_inputs(artifact_paths))
    screenshots = _screenshot_paths(screenshots_dir)
    if max_screenshots is not None:
        screenshots = screenshots[: max(0, max_screenshots)]
    if not screenshots:
        missing_inputs.append("no rendered slide screenshots found")

    input_manifest = _input_manifest(
        workspace=workspace,
        requirements_path=requirements_path,
        narration_script_path=narration_script_path,
        screenshots_dir=screenshots_dir,
        screenshots=screenshots,
        artifact_paths=artifact_paths,
    )
    if missing_inputs:
        _write_text_atomic(
            prompt_output,
            "Presentation review blocked before prompt construction.\n\nMissing inputs:\n"
            + "\n".join(f"- {item}" for item in missing_inputs)
            + "\n",
        )
        report = _blocked_report(
            presentation_id=presentation_id,
            presentation_type=presentation_type,
            topic=topic,
            input_manifest=input_manifest,
            prompt_output=prompt_output,
            workspace=workspace,
            reason="; ".join(missing_inputs),
            reviewers=reviewers,
        )
        _write_outputs(report_output, markdown_output, report)
        return report

    prompt = build_review_prompt(
        workspace=workspace,
        presentation_id=presentation_id,
        presentation_type=presentation_type,
        topic=topic,
        requirements_path=requirements_path,
        narration_script_path=narration_script_path,
        screenshots=screenshots,
        artifact_paths=artifact_paths,
        claim_fidelity_report_paths=claim_fidelity_report_paths,
    )
    _write_text_atomic(prompt_output, prompt)

    digests = slide_digests(screenshots, narration_script_path)
    prior = latest_prior_review(report_output)
    settled = settled_review_slides(prior, digests, input_manifest)

    review_payloads = [
        _run_or_plan_reviewer(
            reviewer,
            prompt=prompt,
            prompt_path=prompt_output,
            workspace=workspace,
            live=live,
            runner=runner,
            screenshots=screenshots,
        )
        for reviewer in reviewers
    ]
    report = _build_report(
        presentation_id=presentation_id,
        presentation_type=presentation_type,
        topic=topic,
        input_manifest=input_manifest,
        prompt_output=prompt_output,
        workspace=workspace,
        live=live,
        reviews=review_payloads,
        settled_slides=settled,
    )
    report["slide_digests"] = {str(number): digest for number, digest in digests.items()}
    report["settled_slides"] = sorted(settled)
    if prior is not None:
        report["prior_review_generated_at"] = prior.get("generated_at")
    _write_outputs(report_output, markdown_output, report)
    return report


def _slide_number(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def slide_digests(screenshots: Sequence[Path], narration_script_path: Path) -> dict[int, str]:
    """One digest per rendered slide: its screenshot and its own spoken text.

    The spoken text comes from the deck's narration source of truth
    (engaging_narration_raw.json, both variants) in deck-contract order; without
    it every slide depends on the whole narration script.
    """

    deck_dir = narration_script_path.parent
    per_slide: dict[str, str] | None = None
    order: list[str] = []
    try:
        raw = json.loads((deck_dir / "engaging_narration_raw.json").read_text(encoding="utf-8"))
        contract = json.loads((deck_dir / "presentation_deck.json").read_text(encoding="utf-8"))
        per_slide = {
            str(slide.get("slide_id")): json.dumps([slide.get("single"), slide.get("two_host")], sort_keys=True)
            for slide in raw.get("slides") or [] if isinstance(slide, dict) and slide.get("slide_id")
        }
        order = [str(slide.get("slide_id") or "") for slide in contract.get("slides") or [] if isinstance(slide, dict)]
    except (OSError, ValueError, AttributeError):
        per_slide = None
    whole = _file_sha256(narration_script_path) if narration_script_path.is_file() else ""
    digests: dict[int, str] = {}
    for number, path in enumerate(screenshots, start=1):
        if per_slide is not None and len(order) == len(screenshots):
            narration = per_slide.get(order[number - 1], "")
        else:
            narration = whole
        payload = {"screenshot": _file_sha256(path), "narration": narration}
        digests[number] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digests


def latest_prior_review(report_output: Path) -> dict | None:
    """The newest real review of this deck: current, superseded, or a preserved attempt."""

    deck_dir = report_output.parent
    paths = {report_output, *deck_dir.glob("presentation_review.superseded-*.json"), *(deck_dir / "gate_a_attempts").glob("*.json")}
    candidates = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("status") in ("passed", "failed") and payload.get("reviews"):
            candidates.append(payload)
    return max(candidates, key=lambda item: str(item.get("generated_at") or ""), default=None)


def settled_review_slides(prior: dict | None, digests: Mapping[int, str], input_manifest: Mapping[str, object]) -> set[int]:
    """Slides unchanged since the prior review, which held no blocking finding on them.

    Three LLM reviewers re-reading a whole deck raise new items on slides nobody
    touched (ICML 2026 deck 04, 2026-11-06/07: each pass after an edit elsewhere
    failed on fresh findings). A new blocking finding on a settled slide is
    recorded, not blocking; changed slides and slides still blocked are judged
    as before.
    """

    if not isinstance(prior, dict):
        return set()
    feedback = prior.get("aggregated_feedback") if isinstance(prior.get("aggregated_feedback"), dict) else {}
    blockers = feedback.get("blockers") or []
    if any(_slide_number(item.get("slide")) is None for item in blockers if isinstance(item, dict)):
        # A deck-level blocker was open: nothing is settled.
        return set()
    blocked = {_slide_number(item.get("slide")) for item in blockers if isinstance(item, dict)}
    recorded = prior.get("slide_digests")
    if isinstance(recorded, dict):
        unchanged = {number for number, digest in digests.items() if recorded.get(str(number)) == digest}
    else:
        # A report from before per-slide digests: a slide is unchanged when its
        # screenshot is, and so is the whole narration script.
        prior_inputs = prior.get("inputs") if isinstance(prior.get("inputs"), dict) else {}
        prior_narration = (prior_inputs.get("narration_script") or {}).get("sha256")
        current_narration = (input_manifest.get("narration_script") or {}).get("sha256")
        if not prior_narration or prior_narration != current_narration:
            return set()
        prior_shots = {_slide_number(row.get("slide")): row.get("sha256") for row in prior_inputs.get("screenshots") or [] if isinstance(row, dict)}
        current_shots = {_slide_number(row.get("slide")): row.get("sha256") for row in input_manifest.get("screenshots") or [] if isinstance(row, dict)}
        unchanged = {number for number in digests if current_shots.get(number) and prior_shots.get(number) == current_shots.get(number)}
    return unchanged - blocked


def prior_review_blockers(deck_dir: Path) -> list[dict]:
    """Read preserved rounds, including legacy superseded reports."""
    paths = sorted(set((deck_dir / "gate_a_attempts").glob("*.json")) |
                   set(deck_dir.glob("presentation_review.superseded-*.json")))
    result = []
    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"production_blockers", "blocking_findings", "blockers"} and child:
                    result.append({"attempt": path.name, "blockers": child})
                elif isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for path in paths:
        try:
            collect(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return result


def build_review_prompt(
    *,
    workspace: Path,
    presentation_id: str,
    presentation_type: str,
    topic: str | None,
    requirements_path: Path,
    narration_script_path: Path,
    screenshots: Sequence[Path],
    artifact_paths: Mapping[str, Path] | None = None,
    claim_fidelity_report_paths: Sequence[Path] = (),
) -> str:
    """Build the reviewer prompt from requirements, narration, and screenshots."""

    max_requirements_chars = _env_int("PRESENTATION_REVIEW_MAX_REQUIREMENTS_CHARS", 60000)
    max_narration_chars = _env_int("PRESENTATION_REVIEW_MAX_NARRATION_CHARS", 60000)
    requirements_text = _read_text_limited(requirements_path, max_requirements_chars)
    narration_text = _read_text_limited(narration_script_path, max_narration_chars)
    artifact_paths = artifact_paths or {}
    artifact_sections = _artifact_prompt_sections(
        {key: path for key, path in artifact_paths.items() if not key.startswith("claim_fidelity_report_")}, workspace
    )
    fidelity_note = _claim_fidelity_note(claim_fidelity_report_paths, workspace)
    from conference_pipeline.presentation.claim_fidelity import bound_source_records
    binding_path = artifact_paths.get("source_binding") or requirements_path.parent / "source_binding_report.json"
    binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
    source_records = json.dumps(bound_source_records(binding), ensure_ascii=False, indent=2)
    prior_findings = json.dumps(prior_review_blockers(requirements_path.parent), ensure_ascii=False, indent=2)
    acceptance_note = _acceptance_note(artifact_paths, workspace)
    advisory_note = _advisory_note(artifact_paths)
    # Without a video receipt this is the deck-acceptance gate, which runs
    # BEFORE any TTS or video is made (a failed review must be able to stop
    # the expensive step). On CVPR 2026 the reviewers read leftover build
    # files and moved-aside old videos as "stale final videos" and blocked
    # every deck for not having what this gate exists to precede.
    pre_video_note = (
        ""
        if "video_receipt" in artifact_paths
        else (
            "\n- This is the PRE-VIDEO deck-acceptance review: TTS audio, the two-host "
            "script.json, both MP4s and the video-sync receipts are produced AFTER it "
            "passes and are reviewed then. Missing, stale or superseded audio/video "
            "artifacts (anything under video_build*/ or superseded_*/) are expected "
            "here and are NOT blockers; judge the slides, narration text, sources "
            "and contracts."
        )
    )
    screenshot_rows = "\n".join(
        f"- slide {index}: `{_relative(path, workspace)}` "
        f"(sha256: {_file_sha256(path)[:16]})"
        for index, path in enumerate(screenshots, start=1)
    )
    criteria_rows = "\n".join(
        f"- `{criterion['id']}`: {criterion['question']}" for criterion in REVIEW_CRITERIA
    )
    topic_line = f"\nTopic: {topic}" if topic else ""
    return f"""You are reviewing a rendered technical conference presentation.

Presentation ID: {presentation_id}
Presentation type: {presentation_type}{topic_line}

Inputs:
- Requirements: `{_relative(requirements_path, workspace)}`
- Narration script: `{_relative(narration_script_path, workspace)}`
- Slide screenshots listed below. Inspect every screenshot path when your runtime can open images.

Review criteria:
{criteria_rows}

Return ONLY a JSON object with this schema:
{{
  "status": "pass" | "warn" | "fail",
  "summary": "one concise paragraph",
  "criteria": {{
    "source_fidelity": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "guideline_adherence": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "technical_audience_pleasantness": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "production_readiness": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "aesthetic_pleasantness": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "clarity_and_congestion": {{"status": "pass|warn|fail", "comments": ["..."]}},
    "artifact_contract_adherence": {{"status": "pass|warn|fail", "comments": ["..."]}}
  }},
  "production_blockers": [
    {{"category": "factual|attribution|visual_requirement|lineage|presentation", "slide": 1, "issue": "specific blocker"}}
  ],
  "slide_comments": [
    {{"slide": 1, "severity": "pass|warn|fail", "category": "factual|attribution|visual_requirement|lineage|presentation", "comment": "specific visual or narration issue"}}
  ],
  "suggested_fixes": ["highest leverage fixes"],
  "confidence": 0.0
}}

A figure carrying a synthesis label is a declared combination; judge it as the talk's synthesis, not as a reported experiment.

Perform an exhaustive per-slide sweep before summarising. List EVERY factual
and attribution blocker on EVERY slide in this round; do not stop after a few.
Verify each prior blocker is fixed, then sweep again for new problems. Source
notes, not slide claims or prior approval, establish what the work supports.
Check unsupported absence claims, cross-source method composition and position
papers presented as demonstrated results.

Bound source records (same evidence for every reviewer):
{source_records}

Prior attempts' blockers (verify each is fixed, then sweep again):
{prior_findings}

Be direct. Flag production blockers when slides are congested, unclear, visually
unpleasant, inconsistent with requirements, or unlikely to work for a technical
audience. Treat vague compliments without slide-specific evidence as low quality.
Treat a slide-count expansion that is NOT recorded as approved in the deck
contract, mismatch among deck/render/delivery/video segments, missing or stale
source-ledger IDs, narration that is factually wrong or disconnected from the
slide's actual claim, unreadable visual evidence, or genuine attribution drift as
production blockers. Literal SRC-* IDs do not need to be spoken aloud, but they
must be present in machine-readable contracts and review context.

Conventions and approved deviations for this pipeline (verified facts about how
these contracts are structured — do NOT raise these as blockers):
- A slide-count expansion is acceptable when the deck contract records it under
  `approved_expansion`. Only an expansion absent from that field is a blocker;
  check the field before flagging the slide count.
- Source labels use two distinct namespaces. `SRC-NNN` is the approved,
  viewer-facing research-source ledger; `PLAN-NNN` is internal planning or
  governance provenance. A `PLAN-NNN` label (for example on a title slide) is
  intentional and is NOT a mis-assigned `SRC-001`. Judge source-ledger ordering
  using the `SRC-NNN` entries only.
- `blueprint_sha256` is the hash of the blueprint contract. Source-paper content
  digests are different values; a source digest that differs from
  `blueprint_sha256` is not a blueprint version mismatch.
- The approved source ledger is the source plan referenced by
  `source_binding_report.approved_source_plan_sha256` (the editorial source
  board) together with the per-slide bindings in `source_binding_report`. The
  blueprint `source_paper_plan` may be a narrower seed subset and is NOT the
  source-approval authority. Judge whether the deck's sources are approved and
  tracked against the binding report and its approved source plan, not against
  the count of blueprint seed entries; a deck source that is bound there is
  approved even if it is absent from the blueprint `source_paper_plan`.
- Approved narration style: this deck uses an engaging two-host (or single
  presenter) style that conveys each slide's substance through plain explanation,
  analogies, and concrete examples, and introduces each source once by name. It
  deliberately does NOT mechanically describe the visual layout ("a two-column
  figure with panels..."). That mechanical visual narration was explicitly
  removed as a creative decision. Do NOT treat the absence of layout description
  as a defect. Flag narration only when it is factually wrong, contradicts the
  slide, or omits the slide's actual claim — not when it simply declines to
  enumerate on-screen shapes.
- A quantitative figure is source-grounded when the slide carries a source chip /
  `source_id` naming the work the numbers come from; the on-slide source ledger
  is the grounding. Flag specific figures only when the named source does not
  support the values or when an unlabeled chart is presented as results with no
  source on the slide at all.
- A blueprint with `revision` above 1 is a recorded repair: `repair_notes` says
  what changed and its generation receipt's `repair.changed_paths` lists every
  changed field. A deck contract or render spec still carrying the parent
  blueprint's `blueprint_sha256` is NOT drift when the change was wording only
  (narration beats, visible text, titles) and no source binding changed; judge
  the slides against the current blueprint.
- The human-approval receipt (`human_review_receipt.json`) and the deck-level
  release gate are applied AFTER this review; their absence at review time is
  expected and is NOT a production blocker for the produced presentation itself.
  Assess the produced artifacts on their own merits.{pre_video_note}
- Categorize every production blocker and every failing slide comment:
  `factual` (a claim, number, mechanism or result that the bound source does not
  support or contradicts), `attribution` (a work, method, person or finding credited
  to the wrong source, or named without a bound source), `visual_requirement` (the
  slide departs from the blueprint's requested visual, or a figure is schematic
  where a chart was asked for), `lineage` (receipts, hashes, digests, contract
  bookkeeping), `presentation` (layout, wording, density, aesthetics). Only
  `factual` and `attribution` findings block the deck; the rest are recorded.
{screenshot_rows}

Requirements (the approved blueprint). Its per-slide `visual_figure_brief` is the
PLAN the figure was authored from, not the rendered figure: judge figures by the
screenshots and figure specs. A brief that names a combination the render no
longer shows is not a finding against the deck.
```text
{requirements_text}
```

Narration script:
```text
{narration_text}
```

Produced artifact contracts and receipts:
{artifact_sections}
{acceptance_note}{advisory_note}{fidelity_note}"""


def _claim_fidelity_note(paths: Sequence[Path], workspace: Path) -> str:
    if not paths:
        return ""
    sections = ["\nADVISORY CLAIM-FIDELITY FINDINGS — confirm or dismiss each; do not treat as established",
                "These reports are review leads, not evidence or established verdicts. Confirm or dismiss each "
                "against the bound work. Contribution notes are short summaries, not the whole paper; "
                "missing detail alone is not a contradiction. Labelled synthesis, pedagogical explanation, "
                "and wording scoped to the talk are permitted. Check ledger_mismatch findings first."]
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        findings = report.get("findings") or []
        ledger = [row for row in findings if row.get("code") == "ledger_mismatch"]
        sections.append(f"\nReport: {_relative(path, workspace)}\n" + json.dumps({
            "phase": report.get("phase"), "status": report.get("status"),
            "finding_counts": report.get("finding_counts"),
            "LEDGER MISMATCH — verify source records first": ledger,
            "findings": findings,
        }, ensure_ascii=False, indent=2))
    return "\n".join(sections)


def _acceptance_note(artifact_paths: Mapping[str, Path], workspace: Path) -> str:
    """Tell reviewers which visual-QC findings the operator already accepted.

    The runner writes ``visual_qc_acceptance`` only under an explicit
    ``--accept-visual-qc-findings``. Reviewers must not fail production
    readiness solely on those findings -- that decision was taken, on the
    record, before they were asked -- but they report them as residual risk
    and are free to fail the deck on anything else.
    """

    path = artifact_paths.get("visual_qc_acceptance")
    if path is None or not path.is_file():
        return ""
    return (
        "\nOPERATOR ACCEPTANCE ON RECORD: the `visual_qc_acceptance` artifact above lists "
        "vision-QC findings the operator explicitly accepted for this run, verbatim. Do NOT "
        "fail `production_readiness` solely because `visual_qc_report` records those "
        "findings or reads as failed on their account; report them under warnings as "
        "accepted residual risk instead. Every other defect you find is still yours to fail.\n"
    )


def _advisory_note(artifact_paths: Mapping[str, Path]) -> str:
    """Tell reviewers how to read the advisory second-opinion visual QC."""

    path = artifact_paths.get("visual_qc_advisory")
    if path is None or not path.is_file():
        return ""
    return (
        "\nADVISORY SECOND OPINION: the `visual_qc_advisory` artifact above lists findings from a "
        "second vision model, run for recall. It reports more than `visual_qc_report` and over-states "
        "severity; it has mistaken a datastore glyph for an overlap and a striped table for a split "
        "panel. Treat each finding as a hint: look at that screenshot, report only what you can see, "
        "at the severity you judge. A finding you cannot confirm is not a defect, and a confirmed one "
        "is yours to report whether or not the primary report agrees.\n"
    )


def _run_or_plan_reviewer(
    reviewer: ReviewerSpec,
    *,
    prompt: str,
    prompt_path: Path,
    workspace: Path,
    live: bool,
    runner: ReviewerRunner | None,
    screenshots: Sequence[Path] = (),
) -> dict[str, object]:
    command = reviewer.command()
    payload: dict[str, object] = {
        "reviewer_id": reviewer.reviewer_id,
        "display_name": reviewer.display_name,
        "provider": reviewer.provider,
        "model": reviewer.model,
        "transport": reviewer.transport,
        "command": list(command),
        "prompt_path": _relative(prompt_path, workspace),
    }
    if not live:
        return {
            **payload,
            "status": "planned",
            "parsed": None,
            "raw_text": "",
            "parse_error": None,
        }
    started = time.monotonic()
    try:
        if runner is not None:
            raw_text = runner(reviewer, prompt, prompt_path, workspace)
        elif reviewer.transport == "api":
            raw_text = _run_api_reviewer(
                reviewer, prompt=prompt, screenshots=screenshots, workspace=workspace
            )
        else:
            raw_text = _run_reviewer_command(command, prompt=prompt, workspace=workspace)
        parsed = _parse_reviewer_json(raw_text)
        return {
            **payload,
            "status": _normalized_status(parsed.get("status")),
            "duration_seconds": round(time.monotonic() - started, 3),
            "parsed": parsed,
            "raw_text": raw_text,
            "parse_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - gate must preserve all reviewer failures.
        return {
            **payload,
            "status": "blocked",
            "duration_seconds": round(time.monotonic() - started, 3),
            "parsed": None,
            "raw_text": "",
            "parse_error": str(exc),
        }


# The Gemini CLI reads "@<token>" in its input as a file reference and splices
# the file in. Narration that says "F1@50" therefore reached Gemini as
# "F1" plus whatever "@50..." resolved to -- on CVPR 2026 deck 09 that was a
# real provider-call receipt, which Gemini then reported, three reviews in a
# row, as a "leaked file path" in a contribution note that contained no such
# thing. The fullwidth "＠" reads the same to a model and is not a CLI trigger.
_GEMINI_AT_TOKEN_RE = re.compile(r"@(?=\S)")


def gemini_safe_prompt(prompt: str) -> str:
    """Neutralize file-reference-shaped @tokens for the Gemini CLI."""

    return _GEMINI_AT_TOKEN_RE.sub("\uff20", prompt)


def _review_image(path: Path, *, max_dim: int) -> tuple[bytes, str]:
    """A screenshot as PNG bytes, shrunk so a whole deck fits in one request."""

    import io

    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def _run_api_reviewer(
    reviewer: ReviewerSpec, *, prompt: str, screenshots: Sequence[Path], workspace: Path
) -> str:
    """Review through the governed API: the prompt plus every screenshot as an image.

    A CLI reviewer opens the screenshot paths itself; an API model cannot, so
    they are attached in the order the prompt lists them. A deck with more
    screenshots than the request limit is refused rather than silently
    truncated -- a review of part of a deck must not read as a review of it.
    """

    max_images = _env_int("PRESENTATION_REVIEW_API_MAX_IMAGES", 100)
    if len(screenshots) > max_images:
        raise RuntimeError(
            f"{len(screenshots)} screenshots exceed the {max_images}-image API request limit; "
            "raise PRESENTATION_REVIEW_API_MAX_IMAGES or review this deck with CLI members"
        )
    max_dim = _env_int("PRESENTATION_REVIEW_API_IMAGE_MAX_DIM", 1600)
    images = [_review_image(path, max_dim=max_dim) for path in screenshots]
    note = (
        f"\n\nThe {len(images)} slide screenshots listed above are ATTACHED to this message as "
        "images, in the same order (image 1 is screenshot 1). You cannot open file paths; judge "
        "the slides from the attached images and from the text above.\n"
    )
    timeout_seconds = _env_int("PRESENTATION_REVIEW_TIMEOUT_SECONDS", 900)
    with provider_receipt_context(
        workspace_root=workspace, stage_id=DEFAULT_STAGE_ID, task_id=MODEL_TASK_ID, caller=__file__
    ):
        result = complete(
            MODEL_TASK_ID,
            prompt=prompt + note,
            images=images,
            provider=reviewer.provider,
            model=reviewer.model,
            # Kimi K3 reasons before it writes, inside the same ceiling: at 16000 it spent
            # all of it on ICML deck 01 (2026-10-04) and the verdict JSON stopped mid-string.
            max_tokens=_env_int("PRESENTATION_REVIEW_API_MAX_TOKENS", 40000),
            timeout=None if timeout_seconds <= 0 else float(timeout_seconds),
            caller=__file__,
        )
    if result.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{reviewer.model} used its whole {result.output_tokens}-token ceiling before the review "
            "was complete (reasoning counts toward it); raise PRESENTATION_REVIEW_API_MAX_TOKENS"
        )
    if not result.text.strip():
        raise RuntimeError(
            f"{reviewer.model} returned no review text (stop_reason={result.stop_reason!r}, "
            f"content_blocks={list(result.block_types)})"
        )
    return result.text.strip()


def _run_reviewer_command(command: Sequence[str], *, prompt: str, workspace: Path) -> str:
    if not command:
        raise ValueError("reviewer command is empty")
    if Path(str(command[0])).name == "gemini":
        prompt = gemini_safe_prompt(prompt)
    timeout_seconds = _env_int("PRESENTATION_REVIEW_TIMEOUT_SECONDS", 900)
    # The Gemini CLI refuses to run headless in a directory it has not been
    # told to trust, exiting 55 before reading the prompt, and names this
    # variable as the fix for automated environments. Without it the ensemble
    # silently degraded to a single reviewer: every CVPR 2026 deck's review
    # ran with Claude alone while recording Gemini as "unavailable". The
    # workspace is the one the caller already passed as cwd, so this asserts
    # nothing broader than the directory this process was pointed at.
    env = dict(os.environ)
    env.setdefault("GEMINI_CLI_TRUST_WORKSPACE", "true")
    completed = subprocess.run(
        list(command),
        input=prompt,
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
        timeout=None if timeout_seconds <= 0 else timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"reviewer command exited {completed.returncode}: {stderr[:1000] or '<no stderr>'}"
        )
    return completed.stdout.strip()


def _build_report(
    *,
    presentation_id: str,
    presentation_type: str,
    topic: str | None,
    input_manifest: dict[str, object],
    prompt_output: Path,
    workspace: Path,
    live: bool,
    reviews: Sequence[dict[str, object]],
    settled_slides: Iterable[int] = (),
) -> dict[str, object]:
    blockers = _collect_blockers(reviews)
    recorded_findings = _collect_recorded_findings(reviews)
    settled = set(settled_slides)
    demoted = [
        {**item, "demoted_reason": "slide unchanged since a prior review that found no blocker on it"}
        for item in blockers if _slide_number(item.get("slide")) in settled
    ]
    if demoted:
        blockers = [item for item in blockers if _slide_number(item.get("slide")) not in settled]
        recorded_findings = recorded_findings + demoted
    warnings = _collect_warnings(reviews)
    unavailable = _collect_unavailable(reviews)
    ran_reviewers = [review for review in reviews if isinstance(review.get("parsed"), dict)]
    min_reviewers = max(1, _env_int("PRESENTATION_REVIEW_MIN_REVIEWERS", 2))
    # Warnings are advisory: only genuine hard-fails block production readiness.
    # Unavailable reviewers are surfaced as advisory warnings (never silently
    # dropped) but do not hard-block, provided enough reviewers actually ran.
    advisory_warnings = warnings + unavailable
    if not live:
        status = "planned"
    elif blockers:
        status = "failed"
    elif len(ran_reviewers) < min_reviewers:
        # Too few reviewers produced a verdict to certify readiness.
        status = "blocked"
    else:
        status = "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_id": DEFAULT_STAGE_ID,
        "model_task_id": MODEL_TASK_ID,
        "status": status,
        "production_ready": bool(live and status == "passed"),
        "reviewer_coverage": {
            "ran": len(ran_reviewers),
            "required": min_reviewers,
            "total": len(list(reviews)),
            "unavailable": [str(notice["reviewer_id"]) for notice in unavailable],
        },
        "generated_at": _now(),
        "presentation_id": presentation_id,
        "presentation_type": presentation_type,
        "topic": topic,
        "criteria": list(REVIEW_CRITERIA),
        "inputs": input_manifest,
        "prompt_path": _relative(prompt_output, workspace),
        "reviewers": [
            {
                "reviewer_id": review["reviewer_id"],
                "display_name": review["display_name"],
                "provider": review["provider"],
                "model": review["model"],
                "transport": review.get("transport", "cli"),
                "command": review["command"],
            }
            for review in reviews
        ],
        "reviews": list(reviews),
        "aggregated_feedback": {
            "blockers": blockers,
            "recorded_non_blocking_findings": recorded_findings,
            "blocking_categories": sorted(BLOCKING_CATEGORIES),
            "warnings": advisory_warnings,
            "summary": _aggregate_summary(reviews),
        },
    }


def _blocked_report(
    *,
    presentation_id: str,
    presentation_type: str,
    topic: str | None,
    input_manifest: dict[str, object],
    prompt_output: Path,
    workspace: Path,
    reason: str,
    reviewers: Sequence[ReviewerSpec],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_id": DEFAULT_STAGE_ID,
        "model_task_id": MODEL_TASK_ID,
        "status": "blocked",
        "production_ready": False,
        "generated_at": _now(),
        "presentation_id": presentation_id,
        "presentation_type": presentation_type,
        "topic": topic,
        "criteria": list(REVIEW_CRITERIA),
        "inputs": input_manifest,
        "prompt_path": _relative(prompt_output, workspace),
        "reviewers": [
            {
                "reviewer_id": reviewer.reviewer_id,
                "display_name": reviewer.display_name,
                "provider": reviewer.provider,
                "model": reviewer.model,
                "transport": reviewer.transport,
                "command": list(reviewer.command()),
            }
            for reviewer in reviewers
        ],
        "reviews": [],
        "aggregated_feedback": {
            "blockers": [
                {
                    "reviewer_id": "presentation_review",
                    "criterion": "inputs_available",
                    "comment": reason,
                }
            ],
            "warnings": [],
            "summary": reason,
        },
    }


# Operator decision 2026-09-25: only errors the audience would see or hear block
# a deck. With three reviewers and any-blocker-fails, CVPR 2026 decks went five
# rounds on findings such as receipt digests, charts the sources had no numbers
# for, and a title that rendered correctly; those are recorded, not blocking.
BLOCKING_CATEGORIES = frozenset({"factual", "attribution"})


def _finding_category(item: object) -> str | None:
    if isinstance(item, dict):
        category = str(item.get("category") or "").strip().lower()
        return category or None
    return None


def _finding_blocks(item: object) -> bool:
    """Uncategorized findings still block (a reviewer on the old schema)."""
    category = _finding_category(item)
    return category is None or category in BLOCKING_CATEGORIES


def _classified_findings(reviews: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(blocking, recorded) findings from production blockers and failing slide comments."""

    blocking: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []
    for review in reviews:
        reviewer_id = str(review.get("reviewer_id") or "")
        parsed = review.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for blocker in parsed.get("production_blockers") or []:
            if isinstance(blocker, dict):
                comment = str(blocker.get("issue") or blocker.get("comment") or "")
                finding = {"reviewer_id": reviewer_id, "criterion": "production_readiness", "comment": comment,
                           "category": _finding_category(blocker), "slide": blocker.get("slide")}
            elif str(blocker).strip():
                finding = {"reviewer_id": reviewer_id, "criterion": "production_readiness", "comment": str(blocker),
                           "category": None}
            else:
                continue
            (blocking if _finding_blocks(blocker) else recorded).append(finding)
        for slide_comment in parsed.get("slide_comments", []) or []:
            if not isinstance(slide_comment, dict):
                continue
            if _normalized_status(slide_comment.get("severity")) != "fail":
                continue
            finding = {
                "reviewer_id": reviewer_id,
                "criterion": "slide_comment",
                "comment": str(slide_comment.get("comment") or "slide failed review"),
                "slide": slide_comment.get("slide"),
                "category": _finding_category(slide_comment),
            }
            (blocking if _finding_blocks(slide_comment) else recorded).append(finding)
    return blocking, recorded


def _collect_blockers(reviews: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Hard blockers: factual or attribution findings (or uncategorized ones).

    A reviewer's overall or per-criterion "fail" is a summary of its findings,
    so it blocks only through them; a reviewer that fails a deck without any
    categorized finding is still honoured as a blocker.
    """

    blocking, recorded = _classified_findings(reviews)
    for review in reviews:
        reviewer_id = str(review.get("reviewer_id") or "")
        parsed = review.get("parsed")
        if not isinstance(parsed, dict) or _normalized_status(parsed.get("status")) != "fail":
            continue
        has_findings = any(item["reviewer_id"] == reviewer_id for item in blocking + recorded)
        if not has_findings:
            blocking.append(
                {
                    "reviewer_id": reviewer_id,
                    "criterion": "overall",
                    "comment": str(parsed.get("summary") or "reviewer marked presentation fail"),
                    "category": None,
                }
            )
    return blocking


def _collect_recorded_findings(reviews: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return _classified_findings(reviews)[1]


def _collect_warnings(reviews: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for review in reviews:
        reviewer_id = str(review.get("reviewer_id") or "")
        parsed = review.get("parsed")
        if not isinstance(parsed, dict):
            continue
        if _normalized_status(parsed.get("status")) == "warn":
            warnings.append(
                {
                    "reviewer_id": reviewer_id,
                    "criterion": "overall",
                    "comment": str(parsed.get("summary") or "reviewer marked presentation warn"),
                }
            )
        for criterion_id, criterion_payload in _criteria_items(parsed):
            if _normalized_status(criterion_payload.get("status")) == "warn":
                warnings.append(
                    {
                        "reviewer_id": reviewer_id,
                        "criterion": criterion_id,
                        "comment": "; ".join(_string_list(criterion_payload.get("comments"))),
                    }
                )
        for slide_comment in parsed.get("slide_comments", []):
            if not isinstance(slide_comment, dict):
                continue
            if _normalized_status(slide_comment.get("severity")) == "warn":
                warnings.append(
                    {
                        "reviewer_id": reviewer_id,
                        "criterion": "slide_comment",
                        "comment": str(slide_comment.get("comment") or ""),
                        "slide": slide_comment.get("slide"),
                    }
                )
    return warnings


def _collect_unavailable(reviews: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Reviewers whose command could not produce a parseable verdict.

    Recorded as advisory notices so an unavailable provider (for example a model
    the installed CLI does not recognize) is never silently dropped, while also
    not hard-blocking production readiness on its own. Certification still
    requires a minimum number of reviewers to have actually run; see
    ``_build_report``.
    """
    notices: list[dict[str, object]] = []
    for review in reviews:
        if str(review.get("status") or "").lower() == "planned":
            continue
        if isinstance(review.get("parsed"), dict):
            continue
        notices.append(
            {
                "reviewer_id": str(review.get("reviewer_id") or ""),
                "criterion": "reviewer_available",
                "comment": str(
                    review.get("parse_error")
                    or "reviewer command did not return a parseable review"
                ),
            }
        )
    return notices


def _aggregate_summary(reviews: Sequence[dict[str, object]]) -> str:
    summaries = []
    for review in reviews:
        parsed = review.get("parsed")
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or "").strip()
            if summary:
                summaries.append(f"{review.get('display_name')}: {summary}")
    if summaries:
        return "\n".join(summaries)
    if all(str(review.get("status")) == "planned" for review in reviews):
        return "Review commands planned; live reviewer execution has not run."
    return "No parseable reviewer summaries were collected."


def _criteria_items(parsed: dict[str, object]) -> Iterable[tuple[str, dict[str, object]]]:
    criteria = parsed.get("criteria")
    if not isinstance(criteria, dict):
        return ()
    items: list[tuple[str, dict[str, object]]] = []
    valid_ids = {criterion["id"] for criterion in REVIEW_CRITERIA}
    for criterion_id, payload in criteria.items():
        if str(criterion_id) not in valid_ids or not isinstance(payload, dict):
            continue
        items.append((str(criterion_id), payload))
    return tuple(items)


def _parse_reviewer_json(raw_text: str) -> dict[str, object]:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("presentation review output must be a JSON object")
    return _normalize_reviewer_payload(payload)


def _normalize_reviewer_payload(payload: dict[str, object]) -> dict[str, object]:
    criteria = payload.get("criteria")
    normalized_criteria: dict[str, object] = {}
    source_criteria = criteria if isinstance(criteria, dict) else {}
    for criterion in REVIEW_CRITERIA:
        criterion_id = criterion["id"]
        raw = source_criteria.get(criterion_id)
        raw_dict = raw if isinstance(raw, dict) else {}
        normalized_criteria[criterion_id] = {
            "status": _normalized_status(raw_dict.get("status")),
            "comments": _string_list(raw_dict.get("comments")),
        }
    return {
        "status": _normalized_status(payload.get("status")),
        "summary": str(payload.get("summary") or "").strip(),
        "criteria": normalized_criteria,
        "production_blockers": _normalize_production_blockers(payload.get("production_blockers")),
        "slide_comments": _normalize_slide_comments(payload.get("slide_comments")),
        "suggested_fixes": _string_list(payload.get("suggested_fixes")),
        "confidence": _confidence(payload.get("confidence")),
    }


def _normalize_slide_comments(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    comments: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        comments.append(
            {
                "slide": item.get("slide"),
                "severity": _normalized_status(item.get("severity")),
                "comment": str(item.get("comment") or "").strip(),
                "category": str(item.get("category") or "").strip().lower() or None,
            }
        )
    return comments


def _normalize_production_blockers(value: object) -> list[object]:
    """Keep categorized blockers as objects (the category decides whether they
    block); plain strings stay strings and still block."""

    if not isinstance(value, list):
        return _string_list(value)
    blockers: list[object] = []
    for item in value:
        if isinstance(item, dict):
            issue = str(item.get("issue") or item.get("comment") or "").strip()
            if issue:
                blockers.append(
                    {
                        "category": str(item.get("category") or "").strip().lower() or None,
                        "slide": item.get("slide"),
                        "issue": issue,
                    }
                )
        elif str(item).strip():
            blockers.append(str(item).strip())
    return blockers


def _normalized_status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in BLOCKING_STATUSES:
        return "fail"
    if normalized in {"warn", "warning", "warnings"}:
        return "warn"
    if normalized in {"pass", "passed", "ok", "ready"}:
        return "pass"
    return "warn" if normalized else "warn"


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _input_manifest(
    *,
    workspace: Path,
    requirements_path: Path,
    narration_script_path: Path,
    screenshots_dir: Path,
    screenshots: Sequence[Path],
    artifact_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    artifact_paths = artifact_paths or {}
    return {
        "requirements": _file_record(requirements_path, workspace),
        "narration_script": _file_record(narration_script_path, workspace),
        "screenshots_dir": _relative(screenshots_dir, workspace),
        "screenshot_count": len(screenshots),
        "screenshots": [
            {
                "slide": index,
                **_file_record(path, workspace),
            }
            for index, path in enumerate(screenshots, start=1)
        ],
        "artifacts": {
            label: _file_record(path, workspace)
            for label, path in sorted(artifact_paths.items())
        },
    }


def _file_record(path: Path, workspace: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": _relative(path, workspace),
        "exists": path.exists(),
    }
    if path.is_file():
        record["sha256"] = _file_sha256(path)
        record["bytes"] = path.stat().st_size
    return record


def _missing_inputs(requirements_path: Path, narration_script_path: Path, screenshots_dir: Path) -> list[str]:
    missing = []
    if not requirements_path.is_file():
        missing.append(f"requirements file missing: {requirements_path}")
    if not narration_script_path.is_file():
        missing.append(f"narration script missing: {narration_script_path}")
    if not screenshots_dir.is_dir():
        missing.append(f"screenshots directory missing: {screenshots_dir}")
    return missing


def _resolve_artifact_paths(
    artifact_paths: Mapping[str, Path],
    workspace: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for label, path in artifact_paths.items():
        if path is None:
            continue
        resolved[str(label)] = _resolve_inside(path, workspace, label=f"artifact:{label}")
    return resolved


def _missing_artifact_inputs(artifact_paths: Mapping[str, Path]) -> list[str]:
    missing: list[str] = []
    for label, path in sorted(artifact_paths.items()):
        if not path.is_file():
            missing.append(f"{label} artifact missing: {path}")
    return missing


def _artifact_prompt_sections(artifact_paths: Mapping[str, Path], workspace: Path) -> str:
    if not artifact_paths:
        return "No additional produced artifact contracts were supplied."
    max_chars = _env_int("PRESENTATION_REVIEW_MAX_ARTIFACT_CHARS", 18000)
    rows: list[str] = []
    for label, path in sorted(artifact_paths.items()):
        record = _file_record(path, workspace)
        header = (
            f"\n## {label}\n"
            f"- path: `{record['path']}`\n"
            f"- sha256: `{record.get('sha256', '')}`\n"
            f"- bytes: `{record.get('bytes', 0)}`\n"
        )
        if path.suffix.lower() in {".json", ".md", ".txt", ".html"}:
            rows.append(header + "```text\n" + _read_text_limited(path, max_chars) + "\n```")
        else:
            rows.append(header + "Binary or image artifact; inspect the path directly when possible.")
    return "\n".join(rows).strip()


def _screenshot_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SCREENSHOT_SUFFIXES
    )


def _read_text_limited(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated by presentation_review prompt budget]"


def _resolve_inside(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(
    report_output: Path,
    markdown_output: Path | None,
    report: dict[str, object],
) -> None:
    write_json_atomic(report_output, report)
    if markdown_output is not None:
        _write_text_atomic(markdown_output, render_markdown_report(report))


def render_markdown_report(report: dict[str, object]) -> str:
    """Render a compact Markdown summary for human review packets."""

    feedback = report.get("aggregated_feedback")
    feedback_dict = feedback if isinstance(feedback, dict) else {}
    lines = [
        "# Presentation Review",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Production ready: `{report.get('production_ready')}`",
        f"- Presentation: `{report.get('presentation_id')}`",
        f"- Prompt: `{report.get('prompt_path')}`",
        "",
        "## Reviewers",
        "",
    ]
    for reviewer in report.get("reviewers", []):
        if not isinstance(reviewer, dict):
            continue
        lines.append(
            f"- {reviewer.get('display_name')} "
            f"(`{reviewer.get('provider')}:{reviewer.get('model')}`)"
        )
    lines.extend(["", "## Blockers", ""])
    blockers = feedback_dict.get("blockers")
    if isinstance(blockers, list) and blockers:
        for blocker in blockers:
            if isinstance(blocker, dict):
                lines.append(
                    f"- `{blocker.get('reviewer_id')}` "
                    f"`{blocker.get('criterion')}`: {blocker.get('comment')}"
                )
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = feedback_dict.get("warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            if isinstance(warning, dict):
                lines.append(
                    f"- `{warning.get('reviewer_id')}` "
                    f"`{warning.get('criterion')}`: {warning.get('comment')}"
                )
    else:
        lines.append("- None")
    summary = str(feedback_dict.get("summary") or "").strip()
    if summary:
        lines.extend(["", "## Summary", "", summary])
    lines.append("")
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
