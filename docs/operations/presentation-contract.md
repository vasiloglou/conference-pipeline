# Presentation Contract Runbook

Use this runbook to inspect the workspace-level presentation manifest, read
stage status, and backfill historical deck artifacts into the Phase 22
contract.

Phase 22 defines the presentation contract before the canonical workflow
registry consumes it. This runbook is about schema inspection and backfill, not
about claiming end-to-end `workflow run` support for the full presentation
lifecycle.

## Scope

The shared slide template reserves the full inner header width for content,
events, and resources titles. It does not place a logo at the top right.
The opening slide retains its title logo, and the other slides retain the
bottom-right brand mark. This policy applies to every conference and both
topic and executive decks on subsequent HTML/PDF and video-frame renders.
Existing exported files require rendering again to adopt the new layout.

This runbook covers:

- locating `conferences/<slug>/decks/presentation_manifest.json`
- reading stage status, support metadata, and artifact digests
- backfilling a legacy workspace from existing deck artifacts
- troubleshooting fail-closed presentation-manifest validation errors

## Manifest Location

The manifest lives at `conferences/<slug>/decks/presentation_manifest.json`.
For example:

```text
conferences/neurips2025/decks/presentation_manifest.json
```

## Inspecting the Manifest

The manifest has this top-level shape:

```json
{
  "schema_version": 1,
  "generated_at": "2026-04-23T20:00:00Z",
  "generator": "scripts.backfill_presentation_manifest@<commit>",
  "workspace": "/abs/path/to/conferences/neurips2025",
  "content_sha256": "...",
  "stage_specs": [],
  "presentations": []
}
```

When reading `presentations[*].stage_states`, remember:

- `missing` means no canonical artifact evidence was found for that stage
- `partial` means some stage evidence exists, but the full required artifact
  set does not
- `complete` means every required artifact for that stage exists

For historical backfill, `missing` and `partial` are valid outcomes. They are
the truth-preserving states, not errors.

## Backfilling a Historical Workspace

Use the backfill command:

```bash
python3 scripts/backfill_presentation_manifest.py --workspace conferences/neurips2025
```

The backfill script:

- writes exactly one workspace-level manifest under `decks/`
- includes numbered topic deck directories plus `master_presentation` and
  `executive_summary`
- warns and skips non-canonical directories such as `special-topics/`
- records `missing`, `partial`, and `complete` stage states from existing
  artifacts only

The script validates the manifest immediately after writing it. A successful run
prints the number of backfilled presentations and a shortened content digest.

## Phase 24 Command Contract

First-class presentation workflow stages now use the workspace manifest to plan
core owner-script argv. The workflow selects one entry from
`decks/presentation_manifest.json` with `--presentation-type` and `--topic`
where required, then resolves declared artifacts under `decks/<presentation>/`.

Workflow-planned presentation scripts use decks/<presentation>/... paths; presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning.
Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane.

Presentation stage dry-run receipts include command_metadata with exact argv, selected scope, input artifacts, output artifacts, support level, provider/tool requirements, and unsupported_reason.
Unsupported reusable stages expose unsupported_reason instead of claiming script execution.
Video generation is dry-run/planning-only in Phase 24 and does not prove final MP4 generation, visual validation, or cross-conference proof.

Unsupported exceptions are explicit support boundaries:

- `extract_topic_content.py` remains unsupported for reusable workspace execution until manifest-backed source-attribution inputs are supplied.
- Legacy `pdf_export` is compatibility-only; structured `render_export` owns reusable HTML/PDF export proof.
- `human_review` remains a manual checkpoint and does not claim automated script execution.

## Phase 25 Presentation Validation

Phase 25 validates structural readiness of manifest-registered presentation
artifacts; v1.63 adds a governed vision-LLM visual QC report over rendered
screenshots as a content gate for production decks.

Phase 25 does not re-run live generation or assert subjective visual quality
outside governed content gates.

## Presentation Blueprint Contract

Phase 14 makes `presentation_blueprint` the first production presentation
checkpoint. The blueprint is a source-grounded story contract over the full
analysis document and conference corpus, not a slide-count expansion brief. It
must define thesis, audience promise, telescopic section map, subtopic clusters,
source/paper plan, slide intents, visual roles, narration beats, conclusion,
and a duration budget before any deck contract is generated.

Deck contract generation, source binding, visual planning, render/export,
delivery generation, TTS, video rendering, release, proof, and bulk rollout require an
approved `blueprint_review.json`. The approval token is valid only when
`approved_for_draft_generation: true` is recorded with the current
`blueprint_sha256` and explicit gate evidence for gate type: deterministic,
gate type: LLM-judge, and gate type: human. A hash mismatch between
`presentation_blueprint.json` and `blueprint_review.json` blocks
`deck_contract_generation`.

mock/dry-run blueprint artifacts are planning/test evidence only; they cannot
close production or substitute for a live, reviewed presentation blueprint.
Topic blueprints should plan for <=38 min before TTS so final rendered
single-presenter and NotebookLM videos stay under the hard >40 min failure
threshold. Phase 13 owns final rendered duration and MP4/audio receipt checks;
Phase 14 owns the upstream story, source, visual-role, and timing contract that
makes those render gates achievable.

Topic-deep-dive blueprint generation must start from an explicit audience
frame after the analysis synthesis exists. `presentation_frame.json` is the
post-analysis handoff that separates the retrieval/topic-strategy title from the
public deck title. It must include `retrieval_title`, `audience_title`,
`field_thesis`, at least three `general_questions`, `source_analysis_path`, and
an `opening_anchor_policy` before `presentation.generate_editorial_blueprint`
runs. Production plans must pass it with `--presentation-frame
decks/<id>/presentation_frame.json`; missing, invalid, or deck/topic-mismatched
frames fail closed instead of falling back to the retrieval title.

The opening section must establish field-level questions and stakes before it
dives into specialist paper mechanisms. `presentation_frame_validation` is
recorded in `presentation_blueprint.json` and must pass before
`deck_contract_generation`. A deck that starts with method terms before the
audience frame, or that ignores the frame's audience title/general questions,
is not ready for deck-contract generation even when the source evidence is
otherwise complete.

Bulk regeneration remains blocked until a canary deck has passed the full
blueprint-first flow and human review. The selected canary must be approved
with current source, render, video, artifact-truth, and human-review evidence
before bulk presentation regeneration begins.

## Source-Led Canary Acceptance Contract

Canary acceptance is a source-led and artifact-truth gate, not a receipt-only
checkpoint. Before canary acceptance, production must have a current source
corpus inventory, speaker/author/work ledger, source repair findings, research
cluster plan, and artifact-truth validation for the selected presentation.
These artifacts prove that visible people, works, claims, source clusters,
rendered files, media files, and approval receipts are tied to current source
evidence before a deck can advance to canary review, bulk readiness, release,
or proof.

The source corpus inventory and speaker/author/work ledger are the upstream
eligibility records. Source repair findings record unresolved or corrected
source candidates instead of silently dropping them. The research cluster plan
groups the eligible sources used by blueprinting and deck/source-binding
stages. Artifact-truth validation then checks the generated artifacts and
receipt chain directly before canary acceptance.

These gates cannot be replaced by scaffold outputs, old numbered Markdown files,
generic source bindings, blank PDFs, flat/silent videos, synthetic fixtures,
or receipt-only edits. When a local canary exposes conference-specific
failure facts, record those facts in the owning Phase planning context rather
than in root platform docs.

## Per-Step Lineage Ledger

`decks/<deck_id>/step_lineage.json` (schema 1) records, per presentation
step, the digests of the inputs it read (`inputs`), the outputs it wrote
(`outputs`) and the switches it was given (`values_sha256`), plus a per-path
`history` of `(step, sha256)` writes. `conference_pipeline.presentation.step_lineage`
decides `should_skip` from it (see the resume section of the
[pipeline workflow](../workflows/pipeline-workflow.md)); the stage receipt
records `checks.step_lineage[deck_id]` (`ledger`, `ledger_sha256`,
`output_digest`, steps run / reused / adopted, `decisions`) and
`checks.step_lineage_output_digest`, which the receipt's own `output_digest`
folds in. `decisions` says why a resumed run re-ran a step
(`input_changed:<path>`, `output_changed:<path>`, `status_not_accepted:<path>`,
`values_changed`, `inputs_changed:declared`, `no_sidecar`).

## Structured Presentation Production Contract

Phase 14 production evidence is structured. The production artifact chain after
an approved blueprint is:

- `presentation_frame.json`
- `presentation_deck.json`
- `presentation_deck_receipt.json`
- `source_binding_report.json`
- `visual_plan.json`
- `asset_inventory.json`
- `render_spec.json`
- `delivery_single_speaker.json`
- `delivery_notebooklm.json`
- `6_presenter_script.md`
- `video_sync_validation_single_speaker.json`
- `video_sync_validation_notebooklm.json`

`presentation_deck.json` is the deck contract consumed by source binding,
visual planning, render specification, render/export, delivery, and video-sync
validation. `source_binding_report.json` proves claims, slide intents, and
visual roles are bound to source artifacts before rendering. `visual_plan.json`
and `asset_inventory.json` are the only production visual planning proof.

The conference runners produce this chain through the shared
`conference_pipeline.production.presentation_stage` (see
[Shared Production Stages](../reference/stage-owner-audit.md#shared-production-stages)).
Per deck it runs, in order: paper matching; source corpus inventory,
speaker/author/work ledger, source repair and research cluster plan; the
editorial blueprint plus its review (LLM-judge artifact when present); the deck
contract; source pack and source binding; figures by route (`compiler`: spec
critic and numeric-claims ledger, compile gate, variety quota; `drawsvg`:
two-pass codegen plus semantic review); `validate_svgs`; render spec; bookend
slides; HTML, PDF and PPTX; screenshot inspection; the vision judge with a
bounded repair loop (or a recorded archetype regression on the compiler route)
and the operator's `visual_qc_acceptance.json`; artifact truth. The delivery
substage adds the presenter script, engaging narration with its raw-JSON sync
guard, presenter and two-host narration lint/review/fix plus series review,
the Gate A whole-presentation review before any TTS, both videos and video
sync validation. Executive decks are verified against the digest chain's
artifacts (`build_executive_summary.py`) and the same render gates rather than
rebuilt. Every step is skipped on resume only while its per-deck step ledger
says its inputs and outputs are unchanged.
`render_spec.json` is the render/export source of truth for `presentation.html`,
`presentation.pdf`, and `presentation_slide_mapping.json`. The delivery
contracts, `delivery_single_speaker.json` and `delivery_notebooklm.json`, are the
production narration inputs for the two video families. The single-speaker
`6_presenter_script.md` is declared narration evidence for whole-presentation
review, and both
`video_sync_validation_single_speaker.json` and
`video_sync_validation_notebooklm.json` must pass before release or Phase 13
proof can close.

Old numbered slide draft Markdown files are legacy-only numbered Markdown. They may remain
as debug-export-only or local repair aids with `production_evidence: false`, but
they do not prove production readiness, Phase 13 canary status, render source
freshness, or delivery completeness. Production command planning, manifests,
receipts, Phase 13 evidence, generated-plan audits, docs, and skills must reject
old numbered Markdown production proof and report
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED` from `legacy_markdown_gate.py` when the
old ladder is used as production evidence.

Run the full validation harness through the top-level wrapper:

```bash
conference-pipeline presentation-validate --workspace conferences/neurips2025 --presentation-type topic-deep-dive --topic 01_tools_and_open_source --mode dry-run --run-id phase25-presentation-validate --summary
```

The wrapper delegates to the same receipt-backed workflow machinery as
`workflow run`. It selects the validation stages, binds receipts to the
presentation manifest content digest and selected scope, and writes
`presentation_validation_report.json` under
`storage/workflow-runs/<run-id>/`.

The report top-level fields are:

- `schema_version`
- `generator`
- `short_commit`
- `run_id`
- `workspace`
- `manifest_path`
- `manifest_content_sha256`
- `selected_scope`
- `strict_required_artifacts`
- `totals`
- `checks`

Per-check status values are pass, fail, warn, and skipped_missing.
Per-check severity values are structural and content.
Each check record uses these values to distinguish deterministic structural
checks from content observations.
Structural failures fail the owning validation stage. Content observations can
also fail the stage when a governed content gate is required, such as
vision-LLM visual QC over rendered slide screenshots.
Topic-deep-dive presentation entries may declare `target_hyperlinks_per_slide`
and `target_svgs_per_slide`. Validation enforces those targets with
`hyperlink_density_meets_target` on `4.5_draft_slides_with_hyperlinks.md`,
`hyperlink_preservation_4_5_to_5` between the hyperlink and figure-enriched
markdown files, and `svg_density_meets_target` on
`5_draft_slides_with_images.md`.

Hyperlink density is a content gate, not permission to add unrelated sources.
`presentation.enrich_hyperlinks` may only add density links when the candidate
label or keywords match the slide text. If the source pack cannot provide enough
matching links, the deck should fail validation or carry an explicit blocker
instead of inserting arbitrary references.

Every production deck closeout must run `presentation.link_audit` after HTML/PDF
export. The audit rejects `file:///`, `/Users/...`, `../analysis/...`,
`decks/analysis/...`, and repo-local `conferences/<slug>/...` links in both
`presentation.html` and `presentation.pdf`. Topic decks can use public web
sources such as OpenReview/arXiv. Conference executive overview decks require
externally shareable source links, normally Google Drive URLs for the analysis
markdown sources.
Reveal.js validation also fails rendered decks whose visible source chips are
generic labels such as `Ref 1`, whose opening title slide lacks the
RelationalAI reference treatment, or whose content/figure slides lack a visible
RelationalAI brand mark. The corresponding check IDs are
`revealjs_reference_labels_visible`, `revealjs_title_slide_branded`, and
`revealjs_content_brand_mark_present`.
When `strict_required_artifacts` is true, missing in-scope figure-generation,
video-generation, PDF-export, visual-QC, or human-review artifacts are converted
from `skipped_missing` to `fail`.

Production video closeout must validate final MP4 timing before release or
human approval for TTS/rendered video. `video_stream_durations_match` compares
the final media container's audio and video stream durations so a player cannot
continue audio over a black or ended video track. `video_duration_matches_build_receipts`
compares the final media duration against build evidence such as
`video_build*/narration_track.m4a`, `slides_concat.txt`, and duration metadata
JSON. `video_tail_not_black` inspects the final tail of each MP4 for sustained
black frames so release checks catch player-visible black endings even when
container durations look valid. `video_duration_under_40_minutes` fails any
rendered presentation video above 2400 seconds. These checks apply to every
rendered presentation video style declared by the manifest, including
single-presenter and two-host variants.
The shared production chain (`conference_pipeline.production`) runs the same
four checks path-based, without a presentation manifest, right after
`validate_video_sync.py`: `scripts/presentations/validate_video_delivery.py`
writes `decks/<id>/video_delivery_validation.json` for the presenter and
NotebookLM MP4s of every topic and executive deck, and any `fail` check
blocks the deck.

Before TTS, topic decks must also pass `presentation_editorial_gate.py`. On the
legacy markdown route the gate requires an `editorial_storyboard.json`
compression artifact; on the compiled route the shared production chain runs it
with `--narration-only --advisory` (no slide markdown or storyboard exists
there), records `decks/<id>/presentation_editorial_gate_report.json` on the deck
record,
and leaves the budget call to the operator before TTS. The gate checks that
single-presenter and NotebookLM estimates stay under 40 minutes, warns above
the 34-38 minute working target, blocks excessive figure-detail/zoom density,
flags obvious redundant claim clusters, rejects repeated generic SVG patterns,
and requires important FigureSpec entities to appear in narration. Figure
references should usually be integrated into parent slide narration; separate
zoom/detail beats are reserved for complex or central visuals.

Before bulk presentation regeneration, production must create one-topic
Phase 13 canary objective evidence with
`build_phase13_canary_objective_evidence.py`, then attach that evidence to the
canary review packet with `build_phase13_canary_review_packet.py`. The
objective evidence probes both video styles for MP4 duration, audio duration,
slide-concat receipt duration, first/middle/tail frame luma, and a contact
sheet; it fails over-40-minute videos, receipt drift, missing artifacts, and
black tail frames. The review packet summarizes required artifacts, MP4
duration probes, editorial gate status, strict-validation state, parsed
objective-evidence status/contact-sheet presence, and
`human_review_receipt.json` approval state. Bulk rollout remains
blocked until the canary receipt records an approved decision and the canary
packet is regenerated to `approved_for_bulk_rollout` with parsed objective
evidence still passing. The approval receipt must capture SHA-256 digests and
status snapshots for the human review markdown, canary review packet, and
objective evidence report so later bulk rollout proves exactly what the
operator approved. Regenerating the canary packet after approval must validate
those receipt digests against the current human review markdown and objective
evidence report; stale or mismatched approval evidence blocks rollout.
`run_phase13_rollout.py` is the guarded pre-TTS rollout owner: it blocks
non-canary decks without approved canary review plus an approved canary packet,
records provider environment key presence without secret values, runs the
single-presenter and NotebookLM narration lint/review gates, writes the cross-series
`presenter_narration_series_review.{json,md}` and
`narration_series_review.{json,md}` receipts with
`review_narration_series.py`, and uses `--stop-after-script-json` plus
`--stop-after-transform` so it does not synthesize TTS or render MP4s.
After approval and a refreshed rollout plan, run
`verify_phase13_bulk_readiness.py`; non-canary `run_phase13_rollout.py
--execute` is not accepted until the readiness report is
`ready_for_bulk_rollout`.
After bulk rollout and final strict validation, run
`verify_phase13_completion.py`; Phase 13 completion is accepted only when that
report is `complete`.

Visual QC has two required production artifacts: the Playwright rendered
screenshot report (`visual_qc_report.md`) and the governed vision-LLM review
report (`visual_llm_qc_report.json`). The LLM report is generated from rendered
slide screenshots with model-governance task `presentation.visual_qc` and fails
closed when the provider/model is unavailable, the report is missing, JSON is
not parseable, or any blocker issue is reported.

Whole-presentation review is a separate production gate. The
`presentation.presentation_review` owner consumes the presentation
requirements, generated narration script, rendered slide screenshots, blueprint
review, deck contract, render spec, source binding, delivery contract, video
sync receipt, visual-QC report, and contact-sheet evidence, then asks Claude
Code Opus 4.8, Codex 5.5, and Gemini to review guideline adherence,
technical-audience pleasantness, production readiness, aesthetic pleasantness,
slide clarity versus congestion, and artifact-contract adherence. The required
artifacts are `presentation_review.json`, `presentation_review.md`, and
`presentation_review.prompt.md`; dry-run evidence may plan reviewer commands,
but production readiness requires live reviewer outputs with no blockers.
Phase 13 canary review packets and human approvals must snapshot the current
`presentation_review.json` digest before bulk rollout.

The single-speaker `6_presenter_script.md` Markdown file is declared delivery
evidence for this whole-presentation review gate. It remains paired with the
structured `delivery_single_speaker.json` and `delivery_notebooklm.json`
contracts; older numbered slide Markdown stays legacy/debug-only.

Presentation figure generation must be programmatic SVG generation through a
matplotlib, Plotly, or svgwrite owner before it can satisfy reusable production
completion. Legacy paper-figure extraction/insertion scripts are deprecated for
presentation completion evidence and must not be selected as
`presentation.svg_figure_generation` owner commands.

Production figure generation must also write source-grounded `FigureSpec`
records before rendering SVGs. Each spec must tie the requested figure to slide
claims, ranked papers or source attributions, a depiction type, required
entities, visual encoding, source trace, and the governed
`presentation.svg_figure_generation` model. The required code/spec generation
model is Anthropic `claude-sonnet-5`; generated inventories and semantic
review receipts must record that model id. Bullet-box density fallbacks,
generic scaffold diagrams, copied slide prose, and placeholder chart ticks do
not satisfy figure generation.

Figure validation is a two-part gate: `validate_svgs.py` checks structure,
layout, fonts, contrast, and XML validity, while
`review_programmatic_figures.py` checks semantic informativeness against the
FigureSpec and asset inventory. Both reports must pass before Reveal.js render,
visual QC, video generation, or human approval.

When validation finds a renderer-only defect after a live FigureSpec run has
already produced `figure_specs.json` and `sonnet_figure_spec_receipt.json`,
the repair loop may rerender from those existing specs with
`--figure-specs-input decks/<id>/assets/generated/figure_specs.json` and
`--figure-specs-output decks/<id>/assets/generated/figure_specs.json`. This
path must preserve the live Sonnet receipt, mark the asset inventory governance
status as `live_sonnet_specs_reused_local_render`, clean stale generated SVGs,
and rerun both structural and semantic validation before any render or video
stage.

The canonical per-tool usage pointers for presentation scripts live in the
[Stage Script Registry Tool Usage Index](../reference/stage-owner-audit.md#stage-script-registry-tool-usage-index).
Planning skills and generated plans should cite those registry `usage_doc_refs`
for command shape and proof evidence instead of copying local notes.

## Conference-Level Executive Overview Decks

`analysis/executive_summary.md` and `analysis/trend_report.md` are source
reports. They do not satisfy the presentation deliverable when the brief asks
for executive summary presentations.

When conference overview decks are in scope, production must create two
manifest-backed decks:

- `decks/executive_summary_short`
- `decks/executive_summary_long`

The reusable owner is:

```bash
scripts/presentations/generate_executive_overview_decks.py \
  --workspace conferences/<slug> \
  --analysis-manifest analysis/analysis_manifest.json \
  --executive-summary analysis/executive_summary.md \
  --trend-report analysis/trend_report.md \
  --source-link-map decks/executive_source_link_map.json \
  --require-source-link-map \
  --receipt-output decks/executive_overview_decks_receipt.json
```

Both decks must then be registered in `decks/presentation_manifest.json` with
`presentation_type: executive-summary`. The generator writes the structured
presentation chain for each deck: `presentation_frame.json`,
`presentation_blueprint.json`, `presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `asset_inventory.json`,
`delivery_single_speaker.json`, `delivery_notebooklm.json`, and
`render_spec.json`. Render with `md_to_revealjs.py --render-spec`, export with
`export_revealjs_pdf.py --render-spec`, and check rendered-slide QC before
video generation, release, or proof. The retired numbered Markdown ladder is
debug-only and cannot satisfy executive-summary production proof. A topic-only
`topic-deep-dive` manifest is incomplete when short/long executive overview
decks are requested.

Executive overview decks must use the same bookend semantics as topic decks:
slide 2 is `slide_kind: events`, titled as the source surface for the readout,
and the final slide is `slide_kind: resources`. Let `x` be the number of
executive synthesis points in the deck spine; slide 2 must carry no more than
`x` important events on the left and at least `x` key resources on the right.
Important events include keynotes, invited talks, workshops, oral sessions or
oral talks, panels, competitions, and best-paper or awards contexts when
available. The final resources slide must include the presentation and analysis
report link for every in-scope topic identified by the conference milestone
context or production brief. Production-ready public presentation links should
come from `decks/executive_presentation_link_map.json` via
`--presentation-link-map` and `--require-presentation-link-map`; workspace-local
presentation links are allowed only for local review rebuilds and must not close
the presentation link audit.

The executive overview narrative must not be an index wrapped around an index.
Conference-local context may define a candidate lesson pool, target point count,
duration, and short-versus-long expansion strategy, but the produced executive
decks must select from that pool rather than reciting the full inventory. Each
selected point should bind to transcript-backed evidence: short quotes from
keynotes, workshops, oral sessions, panels, or talks when available, and concise
grounded paraphrases otherwise. Slide 2 and the final reading slide remain useful
bookends, but they cannot substitute for a quote-backed synthesis spine.

The `--require-source-link-map` flag is the Google Drive sync contract. If a
source markdown file has not yet synced or been uploaded to Drive, production
must stop with `source_link_contract_blocked` and remediation to create
`decks/executive_source_link_map.json`; it must not emit local markdown links as
placeholders because PDF export can turn those into unusable file links.

Example report shape:

```json
{
  "schema_version": 1,
  "generator": "conference-pipeline/presentation-validate@1",
  "short_commit": "dev",
  "run_id": "phase25-presentation-validate",
  "workspace": "neurips2025",
  "manifest_path": "decks/presentation_manifest.json",
  "manifest_content_sha256": "...",
  "selected_scope": {
    "presentation_id": "01_tools_and_open_source",
    "presentation_type": "topic-deep-dive",
    "topic": "01_tools_and_open_source"
  },
  "strict_required_artifacts": false,
  "totals": {
    "fail": 0,
    "pass": 7,
    "skipped_missing": 2,
    "warn": 1
  },
  "checks": [
    {
      "check_id": "slide_mapping_schema",
      "status": "pass",
      "severity": "structural",
      "message": "presentation_slide_mapping.json parses",
      "evidence_paths": [
        "decks/01_tools_and_open_source/presentation_slide_mapping.json"
      ],
      "validation_stage_id": "render_spec_generation_validation",
      "checkpoint_stage_id": "render_spec_generation"
    }
  ]
}
```

## Phase 26 Proof Receipts

Phase 26 records receipt-backed presentation proof evidence without claiming
live cross-conference production. The proof driver writes
`presentation-proof-report.json` alongside workflow stage receipts and records
known limitations in the report.

Use the NeurIPS reference proof command:

```bash
conference-pipeline presentation-proof --workspace conferences/neurips2025 --presentation-type topic-deep-dive --topic 01_tools_and_open_source --mode dry-run --run-id <run-id>
```

Use the ICLR scaffold routing proof command:

```bash
conference-pipeline presentation-proof --workspace conferences/iclr2025 --presentation-type topic-deep-dive --topic 01_iclr_2025_proof --mode dry-run --stages deck_contract_generation --run-id <run-id>
```

The report path is
`conferences/<slug>/storage/workflow-runs/<run-id>/presentation-proof-report.json`.
The consolidated proof summary is
`.planning/milestones/v1.3-phases/26-neurips-backfill-and-iclr-presentation-proof/26-PROOF-SUMMARY.md`.

Committed Phase 26 run ids:

- NeurIPS dry-run: `phase26-neurips-presentation-proof`
- ICLR dry-run scaffold routing: `phase26-iclr-presentation-dryrun`
- ICLR live-denied preflight check: `phase26-iclr-presentation-live-denied`

## What Phase 22 Does Not Claim

Phase 22 defines the contract first. It does **not** claim that the canonical
workflow registry already consumes `presentation_manifest.json`, nor that
`workflow run` already executes every stage in `deck_contract_generation` through
`human_review`.

That separation is intentional: the contract must exist before later phases can
wire stage registration, receipts, proof paths, and generalized scripts onto
it.

## Troubleshooting

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `Missing presentation_manifest.json` | The workspace has not been backfilled yet, or the manifest was deleted | Run the backfill command above |
| `unknown stage id` | The manifest was edited with an unsupported stage name | Re-run the backfill command or regenerate with the current emitter |
| A resumed deck re-ran a step you expected it to skip | An input, output or switch the ledger recorded moved | Read `checks.step_lineage[<deck>].decisions` on the stage receipt (or `last_decision` in `decks/<deck>/step_lineage.json`) |
| A deck built before v5.1 re-runs everything on `--resume` | No `step_lineage.json` exists for it | Run the presentation stage once with `--adopt-existing-outputs` |
| `schema_version N is not supported` | The manifest was written by an incompatible emitter | Re-run the backfill command to emit schema version 1 |
| `must stay inside the conference workspace` | A declared artifact path is absolute or escapes the workspace via `../` | Re-emit the manifest from trusted workspace files |
| `sha256 mismatch` | A declared file or directory artifact changed after the manifest was written | Re-run the backfill command to refresh digests |

## Phase 01 Canary Acceptance Guidance

Tests in `tests/test_presentation_brand_template.py` are guardrails (regression
guardrails), not substitutes for viewer inspection. These tests are guardrails
against known failure modes and do not replace visual review. `01-HUMAN-REVIEW-PACKET.md` is review
evidence; explicit human approval is still required before any Topic 4 canary
artifacts are promoted. Bulk rollout remains blocked until a separate rollout
plan is created.

## Related Documentation

- [Presentation Manifest Schema](../design/presentation-manifest-schema.md)
- [Pipeline Workflow](../workflows/pipeline-workflow.md)
- [Source Of Truth](../source-of-truth.md)
