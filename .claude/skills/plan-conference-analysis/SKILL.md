---
name: plan-conference-analysis
description: Plan a complete conference-specific GSD milestone from this repository's conference pipeline contracts. Use when asked to create or update planning for a conference analysis milestone, scaffold conference planning artifacts, validate dry-run/mock support boundaries, or turn a PRD-style conference brief into GSD context, requirements, roadmap, state, and workspace scaffold files.
---

# Plan Conference Analysis

Create a conference-specific GSD milestone from a PRD-style conference brief
using this repository's platform contracts. Keep root guidance platform-general;
put conference assumptions and proof status in planning-owned context packets.

## Required Reads

Read these before drafting:

- `CLAUDE.md`
- `AGENTS.md`
- `docs/source-of-truth.md`
- `docs/reference/pipeline-capabilities.md`
- `docs/reference/stage-owner-audit.md`
- `docs/workflows/pipeline-workflow.md`
- `docs/workflows/conference-analysis-playbook.md`
- `docs/extraction/exhaustive-extraction-guide.md`
- `docs/operations/analysis-contract.md`
- `docs/operations/presentation-contract.md`
- `docs/operations/future-conference-storage.md`
- `docs/design/analysis-manifest-schema.md`
- `docs/design/presentation-manifest-schema.md`
- `docs/cli.md`
- `src/conference_pipeline/model_governance/registry.yaml`
- `src/conference_pipeline/model_governance/callsite_inventory.json`
- `.planning/milestones/*-CONTEXT.md`

Read references as needed:

- `references/brief-contract.md`
- `references/planning-artifacts.md`
- `references/roadmap-backbone.md`
- `references/official-workflow-mapping.md`
- `references/support-boundaries.md`

## Brief Intake

Prefer a PRD-style conference brief. Require conference identity, source URLs
and auth notes, target outputs, proof mode, topic strategy, provider
preferences, provider-cost approval posture, storage/share targets, acceptance
criteria, platform version, and conference name.

Ask targeted questions for missing truth-critical fields. Do not write
canonical planning artifacts until required fields are resolved. Infer from an
existing workspace only when the source is visible and the inference is recorded
in the draft.

Always ask the operator which research engine runs the deep-dive research
lanes (07C), and never assume one. Both run on the conference's own hybrid
index and local model:

- `gpt_researcher` -- the GPT-Researcher package with our retriever plugin:
  broader retrieval (planning, sub-queries), passage-linked citations.
- `inhouse` -- one retrieval and one local-model synthesis per lane: better
  grounded and quotes its sources in the ICML 2026 topic benchmark
  (`conferences/icml2026/storage/workflow-runs/icml2026-v4.0-live-production-20260711/engine-comparison-20260929/`).

Record the answer in the draft and in `conferences/<slug>/config.yaml` as
`workflow.research_engine` with a `workflow.research_engine_decision` note
(who, when, why). The stage refuses to run without it
(`conference_pipeline.research.engine_choice`).

When exhaustive extraction is in scope, also require:

- local LLM endpoint label (configures `EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL`)
- local LLM base URL (configures `EXTRACTION_LOCAL_LLM_BASE_URL`; must be a
  loopback address — non-local endpoints are rejected)
- local LLM model name (configures `EXTRACTION_LOCAL_LLM_MODEL`)

## Draft Then Finalize

Default mode writes reviewable drafts first. The approval summary must show:

- generated milestone identity
- support claims
- roadmap phases
- changed files

Finalize only after approval. `--autonomous` may skip the approval gate only
when explicitly requested; it must still preserve the draft/finalizer boundary
and exact support labels.

After every generated draft and after every manual planning edit, run the
generated-plan audit, the workspace gitignore policy generator/check, the
stage owner registry audit, and the unified stage script registry verifier:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py --workspace conferences/<slug> --write --check --receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json
PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit --plan-root <draft-root>
PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py --plan-root <draft-root> --skills-root .claude/skills --docs-root docs --strict
```

Do not finalize while the gitignore check or either registry audit is blocked.
`scripts/generate_workspace_gitignore.py` is mandatory because workspace
artifact Git policy must be derived from `storage/storage_manifest.json`:
`storage_class: git` stays trackable, while `public`, `backup`, and `cache`
families are ignored unless a small proof/control file is declared in
`gitignore_exceptions`. The unified registry in
`src/conference_pipeline/stage_owner_registry.json` is mandatory because the
plan must use approved required scripts, reject deprecated completion owners,
and carry a registry digest that production receipts can verify later.

Generated plans must preserve the v1.63 Runtime Stage Contract Gate Pattern:
every live production stage that can write `status: "succeeded"` must be
registry-bound through `stage_bindings.<stage>.runtime_contract`, call
`conference_pipeline.stage_contracts.verify_stage_contract`, and write
`stage_contract.accepted_evidence` plus `stage_contract.blockers` into its
stage receipt. A stage may be explicitly marked non-production only when the
runner cannot produce live completion evidence. Receipts also carry v5.1
receipt lineage (`output_digest` + `lineage`, bound through
`conference_pipeline.receipt_lineage.finalize_receipt`) so resume fails closed
when an upstream stage's output moved.

If the target workspace or milestone context exists, ask whether to update the
existing plan or create a new version. Never silently merge into existing
planning state.

### Anti-Tautology Runtime-Contract Coverage And Receipt-Collision Guard

A stage binding's `required_script_ids` must never be a paper tiger. The
generated-plan audit runs `stage_contract_coverage.audit_runtime_contract_coverage`
and `stage_contract_coverage.audit_receipt_name_collisions` against the live
`src/conference_pipeline/stage_owner_registry.json` on every audit pass:

- **Coverage rule:** every owner id in a stage binding's `required_script_ids`
  must be evidenced by a `required_receipts`/`required_outputs`/`required_checks`
  entry in that binding's `runtime_contract` that names an artifact only the
  real owner produces -- not just an echo of the owner id back onto the
  receipt. An owner id with no evidencing entry blocks the plan audit unless
  the binding governs it with a `coverage_exempt_script_ids` entry plus a
  matching non-empty `coverage_exempt_rationale` string (e.g. heuristic
  prefilters, `*_policy` modules, and combined-runner aliases that have no
  distinct artifact of their own). An exemption without rationale is itself a
  blocker.
- **Receipt-name collision guard:** within one stage binding, two distinct
  required-receipt paths must not share a leading stage-number prefix (e.g.
  `14A`) with a different trailing slug -- that is exactly the naming
  collision that let a legacy heuristic receipt masquerade as a different
  owner's evidence. New collisions are blocked; only the registry-governed
  `receipt_prefix_collision_allow` allowlist (with rationale) may permit a
  historical pair to remain, and no existing receipt family is ever renamed
  to close a collision.

### Pre-Spend Contract Gate And Resume Economics

Generated plans must require, and the generated-plan audit enforces, that:

- Every paid provider stage evaluates its stage binding's `runtime_contract`
  `required_receipts`/`required_checks` in **precondition mode BEFORE the
  first provider call** -- dependency-order enforcement distinct from the
  budget/cost-estimate gate. A precondition failure writes a zero-spend
  blocker (`precondition_contract_unmet`) and stops the stage before any
  provider is invoked.
- Multi-unit stages refresh their top-level receipt after every completed
  unit (or write per-unit receipts plus a rollup), so a mid-run kill cannot
  mask N/M completed units as 0/M.
- Paid artifacts are reuse-by-default on `--resume`; any explicit
  regeneration must be named in the plan -- a silent regenerate-all pattern
  is forbidden.

### Provider-Substitution Governance And Credential Preflight

Use `references/provider-substitution-waiver.md`. Generated plans must
require, and the generated-plan audit enforces, that:

- Every provider-backed stage's `runtime_contract` equals-checks the
  receipt's actual `provider` and `model` against the approved PC-* contract
  list and requires `provider_calls_made: true` (and `boundary_method` where
  applicable), reusing the existing `equals`/`one_of`/`not_equals` predicate
  vocabulary -- no new predicates.
- The ONLY governed path around an approved provider contract is a
  `provider_substitution_waiver.json` matching the
  `references/provider-substitution-waiver.md` schema (`task_id`, `from`
  {provider, model}, `to` {provider, model}, `approver`, `rationale`,
  `cost_impact`, `expiry`). A bare manual-ack UnblockReceipt does NOT count.
- A Phase-1 cloud-credential preflight probe makes one cheap live call per
  provider named in Task Model Defaults, with a receipt, before a multi-day
  run -- mirroring the already-mandated local-LLM readiness preflight.
- Blocker classes `unauthorized_provider_substitution` and
  `credential_invalid` are named for an unapproved provider swap or an
  invalid/expired credential.

### Structured Gates

Use `references/structured-gates.md`. `production_handoff()` emits a
machine-checkable `structured_gates` field (gate id -> receipt path -> field
checks) alongside the existing prose gate table. `PG-TOPIC-STRATEGY`,
`PG-QUERY`, `PG-GPT-RESEARCHER`, and `PG-PAPER-MATCHING` must each have a
distinct `receipt_path` -- collapsing them to one receipt makes per-gate
checking impossible. The generated-plan audit fails the plan when
`structured_gates` is missing, malformed, or those four gates collapse to
one receipt path.

## Registry Usage Documentation

Use registry `usage_doc_refs` for tool instructions and registry `diagnostic_profile`/`diagnostic.next_actions` for recovery guidance. Generated plans must include these anchors when they name registry tools; raw script paths alone are not sufficient planning guidance.

| Script ID | Status | Path | Usage docs |
|-----------|--------|------|------------|
| `planning.analysis.scaffold` | `required` | `.claude/skills/plan-conference-analysis/scripts/scaffold_plan.py` | `docs/reference/stage-owner-audit.md#planning-analysis-scaffold` |
| `planning.analysis.audit` | `required` | `.claude/skills/plan-conference-analysis/scripts/audit_generated_plan.py` | `docs/reference/stage-owner-audit.md#planning-analysis-audit` |
| `gitignore.workspace_policy` | `required` | `scripts/generate_workspace_gitignore.py` | `docs/reference/stage-owner-audit.md#gitignore-workspace-policy` |
| `verification.stage_owner_audit` | `required` | `src/conference_pipeline/stage_owner_audit.py` | `docs/reference/stage-owner-audit.md#verification-stage-owner-audit` |
| `verification.stage_script_usage` | `required` | `scripts/verify_stage_script_usage.py` | `docs/reference/stage-owner-audit.md#verification-stage-script-usage` |
| `acquisition.workshop_acquisition_pipeline` | `required` | `src/conference_pipeline/acquisition/workshop_pipeline.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-acquisition-pipeline` |
| `acquisition.workshop_discovery` | `required` | `scripts/discover_openreview_workshops.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-discovery` |
| `acquisition.workshop_paper_download` | `required` | `scripts/download_openreview_workshop_papers.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-paper-download` |
| `acquisition.slideslive_media_recovery` | `required` | `src/conference_pipeline/slideslive_recovery.py`; `scripts/recover_iclr2026_slideslive_media.py` | `docs/reference/stage-owner-audit.md#acquisition-slideslive-media-recovery` |
| `acquisition.slideslive_slide_download` | `required` | `src/conference_pipeline/slideslive_slides.py`; `scripts/recover_slideslive_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slideslive-slide-download` |
| `acquisition.slide_reference_detection` | `required` | `src/conference_pipeline/slide_cleanup.py`; `scripts/detect_reference_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slide-reference-detection` |
| `acquisition.slide_cleanup` | `required` | `src/conference_pipeline/slide_cleanup.py`; `scripts/cleanup_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slide-cleanup` |
| `acquisition.browser_cookie_export` | `allowed` | `scripts/export_browser_cookies.py` | `docs/reference/stage-owner-audit.md#acquisition-browser-cookie-export` |
| `acquisition.cvf_openaccess_workshop_paper_download` | `allowed` | `scripts/download_cvf_openaccess_workshop_papers.py` | `docs/reference/stage-owner-audit.md#acquisition-cvf-openaccess-workshop-paper-download` |
| `analysis.topic_strategy` | `required` | `src/conference_pipeline/analysis/topic_strategy.py`; `scripts/build_topic_strategy.py` | `docs/reference/stage-owner-audit.md#analysis-topic-strategy` |
| `analysis.llm_full_read_extraction` | `required` | `src/conference_pipeline/extractions/llm_full_read.py`; `scripts/run_llm_full_read_extraction.py` | `docs/reference/stage-owner-audit.md#analysis-llm-full-read-extraction` |
| `analysis.gpt_researcher_stage` | `required` | `src/conference_pipeline/research/researcher.py` | `docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage` |
| `analysis.gpt_researcher_heuristic_prefilter` | `required` | `src/conference_pipeline/research/heuristic_prefilter.py`; `scripts/run_heuristic_prefiltered_research.py` | `docs/reference/stage-owner-audit.md#analysis-gpt-researcher-heuristic-prefilter` |
| `analysis.deep_dive_question_selection` | `required` | `src/conference_pipeline/research/deep_dive_lanes.py`; `scripts/select_deep_dive_questions.py` | `docs/reference/stage-owner-audit.md#analysis-deep-dive-question-selection` |
| `analysis.dual_deep_dive_research_lanes` | `required` | `src/conference_pipeline/research/deep_dive_lanes.py`; `scripts/run_dual_deep_dive_research.py` | `docs/reference/stage-owner-audit.md#analysis-dual-deep-dive-research-lanes` |
| `analysis.topic_reconciliation` | `required` | `src/conference_pipeline/analysis/topic_reconciliation.py`; `scripts/reconcile_topic_analysis.py` | `docs/reference/stage-owner-audit.md#analysis-topic-reconciliation` |
| `analysis.topic_reconciliation_reference_workflow` | `required` | `src/conference_pipeline/analysis/topic_reconciliation_reference_workflow.py`; `scripts/run_topic_reconciliation_reference_workflow.py` | `docs/reference/stage-owner-audit.md#analysis-topic-reconciliation-reference-workflow` |
| `analysis.topic_reference_validation` | `required` | `src/conference_pipeline/analysis/topic_reference_validation.py`; `scripts/validate_topic_reconciliation_references.py` | `docs/reference/stage-owner-audit.md#analysis-topic-reference-validation` |
| `analysis.topic_reference_candidate_resolution` | `required first-line remediation` | `src/conference_pipeline/analysis/topic_reference_candidate_resolution.py`; `scripts/resolve_topic_reference_candidates.py` | `docs/reference/stage-owner-audit.md#analysis-topic-reference-candidate-resolution` |
| `analysis.topic_reference_repair` | `fallback remediation` | `src/conference_pipeline/analysis/topic_reference_repair.py`; `scripts/repair_topic_reconciliation_references.py` | `docs/reference/stage-owner-audit.md#analysis-topic-reference-repair` |
| `podcast.focused_script` | `required` | `scripts/generate_focused_podcast.py` | `docs/reference/stage-owner-audit.md#podcast-focused-script` |
| `podcast.notebooklm_audio_overview` | `required` | `scripts/generate_notebooklm_podcast.py` | `docs/reference/stage-owner-audit.md#podcast-notebooklm-audio-overview` |
| `paper_matching.artifact_keypoints` | `required` | `scripts/match_papers_from_artifact_keypoints.py` | `docs/reference/stage-owner-audit.md#paper-matching-artifact-keypoints` |
| `presentation.source_pack` | `required` | `scripts/presentations/build_deck_source_pack.py` | `docs/reference/stage-owner-audit.md#presentation-source-pack` |
| `presentation.bind_presentation_sources` | `required` | `scripts/presentations/bind_presentation_sources.py` | `docs/reference/stage-owner-audit.md#presentation-bind-presentation-sources` |
| `presentation.executive_overview_decks` | `required` | `scripts/presentations/generate_executive_overview_decks.py` | `docs/reference/stage-owner-audit.md#presentation-executive-overview-decks` |
| `presentation.editorial_gate` | `required` | `scripts/presentations/presentation_editorial_gate.py` | `docs/reference/stage-owner-audit.md#presentation-editorial-gate` |
| `presentation.phase13_canary_objective_evidence` | `required` | `scripts/presentations/build_phase13_canary_objective_evidence.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-canary-objective-evidence` |
| `presentation.phase13_canary_review_packet` | `required` | `scripts/presentations/build_phase13_canary_review_packet.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-canary-review-packet` |
| `presentation.phase13_rollout` | `required` | `scripts/presentations/run_phase13_rollout.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-rollout` |
| `presentation.phase13_bulk_readiness` | `required` | `scripts/presentations/verify_phase13_bulk_readiness.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-bulk-readiness` |
| `presentation.phase13_completion` | `required` | `scripts/presentations/verify_phase13_completion.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-completion` |
| `presentation.generate_editorial_blueprint` | `required` | `scripts/presentations/generate_editorial_blueprint.py`; `src/conference_pipeline/presentation/presentation_frame.py` | `docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint` |
| `presentation.generate_presentation_blueprint` | `legacy-debug` | `scripts/presentations/generate_presentation_blueprint.py` | `docs/reference/stage-owner-audit.md#presentation-generate-presentation-blueprint` |
| `presentation.review_presentation_blueprint` | `required` | `scripts/presentations/review_presentation_blueprint.py` | `docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint` |
| `presentation.repair_presentation_blueprint` | `repair` | `scripts/presentations/repair_presentation_blueprint.py` | `docs/reference/stage-owner-audit.md#presentation-repair-presentation-blueprint` |
| `presentation.generate_deck_contract` | `required` | `scripts/presentations/generate_deck_contract.py` | `docs/reference/stage-owner-audit.md#presentation-generate-deck-contract` |
| `presentation.generate_render_spec` | `required` | `scripts/presentations/generate_render_spec.py` | `docs/reference/stage-owner-audit.md#presentation-generate-render-spec` |
| `presentation.generate_programmatic_figures` | `required` | `scripts/presentations/generate_programmatic_figures.py` | `docs/reference/stage-owner-audit.md#presentation-generate-programmatic-figures` |
| `presentation.validate_svgs` | `required` | `scripts/presentations/validate_svgs.py` | `docs/reference/stage-owner-audit.md#presentation-validate-svgs` |
| `presentation.add_title_slide` | `legacy-debug` | `scripts/presentations/add_title_slide_to_slides_md.py` | `docs/reference/stage-owner-audit.md#presentation-add-title-slide` |
| `presentation.md_to_revealjs` | `required` | `scripts/presentations/md_to_revealjs.py` | `docs/reference/stage-owner-audit.md#presentation-md-to-revealjs` |
| `presentation.add_bookend_slides` | `required` | `scripts/presentations/add_bookend_slides.py` | `docs/reference/stage-owner-audit.md#presentation-add-bookend-slides` |
| `presentation.repair_loop` | `legacy-debug` | `scripts/presentations/run_presentation_repair_loop.py` | `docs/reference/stage-owner-audit.md#presentation-repair-loop` |
| `presentation.delivery_generation` | `required` | `scripts/presentations/generate_presenter_script.py` | `docs/reference/stage-owner-audit.md#presentation-presenter-script` |
| `presentation.presenter_narration_review` | `required` | `scripts/presentations/lint_presenter_narration.py`; `scripts/presentations/review_presenter_narration.py` | `docs/reference/stage-owner-audit.md#presentation-presenter-narration-review` |
| `presentation.presenter_narration_fix` | `required` | `scripts/presentations/fix_presenter_narration.py` | `docs/reference/stage-owner-audit.md#presentation-presenter-narration-fix` |
| `presentation.narration_script_review` | `required` | `scripts/presentations/lint_narration_script.py`; `scripts/presentations/review_narration_script.py` | `docs/reference/stage-owner-audit.md#presentation-narration-script-review` |
| `presentation.narration_series_review` | `required` | `scripts/presentations/review_narration_series.py` | `docs/reference/stage-owner-audit.md#presentation-narration-series-review` |
| `presentation.narration_series_fix` | `required` | `scripts/presentations/fix_narration_series.py` | `docs/reference/stage-owner-audit.md#presentation-narration-series-fix` |
| `presentation.narration_script_fix` | `required` | `scripts/presentations/fix_narration_script.py` | `docs/reference/stage-owner-audit.md#presentation-narration-script-fix` |
| `presentation.engaging_narration` | `required` | `scripts/presentations/generate_engaging_narration.py` | `docs/reference/stage-owner-audit.md#presentation-engaging-narration` |
| `presentation.rebuild_delivery_engaging` | `required` | `scripts/presentations/rebuild_delivery_engaging.py` | `docs/reference/stage-owner-audit.md#presentation-rebuild-delivery-engaging` |
| `presentation.reconcile_contract_hashes` | `required` | `scripts/presentations/reconcile_contract_hashes.py` | `docs/reference/stage-owner-audit.md#presentation-reconcile-contract-hashes` |
| `presentation.build_presenter_video` | `required` | `scripts/presentations/build_presenter_video.py` | `docs/reference/stage-owner-audit.md#presentation-build-presenter-video` |
| `presentation.build_notebooklm_video` | `required` | `scripts/presentations/build_notebooklm_style_presenter_video.py` | `docs/reference/stage-owner-audit.md#presentation-build-notebooklm-video` |
| `presentation.export_pdf` | `required` | `scripts/presentations/export_revealjs_pdf.py` | `docs/reference/stage-owner-audit.md#presentation-export-pdf` |
| `presentation.inspect_rendered_slides` | `required` | `scripts/presentations/inspect_rendered_slides.py` | `docs/reference/stage-owner-audit.md#presentation-inspect-rendered-slides` |
| `presentation.inspect_rendered_slides_llm` | `required` | `scripts/presentations/inspect_rendered_slides_llm.py` | `docs/reference/stage-owner-audit.md#presentation-inspect-rendered-slides-llm` |
| `planning.analysis.install` | `allowed` | `.claude/skills/plan-conference-analysis/scripts/install_skill.py` | `docs/reference/stage-owner-audit.md#planning-analysis-install` |

## Roadmap Backbone

Use `references/roadmap-backbone.md`. Generate phase-level roadmap entries with
Goal, Requirements, Success Criteria, and Canonical References. Do not over-plan
task details before downstream GSD discuss/plan phases.

Generated ROADMAP.md must include an Official Workflow Mapping table before phase blocks, with every official workflow step from docs/workflows/pipeline-workflow.md mapped, blocked, or out_of_scope.

## Editorial And Narration Review Phase

Generated conference plans that include deck/presentation production must add a
Presentation Blueprint Contract before `presentation.deck_contract_generation`.
`presentation_blueprint` is produced by
`presentation.generate_editorial_blueprint`, approved by
`presentation.review_presentation_blueprint`, and repaired with
`presentation.repair_presentation_blueprint` when review blocks.
Before blueprint generation, generated plans must require a
`decks/<id>/presentation_frame.json` audience-frame handoff. The frame must
separate the retrieval/topic-strategy title from the audience-facing
`audience_title`, include a field-level `field_thesis`, at least three broad
`general_questions`, `source_analysis_path`, and an `opening_anchor_policy`.
Plans must route it to `generate_editorial_blueprint.py` with
`--presentation-frame decks/<id>/presentation_frame.json`, fail closed when the
frame is missing, invalid, or deck/topic-mismatched, and require the first
slides to establish field-level questions before specialist paper mechanisms.
`scripts/presentations/generate_deck_contract.py` is the only production deck
contract path. The approval token must record
`approved_for_deck_contract_generation` (or the historical
`approved_for_draft_generation` alias), current `blueprint_sha256`, gate type:
deterministic, gate type: LLM-judge, gate type: human evidence, and passing
`presentation_frame_validation`. Generated plans must cite
`docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint`,
`docs/reference/stage-owner-audit.md#presentation-generate-presentation-blueprint`
and
`docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint`.
No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.

After blueprint approval, generated conference and production plans must use
structured presentation proof: `presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `render_spec.json`,
`delivery_single_speaker.json`, `delivery_notebooklm.json`,
`presentation_frame.json`,
`video_sync_validation_single_speaker.json`, and
`video_sync_validation_notebooklm.json`. Old numbered Markdown is legacy-only
debug export with `production_evidence: false`; it must not be cited as render
proof, Phase 13 canary proof, or delivery completion. Plans must name
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED` when the legacy path is detected.

Generated conference plans that include presenter videos, NotebookLM-style
videos, podcast-style narration, or TTS must include a dedicated editorial and
narration review/fix phase after `presentation.delivery_generation` and before any
TTS, PDF/video render, release, or proof. Before either TTS path, the phase
must run `presentation.editorial_gate` for every deck, require topic
`editorial_storyboard.json` artifacts with clusters, representative papers,
retained/merged/dropped slide rationale, figure role, and narration beat, target
34-38 minutes for each video style, fail any single-presenter or NotebookLM
script estimate above 40 minutes, merge obvious duplicate paper clusters without
dropping materially distinct papers, limit explicit zoom/detail beats to figures
marked complex or central, and enforce redundancy, visual variety, and
figure-narration alignment.

The phase must produce one representative topic canary end to end before bulk
regeneration. It must build `13-CANARY-OBJECTIVE-EVIDENCE.{json,md}` with
`presentation.phase13_canary_objective_evidence` to prove MP4/audio/slide
receipt duration alignment, non-black first/middle/tail frames, and contact
sheet evidence for both video styles. It must then build
`13-CANARY-REVIEW-PACKET.{json,md}` with
`presentation.phase13_canary_review_packet`, require a canary
`human_review_receipt.json` that records approval, regenerate the packet to
`approved_for_bulk_rollout`, and route bulk pre-TTS work through
`presentation.phase13_rollout --require-approved-canary --canary-review-packet <packet.json>`. The rollout
runner must include deck-level narration reviews plus cross-series
`review_narration_series.py` receipts and may generate reviewable markdown/JSON
artifacts, but it must stop before TTS/video until the canary approval gate is
closed and `presentation.phase13_bulk_readiness` reports
`ready_for_bulk_rollout`. Final phase closeout must run
`presentation.phase13_completion` and require `complete`.

The phase must first generate single-presenter
`video_build/script.json` with `build_presenter_video.py --stop-after-script-json`,
then run `presentation.presenter_narration_review` for every topic deck and
executive summary deck, writing `presenter_narration_lint_report.json`,
`presenter_narration_review.md`, and `presenter_narration_review.json`. It must
also run `presentation.narration_series_review` with
the single-presenter delivery contract to catch cross-deck single-presenter
figure repetition. Single-presenter TTS is blocked unless every presenter
review JSON has `approve_for_tts=true` and the presenter series review has no
unaccepted failure.

For NotebookLM-style videos, the phase must run
`presentation.narration_script_review` for every topic deck and executive
summary deck, write `narration_lint_report.json`, `narration_review.md`, and
`narration_review.json`, then run `presentation.narration_series_review` across
the full deck set to write `narration_series_review.md` and
`narration_series_review.json`. NotebookLM TTS is blocked unless every deck
review JSON has `approve_for_tts=true` and the series review has no unaccepted
failure.

If single-presenter deck-level review fails, the plan must route through
`presentation.presenter_narration_fix` or an explicit LLM/manual revision skill.
If NotebookLM deck-level review fails, the plan must route through
`presentation.narration_script_fix` or an explicit LLM/manual revision skill.
If series review fails, the plan must route through
`presentation.narration_series_fix` or an explicit governed cross-deck revision
skill. All fix paths must keep the relevant delivery contract, optional debug
markdown export, and `script.json` synchronized. The review criteria must
cover factual fidelity, truncation/hanging fragments, smooth transitions,
cohesion across slides, boring/fluffy/mechanical language, repeated question
hooks, repeated welcome/closing formulas, executive-summary bullet-stitch
templates such as `That connects directly to`, `The operational test is`, and
`That sets up`, and filler-word overuse such as repeated `critical`,
`crucial`, `key`, or `fundamental`, closing grammar joins such as
`The next step is We need...`, small fixed executive-summary transition
rotations, excessive conference-label repetition,
cross-deck duplicated figure
narration, and whether the final video will sound accurate, pleasant, and
human.

## Workshop Coverage Contract

Workshops are not a single event-family bullet. The brief must record
`workshop_coverage_posture` (one of `all_openreview_with_pdf`,
`all_workshops_best_effort`, `selective`, `out_of_scope`) and the analysis
scaffold must call
`scripts/discover_openreview_workshops.py --venue-prefix <openreview_workshop_parent_venue>`
to build `conferences/<slug>/storage/workshops/inventory.json`. Each row
carries `venueid`, `paper_count`, `tiers`, and a `pdf_availability_class`
(`openreview_with_pdf`, `openreview_metadata_only`, or `no_openreview_record`),
plus an explicit `decision` (`approved`, `metadata_only_acceptable`,
`external_site_required`, or `excluded`). The follow-on owner
`scripts/download_openreview_workshop_papers.py` downloads PDFs for
`openreview_with_pdf` venues at ≤1.4 rps with HTTP 429 backoff and resume; per-paper
receipts must record `content_type=application/pdf`, `sha256`, and
`byte_count > 0`. For workshops classified `openreview_metadata_only` or
`no_openreview_record` the analysis scaffold writes an external-site
downloader stub under
`scripts/conference/<slug>/workshop_site_downloaders/<workshop_slug>.py` and a
tracker entry at
`conferences/<slug>/storage/workshops/workshop_site_tracker.json`. Production
audit rejects stub-only scaffolds when those workshops are in scope.

### CVF Open Access Workshop Proceedings

For conferences published by the Computer Vision Foundation (CVPR, ICCV,
WACV), OpenReview is not the workshop paper source of record: most CVF
workshops post no papers there (CVPR 2026: 74 of 113 OpenReview workshop
venues had no papers, and ~40 calendar workshops had no venue at all), while
the archival proceedings of ~100 workshops are public at
`openaccess.thecvf.com/<Venue><Year>_workshops/menu`. The brief must record
`urls.openaccess_workshops` in the workspace `config.yaml` and the workshop
acquisition phase must run the owner
`scripts/download_cvf_openaccess_workshop_papers.py --workspace conferences/<slug>`
(`--dry-run` first) after OpenReview workshop download and before any
external-site stub is written or counted. It annotates each CVF workshop with
its OpenReview venue, skips papers already downloaded from OpenReview (exact
title, or a retitled camera-ready sharing most authors), and writes per-paper
receipts in the same `conference_pipeline.workshop_paper_receipt` schema.
Only workshops with no CVF listing, and non-archival tracks, remain
`external_site_required`. Plans must not treat `no_openreview_record` as a
coverage boundary for a CVF conference until this receipt exists.

For every conference (CVF or not), the workshop acquisition phase then runs
the `conference-workshop-paper-discovery` skill
(`.claude/skills/conference-workshop-paper-discovery/SKILL.md`): coverage
audit against the official schedule
(`scripts/audit_workshop_paper_coverage.py`), gap research, title resolution
(`scripts/resolve_paper_titles.py`) and verified workshop-site downloads
(`scripts/download_workshop_site_papers.py`). Coverage claims cite
`storage/workshops/workshop_paper_coverage.json`.

The narrowed OpenReview metadata clause is: OpenReview metadata-only records
cannot substitute for video/session/media downloads or for transcript-backed
analysis; OpenReview PDF downloads are acceptable paper-acquisition evidence
when accompanied by per-paper receipts carrying `content_type=application/pdf`,
`sha256`, and `byte_count > 0`.

### OpenReview Authentication And Browser Cookies

Since roughly 2026-07 the OpenReview API is challenge-gated: unauthenticated
non-browser clients receive HTTP 403 `ChallengeRequiredError` ("Verifying your
browser") on the notes API and on both PDF endpoints
(`openreview.net/pdf` and `api2.openreview.net/attachment`). A zero-notes or
403 result from an unauthenticated fetch is a gated result, not evidence that a
venue has no papers or no PDFs; it must never be classified
`no_openreview_record`.

Generated plans that acquire OpenReview papers (main-conference or workshop)
must record the authenticated posture. Both registry owners
`scripts/discover_openreview_workshops.py` and
`scripts/download_openreview_workshop_papers.py` accept `--openreview-auth`
(`auto` default, `required`, or `off`) and mint a Bearer token from the
environment variables `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` (sourced
from the repo-root `.env`; values never recorded — receipts carry only
`openreview_auth_mode` and `openreview_auth_env_names`). Authenticated download
runs fetch PDFs from `api2.openreview.net/attachment`, which the token unlocks.
Plans that expect a complete OpenReview corpus must require
`--openreview-auth required` and treat `openreview_auth_mode: unauthenticated`
plus 403/zero-note venues as a fail-stop blocker, not a coverage boundary.

Conference virtual sites (icml.cc, iclr.cc, neurips.cc) gate some item pages
and embedded media behind an attendee login. Reuse an existing browser session
with `scripts/export_browser_cookies.py --browser <browser> --domain <site>
--output <path-outside-repo>`, which reads the logged-in profile's cookies via
yt-dlp, keeps only the requested domain, and writes a Netscape cookie file
(owner-only permissions; cookie names only in stdout). Point the conference's
`*_COOKIES_FILE` env var (e.g. `ICML_COOKIES_FILE`) at that path. Cookie files
and OpenReview credentials live only in `.env`/local files; generated artifacts
record env-var names and cookie names, never values.

## Benchmark-Aware Provider Contracts

Generated ROADMAP.md must include `## Benchmark-Aware Provider Contracts`.
Generated ROADMAP.md must also include `## Task Model Defaults`, an
operator-facing table with every governed registry task id, stage, default
provider/model, fallback provider/model, selection status, and benchmark/receipt
requirements. This table is required even when the same information is present
in machine-readable handoff JSON; operators must not have to infer the default
ASR/OCR/LLM choices from `approval-summary.json` alone.
Generated `approval-summary.json` must carry the same rows under
`"benchmark-aware provider contracts"` and `"task model defaults"`. The
structured `production handoff.provider_contracts` and
`production handoff.task_model_defaults` values must carry the same rows so
production planning preserves analysis-approved provider contracts and default
model choices.

Planning prose alone cannot supersede a provider/model contract; generated
artifacts must name registry snapshots, benchmark receipt requirements,
provider/model selection status, local proof posture when relevant, and
operator/provider approval posture. No provider family, including OpenAI,
Anthropic, Gemini, OpenRouter, or local models, may be described as
categorically best without benchmark receipt evidence.

Exhaustive extraction defaults to the configured local OpenAI-compatible LLM
endpoint and model for topic classification and evidence mapping over completed
artifacts. Deterministic heuristics serve as prefilter and confidence support,
and as fallback when the local endpoint is unavailable. Cloud LLMs are reserved
for low-confidence or ambiguous records that neither heuristics nor the local
LLM can resolve with sufficient confidence, and must not be the default
extraction path. This policy applies only to exhaustive extraction.
GPT-Researcher has its own local-default evidence contract: use local Qwen3.6 35B-A3B
(`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`) with local BGE-M3 embeddings
(`mlx-community/bge-m3-mlx-8bit`, 1024 dim), BM25+vector hybrid retrieval, and
web search disabled unless a plan explicitly opts into Tavily. The default
GPT-Researcher parameters are `MAX_ITERATIONS=5`,
`DEEP_RESEARCH_BREADTH=4`, `DEEP_RESEARCH_DEPTH=3`, `MAX_SUBTOPICS=8`,
`RETRIEVAL_K=18`, `PAPER_MATCHING_RETRIEVAL_K=18`,
`CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, `TOTAL_WORDS=3000`,
`SMART_TOKEN_LIMIT=8192`, `STRATEGIC_TOKEN_LIMIT=8192`,
`FAST_TOKEN_LIMIT=4096`, and `TEMPERATURE=0.0`. The final public narrative
remains governed by `analysis.deep_narrative_synthesis`
(`openai:gpt-5.6-luna`; no separate live fallback — the retired `gpt-5.4`
fallback pairing was collapsed since it resolved to the same model) unless
local deep-validation receipts pass.

## Title-Spine Topic Strategy

Generated analysis plans must make `analysis.topic_strategy` the owner of topic
selection before query planning. The owner command is
`.venv/bin/python scripts/build_topic_strategy.py --workspace conferences/<slug>
--run-id <run-id> --mode live --provider-cost-approved --model openai:gpt-5.6-sol
--model anthropic:claude-opus-4-8 --model
gemini:gemini-3.1-pro-preview --merge-model openai:gpt-5.6-sol --summary`.

The stage must collect invited/keynote, main oral, session, workshop, and
segmented workshop talk titles into
`storage/analysis/topic-strategy/<run-id>/title_spine.json`, record the
title-spine sha256, obtain three independent model recommendations, and write
exactly 12 merged `final_topics` to `analysis/topic_strategy.json`. Mock mode is
planning/test evidence only. Stale candidate interest groups, a single-model
topic choice, or prose-only topic strategy without the title-spine receipt are
rejected.

Enhanced-indexing and GPT-Researcher embedding contracts must require local
embedding readiness receipt fields. A future conference plan must not accept a
local embedding blocker unless the receipt proves the configured embedding
client was constructed and first-used (`build_embedding_client` plus
`embed_texts`) and records `embedding_readiness` plus
`client_start_or_load_attempted`.

Generated analysis specs must also hand off the token-bounded Stage 6 gate:
`PG-CORPUS-TOKEN-INDEXING`. Production must record
`token_budget_policy_digest`, `embedding_policy_digest`,
`EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, and
`vector_embeddings_written: true`, with embedded chunk counts matching sidecar
counts. The reusable owner must write durable embedding sidecar manifests before
bulk Chroma load and receipt `embedding_sidecar_manifest_path`,
`embedding_sidecar_manifest_sha256`, `embedding_sidecar_format`,
`embedding_sidecar_shard_count`, and `chroma_insert_batch_size`.
`CORPUS_VECTOR_EMBED_BATCH_SIZE` and Chroma insert batch size are operational
tuning evidence, not embedding identity; generated plans must not force
reembedding only because a batch size changes. Word-count-only sidecars,
readiness-only probes, partial Chroma rows, conference-local embedding
reimplementations, and historical exhaustive extraction output without registry
receipt are rejected evidence. Cite
`docs/reference/stage-owner-audit.md#indexing-token-bounded-chunking` and
`docs/reference/stage-owner-audit.md#indexing-local-embedding-materialization-policy`
when handing off the gate.

Generated analysis specs must also hand off
`PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION` whenever exhaustive extraction is in
scope. Production must run
`scripts/run_llm_full_read_extraction.py --scope all --resume --model
mlx-community/Qwen3.5-35B-A3B-4bit` after the 12-topic strategy exists and
before query planning. The receipt must prove `scope: all`, `limit: null`,
`checks.completion_state: complete`, `completed_unit_count ==
expected_unit_count`, `failed_count: 0`, transcript windows first, paper
abstracts last, and by-unit/by-topic outputs. Heuristic 14A receipts alone,
25-unit pilots, `--limit` runs, transcript-only runs, metadata-only receipts,
and transcript artifacts backed by empty sidecars are rejected evidence.

Workshop speech transcript plans must preserve talk metadata. When workshops
are in scope, first require workshop talk-media acquisition proof at
`storage/workshops/talk_media_acquisition_receipt.json` or a formal
`storage/workshops/workshop_talk_media_scope_waiver.json`; a zero-count
workshop speech audit is not evidence that workshop talks were out of scope.
Plans must name `acquisition.workshop_talk_media` and the reusable
`acquisition.embedded_media_recovery` owner
(`src/conference_pipeline/acquisition/embedded_media.py`) so agenda crawling is
public-first/authenticated retry, extracts embedded-player state with
`embedded_media_links`, sanitizes media URLs with `sanitize_media_url`, and uses
yt-dlp recovery before declaring operator-visible media inaccessible. Generated
plans must also name the conference-local embedded host allowlist; for ICLR
production this is `--embedded-media-host slideslive.com`, and receipts must
record `allowed_hosts`. Do not plan YouTube, Vimeo, Zoom, social/profile, or
unrelated embeds as ICLR workshop presentation media unless a future
conference-specific phase explicitly widens that allowlist. Cite
`docs/reference/stage-owner-audit.md#acquisition-workshop-talk-media` and
`docs/reference/stage-owner-audit.md#acquisition-embedded-media-recovery`.
Name
`PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY` and
`transcription_ocr.workshop_talk_boundary_discovery` when no vetted
`talks.json` exists: it uses Gemini 2.5 Pro (`gemini-2.5-pro`), writes `*_talks.json`, emits
5-minute talk-context batches, records `provider_receipt_paths`, and uses
`cost_task_id=workshop_talk_boundary_discovery`. Then name
`transcription_ocr.workshop_talk_segmentation`, preserve `workshop_id`,
`workshop_title`, `talk_id`, `talk_title`, and `talk_speaker`, and block with
`talk_segmentation_provider_not_configured` when no `talks.json` or governed
discovery path exists. audio-only recovery is rejected evidence for talk-aware
transcripts. Cite
`docs/reference/stage-owner-audit.md#transcription-ocr-workshop-talk-boundary-discovery`
and
`docs/reference/stage-owner-audit.md#transcription-ocr-workshop-talk-segmentation`
with `conference_pipeline.workshop_talk_segmentation`,
`src/conference_pipeline/workshop_talk_segmentation.py`,
`scripts/discover_workshop_talks.py`, and
`scripts/segment_workshop_talks.py`.

Generated conference context packets must include a Backup Posture section:
`My Drive/<Conference> <Year>-raw`, an explicit `--remote-path` to the mounted
conference backup leaf, manifest source, backup profile, Drive setup with
`scripts/sync_to_gdrive.py --profile backup --remote-path <mounted-leaf>
--init-only`, continuous per-stage artifact offload with manifest-bounded
rsync-incremental backup sync, restore-verification posture, and the no-secret
rule. The point is to keep generated payloads out of Git and move only
manifest-selected durable families to the mounted Drive target as they are
produced; cache/unselected families must not bloat Drive.

## Content Fidelity Gates

Conference plans that request analysis, presentations, slides, figures, or
exhaustive reading must define explicit content-fidelity gates. These gates are
part of the specification, not optional production polish:

- **Extracted Content Provenance:** `extracted_content.json` must be backed by
  the exhaustive source-reading workflow in
  `docs/extraction/exhaustive-extraction-guide.md`. Selected-corpus metadata,
  paper titles, abstracts, or PDF-only summaries cannot satisfy exhaustive
  analysis or presentation completion.
- **Slide/Visual OCR:** slide, poster, image, or video-derived evidence is
  classified as one of three states based on a source-coverage assessment:
  `required` (corpus lacks sufficient text-bearing artifacts, or the brief
  requires slide/visual evidence), `skipped_with_rationale` (available text
  sources provide sufficient coverage and a receipt-backed rationale is
  recorded), or `deferred_blocked` (OCR decision requires upstream information
  not yet available). Unconditional OCR requirements without a source-coverage
  assessment are not acceptable.
- **Figure/SVG Quality:** slides marked `FIGURE_NEED` require actual embedded
  programmatic SVG figures generated through matplotlib, Plotly, or svgwrite,
  or explicit per-slide waivers. Programmatic output is necessary but not
  sufficient: generated plans must require source-grounded `FigureSpec`
  records, Anthropic `claude-sonnet-5` as the governed
  `presentation.svg_figure_generation` code/spec generation model,
  `--live-sonnet-specs` for live production unless cost is explicitly waived,
  content-aware, slide-specific editorial SVGs that bind to slide evidence,
  select an appropriate visual template (matrix, flow, benchmark curve, radar,
  memory breakdown, attention map, etc.), and beat generic placeholder/scaffold diagrams.
  Renderer-only repairs after an existing live FigureSpec receipt may reuse
  `figure_specs.json` with `--figure-specs-input`, but plans must require that
  the Sonnet receipt is preserved, stale SVGs are cleaned, the asset inventory
  records `live_sonnet_specs_reused_local_render`, and both structural plus
  semantic SVG validation are rerun.
  Paper-extracted/cropped figures may be supplemental evidence only; they must
  not satisfy `presentation.svg_figure_generation`. SVG quality checks must
  render, inspect, repair, and re-render; zero SVG output, repeated boilerplate
  visuals, or generic scaffold figures are blockers.
- **Figure Validation:** generated SVGs must run
  `.venv/bin/python scripts/presentations/validate_svgs.py --workspace <workspace> --presentation-id <id> --dir decks/<id>/assets/generated --report-output decks/<id>/assets/generated/validation_report.md`
  and
  `.venv/bin/python scripts/presentations/review_programmatic_figures.py --workspace <workspace> --presentation-id <id> --presentation-type <type> --topic <topic> --dir decks/<id>/assets/generated --asset-inventory decks/<id>/asset_inventory.json --report-output decks/<id>/assets/generated/semantic_review.json --markdown-output decks/<id>/assets/generated/semantic_review.md`
  before render/QC can be accepted. Semantic review blocks copied bullets,
  generic template phrases, density fallback, placeholder chart ticks, and
  missing required FigureSpec entities.
- **Vision-LLM Visual QC:** presentation visual QC must include both rendered
  screenshot heuristics and governed vision-LLM review. Plans must run
  `scripts/presentations/inspect_rendered_slides.py` to create screenshots and
  `scripts/presentations/inspect_rendered_slides_llm.py` to create
  `assets/generated/visual_llm_qc_report.json` with model-governance task
  `presentation.visual_qc` (default `openai:gpt-5.6-luna`, fallback
  `openai:gpt-5.6-luna`). Missing provider/model, missing report, parse failure, or
  blocker findings fail closed; operator-only or heuristic-only visual QC cannot
  satisfy production.
- **LLM-Feedback Visual Repair:** plans must require
  `scripts/presentations/run_presentation_repair_loop.py` for
  `presentation.repair` whenever `visual_llm_qc_report.json` contains blocker
  findings. The repair loop must read the LLM findings, map screenshot slide
  numbers back through `presentation_slide_mapping.json`, write
  `assets/generated/visual_repair_feedback.json`, rerun
  `scripts/presentations/generate_programmatic_figures.py --repair-feedback`,
  rebuild `presentation.html`, rerun rendered screenshot QC, and then require a
  fresh governed vision-LLM QC pass. Mechanical SVG fixes alone cannot satisfy
  this repair contract.
- **Presentation Density Targets:** deck plans must carry NeurIPS-style source
  density targets into `workflow.deck` and `presentation_manifest.json`:
  `target_hyperlinks_per_slide` defaults to `2.0`, `target_svgs_per_slide`
  defaults to `0.70`, and conference-specific overrides must be explicit.
  Validation must include `source_binding_density_meets_target`,
  `visible_reference_coverage_meets_target`, and `svg_density_meets_target`.
  Source density is measured from `source_binding_report.json`; old numbered
  Markdown files are legacy/debug exports and cannot satisfy production proof.
- **Executive Overview Decks:** when target outputs include executive summary
  presentations, generated plans must create both manifest-backed aggregate
  decks: `decks/executive_summary_short` and
  `decks/executive_summary_long`. These are distinct from
  `analysis/executive_summary.md` and `analysis/trend_report.md`, which are
  source synthesis reports only. Plans must include
  `scripts/presentations/generate_executive_overview_decks.py`, register both
  decks in `decks/presentation_manifest.json` with
  `presentation_type: executive-summary`, render/export them through the same
  branded Reveal.js path, and run rendered-slide QC before videos, release, or
  proof. A topic-only `topic-deep-dive` manifest is incomplete when short/long
  executive overview decks are requested. Executive overview decks must follow
  the topic bookend format: slide 2 is a `slide_kind: events` source-surface
  slide and the last slide is `slide_kind: resources`. Let `x` be the number of
  executive synthesis points in the deck spine; slide 2 may show at most `x`
  important events on the left and must show at least `x` key resources on the
  right. Important events include keynotes, invited talks, workshops, oral
  sessions/talks, panels, competitions, awards, and best-paper contexts when
  available. The final resources slide must link all 12 topic presentations and
  all 12 topic analysis reports. Production plans should require a public
  presentation-link map (`decks/executive_presentation_link_map.json`) for those
  topic-presentation links; workspace-local presentation links are review-only
  and cannot close the final link audit. The executive overview story itself
  must be a quote-backed synthesis, not a repeated resource index: generated
  plans may extract up to 24 candidate conference lessons, roughly two per
  topic when all 12 topics are in scope, but the produced structure is
  operator-locked: a short deck with around 8 executive points and about 15
  minutes of narration, and a long deck with the same 8-point spine plus 3
  finer-grained expansions, around 11 points total, and about 30 minutes of
  narration. Bind each selected point to transcript quotes or tightly grounded
  transcript paraphrases from keynotes, workshops, oral sessions, panels, or
  talks.
- **Structured Source Binding:** topic-deep-dive decks must run
  `presentation.source_binding` after `presentation_deck.json` exists.
  Required owners are `scripts/presentations/build_deck_source_pack.py` and
  `scripts/presentations/bind_presentation_sources.py`; completion evidence is
  `source_pack_receipt.json`, `ranked_papers.json`, and
  `source_binding_report.json`. Deep-research matched paper titles,
  OpenReview/arXiv/PDF URLs, speakers, talks, system names, and resource links
  must be bound to slide IDs and visible reference slots. Empty URL evidence,
  unresolved placeholders, or missing author/speaker/system references are
  blockers.
- **Conference-Adaptive Source-Link Adapter:** when deck/presentation output is
  in scope, generated plans must add an implementation checkpoint after source
  binding and before bookend insertion, HTML render, PDF export, video release,
  or proof. The checkpoint must inspect the actual conference corpus and source
  metadata (OpenReview/forum ids, conference program pages, workshop/media
  inventories, transcript extraction metadata, poster/media paths, Drive source
  maps, and executive source-link maps), then update or extend resolver/render
  and audit code as needed. Expected edit surfaces include
  `src/conference_pipeline/presentation/source_links.py`,
  `scripts/presentations/add_bookend_slides.py`,
  `src/conference_pipeline/presentation/structured_video.py`,
  `scripts/presentations/audit_presentation_links.py`, or a clearly named
  conference adapter/config plus regression tests. Plans must not assume a
  universal registry lookup will resolve every public source link; unresolved
  visible references are blockers unless a receipt records that no public URL is
  available in the corpus.
- **RelationalAI Brand Reference And Title Metadata:** presentation plans must
  require title, author, and report-link metadata inside `render_spec.json`
  before `presentation.render_export`. Missing title/author/report-link
  metadata is a blocker because video sync, attribution, and brand rhythm
  depend on it. Reveal.js render must use the exact RelationalAI NeurIPS 2025
  brand reference from
  `assets/brand/relationalai/`; an optional PPTX template may be supplied only
  if it exposes the same logo, bug, and title artwork. Approximate palettes,
  generic themes, or unbranded renders cannot satisfy production.
- **Presenter And Video Slide-Sync:** presenter-script generation must run
  `scripts/presentations/generate_presenter_script.py` after the structured
  render path exists, with `--render-spec decks/<id>/render_spec.json`,
  `--slide-mapping decks/<id>/presentation_slide_mapping.json`,
  `--conference-label <conference>`,
  `--delivery-single-speaker decks/<id>/delivery_single_speaker.json`, and
  `--delivery-notebooklm decks/<id>/delivery_notebooklm.json`.
  Markdown narration exports are debug-only and cannot satisfy delivery proof.
  Video generation must run both `scripts/presentations/build_presenter_video.py`
  and `scripts/presentations/build_notebooklm_style_presenter_video.py` with
  `--slide-mapping decks/<id>/presentation_slide_mapping.json`; the NotebookLM
  builder must also pass `--conference-label <conference>` so generated dialogue
  cannot leak a prior conference name. These builders are required evidence
  owners because they fail closed when mapped Slide `0` or any later mapped slide
  lacks narration/timing coverage.
- **PDF Export:** presentation plans must include PDF export under
  `presentation.render_export` whenever deck deliverables are in scope. The
  required owner command is
  `scripts/presentations/export_revealjs_pdf.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --input-html decks/<id>/presentation.html --output-pdf decks/<id>/presentation.pdf`.
  Manual browser print-to-PDF, ad hoc Chrome commands, or screenshot-based
  exports cannot satisfy completion without an explicit registry-approved
  waiver.
- **Presentation Link Audit And Drive Source Links:** after HTML/PDF export,
  plans must run `scripts/presentations/audit_presentation_links.py --workspace
  <workspace> --decks-dir decks --conference-slug <slug> --receipt-output
  decks/presentation_link_audit_receipt.json`. A passed Stage 10 must reject
  `file:///`, `/Users/...`, repo-local, and `../analysis/...` deck links. The
  receipt must enumerate expected public links from slide-2 events/papers,
  final-slide resources, content-slide source chips, and executive summary
  decks; verify matching HTML hrefs and PDF `/URI` annotations; and report
  `unresolved_visible_refs`, `bad_local_hrefs`, `missing_html`, and
  `missing_pdf` counts. If the audit script cannot prove those dimensions for
  the current deck format, the plan must include resolver/audit code changes
  before closeout rather than waiving the link check.
  Executive overview decks must run
  `scripts/presentations/generate_executive_overview_decks.py` with
  `--source-link-map decks/executive_source_link_map.json
  --require-source-link-map`; if Google Drive has not synced or uploaded the
  analysis markdown source, the plan must record an explicit blocked source-link
  contract instead of allowing local source-link fallbacks.
- **Implementation Inventory:** before naming stage owners, generated plans
  must inventory existing scripts/modules and bind each governed task id to a
  concrete owner command or an explicit blocker. Deprecated presentation
  paper-figure scripts (`scripts/presentations/extract_topic_figures.py` and
  `scripts/presentations/add_images_to_md.py`) are rejected as owner commands
  for `presentation.svg_figure_generation`. Generated plans must cite the
  unified stage script registry SHA-256, registry-owned `usage_doc_refs`,
  agent-facing `diagnostic_profile` guidance, and final verifier
  `scripts/verify_stage_script_usage.py`. For manifest-backed
  `podcast.script_generation`, generated plans must use
  `.venv/bin/python scripts/generate_focused_podcast.py analysis/<id>.md --fact-ranks audio/<id>/ranks.json --output-dir audio/<id>/focused_podcast`
  or declare an explicit blocker if the analysis manifest lacks a topic
  `fact_ranks_path`.
  For `podcast.tts`, generated plans must then use
  `.venv/bin/python scripts/render_podcast_audio.py --script-json audio/<id>/focused_podcast/script.json --output-dir audio/<id>/focused_podcast --model gpt-4o-mini-tts`
  and must require segment-level provider receipts plus a rendered
  `audio/<id>/focused_podcast/podcast.mp3`
  (`docs/reference/stage-owner-audit.md#podcast-render-audio`).
  When the brief's target outputs include per-deck NotebookLM podcasts
  (`podcast_notebooklm: true`), the production handoff must name
  `podcast.notebooklm_audio_overview` and
  `.venv/bin/python scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> --live --ack-live-notebooklm --format deep-dive --length long`
  as the owner of one Google NotebookLM Audio Overview per topic and executive
  deck, sourced from the deck's analysis spine plus blueprint-cited sources and
  bound by `audio/<id>/notebooklm_podcast/notebook_manifest.json`
  (`docs/reference/stage-owner-audit.md#podcast-notebooklm-audio-overview`);
  it requires a Google account with NotebookLM Pro and the `notebooklm` extra.
  Conference-specific podcast scaffold writers and script-only
  `audio_ready_package.json` emitters are rejected as completion owners. For
  manifest-backed `presentation.deck_contract_generation`, generated plans must first
  run `presentation_blueprint` with
  `scripts/presentations/generate_editorial_blueprint.py --presentation-frame decks/<id>/presentation_frame.json --final-report-url <url> --provider-cost-approved`
  and `scripts/presentations/review_presentation_blueprint.py`, repair blocked
  reviews with `scripts/presentations/repair_presentation_blueprint.py`, require
  passing `presentation_frame_validation`, and
  then use
  `.venv/bin/python scripts/presentations/generate_deck_contract.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_deck.json --receipt-output decks/<id>/presentation_deck_receipt.json`.
  Reject `render_slides(...)`, conference-specific scaffold writers,
  `scripts/presentations/generate_slides_simple.py`, and
  `scripts/generate_slides_agentic.py` as completion owners for
  manifest-backed deck contract generation.
  For manifest-backed `presentation.render_export`, generated plans must use
  `.venv/bin/python scripts/presentations/md_to_revealjs.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --output-html decks/<id>/presentation.html --slide-mapping decks/<id>/presentation_slide_mapping.json --require-brand-assets`.
  For manifest-backed `presentation.delivery_generation`, generated plans must use
  `.venv/bin/python scripts/presentations/generate_presenter_script.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --slide-mapping decks/<id>/presentation_slide_mapping.json --conference-label <conference> --delivery-single-speaker decks/<id>/delivery_single_speaker.json --delivery-notebooklm decks/<id>/delivery_notebooklm.json`.
  For manifest-backed `presentation.video_generation`, generated plans must run
  both `.venv/bin/python scripts/presentations/build_presenter_video.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --delivery-contract decks/<id>/delivery_single_speaker.json --render-spec decks/<id>/render_spec.json --build-dir decks/<id>/video_build --output decks/<id>/presentation_video.mp4` and
  `.venv/bin/python scripts/presentations/build_notebooklm_style_presenter_video.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --source-delivery-contract decks/<id>/delivery_notebooklm.json --render-spec decks/<id>/render_spec.json --build-dir decks/<id>/video_build_notebooklm --output decks/<id>/presentation_video_notebooklm.mp4 --duration-reference-script-json decks/<id>/video_build/script.json --conference-label <conference>`.
  For manifest-backed `presentation.render_export`, generated plans must use
  `.venv/bin/python scripts/presentations/export_revealjs_pdf.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --input-html decks/<id>/presentation.html --output-pdf decks/<id>/presentation.pdf`.
  For manifest-backed `presentation.link_audit`, generated plans must use
  `.venv/bin/python scripts/presentations/audit_presentation_links.py --workspace <workspace> --decks-dir decks --conference-slug <slug> --receipt-output decks/presentation_link_audit_receipt.json`.
  For manifest-backed `presentation.repair`, generated plans must repair
  structured evidence: update `visual_plan.json`, `asset_inventory.json`,
  `figure_specs.json`, or `render_spec.json` as appropriate, rerun
  `presentation.render_export`, and then run `.venv/bin/python
  scripts/presentations/inspect_rendered_slides_llm.py --workspace <workspace>
  --presentation-id <id> --presentation-type topic-deep-dive --topic <topic>
  --screenshots-dir decks/<id>/assets/generated/qc/screenshots
  --report-output decks/<id>/assets/generated/visual_llm_qc_report.after_repair.json`.
  Markdown repair-loop commands are legacy/debug-only and cannot close
  production blockers by themselves.
  Plans must reject `validate_svgs.py --fix` as sufficient completion evidence
  when LLM visual QC reports blocker findings.
- **Source Traceability IDs:** durable artifact IDs and local paths remain
  usable in receipts and extraction records. Sanitizers redact secrets, cookies,
  authorization headers, and base64 payloads, but must not redact ordinary media
  artifact IDs into unusable `REDACTED` traceability gaps.
- **Narrative Analysis Contract:** public `analysis/<topic>.md` files are deep
  analysis documents, not slide briefs. They must include `Executive Summary`,
  `Key Findings`, `Detailed Analysis`, `References And Sources`,
  `Cross-References`, `Recommendations`, `Questions For Further Investigation`,
  and a compact `Slide Brief` section. Deck-facing summaries live in
  `analysis/slide_briefs/<topic_id>.md`. Raw extraction, vector chunk,
  research, matched-paper, and quote ledgers belong in
  `analysis/provenance/<topic_id>.md`,
  `analysis/provenance/<topic_id>.json`, and
  `analysis/analysis_citation_manifest.json`. Generated plans must require
  `analysis.deep_narrative_synthesis` with default `openai:gpt-5.6-luna` (no
  separate live fallback), iterative GPT-Researcher/deep-query plans, deep
  validation gates, and fail-closed production behavior when the LLM/model or
  evidence is missing. Generated plans must reject `Source Basis`, `Verbatim
  transcript/OCR evidence`, `Matched Paper Evidence`, and `Source Gaps` as
  top-level public-report sections, and must reject plans that describe public
  `analysis/<topic>.md` as only slide-ready `Major Theme` documents. Local
  GPT-Researcher findings can feed this source pack, but cannot close the
  public narrative step unless they pass the same schema, provenance,
  source-diversity, keypoint, and editorial-quality gates.
- **Presentation Narrative Gate:** deck/presentation phases must include a
  reusable presentation narrative-content gate that verifies draft decks consume
  `analysis/slide_briefs/<topic_id>.md`, bind to the current slide-brief/deep
  analysis digest, reject stale scaffold/public-ledger language, and fail
  identical deck checkpoint files unless an explicit valid stage waiver is
  present.

## Extraction Provider Policy

Generated exhaustive extraction plans default to the configured local
OpenAI-compatible LLM endpoint and model for topic classification and evidence
mapping over completed artifacts. Deterministic heuristics serve as prefilter
and fallback when the local endpoint is unavailable or when heuristic confidence
meets the classification threshold. Cloud LLMs are reserved for low-confidence
or ambiguous records that neither heuristics nor the local LLM can resolve with
sufficient confidence.

The LLM full-read lane is separate from that heuristic/local classifier lane:
it uses Qwen 35B-A3B to read every eligible transcript window and paper abstract
against the finalized 12-topic strategy. It is not complete unless it runs with
`--scope all` and no `--limit`; partial/pilot receipts must remain blockers
even if the heuristic 14A receipt succeeded.

For conferences where the full-read by-topic evidence is compact enough to
inspect in long-context model windows, plans may add a provisional
`analysis.exhaustive_topic_summarization` phase before formal deep analysis:
`.venv/bin/python scripts/summarize_llm_full_read_topics.py --workspace
conferences/<slug> --run-id <run-id> --mode dry-run --provider gemini --model
gemini-3.1-pro-preview --max-input-tokens 400000 --summary` to size the work,
then live mode with `--provider-cost-approved` to write
`analysis/exhaustive_summaries/<topic_id>__transcripts.*` and
`analysis/exhaustive_summaries/<topic_id>__papers.*`. Treat these summaries as
inspection/source-pack inputs until a later milestone promotes them into a
formal required stage.

When exhaustive extraction is in scope, the following fields are truth-critical
in the brief:

- Local LLM endpoint label (env var `EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL`)
- Local LLM base URL (env var `EXTRACTION_LOCAL_LLM_BASE_URL` — must be a
  loopback or local address; non-local endpoints are rejected at runtime)
- Local LLM model name (env var `EXTRACTION_LOCAL_LLM_MODEL`)
- Whether the local endpoint is reachable and the model is available

Plans must declare local LLM readiness as a prerequisite: endpoint reachable,
model configured, and sample inference succeeds before production extraction
begins. If the local endpoint is not configured or not reachable, the plan must
record this as a blocker rather than silently routing to cloud LLM.

Non-extraction generation providers are not covered by the
extraction-local-default policy. The approved GPT-Researcher evidence lane is
local Qwen3.6 35B-A3B plus local BGE-M3 embeddings with corpus-only hybrid retrieval;
ASR, TTS, ranking, podcast, presentation, OCR, and final report synthesis
continue to use their own benchmark-governed contracts.

## GPT-Researcher Retrieval And Parameter Policy

Generated analysis plans must treat
`conference_pipeline.iclr2025_live.run_query_planning_gpt_researcher`,
`conference_pipeline.research.researcher`,
`conference_pipeline.research.query_planner`,
`conference_pipeline.research.config`, and
`conference_pipeline.research.walker` as the registry-owned GPT-Researcher
stage (`docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage`).
The default GPT-Researcher evidence lane is local/corpus-only:
`GPTR_LOCAL_LLM_RUNTIME_KIND=mlx`,
`GPTR_LOCAL_LLM_SMART_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,
`GPTR_LOCAL_LLM_FAST_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,
`EMBEDDING_PROVIDER=mlx`, and
`EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit`. The default retrieval mode is
BM25+vector hybrid (`workflow.research.gpt_researcher.retrieval_mode: hybrid`);
vector-only is an explicit override, not the default.

When generated analysis includes deep-dive topics, plans must first run
`analysis.deep_dive_question_selection` with GPT-5.6 Luna to compress exhaustive
topic summary bullets into a bounded set of 4-6 selected questions per topic:
`.venv/bin/python scripts/select_deep_dive_questions.py --workspace conferences/<slug>
--run-id <run-id> --mode live --model <registry-default> --provider-cost-approved
--min-questions-per-topic 4 --max-questions-per-topic 6 --summary`, where
`<registry-default>` is `default_model("analysis.deep_dive_question_selection")`
from the model registry (currently `gpt-5.6-luna`; read it from the registry
rather than hardcoding it). Plans must reject one-GPT-Researcher-run-per-key-point
expansion.

For each selected question, plans must require two
GPT-Researcher lanes. The first lane is the
global, no-topic-id GPT-Researcher pass over the whole corpus or stable source
scopes. The second lane is the heuristic keyword-prefiltered pass owned by
`analysis.gpt_researcher_heuristic_prefilter`,
`conference_pipeline.research.heuristic_prefilter`,
`src/conference_pipeline/research/heuristic_prefilter.py`, and
`scripts/run_heuristic_prefiltered_research.py`
(`docs/reference/stage-owner-audit.md#analysis-gpt-researcher-heuristic-prefilter`).
The heuristic lane must record
`workflow.research.heuristic_prefiltered_gpt_researcher`,
`heuristic_keyword_prefilter`, `keyword_seed_source`,
`prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`,
`stores_searched`, `source_family_scope`, `retrieval_scope`, and
`invalid_citation_ids`. Topic-id filtered deep dives, unrecorded keyword lists,
source-family leakage, and citation-invalid heuristic reports are rejected
live evidence. The combined runner is `analysis.dual_deep_dive_research_lanes`
via `scripts/run_dual_deep_dive_research.py --resume` with
`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`.

Before final public narrative synthesis or slide synthesis, plans must run the
packaged reconciliation and reference-governance owner
`analysis.topic_reconciliation_reference_workflow`:
`.venv/bin/python scripts/run_topic_reconciliation_reference_workflow.py
--workspace conferences/<slug> --run-id <run-id> --mode live --model
mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary`. This workflow is
the normal production entrypoint because it packages the order future
conferences must not accidentally skip: 07D topic reconciliation, immediate 07E
deterministic reference validation, deterministic mechanical safety fixes,
bounded 07G candidate resolution when needed, broader 07F repair only after
candidate blockers remain, and final 07E validation. It writes
`07H-topic-reconciliation-reference-workflow.json`, and final narrative or slide
synthesis requires `final_validation_status: succeeded`, positive ledger and
citation counts, and zero `final_invalid_reference_count`,
`final_unknown_reference_count`, `final_title_mismatch_count`, and
`final_bare_reference_warning_count`.

The individual owners remain registered evidence and fallback/debug entrypoints.
`analysis.topic_reconciliation` merges exhaustive paper/transcript summaries,
the current narrative source pack, and both deep-dive lanes into
`analysis/topic_reconciliations/<topic_id>.md` plus source-pack
`topic_reconciliation` fields. `analysis.topic_reference_validation` builds
`analysis/topic_reference_ledgers/<topic_id>.json`, writes
`analysis/topic_reference_validation/<topic_id>.json`, records
`07E-topic-reference-validation.json`, and blocks unknown ids, title/id
mismatches, empty required ledgers, and uncanonicalized bare ids. If 07E blocks,
`analysis.topic_reference_candidate_resolution` is the first LLM-assisted repair
lane and may apply only validator-clean bounded packet decisions. Only when 07G
records unresolved blockers should plans use the broader
`analysis.topic_reference_repair` fallback. Generated plans must not defer the
first reconciliation of parallel analysis products to slide synthesis or rely on
manual memory of the 07D/07E/07G/07F ordering.

Plans must expose and record the GPT-Researcher research-budget knobs under
`workflow.research.gpt_researcher` and compatible env aliases:
`MAX_ITERATIONS`, `DEEP_RESEARCH_BREADTH`, `DEEP_RESEARCH_DEPTH`,
`DEEP_RESEARCH_CONCURRENCY`, `MAX_SUBTOPICS`, `TOTAL_WORDS`, token limits,
`RETRIEVAL_K`, `PAPER_MATCHING_RETRIEVAL_K`, `CANDIDATE_MULTIPLIER`, `RRF_K`,
and `TEMPERATURE`. Defaults are `5/4/3/2/8`, `TOTAL_WORDS=3000`,
`SMART_TOKEN_LIMIT=8192`, `STRATEGIC_TOKEN_LIMIT=8192`,
`FAST_TOKEN_LIMIT=4096`, `RETRIEVAL_K=18`,
`PAPER_MATCHING_RETRIEVAL_K=18`, `CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, and
`TEMPERATURE=0.0`. The behavioral resume digest is driven by knobs that
currently change execution or the synthesis prompt; concurrency and
fast/strategic token limits remain receipt-only until separate parallel or
fast-model calls exist.

The local GPT-Researcher lane must use loopback-only `LocalLLMResearcher`,
record actual provider/model provenance, and disable Tavily for strict local
runs unless the plan explicitly allows web search. Plans must state that this
local lane is eligible for store-grounded retrieval, evidence expansion,
parameter sweeps, and draft GPT-Researcher findings, while production
`analysis/<topic>.md` still uses governed GPT-5.6 Luna/GPT-5.4 deep narrative
synthesis unless local validation passes.

## Dual-Store Paper Matching

When a conference plan includes presentations, decks, or presentation paper
matching, generated planning must require the v1.91 dual-store architecture:

- Build two vector stores: `papers` and `artifacts`.
- Build corpus stores through the reusable
  `conference_pipeline.indexing.corpus_vector_store` asset; conference-local
  production runners may orchestrate it but must not reimplement chunking or
  embedding materialization inline. The reusable asset writes embedding
  sidecar manifests first, then bulk-loads Chroma from those sidecars.
- Put only `paper_pdf`, `paper_text`, and `paper_metadata_fallback` chunks in
  the paper store.
- Put transcripts, slide OCR, visual descriptions, session/talk evidence, and
  other non-paper artifacts in the artifact store.
- Run GPT-Researcher on the artifact store before paper matching.
- Extract frontier-model keypoints from artifact GPT-Researcher findings.
  Generated config must include `workflow.paper_matching.keypoint_provider`
  and `workflow.paper_matching.keypoint_model` (default frontier model:
  `gpt-5.6-sol` unless model governance requires a replacement).
- Run GPT-Researcher on the paper store using those artifact-derived
  keypoints.
- Keep paper titles returned by the paper-store GPT-Researcher pass as the
  presentation paper matches in `matched_papers.json`.
- Full paper PDF/text chunks must prepend paper title, section title, and
  subsection title when known. Title-plus-abstract fallback must be recorded in
  metadata.
- `scripts/rank_papers.py` is legacy/deprecated for NeurIPS 2025 compatibility
  only; generated plans must point to
  `scripts/match_papers_from_artifact_keypoints.py` as the reusable command.

## Conditional Slide OCR

Before committing to slide OCR, plans must run or consume a source-coverage
assessment comparing these four sources against the evidence needs of downstream
deliverables: transcripts, paper text, source files (existing PDF text layer),
and available media metadata.

Slide/Visual OCR is classified as one of three states:

- `required`: the corpus lacks sufficient text-bearing artifacts, the brief
  explicitly requires slide/visual evidence, or downstream deliverables need
  slide-level citations.
- `skipped_with_rationale`: available text sources provide sufficient coverage.
  The plan must record a receipt-backed rationale with these fields:
  `available_text_sources` (list), `expected_coverage_impact` (str),
  `reopen_conditions` (list).
- `deferred_blocked`: the OCR decision requires upstream information not yet
  available.

When OCR is `required`, the plan must record: `why_required` (one of the three
triggers above), `expected_receipt_paths` (list), `provider_runtime_contract`
(str), `cost_rate_limits` (str), `downstream_blockers` (list).

Unconditional OCR requirements without a source-coverage assessment are not
acceptable.

## NotebookLM Source Export

When a plan promises a NotebookLM source bundle for human deep-dive, it must use
the filesystem-only packager
`scripts/package_conference_for_notebook_lm_export.py --workspace
conferences/<slug>`. The packager reads already-produced artifacts (no provider
calls) and writes `exports/notebooklm_export/` with chunked `notebooklm/*.md`
parts, a machine-readable `manifest.json`, and `index.csv`. Parts are packed to
stay under one NotebookLM source (default `--max-words 450000` /
`--max-bytes 4000000`, a safety margin below NotebookLM's ~500k-word per-source
limit); paper chunks honor `--papers-max-words` / `--papers-max-bytes`.

`--paper-mode` selects paper content and is the contract a plan must state:

- `abstract` (default): metadata + abstract chunks, prefix `paper_abstracts`.
- `full`: cleaned full paper text grouped by primary topic, one prefix per topic
  area (`papers_full_<NN_topic>`); abstract chunks are not emitted.
- `both`: emits both.

Full-text mode is filesystem-only and content-addressed. It indexes
`paper_text` / `workshop_paper_text` extraction records by `text_sha256`
(the on-disk extracted-text path is retention-redacted, so content hashing is
the join key) and joins to local OpenReview metadata through the PDF hash for
clean titles (records that do not join fall back to head-extracted titles, with
full text preserved). Cleaning strips the recurring conference header and
references/bibliography list while keeping appendices, de-hyphenates line breaks,
and collapses whitespace.

Primary-topic grouping is governed by `--paper-topic-source` (default `llm`).
The default prefers the exhaustive `llm_full_read` semantic assignment
(`llm-full-read/by-unit/paper_abstract_*.json`, keyed by forum id) and falls back
to the confidence-ranked keyword heuristic (`topic_ids[0]`, which the upstream
classifier already orders by `score / keyword_count`) for papers the LLM lane did
not cover — typically workshop and calendar-sourced papers. Do not group by the
raw `topic_scores` argmax: it over-counts generic vocabulary and collapses most
papers into a couple of topics. `--paper-topic-source heuristic` forces the
keyword path. The manifest records provenance via
`counts.full_text_topic_from_llm_full_read` and
`counts.full_text_topic_from_heuristic`, and each paper block carries a
`Topic source` line.

Plans that promise full-text export must state `--paper-mode full`
(and `--paper-topic-source` when not the default), cite the per-topic part
counts and total NotebookLM source count from `manifest.json`
(`counts.papers_full_source_files` plus transcript and analysis parts) against
the target NotebookLM tier's sources-per-notebook limit, and rely on
`tests/test_notebooklm_conference_export.py` as the behavioral contract.
Conference-specific source counts and tier assumptions belong in the milestone
context packet, not this skill.

Unless `--skip-slides` is set, the packager also includes each topic's
structured presentation contracts (`presentation_deck.json`,
`render_spec.json`, and delivery contracts when present) as per-topic sources
(prefix `presentation_<NN_topic>`), counted in
`counts.topic_presentation_contracts` / `counts.presentation_source_files` and
`parts.presentations`.
`--decks-dir` overrides the deck root.

Unless `--skip-conference-overview` is set, the packager writes a
`notebooklm/00_conference_overview.md` orientation source (corpus scale, the
executive-summary throughline and ranked cross-topic conclusions, the 12 topic
areas with full-text paper counts and one-line characterizations, and a
how-to-use guide). It is the first NotebookLM source and is recorded at
`manifest.json#conference_overview`.

Unless `--skip-presentation-prompts` is set, the packager writes
`exports/notebooklm_export/presentation_prompts/` — for each topic a long
(`<NN_topic>_long.txt`, ~60 min) and a short (`<NN_topic>_short.txt`, ~20 min)
NotebookLM presentation prompt, plus conference-wide `00_global_conference_long`
/`_short` variants (26 files total). These are paste-in prompts (not upload
sources): the long variant requests a full sectioned talk over all grounding
points, the short variant a focused headline talk; both reference the topic's
slide deck and full-text sources. They are grounded in the exhaustive LLM
summaries (per-topic Key Points / Tensions / Open Questions) and the executive
summary's cross-topic conclusions, with topic titles from
`analysis/topic_strategy.json`, and are listed in
`manifest.json#presentation_prompts`.

## Media-Family Scope Decision

Media acquisition scope is opt-out, never opt-in. The default scope is ALL
media-bearing event families the conference program declares or the
event-family inventory discovers — tutorials, keynotes/invited talks, orals,
workshops and segmented workshop talks, panels, spotlights, expo talks, expo
demonstrations, expo workshops, affinity events, town halls, award talks, and
socials with recorded media. Planning MUST ask the user one explicit question:

> "Media scope defaults to ALL media-bearing event families:
> <enumerated list from the inventory>. Should any family be EXCLUDED?"

The answer is recorded in the generated plan as a `media_family_scope`
handoff field with `default: all_media_bearing_families`, the user's answer
verbatim as `decision_source`, and `excluded_families` (empty unless the user
explicitly excluded families). Every exclusion also gets an Exclusion Records
entry with scope, reason, timestamp, and downstream effect.

Fail-closed consequences the generated plan must state:

- Acquisition code must never allowlist event families by URL shape; every
  virtual-site item page is an acquisition target by default
  (`conference_pipeline.program_coverage` is the platform owner of this rule).
- Production must re-confirm the same question before the acquisition stage
  runs (see `plan-conference-production`), and the download/corpus stage must
  fail closed when a declared media-bearing family yields zero discovered
  item pages without a receipt-recorded exclusion
  (`media_family_coverage` payload, `program_family_media_items_missing`
  blockers).
- A reachable family index page with zero extracted item links is a discovery
  gap (JS-rendered index, bare-ID links, new URL shape) — never evidence the
  family has no media. ICML 2026 lost all 10 tutorials and every invited talk
  to exactly this: the tutorials index linked items as bare-ID
  `/virtual/2026/<id>` URLs and the classifier only recognized
  poster/paper/oral paths.

## Production Execution Handoff

This skill creates the conference analysis specification. It should not pretend
that specification execution has happened. When the user wants the next step
that creates live-only runnable work to actually produce the conference
analysis, use the companion `plan-conference-production` skill.

`plan-conference-production` is live-only production planning. It must reject
dry-run, mock, scaffold, readiness-only, metadata-only, and `needs live proof`
artifacts as production completion evidence, even when this analysis skill used
those artifacts for planning or UAT.

Generated context packets and `approval-summary.json` must include an explicit
next-step handoff naming `plan-conference-production`. The handoff must say that
`plan-conference-production` creates the live execution phase or milestone; it
must not imply that production has already run or that production can run
without approval, credentials, cost gates, and receipts.

The handoff should identify:

- `plan-conference-production` as the next skill to call after this analysis
  specification is finalized
- the finalized milestone context, requirements, roadmap, and state files
- target workspace and conference slug
- a full event-family inventory covering papers, posters, workshops, keynotes,
  tutorials, competitions, panels, sessions, and conference-specific event
  types surfaced by source pages
- the media-family scope decision (see `## Media-Family Scope Decision`):
  `media_family_scope` with `default: all_media_bearing_families`, the
  explicit user answer to the exclusion question, and an
  `excluded_families` list that is empty unless the user explicitly excluded
  a family
- proposed interest groups with evidence notes plus explicit user decisions:
  approved, edited, rejected, or complete
- exclusion records with scope, reason, timestamp or decision context,
  source/evidence note, and downstream effect before expensive work starts
- stage support labels and known unsupported stages
- required outputs and proof boundaries
- provider-cost posture and whether expensive provider work requires an
  estimate before production approval
- registry task ids, approved provider/model choices, pending research,
  benchmark requirements, `not approved yet` statuses, and local-provider
  deferrals for ASR, OCR/image parsing, extraction, research, ranking,
  podcast, presentation, TTS, and failover providers
- whether OCR/image parsing should include a model benchmark before bulk live
  provider calls, using:
  `.venv/bin/python -m conference_pipeline.ocr_benchmark run --workspace <workspace> --sample-size 12 --budget-cap-usd 30 --per-call-timeout-seconds 60 --json`
- extracted-content provenance expectations, including whether
  `extracted_content.json` must represent exhaustive source reading or a smaller
  explicitly waived scope
- narrative analysis expectations, including `analysis/narrative_source_packs`,
  public narrative shape, provenance sidecar paths, and manifest-backed citation
  validation
- slide/visual OCR expectations and any allowed per-asset waiver policy
- figure/SVG generation, render-inspection, repair-loop, and zero-output blocker
  expectations for presentation outputs
- vision-LLM visual QC expectations, including screenshot source, governed
  provider/model, blocker criteria, and `visual_llm_qc_report.json` receipt path
- the programmatic figure owner-command expectation: matplotlib/Plotly/svgwrite
  generation is required for `presentation.svg_figure_generation`, FigureSpec
  records plus `review_programmatic_figures.py` semantic review are required,
  renderer-only reuse repairs must keep the live Sonnet receipt and rerun both
  SVG gates, generic placeholder SVGs are rejected, and the deprecated
  paper-figure extraction/insertion scripts are forbidden as completion owners
- implementation inventory expectations tying every provider-backed task id to
  an existing owner, required implementation task, or explicit blocker
- source traceability ID preservation expectations for media artifacts and local
  paths
- embedded media recovery posture for ICLR-style authenticated item pages:
  cached `item_page_auth_probe` HTML may be acquisition evidence, but any
  SlidesLive audio recovery belongs before speech transcription, must use the
  registry-owned `acquisition.slideslive_media_recovery` owner or a
  conference-local equivalent, and must not redownload source pages, paper PDFs,
  workshop PDFs, poster images, or unrelated media families
- SlidesLive slide acquisition posture: visual/OCR-scoped plans must treat
  SlidesLive slide images as a separate lane from audio, use
  `acquisition.slideslive_slide_download`, then run
  `acquisition.slide_reference_detection` and `acquisition.slide_cleanup`
  before visual OCR/indexing; audio-only recovery cannot satisfy slide evidence
- token-bounded indexing posture: production handoff must require
  `PG-CORPUS-TOKEN-INDEXING`, `token_budget_policy_digest`,
  `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`,
  `CORPUS_VECTOR_EMBED_BATCH_SIZE`, `embedding_sidecar_manifest_path`,
  `embedding_sidecar_manifest_sha256`, `chroma_insert_batch_size`, and
  `vector_embeddings_written: true` from
  `conference_pipeline.indexing.corpus_vector_store`; base vector stores are
  topic-agnostic and must carry stable source metadata such as
  `content_kind`, `conference_scope`, `source_url`, `paper_id`,
  `session_title`, `workshop_id`, `talk_id`, timestamps, and media provenance
- workshop talk media posture: production handoff must require
  `acquisition.workshop_talk_media` and
  `acquisition.embedded_media_recovery`
  (`src/conference_pipeline/acquisition/embedded_media.py`), with
  public-first/authenticated retry, embedded-player extraction, yt-dlp recovery,
  a conference-local embedded host allowlist, and authenticated
  agenda/talk-media acquisition where conference sites gate videos/slides behind
  login. For ICLR production the allowlist is `--embedded-media-host
  slideslive.com`; generated plans must reject YouTube, Vimeo, Zoom,
  social/profile, and unrelated embeds as ICLR workshop presentation media
  unless an explicit future phase widens the allowlist. Require
  `storage/workshops/talk_media_acquisition_receipt.json` or
  `storage/workshops/workshop_talk_media_scope_waiver.json` before Stage 5/6
  when workshops are in scope; unauthenticated zero-media blockers are not
  final evidence when an authenticated operator can see media
- workshop talk segmentation posture: production handoff must require
  `transcription_ocr.workshop_talk_segmentation`,
  `talk_segmentation_provider_not_configured`, and the statement that
  audio-only recovery is rejected evidence for talk-aware transcripts
- whether the user wants one execution phase or a multi-phase production
  milestone
- the default production corpus scope: all discovered conference items unless
  explicitly excluded before expensive provider work
- the production cost-gate command:
  `.venv/bin/python -m conference_pipeline.costs estimate --workspace <workspace> --scope full --scenario base --json`

## Conference-Portability Requirement

Generated plans must carry a portability requirement: reusable platform code
(`src/conference_pipeline/**` and shared `scripts/`) must read all
conference-specific values — conference name, year, program URLs/domains, topic
taxonomy, audience framing — from the workspace's generated files
(`config.yaml` via `conference_pipeline.conference_identity`,
`analysis/analysis_manifest.json`, `decks/<id>/presentation_frame.json`), never
from hardcoded literals. Roadmaps must include, as a milestone-completion gate,
running the `conference-hardcoding-audit` skill
(`python scripts/audit_conference_hardcoding.py` exiting 0 plus the manual
`scripts/` sweep). A new conference must be servable by the same code with only
its workspace files changing.

Generated plans must also carry a separate, independent model-selection gate:
any milestone that touches model selection must additionally run the
`model-registry-audit` skill (`python scripts/audit_model_registry_hardcoding.py`
exiting 0) before close. This gate is in addition to, not a replacement for,
`conference-hardcoding-audit` — the two cover different literal families
(model names vs. conference identity).

## Conference Quirk Inventory For Thin Orchestrators

Each conference keeps its own production orchestrator. The operator made this
decision on 2026-09-30, and the analysis spec shapes the orchestrator. It must
be thin: it imports shared stage owners from `src/conference_pipeline/` and
implements only the conference's quirks, behind declared hooks. See
`.planning/proposals/thin-conference-orchestrators-20260930.md` and
`## Thin Conference Orchestrator Composition` in `plan-conference-production`.

The analysis spec must therefore record a **conference quirk inventory**, so
production planning can map every stage to a shared component or a hook.

- For each stage (runner-preflight through proof), state one of two things:
  - "shared, no quirk", or
  - the quirk, the evidence that makes it conference-specific, and the hook
    family it needs. Hook families: provider probes, source-family adapter,
    corpus exclusions, auth posture, acquisition lane, paper priority, request
    pacer, transcription lane, retry authorization, record builder,
    source-pack policy, extra CLI command.
- Typical quirks to probe for during brief intake:
  - the auth model: OpenReview Bearer login vs. cookies vs. registration-gated
    media
  - where talks live: SlidesLive, organizer YouTube channels, a virtual site
  - multi-talk recordings that need event-talk splitting
  - proceedings hosts outside OpenReview (CVF Open Access, workshop sites)
  - program-spine sources
- Do not label a behavior a quirk if a shared module already covers it with
  configuration, for example a priority-terms file or workspace `config.yaml`.
  A quirk is behavior, not a value. Values belong in workspace files per the
  conference-portability requirement.
- The production handoff must carry the inventory as `orchestrator_quirks`,
  a list of `{stage, hook_family, description, evidence, paid}`.
- The roadmap must include an orchestrator-composition task that creates
  `scripts/run_<slug>_production.py` as a thin orchestrator. The task starts
  from the shared stage table and orchestrator scaffolding and adds only the
  inventoried hooks. It must never start by copying another conference's
  runner.
- A stage with no shared owner yet becomes a lift task in the roadmap. The
  task lifts the existing implementation into `src/conference_pipeline/`
  with golden-receipt tests. It must not produce a new local copy.

### Acquisition Lanes Recommended By The Lane Catalog

Stage 4 quirks map to catalogued acquisition lanes
(`src/conference_pipeline/acquisition/lane_catalog.py`). During brief intake,
list the lanes recommended for the conference family and probe each one's
quirk, instead of re-deriving the acquisition design:

```bash
PYTHONPATH=src python -m conference_pipeline.acquisition.lane_catalog --conference "<Conference Name YYYY>"
```

- For each recommended lane, the spec records whether the quirk applies to
  this edition (with evidence: an auth wall observed, talks found on organizer
  channels, workshop papers outside OpenReview, ...). A confirmed lane enters
  `orchestrator_quirks` with `hook_family: acquisition lane` (or `auth
  posture`, `paper priority`, `request pacer`) and its catalog `lane_id`.
- A new CVPR starts from the CVPR-recommended lanes; a new ICML from the
  ICML-recommended lanes. A lane listed only under "may fit" (for example a
  CVF lane for ICCV or WACV) is proposed as a question for intake, not
  assumed.
- A conference family with no recommended lanes records "shared, no quirk"
  for download-corpus unless intake finds evidence for a catalogued or new
  lane. A new lane is described by capability (what it acquires and why the
  shared frame alone cannot), never by conference year.
- The production handoff carries the catalog lane ids in
  `orchestrator_quirks`, so `plan-conference-production` binds the existing
  lanes rather than writing new acquisition code.

## Corpus Source-Metadata Coverage Requirement

Every generated analysis plan and its downstream production handoff must carry
the `corpus source-metadata enrichment` stage and its coverage-audit
completion gate, `PG-CORPUS-SOURCE-METADATA`. The gate's live command is
`python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug>
--json --write-receipt storage/workflow-runs/<run-id>`, and it writes the
coverage receipt `corpus_source_metadata_coverage.json`. See the
`corpus-metadata-audit` skill
(`.claude/skills/corpus-metadata-audit/SKILL.md`) for how the runtime gate is
run and interpreted (per-family floors, the `artifacts` family's
`structurally_authorless` posture, and the no-silent-lowering
`floor_override_unjustified` refusal path).

This planning-skill requirement and its own `audit_generated_plan.py` check
are weaker than that runtime gate, and the difference must never be
papered over. `audit_generated_plan.py` (and `plan-conference-production`'s
`audit_generated_production_plan.py`) check that a GENERATED PLAN'S TEXT
declares the stage and the gate. They do not and cannot verify that a future
conference's live run actually produced real coverage — that stronger,
runtime-scanning guarantee comes only from running
`corpus-metadata-audit` against real on-disk artifacts. See
`.planning/workstreams/icml-2026/audits/skill-enforcement-review-20260713.md`
for the empirical record of a plan-text audit reporting a full pass
(`status "passed", 81/81 assertions, 0 failures`) against a production plan
whose live execution still failed for reasons the plan's own text correctly
forbade. Treat a plan-text pass here as "the plan says the right words," not
as proof a future conference's corpus will meet the coverage floor.

Roadmaps must add `python scripts/audit_corpus_source_metadata.py --workspace
conferences/<slug>` to the milestone-completion gate list, alongside
(not instead of) `conference-hardcoding-audit` and `model-registry-audit`.

## Fail-Stop Pipeline Policy

Generated conference milestones must declare that blocked or unavailable
required work halts the whole pipeline before downstream stages run. A required
stage result with `status: blocked`, `status: deferred_blocked`,
`status: failed`, `status: unavailable`, or `status: unsupported` is not a
successful completion boundary.

Plans must require runtime `BlockerReceipt` evidence for fail-stop states,
including `ocr_deferred_blocked`, `stage_blocked`, `stage_unavailable`,
`local_llm_not_ready`, `missing_receipt`, `threshold_violation`, and
`stale_receipt`. Resume is blocked while active blockers exist and may continue
only after an `UnblockReceipt` written through `workflow unblock` or
`conference_pipeline.workflow_unblock`. Summaries must expose `halt_status`,
`halt_stage_id`, `blocker_receipt_path`, and `unblock_acknowledgements`.

Generated roadmaps and production handoffs must explicitly say that provider
availability failures, missing live owner commands, unsupported live stages,
missing required receipts, stale receipts, and failed required validation stop
downstream podcast, deck, release, proof, and publication work.

## Support Labels

Use exactly these support labels:

- `live`
- `dry-run`
- `mock`
- `unsupported`
- `planned`
- `dry-run/mock only`
- `needs live proof`

New conferences default to dry-run/mock proof unless live proof receipts exist.
Do not imply live crawl, transcription, analysis, podcast, or presentation
support without proof receipts.

## Install Targets

Canonical repo source:

- `.claude/skills/plan-conference-analysis`

User sync targets:

- `~/.codex/skills/plan-conference-analysis`
- `~/.claude/skills/plan-conference-analysis`

Use `scripts/install_skill.py` to sync. Do not copy secrets, generated UAT
drafts, caches, or proof artifacts into user skill directories.

## local_proof_posture (required, Phase 13+)

The analysis scaffold includes `local_proof_posture` in every structured
production handoff. Each task_id maps to either `"deferred"` when local proof is
empty or incomplete, or to a proof reference dict with `research_ref`,
`hardware_fit_ref`, `benchmark_receipt_ref`, `operator_approval_ref`,
`approved_date`, and `scope`.

Production planning uses this field to distinguish approved-local tasks from
local deferrals. The generated audit must find exactly one posture for each
local-provider task: approved proof or deferral, never both.

## UAT Proof

Use `scripts/scaffold_plan.py` to draft planning artifacts from a brief and
`scripts/audit_generated_plan.py` to verify roadmap coverage, source-of-truth
placement, exact support labels, content-fidelity gates, and secret-free
generated outputs. Then run
`PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit --plan-root <draft-root>`
as the mandatory owner-registry check before approval/finalization.

Write durable UAT proof as JSON plus Markdown under the active planning phase
directory, such as `.planning/workstreams/<workstream-id>/phases/<NN>-<phase-name>/`
for the phase under review.
UAT must be dry-run/mock-first and must not claim unsupported live support.
