---
name: plan-conference-production
description: Generate an executable live-only GSD phase or milestone from an existing plan-conference-analysis specification. Use after plan-conference-analysis when asked to turn a conference analysis spec into runnable production work that executes all pipeline stages end-to-end, including runner commands, receipts, validation, completion gates, recovery behavior, and rejected non-live evidence boundaries.
---

# Plan Conference Production

Create execution-ready live production GSD work from an existing conference
analysis specification. This skill is the companion to
`plan-conference-analysis`: that skill defines what the conference milestone
should produce; this skill defines the phase or milestone that actually runs the
pipeline stages to produce it.

Production output is live-only. Dry-run, mock, scaffold, readiness-only,
metadata-only, and `needs live proof` artifacts may be mentioned as historical
or rejected evidence, but they must not satisfy production completion.

## Required Reads

Read these before writing or updating production execution plans:

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
- `src/conference_pipeline/costs/pricing_catalog.json`
- `src/conference_pipeline/costs/task_specs.yaml`
- active `.planning/milestones/*-CONTEXT.md`
- active `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, and `.planning/STATE.md`
- target workspace `config.yaml`, `storage/storage_manifest.json`,
  `analysis/analysis_manifest.json`, and `decks/presentation_manifest.json`

## Generator And Audit Workflow

Use the deterministic scripts for production phase generation:

1. Draft production artifacts with
   `scripts/scaffold_production_plan.py --spec-root <analysis-spec-root> --draft-root <draft-root>`.
2. Audit the draft with
   `scripts/audit_generated_production_plan.py <generated-production-dir> --json-output <workstream>/audits/generated-production-audit.json`.
3. Run the mandatory stage owner registry audit:
   `PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit --plan-root <generated-production-dir>`.
4. Run the unified stage script registry verifier:
   `PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py --plan-root <generated-production-dir> --skills-root .claude/skills --docs-root docs --strict`.
5. Verify workspace gitignore policy from the storage manifest:
   `PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py --workspace <workspace> --check`.
6. Do not finalize or execute generated work until the production audit, the
   gitignore policy check, the stage owner registry audit, and the stage script
   registry verifier all pass.

Generated production work must preserve the v1.63 Runtime Stage Contract Gate
Pattern: every live runner stage that can write `status: "succeeded"` must be
registry-bound through `stage_bindings.<stage>.runtime_contract`, call
`conference_pipeline.stage_contracts.verify_stage_contract` immediately before
writing the receipt, and embed `stage_contract.accepted_evidence` plus
`stage_contract.blockers` in that receipt. If the runner cannot produce live
completion evidence for a stage, the registry must mark that stage explicitly
non-production or unavailable instead of allowing a silent success.
Every such receipt must also carry v5.1 receipt lineage: bind it on the
provisional receipt with `conference_pipeline.receipt_lineage.finalize_receipt`
(`bind=True` on a fresh run, the recorded `lineage=` on any resume path that
rebuilds a receipt), declare the stage's `receipt_filename` and
`output_artifact_checks` / `input_artifact_checks` in its registry binding,
and verify persisted receipts as-is on resume (`require_lineage=True`), so a
re-run of any upstream stage invalidates the stages below it.

The generator consumes the `plan-conference-analysis` spec and emits a
multi-phase live production milestone by default. It may emit one execution
phase only when every in-scope stage has a real existing owner command and the
caller explicitly asserts that with `--all-owner-commands-exist`.

Read `references/production-workflow-gates.md` when changing generated
workflow gates, audit assertions, or production closeout rules.

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

### Count-Free Contracts And Appendable Transcript Evidence Lanes

Generated production plans must honor two platform rules that exist because
CVPR 2026 lost three plenary keynotes and ~460 paid ASR transcripts to a
stage contract that pinned one run's counts:

- **Contracts verify structure and lineage, never run counts.** A
  `runtime_contract` or `stage_contracts.py` check may pin an upstream
  receipt's sha256 and require internal consistency (inventory counts equal
  list lengths equal receipt checks), but it must not assert a literal
  number of routes, captions, or gaps. Derive expected counts from the
  digest-bound upstream at verification time. `scripts/audit_conference_hardcoding.py`
  fails on any integer literal `>= 100` in a non-allowlisted `*contract*`
  module under `src/conference_pipeline/` unless the line carries
  `# run-count-literal-ok: <reason>`.
- **Evidence that arrives after a harvest is appended, not parked.** Every
  transcript recovery that runs after the base caption inventory (caption
  rechecks, governed ASR fallback, hand-recovered keynotes) must end in a
  digest-bound lane receipt registered through
  `conference_pipeline.indexing.transcript_evidence_lanes`, and the
  conference's corpus-inventory-index stage must call
  `load_admitted_lane_records` and append the result. A recovery script
  with no downstream consumer is not done, whatever its own receipt says.
  The base inventory keeps precedence, lanes follow registry order, and a
  registered lane that fails digest validation blocks Stage 6 rather than
  being skipped. Reference implementation:
  `scripts/build_cvpr2026_transcript_evidence_lanes.py` feeding
  `scripts/run_cvpr2026_production.py --stage corpus-inventory-index`.

Generated plans must name the lane-registration step for any phase that
recovers transcripts after Stage 5 and must list the corpus-inventory
rebuild as that phase's completion gate (`PG-TRANSCRIPT-EVIDENCE-LANES`).

### Pre-Spend Contract Gate And Resume Economics

Generated production plans must require, and the generated-plan audit
enforces, that:

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

Use `references/provider-substitution-waiver.md`. Generated production plans
must require, and the generated-plan audit enforces, that:

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

The analysis handoff's `structured_gates` field (schema documented in the
analysis skill's `references/structured-gates.md`) is validated by
`validate_structured_gates()` (called from `validate_analysis_handoff`) and
propagated verbatim into this skill's own `approval-summary.json` under
`structured_gates` for consumption by the corresponding runtime_contract /
gate entries. The generated-plan audit fails when the propagated field is
missing or malformed.

## Registry Usage Documentation

Use registry `usage_doc_refs` for tool instructions and registry `diagnostic_profile`/`diagnostic.next_actions` for recovery guidance. Generated plans must include these anchors when they name registry tools; raw script paths alone are not sufficient planning guidance.

| Script ID | Status | Path | Usage docs |
|-----------|--------|------|------------|
| `planning.production.scaffold` | `required` | `.claude/skills/plan-conference-production/scripts/scaffold_production_plan.py` | `docs/reference/stage-owner-audit.md#planning-production-scaffold` |
| `planning.production.audit` | `required` | `.claude/skills/plan-conference-production/scripts/audit_generated_production_plan.py` | `docs/reference/stage-owner-audit.md#planning-production-audit` |
| `gitignore.workspace_policy` | `required` | `scripts/generate_workspace_gitignore.py` | `docs/reference/stage-owner-audit.md#gitignore-workspace-policy` |
| `verification.stage_owner_audit` | `required` | `src/conference_pipeline/stage_owner_audit.py` | `docs/reference/stage-owner-audit.md#verification-stage-owner-audit` |
| `verification.stage_script_usage` | `required` | `scripts/verify_stage_script_usage.py` | `docs/reference/stage-owner-audit.md#verification-stage-script-usage` |
| `indexing.transcript_evidence_lanes` | `required` | `src/conference_pipeline/indexing/transcript_evidence_lanes.py`; conference binder e.g. `scripts/build_cvpr2026_transcript_evidence_lanes.py` | `docs/reference/stage-owner-audit.md#indexing-transcript-evidence-lanes` |
| `backup.sync_to_gdrive` | `required` | `scripts/sync_to_gdrive.py` | `docs/reference/stage-owner-audit.md#backup-sync-to-gdrive` |
| `acquisition.workshop_acquisition_pipeline` | `required` | `src/conference_pipeline/acquisition/workshop_pipeline.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-acquisition-pipeline` |
| `acquisition.workshop_discovery` | `required` | `scripts/discover_openreview_workshops.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-discovery` |
| `acquisition.workshop_paper_download` | `required` | `scripts/download_openreview_workshop_papers.py` | `docs/reference/stage-owner-audit.md#acquisition-workshop-paper-download` |
| `acquisition.slideslive_media_recovery` | `required` | `src/conference_pipeline/slideslive_recovery.py`; `scripts/recover_iclr2026_slideslive_media.py` | `docs/reference/stage-owner-audit.md#acquisition-slideslive-media-recovery` |
| `acquisition.slideslive_slide_download` | `required` | `src/conference_pipeline/slideslive_slides.py`; `scripts/recover_slideslive_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slideslive-slide-download` |
| `acquisition.slide_reference_detection` | `required` | `src/conference_pipeline/slide_cleanup.py`; `scripts/detect_reference_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slide-reference-detection` |
| `acquisition.slide_cleanup` | `required` | `src/conference_pipeline/slide_cleanup.py`; `scripts/cleanup_slides.py` | `docs/reference/stage-owner-audit.md#acquisition-slide-cleanup` |
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
| `podcast.notebooklm_delivery` | `required` | `scripts/deliver_notebooklm_podcasts.py` | `docs/reference/stage-owner-audit.md#podcast-notebooklm-delivery` |
| `presentation.source_pack` | `required` | `scripts/presentations/build_deck_source_pack.py` | `docs/reference/stage-owner-audit.md#presentation-source-pack` |
| `presentation.bind_presentation_sources` | `required` | `scripts/presentations/bind_presentation_sources.py` | `docs/reference/stage-owner-audit.md#presentation-bind-presentation-sources` |
| `presentation.executive_overview_decks` | `required` | `scripts/presentations/generate_executive_overview_decks.py` | `docs/reference/stage-owner-audit.md#presentation-executive-overview-decks` |
| `presentation.generate_editorial_blueprint` | `required` | `scripts/presentations/generate_editorial_blueprint.py`; `src/conference_pipeline/presentation/presentation_frame.py` | `docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint` |
| `presentation.review_presentation_blueprint` | `required` | `scripts/presentations/review_presentation_blueprint.py` | `docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint` |
| `presentation.repair_presentation_blueprint` | `repair` | `scripts/presentations/repair_presentation_blueprint.py` | `docs/reference/stage-owner-audit.md#presentation-repair-presentation-blueprint` |
| `presentation.generate_deck_contract` | `required` | `scripts/presentations/generate_deck_contract.py` | `docs/reference/stage-owner-audit.md#presentation-generate-deck-contract` |
| `presentation.generate_render_spec` | `required` | `scripts/presentations/generate_render_spec.py` | `docs/reference/stage-owner-audit.md#presentation-generate-render-spec` |
| `presentation.generate_programmatic_figures` | `required` | `scripts/presentations/generate_programmatic_figures.py` | `docs/reference/stage-owner-audit.md#presentation-generate-programmatic-figures` |
| `presentation.validate_svgs` | `required` | `scripts/presentations/validate_svgs.py` | `docs/reference/stage-owner-audit.md#presentation-validate-svgs` |
| `presentation.add_title_slide` | `required` | `scripts/presentations/add_title_slide_to_slides_md.py` | `docs/reference/stage-owner-audit.md#presentation-add-title-slide` |
| `presentation.md_to_revealjs` | `required` | `scripts/presentations/md_to_revealjs.py` | `docs/reference/stage-owner-audit.md#presentation-md-to-revealjs` |
| `presentation.add_bookend_slides` | `required` | `scripts/presentations/add_bookend_slides.py` | `docs/reference/stage-owner-audit.md#presentation-add-bookend-slides` |
| `presentation.repair_loop` | `required` | `scripts/presentations/run_presentation_repair_loop.py` | `docs/reference/stage-owner-audit.md#presentation-repair-loop` |
| `presentation.editorial_gate` | `required` | `scripts/presentations/presentation_editorial_gate.py` | `docs/reference/stage-owner-audit.md#presentation-editorial-gate` |
| `presentation.phase13_canary_objective_evidence` | `required` | `scripts/presentations/build_phase13_canary_objective_evidence.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-canary-objective-evidence` |
| `presentation.phase13_canary_review_packet` | `required` | `scripts/presentations/build_phase13_canary_review_packet.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-canary-review-packet` |
| `presentation.phase13_rollout` | `required` | `scripts/presentations/run_phase13_rollout.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-rollout` |
| `presentation.phase13_bulk_readiness` | `required` | `scripts/presentations/verify_phase13_bulk_readiness.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-bulk-readiness` |
| `presentation.phase13_completion` | `required` | `scripts/presentations/verify_phase13_completion.py` | `docs/reference/stage-owner-audit.md#presentation-phase13-completion` |
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
| `planning.production.install` | `allowed` | `.claude/skills/plan-conference-production/scripts/install_skill.py` | `docs/reference/stage-owner-audit.md#planning-production-install` |

## Presentation Blueprint Gate

Generated production plans must run `presentation_blueprint` before
`presentation.deck_contract_generation`. The owner commands are
`presentation.generate_editorial_blueprint`,
`presentation.review_presentation_blueprint`, and, when review blocks,
`presentation.repair_presentation_blueprint`.
Before blueprint generation, production plans must create or consume
`decks/<id>/presentation_frame.json` as the audience-frame handoff. The frame
must distinguish the retrieval/topic-strategy title from the public
`audience_title`, include `field_thesis`, at least three broad
`general_questions`, `source_analysis_path`, and `opening_anchor_policy`, and be
passed with `--presentation-frame decks/<id>/presentation_frame.json`. Missing,
invalid, stale, or deck/topic-mismatched frames block production instead of
falling back to topic words. The opening slides must answer broad field-level
questions before introducing specialist paper mechanisms.
`scripts/presentations/generate_deck_contract.py` is the only production deck
contract path. The approved review must record
`approved_for_deck_contract_generation` (or the historical
`approved_for_draft_generation` alias), the current `blueprint_sha256`, gate type:
deterministic, gate type: LLM-judge, gate type: human evidence, and passing
`presentation_frame_validation`. Generated production plans must cite
`docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint`
and `docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint`.
They must not cite `presentation.generate_presentation_blueprint` or
`docs/reference/stage-owner-audit.md#presentation-generate-presentation-blueprint`
as a production completion owner; that path is historical/debug evidence only.
No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.

**Rebuilding after a deck change.** When a deck changes after its videos were
built, decide re-encode versus re-record from the delivery contract, not from
the screen. If the contract's `presentation_deck_sha256` still matches
`presentation_deck.json`, the segment count still matches the slide count, and
the figures' `<text>` nodes are unchanged (compile the same specs with the
previous revision and diff -- do not assume), then the recorded audio is still
correct: re-render frames and re-encode. TTS segments are cached by a
fingerprint of (text, model, voice), so the rebuild reuses every segment at no
provider cost. If any narration text changed, redo narration review, TTS and
video in that order; never splice new audio into an old timeline.

Figures, narration, TTS, PDF export, video rendering, release, and proof are
blocked until the approved blueprint review is current.

Production presentation proof after blueprint approval is structured:
`presentation_frame.json`, `presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `render_spec.json`,
`delivery_single_speaker.json`,
`delivery_notebooklm.json`, `video_sync_validation_single_speaker.json`, and
`video_sync_validation_notebooklm.json` are required evidence. Old numbered
Markdown is legacy-only/debug-export evidence with `production_evidence: false`;
generated production plans must reject it as production proof and name
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED` when the legacy path appears.

## Generated Production Workflow Gates

Every generated production roadmap must include `## Production Workflow Gates`
and every generated approval summary must carry the same rows as structured
`production workflow gates` data.

Required fields are exactly:

- `gate_id`
- `workflow_step`
- `owner_command`
- `receipt_path`
- `pass_criteria`
- `fail_criteria`
- `downstream_blockers`
- `stale_evidence_rule`
- `rejected_evidence`
- `source_ref`

Required gate IDs are `PG-WORKSHOP-INDEX`, `PG-WORKSHOP-CORPUS`,
`PG-MEDIA-FAMILY-COVERAGE`,
`PG-EMBEDDED-MEDIA-RECOVERY`, `PG-SLIDESLIVE-SLIDE-ACQUISITION`,
`PG-WORKSHOP-TALK-MEDIA-ACQUISITION`,
`PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY`, `PG-CORPUS-TOKEN-INDEXING`,
`PG-INDEX`, `PG-TOPIC-STRATEGY`,
`PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION`, `PG-QUERY`, `PG-GPT-RESEARCHER`, `PG-PAPER-MATCHING`,
`PG-COVERAGE-GAP-FILL`, `PG-SYNTHESIS`, `PG-NARRATION-SCRIPT-REVIEW`,
and `PG-OPTIONAL-PIPELINES`.
`PG-WORKSHOP-INDEX` and
`PG-WORKSHOP-CORPUS` are required whenever the analysis handoff's
`workshop_coverage_posture` is any value other than `out_of_scope`.

`PG-MEDIA-FAMILY-COVERAGE` is the fail-closed media-family coverage gate:
the download/corpus stage receipt must carry a `media_family_coverage`
payload (via `conference_pipeline.program_coverage`) in which every declared
media-bearing program family (tutorials, keynotes/invited talks, orals,
workshops, expo) has at least one discovered item page, or a
receipt-recorded explicit operator exclusion. Zero item pages for a
non-excluded family emits `program_family_media_items_missing` and blocks
the stage. Acquisition must never allowlist event families by URL shape —
typed and bare-ID virtual-site item pages are targets by default.

Downstream podcast, deck/presentation, release, and proof gates fail closed
when required upstream gate receipts are missing, failed, stale, or
digest-mismatched. Dry-run/mock, readiness-only, metadata-only, paper-only,
planned, and historical proof artifacts are rejected evidence for live
production completion.

## Narration Review And Fix Gate

Generated production work must split single-presenter and NotebookLM-style
narration into review-safe steps before TTS:

0. Produce one representative topic canary before bulk regeneration. Generate
   `13-CANARY-OBJECTIVE-EVIDENCE.{json,md}` with
   `presentation.phase13_canary_objective_evidence` to prove MP4/audio/slide
   receipt duration alignment, non-black first/middle/tail frames, and contact
   sheet evidence for both video styles. Then generate
   `13-CANARY-REVIEW-PACKET.{json,md}` with
   `presentation.phase13_canary_review_packet`; block bulk rollout until the
   packet has no technical blockers, the canary `human_review_receipt.json`
   records approval, and the packet is regenerated to
   `approved_for_bulk_rollout`. Use `presentation.phase13_rollout` as the
   guarded pre-TTS bulk entrypoint; it must plan or execute steps with
   `--require-approved-canary --canary-review-packet <packet.json>`, include
   both deck-level narration reviews and cross-series
   `review_narration_series.py` receipts, and stop before
   TTS/video. Before non-canary `--execute`, run
   `presentation.phase13_bulk_readiness` and require `ready_for_bulk_rollout`.
   Before closing the phase, run `presentation.phase13_completion` and require
   `complete`.
1. Generate or refresh single-presenter `video_build/script.json` with
   `build_presenter_video.py --stop-after-script-json`, then run
   `presentation.presenter_narration_review` across every topic deck and
   executive summary deck. Single-presenter TTS is blocked unless each
   `presenter_narration_review.json` records `approve_for_tts=true` and the
   single-presenter series review passes against the single-presenter delivery
   contract.
2. Generate or refresh `delivery_notebooklm.json` and
   `video_build_notebooklm/script.json` with
   `build_notebooklm_style_presenter_video.py --stop-after-transform`.
3. Run `presentation.editorial_gate` on each deck after both reviewable script
   JSON files exist and before either TTS path. Use
   `scripts/presentations/presentation_editorial_gate.py --deck-dir decks/<id> --deck-kind <topic|executive_short|executive_long> --report-out decks/<id>/presentation_editorial_gate_report.json`.
   Topic-deck TTS/video/PDF/release/proof is blocked if the gate finds missing
   storyboard evidence, duration estimates above 2400 seconds, excessive
   figure-detail/zoom density, obvious redundant claim clusters, generic
   repeated SVG patterns, or important figure entities missing from narration.
   The generation plan must not force a separate figure-detail/zoom beat for
   every SVG; most visual references should be integrated into the parent
   slide narration, with explicit zoom/detail beats reserved for complex or
   central visuals.
4. Run `presentation.narration_script_review` across every topic deck and
   executive summary deck, then run `presentation.narration_series_review`
   across the full deck set. TTS/video rendering is blocked unless each
   `narration_review.json` records `approve_for_tts=true` and the series
   review has no unaccepted failure.
5. **Accept the DECK before rendering any video.** Run
   `presentation.presentation_review` (the multi-model produced-deck review)
   against the rendered screenshots, PDF and `visual_llm_qc_report.json`, and
   block TTS and both video builds until every deck passes. The review needs
   the blueprint, blueprint review, deck contract, render spec, source binding,
   visual-QC report and slide screenshots -- it needs no video, no audio and no
   delivery contract, so nothing expensive has to exist first. See
   `conference-rough-cut-review` Gate A.

   This step is ordered here because of a measured failure. On CVPR 2026 the
   review ran only at the END of the delivery/video substage: 24 videos were
   rendered over two and a half hours, the review then failed all twelve decks
   on figure defects its reviewers had confirmed in the screenshots, and the
   figure fix that followed invalidated every video. Canary the gate on ONE
   deck first and read the per-reviewer statuses, because a reviewer that fails
   for an environment reason (an untrusted working directory, a model the
   account cannot use) degrades the ensemble silently while still returning a
   verdict.

When single-presenter deck-level review fails, production work must run
`presentation.presenter_narration_fix` or a governed LLM/manual revision skill.
When NotebookLM deck-level review fails, production work must run
`presentation.narration_script_fix` or a governed LLM/manual revision skill.
When series review fails, production work must run
`presentation.narration_series_fix` or an explicit governed cross-deck revision
pass. Then rerun the deck-level and series reviews. Fixes must keep the markdown
and `script.json` synchronized. Required
review dimensions are accuracy, cohesion, smooth transitions,
non-fluffy/non-mechanical language, interesting but calm delivery, question-hook
control, truncation detection, excessive conference-label repetition,
cross-deck duplicated figure narration, duration budget, redundancy, visual
variety, figure/narration alignment, repeated welcome/closing formulas,
executive-summary stitch
templates such as `That connects directly to`, `The operational test is`, and
`That sets up`, filler-word overuse such as repeated `critical`, `crucial`,
`key`, or `fundamental`, closing grammar joins such as `The next step is We
need...`, small fixed executive-summary transition rotations, and overall
pleasantness for the final video.

## Engaging Narration And Bookend Slides

The Phase-10 human review of the 04 canary established a deck-polish path that
makes the delivered video less mechanical and gives every cited source a single,
natural on-screen introduction. Generated production plans should apply it after
the render spec exists and before TTS. Event/source selection must remain
topic-agnostic and reproducible, but public-link resolution is
conference-adaptive code, not a universal registry lookup:

1. `presentation.add_bookend_slides` injects the opening "relevant events +
   most-cited papers + prepared-by" slide and the closing hyperlinked
   "sources & further reading" slide, derived deterministically from the deck's
   own frequency-ranked `render_spec.json` source chips. Per-deck curated event
   labels live in an optional `bookend_event_overrides.json` (speaker →
   `{role, detail}`); decks without it fall back to derived talk titles, so the
   step is not topic-hardcoded.
   Before bookend injection, Reveal.js render, PDF export, or video release,
   production plans must include a source-link adapter implementation checkpoint:
   inspect the current conference corpus and source metadata (OpenReview/forum
   ids, conference program pages, workshop/media inventories, transcript
   extraction metadata, poster/media paths, Drive source maps, and executive
   source-link maps), then update or extend the resolver/render/audit code as
   needed. Expected edit surfaces include
   `src/conference_pipeline/presentation/source_links.py`,
   `scripts/presentations/add_bookend_slides.py`,
   `src/conference_pipeline/presentation/structured_video.py`,
   `scripts/presentations/audit_presentation_links.py`, or a clearly named
   conference adapter/config plus tests. Do not treat missing links as a
   registry-data problem only; if the current corpus exposes public URLs and
   the deck cannot render them, the production plan must patch code and rerun
   tests before proceeding.
2. `presentation.engaging_narration` drafts two-host narration (named hosts,
   `ENGAGING_NARRATION_HOST_A`/`HOST_B`, default Iapetus/Erinome) for every
   slide: each source introduced once instead of repeated mechanically, no wrong
   or mechanical visual descriptions, analogies and concrete examples drawn from
   the analysis, ~20-25 minute target. It uses Claude Opus via
   `ANTHROPIC_API_KEY` (source the repo `.env` first).
3. `presentation.rebuild_delivery_engaging` rewrites both delivery contracts to
   carry that narration over all render-spec slides (including the bookends),
   wiring the two-host voices and word-count-based timing.
4. `presentation.reconcile_contract_hashes` re-stamps the cross-referenced
   contract hashes (`render_spec.input_hashes`,
   `delivery_*.presentation_deck_sha256`, source-binding, visual-plan) after the
   bookend and narration edits so the produced-presentation review does not flag
   artifact drift.

The narration produced here still passes through the standard
`presentation.narration_script_review` / `presentation.presenter_narration_review`
and series-review gates above before TTS; the engaging path changes the content,
not the review contract. NotebookLM TTS must use the multi-speaker voice config
(`build_presenter_video.py` derives the host voices from the delivery `style`),
so the two hosts are not silently collapsed into a single voice.

## Benchmark-Aware Provider Contracts

Production ROADMAP.md must include `## Benchmark-Aware Provider Contracts`.
Production ROADMAP.md must also include `## Task Model Defaults`, preserving the
operator-facing default ASR/OCR/LLM/embedding matrix from the source analysis
handoff. This is a hard gate so the transcription/OCR phase can see defaults
such as `transcription.session_speech` without spelunking through JSON.
Production `approval-summary.json` must carry rows under
`"benchmark-aware provider contracts"` and `"task model defaults"`, and those
rows must preserve `source analysis handoff.provider_contracts` and
`source analysis handoff.task_model_defaults` exactly.

The production audit must fail before execution when provider-contract rows are
missing, changed, locally substituted outside approved scope, or superseded by
prose without named benchmark receipts. Production generation must not reselect
or rewrite providers from the analysis handoff.

Production generation must preserve the local-LLM-default extraction contract
from the analysis handoff: exhaustive extraction uses the configured local
OpenAI-compatible LLM endpoint as the default path; heuristics prefilter and
serve as fallback; cloud LLMs are escalation-only for low-confidence or ambiguous
records where local LLM and heuristics produce insufficient confidence.
Production generation must not make cloud LLM the default extraction path or
promote report synthesis, OCR benchmarking, podcast, deck, release, or other
non-extraction stages into unsupported local provider paths. GPT-Researcher
retrieval is a separate analysis-stage contract: local Qwen3.6 35B-A3B
(`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`) plus local BGE-M3 embeddings
(`mlx-community/bge-m3-mlx-8bit`, 1024 dim), BM25+vector hybrid retrieval, and
web search disabled are the default evidence lane. The production handoff must
preserve the parameter defaults `MAX_ITERATIONS=5`,
`DEEP_RESEARCH_BREADTH=4`, `DEEP_RESEARCH_DEPTH=3`, `MAX_SUBTOPICS=8`,
`RETRIEVAL_K=18`, `PAPER_MATCHING_RETRIEVAL_K=18`,
`CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, `TOTAL_WORDS=3000`,
`SMART_TOKEN_LIMIT=8192`, `STRATEGIC_TOKEN_LIMIT=8192`,
`FAST_TOKEN_LIMIT=4096`, and `TEMPERATURE=0.0`. Final public narrative
synthesis remains governed by `analysis.deep_narrative_synthesis`
(`openai:gpt-5.6-luna`; no separate live fallback — the retired `gpt-5.4`
fallback pairing was collapsed since it resolved to the same model) unless
local deep-validation receipts pass.

Before query planning, generated production work must run the title-spine topic
strategy owner:
`.venv/bin/python scripts/build_topic_strategy.py --workspace conferences/<slug>
--run-id <run-id> --mode live --provider-cost-approved --model openai:gpt-5.6-sol
--model anthropic:claude-opus-4-8 --model
gemini:gemini-3.1-pro-preview --merge-model openai:gpt-5.6-sol --summary`.
`PG-TOPIC-STRATEGY` passes only when the receipt records a title-spine sha256,
invited/keynote, main oral, session, workshop, and segmented workshop talk
source families, three recommendation outputs, exactly 12 final topics in
`analysis/topic_strategy.json`, and live provider receipt paths. Mock mode,
stale candidate interest groups, single-model topic choices, and prose-only
topic lists are rejected production evidence.

After topic strategy and before query planning, generated production work must
run the Qwen full-read exhaustive extraction owner:
`.venv/bin/python scripts/run_llm_full_read_extraction.py --workspace
conferences/<slug> --run-id <run-id> --scope all --resume --model
mlx-community/Qwen3.5-35B-A3B-4bit`. `PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION`
passes only when `14A-llm-full-read-extraction-receipt.json` has
`status: succeeded`, `scope: all`, `limit: null`,
`checks.completion_state: complete`, `completed_unit_count ==
expected_unit_count`, `failed_count: 0`,
`checks.transcript_source_coverage.unread_input_count: 0`, and by-unit/by-topic
output families. `completed_unit_count == expected_unit_count` is not coverage
evidence: both counts come from the same unit iterator, so a transcript that
yields no unit lowers both and the run still reads "complete" (CVPR 2026's
2026-09-19 full read did, reading none of its talk-annotated keynotes,
tutorials, workshops, or oral sessions). Only `transcript_source_coverage`
names transcripts that carry text and produced no read unit; review its
`by_storage_lane` rows so every registered transcript evidence lane shows
`unit_count > 0`, and treat a large `pruned_obsolete_unit_count` as a finding
to explain before promotion.
Heuristic 14A receipts alone, pilot runs, `--limit` runs, transcript-only runs,
metadata-only receipts, and empty transcript sidecars are rejected production
evidence.

When the analysis handoff requests provisional exhaustive summary packs, generated
production work may insert `analysis.exhaustive_topic_summarization` immediately
after the full-read gate and before Stage 7 narrative/source-pack promotion.
Use dry-run first to size the 12 topic x 2 category workload, then live Gemini
mode only with explicit provider-cost approval:
`.venv/bin/python scripts/summarize_llm_full_read_topics.py --workspace
conferences/<slug> --run-id <run-id> --mode live --provider gemini --model
gemini-3.1-pro-preview --provider-cost-approved --max-input-tokens 400000
--summary`. The outputs under `analysis/exhaustive_summaries/` are inspection
inputs until a future platform phase makes them a formal production stage.

Before live artifact generation begins, generated production work must initialize
the mounted Drive backup leaf with `scripts/sync_to_gdrive.py --profile backup
--remote-path <mounted-conference-leaf> --init-only` and a JSON
`--receipt-path`. Every artifact-producing generated phase must then include a
manifest-driven backup checkpoint immediately after the stage writes outputs
and before downstream stages consume them: `scripts/sync_to_gdrive.py --profile
backup --remote-path <mounted-conference-leaf>` dry-run, write mode, JSON
`--receipt-path`, stale-artifact failure rule, and fail-closed verification.
The production audit must also validate workflow
registry dispatch coverage for declared workflow stages, including dispatch
kind, implementation reference, receipt path, cost accounting, backup
checkpoint, and verification contract. Final proof must also run
`scripts/generate_workspace_gitignore.py --workspace <workspace> --check` and
`scripts/verify_stage_script_usage.py --workspace <workspace> --run-id <run-id>
--docs-root docs --strict` so completed live receipts prove required owner scripts, reject
deprecated completion owners, and prove that Google Drive/cache-owned artifact
families are not treated as git-owned payload.

## Extraction Provider Policy

Production generation preserves the local-LLM-default extraction contract from
the analysis handoff. Exhaustive extraction uses the configured local
OpenAI-compatible LLM endpoint as the default path. Heuristics serve as prefilter
and fallback when the local endpoint is unavailable or confidence meets the
classification threshold. Cloud LLMs are escalation-only for low-confidence or
ambiguous records that neither heuristics nor the local LLM can resolve with
sufficient confidence.

Production plans must include an executable local LLM readiness preflight before
extraction: confirm the endpoint is reachable, the model is available, and sample
inference succeeds. If the local endpoint is absent or unreachable, the plan must
fail closed rather than silently routing to cloud LLM.

Slide/Visual OCR decisions must be classified as one of three states based on a
source-coverage assessment: `required`, `skipped_with_rationale`, or
`deferred_blocked`. Production plans must not require unconditional OCR without
a coverage assessment justifying it.

Non-extraction generation providers use their own benchmark-governed contracts.
GPT-Researcher evidence defaults to local Qwen3.6 35B-A3B plus local BGE-M3 embeddings
with corpus-only hybrid retrieval; ASR, TTS, ranking, podcast, presentation,
OCR, and final report synthesis are not changed by the extraction-local-default
policy.

## Indexing Embedding Readiness

Production `PG-CORPUS-TOKEN-INDEXING` is required before query planning,
GPT-Researcher, paper matching, synthesis, podcast, deck/presentation, release,
or proof. The generated plan must require Stage 6 receipts and index digests to
record `token_budget_policy_digest`, `embedding_policy_digest`,
`EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, and
`vector_embeddings_written: true`. Embedded chunk counts must match per-store
sidecar chunk counts. The reusable owner must write durable embedding sidecar
manifests before bulk Chroma load and receipt
`embedding_sidecar_manifest_path`, `embedding_sidecar_manifest_sha256`,
`embedding_sidecar_format`, `embedding_sidecar_shard_count`, and
`chroma_insert_batch_size`. `CORPUS_VECTOR_EMBED_BATCH_SIZE` and Chroma insert
batch size are runtime tuning evidence, not embedding identity; production
plans must not force reembedding only because a batch size changes.
Word-count-only sidecars, readiness-only probes, partial Chroma rows,
conference-local embedding reimplementations, and historical exhaustive
extraction output without registry receipt are rejected evidence. Cite
`docs/reference/stage-owner-audit.md#indexing-token-bounded-chunking` and
`docs/reference/stage-owner-audit.md#indexing-local-embedding-materialization-policy`.
The Stage 6 repair must not redownload source pages, papers, workshop PDFs,
audio, slides, video, or poster media.

Production `PG-INDEX` must not treat local embedding unavailability as a blocker
unless the stage has tried to make the configured local embedding client usable.
The generated plan must require a receipt proving:

- `build_embedding_client(...)` was called for the configured provider/model/dim
- first use was forced with `embed_texts(...)`, so MLX loads or Ollama
  starts/probes
- `embedding_readiness.status`, `provider`, `model`, `embedding_dim`, and
  `client_start_or_load_attempted` were recorded
- failures use a blocker such as `blocked_local_embedding_client_start_failed`
  after the startup/load probe fails

Chunk sidecars may be written before or alongside this probe for resumability,
but downstream vector retrieval cannot be claimed from sidecars alone.

Workshop speech transcripts must be talk-aware before indexing or analysis when
workshops are in scope. Before Stage 5/6, plans must prove workshop talk-media
acquisition explicitly: run workshop agenda/site/talk media acquisition and
write `storage/workshops/talk_media_acquisition_receipt.json` with
workshop/talk/media metadata, or write a formal
`storage/workshops/workshop_talk_media_scope_waiver.json` with decision,
rationale, approver, and downstream impact. A zero-count workshop speech audit
is rejected evidence for deciding workshop talks were out of scope.
The acquisition plan must name `acquisition.workshop_talk_media` plus
`acquisition.embedded_media_recovery`
(`src/conference_pipeline/acquisition/embedded_media.py`): crawl public pages
first, retry with authentication when blocked, run embedded-player extraction
with `embedded_media_links`, strip secrets with `sanitize_media_url`, and use
yt-dlp recovery before accepting inaccessible-media blockers. The plan must
bind a conference-local embedded host allowlist via `--embedded-media-host
<conference-local-host>` (for ICLR production this is
`--embedded-media-host slideslive.com`; other conferences must name their own
verified embedded-video host rather than reusing ICLR's), and receipts must
record `allowed_hosts`. Do not plan YouTube, Vimeo, Zoom, social/profile, or
unrelated embeds as workshop presentation media for any conference unless a
conference-specific phase explicitly widens that conference's allowlist. Cite
`docs/reference/stage-owner-audit.md#acquisition-workshop-talk-media` and
`docs/reference/stage-owner-audit.md#acquisition-embedded-media-recovery`.

If no vetted `talks.json` exists after acquisition, plans must run
`transcription_ocr.workshop_talk_boundary_discovery` with Gemini 2.5 Pro
(`gemini-2.5-pro`), write `*_talks.json`, emit 5-minute talk-context batches, record
`provider_receipt_paths`, and bind cost evidence to
`cost_task_id=workshop_talk_boundary_discovery`. Plans must then name
`transcription_ocr.workshop_talk_segmentation`, preserve `workshop_id`,
`workshop_title`, `talk_id`, `talk_title`, and `talk_speaker`, and block with
`talk_segmentation_provider_not_configured` when neither `talks.json` nor a
governed discovery path is available. audio-only recovery is rejected evidence for talk-aware transcripts.
Use `conference_pipeline.workshop_talk_segmentation`,
`src/conference_pipeline/workshop_talk_segmentation.py`, and
`scripts/discover_workshop_talks.py` plus `scripts/segment_workshop_talks.py`,
and cite
`docs/reference/stage-owner-audit.md#transcription-ocr-workshop-talk-boundary-discovery`
and
`docs/reference/stage-owner-audit.md#transcription-ocr-workshop-talk-segmentation`.

## GPT-Researcher Retrieval And Parameter Policy

Production plans must carry the registry-owned GPT-Researcher analysis stage:
`analysis.gpt_researcher_stage`
(`docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage`) backed by
`conference_pipeline.iclr2025_live.run_query_planning_gpt_researcher`,
`conference_pipeline.research.researcher`,
`conference_pipeline.research.query_planner`,
`conference_pipeline.research.config`, and
`conference_pipeline.research.walker`.

The default GPT-Researcher evidence lane is local/corpus-only:
`GPTR_LOCAL_LLM_RUNTIME_KIND=mlx`,
`GPTR_LOCAL_LLM_SMART_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,
`GPTR_LOCAL_LLM_FAST_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,
`EMBEDDING_PROVIDER=mlx`, and
`EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit`. The base vector store is
topic-agnostic; generated deep research must retrieve over the whole corpus or
stable source scopes (`papers`, `artifacts`, `workshop`, `main_conference`),
not mutable topic IDs. The default retrieval mode is
`workflow.research.gpt_researcher.retrieval_mode: hybrid`, which uses BM25 plus
vector reciprocal-rank fusion. Vector-only retrieval is an explicit override and
must be visible in plan config and receipts. Generated work must expose and
record `MAX_ITERATIONS`, `DEEP_RESEARCH_BREADTH`, `DEEP_RESEARCH_DEPTH`,
`DEEP_RESEARCH_CONCURRENCY`, `MAX_SUBTOPICS`, `TOTAL_WORDS`, token limits,
`RETRIEVAL_K`, `PAPER_MATCHING_RETRIEVAL_K`, `CANDIDATE_MULTIPLIER`, `RRF_K`,
and `TEMPERATURE`, with defaults `5/4/3/2/8`, `TOTAL_WORDS=3000`,
`SMART_TOKEN_LIMIT=8192`, `STRATEGIC_TOKEN_LIMIT=8192`,
`FAST_TOKEN_LIMIT=4096`, `RETRIEVAL_K=18`,
`PAPER_MATCHING_RETRIEVAL_K=18`, `CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, and
`TEMPERATURE=0.0`.

For deep-dive topics, generated production work must first run and receipt
`analysis.deep_dive_question_selection` with GPT-5.6 Luna to compress exhaustive
summary key points, tensions, and open questions into 4-6 selected deep dives
per topic. The owner command is
`.venv/bin/python scripts/select_deep_dive_questions.py --workspace conferences/<slug>
--run-id <run-id> --mode live --model <registry-default> --provider-cost-approved
--min-questions-per-topic 4 --max-questions-per-topic 6 --summary`, where
`<registry-default>` is `default_model("analysis.deep_dive_question_selection")`
from the model registry (currently `gpt-5.6-luna`; read it from the registry
rather than hardcoding it), and one-run-per-summary-bullet expansion is rejected
live evidence.

For each selected question, generated production work must run and receipt two
GPT-Researcher lanes before promoting findings: the global no-topic-id lane
owned by `analysis.gpt_researcher_stage`, and the keyword-prefiltered lane
owned by `analysis.gpt_researcher_heuristic_prefilter`,
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
source-family leakage, and citation-invalid heuristic reports are rejected live
evidence. The combined runner is `analysis.dual_deep_dive_research_lanes` via
`scripts/run_dual_deep_dive_research.py --resume` and must use
`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` as the default local GPT-Researcher
model.

Generated production work must then run the packaged reconciliation and
reference-governance owner before rerunning final narrative synthesis or
starting slides:
`.venv/bin/python scripts/run_topic_reconciliation_reference_workflow.py
--workspace conferences/<slug> --run-id <run-id> --mode live --model
mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary`. The receipt
`07H-topic-reconciliation-reference-workflow.json` must show 07D
reconciliation, immediate 07E validation, deterministic mechanical safety
fixes, bounded 07G candidate resolution when needed, 07F broad repair only
after unresolved candidate blockers, and final 07E validation. Final narrative
or slide synthesis requires `final_validation_status: succeeded`, positive
ledger and citation counts, and zero `final_invalid_reference_count`,
`final_unknown_reference_count`, `final_title_mismatch_count`, and
`final_bare_reference_warning_count`.

The individual owners remain registered fallback/debug evidence. 07D must show
per-topic prompt token estimates under the local context ceiling,
`analysis/topic_reconciliations/*.md`, local provider receipts, and enriched
source packs. 07E must show per-topic ledgers and validation sidecars. If 07E
blocks, 07G candidate resolution is the first LLM-assisted repair lane and may
apply only packet-bound Qwen decisions with clean `final_validation`. Only when
07G records unresolved blockers should 07F broad repair run, and final 07E must
still succeed before publication. Slide synthesis must consume reconciled,
validated analysis and slide briefs, not raw parallel deep-dive reports as the
first reconciliation point.

The local lane is an evidence/draft substitution, not automatic
production-final analysis; production `analysis/<topic>.md` continues through
governed GPT-5.6 Luna/GPT-5.4 deep narrative synthesis unless the local report
passes the same validation gates.

## Dual-Store Paper Matching

When presentation/deck paper matching is in scope, production plans must fail
closed unless the live run includes the v1.91 paper-matching chain:

- final indexing writes separate `papers` and `artifacts` stores;
- full paper chunks include title, section title, and subsection title when
  known and are built through `conference_pipeline.indexing.chunking`;
- artifact-store GPT-Researcher completes before keypoint extraction;
- a configured frontier model extracts keypoints from artifact findings;
- paper-store GPT-Researcher uses those keypoints;
- `matched_papers.json` records the returned paper titles with traceability to
  keypoints;
- `ranked_papers.json` is treated only as a compatibility output;
- `scripts/rank_papers.py` is rejected as the reusable production mechanism and
  may appear only as deprecated NeurIPS 2025 legacy evidence.

## Conditional Slide OCR

Before committing to slide OCR, production plans must run or consume a
source-coverage assessment comparing these four sources against the evidence
needs of downstream deliverables: transcripts, paper text, source files
(existing PDF text layer), and available media metadata.

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

When OCR is `required`, the production plan must record: `why_required` (one of
the three triggers above), `expected_receipt_paths` (list),
`provider_runtime_contract` (str), `cost_rate_limits` (str),
`downstream_blockers` (list).

Production plans must not require unconditional OCR without a source-coverage
assessment justifying the `required` classification.

## NotebookLM Source Export

When production work delivers a NotebookLM source bundle for human deep-dive, it
must run and receipt the filesystem-only packager
`.venv/bin/python scripts/package_conference_for_notebook_lm_export.py
--workspace conferences/<slug> --paper-mode <abstract|full|both>`. The packager
makes no provider calls; it reads already-produced artifacts and writes
`exports/notebooklm_export/` with chunked `notebooklm/*.md` parts,
`manifest.json` (treat as a machine-readable contract), and `index.csv`. Parts
stay under one NotebookLM source via `--max-words` / `--max-bytes` (defaults
450000 / 4000000, a safety margin below NotebookLM's ~500k-word per-source
limit), with `--papers-max-words` / `--papers-max-bytes` for paper chunks.

`--paper-mode full` is the live-only full-text deliverable: cleaned full paper
text grouped by primary topic (`papers_full_<NN_topic>` prefixes), no abstract
chunks. It is content-addressed and deterministic — it indexes
`paper_text` / `workshop_paper_text` extraction records by `text_sha256` (the
on-disk extracted-text path is retention-redacted, so the content hash is the
join key) and joins to local OpenReview metadata via the PDF hash for titles
(unjoined records fall back to head-extracted titles with full text preserved).
Cleaning drops the recurring conference header and references/bibliography list,
keeps appendices, de-hyphenates line breaks, and collapses whitespace.

Primary-topic grouping is set by `--paper-topic-source` (default `llm`): it
prefers the exhaustive `llm_full_read` semantic assignment (keyed by forum id)
and falls back to the confidence-ranked keyword heuristic (`topic_ids[0]`) for
papers the LLM lane did not cover (workshop and calendar-sourced papers). Do not
ship the raw `topic_scores` argmax grouping — it collapses most papers into a
couple of topics. Provenance is recorded in
`counts.full_text_topic_from_llm_full_read` /
`counts.full_text_topic_from_heuristic` and a per-paper `Topic source` line.

Unless `--skip-slides` is set, the packager includes each topic's structured
presentation contracts (`presentation_deck.json`, `render_spec.json`, and
delivery contracts when present) as per-topic sources (prefix
`presentation_<NN_topic>`), counted in `counts.topic_presentation_contracts` /
`counts.presentation_source_files` and `parts.presentations`. The packager also writes a
`notebooklm/00_conference_overview.md` orientation source (corpus scale, the
executive-summary throughline and cross-topic conclusions, the 12 topic areas
with paper counts and one-line characterizations, and a how-to-use guide),
recorded at `manifest.json#conference_overview` (`--skip-conference-overview`
disables it). Unless `--skip-presentation-prompts` is set, it also writes
`exports/notebooklm_export/presentation_prompts/` — a long (~60 min) and a short
(~20 min) NotebookLM presentation prompt for each topic plus conference-wide
long/short variants (26 files), grounded in the exhaustive LLM summaries and
executive summary. The prompts are paste-in instructions, not upload sources,
and are listed in `manifest.json#presentation_prompts`.

The whole bundle reaches Google Drive through the backup sync: the
`export_notebooklm` storage-manifest family (`exports/notebooklm_export`,
`sync_profiles.backup: true`) is what `scripts/sync_to_gdrive.py --profile backup`
copies. New conference workspaces inherit it from the TEMPLATE manifest; verify
it is present before claiming the deep-dive bundle is backed up.

Completion for a full-text export deliverable requires: the packager command in
the run log, a regenerated `manifest.json` showing `paper_mode: full` with
`counts.full_text_main_papers`, `counts.full_text_workshop_papers`,
`counts.papers_full_source_files`, the topic-source provenance counts, and
`counts.topic_slide_decks`; every `parts.papers_full[*]` part at or under the
configured `max_words`; a non-null `manifest.json#conference_overview`, the 12
per-topic slide sources, and the 26 `presentation_prompts` entries present
(unless explicitly skipped); and the resulting total NotebookLM source count
(conference overview + papers_full + slides + transcripts + analysis parts)
recorded against the target tier's sources-per-notebook limit.
`tests/test_notebooklm_conference_export.py` is the behavioral gate.
Conference-specific counts and tier assumptions live in the milestone context
packet, not this skill.

## Inputs To Confirm

Require or infer from visible planning artifacts:

- conference slug and workspace path
- milestone/version identity
- desired output form: one execution phase, a multi-phase milestone, or auto
- execution mode: live-only production
- target corpus bounds and topic scope
- source/auth posture and provider/runtime credential requirements
- provider-cost approval posture, including whether the full-pipeline estimate
  has been run for the target workspace
- analysis handoff fields: evidence class, corpus bounds, event-family
  inventory status, media-family scope (`media_family_scope`),
  interest-group decisions, exclusions, provider/model posture, benchmark
  requirements, and local-provider deferrals
- the media-family scope re-confirmation: ALWAYS re-ask, at execution
  planning time, whether any media-bearing event family is being excluded;
  the default is all families included, and this question is truth-critical
  even when the analysis handoff already recorded an answer
- registry task ids plus approved provider/model choices, pending current
  research, stage-local benchmark requirements, `not approved yet` statuses,
  and failover providers for ASR, OCR/image parsing, extraction, research,
  ranking, podcast, presentation, TTS, and repair tasks
- whether an OCR/image parsing benchmark is required before bulk live OCR
  provider calls
- required outputs: downloads, transcripts/OCR, analysis, QA, podcast audio,
  deck/presentation exports, release bundle, proof packet
- storage/share targets and publication boundaries

Ask only truth-critical questions. If a field is unknown but can be expressed as
a blocker, record it as a blocker instead of inventing completion evidence.

## Output Mode

Choose the output shape deliberately:

- **One phase:** Use when the user asks for one phase and the repo already has
  real executable owners for every in-scope stage. The generator requires
  `--all-owner-commands-exist`.
- **Milestone:** Use when stages must be implemented, wired, or proven before
  execution can honestly complete.
- **Auto:** If any in-scope stage is unsupported or has no owner command, choose
  a milestone or include an implementation task before the execution task.

Do not produce a plan whose main command does not exist unless the same plan
also implements that command before running it.

## Stage Checklist

Every production plan must account for these stages, even if a stage is
explicitly out of scope:

1. Source discovery or source import.
2. Full corpus freeze: all discovered conference items unless explicitly
   excluded before expensive provider work.
   Live production source/corpus commands must not use silent item-count caps
   such as `--max-papers`; capped dry-run or development commands cannot
   satisfy the production corpus freeze.
   Before this stage runs, re-ask the media-family scope question even when
   the analysis handoff already answered it: "Media scope defaults to ALL
   media-bearing event families (<list from the handoff inventory>). Confirm
   none are being excluded, or name exclusions." Record the answer in the
   corpus-freeze receipt. A silent exclusion discovered later is a
   production-stopping defect, not a tolerable gap.
3. Paper/source/session/media download with local paths and digests, gated
   by `PG-MEDIA-FAMILY-COVERAGE`: every declared media-bearing family must
   yield at least one discovered item page or carry a receipt-recorded
   exclusion; zero item pages for a non-excluded family blocks the stage
   (`program_family_media_items_missing`).
4. Transcription, OCR, and media processing for in-scope assets.
5. Corpus inventory with provenance and unreachable-source receipts.
6. Enhanced indexing with canonical source-aware chunking and stable source
   metadata. The production owner must use the reusable
   `conference_pipeline.indexing.corpus_vector_store` asset, backed by
   `conference_pipeline.indexing.chunking`, to build full source-backed paper,
   transcript, slide/OCR, and generic artifact chunks. Conference-specific
   runners may orchestrate this asset but must not reimplement chunking or
   embedding materialization inline. Snippet-only / first-240-character
   vectorization and ASR receipt JSON chunks are rejected production evidence.
   Base corpus vector stores must be topic-agnostic and carry stable metadata
   such as `content_kind`, `conference_scope`, `source_url`, `paper_id`,
   `session_title`, `workshop_id`, `talk_id`, timestamps, and media provenance.
   The reusable owner must write durable embedding sidecar manifests before
   Chroma load, then bulk-load Chroma from those sidecars; batch-size tuning is
   recorded in receipts but not part of `embedding_policy_digest`.
   If local embeddings are expected, the owner must build
   the configured embedding client and force first use with `embed_texts()`
   before declaring the local embedding lane blocked. Static "client
   unavailable" markers without `embedding_readiness` and
   `client_start_or_load_attempted` receipt fields are rejected. Stage 6 must
   also carry `PG-CORPUS-TOKEN-INDEXING` evidence: `token_budget_policy_digest`,
   `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`,
   `CORPUS_VECTOR_EMBED_BATCH_SIZE`, `embedding_sidecar_manifest_path`,
   `embedding_sidecar_manifest_sha256`, `chroma_insert_batch_size`,
   `vector_embeddings_written: true`, and embedded chunk counts matching
   sidecar counts.
7. Query planning and Analysis generation through topic analysis generation.
8. Dual-store paper matching when presentation paper matching is in scope:
   artifact-store GPT-Researcher, frontier-model keypoints, paper-store
   GPT-Researcher, canonical `matched_papers.json`, and compatibility
   `ranked_papers.json` only when existing deck scripts require it.
9. Narrative analysis generation: public `analysis/<topic>.md` files are deep analysis documents with Executive Summary, Key Findings, Detailed Analysis,
   References/Sources, Cross-References, Recommendations, Questions, and a
   compact Slide Brief section. Deck-facing summaries live in
   `analysis/slide_briefs/<topic_id>.md`, while raw extraction/chunk/research/
   matched-paper ledgers live in `analysis/provenance/<topic_id>.md`,
   `analysis/provenance/<topic_id>.json`, and
   `analysis/analysis_citation_manifest.json`. Production uses
   `analysis.deep_narrative_synthesis` (`openai:gpt-5.6-luna`, no separate
   live fallback) and must fail closed if the LLM/model, iterative
   GPT-Researcher evidence, or deep validation gates are missing. `Source
   Basis`, `Verbatim transcript/OCR evidence`, `Matched Paper Evidence`, and
   `Source Gaps` are rejected as top-level public-report sections. Local
   BGE+Qwen GPT-Researcher outputs may satisfy the evidence-input side, but not
   the final narrative synthesis side unless local deep-validation receipts
   pass.
10. Analysis manifest refresh and digest validation, including optional
   `deep_analysis_path`, `slide_brief_path`, `analysis_profile`,
   `narrative_model`, `deep_validation`, `provenance_path`, and
   `provenance_json_path` topic fields.
11. QA, traceability, source-gap checks, coverage repair, and human correction
   loop.
12. Gap-fill and synthesis, including executive or trend reports when in scope.
13. Podcast script generation plus rendered audio when podcast output is in
   scope. When per-deck NotebookLM podcasts are in scope, add one Google
   NotebookLM Audio Overview per topic deck and per executive deck through
   `scripts/generate_notebooklm_podcast.py --live --ack-live-notebooklm --format deep-dive --length long`
   (owner `podcast.notebooklm_audio_overview`, receipt
   `stage-10-notebooklm-podcast.json`), sourced from each deck's analysis spine
   plus the blueprint-cited paper texts and talk transcripts and bound by
   `audio/<id>/notebooklm_podcast/notebook_manifest.json`; then deliver the
   curated Drive folder with `scripts/deliver_notebooklm_podcasts.py`
   (owner `podcast.notebooklm_delivery`). Dry-run or `--fake-engine`
   manifests and the public-bundle `scripts/generate_notebooklm_audio.py`
   planner are rejected completion evidence.
14. Deck/presentation generation, render/export, visual/structural validation,
   and presenter script when presentation output is in scope. If the conference
   output scope includes executive summary presentations, production must create
   and validate both `decks/executive_summary_short` and
   `decks/executive_summary_long` through
   `scripts/presentations/generate_executive_overview_decks.py`; the source
   reports `analysis/executive_summary.md` and `analysis/trend_report.md` do
   not satisfy those deck deliverables. Figure generation
   must map `presentation.svg_figure_generation` to a concrete programmatic
   matplotlib/Plotly/svgwrite owner command,
   `scripts/presentations/generate_programmatic_figures.py`, and that command
   must produce content-aware, slide-specific editorial SVGs rather than generic
   placeholder diagrams. If the owner cannot beat scaffold-level visuals, the
   phase remains blocked.
15. Release bundle with inventory and review status. Its public release runs through
   `conference-pipeline release` (v6.0-Public-Release Phase 6), never through hand-named
   publishing scripts:
   - `release init <slug>` drafts `storage/public_export_manifest.json` with every family
     excluded; `release init <slug> --refresh` adds storage families a manifest does not
     mention yet. The owner rules each family's license and `caution` acknowledgement
     before a family is promoted.
   - `release families <slug>` builds the reader families from
     `storage/release/reader_sources.json`: the paper catalog, topic papers, excluded ids,
     statistics (talks included), narratives, and the limitations record from
     `storage/release/limitations_source.json`, each quote checked against the close
     record.
   - `release plan`, then `release build` (non-live), then `release check` (exit 0, 0
     private-marker findings). Bundle-copy redaction is declared per family in
     `redaction_rules`; workspace receipts are never edited.
   - `release publish <slug> --approved-by <name> --surface ...` only with the owner's
     approval for that release.
16. Provider cost estimate, actual receipt scan after paid runs, and cost-gate
   evidence in approval/proof docs.
17. Final proof packet and support-label promotion.

## Thin Conference Orchestrator Composition

Each conference keeps its own production orchestrator
(`scripts/run_<slug>_production.py`). The operator made this decision on
2026-09-30. Conferences have real quirks: organizer-channel YouTube harvest,
SlidesLive recovery and event-talk splitting, OpenReview auth differences,
registration-gated media, and so on. A single fully generic runner is not the
goal. The orchestrator must still be **thin**: it imports the shared stage
owners from `src/conference_pipeline/` and implements only its conference
quirks, behind declared hooks. Background, measurements, and the migration
plan are in `.planning/proposals/thin-conference-orchestrators-20260930.md`.

Generated production plans must follow these rules:

- **Never fork a runner.** Do not copy an existing conference runner as the
  starting point. Do not copy functions between runners. The ICML 2026 runner
  began as an ICLR 2026 fork and ended up with about 7,400 duplicated lines,
  about 1,300 dead lines, and about 800 lines of stale forks of `src/` modules.
- **Never re-implement a shared function locally.** If a function with the
  same job exists in `src/conference_pipeline/**`, import it. If it needs a
  change, change it in `src/` for every conference.
- **Call shared stages through adapters.** Each stage handler in the
  orchestrator is a short adapter (target ≤ 40 lines). It builds the
  `conference_pipeline.production.ProductionRunnerContext` and calls the
  shared stage module. The Stage 10–12 adapters
  (`production.presentation_stage`, `notebooklm_stage`, `release_stage`,
  `proof_stage`) are the reference shape.
- **Put quirks behind named hooks.** A conference quirk is an acquisition
  lane, an auth posture, a record builder, a retry authorization, a priority
  ordering, or an extra CLI command. Each one is implemented locally and
  registered under a named hook id (for example
  `download-corpus.acquisition_lane`). It is never implemented as an inline
  branch inside a copied stage body. Every hook default is empty, so a
  conference with no quirks passes an empty hook set.
- **Keep the orchestrator as the receipt owner.**
  - `code_identity` is `{runner, runner_path}` of the orchestrator.
  - Receipt types keep the conference's own prefix, supplied through the
    context. Shared modules never contain a conference literal;
    `scripts/audit_conference_hardcoding.py` scans `src/`.
  - Every stage still calls, in order, `upstream_freshness_blockers`, then
    `verify_stage_contract`, then `finalize_receipt`, then
    `preserve_prior_receipt`, then an atomic write.
  - On `--resume`, verify the persisted receipt as-is.
- **Plan registry changes as one batch.** Moving code into a shared module is
  registry-neutral: the orchestrator stays the owner of record. Any change to
  `src/conference_pipeline/stage_owner_registry.json` goes into a single batch
  in a window with no active run, because every edit changes
  `script_registry_sha256` on every receipt. Batch contents:
  - new owner paths and `refs` naming both orchestrators
  - `<stage>.<slug>` quirk-owner ids
  - no new `required_script_ids`
  After the batch, the stage-owner audit must stay at its recorded blocker
  baseline.

Every generated production plan must include an **Orchestrator Composition
Checklist** table. Each row is: stage → shared component → hooks used → local
quirk lines and justification. Start from this mapping. "Proposed" means the
shared stage module is planned in the proposal above. Until it exists, the
plan names the current owner and must not add a new local copy.

| Stage | Shared component(s) (import, do not copy) | Typical conference hooks |
|---|---|---|
| runner-preflight | `production_preflight` (approval scope, estimate, capacity, probes), `model_governance`, `costs.estimator`; proposed `production/preflight_stage.py` | extra provider probes, extra preflight checks |
| source-discovery | `source_discovery`; proposed `production/source_discovery_stage.py` | source-family adapters (OpenReview venues, CVF proceedings, program spine) |
| corpus-freeze | proposed `production/corpus_freeze_stage.py` | operator exclusions; media-family scope answer recorded in receipt |
| download-corpus | `acquisition.*` (`workshop_pipeline`, `workshop_paper_sources`, `proceedings_paper_sources`, `public_material_acquisition`, `youtube_talks`, `youtube_transcripts`, `youtube_audio_recovery`, `embedded_media`), `openreview_auth`, `slideslive_recovery`; proposed `production/download_corpus_stage.py` frame | acquisition lanes, auth posture, paper priority ordering, request pacer |
| transcription-ocr | `transcription_ocr`, `indexing.youtube_harvest_lanes`; proposed `production/transcription_ocr_stage.py` | transcription lanes, retry authorization |
| corpus-inventory-index | `indexing.corpus_vector_store`, `chunk_manifest`, `chunking`, `record_titles`, `keyword_index`, `talk_maps`, `event_talk_splits`, `workshop_talk_annotation_bindings`, `transcript_evidence_lanes`, `paper_source_metadata`, `paper_corpus_curation`, `index_coverage`; proposed `production/corpus_index_stage.py` | record builders (proceedings, workshop, caption transcript, event-talk maps) |
| exhaustive-topic-extraction | `scripts/run_llm_full_read_extraction.py`, `analysis.exhaustive_summaries`; proposed `production/exhaustive_extraction_stage.py` | none expected |
| analysis | `scripts/summarize_llm_full_read_topics.py`, `scripts/select_deep_dive_questions.py`, `scripts/run_dual_deep_dive_research.py`, `scripts/match_papers_from_artifact_keypoints.py`, `scripts/run_topic_reconciliation_reference_workflow.py`, `analysis.narrative`, `analysis.source_pack_selection`, `analysis.full_read_evidence_selection`, `research.engine_choice`; proposed `production/analysis_stage.py` | source-pack policy, matched-papers loader |
| qa-gap-fill-synthesis | proposed `production/qa_synthesis_stage.py` (ICML and ICLR copies are byte-identical today) | none expected |
| podcast | proposed `production/podcast_stage.py` | none expected |
| presentation | `production.presentation_stage` (+ `presentation_chain`, `presentation_steps`, `decks`) | none (deck scope via CLI) |
| notebooklm-podcast | `production.notebooklm_stage` | none |
| release | `production.release_stage` | none |
| public release | `publishing.release_cli` (`conference-pipeline release init`, `families`, `plan`, `build`, `check`, `publish`), `publishing.reader_families`, `publishing.bundle_redaction` | the workspace's `reader_sources.json`, `limitations_source.json` and manifest `redaction_rules` |
| proof | `production.proof_stage` | none |
| cross-cutting | `production.context`, `production.receipts`, `receipt_lineage`, `stage_contracts`, `stage_script_registry`, `workflow_runs`, `provider_receipts`; proposed `production/orchestrator.py` (CLI, dispatch, summary, active-blocker halt) and `production/stage_table.py` (stage table from registry `stage_bindings`) | extra CLI args and commands |

Plan requirements for the orchestrator:

- A plan that needs a new quirk names the hook id, the lane or builder, its
  receipt fields (secret-free), and whether it is paid. A paid lane must pass
  the shared preflight authorization.
- A plan that finds a stage has no shared owner yet adds a task to lift one.
  The task lifts from the most complete existing implementation, merges fixes
  from both sides, and uses golden-receipt tests on an ICML-shaped and a
  CVPR-shaped fixture workspace. A plan must not create a third local copy.
- **Orchestrator share gate (milestone close).** Report local vs. shared lines
  per stage, and list shadow copies of `src/` functions and clones of other
  orchestrators' functions. A new orchestrator must have zero shadows and zero
  clones. Use the proposed `python -m conference_pipeline.orchestrator_share_audit`
  once it exists, and a hand-filled Composition Checklist until then. This
  gate is independent of, and in addition to, the conference-portability and
  model-registry gates.

### Stage 4 Acquisition Lanes From The Lane Catalog

The Stage 4 frame exists: `src/conference_pipeline/production/download_corpus_stage.py`
owns lane ordering (every `acquire` lane in declaration order, then every
`gate` lane), paid-lane authorization, receipts, lineage pins, capacity
checks, prior-inventory carry-forward and the Stage 4 ledger. A conference's
download-corpus quirks are declared lanes
(`ConferenceHooks.acquisition_lanes`, each an `AcquisitionLane`) plus the
`download-corpus.auth`, `download-corpus.paper_priority_key` and
`download-corpus.request_pacer` hooks. The lanes are written once, in
capability-named modules under `src/conference_pipeline/acquisition/`, each
with a `RECOMMENDED_FOR` annotation and a docstring naming the stage/quirk it
covers and the other conferences it may fit.

When generating a production plan, consult the lane catalog before writing any
Stage 4 task:

```bash
PYTHONPATH=src python -m conference_pipeline.acquisition.lane_catalog --conference "<Conference Name YYYY>"
PYTHONPATH=src python -m conference_pipeline.acquisition.lane_catalog --format json --conference <slug>
```

- **Propose the recommended lanes.** Every lane the catalog recommends for the
  conference family becomes a row in the Composition Checklist's
  download-corpus line (lane id, module, hook id, paid or not). A new CVPR
  gets the CVPR-recommended lanes (public program items, CVF workshop owner,
  organizer-channel talks, the paid priority-gap search, program spine and
  focused recovery, registration media, event media); a new ICML gets the
  ICML-recommended lanes (OpenReview Bearer auth, SlidesLive carry-forward,
  workshops first, virtual-site crawl, accepted papers with priority
  ordering, request pacer, media-family coverage gate).
- **Confirm each proposed lane against the analysis spec's
  `orchestrator_quirks`.** Keep a recommended lane only when the quirk
  inventory records the matching evidence; drop it with a one-line reason
  otherwise. A lane listed only under "may fit" is a candidate, never a
  default: adopt it only when the quirk inventory names it.
- **Bind, do not copy.** The orchestrator binds a lane with its conference
  values (venue ids, labels, paths, run pins) through the lane's settings
  dataclass; it never copies lane code. A paid lane (`paid=True`) runs only
  through the frame's `authorize_paid` preflight hook.
- **A conference with no recommended lanes** starts from the frozen-corpus
  frame with `ConferenceHooks()` (no lanes) and adds lanes from the full
  catalog only where the quirk inventory records them.
- **A genuinely new acquisition behavior** becomes a new capability-named
  module under `acquisition/` with `RECOMMENDED_FOR`, a docstring
  "Recommended for" section and a catalog entry
  (`acquisition/lane_catalog.py`; `catalog_problems()` must stay empty). Never
  name a module by conference year.
- **Registry ownership is unchanged by a lane move.** Moving a lane into a
  shared module stays registry-neutral; any owner-path change goes into the
  single registry batch (see above).

## Media, Transcription, And Analysis Truth Gates

Production plans must fail closed on media and transcript claims:

- Video/media downloads cannot be substituted with paper PDFs, HTML pages,
  abstracts, OpenReview metadata-only records, or source-import receipts.
  Transcript-backed analysis cannot use those artifacts as speech/media
  substitutes. If the analysis spec requires talks, sessions, posters, slides,
  audio, or video, the plan must discover those asset URLs, attempt downloads,
  and record local media paths, MIME types, byte counts, digests, or
  unreachable-source blockers. OpenReview PDF downloads
  (`https://openreview.net/pdf?id=<note_id>`) are acceptable paper-acquisition
  evidence for both main-conference and workshop papers when each per-paper
  receipt records `content_type=application/pdf`, a `sha256`, and a
  `byte_count > 0`; OpenReview metadata records without those bytes are not
  acceptable evidence.
- If authenticated item pages were cached as `item_page_auth_probe` artifacts
  and those pages reveal embedded SlidesLive media that did not merge into
  `download_corpus.json`, production must treat focused embedded media recovery
  as acquisition work before transcription/OCR. Use
  `acquisition.slideslive_media_recovery` or a conference-local equivalent,
  write `storage/download-corpus/slideslive_media_recovery.json`, and preserve
  the no-redownload boundary: no source pages, paper PDFs, workshop PDFs,
  poster images, or unrelated media families are refreshed by this recovery.
- If visual/OCR evidence is in scope and cached item pages reveal SlidesLive
  embeds, production must treat slide images as a separate acquisition lane.
  Use `acquisition.slideslive_slide_download`, then
  `acquisition.slide_reference_detection` and `acquisition.slide_cleanup`
  before visual OCR/indexing; audio-only recovery, paper PDFs, and poster
  thumbnails do not satisfy SlidesLive slide acquisition.
- Transcription means speech-to-text or OCR over downloaded audio/video/slides
  or images. Local PDF text extraction is a useful `paper_text_extraction`
  artifact, but it is not proof of session/media transcription.
- Analysis completion requires the requested final evidence class. If the
  production brief calls for video/session insight, generated reports must cite
  transcript, OCR, or media segment evidence, not just paper titles, abstracts,
  metadata, or PDF text.
- Downstream podcasts, decks, executive summaries, releases, and final proof
  inherit upstream incompleteness. A rendered audio file or deck may exist, but
  it cannot close production if it was generated from incomplete or
  metadata-only analysis.

Every generated production plan must include explicit assertions for these
gates when media or transcript-backed analysis is in scope:

- count of downloaded video/audio/session media artifacts is greater than zero,
  or every expected media source has an unreachable/blocker receipt
- count of transcript/OCR artifacts for in-scope media is greater than zero
- provider/local transcription run receipts record execution mode, cost/rate
  limits, retry state, and credential safety
- provider approval cites the full-pipeline cost estimate command and scenario,
  not only ASR/OCR subtotals
- analysis reports reference final transcript/OCR/media evidence, or the phase
  remains incomplete with a blocker; public reports must be narrative-shaped
  and raw citation ledgers must be in provenance sidecars
- final support labels contain no `live` promotion for media download,
  transcription, transcript-backed analysis, release, or proof unless those
  exact gates passed

## Content Fidelity Gates

Production plans must also fail closed on the ICLR-style content-quality
failure modes:

- extracted_content.json is built from exhaustive source reading over final
  downloaded, transcribed, OCR'd, indexed, and gap-filled assets as described in
  `docs/extraction/exhaustive-extraction-guide.md`. Selected-corpus metadata,
  paper titles, abstracts, or PDF-only summaries cannot stand in for exhaustive
  extraction unless the plan records an explicit narrowed-scope waiver.
- Slide/image evidence for slide, poster, image, or video-derived assets is
  classified as one of three states based on a source-coverage assessment:
  `required` (corpus lacks sufficient text-bearing artifacts or the brief
  requires slide/visual evidence), `skipped_with_rationale` (available text
  sources provide sufficient coverage and a receipt-backed rationale is
  recorded), or `deferred_blocked` (OCR decision requires upstream information
  not yet available). Production plans must not impose unconditional OCR without
  a source-coverage assessment.
- Slides marked `FIGURE_NEED` require actual embedded programmatic SVG figures
  generated with matplotlib, Plotly, or svgwrite, or explicit per-slide waivers.
  A programmatic matplotlib/Plotly/svgwrite owner command remains mandatory:
  `scripts/presentations/generate_programmatic_figures.py`.
  The generation owner must write source-grounded `FigureSpec` records and
  record Anthropic `claude-sonnet-5` as the governed
  `presentation.svg_figure_generation` code/spec generation model. Live
  production plans must include `--live-sonnet-specs`; deterministic local
  fallback may be used only for dry-run, repair scaffolding, or explicit
  operator-waived cost control.
  When a live run has already produced `figure_specs.json` plus
  `sonnet_figure_spec_receipt.json`, renderer-only repairs may reuse those specs
  with `--figure-specs-input` and must preserve the live receipt, mark
  `live_sonnet_specs_reused_local_render`, clean stale SVGs, and rerun both SVG
  validation commands.
  Programmatic output is necessary but not sufficient: figures must be
  content-aware, slide-specific, and editorial-quality, with template selection
  tied to slide intent (matrix, flow, benchmark curve, radar, memory breakdown,
  attention map, etc.) and rendered inspection that rejects repeated boilerplate
  or generic scaffold diagrams. Paper-extracted/cropped figures may supplement
  a slide, but they must not satisfy `presentation.svg_figure_generation`.
  Generated plans must name the programmatic matplotlib/Plotly/svgwrite owner
  command. Zero SVG output, zero figure output, or generic placeholder SVGs are
  failed validation results.
- Generated SVGs must run
  `.venv/bin/python scripts/presentations/validate_svgs.py --workspace <workspace> --presentation-id <id> --dir decks/<id>/assets/generated --report-output decks/<id>/assets/generated/validation_report.md`
  and
  `.venv/bin/python scripts/presentations/review_programmatic_figures.py --workspace <workspace> --presentation-id <id> --presentation-type <type> --topic <topic> --dir decks/<id>/assets/generated --asset-inventory decks/<id>/asset_inventory.json --report-output decks/<id>/assets/generated/semantic_review.json --markdown-output decks/<id>/assets/generated/semantic_review.md`
  before render/QC can be accepted. Semantic review blocks copied bullets,
  generic template phrases, density fallback, placeholder chart ticks, and
  missing required FigureSpec entities.
- Visual QC requires two owners: `scripts/presentations/inspect_rendered_slides.py`
  for Playwright screenshots/heuristics and
  `scripts/presentations/inspect_rendered_slides_llm.py` for governed
  vision-LLM review. The LLM report path is
  `assets/generated/visual_llm_qc_report.json`, uses task
  `presentation.visual_qc` with default `openai:gpt-5.6-luna` and fallback
  `openai:gpt-5.6-luna`, and fails closed on missing provider/model, missing report,
  unparseable JSON, or blocker findings.
- LLM-feedback visual repair requires
  `scripts/presentations/run_presentation_repair_loop.py` for
  `presentation.repair` whenever `visual_llm_qc_report.json` has blocker
  findings. The repair owner must normalize findings through
  `presentation_slide_mapping.json`, write
  `assets/generated/visual_repair_feedback.json`, rerun
  `scripts/presentations/generate_programmatic_figures.py --repair-feedback`,
  rebuild `presentation.html`, rerun rendered screenshot QC, and then run
  `.venv/bin/python scripts/presentations/inspect_rendered_slides_llm.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --screenshots-dir decks/<id>/assets/generated/qc/screenshots --report-output decks/<id>/assets/generated/visual_llm_qc_report.after_repair.json`.
  `validate_svgs.py --fix` is only a
  mechanical helper and cannot complete LLM visual repair by itself.
- Deck/presentation production must carry configured density targets from the
  analysis handoff into `workflow.deck` and `presentation_manifest.json`.
  Defaults are `target_hyperlinks_per_slide: 2.0` and
  `target_svgs_per_slide: 0.70`; validation must run
  `source_binding_density_meets_target`,
  `visible_reference_coverage_meets_target`, and `svg_density_meets_target`.
  Source density is measured from `source_binding_report.json`; old numbered
  Markdown files are legacy/debug exports and cannot satisfy production proof.
- Conference-level executive overview decks are required whenever the brief or
  target outputs request executive summary presentations. Production must run
  `scripts/presentations/generate_executive_overview_decks.py --workspace <workspace> --analysis-manifest analysis/analysis_manifest.json --executive-summary analysis/executive_summary.md --trend-report analysis/trend_report.md --receipt-output decks/executive_overview_decks_receipt.json`,
  then register `executive_summary_short` and `executive_summary_long` in
  `decks/presentation_manifest.json` with
  `presentation_type: executive-summary`. Both must pass branded Reveal.js
  render, PDF export, and rendered-slide QC before presentation closeout, video
  generation, release, or proof. A topic-only manifest cannot close production
  when the overview decks are in scope. The executive decks must follow topic
  bookend semantics: slide 2 is `slide_kind: events`, with no more than `x`
  important events where `x` is the number of executive synthesis points and at
  least `x` key resources on the right; the last slide is `slide_kind:
  resources` and links the 12 topic presentations plus the 12 topic analysis
  reports. Production must pass
  `--source-link-map decks/executive_source_link_map.json --require-source-link-map`
  and, once topic decks have been uploaded or published,
  `--presentation-link-map decks/executive_presentation_link_map.json
  --require-presentation-link-map`; workspace-local topic-presentation links are
  review-only evidence and cannot satisfy the final presentation link audit.
  Production executive overview decks may use up to 24 candidate conference
  lessons during selection, roughly two per topic when all 12 topics are in
  scope, but the produced structure is operator-locked: a short deck with
  around 8 executive points and about 15 minutes of narration, and a long deck
  with the same 8-point spine plus 3 finer-grained expansions, around 11 points
  total, and about 30 minutes of narration. Each selected point must be backed
  by transcript quotes or tightly grounded transcript paraphrases. Slide 2 and
  the final reading slide are support bookends; a repeated beginning/end
  resource index is not production-quality synthesis. The generator must emit
  duration proof such as `duration_contract.json` and matching delivery totals;
  slide count alone cannot satisfy the 15-minute short or 30-minute long
  requirement, and neither can a timing JSON that merely sums to the target.
  The build must emit `narration_duration_verification.json` from the actual
  script text and a sampled TTS/audio extrapolation receipt; if extrapolated
  duration falls below target, redesign and rebuild narration before export.
  Executive decks must include at least eight GenAI-generated raster images per
  deck via `premium_image_receipt.json`, with source-bound prompts, provenance,
  and Gemini/GPT-5/Claude/Qwen review requirements. Produced executive decks
  must be screenshot/contact-sheet reviewed so body slides prove visible
  lesson/evidence/source/decision synthesis instead of generic bullet-card
  visuals.
- Structured source binding is mandatory for topic-deep-dive decks.
  Production plans must run `presentation.source_binding` after
  `presentation_deck.json` exists. Required owners are
  `scripts/presentations/build_deck_source_pack.py` and
  `scripts/presentations/bind_presentation_sources.py`; completion evidence is
  `source_pack_receipt.json`, `ranked_papers.json`, and
  `source_binding_report.json`. Unresolved placeholders, empty URL evidence, or
  missing author/speaker/system references block the production deck.
- Reveal.js render must consume `render_spec.json` with title, author, and
  report-link metadata, then run `scripts/presentations/md_to_revealjs.py
  --render-spec decks/<id>/render_spec.json --require-brand-assets` with the
  exact RelationalAI NeurIPS 2025 brand reference from
  `assets/brand/relationalai/`. Optional PPTX templates may be supplied only if
  they expose the same logo, bug, and title artwork. Missing title/author/report
  metadata, generic/approximate themes, or unbranded renders are production
  blockers.
- Presenter-script generation must run
  `scripts/presentations/generate_presenter_script.py` with
  `--render-spec decks/<id>/render_spec.json`,
  `--slide-mapping decks/<id>/presentation_slide_mapping.json`,
  `--conference-label <conference>`,
  `--delivery-single-speaker decks/<id>/delivery_single_speaker.json`, and
  `--delivery-notebooklm decks/<id>/delivery_notebooklm.json`
  after the structured render path is current. Markdown narration exports are
  debug-only and cannot satisfy delivery proof. Video generation must run both
  `scripts/presentations/build_presenter_video.py` and
  `scripts/presentations/build_notebooklm_style_presenter_video.py` with
  `--slide-mapping decks/<id>/presentation_slide_mapping.json`; the NotebookLM
  builder must also pass `--conference-label <conference>` so generated dialogue
  cannot leak a prior conference name. These are required owners because they
  block stale presenter scripts that omit mapped Slide `0` or any later mapped
  slide.
- PDF export must run
  `scripts/presentations/export_revealjs_pdf.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --input-html decks/<id>/presentation.html --output-pdf decks/<id>/presentation.pdf`.
  Manual browser print-to-PDF, ad hoc Chrome commands, or screenshot-based
  exports cannot satisfy `presentation.render_export` without an explicit
  registry-approved waiver.
- After HTML/PDF export, production must run
  `scripts/presentations/audit_presentation_links.py --workspace <workspace>
  --decks-dir decks --conference-slug <slug> --receipt-output
  decks/presentation_link_audit_receipt.json`. The receipt must pass before
  closeout and must reject `file:///`, `/Users/...`, repo-local, and
  `../analysis/...` links. It must enumerate expected public links from slide-2
  events/papers, final-slide resources, content-slide source chips, and
  executive summary decks; verify matching HTML hrefs and PDF `/URI`
  annotations; and report `unresolved_visible_refs`, `bad_local_hrefs`,
  `missing_html`, and `missing_pdf` counts. If the audit script cannot prove
  those dimensions for the current deck format, production must update the
  resolver/audit code and rerun export rather than waive the check. Executive
  overview deck generation must pass
  `--source-link-map decks/executive_source_link_map.json
  --require-source-link-map`; missing Google Drive sync/upload is an explicit
  blocker, not permission to emit local analysis markdown links.
- SVG assets render, are inspected, repaired, and re-rendered before deck
  completion; presentation structural validation alone is insufficient.
- Podcast script generation must use
  `.venv/bin/python scripts/generate_focused_podcast.py analysis/<id>.md --fact-ranks audio/<id>/ranks.json --output-dir audio/<id>/focused_podcast`
  or declare a blocker if the manifest lacks `fact_ranks_path`.
  Podcast TTS must then use
  `.venv/bin/python scripts/render_podcast_audio.py --script-json audio/<id>/focused_podcast/script.json --output-dir audio/<id>/focused_podcast --model gpt-4o-mini-tts`
  and must produce `audio/<id>/focused_podcast/podcast.mp3`,
  `audio_manifest.json`, and segment-level provider receipts
  (`docs/reference/stage-owner-audit.md#podcast-render-audio`).
  Per-deck NotebookLM podcasts, when in scope, must use
  `.venv/bin/python scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> --live --ack-live-notebooklm --format deep-dive --length long`
  (or the conference runner's `--stage notebooklm-podcast`), produce
  `audio/<id>/notebooklm_podcast/podcast.m4a`, `notebook_manifest.json`,
  `prompt.txt`, and `show_notes.md` for every topic and executive deck, and
  record `notebooklm_podcast_aggregate_sha256`, `deck_success_count ==
  deck_scope_full_count`, `storage_state_present`, and `engine_version`
  (`docs/reference/stage-owner-audit.md#podcast-notebooklm-audio-overview`).
  Every notebook must carry an `Episode brief` run sheet that credits the
  deck creator, discloses from receipts how the material was harvested,
  read end to end by a small language model, and distilled, and names the
  topic's workshops; executive episodes must cover every topic with
  explicit transitions. Hosts must never claim to have attended. Each live
  episode must pass the local listening review (`review.json`, governance
  task `podcast.notebooklm_review_transcription`) or carry a recorded
  operator acceptance.
  Auth is `notebooklm login --browser-cookies chrome` on the operator's
  NotebookLM Pro account; `storage_state.json` is a secret and never enters
  artifacts or receipts. Quota halts (`notebooklm_quota_exhausted`) are
  resumable; `--deck-ids` canaries are always blocked receipts. The curated
  Drive folder is then written by
  `.venv/bin/python scripts/deliver_notebooklm_podcasts.py --workspace conferences/<slug> --run-id <run-id> --drive-root '<mounted-drive-root>/<Conference> Podcasts'`
  (`docs/reference/stage-owner-audit.md#podcast-notebooklm-delivery`).
  Conference-specific podcast scaffold writers, `write_podcast_package(...)`,
  and script-only `audio_ready_package.json` emitters are rejected as
  production completion owners.
- Manifest-backed deck contract generation must first run
  `presentation_blueprint` with
  `scripts/presentations/generate_editorial_blueprint.py --presentation-frame decks/<id>/presentation_frame.json --final-report-url <url> --provider-cost-approved`
  and `scripts/presentations/review_presentation_blueprint.py`, repair blocked
  reviews with `scripts/presentations/repair_presentation_blueprint.py`, require
  passing `presentation_frame_validation`, and
  then use
  `.venv/bin/python scripts/presentations/generate_deck_contract.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_deck.json --receipt-output decks/<id>/presentation_deck_receipt.json`.
  `render_slides(...)`, conference-specific scaffold writers,
  `scripts/presentations/generate_slides_simple.py`, and
  `scripts/generate_slides_agentic.py` are rejected as completion owners for
  manifest-backed deck contract generation.
- Deck/presentation phases must include a narrative-content gate that checks
  slide-brief/deep-analysis digest binding, rejects stale scaffold/public-ledger
  language, and fails identical deck checkpoint files without a valid waiver.
- Deprecated paper-figure presentation scripts
  `scripts/presentations/extract_topic_figures.py` and
  `scripts/presentations/add_images_to_md.py` are rejected as production owner
  commands. Generated plans that route figure generation through either script
  must fail audit.
- Source traceability IDs remain unredacted for artifact IDs and local paths
  while secrets, cookies, authorization headers, and base64 payloads stay
  redacted. Over-redacted media IDs are blockers because they break provenance.

## Execution Plan Requirements

The generated phase or milestone must include:

- exact command line(s) to run, with `--workspace`, `--run-id`, stage selectors,
  and resume behavior
- no silent item-count caps on live source discovery, corpus freeze, or corpus
  download commands; intentional exclusions must be explicit, receipt-backed,
  and recorded before expensive work
- for large provider-backed transcription/OCR stages, include explicit
  resumable tranche controls when the runner supports them, such as
  `--max-media-items`, `--media-kind audio|image|pdf`, and
  `--max-pdf-pages-per-item`
- a preferred single runner command when appropriate, for example
  `.venv/bin/python scripts/run_<slug>_<version>_production.py ...`
- implementation tasks for the runner if no existing command executes all
  in-scope stages
- a command-fidelity inventory that maps each governed task id, especially
  `presentation.deck_contract_generation` and `presentation.svg_figure_generation`, to
  an existing executable owner, implementation task, or explicit blocker before
  execution. Presentation deck work requires the manifest-backed
  `generate_deck_contract.py` owner and figure work requires a programmatic
  matplotlib/Plotly/svgwrite owner command with content-aware editorial SVG
  quality gates before either can satisfy production completion.
- per-stage receipt paths under `conferences/<slug>/storage/workflow-runs/`
- manifest updates and digest/provenance requirements
- environment variables and provider credentials required at runtime, without
  copying secret values into durable files
- cost/rate limits, retry rules, resume/idempotency behavior, and failure modes
- the full-pipeline cost command:
  `.venv/bin/python -m conference_pipeline.costs estimate --workspace <workspace> --scope full --scenario base --json`
- the OCR/image parsing benchmark command when OCR model selection is not final
  or the user asks for a quality/cost/speed experiment:
  `.venv/bin/python -m conference_pipeline.ocr_benchmark run --workspace <workspace> --sample-size 12 --budget-cap-usd 30 --per-call-timeout-seconds 60 --json`
- the post-run actual reconciliation command:
  `.venv/bin/python -m conference_pipeline.costs actual --workspace <workspace> --scope full --json`
- validation commands and pass/fail criteria
- extracted-content provenance checks, slide/visual OCR checks, figure/SVG
  render-inspection repair loops, vision-LLM visual QC, and source traceability
  ID checks
- stage-to-requirement traceability
- done conditions that distinguish final output from interim artifacts

## Failure And Recovery Requirements

Generated phases must include explicit recovery policy:

- Fail-stop blockers: any required stage result with `status: blocked`,
  `status: deferred_blocked`, `status: failed`, `status: unavailable`, or
  `status: unsupported` stops the whole pipeline before downstream podcast,
  deck, release, proof, or publication work. The runtime must write a
  `BlockerReceipt` with the applicable class: `ocr_deferred_blocked`,
  `stage_blocked`, `stage_unavailable`, `local_llm_not_ready`,
  `missing_receipt`, `threshold_violation`, or `stale_receipt`.
- Resume/unblock: active blockers prevent resume until an `UnblockReceipt` is
  written through `workflow unblock` or `conference_pipeline.workflow_unblock`;
  summaries must expose `halt_status`, `halt_stage_id`,
  `blocker_receipt_path`, and `unblock_acknowledgements`.
- Network/timeouts/rate limits: retry with bounded exponential backoff, persist
  partial receipts, then resume from compatible succeeded receipts.
- Batch caps: record the first next-item blocker, keep scanning compatible
  existing transcript/OCR outputs for receipt reconciliation, and separate new
  provider calls from represented provider calls.
- Auth/cookies/permissions: attempt documented recovery once when safe, then
  stop and ask the user for credentials or source guidance.
- Provider/cost gates: require a full-pipeline provider estimate and live
  provider approval before transcription, OCR, LLM, TTS, or
  presentation-provider calls. Estimates are approval forecasts; provider
  receipts/invoices remain actual-cost truth.
- Model-governance gates: every provider-backed task cites a registry task id
  from `src/conference_pipeline/model_governance/registry.yaml`; current model
  research and benchmark receipts are staged where realistic inputs exist. If
  inputs or receipts are missing, record an explicit blocker with the next
  action instead of a placeholder receipt.
- OCR model gates: if a benchmark is requested, dry-run it first, then run live
  only with an explicit budget cap and receipt path. Benchmark receipts compare
  cost, latency, JSON validity, schema completeness, locator/numeric recall,
  and similarity against the authority parse.
- Disk/storage: run preflight before bulk downloads; block and ask before
  deleting or pruning.
- QA/manifest failures: fail closed, write failed receipts and summary, and
  require correction before podcast, deck, release, or proof.

## Completion Rules

Completion requires actual receipt-backed execution for each in-scope stage.

Do not mark a production phase or milestone complete using:

- dry-run/mock receipts
- readiness files without execution receipts
- audio-ready scripts when rendered audio is required
- scaffolded decks when rendered/exported presentations are required
- `planned`, `unsupported`, or `needs live proof` labels for in-scope
  deliverables

If a stage cannot run, keep the phase or milestone incomplete and record the
blocker, owner command needed, and next action.

### Conference-portability gate (required before milestone close)

Before a production phase or milestone is marked complete, run the
`conference-hardcoding-audit` skill: `python scripts/audit_conference_hardcoding.py`
must exit 0 (no hardcoded conference names, years, program domains, or topic
taxonomies in the reusable library), and the skill's manual `scripts/` sweep must
find no unresolved violations. Reusable code must read conference identity from
`config.yaml` (via `conference_pipeline.conference_identity`),
`analysis_manifest.json`, and `presentation_frame.json` — never from literals.
Record the audit result in the milestone completion notes; a failing audit blocks
completion.

### Model-registry gate (required before milestone close, independent of the above)

Before a production phase or milestone touching model selection is marked
complete, also run the `model-registry-audit` skill:
`python scripts/audit_model_registry_hardcoding.py` must exit 0 (no hardcoded
LLM model-name literals in `src/conference_pipeline/**`, `scripts/**`, or
`.claude/skills/**` command examples). Code and skills must reference a
governed task's model by task id through
`conference_pipeline.model_governance.registry`'s `default_model` /
`default_model_for_provider`, never a literal. This is a second, independent
gate alongside `conference-hardcoding-audit`, not a replacement for it. Record
the audit result in the milestone completion notes; a failing audit blocks
completion.

### Corpus source-metadata gate (required before milestone close, independent of the above)

Every generated production plan must include the `corpus source-metadata
enrichment` stage and its completion gate, `PG-CORPUS-SOURCE-METADATA`. Its
live command is `python scripts/audit_corpus_source_metadata.py --workspace
conferences/<slug> --json --write-receipt storage/workflow-runs/<run-id>`,
which writes the coverage receipt `corpus_source_metadata_coverage.json`. See
the `corpus-metadata-audit` skill
(`.claude/skills/corpus-metadata-audit/SKILL.md`) for how the gate is run and
interpreted — per-family (`papers_main`, `papers_workshop`, `artifacts`)
coverage floors, the `artifacts` family's `structurally_authorless` posture,
and the no-silent-lowering `floor_override_unjustified` refusal path.

Be explicit about what this SKILL.md's own audit can and cannot prove.
`audit_generated_production_plan.py` (and `plan-conference-analysis`'s
`audit_generated_plan.py`) check that a GENERATED PLAN'S TEXT declares this
stage and gate. They do not and cannot verify that a future conference's live
run actually produced real coverage — that stronger, runtime-scanning
guarantee comes only from running `corpus-metadata-audit` against real
on-disk artifacts, and only the operator running it can produce that
evidence. See
`.planning/workstreams/icml-2026/audits/skill-enforcement-review-20260713.md`
for the empirical record of `audit_generated_production_plan.py` reporting a
full pass (`status "passed", 81/81 assertions, 0 failures`) against the exact
ICML production draft whose live execution failed for reasons the plan's own
text correctly forbade. A plan-text pass from this skill's own audit is never
equivalent to a passing `corpus-metadata-audit` run; do not treat the weaker
gate as the stronger one.

Add `python scripts/audit_corpus_source_metadata.py --workspace
conferences/<slug>` to the milestone completion notes alongside (not instead
of) `conference-hardcoding-audit` and `model-registry-audit`; a failing audit
blocks completion the same as the other two.

## Source-Of-Truth Rules

- Root docs stay platform-general.
- Conference-specific execution assumptions, source URLs, proof status,
  limitations, and support labels belong in `.planning/milestones/*-CONTEXT.md`
  and the conference workspace.
- Update `docs/reference/pipeline-capabilities.md` only when reusable platform
  behavior actually changes.
- Workspace READMEs should point to planning-owned truth rather than owning it.

## Install Targets

Canonical repo source:

- `.claude/skills/plan-conference-production`

User sync targets:

- `~/.codex/skills/plan-conference-production`
- `~/.claude/skills/plan-conference-production`

Use `scripts/install_skill.py` to sync. Do not copy secrets, generated
conference artifacts, caches, or provider output into user skill directories.

## Approved-Local Proof Gate (Phase 10+)

Production planning checks each task's `local_proof` block in the registry via
`approved_local_selection()`. A task may use local execution only when all four
proof classes are present and the referenced benchmark receipt validates as a
live, local, MLX, passed receipt for the same task and model:

1. `research_ref` — Phase 8 research evidence artifact
2. `hardware_fit_ref` — Phase 7/9 target-Mac hardware-fit proof artifact
3. `benchmark_receipt_ref` — Phase 9+ benchmark receipt with
   `selection_decision=passed`
4. `operator_approval_ref` — Phase 10 operator approval record

Missing any proof class or receipt validation: task remains on the
cloud/default path and a `local-proof-missing-*` assertion is emitted.
Generated audit requires each local-provider task to be covered by either an
approved-local proof snapshot or a local deferral, and rejects overlap between
the two with `local-proof-disagree-fail-closed`.
Raw local model literal in plan: `local-raw-model-bypass` assertion emitted.

## UAT Proof

Use `scripts/scaffold_production_plan.py` to draft live production artifacts
from an analysis spec and `scripts/audit_generated_production_plan.py` to verify
full pipeline coverage, live-only completion gates, recovery behavior,
media/transcription truth gates, content-fidelity gates, and secret-free
generated outputs.
