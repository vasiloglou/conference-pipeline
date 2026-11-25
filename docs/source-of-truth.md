# Source Of Truth Contract

This repository is one platform with many conference workspaces. Keep durable
platform rules in root docs and keep conference-local context in the conference
workspace.

## Canonical Layers

| Layer | Source of truth | Belongs here | Does not belong here |
|-------|-----------------|--------------|----------------------|
| Agent entrypoints | `CLAUDE.md`, `AGENTS.md` | Universal repo orientation, safety rules, links to canonical docs | Conference topic inventories, one-off analysis narratives |
| Platform docs | `docs/` | Architecture, CLI, workflow contracts, operation runbooks, schemas, reusable capability docs | Local source URLs, conference-specific credentials, conference proof notes |
| Code capability catalog | `docs/reference/pipeline-capabilities.md` | Reusable capabilities grounded in actual modules/scripts | Aspirational features without implementation owners |
| Workflow playbook | `docs/workflows/conference-analysis-playbook.md` | Reusable conference analysis lifecycle | A single conference's complete topic findings |
| Conference milestone context | `.planning/milestones/<version>-<Conference>-CONTEXT.md` | Milestone identity, local assumptions, source URLs, proof status, canonical local docs | Shared platform policy that should apply to every conference |
| Generated analysis plans | `conferences/<slug>/analysis/` | Conference-specific analysis plan, topic docs, manifests, QA reports | Root agent rules or platform capability definitions |
| Script/API docs | `docs/api/`, `docs/cli.md`, module docstrings | Stable interfaces, command contracts, receipt paths | Manual notes that drift from command behavior |
| Extraction provider policy | `docs/reference/pipeline-capabilities.md` | Reusable extraction provider policy (local LLM default, cloud escalation, conditional OCR) | Conference-specific source URLs, credentials, or proof claims |
| Local LLM backend selection | `src/conference_pipeline/extractions/backends.py`, `docs/reference/pipeline-capabilities.md` | Runtime backend selection, MLX-vs-HTTP rules, endpoint-label contract | One conference's proof packet or operator note |
| Quality-sample thresholds | `src/conference_pipeline/extractions/thresholds.py`, `docs/reference/pipeline-capabilities.md` | Canonical defaults, override path, receipt contract | Untracked threshold tweaks in workspace notes |
| Fail-stop blocker policy | `src/conference_pipeline/provider_receipts.py`, `src/conference_pipeline/workflow_runs.py`, `src/conference_pipeline/workflow_unblock.py` | Blocker/unblock receipt contract, halt/resume summary fields, operator acknowledgement path | Conference-local proof status or blocker history outside the context packet |
| Local LLM and embedding lane (v1.9) | [`docs/local-llm.md`](local-llm.md) | Operator reference for MLX / Ollama embeddings + `LocalLLMResearcher`, env var contract, dim migration policy, `vector-store rebuild` CLI, `conference-smoketest --local` smoke contract, `manage_server=True` lifecycle | Conference-specific local-model selections; one-off operator log notes |
| Local visual LLM and OCR lane (v1.61) | [`docs/local-llm.md`](local-llm.md) (Vision & Slide Analysis section), `src/conference_pipeline/slide_analysis/`, `src/conference_pipeline/model_governance/mlx_vision_client.py`, `src/conference_pipeline/model_governance/fastvlm_native_client.py` | `SlideAnalysisEngine` + `LLMOcrEngine(provider='local')` contract, `LocalVisionCallReceipt` schema, `vllm-mlx` server adapter and FastVLM-MLX native adapter, registry catalog and promotion CLI. Status: `local-supported, benchmark_pending` until per-task receipts promote a candidate via `promote-vision-candidate`. | Conference-local slide fixtures, operator-specific UAT receipts, hardware-specific UM-floor notes |
| Presentation visual QC (v1.63) | [`docs/operations/presentation-contract.md`](operations/presentation-contract.md), `src/conference_pipeline/presentation/vision_qc.py`, `scripts/presentations/inspect_rendered_slides_llm.py`, `src/conference_pipeline/stage_owner_registry.json` | Governed `presentation.visual_qc` vision-LLM review over rendered screenshots, `visual_llm_qc_report.json` schema, fail-closed validation behavior, and approved owner-command registry binding. | Conference-local waived slide issues, one-off subjective review notes not backed by the report |
| Presentation audience framing contract | [`docs/operations/presentation-contract.md`](operations/presentation-contract.md), [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md), `src/conference_pipeline/presentation/presentation_frame.py`, `scripts/presentations/generate_editorial_blueprint.py`, `src/conference_pipeline/stage_owner_registry.json` | `presentation_frame.json` as the post-analysis, pre-blueprint handoff that separates retrieval topic titles from audience-facing deck titles, requires a field-level thesis and broad opening questions, and records `presentation_frame_validation` before deck-contract generation. | Treating topic strategy words as the public title, starting topic decks with narrow paper mechanisms before field-level questions, or accepting missing/mismatched presentation frames as production evidence |
| Token-bounded vector-store chunking and embeddings | [`docs/reference/pipeline-capabilities.md`](reference/pipeline-capabilities.md), [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), `src/conference_pipeline/indexing/token_policy.py`, `src/conference_pipeline/indexing/chunking.py`, `src/conference_pipeline/indexing/embeddings.py`, `src/conference_pipeline/indexing/mlx_embeddings.py` | Reusable source-aware token-bounded chunk builders for papers, PDFs/text, transcripts, slides/OCR/images, generic artifacts; `PG-CORPUS-TOKEN-INDEXING`; `token_budget_policy_digest`; `embedding_policy_digest`; `EMBEDDING_MAX_LENGTH`; `CORPUS_VECTOR_EMBED_BATCH_SIZE`; production rejection of snippet-only/first-240-character vectorization, word-count-only sidecars, readiness-only probes, and partial Chroma rows | Per-conference ad hoc chunking rules, one-off snippet/vector-store exceptions, or treating `scripts/setup_vector_store_enhanced.py` as a future-conference owner command |
| Talk-aware workshop transcripts | [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md), [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), `src/conference_pipeline/acquisition/embedded_media.py`, `src/conference_pipeline/workshop_talk_segmentation.py`, `scripts/discover_workshop_talks.py`, `scripts/segment_workshop_talks.py` | When workshops are in scope, production must first prove workshop talk-media acquisition with `storage/workshops/talk_media_acquisition_receipt.json` or a formal `storage/workshops/workshop_talk_media_scope_waiver.json`; `workshop_speech_count: 0` is inventory evidence only, not a waiver. Workshop talk acquisition must use deterministic public-first/authenticated retry and reusable embedded-player recovery (`acquisition.embedded_media_recovery`) before declaring media inaccessible. `transcription_ocr.workshop_talk_boundary_discovery` uses Gemini 2.5 Pro to write `*_talks.json`, 5-minute talk-context batches, `provider_receipt_paths`, and `cost_task_id=workshop_talk_boundary_discovery` when no vetted talk boundaries exist; `transcription_ocr.workshop_talk_segmentation` then annotates transcript segments with talk metadata (`workshop_id`, `workshop_title`, `talk_id`, `talk_title`, `talk_speaker`). The blocker code remains `talk_segmentation_provider_not_configured` when neither `talks.json` nor governed provider discovery exists. audio-only recovery is rejected evidence for talk-aware transcripts. | Treating workshop paper acquisition as workshop talk-media acquisition, treating zero workshop speech as proof workshop talks were out of scope, treating workshop audio recovery alone as talk segmentation, dropping talk metadata during indexing, silently merging whole-workshop transcripts without a segmentation/discovery receipt, direct-link-only workshop crawling, or using historical `scripts/exhaustive_extraction` as production evidence |
| GPT-Researcher hybrid retrieval and parameters | [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), [`docs/integrations/gpt-researcher/quickstart.md`](integrations/gpt-researcher/quickstart.md), `src/conference_pipeline/research/config.py`, `src/conference_pipeline/research/researcher.py`, `src/conference_pipeline/research/heuristic_prefilter.py`, `src/conference_pipeline/stage_owner_registry.json` | Default BM25+vector retrieval for `query-planning-gpt-researcher`, vector-only override semantics, GPT-Researcher-style knobs, behavioral/full parameter digests, provider/model provenance, and the dual deep-dive contract: global no-topic-id retrieval plus `analysis.gpt_researcher_heuristic_prefilter` keyword packs with `prefilter_terms_sha256` and `selected_evidence_ids` | One-off parameter-sweep outcomes, conference-local chosen local model ids, topic-id filtered deep dives, unrecorded keyword lists, or replacing provider defaults without receipts |
| Topic reconciliation and reference validation | [`docs/operations/analysis-contract.md`](operations/analysis-contract.md), [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md), `src/conference_pipeline/analysis/topic_reconciliation_reference_workflow.py`, `src/conference_pipeline/analysis/topic_reconciliation.py`, `src/conference_pipeline/analysis/topic_reference_validation.py`, `src/conference_pipeline/stage_owner_registry.json` | Packaged 07H workflow after exhaustive summaries and dual deep-dive lanes; local Qwen3.6 topic reconciliation; deterministic per-topic reference ledgers; deterministic safety fixes; bounded candidate-resolution repair before broad repair; `07D-topic-reconciliation.json`; final `07E-topic-reference-validation.json`; `07H-topic-reconciliation-reference-workflow.json`; rejection of unknown ids, title/id conflations, and unvalidated reconciliation before final narrative or slide synthesis | One-off comparison reports, hand-edited local-model briefs without reference validation, manual 07D/07E/07G/07F ordering without a 07H receipt, or mutable topic-id filters |
| SlidesLive slide acquisition and cleanup | [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md), `src/conference_pipeline/slideslive_slides.py`, `src/conference_pipeline/slide_cleanup.py`, `src/conference_pipeline/stage_owner_registry.json` | Cached-item-page SlidesLive slide-image recovery, slide sync metadata, reference/banner detection, duplicate cleanup, non-destructive `skip_ocr` markers, and registry-owned planning gates before visual OCR/indexing | Conference-local slide waivers, physical deletion as cleanup evidence, treating audio-only recovery as slide acquisition, or using poster thumbnails as SlidesLive slide proof |
| Dual-store paper matching (v1.91) | [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), [`docs/cli.md`](cli.md), `src/conference_pipeline/paper_matching/` | Reusable dual vector-store contract, `paper-matching` workflow stage, `workflow.paper_matching` config, artifact-keypoint-to-paper-store matching flow, canonical `matched_papers.json` output, `rank_papers.py` deprecation | One conference's selected paper list, topic-specific paper rationale, or private/source-gated paper URLs |
| Deep narrative analysis contract (v1.63) | [`docs/operations/analysis-contract.md`](operations/analysis-contract.md), [`docs/workflows/pipeline-workflow.md`](workflows/pipeline-workflow.md), `src/conference_pipeline/analysis/narrative.py`, `src/conference_pipeline/model_governance/registry.yaml` | Platform-wide deep report shape, slide-brief sidecars, manifest fields, `analysis.deep_narrative_synthesis` model governance, validation gates, production fail-closed behavior | One conference's topic conclusions, local rebuild receipts, or conference-specific provider override notes |
| Unified stage script registry and Stage owner registry audit | [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md), `src/conference_pipeline/stage_owner_registry.json`, `src/conference_pipeline/stage_script_audit.py`, `scripts/verify_stage_script_usage.py` | Canonical required/allowed/deprecated/forbidden script registry, registry-owned `usage_doc_refs` and `diagnostic_profile` messages for every tool, planning-skill script contracts, post-planning verifier, run-receipt script usage proof, runner stage-owner categories | Conference-local scaffold exceptions without explicit blockers, undocumented tool instructions copied into skills, one-off proof claims that bypass the registry |
| Runtime stage contract gates | [`docs/reference/stage-owner-audit.md`](reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern), `src/conference_pipeline/stage_owner_registry.json`, `src/conference_pipeline/stage_contracts.py`, stage receipts under `storage/workflow-runs/<run-id>/` | Reusable rule that every production runner capable of writing a succeeded receipt must call `verify_stage_contract` and prove registry-required owner receipts, output families, freshness bindings, required checks, and blockers for the current workspace/run id before completion | Treating roadmap prose, skill prose, stale receipts, mock receipts, dry-run receipts, or unexecuted owner refs as production completion evidence |
| Manifest-driven workspace gitignore policy | [`docs/operations/future-conference-storage.md`](operations/future-conference-storage.md), `docs/storage_manifest.schema.json`, `scripts/generate_workspace_gitignore.py`, `conferences/<slug>/storage/storage_manifest.json` | Reusable rule that `storage_class` drives `conferences/<slug>/.gitignore`: `git` families stay trackable, human-reviewable Markdown deliverables remain Git-owned even when mirrored to Drive, and `public`/`backup`/`cache` payload families stay out of Git unless `gitignore_exceptions` declares a small control file | Hand-edited per-conference ignore rules, committing Google Drive/cache-owned payload artifacts, or broad root `.gitignore` lane policy |
| Continuous Drive artifact offload | [`docs/operations/private-backup.md`](operations/private-backup.md), [`docs/operations/future-conference-storage.md`](operations/future-conference-storage.md), `scripts/sync_to_gdrive.py`, `conferences/<slug>/storage/storage_manifest.json` | New conference Drive leaf setup with `--init-only`; manifest-bounded, rsync-incremental backup sync immediately after artifact-producing stages; receipts include manifest and artifact snapshot metadata | End-of-run-only artifact dumps, copying cache/unselected families, committing payloads to Git instead of offloading, or ad hoc Drive folders outside the manifest profile |
| Drive artifact symlink restore | [`docs/operations/drive-symlink-restore.md`](operations/drive-symlink-restore.md), `scripts/restore_drive_symlinks.py`, `conferences/<slug>/storage/storage_manifest.json` | Manifest-bounded symlink restore of public/backup Drive artifacts after local cleanup; receipt-backed verify run; skips git and cache families by storage_class; dry-run/apply/verify modes with exit-code contract | Copying payloads locally (use rsync for that), committing symlinks to git, pointing symlinks at cache/git-owned families, or treating a missing Drive mount as a safe no-op |

## Conference Context Packet Schema

Every conference milestone context must live under `.planning/milestones/` and
include these fields near the top:

```yaml
conference_slug: "<slug>"
conference_name: "<display name>"
platform_version: "vX.Y"
conference_milestone_name: "vX.Y-<Conference Name>"
proof_status: "complete|dry-run-proven|scaffolded|unproven"
planning_owner: ".planning/milestones/vX.Y-<Conference>-CONTEXT.md"
workspace: "conferences/<slug>"
```

The body must also include:

- local source URLs and auth notes
- analysis profile and topic strategy
- supported pipeline stages, separated by live, dry-run, and mock support
- canonical local docs and generated manifests
- proof receipts or explicit reason no proof exists yet
- known conference-specific decisions and limitations

Conference-specific milestone display names use
`{platform_version}-{conference name}`. Archived GSD filenames do not need to be
renamed when the display identity changes.

## Update Rules

- If a fact applies to all conferences, update `docs/` or root instructions.
- If a fact applies to one conference, update that conference's formal
  `.planning/milestones/*-CONTEXT.md` record. Workspace READMEs should only
  point to the planning-owned context.
- Provider credentials and API keys are a repository-wide local operator rule:
  keep them in the active worktree's repository-root `.env`, source that file
  before live provider runs, and never place secret values in
  docs, plans, receipts, or summaries.
- If a workflow behavior changes, update CLI/API docs and add or update a
  docs-contract test.
- If a proof is run, record the run id and report path in the conference
  context packet.
- If a proof is deferred, record the reason directly and point to the
  conference-local proof packet or deferred summary under
  `conferences/<slug>/storage/workflow-runs/`.
- If live support is missing, say so directly and prefer dry-run or mock proof
  language.

## Current Milestone Identities

| Workspace | Conference milestone | Status |
|-----------|----------------------|--------|
| `conferences/neurips2025` | `v1.0-NeurIPS 2025` | `.planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md` |
| Platform-wide workflow | `v1.1 Reusable Conference Production Pipeline` | Conference-neutral platform milestone |
| `conferences/iclr2025` | `v1.2-ICLR 2025` | `.planning/milestones/v1.2-ICLR-2025-CONTEXT.md` |
| `conferences/iclr2026` | `v2.0-ICLR 2026` | Analysis spec: `.planning/milestones/v2.0-ICLR 2026-CONTEXT.md` |
| `conferences/iclr2026` | `v2.1-ICLR 2026 Production` | Production context: `.planning/milestones/v2.1-ICLR 2026 Production-CONTEXT.md`; Topic 4 canary recovery promoted to v2.2 |
| `conferences/iclr2026` | `v2.2-ICLR 2026 Topic 4 Presentation Canary Recovery` | Canary recovery context: `.planning/milestones/v2.2-ICLR 2026 Topic 4 Presentation Canary Recovery-CONTEXT.md` |
| `conferences/iclr2026` | `v3.1-ICLR 2026 Public Publishing` | Active public publishing context: `.planning/milestones/v3.1-ICLR 2026 Public Publishing-CONTEXT.md` |
