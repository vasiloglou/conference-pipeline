# Pipeline Capabilities

This catalog describes capabilities that exist in the repository today. It is
grounded in code owners so operators can distinguish proven platform behavior
from conference-specific practice.

## Support Levels

| Level | Meaning |
|-------|---------|
| `conference-general` | Validates or runs against any workspace that satisfies the documented config/manifest contract. |
| `non-live-general` | Dry-run or mock receipts work across workspaces, but live execution may still be NeurIPS-bound. |
| `neurips-live-proven` | Live or historical proof exists for `conferences/neurips2025`. |
| `workspace-local` | Behavior depends on the selected conference workspace and local manifests. |

## Capability Catalog

| Capability | Support | Owners | Canonical docs |
|------------|---------|--------|----------------|
| Workspace resolution | `conference-general` | `src/conference_pipeline/workflow.py`, `conferences/README.md` | [Workflow API](../api/workflow.md), [Future Conference Storage](../operations/future-conference-storage.md) |
| Storage manifest governance | `conference-general` | `docs/storage_manifest.schema.json`, `scripts/generate_workspace_gitignore.py`, `scripts/inventory_workspace.py`, `scripts/sync_to_gdrive.py`, `scripts/prune_conference_workspace.py` | [Operations](../operations/index.md) |
| Drive artifact symlink restore | `conference-general`; dry-run/apply/verify modes with exit-code contract; skips `git` and `cache` storage-class families; receipt records per-family action counts and embedded verification block; ICLR 2026 live proof: 4,378 links checked, 0 unresolved, 0 git-control violations, exit 0 | `scripts/restore_drive_symlinks.py` | [Drive Symlink Restore](../operations/drive-symlink-restore.md) |
| Named workflow registry | `conference-general` for registry and validation; live support varies by stage | `src/conference_pipeline/workflow.py`, `src/conference_pipeline/cli.py` | [Workflow Runbook](../workflows/pipeline-workflow.md), [CLI](../cli.md) |
| Unified stage script registry and owner audit | `conference-general`; generated plans, planning skills, docs, and run receipts can fail closed when required scripts are missing, registry-owned `usage_doc_refs`/`diagnostic_profile` entries are stale, or deprecated owners claim completion | `src/conference_pipeline/stage_owner_registry.json`, `src/conference_pipeline/stage_script_audit.py`, `src/conference_pipeline/stage_owner_audit.py`, `scripts/verify_stage_script_usage.py` | [Stage Owner Audit](stage-owner-audit.md), [CLI](../cli.md) |
| Model-governance registry | `conference-general` planning and validation contract; live selections still require milestone receipts | `src/conference_pipeline/model_governance/registry.yaml`, `src/conference_pipeline/model_governance/registry.py` | [Model Governance Registry](model-governance.md), [Pipeline Workflow](../workflows/pipeline-workflow.md) |
| Mode-aware receipts and resume | `conference-general` | `src/conference_pipeline/workflow_runs.py` | [Workflow API](../api/workflow.md) |
| Dry-run and mock execution | `non-live-general` | `src/conference_pipeline/workflow_runs.py` | [CLI](../cli.md), [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Provider cost estimation | `conference-general` | `src/conference_pipeline/costs/`, `scripts/estimate_pipeline_costs.py` | [CLI](../cli.md), [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Receipt-bound public-media/material recovery and cumulative paid-call reservation | `conference-general` local contract; live use remains conference-plan-gated | `src/conference_pipeline/acquisition/youtube_audio_recovery.py`, `src/conference_pipeline/acquisition/public_material_acquisition.py`, `src/conference_pipeline/costs/estimator.py`, `src/conference_pipeline/costs/budget_enforcement.py` | [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Acquisition stages: discover, download, transcribe | `neurips-live-proven`; non-live receipts are general | `src/conference_pipeline/pipeline.py`, `src/conference_pipeline/workflow.py`, `scripts/discover_sessions.py`, `scripts/download_from_sessions.py`, `scripts/transcribe_sessions.py` | [Workflow Runbook](../workflows/pipeline-workflow.md) |
| SlidesLive slide acquisition and cleanup | `conference-general` for cached item-page probes; live support depends on the workspace carrying authenticated `item_page_auth_probe` artifacts | `src/conference_pipeline/slideslive_slides.py`, `src/conference_pipeline/slide_cleanup.py`, `scripts/recover_slideslive_slides.py`, `scripts/detect_reference_slides.py`, `scripts/cleanup_slides.py` | [Stage Owner Audit](stage-owner-audit.md), [Workflow Runbook](../workflows/pipeline-workflow.md), [CLI](../cli.md) |
| Analysis manifest contract | `conference-general` for manifest validation and backfill | `src/conference_pipeline/analysis/manifest.py`, `scripts/backfill_analysis_manifest.py` | [Analysis Contract](../operations/analysis-contract.md), [Manifest Schema](../design/analysis-manifest-schema.md) |
| Deep narrative analysis contract | `conference-general` when `analysis/narrative_source_packs/*.json` are present; production uses `analysis.deep_narrative_synthesis`, writes deep reports, provenance sidecars, and `analysis/slide_briefs/` for decks | `src/conference_pipeline/analysis/narrative.py`, `src/conference_pipeline/analysis/orchestrator.py` | [Analysis Contract](../operations/analysis-contract.md), [Pipeline Workflow](../workflows/pipeline-workflow.md) |
| Analyze stage orchestration | `neurips-live-proven`; non-live receipts are general | `src/conference_pipeline/analysis/orchestrator.py`, `src/conference_pipeline/analysis/steps/` | [Workflow API](../api/workflow.md) |
| Podcast command construction | `workspace-local`; manifest-backed command previews are general when declared ranks/audio paths exist | `src/conference_pipeline/workflow.py`, `scripts/generate_focused_podcast.py` | [Analysis Contract](../operations/analysis-contract.md) |
| Per-deck NotebookLM podcast production | `workspace-local`; one Google NotebookLM Audio Overview per topic and executive deck from the analysis spine plus blueprint-cited sources, manifest-bound and fail-closed; promoted only by a live conference receipt | `src/conference_pipeline/podcast_notebooklm.py`, `scripts/generate_notebooklm_podcast.py`, `scripts/deliver_notebooklm_podcasts.py` | [Stage Owner Audit](stage-owner-audit.md), [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Deck command construction | `non-live-general`; legacy deck command previews remain backward-compatible | `src/conference_pipeline/workflow.py`, `scripts/presentations/` | [Workflow API](../api/workflow.md) |
| Presentation stage registration and receipts | `workspace-local`; manifest-backed command previews, receipt scope, command metadata, and safe dry-run script contracts are available for declared presentation manifests | `src/conference_pipeline/workflow.py`, `src/conference_pipeline/workflow_runs.py`, `src/conference_pipeline/presentation/manifest.py` | [Workflow Runbook](../workflows/pipeline-workflow.md), [Workflow API](../api/workflow.md) |
| Presentation validation | `workspace-local`; structural validation plus governed vision-LLM visual QC over rendered screenshots is receipt-backed and command-runnable | `src/conference_pipeline/presentation/validation.py`, `src/conference_pipeline/presentation/vision_qc.py`, `src/conference_pipeline/workflow.py`, `src/conference_pipeline/workflow_runs.py`, `src/conference_pipeline/cli.py` | [Presentation Contract](../operations/presentation-contract.md), [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Presentation proof | `workspace-local`; receipt-backed dry-run proof reports are available through `conference-pipeline presentation-proof` | `src/conference_pipeline/presentation_proof.py`, `src/conference_pipeline/cli.py`, `src/conference_pipeline/workflow_runs.py` | [Presentation Contract](../operations/presentation-contract.md), [Workflow Runbook](../workflows/pipeline-workflow.md) |
| Presentation production | `workspace-local`; workspace-aware command planning, safe dry-run script contracts, structural validation, and receipt-backed proof reports are supported; live cross-conference production still requires conference-local proof | `conferences/neurips2025/decks/`, `scripts/presentations/` | [Workflow Runbook](../workflows/pipeline-workflow.md), [Source Of Truth](../source-of-truth.md) |
| Local control plane | `workspace-local`, defaulting to NeurIPS | `src/conference_pipeline/control_plane/` | [Control Plane Runbook](../operations/control-plane.md) |
| Exhaustive topic extraction | `workspace-local`; live-proven for `conferences/iclr2025` under v1.5, with v1.8 backend selection, quality-threshold, and fail-stop policy support | `src/conference_pipeline/extractions/`, `src/conference_pipeline/iclr2025_live.py::run_exhaustive_topic_extraction` | [Analysis Contract](../operations/analysis-contract.md) |
| Historical NeurIPS smoke proof | `neurips-live-proven` for dry-run proof contract | `src/conference_pipeline/smoke_test.py`, `src/conference_pipeline/cli.py` | [CLI](../cli.md) |
| Generic conference smoke proof | `non-live-general` | `src/conference_pipeline/smoke_test.py`, `src/conference_pipeline/cli.py` | [CLI](../cli.md), [Conference Analysis Playbook](../workflows/conference-analysis-playbook.md) |
| Docs link validation | `conference-general` | `scripts/check_docs_links.py`, docs-contract tests under `tests/` | [Source Of Truth](../source-of-truth.md) |
| Vector-store embeddings (cloud + local lanes) | `conference-general`; GPT-Researcher corpus retrieval defaults to local MLX BGE-M3 (`mlx-community/bge-m3-mlx-8bit`, 1024 dim); OpenAI `text-embedding-3-small` remains a compatibility fallback when local embedding receipts fail; Ollama remains available for local experiments; production plans must prove `embedding_readiness.client_start_or_load_attempted` after `build_embedding_client` plus first-use `embed_texts` before accepting a local embedding blocker; production receipts must record `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, `vector_embeddings_written: true`, and embedded chunk counts matching sidecar chunk counts | `src/conference_pipeline/indexing/embeddings.py`, `src/conference_pipeline/indexing/mlx_embeddings.py`, `src/conference_pipeline/indexing/ollama_embeddings.py`, `src/conference_pipeline/indexing/vector_store_naming.py` | [Local LLM Lane](../local-llm.md), [Stage Owner Audit](stage-owner-audit.md#indexing-local-embedding-materialization-policy) |
| Canonical vector-store chunking | `conference-general`; token-bounded source-aware chunking for papers, transcripts, slides/OCR, and generic artifacts; rejects snippet-only / first-240-character vectorization, word-count-only sidecars, readiness-only probes, and partial Chroma rows for production; `PG-CORPUS-TOKEN-INDEXING` requires `token_budget_policy_digest`, `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`, and `CORPUS_VECTOR_EMBED_BATCH_SIZE` before query planning | `src/conference_pipeline/indexing/token_policy.py`, `src/conference_pipeline/indexing/chunking.py`, `src/conference_pipeline/indexing/walker.py`; pinned NeurIPS reference strategy: `scripts/setup_vector_store_enhanced.py` | [Pipeline Workflow](../workflows/pipeline-workflow.md), [GPT-Researcher Vector Store Guide](../integrations/gpt-researcher/vector-store-guide.md), [Stage Owner Audit](stage-owner-audit.md#indexing-token-bounded-chunking) |
| Receipt lineage (content-addressed invalidation) | `conference-general`; every stage receipt of the generic workflow and the CVPR 2026 / ICLR 2026 / ICML 2026 runners records `output_digest` and `lineage`; `verify_stage_contract` and every resume short-circuit fail closed when an upstream's output or a recorded input moved (transitively); presentation per-deck steps skip on a per-deck ledger of input/output digests; `workflow stage-status` reports staleness read-only; `workflow bind-lineage --backfill` migrates pre-v5.1 runs | `src/conference_pipeline/receipt_lineage.py`, `src/conference_pipeline/presentation/step_lineage.py`, `src/conference_pipeline/stage_contracts.py`, `src/conference_pipeline/cli.py` | [Stage Owner Audit](stage-owner-audit.md#receipt-lineage), [Pipeline Workflow](../workflows/pipeline-workflow.md) |
| Shared production stages (presentation, NotebookLM podcast, release, proof) | `conference-general` code; live support stays per conference (`cvpr2026` receipts exist; `icml2026` adopts the same modules and needs its own live receipts); a runner supplies a `ProductionRunnerContext` (stage table, receipt type, code identity, env loader, summary and cost-approval hooks) and the stage writes receipts in that runner's own shape; the presentation stage runs the full per-deck chain with per-deck step ledgers, canary scope (`--deck-ids`, never `succeeded`), operator acceptance records, Gate A before TTS, and executive digest-chain verification | `src/conference_pipeline/production/` (`context.py`, `receipts.py`, `decks.py`, `presentation_steps.py`, `presentation_chain.py`, `presentation_stage.py`, `notebooklm_stage.py`, `release_stage.py`, `proof_stage.py`), `scripts/run_cvpr2026_production.py`, `scripts/run_icml2026_production.py` | [Stage Owner Audit](stage-owner-audit.md#shared-production-stages), [Presentation Contract](../operations/presentation-contract.md#structured-presentation-production-contract) |
| Public release (`conference-pipeline release`) | `conference-general` for `init`, `plan`, `build`, `check` and isolated rebuilds; `publish` is live-proven for `neurips2025` and `iclr2026` only, and always needs the owner's named approval | `src/conference_pipeline/publishing/`, `scripts/export_for_public.py`, `docs/public_export_manifest.schema.json` | [Public Release Runbook](../operations/public-release.md), [CLI](../cli.md), [Scrub Policy](../operations/public-release-scrub-policy.md) |
| Late paper-set admission (`conference-pipeline papers admit`) | `conference-general`; workspaces declare their existing-corpus adapter and lanes in `storage/papers/late_paper_lanes.json`; run on NeurIPS 2025 and ICLR 2025 in v6.0 Phase 5 | `src/conference_pipeline/production/paper_admission.py`, `src/conference_pipeline/indexing/paper_identity.py`, `src/conference_pipeline/indexing/paper_lanes.py`, `scripts/copy_late_paper_sets.py` | [CLI](../cli.md) |
| Incremental corpus indexing | `conference-general`; Stage 6 keeps a per-artifact `chunk_manifest.json` per store and artifact-keyed embedding buckets (`npz-buckets-v2`), so adding, changing, or removing artifacts re-reads, re-chunks, and re-embeds only those artifacts; vectors are reused by embedding digest across buckets, legacy positional shards, and sibling run ids; receipts record `checks.index_update` and `checks.embedding_update`; unchanged artifacts are confirmed by content hash (`--trust-file-stat` opts out); `--full-rebuild` is explicit; legacy stores are adopted without embedding by `scripts/migrate_corpus_index_to_incremental.py` | `src/conference_pipeline/indexing/chunk_manifest.py`, `src/conference_pipeline/indexing/corpus_vector_store.py`, `src/conference_pipeline/indexing/corpus_index_migration.py`, `scripts/migrate_corpus_index_to_incremental.py` | [Pipeline Workflow](../workflows/pipeline-workflow.md#incremental-corpus-indexing) |
| Talk-aware workshop transcripts | `conference-general`; optional when workshops are out of scope, required when workshop speech transcripts are indexed; `transcription_ocr.workshop_talk_boundary_discovery` uses Gemini 2.5 Pro to identify talk boundaries from long workshop transcripts plus slide timestamps, writes `*_talks.json`, emits 5-minute talk-context batches, and records `provider_receipt_paths` plus `cost_task_id=workshop_talk_boundary_discovery`; `transcription_ocr.workshop_talk_segmentation` annotates transcript segments with `workshop_id`, `workshop_title`, `talk_id`, `talk_title`, and `talk_speaker`; blocks with `talk_segmentation_provider_not_configured` when no `talks.json` or governed discovery path exists; audio-only recovery is rejected evidence for talk-aware transcripts | `src/conference_pipeline/workshop_talk_segmentation.py`, `scripts/discover_workshop_talks.py`, `scripts/segment_workshop_talks.py` | [Discovery Owner](stage-owner-audit.md#transcription-ocr-workshop-talk-boundary-discovery), [Segmentation Owner](stage-owner-audit.md#transcription-ocr-workshop-talk-segmentation), [Pipeline Workflow](../workflows/pipeline-workflow.md) |
| GPT-Researcher synthesis (cloud + local lanes) | `conference-general`; the GPT-Researcher package (0.16.1) runs research through `research.gptr_engine` on our stored hybrid index only (`local_hybrid` retriever plugin) and the registry's local Qwen model only, inside a loopback-only network guard; the in-house `LocalLLMResearcher` synthesis remains selectable (`--engine inhouse`). Evidence/draft synthesis defaults to local Qwen3.6 35B-A3B (`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`) through that engine with `WEB_SEARCH_ENABLED=false`, local BGE-M3 embeddings, and BM25+vector hybrid retrieval; OpenAI/Tavily is an explicit web-backed or local-runtime escalation; final public report composition remains governed by `analysis.deep_narrative_synthesis` (`openai:gpt-5.6-sol`, fallback `openai:gpt-5.6-terra`) | `src/conference_pipeline/research/config.py`, `src/conference_pipeline/research/researcher.py`, `src/conference_pipeline/research/local_researcher.py` | [Local LLM Lane](../local-llm.md), [Pipeline Workflow](../workflows/pipeline-workflow.md) |
| Topic evidence reconciliation and reference validation | `conference-general`; before slide synthesis, the packaged 07H workflow runs Qwen3.6 35B-A3B topic reconciliation, deterministic reference validation, deterministic safety fixes, bounded candidate-resolution repair when 07E blocks, broad repair only after unresolved candidate blockers, and final validation; final narrative/slides require 07H plus a clean final 07E receipt | `src/conference_pipeline/analysis/topic_reconciliation_reference_workflow.py`, `src/conference_pipeline/analysis/topic_reconciliation.py`, `src/conference_pipeline/analysis/topic_reference_validation.py`, `src/conference_pipeline/analysis/topic_reference_candidate_resolution.py`, `src/conference_pipeline/analysis/topic_reference_repair.py`, `scripts/run_topic_reconciliation_reference_workflow.py`, `scripts/reconcile_topic_analysis.py`, `scripts/validate_topic_reconciliation_references.py`, `scripts/resolve_topic_reference_candidates.py`, `scripts/repair_topic_reconciliation_references.py`, `src/conference_pipeline/analysis/narrative.py` | [Analysis Contract](../operations/analysis-contract.md), [Packaged Workflow Owner](stage-owner-audit.md#analysis-topic-reconciliation-reference-workflow), [Reconciliation Owner](stage-owner-audit.md#analysis-topic-reconciliation), [Reference Validation Owner](stage-owner-audit.md#analysis-topic-reference-validation), [Candidate Resolution Owner](stage-owner-audit.md#analysis-topic-reference-candidate-resolution) |
| Dual-store paper matching | `conference-general`; v1.91 adds the `paper-matching` workflow stage, builds separate `papers` and `artifacts` stores, extracts frontier-model keypoints from artifact GPT-Researcher findings, then runs paper-store GPT-Researcher to produce canonical `matched_papers.json` and optional legacy `ranked_papers.json` | `src/conference_pipeline/indexing/walker.py`, `src/conference_pipeline/paper_matching/`, `scripts/match_papers_from_artifact_keypoints.py`, `scripts/rank_papers.py` | [Pipeline Workflow](../workflows/pipeline-workflow.md), [CLI](../cli.md) |
| Vector-store rebuild CLI | `conference-general`; idempotent via source-manifest SHA-256; per-provider+model+dim collection naming so cloud and local coexist | `src/conference_pipeline/cli.py::vector_store_rebuild`, `src/conference_pipeline/indexing/vector_store_naming.py` | [Local LLM Lane](../local-llm.md) |
| `conference-smoketest --local` lane | `conference-general`; refuses to run if `OPENAI_API_KEY` is set; sets `EMBEDDING_PROVIDER` and `GPTR_LOCAL_LLM_RUNTIME_KIND` env for the local-only path | `src/conference_pipeline/cli.py::conference_smoketest` | [Local LLM Lane](../local-llm.md) |
| Local visual LLM lane (slide analysis + OCR) | `local-supported, benchmark_pending` (v1.61); paid OpenAI/Anthropic vision lane stays the production default until per-task receipts promote a local candidate to `approved` via `promote-vision-candidate`; loopback-only and image-bytes-never-persisted; reuses `--local` enforcement | `src/conference_pipeline/slide_analysis/`, `src/conference_pipeline/model_governance/mlx_vision_client.py`, `src/conference_pipeline/model_governance/fastvlm_native_client.py`, `src/conference_pipeline/benchmarks/vision_scoring.py`, `src/conference_pipeline/benchmarks/promote_vision.py`, `src/conference_pipeline/cli.py::conference_smoketest` | [Local LLM Lane — Vision & Slide Analysis section](../local-llm.md) |
| Paper source metadata (title/author) extraction and join | `conference-general` for resolution, propagation, and the audit gate (Phase 10.1); no provider call in the default path; see [Paper Source Metadata](#paper-source-metadata-titleauthor-extraction-and-join) below for per-conference support status | `src/conference_pipeline/indexing/paper_source_metadata.py`, `scripts/audit_corpus_source_metadata.py`, `scripts/run_cvpr2026_production.py`, `scripts/run_iclr2026_production.py` | [Corpus Metadata Audit skill](../../.claude/skills/corpus-metadata-audit/SKILL.md), [Pipeline Workflow](../workflows/pipeline-workflow.md) |

## Exhaustive Topic Extraction Stage

Stage id: `exhaustive-topic-extraction`. Phase 14A in the v1.5 ICLR roadmap.
Owner: `conference_pipeline.iclr2025_live.run_exhaustive_topic_extraction`,
dispatched through `WORKFLOW_STAGE_BY_ID["exhaustive-topic-extraction"]`.

Owner command:

```bash
conference-pipeline workflow run exhaustive-topic-extraction \
    --workspace conferences/iclr2025 \
    --run-id v15-iclr2025-production
```

The default heuristic-only path requires no provider keys and makes no paid
provider calls for confident matches. To opt into local LLM-assisted
reclassification for ambiguous records only, set `EXTRACTION_LLM_ASSIST=1` or
`EXTRACTION_LOCAL_LLM_ASSIST=1`, `EXTRACTION_LOCAL_LLM_BASE_URL` to a loopback
OpenAI-compatible endpoint, and `EXTRACTION_LOCAL_LLM_MODEL`. Calls are capped
by `LLM_ASSIST_MAX_CALLS` (default 300) and receipt metadata records provider,
model, endpoint label or redacted loopback base URL identity, call counts,
token counts when available, confidence path, escalation reason, paid-cost cap,
estimated paid cost, and missing-server behavior without storing secrets.
Hard or unresolved records preserve the approved remote escalation/failover
contract instead of treating local inference as a universal replacement.

Inputs (joined by `iterate_extraction_inputs`):

- `storage/transcription-ocr/v15_transcription_ocr_receipts.json`
- `storage/corpus/v15_corpus_inventory.json`
- `storage/acquisition/v15_download_receipts.json`
- `analysis/iclr2025_taxonomy.json`

Outputs (per run id):

- `storage/extractions/<run-id>/by-artifact/<artifact_id>.json` — one record
  per source artifact, with topic ids, evidence span, confidence, and the
  source artifact digest.
- `storage/extractions/<run-id>/by-topic/<topic_id>.json` — per-topic
  aggregation index (one file per non-proof family in the v1.4 taxonomy).
- `storage/extractions/<run-id>/v15_extraction_coverage.json` — coverage
  buckets (`covered`, `skipped_unusable`, `unclassifiable`, `blocked`,
  `not_attempted`), `coverage_ratio`, and thresholds.
- `storage/workflow-runs/<run-id>/14A-exhaustive-topic-extraction-receipt.json`
  — stage receipt with `details.coverage`, `media_prep_status`,
  `paper_reference_scope`, `token_usage`, and `resume_stats`.

Resume semantics: per-artifact records are reused when the source artifact
digest still matches (`existing_digest_matches`); pass `--rebuild` to force
re-extraction.

Downstream contract: every phase from analysis through proof fails closed
through `analysis_extraction_gate` when this stage receipt is missing,
not succeeded, stale relative to the transcription-ocr receipt, has
`not_attempted > 0`, or cites unknown record ids. See the
[Analysis Contract Runbook](../operations/analysis-contract.md#pre-analysis-extraction-requirement)
for the full clause.

Backup checkpoint: production initializes the mounted Drive leaf with
`scripts/sync_to_gdrive.py --profile backup --remote-path <mounted-conference-leaf>
--init-only` before artifact generation. BACKUP-01 closes Phase 14A by running
`scripts/sync_to_gdrive.py --profile backup --remote-path <mounted-conference-leaf>`
immediately after artifacts are written (dry-run, then write-mode) and
persisting the receipt at
`storage/workflow-runs/<run-id>/backup/14A-backup-receipt.json` with
`extractions` in `included_families`, manifest/artifact snapshot metadata, and
both exit codes 0.

## Extraction Provider Policy

Policy version: v1.7. This section is platform-general and applies to every
conference workspace that runs exhaustive extraction.

**Default path:** local LLM inference is the default extraction path for
topic classification and evidence mapping over completed artifacts. Configure
the local model with `EXTRACTION_LOCAL_LLM_MODEL`; backend selection happens at
runtime through `select_local_llm_backend()`. On Apple Silicon with `mlx_lm`
installed and the model present locally, the pipeline prefers MLX. Otherwise it
falls back to an OpenAI-compatible loopback HTTP endpoint configured by
`EXTRACTION_LOCAL_LLM_BASE_URL` and (optionally)
`EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL`. A local LLM readiness preflight must
pass before production extraction begins: selected backend reachable, model
available, sample inference succeeds, context/window limits recorded, and
fallback behavior explicit.

**Heuristics role:** heuristics serve as prefilter for confident matches and
as fallback when the local LLM endpoint is unavailable. Heuristic-only
extraction cannot satisfy production exhaustive extraction when a local LLM
endpoint is configured and reachable.

**Cloud escalation:** cloud LLM escalation only for low-confidence or ambiguous
records after local LLM and heuristic processing has produced insufficient
confidence. Cloud escalation requires explicit provider/model/cost caps,
escalation reasons, per-record receipt metadata (provider, model, cost
estimate, escalation reason, affected record ids), and fail-closed downstream
behavior when required receipts are missing or stale.

**Conditional OCR:** slide OCR is optional by default. Generated plans classify
OCR posture as `required`, `skipped_with_rationale`, or `deferred_blocked`
based on a source-coverage assessment. OCR is required only when corpus text
coverage is insufficient, the brief requires slide/visual evidence, or
downstream deliverables need slide-level citations.

**Non-extraction providers:** Non-extraction generation providers preserve
benchmark-governed contracts unless a receipt-backed override is approved. This
policy applies only to the exhaustive extraction classification and
evidence-mapping stage; GPT-Researcher retrieval itself is governed separately
by the BM25+vector hybrid retrieval contract.

**Operator status:** `conference-pipeline extraction-status --workspace <path>`
shows the selected extraction mode, local LLM readiness from the most recent
preflight receipt, cloud escalation count, OCR posture, receipt links, and
blockers — sourced entirely from on-disk receipts without live endpoint calls.

## Backend Selection (v1.8)

The platform selects the local extraction backend at runtime through
`src/conference_pipeline/extractions/backends.py::select_local_llm_backend`.

- MLX-first on Apple Silicon when `mlx_lm` is installed and the requested
  model is available locally.
- OpenAI-compatible HTTP fallback otherwise for loopback servers such as
  Ollama, llama.cpp, or vLLM.

The selected backend is recorded on `LocalLLMReceipt.endpoint_label` as either
`mlx://<model>` or a redacted loopback HTTP label. Force backend selection with
`EXTRACTION_LOCAL_LLM_BACKEND=mlx|http`; `http` is honored as an explicit
loopback-server override, while invalid values and incompatible forced `mlx`
combinations fail closed with `BackendSelectionError`.

For Qwen-family MLX instruction models, `MLXBackend` renders prompts through
the tokenizer chat template with thinking disabled when supported. This keeps
structured extraction calls from emitting `<think>` traces before the JSON
payload. The classifier also recovers fenced or verbose JSON responses and
production recovery receipts block when a required local-Qwen lane records no
local LLM attempt or any failed local LLM attempt.

## Quality Sample Thresholds (v1.8)

The exhaustive extraction quality sample uses canonical defaults from
`src/conference_pipeline/extractions/thresholds.py`:

- min mean confidence: `0.65`
- max failure-class rate: `0.15`
- min record coverage: `0.80`
- cloud escalation trigger: `0.50`

Workspace-local overrides may replace any subset of these values in
`config/extraction-thresholds.json`. Effective values for each run are recorded
on `LocalLLMReceipt.effective_thresholds`.

When the sample mean confidence falls below the cloud escalation trigger,
`run_quality_sample()` makes a real cloud provider call (OpenAI by default) and
writes an `EscalationReceipt` that wraps the cloud `ProviderCallReceipt`.
Configure the provider with `EXTRACTION_CLOUD_ESCALATION_PROVIDER`, model with
`EXTRACTION_CLOUD_ESCALATION_MODEL`, and retry budget with
`EXTRACTION_CLOUD_ESCALATION_RETRIES`. The paid-cloud proof lane is opt-in:
`RUN_LIVE_CLOUD_ESCALATION=1` runs the live escalation receipt test and fails
closed if provider credentials are missing. If the cloud LLM cannot be reached
after the retry budget, `CloudEscalationError` fails the workflow stage through
the normal resumable stage receipt path; it does not write a `BlockerReceipt`.
If escalation succeeds but the quality sample remains below fail-closed
thresholds, `QualitySampleThresholdError` fails the stage through that same
resumable, non-blocker path.

## Fail-Stop Blocker Policy (v1.92)

Autonomous workflow runs halt deterministically on any required blocked or
unavailable stage. Stage results with `status: blocked`,
`status: deferred_blocked`, `status: failed`, `status: unavailable`, or
`status: unsupported` are not completion evidence. A `BlockerReceipt` is
written under `storage/workflow-runs/<run-id>/blocker-receipts/<stage>/`, and
resumption requires an `UnblockReceipt`.

Blocker classes:

- `ocr_deferred_blocked`
- `stage_blocked`
- `stage_unavailable`
- `local_llm_not_ready`
- `missing_receipt`
- `threshold_violation` for invalid threshold configuration, not cloud outage
  escalation failures
- `stale_receipt`

Provider availability failures, missing live owner commands, unsupported live
stages, missing or stale required receipts, and failed required validation all
stop downstream podcast, deck, release, proof, and publication work.

Manual acknowledgement is available through either
`.venv/bin/python -m conference_pipeline.cli workflow unblock ...` or
`python -m conference_pipeline.workflow_unblock ...`. Updated workflow
summaries surface `halt_status`, `halt_stage_id`, `blocker_receipt_path`, and
`unblock_acknowledgements`.

## Per-Conference OCR Proof Status (v1.8)

| Conference | v1.8 OCR proof posture | Proof packet |
|------------|------------------------|--------------|
| `conferences/neurips2025` | Deferred. Historical workflow receipts are present, but no committed transcription/OCR workspace proof packet predating v1.8 was available to promote as live three-state OCR proof. | `conferences/neurips2025/storage/workflow-runs/ocr-proof-v1.8-neurips2025-20260430/summary.json` |
| `conferences/iclr2025` | Receipt-backed proof packet assembled from committed v15 transcription/OCR artifacts, with required, skipped-with-rationale, and deferred-blocked examples plus manual acknowledgement evidence. | `conferences/iclr2025/storage/workflow-runs/ocr-proof-v1.8-iclr2025-20260430/summary.json` |

## Paper Source Metadata (Title/Author) Extraction and Join

Phase 10.1 (PROD-15). Corpus-inventory records for the `papers_main` and
`papers_workshop` source families carry structured `paper_title` and
`authors` fields, resolved by
`src/conference_pipeline/indexing/paper_source_metadata.py` and propagated
through chunking, the vector store, narrative source packs, and the topic
reference / speaker-author-work ledgers. This is a platform module: every
conference workspace's runner routes its corpus-record construction through
the same shared entry point (`enrich_paper_record` / `enrich_paper_records`);
no conference contributes its own title/author extraction logic.

**Resolution order (deterministic, no provider call in the default path):**

1. An authoritative join or receipt lift for the conference's own source
   shape — `resolve_main_paper_metadata` (source-discovery program-item
   join), `resolve_workshop_paper_metadata` (per-paper workshop receipt
   lift), or `resolve_submission_system_metadata` (a receipt-lift for a
   conference whose submission platform, e.g. OpenReview, already bundles
   normalized title/authors directly onto the download artifact, with no
   program-item join structure available).
2. A deterministic, bounded PDF-header fallback parser
   (`parse_paper_header_metadata`) for records the authoritative path leaves
   unresolved, only invoked when a `header_text_reader` is supplied.
3. Honest non-resolution: an artifact that no path resolves keeps
   `source_metadata_provenance` set to `unresolved` on the missing field(s)
   and is never fabricated. An unresolved individual record is a coverage
   number, not a stage blocker.

Every resolution records a `source_metadata_provenance` label per field
(`source_discovery_join`, `listing_html`, `workshop_receipt`,
`submission_system_metadata`, `pdf_header_parse`, or `unresolved`) so coverage
auditing can distinguish an authoritative join from a heuristic parse from an
unresolved field, and coverage is always measured and reported per source
family — `papers_main`, `papers_workshop`, and `artifacts` are never blended
into one aggregate number. The `artifacts` family (video-caption transcripts
and similar) is structurally author-less and is reported but never gated on
author coverage; its title coverage is still measured for visibility.

**Completion gate:** `scripts/audit_corpus_source_metadata.py --workspace
conferences/<slug>` (also `--json`, `--write-receipt <run-dir>`) fails closed
when a gated family's measured coverage falls below its justified floor, when
a required ledger artifact is missing, or when a floor override lacks a
justification. See the [Corpus Metadata Audit
skill](../../.claude/skills/corpus-metadata-audit/SKILL.md) for the full
per-family interpretation and the no-silent-lowering floor-override rule.

**Per-conference support status:**

| Conference | Status | Evidence |
|------------|--------|----------|
| `conferences/cvpr2026` | Backfilled and receipt-measured: the main-conference and workshop builders in `scripts/run_cvpr2026_production.py` route every record-construction branch through the shared resolvers, and the corpus-inventory Stage 6 receipt records `source_metadata_coverage` per family. This is the proving-ground conference for the fix. | `conferences/cvpr2026/storage/workflow-runs/<run-id>/stage-06-corpus-inventory-index.json` checks `source_metadata_coverage`; see plan `10.1-05-SUMMARY.md`. |
| `conferences/iclr2026` | Sibling-generality proof only, not a live production re-run. `scripts/run_iclr2026_production.py`'s corpus-inventory-index stage routes both paper families through the identical shared `enrich_paper_records` entry point and writes the same `source_metadata_coverage` receipt key. Proven at fixture level (unit tests asserting resolved/unresolved behavior) and by a real, on-disk measurement against all 5,352 real ICLR2026 accepted-paper records already committed at `conferences/iclr2026/storage/download-corpus/openreview_accepted_papers.json` (no Drive fetch), showing 100% title/author resolution via `resolve_submission_system_metadata`. **Not proven:** a live end-to-end production re-run of the ICLR2026 corpus-inventory-index stage against real downloaded PDFs and transcription output, since neither `storage/download-corpus/download_corpus.json` nor `storage/corpus/corpus_inventory.json` exists locally for this workspace (Drive-backed, not restored) and re-running the full ICLR2026 production pipeline is out of this phase's scope (a separately owned milestone). A future ICLR2026 production re-run proves this end-to-end via `.venv/bin/python scripts/audit_corpus_source_metadata.py --workspace conferences/iclr2026 --write-receipt storage/workflow-runs/<run-id>` against a real `--stage corpus-inventory-index` run's output. | plan `10.1-07-SUMMARY.md`. |

## Transcript Evidence Lanes (post-harvest transcript admission)

`conference_pipeline.indexing.transcript_evidence_lanes` is the platform
contract for transcripts that arrive after a conference's base caption
inventory was frozen: captions YouTube publishes a day late, a governed ASR
fallback run, a hand-recovered keynote. Each recovery becomes a *lane*: a
digest-bound `lane_receipt.json` listing every transcript (video id, path,
sha256, source, event URLs, provenance) plus explicit coverage gaps, registered
in precedence order in `storage/transcription-ocr/evidence_lanes.json`. A
conference runner's corpus-inventory-index stage calls
`load_admitted_lane_records(...)`, which returns ordinary `speech_transcript`
artifact records; the base inventory keeps precedence, lanes follow registry
order, and any digest or path failure blocks the stage.

Why: on CVPR 2026 the r6 ASR run (467 transcripts) and the r7 caption recheck
(66 captions, including all three plenary keynotes) were written beside the
immutable r5 inventory and never consumed, because the Stage 6 admission list
and the Stage 5 contract both pinned r5 and its exact counts. Lanes make
later evidence appendable without touching the base inventory or its
contract. The companion rule, enforced by `scripts/audit_conference_hardcoding.py`,
is that contract modules verify structure and lineage and never pin run
counts as literals.

| Conference | Lanes registered | Proof |
|------------|------------------|-------|
| `conferences/cvpr2026` | `r7-caption-recovery` (66 captions), `r6-asr-fallback` (467 ASR transcripts, 54 gaps) via `scripts/build_cvpr2026_transcript_evidence_lanes.py`; since 2026-09-17 also six `yt-*-20260917-{caption,gemini}` lanes (1,365 harvest transcripts) via `scripts/build_youtube_harvest_evidence_lanes.py` | `storage/corpus/corpus_inventory.json` `upstream.transcript_evidence_lanes` and `stage-06-corpus-inventory-index.json` `checks.transcript_evidence_lanes` after the 2026-09-17 rebuild. |
| `conferences/neurips2025` | `deep-dive-slideslive` (119 recordings from 129 deep-dive talk files; 10 named gaps) via `conference-pipeline talks bind neurips2025` | `storage/corpus/corpus_statistics.json` `talks`, admitted by `conference-pipeline papers admit` (v6.0 Phase 5.1). |
| `conferences/iclr2025` | `v15-session-asr` (751 transcripts from the v1.5 transcription receipt; 1 named gap) via `conference-pipeline talks bind iclr2025` | `storage/corpus/corpus_statistics.json` `talks` (v6.0 Phase 5.1). |
| other workspaces | none yet; the module is conference-neutral and any runner may adopt it | — |

YouTube harvest directories bind through the conference-neutral
`conference_pipeline.indexing.youtube_harvest_lanes` module
(`scripts/build_youtube_harvest_evidence_lanes.py`): one lane per transcript
tier, rolling auto-caption cues collapsed, timestamped segment JSON
materialized in the lane, and typed coverage gaps for unavailable, failed, and
speechless videos. Lane records may carry an annotated transcript and talk
map for a split multi-talk recording (`conference_pipeline.workshop_talk_discovery_windowed`
via `scripts/discover_harvest_talk_boundaries.py`); both files are digest-bound
in the lane receipt, and the talk map must name the transcript digest it was
derived from.

### Talks a workspace already holds (`talks bind`), and the coverage guard

`conference_pipeline.indexing.talk_source_lanes` binds transcripts that exist on disk but
were never admitted. A workspace declares its sources in
`storage/transcription-ocr/talk_sources.json`; readers are named by format
(`deep_dive_presentations`, `transcription_receipt_v15`). Named exclusions with their
rulings go in `talk_exclusions.json`. An absent or blank transcript is a gap
`empty_transcript`, and events that share one recording keep one record
(`duplicate_recording`). `conference-pipeline papers admit <slug>` admits the lanes beside
the papers and writes a `talks` block (listed, admitted, excluded by reason, per kind) into
`corpus_statistics.json`.

`scripts/audit_talk_coverage.py --workspace conferences/<slug>`
(`conference_pipeline.indexing.talk_coverage`) checks that every transcript on disk is
admitted or excluded by name. It exits 0 when that holds; 1 when something is unaccounted,
or when talks are on disk but there is no corpus inventory; 2 when a file is unreadable. It
reads four formats:
- v1.5 transcription receipts;
- the stage transcription inventory, in its `output_inventory` and `caption_transcripts`
  shapes;
- evidence lane receipts;
- the deep-dive export.

It cannot see a transcript in any other format, so **a new transcript producer must write
one of these formats or bind its output as an evidence lane.** On 2026-11-17 it exited 0 on
NeurIPS 2025, ICLR 2025, ICML 2026 and CVPR 2026. ICLR 2026's Drive-backed
`storage/transcription-ocr` was unreadable from the measuring process (exit 2).

## Current Boundaries

- Phase 25 validates structural readiness of manifest-registered presentation artifacts; v1.63 adds a governed vision-LLM visual QC report over rendered screenshots as a content gate for production decks.
- Presentation validation also enforces configured hyperlink and generated-SVG density targets for topic-deep-dive decks when those targets are declared in `presentation_manifest.json`.
- Root docs are universal, but some runtime class names still contain
  `NeuripsPipeline`; that is an implementation boundary, not permission to put
  NeurIPS-specific guidance in root agent files.
- Live acquisition stages for new conferences need conference-local proof before
  they are marked live-supported.
- `conference-smoketest` is the generic proof command. It can validate a
  non-NeurIPS workspace through dry-run or mock receipts before live providers
  are configured.
- `workflow deck` remains a backward-compatible alias for legacy deck command construction; it is not the full presentation lifecycle.
- First-class presentation stages are presentation_blueprint, deck_contract_generation, source_binding, visual_plan_generation, render_spec_generation, render_export, visual_qc, delivery_generation, presentation_review, video_generation, video_sync_validation, and human_review.
- Topic-deep-dive presentation blueprinting uses `presentation_frame.json` as the post-analysis audience-frame handoff: it separates retrieval topic title from public deck title, requires field-level thesis/questions, and records `presentation_frame_validation` before deck-contract generation.
- First-class presentation stages now provide workspace-aware command planning, safe dry-run script contracts, structural validation, and receipt-backed proof reports; live cross-conference production still requires conference-local proof.
- Phase 25 validates structural readiness of manifest-registered presentation artifacts and does not re-run live generation or assert subjective visual quality.
- Presentation production is not yet `conference-general`: current reusable support covers `Deck command construction` plus manifest-backed presentation stage registration and receipts, workspace-aware command planning, safe dry-run script contracts, structural validation, and receipt-backed Phase 26 proof reports; live cross-conference production still requires conference-local proof.
- Phase 26 records receipt-backed proof reports through `conference-pipeline presentation-proof --workspace conferences/neurips2025 --presentation-type topic-deep-dive --topic 01_tools_and_open_source --mode dry-run --run-id <run-id>` and `conference-pipeline presentation-proof --workspace conferences/iclr2025 --presentation-type topic-deep-dive --topic 01_iclr_2025_proof --mode dry-run --stages deck_contract_generation --run-id <run-id>`.
- Phase 26 proof receipts are workspace-local dry-run/scaffold evidence, not a live cross-conference production claim.
- Proof reports are written to `conferences/<slug>/storage/workflow-runs/<run-id>/presentation-proof-report.json`; the consolidated proof summary is `.planning/milestones/v1.3-phases/26-neurips-backfill-and-iclr-presentation-proof/26-PROOF-SUMMARY.md`.
- Committed Phase 26 run ids are `phase26-neurips-presentation-proof`, `phase26-iclr-presentation-dryrun`, and `phase26-iclr-presentation-live-denied`.
- Richer NeurIPS Phase 7b media scripts such as `generate_slides_agentic.py` and `generate_podcast_gemini.py` are NeurIPS-local, historical, or experimental until workflow ownership, validation, and proof receipts establish broader support.
- Presentation stage dry-run receipts include command_metadata with exact argv, selected scope, input artifacts, output artifacts, support level, provider/tool requirements, and unsupported_reason.
- Workflow-planned presentation scripts use decks/<presentation>/... paths; presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning.
- Unsupported reusable stages expose unsupported_reason instead of claiming script execution.
- Video generation is dry-run/planning-only in Phase 24 and does not prove final MP4 generation, visual validation, or cross-conference proof.
- `small-neurips-smoketest` remains the historical v1.1 proof contract for
  `conferences/neurips2025`.
