# Production Workflow Gates

Generated live-production milestones must preserve explicit receipt gates for
workflow steps that are easy to flatten into broad "analysis" or "QA" language.
The roadmap table and `approval-summary.json` rows are the durable contract;
neither broad stage coverage nor narrative prose is enough to close these
steps.

Every live runner stage that can write `status: "succeeded"` must also pass the
runtime stage contract gate in `conference_pipeline.stage_contracts`. The stage
binding's `runtime_contract` owns required receipts, required outputs, required
checks, freshness bindings, forbidden evidence, and explicit non-production
markers. Production receipts must include `script_registry_sha256`,
`required_script_ids`, and `stage_contract.accepted_evidence` /
`stage_contract.blockers`.

## Required Fields

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

## Required Gates

| Gate ID | Workflow step | Required evidence boundary |
|---------|---------------|----------------------------|
| `PG-WORKSHOP-INDEX` | Workshop venue discovery against the conference's OpenReview parent group | Owner command `scripts/discover_openreview_workshops.py` and `storage/workshops/discovery_receipt.json` prove the sub-venue listing, per-workshop paper count, tier breakdown, and `pdf_availability_class`. Required whenever the analysis handoff's `workshop_coverage_posture` is not `out_of_scope`. |
| `PG-WORKSHOP-CORPUS` | Workshop paper PDF acquisition (OpenReview + external sites) | Owner command `scripts/download_openreview_workshop_papers.py` plus per-workshop external-site downloaders prove each accepted-paper PDF has a receipt with `content_type=application/pdf`, `sha256`, and `byte_count > 0`. Stub external-site scaffolds and OpenReview metadata records without bytes are rejected evidence. |
| `PG-EMBEDDED-MEDIA-RECOVERY` | Embedded media recovery from cached authenticated item pages | Required when `download_corpus.json` contains successful `item_page_auth_probe` artifacts whose cached HTML exposes embedded SlidesLive media but no local audio/video artifact was merged. Owner command `scripts/recover_iclr2026_slideslive_media.py` or a conference-local equivalent must write `storage/download-corpus/slideslive_media_recovery.json`, merge recovered audio into `download_corpus.json`, and prove no source pages, paper PDFs, workshop PDFs, poster images, or unrelated media families were redownloaded. |
| `PG-SLIDESLIVE-SLIDE-ACQUISITION` | SlidesLive slide-image acquisition and cleanup | Required when cached `item_page_auth_probe` artifacts expose SlidesLive embeds and visual/OCR evidence is in scope. Owner commands `scripts/recover_slideslive_slides.py`, `scripts/detect_reference_slides.py`, and `scripts/cleanup_slides.py` must write `storage/download-corpus/slideslive_slide_recovery.json`, `slide_reference_detection.json`, and `slide_cleanup.json`, merge slide images into `download_corpus.json`, and mark duplicate/generic banner slides with `skip_ocr` rather than deleting raw files. |
| `PG-WORKSHOP-TALK-MEDIA-ACQUISITION` | Workshop talk-media acquisition or explicit scope waiver | Required whenever workshops are in scope before Stage 5/6. A conference-local workshop agenda/talk media acquisition owner must write `storage/workshops/talk_media_acquisition_receipt.json` with workshop/talk/media metadata and `allowed_hosts`, or a formal `storage/workshops/workshop_talk_media_scope_waiver.json` must record decision, rationale, approver, and downstream impact. The plan must bind a conference-local embedded host allowlist via `--embedded-media-host <conference-local-host>` (for ICLR production this is `--embedded-media-host slideslive.com`; other conferences must name their own verified embedded-video host rather than reusing ICLR's); YouTube, Vimeo, Zoom, social/profile, and unrelated embeds are rejected unless a conference-specific phase explicitly widens that conference's allowlist. A zero-count workshop speech audit is rejected evidence for deciding workshop talks are out of scope. |
| `PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY` | Gemini workshop talk-boundary discovery before talk-aware transcript segmentation | Required when workshop transcripts/audio are in scope and no vetted `talks.json` exists. Owner command `scripts/discover_workshop_talks.py` must run `transcription_ocr.workshop_talk_boundary_discovery` with Gemini 2.5 Pro (`gemini-2.5-pro`), write `*_talks.json`, emit 5-minute talk-context batches, and record `model_governance_task_id`, `provider_receipt_paths`, and `cost_task_id=workshop_talk_boundary_discovery` before segmentation or indexing. |
| `PG-CORPUS-TOKEN-INDEXING` | Token-bounded corpus chunking and embedding materialization | Owner command and `stage-06-corpus-inventory-index.json` prove current token-bounded chunk sidecars, durable embedding sidecar manifests, and persisted embeddings through `conference_pipeline.indexing.corpus_vector_store`, not conference-local reimplementation. Receipts and index digests must record `token_budget_policy_digest`, `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, `embedding_sidecar_manifest_path`, `embedding_sidecar_manifest_sha256`, `embedding_sidecar_format`, `embedding_sidecar_shard_count`, `chroma_insert_batch_size`, `vector_embeddings_written: true`, embedded chunk counts matching sidecar chunk counts, indexed text kind counts, and stable source metadata coverage. Batch-size tuning is operational evidence, not part of `embedding_policy_digest`. Word-count-only sidecars, readiness-only probes, partial Chroma rows, chunk-sidecar-only runs, ASR receipt JSON chunks, historical exhaustive extraction output without registry receipt, mutable topic-id filtering as the base retrieval scope, conference-local embedding/chunking reimplementations, and any redownload of source pages, papers, workshop PDFs, audio, slides, video, or poster media are rejected evidence. |
| `PG-TRANSCRIPT-EVIDENCE-LANES` | Post-harvest transcript recovery bound into the corpus | Required whenever any transcript recovery runs after the base caption inventory (caption rechecks, governed ASR fallback, hand-recovered keynotes). The recovery must write its own receipt, a lane receipt must be built and registered through `conference_pipeline.indexing.transcript_evidence_lanes` (`storage/transcription-ocr/evidence_lanes.json` plus `evidence-lanes/<lane_id>/lane_receipt.json`, every transcript sha256-bound), and the corpus-inventory-index stage must be re-run so `stage-06-corpus-inventory-index.json` `checks.transcript_evidence_lanes[]` lists each lane with `admitted_count`. Recovered transcripts sitting in a side directory with no registered lane, an unregistered lane, or a stage contract that pins run counts as literals are rejected evidence. |
| `PG-INDEX` | Enhanced indexing/vector-store construction | Owner command and `stage-06-corpus-inventory-index.json` prove final corpus inputs, vector-store outputs, input digests, canonical source-aware chunking, and local embedding readiness. If embeddings are unavailable, the receipt must prove the owner called `build_embedding_client` and forced first use with `embed_texts`, recording `embedding_readiness.client_start_or_load_attempted`, before writing a local embedding blocker such as `blocked_local_embedding_client_start_failed`. |
| `PG-TOPIC-STRATEGY` | Title-spine topic strategy before query planning | Owner command `scripts/build_topic_strategy.py` and `storage/analysis/topic-strategy/<run-id>/topic_strategy_receipt.json` prove invited/keynote, main oral, session, workshop, and segmented workshop talk title collection, title-spine sha256, three independent model recommendations, exact 12-topic merged `analysis/topic_strategy.json`, and live provider receipts. Stale candidate interest groups, single-model-only topic choice, or planning prose without title-spine receipt are rejected. |
| `PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION` | Qwen full-read exhaustive extraction before query planning | Owner command `scripts/run_llm_full_read_extraction.py --scope all --resume --model mlx-community/Qwen3.5-35B-A3B-4bit` and `storage/workflow-runs/<run-id>/14A-llm-full-read-extraction-receipt.json` prove every eligible transcript window and paper abstract was read against the finalized 12-topic strategy. The receipt must have `status: succeeded`, `scope: all`, `limit: null`, `checks.completion_state: complete`, `completed_unit_count == expected_unit_count`, `failed_count: 0`, `checks.transcript_source_coverage.unread_input_count: 0`, and by-unit/by-topic output families. Completed-equals-expected is not coverage evidence (both counts come from the same unit iterator, so a transcript that yields no unit lowers both); `transcript_source_coverage` is the only field that names transcripts with text and no read unit, and every registered transcript evidence lane must show `unit_count > 0` in its `by_storage_lane` row. Heuristic-only receipts, pilot runs, `--limit` runs, transcript-only runs, metadata-only outputs, and empty transcript sidecars are rejected. |
| `PG-QUERY` | Query planning | Owner command and `stage-07-analysis.json` prove query-plan generation before topic execution. |
| `PG-GPT-RESEARCHER` | GPT Researcher hybrid retrieval/deep-query execution | Owner command and `stage-07-analysis.json` prove BM25+vector hybrid retrieval by default, resolved GPT-Researcher parameters such as `MAX_ITERATIONS`, provider/model provenance, and source-backed deep-query execution; local BGE+Qwen can satisfy this evidence/draft layer only, while final public narrative synthesis stays under `analysis.deep_narrative_synthesis` unless local deep validation passes; metadata-only, paper-only, vector-only-by-default, or missing-provenance evidence is rejected. |
| `PG-NOTEBOOKLM-PODCAST` | Per-deck NotebookLM Audio Overview podcasts (when in scope) | Owner command `scripts/generate_notebooklm_podcast.py --live --ack-live-notebooklm --format deep-dive --length long` (or the runner's `--stage notebooklm-podcast`) and `stage-10-notebooklm-podcast.json` prove one live Google NotebookLM Audio Overview per topic and executive deck, each bound by `audio/<id>/notebooklm_podcast/notebook_manifest.json` (notebook id, every source sha256, prompt sha256, audio sha256) with `deck_success_count == deck_scope_full_count`, `deck_blocker_count: 0`, `notebooklm_podcast_aggregate_sha256`, `storage_state_present`, and `engine_version`; the curated Drive folder is proven by `scripts/deliver_notebooklm_podcasts.py`'s `notebooklm-podcast-delivery-receipt.json` with `delivered_count == expected_count`. Dry-run manifests, `--fake-engine` runs, `--deck-ids` canary receipts, TTS substitutes, and the public-bundle `scripts/generate_notebooklm_audio.py` planner are rejected evidence. |
| `PG-COVERAGE-GAP-FILL` | Coverage analysis and gap fill | Owner command and `stage-08-qa-gap-fill-synthesis.json` prove coverage repair before synthesis, podcast, deck/presentation, release, or proof. |
| `PG-SYNTHESIS` | Executive/trend synthesis | Owner command and `stage-08-qa-gap-fill-synthesis.json` prove synthesis outputs before release or proof. |
| `PG-NARRATION-SCRIPT-REVIEW` | Narration review before TTS | Owner command and `stage-10-deck-presentation-production.json` prove every topic and executive narration has fresh single-presenter `presenter_narration_lint_report.json`, `presenter_narration_review.json`, and `presenter_narration_series_review.json` artifacts for `video_build/script.json`, plus fresh NotebookLM `narration_lint_report.json`, `narration_review.json`, and `narration_series_review.json` artifacts for `video_build_notebooklm/script.json`; every relevant review has `approve_for_tts=true`; failed single-presenter deck reviews route through presenter narration fix, failed NotebookLM deck reviews route through narration script fix, and failed cross-deck reviews route through narration series fix or an explicit governed revision before TTS/video. |
| `PG-OPTIONAL-PIPELINES` | Optional pipelines | Owner command and `stage-12-proof.json` prove explicit receipts or waivers for exhaustive extraction, podcast, deck/presentation, and paper-level analysis. |

## Sources

- `docs/workflows/pipeline-workflow.md#overview`
- `docs/operations/analysis-contract.md`
- `.claude/skills/plan-conference-production/SKILL.md`

Workshop speech transcripts also require
explicit workshop talk-media acquisition proof, with authenticated access when
conference workshop videos/slides are login-gated, or a formal
`workshop_talk_media_scope_waiver.json` before Stage 5/6 when workshops are in
scope. A `workshop_speech_count: 0` audit only describes current inventory; it
does not prove workshop talk media was unnecessary. Unauthenticated zero-media
blockers are not final evidence when an authenticated operator can see media.
Generated plans must bind a conference-local embedded host allowlist via
`--embedded-media-host <conference-local-host>` and record `allowed_hosts`;
for ICLR production this is `--embedded-media-host slideslive.com`, but other
conferences must name their own verified embedded-video host rather than
reusing ICLR's. YouTube, Vimeo, Zoom, social/profile, and unrelated embeds
are rejected for any conference unless a conference-specific phase explicitly
widens that conference's allowlist.

Workshop speech transcripts then require
`transcription_ocr.workshop_talk_boundary_discovery` when no vetted
`talks.json` exists, followed by the separate
`transcription_ocr.workshop_talk_segmentation` owner when workshop speech is in
scope. Missing `talks.json` or governed discovery path blocks with
`talk_segmentation_provider_not_configured`; audio-only recovery is rejected
evidence for talk-aware transcripts.
