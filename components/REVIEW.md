# Component index review, 2026-11-15

Scope: the whole index, at its first build (v6.0-Public-Release Phase 1).
Code read at commit `882468e63`. 478 components and 6 subsystems after the build (none
before); 329 package modules, all assigned, and 427 of 428 files under `scripts/`
(the one left out is `scripts/presentations/run_full_pipeline.sh`, a shell wrapper of
`conference-pipeline workflow run`). The package modules and scripts were read by 35
readers, most of them in part: 328 entries carry `checked: partial`, and every reader
listed the line ranges it skipped. These are suggestions; nothing below has been changed
in the code.

"Checked how" says who confirmed a finding. **Writer** means the writer reproduced it with
a grep or read before writing it here. **Reader** means one reader reported it with a file
and line and the writer did not reproduce it. Treat those as leads.

## What was repaired in the index
- `figures/errors.py` moved from `figures.compiler` to `figures.spec`, and `figures/tokens.py` from
  `figures.svg_canvas` to `figures.text_fitting`. The spec, layouts and compiler all import the
  first; text measurement and the canvas both read the second. The moves removed three false
  mutual-import pairs.
- Five test files that import only one component were added to that component's `tests`.
- Duplicate ids proposed by several readers were merged into one entry each: the
  `conference.<conf>_one_off` entries, `publishing.release_cli_wrappers` and
  `research.gemini_browser_query`.
- `publishing.closeout` was taken out of `subsystem.public_release`'s flow: it runs as a later
  step (`scripts/record_public_closeout.py`), not from the release orchestrator.

## Suggested code changes
| # | Kind | What | Where | Size | Checked how |
|---|---|---|---|---|---|
| 1 | Drift | The binding contract's blocked-status set lacks `off_topic_artifact`, which the ledger's set has, so the binding accepts a reference the ledger blocks | presentation/source_binding.py:13, presentation/source_inventory.py:37-46 | small | writer: read both sets |
| 2 | Duplicate | `sanitize_for_receipt` is defined twice | provider_receipts.py:293, iclr2025_live.py:946 | small | writer: grep |
| 3 | Duplicate | The figure semantic review exists in the package and in a script; the runner uses the script | presentation/semantic_svg_review.py:54, scripts/presentations/review_programmatic_figures.py:55 | medium | reader (s01, c-pres-06) |
| 4 | Dead code | `build_structured_delivery_video` (a sine-tone placeholder MP4) is imported by two video scripts and called by neither | presentation/structured_video.py:1863, scripts/presentations/build_presenter_video.py:50, build_notebooklm_style_presenter_video.py:43 | small | writer: grep finds imports only |
| 5 | Dead code | Eight legacy `_build_*` stage builders, about 600 lines, are absent from the builder table | presentation/commands.py:1163-1773 | medium | reader (s03) |
| 6 | Duplicate | `MAX_FAMILY_SHARE = 0.45` is assigned twice in one module | figures/selection.py:71, :425 | small | writer: grep |
| 7 | Defect | Production cost rollups are built without estimates, so they never show drift | workflow_runs.py:356 | small | writer: the call passes only workspace and run id; drift claim is the reader's (s13) |
| 8 | Layering | About 20 shared modules import `iclr2025_live` for `sanitize_for_receipt`, `utc_now` and extraction helpers, which makes 13 of the 30 mutual-import pairs in `review` | iclr2025_live.py; see `review` "Components that import each other" | large | writer: `review` output |
| 9 | Defect | The analysis orchestrator's embedding clients define `embed()`, but the vector-store setup step calls `embed_texts`; tests stub the step | analysis/steps/vector_store_setup.py, analysis/orchestrator.py | small | reader (s08) |
| 10 | Defect | The setup step writes a `neurips_presentations__...` collection while traceability reads `<slug>_presentations` | analysis/steps/ | small | reader (s08) |
| 11 | Defect | `merge_transcripts` says it removes chunk overlap but never does | audio_chunker.py | small | reader (s16) |
| 12 | Governance | Model calls that bypass `llm_client`: the paid source-discovery join (OpenAI SDK), the Stage 8 synthesis client (Anthropic and OpenAI SDKs), windowed talk discovery (Gemini), reference and figure captioning, executive narration scripts | production/source_discovery_stage.py, production/qa_synthesis_stage.py, workshop_talk_discovery_windowed.py, reference_extractor.py, figure_extractor.py, scripts/presentations/generate_executive_*.py | large | reader (s06, s07, s17, c-pres-04) |
| 13 | Governance | Embedding model names are hardcoded outside the model registry | indexing/embeddings.py, indexing/embedding_resolver.py, indexing/walker.py, publishing/public_index.py:28-31 | medium | writer: public_index.py:28-31 read; others reader (s10) |
| 14 | Public readiness | The owner's personal name is hardcoded in shared contracts and defaults | presentation/deck_contract.py:185-186, presentation/brand_template.py:218-222, presentation/editorial_contract.py:98, presentation/structured_video.py (~1373), scripts/presentations/generate_render_spec.py:158, generate_deck_contract.py:334, :380, run_topic_presentation_build.py:1232 | medium | reader (s03, s02, c-pres-03, c-pres-05, c-pres-07) |
| 15 | Public readiness | Absolute home-directory paths in code defaults (mostly `--env-file`) | publishing/public_release.py:28, publishing/live_deployment.py:35, analysis/topic_strategy.py:54, scripts/run_iclr2026_production.py:178, :1571, scripts/recover_iclr2026_slideslive_media.py:22, scripts/presentations/run_phase13_rollout.py:35, run_topic_presentation_build.py:48, the smart_sync scripts, scripts/run_all_rankings.sh:7 | medium | reader (s11, s08, c-topconf-01/02, c-top-03/04/05, c-pres-07) |
| 16 | Public readiness | Drive folder names in code literals | presentation/commands.py:639-640, iclr2025_live.py:638, scripts/restore_drive_symlinks.py:176, :511 | small | reader (s03, s19, c-top-04) |
| 17 | Secrets | `gemini_deep_research.py` reads API keys from the shell profile | scripts/gemini_deep_research.py | small | reader (c-top-02) |
| 18 | Duplicate | The two presenter-video builders share about ten helpers | scripts/presentations/build_presenter_video.py, build_notebooklm_style_presenter_video.py | medium | reader (c-pres-02) |
| 19 | Duplicate | `run_topic_presentation_build.py` is an older parallel of the shared presentation chain; both define `write_qc_figure_feedback` | scripts/presentations/run_topic_presentation_build.py, production/presentation_chain.py | large | reader (s06, c-pres-07) |
| 20 | Dead code | 92 of 93 workshop-site downloaders are identical `NotImplementedError` stubs | scripts/conference/cvpr2026/workshop_site_downloaders/, scripts/conference/icml2026/workshop_site_downloaders/ | small | reader (c01, read all 93) |
| 21 | Dead code | `vision_scoring` has no caller, and the benchmark harness never writes the vision metrics that promotion checks | benchmarks/vision_scoring.py, benchmarks/promotion.py | medium | reader (s18) |
| 22 | Layering | Package code depends on scripts: the r6 media owner imports from `scripts/`, and the repair loop loads `md_to_revealjs.py` by path | acquisition/_cvpr_r6_media_owner_impl.py, presentation/repair_loop.py:551-558 | medium | reader (s04, s02) |
| 23 | Duplicate | Two local-LLM transports and two OCR implementations | extractions/backends.py vs model_governance/mlx_client.py; slide_ocr.py vs ocr_engine.py | medium | reader (s13, s16) |
| 24 | Split | Files that hold several capabilities | presentation/source_inventory.py (4,670 lines), presentation/programmatic_figures.py (4,533), iclr2025_live.py (18,939), podcast_notebooklm.py (3,284), figures/layouts/topology.py (2,606), cli.py, presentation/structured_video.py | large | reader; line counts writer |
| 25 | Hardcoding | Conference and content literals the hardcoding audit does not flag: paper and topic keyword triggers, CVPR family ids, NeurIPS prompts and default paths | presentation/programmatic_figures.py:2612-2690, source_discovery.py:89-99, scripts/exhaustive_extraction/prompts, production/proof_stage.py ("neurips-live-proven") | medium | reader (s01, s17, c02, s07) |

### 1. Blocked statuses drift between the ledger and the binding contract
`source_inventory.py` blocks `off_topic_artifact` sources from visible use; `source_binding.py`
keeps its own set without it, so a deck can bind a reference the ledger already rejected.
Suggest one shared constant. The binding tests would need a case with that status.

### 8. Shared code depends on the ICLR 2025 driver
`iclr2025_live.py` holds general helpers (`sanitize_for_receipt`, `utc_now`,
`run_exhaustive_topic_extraction`) that indexing, research, extraction and matching import,
and it imports them back. Moving the helpers into a neutral module would remove most
mutual-import pairs and let the driver be marked experimental.

### 14-17. Public-readiness items
These are the private markers that never reached the generated pages (they are in code,
not docstrings; Phase 1 test T8 checks the pages) but would ship in a public code export.
They belong to the milestone's public-readiness audit (Phase 2).

## Not confirmed
- **Refuted:** the reported duplicate `"render_export_validation"` key in
  `presentation/validation.py:100-101`. The key appears once.
- The 30 mutual-import pairs and 2 loops `review` prints are left as they are. Apart from the
  three figure pairs repaired above, they are imports in the code, not misplaced modules.
- Model literals without a `model-literal-ok` marker were reported in
  research/local_researcher.py:51-52, scripts/exhaustive_extraction/models.py:175 and
  slide_analysis/engine.py:243. The model-registry audit was not run in this phase, so
  whether it flags them is open.

## Searches tried
- "publish conference release to hugging face" -> publishing.release_cli_wrappers
- "upload to google drive backup" -> storage.drive_sync
- "check public url liveness" -> publishing.cdn
- "build public search index" -> publishing.public_index
- "estimate provider cost" -> costs.estimate_command (after the CVPR one-off entry stopped claiming no provider spend)
- "resume workflow from receipts" -> workflow.lineage
- "render narrated video" -> video.presenter_video_build (a 3/3 tie)
- "transcribe talk audio" -> media.speech_transcription
- "generate figure svg" -> figures.programmatic_svg
- "detect hardcoded conference names" -> audit.conference_hardcoding
- "run production stages" -> subsystem.production_run first; first component audit.orchestrator_share (a 3/3 tie, alphabetical)
- "train a reinforcement learning policy on a robot" -> component hits share one word at most; one module hit, src/conference_pipeline/conference_identity.py, shares two (learning, policy) through its docstring
