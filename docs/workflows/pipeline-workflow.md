# Conference Data + Analysis Workflow

This document describes the reusable conference workflow surface. NeurIPS 2025
is the historical reference workspace; ICLR 2025 is the v1.2 generic dry-run
proof workspace.

## Canonical Workflow CLI

The named workflow surface is the canonical operator entrypoint:

```bash
.venv/bin/python -m conference_pipeline.cli workflow list --workspace conferences/neurips2025
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages analyze,podcast,deck
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe --mode dry-run --run-id preview-run
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages transcribe --mode mock --run-id mock-transcribe
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe --run-id preview-run --resume
.venv/bin/python -m conference_pipeline.cli workflow resume --workspace conferences/neurips2025 --run-id preview-run --stages discover,download,transcribe
.venv/bin/python -m conference_pipeline.cli workflow summary --workspace conferences/neurips2025 --run-id preview-run
.venv/bin/python -m conference_pipeline.cli workflow unblock --workspace conferences/neurips2025 --run-id preview-run --blocker-path conferences/neurips2025/storage/workflow-runs/preview-run/blocker-receipts/discover/20260430T120000Z-local_llm_not_ready.json --operator operator-1 --note "manual ack"
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages deck_contract_generation --presentation-type topic-deep-dive --topic 01 --mode dry-run --run-id presentation-topic-preview
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages render_export --presentation-type master --mode dry-run --run-id presentation-master-preview
```

`.venv/bin/python -m conference_pipeline.cli run-all` is the legacy acquisition alias.

The workflow registry validates the conference workspace, exposes stable stage
ids, and delegates to the current stage implementations. Phase 11 adds
mode-aware receipts under
`conferences/<slug>/storage/workflow-runs/<run-id>/summary.json`; for example,
`conferences/neurips2025/storage/workflow-runs/preview-run/summary.json`.

Live provider stages must load credentials from the active worktree's
repository-local `.env` file. From the worktree root, run
`set -a; source .env; set +a` before invoking live
commands, or use scripts that explicitly load that file. Receipts and docs must
record only requirement names such as `env:GEMINI_API_KEY|GOOGLE_API_KEY`, never
secret values.

Dry-run records planned stage work without constructing live browser, network, or provider clients.
Mock mode records deterministic local success receipts without live browser, network, or provider calls.
Stage receipt resume skips only compatible succeeded receipts for the same workspace, stage, mode, topic, implementation, config, and command preview.
A compatible receipt must also carry `lineage` and `output_digest` whose `lineage_digest` matches what the stage would consume now: re-running an upstream stage with different outputs re-runs everything below it (v5.1 receipt lineage; see [Stage Owner Audit](../reference/stage-owner-audit.md#receipt-lineage)).
Existing workflow-runs/<run-id>/ directories are rejected unless --resume is set.
Every attempted stage writes a receipt, and failed stages persist summary.json before the command exits nonzero.

Use `workflow stage-status` to see, read-only, which stages of a run are
FRESH, STALE (and why -- which upstream moved, through which chain), MISSING
or BLOCKED, and the next stage to run; never delete receipts by hand to force
a re-run. Use `workflow bind-lineage --backfill` once to stamp lineage onto a
run written before v5.1 (dry run by default; `--apply` writes).
Use `workflow summary` to inspect the machine-readable `summary.json` for a
run. Use `workflow run --resume` or `workflow resume` to continue from stage
receipts; `workflow resume` requires `--stages` so operators explicitly choose
the stage subset to resume. Top-level resume uses presentation-status
checkpoints; workflow resume uses stage receipts.

#### Presentation per-step resume

Inside the presentation stage each per-deck step (source inventory,
blueprint, deck contract, figures, render, narration, review, video) is
skipped on `--resume` only while the deck ledger,
`decks/<deck_id>/step_lineage.json`, records it and every input it read and
every output it wrote still carries the recorded digest -- never because an
output file merely exists. Inputs are derived from the step's argv plus
declared implicit reads; in-place editors (bookend slides, delivery rebuild,
hash reconcile, the `fix_*` narration steps) append to a per-path writer
history so readers that ran before them keep skipping; sanctioned
operator-edit surfaces (`engaging_narration_raw.json`, the `*.ENGAGING.md`
drafts, a repaired blueprint) are adopted rather than regenerated, and only
their consumers re-run. A succeeded Stage 10 receipt is reused only while no
deck ledger reports drift. `--adopt-existing-outputs` bootstraps ledgers for
decks built before they existed (one-time; without it the first resume
re-runs every step, paid ones included). The ICLR 2026 and ICML 2026 runners'
per-command resume check uses the same ledger and no longer consults mtimes.

Halt and acknowledge: required stages that return `blocked`,
`deferred_blocked`, `failed`, `unavailable`, or `unsupported` write a
`BlockerReceipt`, persist `halt_status`, `halt_stage_id`, and
`blocker_receipt_path` on `summary.json`, and stop the autonomous run before
downstream stages. Provider availability failures, unavailable live owners,
missing/stale receipts, and failed required validation use the same fail-stop
contract. Resume after acknowledgement through
`.venv/bin/python -m conference_pipeline.cli workflow unblock ...` or
`python -m conference_pipeline.workflow_unblock ...`; refreshed summaries add
`unblock_acknowledgements` and switch `halt_status` to `resumed` once the
active blocker is cleared.

### Post-Full-Read Analysis Chain (ICML / CVPR Production Runners)

Once the Stage 14A LLM full-read extraction receipt is complete and
`workflow stage-status` reports it FRESH, the analysis sub-steps and Stage 7/8
run in this fixed order. Each step is its own command with its own receipt and
lineage; get operator cost approval before every paid step, and run each one by
hand (there is no supervisor script: `scripts/supervise_icml2026_post_14a.py`
is retired because it ran reconciliation before paper matching, forced
`--max-input-tokens 400000`, hardcoded model literals, and continued into
presentation).

1. **07A exhaustive topic summaries** -- `scripts/summarize_llm_full_read_topics.py`
   (dry-run first, then `--mode live --provider-cost-approved --resume`).
   **Stop point:** the model is the operator's choice, never a default. It
   comes from `workflow.exhaustive_summary_model` in the conference
   `config.yaml` (a catalog alias the registry allows for the task:
   `luna`, `kimi_openrouter`, `gemini_pro`, `sol`; add a
   `exhaustive_summary_model_decision` note) or from `--provider`; with
   neither the stage refuses before touching any receipt and asks
   (`conference_pipeline.analysis.summary_model_choice`). The receipt records
   `summary_model_choice`. The 2026-10-01 ICML 2026 topic-09 benchmark
   (`07A-model-benchmark-20261001.json`): Kimi via OpenRouter, the registry
   default, gives the densest packs at ~5x the latency; Luna, the alternative,
   is grounded, fast and ~40x cheaper than Gemini Pro; Gemini Pro is grounded
   but thin and the most expensive; GLM 5.3 was rejected. Beware that a dry run
   rewrites the live 07A receipt and `summary_plan.json`. Do not pass
   `--max-input-tokens`: the default chunk ceiling is tier-aware, kept below
   the selected model's high price tier from the pricing catalog
   (`resolve_max_input_tokens`). Forcing 400k made most ICML 2026 calls bill
   at the high-tier rate.
2. **07B deep-dive question selection** -- `scripts/select_deep_dive_questions.py`.
3. **07C dual deep-dive research lanes** -- `scripts/run_dual_deep_dive_research.py`.
   The engine comes from `workflow.research_engine` in the conference
   `config.yaml` (or `--engine`). The stage refuses when no engine has been
   chosen; it never picks a default.
4. **Per-topic paper matching** -- `scripts/match_papers_from_artifact_keypoints.py
   --topic-id <topic>` for every topic in `analysis/topic_strategy.json`.
5. **Stage 7 analysis, pass 1, without cost approval** -- run the conference
   runner (`scripts/run_<slug>_production.py --stage analysis`) without
   `--provider-cost-approved`. It costs $0 and rebuilds the source packs from the
   07A-07C outputs and matched papers. It is expected to stop on the lineage
   check because the 07D-07H receipts do not exist yet or are stale.
6. **07D-07H reconciliation and reference workflow** --
   `scripts/run_topic_reconciliation_reference_workflow.py` (local model). It
   reads and enriches the pass-1 source packs.
7. **Stage 7 analysis, pass 2, with cost approval** -- the same runner command
   with `--provider-cost-approved`: the paid narrative pass over the reconciled
   source packs.
8. **Stage 8 QA and gap-fill synthesis** -- `--stage qa-gap-fill-synthesis`.

Before each paid step, check `conference-pipeline workflow stage-status
--workspace <workspace> --run-id <run>`. The runners' pre-spend gate blocks
when an upstream is stale. That includes an upstream stage whose recorded
outputs were edited on disk after its receipt was written
(`upstream_output_changed`). In that case stage-status shows the edited stage
itself as STALE (`output_changed [paths]`), and that stage must be re-run.

### Analysis Contract Flow

The analyze stage produces `conferences/<slug>/analysis/analysis_manifest.json`
containing declared topics, paths, and digests. Podcast and deck stages read
this manifest as their source of truth: no filesystem globbing, no
topic-specific glue.

For already-populated conference workspaces, live analyze is preservation-first:
topic reports, coverage/mapping files, QA reports, and traceability addenda are
not overwritten by the workflow entrypoint. Use this path for UAT and receipt
refreshes; destructive regeneration requires an explicit Python API opt-in.

Deck generation is manifest-backed for workflow selection (topic enumeration and common-arg plumbing); direct script manifest consumption is deferred to a later phase.

`workflow deck` is the current deck command-construction stage, not full
presentation production. `workflow deck` remains a backward-compatible alias for
legacy deck command construction; it is not the full presentation lifecycle.
Full presentation production is the structured lifecycle that includes
presentation blueprint generation/review, deck contract generation, source
binding, visual planning, render specification, render/export, visual QC,
delivery contract generation, whole-presentation review, video generation,
video-sync validation, and human review checkpoints. First-class presentation stages are
presentation_blueprint, deck_contract_generation, source_binding,
visual_plan_generation, render_spec_generation, render_export, visual_qc,
delivery_generation, presentation_review, video_generation,
video_sync_validation, and human_review.
First-class presentation stages now provide workspace-aware command planning,
safe dry-run script contracts, structural validation, and receipt-backed proof
reports; live cross-conference production still requires conference-local
proof. Presentation Blueprint Contract: `presentation_blueprint` must first
consume `presentation_frame.json` through `presentation.generate_editorial_blueprint`.
The frame separates retrieval topic title from public `audience_title`, records
a `field_thesis`, and supplies broad `general_questions` so the opening section
establishes area-level stakes before specialist paper mechanisms. The blueprint
stage must produce `presentation_blueprint.json`,
`presentation_blueprint_receipt.json`, `presentation_frame_validation`, and
`blueprint_review.json`; production `deck_contract_generation` must consume the
approved blueprint only after `approved_for_draft_generation: true`, current
`blueprint_sha256`, passing frame validation, and gate type: deterministic,
gate type: LLM-judge, and gate type: human evidence are recorded. mock/dry-run
blueprint artifacts are planning/test evidence only. Topic blueprints target
<=38 min and block downstream plans above >40 min before TTS; Bulk regeneration
is blocked by the current canary rule: No bulk presentation regeneration before
approved 04_safety_alignment_privacy canary.

The production artifact chain is `presentation_frame.json`, `presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `render_spec.json`,
`delivery_single_speaker.json`, `delivery_notebooklm.json`,
`6_presenter_script.md`, `video_sync_validation_single_speaker.json`, and
`video_sync_validation_notebooklm.json`. Old numbered slide draft Markdown files
are legacy-only numbered Markdown: they may remain as
debug-export-only artifacts with `production_evidence: false`, but they cannot
serve as production proof. Production plans, manifests, receipts, Phase 13
evidence, generated-plan audits, and stage-owner docs must treat numbered
Markdown proof as a blocked legacy path and report
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED` when it is used as production
evidence. First-class presentation `visual_plan_generation` must use
programmatic matplotlib/Plotly/svgwrite SVG owners, write source-grounded
FigureSpec records, and record Anthropic `claude-sonnet-5` as the governed
code/spec generation model; paper-figure extraction scripts and generic
bullet-box SVGs are deprecated and must not be used as completion evidence.
Renderer-only repairs after a live FigureSpec receipt may reuse
`figure_specs.json` with `--figure-specs-input`, but must preserve the Sonnet
receipt, mark `live_sonnet_specs_reused_local_render`, clean stale SVGs, and
rerun structural plus semantic validation. First-class presentation validation
additionally includes semantic figure review before render and visual QC.
First-class presentation validation stages are
presentation_blueprint_validation, deck_contract_generation_validation,
source_binding_validation, visual_plan_generation_validation,
render_spec_generation_validation, render_export_validation,
visual_qc_validation, delivery_generation_validation,
video_generation_validation, video_sync_validation_receipt, and
human_review_validation.

Presentation workflow runs use `--presentation-type` with the existing
`--topic` selector. `--presentation-type topic-deep-dive` requires `--topic`;
`--presentation-type master` and `--presentation-type executive-summary`
select a single manifest entry without a topic. Presentation stage receipts
include selected scope and presentation_manifest digests, and resume
compatibility binds the selected scope, manifest content, config, and command
preview.

Phase 24 command planning resolves `decks/presentation_manifest.json`, selects
one manifest entry from `--presentation-type` and `--topic` where required,
and builds workspace-aware argv for the core presentation owner scripts.
Workflow-planned presentation scripts use decks/<presentation>/... paths; presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning.

Presentation stage dry-run receipts include command_metadata with exact argv, selected scope, input artifacts, output artifacts, support level, provider/tool requirements, and unsupported_reason.
Unsupported reusable stages expose unsupported_reason instead of claiming script execution.
Video generation is dry-run/planning-only in Phase 24 and does not prove final MP4 generation, visual validation, or cross-conference proof.
The approved command owners and per-tool usage documentation for presentation,
analysis, podcast, backup, and verification scripts are registry-owned in the
[Stage Script Registry Tool Usage Index](../reference/stage-owner-audit.md#stage-script-registry-tool-usage-index).
Generated plans must cite those `usage_doc_refs` anchors when naming required
tools. When a registry tool blocks, agents must read its emitted
`diagnostic.next_actions` from the registry-owned `diagnostic_profile` before
retrying or moving to the next stage.

Presentation validation checks are stable by `check_id`:
`slide_mapping_schema`,
`slide_mapping_markdown_count_equals_mapping`,
`slide_mapping_revealjs_section_count_equals_mapping`,
`presenter_script_slide_ids_resolve`,
`hyperlink_density_meets_target`,
`hyperlink_preservation_4_5_to_5`,
`svg_density_meets_target`,
`svg_well_formed`,
`pdf_parses`,
`pdf_page_count_matches_target_slides`,
`video_artifacts_nonempty`,
`video_duration_readable`,
`video_stream_durations_match`,
`video_duration_matches_build_receipts`,
`video_tail_not_black`,
`visual_qc_report_present`,
`vision_llm_visual_qc_passed`, and
`revealjs_html_parses`,
`revealjs_reference_labels_visible`,
`revealjs_title_slide_branded`, and
`revealjs_content_brand_mark_present`.

Use `conference-pipeline presentation-validate` when an operator wants the
full validation harness for one manifest-selected presentation. Individual
validation stages can also be selected through `workflow run --stages <stage>`.

Phase 26 records receipt-backed presentation proof evidence through
`conference-pipeline presentation-proof`. The NeurIPS reference command is
`conference-pipeline presentation-proof --workspace conferences/neurips2025 --presentation-type topic-deep-dive --topic 01_tools_and_open_source --mode dry-run --run-id <run-id>`.
The ICLR scaffold routing command is
`conference-pipeline presentation-proof --workspace conferences/iclr2025 --presentation-type topic-deep-dive --topic 01_iclr_2025_proof --mode dry-run --stages deck_contract_generation --run-id <run-id>`.
Proof reports are written to
`conferences/<slug>/storage/workflow-runs/<run-id>/presentation-proof-report.json`;
the consolidated proof summary is
`.planning/milestones/v1.3-phases/26-neurips-backfill-and-iclr-presentation-proof/26-PROOF-SUMMARY.md`.
Committed Phase 26 run ids are `phase26-neurips-presentation-proof`,
`phase26-iclr-presentation-dryrun`, and
`phase26-iclr-presentation-live-denied`.

Examples:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages render_export --presentation-type master --mode dry-run --run-id presentation-master-preview
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages video_generation --presentation-type master --mode dry-run --run-id presentation-video-preview
conference-pipeline presentation-validate --workspace conferences/neurips2025 --presentation-type topic-deep-dive --topic 01_tools_and_open_source --mode dry-run --run-id phase25-presentation-validate --summary
```

Podcast and deck stages fail closed when analysis_manifest.json is missing, schema_version is incompatible, or any declared path does not exist.

Podcast and deck resume skips only succeeded receipts whose recorded manifest_content_sha256 matches the current manifest.

For a legacy workspace, backfill the manifest with:

```bash
python3 scripts/backfill_analysis_manifest.py --workspace conferences/neurips2025
```

See the [Analysis Contract Runbook](../operations/analysis-contract.md).

The workflow registry and non-live receipt path are conference-general for any
workspace with `config.yaml` and `storage/storage_manifest.json`. Live stage
support still varies by stage and conference; consult the local
`.planning/milestones/*-CONTEXT.md` packet before claiming live support.

### Acquisition Cache And Embedded Media Recovery

Stage-04 acquisition may persist source-page and item-page probes under
`conferences/<slug>/cache/acquisition-downloads/`. These files are
workspace-local acquisition evidence, not old-conference fixtures: they are
created by the live runner for the selected workspace and are excluded from git
unless a small control file is explicitly whitelisted.

For ICLR-style sites, paper/session item pages can expose embedded SlidesLive
players after authenticated page probing. If `download_corpus.json` contains
successful `item_page_auth_probe` artifacts but no merged local media artifact,
use the registry-owned `acquisition.slideslive_media_recovery` component before
advancing to speech transcription. The recovery scans only cached item-page
probe HTML, strips secret-bearing embed query parameters from receipts,
downloads or resumes only the corresponding SlidesLive audio, merges those
audio records into `storage/download-corpus/download_corpus.json`, and writes
`storage/download-corpus/slideslive_media_recovery.json`.

Slides are a separate acquisition lane, not proof that audio recovery is
complete. When cached `item_page_auth_probe` HTML exposes SlidesLive embeds and
the conference plan needs visual/OCR evidence, run the registry-owned
`acquisition.slideslive_slide_download` owner before visual OCR/indexing. The
owner uses SlidesLive `slides.json` first, falls back to yt-dlp thumbnail
metadata for `/slides/original/` images, can use `--browser-fallback` for
browser-intercepted slide image metadata when structured sources fail, writes
per-presentation `sync.json` files, merges slide images into `download_corpus.json`, and writes
`storage/download-corpus/slideslive_slide_recovery.json`.

After slide download, run `acquisition.slide_reference_detection` and
`acquisition.slide_cleanup` so repeated generic/banner slides and consecutive
duplicates are removed from OCR/indexing by non-destructive `skip_ocr` markers.
Raw slide images remain in the workspace for audit. Poster thumbnails or paper
PDFs are rejected evidence for SlidesLive slide acquisition.

Both focused recoveries must preserve these boundaries: no source pages, paper
PDFs, workshop PDFs, poster images, audio, video, or unrelated media families
are redownloaded outside the requested lane, and no provider calls are made. Use
`--inventory-only` when an operator needs to audit embedded SlidesLive ids
without downloading assets or mutating the corpus inventory. The audio wrapper
is ICLR 2026-specific; the slide wrapper is conference-general for workspaces
with cached item-page probes.

## Generic Conference Smoke Test

Use the generic proof command for a new conference before live provider work:

```bash
conference-pipeline conference-smoketest --workspace conferences/iclr2025 --stages analyze,deck --topic 01 --mode dry-run --run-id phase20-iclr2025-smoketest --summary
```

The command writes:

- `conferences/<slug>/storage/workflow-runs/<run-id>/summary.json`
- `conferences/<slug>/storage/workflow-runs/<run-id>/conference-smoketest-report.json`

Use `analyze,podcast,deck` only when `analysis_manifest.json` declares the
topic-level ranks and audio paths required by the podcast adapter. See the
[Conference Analysis Playbook](conference-analysis-playbook.md).

## Overview

The workflow has two tracks:

Data acquisition (Steps 1-7):
1. **Discover** sessions from the NeurIPS virtual site
2. **Filter** sessions manually by editing JSON
3. **Download** audio and slide images
4. **Transcribe** audio
5. **Clean slides** (optional but recommended)
6. **OCR** slides using vision LLMs
7. **(Optional) Reference extraction + paper download** (not required for analysis plan)

Analysis and synthesis (Steps 8-14):
8. **Enhanced indexing** with smart chunking + entity tracking
9. **Title-spine topic strategy** (three-model consensus to exactly 12 topics)
10. **Query planning** (topic questions + synonym expansion)
11. **Topic execution** using GPT Researcher + local search
12. **Traceability + QA** (timestamps, slide IDs, source gaps)
13. **Coverage analysis** and gap filling
14. **Synthesis** (executive summary + trend report)
15. **Optional pipelines** (exhaustive extraction, podcast, paper-level analysis)

## Step-by-Step Workflow

### Step 1: Initial Setup

```bash
# Clone/download the repository
cd neurips-foundation-conf-pipeline

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies
brew install ffmpeg playwright  # macOS
# or: apt-get install ffmpeg  # Linux

# Install Playwright browsers
playwright install chromium

# Configure settings
cp .env.example .env
# Recommended for the NeurIPS 2025 acquisition runtime:
# STORAGE_ROOT=./data/neurips_2025
# OPENAI_API_KEY=...
# TAVILY_API_KEY=...
```

The committed workspace at `conferences/neurips2025/` owns the lane-based
`analysis/`, `audio/`, and `decks/` artifacts. Acquisition state such as
`sessions.json`, `metadata.sqlite`, and runtime logs lives under
`$STORAGE_ROOT`.

### Step 2: Authentication Setup

To access protected NeurIPS content, extract your session cookie:

```bash
# Run the interactive cookie extraction helper
python3 conferences/neurips2025/scripts/create_cookies_file.py
```

Follow the prompts:
1. Log in to https://neurips.cc in your browser
2. Open Developer Tools (F12)
3. Go to Application tab → Cookies → https://neurips.cc
4. Find and copy the `sessionid` cookie value
5. Paste it when prompted by the script

This creates `cookies.json` which the pipeline will use automatically.

**Detailed instructions**: See `conferences/neurips2025/scripts/get_cookies_instructions.md`

### Step 3: Discover Sessions

Discover all available sessions from NeurIPS 2025:

```bash
python3 scripts/discover_sessions.py \
  --year 2025 \
  --output "$STORAGE_ROOT/sessions.json"
```

**What this does:**
- Logs in to NeurIPS using your cookies
- Scrapes the virtual schedule for all session types:
  - Tutorials
  - Workshops
  - Keynotes
  - Oral presentations
  - Awards (Hinton, Test of Time)
  - Competitions
- Saves results to `sessions.json`

**Output:** `$STORAGE_ROOT/sessions.json` with discovered sessions

### Step 4: Manual Filtering

Review and filter the discovered sessions:

```bash
# Open sessions.json in your text editor
nano "$STORAGE_ROOT/sessions.json"  # or vim, vscode, etc.
```

For each session, set the `chosen` field:
- `"chosen": "Y"` - Download this session
- `"chosen": "N"` - Skip this session
- `"chosen": null` - Undecided (treated as "N")

**Example:**
```json
{
  "id": "128329",
  "title": "Sejnowski-Hinton Award",
  "session_type": "hinton_award",
  "url": "https://neurips.cc/virtual/2025/award/128329",
  "chosen": "Y"   ← Set this field
}
```

**Tip:** You can use the helper scripts in `scripts/helpers/` to batch-update sessions by category.

### Step 5: Download Assets

Download audio, slide videos, and extract slide images for all chosen sessions:

```bash
python3 scripts/download_from_sessions.py \
  --sessions-file "$STORAGE_ROOT/sessions.json"
```

**What this does for each chosen session:**

1. **Find the presentation** - Navigate to the session URL and locate the SlidesLive video embed
   - For oral presentations: automatically follows nested links (oral → poster page) and handles timezone redirects
   - Performs intelligent page reloads to trigger lazy-loaded SlidesLive embeds
   - Extracts final presentation URL with location-specific path (e.g., `/loc/san-diego/poster/...`)
2. **Download audio** - Extract MP3 audio (~104MB, full presentation audio)
3. **Scrape slide images** - Use browser automation to navigate through ALL slides in the SlidesLive player, screenshot each one, and crop out player controls (bottom 80px)
4. **Save metadata** - Track all downloads with provenance (source URLs, timestamps)

**Note:** The scraper captures complete slide decks (100-200+ slides for keynotes/orals) by clicking through the SlidesLive player, ensuring we get all slides instead of just a subset.

**Progress tracking:**
- Console output shows current presentation
- `download_progress.log` - Full console output
- `$STORAGE_ROOT/logs/session_asset_download.log` - Detailed structured logs

**Resumability:** If interrupted, simply re-run the command. Already-downloaded presentations are skipped.

**Testing:** Use `--limit N` to test with first N presentations:
```bash
python3 scripts/download_from_sessions.py \
  --sessions-file "$STORAGE_ROOT/sessions.json" \
  --limit 5
```

### Adding transcripts after the harvest (evidence lanes)

Captions often appear on YouTube a day or more after a harvest, and ASR
fallback runs finish later still. Do not edit the base caption inventory and
do not leave recovered transcripts in a side directory. Bind them as a
transcript evidence lane and rebuild the corpus inventory:

```bash
# 1. The recovery run writes its own receipt (example: CVPR 2026 r7 recheck).
.venv/bin/python scripts/recover_cvpr2026_r7_captions.py --finalize-existing

# 2. Build digest-bound lane receipts and register them (conference-local binder
#    calling conference_pipeline.indexing.transcript_evidence_lanes).
.venv/bin/python scripts/build_cvpr2026_transcript_evidence_lanes.py --dry-run
.venv/bin/python scripts/build_cvpr2026_transcript_evidence_lanes.py

# 3. Re-run the corpus inventory + index stage; lanes are admitted after the base
#    inventory and reported under checks.transcript_evidence_lanes.
.venv/bin/python scripts/run_cvpr2026_production.py --stage corpus-inventory-index --resume
```

Every downstream analysis receipt binds the index digest, so re-run the
analysis stages that should see the new transcripts afterwards.

Harvest runs from `scripts/harvest_youtube_talks.py` (a directory with
`video_transcript_inventory.json`) use the conference-neutral binder instead
of a conference-local script. It writes one lane per transcript tier
(`<label>-caption`, `<label>-gemini`), normalizes rolling auto-caption cues,
materializes timestamped segment JSON in the lane, turns unavailable, failed,
and speechless videos into coverage gaps, and registers caption lanes ahead of
ASR lanes and model-transcription lanes last:

```bash
.venv/bin/python scripts/build_youtube_harvest_evidence_lanes.py \
  --workspace conferences/<slug> \
  --harvest-dir storage/youtube-talks/<run-a> \
  --harvest-dir storage/youtube-talks/<run-b> --dry-run
# same command without --dry-run binds and registers the lanes
```

Long recordings that hold several talks (workshop streams, oral sessions,
tutorials) are split before indexing, so each chunk names its talk. The split
keeps one artifact per recording: an annotated transcript whose segments
carry `talk_id`, `talk_title`, `talk_speaker`, and workshop fields, which the
chunker turns into talk-attributed chunks with recording timestamps.

```bash
set -a; source .env; set +a
# 1. cost estimate (no provider call); needs operator approval to proceed
.venv/bin/python scripts/discover_harvest_talk_boundaries.py --workspace conferences/<slug> --estimate
# 2. windowed discovery with the governed boundary-discovery model; halts before the task budget cap
.venv/bin/python scripts/discover_harvest_talk_boundaries.py --workspace conferences/<slug> --provider-cost-approved --resume
# 3. spot-check the talk maps
.venv/bin/python scripts/discover_harvest_talk_boundaries.py --workspace conferences/<slug> --report-only
# 4. bind the talk maps into the lanes (re-pins the lane receipts), then re-run Stage 6
.venv/bin/python scripts/build_youtube_harvest_evidence_lanes.py --workspace conferences/<slug> \
  --harvest-dir storage/youtube-talks/<run-a> --attach-talk-maps
```

Candidates come from the corpus inventory, not only from harvest lanes: every
admitted speech transcript (base caption inventory, r-lanes, harvest lanes;
segment JSON and caption files alike) gets an eligibility decision in the
estimate receipt, and a long recording that cannot be split (no timing) is
listed rather than skipped silently. Talk maps are written next to the
transcript: a lane's `talks/` directory, or
`storage/transcription-ocr/talk-maps/base/` for base captions. Stage 6
attaches any map whose chain verifies (transcript digest -> talk map ->
annotation `talks_sha256`, same `video_id`) and blocks on a stale one; harvest
lanes that pin maps in their lane receipt are re-bound with step 4.

Recordings whose transcripts carry no timing at all (plain-text SlidesLive
transcripts, for example) are listed as untimed by that estimate. Split them
by event instead, from the site's own event pages:

```bash
set -a; source .env; set +a
# estimate (no provider call): choose recordings by event kind and/or item id
.venv/bin/python scripts/split_event_recordings.py --workspace conferences/<slug> \
  --kind oral --kind affinity-event --item-id <event-item-id> --estimate
# run with the same selection after operator approval; --resume skips current maps
.venv/bin/python scripts/split_event_recordings.py --workspace conferences/<slug> \
  --kind oral --kind affinity-event --item-id <event-item-id> --provider-cost-approved --resume
```

When an event page links its talks (an oral session lists its orals in
program order), those titles go to the model as the program and each talk it
finds is named with its scheduled title verbatim, which is what indexing and
embedding key on. Untimed transcripts get evenly timed one-minute segments from
the audio duration, so a boundary falls within a minute. Output is the same
verified base-scope talk map as above; a map made for a changed transcript
never attaches and blocks Stage 6 instead. A talk map supersedes an older
inline workshop split for the same recording.

Discovery tiles each recording into 90-minute windows with 15-minute overlap
and merges window candidates. A distinct named talk, keynote, or panel is kept
down to 30 seconds (lightning talks); only unnamed fragments under two minutes
fold into a neighbour, and a gap over a minute becomes an unattributed segment
rather than silently extending the previous talk. Model-transcribed recordings
with only coarse window timing get synthetic cues weighted by characters, so
unspaced scripts are timed sensibly. A workshop is stamped onto talks only
when the recording's event page names it; a fuzzy title match is passed to
the model as an unverified hint and never becomes metadata. Maps are validated
for coverage, talk length, and titles; any failure, budget halt, or missing
approval falls back to one whole-recording talk, which is not attached. Every
map stores the raw per-window candidates and a `discovery_algorithm_version`;
`--resume` reuses only a complete map of the current version (receipt written
last, all digests matching). Because indexing is incremental, attaching talk
maps later re-embeds only the recordings that were split.

A conference's main proceedings can be split across listings: CVPR 2026
published 942 Findings-track papers at `openaccess.thecvf.com/CVPR2026_findings`,
separately from the 4,068-paper main listing the Stage 4 ledger enumerated, and
the whole track was missing until a poster-vs-index audit found it. Check the
proceedings host's menu for extra tracks, then download each as a proceedings
source (papers already in the corpus by title are skipped as duplicates):

```bash
.venv/bin/python scripts/download_cvf_proceedings_papers.py --workspace conferences/<slug> \
  --source-id cvf_findings --track findings --venue "<Conference YYYY> Findings" \
  --listing-url "https://openaccess.thecvf.com/<Conf>_findings?day=all"
```

Stage 6 admits registered tracks (`storage/papers/proceedings_paper_sources.json`,
receipts pinned by sha256) as main-conference papers (`source_family_id:
paper_text`, `conference_scope: main_conference`) carrying `venue` and
`venue_track`; their chunks add a `Venue:` line under the title.

Before indexing, Stage 6 curates the paper records
(`conference_pipeline.indexing.paper_corpus_curation`):

- An image-only PDF, one whose extracted text is under 5,000 bytes (only the
  publisher watermark or figure labels), is excluded as
  `image_only_text_layer`.
- Copies of one paper, by identical PDF bytes or normalized title, keep one
  canonical copy: main track first, then supplementary proceedings tracks,
  then workshop copies. The kept copy lists the others under
  `also_presented_at`.

Excluded papers stay in `corpus_inventory.json` as audit-only records
(`indexable_text: false`, `index_exclusion: {reason, kept_artifact_id}`), and
the inventory's `paper_curation` block lists every exclusion. An incremental
run deletes their chunks and embeds nothing.

Model transcripts can be fluent and invented: on CVPR 2026, silent poster clips
came back from Gemini as recipes and product reviews. Reviewed rejects go in
`storage/transcription-ocr/evidence-lanes/transcript_exclusions.json`
(`video_id`, `reason`, `evidence`), which the harvest lane binder turns into
`excluded:<reason>` coverage gaps; the binder also lists short model
transcripts that share no word with their title as `suspect_model_transcripts`
for review (flagged, not excluded).

Workshop papers whose source receipt has no authors can be completed with
`scripts/recover_paper_authors_by_title.py --workspace conferences/<slug>`
(free arXiv and Crossref title lookups). A hit is accepted only when the
normalized titles match and at least half of the surnames appear on the
paper's own first page; Stage 6 applies accepted entries with provenance
`title_lookup_verified` and binds the recovery file by digest.

## Output Structure

### Directory Layout

Presentation directories use human-readable slugified titles (e.g., `sejnowski-hinton-award`) instead of numeric IDs for easier navigation. Examples below assume `STORAGE_ROOT=./data/neurips_2025` (set in `.env`).

```
$STORAGE_ROOT/
├── presentations/
│   ├── sejnowski-hinton-award/
│   │   ├── audio/
│   │   │   └── raw.mp3           # 104MB, ~2hr audio
│   │   ├── slides/
│   │   │   ├── images/
│   │   │   │   ├── slide_001.png  # Clean slide 1 (browser screenshot)
│   │   │   │   ├── slide_002.png  # Clean slide 2
│   │   │   │   └── ...            # 195 images (~50MB total, full deck)
│   │   │   └── sync.json          # Slide timestamps for synchronization
│   │   ├── transcripts/
│   │   │   ├── transcript.txt     # Plain text transcript
│   │   │   ├── transcript.json    # TranscriptChunk list (pipeline output)
│   │   │   └── segments.json      # Optional segments (transcribe_sessions.py + whisper-1)
│   │   ├── ocr/
│   │   │   ├── slide_001.json     # Structured OCR output
│   │   │   ├── slide_002.json
│   │   │   └── ...
│   │   └── meta.json              # Presentation metadata
│   ├── building-foundational-models-for-robotics-at-tesla/
│   ├── beyond-benchmarks-rethinking-reasoning-in-language-models/
│   └── ...                  # More presentations
├── reference_slides/            # Generic slide patterns
│   ├── reference_01.png         # Detected reference slide
│   └── metadata.json            # Detection metadata
├── logs/
│   ├── session_asset_download.log
│   ├── transcription.log
│   └── ocr.log
├── transcription_status.json    # Tracks transcription progress
├── ocr_status.json              # Tracks OCR progress
└── metadata.sqlite              # SQLite database
```

**sync.json format** (includes skip flags after cleanup):
```json
{
  "slides": [
    {
      "index": 1,
      "image": "slide_001.png",
      "timestamp_sec": 0.0,
      "skip_ocr": true,
      "skip_reason": "generic_conference_slide"
    },
    {
      "index": 2,
      "image": "slide_002.png",
      "timestamp_sec": 60.5
    }
  ]
}
```

**Directory naming**: Titles are converted to lowercase slugs with special characters replaced by hyphens, truncated to 80 characters max.

### Database Schema

The SQLite database (`metadata.sqlite`) tracks core pipeline artifacts. Key tables:
- **presentations** - Metadata + status
- **audio_files** / **video_files** - Media with provenance
- **slide_images** - Slide images + timestamps
- **transcripts** - transcript.txt + transcript.json (pipeline), optional segments.json (scripts)
- **ocr_results** - OCR output per slide
- **reference_candidates** - Detected reference candidates
- **papers** / **presentation_papers** - Resolved papers and links

For the canonical schema, see `src/conference_pipeline/storage.py`.

### Status Tracking

Each presentation has a status in the database:
- `discovered` - Found on NeurIPS site
- `assets_downloaded` - Audio + slides downloaded
- `transcription_in_progress` - Partial transcription
- `transcribed` - Audio transcribed
- `slides_extracted` - Slides extracted
- `ocr_done` - Slides OCR'd
- `references_extracted` - References extracted
- `papers_downloaded` - Papers downloaded
- `error` - Processing failed (check logs)

Query status:
```bash
sqlite3 "$STORAGE_ROOT/metadata.sqlite" "SELECT status, COUNT(*) FROM presentations GROUP BY status;"
```

**Progress tracking files:**
- `transcription_status.json` - Tracks which presentations have been transcribed
- `ocr_status.json` - Tracks which presentations have been OCR'd

These JSON files enable resume functionality - re-running scripts skips completed work.

## Storage Requirements

Rough order-of-magnitude estimates (varies by slide count and audio length):
- **Audio**: ~80-120MB per presentation
- **Slide images**: ~30-120MB per presentation (full decks)
- **OCR JSON**: ~5-20MB per presentation
- **Total for 129 presentations**: plan for ~20-40GB if keeping all artifacts

**Note:** Slide images are PNG screenshots captured via browser automation, providing complete slide decks instead of just a subset.

## Performance

**Per-presentation timing:**
- Audio download: ~8 seconds
- Slide scraping: ~2-5 minutes (depends on number of slides, ~0.5 seconds per slide)
- **Total**: ~3-6 minutes per presentation

**For 46 presentations:**
- Estimated time: ~2.5-4.5 hours
- Timing varies based on slide counts (keynotes with 200 slides take longer than workshops with 50 slides)

## Troubleshooting

### Authentication Failures

**Symptom:** "403 Forbidden" errors, "No SlidesLive ID found"

**Solution:**
1. Verify you're logged in to https://neurips.cc in your browser
2. Extract fresh cookies: `python3 conferences/neurips2025/scripts/create_cookies_file.py`
3. Ensure `cookies.json` exists in project root
4. Check cookie expiration (re-extract if needed)

### Timeouts

**Symptom:** "Timeout 30000ms exceeded"

**Solution:** These are usually harmless - the download proceeds using direct URLs even if browser scraping times out.

### Disk Space

**Symptom:** "No space left on device"

**Solution:**
- Free up ~6GB of disk space
- Or reduce the number of chosen sessions
- Or delete slide videos after extraction

### Download Failures

**Symptom:** Presentation marked as "error" in database

**Solution:**
1. Check logs: `$STORAGE_ROOT/logs/session_asset_download.log`
2. Re-run download script (it will retry errors)
3. For specific presentation, check if video is available on NeurIPS site

### Step 6: Transcribe Audio

Transcribe all downloaded audio using the configured transcription provider:

```bash
python3 scripts/transcribe_sessions.py
```

**What this does:**
- Finds all presentations with downloaded audio
- Automatically chunks large files (>25MB) into ~10 minute segments
- Transcribes using OpenAI's `gpt-4o-transcribe-diarize` model by default for speaker labels
- Generates segments.json only when the transcription provider returns segments (whisper-1)
- Saves transcript.txt and, when available, timestamped segments

**Options:**
```bash
# Process in parallel (faster)
python3 scripts/transcribe_sessions.py --parallel 3

# Retry failed transcriptions
python3 scripts/transcribe_sessions.py --retry-errors

# Test with first N presentations
python3 scripts/transcribe_sessions.py --limit 5
```

**Output per presentation:**
- `transcripts/transcript.txt` - Plain text transcript
- `transcripts/segments.json` - Optional timestamped segments with start/end times

**Note:** The pipeline (`neurips-pipeline transcribe`) writes `transcripts/transcript.json`
with chunk metadata (and embedded segment timestamps when available). The standalone
`transcribe_sessions.py` script writes `segments.json` only when Whisper returns
segment timestamps (typically `whisper-1`).

**Resumability:** Re-running skips already-transcribed presentations.

**Cost estimate:** Use `.venv/bin/python -m conference_pipeline.costs estimate --workspace conferences/<slug> --scope full --scenario base` before running paid provider work.

### Step 6.5: Clean Up Slides

Before running OCR, mark duplicate and generic slides for skipping to reduce API costs
and avoid indexing the same conference banner everywhere:

```bash
# Detect generic conference slides automatically from a generic slide root
python3 scripts/detect_reference_slides.py \
  --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset \
  --output-dir conferences/<slug>/storage/download-corpus/reference_slides \
  --receipt-path conferences/<slug>/storage/download-corpus/slide_reference_detection.json

# Mark duplicates and generic slides in all presentations
python3 scripts/cleanup_slides.py \
  --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset \
  --workspace conferences/<slug> \
  --download-corpus-path storage/download-corpus/download_corpus.json \
  --receipt-path conferences/<slug>/storage/download-corpus/slide_cleanup.json
```

**What this does:**
- Detects recurring conference title/banner slides across presentations
- Identifies consecutive duplicate slides (common in SlidesLive recordings)
- Marks slides with `skip_ocr: true` in `sync.json` and download-corpus artifacts
  (does NOT delete any files)

**Typical savings:** ~34% fewer slides to OCR, significantly reducing API costs.

**Preview first:**
```bash
python3 scripts/cleanup_slides.py --dry-run --verbose \
  --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset
```

### Step 7: OCR Slides

Run OCR on all slide images using vision LLMs:

```bash
python3 scripts/ocr_slides.py
```

**What this does:**
- Processes slide images using a configured vision model
- Uses transcript context to improve understanding (syncs slide timestamps with transcript segments)
- Extracts structured content: title, bullet points, equations, visual elements, key concepts, references
- Outputs JSON files with rich metadata

**Options:**
```bash
# Process slides in parallel (faster)
python3 scripts/ocr_slides.py --parallel 5

# Use the default production primary model
python3 scripts/ocr_slides.py --provider openai --model gpt-5.6-luna

# Use Gemini only as a quality fallback for low-confidence/failed primary parses
python3 scripts/ocr_slides.py --fallback-provider gemini --fallback-model gemini-3.1-flash-lite-preview

# Use Claude directly
python3 scripts/ocr_slides.py --provider anthropic

# Retry failed OCRs
python3 scripts/ocr_slides.py --retry-errors

# Test with first N presentations
python3 scripts/ocr_slides.py --limit 5
```

**Output per presentation:**
- `ocr/slide_001.json`, `ocr/slide_002.json`, etc.

**JSON structure:**
```json
{
  "slide_index": 5,
  "timestamp_sec": 60.56,
  "slide_text": {
    "title": "Slide Title",
    "bullet_points": ["Point 1", "  - Sub-point"],
    "equations": ["\\alpha + \\beta"],
    "urls": ["https://example.com"]
  },
  "visual_elements": [
    {"type": "chart", "description": "...", "labels": [...]}
  ],
  "key_concepts": [
    {"concept": "Diffusion Models", "explanation": "..."}
  ],
  "references": [
    {"title": "Paper Title", "authors": "...", "venue": "NeurIPS", "year": "2025"}
  ]
}
```

**Resumability:** Re-running skips already-processed presentations.

Before live OCR, run the model experiment in dry-run mode and then in live mode
only after provider/budget approval:

```bash
.venv/bin/python -m conference_pipeline.ocr_benchmark run \
  --workspace conferences/<slug> \
  --sample-size 12 \
  --budget-cap-usd 30 \
  --per-call-timeout-seconds 60 \
  --json
```

The experiment compares cheaper OpenAI, Anthropic, and Gemini image parsers
against an OpenAI authority (`gpt-5.6-sol`, falling back to `gpt-5.6-terra`) for cost,
latency, JSON validity, schema completeness, locator/numeric recall, and text
similarity. Receipts are written under
`storage/transcription-ocr/ocr-model-benchmark/`.

## Provider Cost Gates

Before any paid provider work, estimate the full provider-backed pipeline, not
only transcription and OCR. The canonical registry is
`src/conference_pipeline/costs/pricing_catalog.json`, refreshed on
`2026-04-26`, and task coverage lives in
`src/conference_pipeline/costs/task_specs.yaml`.

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/iclr2025 \
  --scope full \
  --scenario base \
  --json
```

For low/base/high model scenarios:

```bash
.venv/bin/python scripts/estimate_pipeline_costs.py \
  --workspace conferences/iclr2025 \
  --compare-models
```

The full-scope estimate includes ASR transcription, visual OCR, PDF text
follow-up, exhaustive extraction through direct providers and OpenRouter
failover, GPT Researcher LLM calls, embeddings, Tavily search/extract credits,
topic reports, traceability recovery, ranking passes, podcast scripts/TTS, deck
generation, enrichment, consolidation, figure captioning, programmatic
Plotly/Matplotlib/svgwrite SVG figure code generation and repair, presenter
scripts, NotebookLM-style dialogue, and presenter-video TTS.

Local-only downloads, browser rendering, `ffmpeg`, Tesseract, local PDF text
extraction, local Python/Plotly/Matplotlib/svgwrite SVG rendering, and ordinary
filesystem work are reported as `$0 API cost` line items. Historical NeurIPS
ASR/OCR examples in older receipts remain historical evidence; new approvals
should cite the full-pipeline estimate output and the pricing registry source
URLs.

### Rerunning transcription-ocr for new documents

The transcription-ocr stage covers both paper PDFs and talk recordings, so a
rerun to extract newly downloaded PDFs re-checks every recording too. Each run
first reuses the transcripts already on disk; that costs nothing and needs no
option. When every recording has a transcript, the speech lane succeeds with
no provider calls. Only recordings without a transcript need the speech
provider option plus provider cost approval; without them the stage stops
with `speech_inputs_present_provider_path_required` and lists the missing
transcripts under `transcript_reuse_blockers`.

### Receipt-bound public-media recovery and paid-call reservation

When a conference plan permits recovery of an otherwise captionless public
talk, the recovery queue must be a declared receipt—not a filesystem glob or a
search result.  The reusable audio primitive accepts only a direct public
YouTube/youtu.be video identifier, obtains metadata before requesting audio,
and records the canonical URL, video id, declared provenance, digest, MIME,
byte count, measured duration, bounded attempt count, and one terminal state.
Private, login- or age-gated, malformed, and failed sources remain explicit
terminal blockers.  It never accepts browser cookies, conference credentials,
paywall bypass options, or arbitrary hosts.

Direct-public PDFs, decks, and slide images use a separate finite acquisition
boundary. A workspace-selected policy must enumerate the allowed source-page
and target hosts, redirect cap, MIME/class mapping, and byte limit. The
acquisition owner revalidates the source, initial target, every redirect, and
final target; it verifies MIME, content signature, byte count, and SHA-256
while atomically writing a deterministic workspace-local payload. Its receipt
is the only processor handoff: downstream PDF/OCR work receives the declared
local path, MIME, byte count, hash, provenance, and policy digest—not a source
URL, discovery row, or filesystem scan. New hosts, redirects, MIME/signature,
or size failures remain terminal classification records with no payload.

Use a named receipt-bound estimator before any paid speech or OCR work.  It
accepts only explicitly declared acquired rows with unique identities, valid
audio digests, and finite measured durations; missing prices, non-finite
durations, duplicate rows, and path/glob-derived scope fail closed.  The
estimator resolves the governed task's provider/model through the model
registry, records canonical existing spend, and enforces the applicable
all-in and task caps before a provider dispatch.

Every benchmark, speech, or OCR call must also create a durable locked
reservation before dispatch.  The ledger includes canonical prior spend plus
all unsettled maximum charges and settled provider-receipted actuals.  It binds
the run, task, item, benchmark, and estimate identities; duplicate, stale,
unknown, unpriced, or over-cap entries are rejected.  The provider receipt is
sanitized and settled through the ordinary preflight and post-call cap checks
before the next paid call.  These controls are reusable policy: the active
conference plan supplies its own source policy, receipt paths, and cap values.

## Analysis Workflow (Aligned with NeurIPS_Analysis_Plan.md)

This section mirrors `conferences/neurips2025/analysis/NeurIPS_Analysis_Plan.md` and captures the steps used to reproduce the NeurIPS 2025 analysis outputs.

### Analysis script and path support boundaries

| Surface | Reusable workflow label | Command or path | Boundary |
|---------|-------------------------|-----------------|----------|
| Enhanced indexing | reusable workflow stage for production; NeurIPS-local reference script for historical reproduction | `enhanced-indexing-vector-store`; owner modules `src/conference_pipeline/indexing/chunking.py` and `src/conference_pipeline/indexing/walker.py`; pinned reference strategy `scripts/setup_vector_store_enhanced.py` | Future conference production plans must use canonical source-aware chunking through the workflow stage. The NeurIPS script is reference evidence for the adopted strategy, not the cross-conference production owner command. |
| Legacy indexing script | legacy / historical | `scripts/setup_vector_store.py` | Do not present as the current analysis-plan setup command. |
| Vector-store path | intended docs path with implementation drift | `conferences/neurips2025/cache/vector-store/`; drift evidence: `conferences/neurips2025/vector_store/` | Do not claim all scripts are normalized to the cache path until runtime scripts are changed. |
| Phase 1b NotebookLM Export | reusable optional support script | `scripts/package_conference_for_notebook_lm_export.py` | Filesystem-only packaging recipe for NotebookLM sources; not workflow receipt behavior. `scripts/package_deep_research_docs.py` remains a compatibility wrapper. |
| Phase 7 QA and traceability | NeurIPS-local optional support scripts | `scripts/qa_analysis_docs.py`; `scripts/traceability_recovery.py`; `scripts/inline_traceability_references.py` | Local quality/traceability recipes around analysis docs; workflow receipts remain the canonical run evidence. |
| Phase 7b focused podcast | workflow-owned podcast command construction | `scripts/generate_focused_podcast.py` | Current workflow podcast argv target when `analysis_manifest.json` declares topic docs and fact ranks. |
| Phase 7b fact ranking | NeurIPS-local prerequisite/support script | `scripts/rank_facts.py` | Can prepare ranks for focused podcast generation; not a named workflow stage. |
| Phase 7b slides | NeurIPS-local / experimental richer presentation path | `scripts/generate_slides_agentic.py` | Do not claim reusable workflow support before presentation validation and proof. |
| Phase 7b Gemini audio | historical richer NeurIPS audio path | `scripts/generate_podcast_gemini.py` | Not the current workflow-level podcast adapter. |
| Phase 8 paper analysis | NeurIPS-local optional support script | `scripts/phase8_paper_analysis.py` | Optional local paper-analysis recipe, not a required canonical workflow stage. |

### Phase 1: Enhanced Indexing (Required)

Production runs create the vector store through the reusable workflow stage with
canonical source-aware chunking:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
  --workspace conferences/<slug> \
  --stages enhanced-indexing-vector-store \
  --run-id <run-id>
```

The canonical chunking owner is
`src/conference_pipeline/indexing/chunking.py`, called by
`src/conference_pipeline/indexing/walker.py`. It uses title/abstract metadata
chunks for OpenReview paper metadata, section/page-aware or 1,000-token /
100-overlap chunks for long paper/PDF text, about 120-second chunks for
timestamped transcripts, one chunk per slide/page/image for OCR/visual records,
and the same 1,000-token / 100-overlap fallback for generic artifact text.
Production vector-store builds reject snippet-only or first-240-character
vectorization as completion evidence and must write chunk-length statistics
including min, p50, p95, max, and the count of chunks below 300 characters.
`PG-CORPUS-TOKEN-INDEXING` requires the stage receipt and index digest to record
`token_budget_policy_digest`, `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`,
and `CORPUS_VECTOR_EMBED_BATCH_SIZE`. Passing evidence requires
`vector_embeddings_written: true` and embedded chunk counts matching the
sidecar chunk counts for each store. Word-count-only sidecars, readiness-only
probes, partial Chroma rows, and historical exhaustive extraction output
without registry receipt are rejected evidence.
Production plans must also prove local embedding readiness before treating the
local embedding lane as blocked: the stage receipt must show the configured
client was built with `build_embedding_client`, first use was forced with
`embed_texts`, and `embedding_readiness.client_start_or_load_attempted` was
recorded. Sidecar chunks alone cannot be used to claim vector retrieval.

#### Incremental corpus indexing

The corpus-inventory-index stage is incremental. Each store keeps a
`chunk_manifest.json` beside `chunks.jsonl` that records, per artifact, a
fingerprint of everything its chunks depend on (record metadata that reaches
chunk text or locators, the selected text file, the token and embedding policy
digests, and the chunker source), the text file's size, mtime, and digest, and
the artifact's byte range in `chunks.jsonl`. On the next run:

- unchanged artifacts are byte-copied (no read, no chunking, no tokenization);
- added or changed artifacts are chunked; removed ones are dropped;
- `chunks.jsonl` stays in inventory order, which downstream scans rely on;
- vectors live in 256 artifact-keyed hash buckets
  (`embedding-sidecars/<policy>/<store>-bucket-NNNN.npz`, format
  `npz-buckets-v2`), so an insertion rewrites one bucket, and a rewritten
  bucket copies every vector whose embedding digest already exists (in this
  run, in legacy positional shards, or in a sibling run id) and embeds only the
  rest;
- Chroma skips unchanged rows, refreshes metadata only (never the vector) for
  rows whose artifact rows changed without changing their text (for example
  authors enriched later), and deletes ids that left the corpus;
- unchanged artifacts are confirmed by content hash (about a second for a
  500 MB corpus); `--trust-file-stat` trusts size and mtime instead;
- `index_digest.json` never contains its own digest; receipts carry it.

Transcript chunks always embed their title: Stage 6 blocks on an admitted
speech transcript without a `title` (`speech_transcript_title_missing`), and
base caption records take their title from the event page they were found on.
Caption files (WebVTT/SRT) are indexed as timed segments, never as raw cue
markup: `conference_pipeline.caption_text` strips cue timing and tags and
collapses YouTube's rolling auto-caption repeats (only between cues that abut
in time, so a phrase said twice after a pause stays). Scripts written without
spaces (Japanese, Chinese) are cut into fixed pieces before token windowing, so
no speech is dropped at the embedding limit.

A token or embedding policy change, a chunker or caption-normalization source
change, or a tampered
`chunks.jsonl` re-chunks the store (vectors are still reused by digest). The
receipt records what happened under `checks.index_update` (per store: artifacts
unchanged, added, changed, removed, skipped, and chunks reused or re-chunked)
and `checks.embedding_update` (chunks reused, copied, embedded now). Flags:

```bash
# normal incremental run
.venv/bin/python scripts/run_cvpr2026_production.py --stage corpus-inventory-index --resume
# skip content hashing and trust size+mtime (misses same-size, same-mtime rewrites)
.venv/bin/python scripts/run_cvpr2026_production.py --stage corpus-inventory-index --resume --trust-file-stat
# discard the manifest, every stored vector, and the Chroma collection
.venv/bin/python scripts/run_cvpr2026_production.py --stage corpus-inventory-index --resume --full-rebuild
```

A store built before the chunk manifest existed is adopted once, without
embedding, by `scripts/migrate_corpus_index_to_incremental.py --workspace
conferences/<slug> --run-id <stage6-run-id> [--dry-run] [--prune-legacy]`. It
re-chunks into a staging directory, requires byte-identical rows, installs the
manifest, copies every vector into buckets by digest, and requires the
`vector_store_digest` to be unchanged. It never modifies `index_digest.json`,
`chunks.jsonl`, the legacy sidecar manifest, or Chroma. If it cannot adopt a
store, the next stage run re-chunks once and still copies the legacy vectors.

Workshop transcripts must be talk-aware before corpus indexing when workshop
speech is in scope. If no vetted `talks.json` exists, the provider-backed owner
is `transcription_ocr.workshop_talk_boundary_discovery` via
`scripts/discover_workshop_talks.py`; it uses Gemini 2.5 Pro over long workshop
transcripts plus slide timestamps, writes `*_talks.json`, emits 5-minute talk-context
batches, and records `provider_receipt_paths` plus estimated cost under
`cost_task_id=workshop_talk_boundary_discovery`. The offline annotation
owner is `transcription_ocr.workshop_talk_segmentation` via
`scripts/segment_workshop_talks.py`; it records `workshop_id`,
`workshop_title`, `talk_id`, `talk_title`, and `talk_speaker` on segments. If no
vetted `talks.json` or governed discovery path exists, the stage blocks with
`talk_segmentation_provider_not_configured`. audio-only recovery is rejected evidence for talk-aware transcripts.

For historical NeurIPS 2025 reproduction only, the pinned reference strategy is:

```bash
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

That script is retained to document the NeurIPS strategy that was promoted into
the reusable chunker; it is not the production owner command for future
conferences.

Configure GPT Researcher to use the persistent store:

```bash
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

Outputs:
- `conferences/neurips2025/cache/vector-store/chroma_db/`
- `conferences/neurips2025/cache/vector-store/entity_topic_index.json`
- `conferences/neurips2025/cache/vector-store/vector_store_metadata.json`

v1.91 dual-store paper matching adds a reusable platform requirement for
presentation-oriented runs: canonical indexing also persists separate
`storage/vector-store/<run-id>/stores/papers/` and
`storage/vector-store/<run-id>/stores/artifacts/` sidecars. The paper store
contains `paper_pdf`, `paper_text`, and `paper_metadata_fallback` chunks. The
artifact store contains transcripts, slide OCR, visual descriptions,
session/talk evidence, and other non-paper extraction artifacts. Full paper
PDF/text chunks must prepend paper title plus section and subsection context
when known; title-plus-abstract fallback is allowed only when recorded in
metadata. Dual-store paper matching must consume these canonical `papers` and
`artifacts` stores built from full chunk text; `evidence.snippet` previews are
not valid vector text for live corpus indexing.

Configure the split and paper matching in `config.yaml`:

```yaml
workflow:
  vector_store:
    stores:
      papers:
        source_families: ["paper_pdf", "paper_text", "paper_metadata_fallback"]
      artifacts:
        exclude_source_families: ["paper_pdf", "paper_text", "paper_metadata_fallback"]
  paper_matching:
    enabled: true
    keypoint_provider: "openai"
    keypoint_model: "gpt-5.6-sol"
    keypoint_source: "artifact_gpt_researcher_findings"
    paper_research_store: "papers"
    artifact_research_store: "artifacts"
    output_top_n: 20
```

The matching order is locked:

```text
paper-matching workflow stage
  -> artifact store GPT-Researcher using store_kind=artifacts
  -> frontier-model keypoint extraction
  -> paper store GPT-Researcher using store_kind=papers
  -> returned paper titles in matched_papers.json
```

Both GPT-Researcher passes are store-grounded and do not use Tavily/web
fallback. The workflow receipt is written to
`storage/workflow-runs/<run-id>/paper-matching-receipt.json`; per-topic
artifacts are written under
`storage/paper-matching/<run-id>/<topic_id>/` as
`artifact_gpt_researcher_findings.md`, `keypoints.json`,
`paper_gpt_researcher_findings.md`, and `matched_papers.json`.

Paper-count caps must be monitored as part of the workflow contract. The
source-of-truth cap is `workflow.paper_matching.output_top_n`; downstream
analysis matched-paper ingestion and deck paper limits must not silently set a
lower value. Live receipts should include a
`paper_matching_cap_verification` block and fail closed when a hidden cap would
reduce paper evidence before analysis or presentation generation.

The rationale is that presentation papers should match what oral/session
artifacts discuss. They should not come from generic paper ranking or podcast
frequency scoring. `scripts/rank_papers.py` is deprecated and remains only for
NeurIPS 2025 legacy compatibility. Presentation stages consume
`matched_papers.json`; `ranked_papers.json` is generated only as a compatibility
artifact.

#### Every passage names its paper or talk; the keyword index is stored

Two rules hold for every conference's Stage 6 (`corpus-inventory-index`):

**Titles before chunking.** `conference_pipeline.indexing.record_titles.title_corpus_records`
titles every paper, workshop-paper and speech record from sources acquisition
already wrote, with no network call:

- main papers from the submission-system list the download stage imported
  (title, authors, keywords, primary area, venue);
- a paper's slides or poster file from its event page (`material_type: slides`,
  linked to the paper, never dropped as a duplicate);
- workshop papers from their acquisition receipts, with the workshop named by
  the workshop page that links the paper;
- recordings from their event page (`Tutorial: …`, `Invited Talk: …`,
  `Oral Session: …`, `Poster: <paper>`), split recordings adding `Talk` and
  `Speaker` per passage;
- anything else from the PDF's own first page (title only).

The chunker writes these as header lines (`Paper Title`, `Authors` (first ten),
`Keywords`, `Workshop`, `Material`, `Presentation`, `Talk`, `Speaker`) at the
top of every passage, so both dense and keyword retrieval can attribute and
match a passage by its paper's or talk's own words. An indexable transcript
without a title blocks the stage (`speech_transcript_title_missing`); reviewed
speechless transcripts are listed in
`storage/transcription-ocr/evidence-lanes/transcript_exclusions.json` and
become audit-only. The Stage 6 receipt reports titled/untitled counts per
family and their provenance under `artifact_counts.record_titles`.

**Hybrid index = vectors + stored bag of words.** Each store writes, next to
its `chunks.jsonl`, a BM25 index (`keyword_index.json` +
`keyword-index/`) built from the exact passage text Chroma embeds
(`conference_pipeline.indexing.keyword_index`). Its fingerprint is in the
store summary of `index_digest.json` (a registered Stage 6 output, so receipt
lineage covers it) and in the store's `v15_index_digest.json`. The hybrid
retriever loads it instead of rebuilding BM25 in memory and refuses when its
passages differ from the vectors' (`keyword_index_stale_for_passages`,
`keyword_index_manifest_mismatch`); scores are identical to the in-memory
ranker, so fusion (`equal_weight_rrf.v2`) is unchanged.

**Lineage.** The talk-map directory, the transcript exclusions, the event-page
list and the submission-system list are registered `corpus-inventory-index`
lineage inputs; splitting recordings is the `event-talk-splits` sub-step. A
re-split or a new exclusion marks the index stale in `workflow stage-status`.

### Phase 1b: NotebookLM Export (Optional)

Package conference transcripts, paper abstracts, workshop paper metadata, and
analysis documents into upload-friendly NotebookLM Markdown source chunks.
Transcript chunks contain only the talk title, author when available, and the
transcript text. Analysis documents are copied as Markdown without hash
dereferencing so existing ids remain untouched.

```bash
python3 scripts/package_conference_for_notebook_lm_export.py \
  --workspace conferences/iclr2026 \
  --output-dir conferences/iclr2026/exports/notebooklm_export
```

Outputs:
- `notebooklm_export/notebooklm/transcripts_part_###.md`
- `notebooklm_export/notebooklm/paper_abstracts_part_###.md`
- `notebooklm_export/notebooklm/analysis_docs_part_###.md`
- `notebooklm_export/manifest.json`
- `notebooklm_export/index.csv`

Notes:
- Main-conference paper abstracts are loaded from
  `storage/download-corpus/openreview_accepted_papers.json` by default.
- Workshop paper receipts under `storage/workshops/papers/**/receipts/` are
  included and enriched with abstracts when they can be joined to local paper
  metadata by id or title.
- Speech transcripts are loaded from
  `storage/transcription-ocr/speech-transcripts/`.
- Analysis exports include Markdown files by default; use
  `--analysis-extensions` to include additional text artifact types.
- Use `--max-bytes`, `--max-words`, `--papers-max-bytes`, or
  `--papers-max-words` to control chunk sizes.

### Phase 2: Query Preparation

Before running GPT Researcher:
- Run `scripts/build_topic_strategy.py` to collect the title spine, request
  three independent model recommendations, and write
  `analysis/topic_strategy.json` with exactly 12 final topics.
- Pre-formulate questions per topic (01-20)
- Expand synonyms and alternative phrasings
- Use `entity_topic_index.json` to seed queries
- Track any web-only findings as "Source Gaps" for later recovery

### Phase 3-4: Topic Execution (GPT Researcher + Local Search)

**The GPT-Researcher package can run the research (since 2026-09-28).** From
Phase 14C (2026-04-27) until this date the research engine was an in-house
stand-in named after GPT-Researcher: our hybrid retriever plus one local-LLM
synthesis call. The package itself (0.16.1, Python 3.12) now runs it --
planning, sub-queries, deep research, report writing -- with two guarantees
enforced in `conference_pipeline.research.gptr_engine`:

- **Retrieval is our stored hybrid index only.** The `local_hybrid` retriever
  plugin (`research.gptr_retriever`, entry point
  `gpt_researcher.retrievers`) returns indexed passages (dense + stored BM25,
  equal-weight RRF) with their title headers as content and
  `chunk://<store>/<chunk_id>` URLs; nothing is scraped. The engine refuses
  to run if GPT-Researcher resolved any other retriever (it silently falls
  back to a web search when a name does not load).
- **Every model call is the local model.** Fast/smart/strategic LLMs resolve
  from the registry tasks `research.gpt_researcher_fast` /
  `research.gpt_researcher_smart` (`local` provider) on a loopback endpoint
  with a dummy key; `CONTEXT_FILTER=keyword`; the run sits inside a
  loopback-only network guard (`research.egress_guard`) so a hosted scorer,
  embeddings call or telemetry aborts it. Each call is a $0
  `local-openai-compatible` provider receipt.

**The engine is an operator choice per conference, never a default.** The
07C dual deep-dive lanes run either engine: `gpt_researcher` (above) or
`inhouse` (one retrieval plus one local-model synthesis per lane). The choice
comes from `--engine` or `workflow.research_engine` in the conference's
`config.yaml` (with a `research_engine_decision` note); with neither, the stage
stops and asks (`conference_pipeline.research.engine_choice`). ICML 2026 chose
`inhouse` on 2026-09-30 after a topic-level benchmark (in-house better
grounded, 91% vs 74% of cited claims supported; GPT-Researcher broader and
once derailed by instruction-like text in a retrieved paper); the receipt
records `research_engine_choice`. With GPT-Researcher the global lane
researches over the whole index and the heuristic lane within its
keyword-prefiltered pack. Evidence files list exactly the passages the
retriever returned; the lane receipt records `research_engine`, its settings
(also in each lane's resume key) and the guard's report. `--engine inhouse`
keeps the previous single-call synthesis for comparison; `build_researcher`
accepts provider `gpt_researcher` for any other stage.

**Citations are passage links in both engines.** Every lane report cites
`chunk://<store>/<chunk_id>`, so a citation still resolves after the topic
merge (07D reads only the lane Markdown). GPT-Researcher sees short per-run
handles (`chunk://<store>/S<n>`) that the engine rewrites to full ids; the
in-house engine is asked for `[EVIDENCE #n]`, which
`deep_dive_lanes.resolve_evidence_numbers` rewrites to the n-th passage of the
lane's saved evidence list (the prompt numbers passages in that order).
Anything that does not resolve becomes `[unresolved citation: ...]`, never a
guess; lane evidence files and the 07C receipt record `citations.resolved` /
`citations.unresolved`.


Run iterative hybrid queries, then draft topic documents in
`conferences/<slug>/analysis/`. The reusable production contract is retrieval
first: read local extraction/vector chunks, incorporate query research and
matched-paper findings, run broad scan, focus-area subqueries, top-paper deep
dives, benchmark/resource enrichment, and targeted gap-fill searches, and only
then write the deep topic analysis.

Production GPT-Researcher evidence/draft synthesis defaults to local Qwen3.6 35B-A3B
(`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`) with local BGE-M3 embeddings
(`mlx-community/bge-m3-mlx-8bit`, `EMBEDDING_DIM=1024`) and
`WEB_SEARCH_ENABLED=false`. Retrieval uses BM25+vector reciprocal-rank fusion
by default (`workflow.research.gpt_researcher.retrieval_mode: hybrid`) so exact
terms such as model names, acronyms, and tool names can win over semantically
near but wrong vector neighbors. Fusion is equal-weight RRF
(`rrf_fusion: equal_weight_rrf.v2`) unless both `bm25_weight` and
`vector_weight` are set: RRF scores are nearly flat at `RRF_K=60`, and the
former query-length weights (0.3/0.7 for 5+ terms) silently made most queries
vector-only. On CVPR 2026 a verbatim sentence found its paper 36/38 times with
equal-weight fusion, 4/38 with vectors alone. The same config block exposes the experiment
knobs used for deeper local runs: `max_iterations`,
`deep_research_breadth`, `deep_research_depth`,
`deep_research_concurrency`, `max_subtopics`, `total_words`,
`smart_token_limit`, `strategic_token_limit`, `fast_token_limit`, and
`temperature`. Uppercase env aliases such as `MAX_ITERATIONS` and
`DEEP_RESEARCH_BREADTH` override config for one-off runs and are recorded in
query plans and receipts. Current defaults are `MAX_ITERATIONS=5`,
`DEEP_RESEARCH_BREADTH=4`, `DEEP_RESEARCH_DEPTH=3`, `MAX_SUBTOPICS=8`,
`TOTAL_WORDS=3000`, `SMART_TOKEN_LIMIT=8192`,
`STRATEGIC_TOKEN_LIMIT=8192`, `FAST_TOKEN_LIMIT=4096`,
`RETRIEVAL_K=18`, `PAPER_MATCHING_RETRIEVAL_K=18`,
`CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, and `TEMPERATURE=0.0`.
Current behavioral controls are iterations, breadth,
depth, retrieval settings, `max_subtopics`, `total_words`,
`smart_token_limit`, and `temperature`; concurrency and the fast/strategic token
limits are receipt-only until the stage owns separate parallel or fast-model
calls.

The local Qwen+BGE lane is the default at the research/evidence layer, not
automatically at the final public-analysis layer. It owns corpus-grounded
retrieval, evidence expansion, parameter sweeps, and draft GPT-Researcher
findings when the run records hybrid retrieval, source manifests,
provider/model provenance, parameter digests, and local-only receipts. Those
findings remain inputs to deep narrative synthesis. Production
`analysis/<topic>.md` still uses the governed
`analysis.deep_narrative_synthesis` route
(`openai:gpt-5.6-sol`, fallback `openai:gpt-5.6-terra`) unless a local candidate passes
the same schema, provenance, source-diversity, keypoint, and editorial-quality
validation gates.

Repeated identical topic/query/corpus tuples are served from the behavioral
research result cache under `storage/research-cache/`. Cache keys include the
retrieval bundle digest, query plan digest, behavioral parameters, model/provider,
and web-search setting. Cache-hit receipts provide exact reuse proof; only
successful results and intentionally thin source-scarcity results are cached —
transient failures are never cached. See
[docs/local-llm.md](../local-llm.md#gpt-researcher-local-lane-and-research-cache-gptr-0405)
for the full cache key specification and receipt schema.

Deep-dive topics run two GPT-Researcher evidence lanes. The first is the global
no-topic-id hybrid lane over the whole corpus or stable source scopes. The
second is the heuristic keyword-prefiltered lane owned by
`analysis.gpt_researcher_heuristic_prefilter` and
`conference_pipeline.research.heuristic_prefilter`, which builds an explicit
`heuristic_keyword_prefilter` pack from stable vector chunk sidecars before
synthesis. Receipts must record `keyword_seed_source`,
`prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`,
`stores_searched`, `source_family_scope`, `retrieval_scope`, and
`invalid_citation_ids`; topic-id filters and unrecorded keyword lists are
rejected.

```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Your research question"
```

Document structure (per topic):
- Executive Summary
- Key Findings
- Detailed Analysis with multiple named subsections
- References And Sources
- Cross-References
- Recommendations
- Questions For Further Investigation
- compact Slide Brief section with Slide Seeds, plus
  `analysis/slide_briefs/<topic_id>.md`

Format requirements (from the analysis plan):
- Start with bullet points + references
- Keep phase 1 drafts to roughly 5 pages
- Re-evaluate whether a topic needs subdivision before deep dives

Traceability gate: every finding should include Presentation ID + Slide +
Timestamp during drafting. In current production runs, raw ids and transcript
quotes belong in `analysis/provenance/<topic_id>.md`,
`analysis/provenance/<topic_id>.json`, and
`analysis/analysis_citation_manifest.json`. The public
`analysis/<topic_id>.md` is the deep report consumed by synthesis and podcast;
deck stages prefer `analysis/slide_briefs/<topic_id>.md`. Use auto-recovery
only for gaps and log any unresolved items.

Reusable v1.63 narrative generation is available to future workspaces through
`analysis/narrative_source_packs/*.json`. When these source packs are present,
the generic `analyze` stage renders deep narrative reports, writes slide brief
sidecars and provenance sidecars, emits `analysis_citation_manifest.json`, and
enriches `analysis_manifest.json` with optional deep-analysis validation and
model metadata.

When exhaustive summaries and dual deep-dive GPT-Researcher reports are
available, run the packaged `analysis.topic_reconciliation_reference_workflow`
before final narrative rendering:

```bash
.venv/bin/python scripts/run_topic_reconciliation_reference_workflow.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --resume \
  --summary
```

This writes `07H-topic-reconciliation-reference-workflow.json` and packages
07D reconciliation, 07E deterministic validation, deterministic safety fixes,
bounded 07G candidate resolution when needed, 07F broad repair only after
candidate blockers remain, and final 07E validation. Final narrative rendering
or slide generation requires a succeeded 07H receipt with clean final reference
counts.

The individual 07D owner remains available for inspection or fallback:

```bash
.venv/bin/python scripts/reconcile_topic_analysis.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --resume \
  --summary
```

This writes `analysis/topic_reconciliations/<topic_id>.md`, records
`07D-topic-reconciliation.json`, and injects `topic_reconciliation` into
`analysis/narrative_source_packs/<topic_id>.json`.

Before final narrative rendering or slide generation consumes a local-model
brief from the individual 07D path, run the deterministic reference gate:

```bash
.venv/bin/python scripts/validate_topic_reconciliation_references.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --summary
```

This builds `analysis/topic_reference_ledgers/<topic_id>.json`, writes
`analysis/topic_reference_validation/<topic_id>.json`, records
`07E-topic-reference-validation.json`, and enriches
`topic_reconciliation.reference_validation`. The gate must report zero unknown
ids and zero title/id mismatches; bare ids are warning evidence to canonicalize
before publication. Slide generation should consume the final reconciled and
reference-validated analysis plus slide briefs, not reconcile raw parallel
reports for the first time.

### Phase 5: Coverage Analysis

Identify under-referenced presentations:

```bash
.venv/bin/python scripts/analyze_presentation_coverage.py
```

Output: `conferences/neurips2025/analysis/coverage_report.md`

### Phase 6: Gap Filling (Substantive)

Use tooling to map presentations to topics, then do deep reads:

```bash
.venv/bin/python scripts/map_presentations_to_topics.py
.venv/bin/python scripts/fill_coverage_gaps.py
```

Note: `fill_coverage_gaps.py` adds pointer references. The analysis plan requires manual, substantive extraction for underreferenced sessions.

### Phase 7: QA and Synthesis

Quality checks and traceability recovery:

```bash
.venv/bin/python scripts/qa_analysis_docs.py
.venv/bin/python scripts/traceability_recovery.py conferences/neurips2025/analysis
.venv/bin/python scripts/inline_traceability_references.py
```

Synthesis outputs:
- `conferences/neurips2025/analysis/executive_summary.md`
- `conferences/neurips2025/analysis/trend_report.md`

### Phase 7b: Podcast Script Generation (Optional)

The workflow-owned podcast command construction target is `scripts/generate_focused_podcast.py`; `scripts/generate_podcast_gemini.py` remains a historical richer NeurIPS audio path and is not the current workflow-level podcast adapter.

```bash
.venv/bin/python scripts/generate_focused_podcast.py \
  conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --fact-ranks conferences/neurips2025/audio/01_tools_and_open_source/fact_ranks.json \
  --output-dir conferences/neurips2025/audio/01_tools_and_open_source \
  --target-minutes 15
```

### Phase 7c: Per-Deck NotebookLM Podcasts (Optional)

One Google NotebookLM Audio Overview per topic deck and per executive deck,
owned by `scripts/generate_notebooklm_podcast.py`
(`conference_pipeline.podcast_notebooklm`). Each deck gets its own notebook
holding the analysis spine (topic report, talk synthesis, deep dives, and a
spine rendered from the approved `presentation_blueprint.json`) plus the paper
texts and de-duplicated talk transcripts the blueprint cites; executive decks
get the executive summary, trend report, cross-topic digest, frame, and all
topic reports. Generation is steered by a deterministic prompt written to
`audio/<id>/notebooklm_podcast/prompt.txt`.

```bash
# one-time: install the extra and log in with the operator's Chrome session
uv pip install --python .venv/bin/python -e ".[notebooklm]"
.venv/bin/notebooklm login --browser-cookies chrome
.venv/bin/notebooklm auth check --test --passive

# stage sources + prompts for every deck, no network
.venv/bin/python scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> --dry-run --summary

# live canary, then the full run (resumable; completed decks are skipped by manifest sha)
.venv/bin/python scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> \
  --deck-ids <topic-id>,<executive-deck-id> --live --ack-live-notebooklm --format deep-dive --length long --summary
.venv/bin/python scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> \
  --live --ack-live-notebooklm --format deep-dive --length long --summary

# curated Drive folder (after the stage receipt succeeded; the raw mirror is the backup profile)
.venv/bin/python scripts/deliver_notebooklm_podcasts.py --workspace conferences/<slug> --run-id <run-id> \
  --drive-root '<mounted-drive-root>/<Conference> Podcasts' --dry-run
```

Each notebook also receives an `Episode brief` run sheet and a `Program map`. The brief credits the creator (the blueprint `author`), states how the material was made from receipts (corpus counts, the small model that read everything end to end, the number of transcript windows it read), says the hosts did not attend, and names the workshops and sessions the full read ranked for the topic. Executive briefs list every topic in arc order with a transition into each one, and every episode is generated at `long` length so no topic is dropped.

Every live episode is then transcribed locally with MLX Whisper and reviewed before it counts: the creator credit and the method disclaimer must appear in the opening, no attendance may be claimed, topic episodes must name their workshops, and executive episodes must discuss every topic after the opening. Failures block the deck (`notebooklm_podcast_review_failed`); `--accept-review-findings` records operator acceptance, `--no-review` records a waiver, and `--force-regenerate` requests a fresh episode. Reviewed, unchanged episodes are never regenerated or re-reviewed.

Rules: `--live` requires `--ack-live-notebooklm`; an auth failure halts the
run, a quota failure halts resumably (`notebooklm_quota_exhausted`, about 20
audio overviews per day on NotebookLM Pro), and a blueprint-cited work with no
on-disk text blocks its deck. Outputs live under `audio/<id>/notebooklm_podcast/`
(family `podcasts`, Drive-mirrored, gitignored); `notebook_manifest.json` binds
notebook id, source sha256s, prompt sha256, and audio sha256. Dry-run and
`--fake-engine` manifests are never completion evidence.

### Stage Owner Registry Gate

After conference analysis or production planning is generated, run both the
stage owner compatibility audit and the unified stage script registry verifier
before finalizing the plan:

```bash
PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit \
  --plan-root <generated-plan-root>

PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --plan-root <generated-plan-root> \
  --skills-root .claude/skills \
  --strict
```

The registry checks `src/conference_pipeline/stage_owner_registry.json` for
approved owner scripts, deprecated/forbidden completion owners, planning-skill
script contracts, and the registry SHA-256 digest that future receipts must
bind. It is mandatory after planning because artifact-shape checks alone cannot
distinguish a reusable owner from a conference-local scaffold.

Before production proof, run the same verifier against receipts:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --strict
```

### Phase 8: Paper-Level Analysis (Optional, but recommended)

Score and classify papers not captured by presentations:

```bash
.venv/bin/python scripts/phase8_paper_analysis.py --score
.venv/bin/python scripts/phase8_paper_analysis.py --classify
```

Outputs:
- `conferences/neurips2025/analysis/importance_scores.json`
- `conferences/neurips2025/analysis/paper_topic_mapping.json`

### Phase 9: Hypothesis Detection (Planned)

The analysis plan includes field entropy, aspirational language detection, and benchmark coverage analysis. These are planned steps but not yet automated in scripts.

## Optional: Exhaustive Extractions Pipeline

The exhaustive extraction pipeline complements manual analysis by processing all segments with topic guardrails:

Provider policy (v1.7): local LLM inference is the default extraction path; heuristics serve as prefilter for confident matches; cloud LLM escalation applies only for low-confidence or ambiguous records; slide OCR is conditional (required, skipped_with_rationale, or deferred_blocked depending on source coverage).

```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

Source catalog generation:

```bash
python scripts/list_exhaustive_sources.py
```

## Advanced Usage

### Custom Session File

Use a different session file:
```bash
python3 scripts/download_from_sessions.py --sessions-file my_sessions.json
```

### Limit for Testing

Test with first N presentations:
```bash
python3 scripts/download_from_sessions.py --limit 10
```

### Monitor Progress

Watch live progress:
```bash
tail -f download_progress.log
```

Watch detailed logs:
```bash
tail -f "$STORAGE_ROOT/logs/session_asset_download.log"
```

### Check Database

Query presentation status:
```bash
sqlite3 "$STORAGE_ROOT/metadata.sqlite" << EOF
SELECT id, title, status
FROM presentations
ORDER BY status, id;
EOF
```

Count by status:
```bash
sqlite3 "$STORAGE_ROOT/metadata.sqlite" << EOF
SELECT status, COUNT(*) as count
FROM presentations
GROUP BY status;
EOF
```

## Configuration

Key `.env` settings:

```bash
# Conference year
NEURIPS_YEAR=2025

# Storage location for acquisition state
STORAGE_ROOT=./data/neurips_2025

# Keys for indexing + analysis
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...

# Audio format (mp3, m4a, wav)
AUDIO_FORMAT=mp3

# Rate limiting (requests per domain)
RATE_LIMIT_SLIDESLIVE=3

# Browser settings
HEADLESS=true
BROWSER_TIMEOUT=30000
```

See `.env.example` for all configuration options.

## Source References

- conferences/neurips2025/analysis/NeurIPS_Analysis_Plan.md
- scripts/discover_sessions.py
- scripts/download_from_sessions.py
- scripts/transcribe_sessions.py
- scripts/ocr_slides.py
- scripts/cleanup_slides.py
- scripts/detect_reference_slides.py
- src/conference_pipeline/indexing/chunking.py
- src/conference_pipeline/indexing/walker.py
- scripts/setup_vector_store_enhanced.py (pinned NeurIPS reference strategy; not the reusable production owner)
- scripts/setup_gpt_researcher_with_vectorstore.py
- scripts/package_conference_for_notebook_lm_export.py
- scripts/package_deep_research_docs.py (compatibility wrapper)
- scripts/query_gpt_researcher_hybrid.py
- scripts/analyze_presentation_coverage.py
- scripts/map_presentations_to_topics.py
- scripts/fill_coverage_gaps.py
- scripts/qa_analysis_docs.py
- scripts/traceability_recovery.py
- scripts/inline_traceability_references.py
- scripts/rank_facts.py
- scripts/generate_focused_podcast.py
- scripts/generate_podcast_gemini.py
- scripts/generate_slides_agentic.py (historical/deprecated; not a reusable completion owner)
- scripts/phase8_paper_analysis.py
- scripts/run_exhaustive_extraction.py
- scripts/list_exhaustive_sources.py
- src/conference_pipeline/storage.py
