# CLI Reference

The pipeline provides multiple entry points for different workflows.

## Canonical Workflow CLI

The canonical workflow surface for reusable stage execution is:

```bash
.venv/bin/python -m conference_pipeline.cli workflow list --workspace conferences/neurips2025
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages analyze,podcast,deck
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe --mode dry-run --run-id preview-run
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages transcribe --mode mock --run-id mock-transcribe
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe --run-id preview-run --resume
.venv/bin/python -m conference_pipeline.cli workflow resume --workspace conferences/neurips2025 --run-id preview-run --stages discover,download,transcribe
.venv/bin/python -m conference_pipeline.cli workflow summary --workspace conferences/neurips2025 --run-id preview-run
.venv/bin/python -m conference_pipeline.cli workflow stage-status --workspace conferences/neurips2025 --run-id preview-run
.venv/bin/python -m conference_pipeline.cli workflow bind-lineage --workspace conferences/cvpr2026 --run-id cvpr2026-live-production --backfill
.venv/bin/python -m conference_pipeline.cli workflow unblock --workspace conferences/neurips2025 --run-id preview-run --blocker-path conferences/neurips2025/storage/workflow-runs/preview-run/blocker-receipts/discover/20260430T120000Z-local_llm_not_ready.json --operator operator-1 --note "manual ack"
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages deck_contract_generation --presentation-type topic-deep-dive --topic 01 --mode dry-run --run-id presentation-topic-preview
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages render_export --presentation-type master --mode dry-run --run-id presentation-master-preview
```

`.venv/bin/python -m conference_pipeline.cli run-all` is the legacy acquisition alias.

### Conference production runners: shared late stages

The conference runners (`scripts/run_cvpr2026_production.py`,
`scripts/run_icml2026_production.py`) execute Stages 10 through 12 through the
shared `conference_pipeline.production` modules, so the same flags apply to
both:

```bash
# Stage 10: two-deck canary on the figure compiler route (a filtered scope always ends blocked)
.venv/bin/python scripts/run_icml2026_production.py --stage presentation --run-id <run-id> --resume --provider-cost-approved \
  --deck-ids 01_topic,02_topic --figure-route compiler --presentation-substage deck_figures
# Stage 10: operator acceptance of recorded findings, then delivery/video for the full scope
.venv/bin/python scripts/run_icml2026_production.py --stage presentation --run-id <run-id> --resume --provider-cost-approved \
  --accept-visual-qc-findings --accept-presentation-review-findings 01_topic --presentation-substage all
# Stage 10.4: per-deck NotebookLM podcasts (quota-metered Google account)
.venv/bin/python scripts/run_icml2026_production.py --stage notebooklm-podcast --run-id <run-id> --resume --ack-live-notebooklm
# Stage 11 and 12: release inventory (publication stays blocked without both flags), then the proof packet
.venv/bin/python scripts/run_icml2026_production.py --stage release --run-id <run-id> --publish-target <target> --publish-approved-by <operator>
.venv/bin/python scripts/run_icml2026_production.py --stage proof --run-id <run-id>
```

`--adopt-existing-outputs` bootstraps per-deck step ledgers for decks built
before the ledger existed; `--blueprint-approved-by`, `--visual-qc-max-rounds`
and `--allow-figure-variety-findings` tune the blueprint approval identity, the
redraw bound and the archetype-variety quota. See
[Shared Production Stages](reference/stage-owner-audit.md#shared-production-stages).

Use `workflow list` to inspect the ordered stage registry and the current
implementation backing for each stage. Use `workflow run` to target a specific
workspace and a selected subset of stages from one operator-facing surface. The
workflow command writes receipts under
`conferences/<slug>/storage/workflow-runs/<run-id>/summary.json`; for example,
`conferences/neurips2025/storage/workflow-runs/preview-run/summary.json`.

### workflow run options

- `--mode live|dry-run|mock`: live executes stage implementations; dry-run writes planned receipts only; mock writes deterministic success receipts without live calls.
- `--run-id TEXT`: workflow run id used for `storage/workflow-runs/<run-id>/`.
- `--resume`: skip compatible succeeded receipts from the same run id.
- `--summary`: print summary JSON after the run.
- `--presentation-type topic-deep-dive|master|executive-summary`: select the presentation manifest entry for first-class presentation stages. `topic-deep-dive` also requires `--topic`.

Dry-run records planned stage work without constructing live browser, network, or provider clients.
Mock mode records deterministic local success receipts without live browser, network, or provider calls.
Stage receipt resume skips only compatible succeeded receipts for the same workspace, stage, mode, topic, implementation, config, and command preview.
Existing workflow-runs/<run-id>/ directories are rejected unless --resume is set.
Every attempted stage writes a receipt, and failed stages persist summary.json before the command exits nonzero.
Required blocked or unavailable stages also exit nonzero: `blocked`,
`deferred_blocked`, `failed`, `unavailable`, and `unsupported` status payloads
write blocker receipts and prevent downstream stages from running until the
blocker is acknowledged.

### workflow stage-status

Read-only. Prints one line per stage of a run -- `FRESH`, `STALE (<reason>)`,
`MISSING`, `BLOCKED` -- and the next stage to run. `--layout auto|production|generic`
picks the registry `receipt_filename` order (production runners) or the
workflow stage order; `--json` prints the full report including per-stage
`reasons` (blocker code, upstream stage, chain) and `would_invalidate`.

### workflow bind-lineage

One-time migration for runs written before v5.1: `--backfill` stamps
`output_digest` and `lineage` onto succeeded receipts from the evidence
they recorded; read-only unless `--apply`; `--stages a,b` limits the scope;
`--assume-current a,b --operator NAME` records an operator attestation for
stages whose evidence does not resolve; `--force` re-stamps; `--json` prints
the report.

### workflow unblock

Use `workflow unblock` to record a manual acknowledgement for an active
`BlockerReceipt`:

- `--workspace PATH`: conference workspace root.
- `--run-id TEXT`: blocked workflow run id.
- `--blocker-path PATH`: blocker receipt path, absolute or workspace-relative.
- `--operator TEXT`: operator id recorded on the `UnblockReceipt`.
- `--note TEXT`: optional operator note.
- `--summary`: print the refreshed `summary.json` after acknowledgement.

The command writes an `UnblockReceipt`, refreshes `summary.json`, preserves the
original `blocker_receipt_path`, records `unblock_acknowledgements`, and
updates `halt_status` / `halt_stage_id` so downstream readers can tell whether
the run is still halted or has been manually resumed. The module form
`python -m conference_pipeline.workflow_unblock ...` exposes the same
manual-acknowledgement contract.

### stage owner and stage script registry audits

Run the stage owner registry audit after generating or editing conference
planning artifacts. It checks approved owner commands in
`src/conference_pipeline/stage_owner_registry.json` and fails when required
owner refs are missing or rejected scaffold/deprecated owners appear as
completion commands.

```bash
PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit \
  --plan-root <generated-plan-root>
```

Run the unified stage script verifier when plans need registry digest, skill
contract, registry-owned usage documentation, or production-receipt proof:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --plan-root <generated-plan-root> \
  --skills-root .claude/skills \
  --docs-root docs \
  --strict

PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --strict
```

The verifier prints JSON with a top-level `diagnostic` object. Agents should
read `diagnostic.summary`, `diagnostic.checked_inputs`, and
`diagnostic.next_actions` before retrying. Blockers include `expected`,
`observed`, and `next_action` fields so the next repair can target the correct
plan row, receipt, manifest, usage-doc anchor, or command metadata entry.

### focused SlidesLive media recovery

Use the focused recovery only after Stage 4 has produced cached
`item_page_auth_probe` artifacts and the download-corpus inventory still lacks
local audio for embedded SlidesLive presentations:

```bash
PYTHONPATH=src .venv/bin/python scripts/recover_iclr2026_slideslive_media.py \
  --workspace conferences/iclr2026 \
  --run-id iclr2026-live-production \
  --resume \
  --worker-count 2 \
  --summary
```

The script is the ICLR 2026 wrapper for the registry owner
`acquisition.slideslive_media_recovery`. It scans only cached item-page probes,
downloads or resumes matching SlidesLive audio, writes
`storage/download-corpus/slideslive_media_recovery.json`, and merges recovered
audio records into `storage/download-corpus/download_corpus.json`. Use
`--inventory-only` for a no-download audit. Do not add paper, workshop, poster,
or source-page refresh flags to this recovery; those belong to Stage 4 or a
separate acquisition phase.

SlidesLive slide images are a separate acquisition lane:

```bash
PYTHONPATH=src .venv/bin/python scripts/recover_slideslive_slides.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --resume \
  --worker-count 4 \
  --summary
```

The registry owner `acquisition.slideslive_slide_download` scans only cached
`item_page_auth_probe` artifacts, uses SlidesLive `slides.json` first, falls
back to yt-dlp thumbnail metadata for `/slides/original/` images, optionally
uses `--browser-fallback` for browser-intercepted slide image metadata, writes
`storage/download-corpus/slideslive_slide_recovery.json`, and merges slide
image records into `storage/download-corpus/download_corpus.json`. It does not
redownload audio, video, paper PDFs, workshop PDFs, poster images, or source
pages.

Before visual OCR/indexing, remove duplicate and generic/banner slides from the
OCR corpus non-destructively:

```bash
PYTHONPATH=src .venv/bin/python scripts/detect_reference_slides.py \
  --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset \
  --output-dir conferences/<slug>/storage/download-corpus/reference_slides \
  --receipt-path conferences/<slug>/storage/download-corpus/slide_reference_detection.json \
  --summary

PYTHONPATH=src .venv/bin/python scripts/cleanup_slides.py \
  --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset \
  --workspace conferences/<slug> \
  --download-corpus-path storage/download-corpus/download_corpus.json \
  --receipt-path conferences/<slug>/storage/download-corpus/slide_cleanup.json \
  --summary
```

These owners are `acquisition.slide_reference_detection` and
`acquisition.slide_cleanup`. Cleanup writes `skip_ocr` / `skip_reason` markers
instead of deleting raw slide images.

Use repeated `--task-id` flags to narrow the check to specific governed tasks:

```bash
PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit \
  --plan-root <generated-plan-root> \
  --task-id podcast.script_generation \
  --task-id presentation.deck_contract_generation
```

`workflow resume` requires `--stages` so operators explicitly choose the stage
subset to resume. Phase 11 does not auto-purge `storage/workflow-runs/`;
operators can remove obsolete run directories manually after they no longer need
receipts.

Top-level resume uses presentation-status checkpoints; workflow resume uses stage receipts.

First-class presentation stages are presentation_blueprint, deck_contract_generation, source_binding, visual_plan_generation, render_spec_generation, render_export, visual_qc, delivery_generation, presentation_review, video_generation, video_sync_validation, and human_review. `workflow deck` remains a backward-compatible alias for legacy deck command construction; it is not the full presentation lifecycle. Presentation stage receipts include selected scope and presentation_manifest digests, and resume compatibility binds the selected scope, manifest content, config, and command preview. First-class presentation stages now provide workspace-aware command planning, safe dry-run script contracts, structural validation, and receipt-backed proof reports; live cross-conference production still requires conference-local proof.

Topic-deep-dive blueprint generation consumes `presentation_frame.json` through
`presentation.generate_editorial_blueprint`; the frame separates retrieval
topic title from public deck title and records `presentation_frame_validation`
before deck-contract generation.

Presentation stages resolve `decks/presentation_manifest.json` from the selected
workspace and choose one manifest entry through `--presentation-type` plus
`--topic` for topic deep dives. Workflow-planned owner-script argv uses
`decks/<presentation>/...` artifacts and stage-specific flags such as
`--slides-md`, `--output-html`, `--slide-mapping`, `--script-md`,
`--slides-html`, `--build-dir`, `--output`, and `--dry-run`.

Workflow-planned presentation scripts use decks/<presentation>/... paths; presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning.
Presentation stage dry-run receipts include command_metadata with exact argv, selected scope, input artifacts, output artifacts, support level, provider/tool requirements, and unsupported_reason.
Unsupported reusable stages expose unsupported_reason instead of claiming script execution.
Video generation is dry-run/planning-only in Phase 24 and does not prove final MP4 generation, visual validation, or cross-conference proof.

Additional dry-run examples:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages render_export --presentation-type master --mode dry-run --run-id presentation-master-preview
.venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages video_generation --presentation-type master --mode dry-run --run-id presentation-video-preview
```

## Public Release

`conference-pipeline release` drafts, plans, builds, checks and publishes one
workspace's public release. The runbook is
[Public Release Runbook](operations/public-release.md).

```bash
.venv/bin/python -m conference_pipeline.cli release init <slug>
.venv/bin/python -m conference_pipeline.cli release plan <slug> [--bundle-id ID] [--ack-caution-family ID] [--isolated DIR]
.venv/bin/python -m conference_pipeline.cli release build <slug> [--bundle-id ID] [--bundle-only] [--mock | --embedding-cache DIR] [--isolated DIR] [--expect-tree-sha256 SHA]
.venv/bin/python -m conference_pipeline.cli release check <slug> [--bundle-id ID] [--isolated DIR] [--json PATH]
.venv/bin/python -m conference_pipeline.cli release publish <slug> --approved-by NAME --surface github|r2|pages|hf|zenodo [--with-preflight]
.venv/bin/python -m conference_pipeline.cli release hub build --bundle <bundle> [--bundle <bundle> ...] --out DIR [--env-file PATH]
.venv/bin/python -m conference_pipeline.cli release hub check DIR [--env-file PATH]
.venv/bin/python -m conference_pipeline.cli release hub publish DIR --approved-by NAME [--env-file PATH] [--dry-run]
```

| Command | Does | Writes |
|---|---|---|
| `init` | drafts `storage/public_export_manifest.json`: no family published, every storage family in `excluded_families`; never overwrites (`--force-draft-path` drafts beside it) | the draft |
| `plan` | validates the manifest and dry-runs the export; refuses an unpromoted draft (`no_family_promoted`) | validation and dry-run receipts; no bundle |
| `build` | writes the bundle from the reviewed dry run, then the index, site, dataset package and a staged data repository; never overwrites a bundle with live publication receipts | the bundle and its receipts |
| `check` | re-verifies the bundle: tree digest, checksums, private markers (no allowlist), secrets, provider limits per surface | nothing (`--json` writes a redacted report) |
| `publish` | refuses without `--approved-by` (`owner_approval_missing`) before reading `.env`, and refuses a surface over a provider limit; pushes the chosen surfaces, then checks liveness and metadata | live receipts |
| `hub build` / `check` / `publish` | the cross-conference hub from release bundles; its title and Pages project come from `config/public_hub.yaml`; `build` and `check` exit 1 on any link, accessibility, size, private-marker or secrets finding; `publish` needs `--approved-by` and refuses a hub with a finding; all three scan with the extra private markers the env file names (the checkout's `.env` unless `--env-file`), PDFs and zips by their text and members too | the hub directory; a Pages receipt |

Exit codes: 0 done; 1 `check` found a finding, or an isolated build's digest differs
from `--expect-tree-sha256`; 2 refused or blocked. `--isolated DIR` plans and builds
into a directory outside the workspace and writes nothing under `conferences/<slug>/`.
Targets come from an optional `release:` block in the workspace `config.yaml`; flags
override.

## Talks a Workspace Already Holds

`conference-pipeline talks bind <slug>` binds talk transcripts that are on disk but were
never admitted. Each source the workspace declares in
`storage/transcription-ocr/talk_sources.json` becomes a transcript evidence lane, and the
next `papers admit` admits those lanes beside the papers.

```bash
.venv/bin/python -m conference_pipeline.cli talks bind <slug> --dry-run
.venv/bin/python -m conference_pipeline.cli talks bind <slug>
.venv/bin/python -m conference_pipeline.cli papers admit <slug>
PYTHONPATH=src .venv/bin/python scripts/audit_talk_coverage.py --workspace conferences/<slug>
```

- **Sources** are read by format: `deep_dive_presentations` or
  `transcription_receipt_v15`.
- **Exclusions:**
  - an event named in `talk_exclusions.json` is a gap with that reason;
  - an absent or blank transcript is `empty_transcript`;
  - events sharing one recording keep one record, the smallest event id, and the rest are
    `duplicate_recording`.
- **A rebind** that would change only the receipt's timestamp writes nothing.
- **`talks bind` exit codes:** 0 bound (or a dry run reported); 2 refused.
- **`audit_talk_coverage.py` exit codes:** 0 every transcript on disk is admitted or
  excluded by name; 1 something unaccounted; 2 unreadable input.

## Late Paper Sets

`conference-pipeline papers admit <slug>` merges a workspace's existing paper set with the
paper sets added later, by stable paper identity, and writes
`storage/corpus/corpus_inventory.json`, `paper_identity_report.json` and
`corpus_statistics.json`.

```bash
python3 scripts/copy_late_paper_sets.py --source <source tree> --workspace conferences/<slug> --dry-run
python3 scripts/copy_late_paper_sets.py --source <source tree> --workspace conferences/<slug>
.venv/bin/python -m conference_pipeline.cli papers admit <slug> --dry-run
.venv/bin/python -m conference_pipeline.cli papers admit <slug>
.venv/bin/python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug> --scope inventory
```

- The copy follows `storage/papers/late_copy_plan.json`. Every file is checked by sha256
  after the copy, and the source tree is only read. Payloads land in the private backup
  family `storage/papers/late/`.
- `admit` reads the existing corpus through the adapter named in
  `storage/papers/late_paper_lanes.json` (`openreview_acquisition_json` or
  `deep_dive_abstracts_txt`), then every lane in it, and the named exclusions in
  `storage/papers/paper_exclusions.json`.
- Records sharing an OpenReview, arXiv or proceedings id are one paper. A record known
  only by a site id may join a paper by title; two stable ids never merge on title.
- The paper total is the number of distinct papers minus the named exclusions. A paper
  that only gains text, or arrives twice, does not add to it. No existing record is
  removed, and a second run leaves the files byte-identical.
- Exit codes: 0 admitted (or a dry run reported); 2 refused (no workspace, no registry,
  a malformed lane).
- `--scope inventory` measures title and author coverage over the inventory alone, for
  a workspace with no papers vector store or ledgers. It names what it did not measure.

## Paper Matching

Use the v1.91 `paper-matching` workflow stage after the split vector stores
exist:

```bash
conference-pipeline workflow run \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --stages paper-matching \
  --topic <topic_id>
```

The script wrapper exposes the same runtime for one topic:

```bash
python3 scripts/match_papers_from_artifact_keypoints.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --topic-id <topic_id>
```

The command writes canonical
`conferences/<slug>/storage/paper-matching/<run-id>/<topic_id>/matched_papers.json`.
By default it also writes legacy
`conferences/<slug>/decks/<topic_id>/ranked_papers.json` as a compatibility
artifact. Presentation command planning requires the canonical
`matched_papers.json`; `ranked_papers.json` is not the source of truth.
`scripts/rank_papers.py` is deprecated and should be used only for NeurIPS
2025 compatibility.

## Local Control Plane UI

Start the local operator UI with:

```bash
conference-pipeline control-plane --host 127.0.0.1 --port 8765
```

The default workspace is `conferences/neurips2025`, the default mode is
`dry-run`, and the default selected stages are `analyze,podcast,deck`. The UI
uses the same canonical workflow registry and receipt writer as `workflow run`.
It permits one active run at a time, so a second launch is rejected while an
existing run is still active.

The server binds to loopback with `--host 127.0.0.1` by default. Launch and
small-neurips-smoketest preset POSTs use a per-process action token sent as
`x-control-plane-token`; read-only status, artifact, and preview routes remain
ordinary local GET requests.

See [Local Control Plane Runbook](operations/control-plane.md) for the browser
workflow, safety boundaries, receipt behavior, and smoke-test preset contract.

The control plane also surfaces scheduled private-backup status from
`conferences/<slug>/storage/workflow-runs/scheduled-backup/`, including the
latest receipt path, last successful run, last failure, and digest count.

## Private Backup Sweep

Generated production phases and scheduled backup sweeps use the manifest-driven
backup profile documented in [Private Backup](operations/private-backup.md):

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --init-only --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/drive-setup-backup-write-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --dry-run --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-dry-run-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-write-receipt.json
python3 scripts/run_scheduled_backup_sweep.py --conferences-root conferences --target-root "/mounted/Drive"
```

The generated backup target convention is `My Drive/<Conference> <Year>-raw`;
operators must pass `--remote-path` for one workspace or `--target-root` for
scheduled sweeps so every run names the mounted conference leaf explicitly. Scheduled
sweeps with `--target-root` derive each leaf from `conference.name` in
`config.yaml`, falling back to the workspace directory name. Sweeps without
`--target-root` fail closed.

Run `--init-only` before live artifact generation to create the mounted Drive
leaf and manifest-selected family directories without copying payloads. Then run
the dry-run/write backup pair immediately after each artifact-producing stage
and before downstream stages consume those outputs. The sync is manifest-bounded
and rsync-incremental, so cache/unselected families do not move to Drive unless
the manifest explicitly enables the selected profile.

Receipts record workspace, profile, target, manifest sha, included families,
skipped families, exit codes, digest count, artifact snapshot sha, sync item
summaries, timestamp, secret-like path scan results, and `secrets_copied: false`.
Secret-like files such as `.env`, cookies, tokens, passwords, and credential
files fail the sync before any rsync command runs.

## Small NeurIPS Smoke Test

Run the milestone proof path with:

```bash
conference-pipeline small-neurips-smoketest --workspace conferences/neurips2025 --mode dry-run --run-id phase14-smoketest --summary
```

The default smoke-test topic is `01_tools_and_open_source`, and the default
output stages are `analyze,podcast,deck`. The command writes the normal workflow
summary at `conferences/neurips2025/storage/workflow-runs/<run-id>/summary.json`
and the proof report at
`conferences/neurips2025/storage/workflow-runs/<run-id>/small-neurips-smoketest-report.json`.
The required smoke-test path uses dry-run or mock behavior instead of live crawling or live transcription.

## Generic Conference Smoke Test

Run a generic conference proof path with:

```bash
conference-pipeline conference-smoketest --workspace conferences/iclr2025 --stages analyze,deck --topic 01 --mode dry-run --run-id phase20-iclr2025-smoketest --summary
```

The command accepts any workspace that satisfies the conference workspace
contract. It writes the normal workflow summary at
`conferences/<slug>/storage/workflow-runs/<run-id>/summary.json` and the proof
report at
`conferences/<slug>/storage/workflow-runs/<run-id>/conference-smoketest-report.json`.
Use dry-run or mock mode for new conferences until live source URLs, auth, and
provider behavior have conference-local proof.

The v1.2 ICLR 2025 proof uses `analyze,deck` because the proof topic does not
yet declare the ranks/audio paths required by the podcast adapter.

## Analysis Manifest Contract

The `workflow run --stages analyze` command emits
`conferences/<slug>/analysis/analysis_manifest.json`, which the `podcast` and
`deck` stages consume as the source of truth for topics and declared paths.
See [Analysis Contract Runbook](operations/analysis-contract.md) for the
operator workflow and [Analysis Manifest Schema](design/analysis-manifest-schema.md)
for the on-disk format.

Backfill a legacy workspace:

```bash
python3 scripts/backfill_analysis_manifest.py --workspace conferences/neurips2025
```

## Standalone Scripts

### run_cvpr2026_production.py

CVPR's live-only runner has two local preparation commands that do not load
credentials, construct provider clients, issue network requests, or write live
stage receipts. Run them in order before the authorized live preflight:

```bash
.venv/bin/python scripts/run_cvpr2026_production.py \
  --workspace conferences/cvpr2026 \
  --run-id cvpr2026-live-production \
  --prepare-preflight-estimate

.venv/bin/python scripts/run_cvpr2026_production.py \
  --workspace conferences/cvpr2026 \
  --run-id cvpr2026-live-production \
  --prepare-storage-estimate
```

The second command requires the first command's named-run estimate and binds
the resulting capacity input to its corpus, inventory, manifest, and estimate
digests. Its remaining-artifact-family reservations come from the reviewed
conference-local `storage/preflight-storage-policy.json`; live preflight still
checks current free capacity immediately before credential loading.

### download_from_sessions.py

The primary script for downloading assets from chosen sessions.

```bash
# Basic usage - downloads all chosen sessions
python3 scripts/download_from_sessions.py

# Limit to first N sessions (for testing)
python3 scripts/download_from_sessions.py --limit 5

# Use a different sessions file
python3 scripts/download_from_sessions.py --sessions-file my_sessions.json

# Retry sessions that previously failed
python3 scripts/download_from_sessions.py --retry-errors
```

**Features:**
- **Auto-resume**: Re-running skips already completed sessions
- **Error retry**: Use `--retry-errors` to retry failed sessions
- **Partial resume**: Resumes incomplete presentations from their last state

**Options:**
| Option | Description |
|--------|-------------|
| `--sessions-file PATH` | Path to sessions.json file (default: `sessions.json`) |
| `--limit N` | Process only first N sessions |
| `--retry-errors` | Retry sessions that previously failed with errors |

### transcribe_sessions.py

Transcribe audio files for downloaded presentations using OpenAI Whisper.

```bash
# Transcribe all presentations with audio
python3 scripts/transcribe_sessions.py

# Limit to first N presentations (for testing)
python3 scripts/transcribe_sessions.py --limit 5

# Retry presentations that previously failed
python3 scripts/transcribe_sessions.py --retry-errors

# Process multiple sessions in parallel
python3 scripts/transcribe_sessions.py --parallel 3

# Adjust rate limiting
python3 scripts/transcribe_sessions.py --max-rpm 100
```

**Features:**
- **Auto-resume**: Re-running skips already transcribed presentations
- **Error retry**: Use `--retry-errors` to retry failed transcriptions
- **Automatic chunking**: Large audio files (>25MB) are automatically split into chunks
- **Parallel processing**: Process multiple sessions and chunks concurrently
- **Rate limiting**: Built-in rate limiting to avoid API throttling
- **Timestamped output**: Generates segments.json only when Whisper returns segments (whisper-1)
- **Error summary**: Reports all failed transcriptions at the end

**Options:**
| Option | Description |
|--------|-------------|
| `--limit N` | Process only first N presentations |
| `--retry-errors` | Retry presentations that previously failed with errors |
| `--parallel N` | Process N sessions in parallel (default: 1 = sequential) |
| `--max-rpm N` | Maximum API requests per minute (default: 50) |

**Output files** (per presentation):
- `transcripts/transcript.txt` - Plain text transcript
- `transcripts/segments.json` - Optional timestamped segments with start/end times

**Note:** `segments.json` is produced only when the provider returns segment timestamps
(typically `OPENAI_MODEL=whisper-1`). The CLI pipeline command writes
`transcripts/transcript.json` instead.

**Note:** OpenAI Whisper has a 25MB file size limit. Files larger than this are automatically
split into ~10 minute chunks using ffmpeg, transcribed separately, and merged back together
with adjusted timestamps. Chunks within a session are also processed in parallel for faster transcription.

### ocr_slides.py

OCR slide images using vision-capable LLMs to extract structured content.

```bash
# OCR all presentations with slides
python3 scripts/ocr_slides.py

# Limit to first N presentations (for testing)
python3 scripts/ocr_slides.py --limit 5

# Retry presentations that previously failed
python3 scripts/ocr_slides.py --retry-errors

# Process slides in parallel (default: 3)
python3 scripts/ocr_slides.py --parallel 5

# Adjust rate limiting
python3 scripts/ocr_slides.py --max-rpm 60

# Use a specific primary model
python3 scripts/ocr_slides.py --provider openai --model gpt-5.6-luna

# Use Gemini as quality fallback for low-confidence/failed primary parses
python3 scripts/ocr_slides.py --fallback-provider gemini --fallback-model gemini-3.1-flash-lite-preview

# Use Anthropic Claude directly
python3 scripts/ocr_slides.py --provider anthropic
```

**Features:**
- **Auto-resume**: Re-running skips already processed presentations
- **Error retry**: Use `--retry-errors` to retry failed OCRs
- **Parallel processing**: Process multiple slides concurrently with rate limiting
- **Transcript context**: Uses synchronized transcript text to improve OCR understanding
- **Structured JSON output**: Extracts title, bullet points, equations, visual elements, key concepts, and references
- **Error summary**: Reports all failed OCRs at the end

**Options:**
| Option | Description |
|--------|-------------|
| `--limit N` | Process only first N presentations |
| `--retry-errors` | Retry presentations that previously failed with errors |
| `--parallel N` | Number of concurrent API requests (default: 3) |
| `--max-rpm N` | Maximum API requests per minute (default: 30) |
| `--provider` | OCR provider: `openai` (default), `anthropic`, or `gemini` |
| `--model` | Primary OCR model override |
| `--fallback-provider` | Optional fallback provider: `none`, `openai`, `anthropic`, or `gemini` |
| `--fallback-model` | Fallback OCR model override |

**Output files** (per presentation):
- `ocr/slide_001.json`, `ocr/slide_002.json`, etc. - Structured JSON per slide

**JSON Output Structure:**
```json
{
  "slide_index": 5,
  "timestamp_sec": 60.56,
  "image_path": "slide_005.png",
  "slide_text": {
    "title": "Slide Title",
    "bullet_points": ["Point 1", "  - Sub-point"],
    "equations": ["\\alpha + \\beta"],
    "other_text": "Additional text",
    "urls": ["https://example.com"]
  },
  "visual_elements": [
    {
      "type": "chart",
      "description": "Bar chart showing performance",
      "data": "X-axis: models, Y-axis: accuracy",
      "labels": ["Model A", "Model B"]
    }
  ],
  "key_concepts": [
    {
      "concept": "Diffusion Models",
      "explanation": "Models that learn to reverse a diffusion process"
    }
  ],
  "references": [
    {
      "title": "Paper Title",
      "authors": "Author et al.",
      "venue": "NeurIPS 2025",
      "year": "2025",
      "url": "arXiv:2401.12345"
    }
  ]
}
```

**Note:** The OCR uses transcript context when available - it matches slide timestamps with
transcript segments to provide the LLM with what the speaker is saying during each slide,
improving extraction accuracy for technical content.

### detect_reference_slides.py

Automatically detect generic/reference slides that appear across multiple presentations (e.g., conference title slides, sponsor slides).

```bash
# Auto-detect reference slides from all presentations
python3 scripts/detect_reference_slides.py

# Only check keynotes and workshops (more likely to have generic slides)
python3 scripts/detect_reference_slides.py --session-types keynote,workshop,tutorial

# Require higher frequency (at least 50% of presentations)
python3 scripts/detect_reference_slides.py --min-frequency 0.5

# Check more slides from start of each presentation
python3 scripts/detect_reference_slides.py --check-first 10

# Preview without saving
python3 scripts/detect_reference_slides.py --dry-run
```

**Features:**
- **Perceptual hashing**: Uses imagehash to find visually similar slides
- **Cross-presentation analysis**: Groups similar slides across all presentations
- **Frequency filtering**: Only selects slides appearing in many presentations
- **Session type filtering**: Focus on specific presentation types
- **Metadata output**: Saves detection parameters and example presentations

**Options:**
| Option | Description |
|--------|-------------|
| `--data-dir PATH` | Path to data directory (default: `data/neurips_2025`) |
| `--output-dir PATH` | Output directory for reference slides |
| `--check-first N` | Number of slides from start to check (default: 5) |
| `--session-types TYPES` | Comma-separated session types to check |
| `--min-frequency FLOAT` | Minimum frequency threshold (default: 0.3 = 30%) |
| `--limit N` | Maximum presentations to check |
| `--dry-run` | Preview without saving files |

**Output:**
- `reference_slides/reference_01.png`, etc. - Representative slide images
- `reference_slides/metadata.json` - Detection metadata and parameters

### cleanup_slides.py

Mark duplicate and generic slides for skipping during OCR (reduces API costs).

```bash
# First time: detect and save reference slides
python3 scripts/detect_reference_slides.py

# Run cleanup on all presentations
python3 scripts/cleanup_slides.py

# Or create reference from known presentations
python3 scripts/cleanup_slides.py --create-reference

# Preview changes without modifying files
python3 scripts/cleanup_slides.py --dry-run --verbose

# Limit to first N presentations
python3 scripts/cleanup_slides.py --limit 10
```

**Features:**
- **Consecutive duplicate detection**: Uses perceptual hashing to find repeated slides
- **Reference slide matching**: Marks generic conference slides
- **Non-destructive**: Only adds `skip_ocr` flags to sync.json, never deletes files
- **Resume-safe**: Can be re-run safely on already-processed presentations

**Options:**
| Option | Description |
|--------|-------------|
| `--data-dir PATH` | Path to data directory (default: `data/neurips_2025`) |
| `--reference-dir PATH` | Path to reference slides directory |
| `--create-reference` | Create reference slides from sample presentations |
| `--limit N` | Process only first N presentations |
| `--dry-run` | Don't modify files, just show what would be done |
| `--verbose`, `-v` | Show detailed output per presentation |

**Slide marking:**
Slides are marked in `sync.json` with:
```json
{
  "index": 1,
  "image": "slide_001.png",
  "skip_ocr": true,
  "skip_reason": "generic_conference_slide"
}
```

Skip reasons:
- `duplicate_of_previous` - Nearly identical to previous slide
- `generic_conference_slide` - Matches a reference slide pattern

### Whole-Pipeline Provider Cost Estimates

Use the canonical cost estimator before approving paid provider work. It reads
`src/conference_pipeline/costs/pricing_catalog.json` and
`src/conference_pipeline/costs/task_specs.yaml`, both refreshed on
`2026-04-26`.

```bash
# Full base-scenario estimate with machine-readable output
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/iclr2025 \
  --scope full \
  --scenario base \
  --json

# Low/base/high comparison
.venv/bin/python -m conference_pipeline.costs compare \
  --workspace conferences/iclr2025 \
  --scope full

# Compatibility wrapper
.venv/bin/python scripts/estimate_pipeline_costs.py \
  --workspace conferences/iclr2025 \
  --compare-models

# Best-effort receipt scan after a run
.venv/bin/python -m conference_pipeline.costs actual \
  --workspace conferences/iclr2025 \
  --scope full \
  --json

# Dry-run OCR model experiment before live provider approval
.venv/bin/python -m conference_pipeline.ocr_benchmark run \
  --workspace conferences/iclr2025 \
  --sample-size 12 \
  --budget-cap-usd 30 \
  --per-call-timeout-seconds 60 \
  --json
```

**Features:**
- Measures audio duration, image dimensions, PDF pages, text token estimates,
  topic count, and presentation targets from a workspace.
- Estimates ASR, visual OCR, PDF cleanup, exhaustive extraction, OpenRouter
  failover, GPT Researcher, embeddings, Tavily credits, topic reports,
  rankings, podcast/deck/video generation, and TTS.
- Reports local-only downloads, browser rendering, `ffmpeg`, Tesseract, and
  local PDF text extraction as `$0 API cost` line items.
- Keeps historical NeurIPS ASR/OCR estimates as historical evidence instead of
  silently rewriting them with current prices.
- The OCR benchmark compares cheaper OpenAI, Anthropic, and Gemini image
  parsers against an OpenAI authority and writes receipts under
  `storage/transcription-ocr/ocr-model-benchmark/`; add `--live` only after
  budget/provider approval.

### discover_sessions.py

Discover available sessions from NeurIPS schedule.

```bash
python3 scripts/discover_sessions.py --year 2025
```

Creates `sessions.json` with all discovered sessions.

**Options:**
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output JSON file (default: `sessions.json`) |
| `--year`, `-y` | Conference year (default: from settings) |
| `--no-browser` | Don't use browser for scraping (faster but may miss content) |

### show_stats.py

Show download statistics and progress.

```bash
# Summary statistics
python3 conferences/neurips2025/scripts/show_stats.py

# Include largest presentations
python3 conferences/neurips2025/scripts/show_stats.py --detailed

# Output as JSON (for scripting)
python3 conferences/neurips2025/scripts/show_stats.py --json
```

**Options:**
| Option | Description |
|--------|-------------|
| `--detailed` | Show detailed stats including largest presentations |
| `--json` | Output stats as JSON |
| `--data-dir PATH` | Path to data directory (default: `data/neurips_2025`) |
| `--sessions-file PATH` | Path to sessions.json (default: `sessions.json`) |

### sync_to_gdrive.py

Sync one conference workspace to a profile-specific Google Drive destination
using manifest-driven rsync selection.

For release and backup procedure, evidence checks, and destructive-operation
gates, use the focused runbooks:
[public-release.md](operations/public-release.md) and
[private-backup.md](operations/private-backup.md).

```bash
# Preview the public NeurIPS 2025 sync
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public --dry-run

# Preview the backup NeurIPS 2025 sync
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw" --dry-run

# Override the configured destination for one run
python3 scripts/sync_to_gdrive.py --conference-dir conferences/iclr2025 --profile public --remote-path /mounted/Drive/ICLR-2025

# Allow delete mode only after reviewing a matching delete-preview dry run
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw" --dry-run --delete --receipt-path conferences/neurips2025/storage/workflow-runs/<run-id>/backup/delete-preview-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw" --delete --reviewed-dry-run-receipt conferences/neurips2025/storage/workflow-runs/<run-id>/backup/delete-preview-receipt.json
```

**Options:**
| Option | Description |
|--------|-------------|
| `--conference-dir PATH` | Conference workspace directory such as `conferences/neurips2025` |
| `--config PATH` | Conference config path such as `conferences/neurips2025/config.yaml` |
| `--profile {public,backup}` | Required sync profile |
| `--remote-path PATH` | Override the configured destination path for the selected profile without changing manifest filtering |
| `--dry-run` | Show what would be synced without mutating the destination |
| `--init-only` | Create the selected mounted Drive leaf/family directories without copying payload artifacts |
| `--delete` | Delete remote files inside each included family destination that are not in the selected local sync set |
| `--reviewed-dry-run-receipt PATH` | Bind destructive `--delete` to a successful matching `--dry-run --delete` receipt with the same manifest, target, and artifact snapshot |
| `--reviewed-dry-run` | Honor-system acknowledgement that the dry-run output was reviewed before `--delete` |
| `--quiet` | Suppress progress output |

**Destination defaults:**
- `GDRIVE_PUBLIC_PATH` supplies the default public destination
- Backup syncs require an explicit `--remote-path` pointing at the mounted conference leaf
- `GDRIVE_REMOTE_PATH` is deprecated compatibility guidance and should not be used for new flows

**Notes:**
- Family selection comes from `storage_manifest.json`, not from hard-coded path switches.
- `path` and `path_glob` manifest families are supported.
- `--init-only` is for new conference setup and writes only directories.
- During --dry-run, the script reports state but does not create the conference leaf or family directories.
- The dry-run report now includes `Included families:`, `Excluded families:`,
  `Target root status:`, and `Leaf status:` so you can verify both profile
  intent and mounted-root reachability before syncing.
- During `--dry-run`, `Leaf status:` reports `existing` or `missing`; `created`
  is reserved for write mode after the conference leaf is actually
  materialized.
- Legacy or out-of-profile top-level directories already present in the mounted
  leaf remain untouched. `--delete --reviewed-dry-run` only prunes within the
  included family destinations. `--delete --reviewed-dry-run-receipt` applies
  the same scope after a matching `--dry-run --delete` receipt; the older
  acknowledgement is retained for compatibility.

### prune_conference_workspace.py

Recoverability-first local cleanup for rebuildable conference families.

For quarantine, restore verification, recovery, and purge procedure, use the
focused runbook:
[cleanup-and-recovery.md](operations/cleanup-and-recovery.md).

```bash
# Inspect manifest-driven prune candidates
python3 scripts/prune_conference_workspace.py report \
  --conference-dir conferences/neurips2025

# Verify mounted-drive restore integrity for eligible remote-verified families before quarantine
python3 scripts/prune_conference_workspace.py verify-restore \
  --conference-dir conferences/neurips2025 \
  --profile backup \
  --tier remote-verified \
  --remote-path "/mounted/Drive/NeurIPS 2025-raw"

# Move eligible families into repo-local quarantine after verification
python3 scripts/prune_conference_workspace.py execute-quarantine \
  --conference-dir conferences/neurips2025 \
  --quarantine-dir conferences/neurips2025/.quarantine \
  --tier cache

# Permanently purge expired quarantine batches later
python3 scripts/prune_conference_workspace.py purge \
  --conference-dir conferences/neurips2025 \
  --quarantine-dir conferences/neurips2025/.quarantine \
  --min-age-hours 168
```

**Notes:**
- Candidates come from manifest fields such as `rebuildable`, `storage_class`,
  and `sync_profiles`, not from ad hoc path lists.
- Cleanup path resolution follows conference-local config pointers:
  `storage.manifest_path` selects the manifest, and
  `storage.support_material.storage` is the root for `inventory.json` and
  `prune-verification/` evidence.
- `verify-restore` writes profile-specific evidence under
  `conferences/<slug>/storage/prune-verification/`.
- verify-restore fails closed for cache-tier or profile-ineligible families.
- Use `--tier remote-verified` or explicit `--family` values to narrow restore
  verification to eligible families.
- `execute-quarantine` writes a machine-readable `receipt.json` for each batch.
- `conferences/*/.quarantine/` is ignored by default so temporary cleanup work
  does not pollute git status.
- Live family paths are never deleted directly; quarantine and purge remain
  separate operations.

### merge_chosen_sessions.py

Merge `chosen='Y'` values from old sessions.json into new sessions.json.

```bash
python3 conferences/neurips2025/scripts/merge_chosen_sessions.py
```

This preserves your manual filtering choices when regenerating the sessions file.
Expects `sessions_old.json` and `sessions.json` to exist in the current directory.

### create_cookies_file.py

Interactive helper to create `cookies.json` from your browser session.

```bash
python3 conferences/neurips2025/scripts/create_cookies_file.py
```

Follow the prompts to extract and save your NeurIPS session cookie.

### Analysis and Research Scripts

Commands that align with the NeurIPS 2025 analysis plan:

```bash
.venv/bin/python scripts/setup_vector_store_enhanced.py
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Your research question"
.venv/bin/python scripts/analyze_presentation_coverage.py
.venv/bin/python scripts/qa_analysis_docs.py
.venv/bin/python scripts/traceability_recovery.py conferences/neurips2025/analysis
.venv/bin/python scripts/phase8_paper_analysis.py --score
python scripts/run_exhaustive_extraction.py --mode presentations --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

## Module CLI

The pipeline also provides a Typer-based CLI:

```bash
.venv/bin/python -m conference_pipeline.cli [COMMAND] [OPTIONS]
```

Or if installed:

```bash
neurips-pipeline [COMMAND] [OPTIONS]
```

## Commands

### discover

Scrape the NeurIPS schedule and discover all presentations.

```bash
.venv/bin/python -m conference_pipeline.cli discover
```

This creates presentation records in the database with status `DISCOVERED`.

### download-assets

Download audio, video, and slides for presentations.

```bash
# Download for all presentations
.venv/bin/python -m conference_pipeline.cli download-assets

# Download for a specific presentation
.venv/bin/python -m conference_pipeline.cli download-assets paper-12345
```

**Arguments:**
- `PRES_ID`: Process only this presentation ID

### transcribe

Transcribe audio files to text.

```bash
.venv/bin/python -m conference_pipeline.cli transcribe [PRES_ID]
```

Uses the configured transcription provider (OpenAI, Anthropic, or local Whisper).
Skips presentations that are already transcribed or have an existing transcript on disk.
`run-all` will run this stage automatically after assets are downloaded.

**Arguments:**
- `PRES_ID`: Process only this presentation ID

**Outputs:** `transcripts/transcript.txt` and `transcripts/transcript.json` are written
via `StorageBackend`. The CLI does not write `segments.json`.

### extract-slides

Extract slide images from PDFs or video.

```bash
.venv/bin/python -m conference_pipeline.cli extract-slides [PRES_ID]
```

**Arguments:**
- `PRES_ID`: Process only this presentation ID

### ocr-slides

Run OCR on extracted slide images.

```bash
.venv/bin/python -m conference_pipeline.cli ocr-slides [PRES_ID]
```

**Arguments:**
- `PRES_ID`: Process only this presentation ID

### extract-refs

Extract paper references from OCR text and transcripts.

```bash
.venv/bin/python -m conference_pipeline.cli extract-refs [PRES_ID]
```

**Arguments:**
- `PRES_ID`: Process only this presentation ID

### download-papers

Resolve references and download open-access PDFs.

```bash
.venv/bin/python -m conference_pipeline.cli download-papers [PRES_ID]
```

**Arguments:**
- `PRES_ID`: Process only this presentation ID

### run-all

Execute the complete pipeline from discovery to paper download.

```bash
.venv/bin/python -m conference_pipeline.cli run-all [OPTIONS]
```

**Options:**
- `--retry-errors`, `-r`: Retry presentations that previously errored
- `--reprocess-completed`: Reprocess already completed presentations

### resume

Resume the pipeline from where it left off.

```bash
.venv/bin/python -m conference_pipeline.cli resume [OPTIONS]
```

Skips presentations that are already completed (`PAPERS_DOWNLOADED` status).
Top-level resume uses presentation-status checkpoints; workflow resume uses stage receipts.

**Options:**
- `--retry-errors`, `-r`: Also retry errored presentations

### reset-errors

Reset errored presentations to allow retry.

```bash
# Reset all errored presentations
.venv/bin/python -m conference_pipeline.cli reset-errors

# Reset a specific presentation
.venv/bin/python -m conference_pipeline.cli reset-errors paper-12345
```

**Arguments:**
- `PRES_ID`: Reset only this presentation ID

### stats

Show pipeline status and statistics.

```bash
.venv/bin/python -m conference_pipeline.cli stats
```

Displays:
- Total presentations discovered
- Count by status
- Error summary
- Storage statistics

## Examples

### Full Pipeline Run

```bash
# First time: discover and process everything
.venv/bin/python -m conference_pipeline.cli discover
.venv/bin/python -m conference_pipeline.cli run-all
```

### Resume After Interruption

```bash
# Resume from where we left off
.venv/bin/python -m conference_pipeline.cli resume

# Also retry previously failed presentations
.venv/bin/python -m conference_pipeline.cli resume --retry-errors
```

### Process Single Presentation

```bash
# Download assets for one presentation
.venv/bin/python -m conference_pipeline.cli download-assets spotlight-12345

# Run all stages for that presentation
.venv/bin/python -m conference_pipeline.cli transcribe spotlight-12345
.venv/bin/python -m conference_pipeline.cli extract-slides spotlight-12345
.venv/bin/python -m conference_pipeline.cli ocr-slides spotlight-12345
.venv/bin/python -m conference_pipeline.cli extract-refs spotlight-12345
.venv/bin/python -m conference_pipeline.cli download-papers spotlight-12345
```

### Reset and Retry Errors

```bash
# Check status
.venv/bin/python -m conference_pipeline.cli stats

# Reset all errors
.venv/bin/python -m conference_pipeline.cli reset-errors

# Retry
.venv/bin/python -m conference_pipeline.cli run-all --retry-errors
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error (missing credentials, etc.)

## Logging

The CLI respects the `LOG_LEVEL` environment variable:

```bash
# Verbose output
LOG_LEVEL=DEBUG .venv/bin/python -m conference_pipeline.cli discover

# Quiet output
LOG_LEVEL=WARNING .venv/bin/python -m conference_pipeline.cli run-all
```

## Source References

- scripts/download_from_sessions.py
- scripts/transcribe_sessions.py
- scripts/ocr_slides.py
- scripts/setup_vector_store_enhanced.py
- scripts/query_gpt_researcher_hybrid.py
- src/conference_pipeline/cli.py
