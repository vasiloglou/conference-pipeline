# workflow.py and workflow_runs.py - Canonical Workflow Registry and Receipts

Reusable stage registry, workspace resolution, mode-aware workflow execution,
stage receipts, receipt-backed resume, and summary inspection for the operator
surface.

## Location

- `src/conference_pipeline/workflow.py`
- `src/conference_pipeline/workflow_runs.py`

## Responsibilities

- define the ordered canonical stage ids: `discover`, `download`,
  `transcribe`, `analyze`, `podcast`, `deck`, and the first-class
  presentation checkpoint stages
- validate conference workspaces through `config.yaml` plus
  `storage/storage_manifest.json`
- load operator-facing workflow defaults from the conference-local
  `workflow:` section in `config.yaml`
- expose explicit dispatch metadata for direct Python stages and
  subprocess-backed stages
- build the current subprocess command sequences for `analyze`, `podcast`,
  `deck`, and presentation-stage command previews
- execute selected stages in live, dry-run, or mock mode
- write one receipt per attempted stage plus `summary.json` for each run
- halt on blocker-class failures and persist unblock-aware summary fields
- reject run-id reuse unless `--resume` is set
- resume only from compatible succeeded stage receipts

## Canonical Commands

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

Receipts are stored at
`conferences/<slug>/storage/workflow-runs/<run-id>/summary.json`; for example,
`conferences/neurips2025/storage/workflow-runs/preview-run/summary.json`.

Dry-run records planned stage work without constructing live browser, network, or provider clients.
Mock mode records deterministic local success receipts without live browser, network, or provider calls.
Stage receipt resume skips only compatible succeeded receipts for the same workspace, stage, mode, topic, implementation, config, and command preview.
Existing workflow-runs/<run-id>/ directories are rejected unless --resume is set.
Every attempted stage writes a receipt, and failed stages persist summary.json before the command exits nonzero.
Blocked stages also write `BlockerReceipt` artifacts; `workflow unblock` or
`python -m conference_pipeline.workflow_unblock` writes the matching
`UnblockReceipt` and refreshes `summary.json`.

First-class presentation stages are presentation_blueprint, deck_contract_generation, source_binding, visual_plan_generation, render_spec_generation, render_export, visual_qc, delivery_generation, presentation_review, video_generation, video_sync_validation, and human_review. `workflow deck` remains a backward-compatible alias for legacy deck command construction; it is not the full presentation lifecycle. Presentation stage receipts include selected scope and presentation_manifest digests, and resume compatibility binds the selected scope, manifest content, config, and command preview. First-class presentation stages now provide workspace-aware command planning, safe dry-run script contracts, structural validation, and receipt-backed proof reports; live cross-conference production still requires conference-local proof.

Topic-deep-dive blueprint generation consumes `presentation_frame.json` through
`presentation.generate_editorial_blueprint`; the frame separates retrieval
topic title from public deck title and records `presentation_frame_validation`
before deck-contract generation.

For first-class presentation stages, command planning resolves the selected
workspace's `decks/presentation_manifest.json`, applies `--presentation-type`
and `--topic` where required, and builds command metadata without importing or
executing provider-backed owner scripts.

Presentation stage dry-run receipts include command_metadata with exact argv, selected scope, input artifacts, output artifacts, support level, provider/tool requirements, and unsupported_reason.
Workflow-planned presentation scripts use decks/<presentation>/... paths; presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning.
Unsupported reusable stages expose unsupported_reason instead of claiming script execution.
Video generation is dry-run/planning-only in Phase 24 and does not prove final MP4 generation, visual validation, or cross-conference proof.

## Local Control Plane API Routes

The loopback control plane is started with
`conference-pipeline control-plane --host 127.0.0.1 --port 8765` and exposes a
small browser/API surface over the same workflow registry and receipt files:

- `POST /api/runs` launches selected registered stage ids for a workspace.
- `POST /api/presets/small-neurips-smoketest` launches the safe Phase 13 preset.
- `GET /api/runs/{run_id}` reads the receipt-backed run summary and active-state overlay.
- `GET /api/runs/{run_id}/artifacts` returns server-generated artifact records.
- `GET /api/runs/{run_id}/artifacts/{artifact_id}/preview` serves supported previews.

Launch and preset POSTs require the per-process `x-control-plane-token`. The
artifact list combines `receipt` artifacts, `analysis manifest` paths, and an
`allowlisted workspace scan` so the UI can show the files operators expect
without expanding the receipt schema. Preview routes are workspace-contained:
the browser submits an opaque artifact id from the server-generated record and
never submits a raw absolute path. Raw absolute path submission is not accepted
by the control-plane preview contract.

## Small NeurIPS Smoke Test Proof

Run the reusable production-pipeline proof with:

```bash
conference-pipeline small-neurips-smoketest --workspace conferences/neurips2025 --mode dry-run --run-id phase14-smoketest --summary
```

The proof uses topic `01_tools_and_open_source` and stages
`analyze,podcast,deck`. The workflow summary remains at
`conferences/neurips2025/storage/workflow-runs/<run-id>/summary.json`, and the
smoke-test report is written to
`conferences/neurips2025/storage/workflow-runs/<run-id>/small-neurips-smoketest-report.json`.
The required smoke-test path uses dry-run or mock behavior instead of live crawling or live transcription.

## Generic Conference Smoke Test Proof

Run the reusable conference proof command with:

```bash
conference-pipeline conference-smoketest --workspace conferences/iclr2025 --stages analyze,deck --topic 01 --mode dry-run --run-id phase20-iclr2025-smoketest --summary
```

The generic proof writes `conference-smoketest-report.json` next to the workflow
`summary.json` under
`conferences/<slug>/storage/workflow-runs/<run-id>/`. The report path is
`conferences/<slug>/storage/workflow-runs/<run-id>/conference-smoketest-report.json`.
It is intended for
conference-general dry-run or mock proof before a new conference is promoted to
live support. Use `analyze,podcast,deck` only when the selected workspace's
analysis manifest declares the topic-level ranks and audio paths required by
the podcast adapter.

## conference_pipeline.analysis - Analysis Manifest Contract

Phase 12 adds the `conference_pipeline.analysis` subpackage, which owns the
`analysis_manifest.json` contract between analyze, podcast, and deck stages.

Key names:

- `AnalysisManifest`, `TopicEntry`, `LoadedManifest`: frozen dataclasses modeling the schema.
- `emit_analysis_manifest(workspace, *, generator=None)`: atomic emission. Used by `run_analyze_stage` and `scripts/backfill_analysis_manifest.py`.
- `load_and_validate_manifest(workspace, *, require_path_existence=True)`: fail-closed loader used by podcast and deck adapters and by the backfill script for post-write verification.
- `run_analyze_stage(workspace, *, env_file=None, storage_root=None, step_runner=None, overwrite_existing=False)`: Python-stage entrypoint for `analyze`. Registered as `dispatch_kind="python_stage"`, `implementation_ref="conference_pipeline.analysis.run_analyze_stage@v1"`. Existing analysis artifacts are preserved by default; destructive regeneration must opt in with `overwrite_existing=True`.
- `ManifestValidationError`, `AnalysisContractError`: narrow error hierarchy. `AnalysisContractError` is not a `ValueError` subclass.

The manifest file path is `conferences/<slug>/analysis/analysis_manifest.json`
and `schema_version == 1`. See
[Analysis Manifest Schema](../design/analysis-manifest-schema.md).

The analyze stage receipt (Phase 11 schema unchanged) gains one `artifacts`
entry of kind `"analysis_manifest"` containing `path`, `manifest_sha256`
(whole-file digest), `manifest_content_sha256` (content-stable digest; resume
key), and `sub_step_digests`.

## Key Types

### `WorkflowStage`

Registry entry with:

- `id`
- `order`
- `description`
- `dispatch_kind`
- `implementation_ref`
- `config_key`
- `supports_topic_filter`
- `requires_workspace_files`

### `WorkflowWorkspace`

Validated workspace handle with:

- `root`
- `config_path`
- `manifest_path`
- `config_data`
- `workflow`

### `WorkflowExecutionMode`

Execution mode enum with `live`, `dry-run`, and `mock`.

### `WorkflowRunOptions`

Run options record with:

- `run_id`
- `mode`
- `resume`
- `topic`
- `presentation_type`
- `env_file`
- `storage_root`

### `StageReceipt`

Per-stage receipt record persisted as JSON under the run directory.

Required receipt JSON keys:

- `schema_version`
- `run_id`
- `workspace`
- `stage_id`
- `stage_order`
- `mode`
- `topic`
- `selected_scope`
- `presentation_manifest`
- `implementation_ref`
- `status`
- `started_at`
- `ended_at`
- `resumable`
- `skip_reason`
- `resume_compatibility_key`
- `config_digest`
- `command_digest`
- `commands`
- `command_metadata`
- `artifacts`
- `error`
- `next_resume_stage`
- `receipt_path`
- `output_artifacts`
- `output_digest`
- `lineage`

`output_artifacts`, `output_digest` and `lineage` are the v5.1 receipt-lineage
fields: what the stage produced (digested per artifact and as one canonical
`output_digest` over them, never over the receipt file) and what it consumed
(`lineage.upstream_receipts` keyed by workspace-relative receipt path with the
upstream's `output_digest` or `content_digest`, `lineage.input_artifacts` with
file sha256s, `lineage.lineage_digest` over both). See
[Stage Owner Audit — receipt lineage](../reference/stage-owner-audit.md#receipt-lineage).

`commands` stores argv previews for subprocess-backed stages. It records command
arguments, not expanded environment credentials or provider secrets.
`command_metadata` stores JSON-safe presentation command plans. Each entry
contains `argv`, `selected_scope`, `input_artifacts`, `output_artifacts`,
`support_level`, `provider_requirements`, and `unsupported_reason`; provider
requirements are requirement names only, not secret values.
For first-class presentation stages, `selected_scope` contains
`presentation_id`, `presentation_type`, and `topic`, while
`presentation_manifest` contains `path`, `manifest_sha256`, and
`content_sha256`.

### `WorkflowRunSummary`

Machine-readable run index persisted as `summary.json`. It includes run identity,
workspace, mode, topic, presentation type, summary path, next resumable stage,
serialized stage receipts, and the v1.8 halt/unblock fields:

- `halt_status`
- `halt_stage_id`
- `blocker_receipt_path`
- `unblock_acknowledgements`

## Key Functions

```python
ordered_workflow_stages() -> tuple[WorkflowStage, ...]
parse_stage_selection(selection: str | None) -> list[WorkflowStage]
resolve_workspace(workspace: str | Path) -> WorkflowWorkspace
list_workflow_topics(workspace: WorkflowWorkspace) -> list[str]
run_selected_stages(
    workspace: WorkflowWorkspace,
    selected_stages: Sequence[WorkflowStage],
    *,
    topic: str | None = None,
    presentation_type: str | None = None,
    env_file: Path | None = None,
    storage_root: Path | None = None,
    mode: str = "live",
    run_id: str | None = None,
    resume: bool = False,
    write_receipts: bool = False,
) -> WorkflowRunSummary | None
validate_run_id(run_id: str) -> str
workflow_run_dir(workspace: WorkflowWorkspace, run_id: str) -> Path
run_workflow_stages(
    workspace: WorkflowWorkspace,
    selected_stages: Sequence[WorkflowStage],
    options: WorkflowRunOptions,
) -> WorkflowRunSummary
load_workflow_summary(workspace: WorkflowWorkspace, run_id: str) -> dict[str, Any]
run_conference_smoketest(...) -> SmokeTestReport
```

## workflow_runs.py - Workflow Run Receipts

`workflow_runs.py` owns the receipt and resume contract around the Phase 10
registry. `WorkflowExecutionMode` decides whether a stage uses live execution,
planned dry-run receipts, or deterministic mock success receipts.
`WorkflowRunOptions` carries the run id, mode, topic, resume flag, and optional
live runtime inputs. `StageReceipt` is the durable resume unit, and
`WorkflowRunSummary` is the operator and future UI index over those receipts.

`validate_run_id` rejects blank, path-traversal, nested-path, and non-slug run
ids. `workflow_run_dir` resolves the run directory under the selected
workspace's support storage root. `run_workflow_stages` writes receipts and
updates `summary.json` after every attempted or skipped stage.
`load_workflow_summary` reads the persisted summary for `workflow summary`.
Resume compatibility includes the stage's `lineage_digest`: before deciding
to skip, the runner binds what the stage would consume *now* (every earlier
workflow stage's `output_digest` plus registry-declared inputs) and compares
it with the digest the persisted receipt recorded; a receipt without
`lineage` or `output_digest`, or with a different `lineage_digest`, is never
reused, so re-running an upstream stage with different outputs re-runs every
stage below it, while a no-op re-run keeps downstream receipts reusable.
The unblock flow refreshes the same summary file so operators can inspect
`halt_status`, `halt_stage_id`, `blocker_receipt_path`, and
`unblock_acknowledgements` without rerunning the blocked stage first.

## Notes

- The workflow surface standardizes stage naming, mode-aware execution, and
  receipt-backed resume behavior.
- Output-stage defaults come from the workspace `config.yaml` instead of the
  legacy wrapper shell constants.
- Top-level resume uses presentation-status checkpoints; workflow resume uses stage receipts.
