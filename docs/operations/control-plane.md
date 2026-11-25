# Local Control Plane Runbook

The local control plane is a bounded browser console for one operator on one
machine. It wraps the canonical workflow stage APIs, writes the same
workspace-local receipts as the CLI, and previews only server-indexed
artifacts from the selected conference workspace.

## Start The Local Control Plane

Start the server on loopback:

```bash
conference-pipeline control-plane --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` in the same machine's browser. The first screen
is the operator console with launch controls, run status, receipts, artifact
browser, and preview pane.

Defaults:

- Workspace: `conferences/neurips2025`
- Mode: `dry-run`
- Stages: `analyze,podcast,deck`

The server binds to `127.0.0.1` by default and creates a per-process action
token for launch and preset POSTs. The token is rendered into the local form
and sent by the JavaScript enhancement as `x-control-plane-token`.

## Launch Runs

Use `Launch Run` to choose a workspace, mode, stage set, optional topic,
optional run id, and optional resume flag. Stage ids come from the canonical
workflow registry in this order: `discover`, `download`, `transcribe`,
`analyze`, `podcast`, `deck`.

The launch form works without JavaScript. JavaScript only enhances the same
form with token-aware `fetch` submission and redirects the browser to the run
detail view after the API accepts the run.

The local run manager allows one active run at a time. If a run is still active,
another launch is rejected instead of hidden in a queue. Use dry-run for the
default operator preview path and use mock when deterministic local success
receipts are enough.

## Launch Smoke Test

`Launch Smoke Test maps to small-neurips-smoketest with mode=dry-run, workspace=conferences/neurips2025, stages=analyze,podcast,deck, and topic=01_tools_and_open_source; Phase 14 owns the production smoke-test proof.`

The same proof can be run from the CLI:

```bash
conference-pipeline small-neurips-smoketest --workspace conferences/neurips2025 --mode dry-run --run-id phase14-smoketest --summary
```

The command writes the normal workflow summary at
`conferences/neurips2025/storage/workflow-runs/<run-id>/summary.json` and the
proof report at
`conferences/neurips2025/storage/workflow-runs/<run-id>/small-neurips-smoketest-report.json`.
The required smoke-test path uses dry-run or mock behavior instead of live crawling or live transcription.

## Inspect Runs And Receipts

Open `/runs/<run-id>` from the launch response or from the browser after a
successful form launch. The run detail view shows:

- Summary status and mode.
- `next_resume_stage` when the run is resumable.
- Stage timeline in canonical stage order.
- Receipt rows with stage id, status, mode, topic, timestamps, resumable value,
  next resume stage, receipt filename, artifact count, and error text.
- Collapsed command previews and digests under `Details`.

Receipts remain the durable source of truth. The control plane reads
`summary.json` and per-stage receipt JSON under
`storage/workflow-runs/<run-id>/`, then overlays only the in-memory active state
for the currently running local process.

## Browse And Preview Artifacts

Artifact records are server-generated from three provenance sources:

- `receipt` artifacts recorded by workflow stage receipts.
- `analysis manifest` paths from `analysis/analysis_manifest.json`.
- `allowlisted workspace scan` results under supported output roots such as
  `analysis/`, `audio/`, `video/`, `decks/`, `presentations/`, and
  `storage/workflow-runs/<run-id>/`.

Each artifact row shows the label, workspace-relative path, provenance source,
preview type, and controls to preview, copy the preview URL, or open the
server-generated route in a new tab.

Preview routes are workspace-contained. The browser uses server-generated
`/api/runs/{run_id}/artifacts/{artifact_id}/preview` URLs and never submits a
raw absolute path; raw absolute path submission is not part of the control-plane
contract. Markdown and JSON previews are escaped text, audio and video use
native browser controls, document previews use the browser document viewer, and
deck HTML is loaded only in a sandboxed frame.

## Safety Boundaries

- The default command binds to loopback: `conference-pipeline control-plane --host 127.0.0.1 --port 8765`.
- Launch and preset POSTs require the per-process `x-control-plane-token`.
- Stage selection accepts registered stage ids only, not arbitrary commands.
- The smoke preset stays on `mode=dry-run` for Phase 13.
- The artifact browser exposes workspace-relative records and opaque artifact
  ids, not raw filesystem input from the browser.
- Receipt and status rendering is sanitized so environment marker names and
  local secret-file content are not displayed.
- The Phase 13 UI is local and operator-focused; broader product controls stay
  outside this runbook.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Browser cannot connect | Confirm the command is still running and that the URL is `http://127.0.0.1:8765/`. |
| Launch is rejected | Check whether another run is active; the control plane permits one active run at a time. |
| Launch returns a token error | Refresh the page so the form has the current per-process action token. |
| Run detail cannot load | Confirm `storage/workflow-runs/<run-id>/summary.json` exists under the selected workspace. |
| Artifact does not preview | Confirm the file is inside the workspace and has a supported preview extension. |
| Preview is empty | The current checkout may not contain audio, video, deck, document, JSON, or markdown outputs for that run. |
