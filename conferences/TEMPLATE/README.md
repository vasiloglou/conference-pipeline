# Conference Workspace Template

Use this directory as the starting point for a new conference workspace.

## How To Use

```bash
cp -R conferences/TEMPLATE conferences/<conference_slug>
```

Then update:

- `config.yaml`
- this `README.md`
- any conference-specific extraction assumptions

## Minimal Layout

- `config.yaml` - conference metadata, `workspace_layout`, and manifest pointer
- `README.md` - conference-local operating notes
- `storage/` - storage manifest, policy notes, inventory, and verification evidence
- `assets/` - small static assets that should stay in git
- `scripts/` - helper scripts that only belong to this conference

Use these top-level workspace lanes when content is produced:

- `analysis/` - private analysis notes and intermediate writeups
- `decks/` - curated presentation decks and visual release outputs
- `audio/` - podcast, narration, and other release-oriented audio outputs
- `exports/` - private export bundles such as deep-dive or exhaustive extraction output
- `agent-work/` - private agent sidecar work and generated working material
- `cache/` - rebuildable local indexes, experiments, and cache state
- `legacy/` - archived material retained for traceability but not active work

Conference-local subfolders can vary beneath those lanes. Keep the top-level
lane names stable so shared sync, backup, and cleanup docs stay reusable.

`config.yaml` points at `storage/storage_manifest.json`. Update the manifest
when the workspace adds or removes artifact families so public release, private
backup, and local-cache policy remain explicit.

Shared reusable code should stay in the platform layer:

- `src/conference_pipeline/`
- root `scripts/`
- `tests/`
- `docs/`
