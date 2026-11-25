# Future Conference Storage Convention

This is the default workspace convention for future conference directories under
`conferences/<slug>/`. It promotes the NeurIPS 2025 storage model into a
reusable starter contract: stable top-level lanes, support material that stays in
git, and a manifest that drives sync and cleanup behavior.

Use this convention when creating a new conference workspace. Keep
conference-local details inside the lanes rather than inventing new top-level
workspace roots.

## Stable Workspace Lanes

Every conference workspace should use these top-level lanes by default:

| Lane | Default purpose |
|------|-----------------|
| `analysis` | Private analysis notes, intermediate writeups, and retained editorial context |
| `decks` | Curated presentation decks, figures, rendered slides, and other public visual outputs |
| `audio` | Podcast scripts, narration assets, and release-oriented audio outputs |
| `exports` | Private export bundles, including deep-dive and exhaustive extraction outputs |
| `agent-work` | Private agent-generated working material and sidecar outputs |
| `cache` | Rebuildable local indexes, experiments, and generated cache state |
| `legacy` | Archived historical material retained for traceability but not used as active workspace state |

The convention fixes only the top-level lane names. Conference-local
substructure may vary beneath those lanes. For example, one conference might
organize `decks` by topic, while another might organize it by audience or final
delivery format. Both are compatible as long as the manifest identifies the
artifact families that sync and cleanup tooling should consider.

## Required Support Material

Each workspace should also carry the support files and directories that make the
storage contract durable:

| Path | Role |
|------|------|
| `README.md` | Human entrypoint for conference-local operating notes |
| `config.yaml` | Conference metadata plus `storage.manifest_path` and `workspace_layout` |
| `storage/` | Manifest, storage policy, inventory, verification evidence, and receipts |
| `assets/` | Small static assets and references that are intentionally versioned |
| `scripts/` | Conference-local helper scripts that do not belong in shared root scripts |

The required machine-readable pointer is:

```yaml
storage:
  manifest_path: "storage/storage_manifest.json"
```

`support_material.storage` is the root for `inventory.json`,
`prune-verification/` evidence, and related cleanup receipts. The reusable
starter default remains `storage`.

The workspace layout block should name the stable lanes explicitly:

```yaml
storage:
  workspace_layout:
    analysis: "analysis"
    decks: "decks"
    audio: "audio"
    exports:
      root: "exports"
      deep_dive: "exports/deep-dive"
      exhaustive_extractions: "exports/exhaustive-extractions"
    agent_work:
      root: "agent-work"
      claude: "agent-work/claude"
      codex: "agent-work/codex"
      gpt_researcher: "agent-work/gpt-researcher"
      notebook_lm: "agent-work/notebook-lm"
    cache:
      root: "cache"
      vector_store: "cache/vector-store"
      experimental_videos: "cache/experimental-videos"
    legacy: "legacy"
  support_material:
    storage: "storage"
    runtime_logs: "storage/logs"
    assets: "assets"
    scripts: "scripts"
```

## Manifest-Driven Contract

The durable storage contract lives in `storage/storage_manifest.json`. The
manifest defines artifact families, paths, storage classes, share policy, which
sync profiles include each family, and the workspace gitignore policy.

Keep the existing storage classes and sync semantics:

| Class | Meaning |
|-------|---------|
| `git` | Small support material that remains versioned in the repository |
| `public` | Curated deliverables eligible for the public release profile |
| `backup` | Private workspace state eligible for the backup profile |
| `cache` | Rebuildable local state that is not canonical release or backup material |

Each workspace must keep a generated `conferences/<slug>/.gitignore` managed by:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py \
  --workspace conferences/<slug> \
  --write --check \
  --receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json
```

The generator treats `storage_class: git` as trackable and ignores `public`,
`backup`, and `cache` families because those payloads belong in Google Drive or
local rebuildable cache. Human-reviewable Markdown deliverables, including
analysis reports, slide briefs, deck markdown, and presenter scripts, should be
Git-owned even when `sync_profiles` also mirrors them to Drive for sharing.
Small control files inside an ignored family may be listed with
`gitignore_exceptions`; payload artifacts such as SVGs, screenshots, rendered
HTML, videos, audio, and PDFs must not be listed there.

The generator also protects the repository from GitHub's 100 MB blob limit. It
uses a 95 MiB safety threshold: untracked oversized files outside already
ignored `public`, `backup`, or `cache` families are added to the generated
workspace `.gitignore`, while oversized files that are already tracked block
`--check` until they are moved into a non-Git artifact family and untracked.

The canonical usage entry for this generator is
[`gitignore.workspace_policy`](../reference/stage-owner-audit.md#gitignore-workspace-policy)
in the Stage Script Registry Tool Usage Index.

Do not rely on ignored empty directories to define the starter contract. The
README, `config.yaml`, manifest, and tests are the durable source of truth. Lane
directories can appear when real content is produced.

## Continuous Drive Offload

New conference setup should create the mounted Drive backup leaf before live
artifact generation starts:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --init-only --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/drive-setup-backup-write-receipt.json
```

`--init-only` creates the conference leaf and manifest-selected family
directories without copying payload artifacts. After that, production stages
that produce artifacts must immediately run the manifest-bounded backup profile
dry run and write sync before downstream stages consume the new outputs:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --dry-run --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-dry-run-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-write-receipt.json
```

The sync is rsync-incremental and bounded by manifest `sync_profiles`, so it
does not create duplicate timestamped Drive payloads and does not copy cache or
unselected families unless the manifest explicitly enables that profile.
The canonical usage entry for the sync tool is
[`backup.sync_to_gdrive`](../reference/stage-owner-audit.md#backup-sync-to-gdrive).

## NeurIPS 2025 Example

NeurIPS 2025 is the reference workspace for this convention:

- `conferences/neurips2025/config.yaml` declares `workspace_layout` and points
  `storage.manifest_path` at `storage/storage_manifest.json`.
- `conferences/neurips2025/storage/storage_manifest.json` maps public
  presentation output to `decks`, audio output to `audio`, private exports to
  `exports/deep-dive` and `exports/exhaustive-extractions`, agent sidecar work
  to `agent-work/*`, rebuildable indexes to `cache/vector-store`, and historical
  archive material to `legacy`.
- The public and backup runbooks use the manifest output as evidence, rather
  than hard-coded assumptions about directories.
- Newer workspaces (the template, ICML 2026, CVPR 2026) map the
  `exhaustive_extractions` family to `storage/extractions`, where the current
  extraction owner writes `<run-id>/by-topic/`, `by-artifact/` and the
  full-read lane; `exports/exhaustive-extractions` is the NeurIPS 2025
  historical location only. A family whose path no owner writes is skipped by
  the backup profile without an error, so the path must follow the owner.

Future conferences should start from the same lane names and then refine the
manifest only where the conference's actual artifact families differ.

## Starter Surfaces

Use these files when creating or reviewing a future workspace:

- [conferences/README.md](../../conferences/README.md)
- [conferences/TEMPLATE/README.md](../../conferences/TEMPLATE/README.md)
- [conferences/TEMPLATE/config.yaml](../../conferences/TEMPLATE/config.yaml)
- [conferences/TEMPLATE/storage/storage_manifest.json](../../conferences/TEMPLATE/storage/storage_manifest.json)
- `conferences/neurips2025/config.yaml`
- `conferences/neurips2025/storage/storage_manifest.json`
