# Operations Runbooks

Canonical operator guidance for conference storage, release, backup, cleanup,
and recovery workflows lives here. Conference workspaces can link to these
runbooks for local context, but this directory is the source of truth when the
two surfaces differ.

## Runbooks

- [Public Release](public-release.md) - publish curated shareable artifacts
- [Public Release Scrub Policy](public-release-scrub-policy.md) - what must never reach a public byte, how to scan for it, where it is removed
  from a manifest-backed workspace to the public Drive target.
- [Private Backup](private-backup.md) - back up private analysis, exports,
  sidecar outputs, and retained public deliverables to the private Drive
  target.
- [Cleanup And Recovery](cleanup-and-recovery.md) - inspect prune candidates,
  verify restore coverage, quarantine local copies, and purge expired
  quarantine batches.
- [Future Conference Storage](future-conference-storage.md) - adopt the
  lane-based `workspace_layout` and manifest-backed starter contract for new
  conference workspaces.
- [Local Control Plane](control-plane.md) - start the loopback operator UI,
  launch dry-run or mock workflow runs, inspect receipts, and preview
  workspace-contained artifacts.

## Operator Model

The storage model has four artifact classes:

| Class | Purpose | Sync behavior |
|-------|---------|---------------|
| `git` | Small support material such as README, config, scripts, and storage manifests | Stays in the repository |
| `public` | Curated deliverables intended for sharing | Included in the `public` profile and retained in backup where configured |
| `backup` | Private canonical workspace state | Included in the `backup` profile only |
| `cache` | Rebuildable local working state | Not canonical release or backup material |

The machine-readable manifest controls the sync, cleanup, and workspace
gitignore set. `scripts/generate_workspace_gitignore.py` derives
`conferences/<slug>/.gitignore` from `storage_class`: `git` remains trackable,
while `public`, `backup`, and `cache` stay out of Git unless a small control
file is declared in `gitignore_exceptions`. Markdown deliverables should be
Git-owned for reviewability and may still be mirrored to Drive through
`sync_profiles`. Operator runbooks describe how to read the generated evidence;
they do not replace the manifest or the script output.

New conference production initializes the mounted Drive leaf with
`scripts/sync_to_gdrive.py --profile backup --remote-path <mounted-conference-leaf>
--init-only`, then runs the backup profile immediately after each
artifact-producing stage with the same explicit `--remote-path`. The sync is
manifest-bounded and rsync-incremental so generated payloads move to Drive
without duplicating timestamped payload trees or copying cache/unselected
families.

## Hard Gates

- Public and backup syncs start with `--dry-run`.
- Workspace gitignore policy passes
  `scripts/generate_workspace_gitignore.py --workspace conferences/<slug> --check`.
- Public release review must confirm `Included families:`,
  `Excluded families:`, `Target root status:`, and `Leaf status:`.
- `--delete` is only allowed with `--reviewed-dry-run-receipt` after a matching
  successful `--dry-run --delete` receipt has been reviewed. The older
  `--delete --reviewed-dry-run` acknowledgement remains compatibility-only.
- Remote-verified cleanup candidates need passing `verify-restore` evidence
  before quarantine.
- Quarantine and purge are separate operations. A `receipt.json` must exist
  before a quarantine batch is eligible for later purge.

## NeurIPS 2025 Entry Points

- Workspace README:
  `conferences/neurips2025/README.md`
- Storage policy:
  `conferences/neurips2025/storage/storage_policy.md`
- Archived Phase 3 proof summary:
  `.planning/milestones/v1.0-MILESTONE-AUDIT.md`
- Prune evidence directory:
  conferences/neurips2025/storage/prune-verification/ (`conferences/neurips2025/storage/prune-verification`)
