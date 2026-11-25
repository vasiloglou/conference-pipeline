# Private Backup Runbook

Use this runbook to back up canonical private conference state to a private
Drive target. NeurIPS 2025 is the worked example; future conferences should use
the same `backup` profile and evidence checks with their own workspace path.

## Scope

Private backup is for retained workspace state that should survive local
cleanup or machine loss. It is not a publication flow.

For NeurIPS 2025, the backup profile covers private analysis, expensive export
bundles, sidecar agent work, archive material, plus public deliverables that
also need retention.

## Preflight

Review the storage policy and manifest before running backup sync:

- `conferences/<slug>/storage/storage_policy.md`
- `conferences/<slug>/storage/storage_manifest.json`

For NeurIPS 2025:

- Storage policy (`conferences/neurips2025/storage/storage_policy.md`)
- v1.0 milestone audit (`.planning/milestones/v1.0-MILESTONE-AUDIT.md`)
- Backup restore evidence directory (`conferences/neurips2025/storage/prune-verification/backup`)

## Dry Run

Run the backup profile dry run:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --dry-run --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

The preflight report must include:

```text
Included families:
Excluded families:
Target root status:
Leaf status:
```

During --dry-run, the script reports state but does not create the conference leaf or family directories.

During `--dry-run`, `Leaf status:` should report `existing` when the mounted
conference leaf is already present or `missing` when only the mounted parent
root exists. `created` is a write-mode outcome only.

Legacy or otherwise out-of-profile top-level directories already present in the
mounted leaf remain untouched. `--delete --reviewed-dry-run` only prunes within
the included family destinations.

For NeurIPS 2025, the Phase 3 backup dry-run evidence expected the included
families to cover:

- `analysis`
- `claude_autonomy`
- `codex_autonomy`
- `exhaustive_extractions`
- `export_for_deep_dive`
- `gpt_researcher`
- `legacy_codex_paper_influence_archive`
- `notebook_lm_findings`
- `podcasts`
- `presentations`

The expected excluded families include git/support material and rebuildable or
non-canonical local state such as `assets`, `config_yaml`, `readme_md`,
`scripts`, `storage`, `vector_store`, and `experimental_videos`.

## Evidence To Confirm

Before write-mode backup, confirm:

| Evidence | Required result |
|----------|-----------------|
| `Profile:` | `backup` |
| `Target root status:` | `reachable` |
| `Leaf status:` | `existing` or `missing` during dry run; `created` only after a write-mode run materializes the intended private backup target |
| `Included families:` | Private canonical families plus retained public deliverables |
| `Excluded families:` | Git/support material and rebuildable cache-only families stay out |
| `Dry run:` | `enabled` |

Phase 3 recorded passing backup restore proof in:

- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`
- conferences/neurips2025/storage/prune-verification/backup/ (`conferences/neurips2025/storage/prune-verification/backup`)

That restore evidence is the proof source for later cleanup. Do not replace it
with an informal note.

## Write Mode

After reviewing the dry run, run backup sync without `--dry-run`:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

Backup sync requires `--remote-path` for the mounted conference leaf. Keep the
profile as `backup` so manifest filtering does not change.

## Generated Milestone Contract

Generated conference analysis context packets must declare a Backup Posture
section before production planning begins. The reusable default target leaf is
`My Drive/<Conference> <Year>-raw`; the runtime requirement is an explicit
`--remote-path` that points at the mounted private backup leaf.

Before live artifact generation begins, initialize the conference Drive leaf and
manifest-selected family directories without copying payloads:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --init-only --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/drive-setup-backup-write-receipt.json
```

Generated production phases that write artifacts must then run
`scripts/sync_to_gdrive.py --profile backup` once in dry-run mode and once in
write mode immediately after each artifact-producing stage finishes and before
any downstream stage consumes those artifacts, with `--receipt-path` writing
distinct JSON receipts under
`conferences/<slug>/storage/workflow-runs/<run-id>/backup/`:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --dry-run --remote-path "/mounted/Drive/<Conference> <Year>-raw" --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-dry-run-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup --remote-path "/mounted/Drive/<Conference> <Year>-raw" --receipt-path conferences/<slug>/storage/workflow-runs/<run-id>/backup/<stage>-backup-write-receipt.json
```

The sync is manifest-bounded and rsync-incremental: it only moves families
enabled by `sync_profiles`, updates changed files in place, and does not copy
cache/unselected families unless the manifest explicitly enables the selected
profile.

The backup receipt must include workspace, profile, target, manifest sha,
included families, skipped families, dry-run and write exit codes, digest count,
artifact snapshot sha, sync item summaries, timestamp, and `secrets_copied:
false`. The sync helper excludes and preflight-rejects
secret-like files such as `.env`, cookies, tokens, passwords, and credential
files before rsync runs. Missing, failed, stale, or secret-bearing backup
receipts fail closed.

The canonical usage documentation for `scripts/sync_to_gdrive.py` is the
[`backup.sync_to_gdrive`](../reference/stage-owner-audit.md#backup-sync-to-gdrive)
registry entry. Generated plans should cite that registry `usage_doc_refs`
anchor when scheduling backup checkpoints.

## Scheduled Sweep

Run the reusable scheduled backup sweep against active conference workspaces:

```bash
python3 scripts/run_scheduled_backup_sweep.py --conferences-root conferences --target-root "$GDRIVE_BACKUP_ROOT"
```

Add `--write` only after the dry-run behavior and target leaves are reviewed.
When `--target-root` is set, each destination leaf is derived from
`conference.name` in `config.yaml` as `<conference.name>-raw`, falling back to
`<workspace-directory>-raw` only when the display name is absent. Sweeps without
`--target-root` fail closed so every backup leaf is derived from the explicit
mounted Drive root.
Each workspace writes receipts under:

```text
conferences/<slug>/storage/workflow-runs/scheduled-backup/
```

The local control plane exposes the latest scheduled backup status, last
successful run, last failure, digest count, and receipt path without exposing
secret values.

## Delete Gate

Backup delete mode is also destructive.

Hard gate: `--delete` is only allowed with either `--reviewed-dry-run-receipt`
pointing at a matching successful `--dry-run --delete` receipt, or the legacy
`--reviewed-dry-run` acknowledgement. Prefer the receipt-bound gate for all new
production plans.

Only run this after reviewing a matching backup delete-preview dry run:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw" --dry-run --delete --receipt-path conferences/neurips2025/storage/workflow-runs/<run-id>/backup/<stage>-backup-delete-preview-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile backup --remote-path "/mounted/Drive/NeurIPS 2025-raw" --delete --reviewed-dry-run-receipt conferences/neurips2025/storage/workflow-runs/<run-id>/backup/<stage>-backup-delete-preview-receipt.json
```

The receipt must come from the same manifest, target, artifact snapshot, and a
successful `--dry-run --delete` command. If the preview has unexpected missing
families, root-level churn, or private state moving to an unexpected path, stop
before delete mode.

## Restore Verification

Use `verify-restore` before treating remote-verified families as safe for local
cleanup. Example for the NeurIPS 2025 backup target:

```bash
python3 scripts/prune_conference_workspace.py verify-restore --conference-dir conferences/neurips2025 --profile backup --tier remote-verified --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

For the representative backup proof families refreshed in Phase 3 and narrowed
again in Phase 7:

```bash
python3 scripts/prune_conference_workspace.py verify-restore --conference-dir conferences/neurips2025 --profile backup --tier remote-verified --family export_for_deep_dive --family exhaustive_extractions --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

verify-restore fails closed for cache-tier or profile-ineligible families.
Use `--tier remote-verified` or explicit `--family` values to narrow restore
verification to eligible families.

The command writes JSON evidence under:

```text
conferences/<slug>/storage/prune-verification/<profile>/<family>.json
```

For NeurIPS 2025, inspect:

```text
conferences/neurips2025/storage/prune-verification/backup/
```

Each family intended for remote-verified cleanup needs `status: "passed"` in
its evidence file or it must stay local.

## Troubleshooting

### Wrong Target Paths

If `Target root status:` is not `reachable`, the mounted backup parent is wrong
or disconnected. Stop and correct the mount or pass the intended
`--remote-path`.

If the path points at the public release folder, stop. The backup profile may
include private material that must not be written into the public target.

If `Leaf status:` is `missing` during dry run, the mounted parent root is
reachable but the conference leaf has not been created yet. Review the target
path before write mode.

### Suspicious Dry-Run Deltas

If the backup dry run omits private families that should be retained, check the
manifest `sync_profiles.backup` values and confirm the local family paths still
exist.

If the dry run wants to delete or overwrite a large remote subtree, run another
plain `--dry-run` without delete mode and compare the family list against the
Phase 3 backup evidence.

Legacy or otherwise out-of-profile top-level directories already present in the
mounted leaf remain untouched. `--delete --reviewed-dry-run` only prunes within
the included family destinations.

### Failed Verification

If `verify-restore` fails, do not quarantine the family. Refresh the backup
sync, rerun verification, and inspect the generated JSON for file-count, size,
and sample hash mismatches.

### Missing Evidence

If no evidence file exists under `storage/prune-verification/backup/`, cleanup
is not approved for remote-verified families. Generate evidence first with
`verify-restore`.

## Restore Access After Cleanup

After cleanup removes local payload copies, use the Drive symlink restore tool
to re-create read-only symlink access without rehydrating large payloads:

See [Drive Symlink Restore](drive-symlink-restore.md).
