# Drive Symlink Restore Runbook

Restore read-only symlink access to Drive-backed generated conference artifacts
after local cleanup, without rehydrating large payload copies.

## Scope

This runbook covers manifest-bounded symlink restore of `public` and `backup`
artifact families. It is the inverse of `sync_to_gdrive.py`: that tool pushes
local payloads up to a mounted Drive leaf; this tool reads cleaned-up local
paths and plans symlinks pointing down to that mounted leaf so generated
artifacts become browsable again.

**What this is NOT:**

- Not a payload copy. Symlinks point at the mounted Drive leaf; no bytes are
  copied locally.
- Not a Drive backup. For the backup flow, see
  [Private Backup](private-backup.md).
- Not a replacement for `sync_to_gdrive.py`. The backup/offload contract
  is separate from this restore-symlink contract.
- Not a full local restore. If you need real local bytes (for editing or
  offline access), use `rsync` instead (see decision table below).

**Does not restore git-class or cache-class families.** Only `storage_class`
values `public` and `backup` are eligible. See
[Why Some Drive Files Are Not Restored](#why-some-drive-files-are-not-restored)
below.

## Symlink Access vs rsync Copy vs Leave on Drive

Choose the access mode before restoring:

| Mode | Command | When to use |
|------|---------|-------------|
| Symlink access | `python3 scripts/restore_drive_symlinks.py <slug> --apply` | Browse generated artifacts in-place; mount required; no local copy made |
| Local copy (undo symlinks) | `.venv/bin/python scripts/materialize_drive_symlinks.py --workspace conferences/<slug> --apply` | Copy each Drive-linked family back to local disk, verify every file, swap the link for the directory; Drive keeps its copy; `--revert` puts the links back |
| rsync copy | `rsync -a "/Volumes/..." conferences/<slug>/...` | Need editable local copy or offline access; copies real bytes |
| Leave on Drive | (default after cleanup — no command needed) | Artifact untouched; access via Drive UI or re-run the backup flow |

Use **symlink access** when you want the artifact paths to be browsable in the
repo workspace without consuming local disk space. The mounted Drive leaf must
remain available while the symlinks are in use; unmounting Drive will make them
dangling until Drive remounts.

Use **rsync copy** when you need to edit a file, use it offline, or hand it to
a tool that cannot follow symlinks to a Drive mount.

Use **leave on Drive** when you do not need local access: the artifact is safely
retained in Drive and can be retrieved on demand.

## Why Some Drive Files Are Not Restored

The tool selects families strictly by their `storage_class` field in
`conferences/<slug>/storage/storage_manifest.json`. Only two classes are
eligible for symlink restore:

| `storage_class` | Eligible | Reason |
|-----------------|----------|--------|
| `public` | Yes | Manifest-governed, Drive-backed deliverable payloads |
| `backup` | Yes | Manifest-governed, Drive-backed retained payloads |
| `git` | No | Control files are git-tracked; symlinks must never shadow them |
| `cache` | No | Rebuildable outputs; creating symlinks would hide their rebuild status |

**Cache-class exclusion in practice:** rebuildable output families such as
`video_build*/`, `orchestrator_logs/`, and `assets/generated/` carry
`storage_class: "cache"` in the manifest. Even when matching files exist on the
Drive leaf, the tool intentionally skips them. This ensures that a clean
rebuild always writes into the local path rather than finding a dangling or
stale symlink pointing to a Drive copy from an earlier run.

To inspect which families are eligible and which are skipped, read the manifest:

```bash
cat conferences/<slug>/storage/storage_manifest.json | python3 -c "
import json, sys
m = json.load(sys.stdin)
for f in m['artifact_families']:
    cls = f.get('storage_class', '(none)')
    eligible = cls in ('public', 'backup')
    print(f\"{f['id']:40s} storage_class={cls:10s} {'ELIGIBLE' if eligible else 'skipped'}\")"
```

The manifest is the authoritative source for each family's `storage_class`.
Edit the manifest to change eligibility — do not patch the tool.

## Preflight

Before running the restore tool:

1. **Confirm Drive is mounted.** The tool stats the mounted leaf and fails
   closed if it is absent.

2. **Read `conferences/<slug>/DRIVE.md`** for the Drive leaf name and the
   expected mounted path. The tool parses this file automatically, but reading
   it first confirms the mount point and Drive folder.

3. **Read `conferences/<slug>/storage/storage_manifest.json`** to understand
   which families are eligible for symlink restore.

4. **Check for existing local content.** If a family path exists locally as a
   real (non-symlink) file or non-empty directory, the tool will record it as a
   `unsafe_existing_local` blocker and skip it. Existing symlinks that already
   point at the correct Drive target are treated as no-ops (`already_linked`).

## Dry Run

Run the restore planner without creating any symlinks:

```bash
python3 scripts/restore_drive_symlinks.py <slug> --dry-run
```

Or with an explicit conference directory path:

```bash
python3 scripts/restore_drive_symlinks.py --conference-dir conferences/<slug> --dry-run
```

To override the mounted Drive path (useful when the path in DRIVE.md is stale):

```bash
python3 scripts/restore_drive_symlinks.py <slug> --dry-run --drive-path "/Volumes/My Drive/<Conference>-raw"
```

For a machine-readable per-family plan:

```bash
python3 scripts/restore_drive_symlinks.py <slug> --dry-run --json
```

The dry-run output must show:

- **Eligible families** — the `public`/`backup` families that will receive symlinks
- **Skipped families** — the `git`/`cache` families and their reason
- **Planned link count** — per family (path families: one directory-level link;
  glob families: one file-level link per Drive leaf)
- **Mode: dry-run (no filesystem changes)** — confirms the run is read-only

Dry run exits 0 even when blockers are present. No symlinks are created.

## Apply Mode

After reviewing the dry-run plan, create symlinks:

```bash
python3 scripts/restore_drive_symlinks.py <slug> --apply --receipt-path <path>
```

`--receipt-path` is optional; omit it to skip writing the JSON receipt.

**Apply behavior:**

- **Creates symlinks** for every eligible entry where the local path is absent
  and the Drive target exists (`link` action).
- **No-ops** for entries where the local path is already a correct symlink
  pointing at the Drive target (`already_linked` action).
- **Blocks and reports** when a real file or non-empty directory occupies the
  local path, or when the Drive target is missing (`unsafe_existing_local` /
  `missing_remote` action). Blockers do not fail the run (exit 0); they are
  reported and recorded in the receipt.
- **Fails hard** (exit 1) only on an OS-level symlink creation error.

For glob-family entries (file-level leaves), the tool creates any missing
intermediate directories under the conference workspace before creating each
symlink. Intermediate directories are always real directories, never symlinks.

After creating symlinks, apply mode automatically runs post-apply verification.
If verification detects unresolved symlinks or git-control violations, the exit
code becomes 1.

## Verify Mode

Re-check existing symlinks without creating or removing anything:

```bash
python3 scripts/restore_drive_symlinks.py <slug> --verify --receipt-path <path>
```

Verify mode re-resolves every eligible symlink that exists in the workspace and
confirms that no git-tracked or git-class-manifest file was replaced. It reads
the same manifest and DRIVE.md as the other modes.

Exit 0 = verification passed (all symlinks resolve, zero git-control violations).
Exit 1 = one or more unresolved targets or git-control violations.

Run `--verify` after remounting Drive to confirm that previously-created
symlinks are still valid.

## Evidence To Confirm

After apply or verify, check the following:

| Evidence | Required result |
|----------|-----------------|
| Exit code | 0 for success; 1 for hard fault or verification failure |
| `verification.status` in receipt | `"passed"` |
| `blockers` in receipt | Lists `unsafe_existing_local` / `missing_remote` entries; exit 0 is still correct when blockers are present |
| `verification.unresolved_targets` | `[]` (empty) |
| `verification.git_control_violations` | `[]` (empty) |

The receipt schema is described in [Receipt Schema](#receipt-schema) below.

## Drive Mount Failure Recovery

### Mount not present — `ERROR_DRIVE_LEAF_MISSING` or `ERROR_NO_DRIVE_PATH`

`ERROR_DRIVE_LEAF_MISSING` is raised when the tool resolves the Drive leaf path
(from DRIVE.md or `--drive-path`) but finds it does not exist or is not a
directory. `ERROR_NO_DRIVE_PATH` is raised when no mounted path can be
determined at all (DRIVE.md has no `Mounted path, when available:` line and
`--drive-path` is not set).

**Recovery steps:**

1. Check that Google Drive for Desktop is running and signed in.
2. Confirm the conference Drive folder is set to "Available offline" or
   "Stream files" mode in Drive preferences.
3. After remounting, re-run `--verify` to confirm that previously-created
   symlinks still resolve:
   ```bash
   python3 scripts/restore_drive_symlinks.py <slug> --verify
   ```

### Manifest not found — `ERROR_NO_MANIFEST`

`ERROR_NO_MANIFEST` is raised when no `storage/storage_manifest.json` is found
under the conference workspace (and no `storage.manifest_path` config override
points at an alternative file).

**Recovery steps:**

1. Confirm you are running from the repository root.
2. Confirm `conferences/<slug>/storage/storage_manifest.json` exists.
3. If the file is missing, check `git status` and restore it from git history or
   from the Drive backup before proceeding.

### Existing symlinks after Drive unmount

Symlinks created by `--apply` become **dangling** when the Drive mount
disappears. They are not broken — they will resolve again when Drive remounts.
Do not delete dangling symlinks; simply remount Drive and run:

```bash
python3 scripts/restore_drive_symlinks.py <slug> --verify
```

Verify exit 0 after remount confirms all links are intact.

## Receipt Schema

When `--receipt-path` is provided, the tool writes a machine-readable JSON
receipt with the following top-level fields:

| Field | Value |
|-------|-------|
| `receipt_type` | `"conference_pipeline.gdrive_symlink_restore"` |
| `schema_version` | `1` |
| `mode` | `"dry-run"`, `"apply"`, or `"verify"` |
| `status` | `"succeeded"` or `"failed"` |
| `applied` | `true` for apply mode only |
| `exit_code` | `0` or `1` |
| `eligible_families` | List of family ids that were eligible |
| `skipped_families` | List of family ids that were skipped |
| `families` | Per-family `action_counts` (integers only — no per-leaf paths) |
| `totals` | Aggregate action counts across all families |
| `blockers` | List of `unsafe_existing_local` and `missing_remote` entries |
| `verification` | Optional embedded verification block (see below) |
| `generated_at` | ISO 8601 timestamp |

The `verification` block (when present) contains:

| Field | Value |
|-------|-------|
| `status` | `"passed"` or `"failed"` |
| `links_checked` | Integer count of links inspected |
| `unresolved_targets` | List of symlink paths whose targets were unreachable |
| `git_control_violations` | List of symlink paths that land in a git-class family |

`--receipt-path` is optional. Omit it to run without writing a receipt file.

## Troubleshooting

### `Conference workspace does not exist`

The slug did not resolve to a conference directory under `conferences/`. Check
the slug spelling or use `--conference-dir conferences/<slug>` to supply the
path directly.

### Symlinks appear in `git status` as untracked or modified

The manifest-generated `.gitignore` for the workspace should cover `public` and
`backup` payload paths. If symlinks appear in `git status`, confirm that the
workspace `.gitignore` is current:

```bash
python3 scripts/generate_workspace_gitignore.py --conference-dir conferences/<slug>
```

### `unsafe_existing_local` blockers for families that should be empty

A real directory may exist under the conference workspace if a previous workflow
stage partially wrote output before cleanup. Inspect the path, then either
remove the real directory or use the `rsync` copy mode to update it.

For more context on cleanup and recovery, see
[Cleanup And Recovery](cleanup-and-recovery.md).

## Undoing symlinks: materialize to local disk

Pipeline stages that hash inputs (`workflow bind-lineage`, incremental index
migration) and sandboxed agent sessions cannot read a Google Drive mount, so a
workspace that will be re-run needs its payloads local. The inverse of the
restore tool is `scripts/materialize_drive_symlinks.py`:

```bash
.venv/bin/python scripts/materialize_drive_symlinks.py --workspace conferences/<slug>
.venv/bin/python scripts/materialize_drive_symlinks.py --workspace conferences/<slug> --apply
```

The dry run lists every symlink that leaves the repository, with file counts
and bytes, against free disk plus a 20 GiB margin. `--apply` copies each target
into `<name>.materializing` (`rsync -a`, resumable), verifies relative paths and
sizes (`--hash` adds SHA-256), then swaps the link for the directory and records
the original target in `storage/drive-materialization-receipt.json`. A copy that
fails verification, an unreadable target, or a target that lists no files
leaves the link untouched. `--revert` recreates the links from the receipt and
renames the local copy to `<name>.local-<utc>`; nothing is ever deleted. Run it
on the host, where the Drive mount is readable, then regenerate the workspace
gitignore if the manifest changes.

