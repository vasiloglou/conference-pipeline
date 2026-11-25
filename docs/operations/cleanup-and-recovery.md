# Cleanup And Recovery Runbook

Use this runbook to inspect local cleanup candidates, verify remote recovery,
move eligible local paths into quarantine, and purge expired quarantine batches.
The cleanup helper is recoverability-first: live family paths are not deleted
directly.

Cleanup path resolution follows conference-local config pointers:
`storage.manifest_path` selects the manifest, and
`storage.support_material.storage` is the root for `inventory.json` and
`prune-verification/` evidence.

## Scope

Cleanup is appropriate only after the storage manifest and sync evidence show
where a family belongs. Cache-tier families are rebuildable. Remote-verified
families require passing restore evidence before quarantine.

For NeurIPS 2025, local cleanup evidence and restore proof live under:

```text
conferences/neurips2025/storage/prune-verification/
```

The archived Phase 3 proof summary now lives in:

- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`

## Report Candidates

Inspect manifest-driven prune candidates:

```bash
python3 scripts/prune_conference_workspace.py report --conference-dir conferences/neurips2025
```

Use `--tier cache` or `--tier remote-verified` to narrow the list:

```bash
python3 scripts/prune_conference_workspace.py report --conference-dir conferences/neurips2025 --tier cache
```

The report is evidence for what the tool would consider eligible. It does not
move or delete anything.

## Verify Restore

Before quarantining a remote-verified family, generate restore evidence:

```bash
python3 scripts/prune_conference_workspace.py verify-restore --conference-dir conferences/neurips2025 --profile backup --tier remote-verified --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

To verify a specific family or representative proof slice:

```bash
python3 scripts/prune_conference_workspace.py verify-restore --conference-dir conferences/neurips2025 --profile backup --tier remote-verified --family export_for_deep_dive --remote-path "/mounted/Drive/NeurIPS 2025-raw"
```

verify-restore fails closed for cache-tier or profile-ineligible families.
Use `--tier remote-verified` or explicit `--family` values to narrow restore
verification to eligible families.

Evidence is written to:

```text
conferences/<slug>/storage/prune-verification/<profile>/<family>.json
```

With the starter default config this stays under `storage/`, but future
conference workspaces may move inventory and prune evidence together by changing
`storage.support_material.storage`.

The evidence file must contain `status: "passed"` before that family is
eligible for remote-verified quarantine.

## Execute Quarantine

Move eligible families into repo-local quarantine:

```bash
python3 scripts/prune_conference_workspace.py execute-quarantine --conference-dir conferences/neurips2025 --quarantine-dir conferences/neurips2025/.quarantine --tier cache
```

For remote-verified families, run this only after `verify-restore` evidence
passes:

```bash
python3 scripts/prune_conference_workspace.py execute-quarantine --conference-dir conferences/neurips2025 --quarantine-dir conferences/neurips2025/.quarantine --tier remote-verified
```

Each quarantine batch writes a machine-readable receipt:

```text
conferences/<slug>/.quarantine/<batch-id>/receipt.json
```

The `receipt.json` contract records the batch id, source paths, quarantine
paths, cooldown timing, and verified profiles. Keep the receipt until the
batch is purged.

## Purge Expired Quarantine

Purge only acts on quarantine batches that satisfy the minimum age and cooldown
checks. It does not touch live workspace paths.

```bash
python3 scripts/prune_conference_workspace.py purge --conference-dir conferences/neurips2025 --quarantine-dir conferences/neurips2025/.quarantine --min-age-hours 168
```

For a specific quarantine batch:

```bash
python3 scripts/prune_conference_workspace.py purge --conference-dir conferences/neurips2025 --quarantine-dir conferences/neurips2025/.quarantine --batch-id 20260419T201405Z --min-age-hours 168
```

Purge records are written under:

```text
conferences/<slug>/storage/prune-verification/purges/
```

## Recovery

If a quarantined family needs to be restored during the cooldown window:

1. Open the batch `receipt.json`.
2. Find the `source_path` and `quarantine_path` for the family.
3. Move the quarantined path back to its recorded source location.
4. Rerun the relevant workflow or verification command.
5. Keep the receipt with a note in the operator log until the batch is no
   longer needed.

If the quarantine batch has already been purged, restore from the public or
backup target using the family path recorded in
`storage/prune-verification/<profile>/<family>.json`.

For families backed up to Drive before cleanup, use the symlink restore tool to
re-create browsable access without a full local rehydration:

See [Drive Symlink Restore](drive-symlink-restore.md).

## Hard Gates

- Do not run `execute-quarantine` for remote-verified families without passing
  `verify-restore` evidence.
- Do not treat `report` output as restore proof.
- Do not manually delete live family paths instead of using quarantine.
- Do not run `purge` until the quarantine `receipt.json` exists and the
  cooldown window has expired.

## Troubleshooting

### Wrong Target Paths

If `verify-restore` reports a missing remote family path, confirm the
`--remote-path` is the mounted conference leaf, not the Drive parent or a
different conference folder.

If generated evidence points at a public path for a backup-only family, stop and
rerun verification against the backup profile and backup destination.

### Suspicious Candidate Lists

If `report` lists a family that should not be cleanup-eligible, inspect
`storage.manifest_path` for `storage_class`, `rebuildable`, and `sync_profiles`.
Do not edit the quarantine output to compensate for a manifest mistake.

If `report` omits a known cache or backup family, confirm the path exists
locally and the manifest still covers it.

### Failed Verification

If an evidence file has `status: "failed"`, compare the recorded local and
remote file counts, byte sizes, and sample hashes. Refresh the appropriate sync
target, then rerun `verify-restore`.

### Stale Or Superseded Stage Receipts

Never delete stage receipts to force a re-run. Since v5.1 every receipt
records what it consumed (`lineage`) and produced (`output_digest`), and
`--resume` refuses to reuse a receipt whose inputs moved. Read the state
first:

```bash
.venv/bin/python -m conference_pipeline.cli workflow stage-status --workspace conferences/<slug> --run-id <run-id>
```

Each stage prints FRESH, STALE (with the reason -- which upstream changed and
through which chain), MISSING or BLOCKED, followed by the next stage to run.
Re-run from that stage with `--resume`; every stage below it re-runs because
its recorded upstream digest no longer matches. The runners run one stage per
invocation and never schedule an upstream for you: a paid stage started
while an upstream is still stale blocks before any spend with
`upstream_lineage_not_fresh`, naming the stale chain, so run the stages in
`stage-status` order. Below the stages it lists the sub-steps (the LLM full
read and the 07A-07H analysis steps, each run by its own command); a stage
consuming a stale sub-step stays STALE until that sub-step is re-run, starting
from `next sub-step to run`. A run written before v5.1
has no lineage and reads STALE everywhere until it is stamped once:

```bash
.venv/bin/python -m conference_pipeline.cli workflow bind-lineage --workspace conferences/<slug> --run-id <run-id> --backfill          # dry run
.venv/bin/python -m conference_pipeline.cli workflow bind-lineage --workspace conferences/<slug> --run-id <run-id> --backfill --apply
```

The backfill is evidence-based: a receipt binds an upstream only when a
digest it recorded in its own `checks` identifies the upstream's current
version (the upstream's receipt sha or one of its registered output shas --
not a manifest digest both stages merely read), and nothing it recorded
contradicts that version; otherwise the entry stays `unresolved` and the
stage reads STALE. `--assume-current <stage,...> --operator <name>` records
an operator attestation instead. Prior receipts are preserved as
`<stem>.prebackfill-<stamp>.json`.

### Missing Or Bad Receipts

If `execute-quarantine` does not write `receipt.json`, treat the batch as
invalid and inspect the command output before moving anything else. A purge
without a valid receipt breaks recovery traceability.

### Purge Does Nothing

If `purge` returns an empty list, the batch may be younger than
`--min-age-hours`, still inside its cooldown window, or filtered by a different
`--batch-id`. Recheck the receipt timestamps before changing the minimum age.
