# Public release scrub policy

What must never reach a public byte (a release bundle, a public site, a public code
repository), how to find it, and where it is removed. This policy was written in
v6.0-Public-Release Phase 2. The bundle builder applies it to every release.

## What is private

| Kind | Examples | Found by |
|---|---|---|
| Home-directory paths | `/Users/<name>/…`, `/home/<name>/…`, `file:///Users/…` | `audit_private_markers.py`, rule `home_path` |
| Cloud-storage mounts and Drive accounts | `Library/CloudStorage/…`, `GoogleDrive-<address>/…` | rules `cloud_storage`, `drive_account` |
| Email addresses | any address outside the reserved example domains (`example.com`, `*.example`, `*.invalid`, `*.test`) | rule `email` |
| Secrets read from a shell profile | code that opens `~/.zshrc`, `~/.bashrc`, `~/.profile` | rule `shell_profile_secret` |
| The operator's own identifiers | usernames, personal and work addresses | rule `extra`, from `CONFERENCE_PIPELINE_PRIVATE_MARKERS` |
| Concrete Drive folder names in data | a publication target naming the operator's Drive folder | the bundle builder's field rules (below) |
| Credentials | `.env` values, API keys, tokens, cookie files | `gitleaks` (`publishing/export_gate.py`) |
| The env file's name | `.env`, `prod.env`, `.env.local`, `".env"`; not a word such as `changes.environmental` | the publishing token `.env`: one file-name rule (no letter or digit after it) for text, UTF-16 text (in either byte order and alignment, inside any bytes), PDFs (their text, comments and decoded streams) and zip members, member names and comments alike; in binary or compressed bytes, only after a path separator. A PDF or zip gets one verdict in every gate, from the container check, which applies every rule to all of it |

A generic naming convention is not private on its own: for example,
`My Drive/<Conference> <Year>-raw` names no account. A concrete folder name inside
data that ships, such as a publication target, is removed with the rest of that field.

## How to scan

```bash
# The platform tree (tracked files), with the operator's own identifiers:
set -a; source .env; set +a
PYTHONPATH=src .venv/bin/python scripts/audit_private_markers.py

# A staged bundle or exported repository (every file under the directory):
PYTHONPATH=src .venv/bin/python scripts/audit_private_markers.py --tree <staged dir> --json <report.json>
```

- Set `CONFERENCE_PIPELINE_PRIVATE_MARKERS` in the local `.env` as a comma-separated list
  of the operator's usernames and addresses. It is never committed, and the scanner never
  prints the values: a finding shows only `<redacted:extra>`.
- Exit codes: 0 means clean, 1 means findings, 2 means a usage error.
- Generic fixtures in the platform tree are exempted in `.private-markers-allowlist.toml`,
  and each exemption carries a reason. An exemption covers only the match its `match`
  text names, never another marker on the same line.
- **Which allowlist applies to a staged tree** depends on what the tree holds:
  - A **data bundle** (a conference release) is scanned with no allowlist (the default
    for `--tree`). Every finding is removed from the bundle.
  - A **code export** carries the repository's own tests and their generic fixtures. It
    is scanned with the repository's allowlist:
    `--tree <export> --allowlist <export>/.private-markers-allowlist.toml`.

## Where it is removed

- **In the bundle copy, never in the workspace original.** Receipts are
  content-addressed, and their lineage must stay intact. A scrubbed copy of a receipt
  lives in the bundle; the workspace receipt is not edited.
- **Field rules for receipts copied into a bundle:**
  - an absolute path becomes a path relative to the workspace root;
  - a `publication.target` that names a person, an address or a Drive folder becomes
    `<redacted:private_target>`;
  - a field whose only content is private (for example a `workspace` absolute path) is
    replaced by its relative form, or dropped if it has none.
- **Known case (CVPR 2026).** In
  `storage/workflow-runs/cvpr2026-live-production/stage-11-release.json` and
  `final_proof_packet.json`:
  - `publication.target` holds an address and a Drive folder; it is redacted;
  - every `receipt_path`, `release_inventory_path`, `workspace` and evidence path is
    absolute and becomes relative.
- **Before anything is published,** the scanner runs over the finished staged tree with
  `--tree` and must exit 0, and the bundle's gitleaks gate must pass.

## Code repository

The public code repository is exported from the platform tree (`src/`, `scripts/`,
`tests/`, `docs/`, `.claude/`, `conferences/TEMPLATE/` and the root files) with a fresh
history. Before export, the platform-tree scan above must exit 0. Afterwards, the same
scanner runs over the export with the export's own allowlist (see above), and gitleaks
runs over the export's history.
