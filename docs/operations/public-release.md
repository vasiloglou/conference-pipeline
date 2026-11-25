# Public Release Runbook

Use this runbook to publish curated, shareable conference artifacts from a
manifest-backed workspace to a public release target. NeurIPS 2025 is the
worked example; future conferences should use the same profile and evidence
checks with their own `conferences/<slug>` path and Drive destination.

## Scope

Public release is for artifacts that have already been deliberately curated for
sharing. For the NeurIPS 2025 pilot, the public profile includes the `podcasts`
and `presentations` families, which map to `audio/` and `decks/`.

Private analysis, raw exports, sidecar agent work, caches, and legacy material
must stay out of the public release unless a later manifest change intentionally
promotes a curated subset.

## Preflight

Confirm the workspace manifest and policy first:

- `conferences/<slug>/storage/storage_manifest.json`
- `conferences/<slug>/storage/storage_policy.md`

For NeurIPS 2025:

- Storage policy (`conferences/neurips2025/storage/storage_policy.md`)
- Public restore evidence (`conferences/neurips2025/storage/prune-verification/public`)

## The release command

`conference-pipeline release` is the operator path for any conference workspace. It
runs the orchestrator and the stage scripts described below; it adds no stage of its
own. Every command exits 0 when done, 1 when `check` finds a problem (or an isolated
rebuild's digest differs), and 2 when it refuses or a gate blocks it.

```bash
.venv/bin/python -m conference_pipeline.cli release init <slug>
# edit conferences/<slug>/storage/public_export_manifest.json: promote families
.venv/bin/python -m conference_pipeline.cli release plan <slug>
.venv/bin/python -m conference_pipeline.cli release build <slug>
.venv/bin/python -m conference_pipeline.cli release check <slug>
.venv/bin/python -m conference_pipeline.cli release publish <slug> --approved-by <owner> --surface github
```

1. **`init`** drafts `storage/public_export_manifest.json` from the storage manifest.
   Nothing is published by default: `families` is empty and every storage family is
   listed in `excluded_families` with a reason. `init` never overwrites a manifest;
   `--force-draft-path` drafts beside it.
2. **Promote** a family by moving its row from `excluded_families` into `families` and
   filling in `id`, `private_path`, `public_path`, `storage_manifest_id`,
   `redistribution_status`, `license` and `provenance`
   ([schema](../public_export_manifest.schema.json)). `plan` refuses a manifest with no
   promoted family (`no_family_promoted`).
3. **`plan`** validates the manifest and dry-runs the export. It writes the validation
   and dry-run receipts under `storage/workflow-runs/` and no bundle file.
4. **`build`** writes the bundle from the reviewed dry run, then builds the index, the
   site and the dataset package, and stages the data repository last, all without
   publishing. `--bundle-only` stops after the bundle; `--mock` builds the index with
   the offline `deterministic-hash` provider; `--embedding-cache DIR` builds it from
   vectors made on another machine (below). A bundle id with live publication
   receipts is never overwritten. The public URLs the workspace's `release:` block
   names are written into `RELEASE.json` here, so publishing rewrites nothing.
5. **`check`** re-verifies the built bundle: the export's tree digest against its write
   receipt, every `CHECKSUMS.json` row, a private-marker scan with no allowlist
   ([scrub policy](public-release-scrub-policy.md)), the secrets gate, and each
   surface's provider limits (a `surface_limit` finding: a file over GitHub's 100 MiB,
   Pages' 25 MiB or 20,000 files, or an HF folder over 10,000 entries; the table and its
   sources are in `publishing/surface_limits.py`).
6. **`publish`** needs the owner's approval for that release, named with
   `--approved-by`; without it, it refuses (`owner_approval_missing`) before reading
   `.env`. `--surface` picks `github`, `r2`, `pages`, `hf` or `zenodo`. It refuses
   before any provider call when a chosen surface's tree breaks a provider limit, then
   pushes the chosen surfaces and checks URL liveness and metadata agreement.
   `--with-preflight` adds the deployment preflight, which needs the R2 upload receipts.

**What each surface receives** (v6.0 Phase 7.1):

| Surface | Receives |
|---|---|
| `github` | the data repository: the bundle without `site/` and `index/`. An existing repository keeps its history: the version it replaces is tagged with its bundle id, the new one is committed on top and tagged too; nothing is force-pushed |
| `hf` | the bundle without `site/`, `public-repo/`, `index/bm25_index.pkl` and `index/embeddings.npy` (the Parquet copy replaces it), plus a generated dataset card (`README.md`) and `LICENSE`; files a new version dropped are deleted; the push is tagged with the bundle id after tagging the replaced version |
| `pages` | the reading site, whose search is a sharded index (`search/`) and int8 vector shards (`assets/vectors/`) |
| `r2` | the media the bundle's `*.cdn.json` sidecars name, one PUT per object |
| `zenodo` | unchanged; v6.0 makes no new deposits (plan D13) |

**Vectors made on another machine.** The real embedding lanes take hours on one
machine. `scripts/public_embedding_cache.py export --bundle-root <bundle> --out
chunks.jsonl` writes the bundle's chunk texts; `scripts/public_embedding_cache.py embed
--chunks chunks.jsonl --out DIR` runs the two governed models there; `release build
<slug> --embedding-cache DIR` then builds from DIR. A chunk DIR does not hold, or a lane
made by another model, refuses the build.

Each command writes its own top-level receipt,
`storage/workflow-runs/public-release-<bundle>-<command>-receipt.json`. Credentials come
from the checkout's `.env` (`--env-file` to override), read only for live commands, and
receipts record variable names, never values.

**Targets** come from an optional `release:` block in the workspace `config.yaml`
(`bundle_id`, `github_repo`, `pages_project`, `hf_repo_id`, `cdn_base_url`,
`r2_bucket`, `r2_prefix`, `public_site_url`, `previous_version_doi`, `zenodo_sandbox`,
`r2_base_url`); flags override. The data repository's name defaults to
`<conference>-data`. A target that cannot form its URL refuses every command.

**Rebuilding a published bundle for comparison.** `--isolated DIR` plans and builds into
a directory outside the workspace and writes nothing under `conferences/<slug>/`:

```bash
.venv/bin/python -m conference_pipeline.cli release plan <slug> --bundle-id <bundle> --isolated <DIR>
.venv/bin/python -m conference_pipeline.cli release build <slug> --bundle-id <bundle> --isolated <DIR> --expect-tree-sha256 <digest>
```

The reports `DIR/plan-report.json` and `DIR/build-report.json` hold bundle-relative paths
and digests only. With `--expect-tree-sha256`, the build exits 1 on a mismatch and lists
the paths that differ from the bundle's recorded write receipt.

## The reader site and the hub

`release build` writes the site into `<bundle>/site/` with
`conference_pipeline.publishing.static_site`. The site is built from the bundle alone:
- a topics page;
- one page per topic: slides, videos, podcast, narrative, and the papers it cites;
- a papers page;
- a data-files page, the only page that links control files;
- search.

**Copying.** Every linked bundle file is copied under `files/` after its sha256 is checked
against `CHECKSUMS.json`. `index/` files are checked against the index manifest's
`output_checksums` instead. A file whose bytes differ stops the build. Videos stay on the
CDN their sidecars name.

**Search modes.** Semantic search is offered only when the bundle's index has a browser
vector blob. The page shows only the modes that work.

**Reader content paths.** These are the paths the export must fill. A bundle without one
shows "not in this release" for it.

| Path | Content |
|---|---|
| `decks/control/presentation_manifest.json` | topic ids, titles, order |
| `decks/payloads/<topic>/presentation.{html,pdf}`, `*.mp4.cdn.json` | slides, video sidecars |
| `audio/<topic>/…/*.mp3` or `*.mp3.cdn.json` | podcast |
| `narratives/<topic>.md` | the narrative |
| `catalog/papers.jsonl` | one [catalog row](../public_catalog_row.schema.json) per public paper |
| `catalog/topic_papers.json`, `catalog/excluded_ids.json`, `catalog/statistics.json` | the papers each topic cites, the export's exclusions, the corpus statistics and version label |

`RELEASE.json` names the conference for readers through `release.conference_name`, which
comes from the export manifest's `release` block.

**Building the reader families** (v6.0 Phase 6). The exporter copies only files that exist
under a family's `private_path`, so the catalog, statistics, narratives and limitations are
written first, into `storage/release-staging/<bundle_id>/`. A manifest family then exports
that folder to the bundle root.

```bash
.venv/bin/python -m conference_pipeline.cli release families <slug> [--bundle-id <id>]
```

The workspace declares its sources in `storage/release/reader_sources.json`:

| Key | Meaning |
|---|---|
| `catalog` | `{"adapter": "corpus_inventory" \| "openreview_accepted_papers" \| "none", "path": ...}`. A row's `official_url` comes from its id (`openreview:` → the forum, `arxiv:` → the abstract page), else the record's own https link, else null. A named exclusion appears only in `catalog/excluded_ids.json` |
| `statistics` | `{"path": "storage/corpus/corpus_statistics.json"}`, or `{"inventory": ...}` to compute it from a corpus inventory (label `<slug>-inventory-<sha12>`). Talks carry display names; each kind must balance (listed = admitted + excluded) |
| `topics` | `{"validation": <folder>}`: each file's `valid_reference_ids` map to catalog rows by bare id |
| `narratives` | `{"dir": "analysis", "pattern": "[0-9][0-9]_*.md"}` |
| `limitations` | `{"path": "storage/release/limitations_source.json"}`: items citing the close record's lines. Every quote must sit at its lines, or the build refuses (exit 2) |
| `d11_note` | optional text shown with the statistics |
| `presentations` | `{"decks": "decks", "html": "*/presentation.html", "executive": "executive_*"}`: a release presentation manifest listing every deck folder with published HTML, titled from its `presentation_frame.json`; publish it at `decks/control/` in place of a stage snapshot |
| `transcripts` | `{"inventory": "storage/corpus/corpus_inventory.json"}`: each admitted `speech_transcript` as `transcripts/<item_id>.md` (title, kind, talk page, a machine-transcript notice, the text) and `transcripts/index.jsonl`. Publish them as a `caution` family; the build then needs `--ack-caution-family transcripts` |
| `regenerated_pdfs` | `{"family": "presentation_payloads", "html": "*/presentation.html"}`: render each deck's *published* HTML (after that family's redaction rules) to `pdfs/<deck>/presentation.pdf` with Playwright, once. Publish them with a family whose `private_path` is `storage/release-staging/<bundle>/pdfs`, and drop the original PDFs from the payload family. A PDF that still holds a marker is excluded and reported; HTML that still loads a local image is refused |

Check the limitations record row by row against the close record's accepted lines:
`conference_pipeline.publishing.reader_families.check_limitations` exits 0 when every
accepted range is covered, and 1 naming each uncovered range.

**Redaction in the bundle copy.** A family's `redaction_rules` objects rewrite each matching
file's bundle copy before checksumming; the workspace original never changes. The write
receipt lists `redacted_files` and `binary_markers`.

| Kind | Effect |
|---|---|
| `json_field` | the value at `pointer` becomes `replacement`, e.g. `<redacted:private_target>` |
| `absolute_path` | workspace paths become relative; other home paths become `<redacted:path>` |
| `relink` | a Drive link (HTML anchor, any Markdown link form, or a bare URL) is rewritten through `link_map`; an unmapped one is dropped, keeping a link's text |
| `marker_text` | publishing markers become `<redacted:RULE>` |
| `figure_links` | each local figure in deck Markdown becomes a line naming the paper it was cropped from (the deck's `asset_inventory.json`) and linking its official copy through `catalog` (the staged `catalog/papers.jsonl`); paper figures are linked, never copied. A figure the pipeline drew (`assets/generated/`) is named as one |
| `pdf_links` | in a PDF, each link annotation that is not a public https link (a `file://` path the renderer baked in) is removed; the pages are unchanged |
| `drop_preamble` | text above a Markdown file's first `# ` title is dropped (a model's chat line above a script) |
| `inline_assets` | each local image an HTML file loads becomes a `data:` URI, so the deck is one self-contained file. Refused for an image outside the project, reached through a link, missing, or holding a marker. Always runs last |

Apart from `pdf_links`, a binary (PDF, image, media) is never rewritten. If it holds a marker it is listed under
`binary_markers`, to be regenerated from its fixed source or excluded. For deck PDFs,
`regenerated_pdfs` (above) does the regeneration.

`release check` applies the literal publishing tokens (`drive.google.com`, `/Users/`,
`file://`, ...) as well as the marker rules, so it refuses what the site build refuses.

**Checking a built site or hub.** This runs the link check, the accessibility lint and a
375 px Playwright render. It exits 1 on any finding; a skipped render is reported as
skipped.

```bash
.venv/bin/python scripts/check_public_site.py <site dir>
```

**The hub.** It merges several bundles' catalogs under `<release id>/<paper id>`, with
cross-conference full-text search, and lists every release with its DOI, dataset, site
and repository. It reads only bundles. Excluded rows and catalog files outside
`CHECKSUMS.json` are never read.

```bash
.venv/bin/python scripts/build_public_hub.py --bundle-root <bundle> --bundle-root <bundle> --output-dir <hub dir> [--code-repo-url <url>]
```

## Reusable Public Release Orchestrator

`conference-pipeline release` runs this orchestrator. Its script,
`scripts/publish_conference_public_release.py`, remains the lower-level entry point
for single stages. It plans and runs the existing export, index,
site, repo, CDN, dataset, metadata, and liveness scripts without duplicating
their internals. By default, use validation and dry-run stages first.

Generic validation/default dry run:

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --validate-manifest \
  --dry-run \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-validation-receipt.json
```

ICLR 2026 validation/default dry run:

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug iclr2026 \
  --bundle-id iclr2026-v1.0.0 \
  --validate-manifest \
  --dry-run \
  --receipt-path conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-validation-receipt.json
```

Run validation/dry-run, review both child receipts, and then run write-only
publication as separate orchestrator invocations. Write mode is always bound to
a reviewed dry-run receipt, including the first write of a new bundle; the
exporter rechecks the manifest digest, source SHA, planned artifact paths,
delivery modes, sizes, source checksums, and CDN sidecars before materializing
files. Include `--ack-caution-family <family-id>` for every manifest family
whose `redistribution_status` is `caution`.

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --write-bundle \
  --reviewed-dry-run-receipt conferences/<slug>/storage/workflow-runs/public-export-<slug>-v1.0.0-dry-run-receipt.json \
  --export-write-receipt-path conferences/<slug>/storage/workflow-runs/public-export-<slug>-v1.0.0-write-receipt.json \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-write-receipt.json
```

The top-level `--receipt-path` and child `--export-write-receipt-path` are
independent evidence paths. Review the child write receipt directly; the
top-level receipt indexes it but does not replace its artifact, gate, or bundle
verification evidence.

### Receipt confinement and orchestration recovery

Every public-release receipt input and output must remain below the selected
conference's `storage/workflow-runs` tree. This boundary includes top-level and
child outputs, explicit child overrides, reviewed export/public-repo inputs,
and the adjacent orchestration journal. Existing symlinked receipt components
fail closed, and invalid receipt overrides never fall back to the unsafe target.
The shared `write_json_atomic_secure` primitive creates an exclusive no-follow
sibling temporary file, pins and revalidates its parent, and performs the
same-directory replacement; exporter, orchestrator, and journal persistence do
not maintain separate temporary-file schemes.

Blocked and failed receipts use `stable_receipt_exception_code` to retain only
fixed gate/stage labels and a validated exception class. Raw exception messages
never enter blocked receipts or journals. Token/password text, traceback text,
stdout/stderr, resolved symlink targets, and filesystem paths taken from an
exception are not receipt fields, while `secret_values_recorded` remains false.

`PUBLIC_RELEASE_JOURNAL_RECEIPT_TYPE` identifies the orchestration journal.
`public_release_journal_path` derives its deterministic non-aliased path next to
the requested top-level receipt, and `write_public_release_journal` delegates
every transition to `write_json_atomic_secure`. Before the first child runs, the
orchestrator writes an adjacent journal and updates it after every completed
child result. Its stable fields include `status`, `top_level_receipt_path`,
`planned_commands`, `command_results`, `last_completed_stage`, `failure_code`,
`exception_type`, `secret_values_recorded`, and `updated_at`. If the initial
journal cannot be created, no child runs.

After the last child result is durable, the top-level receipt indexes both
`journal_path` and all child receipt paths, then the journal becomes
`finalized`. If top-level receipt replacement fails, the journal remains the
durable recovery record, changes to `failed`, and records only
`top_level_receipt_persistence_failed` plus the sanitized exception class. The
caller receives a structured nonzero result instead of a raw persistence
exception. Child receipt transaction durability and top-level orchestration
journal durability are separate guarantees.

An optional pre-publication rebuild is another separate write-only orchestrator
invocation bound to the same reviewed dry-run receipt. Give that rebuild its own
child and top-level receipt paths so the first write remains independently
reviewable:

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --write-bundle \
  --reviewed-dry-run-receipt conferences/<slug>/storage/workflow-runs/public-export-<slug>-v1.0.0-dry-run-receipt.json \
  --export-write-receipt-path conferences/<slug>/storage/workflow-runs/public-export-<slug>-v1.0.0-rebuild-receipt.json \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-rebuild-receipt.json
```

Live provider stages are blocked unless the operator supplies the matching
`--live-*` flag, the matching `--ack-live-*` flag, provider target arguments,
and a reviewed dry-run/sandbox receipt. The orchestrator loads the configured
env file only for live stages and writes receipts that record environment
variable names/presence only, never secret values.

`--live-r2` needs the owner's named approval; the orchestrator passes it to
`scripts/upload_public_media.py` as `--approved-by`. Run directly, the script's
`--live` needs `--approved-by NAME` too, and it refuses any object over the R2
single-PUT limit before uploading. Use `scripts/upload_public_media.py --dry-run`
for planning; the live receipt records `uploaded_count` and `uploaded_objects`.
The direct GitHub and Pages publishers likewise refuse a tree over their
provider limits.

Generate prerequisite package receipts before deployment preflight. `--stage-repo`
and `--package-dataset` create local public-repo/dataset receipts; they do not
publish provider surfaces.

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --stage-repo \
  --package-dataset \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-packaging-receipt.json
```

After the public repo, site, dataset, R2 upload, and R2 liveness receipts exist,
run deployment preflight:

```bash
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --preflight-live \
  --github-repo <owner>/<repo> \
  --r2-base-url <https-r2-public-base-url> \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-preflight-receipt.json
```

Then source credentials only for the explicit live provider attempt. `--live-github`
currently fails closed with `operator_blocked` until provider push proof is
implemented or supplied by a later provider-specific receipt.

```bash
set -a; source .env; set +a
python3 scripts/publish_conference_public_release.py \
  --conference-slug <slug> \
  --bundle-id <slug>-v1.0.0 \
  --live-github \
  --ack-live-github \
  --github-repo <owner>/<repo> \
  --public-repo-dry-run-receipt conferences/<slug>/storage/workflow-runs/public-repo-<slug>-v1.0.0-dry-run-receipt.json \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-<slug>-v1.0.0-live-github-receipt.json
```

## Public Bundle Export

The public release bundle is produced by `scripts/export_for_public.py`, not by
making a private Drive leaf public. The script reads
`conferences/<slug>/storage/public_export_manifest.json`, runs the fail-closed
publishing gate, and emits a deterministic bundle under
`conferences/<slug>/exports/public-release/<bundle_id>/`.

For focused debugging, the lower-level exporter can reproduce an individual
child receipt. Production review and write boundaries use the orchestrator
sequence above. Start a lower-level dry run with:

```bash
python3 scripts/export_for_public.py \
  --conference-dir conferences/neurips2025 \
  --bundle-id neurips2025-v1.0.0 \
  --dry-run \
  --ack-caution-family transcripts \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-export-neurips2025-v1.0.0-dry-run-receipt.json
```

The dry-run report and receipt must show planned families, excluded artifacts,
`redacted_files` (empty for the NeurIPS 2025 keys-only policy), planned CDN
sidecars, `source_sha`, `export_manifest_sha256`, and license-gating outcome.
Dry run writes no bundle files.

After reviewing the dry-run receipt, lower-level write mode is:

```bash
python3 scripts/export_for_public.py \
  --conference-dir conferences/neurips2025 \
  --bundle-id neurips2025-v1.0.0 \
  --ack-caution-family transcripts \
  --reviewed-dry-run-receipt conferences/neurips2025/storage/workflow-runs/public-export-neurips2025-v1.0.0-dry-run-receipt.json
```

Write mode produces:

| Artifact | Purpose |
|----------|---------|
| `exports/public-release/<bundle_id>/RELEASE.json` | Release identity, source SHA, manifest digest, and included families |
| `exports/public-release/<bundle_id>/CHECKSUMS.json` | SHA-256 checksums for emitted inline files, metadata, and CDN sidecars |
| `storage/workflow-runs/public-export-<bundle_id>-write-receipt.json` | Planned vs actual artifacts, gate receipts, exit code, and license outcome |

Inline source symlinks are dereferenced into portable regular files. Post-write
verification requires no symlink nodes or other non-regular nodes inside the
bundle; the storage-policy ancestor that contains the bundle may remain a
managed symlink. Verification also requires the exact planned artifact set,
independently recomputed checksums and sizes, matching source and manifest
identity, exact CDN URL/object-key sidecars, and repeated redistribution and
secret gates.

Write mode materializes and verifies adjacent staging before it touches the
final path. Publication uses verified staging, atomic replacement, and
prior-final rollback: if replacement fails, the previous verified final is
restored and the failed child receipt records the publication outcome.
`rollback_authority_retained_through_receipt` proves that rollback authority was
kept through final verification and durable child success-receipt persistence.
For a rebuild, the prior-final backup is not removed until that receipt commit
boundary has passed; a final-verification or receipt-persistence failure
restores the prior verified final.

Publication becomes irreversible before destructive backup cleanup begins. A
partial backup cleanup failure preserves the newly verified final and never
restores the partially deleted backup. The failed child evidence reports
`rollback_performed: false`, `cleanup_pending: true`, and the stable failure
point `publication_backup_cleanup`, with `secret_values_recorded: false`.
Operators recover from stable cleanup evidence rather than dumping whole
receipts or filesystem authority. They may remove the remaining adjacent
backup debris only after independently confirming that the newly verified
final still passes the bundle verification contract.

A reviewed matching dry run permits a same-version rebuild only before public
deployment. A non-empty `released_at` or a successful matching live-publication receipt
permanently freezes that bundle version; corrections must mint a new version
instead of replacing published bytes.

Successful write receipts record `bundle_tree_sha256`. A pre-publication
rebuild additionally records `prior_bundle_tree_sha256`,
`current_bundle_tree_sha256`, and `byte_identical`; unchanged source SHA and
manifest digest must produce equal tree digests and byte-identical bundle
contents.

Large-media families produce `.cdn.json` sidecars containing the future CDN URL,
source checksum, byte size, and MIME type. Uploading the large media bytes to a
CDN is a later operator step. Successful child receipts expose
`cdn_upload_sources` and `cdn_upload_sources_sha256`, binding each public
sidecar/object key to a conference-relative private source path, byte size, and
SHA-256 without publishing a resolved private or Drive path in the sidecar.
The later uploader must re-hash the private source contract before upload.
Phase 34 still owns actual upload, prefix repair, and URL liveness.
Placeholder hosts such as `.example.invalid` are allowed for dry-run sidecars
only; write mode rejects placeholder CDN hosts.

Produced-byte scanning is independent at staging and post-publication final
boundaries. Each pass covers `RELEASE.json`, `CHECKSUMS.json`, every inline
artifact, and every emitted CDN sidecar; a source-only scan cannot satisfy
either produced-tree gate. Structural exact-set, checksum, portability, and CDN
identity checks complement the repeated redistribution and secret-content
scans.

## Public Index Build

After the bundle exists, build the precomputed public search index from the
bundle root, not from the private conference workspace:

```bash
python3 scripts/build_public_index.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-index-neurips2025-v1.0.0-receipt.json
```

The builder requires `RELEASE.json` and `CHECKSUMS.json` in the supplied bundle
root and rejects private workspace paths. It indexes textual public-bundle
content only; release metadata, checksum files, CDN sidecars, receipts,
minified assets, and previous `index/` outputs are excluded.

Write mode produces:

| Artifact | Purpose |
|----------|---------|
| `exports/public-release/<bundle_id>/index/corpus_chunks.jsonl` | Bundle-relative public text chunks and provenance |
| `exports/public-release/<bundle_id>/index/bm25_index.pkl` | Offline BM25 payload for keyword retrieval |
| `exports/public-release/<bundle_id>/index/embeddings.npy` | Deterministic offline embedding matrix |
| `exports/public-release/<bundle_id>/index/chunk_ids.json` | Stable chunk ordering contract |
| `exports/public-release/<bundle_id>/index/index_manifest.json` | Source SHA, export manifest digest, chunk policy, capability status, and checksums |
| `storage/workflow-runs/public-index-<bundle_id>-receipt.json` | Inputs, outputs, browser-native capability blockers, and exit code |

Browser-native companion artifacts are best-effort. If optional dependencies
such as `pyarrow` or `sqlite_vec` are unavailable, the manifest records an
explicit capability blocker while the core roadmap artifacts still pass.

## Public Repo, Site, CDN, and Dataset Automation

Phase 08 proves the automation for public publication. The live GitHub,
Cloudflare, Zenodo, and Hugging Face publication steps are operator-gated: review
the dry-run/sandbox receipts first, source the repo-local `.env`, then run the
explicit `--live` command with the required acknowledgement flag.

Stage the read-only public repo package:

```bash
python3 scripts/push_public_repo.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --target-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/public-repo \
  --repo-name neurips2025-data \
  --dry-run \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-repo-neurips2025-v1.0.0-dry-run-receipt.json
```

Live public repo creation or push must re-use a reviewed dry-run receipt whose
`public_safety.status` is `clean`. Each conference gets a fresh public data
repository, so private source-repo history is not part of this publication
surface:

```bash
set -a; source .env; set +a
python3 scripts/push_public_repo.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --target-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/public-repo \
  --repo-name neurips2025-data \
  --reviewed-dry-run-receipt conferences/neurips2025/storage/workflow-runs/public-repo-neurips2025-v1.0.0-dry-run-receipt.json \
  --live \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-repo-neurips2025-v1.0.0-live-receipt.json
```

Generate the static catalog/search site:

```bash
python3 scripts/generate_public_site.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --output-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/site \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-site-neurips2025-v1.0.0-receipt.json
```

Plan CDN/R2 uploads from `.cdn.json` sidecars:

```bash
python3 scripts/upload_public_media.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --provider r2 \
  --bucket neurips-public \
  --prefix conferences/neurips2025 \
  --dry-run \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-cdn-neurips2025-v1.0.0-dry-run-receipt.json
```

After the operator uploads or deploys preview assets, run URL liveness:

```bash
python3 scripts/check_public_urls.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-url-liveness-neurips2025-v1.0.0-receipt.json
```

Package dataset metadata:

```bash
python3 scripts/package_public_dataset.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --metadata-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/metadata \
  --receipt-path conferences/neurips2025/storage/workflow-runs/public-dataset-neurips2025-v1.0.0-receipt.json
```

Prove Zenodo against sandbox before live DOI minting:

```bash
python3 scripts/deposit_zenodo.py \
  --metadata-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/metadata \
  --sandbox \
  --dry-run \
  --receipt-path conferences/neurips2025/storage/workflow-runs/zenodo-neurips2025-v1.0.0-sandbox-receipt.json
```

Live DOI minting requires the private token from `.env` and explicit
acknowledgement:

```bash
set -a; source .env; set +a
python3 scripts/deposit_zenodo.py \
  --metadata-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/metadata \
  --live \
  --ack-live-doi \
  --receipt-path conferences/neurips2025/storage/workflow-runs/zenodo-neurips2025-v1.0.0-live-receipt.json
```

Prepare the Hugging Face Datasets metadata push:

```bash
python3 scripts/push_hf_dataset.py \
  --metadata-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/metadata \
  --repo-id vasiloglou/neurips2025-data \
  --dry-run \
  --receipt-path conferences/neurips2025/storage/workflow-runs/hf-neurips2025-v1.0.0-dry-run-receipt.json
```

Live Hugging Face publication also requires `.env`, `--live`, and
`--ack-live-push`:

```bash
set -a; source .env; set +a
python3 scripts/push_hf_dataset.py \
  --metadata-dir conferences/neurips2025/exports/public-release/neurips2025-v1.0.0/metadata \
  --repo-id vasiloglou/neurips2025-data \
  --live \
  --ack-live-push \
  --receipt-path conferences/neurips2025/storage/workflow-runs/hf-neurips2025-v1.0.0-live-receipt.json
```

After live Zenodo succeeds, back-write the minted DOI into the public release
metadata and rerun the public repo/site packaging proof so badges and citation
metadata agree.

## Live Closeout Verification

Use this sequence after provider publication receipts are written or explicitly
operator-blocked. The commands are reusable for any public bundle; replace
`<slug>`, `<bundle_id>`, receipt names, and provider targets with the current
conference values.

Run the deployment preflight against the reviewed bundle and existing R2 proof:

```bash
python3 scripts/preflight_public_deployment.py \
  --bundle-root conferences/<slug>/exports/public-release/<bundle_id> \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-deployment-preflight-<bundle_id>-receipt.json \
  --github-repo <owner>/<repo> \
  --r2-base-url <https-r2-public-base-url> \
  --public-repo-dry-run-receipt conferences/<slug>/storage/workflow-runs/public-repo-<bundle_id>-dry-run-receipt.json \
  --site-receipt conferences/<slug>/storage/workflow-runs/public-site-<bundle_id>-receipt.json \
  --dataset-receipt conferences/<slug>/storage/workflow-runs/public-dataset-<bundle_id>-receipt.json \
  --r2-upload-receipt conferences/<slug>/storage/workflow-runs/public-cdn-<bundle_id>-r2-live-upload-receipt.json \
  --r2-liveness-receipt conferences/<slug>/storage/workflow-runs/public-url-liveness-<bundle_id>-r2-live-receipt.json
```

Capture provider receipts for every live surface. A provider receipt must either
prove `status: "succeeded"` with its public URL, commit, deployment, DOI, or
dataset identifier, or record `status: "operator_blocked"` with exact blocker
codes and `secret_values_recorded: false`.

After provider receipts are available, restamp release metadata and check
agreement across the public bundle and provider receipts:

```bash
python3 scripts/check_public_release_metadata.py \
  --bundle-root conferences/<slug>/exports/public-release/<bundle_id> \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-release-metadata-agreement-<bundle_id>-receipt.json \
  --preflight-receipt conferences/<slug>/storage/workflow-runs/public-deployment-preflight-<bundle_id>-receipt.json \
  --github-receipt conferences/<slug>/storage/workflow-runs/github-<bundle_id>-live-receipt.json \
  --pages-receipt conferences/<slug>/storage/workflow-runs/cloudflare-pages-<bundle_id>-live-receipt.json \
  --zenodo-receipt conferences/<slug>/storage/workflow-runs/zenodo-<bundle_id>-live-receipt.json \
  --hf-receipt conferences/<slug>/storage/workflow-runs/hf-<bundle_id>-live-receipt.json
```

Republish post-restamp provider content only for providers whose live receipt
succeeded and whose credentials/account scope are approved for the current
operator session. If a provider cannot be updated, write a redacted
post-restamp `operator_blocked` receipt; do not reuse a pre-restamp success
receipt as post-restamp proof.

Verify remote post-restamp provider content before closeout:

```bash
python3 scripts/check_public_urls.py \
  --bundle-root conferences/<slug>/exports/public-release/<bundle_id> \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-url-liveness-<bundle_id>-remote-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/github-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/cloudflare-pages-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/zenodo-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/hf-<bundle_id>-post-restamp-receipt.json
```

The final closeout liveness check should include release metadata, README and
CITATION files, site and dataset metadata, the operator runbook, the closeout
evidence document, the closeout receipt, post-restamp publication receipts, and
all provider receipts:

```bash
python3 scripts/check_public_urls.py \
  --bundle-root conferences/<slug>/exports/public-release/<bundle_id> \
  --receipt-path conferences/<slug>/storage/workflow-runs/public-url-liveness-<bundle_id>-final-receipt.json \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/RELEASE.json \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/public-repo/README.md \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/public-repo/CITATION.cff \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/site/site_manifest.json \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/metadata/croissant.json \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/metadata/datapackage.json \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/metadata/CITATION.cff \
  --extra-metadata conferences/<slug>/exports/public-release/<bundle_id>/metadata/dataset_manifest.json \
  --extra-metadata docs/operations/public-release.md \
  --extra-metadata .planning/phases/<phase-slug>/<phase>-LIVE-CLOSEOUT.md \
  --extra-metadata conferences/<slug>/storage/workflow-runs/phase-live-closeout-<bundle_id>-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/public-release-post-restamp-publication-<bundle_id>-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/public-url-liveness-<bundle_id>-remote-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/github-<bundle_id>-live-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/cloudflare-pages-<bundle_id>-live-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/zenodo-<bundle_id>-live-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/hf-<bundle_id>-live-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/github-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/cloudflare-pages-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/zenodo-<bundle_id>-post-restamp-receipt.json \
  --extra-metadata conferences/<slug>/storage/workflow-runs/hf-<bundle_id>-post-restamp-receipt.json
```

Finish with focused publishing tests. Secret checking for the public GitHub
surface is the staged-content gate inside `scripts/push_public_repo.py`; each
conference public data repository is created from the staged bundle contents
without the private source repository's Git history.

```bash
python3 -m pytest \
  tests/test_public_deployment_closeout.py \
  tests/test_push_public_repo.py \
  tests/test_generate_public_site.py \
  tests/test_public_cdn_liveness.py \
  tests/test_public_dataset_package.py \
  -q
```

## BYOK Tools And Self-Host Kit

Generate the browser BYOK tool surface against the exported public bundle. This
adds `tools.html`, browser assets, a compact public corpus JSON, and a receipt.
The generated browser asset keeps user keys in memory only and calls the user's
chosen provider directly.

```bash
python3 scripts/generate_byok_tooling.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --receipt-path conferences/neurips2025/storage/workflow-runs/byok-tools-neurips2025-v1.0.0-receipt.json
```

Package the self-host kit for GPT-Researcher local-source deep research,
deck/report generation, and podcastfy with Microsoft Edge TTS:

```bash
python3 scripts/package_self_host_tools.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --receipt-path conferences/neurips2025/storage/workflow-runs/self-host-tools-neurips2025-v1.0.0-receipt.json
```

Plan delivered NotebookLM audio with the pinned `notebooklm-py` wrapper. The
default proof is dry-run/operator-gated and does not require Google credentials:

```bash
python3 scripts/generate_notebooklm_audio.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --dry-run \
  --receipt-path conferences/neurips2025/storage/workflow-runs/notebooklm-audio-neurips2025-v1.0.0-dry-run-receipt.json
```

Live NotebookLM generation is optional and operator-gated. It requires the
private `.env` to be sourced, a Google account cookie/profile path, and explicit
acknowledgement. Do not expose this as a hosted public endpoint.

```bash
set -a; source .env; set +a
python3 scripts/generate_notebooklm_audio.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-v1.0.0 \
  --live \
  --ack-live-notebooklm \
  --cookies-path /path/to/operator/notebooklm/cookies.json \
  --receipt-path conferences/neurips2025/storage/workflow-runs/notebooklm-audio-neurips2025-v1.0.0-live-receipt.json
```

SAFETY-04 is documented in
[public-byok-cost-safety.md](public-byok-cost-safety.md). v3.0 ships no
maintainer-funded hosted-key endpoint; any future hosted path must implement a
hard per-session cap and global daily circuit-breaker mapped to
`budget_cap_usd`, never auto-raised.

## Phase 12 Semantic Search: Real-Model Re-Embed and Proof (LOCAL build step)

This section documents the Phase 12 v3.0 semantic-search completion flow.  It is a
**LOCAL build step inside the v3.0 milestone boundary** — no paid providers, no
provider credentials required, key-free throughout.

**Scope (D-12):** re-embed the existing `neurips2025-phase06-proof` pilot bundle with
real dual local models.  Producing a fuller/richer NeurIPS public bundle is a Phase 06
`export_for_public.py` operator step (out of scope here).

### Step 1 — Install the publishing extra

```bash
pip install -e ".[publishing,mlx]"
```

This installs `sqlite-vec` (the PUBWEB-02 durable `.db` deliverable) plus the
build-time `sentence-transformers` gemma runtime.  BGE-M3 weights are already
HF-cached; the `embeddinggemma-300m-ONNX` ONNX weights download once (~23 MB)
on first build.

**Note:** Python's `sqlite3` must support loadable extensions (`enable_load_extension`).
The repo `.venv` (Python 3.11.15, macOS arm64, sqlite lib 3.50.4) supports it.  If a
different env reports `OperationalError: not authorized`, the build records a capability
blocker (`sqlite_vec.status: unavailable`) rather than crashing (RESEARCH Pitfall 2).

Confirm install:
```bash
python -c "import sqlite_vec; import sentence_transformers; print('OK')"
```

### Step 2 — Re-embed and rebuild the pilot index

Run `scripts/build_public_index.py` with the real bundle root and a real domain
query string via `--semantic-proof-query`.  This overwrites the mock `embeddings.npy`
with real BGE-M3 1024-dim vectors, writes the real `vec0` `.db` + packed 768-dim blob,
flips `embedding.implementation` and `sqlite_vec.status` in the manifest, and records
`index/semantic_query_proof.json` (the D-10 completion receipt):

```bash
python scripts/build_public_index.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-phase06-proof \
  --semantic-proof-query "attention mechanisms in foundation models"
```

Choose a query string relevant to the NeurIPS 2025 deck content so the top-K results
are interpretable during the Step 4 eyeball check.

### Step 3 — Rebuild the static site

Re-run `scripts/generate_public_site.py` so the live Semantic/Both search buttons and
the packed blob ship into `site/assets/`:

```bash
python scripts/generate_public_site.py \
  --bundle-root conferences/neurips2025/exports/public-release/neurips2025-phase06-proof \
  --output-dir conferences/neurips2025/exports/public-release/neurips2025-phase06-proof/site
```

### Step 4 — Verify the D-10 completion gate

Check the following after Steps 2-3 complete:

| Check | Expected result |
|-------|-----------------|
| `index_manifest.json` `embedding.implementation` | Real model id (NOT the legacy hash placeholder value) |
| `index_manifest.json` `browser_capabilities.sqlite_vec.status` | `"written"` |
| `index/public_index.sqlite` exists | Yes |
| `index/embeddings_browser.f32` byte length | 66 × 768 × 4 ≈ 203 KB |
| `index/semantic_query_proof.json` exists | Yes |
| `semantic_query_proof.json` `top_k` | Non-empty, descending scores, chunk IDs sensibly related to the query (eyeball) |
| `semantic_query_proof.json` `parity_cosine_python_vs_browser` | Present and `>= 0.95` (the build-time-gemma vs in-browser-gemma drift pass bar) |
| `site/assets/embeddings_browser.f32` | Copied by `generate_public_site.py` |

**REQUIRED parity check:** the `parity_cosine_python_vs_browser` field in
`semantic_query_proof.json` must be `>= 0.95` for the same probe text.  This makes the
build-time-gemma vs in-browser-gemma vector-space drift threat (T-12-07) a checkable
receipt value rather than an operator judgment call (RESEARCH Pitfall 1 / Open
Questions Q1).  A value below 0.95 fails the D-10 gate — switch the build-time gemma
runtime to `optimum[onnxruntime]` running the identical ONNX file (documented in the
Phase 12 Plan 03 runbook) and re-run before approving.

Inspect the manifest and receipt with:
```bash
python -m json.tool \
  conferences/neurips2025/exports/public-release/neurips2025-phase06-proof/index/index_manifest.json \
  | grep -E '"implementation"|"status"'

python -m json.tool \
  conferences/neurips2025/exports/public-release/neurips2025-phase06-proof/index/semantic_query_proof.json
```

### Step 5 — Run the offline regression suite

Confirm nothing regressed after the real rebuild:

```bash
.venv/bin/python scripts/run_offline_regression.py
```

Must exit 0.  The offline suite uses injected fake vectors only — no model download,
no paid provider.

### Credentials and secrets

This flow is key-free and local-only.  No provider credentials are required, and the
`.env` file does not need to be sourced.  Do not copy any `.env` values into proof
receipts or manifests.

---

## Dry Run

Run the public profile dry run:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public --dry-run
```

The preflight report must include and be reviewed for these lines:

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

For NeurIPS 2025, the expected public included families are:

```text
Included families: podcasts, presentations
```

The expected excluded families include private and non-release material such as
`analysis`, `claude_autonomy`, `codex_autonomy`, `export_for_deep_dive`,
`exhaustive_extractions`, `gpt_researcher`, `notebook_lm_findings`,
`storage`, and `vector_store`.

## Evidence To Confirm

Before any write-mode sync, confirm:

| Evidence | Required result |
|----------|-----------------|
| `Profile:` | `public` |
| `Target root status:` | `reachable` |
| `Leaf status:` | `existing` or `missing` during dry run; `created` only after a write-mode run materializes the intended public target |
| `Included families:` | Only curated release families |
| `Excluded families:` | Private, backup-only, cache, storage, and support families are excluded |
| `Dry run:` | `enabled` |
| `Delete mode:` | `disabled` unless the reviewed delete gate below is being used |

NeurIPS 2025 public restore evidence from Phase 3 lives at:

- `conferences/neurips2025/storage/prune-verification/public/presentations.json`
- `conferences/neurips2025/storage/prune-verification/public/podcasts.json`

## Write Mode

After the dry run is reviewed, run the same command without `--dry-run` only
when the included and excluded families match the release intent.

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public
```

## Delete Gate

`--delete` is destructive. It can remove remote files inside the included
family destinations that are not in the selected local manifest set.

Hard gate: `--delete` is only allowed with either
`--reviewed-dry-run-receipt` pointing at a matching successful
`--dry-run --delete` receipt, or the legacy `--reviewed-dry-run`
acknowledgement.

Use delete mode only after saving and reviewing a matching delete-preview
receipt:

```bash
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public --dry-run --delete --receipt-path conferences/neurips2025/storage/workflow-runs/<run-id>/public-delete-preview-receipt.json
python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public --delete --reviewed-dry-run-receipt conferences/neurips2025/storage/workflow-runs/<run-id>/public-delete-preview-receipt.json
```

If the preceding delete-preview dry run did not show the exact intended
release set, do not run delete mode. The older
`--delete --reviewed-dry-run` flow remains available for compatibility only.

## Troubleshooting

### Wrong Target Paths

If `Target root status:` is not `reachable`, the mounted Drive parent is wrong
or unavailable. Stop and fix the mount or pass the intended `--remote-path`.
Do not create a new root that only looks correct by name.

If `Leaf status:` is `missing` during dry run, the mounted parent root is
reachable but the conference leaf has not been created yet. Review the target
path before write mode.

If `Leaf status:` is `created`, confirm that a write-mode run materialized the
intended conference destination. A typo in the final folder name can create a
valid but wrong release location.

### Suspicious Dry-Run Deltas

If public sync wants to copy private lanes, stop and inspect
`storage/storage_manifest.json`. The public profile should not include private
analysis, raw exports, sidecar outputs, caches, or storage internals.

If the remote contains unexpected top-level folders, use a non-delete write
sync first or archive the legacy material deliberately. Do not use delete mode
to guess your way back to a clean target.

Legacy or otherwise out-of-profile top-level directories already present in the
mounted leaf remain untouched. `--delete --reviewed-dry-run` only prunes within
the included family destinations.

### Failed Sync

If `rsync` exits non-zero, keep the dry-run output and command line. Check
mount availability, destination permissions, and whether the local manifest path
still resolves. Retry only after the cause is clear.

### Family Evidence Mismatch

If restore evidence shows failed sample hashes or a remote that is not a
superset of local public deliverables, refresh the public target before treating
that family as recoverable or release-complete.
