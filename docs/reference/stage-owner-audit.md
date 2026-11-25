# Stage Script Registry And Owner Audit

`src/conference_pipeline/stage_owner_registry.json` is the canonical
platform-wide registry for scripts and modules used by analysis and production
stages. It records required, allowed, deprecated, and forbidden stage scripts,
then binds them to workflow stages, governed task ids, and planning skills.

`src/conference_pipeline/stage_owner_audit.py` remains the compatibility guard
for existing owner-command plan checks. `src/conference_pipeline/stage_script_audit.py`
and `scripts/verify_stage_script_usage.py` are the unified verification layer.

## Registry

The registry schema includes:

- `scripts`: canonical script/module entries with refs, path, kind, status,
  scope, existence policy, and registry-owned `usage_doc_refs`.
- `stage_bindings`: workflow, production, and presentation stages mapped to
  required, optional, and forbidden scripts.
- `task_bindings`: governed task ids such as `podcast.script_generation`,
  `presentation.svg_figure_generation`, and `presentation.visual_qc`.
- `skill_contracts`: script requirements explicitly mentioned by
  `plan-conference-analysis` and `plan-conference-production`.
- `evidence_rules`: where usage can be proven, including plan text, approval
  summaries, workflow receipt commands, command metadata, presentation manifest
  owner refs, explicit waivers, and registry usage-documentation anchors.

## Runtime Stage Contract Gate Pattern

Planning text is not sufficient evidence that a production stage enforced its
owners. Every production runner that can write a succeeded stage receipt should
call a runtime stage contract gate immediately before it writes `status:
"succeeded"`. The gate compares the stage's registry-owned contract against the
workspace/run evidence and blocks when a required owner receipt, output family,
lineage binding, or negative proof is missing.

The reusable helper shape is:

```python
verify_stage_contract(
    workspace=workspace,
    run_id=run_id,
    stage_id="analysis",
    required_owner_ids=[
        "analysis.topic_strategy",
        "analysis.exhaustive_topic_extraction",
        "analysis.deep_narrative_module",
    ],
)
```

The helper returns structured accepted evidence and blockers suitable for a
stage receipt. Live success receipts must embed the result under
`stage_contract` alongside `script_registry_sha256` and
`required_script_ids`. It must never rely on roadmap prose alone. If a required
owner is listed in a plan but not enforced by the runner, the runner must block
instead of producing a succeeded receipt.

Required upstream receipts may also carry their own `checks` predicates in the
registry. This prevents a stale or malformed upstream receipt from passing just
because it says `status: succeeded`; for example, the analysis contract can
require the LLM full-read receipt to prove `scope: all`, `limit: null`,
`checks.completion_state: complete`, `completed_unit_count ==
expected_unit_count`, and `failed_count: 0`.

This pattern is intentionally stricter than the existing planning verifiers:

- Planning verification proves the roadmap, skills, docs, and registry mentions
  are aligned.
- Runtime stage contract verification proves the executable runner actually
  enforced those mentions for the current workspace/run id.
- Receipt verification proves the exact required files, digests, statuses, and
  lineage bindings were present when the stage succeeded -- and that every
  input the receipt recorded still carries the digest it recorded.

Existing runners already enforce parts of this pattern: active blocker checks,
download/transcription receipt checks, workshop talk-media gates,
`vector_embeddings_written: true`, topic-strategy receipt validation, manifest
validation, and presentation visual-QC gates. `verify_stage_contract` is the
universal layer that records the accepted evidence and blockers in the receipt.
Future conference runners should express cross-stage proof requirements in the
registry contract before adding stage-local one-offs.

The implementation priority is:

1. Add or update the `runtime_contract` object on the stage binding.
2. Name required receipts, required outputs, required checks, forbidden modes,
   lineage inputs (`lineage.input_artifacts`, `required_upstream_stages`), the
   stage's `receipt_filename` and artifact tables, and non-production/
   unavailable stage status there.
3. Call `verify_stage_contract` immediately before writing live success.
4. Generate negative tests from the registry: missing receipt, failed receipt,
   mock/dry-run receipt, stale digest, missing output family, and missing runner
   enforcement must all block.

## Shared Production Stages

The late production stages are conference-neutral code under
`src/conference_pipeline/production/`, and every conference runner adopts them
by building a `ProductionRunnerContext` (workspace, run id, its own stage
table, receipt type and code identity, plus its env-file loader, run-summary
hook and provider-cost approval hook) rather than by copying the stage:

| Stage | Module | Notes |
|-------|--------|-------|
| `presentation` (Stage 10) | `production/presentation_stage.py`, `production/presentation_chain.py`, `production/presentation_steps.py` | Gates (inputs, lineage, legacy Markdown, cost approval, canary scope), the per-deck chain with per-deck step ledgers, the executive digest-chain verification, the cross-deck link audit and the strict final contract. A per-deck step failure is a returned outcome; the shell records it and continues to the next deck. |
| `notebooklm-podcast` (Stage 10.4) | `production/notebooklm_stage.py` | Gates then `podcast_notebooklm.run_stage`; a `--deck-ids` canary is always `blocked`. |
| `release` (Stage 11) | `production/release_stage.py` | Over `release_bundle`; publication stays blocked without a target and an approver. |
| `proof` (Stage 12) | `production/proof_stage.py` | Over `final_proof`; support labels propose `<workspace slug>-live-proven`, decided against conference-local receipts only. |

The runners `scripts/run_cvpr2026_production.py` and
`scripts/run_icml2026_production.py` call these modules; `release.bundle` and
`proof.final_verification` in the registry point at the shared modules. The
automated blueprint approver id and the phase-13 exemption evidence tokens are
derived from the runner id, never written as a conference literal.

## Receipt Lineage

Since v5.1 every stage receipt records what it consumed and produced, and
`verify_stage_contract` fails closed when any of it moved:

- `output_artifacts` / `output_digest`: the stage's outputs, from the
  binding's `output_artifact_checks` table (`[name, receipt_key, sha_key?]`
  rows with dotted keys from the receipt root, e.g.
  `checks.index_digest_path`); `output_digest` is canonical over the sorted
  artifact digests, never over the receipt file, so a no-op re-run does not
  cascade.
- `lineage.upstream_receipts`: one entry per registry `required_receipts`
  path and per `required_upstream_stages` id (resolved through the binding's
  `receipt_filename`). A `required_upstream_stages` entry is a stage id or
  `{"stage": id, "layouts": ["production"]}` -- a prerequisite only the
  production runners have (the generic workflow binds and verifies with
  `layout="generic"` and never writes `corpus-inventory-index` or
  `presentation` receipts). Entries are keyed by workspace-relative path, recording the
  upstream's `output_digest` (stage receipts) or `content_digest` (sub-step
  receipts, and any spec marked `"lineage": "content"` such as a pinned
  historical receipt). A stage receipt without `output_digest` is a blocker,
  never silently content-hashed.
- `lineage.input_artifacts`: registry `lineage.input_artifacts` plus the
  binding's `input_artifact_checks` rows, always hashed from disk at bind
  time.
- `lineage.lineage_digest`: over the recorded upstream digests and input
  shas (not `code_identity`), keyed by run-independent identity.

Blocker codes: `stage_contract_lineage_missing`,
`stage_contract_lineage_upstream_unbound`, `stage_contract_upstream_changed`,
`stage_contract_upstream_output_digest_missing`,
`stage_contract_upstream_stale` (transitive; carries the `chain`),
`stage_contract_input_artifact_changed` / `_missing`,
`stage_contract_spec_key_unsupported` (the mtime `fresh_after` binding is
gone; the registry loader rejects it). `require_lineage` defaults to `True`
(`stage_contracts.REQUIRE_LINEAGE_DEFAULT`).

Runner recipe: bind lineage on the *provisional* receipt of a fresh run
(`receipt_lineage.finalize_receipt(..., bind=True)`) before the contract
check; verify a *persisted* receipt as-is on every resume short-circuit
(`bind=False`), and pass the recorded `lineage=` through any path that
rebuilds a receipt -- a resume must never mint fresh lineage for a receipt it
did not produce. Write receipts through `preserve_prior_receipt` so the
prior version survives as `<stem>.superseded-<stamp>.json` (or
`.attempt-<stamp>.json` when the prior one had not succeeded).

Sub-step receipts: some stages consume receipts written by commands that
are not registry stages -- the LLM full read (`14A-llm-full-read`) and the
analysis sub-steps 07A-07H. The registry's top-level `lineage_substeps`
declares each one's `receipt` and what it consumes
(`required_upstream_stages`, `required_substeps`, and `input_artifacts` for
inputs with no receipt of their own -- the full read, 07A, 07B and 07D bind
`analysis/topic_strategy.json` (a taxonomy edit makes them stale), and 07D/07E
bind the narrative source packs, projected with `exclude_keys` so the packs' 07D/07E enrichment and
restamped receipt digests do not count as changes; validated: known stages
with a `receipt_filename`, known sub-steps, unique receipts, acyclic). Each
writer calls `receipt_lineage.stamp_substep_receipt` before it persists its
receipt, and the verifier follows a declared sub-step's lineage like a
stage's: a sub-step without lineage is stale (`lineage_missing`), so a
rebuilt corpus index now reaches the full read, every 07x step after it,
and the stages that consume them. A writer that scrubs its receipt on the
way to disk must persist it with `receipt_lineage.sanitize_receipt_for_write`,
never by passing the whole receipt through `sanitize_for_receipt`: the
sanitizer redacts by key name, and a scrubbed `exclude_keys` (the field names
a projection skips) makes verification project a different digest and report
`input_artifact_changed` for an input nothing touched -- it falsely staled the
2026-09-20 CVPR full read, an 11-hour run. Lineage carries digests and
workspace-relative paths only, so it is never a credential path. A downstream binds a declared sub-step
by its `substep_outputs` digest -- the stamp's digest of the registry-declared
`outputs` (JSON canonically, run-time stamps removed) -- the way it binds a
stage by `output_digest`: the receipt's own counts can stay equal while its
outputs change, and its timings and reuse counts change on an unchanged
resume. Only a succeeded, non-dry-run run that covered the whole scope is bound: a
dry run (even one reporting succeeded), a failed attempt, or a
topic/lane/category/question subset (or a 07C run over a selection file
other than 07B's canonical output) leaves the receipt unbound
(`lineage_unbound_reason`), and verification rejects a sub-step upstream
that is not `succeeded` (`stage_contract_upstream_not_succeeded`). Values
derived from digested Markdown (`markdown_sha256`, token estimates) are kept
out of the output digests, and the 07E ledger hashes only the source-pack
fields it reads (`LEDGER_SOURCE_PACK_KEYS`, and only `selected_items` of the
full-read evidence), never the pack's 07D/07E/repair enrichment. Outputs a
complete run no longer produces are moved out of the digested, glob-read
output tree before stamping, never deleted: the full read moves records of
units the corpus no longer has to `llm-full-read.pruned/<stamp>/`; 07C moves
reports and evidence of deep dives 07B no longer selects, and 07A moves a
summary whose category has no evidence left, to
`storage/lineage-pruned/<label>/<stamp>/` (`archive_obsolete_outputs`,
listed in the receipt's `pruned_outputs`). A live 07D resume never reuses a
dry-run placeholder. Sanctioned in-place
edits (07F/07G reference repairs rewriting 07D's
Markdown) call `refresh_substep_outputs`, which re-digests the outputs and
keeps the recorded lineage (never re-binding a stale 07D). Writers
with a resume cache reuse an item only when its recorded input digest still
matches (every writer first marks its previous receipt unbound -- `unbind_substep_receipt`,
`lineage_unbound_reason: in_progress` -- so a re-run that crashes after
rewriting some outputs is stale, not certified; 07A writes a pair's Markdown
before the JSON that carries its resume key and the Markdown's hash, both
checked before reuse;
full read: the unit's `text_sha256`, `input_kind`, `metadata` and
`topic_prompt_sha256` -- records written before that field existed are
matched on text and metadata and counted in `legacy_prompt_unverified_count`;
07A: pair `input_sha256` over the rendered evidence
blocks, topic title and category; 07C: `lane_input_sha256`, which folds in the
corpus index's `output_digest` and is written only after the lane's report,
so an interrupted rebuild is never reused; 07D: its existing
`input_digest_sha256`), so re-stamping a resumed sub-step never binds old
outputs to new inputs. Execution bookkeeping (`reused`, provider receipt
paths, `provider_metadata`, `usage`) is left out of `substep_outputs`, so an
unchanged resume keeps its digest. The topic
strategy receipt stays an undeclared leaf on purpose: it is the frozen topic
decision and must not be invalidated by a corpus change. `workflow
stage-status` lists the sub-steps below the stages, in dependency order,
with `next sub-step to run`.

Pre-spend gate: a fresh run binds its direct upstreams as they are now, so
the final contract check only catches a stale upstream (one whose own
lineage moved) *after* the stage has done its work. Every paid stage
therefore calls `receipt_lineage.upstream_freshness_blockers` after its own
input gates and before it constructs a paid client or writes an output, and
blocks with `upstream_lineage_not_fresh` (the stale chain is recorded under
`checks.upstream_lineage_blockers`). CVPR: `_cvpr_upstream_lineage_checks`
in extraction, analysis, QA/gap-fill, podcast, presentation and NotebookLM
podcast. ICLR/ICML: `block_if_upstream_not_fresh` in extraction, analysis,
QA/gap-fill, podcast and presentation; it writes a blocked stage receipt but
no blocker receipt, because a run-wide active blocker would also halt the
upstream re-run that clears it.

Operations: `conference-pipeline workflow stage-status` classifies every
stage of a run as FRESH / STALE (with the reason and chain) / MISSING /
BLOCKED read-only; `conference-pipeline workflow bind-lineage --backfill`
stamps receipts written before lineage existed, evidence-based and read-only
without `--apply`. Evidence must identify the upstream *version*: its receipt
file sha or a sha from its `output_artifact_checks`, never a value it merely
shares with its consumer; a `stage<N>_*` check that names a different upstream
sha, or a value matching only a `.superseded-*` version, leaves the entry
unresolved. Backfill keeps its own pre-stamp copy as
`<stem>.prebackfill-<stamp>.json` (the same execution, not a superseded one)
and processes stages in `stage_order`, dependencies first on ties.

The default post-planning owner gate still checks:

- `planning.workspace_gitignore_policy`
- `analysis.gpt_researcher_hybrid_retrieval`
- `analysis.gpt_researcher_heuristic_prefilter`
- `analysis.deep_dive_question_selection`
- `analysis.dual_deep_dive_research_lanes`
- `podcast.script_generation`
- `podcast.tts`
- `podcast.notebooklm_audio_overview`
- `presentation.deck_contract_generation`
- `presentation.source_binding`
- `presentation.visual_plan_generation`
- `presentation.render_spec_generation`
- `presentation.render_export`
- `presentation.delivery_generation`
- `presentation.presentation_review`
- `presentation.video_sync_validation`
- `presentation.video_delivery_validation`
- `presentation.svg_figure_generation`
- `presentation.visual_qc`

Production presentation proof is structured. `presentation_frame.json`,
`presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `render_spec.json`,
`delivery_single_speaker.json`, `delivery_notebooklm.json`,
`6_presenter_script.md`, `presentation_review.json`,
`video_sync_validation_single_speaker.json`, and
`video_sync_validation_notebooklm.json` are production evidence. Old numbered
slide draft Markdown is legacy-only numbered Markdown and may appear only as
debug-export-only output with `production_evidence: false`; production
plans and audits must block that path with
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED`.

## Stage Script Registry Tool Usage Index

Every `scripts` entry in `src/conference_pipeline/stage_owner_registry.json` owns at least one `usage_doc_refs` anchor in this section and one `diagnostic_profile` with success, blocking, output, and next-action messages. Planning skills and generated plans must point to these registry-owned anchors for tool instructions instead of copying undocumented command fragments, and agents should use emitted diagnostics before retrying or advancing stages.

### Planning Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="gitignore-workspace-policy"></a>`gitignore.workspace_policy` | `required` / `planning` | Generate and check the workspace `.gitignore` from `storage_manifest.json` storage-class policy. | `scripts/generate_workspace_gitignore.py`. `--write --check --receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json`. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-analysis-audit"></a>`planning.analysis.audit` | `required` / `planning` | Audit generated analysis planning artifacts before approval. | `.claude/skills/plan-conference-analysis/scripts/audit_generated_plan.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-analysis-install"></a>`planning.analysis.install` | `allowed` / `planning` | Install or sync the analysis planning skill files. | `.claude/skills/plan-conference-analysis/scripts/install_skill.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-analysis-scaffold"></a>`planning.analysis.scaffold` | `required` / `planning` | Scaffold analysis milestone drafts from a conference brief. | `.claude/skills/plan-conference-analysis/scripts/scaffold_plan.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-production-audit"></a>`planning.production.audit` | `required` / `planning` | Audit generated production planning artifacts before execution. | `.claude/skills/plan-conference-production/scripts/audit_generated_production_plan.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-production-install"></a>`planning.production.install` | `allowed` / `planning` | Install or sync the production planning skill files. | `.claude/skills/plan-conference-production/scripts/install_skill.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="planning-production-scaffold"></a>`planning.production.scaffold` | `required` / `planning` | Scaffold live production plans from an approved analysis spec. | `.claude/skills/plan-conference-production/scripts/scaffold_production_plan.py`. Run from the skill scripts directory and record pass/fail in generated approval evidence. | [Source of truth](../source-of-truth.md); [pipeline workflow](../workflows/pipeline-workflow.md) |

### Acquisition Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="acquisition-workshop-acquisition-pipeline"></a>`acquisition.workshop_acquisition_pipeline` | `required` / `acquisition` | Conference-general in-process pipeline that orchestrates `acquisition.workshop_discovery` and `acquisition.workshop_paper_download` for a workspace. Per-conference production runners (`scripts/run_<slug>_production.py`) should import and call `run_workshop_discovery_and_download(workspace, parent_venue=..., rate_limit_rps=1.4)` rather than redefining the orchestration locally. | `from conference_pipeline.acquisition import run_workshop_discovery_and_download`. Returns a summary dict with `status`, `discovery`, `download`, `inventory_path`, `discovery_receipt_path`, and `download_receipt_path` suitable for merging into stage-04 receipts. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-workshop-discovery"></a>`acquisition.workshop_discovery` | `required` / `acquisition` | Enumerate OpenReview workshop sub-venues under a conference parent group and write a per-workshop inventory (paper_count, tiers, `pdf_availability_class`). Uses `https://api2.openreview.net`. Since ~2026-07 the notes API is challenge-gated for unauthenticated clients; pass `--openreview-auth required` so requests carry a Bearer token minted from `OPENREVIEW_USERNAME`/`OPENREVIEW_PASSWORD` (via `conference_pipeline.openreview_auth`). Receipts record names-only `openreview_auth_mode`/`openreview_auth_env_names`. | `scripts/discover_openreview_workshops.py --venue-prefix <ICLR.cc/2026/Workshop> --output storage/workshops/inventory.json --receipt-path storage/workshops/discovery_receipt.json --openreview-auth required --workspace conferences/<slug>`. Pass `--venue-prefix` matching the conference's OpenReview parent workshop group. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-workshop-paper-download"></a>`acquisition.workshop_paper_download` | `required` / `acquisition` | Download PDFs for every `openreview_with_pdf` venue at ≤1.4 rps with HTTP 429 backoff and resume; emit per-paper receipts with `content_type=application/pdf`, `sha256`, `byte_count > 0`. Workshops classified `openreview_metadata_only` or `no_openreview_record` are surfaced in `skipped_external_workshops` and require an external-site downloader scaffold. Since ~2026-07 the PDF endpoints are challenge-gated; pass `--openreview-auth required` so authenticated runs fetch PDFs from `api2.openreview.net/attachment` with a Bearer token from `OPENREVIEW_USERNAME`/`OPENREVIEW_PASSWORD`. | `scripts/download_openreview_workshop_papers.py --inventory storage/workshops/inventory.json --output-dir storage/workshops/papers --rate-limit-rps 1.4 --receipt-path storage/workshops/download_receipt.json --openreview-auth required --workspace conferences/<slug>`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-cvf-openaccess-workshop-paper-download"></a>`acquisition.cvf_openaccess_workshop_paper_download` | `allowed` / `acquisition` | CVF-published conferences (CVPR, ICCV, WACV) only. Download archival workshop papers from the CVF Open Access workshop proceedings (`urls.openaccess_workshops` in the workspace `config.yaml`, defaulting to `<urls.openaccess>_workshops/menu`). Fills the archival part of the `no_openreview_record` / `openreview_metadata_only` gap from one public source, annotates each CVF workshop with its matching OpenReview venue, and skips papers already downloaded from OpenReview (exact title, or retitled camera-ready with shared authors). Per-paper receipts use the `conference_pipeline.workshop_paper_receipt` schema under `storage/workshops/papers/openaccess-thecvf-com_*`. Non-archival tracks are not on this site and still need the workshop site or arXiv. | `PYTHONPATH=src .venv/bin/python scripts/download_cvf_openaccess_workshop_papers.py --workspace conferences/<slug> --dry-run`, then without `--dry-run`; receipt `storage/workshops/cvf_openaccess_download_receipt.json` with `totals`, `per_workshop[].duplicates_of_openreview`, and `openreview_venueid`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-openreview-workshop-groups"></a>`acquisition.openreview_workshop_groups` | `allowed` / `acquisition` | Save every OpenReview workshop group's full title, subtitle and website (the inventory has venue ids only). Coverage matching needs them: short names mis-match renamed workshops and sub-tracks. | `PYTHONPATH=src .venv/bin/python scripts/fetch_openreview_workshop_groups.py --workspace conferences/<slug>` (source `.env`; venue prefix from `urls.openreview_workshop_parent`) → `storage/workshops/openreview_workshop_groups.json`. | [Workshop paper discovery skill](../../.claude/skills/conference-workshop-paper-discovery/SKILL.md) |
| <a id="acquisition-workshop-paper-coverage-audit"></a>`acquisition.workshop_paper_coverage_audit` | `allowed` / `acquisition` | Classify every workshop on the official schedule by paper source (OpenReview, proceedings hosts from `acquisition.proceedings_sources`, registered workshop-site/arXiv sources): website, short-name-in-URL and title matching, sub-track grouping, one-to-one assignment. Lists proceedings workshops without an OpenReview venue and unmatched groups. | `PYTHONPATH=src .venv/bin/python scripts/audit_workshop_paper_coverage.py --workspace conferences/<slug> --schedule <schedule.json> --id-field <key> --website-field <key>` → `storage/workshops/workshop_paper_coverage.{json,md}` with `category_counts` and `workshops_with_papers`. | [Workshop paper discovery skill](../../.claude/skills/conference-workshop-paper-discovery/SKILL.md) |
| <a id="indexing-paper-title-resolution"></a>`indexing.paper_title_resolution` | `allowed` / `acquisition` | Resolve paper titles (title similarity ≥ 0.9) to open PDF copies, DOI-registered versions at other venues, and arXiv venue notes. Default backends arXiv + Crossref are free; OpenAlex (metered daily budget) and Semantic Scholar are opt-in with `OPENALEX_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY`. Resumable; a refusing backend is disabled for the run. | `PYTHONPATH=src .venv/bin/python scripts/resolve_paper_titles.py --research-dir <dir> --output <resolutions.json> --workspace conferences/<slug>` (or `--input` with `{key,title}` records). | [Workshop paper discovery skill](../../.claude/skills/conference-workshop-paper-discovery/SKILL.md) |
| <a id="acquisition-workshop-site-paper-download"></a>`acquisition.workshop_site_paper_download` | `allowed` / `acquisition` | Verify and download papers found by workshop gap research (workshop websites, Drive, arXiv via title resolution). Only fetched `%PDF-` bodies count; 401/403 (e.g. private OpenReview notes) are `not_public`. Skips papers already acquired by OpenReview or registered sources; registers source `workshop_site` on success. | `PYTHONPATH=src .venv/bin/python scripts/download_workshop_site_papers.py --workspace conferences/<slug> --research-dir <dir> --resolutions <file> --verify-only`, then without `--verify-only` → `storage/workshops/workshop_site_download_receipt.json`. | [Workshop paper discovery skill](../../.claude/skills/conference-workshop-paper-discovery/SKILL.md) |
| <a id="acquisition-workshop-paper-sources"></a>`acquisition.workshop_paper_sources` | `allowed` / `acquisition` | Registry of supplementary workshop-paper download receipts (CVF Open Access, workshop sites, arXiv) with pinned sha256; corpus-inventory stages admit every registered source through the shared per-paper receipt path and block on drift. | `storage/workshops/workshop_paper_sources.json`; downloaders call `register_source` on success; stages call `load_registered_sources` and bind `workshop_paper_sources` in checks. | [Workshop paper discovery skill](../../.claude/skills/conference-workshop-paper-discovery/SKILL.md) |
| <a id="acquisition-browser-cookie-export"></a>`acquisition.browser_cookie_export` | `allowed` / `acquisition` | Export one domain's cookies from a logged-in local browser profile into a Netscape cookie file, reusing an existing session for conference virtual-site pages/media that are gated behind attendee login (icml.cc, iclr.cc, neurips.cc). Reads cookies via yt-dlp, keeps only the requested `--domain`, writes owner-only permissions, and prints cookie names only (never values). Point the conference's `*_COOKIES_FILE` env var at the output path. | `scripts/export_browser_cookies.py --browser chrome --domain icml.cc --output <path-outside-repo>`. Then set `ICML_COOKIES_FILE=<path>` in `.env`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-workshop-talk-media"></a>`acquisition.workshop_talk_media` | `required` / `acquisition` | Crawl workshop agenda/site pages for talk media after workshop paper download receipts exist. The owner must run a deterministic public-first/authenticated retry, then use `acquisition.embedded_media_recovery` for embedded-player extraction before treating media as inaccessible. ICLR production must keep embedded recovery SlidesLive-only unless a future conference plan explicitly widens the host allowlist. Each workshop inventory row must produce talk media records or explicit per-workshop blockers; paper PDFs, zero-workshop-speech audits, unauthenticated agenda probes, direct-link-only crawls, YouTube/Vimeo/Zoom/social/profile embeds for ICLR, and silent skips are rejected evidence. | `scripts/acquire_iclr2026_workshop_talk_media.py --workspace conferences/<slug> --run-id <run-id> --resume --summary --live-crawl --iclr-cookie-file <local-cookie-file> --embedded-media-host slideslive.com --workshop-worker-count <n> --slide-worker-count <n>`. Writes `storage/workshops/talk_media_acquisition_receipt.json` and `storage/workshops/talk_media_inventory.json`; records include `workshop_id`, `talk_id`, media provenance fields, sanitized cookie-file status, `embedded_audio_recovery.allowed_hosts`, worker counts, and blockers such as `no_site_url`, `inaccessible_agenda`, `no_media_links`, `embedded_media_download_error`, `unsupported_embed`, `auth_wall`, or `download_error`. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Source of truth](../source-of-truth.md) |
| <a id="acquisition-embedded-media-recovery"></a>`acquisition.embedded_media_recovery` | `required` / `acquisition` | Reusable helper for NeurIPS/ICLR-style pages where operator-visible video/slides are exposed through page HTML or JavaScript state rather than direct anchors. It extracts SlidesLive ids plus supported YouTube, Vimeo, and Zoom video URLs from HTML/JS, ignores generic player scripts and social/profile URLs, sanitizes secret-bearing query params, deduplicates candidates, and routes conference-approved hosts to yt-dlp style audio recovery. | `conference_pipeline.acquisition.embedded_media` / `src/conference_pipeline/acquisition/embedded_media.py`. Conference-local acquisition owners should call `embedded_media_links(..., direct_media_predicate=...)`, `is_recoverable_embedded_media_url`, `sanitize_media_url`, and `existing_completed_audio` before emitting `unsupported_embed` or `no_media_links` blockers, and must apply a conference-local host allowlist such as `--embedded-media-host slideslive.com` for ICLR. Receipts must record sanitized media URLs, `allowed_hosts`, and a digest of any unsanitized download URL rather than secret-bearing values. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Source of truth](../source-of-truth.md) |
| <a id="acquisition-slideslive-media-recovery"></a>`acquisition.slideslive_media_recovery` | `required` / `acquisition` | Recover session media when authenticated ICLR item pages contain SlidesLive embeds but no local audio/video artifact was merged into `download_corpus.json`. This owner scans only cached `item_page_auth_probe` HTML, strips secret-bearing embed query params, downloads/resumes matching SlidesLive audio, and merges those audio artifacts into the existing download-corpus inventory. | `scripts/recover_iclr2026_slideslive_media.py --workspace conferences/<slug> --run-id <run-id> --resume --worker-count 2 --summary`. Use `--inventory-only` for a no-download audit and omit `--max-presentations` for production recovery. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-slideslive-slide-download"></a>`acquisition.slideslive_slide_download` | `required` / `acquisition` | Recover SlidesLive slide images when cached authenticated item pages contain SlidesLive embeds. This owner scans only cached `item_page_auth_probe` HTML, uses `slides.json` first, falls back to yt-dlp thumbnail metadata for `/slides/original/` images, can use `--browser-fallback` for browser-intercepted slide image metadata, writes per-presentation `sync.json`, and merges slide-image artifacts into the existing download-corpus inventory. | `scripts/recover_slideslive_slides.py --workspace conferences/<slug> --run-id <run-id> --resume --worker-count 4 --summary`. Use `--inventory-only` for a no-download audit. Receipt: `storage/download-corpus/slideslive_slide_recovery.json`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-slide-reference-detection"></a>`acquisition.slide_reference_detection` | `required` / `acquisition` | Detect generic conference/banner slides across presentation starts using perceptual hashes so cleanup can remove them from OCR/indexing without deleting raw files. Supports legacy `data/<conference>/decks` layouts and generic `--slides-root` trees such as download-corpus slide image roots. | `scripts/detect_reference_slides.py --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset --output-dir conferences/<slug>/storage/download-corpus/reference_slides --receipt-path conferences/<slug>/storage/download-corpus/slide_reference_detection.json --summary`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-slide-cleanup"></a>`acquisition.slide_cleanup` | `required` / `acquisition` | Mark consecutive duplicate slides and detected generic/banner slides with `skip_ocr: true` / `skip_reason` in `sync.json` and matching download-corpus artifacts. This is non-destructive cleanup: raw slide images remain available for audit, while downstream OCR/indexing excludes skip-marked slides. | `scripts/cleanup_slides.py --slides-root conferences/<slug>/cache/acquisition-downloads/media_asset --workspace conferences/<slug> --download-corpus-path storage/download-corpus/download_corpus.json --receipt-path conferences/<slug>/storage/download-corpus/slide_cleanup.json --summary`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="acquisition-youtube-talk-harvest"></a>`acquisition.youtube_talk_harvest` | `required` / `acquisition` | Harvest conference talks from public YouTube against a program spine (keynote/invited and oral families first) when official recordings are paywalled or absent. Bidirectional match: Direction A enumerates channels/playlists/broad searches and matches back to the spine; Direction B runs per-item yt-dlp search plus an optional Tavily lane (`TAVILY_API_KEY`, key never written to receipts) for unmatched items. Transcript-first, two tiers: (1) captions pulled via yt-dlp (manual before auto for provenance), no audio/video download; a YouTube "confirm you're not a bot" gate raises `YouTubeBotCheckBlocked` and stops the caption tier for the rest of the run instead of being recorded as a missing caption; (2) optional paid Gemini tier (`--gemini-fallback --gemini-budget-usd <usd>`, `GOOGLE_API_KEY`) transcribes the public YouTube URL server-side with the `transcription.youtube_url` registry model, in `--gemini-window-seconds` windows (default 900; longer windows silently drop speech) at `--gemini-fps` (default 0.02), duration and playability read from the watch page (taken-down/private videos become `video_unavailable`; offsets past the end return a bare HTTP 500, so the last window stops short), truncated or sparse windows re-asked in halves, every call via `record_provider_call`, halting before a window would cross the budget; provenance `gemini_youtube_url_transcription`. Talks left without a transcript are marked `asr_fallback_required` for `transcription.session_speech`. `--video-list <json>` runs the same tiers over explicit `{video_id,title,channel,duration_seconds}` rows without program matching, checkpointing each video (`video_transcript_inventory.json`, `video_transcript_receipt.json`). Receipts record `allowed_hosts`, sanitized URLs, per-talk match direction/confidence, and explicit gaps with `must_investigate: true` for keynote/oral misses. | `PYTHONPATH=src .venv/bin/python scripts/harvest_youtube_talks.py --workspace conferences/<slug> --run-id <run-id> --conference-label "<Conference YYYY>" --program-spine <spine.json> --source <channel-or-playlist-url> --broad-query "<Conference YYYY> keynote" --tavily --resume --summary`. Outputs `storage/youtube-talks/{talk_inventory.json,harvest_receipt.json,captions/,transcripts/}`. `--pilot-limit` is development-only and recorded in the receipt; live production corpus commands must not cap item counts. | [Pipeline workflow](../workflows/pipeline-workflow.md) |

### Analysis Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="analysis-deep-narrative-module"></a>`analysis.deep_narrative_module` | `required` / `analysis` | Render and validate platform deep analysis reports and slide-brief sidecars. | `conference_pipeline.analysis.narrative`. Use registered refs in plan text, receipts, `command_metadata.owner_ref`, or manifest owner refs. | [Analysis contract](../operations/analysis-contract.md) |
| <a id="analysis-topic-strategy"></a>`analysis.topic_strategy` | `required` / `analysis` | Build the title-spine topic strategy before query planning: collect invited/keynote, main oral, session, workshop, and segmented workshop talk titles; ask OpenAI, Anthropic, and Gemini for independent 12-topic recommendations; merge to exactly 12 final topics. | `.venv/bin/python scripts/build_topic_strategy.py --workspace conferences/<slug> --run-id <run-id> --mode live --provider-cost-approved --model openai:gpt-5.6-luna --model anthropic:claude-opus-4-8 --model gemini:gemini-3.1-pro-preview --merge-model anthropic:claude-opus-4-8 --summary`. Mock mode is valid only for planning/test validation. Live receipts must include `storage/analysis/topic-strategy/<run-id>/title_spine.json`, title-spine sha256, `analysis/topic_strategy.json`, three recommendation outputs, exactly 12 `final_topics`, and `provider_receipt_paths`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-exhaustive-topic-extraction"></a>`analysis.exhaustive_topic_extraction` | `required` / `analysis` | Exhaustively extract topic-relevant evidence across indexed papers, transcripts, slides/OCR, and artifacts before public narrative synthesis. Analysis completion must not skip this stage or rely on topic strategy alone. | `conference_pipeline.iclr2025_live.run_exhaustive_topic_extraction` or the current conference production runner's registered exhaustive-extraction entrypoint. Receipts must include `status: succeeded` and output families under `storage/extractions/<run-id>/by-topic/*.json` and `storage/extractions/<run-id>/by-artifact/*.json`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-llm-full-read-extraction"></a>`analysis.llm_full_read_extraction` | `required` / `analysis` | Run the LLM-assisted full-read lane after heuristic extraction and before final analysis. The lane reads all transcript windows first, then paper abstracts, preserves workshop/talk/timestamp/source/media metadata, ignores non-substantive noise, and keeps heuristic results as comparison evidence. Pilot, limited, transcript-only, and heuristic-only outputs are rejected production evidence. | `.venv/bin/python scripts/run_llm_full_read_extraction.py --workspace conferences/<slug> --run-id <run-id> --scope all --resume --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`. Production receipts must use `scope: all`, no `limit`, `status: succeeded`, `checks.completion_state: complete`, `completed_unit_count == expected_unit_count`, `failed_count: 0`, and output families under `storage/extractions/<run-id>/llm-full-read/by-unit/*.json` and `storage/extractions/<run-id>/llm-full-read/by-topic/*.json`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-exhaustive-topic-summarization"></a>`analysis.exhaustive_topic_summarization` | `experimental` / `analysis` | Build provisional per-topic/category synthesis packs from completed LLM full-read evidence before promoting that evidence into formal Stage 7 source-pack construction. This is an inspection lane, not yet a required production stage. | `.venv/bin/python scripts/summarize_llm_full_read_topics.py --workspace conferences/<slug> --run-id <run-id> --mode live --provider openai --model gpt-5.6-sol --provider-cost-approved --max-input-tokens 400000 --summary`. Dry-run mode writes only `storage/analysis/exhaustive-summaries/<run-id>/summary_plan.json` and `07A-exhaustive-topic-summaries.json`; live mode also writes `analysis/exhaustive_summaries/*.json`, Markdown summaries, an index, and provider receipts. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-deep-dive-question-selection"></a>`analysis.deep_dive_question_selection` | `required` / `analysis` | Compress exhaustive transcript/paper summary bullets into a bounded set of slide-usable deep-research questions. This prevents one GPT-Researcher run per raw key point or tension. | `.venv/bin/python scripts/select_deep_dive_questions.py --workspace conferences/<slug> --run-id <run-id> --mode live --provider anthropic --model claude-opus-4-8 --provider-cost-approved --min-questions-per-topic 4 --max-questions-per-topic 6 --summary`. Receipts must include `07B-deep-dive-question-selection.json`, `selected_deep_dives.json`, Claude Opus 4.8 provider receipts, a per-topic cap, keywords, source categories, and question counts. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-gpt-researcher-stage"></a>`analysis.gpt_researcher_stage` | `required` / `analysis` | Run `query-planning-gpt-researcher` with default local Qwen3.6 35B-A3B, local BGE-M3 embeddings, BM25+vector hybrid retrieval, resolved GPT-Researcher parameters, `WEB_SEARCH_ENABLED=false`, and provider/model provenance. This owns the global no-topic-id evidence/draft layer; final public narrative synthesis remains governed separately by `analysis.deep_narrative_synthesis` unless local validation passes. | `conference_pipeline.iclr2025_live.run_query_planning_gpt_researcher`; `conference_pipeline.research.researcher`; `conference_pipeline.research.query_planner`; `conference_pipeline.research.config`; `conference_pipeline.research.walker`. Plans and receipts must record `workflow.research.gpt_researcher`, `retrieval_mode: hybrid`, no topic-id filter, `MAX_ITERATIONS=5`, `DEEP_RESEARCH_BREADTH=4`, `DEEP_RESEARCH_DEPTH=3`, `MAX_SUBTOPICS=8`, `RETRIEVAL_K=18`, `PAPER_MATCHING_RETRIEVAL_K=18`, `CANDIDATE_MULTIPLIER=3`, `RRF_K=60`, `EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit`, `GPTR_LOCAL_LLM_SMART_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`, and the actual synthesis provider/model. | [Pipeline workflow](../workflows/pipeline-workflow.md); [GPT Researcher quick start](../integrations/gpt-researcher/quickstart.md) |
| <a id="analysis-gpt-researcher-heuristic-prefilter"></a>`analysis.gpt_researcher_heuristic_prefilter` | `required` / `analysis` | Run a second deep-dive lane that builds an explicit keyword-prefilter evidence pack from topic-agnostic vector chunk sidecars, then feeds that pack to the same GPT-Researcher/local-researcher synthesis path. This lane is a comparison and recall tool; it must not replace the global corpus lane or filter by derived topic ids. | `conference_pipeline.research.heuristic_prefilter`; `src/conference_pipeline/research/heuristic_prefilter.py`; `scripts/run_heuristic_prefiltered_research.py`. Plans and receipts must record `workflow.research.heuristic_prefiltered_gpt_researcher`, `heuristic_keyword_prefilter`, `keyword_seed_source`, `prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`, `stores_searched`, `source_family_scope`, `retrieval_scope`, and `invalid_citation_ids`. Rejected evidence includes topic-id filtered deep dives, unrecorded keyword lists, citation-invalid heuristic reports, and source-family leakage. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Analysis contract](../operations/analysis-contract.md) |
| <a id="analysis-dual-deep-dive-research-lanes"></a>`analysis.dual_deep_dive_research_lanes` | `required` / `analysis` | Execute the curated deep-dive questions through both the global no-topic-id GPT-Researcher lane and the heuristic keyword-prefilter lane, using local Qwen3.6 35B-A3B synthesis. | `.venv/bin/python scripts/run_dual_deep_dive_research.py --workspace conferences/<slug> --run-id <run-id> --mode live --local-model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary`. Receipts must include `07C-dual-deep-dive-research-lanes.json`, selected-question binding, global evidence JSON, heuristic prefilter packs, Markdown outputs, local provider receipt paths, and no topic-id filter in the global lane. Long production runs should use `--resume` so completed lane outputs are reused after interruptions. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Analysis contract](../operations/analysis-contract.md) |
| <a id="analysis-topic-reconciliation"></a>`analysis.topic_reconciliation` | `required` / `analysis` | Reconcile each topic's exhaustive summaries, current narrative source pack, and dual GPT-Researcher deep dives before final public narrative synthesis. This is the first place the parallel evidence layers are merged; slide synthesis must consume reconciled analysis rather than doing research reconciliation itself. | `.venv/bin/python scripts/reconcile_topic_analysis.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary`. Receipts must include `07D-topic-reconciliation.json`, per-topic token estimates under the configured context ceiling, `analysis/topic_reconciliations/*.md`, JSON sidecars, local provider receipts, and enriched `analysis/narrative_source_packs/*.json` entries containing `topic_reconciliation`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-topic-reconciliation-reference-workflow"></a>`analysis.topic_reconciliation_reference_workflow` | `required` / `analysis` | Package the production reconciliation sequence so future conferences cannot skip reference validation or repair ordering. This is the normal entrypoint after exhaustive summaries and dual deep-dive lanes: 07D reconciliation, initial 07E deterministic validation, deterministic safety fixes, bounded 07G candidate resolution when needed, 07F broad repair only after unresolved candidate blockers, and final 07E validation. | `.venv/bin/python scripts/run_topic_reconciliation_reference_workflow.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary`. Receipts must include `07H-topic-reconciliation-reference-workflow.json`, the 07D/07E/07G/07F step evidence that actually ran, `final_validation_status: succeeded`, positive final ledger and citation counts, and zero `final_invalid_reference_count`, `final_unknown_reference_count`, `final_title_mismatch_count`, and `final_bare_reference_warning_count`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-topic-reference-validation"></a>`analysis.topic_reference_validation` | `required` / `analysis` | Build deterministic per-topic reference ledgers and validate reconciled local-model briefs before final public narrative or slide synthesis. This catches unknown ids, bare ids that should be canonical, and title/id conflations such as attaching one paper title to another paper id. Empty required ledgers are blockers rather than successful zero-citation receipts. | `.venv/bin/python scripts/validate_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --summary`. Receipts must include `07E-topic-reference-validation.json`, `analysis/topic_reference_ledgers/*.json`, `analysis/topic_reference_validation/*.json`, enriched `topic_reconciliation.reference_validation` source-pack fields, positive `ledger_entry_count` and `citation_count`, and zero `invalid_reference_count`, `unknown_reference_count`, `title_mismatch_count`, and `bare_reference_warning_count`. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-topic-reference-candidate-resolution"></a>`analysis.topic_reference_candidate_resolution` | `required first-line remediation` / `analysis` | When reference validation blocks a local-model reconciliation, deterministically assemble bounded candidate rows from the ledger, source-pack evidence, exhaustive summaries, and validator errors. Local Qwen may choose only among packet candidates; empty, low-confidence, or unresolved ambiguous packets remain blockers. The validator remains deterministic and final: repaired markdown may be applied only after `final_validation` reports zero invalid references. | `.venv/bin/python scripts/resolve_topic_reference_candidates.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --topic <topic_id> --apply --summary`, followed by `.venv/bin/python scripts/validate_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --summary`. Receipts must include `07G-topic-reference-candidate-resolution.json`, `analysis/topic_reference_candidate_resolution/*__packets.json`, decision sidecars, repaired markdown candidates, `topic_reconciliation.reference_candidate_resolution`, Qwen decisions limited to packet ids, and zero `final_validation.invalid_reference_count` before any repaired brief replaces the reconciliation artifact. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="analysis-topic-reference-repair"></a>`analysis.topic_reference_repair` | `fallback remediation` / `analysis` | When candidate resolution records unresolved blockers, run the broader repair owner that lets local Qwen inspect the original reconciliation evidence bundle, focused candidate ledger rows, and validator errors, then propose a repaired brief. The validator remains deterministic and final: repaired markdown may be applied only after `repaired_validation` reports zero invalid references. | `.venv/bin/python scripts/repair_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --topic <topic_id> --apply --summary`, followed by `.venv/bin/python scripts/validate_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --summary`. Receipts must include `07F-topic-reference-repair.json`, `analysis/topic_reference_repairs/*.json`, `analysis/topic_reference_repairs/*.md`, enriched `topic_reconciliation.reference_repair`, and zero `repaired_invalid_reference_count` before any repaired brief replaces the reconciliation artifact. | [Analysis contract](../operations/analysis-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="paper-matching-artifact-keypoints"></a>`paper_matching.artifact_keypoints` | `required` / `analysis` | Match artifact-derived keypoints against the paper vector store and write `matched_papers.json`. | `scripts/match_papers_from_artifact_keypoints.py`. Use registered refs in plan text, receipts, `command_metadata.owner_ref`, or manifest owner refs. | [Analysis contract](../operations/analysis-contract.md) |

### Production Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="indexing-token-bounded-chunking"></a>`indexing.token_bounded_chunking` | `required` / `indexing` | Build source-aware corpus chunks under the governed token budget policy for `PG-CORPUS-TOKEN-INDEXING`. Papers use token-bounded 1,000-token chunks with 100-token overlap after metadata prefixes; dense transcript 120-second windows are split into overlapping token windows; slide/OCR records remain locator-aware. Word-count-only sidecars, first-240-character vectorization, and speech transcript receipt JSON are rejected evidence. Base corpus chunks are topic-agnostic and carry stable source metadata instead. | `src/conference_pipeline/indexing/corpus_vector_store.py`, `src/conference_pipeline/indexing/token_policy.py`, and `src/conference_pipeline/indexing/chunking.py`. Stage receipts and index digests must record `token_budget_policy_digest`, source metadata coverage, indexed text kind counts, and chunk-length stats before query planning. | [Pipeline workflow](../workflows/pipeline-workflow.md#phase-1-enhanced-indexing-required); [Source of truth](../source-of-truth.md) |
| <a id="indexing-local-embedding-materialization-policy"></a>`indexing.local_embedding_materialization_policy` | `required` / `indexing` | Materialize every indexable token-bounded chunk into durable embedding sidecar shards, then bulk-load persisted Chroma vector stores from those sidecars. Readiness-only probes, partial Chroma rows, chunk-sidecar-only runs, conference-local embedding reimplementations, and recomputing embeddings only because an insert batch changed cannot close Stage 6. | `src/conference_pipeline/indexing/corpus_vector_store.py`, `src/conference_pipeline/indexing/embeddings.py`, and `src/conference_pipeline/indexing/mlx_embeddings.py`. Stage receipts and index digests must record `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, `embedding_sidecar_manifest_path`, `embedding_sidecar_manifest_sha256`, `embedding_sidecar_format`, `embedding_sidecar_shard_count`, `chroma_insert_batch_size`, `vector_embeddings_written: true`, and embedded chunk counts matching sidecar chunk counts. Batch-size tuning is operational receipt evidence, not part of `embedding_policy_digest`. Sidecars are artifact-keyed hash buckets (`embedding_sidecar_format: npz-buckets-v2`, `embedding_sidecar_bucket_count`) indexed by the per-store `chunk_manifest.json`; re-embedding a chunk whose embedding digest already exists in any bucket, legacy shard, or sibling run, or re-chunking artifacts whose fingerprint and text are unchanged, is rejected evidence of a correct incremental run. Receipts record `index_update` and `embedding_update`. | [Pipeline workflow](../workflows/pipeline-workflow.md#phase-1-enhanced-indexing-required); [Source of truth](../source-of-truth.md) |
| <a id="indexing-transcript-evidence-lanes"></a>`indexing.transcript_evidence_lanes` | `required` / `indexing` | Bind transcripts recovered after the base caption inventory (late YouTube captions, governed ASR fallback, hand-recovered keynotes) as appendable, digest-bound evidence lanes so the corpus-inventory-index stage admits them without editing the base inventory or its contract. Each lane is a `lane_receipt.json` listing every transcript with sha256, source, event URLs, provenance, and explicit coverage gaps; `storage/transcription-ocr/evidence_lanes.json` orders lanes by precedence and pins each receipt digest. The base inventory wins, lanes follow registry order, and any digest/path failure blocks Stage 6. A recovery script with no registered lane is not done. | `conference_pipeline.indexing.transcript_evidence_lanes` (`build_lane_receipt`, `register_lane`, `load_admitted_lane_records`, `write_plain_transcript`). Conference binder example: `.venv/bin/python scripts/build_cvpr2026_transcript_evidence_lanes.py --dry-run` then without `--dry-run`, followed by `scripts/run_cvpr2026_production.py --stage corpus-inventory-index --resume`. Proof: `corpus_inventory.json` `upstream.transcript_evidence_lanes[]` and `stage-06-corpus-inventory-index.json` `checks.transcript_evidence_lanes[]` with `admitted_count` per lane. | [Pipeline capabilities](pipeline-capabilities.md#transcript-evidence-lanes-post-harvest-transcript-admission); [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="transcription-ocr-local-media-processing"></a>`transcription_ocr.local_media_processing` | `required` / `transcription_ocr` | Run provider-free local transcription/OCR preprocessing over receipt-backed downloaded artifacts. Extracts local PDF text into deterministic text outputs, inventories image OCR and speech lanes separately, preserves page counts on resume, records upstream digests, and emits explicit blockers instead of substituting paper text for media evidence. | `from conference_pipeline.transcription_ocr import run_local_transcription_ocr`. Per-conference runners should validate their upstream stage receipt, call this module, then write stage receipts that include `required_script_ids: ["transcription_ocr.local_media_processing"]`, `script_registry_sha256`, and the returned lane inventory digest. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Source of truth](../source-of-truth.md) |
| <a id="transcription-ocr-workshop-talk-boundary-discovery"></a>`transcription_ocr.workshop_talk_boundary_discovery` | `required` / `transcription_ocr` | Promote the NeurIPS 2025 Gemini workshop analyzer into a governed reusable owner: Gemini 2.5 Pro analyzes long workshop transcripts plus slide timestamps, identifies talk boundaries/speakers/topics, writes `*_talks.json`, and produces 5-minute talk-context batches. This is provider-backed and requires explicit cost approval plus provider-call receipts. | `scripts/discover_workshop_talks.py --workspace conferences/<slug> --input-transcript-json storage/transcription-ocr/speech-transcripts/<workshop>.json --slides-json storage/download-corpus/workshop-slides/<workshop>.json --workshop-id <id> --workshop-title <title> --output-talks-json storage/transcription-ocr/workshop-talks/<id>_talks.json --output-batches-json storage/transcription-ocr/workshop-talks/<id>_batches.json --receipt-path storage/transcription-ocr/workshop-talks/<id>_discovery_receipt.json --provider-cost-approved --summary`. Receipts must record `model_governance_task_id=transcription_ocr.workshop_talk_boundary_discovery`, `cost_task_id=workshop_talk_boundary_discovery`, `provider_receipt_paths`, and estimated cost. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Source of truth](../source-of-truth.md) |
| <a id="transcription-ocr-workshop-talk-segmentation"></a>`transcription_ocr.workshop_talk_segmentation` | `required` / `transcription_ocr` | Normalize workshop `talks.json` boundaries and annotate transcript segments with `workshop_id`, `workshop_title`, `talk_id`, `talk_title`, and `talk_speaker` without making provider calls. If no talks JSON or governed provider path is available, it blocks with `talk_segmentation_provider_not_configured`; audio-only recovery is rejected evidence for talk-aware transcripts. | `scripts/segment_workshop_talks.py --workspace conferences/<slug> --input-transcript-json storage/transcription-ocr/speech-transcripts/<talk>.json --talks-json storage/transcription-ocr/workshop-talks/talks.json --output-path storage/transcription-ocr/workshop-talks/annotated.json --receipt-path storage/transcription-ocr/workshop-talks/receipt.json --summary`. | [Pipeline workflow](../workflows/pipeline-workflow.md); [Source of truth](../source-of-truth.md) |
| <a id="podcast-focused-script"></a>`podcast.focused_script` | `required` / `production` | Generate focused podcast scripts from deep analysis reports and fact ranks. | `scripts/generate_focused_podcast.py`. Use registered refs in plan text, receipts, `command_metadata.owner_ref`, or manifest owner refs. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="podcast-render-audio"></a>`podcast.render_audio` | `required` / `production` | Render an approved podcast `script.json` into segment-level MP3 files, a merged `podcast.mp3`, an `audio_manifest.json`, and provider receipts for TTS calls. Script-only packages are not sufficient Phase 9 evidence. | `scripts/render_podcast_audio.py --script-json audio/<id>/focused_podcast/script.json --output-dir audio/<id>/focused_podcast --model gpt-4o-mini-tts`. Receipts must include `podcast_script_status: pass`, `podcast_audio_status: pass`, `podcast_audio_sha256`, TTS provider receipts, and `audio_ready_package.json`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="podcast-notebooklm-audio-overview"></a>`podcast.notebooklm_audio_overview` | `required` / `production` | Generate one Google NotebookLM Audio Overview per topic deck and per executive deck through `notebooklm-py`: stage the deck's analysis spine (topic report, talk synthesis, deep dives, a spine rendered from the approved blueprint) plus the blueprint-cited paper texts and de-duplicated talk transcripts into `audio/<id>/notebooklm_podcast/sources/`, add an `Episode brief` run sheet (creator credit from the blueprint author, a receipt-derived how-the-material-was-made disclaimer, segment order, named works, and the workshops and sessions the small-model full read ranked for the topic; for executive decks every topic in arc order with explicit transitions) plus a `Program map` source, steer generation with a deterministic prompt that forbids false attendance claims, and bind notebook id, every source sha256, the prompt sha256, and the audio sha256 in `notebook_manifest.json`. Every live episode then passes a local listening review (MLX Whisper, governance task `podcast.notebooklm_review_transcription`) that writes `transcript.txt` and `review.json` and blocks the deck unless the opening credits the creator and discloses the method, no attendance is claimed, topic episodes name their workshops, and executive episodes cover every topic after the opening; `--accept-review-findings` records operator acceptance, `--no-review` records a waiver, and `--force-regenerate` requests a fresh episode. Fails closed on auth (`notebooklm_auth_failed`), halts resumably on quota (`notebooklm_quota_exhausted`), and blocks a deck whose cited work has no on-disk text. Dry-run and `--fake-engine` runs are rejected completion evidence. | `scripts/generate_notebooklm_podcast.py --workspace conferences/<slug> --live --ack-live-notebooklm --format deep-dive --length long` (or the CVPR runner `--stage notebooklm-podcast`). Auth: `notebooklm login --browser-cookies chrome`; never copy `storage_state.json` into artifacts. Receipts must include `notebooklm_podcast_status: pass`, `notebooklm_podcast_aggregate_sha256`, `deck_success_count == deck_scope_full_count`, `deck_blocker_count: 0`, `storage_state_present`, `engine_version`, and `provider_calls_made`. | [Pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="podcast-notebooklm-delivery"></a>`podcast.notebooklm_delivery` | `required` / `backup` | Copy the per-deck NotebookLM podcasts and show notes into a curated, flat, shareable folder on the mounted Google Drive (numbered by topic, executive decks first) with an `index.md`, refusing unless `stage-10-notebooklm-podcast.json` succeeded; never deletes. The raw workspace mirror still goes through `backup.sync_to_gdrive` (family `podcasts`). | `scripts/deliver_notebooklm_podcasts.py --workspace conferences/<slug> --run-id <run-id> --drive-root '<mounted-drive-root>/<Conference> Podcasts' --dry-run` then without `--dry-run`. Receipt `storage/workflow-runs/<run-id>/backup/notebooklm-podcast-delivery-receipt.json` records every copied file's sha256 and `delivered_count == expected_count`. | [Private backup](../operations/private-backup.md); [Pipeline workflow](../workflows/pipeline-workflow.md) |

### Presentation Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="presentation-add-title-slide"></a>`presentation.add_title_slide` | `legacy-debug` / `presentation` | Retired Markdown helper for pre-Phase-14 decks. Production title/author metadata now lives in `render_spec.json` and the delivery contracts. | `scripts/presentations/add_title_slide_to_slides_md.py` may be used only for legacy/debug Markdown exports; it cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-build-notebooklm-video"></a>`presentation.build_notebooklm_video` | `required` / `presentation` | Build the NotebookLM-style two-host presenter video output. | `scripts/presentations/build_notebooklm_style_presenter_video.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-build-presenter-video"></a>`presentation.build_presenter_video` | `required` / `presentation` | Build the primary presenter video from slides, mapping, and presenter script. | `scripts/presentations/build_presenter_video.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-consolidate-slides"></a>`presentation.consolidate_slides` | `legacy-debug` / `presentation` | Retired Markdown consolidation helper. Production deck shape is compiled into `presentation_deck.json` and `render_spec.json`. | `scripts/presentations/consolidate_slides.py` is legacy/debug-only and cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-editorial-gate"></a>`presentation.editorial_gate` | `required` / `presentation` | Pre-TTS editorial gate for concise, non-redundant, visually integrated decks. Blocks topic decks without `editorial_storyboard.json`, estimates above 40 minutes for either single-presenter or NotebookLM styles, excessive figure-detail/zoom density, obvious redundant claim clusters, generic repeated SVG patterns, and important FigureSpec entities missing from narration. | `scripts/presentations/presentation_editorial_gate.py --deck-dir decks/<id> --deck-kind <topic\|executive_short\|executive_long> --report-out decks/<id>/presentation_editorial_gate_report.json`. The gate runs after reviewable `video_build/script.json` and `video_build_notebooklm/script.json` exist and before any TTS, PDF/video render, release, or proof acceptance. On the compiled deck route (no legacy slide markdown, no storyboard) the shared production chain runs it as `--narration-only --advisory` with `--render-spec decks/<id>/render_spec.json`, `--presenter-script-md`/`--notebooklm-script-md` pointing at the ENGAGING narration and `--asset-inventory decks/<id>/asset_inventory.json`: only the narration rules apply (duration estimate against the 40-minute budget and 38-minute target, zoom-beat density, figure entities named in narration), the report is recorded on the deck record, and it never blocks. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-phase13-canary-objective-evidence"></a>`presentation.phase13_canary_objective_evidence` | `required` / `presentation` | Build objective canary proof before operator review. Probes both single-presenter and NotebookLM MP4 durations, audio durations, slide-concat receipt durations, first/middle/tail frame luma, and a contact sheet so black-tail or receipt-drift failures are visible before bulk rollout. | `scripts/presentations/build_phase13_canary_objective_evidence.py --deck-dir decks/<id> --validation-summary storage/workflow-runs/<run>/summary.json --frames-dir <phase>/canary_frames --report-out <phase>/13-CANARY-OBJECTIVE-EVIDENCE.json --markdown-out <phase>/13-CANARY-OBJECTIVE-EVIDENCE.md`. The report fails if videos exceed 2400 seconds, MP4/audio/slide receipts diverge beyond tolerance, required artifacts are missing, or tail frames are black. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-phase13-canary-review-packet"></a>`presentation.phase13_canary_review_packet` | `required` / `presentation` | Summarize the one-topic Phase 13 canary for operator review before bulk rollout. | `scripts/presentations/build_phase13_canary_review_packet.py --deck-dir decks/<id> --validation-summary storage/workflow-runs/<run>/summary.json --objective-evidence-md <phase>/13-CANARY-OBJECTIVE-EVIDENCE.md --objective-evidence-report <phase>/13-CANARY-OBJECTIVE-EVIDENCE.json --report-out <phase>/13-CANARY-REVIEW-PACKET.json --markdown-out <phase>/13-CANARY-REVIEW-PACKET.md`. The packet must show required artifacts, MP4 duration probes, editorial gate status, strict-validation status, passing objective-evidence state, contact-sheet presence, and human-review receipt state. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-phase13-rollout"></a>`presentation.phase13_rollout` | `required` / `presentation` | Plan or execute guarded Phase 13 pre-TTS rollout steps across canary, topic, and executive decks. | `scripts/presentations/run_phase13_rollout.py --workspace conferences/<slug> --all --require-approved-canary --canary-review-packet <phase>/13-CANARY-REVIEW-PACKET.json --report-out <phase>/13-ROLLOUT-PLAN.json --markdown-out <phase>/13-ROLLOUT-PLAN.md`. The runner blocks non-canary decks until both approved `human_review_receipt.json` and approved canary packet with passing objective evidence are present, records provider env key presence without secret values, runs deck-level narration gates, writes `presenter_narration_series_review.json` and `narration_series_review.json` through `review_narration_series.py`, and uses `--stop-after-script-json` / `--stop-after-transform` so it does not synthesize TTS or render MP4s. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-phase13-bulk-readiness"></a>`presentation.phase13_bulk_readiness` | `required` / `presentation` | Verify the post-approval handoff before executing bulk rollout. Confirms the canary human-review receipt is approved, objective evidence still passes, the regenerated canary packet is approved with no receipt evidence mismatch, and the guarded rollout plan is no longer blocked. | `scripts/presentations/verify_phase13_bulk_readiness.py --workspace conferences/<slug> --canary-review-packet <phase>/13-CANARY-REVIEW-PACKET.json --objective-evidence-report <phase>/13-CANARY-OBJECTIVE-EVIDENCE.json --rollout-plan <phase>/13-ROLLOUT-PLAN.json --report-out <phase>/13-BULK-READINESS.json --markdown-out <phase>/13-BULK-READINESS.md`. The report must be `ready_for_bulk_rollout` before running `run_phase13_rollout.py --execute` for non-canary decks. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-phase13-completion"></a>`presentation.phase13_completion` | `required` / `presentation` | Final Phase 13 acceptance verifier. Confirms bulk readiness is ready, guarded rollout completed, final strict validation succeeded, cross-series presenter/NotebookLM narration reviews passed, every deck has required HTML/PDF/video/script/editorial artifacts, and both video styles are under the 40-minute cap. | `scripts/presentations/verify_phase13_completion.py --workspace conferences/<slug> --phase-dir <phase> --final-validation-summary storage/workflow-runs/<run>/summary.json --report-out <phase>/13-COMPLETION-VERIFICATION.json --markdown-out <phase>/13-COMPLETION-VERIFICATION.md`. Phase 13 is not complete until this verifier reports `complete`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-enrich-hyperlinks"></a>`presentation.enrich_hyperlinks` | `legacy-debug` / `presentation` | Retired Markdown hyperlink helper. Production visible references are owned by `presentation.source_binding`. | `scripts/presentations/enrich_hyperlinks_hybrid.py` is legacy/debug-only; production must write `source_binding_report.json`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-enrich-slides"></a>`presentation.enrich_slides` | `legacy-debug` / `presentation` | Retired Markdown enrichment helper. Production source choices are owned by `presentation.source_binding`. | `scripts/presentations/enrich_slides_with_papers.py` is legacy/debug-only and cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-executive-overview-decks"></a>`presentation.executive_overview_decks` | `required` / `presentation` | Generate the conference-level short and long executive overview decks from final synthesis and analysis-manifest evidence. This is separate from `analysis/executive_summary.md`, which is only a source report. | `scripts/presentations/generate_executive_overview_decks.py --workspace conferences/<slug> --analysis-manifest analysis/analysis_manifest.json --executive-summary analysis/executive_summary.md --trend-report analysis/trend_report.md --short-id executive_summary_short --long-id executive_summary_long --source-link-map decks/executive_source_link_map.json --require-source-link-map --receipt-output decks/executive_overview_decks_receipt.json`. For production link-audit readiness, also pass `--presentation-link-map decks/executive_presentation_link_map.json --require-presentation-link-map` after topic presentation artifacts have public URLs. The receipt must list `executive_summary_short` and `executive_summary_long`; each deck must emit `presentation_frame.json`, `presentation_blueprint.json`, `presentation_deck.json`, `source_binding_report.json`, `visual_plan.json`, `asset_inventory.json`, `delivery_single_speaker.json`, `delivery_notebooklm.json`, and `render_spec.json`. Both decks must use topic-style bookends: slide 2 is `slide_kind: events`, capped at no more than `x` important events where `x` is the executive synthesis point count and carrying at least `x` key resources on the right; the final `slide_kind: resources` slide links all 12 topic presentations plus all 12 topic analysis reports. Render with `md_to_revealjs.py --render-spec` and export with `export_revealjs_pdf.py --render-spec`; retired numbered Markdown checkpoints are debug-only and cannot satisfy production proof. Both deck directories must be registered in `presentation_manifest.json`, rendered to PDF, and QC checked before presentation closeout or video rendering. Executive-summary presenter narration must be authored from synthesis prose, not assembled by repeating bullet-stitch templates such as `That connects directly to`, `The operational test is`, or `That sets up`. When Google Drive has not synced source markdown yet, the owner must write a blocked source-link contract receipt instead of falling back to local `../analysis/*.md` links. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-export-pdf"></a>`presentation.export_pdf` | `required` / `presentation` | Export the Reveal.js deck to PDF for shareable static review. | `scripts/presentations/export_revealjs_pdf.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-link-audit"></a>`presentation.link_audit` | `required` / `presentation` | Audit rendered presentation HTML and PDF links before release. Blocks local paths, file URIs, repo-local links, and source-link fallbacks that would escape into shared PDFs. | `scripts/presentations/audit_presentation_links.py --workspace conferences/<slug> --decks-dir decks --conference-slug <slug> --receipt-output decks/presentation_link_audit_receipt.json`. The receipt must pass after all HTML/PDF exports and before Stage 10 success. Optional live external checking can be run as an operator audit, but the deterministic contract always rejects local-path evidence. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-presenter-narration-fix"></a>`presentation.presenter_narration_fix` | `required` / `presentation` | Repair or synchronize single-presenter delivery after review blockers. Keeps `delivery_single_speaker.json`, optional debug markdown, and `video_build/script.json` synchronized and prevents old outline/stage-direction text, executive stitch templates, repeated welcome/closing formulas, closing grammar joins, and filler-word overuse from reaching TTS. | Run the fixer against the current delivery contract and rendered script JSON, then rerun `presentation.presenter_narration_review`; legacy markdown-only repair cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-presenter-narration-review"></a>`presentation.presenter_narration_review` | `required` / `presentation` | Review single-presenter delivery and TTS JSON before audio. Blocks metadata leaks, stage directions, missing SAY blocks, missing transitions, executive stitch-template repetition, closing grammar joins, and listener-quality tics such as tiny repeated executive transition pools. | Review `delivery_single_speaker.json` together with `video_build/script.json`; optional markdown exports are debug-only. Single-presenter TTS is blocked unless every deck has `approve_for_tts=true`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-narration-script-fix"></a>`presentation.narration_script_fix` | `required` / `presentation` | Repair generated NotebookLM delivery after review blockers. Keeps `delivery_notebooklm.json`, optional debug markdown, and `video_build_notebooklm/script.json` synchronized; accepts exact editorial replacements for substantive fixes and deterministic cleanup for repeated mechanical patterns, including conference-label thinning. | Run the fixer against the current NotebookLM delivery contract and transformed script JSON, then rerun `presentation.narration_script_review`; legacy markdown-only repair cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-narration-script-review"></a>`presentation.narration_script_review` | `required` / `presentation` | Review generated NotebookLM delivery before TTS for accuracy, cohesion, smooth transitions, pleasantness, non-mechanical dialogue, truncated fragments, excessive question hooks, and script/transition readiness. | Review `delivery_notebooklm.json` together with `video_build_notebooklm/script.json`; optional markdown exports are debug-only. TTS is blocked unless every deck has `approve_for_tts=true`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-narration-series-review"></a>`presentation.narration_series_review` | `required` / `presentation` | Review the full set of delivery contracts for cross-deck duplicated figure narration, shared chart patter, and repeated technical phrases that only become obvious to multi-video listeners. Supports both NotebookLM and single-presenter delivery contracts. | Run after every deck-level narration review/fix pass and before TTS. Failed series review blocks TTS unless an explicit operator decision accepts the remaining series-level warning. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-narration-series-fix"></a>`presentation.narration_series_fix` | `required` / `presentation` | Repair cross-deck repeated figure-detail phrasing after series review finds duplicated chart patter. Keeps human-readable markdown and TTS `script.json` synchronized across the deck set. | `scripts/presentations/fix_narration_series.py --decks-dir decks --report-out decks/narration_series_fix_report.json`. Rerun both `presentation.narration_script_review` and `presentation.narration_series_review` after every series fix. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="audit-conference-hardcoding"></a>`audit.conference_hardcoding` | `allowed` / `audit` | Fail-closed AST scan of `src/conference_pipeline/**` for hardcoded conference-specific literals (conference name/year, ICLR/ICML/NeurIPS domains, `/virtual/<year>/` paths). Exit 0 means no violations; a finding requires either routing the literal through `conference_identity`/`config.yaml` or an inline `# conference-literal-ok: <reason>` suppression. | `python scripts/audit_conference_hardcoding.py` (human report) or `--json` (machine-readable). Covers `src/conference_pipeline/**` only — `scripts/**` is covered by the milestone's manual sweep per the `conference-hardcoding-audit` skill, not by this scanner. | [conference-hardcoding-audit skill](../../.claude/skills/conference-hardcoding-audit/SKILL.md) |
| <a id="audit-model-registry-hardcoding"></a>`audit.model_registry_hardcoding` | `allowed` / `audit` | Fail-closed scan of `src/conference_pipeline/**` and `scripts/**` `.py` plus `.claude/skills/**` `.md` for hardcoded LLM model-name literals (`gpt-5.*`, `gpt-5-mini`/`nano`, `claude-(opus\|sonnet\|haiku)-<n>`, `gemini-<n>.<n>-(pro\|flash)`) outside `model_governance/registry.yaml`. Exit 0 means no violations; a finding requires either routing the literal through `registry.py`'s `default_model`/`default_model_for_provider` by task id, or an inline `# model-literal-ok: <reason>` suppression. | `python scripts/audit_model_registry_hardcoding.py` (human report) or `--json` (machine-readable). Run alongside, not instead of, `audit.conference_hardcoding` for any milestone touching model selection. | [model-registry-audit skill](../../.claude/skills/model-registry-audit/SKILL.md) |
| <a id="audit-corpus-source-metadata"></a>`audit.corpus_source_metadata` | `allowed` / `audit` | Fail-closed runtime scan of a conference workspace's real on-disk corpus artifacts (`storage/corpus/corpus_inventory.json`, the papers vector-store chunk JSONL, `analysis/topic_reference_ledgers/*.json`, `decks/*/speaker_author_work_ledger.json`) for per-source-family (`papers_main`, `papers_workshop`, `artifacts`) title/author coverage against justified `COVERAGE_FLOORS`. Exit 0 means every gated family meets its floor; a finding is `coverage_below_floor`, `ledgers_missing`, or `floor_override_unjustified` (an unjustified per-workspace floor override). The `artifacts` family is reported `structurally_authorless` and is never gated on authors. | `python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug>` (human report), `--json` (machine-readable), or `--write-receipt <dir>` to also emit `corpus_source_metadata_coverage.json`. This is a runtime scan of real artifacts, the same strength as `audit.conference_hardcoding`/`audit.model_registry_hardcoding` -- run it alongside them, not instead of them, before closing any conference milestone. | [corpus-metadata-audit skill](../../.claude/skills/corpus-metadata-audit/SKILL.md) |
| <a id="export-notebooklm-package"></a>`export.notebooklm_package` | `allowed` / `export` | Package conference materials for NotebookLM import as supplemental review/source material. This is an export helper, not a production completion owner. | `scripts/package_conference_for_notebook_lm_export.py --workspace conferences/<slug> ...`. Use after registered acquisition, analysis, and presentation receipts exist; do not substitute the package for stage-owner proof evidence. | [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-programmatic-figures"></a>`presentation.generate_programmatic_figures` | `required` / `presentation` | Generate source-grounded programmatic figures from structured deck/source contracts. Completion evidence is `visual_plan.json`, `asset_inventory.json`, `figure_specs.json`, optional governed model receipts, and generated assets. | Production must consume `presentation_deck.json` plus `source_binding_report.json`; Markdown input/output flags are legacy/debug-only and cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-review-programmatic-figures"></a>`presentation.review_programmatic_figures` | `required` / `presentation` | Review generated SVGs for semantic informativeness before render/video. Blocks generic templates, copied slide prose, density fallback, placeholder chart ticks, and missing required FigureSpec entities. | `scripts/presentations/review_programmatic_figures.py --workspace conferences/<slug> --presentation-id <id> --presentation-type <type> --topic <topic> --dir decks/<id>/assets/generated --asset-inventory decks/<id>/asset_inventory.json --report-output decks/<id>/assets/generated/semantic_review.json --markdown-output decks/<id>/assets/generated/semantic_review.md`. The JSON report must pass before Reveal.js render and visual QC. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-editorial-blueprint"></a>`presentation.generate_editorial_blueprint` | `required` / `presentation` | Generate the source-grounded, audience-framed editorial Presentation Blueprint before deck contracts. | `scripts/presentations/generate_editorial_blueprint.py --workspace conferences/<slug> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --analysis-md analysis/<id>.md --speaker-author-work-ledger decks/<id>/speaker_author_work_ledger.json --exhaustive-papers analysis/exhaustive_summaries/<id>__papers.json --presentation-frame decks/<id>/presentation_frame.json --final-report-url <url> --output-json decks/<id>/presentation_blueprint.json --output-md decks/<id>/presentation_blueprint.md --receipt-path decks/<id>/presentation_blueprint_receipt.json --provider-cost-approved`. `presentation_frame.json` must separate retrieval title from audience title, include `field_thesis`, at least three broad `general_questions`, and `opening_anchor_policy`; the generated blueprint must record passing `presentation_frame_validation`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-presentation-blueprint"></a>`presentation.generate_presentation_blueprint` | `legacy-debug` / `presentation` | Retired pre-frame blueprint helper for old receipts and debugging only. Production topic builds must use `presentation.generate_editorial_blueprint` with `presentation_frame.json`; this owner cannot satisfy Phase 11 completion evidence. | `scripts/presentations/generate_presentation_blueprint.py` may be used only to inspect historical source-ledger/live blueprint artifacts. New production plans must not cite it as a completion owner. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-review-presentation-blueprint"></a>`presentation.review_presentation_blueprint` | `required` / `presentation` | Review and approve the Presentation Blueprint before `deck_contract_generation`. | `scripts/presentations/review_presentation_blueprint.py --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output-dir decks/<id>`. Production approval requires `approved_for_deck_contract_generation`, current `blueprint_sha256`, and deterministic, LLM-judge, and human evidence gates. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-repair-presentation-blueprint"></a>`presentation.repair_presentation_blueprint` | `required` / `presentation` | Repair blueprint review blockers while preserving versioned review evidence. | `scripts/presentations/repair_presentation_blueprint.py --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_blueprint.repair.json`. Rerun `presentation.review_presentation_blueprint` after every repair. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-source-corpus-inventory"></a>`presentation.source_corpus_inventory` | `required` / `presentation` | Purpose: Build the hash-bound source corpus inventory required before presentation blueprint, deck contract, source-binding, and canary acceptance. | Command: `scripts/presentations/build_presentation_source_corpus_inventory.py --workspace conferences/<slug> --topic-id <id> --deck-dir decks/<id> --source-input <path> --current-artifact <path> --output-json decks/<id>/source_corpus_inventory.json --receipt-path decks/<id>/source_corpus_inventory_receipt.json`. Outputs: `source_corpus_inventory.json` and `source_corpus_inventory_receipt.json`. Diagnostics/recovery: use the registry diagnostic profile to rerun after source-artifact changes and keep the inventory digest current. Rejected evidence: scaffold outputs, old numbered Markdown files, generic source bindings, synthetic fixtures, and receipt-only edits. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-speaker-author-work-ledger"></a>`presentation.speaker_author_work_ledger` | `required` / `presentation` | Purpose: Build the speaker/author/work ledger that resolves visible source eligibility before blueprinting, deck/source binding, and canary acceptance. | Command: `scripts/presentations/build_speaker_author_work_ledger.py --workspace conferences/<slug> --inventory-json decks/<id>/source_corpus_inventory.json --output-json decks/<id>/speaker_author_work_ledger.json --receipt-path decks/<id>/speaker_author_work_ledger_receipt.json`. Outputs: `speaker_author_work_ledger.json` and `speaker_author_work_ledger_receipt.json`. Diagnostics/recovery: use the registry diagnostic profile to repair missing speaker, author, work, title, or contribution eligibility before advancing. Rejected evidence: scaffold outputs, old numbered Markdown files, generic source bindings, synthetic fixtures, and receipt-only edits. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-repair-presentation-sources"></a>`presentation.repair_presentation_sources` | `required` / `presentation` | Purpose: Record source repair findings for unresolved, blocked, or corrected source candidates before blueprint/deck/source-binding and canary acceptance. | Command: `scripts/presentations/repair_presentation_sources.py --workspace conferences/<slug> --inventory-json decks/<id>/source_corpus_inventory.json --ledger-json decks/<id>/speaker_author_work_ledger.json --output-json decks/<id>/source_repair_findings.json --receipt-path decks/<id>/source_repair_receipt.json`. Outputs: `source_repair_findings.json` and `source_repair_receipt.json`. Diagnostics/recovery: use the registry diagnostic profile to surface unresolved candidates, rerun repairs, and preserve blockers instead of silently dropping sources. Rejected evidence: scaffold outputs, old numbered Markdown files, generic source bindings, synthetic fixtures, and receipt-only edits. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-research-cluster-plan"></a>`presentation.research_cluster_plan` | `required` / `presentation` | Purpose: Build the research cluster plan that groups eligible primary and supporting sources before blueprinting, deck/source binding, and canary acceptance. | Command: `scripts/presentations/build_research_cluster_plan.py --workspace conferences/<slug> --inventory-json decks/<id>/source_corpus_inventory.json --ledger-json decks/<id>/speaker_author_work_ledger.json --repair-findings-json decks/<id>/source_repair_findings.json --output-json decks/<id>/research_cluster_plan.json --markdown-output decks/<id>/research_cluster_plan.md --receipt-path decks/<id>/research_cluster_plan_receipt.json`. Outputs: `research_cluster_plan.json`, `research_cluster_plan.md`, and `research_cluster_plan_receipt.json`. Diagnostics/recovery: use the registry diagnostic profile to rebuild clusters when source eligibility or repair findings change. Rejected evidence: scaffold outputs, old numbered Markdown files, generic source bindings, synthetic fixtures, and receipt-only edits. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-artifact-truth-validation"></a>`presentation.artifact_truth_validation` | `required` / `presentation` | Purpose: Run artifact-truth validation before canary acceptance, canary review packet, bulk readiness, release, or proof acceptance. | Command: `scripts/presentations/validate_artifact_truth.py`. Outputs: `artifact_truth_report.json`; the report blocks when HTML, PDF, slide-frame, audio, video, source, artifact truth, or receipt evidence is unverifiable. Diagnostics/recovery: use the registry diagnostic profile to repair missing rendered assets, stale receipts, bad media, or source-evidence drift before retrying. Rejected evidence: scaffold outputs, old numbered Markdown files, generic source bindings, blank PDFs, flat/silent videos, synthetic fixtures, and receipt-only edits. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-deck-contract"></a>`presentation.generate_deck_contract` | `required` / `presentation` | Generate the structured deck contract after blueprint approval. | `scripts/presentations/generate_deck_contract.py --workspace conferences/<slug> --presentation-id <id> --presentation-type <type> --topic <topic> --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_deck.json --receipt-output decks/<id>/presentation_deck_receipt.json`. Outputs `presentation_deck.json` and `presentation_deck_receipt.json`; old numbered Markdown is not production evidence. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-bind-presentation-sources"></a>`presentation.bind_presentation_sources` | `required` / `presentation` | Bind every deck claim, visual role, and cited paper to source evidence. | `scripts/presentations/bind_presentation_sources.py --workspace conferences/<slug> --presentation-id <id> --presentation-deck decks/<id>/presentation_deck.json --source-pack decks/<id>/source_pack_receipt.json --ranked-papers decks/<id>/ranked_papers.json --output decks/<id>/source_binding_report.json`. Production render/export remains blocked until `source_binding_report.json` passes. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-render-spec"></a>`presentation.generate_render_spec` | `required` / `presentation` | Generate the render/export contract consumed by HTML, PDF, and slide mapping owners. | `scripts/presentations/generate_render_spec.py --workspace conferences/<slug> --presentation-id <id> --presentation-deck decks/<id>/presentation_deck.json --source-binding-report decks/<id>/source_binding_report.json --visual-plan decks/<id>/visual_plan.json --output decks/<id>/render_spec.json`. Outputs `render_spec.json` as the render source of truth. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-md-to-pptx"></a>`presentation.md_to_pptx` | `required` / `presentation` | Export the structured render contract to a PPTX-compatible delivery artifact when requested. | `scripts/presentations/md_to_pptx.py --workspace conferences/<slug> --presentation-id <id> --render-spec decks/<id>/render_spec.json --output decks/<id>/presentation.pptx`. PPTX export is an auxiliary render/export artifact and does not replace `render_spec.json`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-video-sync-validation"></a>`presentation.video_sync_validation` | `required` / `presentation` | Validate audio/video duration and slide-sync receipts for both presenter and NotebookLM videos. | `scripts/presentations/validate_video_sync.py --workspace conferences/<slug> --presentation-id <id> --delivery-contract decks/<id>/delivery_single_speaker.json --video decks/<id>/presentation_video.mp4 --receipt-output decks/<id>/video_sync_validation_single_speaker.json`, and rerun for `delivery_notebooklm.json` to write `video_sync_validation_notebooklm.json`. Both receipts must pass before release, human review, or Phase 13 evidence. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-generate-slides-agent"></a>`presentation.generate_slides_agent` | `legacy-debug` / `presentation` | Retired first-draft Markdown generator. Production uses `presentation.generate_deck_contract`. | `scripts/presentations/generate_slides_agent.py` may write debug Markdown only; `1_draft_slides.md` is never production evidence. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-inspect-rendered-slides"></a>`presentation.inspect_rendered_slides` | `required` / `presentation` | Render and screenshot the deck for deterministic visual-QC inputs. | `scripts/presentations/inspect_rendered_slides.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-inspect-rendered-slides-llm"></a>`presentation.inspect_rendered_slides_llm` | `required` / `presentation` | Run governed vision-LLM review over rendered slide screenshots. | `scripts/presentations/inspect_rendered_slides_llm.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-md-to-revealjs"></a>`presentation.md_to_revealjs` | `required` / `presentation` | Render `render_spec.json` to branded Reveal.js HTML and slide mapping. | `scripts/presentations/md_to_revealjs.py --render-spec decks/<id>/render_spec.json --output-html decks/<id>/presentation.html --slide-mapping decks/<id>/presentation_slide_mapping.json`. Markdown input is legacy/debug-only. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-add-bookend-slides"></a>`presentation.add_bookend_slides` | `required` / `presentation` | Add the opening "relevant events + most-cited papers + prepared-by" slide and the closing hyperlinked "sources & further reading" slide to a rendered deck, derived deterministically from the deck's own bound `render_spec.json` source chips (frequency-ranked talks, papers, and sources). Nothing is topic-hardcoded; optional per-deck `bookend_event_overrides.json` only supplies punchier curated event labels. | `scripts/presentations/add_bookend_slides.py --deck-dir decks/<id> [--event-overrides decks/<id>/bookend_event_overrides.json]`. Runs after `presentation.generate_render_spec` and injects two `slide_kind` (`events`, `resources`) slides into both `presentation_deck.json` and `render_spec.json`, re-stamping `approved_expansion.target_slide_count`. Rerun `presentation.reconcile_contract_hashes` afterward so contract hashes stay aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-engaging-narration"></a>`presentation.engaging_narration` | `required` / `presentation` | Draft engaging two-host narration (named hosts, env-overridable) for every render-spec slide: each source introduced once rather than repeated mechanically, no wrong/mechanical visual descriptions, analogies and concrete examples woven in from the analysis, ~20-25 minute target. Uses Claude Opus via `ANTHROPIC_API_KEY`. | `scripts/presentations/generate_engaging_narration.py --deck-dir decks/<id>`. Source the repo `.env` for `ANTHROPIC_API_KEY` first. Writes `engaging_narration_raw.json` plus debug `*.ENGAGING.md` exports; host names come from `ENGAGING_NARRATION_HOST_A`/`ENGAGING_NARRATION_HOST_B` (default Iapetus/Erinome). Follow with `presentation.rebuild_delivery_engaging`. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-rebuild-delivery-engaging"></a>`presentation.rebuild_delivery_engaging` | `required` / `presentation` | Rebuild both delivery contracts to carry the engaging narration over all render-spec slides (including the events/resources bookends), wiring two-host voices and word-count-based timing so the existing video builders synthesize the new narration. | `scripts/presentations/rebuild_delivery_engaging.py --deck-dir decks/<id>`. Runs after `presentation.engaging_narration`; rewrites `delivery_notebooklm.json` and `delivery_single_speaker.json` (segments, speaker/voice style, timing map, source grounding). Host/voice and single-speaker names share the `ENGAGING_NARRATION_*` env vars. Rerun `presentation.reconcile_contract_hashes`, then `presentation.build_presenter_video` for both variants. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-reconcile-contract-hashes"></a>`presentation.reconcile_contract_hashes` | `required` / `presentation` | Re-stamp cross-referenced contract hashes after deck/render-spec/delivery edits (bookends, engaging narration) so the produced-presentation review does not flag artifact drift. Topic-agnostic. | `scripts/presentations/reconcile_contract_hashes.py --deck-dir decks/<id>`. Re-stamps `render_spec.input_hashes` for deck-dir files and `delivery_*.presentation_deck_sha256` / source-binding / visual-plan hashes. Run after any `add_bookend_slides` or `rebuild_delivery_engaging` edit and before render, video build, and produced-presentation review. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-presenter-script"></a>`presentation.delivery_generation` | `required` / `presentation` | Generate structured single-speaker and NotebookLM delivery contracts after render mapping exists. | `scripts/presentations/generate_presenter_script.py --render-spec decks/<id>/render_spec.json --slide-mapping decks/<id>/presentation_slide_mapping.json --delivery-single-speaker decks/<id>/delivery_single_speaker.json --delivery-notebooklm decks/<id>/delivery_notebooklm.json --debug-script-md decks/<id>/6_presenter_script.md`. The single-speaker Markdown script is declared narration evidence for whole-presentation review; the NotebookLM Markdown export remains auxiliary. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-video-delivery-validation"></a>`presentation.video_delivery_validation` | `required` / `presentation` | Path-based delivery checks on both final MP4s: duration budget (`video_duration_under_40_minutes`), audio/video stream alignment (`video_stream_durations_match`), agreement with the build receipts (`video_duration_matches_build_receipts`), and no black tail (`video_tail_not_black`). The shared production chain runs it right after video-sync validation, for topic and executive decks alike. | `scripts/presentations/validate_video_delivery.py --deck-dir decks/<id> --video decks/<id>/presentation_video.mp4 --video decks/<id>/presentation_video_notebooklm.mp4 --report-out decks/<id>/video_delivery_validation.json`. Any `fail` check blocks the deck. | [Presentation contract](../operations/presentation-contract.md) |
| <a id="presentation-presentation-review"></a>`presentation.presentation_review` | `required` / `presentation` | Review the rendered presentation, narration, slide screenshots, and produced artifact contracts with Claude Code Opus 4.8, Codex 5.5, and Gemini before production handoff. | `scripts/presentations/review_presentation.py --workspace conferences/<slug> --presentation-id <id> --presentation-type <type> --topic <topic> --requirements decks/<id>/presentation_blueprint.json --narration-script decks/<id>/6_presenter_script.md --screenshots-dir decks/<id>/assets/generated/qc/screenshots --blueprint decks/<id>/presentation_blueprint.json --blueprint-review decks/<id>/blueprint_review.json --deck-contract decks/<id>/presentation_deck.json --render-spec decks/<id>/render_spec.json --source-binding decks/<id>/source_binding_report.json --delivery-contract decks/<id>/delivery_notebooklm.json --video-receipt decks/<id>/video_sync_validation_notebooklm.json --visual-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json --contact-sheet decks/<id>/render_contact_sheet.png --report-output decks/<id>/presentation_review.json --markdown-output decks/<id>/presentation_review.md --prompt-output decks/<id>/presentation_review.prompt.md`. The report must collect guideline adherence, technical-audience pleasantness, production readiness, aesthetic pleasantness, clarity/congestion, and artifact-contract-adherence feedback from all three reviewers. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-reflow-slides"></a>`presentation.reflow_slides` | `legacy-debug` / `presentation` | Retired Markdown reflow helper. Production section ordering lives in `presentation_deck.json`. | `scripts/presentations/reflow_slides.py` is legacy/debug-only and cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-repair-loop"></a>`presentation.repair_loop` | `legacy-debug` / `presentation` | Retired Markdown repair loop. Production repair must update structured visual/render contracts and rerun `render_export` and visual QC. | `scripts/presentations/run_presentation_repair_loop.py` may repair legacy/debug Markdown exports only; it cannot by itself close production repair findings. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-sanity-check-slides"></a>`presentation.sanity_check_slides` | `legacy-debug` / `presentation` | Retired Markdown sanity checker. Production structural validation runs on typed contracts. | `scripts/presentations/sanity_check_slides.py` is legacy/debug-only and cannot satisfy production proof. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-source-pack"></a>`presentation.source_pack` | `required` / `presentation` | Build the deck source pack from extraction, source, and paper-matching evidence. | `scripts/presentations/build_deck_source_pack.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-validate-svgs"></a>`presentation.validate_svgs` | `required` / `presentation` | Validate generated SVG structure, dimensions, text fit, and report findings. | `scripts/presentations/validate_svgs.py`. Write the declared deck artifact/receipt and keep `presentation_manifest.json` owner refs aligned. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |

### Backup Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="backup-sync-to-gdrive"></a>`backup.sync_to_gdrive` | `required` / `backup` | Initialize or sync manifest-selected workspace families to the mounted Google Drive backup/public leaf. | `scripts/sync_to_gdrive.py`. Backup profile commands must pass explicit `--remote-path` to the mounted conference leaf; use `--init-only` for setup and dry-run/write receipts under `storage/workflow-runs/<run-id>/backup/`. | [Private backup](../operations/private-backup.md); [future storage](../operations/future-conference-storage.md) |

### Publishing Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="publishing-export-for-public"></a>`publishing.export_for_public` | `required` / `publishing` | Produce deterministic public release bundles from `public_export_manifest.json`. | `scripts/export_for_public.py --conference-dir conferences/<slug> --bundle-id <bundle_id> --dry-run`, then write mode after review. Outputs `exports/public-release/<bundle_id>/RELEASE.json`, `CHECKSUMS.json`, and a `conference_pipeline.public_export` receipt. Fails closed on secret findings, prohibited redistribution status, missing caution acknowledgement, or immutable-bundle overwrite without a matching reviewed dry-run receipt. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-build-public-index"></a>`publishing.build_public_index` | `required` / `publishing` | Build browser/search-ready public indexes from an exported public release bundle. | `scripts/build_public_index.py --bundle-root conferences/<slug>/exports/public-release/<bundle_id>`. Outputs `index/corpus_chunks.jsonl`, `bm25_index.pkl`, `embeddings.npy`, `chunk_ids.json`, `index_manifest.json`, and a `conference_pipeline.public_index_build` receipt. Fails closed when the input is not a public bundle root with `RELEASE.json` and `CHECKSUMS.json`. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-push-public-repo"></a>`publishing.push_public_repo` | `required` / `publishing` | Stage the read-only public data repo and gate live push. | `scripts/push_public_repo.py --bundle-root conferences/<slug>/exports/public-release/<bundle_id> --target-dir <staging-dir> --dry-run`. Outputs README, LICENSE, NOTICE, CITATION, `public_repo_manifest.json`, CI liveness workflow, and a public repo push receipt. Live mode requires reviewed dry-run receipt and a Gitleaks scan of the staged public repo content. Each conference gets a fresh public repo, so private source history is not part of this publication gate. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-generate-public-site"></a>`publishing.generate_public_site` | `required` / `publishing` | Generate the static catalog/search site from a public bundle and index. | `scripts/generate_public_site.py --bundle-root conferences/<slug>/exports/public-release/<bundle_id> --output-dir <site-dir>`. Outputs `index.html`, `search.html`, CSS/JS search assets, search data, `site_manifest.json`, and a public-site receipt. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-upload-public-media"></a>`publishing.upload_public_media` | `required` / `publishing` | Plan or run public media CDN uploads from `.cdn.json` sidecars. | `scripts/upload_public_media.py --bundle-root <bundle> --receipt-path <receipt> --dry-run`. Live mode requires object-store settings and explicit operator command. Outputs a CDN upload receipt and planned upload list. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-check-public-urls"></a>`publishing.check_public_urls` | `required` / `publishing` | Verify public artifact URL liveness from a CI/non-operator environment. | `scripts/check_public_urls.py --bundle-root <bundle> --receipt-path <receipt>`. Rejects `file://`, local, and Drive paths before HTTP; fails on non-200 public responses. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-package-public-dataset"></a>`publishing.package_public_dataset` | `required` / `publishing` | Generate Croissant, datapackage, citation, and dataset manifest metadata. | `scripts/package_public_dataset.py --bundle-root <bundle> --metadata-dir <metadata-dir>`. Outputs `croissant.json`, `datapackage.json`, `dataset_manifest.json`, `CITATION.cff`, and a package receipt. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-deposit-zenodo"></a>`publishing.deposit_zenodo` | `required` / `publishing` | Prove sandbox/live-gated Zenodo deposit automation. | `scripts/deposit_zenodo.py --metadata-dir <metadata-dir> --sandbox --dry-run --receipt-path <receipt>`. Live DOI minting requires `.env` token loading, `--live`, and `--ack-live-doi`; receipts redact token values. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-push-hf-dataset"></a>`publishing.push_hf_dataset` | `required` / `publishing` | Prove dry-run/live-gated Hugging Face dataset push automation. | `scripts/push_hf_dataset.py --metadata-dir <metadata-dir> --repo-id <org/repo> --dry-run --receipt-path <receipt>`. Live push requires `.env` token loading, `--live`, and `--ack-live-push`; receipts redact token values. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-generate-byok-tooling"></a>`publishing.generate_byok_tooling` | `required` / `publishing` | Generate browser BYOK Q&A and corpus-only deep-search assets. | `scripts/generate_byok_tooling.py --bundle-root <bundle> --receipt-path <receipt>`. Outputs `tools.html`, `assets/byok-tools.js`, `assets/byok-corpus.json`, `metadata/byok_tools_manifest.json`, and a BYOK tools receipt. Fails closed on key-persistence APIs, maintainer endpoint markers, or private markers. | [Public release runbook](../operations/public-release.md); [BYOK cost safety](../operations/public-byok-cost-safety.md) |
| <a id="publishing-package-self-host-tools"></a>`publishing.package_self_host_tools` | `required` / `publishing` | Package self-host deck, report, GPT-Researcher, and podcastfy/Edge configs. | `scripts/package_self_host_tools.py --bundle-root <bundle> --receipt-path <receipt>`. Outputs `self-host/README.md`, local-source configs, `podcastfy_config.yaml`, and a self-host tools receipt. | [Public release runbook](../operations/public-release.md) |
| <a id="publishing-generate-notebooklm-audio"></a>`publishing.generate_notebooklm_audio` | `required` / `publishing` | Plan or run gated NotebookLM podcast generation. | `scripts/generate_notebooklm_audio.py --bundle-root <bundle> --dry-run --receipt-path <receipt>`. Live mode requires operator acknowledgement and Google account cookie/profile state; receipts redact credential paths. | [Public release runbook](../operations/public-release.md) |

### Verification Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="verification-stage-owner-audit"></a>`verification.stage_owner_audit` | `required` / `verification` | Compatibility audit for legacy owner-command planning checks. | `conference_pipeline.stage_owner_audit`. Run `python -m conference_pipeline.stage_owner_audit --plan-root <dir>` before approval. | [CLI reference](../cli.md); this registry runbook |
| <a id="verification-stage-script-usage"></a>`verification.stage_script_usage` | `required` / `verification` | Unified verifier for registry, plan, skill, docs, and run usage evidence. | `scripts/verify_stage_script_usage.py`. Run with `--plan-root`, `--skills-root`, `--docs-root`, or `--workspace --run-id`; strict mode blocks completion. | [CLI reference](../cli.md); this registry runbook |
| <a id="verification-conference-hardcoding-audit"></a>`verification.conference_hardcoding_audit` | `required` / `verification` | Portability gate for hardcoded conference literals (name/year, program domain, year-pinned `/virtual/<year>/` paths) in `src/conference_pipeline/**`. | `scripts/audit_conference_hardcoding.py` (human report, exit 1 on findings) or `--json`. Must pass before closing any conference milestone. | [conference-hardcoding-audit skill](../../.claude/skills/conference-hardcoding-audit/SKILL.md) |
| <a id="verification-conference-total-cost"></a>`verification.conference_total_cost` | `allowed` / `verification` | Compute the TOTAL cost of a conference — canonical workspace receipts plus attributable misplaced repo-root receipts — with per-model/provider/stage breakdown and reported (never mutated) budget-cap status. | `scripts/compute_conference_total_cost.py --workspace conferences/<slug> [--attribute-segment <segment> ...] [--since <iso>] [--until <iso>] --json`. Never moves, deletes, or rewrites receipt files, and never raises the budget cap. | [Cost calculator skill](../../.claude/skills/conference-cost-calculator/SKILL.md); this registry runbook |

### Deprecated Tools

| Script id | Status | Usage | Command / proof shape | Deeper docs |
|-----------|--------|-------|-----------------------|-------------|
| <a id="presentation-deprecated-add-images"></a>`presentation.deprecated_add_images` | `deprecated` / `presentation` | Deprecated image insertion helper; must not satisfy production presentation completion. | `scripts/presentations/add_images_to_md.py`. Deprecated/legacy only; rejected as completion-owner evidence and allowed only in negative-policy or historical context. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-deprecated-extract-topic-figures"></a>`presentation.deprecated_extract_topic_figures` | `deprecated` / `presentation` | Deprecated paper-figure extraction helper; must not satisfy production figure generation. | `scripts/presentations/extract_topic_figures.py`. Deprecated/legacy only; rejected as completion-owner evidence and allowed only in negative-policy or historical context. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-deprecated-generate-slides-agentic"></a>`presentation.deprecated_generate_slides_agentic` | `deprecated` / `presentation` | Deprecated legacy slide generator; must not satisfy reusable deck generation. | `scripts/generate_slides_agentic.py`. Deprecated/legacy only; rejected as completion-owner evidence and allowed only in negative-policy or historical context. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="presentation-deprecated-generate-slides-simple"></a>`presentation.deprecated_generate_slides_simple` | `deprecated` / `presentation` | Deprecated simple slide generator; must not satisfy reusable deck generation. | `scripts/presentations/generate_slides_simple.py`. Deprecated/legacy only; rejected as completion-owner evidence and allowed only in negative-policy or historical context. | [Presentation contract](../operations/presentation-contract.md); [pipeline workflow](../workflows/pipeline-workflow.md) |
| <a id="ranking-deprecated-rank-papers"></a>`ranking.deprecated_rank_papers` | `deprecated` / `analysis` | Deprecated NeurIPS-era ranking helper; reusable paper matching must use artifact keypoints. | `scripts/rank_papers.py`. Deprecated/legacy only; rejected as completion-owner evidence and allowed only in negative-policy or historical context. | [Analysis contract](../operations/analysis-contract.md) |


## presentation-brand-template-lock

- **Owner ID:** `presentation.brand_template_lock`
- **Path:** `scripts/presentations/lock_brand_template.py`
- **Outputs:** `01-BRAND-REFERENCE-LOCK.md`, `01-BRAND-REFERENCE-LOCK.json`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-brand-template-lock`

Phase 01 brand governance gate. Validates that the Topic 4 deck uses the
canonical RelationalAI template, computes SHA-256 asset hashes, and blocks on
fake logo treatment, missing hash, or scaffold prose. Required before any Phase
01–10 regeneration work.

## presentation-editorial-source-board

- **Owner ID:** `presentation.editorial_source_board`
- **Path:** `scripts/presentations/build_editorial_source_board.py`
- **Outputs:** `01-SOURCE-EDITORIAL-BOARD.json`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-editorial-source-board`

Phase 03 source diversity gate. Builds a named-source editorial board from
workspace and topic analysis before storyboard generation.

## presentation-storyboard-generation

- **Owner ID:** `presentation.storyboard_generation`
- **Path:** `scripts/presentations/build_storyboard.py`
- **Outputs:** `01-STORYBOARD.md`, `01-VISUAL-DIRECTION.md`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-storyboard-generation`

Phase 03 storyboard and visual direction gate. Generates storyboard rows and
visual direction from the editorial source board before storyboard review.

## presentation-storyboard-review

- **Owner ID:** `presentation.storyboard_review`
- **Path:** `scripts/presentations/review_storyboard.py`
- **Outputs:** `01-STORYBOARD-REVIEW.json`, `01-STORYBOARD-REVIEW.md`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-storyboard-review`

Phase 03 storyboard review gate. Records `approval.status` before Phase 04
render planning approval. Unresolved critical or major findings block render
planning.

## presentation-figure-review-page

- **Owner ID:** `presentation.figure_review_page`
- **Path:** `scripts/presentations/build_figure_review_page.py`
- **Outputs:** `figure_review_page.html`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-figure-review-page`

Written by the runner at the canary stop, after the vision judge: every
rendered slide beside its vision findings, the adversarial critic findings
still open after three rounds, and any operator outcome already recorded.
The operator reads it to decide each open figure (figure fidelity policy, item 9).

## presentation-patch-figure-spec

- **Owner ID:** `presentation.patch_figure_spec`
- **Path:** `scripts/presentations/patch_figure_spec.py`
- **Outputs:** `figure_specs/<figure>.patch.json`, `operator_resolution` in `figure_specs/<figure>.critic.json`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-patch-figure-spec`

Operator-run. Records accept, patch or replace for one figure the critic left
open; a patch or replacement is validated and compiled before anything is
written. The acceptance receipt and the runner refuse to accept a deck while
`conference_pipeline.presentation.figure_findings.undecided_figure_findings`
is non-empty.

## presentation-rough-cut-review-packet

- **Owner ID:** `presentation.rough_cut_review_packet`
- **Path:** `scripts/presentations/build_rough_cut_review_packet.py`
- **Outputs:** `01-ROUGH-CUT-VIDEO-REVIEW.md`, `01-ROUGH-CUT-VIDEO-REVIEW.json`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-rough-cut-review-packet`

Phase 09 rough-cut evidence gate. Summarizes render artifacts, frame evidence,
and video sync status for operator review before human approval.

## presentation-rough-cut-review

- **Owner ID:** `presentation.rough_cut_review`
- **Path:** `scripts/presentations/review_rough_cut.py`
- **Outputs:** `01-HUMAN-REVIEW-PACKET.md`, `01-HUMAN-REVIEW-PACKET.json`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-rough-cut-review`

Phase 09 human review packet gate. Produces the human review packet from the
rough-cut review findings. Explicit human approval is still required before
Topic 4 canary artifacts are promoted. Bulk rollout remains blocked until a
separate rollout plan is created.

## presentation-canary-runner

- **Owner ID:** `presentation.canary_runner`
- **Path:** `scripts/presentations/run_topic4_canary.py`
- **Outputs:** `01-GOVERNANCE.md`
- **Usage doc ref:** `docs/reference/stage-owner-audit.md#presentation-canary-runner`

Phase 01 canary boundary gate. Refuses non-Topic-4 deck directories, checks the
storyboard review and cross-AI creative/storyboard synthesis approval fields
before render planning, and keeps bulk rollout blocked until a separate rollout
milestone is planned and approved.

## Planning Gate

Run both registry checks after generating or editing conference planning
artifacts and before finalizing the plan. Also verify the workspace gitignore
policy when the target workspace has a storage manifest:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py \
  --workspace conferences/<slug> \
  --write --check \
  --receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json

PYTHONPATH=src .venv/bin/python -m conference_pipeline.stage_owner_audit \
  --plan-root <generated-plan-root>

PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --plan-root <generated-plan-root> \
  --skills-root .claude/skills \
  --docs-root docs \
  --strict
```

The compatibility owner audit exits `0` when the plan uses approved owner refs.
The unified verifier additionally requires the stage script registry digest and
verifier command to appear in the plan, checks that planning-skill script
mentions are registered, and proves every registry-owned `usage_doc_refs` anchor
resolves under `docs/`.

## Agent Diagnostics

Registry-owned tools are Codex-facing decision tools, not just pass/fail
scripts. Their JSON output includes a `diagnostic` object that agents should
read before retrying or repairing a run:

- `summary`: compact statement of what passed or why the tool blocked.
- `tool_id`: registry tool id or verifier entrypoint that emitted the message.
- `checked_inputs`: plan roots, workspace/run ids, registry paths, docs roots,
  receipt counts, or manifest paths used for the decision.
- `accepted_evidence`: the evidence families accepted by the tool.
- `outputs`: artifacts or receipts the tool produced or verified.
- `blocker_count` and `warning_count`: quick severity scan.
- `next_actions`: concrete recovery or downstream actions.
- `rerun_command`: exact or representative command to rerun when available.

Individual blockers and warnings also carry `expected`, `observed`, and
`next_action` fields. Use those fields to decide whether to edit a generated
plan, add a missing doc anchor, rerun a workflow stage, regenerate a receipt,
refresh the workspace `.gitignore`, or block production completion.

The registry-owned `diagnostic_profile` is the source for tool-specific success
and blocked summaries. Deprecated tools must name replacement `script_id`s in
that profile so verifiers can explain the approved recovery path instead of
only saying that an owner is forbidden.

## Run Gate

Before a workflow run can be treated as production-complete, verify completed
stage receipts against the same registry:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --strict
```

In strict mode, completed in-scope stages fail when required scripts are absent
from receipt commands, `command_metadata.owner_ref`, presentation manifest
`owner_refs`, or explicit approved waivers. Deprecated scripts may appear only
in negative-policy context, never as completion owners.

## Runner Inventory Gate

Conference-specific live runners can expose a stage inventory with
`StageOwnerRecord` entries and call `audit_stage_owner_inventory()`.

Allowed categories are:

- `conference_adapter`: conference-local source adaptation, acquisition, OCR
  gating, or auth handling.
- `reusable_owner`: production output delegated to an approved shared owner.
- `claim_aggregator`: release, proof, QA, or bookkeeping that only aggregates
  existing artifacts and receipts.
- `legacy_scaffold`: old helper code retained for tests or diagnostics. A
  production output stage with this category fails unless it declares an
  explicit blocker.

Use `registry_task_ids_by_stage` to bind runner stages to approved registry
tasks. Visual QC requires both the Playwright screenshot owner and the
vision-LLM reviewer owner; a screenshot-only report is not enough.

### Presentation claim fidelity review

<a id="presentation-claim-fidelity-review"></a>
`presentation.claim_fidelity_review` owns `scripts/presentations/review_claim_fidelity.py`.
Run with `--workspace <workspace> --deck-dir <deck> --phase figures --report-output <deck>/claim_fidelity_review_figures.json`
after figure authoring, before compilation/rendering. Run again with `--phase narration`
and `--report-output <deck>/claim_fidelity_review_narration.json` after narration fixes,
before Gate A. The compiler route enforces both calls through `owner_step` and
reuses only passing reports whose source bindings, spec files and (for the second
pass) both spoken scripts still match. Legacy programmatic figures do not emit
specs and retain their semantic-review gate.

The governed model comes from `default_model("presentation.claim_fidelity_review")`
(the `sol` catalog alias), with a $5 per-call budget cap and one text-only call per
phase. The local price catalog bounds the request before it is sent. Provider
receipts use the governed task id. Missing source notes, malformed/truncated
verdicts fail closed. Only direct contradiction, clear misattribution, unlabelled
composition presented as demonstrated, or position-as-result findings block.
Every finding requires `severity: blocker|advisory`; advisory-only reports pass
and retain severity counts. Contribution notes summarize papers rather than
exhaust their content. Labelled synthesis, pedagogical explanation, and wording
scoped to this talk are permitted. `ledger_mismatch` findings are always advisory
and surfaced separately. Gate A receives each existing report through repeatable
`--claim-fidelity-report` flags, with report files tracked for resume freshness,
and must confirm or dismiss every lead rather than treating it as established.
Reports bind exact input hashes and attach the existing `patch_figure_spec.py` or `fix_narration_script.py`
repair tool to each finding. Apply the suggested source-grounded repair through
those tools and resume; this review never accepts a failed finding automatically.
No live review is needed for offline tests, which inject a stub completion.

### Presentation generate figure specs

<a id="presentation-generate-figure-specs"></a>
`scripts/presentations/generate_figure_specs.py --deck-dir <deck> --receipt <receipt> --critic`
owns both `presentation.figure_spec_authoring` and `presentation.figure_spec_critic`.
The author and critic receive per-source contribution notes and work kinds.
`presentation.engaging_narration` is bound to
`scripts/presentations/generate_engaging_narration.py --deck-dir <deck>`.
Source-absence assertions fail narration lint; notes that omit scores cannot
establish that a paper has no evaluation. Gate A receives the same bound notes,
checks every prior blocker from `gate_a_attempts/`, then sweeps every slide again.
Slide titles over the shared character limit receive one wording repair and
must pass blueprint validation; no words are silently cut.
