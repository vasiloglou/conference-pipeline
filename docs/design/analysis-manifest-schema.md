# Analysis Manifest — Schema Specification v1

> **NOTE:** This document specifies the on-disk format of
> `conferences/<slug>/analysis/analysis_manifest.json` as of Phase 12. Schema
> version is the load-bearing integer for downstream compatibility checks.

---

## Implementation Status (April 2026)

### Completed

- **Schema version:** `1`, integer, checked by `AnalysisManifest.from_json` in `src/conference_pipeline/analysis/manifest.py`.
- **Emit:** `run_analyze_stage` emits the manifest as its final step; `scripts/backfill_analysis_manifest.py` delegates to the same emitter.
- **Validate:** podcast and deck stages call `load_and_validate_manifest` at the start of `build_stage_commands`; they fail closed in live, dry-run, and mock modes.
- **Compatibility:** Supported schema versions: {1}. Values other than 1 fail closed on load. Future bumps add entries to `SUPPORTED_SCHEMA_VERSIONS` in lockstep with migration code.

## File Location

The manifest lives at `conferences/<slug>/analysis/analysis_manifest.json`.
For example: `conferences/neurips2025/analysis/analysis_manifest.json`.

## Canonical Example

```json
{
  "schema_version": 1,
  "generated_at": "2026-04-21T20:00:00Z",
  "generator": "conference_pipeline.analysis.emit_manifest@<commit>",
  "workspace": "conferences/neurips2025",
  "content_sha256": "...",
  "topics": [
    {
      "id": "01_tools_and_open_source",
      "number": 1,
      "slug": "tools_and_open_source",
      "title": "Tools and Open Source",
      "markdown_path": "analysis/01_tools_and_open_source.md",
      "markdown_sha256": "...",
      "fact_ranks_path": "audio/01_tools_and_open_source/ranks.json",
      "fact_ranks_sha256": "...",
      "figures_dir": null,
      "audio_dir": "audio/01_tools_and_open_source",
      "decks_dir": "presentations/01_tools_and_open_source",
      "provenance_path": "analysis/provenance/01_tools_and_open_source.md",
      "provenance_json_path": "analysis/provenance/01_tools_and_open_source.json"
    }
  ]
}
```

## Top-Level Required Fields

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | MUST equal `1` by exact equality. Values outside `SUPPORTED_SCHEMA_VERSIONS` fail closed. |
| `generated_at` | string | ISO-8601 UTC with `Z` suffix. Excluded from `content_sha256` so re-emitting with no content change is stable. |
| `generator` | string | Printable ASCII, at most 200 characters. Identifies the tool that wrote the manifest. |
| `workspace` | string | Absolute path to the conference workspace. |
| `content_sha256` | string | 64-character SHA-256 digest over the manifest content with `generated_at` removed. |
| `topics` | list | Ordered by `number` ascending; may be empty. |

## Per-Topic Required Fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | `NN_slug` form. |
| `number` | int | 1..N; matches the prefix in `id`. |
| `slug` | string | Portion of `id` after `NN_`. |
| `title` | string | First H1 of the topic markdown, or title-cased slug as fallback. |
| `markdown_path` | string | Workspace-relative path to the topic markdown. |
| `markdown_sha256` | string | Streaming SHA-256 of markdown bytes. |

## Per-Topic Optional Fields

| Field | Type | Notes |
|-------|------|-------|
| `fact_ranks_path` | string or null | Null iff `fact_ranks_sha256` is null; paired nullability is enforced. |
| `fact_ranks_sha256` | string or null | Streaming SHA-256 of `ranks.json` bytes. |
| `figures_dir` | string or null | Workspace-relative figures directory. |
| `audio_dir` | string or null | Workspace-relative audio directory. |
| `decks_dir` | string or null | Workspace-relative presentation/deck directory. |
| `provenance_path` | string or null | Optional workspace-relative narrative provenance markdown sidecar. Omitted when no sidecar exists. |
| `provenance_json_path` | string or null | Optional workspace-relative structured provenance JSON sidecar. Omitted when no sidecar exists. |
| `deep_analysis_path` | string or null | Optional workspace-relative deep report path. Defaults to `markdown_path` for backward compatibility. |
| `slide_brief_path` | string or null | Optional workspace-relative deck-facing slide brief sidecar under `analysis/slide_briefs/`. |
| `analysis_profile` | string or null | Optional narrative profile label such as `deep-narrative-v1`. |
| `narrative_model` | string or null | Optional governed synthesis model recorded for the report. |
| `deep_validation` | object or null | Optional deep-analysis validation metadata captured at render time. |

## Receipt Integration (Phase 11 + Phase 12)

The analyze stage's `StageReceipt` (Phase 11 schema unchanged) carries a single
`artifacts` entry of `kind: "analysis_manifest"` with the following fields:

| Field | Type | Notes |
|-------|------|-------|
| `kind` | string | Always `"analysis_manifest"`. |
| `path` | string | Workspace-relative path, e.g. `analysis/analysis_manifest.json`. |
| `manifest_sha256` | string | Whole-file SHA-256 of the on-disk manifest bytes. Changes whenever `generated_at` changes. |
| `manifest_content_sha256` | string | Content-stable SHA-256 excluding `generated_at`. Resume compatibility binds to this value. |
| `sub_step_digests` | object | Map of `{step_name: sha256}` for the analysis steps. The generic six-step chain remains stable; narrative/provenance source-pack runs may add `narrative_provenance`. |

**Resume binding:** podcast and deck stages compute their resume compatibility key over `manifest_content_sha256` (not `manifest_sha256`), so re-running analyze with no content changes does not bust downstream receipts.

## Fail-Closed Contract

`load_and_validate_manifest(workspace, require_path_existence=True)` raises
`ManifestValidationError` on any of:

- Manifest file missing.
- JSON malformed.
- `schema_version` not an int, or not in `SUPPORTED_SCHEMA_VERSIONS`; currently this is exact equality to 1.
- Required field missing or wrong type.
- Any declared non-null path absent from the filesystem.
- Recomputed `content_sha256` does not match the persisted value.

`ManifestValidationError` subclasses `AnalysisContractError(Exception)`, not
`ValueError`, so non-live fallback paths cannot silently swallow it. When raised
during `run_workflow_stages` preflight, a `StageReceipt(status="failed",
error="preflight failed: ...")` is persisted before the exception propagates.

## Related

- [Analysis Contract Runbook](../operations/analysis-contract.md)
- [Workflow API Reference](../api/workflow.md)
