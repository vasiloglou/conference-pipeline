# Presentation Manifest — Schema Specification v1

> **NOTE:** This document specifies the on-disk format of
> `conferences/<slug>/decks/presentation_manifest.json` as of Phase 22. Schema
> version is the load-bearing integer for downstream compatibility checks.

---

## Implementation Status (April 2026)

### Completed

- **Schema version:** `1`, integer, checked by
  `PresentationManifest.from_json` in
  `src/conference_pipeline/presentation/manifest.py`.
- **Emit:** `emit_presentation_manifest(...)` writes the workspace-level
  manifest; `scripts/backfill_presentation_manifest.py` delegates to the same
  emitter after scanning historical deck artifacts.
- **Validate:** `load_and_validate_manifest(...)` fails closed on malformed
  JSON, unsupported schema versions, path escape, digest mismatch, and missing
  declared artifacts.
- **Compatibility:** Supported schema versions: `{1}`. Values other than `1`
  fail closed on load.

## File Location

The manifest lives at `conferences/<slug>/decks/presentation_manifest.json`.
For example: `conferences/neurips2025/decks/presentation_manifest.json`.

## Canonical Example

```json
{
  "schema_version": 1,
  "generated_at": "2026-04-23T20:00:00Z",
  "generator": "scripts.backfill_presentation_manifest@<commit>",
  "workspace": "/abs/path/to/conferences/neurips2025",
  "content_sha256": "...",
  "stage_specs": [
    {
      "id": "render_export",
      "input_artifacts": [
        "decks/<presentation>/presentation_deck.json",
        "decks/<presentation>/source_binding_report.json",
        "decks/<presentation>/visual_plan.json",
        "decks/<presentation>/render_spec.json"
      ],
      "output_artifacts": [
        "decks/<presentation>/presentation.html",
        "decks/<presentation>/presentation.pdf",
        "decks/<presentation>/presentation_slide_mapping.json"
      ],
      "owner_refs": [
        "scripts/presentations/md_to_revealjs.py",
        "scripts/presentations/export_revealjs_pdf.py"
      ],
      "provider_requirements": [
        "tool:node"
      ],
      "cache_behavior": "renders HTML/PDF and slide mapping from structured render_spec.json",
      "support_level": "workspace-local",
      "mode_readiness": {
        "live": "ready",
        "dry_run": "planned",
        "mock": "unsupported"
      }
    }
  ],
  "presentations": [
    {
      "id": "01_tools_and_open_source",
      "presentation_type": "topic-deep-dive",
      "title": "Tools and Open Source",
      "topic_scope": {
        "mode": "single-topic",
        "topics": [
          "01_tools_and_open_source"
        ]
      },
      "target_duration_minutes": 60,
      "target_slide_count": 60,
      "target_hyperlinks_per_slide": 2.0,
      "target_svgs_per_slide": 0.7,
      "source_analysis_artifacts": [
        {
          "kind": "file",
          "path": "analysis/01_tools_and_open_source.md",
          "sha256": "..."
        }
      ],
      "stage_states": {
        "render_export": {
          "status": "complete",
          "artifacts": [
            {
              "kind": "file",
              "path": "decks/01_tools_and_open_source/presentation.html",
              "sha256": "..."
            },
            {
              "kind": "file",
              "path": "decks/01_tools_and_open_source/presentation_slide_mapping.json",
              "sha256": "..."
            }
          ],
          "notes": []
        }
      }
    }
  ]
}
```

## Top-Level Required Fields

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | MUST equal `1`. Values outside `SUPPORTED_SCHEMA_VERSIONS` fail closed. |
| `generated_at` | string | ISO-8601 UTC with `Z` suffix. Excluded from `content_sha256` so re-emitting with no content change is stable. |
| `generator` | string | Printable ASCII, at most 200 characters. Identifies the tool that wrote the manifest. |
| `workspace` | string | Absolute path to the conference workspace root. |
| `content_sha256` | string | 64-character SHA-256 digest over manifest content with `generated_at` removed. |
| `stage_specs` | list | Ordered checkpoint catalog for the full presentation lifecycle. |
| `presentations` | list | Workspace-local presentation entries; may be empty. |

## Exact Stage IDs

The manifest defines these durable checkpoint ids:

- `presentation_blueprint`
- `deck_contract_generation`
- `source_binding`
- `visual_plan_generation`
- `render_spec_generation`
- `render_export`
- `visual_qc`
- `delivery_generation`
- `video_generation`
- `video_sync_validation`
- `human_review`

First-class presentation stages are presentation_blueprint,
deck_contract_generation, source_binding, visual_plan_generation,
render_spec_generation, render_export, visual_qc, delivery_generation,
presentation_review, video_generation, video_sync_validation, and human_review.

`presentation_blueprint` is the required story-grounding checkpoint before
structured deck contract generation. It consumes `presentation_frame.json` so
the public audience title and opening field questions are separated from
retrieval topic words. It outputs `presentation_frame.json`,
`presentation_blueprint.json`, `presentation_blueprint.md`,
`presentation_blueprint_receipt.json`, `presentation_frame_validation`,
`blueprint_review.json`, and `blueprint_review.md`. Production
`deck_contract_generation` may consume the blueprint only when
`blueprint_review.json` records `approved_for_draft_generation: true`, the
current `blueprint_sha256`, the blueprint is hash-bound to the current
`presentation_frame.json`, passing `presentation_frame_validation`, and gate
type: deterministic, gate type: LLM-judge, and gate type: human evidence.
Mock/dry-run blueprint artifacts are
planning/test evidence only.

The structured production artifact chain is `presentation_deck.json`,
`source_binding_report.json`, `visual_plan.json`, `render_spec.json`,
`delivery_single_speaker.json`, `delivery_notebooklm.json`,
`6_presenter_script.md`, `video_sync_validation_single_speaker.json`, and
`video_sync_validation_notebooklm.json`. Old numbered slide draft Markdown files
are legacy-only numbered Markdown. They may be registered only as
debug-export-only artifacts with `production_evidence: false`; using them as
production evidence must be blocked with `PRESENTATION_LEGACY_MARKDOWN_STAGE_USED`.

## Per-Stage Required Fields

Each entry in `stage_specs` declares:

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | One of the exact stage ids above. |
| `input_artifacts` | list[string] | Canonical input artifact patterns or paths for the checkpoint. |
| `output_artifacts` | list[string] | Canonical output artifact patterns or paths for the checkpoint. |
| `owner_refs` | list[string] | Historical owner scripts/modules/docs for the stage. |
| `provider_requirements` | list[string] | Current provider/tool requirements, recorded truthfully. |
| `cache_behavior` | string | Short description of how the stage reuses or refreshes artifacts. |
| `support_level` | string | Existing repo support vocabulary; Phase 22 defaults to `workspace-local`. |
| `mode_readiness` | object | Explicit `live`, `dry_run`, and `mock` readiness values. |

For `figure_generation`, `owner_refs` must not name deprecated paper-figure
extraction/insertion scripts. Reusable production requires a programmatic
matplotlib, Plotly, or svgwrite SVG owner, or the stage stays explicitly
unsupported/blocking.

For `visual_qc`, `owner_refs` must include both
`scripts/presentations/inspect_rendered_slides.py` and
`scripts/presentations/inspect_rendered_slides_llm.py`. The stage outputs both
`assets/generated/visual_qc_report.md` and
`assets/generated/visual_llm_qc_report.json`; the latter is a governed
`presentation.visual_qc` vision-LLM gate over rendered screenshots.

### `mode_readiness`

`mode_readiness` is an object with exact keys:

- `live`
- `dry_run`
- `mock`

Allowed values are:

- `ready`
- `planned`
- `unsupported`

## Per-Presentation Required Fields

Each entry in `presentations` declares:

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Workspace-local presentation id such as `01_tools_and_open_source`, `master_presentation`, `executive_summary`, `executive_summary_short`, or `executive_summary_long`. |
| `presentation_type` | string | One of `topic-deep-dive`, `master`, or `executive-summary`. |
| `title` | string | Human-readable title for operators. |
| `topic_scope` | object | Declares `mode` plus the topic ids covered by the presentation. |
| `target_duration_minutes` | int | Planned duration target for the presentation variant. |
| `target_slide_count` | int | Planned slide-count target for the presentation variant. |
| `target_hyperlinks_per_slide` | number | Optional content-density gate for hyperlink-enriched markdown; `0` disables the check for legacy manifests. |
| `target_svgs_per_slide` | number | Optional content-density gate for generated SVG references in figure-enriched markdown; `0` disables the check for legacy manifests. |
| `source_analysis_artifacts` | list | Upstream analysis artifacts that inform the presentation. |
| `stage_states` | object | One entry per exact stage id. |

When short and long conference overview decks are requested, both
`executive_summary_short` and `executive_summary_long` are required
`presentation_type: executive-summary` entries. Their source analysis artifacts
should include `analysis/executive_summary.md`, `analysis/trend_report.md`, and
the final topic reports they summarize. The source report
`analysis/executive_summary.md` is not itself a deck deliverable.

### `topic_scope`

`topic_scope` uses:

- `mode`
- `topics`

Current modes are:

- `single-topic`
- `all-topics`

## Stage-State Contract

Each `stage_states.<stage_id>` entry declares:

| Field | Type | Notes |
|-------|------|-------|
| `status` | string | One of `missing`, `partial`, or `complete`. |
| `artifacts` | list | Concrete file or directory evidence recorded for the stage. |
| `notes` | list[string] | Human-readable notes about missing or partial evidence. |

Status values:

- `missing`
- `partial`
- `complete`

Historical backfill may legitimately produce `missing` or `partial` states; the
contract is designed to preserve that truth instead of forcing regeneration.

## Artifact Reference Contract

Every artifact entry uses exact fields:

| Field | Type | Notes |
|-------|------|-------|
| `kind` | string | `file` or `directory`. |
| `path` | string | Workspace-relative path. |
| `sha256` | string | File SHA-256 or deterministic directory tree digest. |

Directory digests are computed by hashing sorted child file paths together with
the SHA-256 of each child file. This allows directories such as
`video_build_notebooklm/` to participate in integrity checks.

## Fail-Closed Contract

`load_and_validate_manifest(workspace, require_path_existence=True)` raises
`PresentationManifestValidationError` on any of:

- manifest file missing
- JSON malformed
- `schema_version` unsupported
- required field missing or wrong type
- unknown or incomplete stage catalog
- absolute path or `../` escape outside the workspace
- any declared artifact missing from disk
- any declared file or directory digest mismatch

`PresentationManifestValidationError` subclasses
`PresentationContractError(Exception)`, not `ValueError`, so contract failures
are not silently swallowed by fallback paths.

## Current Boundary

Phase 22 defines the machine-readable presentation contract before the
canonical workflow registry consumes it. This schema is intentionally
load-bearing for future phases, but the document does **not** claim that
`workflow run` already executes these presentation stages today.

## Related

- [Presentation Contract Runbook](../operations/presentation-contract.md)
- [Pipeline Workflow](../workflows/pipeline-workflow.md)
- [Source Of Truth](../source-of-truth.md)
