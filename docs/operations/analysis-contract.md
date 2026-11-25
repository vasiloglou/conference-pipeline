# Analysis Contract Runbook

Use this runbook to run the standardized analysis stage, inspect the emitted
`analysis_manifest.json`, consume it from podcast and deck stages, and backfill
legacy workspaces.

Analysis path resolution follows the manifest contract: the analyze stage
emits `conferences/<slug>/analysis/analysis_manifest.json`, and podcast and
deck stages consume it as the source of truth for topics, declared paths, and
content digests.

## Scope

This runbook covers:

- Running the standardized analyze stage on `conferences/neurips2025/` (`conferences/neurips2025/analysis/analysis_manifest.json`).
- Inspecting manifest content, including schema, topics, and digests.
- Running podcast and deck from the manifest without topic-specific glue.
- Backfilling a legacy workspace that does not yet have a manifest.
- Troubleshooting fail-closed manifest validation errors.

## Running Analyze

Emit a fresh manifest as part of the analyze stage:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
    --workspace conferences/neurips2025 \
    --stages analyze \
    --mode live \
    --run-id analyze-$(date +%Y%m%d)
```

The receipt written under
`conferences/neurips2025/storage/workflow-runs/<run-id>/stage-04-analyze.json`
records the manifest digests and sub-step digests as a single entry in the
receipt's `artifacts` list. The artifact entry exposes two distinct digest
fields: `manifest_sha256` (whole-file SHA-256 of the manifest bytes) and
`manifest_content_sha256` (content-stable SHA-256 excluding the `generated_at`
timestamp so re-emits with identical content keep the same value).

Safety guard: live analyze preserves existing analysis artifacts by default,
including topic reports, coverage/mapping files, QA reports, and traceability
addenda. A run against an already-populated workspace may refresh the manifest
and receipts without replacing the curated analysis. Destructive regeneration is
reserved for explicit Python API calls with `overwrite_existing=True`.

## Inspecting the Manifest

The manifest lives at `conferences/<slug>/analysis/analysis_manifest.json`
(for example, `conferences/neurips2025/analysis/analysis_manifest.json`).
The file has the following top-level shape:

```json
{
  "schema_version": 1,
  "generated_at": "2026-04-21T20:00:00Z",
  "generator": "conference_pipeline.analysis.emit_manifest@<commit>",
  "workspace": "conferences/neurips2025",
  "content_sha256": "...",
  "topics": []
}
```

## Running Podcast and Deck From the Manifest

Podcast, per topic:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
    --workspace conferences/neurips2025 \
    --stages podcast \
    --topic 01
```

Podcast, all topics:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
    --workspace conferences/neurips2025 \
    --stages podcast
```

Deck, per topic:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
    --workspace conferences/neurips2025 \
    --stages deck \
    --topic 01
```

Deck, all topics:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
    --workspace conferences/neurips2025 \
    --stages deck
```

## Deck Scope (Phase 12)

Deck generation is manifest-backed for workflow selection (topic enumeration and common-arg plumbing); direct script manifest consumption is deferred to a later phase.

In practice this means `workflow run --stages deck --topic 01` uses the
manifest to pick topic `01` and to build the `--topic 01` flag passed to each
of the eight deck sub-scripts. The sub-scripts themselves still resolve their
internal paths from the existing `conferences/neurips2025/` layout. Running the
deck stage against a different conference workspace remains blocked until a
future phase teaches the sub-scripts to read the manifest directly.

## Pre-analysis Extraction Requirement

Before query planning or GPT-Researcher topic execution, production analysis
must run the registry-owned title-spine topic strategy:

```bash
.venv/bin/python scripts/build_topic_strategy.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --provider-cost-approved \
  --model openai:gpt-5.6-sol \
  --model anthropic:claude-opus-4-8 \
  --model gemini:gemini-3.1-pro-preview \
  --merge-model openai:gpt-5.6-sol \
  --summary
```

The stage collects invited/keynote, main oral, session, workshop, and
segmented workshop talk titles into
`storage/analysis/topic-strategy/<run-id>/title_spine.json`, records its
sha256, obtains three independent model recommendations, and writes exactly 12
merged topics to `analysis/topic_strategy.json`. Stale candidate-interest-group
lists, single-model-only topic choices, or planning prose without a
title-spine receipt cannot satisfy this boundary. Mock mode is only a
planning/test path.

Phase 14 final analysis MUST consume
`storage/extractions/<run-id>/by-topic/*.json`, vector-store chunks, query
research receipts, and paper-matching outputs before publishing topic
analysis. The public topic file, `analysis/<topic_id>.md`, is now the deep
analyst document, not a slide brief. It must include `Executive Summary`,
`Key Findings`, `Detailed Analysis`, `References And Sources`,
`Cross-References`, `Recommendations`, `Questions For Further Investigation`,
and a compact `Slide Brief` section. Deck-focused summaries live separately at
`analysis/slide_briefs/<topic_id>.md`.

The machine proof chain lives next to the narrative, not inside it:

- `analysis/provenance/<topic_id>.md` preserves the source-backed ledger with
  `<!-- extraction:<artifact_id> -->`, `<!-- chunk:<chunk_id> -->`, and
  `<!-- research:<research_id> -->` markers plus transcript/OCR/source paths.
- `analysis/provenance/<topic_id>.json` records extraction ids, chunk ids,
  research ids, matched paper ids, quotes, and source paths in structured form.
- `analysis/analysis_citation_manifest.json` remains backward compatible and
  adds `provenance_path` and `provenance_json_path` per production topic.
- `analysis_manifest.json` keeps `markdown_path` for the deep report and may
  add `deep_analysis_path`, `slide_brief_path`, `analysis_profile`,
  `narrative_model`, and `deep_validation` for v1.63+ workspaces.

Public production narratives must not use raw ledger sections such as
`Source Basis`, `Verbatim transcript/OCR evidence`, `Matched Paper Evidence`,
or `Source Gaps` as top-level sections. Production synthesis uses the governed
`analysis.deep_narrative_synthesis` model contract (`openai:gpt-5.6-sol`, fallback
`openai:gpt-5.6-terra`). If the source pack is too thin, the LLM client/model is
missing, synthesis fails validation, or evidence pairing is index-only rather
than source-backed, analysis must write a blocked/degraded receipt instead of
publishing shallow prose as `production-final`.

The GPT-Researcher package (0.16.1) is the research engine, restricted to our
stored hybrid index through the `local_hybrid` retriever plugin and to the
local model through the registry, inside a loopback-only network guard
(`conference_pipeline.research.gptr_engine`); see the pipeline workflow's
Phase 3-4 section.

Local Qwen3.6 35B-A3B plus local BGE-M3 is the default GPT-Researcher source-pack
producer when runs are local-only, corpus-grounded, and receipt-backed: it may
build or query the vector/BM25 evidence layer, produce GPT-Researcher findings,
run parameter sweeps, and surface paper/keypoint candidates for the narrative
source pack. Required receipts include `WEB_SEARCH_ENABLED=false`,
`EMBEDDING_PROVIDER=mlx`, `EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit`,
`EMBEDDING_DIM=1024`, `RETRIEVAL_K=18`,
`PAPER_MATCHING_RETRIEVAL_K=18`, `CANDIDATE_MULTIPLIER=3`, and `RRF_K=60`.
They are not by themselves production-final public narratives. A local report
may replace the governed GPT-5.6 Luna/GPT-5.4 synthesis step only after it passes
the same deep report schema, provenance, source-diversity, keypoint, and
editorial-quality gates that the frontier-model route must pass.

Deep-dive topics require two GPT-Researcher evidence lanes before promotion:
the global no-topic-id lane over the topic-agnostic corpus, and the heuristic
keyword-prefiltered lane owned by
`analysis.gpt_researcher_heuristic_prefilter` /
`conference_pipeline.research.heuristic_prefilter`. The heuristic lane builds
an explicit evidence pack from stable vector chunk sidecars and must record
`workflow.research.heuristic_prefiltered_gpt_researcher`,
`heuristic_keyword_prefilter`, `keyword_seed_source`,
`prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`,
`stores_searched`, `source_family_scope`, `retrieval_scope`, and
`invalid_citation_ids`. Topic-id filtered deep dives, unrecorded keyword lists,
source-family leakage, and citation-invalid heuristic reports are rejected
analysis evidence.

Analysis fails closed (via `analysis_extraction_gate` in
`conference_pipeline.iclr2025_live`) when any of the following hold:

1. The extraction stage receipt at
   `storage/workflow-runs/<run-id>/14A-exhaustive-topic-extraction-receipt.json`
   is missing or its `status` is not `succeeded`.
2. The extraction receipt is older (mtime) than the transcription-ocr
   receipt — extraction must be rerun before downstream phases consume it.
3. The coverage report shows `classes.not_attempted > 0`.
4. Any cited record id from the citation manifest or provenance sidecar does
   not resolve to a per-artifact JSON file under
   `storage/extractions/<run-id>/by-artifact/`.

QA, podcast, deck, release, and proof inherit the same fail-closed
contract: every downstream stage entry calls
`analysis_extraction_gate(workspace_root, run_id)` before any other work
and propagates the resulting blockers into its own receipt. The Phase 14A
closing helper `phase_14a_verification` composes this gate with the
BACKUP-01 receipt check so v1.5 promotion logic is single-call.

The same boundary is the analysis instance of the platform-wide
[Runtime Stage Contract Gate Pattern](../reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern).
Roadmap or planning-skill text can declare that exhaustive extraction belongs
to analysis, but it is not completion evidence by itself. An analysis runner
that can write `status: "succeeded"` must prove, for the current workspace and
run id, that the required analysis owners were actually executed or explicitly
blocked. For exhaustive extraction that means the stage cannot close unless
`storage/workflow-runs/<run-id>/14A-exhaustive-topic-extraction-receipt.json`
and the expected `storage/extractions/<run-id>/by-topic/` and
`storage/extractions/<run-id>/by-artifact/` output families satisfy the gate.
When LLM-assisted full-read extraction is in scope, the same analysis gate must
also require
`storage/workflow-runs/<run-id>/14A-llm-full-read-extraction-receipt.json` and
the `storage/extractions/<run-id>/llm-full-read/by-unit/` plus
`storage/extractions/<run-id>/llm-full-read/by-topic/` output families. That
receipt is production-complete only when it records `status: succeeded`,
`scope: all`, `limit: null`, `checks.completion_state: complete`,
`completed_unit_count == expected_unit_count`, and `failed_count: 0`. Heuristic
14A success, pilot runs, transcript-only runs, metadata-only receipts, or empty
transcript sidecars are blockers, not completion evidence.
Paper units are abstracts: from the conference's
`storage/download-corpus/openreview_accepted_papers.json` export when it has
one (ICLR, ICML), otherwise cut from every completed paper text in the corpus
inventory (`iter_corpus_paper_abstract_units`, `extract_paper_abstract`: text
after the Abstract heading -- at a line start, numbered, run-in, or mid-line
after the author block, never the word "Abstract" in the title -- up to the
first section heading; without a heading, the text after the author block's
last email/URL/affiliation (`abstract_source: after_author_block`), else the
opening text (`leading_text_fallback`); ids canonicalize to
`paper_abstract:<body>`). A run
whose scope includes papers but that produces no paper unit while the corpus
has paper texts is blocked with `llm_full_read_paper_source_missing` -- CVPR
2026's first full read reported complete after reading 0 of its papers.
If a future runner includes topic strategy or GPT-Researcher but omits this
14A evidence check, the runner must fail before it emits a succeeded analysis
receipt.

### Provisional Full-Read Summary Packs

When a conference has completed the LLM full-read lane, operators may run the
experimental `analysis.exhaustive_topic_summarization` owner before promoting
the evidence into formal Stage 7 source-pack construction:

```bash
.venv/bin/python scripts/summarize_llm_full_read_topics.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --provider gemini \
  --model gemini-3.1-pro-preview \
  --provider-cost-approved \
  --summary
```

Leave `--max-input-tokens` unset unless there is a reason to override it. The
default chunk ceiling is tier-aware: it reads the selected model's
`high_tier_input_token_threshold` from
`src/conference_pipeline/costs/pricing_catalog.json` and keeps each chunk
below it (with tokenizer headroom and a prompt-scaffold allowance), because a
prompt above the threshold bills every token at the high-tier rate. On ICML
2026 an explicit `--max-input-tokens 400000` put 49 of 76 calls over the
200k-token tier, costing $81.87 of $84.78. The receipt and plan record the
chosen `max_input_tokens`, `max_input_tokens_source`, and
`max_input_tokens_reason`. Changing the ceiling changes each pair's input
digest, so a `--resume` run re-summarizes pairs that were chunked differently.

The owner builds one corpus for each approved topic/category pair:
`transcripts` from transcript-window units and `papers` from paper-abstract
units. It chunks oversized pairs at the configured token budget, asks the model
for a JSON summary, key points, narrative, notable evidence, caveats, and
analysis-feed bullets, and writes:

- `storage/workflow-runs/<run-id>/07A-exhaustive-topic-summaries.json`
- `storage/analysis/exhaustive-summaries/<run-id>/summary_plan.json`
- `analysis/exhaustive_summaries/<topic_id>__transcripts.json`
- `analysis/exhaustive_summaries/<topic_id>__papers.json`
- matching Markdown summaries and `analysis/exhaustive_summaries/index.json`

Dry-run mode writes only the plan and receipt, with no provider calls. Live
Gemini mode requires explicit provider-cost approval and provider-call
receipts. Local Qwen 122B is allowed only as an experimental smaller-chunk
critic until memory and long-context receipts prove it safe for this workload.
These summary packs are inspection evidence today; they are not yet a formal
production stage and do not replace the deep narrative synthesis contract.

### Topic Evidence Reconciliation

When exhaustive summaries and dual deep-dive GPT-Researcher lanes exist, final
topic analysis must reconcile those parallel evidence layers before slide
synthesis. The normal production entrypoint is the packaged
`analysis.topic_reconciliation_reference_workflow` owner:

```bash
.venv/bin/python scripts/run_topic_reconciliation_reference_workflow.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --resume \
  --summary
```

The workflow packages the sequence future conferences must not skip: 07D
topic reconciliation, immediate 07E deterministic validation, deterministic
mechanical safety fixes, bounded 07G candidate resolution when validation
blocks, 07F broad repair only after unresolved candidate blockers, and final
07E validation. It writes
`07H-topic-reconciliation-reference-workflow.json`; final narrative and slide
synthesis require `final_validation_status: succeeded`, positive final ledger
and citation counts, and zero `final_invalid_reference_count`,
`final_unknown_reference_count`, `final_title_mismatch_count`, and
`final_bare_reference_warning_count`.

The individual owners remain reusable fallback/debug entrypoints. The first
step, `analysis.topic_reconciliation`, can be run directly when an operator
needs to inspect reconciliation before the packaged workflow:

```bash
.venv/bin/python scripts/reconcile_topic_analysis.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --resume \
  --summary
```

The owner reads `analysis/exhaustive_summaries/<topic_id>__papers.md`,
`analysis/exhaustive_summaries/<topic_id>__transcripts.md`, the current
`analysis/narrative_source_packs/<topic_id>.json`, and both global plus
heuristic deep-dive reports. It estimates prompt tokens with a fast tokenizer,
builds a deterministic reference ledger for the prompt, blocks if the prompt
exceeds the configured local-Qwen input ceiling, writes
`analysis/topic_reconciliations/<topic_id>.md` plus JSON sidecars, and injects a
`topic_reconciliation` object into the narrative source pack.

Run `analysis.topic_reference_validation` immediately after reconciliation when
using the individual owners:

```bash
.venv/bin/python scripts/validate_topic_reconciliation_references.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --summary
```

This owner writes `07E-topic-reference-validation.json`,
`analysis/topic_reference_ledgers/<topic_id>.json`, and
`analysis/topic_reference_validation/<topic_id>.json`. It blocks final GPT-5.6 Luna
deep narrative synthesis when a reconciliation brief cites unknown ids, uses
bare ids that need canonicalization, has no canonical ledger entries while
citations are required, or conflates a canonical id with a title that belongs to
another ledger entry. The analysis stage contract requires positive
`ledger_entry_count` and `citation_count`; zero `invalid_reference_count`,
`unknown_reference_count`, and `title_mismatch_count`; and zero
`bare_reference_warning_count` before public narratives or slides consume the
local reconciliation. Slide synthesis must not be the first step that reconciles
exhaustive summaries and deep-dive research.

When validation blocks a local-model reconciliation outside the packaged
workflow, first run the bounded candidate-resolution remediation owner. It lets
deterministic code assemble candidate rows from the ledger, source-pack
evidence, exhaustive summaries, and validator errors; local Qwen may only
choose among those candidates:

```bash
.venv/bin/python scripts/resolve_topic_reference_candidates.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --topic <topic_id> \
  --apply \
  --summary
```

`analysis.topic_reference_candidate_resolution` writes
`07G-topic-reference-candidate-resolution.json`,
`analysis/topic_reference_candidate_resolution/<topic_id>__packets.json`,
decision sidecars, a repaired markdown candidate, and final validation reports.
It must block rather than guess when candidate sets are empty, low-confidence,
ambiguous without direct evidence, or when Qwen selects ids outside the packet.
The candidate can replace `analysis/topic_reconciliations/<topic_id>.md` only
when `final_validation` is clean.

If candidate resolution records unresolved blockers, use the broader
`analysis.topic_reference_repair` remediation owner as a fallback rather than
weakening the validator:

```bash
.venv/bin/python scripts/repair_topic_reconciliation_references.py \
  --workspace conferences/<slug> \
  --run-id <run-id> \
  --mode live \
  --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit \
  --topic <topic_id> \
  --apply \
  --summary
```

The fallback repair owner feeds Qwen the original reconciliation evidence
bundle, focused candidate ledger rows, and the structured validator errors. It
writes `07F-topic-reference-repair.json` and
`analysis/topic_reference_repairs/<topic_id>.json` plus a repaired markdown
candidate. The candidate can replace `analysis/topic_reconciliations/<topic_id>.md`
only when its `repaired_validation` is clean. After any applied repair, rerun
`scripts/validate_topic_reconciliation_references.py`; final narrative and slide
synthesis still depend on the deterministic 07E validation receipt succeeding.

Reusable v1.63 analysis helpers live in
`conference_pipeline.analysis.narrative`. ICLR v1.51 uses the same public
deep-analysis/provenance/slide-brief split for vector-store chunks, query
research, matched-paper provenance, media evidence, and narrative quality.
These gates validate `analysis_citation_manifest.json`,
`analysis/provenance/*`, and `analysis/slide_briefs/*` for proof completeness.
Podcast and synthesis stages consume the deep `markdown_path`; deck stages
prefer `slide_brief_path`.

Tool-specific usage documentation for deep narrative synthesis and reusable
paper matching is registry-owned in the
[Stage Script Registry Tool Usage Index](../reference/stage-owner-audit.md#stage-script-registry-tool-usage-index).
Generated plans should cite those `usage_doc_refs` anchors when selecting
analysis owners.

## Fail-Closed Behavior

Podcast and deck stages fail closed when analysis_manifest.json is missing, schema_version is incompatible, or any declared path does not exist.

This boundary is enforced identically in live, dry-run, and mock modes so
operators cannot accidentally record a successful receipt against a workspace
that lacks a valid manifest. When manifest validation fails during preflight,
the workflow runner persists a `StageReceipt` with `status: "failed"` and
`error: "preflight failed: ..."` before re-raising to the CLI, which converts
the error to `typer.BadParameter` with exit code 2.

## Resume Semantics

Podcast and deck resume skips only succeeded receipts whose recorded manifest_content_sha256 matches the current manifest.

Re-running `workflow run --stages analyze` preserves existing analysis artifacts
and regenerates the manifest with a new `generated_at`; the `content_sha256`
stays stable when the analysis content did not change, so a resume hit is still
the correct outcome. The whole-file `manifest_sha256` will differ between
re-emits because `generated_at` changes, which is why resume deliberately binds
to the content variant.

## Backfilling Legacy Workspaces

Workspaces that predate Phase 12 can generate a manifest without running a full
live analyze sweep:

```bash
python3 scripts/backfill_analysis_manifest.py --workspace conferences/neurips2025
```

The backfill script delegates to the same `emit_analysis_manifest` function the
live analyze step uses, so a backfilled manifest is schema-equivalent to a
live-emitted one. The script exits with code 0 on success, 1 on post-write
validation failure, and 2 on an invalid workspace. It also emits a `WARNING`
line for every `analysis/*.md` file that does not match the `NN_slug.md`
pattern.

## Smoke-Test Proof Consumer

The Phase 14 smoke-test proof reads the analysis manifest through the same
workflow boundaries as podcast and deck stages. Run:

```bash
conference-pipeline small-neurips-smoketest --workspace conferences/neurips2025 --mode dry-run --run-id phase14-smoketest --summary
```

The proof report is written to
`conferences/neurips2025/storage/workflow-runs/<run-id>/small-neurips-smoketest-report.json`.

## Troubleshooting

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `Missing analysis_manifest.json` | Workspace has not been analyzed yet, or the manifest was deleted | Run analyze, or run the backfill command above |
| `schema_version N is not supported` | Manifest was written by an older or incompatible analyze stage | Re-run analyze to emit a v1 manifest |
| `markdown_path ... does not exist` | Manifest declares a file that has been moved or deleted | Re-emit the manifest |
| `content_sha256 mismatch` | Manifest has been edited after emission | Re-emit the manifest |

## Related Documentation

- [Analysis Manifest Schema](../design/analysis-manifest-schema.md)
- [Workflow CLI Reference](../cli.md)
- [Pipeline Workflow](../workflows/pipeline-workflow.md)
