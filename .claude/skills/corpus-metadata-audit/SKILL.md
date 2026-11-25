---
name: corpus-metadata-audit
description: Audit a conference workspace's on-disk corpus for paper title and author coverage per source family (main papers, workshop papers, artifacts), scanning real generated artifacts rather than plan text. Run before closing any conference milestone, and any time corpus-inventory, indexing, narrative, or ledger code changes; referenced as a completion gate by the conference planning skills.
---

# Corpus Metadata Audit

## 1. Why an empty-metadata corpus is a defect

The presentation content-sufficiency gates (`MIN_SOURCES`, `people`) count
distinct sources and named people. A corpus with no structured author or title
metadata makes every deck look like a per-topic sparsity problem when the real
defect is upstream: nothing ever extracted or joined the metadata in the first
place. This happened for real — CVPR2026's first live Phase 10 run blocked all
14 decks this way: 0 of 653 reference-ledger entries had any resolved author,
and every `papers`-family entry had an empty title. Full evidence:
`.planning/workstreams/cvpr-2026/phases/10-deck-and-presentation-production/10-BLOCKER-corpus-metadata-gap.md`.
Use this skill as a corpus-integrity gate **before closing a conference
milestone** (and any time corpus-inventory, indexing, narrative-pack, or
ledger code changed).

## 2. Run the automated gate (required)

```bash
python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug>                                       # human report, exit 1 on findings
python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug> --json                                 # machine-readable
python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug> --write-receipt storage/workflow-runs/<run-id>   # + coverage receipt
```

It fail-closes (non-zero exit) when a gated family's measured title or author
coverage — read live from `storage/corpus/corpus_inventory.json`, the papers
vector-store chunk JSONL, `analysis/topic_reference_ledgers/*.json`, and
`decks/*/speaker_author_work_ledger.json` — falls below its justified floor,
or when a required ledger artifact is missing (`ledgers_missing`), or when a
workspace floor override is not accompanied by a justification
(`floor_override_unjustified`). A milestone must not close while this gate
fails.

## 3. Per-family interpretation

| Family | A shortfall means | Gated? |
|---|---|---|
| `papers_main` | The main-conference source-discovery join (or its PDF-header fallback) is not landing title/authors onto the corpus inventory and propagating them through the vector store and ledgers — the exact Phase 10 root cause. Investigate the join first (`paper_source_metadata.resolve_main_paper_metadata`), then propagation (`narrative.compact_chunk_evidence`, `topic_reference_validation.build_reference_ledger`). | Yes |
| `papers_workshop` | The workshop-receipt lift is not reaching every workshop paper, or propagation through the same chain as main papers is broken. | Yes |
| `artifacts` | Video-caption transcripts and similar non-paper artifacts are **structurally author-less** — presenters rarely state a full co-author byline on camera — so this is reported (`authors_classification: structurally_authorless`) but never gated on authors. Its title coverage is still measured and printed for visibility, since a real title-extraction gap here is a separate, legitimate finding even though it doesn't fail the audit. | No |

## 4. Floors and the no-silent-lowering rule

Every entry in `COVERAGE_FLOORS` carries a `justification` string naming the
exact execution-time measurement (numerator, denominator, date, and the
command that produced it) the floor was derived from — never a bare
assertion. Lowering a floor for one workspace requires a per-workspace
override in that workspace's `config.yaml`
(`corpus_metadata_coverage.floor_overrides.<family>.{title_floor,
authors_floor,justification}`) carrying a **non-empty justification**; the
audit prints the accepted override's previous value, new value, and
justification in both human and `--json` output so a lowered floor is always
operator-visible, never buried. An override below the platform default with
no justification is rejected outright with the finding code
`floor_override_unjustified` and a non-zero exit. A silently lowered floor —
one without a recorded, printed justification — is rejected evidence for
milestone closure, exactly like an unmeasured coverage claim.

## 5. What this gate does and does not guarantee

This gate proves that a **specific workspace's on-disk artifacts**, measured
at the moment it ran, meet the coverage floor across the whole extraction →
join → vector-store → narrative → ledger propagation chain. It is a runtime
scan of generated data, the same strength as `conference-hardcoding-audit` and
`model-registry-audit`. It does **not** prove that a future conference's live
production run will produce a corpus meeting the same floor — that is a
weaker, prospective guarantee, and it is what the planning-skill enforcement
in `plan-conference-production` (plan 10.1-08) checks instead, by verifying
plan text describes the right stages. RESEARCH documented a real case where a
plan-text audit reported a full pass against a plan whose live execution still
failed — a runtime-scanning gate and a text-presence gate are different
strengths and must never be conflated or substituted for one another.

## 6. Fix pattern for a real finding

The shared module is
`conference_pipeline.indexing.paper_source_metadata` (join resolvers,
`classify_record_family`, the PDF-header fallback parser). A missing-authors
finding usually points at one of three allowlist/propagation locations:

1. `src/conference_pipeline/source_discovery.py` — is
   `field_evidence.authors` populated on the program item (main papers), and
   has `scripts/rederive_source_discovery_authors.py` actually been run
   (non-dry-run) against this workspace's frozen inventory?
2. `src/conference_pipeline/indexing/corpus_vector_store.py` — does the chunk
   allowlist carry `authors` through to `source_context`?
3. `src/conference_pipeline/analysis/narrative.py` /
   `topic_reference_validation.py` — does `compact_chunk_evidence` /
   `build_reference_ledger` preserve `authors`/`paper_title` instead of
   stripping them during compaction?

## Completion rule

A conference milestone may close only when
`python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug>`
exits 0 for the workspace under close, alongside (not instead of)
`conference-hardcoding-audit` and `model-registry-audit`. Record the coverage
receipt path and the audit result in the milestone completion notes.
