# Conference Analysis Playbook

Use this playbook when starting or proving a conference workspace. It describes
the reusable lifecycle; conference context belongs in formal GSD planning docs
under `.planning/milestones/`.

## Recommended Lifecycle

New conference analysis is a two-step planning flow followed by approved live
execution:

1. **Analysis specification:** run `$plan-conference-analysis` from a
   PRD-style conference brief. This creates or updates the conference-specific
   planning spec: milestone context, requirements, roadmap, state, workspace
   README, support labels, provider contracts, and production handoff.
2. **Production planning:** after the analysis spec is finalized, run
   `$plan-conference-production` against that spec. This creates live-only GSD
   production work that actually produces the conference analysis.
3. **Execution:** run the generated production phases only after approval,
   source/auth readiness, provider-cost gates, backup posture, receipt paths,
   and fail-closed validation gates are in place.

Do not archive the analysis-spec milestone before generating the production
plan; the production planner consumes the finalized analysis spec. After the
production plan is generated, audited, and finalized, the completed analysis
planning milestone can be archived.

Production work should usually live in a new milestone, for example
`vX.Y-<Conference> Production`, rather than reopening the completed
dry-run/mock planning milestone. A separate workstream is acceptable when a
project deliberately keeps one milestone umbrella, but keep the planning-proof
definition of done separate from live production completion.

Dry-run, mock, scaffold, readiness-only, metadata-only, and `needs live proof`
artifacts may inform planning, but they must not satisfy live production
completion.

## 1. Create Or Inspect The Workspace

Start from `conferences/TEMPLATE/` or an existing workspace. The minimum
contract is:

- `README.md`
- `config.yaml`
- `storage/storage_manifest.json`
- lane directories or planned lane paths for `analysis`, `decks`, `audio`,
  `exports`, `agent-work`, `cache`, and `legacy`

Record the conference milestone name in a planning-owned context file named
`.planning/milestones/<version>-<Conference>-CONTEXT.md` using
`{platform_version}-{conference name}`.

When using `$plan-conference-analysis`, the skill scaffolds this planning-owned
context and the workspace README from the approved brief. Manual workspace
edits are still allowed, but durable conference assumptions should remain in the
milestone context.

## 2. Establish Local Context

Fill in:

- source URLs and auth method
- session types and expected site format
- analysis profile and topic strategy
- supported workflow stages by live, dry-run, and mock support
- proof status and known limitations

Keep this information in the formal planning context. The conference workspace
README may link to it, but should not own GSD milestone truth.

The conference brief for `$plan-conference-analysis` should also include target
outputs, proof mode, provider preferences, provider-cost approval posture,
storage/share targets, acceptance criteria, platform version, and conference
name. If exhaustive extraction is in scope, include the local LLM endpoint
label, loopback base URL, and model name before planning production work.

## 3. Generate The Analysis Specification

Run `$plan-conference-analysis` with the conference brief. The generated draft
must pass:

- generated-plan audit
- workspace gitignore policy generator/check
- stage owner registry audit
- unified stage script registry verifier

Finalize only after review. The finalized analysis spec should include an
explicit `plan-conference-production` next-step handoff and a structured
production handoff with provider contracts, support labels, source inventory
requirements, exclusion/interest-group gates, backup posture, and receipt
boundaries.

The analysis specification proves planning readiness. It does not prove that
live source acquisition, provider calls, analysis, podcast, presentation,
release, backup, cleanup, or publication has happened.

## 4. Generate Live Production Work

After the analysis spec is finalized, run `$plan-conference-production` against
the finalized spec root. By default, production generation should emit a
multi-phase live production milestone. Use a one-phase execution plan only when
all in-scope owner commands already exist and the caller explicitly accepts
that shape.

The generated production draft must pass:

- production-plan audit
- workspace gitignore policy check
- stage owner registry audit
- unified stage script registry verifier

Production output is live-only. Non-live analysis-spec evidence can be
referenced as historical context or rejected evidence, but cannot close
production phases.

## 5. Validate Shared Contracts

List available stages:

```bash
.venv/bin/python -m conference_pipeline.cli workflow list --workspace conferences/<slug>
```

Backfill or create `analysis/analysis_manifest.json` before output-stage
previews:

```bash
python3 scripts/backfill_analysis_manifest.py --workspace conferences/<slug>
```

For production-quality analysis, prepare narrative source packs under
`analysis/narrative_source_packs/*.json` before the analyze stage. Each pack
should include the topic id/title, synthesized findings, source chunks,
research id, extraction ids, and matched-paper evidence. When packs are
present, `analyze` writes deep `analysis/<topic>.md` files,
deck-focused `analysis/slide_briefs/<topic>.md` sidecars,
`analysis/provenance/<topic>.md`, `analysis/provenance/<topic>.json`, and
`analysis/analysis_citation_manifest.json`. Production deep synthesis uses the
model-governed `analysis.deep_narrative_synthesis` task by default.

## 6. Run A Generic Non-Live Proof

Use the generic smoke-test command before live provider work:

```bash
conference-pipeline conference-smoketest --workspace conferences/<slug> --stages analyze,deck --topic 01 --mode dry-run --run-id <run-id> --summary
```

The command writes:

- `conferences/<slug>/storage/workflow-runs/<run-id>/summary.json`
- `conferences/<slug>/storage/workflow-runs/<run-id>/conference-smoketest-report.json`

Use `analyze,podcast,deck` only when the conference manifest declares the
topic-level ranks and audio paths required by the podcast adapter.

## 7. Estimate Provider Costs

Before approving paid provider work, run the full-pipeline estimator:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/<slug> \
  --scope full \
  --scenario base \
  --json
```

For model scenario comparisons:

```bash
.venv/bin/python scripts/estimate_pipeline_costs.py \
  --workspace conferences/<slug> \
  --compare-models
```

Record the estimate path or output summary in the planning-owned conference
context. ASR/OCR-only estimates are partial evidence; provider approval should
use the full-pipeline estimate.

## 8. Execute And Promote Support Deliberately

After a dry-run proof passes:

- update the planning-owned context file with the proof report path
- update the capability catalog only if a new reusable capability exists
- add docs-contract tests for new public command wording
- run live stages only after source URLs, auth, provider credentials, and
  conference-specific limitations are documented
- record live receipts before promoting any stage from planned or needs-live
  proof to live-supported

## ICLR 2025 v1.2 Proof Command

The v1.2 proof uses:

```bash
conference-pipeline conference-smoketest --workspace conferences/iclr2025 --stages analyze,deck --topic 01 --mode dry-run --run-id phase20-iclr2025-smoketest --summary
```
