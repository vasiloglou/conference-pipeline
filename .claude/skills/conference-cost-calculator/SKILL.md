---
name: conference-cost-calculator
description: Estimate, compare, or reconcile paid provider costs for this conference pipeline using the repo's canonical cost registry.
---

# Conference Cost Calculator

Use this skill when the user asks for provider cost estimates, model scenario
comparisons, or cost reconciliation for a conference workspace in this repo.

## Workflow

1. From the repository root, run the canonical estimator:

   ```bash
   .venv/bin/python -m conference_pipeline.costs estimate \
     --workspace conferences/iclr2025 \
     --scope full \
     --scenario base \
     --json
   ```

2. For low/base/high model comparisons, run:

   ```bash
   .venv/bin/python scripts/estimate_pipeline_costs.py \
     --workspace conferences/iclr2025 \
     --compare-models \
     --json
   ```

3. After a production run, reconcile local receipts with:

   ```bash
   .venv/bin/python -m conference_pipeline.costs actual \
     --workspace conferences/iclr2025 \
     --scope full \
     --json
   ```

4. For the TOTAL conference cost — canonical workspace receipts PLUS any
   misplaced repo-root receipts you can confidently attribute (see
   docs/reference/stage-owner-audit.md#verification-conference-total-cost) —
   run the total-cost command:

   ```bash
   .venv/bin/python scripts/compute_conference_total_cost.py \
     --workspace conferences/iclr2025 \
     --attribute-segment run_icml2026_production \
     --since 2026-07-01T00:00:00Z \
     --until 2026-07-31T23:59:59Z \
     --json
   ```

   Omit `--attribute-segment`/`--since`/`--until` to see everything reported
   as `misplaced_unattributed` (excluded from `total_usd` by design — never
   silently claim repo-root spend for a conference without an explicit
   attribution filter). The command is read-only: it never moves, deletes,
   or rewrites receipt files, and never raises the budget cap — it only
   reports `budget_cap.breach` against the existing guardrail.

5. For OCR/image parsing model experiments, dry-run first:

   ```bash
   .venv/bin/python -m conference_pipeline.ocr_benchmark run \
     --workspace conferences/iclr2025 \
     --sample-size 12 \
     --budget-cap-usd 30 \
     --per-call-timeout-seconds 60 \
     --json
   ```

   Add `--live` only after provider approval. The receipt compares cheaper
   OpenAI, Anthropic, and Gemini image parsers against an OpenAI authority.

6. Summarize total estimated cost, top provider totals, largest task families,
   warnings, and whether any provider receipts replaced estimates.

## Notes

- Pricing lives in `src/conference_pipeline/costs/pricing_catalog.json`.
- Task assumptions live in `src/conference_pipeline/costs/task_specs.yaml`.
- Provider receipts and invoices are authoritative after a run.
- Local downloads, browser rendering, `ffmpeg`, Tesseract, and local PDF text
  extraction are `$0 API cost` line items.
