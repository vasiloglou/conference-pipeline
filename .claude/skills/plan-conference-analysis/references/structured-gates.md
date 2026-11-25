# Structured Gates

`production_handoff()` emits a `structured_gates` field (and
`structured_gates` is part of `PRODUCTION_HANDOFF_FIELDS`) alongside the
existing prose `PRODUCTION_WORKFLOW_GATES` table -- it is additive, not a
replacement. Prose gate rows remain the human-readable narrative; this field
is the machine-checkable counterpart.

## Schema

A list of objects:

```json
{
  "gate_id": "PG-TOPIC-STRATEGY",
  "receipt_path": "storage/analysis/topic-strategy/<run-id>/topic_strategy_receipt.json",
  "field_checks": [
    {"path": "status", "equals": "succeeded"}
  ]
}
```

- `gate_id`: the gate identifier used throughout the roadmap/production
  workflow gate tables (e.g. `PG-TOPIC-STRATEGY`).
- `receipt_path`: the receipt this gate checks. `PG-TOPIC-STRATEGY`,
  `PG-QUERY`, `PG-GPT-RESEARCHER`, and `PG-PAPER-MATCHING` must each have a
  **distinct** `receipt_path` -- collapsing them to one receipt (the
  historical defect) makes per-gate checking impossible.
- `field_checks`: a list of `{path, predicate...}` objects using the same
  `equals`/`one_of`/`not_equals`/`equals_path`/`truthy` predicate vocabulary
  as `stage_contracts.py` `_check_required_check` -- no new predicates.

## Consumption

`plan-conference-production`'s `validate_structured_gates()` (called from
`validate_analysis_handoff`) validates that `structured_gates` is present,
each entry has `gate_id`/`receipt_path`/`field_checks`, and that the four
named analysis-phase gates carry distinct receipt paths. The production
scaffold then imports the gate field-checks verbatim into its own
`approval-summary.json` under `structured_gates` for propagation into the
corresponding runtime_contract / gate entries.
