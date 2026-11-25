# Provider-Substitution Waiver

A `provider_substitution_waiver.json` is the ONLY governed path around an
approved provider contract row. A bare manual-ack `UnblockReceipt` does NOT
count as a governed substitution -- it acknowledges a blocker, it does not
authorize swapping the provider or model a stage's `runtime_contract`
equals-checks against.

## Schema

```json
{
  "task_id": "analysis.deep_narrative_synthesis",
  "from": {
    "provider": "openai",
    "model": "gpt-5.6-sol"
  },
  "to": {
    "provider": "openai",
    "model": "gpt-5.6-terra"
  },
  "approver": "operator-id-or-name",
  "rationale": "why the approved contract could not be honored",
  "cost_impact": "expected cost delta versus the approved contract",
  "expiry": "ISO-8601 timestamp after which the waiver no longer applies"
}
```

Fields:

- `task_id`: the Task Model Defaults / model-governance task id the waiver
  applies to.
- `from`: the approved `provider`/`model` pair from the contract row being
  waived.
- `to`: the substituted `provider`/`model` pair actually used.
- `approver`: who approved the substitution.
- `rationale`: why the approved contract could not be honored.
- `cost_impact`: the expected cost delta from the substitution.
- `expiry`: when the waiver stops applying; a stage may not rely on an
  expired waiver.

## Enforcement

Every provider-backed stage's `runtime_contract` equals-checks the receipt's
actual `provider` and `model` fields against the approved PC-* contract list
and requires `provider_calls_made: true` (and `boundary_method` where
applicable), reusing the existing `equals`/`one_of`/`not_equals` predicate
vocabulary in `stage_contracts.py` -- no new predicates are introduced. A
receipt that does not match the approved contract AND does not carry a valid,
unexpired `provider_substitution_waiver.json` for that `task_id` blocks with
`unauthorized_provider_substitution`. An invalid or expired credential blocks
with `credential_invalid`.
