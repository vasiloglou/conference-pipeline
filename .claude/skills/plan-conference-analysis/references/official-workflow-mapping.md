# Official Workflow Mapping

Generated conference analysis roadmaps must include the official workflow
mapping before generated phase entries.

| Step ID | Official workflow step | Status | Phase | Support label | Receipt boundary | Rationale | Source |
|---------|------------------------|--------|-------|---------------|------------------|-----------|--------|

Allowed statuses are exactly `mapped`, `blocked`, and `out_of_scope`.

Row fields are exactly `step_id`, `official_step`, `status`, `phase_name`,
`support_label`, `receipt_boundary`, `rationale`, and `source_ref`.

The `source_ref` field must be `docs/workflows/pipeline-workflow.md#overview`
for `WF-01` through `WF-14` and `WF-14a` through `WF-14c`.

Support labels must be exactly `live`, `dry-run`, `mock`, `unsupported`,
`planned`, `dry-run/mock only`, or `needs live proof`.

Each official workflow step must be mapped, blocked, or out of scope. Mapped
steps need a named phase and receipt boundary. Blocked and out-of-scope steps
need a rationale that explains the truthful support boundary without inventing
conference-specific proof.
