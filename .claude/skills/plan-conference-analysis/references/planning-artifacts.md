# Planning Artifacts

Default draft root:

- `.planning/phases/30-plan-conference-analysis-skill-packaging-and-uat/drafts/<conference-slug>/`

Generated conference context:

- `.planning/milestones/<platform_version>-<Conference>-CONTEXT.md`

The context packet must use `{platform_version}-{conference name}` as the
conference milestone display name.

Workspace scaffold:

- `conferences/<slug>/README.md`

Active GSD planning files, finalized only after approval:

- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`

If the target workspace or milestone context already exists, require an
update-vs-new-version decision. Do not silently merge into existing state.
