# Project rules: conference intelligence platform

The skills `start-milestone`, `process-milestone` and `process-phase` are synced from
dev-tools (`.devtools.toml`) and are never edited here: `sync.py check` fails if they are.
This project's own rules for those skills go in this file, which the sync never touches.

The rules below that came out of the move from GSD were ruled by the owner on
2026-11-13, after independent reviews by Codex (`gpt-6-astra`) and Claude Opus 5.5;
`.devtools-migration.md` records each one.

## Roles

- The **owner** is the repository owner. The owner approves plans, specs and closes,
  authorizes paid runs and publication, and holds the **architect** role that
  `architect-overrides.md` refers to. Nobody else holds either role unless the owner
  names them in a milestone's plan-gate record.

## Planning folder and history

- The planning folder is `.planning/`. It exists; there is no scaffold step to run.
- **Maintained**, read and kept current:
  - `.planning/PROJECT.md`, the project one-pager the skills read first;
  - the context packets `.planning/milestones/*-CONTEXT.md`, authoritative for their
    conference: source URLs, assumptions, proof status, scoped authorizations and close
    records;
  - the folders this workflow writes, `.planning/milestones/<milestone-dir>/`.
- **History, not operative**: everything else GSD wrote in `.planning/` up to the
  v5.2 close on 2026-11-14, the last work done under GSD. That includes the root `STATE.md`, `ROADMAP.md`,
  `MILESTONES.md`, `RETROSPECTIVE.md`, `config.json`, `active-workstream`, `phases/`,
  `drafts/`, `proposals/`, `research/`, `codebase/`, `notes/`, `quick/`, `debug/`,
  `forensics/` and `logs/`; every plan in `.planning/workstreams/`; and in
  `.planning/milestones/`, every file and folder that is not a context packet or a
  folder of this workflow (`v*-ROADMAP.md`, `v*-REQUIREMENTS.md`, `*-MILESTONE-AUDIT.md`,
  `*-TRACEABILITY.md`, `*-POST-DOWNLOAD-AUDIT.md`, `*-PRODUCTION-EXECUTION.md`, briefs,
  `v*-phases/`, `ws-*/` and the rest).
  - Never rewrite history, never take a GSD `STATE.md`, `ROADMAP.md` or
    `active-workstream` as the current position, and never write new work into it.
    The root `STATE.md`, `ROADMAP.md` and `MILESTONES.md` open with a notice that says
    so; their paths stay, because tests and records cite them.
  - Never run `src/conference_pipeline/iclr2025_live.py` with `--write-gsd-artifacts`:
    it rewrites the root `.planning/STATE.md` and `ROADMAP.md` whole
    (`update_planning_status`), erasing the notice.
- Nothing carries over by default. When `start-milestone` harvests carryovers, it reads
  the records of the milestones the new one builds on (a context packet's close record
  and follow-ups, `.planning/PROJECT.md` "Active", and every `deferred-items.md` under
  `.planning/`, in workstream phase folders, root `phases/` and `milestones/v*-phases/`)
  and gives each open item a disposition in the obligations table: taken into this
  milestone, left for later with a reason, or closed. GSD's files never become the new
  plan.
- GSD's commands (`/gsd-*`) are no longer the project's workflow.

## Milestone names and folders

- A milestone's **display name** is `{platform_version}-{conference name}`, with a
  suffix for its kind where a conference has more than one milestone
  (`v1.0-NeurIPS 2025`, `v4.0-ICML 2026 Live Production`); platform milestones may stay
  conference-neutral (`docs/source-of-truth.md`).
- Its **folder**, `<milestone-dir>`, is derived from the display name once: replace each
  run of characters other than ASCII letters, digits, `.` and `-` by a single `-`, then
  strip `-` from both ends. `v6.0-NeurIPS 2026` gives `v6.0-NeurIPS-2026`;
  `v6.0-NeurIPS 2026 Live Production` gives `v6.0-NeurIPS-2026-Live-Production`;
  `v3.2-AI4Physics Through NeurIPS, ICLR, and ICML` gives
  `v3.2-AI4Physics-Through-NeurIPS-ICLR-and-ICML`.
  - It must not equal an existing folder or file stem in `.planning/milestones/`,
    compared without regard to case; if it would, stop and ask the owner for a name.
  - It is fixed when the folder is created. If the display name changes later, the
    folder keeps its name; the plan records the display name.
  - Old flat files keep their names, whichever style they use.
- The `milestone:` field of every record holds `<milestone-dir>`, not the display name.
- The `notes/` paths the skills name (`notes/<milestone>-plan.md`,
  `notes/phase-N-spec.md`, review notes) are inside the milestone's folder:
  `.planning/milestones/<milestone-dir>/notes/`. Phase numbers start again in every
  milestone and several milestones run at once, so a shared `notes/` would collide.

## Milestones in this project

- A conference usually takes two milestones: the analysis specification, then live
  production. Several may run at once, each on its own branch or worktree.
- Each conference milestone has its own context packet,
  `.planning/milestones/<display name>-CONTEXT.md`, required reading before any
  conference-specific change (`CLAUDE.md`). A later milestone of the same conference
  reads the earlier packets and writes its own. It never inherits an authorization or a
  delegation recorded in another milestone's packet or records.
- At each milestone close, update `.planning/PROJECT.md`: move shipped requirements to
  Validated, add candidate requirements for the next milestone, re-check the core value
  and out-of-scope assumptions, and update the key decisions with outcomes. (Carried from
  PROJECT.md's own "Evolution" section.)

## The project's own workflow skills

A phase record's `skill:` field names the skill when one of these owns the work.

- **Planning a conference milestone.** `plan-conference-analysis` and
  `plan-conference-production` are the domain planners. They still emit GSD-format
  files, so they run in draft mode inside the milestone's folder, never with
  `--finalize`, and `start-milestone` writes the one canonical plan from their drafts.
  The exact steps, checks and copies are in `references/project-conference-planning.md`,
  to be read before planning a conference milestone. A platform
  milestone (not tied to one conference) is planned by `start-milestone` alone.
- The `conference-*` skills own their pipeline stages (blueprints, figures, narration,
  cost, workshop paper discovery, talk harvest and the rest).
- `conference-hardcoding-audit`, `model-registry-audit`, `model-catalog-upgrade` and
  `corpus-metadata-audit` own their audits.

## Where commands run

- Commands here name `.venv/bin/python` and `.env` at the root of the checkout they run
  in. Both are ignored by git, so a new git worktree has neither.
  - In a worktree, run Python as `<main checkout>/.venv/bin/python` (the main checkout
    is the one whose `.git` is a folder). Never link or copy a `.venv` into a worktree:
    `.gitignore` ignores the folder `.venv/`, not a link of that name.
  - `.env` lives at the root of the checkout that runs live providers (`CLAUDE.md`). A
    worktree that runs them gets its own `.env` from the developer; it is never
    committed, and its values are never copied into a spec, record or review.
- Ignored workspace data (deck ledgers, receipts, media under `conferences/<slug>/`)
  exists only in the checkout that produced it. A run that has to see it, including the
  full suite below, runs in that checkout.

## Test and gate commands

- Test command: `.venv/bin/python -m pytest -q <tests>` (Python 3.12). An ordinary phase
  runs the focused selection (`verification-selection.md`).
- Lanes set apart: tests marked `live_llm` or `local_e2e` are skipped unless
  `RUN_LIVE_LLM=1` or `RUN_LOCAL_E2E=1` is set (`tests/conftest.py`); they need live
  endpoints and never join an ordinary gate. Tests marked `legacy_http_mock` are
  quarantined regression tests, but nothing deselects them: they run in every
  selection that includes their files.
- **Full suite at milestone close (owner ruling, 2026-11-13).** The broad lane is the
  ordinary offline suite, `.venv/bin/python -m pytest -q -n 12 -p no:cacheprovider -rfs tests`
  (two to three minutes; `pytest-xdist` is installed). The live lanes above stay out.
  It is required at every milestone close, not at every phase; the reason is that under
  GSD the closing full run of v5.0-CVPR 2026 phase 10.3 caught a regression focused runs
  had missed.
  - **Selection.** `verification-selection.md` runs the broad lane only on an explicit
    owner selection (the project has no CI job for it), and delegation never authorizes
    it (`process-milestone`, `process-phase`). So the plan names two runs, the baseline
    and the close run, and **the owner's own ruling in the plan-gate record selects
    both for that named milestone**, even when the rest of the milestone runs with
    `auto=true`. A delegated plan approval does not select them; without the owner's
    ruling the close cannot be recorded and the milestone stops there.
  - **Baseline.** The plan says "baseline: pending". The first record after the plan gate
    (`002-baseline.md`) is the baseline run, at the plan-gate commit, before any phase
    changes code.
  - **Where.** Both runs happen in the same checkout: the one where the milestone's work
    and its ignored workspace data live, with its own working `.venv`. Never in a
    throwaway worktree: there, data-dependent tests skip and `.venv`-dependent tests fail,
    so the run proves less than it claims. (A throwaway worktree is still how a known red
    is proven, as `review-and-record.md` says.)
  - **Comparison.** Report the real exit code, never a non-zero run as green. Compare
    with the baseline by node id and cause, for failures **and skips**. A failure or skip
    that is not in the baseline, or that has another cause, is the milestone's to fix.
    Known failures are not an allowance: each is a carryover the plan disposes of.
  - In phase records, the `slow_cadence` block names this command with
    `trigger: "milestone close, selected by the owner in the plan-gate record"`.
  - Tests whose result depends on where they run:
    `tests/test_orchestrator_characterization_cvpr.py::test_cvpr_podcast_success_receipt`
    and
    `tests/test_workspace_gitignore_policy.py::test_starter_workspace_gitignore_policy_checks_pass`
    fail in a checkout without its own `.venv`;
    `tests/test_presentation_source_inventory.py::test_live_transcript_join_leaves_no_placeholder_title_on_real_workspace`
    skips without the CVPR 2026 deck ledgers. At `main` 9f3a6701c, in a worktree with
    the main checkout's interpreter, the run gave 3 failed, 8848 passed, 43 skipped; the
    third failure,
    `tests/test_orchestrator_characterization_icml.py::test_icml_stage_table_and_cli_surface`,
    is real golden-receipt drift (an unexpected `--notebooklm-shared-notebook` option).
- Gates that must exit 0:
  - `PYTHONPATH=src .venv/bin/python scripts/audit_conference_hardcoding.py`: before
    closing any conference milestone, and whenever reusable code changed; record its
    result in the close record (`conference-hardcoding-audit`). Stage contracts verify
    structure and lineage, never run counts.
  - `PYTHONPATH=src .venv/bin/python scripts/audit_model_registry_hardcoding.py`: before
    closing any milestone that touches model selection.
  - `PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py --skills-root .claude/skills --docs-root docs --strict`:
    when a phase changes skills, stage scripts or docs.
  - `PYTHONPATH=src .venv/bin/python scripts/audit_private_markers.py`, with the
    operator's `.env` sourced so `CONFERENCE_PIPELINE_PRIVATE_MARKERS` is set: at every
    milestone close and before any public export (`docs/operations/public-release-scrub-policy.md`).
- Lint: ruff is configured in `pyproject.toml`, but the tree does not pass it
  (2,653 findings at 9f3a6701c), so it is not a gate. A phase may run
  `ruff check <changed files>`.
- Workflow validation starts in dry-run or mock mode unless the task requires live
  providers. A workflow is not called live-supported for a new conference unless its
  context packet and proof receipts show it.

## Review instrument and plan-review bar

- The plan-review bar is the shipped default: one independent reviewer in a different
  tool. Step 4's reviewer must also run in a **different harness** from the one driving
  the work (`agent-delegation.md`), and a fresh session of the same tool is not one. So
  the reviewer depends on what drives the milestone:
  - **Claude Code drives** (the usual case): the reviewer is Codex with `gpt-6-astra`,
    read-only.
  - **Codex drives**: the reviewer is Claude Code in print mode, read-only, with a
    Claude model the owner names (default: the CLI alias `opus`, the current Opus).
  The same rule applies to the plan, every phase's diff (`process-phase` step 4, required
  for every phase), the milestone close, and any spec the owner asks to have reviewed.
- The instrument is the shipped review helper, `scripts/review_tool.py` in this skill
  (`review-tool.md` beside this file says what it does). Its settings come from
  `.devtools.toml`: `reviewer = "auto"` picks the harness by the rule above, with Codex
  on `gpt-6-astra` at effort `high` and Claude on `opus`. The owner may ask for `xhigh`,
  as some GSD-era reviews did: change `codex_effort` there and apply the sync again,
  rather than editing the synced settings file.
- Each round is one folder, `.planning/milestones/<milestone-dir>/notes/reviews/<subject>-r<N>/`
  (`plan-r1`, `phase-3-r1`, `close-r1`, `pr-42-r1`). From the checkout under review:

  ```bash
  python3 .claude/skills/process-phase/scripts/review_tool.py brief <round> --subject phase --base <the phase's base commit> --question "<the question>" --claim "<a criterion>"
  python3 .claude/skills/process-phase/scripts/review_tool.py run <round>
  python3 .claude/skills/process-phase/scripts/review_tool.py status .planning/milestones/<milestone-dir>/notes/reviews
  ```

  `brief` states the question without the author's conclusion, lists each file with its
  SHA-256 and forbids credential files; `run` checks the bytes and the reviewer's login,
  then starts it in the background with stdin closed. Fill the round's `FINDINGS.md`,
  run `sanitize <round>` before committing the folder, and cite a review in a record
  only for the files `status` calls reviewed.
- Preconditions: the reviewer's CLI is installed and logged in. `run` checks
  `codex login status` itself; for Claude, a working `claude -p`. Each uses its own
  login; no key from `.env` is passed to it. If a precondition fails, `run` launches
  nothing: tell the owner.
- A PR review (`--subject pr --pr N`) reads the diff at the PR's head; the `review-pr`
  skill walks its steps. `post <round>` shows the comment; posting it with `--yes` is
  publication and needs the owner's approval for that PR.
- Reviews before 2026-11-15 (the v6.0 plan review among them) used a hand-written brief
  and the commands this section named then; their records and notes stand as written.
- Presentation and narration quality reviews (`conference-presentation-blueprint-review`,
  `conference-narration-review`, `conference-rough-cut-review`) are product stages,
  owned by those skills. They are not the review of a phase's code.

## Paid work, publication and opt-in gates

- A paid provider run needs a current cost estimate for the named run
  (`conference-cost-calculator`) and an authorization that covers it, before it starts.
- An authorization already recorded for a milestone keeps its scope exactly as
  written. v5.0-CVPR 2026 Live Production has one: a current, complete, finite
  named-run estimate at or below USD 800 needs no further authorization, and anything
  above, unpriced, incomplete or changed in scope needs the operator's explicit
  approval (its context packet, the "Provider and cost gates" rule; enforced by the
  production runner). It applies to that milestone only and does not pass to any other.
- Without such a recorded authorization, the owner approves each paid run for a named
  scope. `auto=true` adds no spending authority beyond what is recorded.
- Publication (a Drive publish, a public release, any upload outside the repository)
  needs the owner's explicit approval for that release.
- Opt-in gates (`architect-overrides.md`): the project defines none beyond the full
  suite, which is a required close gate here, selected as above.

## Standing design rules

These bind every phase and live in the project's instruction file, not here:
`CLAUDE.md`, sections "Working Rules" and "Model Governance", and
`docs/source-of-truth.md` for where new guidance belongs. A spec's rule ledger cites
them by section.

## Record evidence

- `storage/storage_manifest.json`, `analysis/analysis_manifest.json` and
  `storage/workflow-runs/<run-id>/summary.json` are machine-readable contracts. For a
  pipeline stage, the evidence a record cites is the stage receipt; `summary.json` is
  display state and does not prove a stage ran (`.planning/PROJECT.md`, Key Decisions).

## Component index

The project keeps one, built in v6.0-Public-Release Phase 1
(`.planning/milestones/v6.0-Public-Release/003-phase-1.md`). The steps of `process-phase`
that use an index apply.

- **Where.** The catalog is `components/COMPONENTS.md`, with generated pages under
  `components/`. It sits outside `docs/` on purpose: the stage-script docs verifier scans
  `docs/` and rejects pages that list deprecated scripts (Phase 1 spec, IF3). The skill is
  `.claude/skills/components/`, and the script is `scripts/components_index.py`, both synced
  from dev-tools. Their settings come from `.devtools.toml`; never edit the synced settings
  file.
- **Search first (plan rule R11).** Every spec searches the index for each capability it
  builds, twice with different words, before designing it. It records the searches and a
  reuse, adapt or reject decision in its Design section (`components` skill, Consult).
- **Update with the code.** A phase that adds, moves or retires a module changes the catalog
  in the same commit and regenerates it with `python3 scripts/components_index.py write`.
  An apply of the sync regenerates it too (`after_apply` in `.devtools.toml`).
- **At phase close.** Run `python3 scripts/components_index.py --check --strict` (exit 0) and
  `python3 scripts/components_index.py changed --base <the phase's base commit> --strict`,
  and list both among the record's gates with their exits. Fix every `fail:`; for each
  `reread:`, edit the entry or say in the record that it is still accurate.
- **Before a PR.** Run `python3 scripts/components_index.py changed --base main` and put
  its summary line in the PR body. No CI job runs it.
- **Partial entries.** Entries marked `checked: partial` were written without reading all
  of their code. Open the code before relying on one, and remove the mark once the entry
  is checked.
- The optional GND rules check (`rules_check.py`) is not set up here.
