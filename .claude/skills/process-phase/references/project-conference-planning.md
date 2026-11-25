# Planning a conference milestone

Project-owned, beside `project.md`; the sync never touches it. `project.md` points here.

`plan-conference-analysis` (from a PRD-style brief) and `plan-conference-production`
(from that analysis specification) are the domain planners: they hold the conference
pipeline's stage contracts, provider and cost gates, model governance, backup
checkpoints and audits. They still emit GSD-format files. Until they are rewritten to
emit this workflow's format, use them as below.

`PY` is `PYTHONPATH=src .venv/bin/python` (in a worktree, the main checkout's
interpreter; see `project.md`, "Where commands run"): the planners' scripts are not executable
files, so they are always run through it. The analysis milestone and the production
milestone each have their own folder; `A` is the analysis milestone's `notes/` and `P`
the production milestone's.

**Analysis milestone**
1. Draft: `PY .claude/skills/plan-conference-analysis/scripts/scaffold_plan.py --brief <brief> --draft-root A/conference-analysis-draft`.
   The draft lands in `A/conference-analysis-draft/<slug>/`.
2. Check the draft, and keep each output:
   - `PY .claude/skills/plan-conference-analysis/scripts/audit_generated_plan.py A/conference-analysis-draft/<slug> --json-output A/conference-analysis-audit.json`;
   - `PY -m conference_pipeline.stage_owner_audit --plan-root A/conference-analysis-draft/<slug>`;
   - `PY scripts/verify_stage_script_usage.py --plan-root A/conference-analysis-draft/<slug> --skills-root .claude/skills --docs-root docs --strict`.
3. Bootstrap the workspace, which the gitignore gate needs in place: the generator
   reads `conferences/<slug>/storage/storage_manifest.json` and refuses a workspace
   outside the repository (`storage_manifest_missing`, `workspace_outside_repo`).
   - Copy the draft's `conferences/<slug>/` (workspace README and workshop tracker)
     and its workshop downloader stubs
     `scripts/conference/<slug>/workshop_site_downloaders/*.py` (the tracker points to
     them) into place.
   - Write `conferences/<slug>/config.yaml` and
     `conferences/<slug>/storage/storage_manifest.json`. The scaffolder writes
     neither; follow `conferences/README.md` (layout and storage classes) and start
     from the closest earlier conference's, as v5.0-CVPR 2026 did
     (`.planning/workstreams/cvpr-2026/drafts/cvpr2026/conferences/cvpr2026/`).
   - Run `PY scripts/generate_workspace_gitignore.py --workspace conferences/<slug> --write --check --receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json`.
   This writes only the new workspace and its stubs; it is part of preparing the plan,
   which a request to start a milestone authorizes (`start-milestone`).
4. Every command of steps 2 and 3 exits 0 before the plan goes to its gate.
5. Once the plan is approved, copy the draft's context packet to
   `.planning/milestones/<display name>-CONTEXT.md`. Never copy its
   `.planning/STATE.md`, `ROADMAP.md` or `REQUIREMENTS.md`.

**Production milestone**
6. Draft: `PY .claude/skills/plan-conference-production/scripts/scaffold_production_plan.py --spec-root <analysis milestone's A>/conference-analysis-draft/<slug> --draft-root P/conference-production-draft`.
   `--spec-root` is the analysis draft's `<slug>` folder, never the repository root.
   The draft lands in `P/conference-production-draft/<slug>-production/`.
7. Check it, and keep each output:
   - `PY .claude/skills/plan-conference-production/scripts/audit_generated_production_plan.py P/conference-production-draft/<slug>-production --json-output P/conference-production-audit.json`;
   - `PY -m conference_pipeline.stage_owner_audit --plan-root P/conference-production-draft/<slug>-production`;
   - `PY scripts/verify_stage_script_usage.py --plan-root P/conference-production-draft/<slug>-production --skills-root .claude/skills --docs-root docs --strict`;
   - `PY scripts/generate_workspace_gitignore.py --workspace conferences/<slug> --check`
     (the workspace is in place since the analysis milestone).
   All exit 0 before the plan goes to its gate.
8. Once the plan is approved, copy the production draft's own context packet
   (`.planning/milestones/<production display name>-CONTEXT.md`, the production
   identity, not the analysis one) and its `conferences/<slug>/PRODUCTION.md` into
   place. Never copy its `.planning/STATE.md`, `ROADMAP.md` or `REQUIREMENTS.md`; its
   `approval-summary.json` stays in the draft, cited by the plan.

**Both**
- **Never run `--finalize`.** It writes the root `.planning/STATE.md` and `ROADMAP.md`,
  which are history, and creates a root `REQUIREMENTS.md`. Without `--update-existing`
  it also stops part-way with `FileExistsError`, leaving a partial write.
- The draft is input, not the plan. `start-milestone` writes the one canonical plan,
  `notes/<milestone-dir>-plan.md` in the milestone's own folder. It lists every draft
  file with its own SHA-256, carries every requirement, stage, gate and audit the drafts
  name, and maps each to a phase id.
- The plan has a section that replaces each GSD lifecycle obligation the drafts state
  with the step of this workflow that discharges it. In particular, the production
  draft's rule "no live execution of Phase N without `phases/NN-*/` PLAN plus
  plan-review and verification artifacts" is met by the phase's approved spec
  (`notes/phase-N-spec.md` with its owner gate), its review notes and its phase record,
  and the plan says so in those words. An obligation with no counterpart stays a gate
  of the phase, written out.
- A dry run of the two scaffolders and their own audits on a test brief (2026-11-13, in
  a scratch folder) exited 0. The other checks, the workspace bootstrap and the copies
  were not part of it, and no real conference milestone has been planned this way yet;
  the first one records anything these steps leave out.
- Rewriting the two planners, their auditors and their tests to emit this workflow's
  format directly is a separate piece of work.
