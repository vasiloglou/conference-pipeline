# Orchestrator Share Audit

Report-only measurement of how thin each per-conference production
orchestrator is (wave W0 of
`.planning/proposals/thin-conference-orchestrators-20260930.md`, section 6).

```bash
PYTHONPATH=src .venv/bin/python -m conference_pipeline.orchestrator_share_audit            # markdown
PYTHONPATH=src .venv/bin/python -m conference_pipeline.orchestrator_share_audit --format json --no-units
PYTHONPATH=src .venv/bin/python -m conference_pipeline.orchestrator_share_audit \
    --orchestrator scripts/run_<slug>_production.py --json-out <report.json> --markdown-out <report.md>
```

- **Static only.** Orchestrators are parsed with `ast`; nothing is imported,
  executed, or sent to a provider.
- **Inputs.** `scripts/run_*_production.py` plus any registry
  `*.production_runner` script path, or explicit `--orchestrator` paths.
  Conference tokens used to normalize clones come from the file names.
- **Per unit.** Lines, the stage it serves (reachability from the
  `stage.id == ...` dispatch in `main`), and its class: `hook`
  (`@conference_hook`), `adapter` (at most 40 body lines referencing a shared
  component), `dead`, or `local`. Overlays: `shadow_of` a `src/` function and
  `clone_of` a unit in another orchestrator (normalized similarity >= 0.80).
- **Shared components.** `conference_pipeline.production.*` and any `src`
  module with a module-level `RECOMMENDED_FOR` constant (decision D4: reusable
  acquisition lanes). Unannotated `src` modules named after the orchestrator's
  conference are reported as conference-local components.
- **Per stage.** `shared`, `shared+hooks`, `conference-local`, `local`, or
  `unrouted`.
- **Output.** JSON with the `attach_agent_diagnostic` envelope, plus a
  markdown table. Always exits 0; no baseline or ratchet yet (that arrives
  after W3).

Fixtures: `tests/fixtures/orchestrators/` (ICML-shaped, CVPR-shaped and
minimal new-conference synthetic workspaces plus a synthetic `repo/`).
Golden-receipt harness: `tests/orchestrator_golden.py`.
