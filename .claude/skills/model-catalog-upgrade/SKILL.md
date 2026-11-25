---
name: model-catalog-upgrade
description: Check whether the governed LLM models are still the latest, propose the upgrades to the user, and apply only the ones the user approves. Use when the user asks to update/upgrade the models or LLMs to their latest versions, asks whether the models are current, or before a milestone that should run on the newest model generation.
---

# Model Catalog Upgrade

`src/conference_pipeline/model_governance/models.yaml` maps stable aliases
(`opus`, `sonnet`, `sol`, `gemini_pro`, ...) to one provider/model pair each,
and every task in `registry.yaml` refers to an alias. A generation upgrade is
therefore a small, reviewable change -- but it changes which model every
dependent task calls, what runs cost, and whether earlier outputs are still
comparable. **This skill proposes; the user decides. Never apply an upgrade
the user has not explicitly approved in this conversation.**

The helper is `scripts/model_catalog_upgrade.py` (run from the repo root with
the repo's `.venv`). Discovery is read-only and free; keys come from `.env`
and are never printed.

## 1. Discover and propose ($0, changes nothing)

```bash
.venv/bin/python scripts/model_catalog_upgrade.py discover --out <scratch>/discovered.json
.venv/bin/python scripts/model_catalog_upgrade.py propose --discovered <scratch>/discovered.json
```

`propose` compares every **active** alias with the provider's current model
list and reports, per alias: the newer same-family model (a higher version,
or the GA release of a preview; never another tier, never a dated snapshot),
how many tasks use the alias as default/fallback, how many literal pairs in
`registry.yaml` / `task_specs.yaml` move with it, whether a price is on file,
and which code/skill files still spell the old model. Aliases named `*legacy*`
are deliberate pins and are skipped unless `--include-legacy`.

Read the output critically before presenting it:
- **Family matching is heuristic.** Confirm each candidate really is the same
  tier's successor (e.g. `gpt-6-sol` for `gpt-5.6-sol`), not a sibling product.
  If unsure, check the provider's model docs (WebFetch on the official page).
- **`NOTE: current model is NOT in the provider's list`** means the pinned model
  may be deprecated -- that alias is urgent, not optional.
- **`ALREADY CATALOGUED as <alias>`**: don't apply; propose a `repoint` (below)
  of this alias's tasks onto that alias, which retires this one.
- **`.env OVERRIDES pin <model>`**: env keys such as `OCR_MODEL` beat the
  registry. After the upgrade they keep -- or, once the model is retired,
  break -- the old model. Name them in the proposal; `.env` is the user's, so
  ask before editing it (usually: delete the line so the registry default applies).
- **Check the tier structure, not just the version.** A new generation can add
  or drop tiers (GPT-6 added `astra` above `sol` and has no `terra`); say so, since
  tasks chose an alias for its tier.
- **Verify prices from the page's own data.** A page summarizer can misread a
  pricing table; confirm the numbers you will record (e.g. `curl` the page and
  read the row) before presenting them. Note scheduled price changes.
- **`BLOCKER: ... also point at`**: two aliases share the old model; resolve which
  should move before applying either.

## 2. Price every candidate

For each proposed model without `price on file`, get input/output USD per 1M
tokens from the provider's **official** pricing page -- the URLs are under
`sources` in `src/conference_pipeline/costs/pricing_catalog.json`
(`anthropic_pricing`, `openai_pricing`, ...). Use the source key as
`--price-source`. OpenRouter's listed price is shown by `propose`; still
confirm it. Never guess a price and never apply without one: an unpriced model
is costed at $0 by preflight estimates.

## 3. Ask the user -- one decision per alias

Present a compact table: alias, current -> proposed, tasks affected, price
change (old vs new per 1M in/out), and the notable risks. Then ask with
AskUserQuestion (`multiSelect: true`), one option per proposed alias, plus the
consequences that apply to this repository:

- **Receipts and lineage**: `registry.yaml`'s digest is bound into
  source-discovery cost authorities; batch all approved upgrades into one
  change and make it BEFORE re-running stages, never during a run.
- **Outputs made with the old model stay as they are.** Say which in-flight
  artifacts were produced by the alias being moved (e.g. deck narration made
  with `opus`) and ask whether they should be regenerated -- that is a separate,
  paid decision the user makes after the upgrade.
- **Cost**: a higher-priced successor raises every task's estimate; mention
  tasks with a `budget_cap_usd` that the new price could push over.

An approval must name the alias. "Upgrade everything" counts only for the
aliases listed in the question you asked.

## 4. Apply the approved aliases only

One alias per call; dry-run first and show the user the change list:

```bash
.venv/bin/python scripts/model_catalog_upgrade.py apply --alias <alias> --to <model> \
    --input-price <usd/1M> --output-price <usd/1M> --price-source <source key> --dry-run
.venv/bin/python scripts/model_catalog_upgrade.py apply --alias <alias> --to <model> \
    --input-price <usd/1M> --output-price <usd/1M> --price-source <source key>
```

To move an alias's tasks onto an existing alias instead (retired or
unlisted model, or a successor that is already catalogued), and to refresh a
stale price:

```bash
.venv/bin/python scripts/model_catalog_upgrade.py repoint --alias <old> --into <existing> --dry-run
.venv/bin/python scripts/model_catalog_upgrade.py price --provider <p> --model <m> \
    --input-price <usd/1M> --output-price <usd/1M> --price-source <source key> --note "<why>"
```

`repoint` keeps the provider route (it refuses to move an OpenRouter task onto a
direct-provider alias) and requires the target to be priced.

`apply` moves the alias in `models.yaml`, keeps the old model as a
`status: retired` entry (so reintroducing it fails closed at
`load_registry()`), rewrites the old literal in `registry.yaml` and
`task_specs.yaml`, adds the price, and runs `load_registry()`. It refuses
unpriced models, aliases that share the old model, and a model already
catalogued under another alias.

## 5. Verify, report, commit

```bash
.venv/bin/python scripts/audit_model_registry_hardcoding.py
.venv/bin/python scripts/audit_conference_hardcoding.py
.venv/bin/python -m pytest -q tests/test_model_governance.py tests/test_model_catalog_upgrade.py \
    tests/test_audit_model_registry_hardcoding.py tests/test_workflow_registry.py
```

`tests/test_model_governance.py::CURRENT_GENERATION` pins alias -> model (so a
merge that reverts the catalog fails a test); update exactly the aliases you
moved there, and nothing else. Other tests should read models through the
catalog (`load_model_catalog()[alias]`) rather than spelling them out. A test
that reads the real `.env` (e.g. the OCR settings default) fails until the
`.env` overrides above are resolved -- report it, don't paper over it.

Then report: what moved, the retired aliases, price deltas, and the code/skill
files `apply` listed as still spelling the old model -- migrate those to
`default_model(<task>)` or mark a deliberate pin `# model-literal-ok: <reason>`;
do not blind-replace them. Commit the governed files **by path** (other
sessions may have uncommitted edits in the same files -- check `git diff`
first and never commit changes you did not make without asking).

## Related

- `model-registry-audit` -- keeps code reading models from the registry.
- `CLAUDE.md` "Model Governance" -- why the catalog is aliased.
