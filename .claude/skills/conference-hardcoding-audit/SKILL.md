---
name: conference-hardcoding-audit
description: Audit reusable code for hardcoded conference-specific values (names, years, program domains, topic taxonomies) that must instead be read from the workspace config.yaml, analysis manifest, and presentation frame. Run before closing any conference milestone, and referenced as a completion gate by the conference planning skills.
---

# Conference Hardcoding Audit

The platform must serve any conference — NeurIPS, ICLR, ICML, or a future one —
from the same code, reading identity from each workspace's generated files, never
from literals baked into reusable modules. Use this skill as a portability gate
**before closing a conference milestone** (and any time reusable code changed).

A hardcoded conference value is a defect because it silently produces
wrong-conference output: an ICML deck that links to `iclr.cc`, a prompt that says
"exhaustive ICLR 2026 evidence" for a NeurIPS run, or a topic taxonomy that
selects the wrong corpus.

## Run the automated gate (required)

```bash
python scripts/audit_conference_hardcoding.py           # human report, exit 1 on findings
python scripts/audit_conference_hardcoding.py --json     # machine-readable
```

It fail-closes (non-zero exit) when the reusable library
(`src/conference_pipeline/**`) contains a functional conference literal — a
conference name/year, a program domain (`iclr.cc`/`icml.cc`/`neurips.cc`), or a
year-pinned `/virtual/<year>/` path — that is not suppressed. A milestone must
not close while this gate fails.

## Then do the manual sweep over scripts (required)

The automated gate scopes to the library. Entry-point and generator scripts are
a mix of active generators, conference-specific entrypoints, and historical
one-off tools, so sweep them by hand with the same patterns and classify:

```bash
grep -rInE "(ICLR|ICML|NeurIPS|CVPR) *20[0-9]{2}|(iclr|icml|neurips)\.cc|cvpr\.thecvf\.com|/virtual/20[0-9]{2}" \
  scripts/presentations/ scripts/*.py
```

Classify every hit:
- **Violation (fix it)** — a literal in a *shared/active generator* that reaches
  generated content or behaviour (prompts, slide/figure/video labels, URL
  construction, topic taxonomies, config-less defaults that flow into output).
- **Conference-specific entrypoint (acceptable, but read config)** — a script
  named for one conference (`run_<conf>_production.py`, `acquire_<conf>_*`,
  `supervise_<conf>_*`). It may name its conference, but any value that should
  vary (labels, URLs, run ids, topic lists) must still be read from `config.yaml`
  / `analysis_manifest.json`, not a literal.
- **Benign** — comments, docstrings, argparse `help=` examples, the fixed
  RelationalAI brand-template lock, or a guarded historical fallback.

## The source-of-truth files (read these, don't hardcode)

| Value | Read from |
|---|---|
| conference name / full name / year / slug / program URLs | `conferences/<slug>/config.yaml` via `conference_pipeline.conference_identity` |
| topic ids / titles / count | `conferences/<slug>/analysis/analysis_manifest.json` (`topics[]`) |
| audience title / field thesis / general questions | `conferences/<slug>/decks/<id>/presentation_frame.json` |
| storage families | `conferences/<slug>/storage/storage_manifest.json` |

## Fix pattern

```python
from conference_pipeline.conference_identity import (
    load_conference_identity,   # -> ConferenceIdentity(name, full_name, year, slug, urls, site_root())
    conference_name,            # conference_name(workspace, default="the conference")
)
```

- A function with workspace/config access: `conference_name(workspace)` or
  `load_conference_identity(workspace).year / .site_root() / .url("workshops")`.
- A function without workspace access: thread a `conference_label: str` (or the
  identity) down from the nearest caller that has the workspace; default to a
  conference-agnostic string (`"the conference"`), never a specific name.
- URL rehydration: build from `identity.site_root()` / `identity.url("virtual")`,
  not a literal host.

## Suppressing an intentional literal

Only for genuinely intentional cases (guarded historical fallback, brand lock,
error-message/help example), append the marker with a reason on the literal's
line:

```python
neurips_base_url: str = "https://neurips.cc"  # conference-literal-ok: historical fallback; config-driven via load_settings(config_path=...)
```

Never suppress a literal that reaches generated content. If in doubt, fix it.

## Deliberately out of scope (documented, not gated)

- `iclr2025_live.py` ICLR-2025 acquisition crawl (venue query, crawl seeds) — not
  on the multi-conference path; new conferences use their own runners and their
  config's OpenReview venue. Excluded by path in the gate.
- The shared vector-store collection base name — renaming it orphans already
  indexed workspaces; each workspace has its own persist directory, so the shared
  name does not cross-contaminate. Documented in `indexing/walker.py`.
- Historical NeurIPS-era one-off scripts (`scripts/collect_papers.py`,
  `scripts/claude_deep_research.py`, etc.) — deprecated, not the active contract.

## Completion rule

A conference milestone may close only when
`python scripts/audit_conference_hardcoding.py` exits 0 **and** the manual scripts
sweep found no unresolved violations. Record the audit result in the milestone
completion notes.
