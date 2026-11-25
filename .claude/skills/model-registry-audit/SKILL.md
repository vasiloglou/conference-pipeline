---
name: model-registry-audit
description: Audit code and planning skills for hardcoded LLM model-name literals that must instead be read from the model-governance registry (registry.yaml) by task id. Run before closing any milestone that touches model selection, alongside conference-hardcoding-audit.
---

# Model Registry Audit

`src/conference_pipeline/model_governance/registry.yaml` is the single
source of truth for which model a governed task uses — across every branch and
worktree of this repository (including sibling conference branches such as
`cvpr2026`, which conform to it rather than the reverse). Code and skills must
reference a governed task's model by task id through
`conference_pipeline.model_governance.registry` accessors, never by hardcoding
the model-name literal. Use this skill as a milestone-closure gate **before
closing any milestone that touches model selection** (a new model swap, a
provider change, a branch merge that could reintroduce a stale literal).

A hardcoded model literal is a defect because it silently drifts from the
registry: a future model swap edits `registry.yaml` but a script or skill still
issues the stale model name, so the actual behavior never changes, or changes
inconsistently across call sites.

## Run the automated gate (required)

```bash
python scripts/audit_model_registry_hardcoding.py           # human report, exit 1 on findings
python scripts/audit_model_registry_hardcoding.py --json     # machine-readable
```

It fail-closes (non-zero exit) when `src/conference_pipeline/**` or
`scripts/**` contains an LLM model-name literal (`gpt-5.*`, `gpt-5-mini`,
`gpt-5-nano`, `claude-(opus|sonnet|haiku)-<n>`, `gemini-<n>.<n>-(pro|flash)`) in
a functional position — an argparse `default=`, a `model=`/`model_id=`/
`model_name=` keyword or assigned default, or a `"--model", "<literal>"`
CLI-flag pair — or when `.claude/skills/**` Markdown contains a `--model
<literal>` command example, and the literal is not suppressed. A milestone
touching model selection must not close while this gate fails. See
[`docs/reference/stage-owner-audit.md#audit-model-registry-hardcoding`](../../../docs/reference/stage-owner-audit.md#audit-model-registry-hardcoding)
for the registered stage-owner usage contract.

## The source-of-truth file (read this, don't hardcode)

| Value | Read from |
|---|---|
| default provider/model for a governed task | `registry.py`'s `default_model(task_id)` |
| default model for a specific provider on a task | `registry.py`'s `default_model_for_provider(task_id, provider)` |
| fallback model for a task | `registry.py`'s `default_fallback_model(task_id)` |
| full task policy (thresholds, benchmarks, local candidates) | `registry.py`'s `load_registry().get(task_id)` |

`registry.yaml` itself holds the `tasks.<task_id>.{provider,model,
fallback_provider,fallback_model,allowed_selections}` fields these accessors
resolve; code and skills never duplicate a value from it as a bare string.

## Fix pattern

```python
from conference_pipeline.model_governance.registry import (
    default_model,
    default_model_for_provider,
)

model = default_model("presentation.presentation_review")
# or, when the call site fixes the provider:
model = default_model_for_provider("ocr.slide_image_parse", "anthropic")
```

- A function/script that always uses the task's registered provider:
  `default_model(task_id)`.
- A function/script that fixes a specific provider (e.g. one arm of a
  multi-provider fallback chain): `default_model_for_provider(task_id,
  provider)`.
- A Markdown command example: cite the registry-governed value symbolically
  (e.g. `--model <registry-default>`, with a note naming the task id) rather
  than pasting the literal, so the example does not go stale when the registry
  changes.

## Suppressing an intentional literal

Only for genuinely intentional cases — a literal that is not itself a governed
API-model selection (a cross-tool CLI reviewer identity, a documented
contract elsewhere that requires a specific model different from the general
registry default) — append the marker with a concrete reason on the literal's
line:

```python
model="claude-opus-4-8",  # model-literal-ok: deliberate cross-tool ensemble reviewer (Claude Code CLI); no anthropic entry exists in this task's registry allowed_selections
```

```markdown
`--model gpt-5.6-luna` <!-- model-literal-ok: illustrative example only -->
```

Never suppress a literal that is simply an unmigrated registry-governed API
call — fix it instead. Every suppression must be enumerated in the closing
milestone's summary for review; a suppression bypasses the fail-closed scan and
is a repudiation risk if it hides an unmigrated call rather than documenting a
genuine exception.

## Completion rule

A milestone that touches model selection may close only when
`python scripts/audit_model_registry_hardcoding.py` exits 0. Run this gate
**alongside**, not instead of, `conference-hardcoding-audit` — they cover
different literal families (model names vs. conference identity) and are both
required, independent milestone-completion gates. Record the audit result in
the milestone completion notes.
