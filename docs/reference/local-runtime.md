# Local Runtime

This document defines the platform contract for local LLM execution in the
conference pipeline. It covers the proof chain required before any local model
reaches production, the typed profile schema, source-of-truth placement, and
the patterns adopted (and rejected) from external reference work.

## Overview

Local runtime support means executing LLM inference directly on operator
hardware — typically Apple Silicon — instead of a cloud provider API. The
platform supports this through a typed profile system that decouples runtime
mechanics (endpoint, server lifecycle, timeouts, concurrency, lock behavior)
from task policy (which tasks may use a local model) and model selection
(which model is approved for which task on which hardware).

Local runtime is intentionally deferred until a complete proof chain exists.
All task policies in `registry.yaml` carry `local_runtime_policy: deferred_*`
until the four-step proof chain below is complete.

## Proof Chain

Before any local model may be used in production for a given task, all four
steps must exist:

1. **Current source research (Phase 8)** — Dated, publicly verifiable research
   confirms which open-weight model families are best-fit candidates for the
   task. No model name from external reference projects is treated as a
   production default without this step.

2. **Target-Mac hardware-fit preflight (Phase 7)** — A preflight check on the
   target operator Mac confirms that optional dependencies are installed, the
   model asset fits in available memory, and the server endpoint is reachable.
   The profile's `hardware_fit_status` must advance to `"verified"` before
   production use.

3. **Task benchmark receipt (Phase 9)** — A benchmark is run against
   representative task inputs and produces a dated, inspectable receipt
   including task id, profile id, model id, hardware profile, prompt/config
   digest, output artifact, rubric score, latency, cost, pass/fail decision,
   and runtime lock metadata.

4. **Explicit operator approval (Phase 10)** — An operator reviews the
   research, preflight, and benchmark receipts and records explicit approval.
   The profile's `approval_status` must advance to `"approved"`. This step
   cannot be satisfied by a planning default or placeholder.

Until all four steps exist for a task, `local_runtime_policy` remains
`deferred_*` and `local_runtime_profile_id` must be absent from that task's
registry entry.

## Profile Schema

Profiles are defined in
`src/conference_pipeline/model_governance/local_runtime_profiles.yaml` and
loaded by `load_profiles()` from
`src/conference_pipeline/model_governance/local_runtime.py`.

Each profile is a `LocalRuntimeProfile` frozen dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `profile_id` | `str` | Unique identifier matching the YAML key |
| `runtime_kind` | `str` | One of `"mlx"`, `"ollama"`, `"custom"` |
| `base_url` | `str` | OpenAI-compatible API base URL, e.g. `"http://127.0.0.1:8080/v1"` |
| `manage_server` | `bool` | If `true`, the platform manages server start/health/stop. Must be `false` unless `base_url` is a loopback address. |
| `optional_deps` | `list[str]` | Python package names required at runtime, e.g. `["mlx", "mlx_lm"]` |
| `startup_timeout_s` | `float` | Seconds to wait for the server health check after startup |
| `request_timeout_s` | `float` | Per-request timeout in seconds |
| `max_concurrency` | `int` | Maximum parallel requests; auto-clamped to safe levels at load time |
| `max_batch_size` | `int` | Maximum batch size per request |
| `lock_scope` | `str` | One of `"heavy"` (machine-wide exclusive), `"light"` (task-family), `"none"` |
| `hardware_fit_status` | `str` | One of `"unknown"`, `"pending"`, `"verified"`, `"rejected"` |
| `approval_status` | `str` | One of `"deferred"`, `"benchmark_pending"`, `"approved"`, `"blocked"` |

`manage_server=true` is allowed only when `base_url` starts with
`http://127.` or `http://localhost`. The loader enforces this at load time.

## Source-of-Truth Placement

| Artifact | Location | Owner |
|---|---|---|
| Reusable profile definitions | `src/conference_pipeline/model_governance/local_runtime_profiles.yaml` | Platform |
| Profile schema and loader | `src/conference_pipeline/model_governance/local_runtime.py` | Platform |
| Task policy registry with `local_runtime_profile_id` extension point | `src/conference_pipeline/model_governance/registry.yaml` | Platform |
| Current source research receipts | Workstream phase artifacts or conference milestone context | Phase 8 |
| Hardware-fit preflight receipts | Workstream phase artifacts or conference milestone context | Phase 7 |
| Task benchmark receipts | Workstream phase artifacts or conference milestone context | Phase 9 |
| Operator approval records | Active conference milestone context | Phase 10 |

Conference-specific benchmark receipts, model decisions, and approvals belong
in the conference milestone context or workspace, not in root platform files.

## Adopted and Rejected Book-Project Patterns

### Adopted

- **Typed config models with explicit field names** — `LocalRuntimeProfile` is
  a frozen dataclass with named, typed fields rather than raw dicts. This
  follows the strict config model pattern used in the reference project.

- **Auto-clamping for modest concurrency and batch settings** — The loader
  validates field ranges. Future Phase 7 preflight logic may clamp
  `max_concurrency` and `max_batch_size` downward for safety, following the
  effective-worker and memory-guard pattern observed in the reference.

- **Managed loopback server lifecycle concept** — The `manage_server` boolean
  and loopback enforcement preserve the managed MLX server idea (start/health
  check/switch/stop), while keeping the actual start/stop implementation in
  Phase 7 where MLX dependencies are present.

- **Machine-wide lock metadata shape** — The lock fields (`lock_scope`,
  `profile_id`) preserve the metadata shape needed for audit and evolution
  (task id, profile id, model id, pid, started_at, receipt path documented in
  Phase 12 lock design).

### Rejected

- **Qwen or other specific open-weight model defaults** — The reference project
  uses specific model names (Qwen, etc.) as defaults. These are not copied here
  because Phase 8 source research must establish the best candidates for the
  conference pipeline's specific tasks and hardware targets before any name
  becomes a default.

- **Multi-runtime framework** — The reference project supports multiple runtime
  back-ends simultaneously. The conference pipeline starts with a simpler
  single-profile selection model and may expand via the `runtime_kind` field.

- **Full benchmark receipt shape from the reference benchmarks script** — The
  reference benchmark runner is useful prior art but is not the conference
  pipeline's receipt contract. Phase 9 defines the canonical receipt shape.

- **Non-loopback managed endpoints** — The reference allows remote managed
  server endpoints in some configurations. The conference pipeline requires
  `manage_server=false` for any non-loopback URL to prevent unintended
  side-effects on shared infrastructure.
