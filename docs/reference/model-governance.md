# Model Governance Registry

The platform-owned model-governance registry lives at
`src/conference_pipeline/model_governance/registry.yaml`. It records durable
task ids, planning-default provider/model pairs, fallback policy, current
research requirements, benchmark placement, budget caps, provenance/schema
requirements, and local-provider deferral status.

## Model Catalog (`models.yaml`)

Task entries do not repeat literal provider/model strings. Each task's
`model_ref` / `fallback_model_ref` names an alias defined once in the sibling
`src/conference_pipeline/model_governance/models.yaml` catalog (`sol`,
`terra`, `luna`, `opus`, `sonnet`, `gemini_pro`, ...), each resolving to a
`{provider, model, status}` entry. `registry.py` resolves the alias at load
time — `TaskPolicy.provider`/`.model` are still plain strings, so no
downstream caller changes. A model-generation upgrade (or retirement) is one
edit to the catalog entry, not one edit per task that uses it.
`load_registry()` raises `ModelGovernanceError` if a task references an
unknown or `status: retired` alias — this is enforced whenever the registry
loads, not only by the offline `model-registry-audit` script. `allowed_selections`
and `benchmark.*` fields stay literal but are validated against the same
catalog. See `models.yaml`'s own header comment for the incident (a merge
silently reverting 27 tasks' model assignments) that motivated this design.

To check whether the catalog is still on the latest models, use the
`model-catalog-upgrade` skill (`scripts/model_catalog_upgrade.py`): it lists
what each provider offers today, proposes same-family successors with their
blast radius and price, and applies only operator-approved aliases — moving
the alias, retiring the old model, rewriting the literal
`allowed_selections`/`benchmark` pairs and `costs/task_specs.yaml`, and
refusing any model without a price in `costs/pricing_catalog.json`.

Model names in the registry are planning defaults, not permanent production
approval. A live conference milestone must still record dated current research,
provider approval, and stage-local benchmark receipts in its planning-owned
milestone context or workspace evidence before provider-backed work starts.

## Required Pattern

- Runnable code should load defaults through
  `conference_pipeline.model_governance` helpers or validate raw overrides with
  `validate_selection(task_id, provider=..., model=...)`.
- Planning generators should cite registry task ids rather than scattering
  provider/model defaults into prose-only instructions.
- Stage-local benchmarks should run where realistic inputs exist: inventory and
  interest groups after source discovery, source reading after corpus material
  exists, OCR/image parsing after representative media exists, and podcast,
  presentation, synthesis, and QA repair near their output stages.
- Missing realistic inputs or benchmark receipts are blockers. Placeholder
  receipts, mock receipts, or local-provider aspirations do not approve a
  production default.

## Local Runtime

Local runtime support (running inference on-device via MLX or Ollama) follows
a separate typed profile contract. Before any local model may be used in
production, a four-step proof chain must be complete: current source research,
hardware-fit preflight, task benchmark receipt, and explicit operator approval.

See [docs/reference/local-runtime.md](local-runtime.md) for:
- The proof chain and its four steps
- The `LocalRuntimeProfile` field schema
- Source-of-truth placement rules for profiles vs. conference-specific receipts
- Adopted and rejected patterns from the reference book-project

The `local_runtime_profile_id` extension point in `registry.yaml` is
documented there along with the conditions under which a task policy may use it.

## Inventory And Exceptions

`src/conference_pipeline/model_governance/callsite_inventory.json` records the
initial in-scope call-site inventory for current runtime and planning surfaces.
`src/conference_pipeline/model_governance/migration_exceptions.yaml` is the
machine-readable allowlist for current-looking model references that remain
historical or out of scope for this migration.

## Local Proof Chain (Phase 10+)

Local model selections require four proof classes before production planning may use them:

| Class | Field | Source Phase |
|-------|-------|-------------|
| Current research | `research_ref` | Phase 8 research receipt |
| Hardware-fit proof | `hardware_fit_ref` | Phase 7 preflight / Phase 9 receipt |
| Benchmark receipt | `benchmark_receipt_ref` | Phase 9 BenchmarkReceipt with selection_decision=passed |
| Operator approval | `operator_approval_ref` | Phase 10 operator approval record |

Use `approved_local_selection(task_id)` to retrieve the registered local
ModelSelection. It raises `ModelGovernanceError` naming the first missing proof
class and validates `benchmark_receipt_ref` before approval. The receipt must
exist, load through the benchmark receipt schema, and be a live/local/MLX passed
receipt for the same task and model.

All registry tasks default to `local_proof: {}` (empty = deferred).
See [local-runtime.md](local-runtime.md) for the proof chain and source-of-truth placement rules.

## Local-First Routing (v3.0 / LOCAL-02)

When an approved local selection exists for a task, the pipeline prefers it
automatically.  Call `resolve_local_first_selection(task_id)` from
`conference_pipeline.model_governance` instead of inspecting the registry
directly.  The resolver applies global and per-task policy and returns a
`LocalFirstRoutingDecision` that records both the selected
provider/model/role and the fallback audit trail.

### Fallback policy

Fallback to the paid provider is **automatic and receipted** — it does not
halt the run.  The Phase-02 `budget_cap_usd` two-point gate still guards any
paid spend that results from a fallback.

| Fallback reason | Condition |
|-----------------|-----------|
| `local_first_disabled` | Global `local_first_enabled=False` passed to the resolver |
| `task_forced_paid` | Per-task `local_first_enabled: false` in the registry policy |
| `approved_local_unavailable` | `approved_local_selection()` raised `ModelGovernanceError` (any missing proof) |
| `local_readiness_failed` | Injected `readiness_check` raised or returned `False` |

This is the deliberate counterpoint to the halt-for-ack pattern used for
budget cap breaches (D-07): a local runtime hiccup should not stop a run.

### Fallback receipts

When `workspace`, `run_id`, and `stage_id` are supplied to the resolver, a
`LocalFallbackReceipt` is written under:

```
<workspace>/storage/workflow-runs/<run_id>/local-fallback-receipts/<stage>/
```

Use `scan_local_fallback_receipts(workspace)` from
`conference_pipeline.provider_receipts` to enumerate all fallback receipts.
The receipt contains `fallback_reason`, `cost_cap_checked: true`, and no
API keys or endpoint URLs (T-03-05 sanitization).

This receipt family is **distinct** from the `--local` smoke-mode flag
(`conference-pipeline ... --local`) which forces local-only execution and
does not fall back.

### Per-task registry override

Add `local_first_enabled: false` to any `TaskPolicy` entry in `registry.yaml`
to force that task to always use the paid provider, even when a valid approved
local selection exists.  This is the D-08 quality-critical-task escape hatch.

### Spine wiring (Phase 10)

Provider *selection* for three task families flows through the resolver at the
spine/factory layer; the engines/services stay pure executors that receive a
concrete `(provider, model)`:

| Task | Seam (factory) | Local executor |
|------|----------------|----------------|
| `transcription.session_speech` | `create_transcription_service(settings, …)` | `LocalWhisperTranscriptionService` |
| `ocr.slide_image_parse` | `create_ocr_engine(settings, …)` (generic `NeuripsPipeline` spine) | `LLMOcrEngine(provider="local")` |
| `vision.slide_analysis` | `create_slide_analysis_engine(settings, …)` | `SlideAnalysisEngine(provider="local_vlm" / "local_fastvlm")` |

Each factory passes `settings.local_first_enabled` as the resolver's
`local_first_enabled`, an injectable per-family readiness probe (from
`model_governance.local_readiness`) as `readiness_check`, and run context
(`workspace`/`run_id`/`stage_id`) so live fallbacks are receipted.

**Routing signal — use `decision.used_local`, never `decision.provider`.** A
local executor is constructed ONLY when `decision.used_local is True` (the
approved-local proof + readiness both passed). The registry shape for
`transcription.session_speech` is `provider: openai` with
`fallback_provider: local` (a *deferred* local candidate). That `local` fallback
is **never** a paid fallback: `_paid_selection_from_policy()` skips local
providers, so a fallback's `decision.provider`/`paid_fallback_provider` is always
non-local. Branching on `decision.provider == "local"` would wrongly route a paid
fallback to local — always key on `used_local`.

For `vision.slide_analysis`, the registry candidate's `runtime_kind` maps to the
engine's local provider (`vllm_mlx`/`mlx_vlm` → `local_vlm`; `mlx_native_fastvlm`
→ `local_fastvlm`). An unknown/unsupported `runtime_kind` falls back to the paid
provider **with a paid/default model** (never the local model id) rather than
raising.

### Known limitations (Phase 10)

- **`iclr2025_live` slide-OCR is paid-only.** The live
  `create_slide_ocr_service(settings)` → `slide_ocr.py` path supports
  `openai`/`anthropic`/`gemini` only — it has **no local executor**. It is
  intentionally NOT routed through the resolver; adding a local executor there is
  new capability for a follow-up phase. Only the generic `NeuripsPipeline`
  `create_ocr_engine` OCR seam is local-capable and wired.
- **`vision.slide_analysis` has no live spine call-site (inactive spine).** The
  `create_slide_analysis_engine` factory wires the engine/factory surface and is
  proven offline with fixtures; there is no production caller in the pipeline
  today. A live call-site is a follow-up phase.
