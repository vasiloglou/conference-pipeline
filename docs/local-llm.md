# Local LLM and Embedding Lane (v1.9)

**Status:** v1.9 Local-First AI Stack — embedding side (Phases 42–44)
plus vector-store provider switch (Phase 45) plus local researcher
(Phase 46) plus this operator reference (Phase 47).

This doc is the operator reference for running the conference pipeline's
embedding and research stages against **local** LLM endpoints (MLX
on Apple Silicon, Ollama on Linux/macOS) instead of the OpenAI cloud
lane. The OpenAI lane remains the default and supported; the local lane
is **opt-in** via configuration and the `--local` smoke flag.

---

## What this enables

- **Vector-store embeddings** computed by `MLXEmbeddingClient` (default
  `mlx-community/bge-m3-mlx-8bit`, 1024-dim) or `OllamaEmbeddingClient`
  (default `nomic-embed-text`, 768-dim) against a loopback endpoint.
- **GPT-Researcher synthesis** via `LocalLLMResearcher` against a local
  OpenAI-compatible endpoint (e.g. `mlx_lm.server` or Ollama's
  OpenAI-compat layer).
- **Coexisting Chroma collections** for cloud and local lanes, suffixed
  with `<provider>_<model_slug>_<dim>` so a workspace can hold multiple
  embedding indexes and switch providers without rebuilding from
  scratch.
- **Receipts at parity** with the cloud lane: every local embedding call
  writes a `LocalEmbeddingReceipt`, every vector-store setup writes a
  `VectorStoreSetupReceipt`, every operator-driven rebuild writes a
  `VectorStoreRebuildReceipt`. All redacted (no secrets, no full
  endpoint URLs — only `endpoint_label` like `http://localhost:11434`
  or `mlx://mlx-community/bge-m3-mlx-8bit`).

---

## Quick start (Apple Silicon, MLX)

```bash
# Install Apple-Silicon optional deps
pip install -e ".[mlx]"

# Pull / load the MLX embedding model on first run (idempotent)
# (mlx-embeddings library handles the HuggingFace download)

# Unset OPENAI_API_KEY to prove the local-only path
unset OPENAI_API_KEY

# Run the smoke-test in the local lane
conference-pipeline conference-smoketest \
    --workspace conferences/iclr2025 \
    --mode mock \
    --local \
    --local-provider mlx
```

## Quick start (Linux / macOS, Ollama)

```bash
# Install ollama (macOS)
brew install ollama
# Or Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull the embedding model
ollama pull nomic-embed-text

# Optional — pre-pull a synthesis model for the LocalLLMResearcher
ollama pull qwen2.5:7b-instruct

# Start the daemon (or set manage_server=True and let the pipeline spawn it)
ollama serve &

# Mark the researcher endpoint to point at Ollama's OpenAI-compat layer
export GPTR_LOCAL_LLM_BASE_URL="http://localhost:11434/v1"
export GPTR_LOCAL_LLM_FAST_MODEL="qwen2.5:7b-instruct"
export GPTR_LOCAL_LLM_SMART_MODEL="qwen2.5:7b-instruct"

unset OPENAI_API_KEY
conference-pipeline conference-smoketest \
    --workspace conferences/iclr2025 \
    --mode mock \
    --local \
    --local-provider ollama
```

---

## Environment variable reference

| Variable | Required when | Default | Scope | Failure mode if misset |
|---|---|---|---|---|
| `EMBEDDING_PROVIDER` | local lane active | `mlx` | embedding | unknown value: `ValueError` from `build_embedding_client`; `openai` without opt-in: `PaidEmbeddingApprovalRequired` |
| `EMBEDDING_MODEL` | optional override | provider default | embedding | unknown model surfaces on first `embed_texts` call |
| `EMBEDDING_BATCH_SIZE` | optional positive int | adaptive heuristic | embedding | non-positive: rejected by factory; ignored if env unset |
| `CONFERENCE_ALLOW_PAID_EMBEDDINGS` | `EMBEDDING_PROVIDER=openai` | unset | embedding paid opt-in | unset with openai provider: `PaidEmbeddingApprovalRequired` raised before any API call (T-03-07) |
| `OLLAMA_BASE_URL` | Ollama provider | `http://127.0.0.1:11434` | embedding | non-loopback: `EmbeddingError("non_loopback_base_url")` before any HTTP |
| `GPTR_LOCAL_LLM_BASE_URL` | local researcher | `http://127.0.0.1:8080/v1` | research | non-loopback: `ResearcherError("non_loopback_base_url")` immediately |
| `GPTR_LOCAL_LLM_FAST_MODEL` | local researcher | `Qwen/Qwen2.5-7B-Instruct` | research | unknown model surfaces on first `run()` |
| `GPTR_LOCAL_LLM_SMART_MODEL` | local researcher | falls back to FAST_MODEL | research | same |
| `GPTR_LOCAL_LLM_ENDPOINT_LABEL` | optional override | `http://localhost:<port>` (redacted) | receipts | string only — used only for display; redacted form still in receipts |
| `GPTR_LOCAL_LLM_RUNTIME_KIND` | optional informational | unset | discovery | informational only — set by `--local --local-provider <p>` |
| `OPENAI_API_KEY` | **must be unset** for `--local` | — | safety | non-empty: `conference-smoketest --local` exits 2 with: `"--local requires OPENAI_API_KEY to be unset to prove the local-only path. Run: unset OPENAI_API_KEY"` |

---

## Local MLX embeddings (default)

The default embedding provider is **`mlx`** (`mlx-community/bge-m3-mlx-8bit`,
1024-dim). No `EMBEDDING_PROVIDER` env var is needed to use the local default.

```bash
# Local default — no env vars required.
conference-pipeline workflow run enhanced-indexing-vector-store \
    --workspace conferences/iclr2025 --run-id my-run
```

---

## Paid OpenAI embeddings (explicit opt-in required)

OpenAI embeddings (`text-embedding-3-small`, 1536-dim) require **explicit
operator opt-in** before they can be selected. Without opt-in, the embedding
resolver raises `PaidEmbeddingApprovalRequired` before making any API call.

To use paid embeddings:

```bash
export EMBEDDING_PROVIDER=openai
export CONFERENCE_ALLOW_PAID_EMBEDDINGS=1
export OPENAI_API_KEY=sk-...
conference-pipeline workflow run enhanced-indexing-vector-store \
    --workspace conferences/iclr2025 --run-id my-run
```

**Paid embedding receipts:** Every paid embedding call writes a
`LocalEmbeddingReceipt` with `provider="openai"` and
`cost_status="estimated"` (or `"unknown"` when pricing is unavailable)
to `storage/local-embedding-receipts/` (or the run-scoped path when a
`run_id` is active). These receipts are distinct from provider-call
receipts and are found by `scan_local_embedding_receipts()`. No API
key or secret material appears in the receipt — only
provider/model/dim/cost_status (T-03-03).

**Dim mismatch:** MLX (1024-dim) and OpenAI (1536-dim) write to separate
Chroma collections (suffixed by provider/model/dim). Switching providers
requires a `vector-store rebuild --provider <new-provider>` to build a
compatible index.

---

## Embedding model defaults and dimensions

| Provider | Default model | Dim | Notes |
|---|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 | paid cloud lane; requires `CONFERENCE_ALLOW_PAID_EMBEDDINGS=1` |
| `mlx` | `mlx-community/bge-m3-mlx-8bit` | 1024 | **default** local embedding; lazy-loads on first `embed_texts` |
| `ollama` | `nomic-embed-text` | 768 | strong general-purpose; one prompt per HTTP call (Ollama's `/api/embeddings` quirk) |
| `fake` | `text-embedding-3-small` | 1536 | deterministic SHA-256-based; for tests |

Configurable via `EMBEDDING_MODEL` env or constructor `model=` arg.

---

## Dimension migration policy (D1 user decision)

**Each provider+model+dim writes to its own Chroma collection.** A
workspace can hold cloud and local indexes simultaneously:

```
storage/vector-store/<run_id>/chroma/
├── neurips_presentations__openai_text-embedding-3-small_1536/
└── neurips_presentations__mlx_mlx-community-bge-m3-mlx-8bit_1024/
```

The collection name is derived by
`derive_collection_name(base, provider, model, dim)` — long model
names are slug-truncated; the prefix and dim are always preserved.

**Dim mismatch fail-stops** with a remediation message:

> `vector store collection 'neurips_presentations__openai_text-embedding-3-small_1536' was built at dim 1536; active embedding provider 'mlx' / model 'mlx-community/bge-m3-mlx-8bit' produces dim 1024. Rebuild with: conference-pipeline vector-store rebuild --provider mlx`

The exception type is `VectorStoreDimensionMismatch`; legacy
collections without `metadata['dim']` (pre-v1.9) pass through the
guard unchanged — the next rebuild populates the metadata.

---

## `vector-store rebuild` command reference

```
conference-pipeline vector-store rebuild \
    --workspace <path> \
    --provider <openai|mlx|ollama|fake> \
    [--collection-base <name>] \
    [--source <storage_manifest.json>] \
    [--force] \
    [--dry-run] \
    [--operator <id>]
```

**Idempotent:** the source manifest's SHA-256 is recorded in the
collection's `metadata['source_manifest_hash']`. A second invocation
with the same hash is a no-op (`outcome="no-op-idempotent"` in the
receipt). `--force` overrides the idempotency check.

**Receipt:** `VectorStoreRebuildReceipt` written to
`<workspace>/storage/vector-store-receipts/rebuilds/<ts>-<provider>-<outcome>.json`
on every invocation regardless of outcome.

**Note:** the CLI manages collection lifecycle only — it drops + recreates
the empty collection. The operator must re-run the analysis pipeline
to populate it. (This separation avoids duplicating the embedding-and-upsert
loop that already lives in `step_vector_store_setup`.)

---

## `conference-smoketest --local` contract

The `--local` flag opts the smoketest into the local lane:

```bash
conference-pipeline conference-smoketest \
    --workspace <path> --mode <live|dry-run|mock> \
    --local [--local-provider mlx|ollama]
```

**Pre-checks** (executed before any pipeline work):

1. **`OPENAI_API_KEY` must be unset.** If non-empty, the CLI exits 2 with:
   > `--local requires OPENAI_API_KEY to be unset to prove the local-only path. Run: unset OPENAI_API_KEY`
   
   This is a deliberate paranoid check — auto-unsetting hides operator
   configuration drift.
2. **`--local-provider` must be `mlx` or `ollama`.** Default `mlx`. Other values exit 2.

**On pass:** sets env vars `EMBEDDING_PROVIDER=<provider>` and
`GPTR_LOCAL_LLM_RUNTIME_KIND=<provider>` before invoking the smoketest
runner. Downstream factories (`build_embedding_client`,
`build_researcher`) read these.

**`local_e2e` opt-in test lane:** structural end-to-end of the local
factories + adapters lives in `tests/test_local_lane_e2e.py` under
the `@pytest.mark.local_e2e` marker. Run with:

```bash
RUN_LOCAL_E2E=1 pytest -m local_e2e
```

Default `pytest` excludes `local_e2e` (mirrors the existing `live_llm`
marker convention).

---

## Exhaustive extraction with local Qwen MLX

The exhaustive topic-extraction classifier can run against a local Qwen
MLX model in-process; no loopback server is required when
`EXTRACTION_LOCAL_LLM_BACKEND=mlx` and `mlx_lm` can load the configured
model.

For Qwen instruction models, the MLX backend must render chat prompts
through the tokenizer's `apply_chat_template(..., enable_thinking=False)`
when that argument is supported. Plain `SYSTEM:` / `USER:` prompts can
leave Qwen thinking mode enabled, causing it to spend the output budget
on `<think>` reasoning before the JSON object. That is not valid proof
for structured extraction.

Runtime contract:

- `src/conference_pipeline/extractions/backends.py::MLXBackend` owns the
  in-process MLX call path and disables Qwen thinking through the chat
  template when possible.
- `src/conference_pipeline/extractions/classifier.py` parses structured
  local-model JSON defensively, including markdown-fenced JSON or verbose
  output with a final JSON object.
- `EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS` is honored for local assist
  but capped inside the classifier to keep one ambiguous artifact from
  hanging a production run.
- When a recovery stage requires the Qwen MLX lane, the receipt must show
  at least one local LLM attempt and zero local LLM failures; otherwise
  the stage blocks rather than passing as heuristic-only completion.

Minimal environment for the in-process MLX lane:

```bash
export EXTRACTION_LOCAL_LLM_ASSIST=1
export EXTRACTION_LOCAL_LLM_BACKEND=mlx
export EXTRACTION_LOCAL_LLM_MODEL=mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit
export EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL=mlx://mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit
export EXHAUSTIVE_EXTRACTION_PROVIDER=mlx
export EXHAUSTIVE_EXTRACTION_MODEL=mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit
export EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES=false
```

Troubleshooting signal: if a receipt shows `local_qwen_assist_failed`,
inspect the linked local provider receipt and the affected per-artifact
record's `provider_metadata.redacted_message`. Common causes are thinking
mode still enabled, insufficient output budget for the final JSON, or a
model/runtime load problem.

---

## Auto-start lifecycle (`manage_server=True`)

For Ollama (and analogously for the MLX research endpoint), the
embedding/research clients can auto-spawn the daemon on first use:

- `OllamaEmbeddingClient(manage_server=True)` (the v1.9 D4 default) calls
  `subprocess.Popen([shutil.which("ollama"), "serve"])` if the
  configured endpoint is unreachable on the initial probe. Subprocess
  output streams to `.planning/logs/ollama-serve.log` (truncated on
  each spawn) so failed startups (port-in-use, missing model store, GPU
  init failure) are debuggable. `atexit.register(_terminate, proc)`
  ensures cleanup on normal Python exit.
- The spawn loop polls `proc.poll()` each iteration; a crashed daemon
  surfaces as `EmbeddingError("ollama_serve_crashed: exited with code N")`
  immediately rather than wasting the full `startup_timeout_s` on a
  misleading "unreachable" timeout.
- `manage_server=False` skips the spawn — the operator is responsible
  for starting `ollama serve` (or the MLX server) externally.

**Caveat:** `atexit` does not fire on `SIGTERM`/`SIGKILL`/`os._exit`.
Operators with a strict cleanup requirement should run an external
watchdog. The `_MANAGED` dict keyed by base_url ensures at most one
spawned process per workstream.

---

## GPT-Researcher local lane and research cache (GPTR-04/05)

### Local GPT-Researcher as evidence/draft layer

`LocalLLMResearcher` (configured via `GPTR_LOCAL_LLM_BASE_URL`,
`GPTR_LOCAL_LLM_FAST_MODEL`, `GPTR_LOCAL_LLM_SMART_MODEL`) runs as the
**evidence/draft layer** for corpus-grounded retrieval and synthesis. It is
not automatically promoted to the final public-analysis layer. Production
`analysis/<topic>.md` continues to use the governed
`analysis.deep_narrative_synthesis` route unless a local candidate passes all
comparison gates (schema validity, citation/provenance validity, source
diversity, editorial-quality, and cost delta).

### Paid comparison requirements (GPTR-04)

The local-vs-paid comparison lane holds the corpus, retrieval bundle, query
plan, and retrieved evidence constant — only the synthesis model/provider
changes. Comparison receipts must prove:

| Field | Required value |
|---|---|
| `same_corpus` | `true` |
| `same_retrieval_bundle` | `true` — same retrieval bundle digest |
| `same_query_plan` | `true` — same query plan digest |
| `same_retrieved_evidence` | `true` — identical `retrieved_evidence_ids` |
| `paid_synthesis_only` | `true` |

Approval is per topic/task family and requires Phase 03 local-first, embedding,
and cost-avoided proof receipts before a real approval can be recorded.

### Research result cache (GPTR-05)

Unchanged topic/query/corpus tuples skip redundant retrieval and synthesis via
the behavioral research result cache. The cache namespace is
`storage/research-cache/` within the conference workspace.

**Cache key components (D-17):**

- Retrieval bundle digest
- Query plan digest
- Behavioral research parameters digest
- Retrieval mode and K
- Expansion/dedup policy
- Model and provider
- Web-search setting (`WEB_SEARCH_ENABLED`)

**Cacheability rules (D-18):** The cache stores only successful results and
intentionally thin results whose receipts prove true source scarcity.
Transient failures — synthesis failures, provider overloads, timeout errors,
retrieval errors — are never cached.

**Cache-hit/miss receipts (D-20):** Every lookup writes a receipt:

- **Hit:** `<receipts_dir>/<topic_id>-cache-hit-receipt.json` — `hit: true`,
  `cache_key` (SHA-256 hex), `reused_status`, `behavioral_key_components`
  (with retrieval bundle digest, query plan digest, model, provider),
  `source_result_path`, `source_receipt_path`.
- **Miss:** `<receipts_dir>/<topic_id>-cache-miss-receipt.json` — `hit: false`,
  `cache_key`, `topic_id`.

Cache-hit receipts are the exact reuse proof for repeated unchanged pipeline
runs. Conference-specific cache entries live under
`conferences/<slug>/storage/research-cache/` and are excluded from git.

---

## Troubleshooting

| Symptom | Likely cause | Diagnosis / fix |
|---|---|---|
| `EmbeddingError: non_loopback_base_url` at construction | `OLLAMA_BASE_URL` or `GPTR_LOCAL_LLM_BASE_URL` points at a non-loopback host | Use `127.0.0.1`, `localhost`, `127.x.y.z`, `::1`, or `[::ffff:127.0.0.1]`. `0.0.0.0` is rejected (binds all interfaces). |
| `EmbeddingError: ollama_unreachable` after spawn | Port 11434 already in use OR Ollama daemon crashed at startup | `tail -f .planning/logs/ollama-serve.log`; `lsof -i :11434`; `ollama list` to verify install |
| `EmbeddingError: ollama_serve_crashed: exited with code N` | Spawned `ollama serve` exited before becoming reachable | See log file path in the error message; common: missing model dir permission, GPU init failure |
| `EmbeddingError: MLXEmbeddingClient requires the mlx-embeddings package` | `mlx-embeddings` not installed in the active env | `pip install -e ".[mlx]"` (silently no-op on non-Mac) |
| `VectorStoreDimensionMismatch` | Switched providers; persisted dim differs from active client | Run `conference-pipeline vector-store rebuild --provider <new-provider>`; the existing cloud collection stays intact (different name suffix) |
| `--local requires OPENAI_API_KEY to be unset` | The smoketest's pre-check fired | `unset OPENAI_API_KEY` then re-run; intentional paranoid check |
| `unknown embedding provider 'cohere'; valid: 'openai', 'mlx', 'ollama', 'fake'` | Typo in `EMBEDDING_PROVIDER` | Fix to one of the four valid values |

---

## Cross-references

- [docs/source-of-truth.md](source-of-truth.md) — top-level platform contract.
- [docs/reference/pipeline-capabilities.md](reference/pipeline-capabilities.md) — stage descriptions including local-lane support.
- v1.9 milestone planning: `.planning/workstreams/v1-9/PROJECT.md`, `.planning/workstreams/v1-9/milestones/v1.9-REQUIREMENTS.md`, `.planning/workstreams/v1-9/ROADMAP.md`.
- Phase verifications (archived with the milestone): Phase 42 (`.planning/workstreams/v1-9/milestones/v1.9-phases/42-embedding-engine-abstraction/42-VERIFICATION.md`), Phase 43 (`.planning/workstreams/v1-9/milestones/v1.9-phases/43-mlx-embedding-adapter/43-VERIFICATION.md`), Phase 44 (`.planning/workstreams/v1-9/milestones/v1.9-phases/44-ollama-embedding-adapter/44-VERIFICATION.md`), Phase 45 (`.planning/workstreams/v1-9/milestones/v1.9-phases/45-vector-store-provider-switch/45-VERIFICATION.md`), Phase 46 (`.planning/workstreams/v1-9/milestones/v1.9-phases/46-local-llm-researcher-adapter/46-VERIFICATION.md`).

---

## Vision & Slide Analysis (v1.61)

v1.61 extends the local-LLM lane with **multimodal slide analysis** —
slides are rendered to PNG/JPEG/WebP images, sent to a local
vision-language model, and parsed into a structured
`SlideAnalysisResult` (salient claims, figures, code blocks, equations,
headings, body text, reading-order hint, confidence). A sibling local
lane is also added to the existing text-only OCR engine
(`LLMOcrEngine(provider="local")`).

The vision lane is **opt-in and additive**. The OpenAI/Anthropic vision
default stays in production until per-task benchmark receipts (v1.6
MODEL-BENCH-01,02) promote a local model to `approved`.

### Runtime choice

| Path | When to use | Code | Deps |
|---|---|---|---|
| **vllm-mlx server** (managed loopback) | Most multimodal VLMs (Qwen3-VL, Qwen2.5-VL, Pixtral, Gemma-3 vision, SmolVLM, Idefics3, Molmo, Phi-3 Vision, LLaVA), plus OCR-specialized GLM-OCR / dots.ocr / DeepSeek-OCR-2. | `model_governance/mlx_vision_client.py` (`MlxVisionServerClient`, `ManagedMlxVisionServer`) | `mlx-vlm` or `vllm-mlx` |
| **Native MLX FastVLM** (in-process) | Apple's `apple/FastVLM-{0.5,1.5,7}B-MLX` — sub-second TTFT, image-only. | `model_governance/fastvlm_native_client.py` (`FastVLMNativeClient`) | `mlx-vlm` |

Both lanes write the same receipt shape (`LocalVisionCallReceipt`); only
the `runtime_kind` differs (`vllm_mlx` vs `mlx_native_fastvlm`).

### Model catalog (Appendix A)

The full v1.61 candidate roster lives in
[registry.yaml](../src/conference_pipeline/model_governance/registry.yaml)
under `vision.slide_analysis`, `vision.figure_caption`,
`vision.chart_summary`, `presentation.visual_qc`, and
`ocr.slide_image_parse`. Defaults:

- **General slide analysis (sweet spot):**
  `mlx-community/Qwen3-VL-8B-Instruct-4bit`
- **Memory-rich (≥48 GB UM):** `mlx-community/Qwen3-VL-32B-Instruct-4bit`
- **Sub-second TTFT:** `apple/FastVLM-7B-MLX`
- **Stylistic alt:** `mlx-community/Pixtral-12B-4bit`
- **Pure OCR:** `mlx-community/GLM-OCR-0.9B-4bit` (default once
  benchmark receipts exist; `mlx-community/dots-ocr-3B-4bit` and
  `mlx-community/DeepSeek-OCR-2-3B-4bit` as alternatives)

All entries enter the registry with `status: benchmark_pending`. They
are promoted to `approved` only via
`conference-pipeline promote-vision-candidate`, which fails closed
without a receipt.

### Environment variables

| Variable | Purpose |
|---|---|
| `GPTR_LOCAL_VISION_MODEL` | Active vision model id (mirror of `--local-vision-model`) |
| `GPTR_LOCAL_VISION_RUNTIME_KIND` | `vllm_mlx` or `fastvlm` |
| `GPTR_LOCAL_VISION_BASE_URL` | Loopback URL of the vllm-mlx server (default: `http://127.0.0.1:8090/v1`) |

### Smoketest example

```bash
unset OPENAI_API_KEY  # required by --local
conference-pipeline conference-smoketest \
    --workspace conferences/iclr2025 \
    --mode dry-run \
    --local \
    --local-vision-model mlx-community/Qwen3-VL-8B-Instruct-4bit \
    --local-vision-provider vllm_mlx
```

For native FastVLM:

```bash
unset OPENAI_API_KEY
conference-pipeline conference-smoketest \
    --workspace conferences/iclr2025 \
    --mode dry-run \
    --local \
    --local-vision-model apple/FastVLM-7B-MLX \
    --local-vision-provider fastvlm
```

The `--local-vision-model` flag requires `--local`; using it without
`--local` fails with exit code 2. Reuses `--local`'s loopback /
`OPENAI_API_KEY=unset` enforcement.

### Receipts

Each call writes a `LocalVisionCallReceipt` to:

- With run id:
  `<workspace>/storage/workflow-runs/<run-id>/local-vision-receipts/<stage>/<ts>-vision.json`
- Without run id:
  `<workspace>/storage/local-vision-receipts/<stage>/<ts>-vision.json`

Receipts record the redacted endpoint label
(`mlx-vision://<model>` or `mlx-fastvlm://<model>`), model id, image
dimensions and byte size, latency, and `cost_usd: 0.0`. Image bytes
themselves are never persisted.

### Promotion CLI

```bash
conference-pipeline promote-vision-candidate \
    --task vision.slide_analysis \
    --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
    --receipt path/to/receipt.json \
    [--thresholds path/to/thresholds.json] \
    [--dry-run]
```

Promotion is **only** applied when the supplied receipt clears all
configured per-metric thresholds (`text_recall`,
`figure_caption_similarity`, `chart_summary_similarity`,
`parse_success`, `overall`). No receipt → fail-closed. Receipt below
thresholds → fail-closed. The registry write is atomic and preserves
default-pick comments.

### Vision troubleshooting

| Symptom | Likely cause | Diagnosis / fix |
|---|---|---|
| `VisionRuntimeUnavailableError: neither mlx_vlm nor vllm_mlx is installed` | `mlx-vlm` not in the active environment | `pip install -e ".[mlx]"` (silently no-op on non-Mac) |
| `ModelGovernanceError: vision server at ... unreachable` | Vision server crashed or never bound | Check logs, ensure `--local-vision-base-url` matches the running port |
| `MlxVisionServerClient requires a loopback base_url` | Non-loopback URL provided | Use `127.0.0.1` / `localhost` / `::1` only |
| `FastVLMTextOnlyUnsupported` | Tried to call FastVLM without an image | FastVLM is image-only; for text use the vllm-mlx lane |
| `LocalVisionRuntimeError: ... OOM` | Model larger than available unified memory | Drop to a smaller candidate (e.g. Qwen3-VL-8B-4bit instead of 32B-4bit) |
| `SlideAnalysisParseError` | Model output not valid JSON or structured markdown | Inspect the truncated raw output on the exception; consider switching `output_format` |
| `PromotionRequiresReceiptError` | Manual promotion attempted without `--receipt` | All promotion goes through receipts; this is by design |

### Status (v1.61)

- ✅ Registry (Phase 65)
- ✅ vllm-mlx adapter + `LocalVisionCallReceipt` (Phase 66)
- ✅ Guard removal + `SlideAnalysisEngine` + `LLMOcrEngine` local branch (Phase 67)
- ✅ Native FastVLM adapter (Phase 68) — code complete; real Apple-Silicon weight load deferred to operator verification
- ✅ Benchmark scoring + `promote-vision-candidate` (Phase 69) — code complete; fixture pack with ground-truth JSON deferred (see [`tests/fixtures/vision/slides/README.md`](../tests/fixtures/vision/slides/README.md))
- ✅ Smoketest CLI flags + operator status + cost registry (Phase 70)
- ✅ This documentation (Phase 71); UAT receipt deferred to operator session

---

*Operator reference for v1.9 Local-First AI Stack and v1.61 Local Visual LLM Stack. Last updated: 2026-05-06.*
