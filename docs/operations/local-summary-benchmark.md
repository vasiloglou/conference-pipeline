# Short-context Stage 07A benchmark

This experimental lane evaluates smaller local models against the existing
exhaustive topic summaries. It never promotes a model or writes into production
`analysis/exhaustive_summaries`. Model choices belong to the governed
`analysis.local_summary_benchmark` task. Conference results belong in the
conference milestone context and its experiment receipts.

## Preparation and execution

Create an isolated output directory under workspace storage. The snapshot
command records every source unit, checks for missing IDs, and verifies the
saved production baseline's exact input fingerprint:

```sh
.venv/bin/python scripts/analysis/prepare_local_summary_benchmark.py \
  --workspace conferences/<slug> --run-id <production-run-id> \
  --pair <topic-id>:papers --pair <topic-id>:transcripts \
  --out <experiment-root>/inputs

.venv/bin/python -m conference_pipeline.analysis.local_summary_benchmark \
  --input <experiment-root>/inputs/<topic-id>__papers.json \
  --out <experiment-root>/screen/local-ledger \
  --budget-root <experiment-root> --alias qwen36_local \
  --strategy ledger --ceiling 16384 --output-tokens 8192 --sample 12
```

Omit `--sample` for complete-pair evaluation. `--order 1` and `--order 2`
provide deterministic reorderings. `--strategy simple` is a plain bounded
map/reduce control. Compare 8192, 16384, and 32768 input-token ceilings; the
ceiling counts the fully rendered chat template. The output cap is separate.
For an overnight local queue, `run_local_summary_campaign.py` accepts a JSON
array of explicit jobs with `input`, `out`, and optional runner settings
(`alias`, `strategy`, `ceiling`, `output_tokens`, `sample`, `order`,
`reread_rounds`). Inspect it without `--execute` first. With `--execute`, it
runs serially, caps wall time at twelve hours, respects the shared compute
budget, records failed/skipped jobs in `campaign-results.json`, and refreshes
the report after each job. `--after-pid` waits for an already-running local
experiment to finish. Cloud evaluation is disabled by default. After explicit
payload/destination authorization, `--cloud-scope papers --credential-file
<trusted-file>` evaluates only paper jobs and runs a bounded cloud final pass
for ledger jobs; transcript jobs remain local. `--cloud-scope all` requires
separate authorization covering transcripts. Cloud calls run between GPU jobs
under the same persisted dollar budget. The campaign never promotes a model.
For narrower transcript authorization, retain `--cloud-scope papers` and pass
each permitted snapshot with `--transcript-input <input.json>`. Other transcript
inputs remain excluded from cloud processing. `--freeze-references` prepares
forty-source full-pair references after any `--after-pid` worker exits and before
the queued local runs. Transcript hybrid synthesis uses complete generated
summary packs (`--hybrid-evidence summary-packs`) rather than the raw-quote ledger.
Local generation disables thinking and uses deterministic sampling. The
current adapters require cached MLX-compatible weights. The heavy runtime lock
and an experiment-level OS advisory lock serialize GPU and budget use.
Use `--reread-rounds 0` only for the extraction-interface ablation; the full
ledger method defaults to two bounded verification rounds. A stopped local
response generated with a smaller output cap can be reused under a larger cap;
a truncated response cannot. Intermediate reductions receive a shorter output
target so their complete contents can fit in later merges.
If two otherwise valid packs cannot fit together, the reducer can re-synthesize
each complete pack with that shorter target. It requires measured shrinkage
and permits at most two such rounds before failing. This can lose information;
all earlier complete levels remain in the receipts for evaluation.

The ledger lane records a processed/excluded disposition for every source
part, selects numbered source sentences, recovers their exact supporting spans, groups claims by normalized theme label,
synthesizes bounded batches, recursively reduces them, and performs at most two
source rereading passes. Each critic round reviews every compact ledger packet,
including sources absent from the draft, then selects at most six sources for
rereading. The current ledger prompt returns sentence IDs, so source quotes are copied by
code rather than regenerated. Legacy pilot quote responses may be revalidated
for whitespace, paired dollar math delimiters, or paired Markdown emphasis
only; original offsets are retained. Neither sentence identity nor formatting
alignment establishes semantic entailment. Independent claim checking remains
necessary.

Every reduction prompt is checked. Oversized source units split with exact
character offsets. Oversized summary records or reductions that cannot shrink
fail rather than truncating. Invalid ledger/summary output gets one repair
attempt. Truncated generation fails. Raw responses remain available even when
validation fails. A later validator may revalidate a cached stopped response;
the original failure is retained alongside a separate revalidation receipt.

## Independent evaluation and hybrid synthesis

The evaluation script imports only named API keys from a trusted credential
file using `dotenv_values`; it never sources arbitrary shell code. Specify the
workflow's documented repository credential file. Cloud calls require explicit
operator authorization for the evidence and provider destination.

```sh
.venv/bin/python scripts/analysis/evaluate_local_summaries.py references \
  --input <input.json> --out <reference.json> --root <experiment-root> \
  --credential-file <trusted-credential-file>

.venv/bin/python scripts/analysis/evaluate_local_summaries.py evaluate \
  --input <input.json> --reference <reference.json> \
  --candidate <baseline-summary.json> --candidate <candidate-summary.json> \
  --out <evaluation.json> --root <experiment-root> \
  --credential-file <trusted-credential-file>
```

Use `--sample 12` consistently across preparation of references, matched sample
baselines, and evaluation when screening. Freeze reference findings before
reviewing candidate summaries. Default references cover 40 position-stratified
units with source-verified quotes and twelve research questions. Independent
judging measures recall per finding and support of claims against supplied
source packets. A bounded pairwise evaluation swaps candidate order. These
are model-assisted judgments with reviewable evidence, not human gold labels.

`baseline` generates a cloud comparison only when the complete selected input
fits its ceiling. Larger pairs use fingerprint-matched saved production
baselines. `hybrid` accepts one candidate summary and its adjacent ledger and
performs one bounded cloud synthesis. If the full ledger exceeds the cloud ceiling, it uses the earliest complete
local reduction level that fits and records that compression in an input-coverage
sidecar. It never silently selects only the first claims. If no complete level
fits, the hybrid call fails.

## Budgets, receipts, and limitations

A shared `budget.json` persists conservative dollar reservations before calls,
including failed calls. Limits are $25 total, $10 synthesis, and $15 evaluation.
Reservations use the canonical pricing catalog, UTF-8 byte input bounds and
output caps with headroom; they are conservative catalog-based estimates, not
provider invoices. Successful calls settle to reported-token catalog cost plus
25% headroom; failed calls retain their full reservation. Original bounds remain
recorded. Retries are not automatic at the API client. Local time is
capped at twelve hours with a fifteen-minute per-call deadline checked during
prefill and generation. Crashed local calls retain their time reservation.
GPU kernels cannot always be interrupted immediately.

Receipts preserve prompt, response, model revision, tokens, time, memory,
swap measurements when available, and errors. Input and prompt hashes support
resume. Payloads are ignored by Git; retain small human-reviewed reports in
milestone documentation. Full corpus results must never be claimed from
`--sample` runs. A skipped pairwise comparison is not a passing comparison.
The numeric-support diagnostic flags numbers absent from cited source text.
It does not establish entailment, correct units, or correct experimental conditions.

Pilot acceptance: complete input accounting and no unknown IDs; at least 95%
audited claim support and no material fabricated number; recall and
contradiction/caveat coverage within five percentage points of the cloud
comparator; synthesis within 0.25/5; and 80% lower paid synthesis cost than the
matched current cloud approach. Report uncertainty, every failed pair, and
local runtime. Do not infer equivalence from output length or valid IDs.

Known experimental limits: theme labels are model-generated and lexical
sorting does not guarantee semantic clustering; ledger-wide rereading can
recover omitted extracted findings but cannot recover findings absent from the ledger;
reference sampling does not exhaustively assess global completeness. These
limits must remain visible in the recommendation and motivate subsequent
ablations if the pilot passes.

## Research basis and scope

The working hypothesis combines exhaustive coverage with bounded synthesis and
source rereading. Recursive Language Models treat source text as addressable
external state rather than requiring every token in one context; GraphRAG
explores community summaries for corpus-level questions. These results motivate
an experiment; they do not establish parity for this pipeline or model size.
The original RLM experiments used frontier models, including a 480B-parameter
open model, and one setup used a stronger root model with cheaper recursive
calls. This supports testing a hybrid final pass, not assuming that a 35B
local model will automatically inherit the published gains.

- [Recursive Language Models](https://arxiv.org/abs/2512.24601)
- [From Local to Global: GraphRAG](https://arxiv.org/abs/2404.16130)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)

Stage 07A currently reconstructs paper **abstracts** and transcript windows
selected by the upstream full-read topic assignments. A matched-input benchmark
cannot establish recall of full paper bodies or detect everything omitted by
upstream topic assignment. Report that scope when interpreting results.

The larger local-model shortlist can include GLM-4.7-Flash (30B-A3B) and
Kimi-Linear-48B-A3B-Instruct; both have community MLX conversions and architecture
implementations in the inspected local runtime. Actual loading, memory, chat
template behavior and quality still need measurement. Flagship Kimi-K3 and
GLM-5.3-Flash are not comparable laptop-sized weights. Candidate additions must
go through the catalog and experiment task selection without changing
production defaults.


## Controlled context comparison

`run_context_summary_study.py` compares a governed local alias at 16384, 65536, and 98304 total tokens against a fresh governed Kimi full-corpus output. Supply `--input`, `--root` (a new isolated study directory), `--api-root` (the existing shared dollar budget), `--reference`, and `--credential-file`. Without `--execute`, it verifies the frozen source fingerprint and prints the protocol without inference. Add `--execute` after checking the protocol.

Extraction and output caps are fixed at 8192 tokens, while synthesis input limits are 8192/57344/90112. Each setting gets at most four local compute hours within a new twelve-hour study allowance. Existing API budgets are reused, never reset. Local jobs run serially and record failures separately; successful summaries receive the same independent reference recall and source-support evaluation. Inspect `study-results.json`, per-setting logs, and `REPORT.md`. Actual measured prompt lengths determine whether a larger ceiling was exercised.
