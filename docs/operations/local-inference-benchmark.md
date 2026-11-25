# Local inference engine comparison

Use `scripts/analysis/benchmark_local_inference.py` to compare inference speed
before a local summary quality campaign. It writes isolated receipts and never
changes production model selections or summary outputs. Model manifests resolve
the allowed `analysis.local_summary_benchmark` selections through the model
registry; workers consume that frozen manifest on another Mac without copying
credentials or the repository.

## Protocol

Prepare the manifest on the authoritative checkout:

```sh
.venv/bin/python scripts/analysis/benchmark_local_inference.py plan \
  --aliases qwen36_local,qwen38_local \
  --contexts 2048 16384 --repetitions 3 --output-tokens 256 \
  --out <benchmark-root>/plan.json
```

Copy only this script, the manifest, and any missing model snapshot to the
benchmark host. Verify the actual snapshot revision matches the manifest.
Use isolated environments for engine dependencies, retaining their exact package
versions in the receipts. Run one heavy GPU job at a time and stop each managed
server before starting another. Do not interrupt unrelated jobs.

Compare the existing direct MLX configuration, direct MLX with a larger prefill
step, and the alternative serving engines. The direct worker uses the same
512 MiB MLX buffer-cache limit as the summary experiment. For example:

```sh
<mlx-python> benchmark_local_inference.py worker \
  --plan <plan.json> --alias <governed-alias> --model-path <snapshot> \
  --engine direct --prefill 512 --out <results-root>/mlx-current

<mlx-python> benchmark_local_inference.py worker \
  --plan <plan.json> --alias <governed-alias> --model-path <snapshot> \
  --engine direct --prefill 2048 --out <results-root>/mlx-prefill-2048

<server-python> benchmark_local_inference.py worker \
  --plan <plan.json> --alias <governed-alias> --model-path <snapshot> \
  --engine vllm-metal --url http://127.0.0.1:<port> \
  --out <results-root>/vllm-metal
```

The API worker supports `vllm-metal` and `vllm-mlx`; both use raw
`/v1/completions` so all engines receive the same pre-rendered chat template.
Disable prefix caching for the serving engines. Use a context limit larger than
the largest prompt plus the output cap and reserve memory headroom for the OS.
Record the actual startup command alongside the worker receipts.

## Interpretation

```sh
<python> benchmark_local_inference.py report --root <results-root>
```

Compare median time to first token, generation rate, and end-to-end output rate
separately. The worker excludes a warmup call, uses deterministic sampling with
thinking disabled, stores every prompt and output hash, and records package
versions and OS memory counters before and after each call. Requested prompt
lengths are nominal; tokenization may differ by a token or two. Compare actual
counts and require matching prompt hashes across engines for each model/trial.

The streaming decode estimate counts all output tokens after the first over
the interval between first and last output events; text buffering can affect it.
Direct MLX also records the runtime's native generation and prompt rates.
The end-to-end metric includes prefill. A token-limit finish is expected for
the bounded speed test and is not proof that a full summary completed.

These are synthetic source-synthesis prompts, not a summary quality evaluation.
Engine bundles may require different MLX versions; any resulting improvement
belongs to the tested complete stack until a version-matched control isolates
the cause. Run a direct MLX control in the winning server's environment when
that distinction changes the recommendation. Single-request results do not
establish concurrency scaling. Before adopting a different model or altered
decoding algorithm, use the independent source-support and recall evaluation
in [the local summary benchmark](local-summary-benchmark.md).

Record competing CPU work as well as GPU work: CPU jobs share memory bandwidth
with the GPU and can affect dispatch. The worker captures system load averages.
Repeat a short baseline at the end of a long sweep to detect performance drift.
If the initial and final baseline disagree materially, retain both results and
avoid attributing small differences to library versions. A shared-machine run
supports a practical recommendation under that load, not an isolated hardware
or kernel ranking.
