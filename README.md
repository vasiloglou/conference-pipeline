# Conference Pipeline

A Python platform that turns a machine-learning conference into something you can read in
an afternoon. It acquires the program (papers, talks, posters), transcribes and indexes it,
analyzes it by topic, and produces decks, narrated videos, podcasts and a searchable public
site per conference, with a receipt for every stage.

It has been run end to end on NeurIPS 2025, ICLR 2025, ICLR 2026, ICML 2026 and CVPR 2026.
The produced data for those conferences is published separately, as data releases; this
repository holds the code, its tests and docs, and an empty starter workspace.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are needed. `uv sync` installs the exact
versions in `uv.lock`, the ones the test suite is checked against, on the Python that
`.python-version` names:

```bash
git clone https://github.com/vasiloglou/conference-pipeline.git
cd conference-pipeline
uv sync --all-extras --frozen
.venv/bin/python -m playwright install chromium   # for the site, slide and PDF checks
```

The release and export gates also need [gitleaks](https://github.com/gitleaks/gitleaks) on
`PATH`. Without it they fail closed, and so do their tests.

`pip install -e ".[all,dev]"` also works, but it resolves the newest versions, which the
suite is not pinned to: a few byte-exact golden tests may then differ. `pip install -e .`
alone installs the core; the extras are listed in `pyproject.toml` (`browser`, `media`,
`ocr`, `transcription`, `llm`, `mlx`, `publishing`, `notebooklm`). The `mlx` packages
install only on Apple Silicon.

## Try it in mock mode

Mock mode needs no API key and spends nothing. It runs the first four stages over the
starter workspace in `conferences/TEMPLATE/`:

```bash
.venv/bin/conference-pipeline workflow run --workspace conferences/TEMPLATE \
  --mode mock --stages discover,download,transcribe,analyze --run-id smoke
```

It exits 0 and writes a receipt per stage, plus
`conferences/TEMPLATE/storage/workflow-runs/smoke/summary.json`. The later stages (decks,
podcasts, videos) need a real analysis, so they need provider keys or local models.

## Run the tests

```bash
.venv/bin/python -m pytest -q -n auto -p no:cacheprovider -rfs tests
```

On macOS, raise the open-file limit first (`ulimit -n 4096`): the default of 256 is too
low for one test worker per core.

Some tests read conference workspaces or planning records that live only in the
maintainer's private checkout. In this repository they are skipped, and each skip names
the missing path (`repo_data: <path> is not in this checkout`). Tests that need a live
model endpoint are skipped unless `RUN_LIVE_LLM=1` or `RUN_LOCAL_E2E=1` is set.

## Start a conference of your own

1. Copy `conferences/TEMPLATE/` to `conferences/<slug>/` and fill in its `config.yaml`
   (see [docs/getting-started.md](docs/getting-started.md) and
   [docs/workflows/pipeline-workflow.md](docs/workflows/pipeline-workflow.md)).
2. Put provider keys in a `.env` at the repository root. `.env.example` lists every
   variable; nothing reads keys from anywhere else.
3. Run stages in `--mode dry-run` first, then `live`. Each stage writes a receipt under
   `storage/workflow-runs/<run-id>/`.
4. Publish with `conference-pipeline release init|plan|build|check|publish <slug>`
   ([docs/operations/public-release.md](docs/operations/public-release.md)). `publish`
   refuses to run without `--approved-by`.

Paid runs are estimated before they start
([docs/reference/pipeline-capabilities.md](docs/reference/pipeline-capabilities.md));
model choices come from one registry,
`src/conference_pipeline/model_governance/registry.yaml`.

## Find your way around

- [docs/index.md](docs/index.md): the documentation index.
- [components/COMPONENTS.md](components/COMPONENTS.md): what the code already does, by
  capability. Search it with `python3 scripts/components_index.py search <words>`.
- `src/conference_pipeline/`: the platform. `scripts/`: command-line entry points.
  `tests/`: the offline suite. `.claude/skills/`: agent skills for the pipeline stages and
  the development workflow.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). To cite this
software, see [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE). The data releases carry their own licenses, recorded per
artifact family in each release.
