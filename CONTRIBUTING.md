# Contributing

Thank you for helping. This file says how to set up, what a change must pass, and where
things go.

## Set up

```bash
uv sync --all-extras --frozen
.venv/bin/python -m playwright install chromium
```

Install [gitleaks](https://github.com/gitleaks/gitleaks) too: the export and release gates
use it and fail closed without it. When you change a dependency, update `uv.lock`
(`uv lock`) in the same pull request.

## Before you open a pull request

Run these from the repository root. Each must exit 0. On macOS, run `ulimit -n 4096`
first; the default open-file limit is too low for parallel tests.

```bash
.venv/bin/python -m pytest -q -n auto -p no:cacheprovider -rfs tests
.venv/bin/python scripts/check_docs_links.py
PYTHONPATH=src .venv/bin/python scripts/audit_private_markers.py --paths . --no-env
PYTHONPATH=src .venv/bin/python scripts/audit_conference_hardcoding.py
PYTHONPATH=src .venv/bin/python scripts/audit_model_registry_hardcoding.py
```

`gitleaks git .` should report nothing.

## Rules the code follows

- **Reusable code has no conference in it.** Platform code lives in
  `src/conference_pipeline/`, `scripts/`, `tests/` and `docs/`. A conference's facts live
  in its workspace, `conferences/<slug>/config.yaml` and its manifests. The hardcoding
  audit enforces this.
- **Models come from the registry.** A model is chosen by task id through
  `src/conference_pipeline/model_governance/registry.py`. Never write a model name in code;
  the model-registry audit enforces this.
- **Dry-run or mock first.** Every workflow change is shown in `--mode dry-run` or
  `--mode mock` before any live run.
- **No secrets or private paths.** Keys go in `.env`, which git ignores. Never commit a
  home-directory path, a personal email address or a key, even in a test: build such a
  value at run time, as the existing tests do.
- **Search before you write.** `python3 scripts/components_index.py search <words>` lists
  what already exists. When you add, move or retire a module, update its entry in
  `components/COMPONENTS.md` and run `python3 scripts/components_index.py write`.

## Tests that need private data

A test that reads files this repository does not carry (a produced conference workspace,
planning records) is marked:

```python
@pytest.mark.repo_data("conferences/<slug>/config.yaml")
```

It is skipped, with the missing path named, wherever that path is absent. New tests
should use fixtures under `tests/fixtures/` instead where they can.

## License

By contributing, you agree that your contribution is licensed under the MIT License
([LICENSE](LICENSE)).
