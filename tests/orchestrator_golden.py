"""Golden-receipt harness for thin-orchestrator lifts.

Characterization tests (wave W1 of the thin-orchestrator plan) run a *current*
runner stage on a fixture workspace, snapshot its receipt, and require the
snapshot to match exactly after the stage is lifted into a shared module. A
snapshot keeps what must be byte-stable across a lift and drops what changes
on every run:

* ``receipt``: the receipt with volatile keys (timestamps, durations, receipt
  ids and paths) removed, ISO timestamps anywhere replaced by
  ``<TIMESTAMP>``, and absolute workspace / repo roots replaced by
  ``<WORKSPACE>`` / ``<REPO>``. The ``lineage`` block stays in, normalized the
  same way, so upstream digests, input shas and code identity are compared.
* ``lineage_digest``: the recorded ``lineage.lineage_digest`` plus
  ``lineage_digest_consistent`` (it equals a recomputation with
  :func:`conference_pipeline.receipt_lineage.lineage_digest`).
* ``output_digest``: the recorded ``output_digest`` plus
  ``output_digest_consistent`` (it equals
  :func:`conference_pipeline.receipt_lineage.compute_output_digest` over the
  recorded ``output_artifacts`` and components).
* ``output_files``: every output artifact (and any extra path) hashed from
  disk now: JSON canonically without run stamps, directories by listing.

Nothing here runs a stage or a provider. W0 ships the harness and the fixture
workspaces only; real stage receipts are pinned in W1.

Set ``ORCHESTRATOR_GOLDEN_UPDATE=1`` to (re)write a golden file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.receipt_lineage import (
    canonical_output_sha256,
    compute_output_digest,
    directory_sha256,
    lineage_digest,
    sha256_file,
    workspace_relpath,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "orchestrators"
FIXTURE_REPO = FIXTURE_ROOT / "repo"
FIXTURE_NAMES = ("icml_shaped", "cvpr_shaped", "minimal_new_conference")
UPDATE_ENV = "ORCHESTRATOR_GOLDEN_UPDATE"

VOLATILE_KEYS = frozenset(
    {
        "started_at",
        "ended_at",
        "finished_at",
        "generated_at",
        "updated_at",
        "bound_at",
        "last_run_at",
        "taken_at",
        "elapsed_seconds",
        "duration_ms",
        "wall_seconds",
        "receipt_id",
        "receipt_path",
        "summary_path",
        "supersedes_receipt_path",
        "superseded_prior_receipt_path",
    }
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$")
TIMESTAMP_PLACEHOLDER = "<TIMESTAMP>"
WORKSPACE_PLACEHOLDER = "<WORKSPACE>"
REPO_PLACEHOLDER = "<REPO>"

_MINIMAL_PDF = (
    "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    "2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
    "% synthetic fixture: {name}\ntrailer<</Root 1 0 R>>\n%%EOF\n"
)


# --------------------------------------------------------------------------
# Fixture workspaces
# --------------------------------------------------------------------------


def load_fixture_manifest(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name / "fixture.json").read_text(encoding="utf-8"))


def materialize_fixture_workspace(name: str, dest: Path) -> Path:
    """Copy a fixture workspace into ``dest/name`` and write its tiny PDFs.

    PDFs are gitignored repo-wide, so the fixture declares them in
    ``fixture.json`` (``generated_pdfs``) and they are generated here,
    deterministically, from their own relative path.
    """

    source = FIXTURE_ROOT / name
    target = Path(dest) / name
    shutil.copytree(source, target)
    for relative in load_fixture_manifest(name).get("generated_pdfs") or ():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_MINIMAL_PDF.format(name=Path(relative).stem).encode("ascii"))
    return target


# --------------------------------------------------------------------------
# Normalization and snapshots
# --------------------------------------------------------------------------


def _root_variants(root: Path | None) -> list[str]:
    if root is None:
        return []
    variants = {str(root), str(Path(root).resolve())}
    # macOS temp dirs appear both as /var/... and /private/var/...
    for item in list(variants):
        if item.startswith("/private/"):
            variants.add(item[len("/private") :])
    return sorted(variants, key=len, reverse=True)


def normalize_receipt(
    payload: Any,
    *,
    workspace: Path | None = None,
    repo_root: Path | None = None,
    volatile_keys: Iterable[str] = VOLATILE_KEYS,
) -> Any:
    """Drop volatile keys, blank timestamps, and relativize absolute roots."""

    volatile = frozenset(volatile_keys)
    replacements = [(variant, WORKSPACE_PLACEHOLDER) for variant in _root_variants(workspace)]
    replacements += [(variant, REPO_PLACEHOLDER) for variant in _root_variants(repo_root)]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {str(key): walk(value) for key, value in node.items() if str(key) not in volatile}
        if isinstance(node, (list, tuple)):
            return [walk(value) for value in node]
        if isinstance(node, str):
            if TIMESTAMP_RE.match(node):
                return TIMESTAMP_PLACEHOLDER
            for root, placeholder in replacements:
                if root and root in node:
                    node = node.replace(root, placeholder)
            return node
        return node

    return walk(payload)


def _output_file_digest(path: Path) -> str | None:
    if path.is_dir():
        return directory_sha256(path)
    if not path.is_file():
        return None
    return canonical_output_sha256(path) or sha256_file(path)


def output_file_digests(
    workspace: Path, receipt: Mapping[str, Any], extra_paths: Iterable[str | Path] = ()
) -> dict[str, str | None]:
    """Workspace-relative path -> digest, hashed from disk now."""

    workspace = Path(workspace)
    paths: list[str | Path] = []
    for item in receipt.get("output_artifacts") or ():
        if isinstance(item, Mapping) and isinstance(item.get("path"), str) and item["path"]:
            paths.append(item["path"])
    paths.extend(extra_paths)
    digests: dict[str, str | None] = {}
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        digests[workspace_relpath(workspace, candidate)] = _output_file_digest(candidate)
    return dict(sorted(digests.items()))


def snapshot_receipt(
    receipt: Mapping[str, Any],
    *,
    workspace: Path,
    repo_root: Path | None = None,
    extra_output_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    lineage = receipt.get("lineage") if isinstance(receipt.get("lineage"), Mapping) else None
    recorded_lineage_digest = lineage.get("lineage_digest") if lineage else None
    artifacts = [item for item in receipt.get("output_artifacts") or () if isinstance(item, Mapping)]
    components = (lineage or {}).get("output_digest_components") or None
    recorded_output_digest = receipt.get("output_digest")
    return {
        "receipt": normalize_receipt(dict(receipt), workspace=workspace, repo_root=repo_root),
        "lineage_digest": recorded_lineage_digest,
        "lineage_digest_consistent": (
            None if lineage is None else recorded_lineage_digest == lineage_digest(lineage)
        ),
        "output_digest": recorded_output_digest,
        "output_digest_consistent": (
            None
            if recorded_output_digest is None
            else recorded_output_digest == compute_output_digest(artifacts, components=components)
        ),
        "output_files": output_file_digests(workspace, receipt, extra_output_paths),
    }


# --------------------------------------------------------------------------
# Comparison and golden files
# --------------------------------------------------------------------------


def compare_snapshots(expected: Any, actual: Any, *, path: str = "") -> list[str]:
    """Every difference between two snapshots, as ``path: expected ... got ...``."""

    label = path or "<root>"
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                differences.append(f"{child}: missing (expected {expected[key]!r})")
            elif key not in expected:
                differences.append(f"{child}: unexpected {actual[key]!r}")
            else:
                differences.extend(compare_snapshots(expected[key], actual[key], path=child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{label}: length {len(expected)} expected, got {len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(compare_snapshots(left, right, path=f"{path}[{index}]"))
        return differences
    if expected != actual:
        return [f"{label}: expected {expected!r}, got {actual!r}"]
    return []


def write_golden(snapshot: Mapping[str, Any], golden_path: Path) -> None:
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_matches_golden(snapshot: Mapping[str, Any], golden_path: Path) -> None:
    """Compare against a golden file; ``ORCHESTRATOR_GOLDEN_UPDATE=1`` rewrites it."""

    golden_path = Path(golden_path)
    if os.environ.get(UPDATE_ENV):
        write_golden(snapshot, golden_path)
        return
    if not golden_path.is_file():
        raise AssertionError(f"golden receipt missing: {golden_path}; rerun with {UPDATE_ENV}=1 to record it")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(snapshot, sort_keys=True))
    differences = compare_snapshots(expected, actual)
    if differences:
        raise AssertionError("golden receipt drift:\n  " + "\n  ".join(differences))
