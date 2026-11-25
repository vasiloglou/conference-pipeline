"""Content-addressed receipt lineage.

A stage receipt records, in ``lineage``, the digest of every upstream receipt
and input artifact it consumed, and in ``output_digest`` a canonical digest of
what it produced. A receipt may be reused on ``--resume`` only while every
recorded input still carries the same digest; because each downstream receipt
binds its upstream's ``output_digest``, re-running any stage invalidates every
receipt below it without anyone editing receipts by hand.

Two digest kinds exist on purpose. ``output_digest`` is computed over a
stage's output artifacts, never over the receipt file: receipts carry
timestamps and ids, so a file digest would force a rebuild of every downstream
stage on each no-op re-run. Non-stage receipts (sub-step receipts such as the
topic-reconciliation or full-read extraction receipts) never carry an
``output_digest``; they bind by ``content_digest``, a canonical digest with
volatile keys removed. A *stage* receipt with no ``output_digest`` is never
silently content-hashed -- it is a blocker until it is re-run or backfilled.

This module deliberately has no module-level dependency on
``stage_contracts`` (which imports it); the registry helpers it needs are
imported lazily inside functions, the same way ``workflow_runs`` defers its
analysis-manifest import.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LINEAGE_SCHEMA_VERSION = 1
DIGEST_KIND_OUTPUT = "output_digest"
DIGEST_KIND_CONTENT = "content_digest"
LINEAGE_MODE_OUTPUT = "output_digest"
LINEAGE_MODE_CONTENT = "content"
LINEAGE_MODES = frozenset({LINEAGE_MODE_OUTPUT, LINEAGE_MODE_CONTENT})
SOURCE_REQUIRED_RECEIPTS = "required_receipts"
SOURCE_REQUIRED_UPSTREAM_STAGES = "required_upstream_stages"
SOURCE_RUNNER_UPSTREAM_STAGES = "runner_upstream_stages"
SOURCE_REQUIRED_SUBSTEPS = "required_substeps"
SOURCE_REGISTRY = "registry"
SOURCE_STAGE = "stage"
# A run's layout: the production runners write registry ``receipt_filename``
# receipts; the generic workflow writes ``stage-NN-<id>.json`` for its own
# stage table. ``required_upstream_stages`` entries may be scoped to layouts.
LAYOUT_PRODUCTION = "production"
LAYOUT_GENERIC = "generic"
LAYOUTS = frozenset({LAYOUT_PRODUCTION, LAYOUT_GENERIC})

# Keys that change on every write of a receipt without its outputs changing.
VOLATILE_RECEIPT_KEYS = frozenset(
    {
        "started_at",
        "ended_at",
        "generated_at",
        "elapsed_seconds",
        "receipt_id",
        "receipt_path",
        "stage_contract",
        "lineage",
        "output_digest",
        "output_artifacts",
        "supersedes_receipt_path",
        "superseded_prior_receipt_path",
        "usage",
        "provider_receipt_paths",
        "local_provider_receipt_paths",
        "cost_summary",
    }
)
# A directory output larger than this is digested by its listing only, so a
# stage whose output is thousands of per-artifact records still binds cheaply;
# the listing digest changes whenever a file is added, removed, or renamed.
DIRECTORY_DIGEST_FILE_LIMIT = 20000
_STAGE_RECEIPT_TYPE_SUFFIX = "production_stage_receipt"
_ARCHIVE_SUFFIX_SUPERSEDED = "superseded"
_ARCHIVE_SUFFIX_ATTEMPT = "attempt"
# Backfill's own copy of a receipt it stamps: the same execution, never a
# superseded one.
_ARCHIVE_SUFFIX_PREBACKFILL = "prebackfill"


# --------------------------------------------------------------------------
# Digest primitives
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_digest(payload: Any, *, excluded_keys: Iterable[str] = ()) -> str:
    excluded = set(excluded_keys)
    if isinstance(payload, Mapping):
        payload = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# Run-time stamps a generated JSON output rewrites on every run even when its
# content is unchanged. Approval/review times are not here: a new approval is
# a real change.
OUTPUT_VOLATILE_KEYS = frozenset(
    {
        "generated_at",
        "source_pack_generated_at",
        "started_at",
        "ended_at",
        "elapsed_seconds",
        "receipt_id",
        "bound_at",
        "last_run_at",
        "last_visit_at",
        "taken_at",
    }
)


def _strip_keys(payload: Any, keys: frozenset[str]) -> Any:
    if isinstance(payload, Mapping):
        return {str(key): _strip_keys(value, keys) for key, value in payload.items() if key not in keys}
    if isinstance(payload, list):
        return [_strip_keys(value, keys) for value in payload]
    return payload


def canonical_output_sha256(path: Path, *, excluded_keys: Iterable[str] = ()) -> str | None:
    """Digest of a JSON output without its run-time stamps (any depth).

    ``None`` when ``path`` is not a JSON file, so the caller falls back to
    the byte digest. A no-op regeneration that only rewrote ``generated_at``
    then keeps the same digest and does not cascade downstream.
    """

    if path.suffix != ".json" or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return canonical_json_digest(_strip_keys(payload, OUTPUT_VOLATILE_KEYS | frozenset(excluded_keys)))


def content_digest(receipt: Mapping[str, Any]) -> str:
    """Digest of a receipt with its volatile bookkeeping removed."""

    return canonical_json_digest(receipt, excluded_keys=VOLATILE_RECEIPT_KEYS)


def directory_sha256(path: Path, *, pattern: str | None = None) -> str:
    """Digest a directory by sorted (relative path, file sha256) pairs.

    Above ``DIRECTORY_DIGEST_FILE_LIMIT`` files only the sorted listing with
    sizes is digested; the receipt records which rule applied.
    """

    root = Path(path)
    files = sorted(
        candidate
        for candidate in (root.rglob(pattern) if pattern else root.rglob("*"))
        if candidate.is_file()
    )
    digest = hashlib.sha256()
    listing_only = len(files) > DIRECTORY_DIGEST_FILE_LIMIT
    digest.update(b"listing-only\n" if listing_only else b"content\n")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        if listing_only:
            digest.update(f"{relative}\t{candidate.stat().st_size}\n".encode("utf-8"))
        else:
            digest.update(f"{relative}\t{sha256_file(candidate)}\n".encode("utf-8"))
    return digest.hexdigest()


def workspace_relpath(workspace: str | Path, path: str | Path) -> str:
    """Workspace-relative posix path; the key every lineage entry uses."""

    workspace_root = Path(workspace).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    try:
        return candidate.resolve().relative_to(workspace_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(candidate.resolve(), workspace_root)).as_posix()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_path(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _load_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, Mapping) else None


# --------------------------------------------------------------------------
# Output artifacts and output digest
# --------------------------------------------------------------------------


def digest_output_artifacts(
    workspace_root: str | Path, artifacts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Normalise ``[{name, path, sha256?}]`` records, hashing files on disk.

    A record that already carries ``sha256``, ``manifest_sha256`` or
    ``content_sha256`` is trusted as-is (the producing stage measured it); any
    other record is hashed from disk, directories by :func:`directory_sha256`.
    A missing path is recorded with ``sha256: null`` and ``missing: true``.
    """

    workspace = Path(workspace_root).resolve()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        name = str(item.get("name") or item.get("kind") or Path(raw_path).name or f"artifact-{index}")
        record: dict[str, Any] = {"name": name, "path": workspace_relpath(workspace, candidate)}
        provided = None
        for key in ("sha256", "manifest_sha256", "content_sha256"):
            value = item.get(key)
            if isinstance(value, str) and value:
                provided = value
                record["sha256_source"] = key
                break
        if provided is not None:
            record["sha256"] = provided
        elif candidate.is_file():
            record["sha256"] = sha256_file(candidate)
            record["kind"] = "file"
        elif candidate.is_dir():
            record["sha256"] = directory_sha256(candidate)
            record["kind"] = "directory"
        else:
            record["sha256"] = None
            record["missing"] = True
        records.append(record)
    return records


def compute_output_digest(
    artifacts: Sequence[Mapping[str, Any]], *, components: Mapping[str, Any] | None = None
) -> str:
    """Canonical digest over sorted ``(name, path, sha256)`` plus components."""

    rows = sorted(
        (
            str(item.get("name") or ""),
            str(item.get("path") or ""),
            item.get("sha256"),
        )
        for item in artifacts
        if isinstance(item, Mapping)
    )
    payload = {"artifacts": rows, "components": dict(components or {})}
    return canonical_json_digest(payload)


def artifacts_from_checks(
    checks: Mapping[str, Any] | None, table: Sequence[Sequence[Any]]
) -> list[dict[str, Any]]:
    """Build artifact records from ``checks`` keys a stage already writes.

    ``table`` rows are ``[name, path_key, sha_key]`` (``sha_key`` may be
    ``null``); keys are dotted paths into ``checks``. Rows whose path key is
    absent are skipped, so a table may list every artifact a stage *can*
    produce.
    """

    checks = checks if isinstance(checks, Mapping) else {}
    records: list[dict[str, Any]] = []
    for row in table:
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) < 2:
            continue
        name = str(row[0])
        path_value = _get_path(checks, str(row[1]))
        if not isinstance(path_value, str) or not path_value:
            continue
        record: dict[str, Any] = {"name": name, "path": path_value}
        if len(row) > 2 and row[2]:
            sha_value = _get_path(checks, str(row[2]))
            if isinstance(sha_value, str) and sha_value:
                record["sha256"] = sha_value
        records.append(record)
    return records


# --------------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------------


def receipt_filenames_from_registry(registry: Mapping[str, Any]) -> dict[str, str]:
    """``{stage_id: receipt_filename}`` for bindings that declare one."""

    result: dict[str, str] = {}
    for stage_id, binding in (registry.get("stage_bindings") or {}).items():
        if not isinstance(binding, Mapping):
            continue
        filename = binding.get("receipt_filename")
        if isinstance(filename, str) and filename:
            result[str(stage_id)] = filename
    return result


def receipt_path_resolver_from_registry(
    workspace: str | Path,
    run_dir: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> Callable[[str], Path | None]:
    """Resolve a production stage id to its receipt path via ``receipt_filename``."""

    from conference_pipeline.stage_script_registry import load_stage_script_registry

    filenames = receipt_filenames_from_registry(load_stage_script_registry(registry_path))
    run_directory = Path(run_dir)

    def resolve(stage_id: str) -> Path | None:
        filename = filenames.get(stage_id)
        return (run_directory / filename) if filename else None

    return resolve


def generic_receipt_path_resolver(run_dir: str | Path) -> Callable[[str], Path | None]:
    """Resolve a generic workflow stage id to ``stage-NN-<id>.json``."""

    from conference_pipeline import workflow
    from conference_pipeline.workflow_runs import stage_receipt_path

    run_directory = Path(run_dir)

    def resolve(stage_id: str) -> Path | None:
        try:
            stage = workflow.get_workflow_stage(stage_id)
        except (KeyError, ValueError):
            return None
        return stage_receipt_path(run_directory, stage)

    return resolve


def is_stage_receipt(
    payload: Mapping[str, Any] | None,
    *,
    path: str | Path | None = None,
    stage_receipt_names: Iterable[str] = (),
) -> bool:
    """Whether a receipt is a stage receipt (binds by ``output_digest``)."""

    if path is not None and Path(path).name in set(stage_receipt_names):
        return True
    if not isinstance(payload, Mapping):
        return False
    receipt_type = payload.get("receipt_type")
    if isinstance(receipt_type, str) and receipt_type.endswith(_STAGE_RECEIPT_TYPE_SUFFIX):
        return True
    return isinstance(payload.get("stage_id"), str) and "stage_order" in payload


def upstream_binding_digest(
    payload: Mapping[str, Any] | None,
    *,
    path: str | Path | None = None,
    lineage_mode: str = LINEAGE_MODE_OUTPUT,
    stage_receipt_names: Iterable[str] = (),
    substep_receipt_names: Iterable[str] = (),
) -> tuple[str, str | None]:
    """``(digest_kind, digest)`` an upstream receipt binds by right now.

    A declared sub-step binds by its ``substep_outputs`` digest, the way a
    stage binds by ``output_digest``: its receipt also carries run
    bookkeeping (timings, reuse counts) that must not invalidate consumers.
    """

    if payload is None:
        return (DIGEST_KIND_CONTENT if lineage_mode == LINEAGE_MODE_CONTENT else DIGEST_KIND_OUTPUT, None)
    if lineage_mode == LINEAGE_MODE_CONTENT:
        return DIGEST_KIND_CONTENT, content_digest(payload)
    if path is not None and Path(path).name in set(substep_receipt_names):
        outputs = payload.get("substep_outputs")
        digest = outputs.get("digest") if isinstance(outputs, Mapping) else None
        return DIGEST_KIND_OUTPUT, digest if isinstance(digest, str) and digest else None
    if is_stage_receipt(payload, path=path, stage_receipt_names=stage_receipt_names):
        digest = payload.get("output_digest")
        return DIGEST_KIND_OUTPUT, digest if isinstance(digest, str) and digest else None
    return DIGEST_KIND_CONTENT, content_digest(payload)


def required_upstream_stage_ids(contract: Mapping[str, Any], *, layout: str = LAYOUT_PRODUCTION) -> list[str]:
    """``required_upstream_stages`` that apply to a run of ``layout``.

    An entry is a stage id (every layout) or ``{"stage": id, "layouts": [...]}``
    -- a prerequisite only one layout has, such as the production runners'
    ``corpus-inventory-index``, which the generic workflow never writes.
    """

    selected: list[str] = []
    for entry in contract.get("required_upstream_stages") or []:
        if isinstance(entry, str):
            selected.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("stage"), str):
            layouts = entry.get("layouts")
            if not layouts or layout in layouts:
                selected.append(str(entry["stage"]))
    return selected


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageInput:
    """One declared input of a stage: an upstream receipt or an input artifact."""

    name: str
    path: Path | None
    kind: str  # "receipt" | "artifact"
    source: str
    stage_id: str | None = None
    optional: bool = False
    lineage_mode: str = LINEAGE_MODE_OUTPUT
    spec_path: str | None = None


def resolve_lineage_inputs(
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    *,
    receipt_path_for_stage: Callable[[str], Path | None] | None = None,
    artifact_id: str | None = None,
    registry_path: str | Path | None = None,
    profile: str | None = None,
    upstream_stage_ids: Sequence[str] = (),
    layout: str = LAYOUT_PRODUCTION,
) -> tuple[list[LineageInput], dict[str, Any]]:
    """Declared inputs from the registry binding for ``stage_id``.

    Returns ``(inputs, binding_source)``; a missing workspace binding is
    recorded in ``binding_source["error"]`` rather than raised, so a runner
    can still write a receipt whose verification then blocks honestly.
    ``upstream_stage_ids`` are stages the *runner* knows precede this one
    (the generic workflow's earlier stages); they join the lineage digest so
    a changed upstream output invalidates resume, but only registry-declared
    ``required_upstream_stages`` are enforced by the verifier, and only those
    scoped to ``layout``.
    """

    from conference_pipeline.stage_contracts import (
        _resolve_workspace_template,
        resolve_stage_contract_binding,
    )
    from conference_pipeline.stage_script_registry import registry_sha256

    workspace_root = Path(workspace).resolve()
    binding_source: dict[str, Any] = {
        "registry_sha256": registry_sha256(registry_path),
        "registry_stage_id": stage_id,
        "profile": profile,
        "layout": layout,
    }
    try:
        binding = resolve_stage_contract_binding(
            workspace=workspace_root,
            stage_id=stage_id,
            registry_path=registry_path,
            profile=profile,
        )
    except ValueError as exc:
        binding_source["error"] = str(exc)
        return [], binding_source
    binding_source["registry_stage_id"] = binding.registry_stage_id
    contract = binding.runtime_contract or {}
    inputs: list[LineageInput] = []
    seen_paths: set[Path] = set()

    for spec in contract.get("required_receipts") or []:
        if not isinstance(spec, Mapping):
            continue
        spec_path = spec.get("path")
        path = _resolve_workspace_template(workspace_root, run_id, spec_path, artifact_id=artifact_id)
        if path is None:
            continue
        seen_paths.add(path)
        inputs.append(
            LineageInput(
                name=Path(str(spec_path)).stem,
                path=path,
                kind="receipt",
                source=SOURCE_REQUIRED_RECEIPTS,
                optional=bool(spec.get("optional")),
                lineage_mode=str(spec.get("lineage") or LINEAGE_MODE_OUTPUT),
                spec_path=str(spec_path),
            )
        )

    for upstream_stage_id in required_upstream_stage_ids(contract, layout=layout):
        path = receipt_path_for_stage(upstream_stage_id) if receipt_path_for_stage else None
        if path is not None:
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
        inputs.append(
            LineageInput(
                name=upstream_stage_id,
                path=path,
                kind="receipt",
                source=SOURCE_REQUIRED_UPSTREAM_STAGES,
                stage_id=upstream_stage_id,
            )
        )

    for upstream_stage_id in upstream_stage_ids:
        upstream_stage_id = str(upstream_stage_id)
        path = receipt_path_for_stage(upstream_stage_id) if receipt_path_for_stage else None
        if path is not None:
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
        inputs.append(
            LineageInput(
                name=upstream_stage_id,
                path=path,
                kind="receipt",
                source=SOURCE_RUNNER_UPSTREAM_STAGES,
                stage_id=upstream_stage_id,
                optional=True,
            )
        )

    lineage_spec = contract.get("lineage")
    lineage_spec = lineage_spec if isinstance(lineage_spec, Mapping) else {}
    for spec in lineage_spec.get("input_artifacts") or []:
        if not isinstance(spec, Mapping):
            continue
        spec_path = spec.get("path")
        path = _resolve_workspace_template(workspace_root, run_id, spec_path, artifact_id=artifact_id)
        if path is None:
            continue
        inputs.append(
            LineageInput(
                name=str(spec.get("name") or Path(str(spec_path)).stem),
                path=path,
                kind="artifact",
                source=SOURCE_REGISTRY,
                optional=bool(spec.get("optional")),
                spec_path=str(spec_path),
            )
        )
    return inputs, binding_source


def _artifact_sha256(path: Path, *, exclude_keys: Iterable[str] = ()) -> str | None:
    """File or directory digest; with ``exclude_keys``, a projected digest.

    A projection digests JSON files canonically without those keys (at any
    depth) and without run-time stamps, so an input a sub-step itself enriches
    (07D/07E writing their results back into the source packs), or one that
    only restamps receipt digests, is compared by what the sub-step read.
    """

    excluded = frozenset(str(key) for key in exclude_keys)
    if not excluded:
        if path.is_file():
            return sha256_file(path)
        if path.is_dir():
            return directory_sha256(path)
        return None

    def _file_digest(candidate: Path) -> str:
        return canonical_output_sha256(candidate, excluded_keys=excluded) or sha256_file(candidate)

    if path.is_file():
        return _file_digest(path)
    if path.is_dir():
        lines = [
            f"{candidate.relative_to(path).as_posix()}\t{_file_digest(candidate)}"
            for candidate in sorted(item for item in path.rglob("*") if item.is_file())
            if not any(part.startswith(".") for part in candidate.relative_to(path).parts)
        ]
        return hashlib.sha256(("projected\n" + "\n".join(lines)).encode("utf-8")).hexdigest()
    return None


def _self_attested_content_sha256(path: Path) -> str | None:
    """Recompute a JSON file's self-attested ``content_sha256``, if it has one.

    Some stage outputs (the analysis manifest) carry their own
    ``content_sha256`` and the stage records that value rather than a file
    digest. Recomputing it with the owner's rule (``generated_at`` dropped,
    ``content_sha256`` blanked) lets an out-of-band edit be detected without
    knowing which stage wrote the file.
    """

    if path.suffix != ".json" or not path.is_file():
        return None
    payload = _load_json_object(path)
    if payload is None or not isinstance(payload.get("content_sha256"), str):
        return None
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable["content_sha256"] = ""
    canonical = json.dumps(stable, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _current_output_digests(
    path: Path, *, exclude_keys: Iterable[str] = (), cache: dict[Any, frozenset[str]] | None = None
) -> frozenset[str]:
    """Every digest form a recorded output artifact may legitimately carry now.

    A recorded ``sha256`` is either measured from disk by
    :func:`digest_output_artifacts` (file or :func:`directory_sha256`, or a
    projection with ``exclude_keys``) or supplied by the stage, which uses a
    file digest, a canonical JSON digest, or a self-attested
    ``content_sha256``. An output still matches when its recorded digest is
    any of these. Empty when the path is absent.
    """

    key = (str(path), tuple(sorted(str(item) for item in exclude_keys)))
    if cache is not None and key in cache:
        return cache[key]
    forms: set[str] = set()
    projected = _artifact_sha256(path, exclude_keys=key[1])
    if projected is not None:
        forms.add(projected)
    if path.is_file():
        if key[1]:
            forms.add(sha256_file(path))
        for candidate in (
            canonical_output_sha256(path, excluded_keys=key[1]),
            _self_attested_content_sha256(path),
        ):
            if candidate:
                forms.add(candidate)
    elif path.is_dir() and key[1]:
        forms.add(directory_sha256(path))
    result = frozenset(forms)
    if cache is not None:
        cache[key] = result
    return result


def _newest_mtime(path: Path) -> float | None:
    try:
        if path.is_file():
            return path.stat().st_mtime
        if path.is_dir():
            return max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=None)
    except OSError:
        return None
    return None


def changed_output_artifacts(
    workspace: str | Path,
    receipt: Mapping[str, Any],
    *,
    receipt_mtime: float | None = None,
    cache: dict[Any, frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    """Recorded ``output_artifacts`` of a receipt whose content moved on disk.

    Re-digests every output the receipt recorded with a digest and returns
    ``[{name, path, recorded}]`` for each that no longer matches -- an
    out-of-band edit after the stage wrote its receipt.

    A digest :func:`digest_output_artifacts` measured from disk must match
    exactly. A digest the stage supplied itself (``sha256_source``) may use a
    scheme only that stage knows (a ``selection_sha256`` over part of a
    file); it matches when it equals any generic form (file, canonical JSON,
    directory, self-attested ``content_sha256``), and otherwise counts as
    changed only when the output was written after the receipt
    (``receipt_mtime``), so an unknown scheme is never mistaken for an edit.

    An output recorded as missing, or absent now (an on-demand Drive-backed
    artifact a consumer only knows by its receipt), is not reported: absence
    is the consuming stage's input gate's concern, not a content change.
    Receipts without ``output_artifacts`` yield nothing.
    """

    workspace_root = Path(workspace).resolve()
    records = receipt.get("output_artifacts") if isinstance(receipt, Mapping) else None
    changed: list[dict[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        rel_path = record.get("path")
        recorded = record.get("sha256")
        if not isinstance(rel_path, str) or not rel_path or not isinstance(recorded, str) or not recorded:
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = workspace_root / path
        if not path.exists():
            continue
        forms = _current_output_digests(path, exclude_keys=record.get("exclude_keys") or (), cache=cache)
        if recorded in forms:
            continue
        if record.get("sha256_source"):
            newest = _newest_mtime(path)
            if receipt_mtime is None or newest is None or newest <= receipt_mtime:
                continue
        changed.append({"name": record.get("name"), "path": rel_path, "recorded": recorded})
    return changed


def bind_lineage(
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    *,
    receipt_path_for_stage: Callable[[str], Path | None] | None = None,
    artifact_id: str | None = None,
    extra_input_artifacts: Sequence[Mapping[str, Any]] = (),
    code_identity: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    profile: str | None = None,
    upstream_stage_ids: Sequence[str] = (),
    layout: str = LAYOUT_PRODUCTION,
) -> dict[str, Any]:
    """Record the current digests of everything ``stage_id`` consumes.

    Never raises for a missing upstream: the entry is recorded with
    ``missing: true`` and ``digest: null`` and the verifier blocks on it.
    ``extra_input_artifacts`` are ``[{name, path, sha256?}]`` records the
    runner declares beyond the registry (for example the ledgers a stage read
    from a sibling run).
    """

    from conference_pipeline.stage_script_registry import (
        load_stage_script_registry,
        registry_sha256,
    )

    workspace_root = Path(workspace).resolve()
    registry = load_stage_script_registry(registry_path)
    stage_receipt_names = set(receipt_filenames_from_registry(registry).values())
    substep_names = substep_receipt_names(registry)
    inputs, binding_source = resolve_lineage_inputs(
        workspace_root,
        run_id,
        stage_id,
        receipt_path_for_stage=receipt_path_for_stage,
        artifact_id=artifact_id,
        registry_path=registry_path,
        profile=profile,
        upstream_stage_ids=upstream_stage_ids,
        layout=layout,
    )

    upstream_receipts: dict[str, dict[str, Any]] = {}
    input_artifacts: dict[str, dict[str, Any]] = {}
    for item in inputs:
        if item.kind == "receipt":
            key, entry = _upstream_receipt_entry(
                workspace_root, item, stage_receipt_names=stage_receipt_names, substep_receipt_names=substep_names
            )
            upstream_receipts[key] = entry
        else:
            input_artifacts[item.name] = _input_artifact_record(
                workspace_root, item.path, source=item.source, optional=item.optional
            )

    for extra in extra_input_artifacts:
        if not isinstance(extra, Mapping):
            continue
        raw_path = extra.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        name = str(extra.get("name") or Path(raw_path).name)
        # Input artifacts are always hashed from disk: the verifier compares
        # against the file, and a stage's own recorded digest for an input is
        # often a canonical (payload) digest, not a file digest.
        record = _input_artifact_record(
            workspace_root,
            candidate,
            source=str(extra.get("source") or SOURCE_STAGE),
            optional=bool(extra.get("optional")),
        )
        declared = extra.get("sha256")
        if isinstance(declared, str) and declared:
            record["declared_sha256"] = declared
        input_artifacts[name] = record

    identity: dict[str, Any] = {"script_registry_sha256": registry_sha256(registry_path)}
    if code_identity:
        identity.update({str(key): value for key, value in code_identity.items()})

    lineage: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "bound_at": _utc_now_iso(),
        "binding_source": binding_source,
        "upstream_receipts": dict(sorted(upstream_receipts.items())),
        "input_artifacts": dict(sorted(input_artifacts.items())),
        "code_identity": identity,
        "output_digest_components": {},
        "backfill": None,
    }
    lineage["lineage_digest"] = lineage_digest(lineage)
    return lineage


def _upstream_receipt_entry(
    workspace_root: Path,
    item: LineageInput,
    *,
    stage_receipt_names: Iterable[str],
    substep_receipt_names: Iterable[str] = (),
) -> tuple[str, dict[str, Any]]:
    """``(key, entry)`` binding one upstream receipt input as it is on disk now."""

    key = workspace_relpath(workspace_root, item.path) if item.path else f"stage:{item.name}"
    entry: dict[str, Any] = {
        "stage_id": item.stage_id,
        "spec_path": item.spec_path,
        "source": item.source,
        "digest_kind": DIGEST_KIND_CONTENT if item.lineage_mode == LINEAGE_MODE_CONTENT else DIGEST_KIND_OUTPUT,
        "digest": None,
        "status": None,
        "optional": item.optional,
        "missing": True,
    }
    payload = _load_json_object(item.path) if item.path and item.path.is_file() else None
    if payload is not None:
        kind, digest = upstream_binding_digest(
            payload,
            path=item.path,
            lineage_mode=item.lineage_mode,
            stage_receipt_names=stage_receipt_names,
            substep_receipt_names=substep_receipt_names,
        )
        entry.update({"digest_kind": kind, "digest": digest, "status": payload.get("status"), "missing": False})
        if entry["stage_id"] is None and isinstance(payload.get("stage_id"), str):
            entry["stage_id"] = payload["stage_id"]
    return key, entry


def upstream_freshness_blockers(
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    *,
    receipt_path_for_stage: Callable[[str], Path | None] | None = None,
    artifact_id: str | None = None,
    registry_path: str | Path | None = None,
    profile: str | None = None,
    layout: str = LAYOUT_PRODUCTION,
) -> list[dict[str, Any]]:
    """Pre-spend gate: why a fresh run of ``stage_id`` would fail lineage verification.

    A fresh run binds its direct upstreams as they are now, so those always
    match. What verification still catches -- only after the stage has done
    its work, paid calls included -- is an upstream that is itself stale (its
    own recorded lineage moved), a stage upstream with no ``output_digest``,
    a missing required upstream, or an unresolvable registry binding. A
    runner calls this before it constructs any paid client and blocks on a
    non-empty result, so re-running a stage above a stale one costs nothing.
    Status checks stay with the runner's own input gates; input artifacts
    are bound fresh and cannot be stale here.
    """

    from conference_pipeline.stage_script_registry import load_stage_script_registry

    workspace_root = Path(workspace).resolve()
    inputs, binding_source = resolve_lineage_inputs(
        workspace_root,
        run_id,
        stage_id,
        receipt_path_for_stage=receipt_path_for_stage,
        artifact_id=artifact_id,
        registry_path=registry_path,
        profile=profile,
        layout=layout,
    )
    if binding_source.get("error"):
        return [
            _blocker(
                "stage_contract_binding_unresolved",
                stage_id=stage_id,
                expected="a registry binding for this stage and workspace",
                observed=binding_source["error"],
            )
        ]
    registry = load_stage_script_registry(registry_path)
    filenames = receipt_filenames_from_registry(registry)
    stage_for_filename = {filename: stage for stage, filename in filenames.items()}
    upstream: dict[str, dict[str, Any]] = {}
    for item in inputs:
        if item.kind != "receipt":
            continue
        key, entry = _upstream_receipt_entry(
            workspace_root,
            item,
            stage_receipt_names=set(filenames.values()),
            substep_receipt_names=substep_receipt_names(registry),
        )
        if entry["stage_id"] is None and item.path is not None:
            # Report-only: name the stage a registry-path upstream belongs to.
            entry["stage_id"] = stage_for_filename.get(item.path.name)
        upstream[key] = entry
    return verify_lineage(
        workspace_root,
        run_id,
        {"upstream_receipts": upstream},
        stage_id=stage_id,
        transitive=True,
        registry_path=registry_path,
    )


def summarize_lineage_blockers(blockers: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact, receipt-safe view of lineage blockers (nested chains dropped)."""

    keys = ("code", "stage_id", "upstream_stage_id", "receipt", "chain", "input", "paths", "expected", "observed")
    return [{key: blocker[key] for key in keys if blocker.get(key) is not None} for blocker in blockers]


# --------------------------------------------------------------------------
# Sub-step receipts
# --------------------------------------------------------------------------
#
# Some stages consume receipts written by commands that are not registry
# stages -- the LLM full read and the 07A-07H analysis sub-steps. Bound by
# content alone they would be leaves: a rebuilt corpus index could never make
# them stale. The registry's ``lineage_substeps`` declares each one's receipt
# and what it consumes; its writer stamps ``lineage`` when it writes, and the
# verifier follows that lineage like a stage's. A declared sub-step receipt
# without lineage is stale.


def substep_specs(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The registry's declared sub-steps, ``{substep_id: spec}``."""

    specs = registry.get("lineage_substeps")
    if not isinstance(specs, Mapping):
        return {}
    return {str(key): value for key, value in specs.items() if isinstance(value, Mapping)}


def _substep_receipt_path(workspace_root: Path, run_id: str, spec: Mapping[str, Any]) -> Path | None:
    template = spec.get("receipt")
    if not isinstance(template, str) or not template:
        return None
    return (workspace_root / template.replace("<run-id>", run_id)).resolve()


def substep_receipt_names(registry: Mapping[str, Any]) -> set[str]:
    """Basenames of every declared sub-step receipt."""

    return {
        Path(str(spec.get("receipt"))).name
        for spec in substep_specs(registry).values()
        if isinstance(spec.get("receipt"), str) and spec.get("receipt")
    }


def substep_receipt_index(workspace: str | Path, run_id: str, registry: Mapping[str, Any]) -> dict[str, str]:
    """``{workspace-relative receipt path: substep_id}`` for one run."""

    workspace_root = Path(workspace).resolve()
    index: dict[str, str] = {}
    for substep_id, spec in substep_specs(registry).items():
        path = _substep_receipt_path(workspace_root, run_id, spec)
        if path is not None:
            index[workspace_relpath(workspace_root, path)] = substep_id
    return index


def bind_substep_lineage(
    workspace: str | Path,
    run_id: str,
    substep_id: str,
    *,
    registry_path: str | Path | None = None,
    code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lineage for a sub-step receipt: the current digests of what it consumed.

    Stage upstreams bind by ``output_digest``; sub-step upstreams by content
    (their own ``lineage`` is excluded from that digest). Raises
    ``KeyError`` for a sub-step the registry does not declare.
    """

    from conference_pipeline.stage_script_registry import (
        load_stage_script_registry,
        registry_sha256,
    )

    workspace_root = Path(workspace).resolve()
    registry = load_stage_script_registry(registry_path)
    specs = substep_specs(registry)
    spec = specs[substep_id]
    filenames = receipt_filenames_from_registry(registry)
    run_directory = workspace_root / "storage" / "workflow-runs" / run_id
    inputs: list[LineageInput] = []
    for stage_id in spec.get("required_upstream_stages") or []:
        filename = filenames.get(str(stage_id))
        inputs.append(
            LineageInput(
                name=str(stage_id),
                path=(run_directory / filename).resolve() if filename else None,
                kind="receipt",
                source=SOURCE_REQUIRED_UPSTREAM_STAGES,
                stage_id=str(stage_id),
            )
        )
    for upstream_substep in spec.get("required_substeps") or []:
        upstream_spec = specs.get(str(upstream_substep)) or {}
        inputs.append(
            LineageInput(
                name=str(upstream_substep),
                path=_substep_receipt_path(workspace_root, run_id, upstream_spec),
                kind="receipt",
                source=SOURCE_REQUIRED_SUBSTEPS,
                stage_id=str(upstream_substep),
            )
        )
    stage_receipt_names = set(filenames.values())
    substep_names = substep_receipt_names(registry)
    upstream_receipts = dict(
        _upstream_receipt_entry(
            workspace_root, item, stage_receipt_names=stage_receipt_names, substep_receipt_names=substep_names
        )
        for item in inputs
    )
    input_artifacts: dict[str, dict[str, Any]] = {}
    for spec_artifact in spec.get("input_artifacts") or []:
        if not isinstance(spec_artifact, Mapping) or not isinstance(spec_artifact.get("path"), str):
            continue
        rel = str(spec_artifact["path"]).replace("<run-id>", run_id)
        exclude_keys = sorted(str(key) for key in spec_artifact.get("exclude_keys") or [])
        sha = _artifact_sha256(workspace_root / rel, exclude_keys=exclude_keys)
        input_artifacts[str(spec_artifact.get("name") or Path(rel).name)] = {
            "path": rel,
            "sha256": sha,
            "source": SOURCE_REGISTRY,
            "optional": bool(spec_artifact.get("optional")),
            "missing": sha is None,
            "exclude_keys": exclude_keys,
        }
    identity: dict[str, Any] = {"script_registry_sha256": registry_sha256(registry_path)}
    if code_identity:
        identity.update({str(key): value for key, value in code_identity.items()})
    lineage: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "bound_at": _utc_now_iso(),
        "binding_source": {"registry_sha256": registry_sha256(registry_path), "substep_id": substep_id},
        "upstream_receipts": dict(sorted(upstream_receipts.items())),
        "input_artifacts": dict(sorted(input_artifacts.items())),
        "code_identity": identity,
        "output_digest_components": {},
        "backfill": None,
    }
    lineage["lineage_digest"] = lineage_digest(lineage)
    return lineage


def current_stage_output_digest(
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    *,
    registry_path: str | Path | None = None,
) -> str | None:
    """The ``output_digest`` a stage's receipt carries now (``None`` if absent).

    Writers with a resume cache fold this into their per-item input digest,
    so a cached item computed against an older upstream is recomputed.
    """

    from conference_pipeline.stage_script_registry import load_stage_script_registry

    filename = receipt_filenames_from_registry(load_stage_script_registry(registry_path)).get(stage_id)
    if not filename:
        return None
    payload = _load_json_object(Path(workspace).resolve() / "storage" / "workflow-runs" / run_id / filename)
    digest = payload.get("output_digest") if payload else None
    return digest if isinstance(digest, str) and digest else None


# Execution bookkeeping a sub-step writes into its outputs: whether an item
# was reused on resume, where its provider receipts went, how long a call
# took. None of it is content, so an unchanged resume keeps the same digest.
SUBSTEP_OUTPUT_EXECUTION_KEYS = frozenset(
    {
        "reused",
        "resumed",
        "receipt_path",
        "provider_receipt_paths",
        "local_provider_receipt_paths",
        "provider_metadata",
        "usage",
        # Derived from Markdown that is itself digested: a repair rewrites the
        # Markdown in place and a later resume re-derives these, so they must
        # not move the digest on their own.
        "markdown_sha256",
        "output_token_estimate",
        "source_pack_topic_reconciliation_sha256",
    }
)


def substep_output_digest(workspace: str | Path, run_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Digest of a sub-step's declared outputs, recorded on its receipt.

    Downstream steps bind a sub-step by the content of its receipt; without
    this, a re-run that wrote different outputs but the same counts would look
    unchanged. JSON files are digested canonically (run-time stamps and
    ``SUBSTEP_OUTPUT_EXECUTION_KEYS`` removed), so an unchanged resume does not
    cascade; hidden temp files are skipped.
    """

    workspace_root = Path(workspace).resolve()
    artifacts: list[dict[str, Any]] = []
    for template in spec.get("outputs") or []:
        rel = str(template).replace("<run-id>", run_id)
        path = workspace_root / rel
        entry: dict[str, Any] = {"path": rel, "sha256": None, "file_count": 0}
        if path.is_file():
            digest = canonical_output_sha256(path, excluded_keys=SUBSTEP_OUTPUT_EXECUTION_KEYS)
            entry.update({"sha256": digest or sha256_file(path), "file_count": 1})
        elif path.is_dir():
            lines: list[str] = []
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                relative = child.relative_to(path).as_posix()
                if any(part.startswith(".") for part in relative.split("/")):
                    continue
                digest = canonical_output_sha256(child, excluded_keys=SUBSTEP_OUTPUT_EXECUTION_KEYS)
                lines.append(f"{relative}\t{digest or sha256_file(child)}")
            entry.update(
                {
                    "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
                    "file_count": len(lines),
                }
            )
        artifacts.append(entry)
    return {"artifacts": artifacts, "digest": canonical_json_digest(artifacts)}


def unbind_substep_receipt(
    workspace: str | Path,
    run_id: str,
    substep_id: str,
    *,
    registry_path: str | Path | None = None,
) -> bool:
    """Mark a sub-step's existing receipt unbound before a re-run rewrites outputs.

    The prior receipt's ``substep_outputs`` digest is a snapshot; a re-run
    that crashes after rewriting some outputs would otherwise leave that
    receipt certifying files it no longer describes. Writers call this before
    their first output write; the new receipt they persist on completion
    replaces it. Returns whether a receipt was changed; never raises.
    """

    try:
        from conference_pipeline.stage_script_registry import load_stage_script_registry

        spec = substep_specs(load_stage_script_registry(registry_path))[substep_id]
        path = _substep_receipt_path(Path(workspace).resolve(), run_id, spec)
        payload = _load_json_object(path) if path is not None and path.is_file() else None
        if payload is None or path is None or ("lineage" not in payload and "substep_outputs" not in payload):
            return False
        unbound = {key: value for key, value in payload.items() if key not in {"lineage", "substep_outputs"}}
        unbound["lineage_unbound_reason"] = "in_progress"
        _write_json_atomic(path, unbound)
        return True
    except Exception:  # noqa: BLE001 - bookkeeping must not stop the run
        return False


def refresh_substep_outputs(
    workspace: str | Path,
    run_id: str,
    substep_id: str,
    *,
    registry_path: str | Path | None = None,
) -> bool:
    """Re-digest a sub-step's outputs after a sanctioned in-place edit.

    Reference repairs (07F/07G) rewrite 07D's Markdown after 07D stamped it;
    without this the next 07E would bind the pre-repair snapshot. Only
    ``substep_outputs`` is recomputed: the recorded ``lineage`` is kept, so a
    repair never re-binds (heals) a 07D whose own upstream moved. Returns
    whether a receipt was refreshed; never raises.
    """

    try:
        from conference_pipeline.stage_script_registry import load_stage_script_registry

        spec = substep_specs(load_stage_script_registry(registry_path))[substep_id]
        path = _substep_receipt_path(Path(workspace).resolve(), run_id, spec)
        payload = _load_json_object(path) if path is not None and path.is_file() else None
        if payload is None or path is None:
            return False
        refreshed = dict(payload)
        refreshed["substep_outputs"] = substep_output_digest(workspace, run_id, spec)
        _write_json_atomic(path, refreshed)
        return True
    except Exception:  # noqa: BLE001 - a repair must not fail on bookkeeping
        return False


def archive_obsolete_outputs(workspace: str | Path, paths: Iterable[Path], *, label: str) -> list[str]:
    """Move outputs a complete run no longer produces out of the output tree.

    A sub-step's declared outputs are digested as a whole and read by globs
    downstream, so a stale file left behind (a deselected deep dive, a summary
    whose category became empty) would be certified current by the next
    stamp. Files and directories move (never deleted) to
    ``storage/lineage-pruned/<label>/<stamp>/<original relative path>``.
    Returns the moved workspace-relative paths.
    """

    workspace_root = Path(workspace).resolve()
    moved: list[str] = []
    stamp = _utc_now_iso().replace(":", "").replace("-", "")
    archive = workspace_root / "storage" / "lineage-pruned" / label / stamp
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.exists():
            continue
        relative = resolved.relative_to(workspace_root)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved.replace(destination)
        moved.append(relative.as_posix())
    return moved


#: Receipt blocks this module owns: digests, registry-derived relative paths
#: and the field *names* a projection skips.  They are machine-verified byte
#: for byte and never carry a credential, so they must reach disk unmodified.
LINEAGE_RECEIPT_KEYS = frozenset(
    {"lineage", "substep_outputs", "lineage_unbound_reason", "lineage_error"}
)


def sanitize_receipt_for_write(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Scrub a receipt for persistence, leaving its lineage blocks intact.

    The receipt sanitizer redacts by *key name*, so a lineage field whose name
    merely resembles a credential is destroyed along with real secrets: the
    2026-09-20 CVPR full read recorded ``exclude_keys`` (the field names a
    projected digest skips) as ``["REDACTED", "REDACTED"]``, which made
    verification project a different digest and report a stale input for an
    11-hour run whose inputs had not changed.  Lineage is built here out of
    digests and workspace-relative paths -- it has no path by which a
    credential could enter it -- so it is written through unscrubbed, and only
    the writer's own payload is sanitized.
    """

    from conference_pipeline.iclr2025_live import sanitize_for_receipt

    payload = {key: value for key, value in receipt.items() if key not in LINEAGE_RECEIPT_KEYS}
    sanitized = sanitize_for_receipt(payload)
    if not isinstance(sanitized, dict):  # pragma: no cover - sanitizer contract
        raise TypeError("sanitize_for_receipt must return a mapping for a receipt")
    for key in LINEAGE_RECEIPT_KEYS:
        if key in receipt:
            sanitized[key] = receipt[key]
    return sanitized


def stamp_substep_receipt(
    receipt: dict[str, Any],
    *,
    workspace: str | Path,
    run_id: str,
    substep_id: str,
    complete: bool = True,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach sub-step ``lineage`` and ``substep_outputs`` to a receipt its
    writer is about to persist.

    Only a succeeded, non-dry-run run that covered the sub-step's whole scope
    is bound (``complete=False`` for a topic/lane/category/question subset): a
    dry run (``mode`` dry-run, even when it reports succeeded), a failed
    attempt or a partial re-run must not bind directory-wide outputs to
    current upstreams it never checked, so such a receipt is left unbound
    (``lineage_unbound_reason``) and verifies as stale.

    Never raises: a writer calls this after its (possibly paid) work, and a
    lost receipt would cost more than a stale one. On failure the receipt
    carries ``lineage_error`` and no lineage, so it verifies as stale.
    """

    status = str(receipt.get("status") or "")
    mode = str(receipt.get("mode") or "")
    unbound_reason = None
    if status != "succeeded":
        unbound_reason = f"status:{status}"
    elif mode in {"dry-run", "dry_run", "mock", "plan"}:
        # A planning run can report succeeded without producing its outputs.
        unbound_reason = f"mode:{mode}"
    elif not complete:
        unbound_reason = "partial_scope"
    if unbound_reason is not None:
        for key in ("lineage", "substep_outputs", "lineage_error"):
            receipt.pop(key, None)
        receipt["lineage_unbound_reason"] = unbound_reason
        return receipt
    receipt.pop("lineage_unbound_reason", None)
    try:
        from conference_pipeline.stage_script_registry import load_stage_script_registry

        spec = substep_specs(load_stage_script_registry(registry_path))[substep_id]
        # Not a volatile key: it is part of the content downstream steps bind.
        receipt["substep_outputs"] = substep_output_digest(workspace, run_id, spec)
        receipt["lineage"] = bind_substep_lineage(workspace, run_id, substep_id, registry_path=registry_path)
        receipt.pop("lineage_error", None)
    except Exception as exc:  # noqa: BLE001 - receipt boundary
        receipt.pop("lineage", None)
        receipt["lineage_error"] = f"{type(exc).__name__}: {exc}"
    return receipt


def _input_artifact_record(
    workspace_root: Path,
    path: Path | None,
    *,
    source: str,
    optional: bool,
    provided_sha256: str | None = None,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "sha256": None, "source": source, "optional": optional, "missing": True}
    sha = provided_sha256 or _artifact_sha256(path)
    return {
        "path": workspace_relpath(workspace_root, path),
        "sha256": sha,
        "source": source,
        "optional": optional,
        "missing": sha is None,
    }


def _upstream_identity(key: str, entry: Mapping[str, Any]) -> str:
    """Run-independent identity of an upstream entry.

    Resolved paths embed the run id, so the digest keys entries by the
    registry template (``<run-id>`` unexpanded) or the upstream stage id;
    otherwise two runs binding the same missing upstream would never share a
    compatibility key.
    """

    spec_path = entry.get("spec_path")
    if isinstance(spec_path, str) and spec_path:
        return spec_path
    stage_id = entry.get("stage_id")
    if isinstance(stage_id, str) and stage_id:
        return f"stage:{stage_id}"
    return str(key)


def lineage_digest(lineage: Mapping[str, Any]) -> str:
    """Digest over recorded upstream digests and input shas (not code identity)."""

    upstream = lineage.get("upstream_receipts")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    inputs = lineage.get("input_artifacts")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    payload = {
        "upstream": {
            _upstream_identity(str(key), entry if isinstance(entry, Mapping) else {}): (
                entry.get("digest") if isinstance(entry, Mapping) else None
            )
            for key, entry in upstream.items()
        },
        "inputs": {
            str(key): (entry.get("sha256") if isinstance(entry, Mapping) else None)
            for key, entry in inputs.items()
        },
    }
    return canonical_json_digest(payload)


def apply_lineage(
    receipt: dict[str, Any],
    *,
    lineage: Mapping[str, Any],
    output_artifacts: Sequence[Mapping[str, Any]],
    components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Set ``output_artifacts``, ``output_digest`` and ``lineage`` on a receipt."""

    lineage_copy = dict(lineage)
    if components:
        lineage_copy["output_digest_components"] = dict(components)
    receipt["output_artifacts"] = [dict(item) for item in output_artifacts if isinstance(item, Mapping)]
    receipt["output_digest"] = compute_output_digest(
        receipt["output_artifacts"], components=lineage_copy.get("output_digest_components") or None
    )
    receipt["lineage"] = lineage_copy
    return receipt


def artifact_tables_for_stage(
    workspace: str | Path,
    stage_id: str,
    *,
    profile: str | None = None,
    registry_path: str | Path | None = None,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """``(output_artifact_checks, input_artifact_checks)`` from the registry.

    Rows are ``[name, receipt_key, sha_key?]`` with dotted keys from the
    receipt root (``checks.index_digest_path``); a row whose key is absent on
    a given runner's receipt is skipped, so one table carries every runner's
    spelling. A missing workspace binding yields empty tables (the verifier
    already blocks on it).
    """

    from conference_pipeline.stage_contracts import resolve_stage_contract_binding

    try:
        binding = resolve_stage_contract_binding(
            workspace=workspace, stage_id=stage_id, registry_path=registry_path, profile=profile
        )
    except ValueError:
        return [], []
    contract = binding.runtime_contract or {}
    outputs = contract.get("output_artifact_checks")
    inputs = contract.get("input_artifact_checks")
    return (
        [list(row) for row in outputs] if isinstance(outputs, list) else [],
        [list(row) for row in inputs] if isinstance(inputs, list) else [],
    )


def finalize_receipt(
    receipt: dict[str, Any],
    *,
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    run_dir: str | Path,
    lineage: Mapping[str, Any] | None = None,
    bind: bool = True,
    profile: str | None = None,
    artifact_id: str | None = None,
    code_identity: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach ``output_artifacts``, ``output_digest`` and ``lineage`` to a runner receipt.

    Output artifacts come from the registry's output table applied to the
    receipt; input artifacts beyond the registry's required receipts come
    from its input table. ``lineage`` is reused when given (a resume path
    handing back the receipt's own recorded lineage); with ``bind=False`` and
    no lineage the receipt is left without one and the verifier blocks -- a
    resume path must never mint fresh lineage for a receipt it did not
    produce (the heal hazard).
    """

    workspace_root = Path(workspace).resolve()
    outputs_table, inputs_table = artifact_tables_for_stage(
        workspace_root, stage_id, profile=profile, registry_path=registry_path
    )
    output_artifacts = digest_output_artifacts(workspace_root, artifacts_from_checks(receipt, outputs_table))
    checks = receipt.get("checks") if isinstance(receipt.get("checks"), Mapping) else {}
    components: dict[str, Any] | None = None
    deck_digest = checks.get("step_lineage_output_digest")
    if isinstance(deck_digest, str) and deck_digest:
        components = {"decks": deck_digest}
    if lineage is None:
        existing = receipt.get("lineage")
        if isinstance(existing, Mapping):
            lineage = existing
        elif bind:
            lineage = bind_lineage(
                workspace_root,
                run_id,
                stage_id,
                receipt_path_for_stage=receipt_path_resolver_from_registry(
                    workspace_root, run_dir, registry_path=registry_path
                ),
                artifact_id=artifact_id,
                extra_input_artifacts=artifacts_from_checks(receipt, inputs_table),
                code_identity=code_identity,
                registry_path=registry_path,
                profile=profile,
            )
    if lineage is None:
        receipt["output_artifacts"] = output_artifacts
        receipt["output_digest"] = compute_output_digest(output_artifacts, components=components)
        return receipt
    return apply_lineage(receipt, lineage=lineage, output_artifacts=output_artifacts, components=components)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _blocker(
    code: str, *, stage_id: str, expected: Any, observed: Any, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "stage_id": stage_id,
        "expected": expected,
        "observed": observed,
        "retryable": True,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def verify_lineage(
    workspace: str | Path,
    run_id: str,
    lineage: Mapping[str, Any] | None,
    *,
    stage_id: str,
    only_path: str | None = None,
    transitive: bool = True,
    registry_path: str | Path | None = None,
    _visited: set[str] | None = None,
    _chain: tuple[str, ...] = (),
    _output_cache: dict[Any, frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    """Blockers for every recorded upstream receipt whose digest moved.

    With ``only_path`` only that upstream entry is checked (the caller walks
    the registry's required receipts one by one); without it every upstream
    entry and every input artifact is checked. ``transitive`` follows each
    upstream *stage* receipt's own lineage.

    An upstream stage receipt whose ``output_digest`` still matches is also
    checked against disk: its recorded ``output_artifacts`` are re-digested
    and any that moved (edited out of band after the stage wrote its
    receipt) yield ``stage_contract_upstream_output_changed`` naming the
    stage and paths. The receipt alone cannot show that edit, so without
    this a paid stage would run on outputs no receipt vouches for.
    """

    if not isinstance(lineage, Mapping):
        return [
            _blocker(
                "stage_contract_lineage_missing",
                stage_id=stage_id,
                expected="receipt.lineage object",
                observed="<missing>",
            )
        ]
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    workspace_root = Path(workspace).resolve()
    registry = load_stage_script_registry(registry_path)
    stage_receipt_names = set(receipt_filenames_from_registry(registry).values())
    substep_names = substep_receipt_names(registry)
    substep_ids = substep_receipt_index(workspace_root, run_id, registry)
    visited = _visited if _visited is not None else set()
    output_cache = _output_cache if _output_cache is not None else {}
    blockers: list[dict[str, Any]] = []
    upstream = lineage.get("upstream_receipts")
    upstream = upstream if isinstance(upstream, Mapping) else {}

    for rel_path, entry in upstream.items():
        rel_path = str(rel_path)
        if only_path is not None and rel_path != only_path:
            continue
        entry = entry if isinstance(entry, Mapping) else {}
        recorded_kind = str(entry.get("digest_kind") or DIGEST_KIND_OUTPUT)
        recorded = entry.get("digest")
        substep_id = substep_ids.get(rel_path)
        upstream_stage = entry.get("stage_id") or substep_id
        path = workspace_root / rel_path
        payload = _load_json_object(path) if path.is_file() else None
        if payload is None:
            if bool(entry.get("optional")) and bool(entry.get("missing")) and recorded is None:
                continue
            blockers.append(
                _blocker(
                    "stage_contract_upstream_changed",
                    stage_id=stage_id,
                    expected=recorded,
                    observed="<missing>",
                    extra={
                        "upstream_stage_id": upstream_stage,
                        "receipt": rel_path,
                        "digest_kind": recorded_kind,
                        "recorded": recorded,
                        "observed": "<missing>",
                    },
                )
            )
            continue
        if substep_id is not None and payload.get("status") != "succeeded":
            # A dry run or failed attempt of a sub-step never counts as current.
            blockers.append(
                _blocker(
                    "stage_contract_upstream_not_succeeded",
                    stage_id=stage_id,
                    expected="succeeded",
                    observed=payload.get("status"),
                    extra={"upstream_stage_id": upstream_stage, "receipt": rel_path},
                )
            )
            continue
        lineage_mode = LINEAGE_MODE_CONTENT if recorded_kind == DIGEST_KIND_CONTENT else LINEAGE_MODE_OUTPUT
        kind, current = upstream_binding_digest(
            payload,
            path=path,
            lineage_mode=lineage_mode,
            stage_receipt_names=stage_receipt_names,
            substep_receipt_names=substep_names,
        )
        if kind == DIGEST_KIND_OUTPUT and current is None:
            blockers.append(
                _blocker(
                    "stage_contract_upstream_output_digest_missing",
                    stage_id=stage_id,
                    expected=f"{rel_path} carries output_digest",
                    observed="<missing>",
                    extra={"upstream_stage_id": upstream_stage, "receipt": rel_path},
                )
            )
            continue
        if recorded is None or current != recorded:
            blockers.append(
                _blocker(
                    "stage_contract_upstream_changed",
                    stage_id=stage_id,
                    expected=recorded,
                    observed=current,
                    extra={
                        "upstream_stage_id": upstream_stage,
                        "receipt": rel_path,
                        "digest_kind": kind,
                        "recorded": recorded,
                    },
                )
            )
            continue
        # Stage receipts and declared sub-step receipts carry lineage to follow;
        # other content-bound receipts (pinned historical inputs) are leaves.
        if transitive and (kind == DIGEST_KIND_OUTPUT or substep_id is not None):
            if rel_path in visited:
                continue
            visited.add(rel_path)
            if kind == DIGEST_KIND_OUTPUT and substep_id is None:
                changed = changed_output_artifacts(
                    workspace_root, payload, receipt_mtime=path.stat().st_mtime, cache=output_cache
                )
                if changed:
                    blockers.append(
                        _blocker(
                            "stage_contract_upstream_output_changed",
                            stage_id=stage_id,
                            expected=f"{rel_path} recorded outputs match disk",
                            observed=[item["path"] for item in changed],
                            extra={
                                "upstream_stage_id": upstream_stage,
                                "receipt": rel_path,
                                "paths": [item["path"] for item in changed],
                                "changed_outputs": changed,
                            },
                        )
                    )
            chain = (*_chain, rel_path)
            sub_lineage = payload.get("lineage")
            sub_stage_id = str(payload.get("stage_id") or upstream_stage or rel_path)
            if not isinstance(sub_lineage, Mapping):
                sub_blockers = [
                    _blocker(
                        "stage_contract_lineage_missing",
                        stage_id=sub_stage_id,
                        expected="receipt.lineage object",
                        observed="<missing>",
                    )
                ]
            else:
                sub_blockers = verify_lineage(
                    workspace_root,
                    run_id,
                    sub_lineage,
                    stage_id=sub_stage_id,
                    transitive=True,
                    registry_path=registry_path,
                    _visited=visited,
                    _chain=chain,
                    _output_cache=output_cache,
                )
            if sub_blockers:
                blockers.append(
                    _blocker(
                        "stage_contract_upstream_stale",
                        stage_id=stage_id,
                        expected=f"{rel_path} lineage verifies",
                        observed=[item.get("code") for item in sub_blockers],
                        extra={
                            "upstream_stage_id": upstream_stage,
                            "receipt": rel_path,
                            "chain": list(chain),
                            "upstream_blockers": sub_blockers,
                        },
                    )
                )

    if only_path is None:
        blockers.extend(verify_input_artifacts(workspace_root, lineage, stage_id=stage_id))
    return blockers


def verify_input_artifacts(
    workspace: str | Path,
    lineage: Mapping[str, Any] | None,
    *,
    stage_id: str,
    names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Blockers for recorded input artifacts that moved or disappeared."""

    if not isinstance(lineage, Mapping):
        return []
    workspace_root = Path(workspace).resolve()
    inputs = lineage.get("input_artifacts")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    selected = set(names) if names is not None else None
    blockers: list[dict[str, Any]] = []
    for name, record in inputs.items():
        if selected is not None and name not in selected:
            continue
        record = record if isinstance(record, Mapping) else {}
        rel_path = record.get("path")
        recorded = record.get("sha256")
        optional = bool(record.get("optional"))
        if not isinstance(rel_path, str) or not rel_path:
            if optional:
                continue
            blockers.append(
                _blocker(
                    "stage_contract_input_artifact_missing",
                    stage_id=stage_id,
                    expected=f"input artifact {name}",
                    observed="<unbound>",
                    extra={"input": name},
                )
            )
            continue
        path = workspace_root / rel_path
        current = _artifact_sha256(path, exclude_keys=record.get("exclude_keys") or ())
        if current is None:
            if optional and recorded is None:
                continue
            blockers.append(
                _blocker(
                    "stage_contract_input_artifact_missing",
                    stage_id=stage_id,
                    expected=rel_path,
                    observed="<missing>",
                    extra={"input": name, "recorded": recorded},
                )
            )
            continue
        if current != recorded:
            blockers.append(
                _blocker(
                    "stage_contract_input_artifact_changed",
                    stage_id=stage_id,
                    expected=recorded,
                    observed=current,
                    extra={"input": name, "path": rel_path},
                )
            )
    return blockers


# --------------------------------------------------------------------------
# Backfill (one-time migration of receipts written before lineage existed)
# --------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha_values(payload: Any, *, _found: set[str] | None = None) -> set[str]:
    """Every sha256-shaped string anywhere in a JSON payload."""

    found = _found if _found is not None else set()
    if isinstance(payload, str):
        if _SHA256_RE.match(payload):
            found.add(payload)
    elif isinstance(payload, Mapping):
        for value in payload.values():
            _sha_values(value, _found=found)
    elif isinstance(payload, list):
        for value in payload:
            _sha_values(value, _found=found)
    return found


def _stage_number(filename: str) -> int:
    match = re.match(r"^stage-(\d+)", filename)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)", filename)
    return int(match.group(1)) if match else 99


def receipt_stage_orders_from_registry(registry: Mapping[str, Any]) -> dict[str, int]:
    """``{stage_id: stage_order}`` for bindings that declare a receipt.

    The runners' stage tables are the truth (``14A-exhaustive-topic-extraction``
    is order 7, beside ``analysis``); a binding without ``stage_order`` falls
    back to the number in its receipt filename.
    """

    orders: dict[str, int] = {}
    for stage_id, binding in (registry.get("stage_bindings") or {}).items():
        if not isinstance(binding, Mapping):
            continue
        filename = binding.get("receipt_filename")
        if not isinstance(filename, str) or not filename:
            continue
        declared = binding.get("stage_order")
        orders[str(stage_id)] = int(declared) if isinstance(declared, int) else _stage_number(filename)
    return orders


_IDENTITY_EXCLUDED_KEYS = ("script_registry_sha256", "output_digest", "output_artifacts", "lineage", "stage_contract")


def _receipt_body_digest(payload: Mapping[str, Any]) -> str:
    """Digest of a receipt minus what backfill adds (same execution, same body)."""

    return canonical_json_digest(
        {key: value for key, value in payload.items() if key not in ("output_digest", "output_artifacts", "lineage")}
    )


def _all_sha_values(payload: Mapping[str, Any]) -> set[str]:
    return _sha_values({key: value for key, value in payload.items() if key not in _IDENTITY_EXCLUDED_KEYS})


def _output_table_shas(
    workspace_root: Path, payload: Mapping[str, Any], outputs_table: Sequence[Sequence[Any]], *, measure_disk: bool
) -> set[str]:
    """Shas of the outputs a receipt version recorded (and, for the current
    version only, of the outputs on disk that it recorded no sha for)."""

    records = artifacts_from_checks(payload, outputs_table)
    if measure_disk:
        records = digest_output_artifacts(workspace_root, records)
    return {str(record["sha256"]) for record in records if isinstance(record.get("sha256"), str) and record["sha256"]}


def _stage_tokens(filename: str) -> set[str]:
    """``stage-06-x.json`` -> {"6", "06"}; ``14A-x.json`` -> {"14a"}."""

    match = re.match(r"^(?:stage-)?(\d+)([A-Za-z]?)", filename)
    if not match:
        return set()
    number, letter = match.group(1), match.group(2).lower()
    return {f"{number}{letter}", f"{int(number)}{letter}"}


def _claims_about(checks: Any, tokens: set[str]) -> dict[str, str]:
    """``{dotted_key: sha}`` for every sha a receipt recorded under a
    top-level ``checks`` key named for the upstream stage (``stage8_receipt_sha256``,
    ``stage6_output_inventory.sha256``)."""

    claims: dict[str, str] = {}
    if not isinstance(checks, Mapping) or not tokens:
        return claims
    prefixes = tuple(f"stage{token}_" for token in tokens)
    for key, value in checks.items():
        if not str(key).lower().startswith(prefixes):
            continue
        for sha in sorted(_sha_values(value)):
            claims[f"{key}:{sha[:12]}" if not isinstance(value, str) else str(key)] = sha
    return claims


def _upstream_evidence(
    workspace_root: Path,
    path: Path,
    payload: Mapping[str, Any],
    *,
    downstream_checks: Any,
    own_evidence: set[str],
    original_file_sha: Mapping[Path, str],
    upstream_stage_id: str | None,
    registry_path: str | Path | None,
) -> dict[str, Any]:
    """Decide whether a downstream receipt consumed the *current* upstream version.

    Version-identifying values of an upstream version are its receipt file
    sha and the shas of its registry output-table artifacts -- never values it
    merely shares with its consumers (the analysis manifest digest that stages
    8 and 9 both record says nothing about which stage-8 output stage 9 read).

    * ``bound``: a downstream check equals a version-identifying value of the
      current upstream, and nothing contradicts it.
    * ``contradicted``: a downstream ``stage<N>_*`` claim names a sha the
      current upstream does not carry, or a downstream value matches only a
      superseded upstream version.
    * ``none``: no evidence either way.
    """

    resolved = path.resolve()
    outputs_table: list[list[Any]] = []
    if upstream_stage_id:
        outputs_table, _ = artifact_tables_for_stage(workspace_root, upstream_stage_id, registry_path=registry_path)

    current_body = _receipt_body_digest(payload)
    current_file_shas = {original_file_sha.get(resolved) or sha256_file(path)}
    for copy in sorted(path.parent.glob(f"{path.stem}.{_ARCHIVE_SUFFIX_PREBACKFILL}-*{path.suffix}")):
        copy_payload = _load_json_object(copy)
        if copy_payload is not None and _receipt_body_digest(copy_payload) == current_body:
            current_file_shas.add(sha256_file(copy))
    version_identity = current_file_shas | _output_table_shas(workspace_root, payload, outputs_table, measure_disk=True)
    full_identity = version_identity | _all_sha_values(payload)

    superseded: list[tuple[str, set[str]]] = []
    for copy in sorted(path.parent.glob(f"{path.stem}.{_ARCHIVE_SUFFIX_SUPERSEDED}-*{path.suffix}")):
        copy_payload = _load_json_object(copy)
        if copy_payload is None or _receipt_body_digest(copy_payload) == current_body:
            continue
        identity = {sha256_file(copy)} | _output_table_shas(workspace_root, copy_payload, outputs_table, measure_disk=False)
        superseded.append((copy.name, identity))

    claims = _claims_about(downstream_checks, _stage_tokens(path.name))
    contradicted = sorted(key for key, sha in claims.items() if sha not in full_identity)
    older_only = own_evidence - version_identity
    for copy_name, identity in superseded:
        for sha in sorted(older_only & identity):
            contradicted.append(f"{copy_name}:{sha[:12]}")
    matched = sorted((own_evidence & version_identity) | {sha for sha in claims.values() if sha in full_identity})
    consumed = next((copy_name for copy_name, identity in superseded if older_only & identity), None)
    if contradicted:
        verdict = "contradicted"
    elif matched:
        verdict = "bound"
    else:
        verdict = "none"
    return {
        "verdict": verdict,
        "matched": matched,
        "contradicted": contradicted,
        "consumed_superseded_copy": consumed,
    }


def _ordered_production_stages(registry: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(stage_id, receipt_filename)`` in ``stage_order``, dependencies first on ties.

    Two stages can share an order (``presentation`` and ``notebooklm-podcast``
    are both 10); a stage must still come after every stage it requires
    (``required_upstream_stages`` in any layout, or a ``required_receipts``
    path naming another stage's receipt), in the root binding, any workspace
    binding, or any profile.
    """

    filenames = receipt_filenames_from_registry(registry)
    orders = receipt_stage_orders_from_registry(registry)
    by_filename = {filename: stage_id for stage_id, filename in filenames.items()}
    deps: dict[str, set[str]] = {stage_id: set() for stage_id in filenames}

    def collect(stage_id: str, contract: Any) -> None:
        if stage_id not in deps or not isinstance(contract, Mapping):
            return
        for entry in contract.get("required_upstream_stages") or []:
            upstream = entry if isinstance(entry, str) else (entry.get("stage") if isinstance(entry, Mapping) else None)
            if isinstance(upstream, str) and upstream in deps and upstream != stage_id:
                deps[stage_id].add(upstream)
        for spec in contract.get("required_receipts") or []:
            if isinstance(spec, Mapping) and isinstance(spec.get("path"), str):
                upstream = by_filename.get(Path(spec["path"]).name)
                if upstream is not None and upstream != stage_id:
                    deps[stage_id].add(upstream)

    for stage_id, binding in (registry.get("stage_bindings") or {}).items():
        if not isinstance(binding, Mapping):
            continue
        collect(str(stage_id), binding.get("runtime_contract"))
        for workspace_binding in (binding.get("workspace_bindings") or {}).values():
            if isinstance(workspace_binding, Mapping):
                collect(str(stage_id), workspace_binding.get("runtime_contract"))
    for profile in (registry.get("runtime_contract_profiles") or {}).values():
        if isinstance(profile, Mapping):
            for stage_id, contract in profile.items():
                collect(str(stage_id), contract.get("runtime_contract") if isinstance(contract, Mapping) and "runtime_contract" in contract else contract)

    priority = {stage_id: (orders.get(stage_id, 99), filenames[stage_id]) for stage_id in filenames}
    ordered: list[str] = []
    remaining = set(filenames)
    while remaining:
        ready = [stage_id for stage_id in remaining if not (deps[stage_id] & remaining)]
        # A dependency cycle cannot be honoured: fall back to plain order.
        chosen = min(ready or remaining, key=lambda stage_id: priority[stage_id])
        ordered.append(chosen)
        remaining.discard(chosen)
    return [(stage_id, filenames[stage_id]) for stage_id in ordered]


def backfill_run_lineage(
    workspace: str | Path,
    run_id: str,
    *,
    stages: Sequence[str] | None = None,
    apply: bool = False,
    assume_current: Sequence[str] = (),
    operator: str | None = None,
    force: bool = False,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Stamp ``output_digest`` and ``lineage`` onto receipts written before lineage existed.

    Evidence-based and read-only unless ``apply``: a receipt may bind an
    upstream stage only when a digest it recorded in its own ``checks`` (an
    index digest, a receipt sha, a manifest digest) matches something the
    upstream's *current* receipt carries -- proof it consumed that version.
    No evidence ⇒ the entry is recorded with ``digest: null`` and
    ``unresolved: true`` and the receipt stays stale; ``assume_current``
    names stages the operator attests are consistent with their current
    upstreams, recorded under ``lineage.backfill``. Only ``succeeded``
    receipts (with a passing ``stage_contract`` when one is present) are
    stamped; already-bound receipts are skipped unless ``force``.
    """

    from conference_pipeline.stage_contracts import verify_stage_contract
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    workspace_root = Path(workspace).resolve()
    run_dir = workspace_root / "storage" / "workflow-runs" / run_id
    registry = load_stage_script_registry(registry_path)
    filenames = receipt_filenames_from_registry(registry)
    selected = set(stages) if stages else None
    attested = set(assume_current)
    resolver = receipt_path_resolver_from_registry(workspace_root, run_dir, registry_path=registry_path)
    stage_receipt_names = set(filenames.values())
    report: dict[str, Any] = {
        "run_id": run_id,
        "workspace": str(workspace_root),
        "applied": bool(apply),
        "operator": operator,
        "stages": {},
    }

    # Receipts stamped earlier in this pass (in dry-run mode they are not on
    # disk yet) so a downstream stage binds the upstream's new output_digest.
    stamped_by_path: dict[Path, dict[str, Any]] = {}
    # Original bytes of every receipt, hashed before this pass writes any:
    # downstream evidence recorded an upstream's receipt sha, and --apply
    # rewrites upstreams before their consumers are examined.
    original_file_sha: dict[Path, str] = {
        (run_dir / filename).resolve(): sha256_file(run_dir / filename)
        for filename in filenames.values()
        if (run_dir / filename).is_file()
    }
    stage_by_filename = {filename: stage_id for stage_id, filename in filenames.items()}
    for stage_id, filename in _ordered_production_stages(registry):
        if selected is not None and stage_id not in selected:
            continue
        path = run_dir / filename
        entry: dict[str, Any] = {"receipt": workspace_relpath(workspace_root, path)}
        report["stages"][stage_id] = entry
        payload = _load_json_object(path) if path.is_file() else None
        if payload is None:
            entry["action"] = "absent"
            continue
        receipt = dict(payload)
        status = str(receipt.get("status") or "")
        contract = receipt.get("stage_contract")
        contract_status = contract.get("status") if isinstance(contract, Mapping) else None
        if status != "succeeded" or (contract_status is not None and contract_status != "pass"):
            entry.update({"action": "refused", "status": status, "stage_contract_status": contract_status})
            continue
        if isinstance(receipt.get("lineage"), Mapping) and not force:
            entry["action"] = "skip_already_bound"
            continue

        outputs_table, inputs_table = artifact_tables_for_stage(workspace_root, stage_id, registry_path=registry_path)
        output_artifacts = digest_output_artifacts(workspace_root, artifacts_from_checks(receipt, outputs_table))
        inputs, binding_source = resolve_lineage_inputs(
            workspace_root, run_id, stage_id, receipt_path_for_stage=resolver, registry_path=registry_path
        )
        # Evidence = digests this stage recorded about what it read (its
        # ``checks``), never the registry digest every receipt shares.
        shared = {str(receipt.get("script_registry_sha256") or "")}
        own_evidence = _sha_values(receipt.get("checks")) - shared
        upstream_receipts: dict[str, dict[str, Any]] = {}
        unresolved: list[str] = []
        for item in inputs:
            if item.kind != "receipt":
                continue
            key = workspace_relpath(workspace_root, item.path) if item.path else f"stage:{item.name}"
            upstream_payload = None
            if item.path is not None:
                upstream_payload = stamped_by_path.get(item.path.resolve()) or (
                    _load_json_object(item.path) if item.path.is_file() else None
                )
            record: dict[str, Any] = {
                "stage_id": item.stage_id,
                "spec_path": item.spec_path,
                "source": item.source,
                "digest_kind": DIGEST_KIND_CONTENT if item.lineage_mode == LINEAGE_MODE_CONTENT else DIGEST_KIND_OUTPUT,
                "digest": None,
                "status": upstream_payload.get("status") if upstream_payload else None,
                "optional": item.optional,
                "missing": upstream_payload is None,
                "backfill_evidence": None,
            }
            if upstream_payload is None:
                if not item.optional:
                    record["unresolved"] = True
                    unresolved.append(key)
                upstream_receipts[key] = record
                continue
            kind, current = upstream_binding_digest(
                upstream_payload,
                path=item.path,
                lineage_mode=item.lineage_mode,
                stage_receipt_names=stage_receipt_names,
                substep_receipt_names=substep_receipt_names(registry),
            )
            record["digest_kind"] = kind
            if item.stage_id is None and isinstance(upstream_payload.get("stage_id"), str):
                record["stage_id"] = upstream_payload["stage_id"]
            if kind == DIGEST_KIND_CONTENT:
                record["digest"] = current
                record["backfill_evidence"] = "content_bound"
            elif stage_id in attested:
                record["digest"] = current
                record["backfill_evidence"] = "operator_attested"
            else:
                evidence = _upstream_evidence(
                    workspace_root,
                    item.path,
                    upstream_payload,
                    downstream_checks=receipt.get("checks"),
                    own_evidence=own_evidence,
                    original_file_sha=original_file_sha,
                    upstream_stage_id=record["stage_id"] or stage_by_filename.get(item.path.name),
                    registry_path=registry_path,
                )
                if evidence["verdict"] == "bound" and current is not None:
                    record["digest"] = current
                    record["backfill_evidence"] = {"matched_digests": evidence["matched"][:4]}
                else:
                    record["unresolved"] = True
                    if evidence.get("consumed_superseded_copy"):
                        record["backfill_evidence"] = {"consumed_superseded_copy": evidence["consumed_superseded_copy"]}
                    elif evidence["verdict"] == "contradicted":
                        record["backfill_evidence"] = {"contradicted_by": evidence["contradicted"][:8]}
                    else:
                        record["backfill_evidence"] = (
                            "upstream_output_digest_missing" if current is None else "no_matching_digest"
                        )
                    unresolved.append(key)
            upstream_receipts[key] = record

        input_artifacts: dict[str, dict[str, Any]] = {}
        for item in inputs:
            if item.kind == "artifact":
                input_artifacts[item.name] = _input_artifact_record(
                    workspace_root, item.path, source=item.source, optional=item.optional
                )
        for extra in artifacts_from_checks(receipt, inputs_table):
            candidate = Path(str(extra["path"]))
            if not candidate.is_absolute():
                candidate = workspace_root / candidate
            input_artifacts[str(extra["name"])] = _input_artifact_record(
                workspace_root, candidate, source=SOURCE_STAGE, optional=False
            )

        lineage: dict[str, Any] = {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "bound_at": _utc_now_iso(),
            "binding_source": binding_source,
            "upstream_receipts": dict(sorted(upstream_receipts.items())),
            "input_artifacts": dict(sorted(input_artifacts.items())),
            "code_identity": {"script_registry_sha256": binding_source.get("registry_sha256"), "runner": "backfill"},
            "output_digest_components": {},
            "backfill": {
                "method": "operator_attested" if stage_id in attested else "evidence_based",
                "operator": operator,
                "at": _utc_now_iso(),
                "unresolved": unresolved,
            },
        }
        lineage["lineage_digest"] = lineage_digest(lineage)
        stamped = apply_lineage(receipt, lineage=lineage, output_artifacts=output_artifacts)
        stamped_by_path[path.resolve()] = stamped
        entry.update(
            {
                "action": "stamp" if apply else "would_stamp",
                "output_digest": stamped["output_digest"],
                "output_artifact_count": len(output_artifacts),
                "upstream_count": len(upstream_receipts),
                "unresolved": unresolved,
                "method": lineage["backfill"]["method"],
            }
        )
        if not apply:
            continue
        archived = preserve_prior_receipt(path, new_status=status, suffix=_ARCHIVE_SUFFIX_PREBACKFILL)
        entry["preserved_prior"] = workspace_relpath(workspace_root, archived) if archived else None
        _write_json_atomic(path, stamped)
        verdict = verify_stage_contract(
            workspace=workspace_root,
            run_id=run_id,
            stage_id=stage_id,
            receipt=stamped,
            mode=str(stamped.get("mode") or "live"),
            registry_path=registry_path,
            require_lineage=True,
        )
        entry["verdict"] = verdict.status
        entry["verdict_blockers"] = [blocker.get("code") for blocker in verdict.blockers][:8]
    return report


# --------------------------------------------------------------------------
# Read-only stage status
# --------------------------------------------------------------------------


def stage_lineage_report(
    workspace: str | Path,
    run_id: str,
    *,
    layout: str = "auto",
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify every stage of a run as fresh / stale / missing / failed, read-only.

    ``layout`` is ``production`` (registry ``receipt_filename`` bindings in
    stage order), ``generic`` (``workflow.WORKFLOW_STAGES``), or ``auto``
    (production when a production receipt is present). A stage is *stale*
    when its receipt has no lineage, when any upstream it recorded moved, or
    when an upstream is itself stale (the chain is reported); ``would_invalidate``
    lists, per stage, the downstream stages that bound it, and
    ``next_stage_to_run`` is the first stage that is not fresh.
    """

    from conference_pipeline.stage_script_registry import load_stage_script_registry

    workspace_root = Path(workspace).resolve()
    run_dir = workspace_root / "storage" / "workflow-runs" / run_id
    registry = load_stage_script_registry(registry_path)
    filenames = receipt_filenames_from_registry(registry)
    resolved_layout = layout
    if layout == "auto":
        production_present = any((run_dir / name).is_file() for name in filenames.values())
        resolved_layout = "production" if production_present else "generic"

    ordered: list[tuple[str, Path]] = []
    if resolved_layout == "production":
        for stage_id, filename in _ordered_production_stages(registry):
            ordered.append((stage_id, run_dir / filename))
    else:
        from conference_pipeline import workflow
        from conference_pipeline.workflow_runs import stage_receipt_path

        for stage in sorted(workflow.WORKFLOW_STAGES, key=lambda item: item.order):
            ordered.append((stage.id, stage_receipt_path(run_dir, stage)))

    stages: dict[str, dict[str, Any]] = {}
    bound_by: dict[str, list[str]] = {}
    next_stage: str | None = None
    output_cache: dict[Any, frozenset[str]] = {}
    for stage_id, path in ordered:
        entry: dict[str, Any] = {"receipt": workspace_relpath(workspace_root, path)}
        payload = _load_json_object(path) if path.is_file() else None
        if payload is None:
            entry["state"] = "missing"
        else:
            status = str(payload.get("status") or "")
            entry["status"] = status
            lineage = payload.get("lineage")
            if isinstance(lineage, Mapping):
                upstream = lineage.get("upstream_receipts")
                for rel_path, record in (upstream or {}).items():
                    upstream_stage = record.get("stage_id") if isinstance(record, Mapping) else None
                    key = str(upstream_stage or rel_path)
                    bound_by.setdefault(key, []).append(stage_id)
            if status != "succeeded":
                entry["state"] = "blocked" if status == "blocked" else ("failed" if status else "unknown")
            else:
                blockers = verify_lineage(
                    workspace_root, run_id, lineage if isinstance(lineage, Mapping) else None,
                    stage_id=stage_id, transitive=True, registry_path=registry_path,
                    _output_cache=output_cache,
                )
                if not isinstance(payload.get("output_digest"), str) or not payload.get("output_digest"):
                    blockers.append(
                        _blocker(
                            "stage_contract_output_digest_missing",
                            stage_id=stage_id,
                            expected="output_digest on the receipt",
                            observed="<missing>",
                        )
                    )
                # The stage's own recorded outputs edited after it ran: re-run it.
                changed = changed_output_artifacts(
                    workspace_root, payload, receipt_mtime=path.stat().st_mtime, cache=output_cache
                )
                if changed:
                    blockers.insert(
                        0,
                        _blocker(
                            "stage_contract_output_changed",
                            stage_id=stage_id,
                            expected="recorded output_artifacts match disk",
                            observed=[item["path"] for item in changed],
                            extra={
                                "receipt": entry["receipt"],
                                "paths": [item["path"] for item in changed],
                                "changed_outputs": changed,
                            },
                        ),
                    )
                if blockers:
                    entry["state"] = "stale"
                    entry["reasons"] = [
                        {
                            "code": blocker.get("code"),
                            "upstream_stage_id": blocker.get("upstream_stage_id"),
                            "receipt": blocker.get("receipt"),
                            "chain": blocker.get("chain"),
                            "input": blocker.get("input"),
                            **({"paths": blocker["paths"]} if blocker.get("paths") else {}),
                        }
                        for blocker in blockers
                    ]
                else:
                    entry["state"] = "fresh"
        if next_stage is None and entry["state"] != "fresh":
            next_stage = stage_id
        stages[stage_id] = entry
    for stage_id, entry in stages.items():
        entry["would_invalidate"] = sorted(set(bound_by.get(stage_id, [])))
    report: dict[str, Any] = {
        "run_id": run_id,
        "workspace": str(workspace_root),
        "layout": resolved_layout,
        "stages": stages,
        "next_stage_to_run": next_stage,
    }
    if resolved_layout == "production":
        substeps = _substep_lineage_report(
            workspace_root, run_id, registry, registry_path=registry_path, output_cache=output_cache
        )
        report["substeps"] = substeps
        report["next_substep_to_run"] = next(
            (substep_id for substep_id, entry in substeps.items() if entry["state"] != "fresh"), None
        )
    return report


def _ordered_substeps(registry: Mapping[str, Any]) -> list[str]:
    """Declared sub-steps, dependencies first (declaration order otherwise)."""

    specs = substep_specs(registry)
    ordered: list[str] = []

    def _visit(node: str) -> None:
        if node in ordered or node not in specs:
            return
        for upstream in specs[node].get("required_substeps") or []:
            _visit(str(upstream))
        ordered.append(node)

    for node in specs:
        _visit(node)
    return ordered


def _substep_lineage_report(
    workspace_root: Path,
    run_id: str,
    registry: Mapping[str, Any],
    *,
    registry_path: str | Path | None = None,
    output_cache: dict[Any, frozenset[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Classify each declared sub-step receipt of a run like a stage."""

    specs = substep_specs(registry)
    report: dict[str, dict[str, Any]] = {}
    for substep_id in _ordered_substeps(registry):
        path = _substep_receipt_path(workspace_root, run_id, specs[substep_id])
        entry: dict[str, Any] = {"receipt": workspace_relpath(workspace_root, path) if path else None}
        payload = _load_json_object(path) if path is not None and path.is_file() else None
        if payload is None:
            entry["state"] = "missing"
        else:
            status = str(payload.get("status") or "")
            entry["status"] = status
            if status != "succeeded":
                entry["state"] = "blocked" if status == "blocked" else ("failed" if status else "unknown")
            else:
                lineage = payload.get("lineage")
                blockers = verify_lineage(
                    workspace_root, run_id, lineage if isinstance(lineage, Mapping) else None,
                    stage_id=substep_id, transitive=True, registry_path=registry_path,
                    _output_cache=output_cache,
                )
                if blockers:
                    entry["state"] = "stale"
                    entry["reasons"] = [
                        {
                            "code": blocker.get("code"),
                            "upstream_stage_id": blocker.get("upstream_stage_id"),
                            "receipt": blocker.get("receipt"),
                            "chain": blocker.get("chain"),
                        }
                        for blocker in blockers
                    ]
                else:
                    entry["state"] = "fresh"
        report[substep_id] = entry
    return report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


# --------------------------------------------------------------------------
# Supersession
# --------------------------------------------------------------------------


def preserve_prior_receipt(
    canonical_path: str | Path, *, new_status: str, stamp: str | None = None, suffix: str | None = None
) -> Path | None:
    """Archive the receipt currently at ``canonical_path`` before it is rewritten.

    A prior ``succeeded`` receipt is kept as ``<stem>.superseded-<stamp>[-n].json``;
    any other prior receipt as ``<stem>.attempt-<stamp>[-n].json``; ``suffix``
    overrides the label (backfill keeps its pre-stamp copy as ``prebackfill``,
    the same execution rather than a superseded one). Returns the
    archive path, or ``None`` when there was nothing to preserve. The caller
    then writes the new receipt to ``canonical_path``.
    """

    path = Path(canonical_path)
    if not path.is_file():
        return None
    prior_bytes = path.read_bytes()
    prior = _load_json_object(path)
    prior_status = str((prior or {}).get("status") or "")
    if suffix is None:
        suffix = _ARCHIVE_SUFFIX_SUPERSEDED if prior_status == "succeeded" else _ARCHIVE_SUFFIX_ATTEMPT
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"{path.stem}.{suffix}-{stamp}{path.suffix}")
    sequence = 1
    while archive.exists():
        archive = path.with_name(f"{path.stem}.{suffix}-{stamp}-{sequence}{path.suffix}")
        sequence += 1
    temp = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    temp.write_bytes(prior_bytes)
    os.replace(temp, archive)
    return archive
