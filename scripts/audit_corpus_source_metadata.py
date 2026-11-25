#!/usr/bin/env python3
"""Fail-closed per-source-family paper title/author coverage audit.

Phase 10's first live deck/figures run blocked all 14 decks on
``MIN_SOURCES``/``people`` thresholds. Root cause was not per-topic sparsity: 0
of 653 reference-ledger entries had any resolved author, and every
``papers``-family entry had an empty title
(``../10-deck-and-presentation-production/10-BLOCKER-corpus-metadata-gap.md``).
This audit exists so that gap can never again ship silently: it scans a
conference workspace's real, on-disk artifacts (the corpus inventory, the
papers vector-store chunk store, the topic reference ledgers, and the
speaker/author work ledgers), measures title and author coverage per source
family, and exits non-zero when a gated family falls below its justified
floor. Unlike the plan-text-presence audits this module is deliberately
modeled on (``audit_conference_hardcoding.py`` / ``audit_model_registry_hardcoding.py``),
this is a RUNTIME scan of generated artifacts, not a text-presence check.

Family classification is delegated entirely to
``conference_pipeline.indexing.paper_source_metadata.classify_record_family``
(plan 10.1-01) -- this module never redefines it. Coverage is always reported
per family (``papers_main`` / ``papers_workshop`` / ``artifacts``) and never
blended into one aggregate percentage; the ``artifacts`` family is reported
but is explicitly non-gating because it is structurally author-less (video
caption transcripts rarely name a co-author on camera).

Run:
    python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug>            # human report
    python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug> --json      # machine report
    python scripts/audit_corpus_source_metadata.py --workspace conferences/<slug> \\
        --write-receipt storage/workflow-runs/<run-id>                                        # + coverage receipt

Fix pattern for a real finding: run the corpus-inventory/source-discovery join
and re-derivation path (``conference_pipeline.indexing.paper_source_metadata``,
``scripts/rederive_source_discovery_authors.py``) so ``paper_title``/``authors``
land on the corpus inventory, then re-index and re-synthesize so the vector
store, ``analysis/topic_reference_ledgers/*.json``, and
``decks/*/speaker_author_work_ledger.json`` carry the same metadata forward.

No LLM call, no paid provider, no conference identity: this audit is
deterministic, free, and conference-neutral.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from conference_pipeline.indexing.corpus_vector_store import PAPER_SOURCE_FAMILIES
from conference_pipeline.indexing.paper_source_metadata import (
    SOURCE_FAMILY_CLASS_ARTIFACT,
    SOURCE_FAMILY_CLASS_MAIN_PAPER,
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER,
    SOURCE_FAMILY_CLASSES,
    classify_record_family,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# `PAPER_SOURCE_FAMILIES` (imported above) is the platform's own, already
# established {"paper_text", "workshop_paper_text"} vocabulary -- reused here
# rather than re-declaring the pair, so this module cannot drift from
# corpus_vector_store's/paper_source_metadata's definition of the two ids.
_WORKSHOP_PAPER_FAMILY_TOKEN = "workshop_paper_text"
_MAIN_PAPER_FAMILY_TOKEN = next(iter(PAPER_SOURCE_FAMILIES - {_WORKSHOP_PAPER_FAMILY_TOKEN}))

STRUCTURALLY_AUTHORLESS = "structurally_authorless"

RECEIPT_TYPE = "conference_pipeline.corpus_source_metadata_coverage"
RECEIPT_DEFAULT_FILENAME = "corpus_source_metadata_coverage.json"

# ---------------------------------------------------------------------------
# Coverage floors -- each justified by an EXECUTION-TIME re-measurement taken
# by the executor of plan 10.1-06 against the real CVPR2026 workspace on
# 2026-07-31 (not copied from the RESEARCH session; see each justification for
# the exact command/numerator/denominator, and the plan's SUMMARY for the full
# re-measurement record including where the fresh number matched or diverged
# from the RESEARCH-session figure).
# ---------------------------------------------------------------------------

COVERAGE_FLOORS: dict[str, dict[str, Any]] = {
    SOURCE_FAMILY_CLASS_MAIN_PAPER: {
        "title_floor": 0.99,
        "authors_floor": 0.99,
        "provisional": False,
        "justification": (
            "Execution-time re-measurement 2026-07-31 against the real CVPR2026 workspace: "
            "resolve_main_paper_metadata (paper_source_metadata, plan 10.1-01) resolves TITLE "
            "for 4068/4068 (100.0%) of on-disk papers_main corpus_inventory.json records via "
            "the source-discovery join (build_program_item_index over the live "
            "source_discovery_inventory.json + download_corpus.json + corpus_inventory.json) "
            "-- byte-identical to the RESEARCH-session figure, zero drift. AUTHOR-join "
            "capability was re-verified the same day via `scripts/rederive_source_discovery_"
            "authors.py --workspace conferences/cvpr2026 --run-id "
            "cvpr2026-live-production-r10-main-paper-corpus --dry-run --json`, which reports "
            "records_authors_added=4068/4068 (100.0%) extractable from the already-cached CVF "
            "listing HTML with network_calls=0 -- the real (non-dry-run) backfill has not yet "
            "been executed on this workspace, so this floor gates a workspace AFTER that "
            "backfill lands, not the current un-backfilled state. Independently corroborated "
            "by a fresh header-parse-fallback re-measurement (`python -m "
            "conference_pipeline.indexing.paper_source_metadata --measure-headers "
            "conferences/cvpr2026/storage/corpus/paper-text --sample 300 --seed 42 --json`), "
            "which reproduced 282/300 (94.0%) title / 251/300 (83.7%) >=2-author resolution on "
            "2026-07-31 -- an EXACT match to the RESEARCH-session baseline, zero drift. ~1pp "
            "headroom applied below the 100.0% join figure."
        ),
    },
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER: {
        "title_floor": 0.99,
        "authors_floor": 0.95,
        "provisional": True,
        "justification": (
            "Execution-time re-measurement 2026-07-31: workshop-paper chunk titles are "
            "7181/7181 (100.0%) present in the current on-disk "
            "storage/vector-store/cvpr2026-live-production/stores/papers/chunks.jsonl "
            "(source_context.paper_title) -- an exact match to the RESEARCH-session figure, "
            "zero drift. Workshop-receipt author presence, which RESEARCH's own note said was "
            "'inspected but never exhaustively counted', was counted EXHAUSTIVELY for the "
            "first time this execution: 505/517 (97.68%) of "
            "storage/workshops/papers/*/receipts/*.json succeeded receipts carry a non-empty "
            "authors list. authors_floor is kept at 0.95 (below, not at, the freshly measured "
            "97.68%) and marked provisional=true because this is the first exhaustive count "
            "and the receipt population can still shift as workshop acquisition continues."
        ),
    },
    SOURCE_FAMILY_CLASS_ARTIFACT: {
        "title_floor": None,
        "authors_floor": None,
        "authors_classification": STRUCTURALLY_AUTHORLESS,
        "provisional": False,
        "justification": (
            "Execution-time re-measurement 2026-07-31: the artifacts family (YouTube caption "
            "transcripts, source_family_id=speech_transcript) is structurally author-less -- "
            "presenters rarely state a full co-author byline on camera -- and no repair for "
            "this family is in phase 10.1's scope. RESEARCH's blocker evidence measured "
            "roughly 12/593 (real titles / topic_reference_ledger entries) for this family; "
            "re-verified live against the current on-disk "
            "analysis/topic_reference_ledgers/*.json this same day: 0/593 artifacts-family "
            "entries have a non-empty 'titles' list -- matches the pre-fix blocker "
            "snapshot byte-for-byte (ledger regeneration reflecting this phase's extraction "
            "fixes has not yet run; that backfill is out of this plan's scope). Coverage is "
            "measured and reported for visibility but does not gate."
        ),
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finding(code: str, message: str, **extra: Any) -> dict[str, Any]:
    finding = {"code": code, "message": message}
    finding.update(extra)
    return finding


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return bool(value)


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 2) if total else 0.0


def _new_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "title_present": 0,
        "authors_present": 0,
        "title_provenance": Counter(),
        "authors_provenance": Counter(),
    }


def _tally(
    bucket: dict[str, Any],
    *,
    title: Any,
    authors: Any,
    provenance: Any,
) -> None:
    bucket["total"] += 1
    if _nonempty(title):
        bucket["title_present"] += 1
    if _nonempty(authors):
        bucket["authors_present"] += 1
    prov = provenance if isinstance(provenance, Mapping) else {}
    bucket["title_provenance"][str(prov.get("title") or "unknown")] += 1
    bucket["authors_provenance"][str(prov.get("authors") or "unknown")] += 1


def _family_token_for_path_based_entry(entry: Mapping[str, Any], path_key: str) -> str | None:
    """Bridge a topic-reference-ledger / speaker-author-work-ledger entry's
    two-way ``source_family`` vocabulary ("papers"/"artifacts") to the exact
    family-id token ``classify_record_family`` expects, WITHOUT redefining
    that function's own 3-way classification decision.

    These downstream ledgers do not carry ``source_family_id`` directly; they
    carry ``source_family`` ("papers" | "artifacts") plus a list of on-disk
    paths. Main-vs-workshop is derived from the platform-wide storage layout
    (workshop papers live under a ``workshops/`` path segment; main papers do
    not) -- a path-shape signal, not a conference-specific one.
    """

    if str(entry.get("source_family")) != "papers":
        return None
    paths = entry.get(path_key) or []
    if any("workshops/" in str(p) for p in paths if p):
        return _WORKSHOP_PAPER_FAMILY_TOKEN
    return _MAIN_PAPER_FAMILY_TOKEN


def _iter_jsonl_lines(path: Path) -> Iterable[str]:
    """Stream non-empty lines from a JSONL file one at a time.

    Never calls ``Path.read_text()`` / ``.readlines()`` on the file -- opens it
    once and iterates line-by-line, so a ~70,000-line chunk store is never
    held whole in memory (threat T-10.1-09).
    """

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield line


# ---------------------------------------------------------------------------
# Artifact readers -- each tolerates a missing file by recording a finding
# ---------------------------------------------------------------------------


def _read_corpus_inventory(
    workspace_path: Path, buckets: dict[str, dict[str, Any]], findings: list[dict[str, Any]]
) -> None:
    path = workspace_path / "storage" / "corpus" / "corpus_inventory.json"
    if not path.is_file():
        findings.append(_finding("corpus_inventory_missing", f"corpus inventory not found at {path}"))
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(_finding("corpus_inventory_unreadable", f"{path}: {exc}"))
        return
    records = data.get("artifact_records") if isinstance(data, Mapping) else None
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        family = classify_record_family(record)
        _tally(
            buckets[family],
            title=record.get("paper_title"),
            authors=record.get("authors"),
            provenance=record.get("source_metadata_provenance"),
        )


def _resolve_run_id(workspace_path: Path, run_id: str | None) -> str | None:
    if run_id:
        return run_id
    vector_store_dir = workspace_path / "storage" / "vector-store"
    if not vector_store_dir.is_dir():
        return None
    candidates = [child for child in vector_store_dir.iterdir() if child.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda child: child.stat().st_mtime, reverse=True)
    return candidates[0].name


def _read_papers_chunks(
    workspace_path: Path,
    run_id: str | None,
    buckets: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    if run_id is None:
        findings.append(_finding("vector_store_missing", "no vector-store run directory found"))
        return
    path = workspace_path / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    if not path.is_file():
        findings.append(_finding("vector_store_missing", f"papers chunk store not found at {path}"))
        return
    for line in _iter_jsonl_lines(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        source_context = record.get("source_context")
        source_context = source_context if isinstance(source_context, Mapping) else {}
        classify_input = source_context if source_context.get("source_family_id") else record
        family = classify_record_family(classify_input)
        title = source_context.get("paper_title") or record.get("paper_title")
        authors = source_context.get("authors") or record.get("authors")
        provenance = source_context.get("source_metadata_provenance") or record.get(
            "source_metadata_provenance"
        )
        _tally(buckets[family], title=title, authors=authors, provenance=provenance)


def _read_topic_reference_ledgers(
    workspace_path: Path, buckets: dict[str, dict[str, Any]], findings: list[dict[str, Any]]
) -> None:
    ledgers_dir = workspace_path / "analysis" / "topic_reference_ledgers"
    paths = sorted(ledgers_dir.glob("*.json")) if ledgers_dir.is_dir() else []
    if not paths:
        findings.append(_finding("ledgers_missing", f"no topic reference ledgers found under {ledgers_dir}"))
        return
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(_finding("ledgers_unreadable", f"{path}: {exc}"))
            continue
        entries = data.get("entries") if isinstance(data, Mapping) else None
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            token = _family_token_for_path_based_entry(entry, "source_paths")
            family = classify_record_family({"source_family_id": token} if token else {})
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
            _tally(
                buckets[family],
                title=entry.get("titles"),
                authors=metadata.get("authors"),
                provenance=None,
            )


def _read_speaker_author_work_ledgers(
    workspace_path: Path, buckets: dict[str, dict[str, Any]], findings: list[dict[str, Any]]
) -> None:
    decks_dir = workspace_path / "decks"
    paths = sorted(decks_dir.glob("*/speaker_author_work_ledger.json")) if decks_dir.is_dir() else []
    if not paths:
        findings.append(
            _finding("ledgers_missing", f"no speaker_author_work_ledger.json found under {decks_dir}")
        )
        return
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(_finding("ledgers_unreadable", f"{path}: {exc}"))
            continue
        entries = data.get("entries") if isinstance(data, Mapping) else None
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            token = _family_token_for_path_based_entry(entry, "evidence_paths")
            family = classify_record_family({"source_family_id": token} if token else {})
            _tally(
                buckets[family],
                title=entry.get("resolved_title"),
                authors=entry.get("authors"),
                provenance=None,
            )


# ---------------------------------------------------------------------------
# measure_workspace_coverage
# ---------------------------------------------------------------------------


SCOPES = ("all", "inventory")
# What --scope inventory leaves out, named in the output and never reported as passed
# (v6.0-Public-Release Phase 5, D4): a workspace whose analysis was not run has no
# papers vector store and no ledgers.
NOT_MEASURED_IN_INVENTORY_SCOPE = ("papers_vector_store", "topic_reference_ledgers", "speaker_author_work_ledgers")


def measure_workspace_coverage(
    workspace: str | Path, *, run_id: str | None = None, scope: str = "all"
) -> dict[str, Any]:
    """Measure real, on-disk per-source-family title/author coverage for a workspace.

    Reads, in order, tolerating a missing later artifact by recording a
    finding rather than raising: the corpus inventory, the papers-store vector
    chunk JSONL (streamed line-by-line), the topic reference ledgers, and the
    speaker/author work ledgers. Every record from every readable artifact
    contributes one more piece of evidence toward (or against) its family's
    coverage tally -- the four artifact types differ in natural granularity
    (paper-level, chunk-level, ledger-entry-level), so this is a combined
    evidence count across the whole propagation chain, not a deduplicated
    per-paper percentage. Never writes to any artifact it reads.
    """

    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    workspace_path = Path(workspace)
    resolved_run_id = _resolve_run_id(workspace_path, run_id) if scope == "all" else None

    buckets = {family_class: _new_bucket() for family_class in SOURCE_FAMILY_CLASSES}
    findings: list[dict[str, Any]] = []

    _read_corpus_inventory(workspace_path, buckets, findings)
    if scope == "all":
        _read_papers_chunks(workspace_path, resolved_run_id, buckets, findings)
        _read_topic_reference_ledgers(workspace_path, buckets, findings)
        _read_speaker_author_work_ledgers(workspace_path, buckets, findings)

    families: dict[str, Any] = {}
    for family_class in SOURCE_FAMILY_CLASSES:
        bucket = buckets[family_class]
        total = bucket["total"]
        family_report = {
            "total": total,
            "title_present": bucket["title_present"],
            "authors_present": bucket["authors_present"],
            "title_coverage_pct": _pct(bucket["title_present"], total),
            "authors_coverage_pct": _pct(bucket["authors_present"], total),
            "title_provenance": dict(bucket["title_provenance"]),
            "authors_provenance": dict(bucket["authors_provenance"]),
        }
        if family_class == SOURCE_FAMILY_CLASS_ARTIFACT:
            family_report["authors_classification"] = STRUCTURALLY_AUTHORLESS
        families[family_class] = family_report

    return {
        "workspace": str(workspace_path),
        "run_id": resolved_run_id,
        "scope": scope,
        "not_measured": list(NOT_MEASURED_IN_INVENTORY_SCOPE) if scope == "inventory" else [],
        "families": families,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Floor overrides + evaluation
# ---------------------------------------------------------------------------


def _load_floor_overrides(workspace_path: Path) -> Mapping[str, Any]:
    """Read an optional per-workspace floor override block from config.yaml.

    Shape:
        corpus_metadata_coverage:
          floor_overrides:
            papers_main:
              authors_floor: 0.90
              justification: "..."
    """

    config_path = workspace_path / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, Mapping):
        return {}
    block = data.get("corpus_metadata_coverage")
    if not isinstance(block, Mapping):
        return {}
    overrides = block.get("floor_overrides")
    return overrides if isinstance(overrides, Mapping) else {}


def _effective_floors(
    floor_overrides: Mapping[str, Any], findings: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    effective: dict[str, dict[str, Any]] = {}
    for family_class in SOURCE_FAMILY_CLASSES:
        base = dict(COVERAGE_FLOORS[family_class])
        override = floor_overrides.get(family_class)
        if isinstance(override, Mapping):
            for key in ("title_floor", "authors_floor"):
                if key not in override or override.get(key) is None:
                    continue
                base_value = base.get(key)
                override_value = override.get(key)
                justification = override.get("justification")
                lowering = base_value is not None and override_value < base_value
                if lowering and not (justification and str(justification).strip()):
                    findings.append(
                        _finding(
                            "floor_override_unjustified",
                            f"{family_class}.{key} override {override_value} is below the "
                            f"platform default {base_value} with no recorded justification",
                            family=family_class,
                            metric=key,
                            previous_value=base_value,
                            new_value=override_value,
                        )
                    )
                    continue
                base[key] = override_value
                base[f"{key}_override"] = {
                    "previous_value": base_value,
                    "new_value": override_value,
                    "justification": justification,
                }
        effective[family_class] = base
    return effective


def evaluate_floors(measurements: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate measured coverage against the (possibly overridden) floors.

    Mutates ``measurements`` in place to attach the resolved
    ``effective_floors`` (so callers such as ``main()``/the receipt writer can
    read the exact floors that were applied), and returns the combined list of
    findings -- both the measurement-time findings already present on
    ``measurements['findings']`` and any floor-evaluation findings discovered
    here (unjustified overrides, below-floor coverage).
    """

    findings = list(measurements.get("findings", []))
    workspace_path = Path(measurements["workspace"])
    floor_overrides = _load_floor_overrides(workspace_path)
    effective_floors = _effective_floors(floor_overrides, findings)

    families = measurements.get("families", {})
    for family_class in SOURCE_FAMILY_CLASSES:
        floor = effective_floors[family_class]
        family_report = families.get(family_class, {})

        title_floor = floor.get("title_floor")
        if title_floor is not None:
            measured = family_report.get("title_coverage_pct", 0.0) / 100.0
            if measured < title_floor:
                findings.append(
                    _finding(
                        "coverage_below_floor",
                        f"{family_class}.title measured {measured:.4f} below floor {title_floor}",
                        family=family_class,
                        metric="title",
                        measured=measured,
                        floor=title_floor,
                        justification=floor.get("justification"),
                    )
                )

        authors_floor = floor.get("authors_floor")
        if authors_floor is not None:
            measured = family_report.get("authors_coverage_pct", 0.0) / 100.0
            if measured < authors_floor:
                findings.append(
                    _finding(
                        "coverage_below_floor",
                        f"{family_class}.authors measured {measured:.4f} below floor {authors_floor}",
                        family=family_class,
                        metric="authors",
                        measured=measured,
                        floor=authors_floor,
                        justification=floor.get("justification"),
                    )
                )

    measurements["effective_floors"] = effective_floors
    return findings


# ---------------------------------------------------------------------------
# Receipt emission
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measured_input_paths(workspace_path: Path, run_id: str | None) -> list[Path]:
    candidates = [Path("storage/corpus/corpus_inventory.json")]
    if run_id:
        candidates.append(Path("storage/vector-store") / run_id / "stores/papers/chunks.jsonl")
    ledgers_dir = workspace_path / "analysis" / "topic_reference_ledgers"
    if ledgers_dir.is_dir():
        for path in sorted(ledgers_dir.glob("*.json")):
            candidates.append(path.relative_to(workspace_path))
    decks_dir = workspace_path / "decks"
    if decks_dir.is_dir():
        for path in sorted(decks_dir.glob("*/speaker_author_work_ledger.json")):
            candidates.append(path.relative_to(workspace_path))
    return candidates


def _input_digest_map(workspace_path: Path, run_id: str | None) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for relative_path in _measured_input_paths(workspace_path, run_id):
        full_path = workspace_path / relative_path
        if full_path.is_file():
            inputs[relative_path.as_posix()] = _sha256_file(full_path)
    return inputs


def _resolve_receipt_destination(workspace_path: Path, given: str | Path) -> Path:
    """Resolve ``--write-receipt``'s argument to a concrete file path.

    A path with a ``.json`` suffix is used as-is (relative to the workspace
    when not absolute); anything else is treated as a directory and the
    default receipt filename is appended.
    """

    candidate = Path(given)
    if not candidate.is_absolute():
        candidate = workspace_path / candidate
    if candidate.suffix != ".json":
        candidate = candidate / RECEIPT_DEFAULT_FILENAME
    return candidate


def write_coverage_receipt(
    path: str | Path,
    *,
    measurements: Mapping[str, Any],
    floors: Mapping[str, Any],
    findings: list[dict[str, Any]],
    workspace: str | Path,
    run_id: str | None,
    status: str,
) -> dict[str, Any]:
    """Write an atomic, digest-bound, leak-free coverage receipt.

    Refuses (raises ``ValueError``, no write) a ``path`` that resolves outside
    ``workspace``. Never includes an author name, a paper title, or any other
    corpus free text -- counts, ratios, codes, paths, and digests only.
    """

    workspace_path = Path(workspace).resolve()
    receipt_path = Path(path)
    if not receipt_path.is_absolute():
        receipt_path = workspace_path / receipt_path
    resolved_receipt_path = receipt_path.resolve()
    if resolved_receipt_path != workspace_path and workspace_path not in resolved_receipt_path.parents:
        raise ValueError(f"receipt path {receipt_path} escapes workspace {workspace_path}")

    receipt = {
        "receipt_type": RECEIPT_TYPE,
        "status": status,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workspace": str(workspace_path),
        "run_id": run_id,
        "secrets_copied": False,
        "families": measurements.get("families", {}),
        "floors": floors,
        "findings": findings,
        "inputs": _input_digest_map(workspace_path, run_id),
    }
    _atomic_write_json(resolved_receipt_path, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------

_FAMILY_LABELS = {
    SOURCE_FAMILY_CLASS_MAIN_PAPER: "papers_main (main-conference papers)",
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER: "papers_workshop (workshop papers)",
    SOURCE_FAMILY_CLASS_ARTIFACT: "artifacts (everything else, incl. video-caption transcripts)",
}


def _print_human_report(
    measurements: Mapping[str, Any],
    findings: list[dict[str, Any]],
    status: str,
) -> None:
    floors = measurements.get("effective_floors", {})
    families = measurements.get("families", {})

    print(f"Corpus source metadata coverage audit -- {status.upper()}")
    if measurements.get("not_measured"):
        print(f"  scope: {measurements.get('scope')}; NOT MEASURED: {', '.join(measurements['not_measured'])}")
    print(f"workspace: {measurements.get('workspace')}   run_id: {measurements.get('run_id')}")
    print()

    for family_class in SOURCE_FAMILY_CLASSES:
        report = families.get(family_class, {})
        floor = floors.get(family_class, {})
        print(f"--- {_FAMILY_LABELS[family_class]} ---")
        print(f"  total records seen: {report.get('total', 0)}")
        print(
            f"  title coverage:   {report.get('title_present', 0)}/{report.get('total', 0)} "
            f"({report.get('title_coverage_pct', 0.0)}%)"
            + (f"  floor={floor.get('title_floor')}" if floor.get("title_floor") is not None else "  (not gated)")
        )
        print(
            f"  authors coverage: {report.get('authors_present', 0)}/{report.get('total', 0)} "
            f"({report.get('authors_coverage_pct', 0.0)}%)"
            + (
                f"  floor={floor.get('authors_floor')}"
                if floor.get("authors_floor") is not None
                else f"  (not gated -- {report.get('authors_classification', '')})"
            )
        )
        print(f"  title provenance: {report.get('title_provenance', {})}")
        print(f"  authors provenance: {report.get('authors_provenance', {})}")
        for key in ("title_floor_override", "authors_floor_override"):
            override = floor.get(key)
            if override:
                print(
                    f"  ACCEPTED OVERRIDE ({key}): {override['previous_value']} -> "
                    f"{override['new_value']}  justification: {override['justification']}"
                )
        print(f"  justification: {floor.get('justification', '')}")
        print()

    if findings:
        print(f"FAIL: {len(findings)} finding(s):")
        for finding in findings:
            print(f"  [{finding['code']}] {finding['message']}")
    else:
        print("PASS: every gated family meets its coverage floor.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, help="path to the conference workspace, e.g. conferences/cvpr2026")
    parser.add_argument("--run-id", default=None, help="vector-store run id; defaults to the most recent one present")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all",
        help="all (default): inventory, papers vector store and ledgers; inventory: the corpus inventory only, the rest reported as not measured",
    )
    parser.add_argument(
        "--write-receipt",
        default=None,
        metavar="PATH",
        help="write the coverage receipt to PATH (a directory or a .json file), relative to --workspace",
    )
    args = parser.parse_args(argv)

    # Resolved to absolute BEFORE any receipt-path joining (Phase 10.1-10 bug
    # fix): ``_resolve_receipt_destination`` prefixes a relative
    # ``--write-receipt`` value with ``workspace_path`` once, and
    # ``write_coverage_receipt`` prefixes AGAIN with its own ``workspace``
    # argument whenever the path it receives is still relative. When
    # ``--workspace`` itself is a relative path (the documented CLI usage,
    # e.g. ``--workspace conferences/cvpr2026``), the first join's result is
    # STILL relative, so the second join silently doubled the workspace
    # segment (``conferences/cvpr2026/conferences/cvpr2026/...``) instead of
    # raising. Every existing unit test passes an already-absolute ``tmp_path``
    # workspace, so this never tripped a test; it only manifests against a
    # real, relative-path workspace invocation. Resolving here once makes
    # every downstream join a no-op the second time, for both call sites.
    workspace_path = Path(args.workspace).resolve()
    measurements = measure_workspace_coverage(workspace_path, run_id=args.run_id, scope=args.scope)
    findings = evaluate_floors(measurements)
    status = "blocked" if findings else "succeeded"

    receipt_written_path: Path | None = None
    if args.write_receipt:
        try:
            receipt_written_path = _resolve_receipt_destination(workspace_path, args.write_receipt)
            write_coverage_receipt(
                receipt_written_path,
                measurements=measurements,
                floors=measurements.get("effective_floors", {}),
                findings=findings,
                workspace=workspace_path,
                run_id=measurements.get("run_id"),
                status=status,
            )
        except ValueError as exc:
            payload = {"status": "fail", "error": str(exc)}
            print(json.dumps(payload, indent=2) if args.json else f"FAIL: {exc}")
            return 1

    if args.json:
        print(
            json.dumps(
                {
                    "status": status,
                    "workspace": measurements.get("workspace"),
                    "run_id": measurements.get("run_id"),
                    "scope": measurements.get("scope"),
                    "not_measured": measurements.get("not_measured", []),
                    "families": measurements.get("families", {}),
                    "floors": measurements.get("effective_floors", {}),
                    "findings": findings,
                    "receipt_path": str(receipt_written_path) if receipt_written_path else None,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        _print_human_report(measurements, findings, status)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
