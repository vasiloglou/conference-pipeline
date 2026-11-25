"""Leakage audit of a workspace's analysis products (v6.1 Phase 2, spec 6.3; plan R16).

Every analysis product is scanned for the works it cites: corpus artifact ids, LLM
full-read unit ids (``paper_abstract:<forum>``), OpenReview forum links and ``"forum"``
fields. Each citation resolves to a work family through the roster; one that does not
is counted as unresolved, never dropped. A product is ``clean`` when none of its cited
families is in the development or sealed-test pool, ``contaminated`` otherwise, and
``unreadable`` when it cannot be read (which the CLI treats as a problem).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

LEAKAGE_SCHEMA = "ontology-leakage/1"
# Analysis product paths, relative to the workspace's analysis directory.
PRODUCT_GLOBS = (
    "[0-9][0-9]_*.md",
    "topic_reference_ledgers/**/*",
    "narrative_source_packs/**/*",
    "exhaustive_summaries/**/*",
    "deep_dive_topics/**/*",
    "trend_report.md",
    "executive_summary.md",
    "topic_strategy.json",
    "analysis_citation_manifest.json",
    "provenance/**/*",
)
CITATION_MANIFEST = "analysis_citation_manifest.json"
# r1 F4: every token of a citation form is collected, well-formed or not; a malformed one
# resolves to no family and is counted as unresolved.
_PREFIXED = re.compile(r"\b(workshop_paper_text|paper_text|speech_transcript|paper_abstract):([A-Za-z0-9_\-]+)")
_HEX_ID = re.compile(r"^[0-9a-f]{12,64}$")
_FORUM_LINK = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_\-]+)")
_FORUM_FIELD = re.compile(r'"forum"\s*:\s*"([^"]+)"')


def product_paths(analysis_dir: Path, extra_globs: tuple[str, ...] = ()) -> list[Path]:
    found: set[Path] = set()
    for pattern in PRODUCT_GLOBS + tuple(extra_globs):
        found.update(p for p in analysis_dir.glob(pattern) if p.is_file())
    return sorted(found)


def citations(text: str) -> list[str]:
    """Every citation token in ``text``, normalized to artifact ids or ``openreview:<forum>``."""
    artifacts: list[str] = []
    forums: list[str] = []
    for family, value in _PREFIXED.findall(text):
        if family == "paper_abstract":
            forums.append(f"openreview:{value}")
        else:
            artifacts.append(f"{family}:{value}")
    for pattern in (_FORUM_LINK, _FORUM_FIELD):
        forums.extend(f"openreview:{forum}" for forum in pattern.findall(text))
    return artifacts + forums


def resolve(tokens: Iterable[str], roster_index: Mapping[str, str]) -> tuple[frozenset[str], list[str]]:
    families: set[str] = set()
    unresolved: set[str] = set()
    for token in tokens:
        if token in roster_index:
            families.add(roster_index[token])
        else:
            unresolved.add(token)
    return frozenset(families), sorted(unresolved)


def cited_families(product: Path, roster_index: Mapping[str, str]) -> tuple[frozenset[str], list[str]]:
    """Families ``product`` cites and its unresolved citations; :class:`OSError` if unreadable."""
    text = product.read_bytes().decode("utf-8", errors="replace")
    return resolve(citations(text), roster_index)


def manifest_citations(manifest: Any) -> dict[str, list[str]]:
    """``{product path relative to analysis/: [citation tokens]}`` from the citation manifest.

    r1 F3: each topic entry declares the sources of the products it names; list fields whose
    name ends in ``paper_ids`` hold bare OpenReview forum ids.
    """
    out: dict[str, list[str]] = {}
    topics = manifest.get("topics") if isinstance(manifest, Mapping) else None
    for topic in topics if isinstance(topics, list) else []:
        if not isinstance(topic, Mapping):
            continue
        tokens: list[str] = []
        for key, value in topic.items():
            if not isinstance(value, list):
                continue
            strings = [v for v in value if isinstance(v, str) and v.strip()]
            if str(key).endswith("paper_ids"):
                tokens += [f"openreview:{v.strip()}" for v in strings]
            else:
                tokens += [t for v in strings for t in citations(v)]
        products = {str(topic[k]).removeprefix("analysis/") for k in ("deep_analysis_path", "markdown_path")
                    if isinstance(topic.get(k), str) and topic[k].strip()}
        for product in products or {CITATION_MANIFEST}:
            out.setdefault(product, []).extend(tokens)
    return out


def scan_products(analysis_dir: Path, roster_index: Mapping[str, str], *,
                  extra_globs: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Scan the analysis products (and any ``extra_globs``, e.g. gold outputs, Phase 3 T13)."""
    rows: list[dict[str, Any]] = []
    for product in product_paths(analysis_dir, extra_globs):
        relative = product.relative_to(analysis_dir).as_posix()
        try:
            raw = product.read_bytes()
        except OSError as exc:
            rows.append({"product": relative, "status": "unreadable", "error": type(exc).__name__,
                         "cited_families": [], "unresolved": []})
            continue
        families, unresolved = resolve(citations(raw.decode("utf-8", errors="replace")), roster_index)
        rows.append({"product": relative, "sha256": hashlib.sha256(raw).hexdigest(), "status": "scanned",
                     "cited_families": sorted(families), "unresolved": unresolved})
    manifest_row = next((r for r in rows if r["product"] == CITATION_MANIFEST and r["status"] == "scanned"), None)
    if manifest_row is not None:
        try:
            manifest = json.loads((analysis_dir / CITATION_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest = None
        by_product = {r["product"]: r for r in rows}
        for product, tokens in sorted(manifest_citations(manifest).items()):
            families, unresolved = resolve(tokens, roster_index)
            row = by_product.get(product)
            if row is None:  # a product the manifest names but no file holds
                row = {"product": product, "status": "manifest-only", "cited_families": [], "unresolved": []}
                rows.append(row)
                by_product[product] = row
            if row["status"] == "unreadable":
                continue
            row["cited_families"] = sorted(set(row["cited_families"]) | families)
            row["unresolved"] = sorted(set(row["unresolved"]) | set(unresolved))
            row["manifest_sources"] = True
        rows.sort(key=lambda r: r["product"])
    return rows


def cited_set(rows: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(family for row in rows for family in row.get("cited_families") or [])


def leakage_report(rows: list[dict[str, Any]], splits: Mapping[str, Any]) -> dict[str, Any]:
    """Classify each scanned product against the splits' pools."""
    pools = splits.get("splits") or {}
    groups = splits.get("groups") or {}
    held_out = {family for name in ("dev", "sealed_test") for g in pools.get(name) or [] for family in groups.get(g, [])}
    products = []
    for row in rows:
        if row["status"] == "unreadable":
            products.append({**row, "contaminated_by": []})
            continue
        leaked = sorted(set(row["cited_families"]) & held_out)  # manifest-only rows are classified too
        products.append({**row, "status": "contaminated" if leaked else "clean", "contaminated_by": leaked})
    return {
        "schema": LEAKAGE_SCHEMA,
        "roster_sha256": splits.get("roster_sha256"),
        "products": products,
        "counts": {
            "products": len(products),
            "clean": sum(p["status"] == "clean" for p in products),
            "contaminated": sum(p["status"] == "contaminated" for p in products),
            "unreadable": sum(p["status"] == "unreadable" for p in products),
            "cited_families": len(cited_set(rows)),
            "unresolved_citations": len({u for row in rows for u in row.get("unresolved") or []}),
        },
    }
