"""Extraction bundle, per-source populations and rebind into the checking root (v6.1 Phase 4, lane X).

The run directory uses GND's interop layout, so GND's own commands and the Phase 3 assurance
adapter read it unchanged: ``extraction.orm.json`` and ``checking.orm.json`` (roots over one
shared vocabulary), ``artifacts/`` (``gnd extract``), and per source
``sources/<source>/``: ``document.md``, ``extracted.population.json`` with its
``.extraction.json`` sidecar and anchors v2, then, after :func:`rebind`, ``population.json``,
its anchors, ``document.txt`` and ``rebind.json`` (``literature-rebind/1``).

The host writes rows; GND verifies them (R6): ``gnd verify --artifacts`` and
``gnd check --anchors --document``. Exits are reported with ``gnd_tool.exit_meaning``, never
collapsed. Anchors and the rebind run in GND's interpreter (``_extraction_gnd.py``) so a span is
located by GND's own matcher and the rebind uses GND's own binding checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily, GndResult

EXTRACTION, CHECKING, ARTIFACTS, SOURCES = "extraction.orm.json", "checking.orm.json", "artifacts", "sources"
EXTRACTED, SIDECAR = "extracted.population.json", "extracted.population.extraction.json"
ANCHORS = "extracted.population.json.anchors.json"
_HELPER = Path(__file__).with_name("_extraction_gnd.py")


class ExtractionError(ValueError):
    """A GND step refused or failed; the message is the reason (adapter exit 3)."""


def _dump(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _helper(run: Path, command: str, config: Mapping[str, Any]) -> dict[str, Any]:
    path = run / f".{command}.config.json"
    path.write_bytes(_dump(dict(config)))
    try:
        result = gnd_tool.run_gnd_python(["-I", str(_HELPER), command, str(path)], family=CommandFamily.OTHER,
                                         cwd=run, timeout_s=300)
    finally:
        path.unlink(missing_ok=True)
    payload = gnd_tool.parse_json_stdout(result)
    if payload is None:
        raise ExtractionError(f"{command}:{result.meaning}")
    if "refused" in payload:
        raise ExtractionError(f"{command}:refused:{payload['refused']}")
    return payload


def bundle(run: Path) -> GndResult:
    """``gnd extract extraction.orm.json --out artifacts`` (offline); refused unless it exits 0."""
    result = gnd_tool.run_gnd(["extract", EXTRACTION, "--out", ARTIFACTS], family=CommandFamily.OTHER, cwd=run)
    if result.exit_code != 0:
        raise ExtractionError(f"extract:{result.meaning}")
    return result


def rows_from_reading(facts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Population rows from accepted reading facts that name their roles; and how many did not."""
    rows, skipped = [], 0
    for fact in facts:
        roles = fact.get("roles")
        if isinstance(roles, Mapping) and roles and all(isinstance(v, str) and v for v in roles.values()):
            rows.append({"fact_type": fact["fact_type"], "roles": dict(roles), "span": fact.get("span")})
        else:
            skipped += 1
    return rows, skipped


def write_source(run: Path, source: str, document: str, rows: Sequence[Mapping[str, Any]], *,
                 executing_model: str, textual_roles: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Write one source's extraction population, sidecar and anchors; rows whose role values the
    document does not contain are left out and counted (``unanchored``). With ``textual_roles``
    (v6.1 Phase 5, ``assertions.TEXTUAL_ROLES``) only those roles are located and anchored; the
    other roles are host ids or enumerated values and carry no anchor."""
    folder = run / SOURCES / source
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "document.md").write_bytes(document.encode("utf-8"))
    located = _helper(run, "anchors", {"model": str(run / EXTRACTION), "document": str(folder / "document.md"),
                                       "rows": [dict(r) for r in rows],
                                       **({"textual_roles": {k: list(v) for k, v in textual_roles.items()}}
                                          if textual_roles is not None else {})})
    missing = set(located["unanchored"])
    facts: dict[str, list[dict[str, str]]] = {}
    for n, row in enumerate(rows):
        if n not in missing and row["roles"] not in facts.setdefault(row["fact_type"], []):
            facts[row["fact_type"]].append(dict(row["roles"]))
    facts = {k: v for k, v in sorted(facts.items()) if v}
    index = json.loads((run / ARTIFACTS / "extraction.orm.artifacts.json").read_bytes())
    population = _dump({"population": index["header_template"], "facts": facts})
    (folder / EXTRACTED).write_bytes(population)
    document_sha = _sha(document.encode("utf-8"))
    (folder / SIDECAR).write_bytes(_dump({
        "schema_version": 1, "model": EXTRACTION, "model_sha256": index["model"]["sha256"],
        "population": EXTRACTED, "population_sha256": _sha(population),
        "document": "document.md", "document_sha256": document_sha,
        "projection_sha256": index["artifacts"]["extraction.orm.projection.schema.json"]["sha256"],
        "skill": "gnd-extract", "skill_version": "1", "executing_model": executing_model,
    }))
    keep = [a for a in located["anchors"] if a["fact_key"]["roles"] in facts.get(a["fact_key"]["fact_type"], [])]
    (folder / ANCHORS).write_bytes(_dump({"schema_version": 2, "population_sha256": _sha(population),
                                          "document_sha256": document_sha,
                                          "vocabulary_digest": located["vocabulary_digest"], "anchors": keep}))
    return {"source": source, "rows": sum(len(v) for v in facts.values()), "unanchored": len(missing),
            "population_sha256": _sha(population)}


def verify(run: Path, source: str) -> GndResult:
    """``gnd verify extraction.orm.json <population> --artifacts artifacts`` (exit reported, not collapsed)."""
    return gnd_tool.run_gnd(["verify", EXTRACTION, f"{SOURCES}/{source}/{EXTRACTED}", "--artifacts", ARTIFACTS,
                             "--format", "json"], family=CommandFamily.CHECK, cwd=run)


def anchor_model(run: Path, root: str = EXTRACTION) -> str:
    """The root without its reference populations: GND 0.4.0 refuses ``--anchors`` on a model that has
    them (Phase 3 finding); the anchors bind the vocabulary digest, which excludes them."""
    model = json.loads((run / root).read_bytes())
    if not model.get("reference_populations"):
        return root
    name = root.replace(".orm.json", ".anchors.orm.json")
    (run / name).write_bytes(_dump({k: v for k, v in model.items() if k != "reference_populations"}))
    return name


def check_anchors(run: Path, source: str, *, rebound: bool = False) -> GndResult:
    """``gnd check <root> <population> --anchors <sidecar> --document <document>``."""
    folder = f"{SOURCES}/{source}"
    population, document = ("population.json", "document.txt") if rebound else (EXTRACTED, "document.md")
    model = anchor_model(run, CHECKING if rebound else EXTRACTION)
    return gnd_tool.run_gnd(["check", model, f"{folder}/{population}", "--anchors", f"{folder}/{population}.anchors.json",
                             "--document", f"{folder}/{document}", "--format", "json"],
                            family=CommandFamily.CHECK, cwd=run)


def rebind(run: Path, source: str) -> dict[str, Any]:
    """Rebind one source into the checking root and write ``rebind.json`` (``literature-rebind/1``)."""
    return _helper(run, "rebind", {"run": str(run), "source": source})["rebind"]
