"""Two-root assurance fixtures for v6.1 Phase 3 (spec 6.7, T17-T20).

The run layout is GND 0.4.0's own interop example (``gnd.examples.literature.interop run``):
an extraction root and a checking root over one shared vocabulary, two sources whose
original extraction is rebound into the checking root (``literature-rebind/1``), and a
composition with a resolution file. The fixture adds what the adapter needs around it:
a split corpus (documents and passages) and an answer key in the checking root's terms,
with the fidelity gold and renderer manifests GND's interop example ships per source.

Narrated expectations (approved with the spec gate, corpus-first for the exit contract):

- ``reference``: the faithful run. Every active stage passes; exit 0. Derivation is
  not applicable (the interop checking root declares no derived fact type).
- ``drop``: two facts removed from survey's original extraction, every digest that binds
  them recomputed (as GND's M2 does). Semantic support recall 4/6 is under the ICML floor
  4/5 (one drop, 5/6, is above it); exit 1, stage ``semantic-support`` fails.
- ``stale``: survey's rebound population rewritten after its sidecars were made (GND's M3),
  the rebind record re-bound to the new bytes (otherwise the adapter refuses at 3, review
  r1 F1). Freshness fails; exit 1.
- ``false-merge``: the answer key says P.I.N.N.s and PINNs are different methods; the run's
  resolution merged them (GND's M7). Identity precision fails; exit 1.
- ``shifted-span``: one rebound anchor's start moved by one character, its excerpt digest
  not recomputed (GND's M6), the rebind record re-bound. Source binding fails; exit 1.
- ``crossed-roots``: survey's rebind record names the extraction root as its checking root.
  Refused before any route; exit 3, reason names ``survey``.
- ``stale-rebind``: replication's rebind record names an extraction-population digest that
  is not the file's. Refused; exit 3, reason names ``replication``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily

SOURCES = ("replication", "survey")
DROPPED = ("paper_submitted_on", "paper_published_on")
EXPECTED = {
    "reference": (0, []),
    "drop": (1, ["semantic-support"]),
    "stale": (1, ["freshness"]),
    "false-merge": (1, ["identity-decisions"]),
    "shifted-span": (1, ["source-binding"]),
    "crossed-roots": (3, "survey"),
    "stale-rebind": (3, "replication"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def interop_run(out: Path) -> Path:
    """Build GND's packaged two-root interop example into ``out`` (real GND, offline)."""
    result = gnd_tool.run_gnd_example("literature.interop", ["run", "--out", str(out)],
                                      family=CommandFamily.OTHER, cwd=out.parent)
    assert result.exit_code == 0, result.stderr
    return out


def split_corpus(root: Path, run: Path) -> Path:
    """An ``ontology-split-corpus/1`` holding the two interop documents, one passage each."""
    (root / "documents").mkdir(parents=True)
    documents, passages = [], []
    for source in SOURCES:
        raw = (run / "sources" / source / "document.txt").read_bytes()
        (root / "documents" / f"{source}.md").write_bytes(raw)
        text = raw.decode("utf-8")
        documents.append({"id": source, "path": f"documents/{source}.md", "sha256": _sha(raw),
                          "domain": "area-x", "work": {"identifier": f"example:{source}", "title": source}})
        passages.append({"id": f"{source}.p1", "document": source, "span": {"start": 0, "end": len(text)}})
    corpus = {"schema": "ontology-split-corpus/1", "split": "dev", "documents": documents, "passages": passages}
    (root / "corpus.json").write_bytes(_dump(corpus))
    return root


def answers(root: Path, run: Path, checking_corpus_sha256: str, *, merge_label: str = "same") -> Path:
    """The answer key in the checking root's terms, plus fidelity gold and renderer manifests."""
    (root / "gold").mkdir(parents=True)
    (root / "render").mkdir(parents=True)
    assertions, anchored, n = [], [], 0
    for source in SOURCES:
        folder = run / "sources" / source
        shutil.copyfile(folder / "source.population.json", root / "gold" / f"{source}.population.json")
        shutil.copyfile(folder / "document.manifest.json", root / "render" / f"{source}.manifest.json")
        gold = json.loads((folder / "source.population.json").read_text())
        for fact_type, rows in sorted(gold["facts"].items()):
            for roles in rows:
                n += 1
                assertions.append({"id": f"a{n:02d}", "passage": f"{source}.p1", "support": "supported",
                                   "fact": {"fact_type": fact_type, "roles": roles}, "label_disagreement": None,
                                   "participants": sorted(str(v) for v in roles.values()), "predicate": fact_type})
        sidecar = json.loads((folder / "population.json.anchors.json").read_text())
        for i, anchor in enumerate(sidecar["anchors"]):
            anchored.append({"id": f"f-{source}-{i}", "source": source, "fact": anchor["fact_key"],
                             "role": anchor["role"], "span": anchor["span"]})
    key = {
        "schema": "literature-answer-key/1", "corpus_sha256": checking_corpus_sha256,
        "labeller": {"kind": "fixture-author", "independent": False, "note": "v6.1 Phase 3 fixture"},
        "assertions": assertions, "anchored_facts": anchored,
        # Each source checked alone, before resolution: survey's "P.I.N.N.s" is not in the
        # catalog (exit 1); replication's anchored roles are unchecked without its document
        # (exit 4). GND's own answer key records per-source exits the same way.
        "populations": [{"source": "replication", "expected_check_exit": 4},
                        {"source": "survey", "expected_check_exit": 1}],
        "identity_pairs": [{"id": "i01", "label": merge_label, "passages": ["survey.p1", "replication.p1"],
                            "left": {"source": "survey", "object_type": "method", "value": "P.I.N.N.s"},
                            "right": {"source": "replication", "object_type": "method", "value": "PINNs"}}],
        "name_pairs": [], "chains": [], "reuse_cases": [],
    }
    (root / "answer-key.json").write_bytes(_dump(key))
    return root / "answer-key.json"


# --- planted faults -------------------------------------------------------------------


def _rebind(run: Path, source: str, **changes: str) -> None:
    path = run / "sources" / source / "rebind.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_bytes(_dump(record))


def plant_drop(run: Path) -> None:
    """Drop survey's two date facts from its original extraction; recompute its digests."""
    folder = run / "sources" / "survey"
    population = json.loads((folder / "extracted.population.json").read_text())
    for fact_type in DROPPED:
        population["facts"].pop(fact_type)
    raw = json.dumps(population, sort_keys=True, separators=(",", ":")).encode()
    (folder / "extracted.population.json").write_bytes(raw)
    for name in ("extracted.population.json.anchors.json", "extracted.population.extraction.json"):
        sidecar = json.loads((folder / name).read_text())
        sidecar["population_sha256"] = _sha(raw)
        if "anchors" in sidecar:
            sidecar["anchors"] = [a for a in sidecar["anchors"] if a["fact_key"]["fact_type"] not in DROPPED]
        (folder / name).write_bytes(_dump(sidecar))
    _rebind(run, "survey", extraction_population_sha256=_sha(raw),
            extraction_anchors_sha256=_sha((folder / "extracted.population.json.anchors.json").read_bytes()),
            extraction_sidecar_sha256=_sha((folder / "extracted.population.extraction.json").read_bytes()))


def _rebind_current(run: Path, source: str) -> None:
    """Re-bind a source to its current files, as a producer that re-rebinds would (review r1 F1)."""
    folder = run / "sources" / source
    _rebind(run, source, rebound_population_sha256=_sha((folder / "population.json").read_bytes()),
            rebound_anchors_sha256=_sha((folder / "population.json.anchors.json").read_bytes()))


def plant_stale(run: Path) -> None:
    path = run / "sources" / "survey" / "population.json"
    path.write_bytes(json.dumps(json.loads(path.read_bytes())).encode() + b"\n")
    _rebind_current(run, "survey")


def plant_shifted_span(run: Path) -> None:
    path = run / "sources" / "survey" / "population.json.anchors.json"
    sidecar = json.loads(path.read_text())
    anchor = next(a for a in sidecar["anchors"] if a["role"] == "method")
    anchor["span"] = {"start": anchor["span"]["start"] + 1, "end": anchor["span"]["end"]}
    path.write_bytes(_dump(sidecar))
    _rebind_current(run, "survey")


def plant_crossed_roots(run: Path) -> None:
    _rebind(run, "survey", checking_root_sha256=_sha((run / "extraction.orm.json").read_bytes()))


def plant_stale_rebind(run: Path) -> None:
    _rebind(run, "replication", extraction_population_sha256="0" * 64)


PLANTS = {"drop": plant_drop, "stale": plant_stale, "shifted-span": plant_shifted_span,
          "crossed-roots": plant_crossed_roots, "stale-rebind": plant_stale_rebind}
