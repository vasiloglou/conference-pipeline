"""Runs inside GND 0.4.0's interpreter for ``ontology.extraction`` (v6.1 Phase 4, lane X).

Two commands, each reading a JSON config path and writing one JSON object to stdout:

- ``anchors``: anchors v2 for a source's rows, located with GND's own lexical matcher
  (``gnd.anchors.lexical_ranges``: NFC, case-folded, word-bounded), only an
  occurrence inside the row's evidence span (none there: the row is unanchored); the sidecar binds GND's vocabulary digest.
- ``rebind``: the host form of ``gnd.examples.literature.interop._rebind`` (0.4.0
  :180-222). Same preconditions (both roots bind the same vocabulary pins and catalogs,
  distinct model ids, the extraction root still matches its artifact index, ``gnd verify``
  passes, no input changes during the run) and the same ``literature-rebind/1`` fields; the
  files are written to the host's run directory. A refusal is ``{"refused": reason}``.

GND internals used (pinned to 0.4.0 by ``gnd_tool.require_gnd``): ``model._included`` and
``model._references`` (as the example does) and ``gnd.extract.service._verify_artifact_index``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

from gnd.anchors import (
    bind_anchors,
    canonical_document_text,
    lexical_ranges,
    parse_anchor_sidecar,
    vocabulary_digest,
)
from gnd.artifacts import canonical_json_bytes, sha256_hex
from gnd.extract.bundle import bundle_stem
from gnd.extract.service import _verify_artifact_index
from gnd.model import load_model
from gnd.population import load_population_bytes

EXTRACTION, CHECKING, ARTIFACTS, SOURCES = "extraction.orm.json", "checking.orm.json", "artifacts", "sources"
EXTRACTED, SIDECAR = "extracted.population.json", "extracted.population.extraction.json"
ORIGINAL_ANCHORS, REBOUND, REBOUND_ANCHORS = "extracted.population.json.anchors.json", "population.json", \
    "population.json.anchors.json"


class RefusedError(ValueError):
    pass


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RefusedError(reason)


def anchors(config: dict) -> dict:
    model = load_model(Path(config["model"]))
    document = Path(config["document"]).read_bytes()
    source = canonical_document_text(document)
    original = document.decode("utf-8")

    def nfc_offset(at: int) -> int:  # review r2 F3: a reading span is in the raw text; GND matches in NFC
        return len(unicodedata.normalize("NFC", original[:at]))
    anchored, unanchored = [], []
    # v6.1 Phase 5 (spec lane E): with "textual_roles", only those roles of each fact type are
    # located; a fact type it does not name is kept with no anchor (host ids, enumerated values).
    textual = config.get("textual_roles")
    for n, row in enumerate(config["rows"]):
        if textual is not None:
            named = set(textual.get(row["fact_type"]) or ())
            row = {**row, "roles_to_anchor": {k: v for k, v in row["roles"].items() if k in named}}
        else:
            row = {**row, "roles_to_anchor": row["roles"]}
        values = {str(v).casefold() for v in row["roles_to_anchor"].values()}
        found = lexical_ranges(source, values)
        span = row.get("span") or {}
        lo = nfc_offset(span["start"]) if "start" in span else 0
        hi = nfc_offset(span["end"]) if "end" in span else len(source)
        rows = []
        for role, value in sorted(row["roles_to_anchor"].items()):
            ranges = found.get(str(value).casefold()) or []
            if not ranges:
                rows = None
                break
            inside = [i for i, (a, b) in enumerate(ranges) if lo <= a and b <= hi]
            if not inside:  # review F2: an occurrence outside the reviewed evidence is not this fact's
                rows = None
                break
            pick = inside[0]
            a, b = ranges[pick]
            rows.append({"fact_key": {"fact_type": row["fact_type"], "roles": row["roles"]}, "role": role,
                         "span": {"start": a, "end": b}, "occurrence": pick,
                         "excerpt_sha256": sha256_hex(source[a:b].encode("utf-8"))})
        if rows is None:
            unanchored.append(n)
        else:
            anchored.extend(rows)
    return {"vocabulary_digest": vocabulary_digest(model), "anchors": anchored, "unanchored": unanchored}


def _digest(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _roots(run: Path):
    extraction, checking = load_model(run / EXTRACTION), load_model(run / CHECKING)
    pins = [[{"id": e.id, "sha256": e.sha256} for e in m._included] for m in (extraction, checking)]
    _need(bool(pins[0]) and extraction._references is not None and checking._references is not None,
          "both roots must bind vocabulary and catalog inputs")
    catalogs = [[{"id": o.id, "raw_sha256": o.raw_sha256} for o, _, _ in m._references._snapshots]
                for m in (extraction, checking)]
    _need(pins[0] == pins[1] and vocabulary_digest(extraction) == vocabulary_digest(checking),
          "root vocabulary pins differ")
    _need(catalogs[0] == catalogs[1] and extraction._references._owners == checking._references._owners,
          "root catalog bindings differ")
    _need(extraction.model_id != checking.model_id, "the model roots must have distinct identities")
    return extraction, checking, pins[0], catalogs[0]


def rebind(config: dict) -> dict:
    run, source = Path(config["run"]), config["source"]
    extraction, checking, pins, catalogs = _roots(run)
    folder = run / SOURCES / source
    stem = bundle_stem(EXTRACTION)
    index = f"{ARTIFACTS}/{stem}.artifacts.json"
    inputs = {
        "extraction_root_sha256": extraction._authored_sha256, "checking_root_sha256": checking._authored_sha256,
        "artifact_index_sha256": _digest(run / index), "document_sha256": _digest(folder / "document.md"),
        "extraction_population_sha256": _digest(folder / EXTRACTED),
        "extraction_sidecar_sha256": _digest(folder / SIDECAR),
        "extraction_anchors_sha256": _digest(folder / ORIGINAL_ANCHORS),
    }
    bundle = _verify_artifact_index(str(run / EXTRACTION), str(run / ARTIFACTS),
                                    (f"{stem}.projection.schema.json", f"{stem}.applicable.json"))
    _need(bundle.root_sha256 == inputs["extraction_root_sha256"], "the extraction root changed")
    verify = subprocess.run([sys.executable, "-m", "gnd", "verify", EXTRACTION, f"{SOURCES}/{source}/{EXTRACTED}",
                             "--artifacts", ARTIFACTS, "--format", "json"], cwd=run, capture_output=True)
    _need(verify.returncode == 0, f"verify exited {verify.returncode}")
    original, original_anchors = (folder / EXTRACTED).read_bytes(), (folder / ORIGINAL_ANCHORS).read_bytes()
    document = (folder / "document.md").read_bytes()
    snapshots = {"extraction_population_sha256": original, "extraction_anchors_sha256": original_anchors,
                 "document_sha256": document, "extraction_sidecar_sha256": (folder / SIDECAR).read_bytes(),
                 "artifact_index_sha256": (run / index).read_bytes()}
    _need(all(sha256_hex(raw) == inputs[k] for k, raw in snapshots.items()),
          "extraction evidence changed during verification")
    _need((_digest(run / EXTRACTION), _digest(run / CHECKING)) ==
          (inputs["extraction_root_sha256"], inputs["checking_root_sha256"]), "a root changed during rebind")
    population = json.loads(original)
    population["population"].update(model=CHECKING, model_id=checking.model_id)
    rebound = canonical_json_bytes(population)
    anchor_doc = json.loads(original_anchors)
    anchor_doc.update(population_sha256=sha256_hex(rebound), vocabulary_digest=vocabulary_digest(checking))
    anchor_bytes = canonical_json_bytes(anchor_doc)
    validated = load_population_bytes(rebound, source=REBOUND, model=checking)
    bind_anchors(parse_anchor_sidecar(anchor_bytes), checking, rebound, document, validated)
    record = canonical_json_bytes({
        "schema": "literature-rebind/1", "source": source, **inputs, "includes": pins, "references": catalogs,
        "rebound_population_sha256": sha256_hex(rebound), "rebound_anchors_sha256": sha256_hex(anchor_bytes),
    })
    for name, raw in ((REBOUND, rebound), (REBOUND_ANCHORS, anchor_bytes), ("document.txt", document),
                      ("rebind.json", record)):
        (folder / name).write_bytes(raw)
    return {"rebind": json.loads(record)}


def main(argv: list[str]) -> int:
    command, config = argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    try:
        result = {"anchors": anchors, "rebind": rebind}[command](config)
    except RefusedError as exc:
        result = {"refused": str(exc)}
    except Exception as exc:  # noqa: BLE001 - GND's own refusals (anchors, population) become reasons
        result = {"refused": f"{type(exc).__name__}: {exc}"[:400]}
    sys.stdout.write(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
