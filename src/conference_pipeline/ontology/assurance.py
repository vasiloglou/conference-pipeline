"""Two-root assurance adapter around GND 0.4.0's staged assurance report (v6.1 Phase 3, 6.7).

GND's packaged scorer (``gnd.examples.literature.assurance_report``) binds one model for
every route and reads fidelity inputs from the same population it checks. This pipeline
keeps the original extraction (extraction root) and a rebound checking population
(checking root, ``literature-rebind/1``). The adapter:

1. checks, per source, that the rebind record binds the run's two roots, its original
   extraction and its document (a crossed or stale binding is refused, exit 3, the
   source named) -- digest bookkeeping only; every ORM check stays GND's (R6);
2. builds two views GND's scorer accepts: a *checking* view (rebound populations, their
   anchors, the composition) and an *extraction* view (original extraction and its
   sidecars, for the fidelity route);
3. runs ``_assurance_gnd.py`` in GND's interpreter, which scores both views with the
   given floors and merges them into one ``literature-assurance-report/1``-shaped report.

The run layout (``ontology-two-root-run/1``) is the one GND's interop example writes:
``extraction.orm.json``, ``checking.orm.json``, their includes and reference files,
``artifacts/`` (extraction artifacts), ``sources/manifest.json`` with ``resolution.json``,
``sources/<id>/{document.txt, extracted.population.json, extracted.population.extraction.json,
extracted.population.json.anchors.json, population.json, population.json.anchors.json,
rebind.json}``, ``composed.population.json`` and ``composition.json``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily

REPORT_SCHEMA = "literature-assurance-report/1"
FLOORS_SCHEMA = "ontology-floors/1"
SPLIT_CORPUS_SCHEMA = "ontology-split-corpus/1"
# The metrics GND 0.4.0's scorer has a floor for, per stage (assurance_report.FLOORS).
PACKAGED_FLOOR_SHAPE: Mapping[str, frozenset[str]] = {
    "model-population-validity": frozenset({"agreement"}),
    "source-binding": frozenset({"precision", "recall"}),
    "semantic-support": frozenset({"precision", "recall", "abstention_max"}),
    "identity-decisions": frozenset({"precision", "recall", "abstention_max"}),
    "derivation": frozenset({"precision", "recall", "abstention_max"}),
    "freshness": frozenset({"precision", "recall"}),
}
EXTRACTION_ARTIFACTS = {"applicable": "extraction.orm.applicable.json",
                        "projection": "extraction.orm.projection.schema.json"}
_SCRIPT = Path(__file__).with_name("_assurance_gnd.py")
# The per-source files a literature-rebind/1 record binds besides the roots and document.
REBIND_FILES = {
    "rebound_population_sha256": "population.json",
    "rebound_anchors_sha256": "population.json.anchors.json",
    "extraction_anchors_sha256": "extracted.population.json.anchors.json",
    "extraction_sidecar_sha256": "extracted.population.extraction.json",
}
# The checking root without its reference populations, for the anchors route only.
ANCHOR_MODEL = "checking.anchors.orm.json"
SPLIT_VALUES = {"dev": "dev", "sealed_test": "heldout", "heldout": "heldout"}


class AssuranceInputError(ValueError):
    """A floors file, split corpus or run layout is malformed."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssuranceInputError(f"unreadable: {path.name}: {exc}") from exc


# --------------------------------------------------------------------------- floors


def load_floors(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Floors as exact fractions; every packaged stage and metric, nothing else."""
    raw = _read(path)
    if not isinstance(raw, Mapping) or raw.get("schema") != FLOORS_SCHEMA or not isinstance(raw.get("stages"), Mapping):
        raise AssuranceInputError(f"{path.name} is not {FLOORS_SCHEMA}")
    stages = raw["stages"]
    if set(stages) != set(PACKAGED_FLOOR_SHAPE):
        missing = sorted(set(PACKAGED_FLOOR_SHAPE) ^ set(stages))
        raise AssuranceInputError(f"floors must name exactly the scored stages; differs at {', '.join(missing)}")
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for stage, metrics in stages.items():
        if not isinstance(metrics, Mapping) or set(metrics) != PACKAGED_FLOOR_SHAPE[stage]:
            got = sorted(metrics) if isinstance(metrics, Mapping) else metrics
            raise AssuranceInputError(f"{stage}: metrics must be {sorted(PACKAGED_FLOOR_SHAPE[stage])}, got {got} "
                                      f"(extra or missing: {sorted(set(got or []) ^ PACKAGED_FLOOR_SHAPE[stage])})")
        out[stage] = {}
        for metric, pair in metrics.items():
            ok = (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(type(v) is int for v in pair)
                  and pair[1] > 0 and 0 <= pair[0] <= pair[1])
            if not ok:
                raise AssuranceInputError(f"{stage}.{metric} must be [numerator, denominator] with 0 <= n <= d, d > 0")
            out[stage][metric] = (pair[0], pair[1])
    return out


def floors_sha256(floors: Mapping[str, Mapping[str, tuple[int, int]]]) -> str:
    return _sha(_dump({s: {m: list(v) for m, v in ms.items()} for s, ms in floors.items()}))


def floor_value(floors: Mapping[str, Mapping[str, tuple[int, int]]], stage: str, metric: str) -> Fraction:
    num, den = floors[stage][metric]
    return Fraction(num, den)


# --------------------------------------------------------------------------- binding


def _sources(run: Path) -> list[str]:
    manifest = _read(run / "sources" / "manifest.json")
    sources = manifest.get("sources") if isinstance(manifest, Mapping) else None
    if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
        raise AssuranceInputError("sources/manifest.json has no sources list")
    return sorted(sources)


def binding_refusal(run: Path) -> str | None:
    """The first rebind defect, as a value-free reason naming its source; ``None`` when bound."""
    extraction_root = _sha_file(run / "extraction.orm.json")
    checking_root = _sha_file(run / "checking.orm.json")
    for source in _sources(run):
        folder = run / "sources" / source
        record = _read(folder / "rebind.json") if (folder / "rebind.json").is_file() else None
        if not isinstance(record, Mapping) or record.get("schema") != "literature-rebind/1" or record.get("source") != source:
            return f"rebind:absent:{source}"
        population = _read(folder / "population.json")
        model = population.get("population", {}).get("model") if isinstance(population, Mapping) else None
        if record.get("checking_root_sha256") == extraction_root or model == "extraction.orm.json":
            return f"rebind:crossed-roots:{source}"
        if record.get("checking_root_sha256") != checking_root:
            return f"rebind:checking-root-stale:{source}"
        if record.get("extraction_root_sha256") != extraction_root:
            return f"rebind:extraction-root-stale:{source}"
        if record.get("extraction_population_sha256") != _sha_file(folder / "extracted.population.json"):
            return f"rebind:stale-extraction:{source}"
        if record.get("document_sha256") != _sha_file(folder / "document.txt"):
            return f"rebind:document-stale:{source}"
        for field, name in REBIND_FILES.items():  # r1 F1: every digest the record binds
            if record.get(field) != _sha_file(folder / name):
                return f"rebind:digest-mismatch:{field}:{source}"
    return None


# --------------------------------------------------------------------------- views


def _split_corpus(pair: Path) -> dict[str, Any]:
    corpus = _read(pair / "corpus.json")
    if not isinstance(corpus, Mapping) or corpus.get("schema") != SPLIT_CORPUS_SCHEMA:
        raise AssuranceInputError(f"corpus.json is not {SPLIT_CORPUS_SCHEMA}")
    if corpus.get("split") not in SPLIT_VALUES:
        raise AssuranceInputError(f"unknown split {corpus.get('split')!r}")
    return dict(corpus)


def view_corpus(pair: Path, run: Path, root: str) -> dict[str, Any]:
    """The ``literature-corpus/1`` the scorer reads for ``root`` (checking or extraction)."""
    split = _split_corpus(pair)
    model = {"checking": "checking.orm.json", "extraction": "extraction.orm.json"}[root]
    return {
        "schema": "literature-corpus/1",
        "model": model, "model_sha256": _sha_file(run / model),
        "artifacts": {"path": "artifacts",
                      "applicable_sha256": _sha_file(run / "artifacts" / EXTRACTION_ARTIFACTS["applicable"]),
                      "projection_sha256": _sha_file(run / "artifacts" / EXTRACTION_ARTIFACTS["projection"])},
        "documents": [{"id": d["id"], "path": f"documents/{d['id']}.md", "sha256": d["sha256"],
                       "domain": d["domain"], "split": SPLIT_VALUES[split["split"]], "work": d.get("work", {})}
                      for d in split["documents"]],
        "passages": list(split["passages"]),
    }


def checking_corpus_sha256(pair: Path, run: Path) -> str:
    """The digest an answer key for this pair and these roots must bind (``corpus_sha256``)."""
    return _sha(_dump(view_corpus(pair.resolve(), run.resolve(), "checking")))


_RUN_FILES = ("extraction.orm.json", "checking.orm.json", "vocabulary", "reference", "artifacts")


def _write_view(work: Path, pair: Path, run: Path, key: dict[str, Any], root: str, sources: list[str]) -> dict[str, str]:
    base = work / root
    corpus_root = base / "corpus-root"
    (corpus_root / "corpus").mkdir(parents=True)
    for name in _RUN_FILES:
        source = run / name
        if source.is_dir():
            shutil.copytree(source, corpus_root / name)
        elif source.is_file():
            shutil.copyfile(source, corpus_root / name)
    split = _split_corpus(pair)
    (corpus_root / "documents").mkdir()
    for document in split["documents"]:
        shutil.copyfile(pair / document["path"], corpus_root / "documents" / f"{document['id']}.md")
    if root == "checking":
        model = _read(run / "checking.orm.json")
        if isinstance(model, dict) and model.get("reference_populations"):
            anchor_model = {k: v for k, v in model.items() if k != "reference_populations"}
            (corpus_root / ANCHOR_MODEL).write_bytes(_dump(anchor_model))
    corpus_raw = _dump(view_corpus(pair, run, root))
    (corpus_root / "corpus" / "corpus.json").write_bytes(corpus_raw)
    answers = base / "answers"
    answers.mkdir()
    for folder in ("gold", "render"):
        if (key["_dir"] / folder).is_dir():
            shutil.copytree(key["_dir"] / folder, answers / folder)
    view_key = {k: v for k, v in key.items() if k != "_dir"}
    view_key["corpus_sha256"] = _sha(corpus_raw)
    (answers / "answer-key.json").write_bytes(_dump(view_key))
    run_dir = base / "run"
    (run_dir / "composition").mkdir(parents=True)
    for source in sources:
        src, dst = run / "sources" / source, run_dir / "composition" / source
        dst.mkdir()
        shutil.copyfile(src / "document.txt", dst / "document.txt")
        if root == "checking":
            shutil.copyfile(src / "population.json", dst / "population.json")
            shutil.copyfile(src / "population.json.anchors.json", dst / "population.json.anchors.json")
        else:
            shutil.copyfile(src / "extracted.population.json", dst / "population.json")
            shutil.copyfile(src / "extracted.population.json.anchors.json", dst / "population.json.anchors.json")
            shutil.copyfile(src / "extracted.population.extraction.json", dst / "population.json.extraction.json")
    if root == "checking":
        for name in ("manifest.json", "resolution.json"):
            if (run / "sources" / name).is_file():
                shutil.copyfile(run / "sources" / name, run_dir / "composition" / name)
        (run_dir / "composed").mkdir()
        shutil.copyfile(run / "composed.population.json", run_dir / "composed" / "population.json")
        shutil.copyfile(run / "composition.json", run_dir / "composed" / "record.json")
    run_doc = {"schema": "literature-run/1", "corpus_sha256": _sha(corpus_raw), "sources": sources,
               "abstentions": list(key.get("_abstentions", []))}
    (run_dir / "run.json").write_bytes(_dump(run_doc))
    out = {"corpus_root": str(corpus_root), "answers": str(answers / "answer-key.json"), "run": str(run_dir)}
    if (corpus_root / ANCHOR_MODEL).is_file():
        out["anchor_model"] = str(corpus_root / ANCHOR_MODEL)
    return out


def has_derivations(model_path: Path) -> bool:
    """Whether a model root, or any model it includes, declares a derived fact type."""
    model = _read(model_path)
    if not isinstance(model, Mapping):
        return False
    if any(isinstance(f, Mapping) and f.get("derivation_rule") for f in model.get("fact_types") or []):
        return True
    return any(has_derivations(model_path.parent / inc["path"]) for inc in model.get("includes") or []
               if isinstance(inc, Mapping) and isinstance(inc.get("path"), str))


def _refused(reason: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "exit_code": 3, "reason": reason, "stages": [], "failed_stages": []}


def score_two_root(pair: Path, answer_key: Path, run: Path, *, floors: Mapping[str, Mapping[str, tuple[int, int]]],
                   abstentions: list[str] | None = None, timeout_s: float = 1200) -> dict[str, Any]:
    """Score a two-root run against one split's answer key under ``floors``; never raises for inputs."""
    pair, answer_key, run = pair.resolve(), answer_key.resolve(), run.resolve()
    try:
        reason = binding_refusal(run)
        if reason is not None:
            return _refused(reason)
        key = _read(answer_key)
        if not isinstance(key, dict) or key.get("schema") != "literature-answer-key/1":
            return _refused("answers:schema")
        if key.get("corpus_sha256") != checking_corpus_sha256(pair, run):
            return _refused("answers:corpus-mismatch")
        key = {**key, "_dir": answer_key.parent, "_abstentions": abstentions or []}
        sources = _sources(run)
    except AssuranceInputError as exc:
        return _refused(f"input:{exc}")
    with tempfile.TemporaryDirectory(prefix="ontology-assurance-") as tmp:
        work = Path(tmp).resolve()
        try:
            views = {root: _write_view(work, pair, run, key, root, sources) for root in ("checking", "extraction")}
        except (OSError, AssuranceInputError) as exc:
            return _refused(f"input:view:{type(exc).__name__}")
        config = work / "config.json"
        config.write_bytes(_dump({"views": views, "sources": sources,
                                  "anchor_model": views["checking"].get("anchor_model"),
                                  "checking_has_derivations": has_derivations(run / "checking.orm.json"),
                                  "floors": {s: {m: list(v) for m, v in ms.items()} for s, ms in floors.items()}}))
        result = gnd_tool.run_gnd_python(["-I", str(_SCRIPT), str(config)], family=CommandFamily.OTHER,
                                         cwd=work, timeout_s=timeout_s)
    report = gnd_tool.parse_json_stdout(result)
    if report is None:
        return _refused(f"adapter:{result.meaning}")
    report["floors_sha256"] = floors_sha256(floors)
    report["answers_sha256"] = _sha_file(answer_key)
    return report
