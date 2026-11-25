"""Admission of extracted assertions by GND's judgment route, on the seed's own model (v6.1 Phase 4, lane A).

P4-D3 (b), owner-approved 2026-11-20: no translation into the GND example's 14-role flat
statement (spec F2a). The checking root is judged as it is:

1. **Flatten.** ``gnd judgments check`` refuses a model that declares includes (GND 0.4.0
   ``judgments.py:470``), so the root is flattened by GND itself: ``gnd extract`` writes
   ``<stem>.flat.orm.json`` and a ``gnd-flattening/1`` record. A root ``gnd extract`` refuses
   (temporal models) is refused here with its reason (implementation findings, spec §8). A root
   that binds reference populations keeps them in the flat model; their files are copied beside
   it and each receipt carries GND's ``reference_identity`` (review F3).
2. **Judge.** One target per assertion: its row in the shape's target fact type
   (``assertion_has_predicate`` in the seed). Each verdict from the reading review becomes a
   consumer-pinned receipt quoting the fact's evidence bytes (``_admission_gnd.py``, GND's own
   digests); ``gnd judgments check <flat> <raw> --policy --receipts --trusted-root`` judges them
   and runs the mechanical check.
3. **Host stages**, the example's two with its reason strings: ``semantic-support`` (``missing``,
   ``rejected``, ``abstained``, ``approval-or-mechanical-floor``) and ``positive-own-findings``
   (``negated``, ``speculated``, ``cited``) from the seed's polarity and attribution facts.
4. **Dispositions.** ``admitted`` (both pass); ``supported-not-admitted`` (supported but not a
   positive own finding: kept in the population, flagged in the record, never counted as an
   admitted supported fact); ``rejected`` (removed with every row that hangs off it).

The output population keeps every fact type without an assertion role byte for byte and every
row that names a kept assertion (participants, conditions, quantities, ...), so nothing the
gold holds is lost in a projection. ``admission-record/1`` binds every input and output digest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily

SCHEMA = "admission-record/1"
_HELPER = Path(__file__).with_name("_admission_gnd.py")
VERDICTS = {"supported": "approve", "unsupported": "reject", "abstain": "abstain"}


class AdmissionError(ValueError):
    """A refusal; the message is the reason (adapter exit 3)."""


@dataclass(frozen=True)
class AssertionShape:
    """Where an assertion lives in the model: the seed's fact types by default."""

    entity_role: str = "assertion"
    target: str = "assertion_has_predicate"
    polarity: tuple[str, str] | None = ("assertion_has_polarity", "polarity")
    attribution: tuple[str, str] | None = ("assertion_attributed_to", "attribution")
    positive: tuple[str, str] = ("asserted", "own")

    def reason(self, polarity: str | None, attribution: str | None) -> str | None:
        if self.polarity and polarity != self.positive[0]:
            return {"negated": "negated", "speculated": "speculated"}.get(polarity or "", f"polarity:{polarity}")
        if self.attribution and attribution != self.positive[1]:
            return {"cited": "cited"}.get(attribution or "", f"attribution:{attribution}")
        return None


def _dump(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def flatten(root: Path, work: Path) -> tuple[Path, bytes]:
    """GND's own flattening of ``root``: the flat model path and the ``gnd-flattening/1`` record bytes.

    A root with no includes is already flat: GND writes no flat model for it (review r2 F6), so it
    is judged in place, its reference files where they are, and the record bytes are empty.
    """
    import yaml

    if not (yaml.safe_load(root.read_text(encoding="utf-8")) or {}).get("includes"):
        return root, b""
    out = work / "flat"
    result = gnd_tool.run_gnd(["extract", str(root), "--out", str(out)], family=CommandFamily.OTHER, cwd=root.parent)
    if result.exit_code != 0:
        reason = "temporal-unsupported" if "temporal" in result.stderr else result.meaning
        raise AdmissionError(f"admission:flatten:{reason}")
    stem = root.name.removesuffix(".json").removesuffix(".yaml")
    flat = out / f"{stem}.flat.orm.json"
    # Phase 4 review F3: the flat model keeps the root's reference_populations with their relative
    # paths, so the catalog files go beside it; GND checks every byte against the declared digest.
    for ref in json.loads(flat.read_bytes()).get("reference_populations") or []:
        target = (out / ref["path"]).resolve()
        if out.resolve() not in target.parents:
            raise AdmissionError(f"admission:reference-path-escapes:{ref['id']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root.parent / ref["path"], target)
    return flat, (out / f"{stem}.flattening.json").read_bytes()


def admit(root: Path, raw: Path, verdicts: Mapping[str, Mapping[str, Any]], trusted_root: Path, out: Path, *,
          shape: AssertionShape = AssertionShape(), judge: str = "reviewer") -> dict[str, Any]:
    """Judge ``raw`` (a population of ``root``) and write the admitted population and its record to ``out``.

    ``verdicts``: assertion id -> ``{"verdict": supported|unsupported|abstain, "path": file under
    trusted_root, "start", "end"}`` (byte offsets of the evidence). An assertion with no entry has
    no receipt (``missing``).
    """
    if out.exists():
        raise AdmissionError("output:exists")
    work = out.with_name(out.name + ".work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        flat, flattening = flatten(root, work)
        raw_bytes = raw.read_bytes()
        population = json.loads(raw_bytes)
        rows = population["facts"].get(shape.target, [])
        ids = [row[shape.entity_role] for row in rows]
        if len(set(ids)) != len(ids):
            raise AdmissionError("input:assertion-identities")
        targets = []
        for index, aid in enumerate(ids):
            v = verdicts.get(aid)
            targets.append({"fact_type": shape.target, "index": index,
                            **({"verdict": VERDICTS[v["verdict"]], "path": v["path"], "start": v["start"],
                                "end": v["end"]} if v else {})})
        config = work / "receipts.config.json"
        config.write_bytes(_dump({"model": str(flat), "population": str(raw), "trusted_root": str(trusted_root),
                                  "judge": judge, "policy_id": f"admission:{root.name}", "targets": targets,
                                  "out": str(work)}))
        made = gnd_tool.parse_json_stdout(gnd_tool.run_gnd_python(
            ["-I", str(_HELPER), "receipts", str(config)], family=CommandFamily.OTHER, cwd=work))
        if made is None or "refused" in made:
            raise AdmissionError(f"receipts:{(made or {}).get('refused', 'failed')}")
        result = gnd_tool.run_gnd(["judgments", "check", str(flat), str(raw), "--policy", str(work / "policy.json"),
                                   "--receipts", str(work / "receipts.json"), "--trusted-root", str(trusted_root),
                                   "--format", "json"], family=CommandFamily.CHECK, cwd=work)
        report = gnd_tool.parse_json_stdout(result)
        if result.exit_code not in (0, 1) or not report or report.get("error") is not None:
            raise AdmissionError(f"judgments:refused:{result.meaning}")
        judged = {t["index"]: t for t in report["judgment_policy"]["targets"]}
        record = _record(population, rows, ids, judged, shape)
        admitted_bytes = _dump(_output(population, {d["assertion"] for d in record if d["disposition"] != "rejected"},
                                       shape))
        out.mkdir(parents=True)
        (out / "admitted.population.json").write_bytes(admitted_bytes)
        report_bytes = _dump(report)
        (out / "judgment-report.json").write_bytes(report_bytes)
        doc = {"schema": SCHEMA, "root_sha256": _sha(root.read_bytes()), "flat_model_sha256": _sha(flat.read_bytes()),
               "flattening_sha256": _sha(flattening) if flattening else None, "raw_sha256": _sha(raw_bytes),
               "policy_sha256": _sha((work / "policy.json").read_bytes()),
               "receipts_sha256": _sha((work / "receipts.json").read_bytes()),
               "judgments_sha256": _sha(report_bytes), "admitted_sha256": _sha(admitted_bytes),
               "gnd": {"exit": result.exit_code, "meaning": result.meaning},
               "counts": {k: sum(d["disposition"] == k for d in record)
                          for k in ("admitted", "supported-not-admitted", "rejected")},
               "assertions": record}
        (out / "admission.json").write_bytes(_dump(doc))
        return doc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _value(population: Mapping[str, Any], pair: tuple[str, str] | None, entity_role: str, aid: str) -> str | None:
    if pair is None:
        return None
    ft, role = pair
    found = [r[role] for r in population["facts"].get(ft, []) if r.get(entity_role) == aid]
    return found[0] if len(found) == 1 else None


def _record(population, rows, ids, judged, shape: AssertionShape) -> list[dict[str, Any]]:
    out = []
    for index, aid in enumerate(ids):
        target = judged.get(index)
        verdicts = {j["verdict"] for j in (target or {}).get("judgments", [])}
        if target is not None and target["admitted"]:
            support = None
        elif not target or "missing" in verdicts:
            support = "missing"
        elif "reject" in verdicts:
            support = "rejected"
        elif "abstain" in verdicts:
            support = "abstained"
        else:
            support = "approval-or-mechanical-floor"
        positive = None if support else shape.reason(_value(population, shape.polarity, shape.entity_role, aid),
                                                     _value(population, shape.attribution, shape.entity_role, aid))
        disposition = "rejected" if support else "supported-not-admitted" if positive else "admitted"
        out.append({"assertion": aid, "index": index, "disposition": disposition,
                    "stages": {"semantic-support": support, "positive-own-findings": positive}})
    return out


def _output(population: Mapping[str, Any], kept: set[str], shape: AssertionShape) -> dict[str, Any]:
    facts = {}
    for ft, rows in population["facts"].items():
        if rows and shape.entity_role in rows[0]:
            facts[ft] = [r for r in rows if r[shape.entity_role] in kept]
        else:
            facts[ft] = rows
    return {**{k: v for k, v in population.items() if k != "facts"}, "facts": facts}


def composition_refusal(sources: Mapping[str, Path]) -> str | None:
    """For a composition manifest: the first source whose admitted population is not the bytes its
    admission record bound (``sources``: name -> admission output directory)."""
    for name, folder in sorted(sources.items()):
        record = json.loads((folder / "admission.json").read_bytes())
        if record.get("admitted_sha256") != _sha((folder / "admitted.population.json").read_bytes()):
            return f"composition:admitted-digest-mismatch:{name}"
    return None


def byte_span(text: str, span: Mapping[str, int]) -> dict[str, int]:
    """A reading span (characters) as the byte offsets a receipt quotes."""
    return {"start": len(text[:span["start"]].encode("utf-8")), "end": len(text[:span["end"]].encode("utf-8"))}
