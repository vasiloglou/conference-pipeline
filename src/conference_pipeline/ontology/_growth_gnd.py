"""Runs inside GND 0.4.0's interpreter for ``ontology.growth`` (v6.1 Phase 4, lane G).

``round <config>``: translates one growth reply (GND ``unattended-growth.md`` "The reply shape")
into a schema-version-3 evolution proposal over the current model, as GND's own tests build
them (``gnd.evolve.EvolutionProposal``, ``gnd.discover.output.candidate_document``):

- every ``quote`` is located in the document's NFC text; none, or more than one without
  ``quote_occurrence``, drops the entry and logs it (never a guessed offset);
- ids already in the model are reused, never re-added;
- a new document opens the proposal with one ``source-document`` ``add``;
- constraints are written ``llm-inferred``, ``proposed`` and ``deontic`` whatever their basis
  (``gnd evolve`` refuses any other source for a new constraint, its D-12), the basis and cue kept
  in ``rationale_world_fact``, and fold into their host fact type;
- ``rows`` become ``sample_population`` with one ``sample_spans`` entry each.

Writes ``proposal.json`` and ``candidate.orm.yaml`` into ``out`` and prints
``{"added", "reused", "dropped", "new_fact_types", "constrained", "open_questions"}`` or
``{"refused": reason}``; the host re-applies R15 to what survived the mapping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from gnd.discover import output
from gnd.evolve import (
    CommittedText,
    ElementLocator,
    EvolutionOperation,
    EvolutionOperationAudit,
    EvolutionOperationKind,
    EvolutionProposal,
    SourceDocumentLocator,
)
from ruamel.yaml import YAML

SOURCES = {"text-cue": "llm-inferred", "rows": "sample", "library": "common-sense"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class _Spans:
    def __init__(self, raw: bytes):
        self.raw, self.text, self.sha = raw, unicodedata.normalize("NFC", raw.decode("utf-8")), _sha(raw)
        self.dropped: list[dict] = []

    def span(self, quote, where: str, occurrence=None):
        if not isinstance(quote, str) or not quote:
            self.dropped.append({"entry": where, "reason": "no-quote"})
            return None
        starts, at = [], self.text.find(quote)
        while at >= 0:
            starts.append(at)
            at = self.text.find(quote, at + 1)
        if not starts:
            self.dropped.append({"entry": where, "reason": "quote-not-found"})
            return None
        if len(starts) > 1 and occurrence is None:
            self.dropped.append({"entry": where, "reason": "quote-ambiguous"})
            return None
        start = starts[occurrence or 0] if (occurrence or 0) < len(starts) else None
        if start is None:
            self.dropped.append({"entry": where, "reason": "quote-occurrence-out-of-range"})
            return None
        return {"document_sha256": self.sha, "start": start, "end": start + len(quote),
                "excerpt_sha256": _sha(quote.encode("utf-8"))}


def _op(n, kind, locator, **values):
    return EvolutionOperation(operation_id=f"op-{n}", kind=EvolutionOperationKind(kind), locator=locator,
                              **{k: copy.deepcopy(v) for k, v in values.items()})


def round_(config: dict) -> dict:
    model_path, doc_path, out = Path(config["model"]), Path(config["document"]), Path(config["out"])
    before_text = model_path.read_text(encoding="utf-8")
    before = YAML(typ="safe").load(before_text)
    reply, spans = config["reply"], _Spans(doc_path.read_bytes())
    existing = {("object_types", e["id"]) for e in before.get("object_types") or []} | \
               {("fact_types", e["id"]) for e in before.get("fact_types") or []}
    ops, added, reused, new_fact_types, constrained = [], [], [], [], []
    declared = {d["sha256"] for d in (before.get("source") or {}).get("documents") or []}
    if spans.sha not in declared:
        ops.append(_op(len(ops), "add", SourceDocumentLocator(locator_kind="source-document", sha256=spans.sha),
                       after={"sha256": spans.sha, "label": config["label"]}))
    constraints: dict[str, list[dict]] = {}
    for n, c in enumerate(reply.get("constraints") or []):
        span = spans.span(c.get("quote"), f"constraints[{n}]", c.get("quote_occurrence"))
        if span is None or c.get("basis") not in SOURCES:
            if span is not None:
                spans.dropped.append({"entry": f"constraints[{n}]", "reason": "unknown-basis"})
            continue
        # gnd evolve admits only llm-inferred/proposed/deontic for a new constraint (its D-12), although
        # unattended-growth.md maps rows -> sample and library -> common-sense: every basis is written
        # llm-inferred and the basis is kept in the world-fact rationale (Phase 4 finding).
        cue = c.get("cue") or c.get("library_entry") or ""
        body = {**c["constraint"], "source": "llm-inferred", "status": "proposed", "modality": "deontic",
                "rationale": c.get("rationale") or "",
                "provenance": {"spans": [span], "rationale_world_fact": f"{c['basis']}: {cue}".rstrip(": ")}}
        constraints.setdefault(c["host"], []).append(body)
    links = {s["subtype"]: s for s in reply.get("subtype_links") or []}
    for collection in ("object_types", "fact_types", "derived_fact_types"):
        target = "fact_types" if collection == "derived_fact_types" else "object_types" if collection == "object_types" \
            else "fact_types"
        for n, entry in enumerate(reply.get(collection) or []):
            eid = entry.get("id")
            if (target, eid) in existing:
                reused.append(eid)
                continue
            span = spans.span(entry.get("quote"), f"{collection}[{n}]", entry.get("quote_occurrence"))
            if span is None:
                continue
            element = {k: v for k, v in entry.items() if k not in ("quote", "quote_occurrence", "rows", "rationale")}
            element["provenance"] = {"spans": [span], **({"rationale": entry["rationale"]} if entry.get("rationale")
                                                          else {})}
            if collection == "fact_types" and entry.get("rows"):
                kept, sample_spans = [], []
                for r, row in enumerate(entry["rows"]):
                    rs = spans.span(row.get("quote"), f"{collection}[{n}].rows[{r}]", row.get("quote_occurrence"))
                    if rs is not None:
                        sample_spans.append({"row": len(kept), "span": rs})
                        kept.append(row["values"])
                if kept:
                    element["sample_population"] = kept
                    element["provenance"]["sample_spans"] = sample_spans
            if eid in constraints:
                element["constraints"] = constraints.pop(eid)
                constrained.append(eid)
            if target == "fact_types":  # review r2 F5: derived fact types are new fact types too
                new_fact_types.append(eid)
            if collection == "object_types" and eid in links:
                link = links[eid]
                element["subtype_of"] = link["supertype"]
                element["inherits_reference"] = bool(link.get("inherits_reference"))
            ops.append(_op(len(ops), "add", ElementLocator(collection=target, element_id=eid), after=element))
            added.append(eid)
    for host in constraints:  # constraints on existing fact types need --check-population (not in Phase 4)
        spans.dropped.append({"entry": f"constraints@{host}", "reason": "host-not-added-this-round"})
    if not ops:
        return {"refused": "nothing to apply"}
    elements = tuple(op for op in ops if op.locator.locator_kind != "source-document")
    candidate = output.candidate_document(before, elements) if elements else copy.deepcopy(before)
    for op in ops:
        if op.locator.locator_kind == "source-document":
            candidate.setdefault("source", {}).setdefault("documents", []).append(dict(op.after))
    text = output.candidate_text(candidate)
    proposal = EvolutionProposal(
        proposal_id=config["proposal_id"], expected_model_sha256=_sha(before_text.encode("utf-8")),
        candidate_model=CommittedText.from_text(text, media_type="application/yaml"), operations=tuple(ops),
        rationale=f"Unattended growth round over {config['label']}.",
        audit=EvolutionOperationAudit(actor_id=config["actor"], origin="conference_pipeline.ontology.growth",
                                      requested_at=datetime.fromisoformat(config["requested_at"]).astimezone(UTC)))
    (out / "proposal.json").write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    (out / "candidate.orm.yaml").write_text(text, encoding="utf-8")
    return {"added": added, "reused": reused, "dropped": spans.dropped, "new_fact_types": new_fact_types,
            "constrained": constrained, "open_questions": reply.get("open_questions") or []}


def main(argv: list[str]) -> int:
    config = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    try:
        result = {"round": round_}[argv[1]](config)
    except Exception as exc:  # noqa: BLE001 - GND's refusals become reasons
        result = {"refused": f"{type(exc).__name__}: {exc}"[:400]}
    sys.stdout.write(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
