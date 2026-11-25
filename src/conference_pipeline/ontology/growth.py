"""Unattended vocabulary growth host, one round per input document (v6.1 Phase 4, lane G).

Follows GND's ``unattended-growth.md`` ("What one round is"): the model is asked for Steps 1-7
on one document and replies in GND's reply shape (R15); the host refuses a reply that brings a
new fact type with no constraint and no open question about it (Steps 3-7 missing), translates
the rest into a schema-version-3 evolution proposal in GND's interpreter (``_growth_gnd.py``: quotes
located in the NFC text, a new document opens with one ``source-document`` add), then gates it:
``gnd validate <candidate> --document ... --require-documents`` → ``gnd evolve preview --policy
additive --shape-report`` → ``gnd evolve apply --policy additive``. Every tenth round also asks
for a consolidation reply and writes it for review without applying it.

Per strategy workspace: ``model.orm.yaml``, ``documents/``, ``records/`` (GND's evolution
evidence), ``rounds/NNNN/`` (prompt digest, reply, proposal, candidate, preview, gate outputs),
``growth-history.json`` (``ontology-growth-history/1``) and ``rejected-mappings.json``.
A work in the dev or sealed-test split is refused before any call (R16).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily
from conference_pipeline.ontology.gold_label import SECURITY
from conference_pipeline.ontology.producer import Producer, outcome_of, parse_json_object

HISTORY, LEDGER = "ontology-growth-history/1", "ontology-growth-rejected/1"
REPLY_KEYS = ("object_types", "fact_types", "constraints", "open_questions")
CONSOLIDATE_EVERY = 10
_HELPER = Path(__file__).with_name("_growth_gnd.py")


class GrowthError(ValueError):
    """A refused round; the message is the reason."""


def _dump(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def growth_prompt(model_text: str, document: str, label: str) -> str:
    """GND's four moves for one document, with the reply shape it must return (R7, R15)."""
    return (f"{SECURITY}\n\nYou grow an ORM model unattended, one document per round (GND unattended-growth). "
            "Read the current model first and reuse its object type and fact type ids. For this document: "
            "Steps 1-2 (elementary facts, object types, fact types, sample rows), then Steps 3-6 on this round's "
            "new types only (combine or derive; uniqueness and arity; mandatory roles; value, set-comparison, "
            "ring and subtyping), Step 7 (other constraints). Every proposal cites a verbatim quote from the "
            "document. Every constraint is a proposal: give its basis (text-cue, rows or library), the cue, the "
            "quote and a rationale. Anything the evidence does not settle goes to open_questions. Never emit an "
            "operation. A new fact type with no constraint must have an open question saying why.\n\n"
            'Return one JSON object: {"document", "object_types", "fact_types" (with "rows": [{"values", "quote"}]), '
            '"subtype_links", "constraints": [{"host", "basis", "constraint", "cue", "quote", "rationale"}], '
            '"derived_fact_types", "open_questions": [{"step", "note", "quote"}]}.\n\n'
            f"Current model:\n<<<\n{model_text}\n>>>\n\nDocument {label}:\n<<<\n{document}\n>>>")


def _bare(fact_types, constrained, reply: Mapping[str, Any]) -> list[str]:
    """R15: new fact types with neither a constraint nor an open question naming them."""
    asked = " ".join(str(q.get("note", "")) for q in reply["open_questions"] if isinstance(q, Mapping))
    return [f for f in fact_types if f not in constrained and f not in asked]


def reply_problems(reply: Mapping[str, Any], existing_fact_types: set[str]) -> list[str]:
    """R15: the reply covers Steps 1-7; each new fact type carries a constraint or an open question."""
    problems = [f"missing:{k}" for k in REPLY_KEYS if not isinstance(reply.get(k), list)]
    if problems:
        return problems
    hosts = {c.get("host") for c in reply["constraints"] if isinstance(c, Mapping)}
    new = [ft.get("id") for ft in [*reply["fact_types"], *(reply.get("derived_fact_types") or [])]  # review r2 F5
           if isinstance(ft, Mapping) and ft.get("id") and ft.get("id") not in existing_fact_types]
    return [f"steps-3-7-missing:{f}" for f in _bare(new, hosts, reply)]


def split_blocker(splits: Mapping[str, Any]) -> Callable[[str], bool]:
    """R16: True for a work whose family is in the dev or sealed-test split (``ontology-splits/1``)."""
    held = {work for split in ("dev", "sealed_test") for group in splits["splits"][split]
            for work in splits["groups"].get(group, [])}
    return held.__contains__


def _model_ids(model_path: Path) -> set[str]:
    import yaml

    model = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
    return {f["id"] for f in model.get("fact_types") or []}


def _declared(model_path: Path) -> list[str]:
    import yaml

    model = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
    return [d["label"] for d in (model.get("source") or {}).get("documents") or []]


def _load(path: Path, default):
    return json.loads(path.read_bytes()) if path.is_file() else default


def run_round(workspace: Path, document: Path, *, round_no: int, producer: Producer, strategy: str,
              work: str, blocked: Callable[[str], bool], requested_at: str) -> dict[str, Any]:
    """One growth round over ``document``; returns its history row (also appended to the history)."""
    model = workspace / "model.orm.yaml"
    history_path, ledger_path = workspace / "growth-history.json", workspace / "rejected-mappings.json"
    history = _load(history_path, {"schema": HISTORY, "strategy": strategy, "rounds": []})
    ledger = _load(ledger_path, {"schema": LEDGER, "strategy": strategy, "rows": []})
    folder = workspace / "rounds" / f"{round_no:04d}"
    raw = document.read_bytes()
    row: dict[str, Any] = {"round": round_no, "work": work, "document": document.name, "document_sha256": _sha(raw)}

    docs = workspace / "documents"
    kept = docs / document.name
    copied = False

    def reject(stage: str, reason: str, **extra) -> dict[str, Any]:
        if copied:  # review F4: a rejected round leaves the bound documents as they were
            kept.unlink()
        ledger["rows"].append({"round": round_no, "work": work, "stage": stage, "reason": reason, **extra})
        ledger_path.write_bytes(_dump(ledger))
        row.update(status="rejected", stage=stage, reason=reason)
        history["rounds"].append(row)
        history_path.write_bytes(_dump(history))
        return row

    if blocked(work):
        return reject("leakage", "dev-or-sealed-test-work")
    if kept.is_file() and _sha(kept.read_bytes()) != row["document_sha256"]:
        # review F4: another document already holds this label; never overwrite bound evidence
        return reject("document", "label-bound-to-other-bytes")
    folder.mkdir(parents=True, exist_ok=False)
    model_text = model.read_text(encoding="utf-8")
    prompt = growth_prompt(model_text, raw.decode("utf-8"), document.name)
    response = producer("grower", "grow", f"{strategy}#{work}", prompt)
    (folder / "reply.json").write_bytes(_dump({"text": response.text, "stop_reason": response.stop_reason,
                                               "provider": response.provider, "model": response.model}))
    row.update(prompt_sha256=_sha(prompt.encode()), reply_sha256=_sha((folder / "reply.json").read_bytes()))
    stop = outcome_of(response)
    reply = None if stop else parse_json_object(response.text)
    if reply is None:
        return reject("reply", stop or "unparsable")
    problems = reply_problems(reply, _model_ids(model))
    if problems:
        return reject("reply", "r15-refused", problems=problems)
    docs.mkdir(exist_ok=True)
    if not kept.is_file():
        shutil.copyfile(document, kept)
        copied = True
    config = folder / "round.config.json"
    config.write_bytes(_dump({"model": str(model), "document": str(docs / document.name), "label": document.name,
                              "reply": reply, "out": str(folder), "proposal_id": f"{strategy}-{round_no:04d}",
                              "actor": f"growth:{strategy}", "requested_at": requested_at}))
    made = gnd_tool.parse_json_stdout(gnd_tool.run_gnd_python(["-I", str(_HELPER), "round", str(config)],
                                                              family=CommandFamily.OTHER, cwd=folder))
    if made is None or "refused" in made:
        return reject("proposal", str((made or {}).get("refused", "failed")))
    for drop in made["dropped"]:
        ledger["rows"].append({"round": round_no, "work": work, "stage": "mapping", **drop})
    # review F5: R15 again after the mapping, which may have dropped a new fact type's constraints
    bare = _bare(made["new_fact_types"], set(made["constrained"]), reply)
    if bare:
        return reject("proposal", "r15-refused", problems=[f"steps-3-7-missing:{f}" for f in bare])
    row.update(added=made["added"], reused=made["reused"], dropped=len(made["dropped"]),
               open_questions=made["open_questions"])
    labels = sorted(set(_declared(model)) | {document.name})
    validate = gnd_tool.run_gnd(["validate", str(folder / "candidate.orm.yaml"),
                                 *[a for name in labels for a in ("--document", str(docs / name))],
                                 "--require-documents"], family=CommandFamily.CHECK, cwd=workspace)
    rel = folder.relative_to(workspace)
    preview = gnd_tool.run_gnd(["evolve", "preview", "model.orm.yaml", str(rel / "proposal.json"), "--workspace", ".",
                                "--records", "records", "--out", str(rel / "preview.json"), "--policy", "additive",
                                "--shape-report", "--format", "json"], family=CommandFamily.OTHER, cwd=workspace) \
        if validate.exit_code == 0 else None
    row["exits"] = {"validate": validate.exit_code, "preview": preview.exit_code if preview else None}
    if validate.exit_code != 0 or preview is None or preview.exit_code != 0:
        stage = "validate" if validate.exit_code != 0 else "preview"
        detail = (validate if stage == "validate" else preview).stderr[-300:]
        return reject(stage, f"exit-{(validate if stage == 'validate' else preview).exit_code}", detail=detail)
    if round_no % CONSOLIDATE_EVERY == 0:
        consolidation = producer("grower", "consolidate", f"{strategy}#round-{round_no:04d}",
                                 f"{SECURITY}\n\nConsolidation round (GND unattended-growth, 'The consolidation "
                                 f"round') over the whole model. Propose merges and whole-model checks only; nothing "
                                 f"is applied without review.\n\nModel:\n<<<\n{model_text}\n>>>")
        (workspace / "consolidation").mkdir(exist_ok=True)
        (workspace / "consolidation" / f"{round_no:04d}.proposal.json").write_bytes(_dump(
            {"round": round_no, "model_sha256": _sha(model_text.encode()), "status": "for-review-not-applied",
             "reply": consolidation.text}))
        row["consolidation"] = f"consolidation/{round_no:04d}.proposal.json"
    apply = gnd_tool.run_gnd(["evolve", "apply", "model.orm.yaml", str(rel / "proposal.json"), "--preview",
                              str(rel / "preview.json"), "--workspace", ".", "--records", "records", "--policy",
                              "additive", "--format", "json"], family=CommandFamily.OTHER, cwd=workspace)
    row["exits"]["apply"] = apply.exit_code
    if apply.exit_code != 0:
        return reject("apply", f"exit-{apply.exit_code}", detail=apply.stderr[-300:])
    row.update(status="applied", model_sha256=_sha(model.read_bytes()))
    history["rounds"].append(row)
    history_path.write_bytes(_dump(history))
    ledger_path.write_bytes(_dump(ledger))
    return row
