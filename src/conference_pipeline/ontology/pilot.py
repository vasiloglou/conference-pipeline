"""Model pilots: one sample, four arms, every candidate under the same settings (v6.1 Phase 5).

Spec ``notes/phase-5-spec.md`` §3. The reading arms call the reading host's own prompt and
parser (``reading.extract_prompt`` / ``read_facts`` in the ``assertions`` shape) directly, one
unit at a time, so an arm measures one task and its score does not depend on another arm's
pick:

- ``broad``: one call per work over its tier-A sections, at most ``reading.MAX_SECTIONS_PER_CALL``
  per call (``ontology.read_broad``);
- ``targeted``: one call per tier-B section, the first six of each work in document order
  (``ontology.read_targeted``);
- ``review``: one call per work judging that work's gold statements as candidate facts
  (``ontology.read_review``);
- ``growth``: one growth round per growth-split work, from a fresh unattended root with an
  empty vocabulary (``growth_root``; the seed cannot be one, finding F9)
  (``ontology.grow_round``), through ``growth.run_round``.

Every call goes through the given producer (``OntologyProducer`` live, ``SavedProducer`` for a
replay with no call). The reading arms read the dev pair only; the growth arm reads growth works
only (R16). Nothing here opens the sealed-test pair.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from conference_pipeline.ontology import extraction, reading
from conference_pipeline.ontology.assertions import TEXTUAL_ROLES, reading_rows
from conference_pipeline.ontology.producer import Producer, Response

SAMPLE_SCHEMA, ARM_SCHEMA = "ontology-pilot-sample/1", "ontology-pilot-arm/1"
ARMS = ("broad", "targeted", "review", "growth")
TASKS = {"broad": "ontology.read_broad", "targeted": "ontology.read_targeted", "review": "ontology.read_review",
         "growth": "ontology.grow_round"}
ROLES = {"broad": "reader", "targeted": "reader", "review": "reviewer", "growth": "grower"}
TARGETED_PER_WORK = 6
SEED_LABEL = "seed-v1.orm.yaml"


class PilotError(ValueError):
    """An input is refused (exit 3 at the command line)."""


def _dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _salted(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}\n{key}".encode()).hexdigest()


def digest(manifest: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------- samples (lane S)


def select_sample(corpus: Mapping[str, Any], *, salt: str, n: int = 20) -> dict[str, Any]:
    """One dev work per area (lowest salted hash), then the rest by salted hash; ordered."""
    if corpus.get("split") != "dev":
        raise PilotError(f"the pilot samples the dev split only, not {corpus.get('split')!r}")
    docs = sorted(corpus["documents"], key=lambda d: _salted(salt, d["work"]["identifier"]))
    picked, areas = [], set()
    for doc in docs:
        if doc["domain"] not in areas:
            areas.add(doc["domain"])
            picked.append(doc)
    picked = picked[:n] + [d for d in docs if d not in picked][:max(0, n - len(picked))]
    if len(picked) < n:
        raise PilotError(f"only {len(picked)} dev works")
    works = [{"work": d["work"]["identifier"], "document": d["id"], "path": d["path"], "sha256": d["sha256"],
              "domain": d["domain"]} for d in sorted(picked, key=lambda d: d["id"])]
    manifest = {"schema": SAMPLE_SCHEMA, "kind": "dev", "salt": salt, "n": n, "works": works}
    return {**manifest, "sha256": digest(manifest)}


def growth_sample(splits: Mapping[str, Any], roster: Mapping[str, Any], text_of: Callable[[str], str], *,
                  salt: str, n: int = 5, min_chars: int = 20_000) -> dict[str, Any]:
    """``n`` growth-split main-paper families with at least ``min_chars`` of text, by salted hash."""
    growth = {f for g in splits["splits"]["growth"] for f in splits["groups"].get(g, [])}
    mains = sorted((f["family_key"] for f in roster["families"]
                    if f["family_key"] in growth and f.get("kind") == "main-paper"), key=lambda k: _salted(salt, k))
    works = []
    for key in mains:
        text = text_of(key)
        if len(text) >= min_chars:
            works.append({"work": key, "sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)})
        if len(works) == n:
            manifest = {"schema": SAMPLE_SCHEMA, "kind": "growth", "salt": salt, "n": n, "works": works}
            return {**manifest, "sha256": digest(manifest)}
    raise PilotError(f"only {len(works)} growth works with text")


def check_sample(sample: Mapping[str, Any]) -> None:
    if sample.get("schema") != SAMPLE_SCHEMA or sample.get("sha256") != digest(sample):
        raise PilotError("sample manifest digest differs from its content")


def load_dev(pair: Path, sample: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, dict], dict[str, Any]]:
    """(texts, golds, links) of the sampled works, read from the dev pair directory only."""
    check_sample(sample)
    if sample["kind"] != "dev":
        raise PilotError("a reading arm needs a dev sample")
    corpus = json.loads((pair / "corpus.json").read_bytes())
    if corpus.get("split") != "dev":
        raise PilotError("the pair is not the dev pair")
    texts, golds = {}, {}
    for w in sample["works"]:
        raw = (pair / w["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != w["sha256"]:
            raise PilotError(f"{w['document']} text differs from the sample")
        texts[w["work"]] = raw.decode("utf-8")
        golds[w["work"]] = json.loads((pair / "gold" / f"{w['document']}.gold.json").read_bytes())
    return texts, golds, json.loads((pair / "links.json").read_bytes())


# --------------------------------------------------------------------------- reading arms


def _sections(text: str, tier: str) -> list[dict[str, Any]]:
    return [s for s in reading.inventory(text)["sections"] if s["tier"] == tier and s["kind"] in ("section", "window")]


def units(arm: str, text: str) -> list[list[dict[str, Any]]]:
    """The section groups one work's calls read in this arm."""
    if arm == "broad":
        opening = _sections(text, "A")
        size = reading.MAX_SECTIONS_PER_CALL
        return [opening[i:i + size] for i in range(0, len(opening), size)]
    if arm == "targeted":
        return [[s] for s in _sections(text, "B")[:TARGETED_PER_WORK]]
    raise PilotError(f"no section units for arm {arm!r}")


def _response(r: Response) -> dict[str, Any]:
    return {"text": r.text, "stop_reason": r.stop_reason, "provider": r.provider, "model": r.model}


def _call(producer: Producer, role: str, kind: str, key: str, prompt: str) -> dict[str, Any] | None:
    """One call; a provider error is that unit's ``error`` outcome (charged by the producer), never
    a retry and never the end of the candidate's run. A budget halt still stops the run."""
    from conference_pipeline.ontology.producer import BudgetExceededError

    try:
        return _response(producer(role, kind, key, prompt))
    except BudgetExceededError:
        raise
    except Exception:  # noqa: BLE001 - recorded as the unit's outcome (R8: no silent retry)
        return None


def read_arm(arm: str, texts: Mapping[str, str], *, producer: Producer, fact_types: Sequence[str]) -> dict[str, Any]:
    """Run the broad or targeted arm over every work; per work its calls, claims and drops."""
    out: dict[str, Any] = {}
    for work in sorted(texts):
        text, calls, claims, drops, read = texts[work], [], [], [], []
        for k, group in enumerate(units(arm, text)):
            ids = "+".join(s["id"] for s in group)
            raw = _call(producer, ROLES[arm], arm, f"{work}#{ids}",
                        reading.extract_prompt(text, group, fact_types, (), "assertions"))
            if raw is None:
                calls.append({"sections": [s["id"] for s in group], "outcome": "error", "stop_reason": "error"})
                continue
            outcome, facts, dropped, _keys = reading.read_facts(raw, text, group, fact_types, f"{arm[0]}{k + 1}-",
                                                                "assertions")
            calls.append({"sections": [s["id"] for s in group], "outcome": outcome, "stop_reason": raw["stop_reason"]})
            claims += facts
            drops += dropped
            read += [{"id": s["id"], "start": s["start"], "end": s["end"]} for s in group]
        out[work] = {"calls": calls, "claims": claims, "drops": drops, "sections_read": read}
    return out


def gold_as_facts(gold: Mapping[str, Any], text: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """A work's gold statements as candidate facts for the reviewer, each in the section holding it."""
    sections = reading.inventory(text)["sections"]
    facts, by_id = [], {}
    for st in gold.get("statements") or []:
        sp = st["span"]
        section = next((s for s in sections if s["start"] <= sp["start"] < s["end"]), None)
        if section is None:
            continue
        by_id[section["id"]] = section
        facts.append({"id": st["id"], "fact_type": "assertion", "section": section["id"], "predicate": st["predicate"],
                      "participants": [{"name": p["name"], "role": p["role"]} for p in st["participants"]],
                      "conditions": list(st.get("conditions") or []), "attribution": st["attribution"],
                      "polarity": st["polarity"], "quote": text[sp["start"]:sp["end"]], "roles": {},
                      "_support": st["support"]})
    return facts, by_id


def review_arm(texts: Mapping[str, str], golds: Mapping[str, Mapping[str, Any]], *, producer: Producer
               ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for work in sorted(texts):
        facts, by_id = gold_as_facts(golds[work], texts[work])
        shown = [{k: v for k, v in f.items() if not k.startswith("_")} for f in facts]
        raw = _call(producer, "reviewer", "review", f"{work}#gold", reading.review_prompt(texts[work], shown, by_id))
        verdicts = reading.read_verdicts(raw, shown) if raw is not None else dict.fromkeys((f["id"] for f in shown),
                                                                                            "missing")
        raw = raw or {"stop_reason": "error"}
        out[work] = {"stop_reason": raw["stop_reason"],
                     "judged": [{"id": f["id"], "gold": f["_support"], "verdict": verdicts[f["id"]]} for f in facts]}
    return out


# --------------------------------------------------------------------------- growth arm


def growth_root(label: str, sha256: str) -> str:
    """An unattended growth root with an empty vocabulary that declares its first document.

    Implementation finding F9 (spec): GND 0.4.0 refuses the owner-confirmed seed as, or as an
    include of, an unattended root (rules unattended-u2..u5: confirmed constraints without a
    reviewer, ``explicit`` sources, elements without source spans), and ``evolve preview``
    refuses a current model that declares no source document (unattended-u1). So the growth arm
    grows from an empty vocabulary whose one declared document is the first growth work."""
    return yaml.safe_dump({"version": 4, "model_id": "urn:conference-pipeline:ontology:growth-pilot",
                           "source": {"kind": "prose", "mode": "unattended", "reviewer": "none",
                                      "documents": [{"sha256": sha256, "label": label}]},
                           "object_types": [], "fact_types": []}, sort_keys=False)


# --------------------------------------------------------------------------- verify (seed run)


def seed_run(run: Path, seed_path: Path) -> Path:
    """A GND interop run directory whose extraction and checking roots are the seed, bundled."""
    if run.exists():
        raise PilotError("verify run directory already exists")
    run.mkdir(parents=True)
    model = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    for root in (extraction.EXTRACTION, extraction.CHECKING):
        (run / root).write_text(json.dumps(model, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    extraction.bundle(run)
    return run


def verify_claims(run: Path, work: str, text: str, claims: Sequence[Mapping[str, Any]], *,
                  executing_model: str) -> dict[str, Any]:
    """Write one work's claims as seed rows (textual roles anchored) and run ``gnd verify``."""
    source = work.replace(":", "-").replace("/", "-")
    written = extraction.write_source(run, source, text, reading_rows(work, claims), executing_model=executing_model,
                                      textual_roles=TEXTUAL_ROLES)
    result = extraction.verify(run, source)
    return {**written, "verify_exit": result.exit_code, "verify_meaning": result.meaning}



# --------------------------------------------------------------------------- one arm, one candidate


def run_arm(arm: str, sample: Mapping[str, Any], out: Path, *, producer: Any, inputs: Mapping[str, Any],
            stop_app: Callable[[], dict[str, Any]] | None = None, clock: Callable[[], float] | None = None
            ) -> dict[str, Any]:
    """Run one candidate over the sample, work by work; a budget halt keeps what was done.

    ``inputs`` holds what the arm reads: ``texts`` and ``fact_types`` (reading arms), ``golds``
    (review), or ``text_of``, ``blocked`` and ``requested_at`` (growth). Writes
    ``arm.json`` and returns the summary; ``stop_app`` (a Modal app stop) runs in a ``finally``
    on every path (013 decision 1)."""
    import time

    from conference_pipeline.ontology.producer import BudgetExceededError

    if arm not in ARMS:
        raise PilotError(f"unknown arm {arm!r}")
    check_sample(sample)
    if (arm == "growth") != (sample["kind"] == "growth"):
        raise PilotError(f"arm {arm} cannot read a {sample['kind']} sample")
    if out.exists():
        raise PilotError(f"{out} already exists")
    out.mkdir(parents=True)
    clock = clock or time.monotonic
    started, results, done, status, halt = clock(), {}, [], "complete", None
    stop: dict[str, Any] | None = None
    try:
        for w in sample["works"]:
            work = w["work"]
            try:
                if arm in ("broad", "targeted"):
                    results.update(read_arm(arm, {work: inputs["texts"][work]}, producer=producer,
                                            fact_types=inputs["fact_types"]))
                elif arm == "review":
                    results.update(review_arm({work: inputs["texts"][work]}, {work: inputs["golds"][work]},
                                              producer=producer))
                else:
                    from conference_pipeline.ontology import growth

                    workspace = out / "growth"
                    label = work.replace(":", "-").replace("/", "-") + ".md"
                    text = inputs["text_of"](work)
                    if not workspace.exists():
                        for d in ("records", "documents", "inputs"):
                            (workspace / d).mkdir(parents=True)
                        if inputs.get("seed_raw") is not None:  # P5-D9: the seed's unattended working copy
                            from conference_pipeline.ontology.seed_package import unattended_seed

                            (workspace / "documents" / SEED_LABEL).write_bytes(inputs["seed_raw"])
                            root = unattended_seed(inputs["seed_raw"], label=SEED_LABEL,
                                                   artifact=inputs["seed_artifact"], commit=inputs["seed_commit"])
                        else:
                            (workspace / "documents" / label).write_text(text, encoding="utf-8")
                            root = growth_root(label, hashlib.sha256(text.encode("utf-8")).hexdigest())
                        (workspace / "model.orm.yaml").write_text(root, encoding="utf-8")
                    doc = workspace / "inputs" / label
                    doc.write_text(text, encoding="utf-8")
                    results[work] = growth.run_round(workspace, doc, round_no=len(done) + 1, producer=producer,
                                                     strategy=inputs.get("strategy", "pilot"), work=work,
                                                     blocked=inputs["blocked"], requested_at=inputs["requested_at"])
            except BudgetExceededError as exc:
                status, halt = "budget-limited", str(exc)
                break
            done.append(work)
    finally:
        if stop_app is not None:
            stop = stop_app()
    summary = {"schema": ARM_SCHEMA, "arm": arm, "sample_sha256": sample["sha256"], "status": status,
               "halt": halt, "works_done": done, "works_not_reached": [w["work"] for w in sample["works"]
                                                                        if w["work"] not in done],
               "spent_usd": round(float(getattr(producer, "spent_usd", 0.0)), 6),
               "calls": int(getattr(producer, "calls", 0)), "wall_seconds": round(clock() - started, 3),
               "modal_stop": stop}
    (out / "arm.json").write_bytes(_dump({"summary": summary, "results": results}))
    return summary
