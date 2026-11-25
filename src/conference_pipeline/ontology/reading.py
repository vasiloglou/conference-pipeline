"""Multi-pass reading host over plain paper text (v6.1 Phase 4, lane R; spec §3, P4-D1, P4-D2).

GND 0.4.0's reading example cannot be driven from a host (no producer hook, one global stop,
fixture-bound replay), so this host re-implements its six stages and writes its formats:
``literature-reading-run/1`` (the run manifest with every artifact's digest),
``stage-N.json`` (stage, purpose, inputs, ``predecessor_sha256``, ``payload_sha256``),
``payload-N.json`` and ``literature-reading-coverage/1`` rows in GND's eight states.

Per document: **inventory** (sections from numbered headings, tables from ``Table n.``
captions, equations from lines ending ``(n)``; fixed windows when the text has no heading)
→ **broad-extraction** (tier A in one call) → **selection** (sections of the run's tiers
whose text contains an evidence term of a question still unanswered) → **targeted-extraction**
(one call per selected section) → **review** (a second model confirms each fact against its
excerpt) → **recheck** (earlier sections that share a key a targeted reply proposed, or that
lost a fact in review, are read again and their new facts reviewed). Tables and equations are
approximate: PDF text keeps a caption line and run-together cells, and equations only as
flattened lines (P4-D5).

A section larger than the document's remaining byte budget is ``omitted``. Each document stops
on its own (P4-D2) with ``required-coverage-completed``, ``budget-exhausted``,
``targeted-pass-added-zero`` or ``available-evidence-exhausted``. The host locates every quote
in its excerpt (a fact whose quote is not there is dropped), so a span never comes from a model.
``replay`` re-derives every stage from the retained responses and makes no call; any digest
mismatch, missing file or changed input is exit 3 with GND's reason strings.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.ontology.gold_label import HEADING, SECURITY, locate_quote, passages
from conference_pipeline.ontology.producer import Producer, Response, outcome_of, parse_json_object

RUN_FORMAT, COVERAGE_FORMAT, REPORT_FORMAT = ("literature-reading-run/1", "literature-reading-coverage/1",
                                              "literature-reading-report/1")
STAGES = ("inventory", "broad-extraction", "selection", "targeted-extraction", "review", "recheck")
PURPOSES = (
    "Identify the document's sections, tables and equations",
    "Read opening material and retain unanswered questions",
    "Select sections for unresolved questions",
    "Recover targeted facts and name keys to revisit",
    "Review every extracted fact against its excerpt",
    "Re-read sections affected by proposals or review",
)
COVERAGE_STATES = ("supported", "explicitly-denied", "inspected-not-found", "unavailable",
                   "not-applicable", "omitted", "producer-refused", "empty-extraction")
STOPS = ("required-coverage-completed", "budget-exhausted", "targeted-pass-added-zero",
         "available-evidence-exhausted")
ROLES = {"extract": "reader", "review": "reviewer"}
DEFAULT_BUDGET = 200_000  # bytes of excerpt per document
# v6.1 Phase 5 (spec lane E, P5-D3): "fact-types" is the Phase 4 reply (one row per fact type);
# "assertions" asks for whole claims in the gold's shape, which the host expands into seed rows
# (``assertions.reading_rows``). The shape is pinned in a run's inputs only when it is not the
# Phase 4 default, so Phase 4 runs replay unchanged.
REPLY_SHAPES = ("fact-types", "assertions")
MAX_SECTIONS_PER_CALL = 6  # 013 decision 2: at most 6 sections per reading call (assertions shape)
POLARITIES, ATTRIBUTIONS = ("asserted", "negated", "speculated"), ("own", "cited")
TIERS = ("A", "B", "C")
_TIER_A = re.compile(r"\b(abstract|introduction|conclusions?|summary|contributions?)\b", re.I)
_TIER_B = re.compile(r"\b(experiments?|experimental|results?|evaluations?|ablations?|benchmarks?|empirical)\b", re.I)
_BARE = re.compile(r"(?im)^[ \t]*(abstract|references|bibliography|acknowledge?ments?|appendix\b[^\n]{0,60})[ \t]*$")
_TABLE = re.compile(r"(?m)^[ \t]*Table[ \t]+[A-Z]?\d+[.:]")
_EQUATION = re.compile(r"(?m)^[^\n]{3,}\(\d{1,3}\)[ \t]*$")
_TERM = re.compile(r'"([^"]+)"')
_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


class ReadingError(ValueError):
    """A replay or input check failed; the message is the reason string (exit 3)."""


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ReadingError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- inputs


def question_terms(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reading questions from a competency-questions file: id, lower-case evidence terms, required (core)."""
    out = []
    for q in document["questions"]:
        if q.get("merged_into"):
            continue
        terms = sorted({t.strip().lower() for quoted in _TERM.findall(q.get("evidence") or "")
                        for t in quoted.split("|") if len(t.strip()) >= 3})
        out.append({"id": q["id"], "text": q["text"], "terms": terms, "required": q.get("priority") == "core"})
    return out


def _tier(heading: str) -> str:
    return "A" if _TIER_A.search(heading) else "B" if _TIER_B.search(heading) else "C"


def inventory(text: str) -> dict[str, Any]:
    """Sections (heading to next heading), tables (caption to next blank line) and equations, with spans."""
    marks = sorted({(m.start(), m.group(0).strip()) for m in [*HEADING.finditer(text), *_BARE.finditer(text)]})
    sections: list[dict[str, Any]] = []
    if not marks:
        for p in passages(text):
            sections.append({"id": f"w{len(sections) + 1:02d}", "kind": "window", "heading": p["section"],
                             "start": p["span"]["start"], "end": p["span"]["end"], "tier": "A"})
        return {"structure": "windows", "sections": sections}
    top: dict[str, str] = {}
    for n, (start, heading) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(text)
        number = heading.split()[0].rstrip(".")
        tier = _tier(heading)
        if number[:1].isdigit():
            if "." not in number:
                top[number] = tier
            elif tier == "C":
                tier = top.get(number.split(".")[0], "C")
        if re.match(r"(?i)references|bibliography", heading):
            tier = "-"  # never read
        sections.append({"id": f"s{n + 1:02d}", "kind": "section", "heading": heading[:120], "start": start,
                         "end": end, "tier": tier})
    if marks[0][0] > 0 and text[:marks[0][0]].strip():
        has_abstract = any(s["heading"].lower() == "abstract" for s in sections)
        sections.insert(0, {"id": "s00", "kind": "section", "heading": "front matter", "start": 0,
                            "end": marks[0][0], "tier": "C" if has_abstract else "A"})
    heads = [s["start"] for s in sections]
    for kind, pattern in (("table", _TABLE), ("equation", _EQUATION)):
        for n, m in enumerate(pattern.finditer(text)):
            start = m.start()
            if kind == "table":
                blank = text.find("\n\n", m.end())
                nxt = [h for h in heads if h > start] + [x.start() for x in _TABLE.finditer(text, m.end())]
                end = min([blank if blank >= 0 else len(text), *nxt, len(text)])
            else:
                end = m.end()
            sections.append({"id": f"{kind[0]}{n + 1:02d}", "kind": kind, "heading": m.group(0).strip()[:120],
                             "start": start, "end": end, "tier": "B" if kind == "table" else "C"})
    return {"structure": "headings", "sections": sections}


# --------------------------------------------------------------------------- prompts and replies


def _excerpts(text: str, sections: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(f"[{s['id']}] {s['heading']}\n<<<\n{text[s['start']:s['end']]}\n>>>" for s in sections)


def extract_prompt(text: str, sections: Sequence[Mapping[str, Any]], fact_types: Sequence[str],
                   questions: Sequence[Mapping[str, Any]] = (), shape: str = "fact-types") -> str:
    asked = ("\n\nThese questions are still unanswered for this paper; prefer facts that answer them:\n"
             + "\n".join(f"- {q['id']}: {q['text']}" for q in questions)) if questions else ""
    if shape == "assertions":
        return (f"{SECURITY}\n\nRead the excerpts below from one research paper. Report each claim the paper "
                "states: what it says a method, model, dataset, benchmark, metric or other named thing does, "
                "achieves, outperforms, requires or lacks. Each claim is stored as these fact types (ids): "
                + ", ".join(fact_types) + ". For each claim give the section id it comes from; an exact quote "
                "of at most two sentences from that section that states it; the predicate in a few words "
                "copied from the quote; its participants, each a name copied exactly from the quote with a "
                "short role such as method, baseline, dataset, metric, model, subject or object; conditions "
                "under which it holds, each copied exactly from the quote; polarity (asserted, negated or "
                "speculated); attribution (own for the paper's own claim, cited for a claim it reports from "
                "other work); and the experiment it is measured in, if the quote names one. Report only what "
                "the excerpt says. Also list proposal_keys: short lower-case words naming concepts an earlier "
                "section should be re-read for." + asked +
                '\n\nReturn {"assertions": [{"section", "quote", "predicate", "participants": [{"name", "role"}], '
                '"conditions": [], "polarity", "attribution", "experiment"}], "proposal_keys": []}.\n\n'
                + _excerpts(text, sections))
    return (f"{SECURITY}\n\nRead the excerpts below from one research paper. Report each fact the paper states, "
            "as one of these fact types (use the id exactly): " + ", ".join(fact_types) + ". For each fact give "
            "the section id it comes from and an exact quote of at most two sentences from that section that "
            "states it. Report only what the excerpt says. Also list proposal_keys: short lower-case words naming "
            "concepts an earlier section should be re-read for." + asked +
            ' For roles, give each role of the fact type its value copied exactly from the quote.'
            '\n\nReturn {"facts": [{"fact_type", "section", "roles": {"<role name>": "<exact text>"}, "predicate", '
            '"participants": [{"role", "name"}], "polarity": "asserted|negated|speculated", "quote"}], '
            '"proposal_keys": []}.\n\n' + _excerpts(text, sections))


def review_prompt(text: str, facts: Sequence[Mapping[str, Any]], by_id: Mapping[str, Mapping[str, Any]]) -> str:
    rows = "\n".join(json.dumps({"id": f["id"], "fact_type": f["fact_type"], "roles": f.get("roles") or {},
                                 "predicate": f.get("predicate"),
                                 "participants": f.get("participants"), "polarity": f.get("polarity"),
                                 **{k: f[k] for k in ("conditions", "attribution") if k in f},
                                 "quote": f["quote"], "section": f["section"]}, ensure_ascii=False) for f in facts)
    used = [by_id[s] for s in sorted({f["section"] for f in facts})]
    return (f"{SECURITY}\n\nAn extractor reported the facts below from a research paper. For each id, judge only "
            "whether the quoted excerpt supports the fact as stated (roles, participants, predicate and polarity). "
            'Return {"verdicts": [{"id", "verdict": "supported|unsupported", "why"}]}.\n\nFacts:\n' + rows +
            "\n\nExcerpts:\n" + _excerpts(text, used))


def _response(response: Response) -> dict[str, Any]:
    return {"text": response.text, "stop_reason": response.stop_reason, "provider": response.provider,
            "model": response.model}


_SOFT_BREAK = re.compile(r"-[ \t]*\n\s*")
_SPACE = re.compile(r"\s+")


def flat(value: str) -> str:
    """Text as compared for a claim: PDF line-break hyphens and all whitespace removed, casefolded.
    Paper text keeps its PDF line breaks ("re-\\ncent", "more\\neasily"); a model quotes it clean
    (v6.1 Phase 5 pilot, 2026-11-21: most claims were dropped for this alone)."""
    return _SPACE.sub("", _SOFT_BREAK.sub("", value)).casefold()


# Words a predicate may add or drop without changing what it says (P5-D10). Negations are
# deliberately absent: they are content, and NEGATIONS treats them separately.
PREDICATE_STOP_WORDS = frozenset(
    "a an the of to in on at by for with from over under across into onto than as and or but while whereas "
    "is are was were be been being has have had do does did it its this that these those their them they "
    "which who whom whose when where via per against between among about up out off so such also both "
    "each any all some more most".split())
# Negations, after contractions are replaced by their words ("doesn't" and "cannot" give "not").
# PR 50 review F7: a predicate's negation must be a word of the quote, never matched through a stem.
NEGATIONS = frozenset("not no never nor neither none nothing nobody nowhere without".split())


# UAX #29 lists the colons in MidLetter (for Swedish "c:a") and notes that implementations may tailor
# them out; here they always separate words, since PDF text glues "Harmfulness:measured".
_COLONS_SEPARATE = str.maketrans({":": " ", "\ufe13": " ", "\ufe55": " ", "\uff1a": " "})


@functools.cache
def _contractions() -> dict[str, tuple[str, ...]]:
    """English contractions and their normalized words, from spaCy's tokenizer exception table
    ("doesn't" -> does, not; "can't" -> can, not; "won't" -> will, not); casefolded."""
    from spacy.lang.en import English
    from spacy.symbols import NORM, ORTH

    return {text.casefold(): tuple((piece.get(NORM) or piece[ORTH]).casefold() for piece in pieces)
            for text, pieces in English.Defaults.tokenizer_exceptions.items() if len(pieces) > 1}


@functools.cache
def _stemmer() -> Any:
    """NLTK's Snowball (Porter 2) English stemmer; it needs no downloaded data."""
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("english")


def _words(value: str) -> list[str]:
    """The casefolded words of ``value``, PDF line-break hyphens joined first. Words are Unicode
    default word boundaries (UAX #29, the ``regex`` module's V1 word mode), which also split the
    punctuation PDF extraction glues to words ("4.5)fails", "1.Performance"), with colons tailored
    to separate (``_COLONS_SEPARATE``); a contraction is replaced by its words (``_contractions``)."""
    import regex

    words: list[str] = []
    for segment in regex.split(r"(?V1w)\b", _SOFT_BREAK.sub("", value).translate(_COLONS_SEPARATE)):
        if any(ch.isalnum() for ch in segment):
            words.extend(_contractions().get(segment.casefold(), (segment.casefold(),)))
    return words


def predicate_in_quote(predicate: str, quote: str) -> bool:
    """P5-D10 (owner, 2026-11-22): every content word of the predicate is a word of the quote, up to
    its ending ("uses" for "use", "restores" for "restore"); small words are ignored, so a model may
    compress the phrasing but may not bring in a verb or noun the sentence does not have. Endings are
    removed by the Snowball stemmer. As P5-D10 allows, two stems of five letters or more also match
    when one begins the other, because extracted PDF text glues words ("proposeSafe-Ablated").
    A negation must be in the quote as a word of its own, never through a stem (PR 50 review F7)."""
    words = _words(quote)
    stems = {_stem(w) for w in words}
    for word in _words(predicate):
        if word in PREDICATE_STOP_WORDS:
            continue
        if not (word in words if word in NEGATIONS else _stem_in(_stem(word), stems)):
            return False
    return True


def _stem(word: str) -> str:
    return _stemmer().stem(word)


def _stem_in(stem: str, stems: set[str]) -> bool:
    return stem in stems or (len(stem) >= 5 and any(len(s) >= 5 and (s.startswith(stem) or stem.startswith(s))
                                                     for s in stems))


def _read_assertions(payload: Mapping[str, Any], text: str, by_id: Mapping[str, Mapping[str, Any]], prefix: str
                     ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Claims in the gold's shape; every name and condition must be in the located quote, and every
    content word of the predicate (``predicate_in_quote``)."""
    facts, drops = [], []
    items = payload.get("assertions") if isinstance(payload.get("assertions"), list) else []
    for n, item in enumerate(items):
        ident = f"{prefix}{n + 1:02d}"
        section = by_id.get(item.get("section")) if isinstance(item, Mapping) else None
        span = locate_quote(text, {"start": section["start"], "end": section["end"]}, item.get("quote")) \
            if section else None
        if span is None:
            drops.append({"fact": ident, "outcome": "quote-not-in-section"})
            continue
        quote = text[span["start"]:span["end"]]
        people = item.get("participants") if isinstance(item.get("participants"), list) else []
        participants = [{"name": p["name"].strip(), "role": p["role"].strip()} for p in people
                        if isinstance(p, Mapping) and isinstance(p.get("name"), str) and p["name"].strip()
                        and isinstance(p.get("role"), str) and p["role"].strip()]
        conditions = [c.strip() for c in item.get("conditions") or [] if isinstance(c, str) and c.strip()] \
            if isinstance(item.get("conditions"), list) else []
        predicate = item["predicate"].strip() if isinstance(item.get("predicate"), str) else ""
        if not participants or len(participants) != len(people):
            drops.append({"fact": ident, "outcome": "no-participants" if not participants else "invalid-participant"})
            continue
        if any(flat(v) not in flat(quote) for v in [*(p["name"] for p in participants), *conditions] if v) \
                or not predicate_in_quote(predicate, quote):
            drops.append({"fact": ident, "outcome": "role-not-in-quote"})
            continue
        if item.get("polarity") not in POLARITIES or item.get("attribution") not in ATTRIBUTIONS:
            drops.append({"fact": ident, "outcome": "invalid-value"})
            continue
        experiment = item["experiment"].strip() if isinstance(item.get("experiment"), str) else ""
        facts.append({"id": ident, "fact_type": "assertion", "section": section["id"], "predicate": predicate or None,
                      "participants": participants, "conditions": conditions, "roles": {},
                      "polarity": item["polarity"], "attribution": item["attribution"], "experiment": experiment or None,
                      "quote": quote, "span": span})
    return facts, drops


def read_facts(raw: Mapping[str, Any], text: str, sections: Sequence[Mapping[str, Any]], fact_types: Sequence[str],
               prefix: str, shape: str = "fact-types") -> tuple[str, list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """(outcome, facts with host-located spans, per-fact drops, proposal keys) from one extraction reply."""
    stop = outcome_of(Response(**raw))
    if stop:
        return stop, [], [], []
    payload = parse_json_object(raw["text"])
    if payload is None:
        return "unparsable", [], [], []
    if shape == "assertions":
        facts, drops = _read_assertions(payload, text, {s["id"]: s for s in sections}, prefix)
        keys = sorted({k.strip().lower() for k in payload.get("proposal_keys") or [] if isinstance(k, str) and k.strip()})
        return ("accepted" if facts else "empty"), facts, drops, keys
    by_id = {s["id"]: s for s in sections}
    facts, drops = [], []
    for n, item in enumerate(payload.get("facts") if isinstance(payload.get("facts"), list) else []):
        ident = f"{prefix}{n + 1:02d}"
        if not isinstance(item, Mapping) or item.get("fact_type") not in fact_types:
            drops.append({"fact": ident, "outcome": "unknown-fact-type"})
            continue
        section = by_id.get(item.get("section"))
        span = locate_quote(text, {"start": section["start"], "end": section["end"]}, item.get("quote")) \
            if section else None
        if span is None:
            drops.append({"fact": ident, "outcome": "quote-not-in-section"})
            continue
        quote = text[span["start"]:span["end"]]
        roles = {str(k): v for k, v in item["roles"].items() if isinstance(v, str) and v} \
            if isinstance(item.get("roles"), Mapping) else {}
        if any(v.casefold() not in quote.casefold() for v in roles.values()):
            # review F2: a role value the quote does not hold would become an unreviewed row
            drops.append({"fact": ident, "outcome": "role-not-in-quote"})
            continue
        facts.append({"id": ident, "fact_type": item["fact_type"], "section": section["id"],
                      "predicate": item.get("predicate"), "participants": item.get("participants") or [],
                      "roles": roles,
                      "polarity": item.get("polarity") if item.get("polarity") in ("asserted", "negated", "speculated")
                      else "asserted", "quote": quote, "span": span})
    keys = sorted({k.strip().lower() for k in payload.get("proposal_keys") or [] if isinstance(k, str) and k.strip()})
    outcome = "accepted" if facts else "empty"
    return outcome, facts, drops, keys


def read_verdicts(raw: Mapping[str, Any], facts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """fact id -> supported | unsupported | missing (a refused or unreadable review leaves every fact missing)."""
    out = dict.fromkeys((f["id"] for f in facts), "missing")
    if outcome_of(Response(**raw)):
        return out
    payload = parse_json_object(raw["text"]) or {}
    answers: dict[str, list[str]] = {}
    for v in payload.get("verdicts") if isinstance(payload.get("verdicts"), list) else []:
        if isinstance(v, Mapping) and v.get("id") in out and v.get("verdict") in ("supported", "unsupported"):
            answers.setdefault(v["id"], []).append(v["verdict"])
    for ident, said in answers.items():  # review F7: more than one answer, of any count: none stands
        out[ident] = said[0] if len(said) == 1 else "missing"
    return out


# --------------------------------------------------------------------------- one document


class _Calls:
    """Wraps a producer: counts calls per stage, records every response for the payload."""

    def __init__(self, producer: Producer, document: str):
        self.producer, self.document = producer, document
        self.counts = {STAGES[i]: 0 for i in (1, 3, 4, 5)}

    def __call__(self, stage: str, kind: str, key: str, prompt: str) -> dict[str, Any]:
        if not isinstance(self.producer, _Retained):  # a replay answers from retained evidence: no call
            self.counts[stage] += 1
        role = ROLES["review"] if kind.endswith("review") else ROLES["extract"]
        return _response(self.producer(role, kind, f"{self.document}#{key}", prompt))


def _keys(heading: str) -> set[str]:
    return set(_WORD.findall(heading.lower()))


def _matches(fact: Mapping[str, Any], terms: Sequence[str]) -> bool:
    hay = f"{fact.get('quote', '')} {fact.get('predicate', '')}".lower()
    return any(t in hay for t in terms)


def read_document(document: str, text: str | None, *, questions: Sequence[Mapping[str, Any]],
                  fact_types: Sequence[str], producer: Producer, budget: int = DEFAULT_BUDGET,
                  tiers: Sequence[str] = ("A", "B"), shape: str = "fact-types"
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    """The six stage payloads, the coverage rows and the document's report."""
    calls = _Calls(producer, document)
    if text is None:
        rows = [{"document": document, "question": q["id"], "section": None, "state": "unavailable",
                 "inspected": False} for q in questions]
        payloads = [{"document": document, "unavailable": True}, {"attempts": [], "omitted": []},
                    {"selected": [], "unresolved": sorted(q["id"] for q in questions)},
                    {"attempts": [], "omitted": []}, {"verdicts": {}}, {"attempts": [], "omitted": [], "verdicts": {}}]
        return payloads, rows, {"document": document, "stop": "available-evidence-exhausted",
                                "producer_calls": calls.counts, "facts": 0, "reading_bytes": 0}
    inv = inventory(text)
    by_id = {s["id"]: s for s in inv["sections"]}
    left = budget
    attempted: dict[str, str] = {}  # section -> outcome of its last read
    omitted: list[str] = []

    def take(chosen: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        nonlocal left
        kept = []
        for s in chosen:
            size = len(text[s["start"]:s["end"]].encode("utf-8"))
            if size > left:
                omitted.append(s["id"])
                continue
            left -= size
            kept.append(s)
        return kept

    def extract(stage: str, kind: str, group: Sequence[Mapping[str, Any]], prefix: str, asked=()) -> dict[str, Any]:
        raw = calls(stage, kind, "+".join(s["id"] for s in group), extract_prompt(text, group, fact_types, asked, shape))
        outcome, facts, drops, keys = read_facts(raw, text, group, fact_types, prefix, shape)
        for s in group:
            attempted[s["id"]] = outcome
        return {"sections": [s["id"] for s in group], "excerpt_sha256": sha256(_excerpts(text, group).encode()),
                "response": raw, "outcome": outcome, "facts": facts, "dropped": drops, "proposal_keys": keys}

    # stage 1: broad (tier A, one call)
    opening = take([s for s in inv["sections"] if s["tier"] == "A" and "A" in tiers
                    and s["kind"] in ("section", "window")])  # review F8: the run's tiers bind stage 1 too
    if shape == "assertions":  # spec T-E5: at most MAX_SECTIONS_PER_CALL sections per broad call
        groups = [opening[i:i + MAX_SECTIONS_PER_CALL] for i in range(0, len(opening), MAX_SECTIONS_PER_CALL)]
    else:
        groups = [opening] if opening else []
    broad = {"attempts": [extract(STAGES[1], "broad", g, "b" if k == 0 else f"b{k + 1}-") for k, g in enumerate(groups)],
             "omitted": list(omitted)}
    facts = [f for a in broad["attempts"] for f in a["facts"]]

    # stage 2: selection
    def unresolved(found):
        return [q for q in questions if q["terms"] and not any(_matches(f, q["terms"]) for f in found)]
    open_qs = unresolved(facts)
    selected = [s for s in inv["sections"] if s["tier"] in tiers and s["tier"] != "A" and s["id"] not in attempted
                and any(t in text[s["start"]:s["end"]].lower() for q in open_qs for t in q["terms"])]
    selection = {"selected": [s["id"] for s in selected], "unresolved": [q["id"] for q in open_qs]}

    # stage 3: targeted (one call per section)
    before = len(omitted)
    targeted = {"attempts": [], "omitted": []}
    for s in take(selected):
        asked = [q for q in open_qs if any(t in text[s["start"]:s["end"]].lower() for t in q["terms"])]
        targeted["attempts"].append(extract(STAGES[3], "targeted", [s], f"t{s['id']}-", asked))
    targeted["omitted"] = omitted[before:]
    targeted_facts = [f for a in targeted["attempts"] for f in a["facts"]]

    # stage 4: review
    def review(stage: str, kind: str, found: list[dict[str, Any]]) -> dict[str, str]:
        if not found:
            return {}
        raw = calls(stage, kind, "review" if kind == "review" else "recheck-review", review_prompt(text, found, by_id))
        return {"response": raw, "verdicts": read_verdicts(raw, found)}  # type: ignore[return-value]
    reviewed = review(STAGES[4], "review", facts + targeted_facts)
    verdicts = dict(reviewed.get("verdicts", {}))

    # stage 5: recheck
    proposed = {k for a in targeted["attempts"] for k in a["proposal_keys"]}
    lost = {f["section"] for f in facts + targeted_facts if verdicts.get(f["id"]) == "unsupported"}
    # review F6 (R8): a section whose read was refused, empty, truncated or unparsable is not retried
    early = [s for s in opening if attempted.get(s["id"]) == "accepted"
             and ((_keys(s["heading"]) & proposed) or s["id"] in lost)]
    before = len(omitted)
    recheck: dict[str, Any] = {"attempts": [], "omitted": []}
    for s in take(early):
        recheck["attempts"].append(extract(STAGES[5], "recheck", [s], f"r{s['id']}-"))
    recheck["omitted"] = omitted[before:]
    recheck_facts = [f for a in recheck["attempts"] for f in a["facts"]]
    rechecked = review(STAGES[5], "recheck-review", recheck_facts)
    recheck["review"] = rechecked
    verdicts.update(rechecked.get("verdicts", {}))

    final = [f for f in facts + targeted_facts + recheck_facts if verdicts.get(f["id"]) == "supported"]
    rows = []
    for q in questions:
        hit = next((f for f in final if _matches(f, q["terms"])), None)
        relevant = [s for s in inv["sections"] if q["terms"]
                    and any(t in text[s["start"]:s["end"]].lower() for t in q["terms"])]
        read = [s["id"] for s in relevant if s["id"] in attempted]
        if hit:
            state, section = ("explicitly-denied" if hit["polarity"] == "negated" else "supported"), hit["section"]
        elif not relevant:
            state, section = "not-applicable", None
        elif not read:
            state, section = ("omitted" if any(s["id"] in omitted for s in relevant) else "not-applicable",
                              relevant[0]["id"])
        else:
            outs = {attempted[i] for i in read}
            section = read[0]
            state = ("producer-refused" if outs == {"refused"} else "empty-extraction"
                     if outs <= {"empty", "truncated", "unparsable", "missing"} and "accepted" not in outs
                     else "inspected-not-found")
        rows.append({"document": document, "question": q["id"], "section": section, "state": state,
                     "inspected": bool(read) or hit is not None})
    resolved = {r["question"] for r in rows if r["state"] in ("supported", "explicitly-denied")}
    required = {q["id"] for q in questions if q["required"]}
    if required <= resolved:
        stop = "required-coverage-completed"
    elif omitted:
        stop = "budget-exhausted"
    elif selected and not any(verdicts.get(f["id"]) == "supported" for f in targeted_facts):
        stop = "targeted-pass-added-zero"
    else:
        stop = "available-evidence-exhausted"
    payloads = [{"document": document, **inv}, broad, selection, targeted,
                {"response": reviewed.get("response"), "verdicts": reviewed.get("verdicts", {})}, recheck]
    report = {"document": document, "stop": stop, "producer_calls": calls.counts, "facts": len(final),
              "reading_bytes": budget - left, "omitted": omitted,
              "attempt_counts": {k: sum(a["outcome"] == k for p in (broad, targeted, recheck) for a in p["attempts"])
                                 for k in ("accepted", "empty", "refused", "truncated", "unparsable", "missing")}}
    return payloads, rows, report


# --------------------------------------------------------------------------- a run


def _pins(documents: Mapping[str, str | None], questions, fact_types, budget, tiers, shape="fact-types") -> dict[str, Any]:
    pins = {"documents": {d: (sha256(t.encode()) if t is not None else None) for d, t in sorted(documents.items())},
            "questions_sha256": sha256(canonical(list(questions))), "fact_types_sha256": sha256(canonical(list(fact_types))),
            "budget_bytes": budget, "tiers": list(tiers)}
    if shape != "fact-types":
        pins["reply_shape"] = shape
    return pins


def _evaluate(documents, questions, fact_types, producer, budget, tiers, shape="fact-types"):
    artifacts: dict[str, bytes] = {}
    coverage, reports = [], []
    pins = _pins(documents, questions, fact_types, budget, tiers, shape)
    for document in sorted(documents):
        payloads, rows, report = read_document(document, documents[document], questions=questions,
                                               fact_types=fact_types, producer=producer, budget=budget, tiers=tiers,
                                               shape=shape)
        predecessor = None
        for i, (stage, purpose, payload) in enumerate(zip(STAGES, PURPOSES, payloads, strict=True)):
            raw = artifacts[f"documents/{document}/payload-{i}.json"] = canonical(payload)
            record = {"stage": stage, "purpose": purpose, "document": document, "inputs": pins,
                      "predecessor_sha256": predecessor, "payload_sha256": sha256(raw),
                      "inspected": [sid for a in payload.get("attempts", []) for sid in a["sections"]]}
            raw = artifacts[f"documents/{document}/stage-{i}.json"] = canonical(record)
            predecessor = sha256(raw)
        coverage += rows
        reports.append(report)
    artifacts["coverage.json"] = canonical({"format": COVERAGE_FORMAT, "rows": coverage})
    manifest = {"format": RUN_FORMAT, "inputs": pins, "artifacts": {k: sha256(v) for k, v in sorted(artifacts.items())}}
    report = {"format": REPORT_FORMAT, "documents": reports, "coverage": coverage,
              "producer_calls": {s: sum(r["producer_calls"][s] for r in reports) for s in (STAGES[1], STAGES[3],
                                                                                         STAGES[4], STAGES[5])},
              "stops": {s: sum(r["stop"] == s for r in reports) for s in STOPS},
              "evidence_sha256": sha256(canonical(manifest))}
    return manifest, artifacts, report


def run(out: Path, documents: Mapping[str, str | None], *, questions, fact_types, producer: Producer,
        budget: int = DEFAULT_BUDGET, tiers: Sequence[str] = ("A", "B"), shape: str = "fact-types") -> dict[str, Any]:
    """Read every document and retain the complete evidence under ``out`` (which must not exist)."""
    _need(not out.exists() and not out.is_symlink(), "reading output already exists")
    _need(type(budget) is int and budget >= 0 and set(tiers) <= set(TIERS) and shape in REPLY_SHAPES,
          "invalid reading settings")
    manifest, artifacts, report = _evaluate(documents, questions, fact_types, producer, budget, tuple(tiers), shape)
    for name, raw in artifacts.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(raw)
    (out / "run.json").write_bytes(canonical(manifest))
    return report


class _Retained:
    """Answers each call from the retained payloads; counts no producer call; refuses an unknown call."""

    calls = 0

    def __init__(self, responses: Mapping[tuple[str, str], Mapping[str, Any]]):
        self.responses = responses

    def __call__(self, role: str, kind: str, work: str, prompt: str) -> Response:
        raw = self.responses.get((kind, work))
        _need(raw is not None, "retained producer response missing")
        return Response(**raw)  # type: ignore[arg-type]


def _retained(run_dir: Path, manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    responses: dict[tuple[str, str], Mapping[str, Any]] = {}
    for document in manifest["inputs"]["documents"]:
        def load(i):
            return json.loads((run_dir / f"documents/{document}/payload-{i}.json").read_bytes())
        for i, kind in ((1, "broad"), (3, "targeted"), (5, "recheck")):
            for a in load(i)["attempts"]:
                responses[(kind, f"{document}#{'+'.join(a['sections'])}")] = a["response"]
        for i, kind, key in ((4, "review", "review"), (5, "recheck-review", "recheck-review")):
            payload = load(i)
            raw = payload.get("response") if i == 4 else (payload.get("review") or {}).get("response")
            if raw:
                responses[(kind, f"{document}#{key}")] = raw
    return responses


def replay(run_dir: Path, documents: Mapping[str, str | None], *, questions, fact_types,
           reply_shape: str | None = None) -> dict[str, Any]:
    """Re-derive the run from its retained responses with zero producer calls; ``ReadingError`` on any drift."""
    try:
        raw_manifest = (run_dir / "run.json").read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, ValueError) as exc:
        raise ReadingError("incomplete stage roster") from exc
    _need(manifest.get("format") == RUN_FORMAT and raw_manifest == canonical(manifest), "noncanonical reading manifest")
    for name, digest in manifest["artifacts"].items():
        path = run_dir / name
        _need(path.is_file(), "incomplete stage roster")
        _need(sha256(path.read_bytes()) == digest, "stale retained reading stage" if "/stage-" in name
              else "stale retained reading output")
    inputs = manifest["inputs"]
    shape = inputs.get("reply_shape", "fact-types")
    _need(reply_shape in (None, shape), "stale reading input")  # spec T-E4: the caller's shape is the run's
    _need(inputs == _pins(documents, questions, fact_types, inputs["budget_bytes"], inputs["tiers"], shape),
          "stale reading input")
    observed, artifacts, report = _evaluate(documents, questions, fact_types, _Retained(_retained(run_dir, manifest)),
                                            inputs["budget_bytes"], tuple(inputs["tiers"]), shape)
    _need(observed == manifest and all((run_dir / k).read_bytes() == v for k, v in artifacts.items()),
          "re-derived reading evidence differs")
    return report


# --------------------------------------------------------------------------- command line


def _load_inputs(args) -> tuple[dict[str, str | None], list, list]:
    documents: dict[str, str | None] = {}
    for line in Path(args.documents).read_text(encoding="utf-8").splitlines():  # one "<id>\t<path>" per line
        if line.strip():
            ident, _, path = line.partition("\t")
            documents[ident] = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else None
    questions = question_terms(json.loads(Path(args.questions).read_text(encoding="utf-8")))
    fact_types = json.loads(Path(args.fact_types).read_text(encoding="utf-8"))
    return documents, questions, fact_types


def main(argv: list[str] | None = None) -> int:
    """``run`` (saved responses only in Phase 4) or ``replay``; exit 0, 3 (incomplete) or 64 (usage)."""
    from conference_pipeline.ontology.producer import SavedProducer

    parser = argparse.ArgumentParser(prog="python -m conference_pipeline.ontology.reading", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "replay"):
        sub = commands.add_parser(name)
        sub.add_argument("--documents", required=True, help="file of '<id>\\t<text path>' lines")
        sub.add_argument("--questions", required=True)
        sub.add_argument("--fact-types", required=True, help="JSON list of fact type ids")
        if name == "run":
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--responses", type=Path, required=True)
            sub.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET)
            sub.add_argument("--tiers", default="A,B")
        else:
            sub.add_argument("run_dir", type=Path)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 64
    try:
        documents, questions, fact_types = _load_inputs(args)
        if args.command == "run":
            result = run(args.out, documents, questions=questions, fact_types=fact_types,
                         producer=SavedProducer(args.responses), budget=args.budget_bytes,
                         tiers=tuple(t for t in args.tiers.split(",") if t))
        else:
            result = replay(args.run_dir, documents, questions=questions, fact_types=fact_types)
    except (ReadingError, OSError, ValueError, KeyError, TypeError) as exc:
        sys.stdout.buffer.write(canonical({"status": "incomplete", "reason": str(exc)}))
        return 3
    sys.stdout.buffer.write(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
