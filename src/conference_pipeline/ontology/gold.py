"""Schema-neutral gold statements and the dev / sealed split pairs (v6.1 Phase 3, 6.2 and 6.5).

A gold file (``ontology-gold/1``) holds one work's passages and statements in plain words:
participants with roles, a predicate, conditions, an experiment, an exact quantity
(Integer coefficient and scale), polarity, attribution and one of GND's four support
labels, each with a character span in the work's text. A links file
(``ontology-gold-links/1``) holds one split's cross-work labels. ``build_split_pair``
writes one split's ``ontology-split-corpus/1`` (documents and passages) and gold bundle;
the two splits share nothing (``pair_problems``). ``sealed_guard`` makes any in-process
read under the sealed root fail.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

GOLD_SCHEMA = "ontology-gold/1"
LINKS_SCHEMA = "ontology-gold-links/1"
SPLIT_CORPUS_SCHEMA = "ontology-split-corpus/1"
SPLITS = ("dev", "sealed_test")
SUPPORT = frozenset({"supported", "stretched", "contradicted", "insufficient-evidence"})
POLARITY = frozenset({"asserted", "negated", "speculated"})
ATTRIBUTION = frozenset({"own", "cited"})
IDENTITY_LABELS = frozenset({"same", "different"})
NAME_KINDS = frozenset({"homonym", "synonym", "family"})
CHAIN_LABELS = frozenset({"supported", "false"})


class SealedReadError(PermissionError):
    """An in-process read touched the sealed root."""


def _span_ok(span: Any, length: int) -> bool:
    return (isinstance(span, Mapping) and type(span.get("start")) is int and type(span.get("end")) is int
            and 0 <= span["start"] < span["end"] <= length)


def _inside(inner: Mapping[str, int], outer: Mapping[str, int]) -> bool:
    return outer["start"] <= inner["start"] and inner["end"] <= outer["end"]


def quantity_problems(quantity: Any) -> list[str]:
    """A quantity is ``{"coefficient": int, "scale": int >= 0, "unit": str}`` (plan §2.6)."""
    if quantity is None:
        return []
    if not isinstance(quantity, Mapping):
        return ["quantity is not an object"]
    problems = []
    if type(quantity.get("coefficient")) is not int:
        problems.append("quantity coefficient must be an Integer")
    if type(quantity.get("scale")) is not int or quantity.get("scale", -1) < 0:
        problems.append("quantity scale must be a non-negative Integer")
    if not isinstance(quantity.get("unit"), str) or not quantity.get("unit"):
        problems.append("quantity unit must be named")
    return problems


def quantity_value(quantity: Mapping[str, Any]) -> tuple[int, int]:
    """The exact value as (numerator, denominator): coefficient / 10**scale."""
    return quantity["coefficient"], 10 ** quantity["scale"]


def gold_problems(gold: Mapping[str, Any], text: str) -> list[str]:
    """Empty when ids are unique, spans are inside the text and their passage, enums valid."""
    problems: list[str] = []
    if gold.get("schema") != GOLD_SCHEMA:
        problems.append(f"schema is not {GOLD_SCHEMA}")
    if gold.get("split") not in SPLITS:
        problems.append(f"unknown split {gold.get('split')!r}")
    passages = {}
    for passage in gold.get("passages") or []:
        pid = passage.get("id") if isinstance(passage, Mapping) else None
        if pid in passages:
            problems.append(f"passage {pid} is duplicated")
        if not _span_ok(passage.get("span") if isinstance(passage, Mapping) else None, len(text)):
            problems.append(f"passage {pid} span is outside the text")
            continue
        passages[pid] = passage["span"]
    seen: set[str] = set()
    for statement in gold.get("statements") or []:
        sid = statement.get("id")
        if sid in seen:
            problems.append(f"statement {sid} is duplicated")
        seen.add(sid)
        span, pid = statement.get("span"), statement.get("passage")
        if pid not in passages:
            problems.append(f"statement {sid} names unknown passage {pid}")
        elif not _span_ok(span, len(text)):
            problems.append(f"statement {sid} span is outside the text")
        elif not _inside(span, passages[pid]):
            problems.append(f"statement {sid} span is outside passage {pid}")
        if statement.get("support") not in SUPPORT:
            problems.append(f"statement {sid} support {statement.get('support')!r} is not one of GND's four")
        if statement.get("polarity") not in POLARITY:
            problems.append(f"statement {sid} polarity {statement.get('polarity')!r} is invalid")
        if statement.get("attribution") not in ATTRIBUTION:
            problems.append(f"statement {sid} attribution {statement.get('attribution')!r} is invalid")
        if not isinstance(statement.get("predicate"), str) or not statement["predicate"].strip():
            problems.append(f"statement {sid} has no predicate")
        participants = statement.get("participants")
        if not isinstance(participants, list) or not participants or not all(
                isinstance(p, Mapping) and p.get("role") and p.get("name") for p in participants):
            problems.append(f"statement {sid} needs participants with a role and a name")
        problems += [f"statement {sid}: {p}" for p in quantity_problems(statement.get("quantity"))]
    return problems


def links_problems(links: Mapping[str, Any], works: Mapping[str, str]) -> list[str]:
    """``works`` maps work key -> split; every link stays inside the links file's split."""
    problems: list[str] = []
    split = links.get("split")
    if links.get("schema") != LINKS_SCHEMA:
        problems.append(f"schema is not {LINKS_SCHEMA}")

    def check(owner: str, work: Any) -> None:
        if work not in works:
            problems.append(f"{owner} names unknown work {work!r}")
        elif works[work] != split:
            problems.append(f"{owner} names {work}, which is in {works[work]}, not {split}")

    for pair in links.get("identity_pairs") or []:
        if pair.get("label") not in IDENTITY_LABELS:
            problems.append(f"identity pair {pair.get('id')} label is invalid")
        for side in ("left", "right"):
            check(f"identity pair {pair.get('id')}", (pair.get(side) or {}).get("work"))
    for pair in links.get("name_pairs") or []:
        if pair.get("kind") not in NAME_KINDS:
            problems.append(f"name pair {pair.get('id')} kind is invalid")
    for chain in links.get("chains") or []:
        if chain.get("label") not in CHAIN_LABELS:
            problems.append(f"chain {chain.get('id')} label is invalid")
        for work in chain.get("works") or []:
            check(f"chain {chain.get('id')}", work)
    return problems


# --------------------------------------------------------------------------- split pairs


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _doc_id(work: str) -> str:
    """A GND-safe document id for a work key (``[A-Za-z0-9][A-Za-z0-9._-]*``)."""
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in work).strip("-.")
    return out[:128] or "work"


def build_split_pair(out: Path, split: str, golds: Iterable[tuple[Mapping[str, Any], bytes, str]],
                     links: Mapping[str, Any]) -> dict[str, str]:
    """Write one split's corpus and gold bundle under ``out``.

    ``golds`` holds (gold file, the work's text bytes, its area). Returns the digests.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "documents").mkdir()
    (out / "gold").mkdir()
    documents, passages = [], []
    for gold, raw, area in sorted(golds, key=lambda g: g[0]["work"]):
        if gold.get("split") != split:
            raise ValueError(f"{gold.get('work')} is in {gold.get('split')}, not {split}")
        doc = _doc_id(str(gold["work"]))
        (out / "documents" / f"{doc}.md").write_bytes(raw)
        documents.append({"id": doc, "path": f"documents/{doc}.md", "sha256": _sha(raw), "domain": area,
                          "work": {"identifier": gold["work"]}})
        passages += [{"id": f"{doc}.{p['id']}", "document": doc, "span": p["span"]} for p in gold["passages"]]
        (out / "gold" / f"{doc}.gold.json").write_bytes(_dump(gold))
    corpus_raw = _dump({"schema": SPLIT_CORPUS_SCHEMA, "split": split, "documents": documents, "passages": passages})
    (out / "corpus.json").write_bytes(corpus_raw)
    links_raw = _dump(links)
    (out / "links.json").write_bytes(links_raw)
    bundle = hashlib.sha256()
    for path in sorted(out.rglob("*")):
        if path.is_file():
            bundle.update(path.relative_to(out).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return {"split": split, "corpus_sha256": _sha(corpus_raw), "links_sha256": _sha(links_raw),
            "bundle_sha256": bundle.hexdigest()}


def _pair_contents(root: Path) -> dict[str, set[str]]:
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    links = json.loads((root / "links.json").read_text(encoding="utf-8"))
    return {
        "document": {d["sha256"] for d in corpus["documents"]},
        "work": {d["work"]["identifier"] for d in corpus["documents"]},
        "passage": {p["id"] for p in corpus["passages"]},
        "label group": {f"{group}:{item.get('id')}" for group in ("identity_pairs", "name_pairs", "chains", "reuse_cases")
                        for item in links.get(group) or []},
    }


def pair_problems(dev: Path, sealed: Path) -> list[str]:
    """Empty when the two split pairs share no document, work, passage or label group."""
    a, b = _pair_contents(dev), _pair_contents(sealed)
    return [f"the dev and sealed pairs share {kind} {item}" for kind in a for item in sorted(a[kind] & b[kind])]


def move_sealed(pair: Path, destination: Path) -> str:
    """Move the sealed pair out of the reader's tree; return its bundle digest (D7)."""
    digest = hashlib.sha256()
    for path in sorted(pair.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(pair).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pair), str(destination))
    return digest.hexdigest()


@contextmanager
def sealed_guard(sealed_root: Path) -> Iterator[list[str]]:
    """Inside the block, any in-process ``open`` under ``sealed_root`` raises SealedReadError.

    Audit hooks cannot be removed, so the hook stays installed and is switched off on exit.
    """
    root = str(sealed_root.resolve())
    state = {"on": True}
    attempts: list[str] = []

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if not state["on"] or event != "open" or not args:
            return
        target = args[0]
        if isinstance(target, (str, bytes, Path)):
            path = str(Path(target.decode() if isinstance(target, bytes) else target).resolve())
            if path == root or path.startswith(root + "/"):
                attempts.append(path)
                raise SealedReadError(f"sealed read refused: {path}")

    sys.addaudithook(hook)
    try:
        yield attempts
    finally:
        state["on"] = False


def gold_statistics(golds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    golds = list(golds)
    supports = Counter(s.get("support") for g in golds for s in g.get("statements") or [])
    return {"works": len(golds), "statements": sum(supports.values()), "by_support": dict(sorted(supports.items()))}
