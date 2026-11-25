"""The seed vocabulary packaged as an unattended growth root (v6.1 Phase 5, spec IF8 / P5-D9).

GND 0.4.0 refuses an owner-confirmed model as, or as an include of, an unattended root:
nothing in unattended mode may be ``confirmed`` (U2) or ``explicit`` / ``user-supplied`` (U3),
and every element must cite a span of a declared document (U4, U5). Unattended means nobody
confirmed anything in this file, and GND holds a growth model to that.

So a growth strategy starts from a **working copy** of the seed in GND's own terms for carried-in
claims. The canonical seed file stays the authority and its confirmations stay there:

- the seed file's exact bytes are the root's first declared source document;
- every object type and fact type cites the span of its own ``id`` line in that file;
- every constraint is ``source: imported`` (``framework``, ``artifact``, ``commit``, ``sha256``,
  ``anchor``: its verbatim rationale sentence) and, as GND's D-03 table requires of an import
  that this model has not reviewed, ``status: proposed`` and ``modality: deontic``, citing the
  span of its ``id`` line.

The reconciled vocabulary edition (plan Phase 7) is curated, not unattended, and includes the
canonical seed, so the owner's confirmations are back in force there.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import yaml

FRAMEWORK = "conference-pipeline ontology seed vocabulary"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _span(text: str, needle: str, doc_sha: str, after: int = 0) -> tuple[dict[str, Any], int]:
    """GND span of the first occurrence of ``needle`` at or after ``after`` (code points of NFC)."""
    at = text.find(needle, after)
    if at < 0:
        raise ValueError(f"seed text has no {needle!r}")
    start, end = at, at + len(needle)
    return {"document_sha256": doc_sha, "start": start, "end": end,
            "excerpt_sha256": _sha(text[start:end].encode("utf-8"))}, end


def _id_line(text: str, ident: str, after: int = 0) -> str:
    match = re.compile(rf"(?:- |\{{)id: {re.escape(ident)}\b").search(text, after)
    if match is None:
        raise ValueError(f"seed text has no id line for {ident!r}")
    return match.group(0)


def unattended_seed(seed_raw: bytes, *, label: str, artifact: str, commit: str) -> str:
    """The unattended working root for one growth strategy, from the seed file's exact bytes."""
    text = unicodedata.normalize("NFC", seed_raw.decode("utf-8"))
    doc_sha = _sha(seed_raw)
    model = yaml.safe_load(seed_raw)

    def cite(ident: str) -> dict[str, Any]:
        needle = _id_line(text, ident)
        span, _ = _span(text, needle, doc_sha)
        return span

    def imported(constraint: dict[str, Any]) -> dict[str, Any]:
        anchor = str(constraint.get("rationale") or "").strip()
        if not anchor or anchor not in text:
            raise ValueError(f"constraint {constraint['id']} has no verbatim rationale to anchor on")
        out = {k: v for k, v in constraint.items() if k not in ("source", "status", "modality", "provenance")}
        out.update(source="imported", status="proposed", modality="deontic",
                   provenance={"framework": FRAMEWORK, "artifact": artifact, "commit": commit, "sha256": doc_sha,
                               "anchor": anchor, "spans": [cite(constraint["id"])]})
        return out

    for ot in model.get("object_types") or []:
        ot["provenance"] = {"spans": [cite(ot["id"])]}
    for ft in model.get("fact_types") or []:
        ft["provenance"] = {"spans": [cite(ft["id"])]}
        ft["constraints"] = [imported(c) for c in ft.get("constraints") or []]
    model["constraints"] = [imported(c) for c in model.get("constraints") or []]
    model["source"] = {"kind": "prose", "mode": "unattended", "reviewer": "none",
                       "documents": [{"sha256": doc_sha, "label": label}]}
    return yaml.safe_dump(model, sort_keys=False, allow_unicode=True)
