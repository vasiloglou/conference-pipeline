"""Competency questions: the reviewed markdown drafts as one machine-readable file (spec 6.6).

Each draft file holds questions as ``### Q-<area>-<n>. <text>`` headings followed by
``- Kind:``, ``- Forces:``, ``- Tier:``, ``- Priority:`` and ``- Evidence:`` bullets. An
edits file lists proposed merges and rewordings; only edits the owner approved
(``"status": "approved"``) change the output (D7). A merged question keeps its id and
names the question it was merged into; a reworded question keeps its original text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

QUESTIONS_SCHEMA = "ontology-competency-questions/1"
EDIT_STATUSES = frozenset({"proposed", "approved", "rejected"})
_HEADING = re.compile(r"^###\s+(Q-[0-9A-Z]+-\d+)\.\s+(.+?)\s*$")
_BULLET = re.compile(r"^-\s+(Kind|Forces|Tier|Priority|Evidence):\s*(.*?)\s*$")
_AREA = re.compile(r"^Q-([0-9A-Z]+)-\d+$")


class QuestionsError(ValueError):
    """A question file or edit is malformed (duplicate id, unknown target, bad status)."""


def parse_file(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := _HEADING.match(line):
            qid = match.group(1)
            current = {"id": qid, "area": _AREA.match(qid).group(1), "text": match.group(2),  # type: ignore[union-attr]
                       "source": path.name}
            questions.append(current)
        elif current is not None and (bullet := _BULLET.match(line)):
            current[bullet.group(1).lower()] = bullet.group(2)
    return questions


def parse_drafts(directory: Path) -> list[dict[str, Any]]:
    """Every question in ``directory``'s ``area-*.md`` and ``cross-cutting.md`` files."""
    files = sorted(directory.glob("area-*.md")) + sorted(directory.glob("cross-cutting.md"))
    questions = [q for path in files for q in parse_file(path)]
    seen: dict[str, str] = {}
    for question in questions:
        if question["id"] in seen:
            raise QuestionsError(f"{question['id']} appears in {seen[question['id']]} and {question['source']}")
        seen[question["id"]] = question["source"]
    return questions


def load_edits(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    edits = payload.get("edits") if isinstance(payload, Mapping) else None
    if not isinstance(edits, list):
        raise QuestionsError(f"{path} has no edits list")
    for edit in edits:
        if not isinstance(edit, Mapping) or edit.get("status") not in EDIT_STATUSES:
            raise QuestionsError(f"edit with a missing or unknown status: {edit!r}")
        if edit.get("kind") not in {"merge", "reword"}:
            raise QuestionsError(f"edit of unknown kind: {edit!r}")
    return [dict(e) for e in edits]


def apply_edits(questions: list[dict[str, Any]], edits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {q["id"]: q for q in questions}
    out = {q["id"]: {**q, "merged_into": None, "original_text": None} for q in questions}
    for edit in edits:
        targets = [edit.get("question"), edit.get("into")] if edit["kind"] == "merge" else [edit.get("question")]
        for target in targets:
            if target not in by_id:
                raise QuestionsError(f"edit names an unknown question: {target!r}")
        if edit["status"] != "approved":
            continue
        question = out[str(edit["question"])]
        if edit["kind"] == "merge":
            if edit["into"] == edit["question"] or out[str(edit["into"])]["merged_into"]:
                raise QuestionsError(f"merge target is itself or already merged: {edit!r}")
            question["merged_into"] = edit["into"]
        else:
            if not isinstance(edit.get("text"), str) or not edit["text"].strip():
                raise QuestionsError(f"rewording without text: {edit!r}")
            question["original_text"], question["text"] = question["text"], edit["text"].strip()
    return [out[q["id"]] for q in questions]


def build_questions(directory: Path, edits_path: Path | None = None) -> dict[str, Any]:
    questions = parse_drafts(directory)
    edits = load_edits(edits_path)
    applied = apply_edits(questions, edits)
    return {
        "schema": QUESTIONS_SCHEMA,
        "questions": applied,
        "edits": edits,
        "counts": {
            "questions": len(applied),
            "core": sum(q.get("priority") == "core" for q in applied),
            "core_active": sum(q.get("priority") == "core" and not q["merged_into"] for q in applied),
            "merged": sum(bool(q["merged_into"]) for q in applied),
            "reworded": sum(bool(q["original_text"]) for q in applied),
            "edits_pending": sum(e["status"] == "proposed" for e in edits),
        },
    }


def question_problems(document: Mapping[str, Any], core_ids: Iterable[str]) -> list[str]:
    """Empty when every listed core id is present as a core question and priorities are valid."""
    problems: list[str] = []
    by_id = {q["id"]: q for q in document.get("questions") or []}
    for qid in sorted(set(core_ids)):
        if qid not in by_id:
            problems.append(f"core question {qid} is missing")
        elif by_id[qid].get("priority") != "core":
            problems.append(f"{qid} is listed as core but has priority {by_id[qid].get('priority')!r}")
    for qid, question in sorted(by_id.items()):
        if question.get("priority") not in {"core", "extended"}:
            problems.append(f"{qid} has no valid priority")
    return problems


def core_ids_from_set(core_set_md: Path) -> list[str]:
    """Question ids listed as ``- **Q-…**`` bullets in ``core-set.md``."""
    return re.findall(r"^- \*\*(Q-[0-9A-Z]+-\d+)\*\*", core_set_md.read_text(encoding="utf-8"), flags=re.M)
