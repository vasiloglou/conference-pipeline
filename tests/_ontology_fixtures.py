"""Synthetic workspaces for the v6.1 Phase 2 ontology-input tests.

One record per join rule of spec 6.1, plus malformed, keyless, excluded and
non-indexable records. ``prefix`` changes every track, topic and workshop name, so the
same builder makes the second workspace of T22 (R2).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

POSTER_BOARD = "paper_text:4284aa957bbcbaf0eafc724b"
TALKS_REL = "storage/transcription-ocr/talk-maps/base/900_talks.json"
MISSING_TALKS_REL = "storage/transcription-ocr/workshop-talks/missing-talks.json"

TITLE_A = "Alpha method for sparse attention kernels"
TITLE_B = "Beta study of optimizer schedules at scale"
TITLE_C = "Gamma shared title across two tracks of work"
TITLE_D = "Delta oral paper about diffusion samplers"
TITLE_E = "Epsilon oral paper about kv cache eviction"


_SHORT = re.compile(r"^(paper_text|workshop_paper_text|speech_transcript|media_asset):([a-z0-9]{3})$")


def aid(short: str) -> str:
    """The 24-hex artifact id a short fixture id stands for (``"paper_text:a01"``)."""
    family, tag = short.split(":")
    return f"{family}:{hashlib.sha256(tag.encode()).hexdigest()[:24]}"


def _rec(artifact_id: str, family: str, **fields: Any) -> dict[str, Any]:
    meta = {k: v for k, v in fields.items() if k in {"paper_title", "item_id", "item_kind", "openreview_id",
                                                       "workshop_id", "authors", "talk_map_talks_json_path",
                                                       "workshop_talks_json_path", "title"}}
    record = {"artifact_id": artifact_id, "source_family_id": family, "indexable_text": True,
              "metadata": meta, **{k: v for k, v in fields.items() if k not in {"workshop_id",
                                                                                 "talk_map_talks_json_path",
                                                                                 "workshop_talks_json_path"}}}
    return record


def records(prefix: str = "") -> list[Any]:
    out = _records(prefix)
    for record in out:
        if isinstance(record, dict) and _SHORT.match(str(record.get("artifact_id"))):
            record["artifact_id"] = aid(record["artifact_id"])
    return out


def _records(prefix: str) -> list[Any]:
    ws1 = f"{prefix}WS/One"
    return [
        _rec("paper_text:a01", "paper_text", material_type="paper", openreview_id="FORUMAAAA1", paper_title=TITLE_A,
             source_path="cache/main/a01.pdf"),
        # T1: no openreview_id field; the id comes from the OpenReview attachment URL
        {**_rec("paper_text:b01", "paper_text", material_type="paper", paper_title=TITLE_B,
                source_path="cache/main/b01.pdf"),
         "provenance": {"download_artifact": {"final_url": "https://api2.openreview.net/attachment?id=FORUMBBBB2&name=pdf",
                                              "source_family_id": "openreview-venue"}}},
        _rec("paper_text:c01", "paper_text", material_type="paper", openreview_id="FORUMCCCC3", paper_title=TITLE_C),
        _rec("paper_text:d01", "paper_text", material_type="paper", openreview_id="FORUMDDDD4", paper_title=TITLE_D),
        _rec("paper_text:e01", "paper_text", material_type="paper", openreview_id="FORUMEEEE5", paper_title=TITLE_E),
        # T2 / T3b: workshop papers; w02 shares paper C's title (a work in two tracks)
        _rec("workshop_paper_text:w01", "workshop_paper_text", material_type="paper", openreview_id="WSPAPER001",
             paper_title="Workshop one paper on agents that plan", workshop_id=ws1, authors=["Wen Ye", "Ola Ade"]),
        _rec("workshop_paper_text:w02", "workshop_paper_text", material_type="paper", openreview_id="WSPAPER002",
             paper_title=TITLE_C, workshop_id=ws1, authors=["Cam Dee"]),
        # T3: poster slides and poster transcript join paper A by item id and title
        _rec("paper_text:p01", "paper_text", material_type="slides", item_kind="poster", item_id="poster-101",
             paper_title=TITLE_A),
        _rec("speech_transcript:s01", "speech_transcript", item_kind="poster", item_id="poster-101", paper_title=TITLE_A),
        # a poster whose paper is not in the corpus: a companion-only family
        _rec("speech_transcript:s02", "speech_transcript", item_kind="poster", item_id="poster-102",
             paper_title="Orphan poster about something else entirely"),
        # a paper PDF with no stable id: a site paper
        {**_rec("paper_text:c90", "paper_text", material_type="paper",
                paper_title="Calendar paper without an identifier at all", source_path="cache/calendar/c90.pdf"),
         "provenance": {"download_artifact": {"source_family_id": "calendar"}}},
        # T6b: oral session recording with a talk map naming two papers (one by item path, one by title)
        _rec("speech_transcript:s03", "speech_transcript", item_kind="oral", item_id="session-900",
             paper_title=f"{prefix}Oral 1A Session", talk_map_talks_json_path=TALKS_REL),
        _rec("paper_text:o01", "paper_text", material_type="slides", item_kind="oral", item_id="oral-201",
             paper_title=TITLE_D),
        # single-talk events; a tutorial's slides and recording are one event
        _rec("speech_transcript:s04", "speech_transcript", item_kind="invited-talk", item_id="invited-1",
             title="Invited Talk: Agents in the world"),
        _rec("paper_text:t01", "paper_text", material_type="slides", item_kind="tutorial", item_id="virtual-5",
             paper_title="Tutorial: Proving things"),
        _rec("speech_transcript:s05", "speech_transcript", item_kind="tutorial", item_id="virtual-5",
             title="Tutorial: Proving things"),
        # a workshop recording whose talk map is missing: an event with unlinked talks
        _rec("speech_transcript:s06", "speech_transcript", item_kind="workshop", item_id="workshop-77",
             title="Workshop: One", workshop_talks_json_path=MISSING_TALKS_REL, workshop_id=ws1),
        # T5: the poster-board record (no title, no item): excluded by the workspace's list
        {"artifact_id": POSTER_BOARD, "source_family_id": "paper_text", "indexable_text": True, "metadata": {}},
        # T4: non-indexable, malformed and keyless records
        {"artifact_id": "media_asset:m01", "source_family_id": "media_asset", "indexable_text": False},
        "not a record",
        {"source_family_id": "paper_text", "indexable_text": True},
        {"artifact_id": "paper_text:k01", "source_family_id": "paper_text", "indexable_text": True,
         "material_type": "paper", "metadata": {}},
        {"artifact_id": "speech_transcript:k02", "source_family_id": "speech_transcript", "indexable_text": True,
         "metadata": {}},
    ]


def openreview_papers(prefix: str = "") -> list[dict[str, Any]]:
    def paper(forum: str, title: str, authors: list[str], venue: str) -> dict[str, Any]:
        return {"forum": forum, "id": forum, "title": title, "authors": authors, "venue": f"{prefix}{venue}"}
    return [
        paper("FORUMAAAA1", TITLE_A, ["Ann Alpha", "Bo Beta"], "Conf regular"),
        paper("FORUMBBBB2", TITLE_B, ["Cy Gamma"], "Conf spotlight"),
        paper("FORUMCCCC3", TITLE_C, ["Cam Dee", "Dan Eff"], "Conf regular"),
        paper("FORUMDDDD4", TITLE_D, ["Di Delta"], "Conf regular"),
        paper("FORUMEEEE5", TITLE_E, ["Ed Eps"], "Conf regular"),
    ]


TOPICS = {"FORUMAAAA1": ["t01_alpha"], "FORUMBBBB2": ["t02_beta", "t01_alpha"], "FORUMCCCC3": ["t01_alpha"],
          "FORUMDDDD4": ["t03_gamma"], "FORUMEEEE5": []}


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")
    return path


def build_workspace(root: Path, *, prefix: str = "", exclusions: bool = True, analysis: bool = True) -> Path:
    workspace = root / "conferences" / f"{prefix.strip('/').lower() or 'conf'}"
    write_json(workspace / "storage/corpus/corpus_inventory.json", {"artifact_records": records(prefix)})
    write_json(workspace / "storage/download-corpus/openreview_accepted_papers.json",
               {"papers": openreview_papers(prefix)})
    for forum, topics in TOPICS.items():
        write_json(workspace / f"storage/extractions/run1/llm-full-read/by-unit/paper_abstract_{forum}.json",
                   {"topic_ids": [f"{prefix}{t}" for t in topics]})
    write_json(workspace / TALKS_REL, {
        "program_hints": {"scheduled_talks": [{"item_kind": "oral", "item_path": "/virtual/2026/oral/201",
                                               "title": "a scheduled title that matches nothing at all"}]},
        "talks": [{"talk_title": TITLE_E}, {"talk_title": "Session opening and introduction remarks"}],
    })
    if exclusions:
        write_json(workspace / "ontology/exclusions.json",
                   {"exclusions": [{"artifact_id": POSTER_BOARD, "reason": "not-a-work"}]})
    if analysis:
        (workspace / "analysis/deep_dive_topics").mkdir(parents=True, exist_ok=True)
        (workspace / "analysis/01_alpha.md").write_text(
            f"Cites {aid('paper_text:a01')} and https://openreview.net/forum?id=FORUMBBBB2 and "
            "paper_text:ffffffffffffffffffffffff (not in the corpus).\n", encoding="utf-8")
        write_json(workspace / "analysis/topic_strategy.json", {"sources": [{"forum": "FORUMCCCC3"}]})
        (workspace / "analysis/deep_dive_topics/x.md").write_text(
            f"From the full read: paper_abstract:FORUMDDDD4 and {aid('speech_transcript:s04')}:segment:00001.\n",
            encoding="utf-8")
        (workspace / "analysis/notes_not_a_product.md").write_text(f"{aid('paper_text:e01')}\n", encoding="utf-8")
    return workspace
