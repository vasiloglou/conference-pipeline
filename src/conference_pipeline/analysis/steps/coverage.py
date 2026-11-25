"""Deterministic presentation coverage analysis step."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from conference_pipeline.analysis._digest import sha256_file


@dataclass
class PresentationInfo:
    """Information about a presentation."""

    folder_name: str
    title: str
    duration_minutes: float
    expected_refs: float
    actual_refs: int = 0
    referencing_docs: list[str] = field(default_factory=list)

    @property
    def coverage_score(self) -> float:
        if self.expected_refs <= 0:
            return float("inf") if self.actual_refs > 0 else 0.0
        return self.actual_refs / self.expected_refs

    @property
    def is_unreferenced(self) -> bool:
        return self.actual_refs == 0

    @property
    def is_underreferenced(self) -> bool:
        return self.coverage_score < 0.5 and self.actual_refs > 0


@dataclass(frozen=True)
class CoverageInputs:
    analysis_dir: Path
    presentations_dir: Path
    overwrite_existing: bool = True


@dataclass(frozen=True)
class CoverageOutputs:
    report_path: Path
    mapping_path: Path
    underreferenced_count: int
    unreferenced_count: int
    digest: str


def normalize_title(title: str) -> str:
    normalized = re.sub(r"[^\w\s]", "", title.lower()).strip()
    normalized = re.sub(r"\b(the|a|an|and|or|in|on|at|for|to|of|with)\b", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_presentations(presentations_dir: Path) -> dict[str, PresentationInfo]:
    presentations: dict[str, PresentationInfo] = {}
    if not presentations_dir.exists():
        return presentations

    for pres_dir in sorted(p for p in presentations_dir.iterdir() if p.is_dir()):
        folder_name = pres_dir.name
        title = folder_name.replace("-", " ").title()
        meta_file = pres_dir / "meta.json"
        if meta_file.exists():
            try:
                meta = _load_json(meta_file)
                if isinstance(meta, dict):
                    title = str(meta.get("title") or title)
            except (OSError, json.JSONDecodeError):
                pass

        duration_minutes = 0.0
        segments_file = pres_dir / "transcripts" / "segments.json"
        if segments_file.exists():
            try:
                segments = _load_json(segments_file)
                if isinstance(segments, list) and segments:
                    last = segments[-1]
                    if isinstance(last, dict):
                        duration_minutes = float(last.get("end") or 0) / 60
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        presentations[folder_name] = PresentationInfo(
            folder_name=folder_name,
            title=title,
            duration_minutes=duration_minutes,
            expected_refs=max(0.5, duration_minutes / 30),
        )
    return presentations


def _title_lookup(presentations: dict[str, PresentationInfo]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for folder_name, info in presentations.items():
        values = {
            folder_name,
            folder_name.replace("-", " "),
            normalize_title(folder_name),
            info.title,
            normalize_title(info.title),
        }
        for value in values:
            if value:
                lookup[value.lower()] = folder_name
    return lookup


def extract_references_from_doc(
    doc_path: Path,
    presentations: dict[str, PresentationInfo],
) -> set[str]:
    content = doc_path.read_text(encoding="utf-8")
    lower_content = content.lower()
    doc_name = doc_path.name
    referenced: set[str] = set()
    lookup = _title_lookup(presentations)

    backtick_pattern = r"`(?:presentations/)?([a-z0-9\-]+(?:/[a-z0-9\-]+)?)`"
    for match in re.finditer(backtick_pattern, content, re.IGNORECASE):
        ref = match.group(1).split("/")[0].lower()
        if ref in lookup:
            referenced.add(lookup[ref])

    source_pattern = r"\*\*Source:\*\*\s*[\"']?([^\"'\n]+)[\"']?"
    for match in re.finditer(source_pattern, content, re.IGNORECASE):
        normalized = normalize_title(match.group(1).strip().rstrip(".")).lower()
        if normalized in lookup:
            referenced.add(lookup[normalized])
            continue
        for key, folder in lookup.items():
            if len(key) > 5 and (key in normalized or normalized in key):
                referenced.add(folder)
                break

    for folder_name in presentations:
        if len(folder_name) > 10 and folder_name.lower() in lower_content:
            referenced.add(folder_name)

    for folder in referenced:
        docs = presentations[folder].referencing_docs
        if doc_name not in docs:
            docs.append(doc_name)
    return referenced


def generate_coverage_report(
    presentations: dict[str, PresentationInfo],
    *,
    documents_scanned: int,
) -> str:
    total = len(presentations)
    unreferenced = sorted(
        [p for p in presentations.values() if p.is_unreferenced],
        key=lambda p: (-p.duration_minutes, p.folder_name),
    )
    underreferenced = sorted(
        [p for p in presentations.values() if p.is_underreferenced],
        key=lambda p: (p.coverage_score, p.folder_name),
    )
    referenced = sorted(
        [p for p in presentations.values() if p.actual_refs > 0],
        key=lambda p: (-p.coverage_score, p.folder_name),
    )
    total_duration = sum(p.duration_minutes for p in presentations.values())

    lines = [
        "# Coverage Report",
        "",
        "## Summary",
        "",
        f"- Total presentations: {total}",
        f"- Referenced presentations: {len(referenced)}",
        f"- Unreferenced presentations: {len(unreferenced)}",
        f"- Underreferenced presentations: {len(underreferenced)}",
        f"- Total content hours: {total_duration / 60:.1f}",
        f"- Analysis documents scanned: {documents_scanned}",
        "",
        "## Unreferenced Presentations",
        "",
        "| Presentation | Duration | Expected Refs |",
        "|---|---:|---:|",
    ]
    for p in unreferenced:
        lines.append(f"| `{p.folder_name}` | {p.duration_minutes:.0f} min | {p.expected_refs:.1f} |")

    lines.extend(
        [
            "",
            "## Underreferenced Presentations",
            "",
            "| Presentation | Refs | Expected | Score |",
            "|---|---:|---:|---:|",
        ]
    )
    for p in underreferenced:
        lines.append(f"| `{p.folder_name}` | {p.actual_refs} | {p.expected_refs:.1f} | {p.coverage_score:.2f} |")

    lines.extend(
        [
            "",
            "## Reference Distribution by Document",
            "",
            "| Document | References |",
            "|---|---:|",
        ]
    )
    doc_refs: dict[str, int] = defaultdict(int)
    for p in presentations.values():
        for doc in p.referencing_docs:
            doc_refs[doc] += 1
    for doc, count in sorted(doc_refs.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {doc} | {count} |")

    return "\n".join(lines).rstrip() + "\n"


def _topic_from_doc_name(doc_name: str) -> str:
    if doc_name.endswith(".md"):
        return doc_name[:-3]
    return doc_name


def step_coverage(inputs: CoverageInputs) -> CoverageOutputs:
    inputs.analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = inputs.analysis_dir / "coverage_report.md"
    mapping_path = inputs.analysis_dir / "presentation_topic_mapping.json"
    if not inputs.overwrite_existing and report_path.exists() and mapping_path.exists():
        return CoverageOutputs(
            report_path=report_path,
            mapping_path=mapping_path,
            underreferenced_count=0,
            unreferenced_count=0,
            digest=sha256_file(report_path),
        )

    presentations = load_presentations(inputs.presentations_dir)
    docs = [
        path
        for path in sorted(inputs.analysis_dir.glob("*.md"))
        if not path.name.startswith(("NeurIPS_Analysis_Plan", "coverage_report", "qa_report", "pilot_"))
    ]

    mappings: list[dict[str, object]] = []
    for doc in docs:
        refs = extract_references_from_doc(doc, presentations)
        for folder in sorted(refs):
            info = presentations[folder]
            mappings.append(
                {
                    "folder": folder,
                    "topic": _topic_from_doc_name(doc.name),
                    "title": info.title,
                    "duration_minutes": info.duration_minutes,
                }
            )
    for folder in {mapping["folder"] for mapping in mappings}:
        presentations[str(folder)].actual_refs = len(presentations[str(folder)].referencing_docs)

    report_path.write_text(
        generate_coverage_report(presentations, documents_scanned=len(docs)),
        encoding="utf-8",
    )

    topic_counts: dict[str, int] = defaultdict(int)
    for mapping in mappings:
        topic_counts[str(mapping["topic"])] += 1
    mapping_path.write_text(
        json.dumps(
            {
                "total_presentations": len(mappings),
                "topic_counts": dict(sorted(topic_counts.items())),
                "mappings": mappings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return CoverageOutputs(
        report_path=report_path,
        mapping_path=mapping_path,
        underreferenced_count=len([p for p in presentations.values() if p.is_underreferenced]),
        unreferenced_count=len([p for p in presentations.values() if p.is_unreferenced]),
        digest=sha256_file(report_path),
    )
