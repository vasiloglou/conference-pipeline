"""Deterministic QA analysis step."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from conference_pipeline.analysis._digest import sha256_file


@dataclass
class QAResult:
    """QA results for a single document."""

    filename: str
    line_count: int = 0
    word_count: int = 0
    presentation_refs: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    github_links: list[str] = field(default_factory=list)
    arxiv_links: list[str] = field(default_factory=list)
    paper_refs: list[str] = field(default_factory=list)
    web_links: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    has_summary: bool = False
    has_key_findings: bool = False
    has_references_section: bool = False
    has_cross_references_section: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        score = 0
        if self.presentation_refs:
            score += 15
        if self.timestamps:
            score += 15
        if self.github_links or self.arxiv_links:
            score += 10
        if self.paper_refs or self.web_links:
            score += 10
        if self.cross_refs:
            score += 15
        if self.has_summary:
            score += 10
        if self.has_key_findings:
            score += 10
        if self.has_references_section:
            score += 10
        if self.has_cross_references_section:
            score += 5
        return score


@dataclass(frozen=True)
class QAInputs:
    analysis_dir: Path
    overwrite_existing: bool = True


@dataclass(frozen=True)
class QAOutputs:
    report_path: Path
    per_doc_scores: dict[str, int]
    digest: str


def analyze_document(filepath: Path) -> QAResult:
    content = filepath.read_text(encoding="utf-8")
    result = QAResult(
        filename=filepath.name,
        line_count=len(content.splitlines()),
        word_count=len(content.split()),
    )

    source_pattern = r"\*\*Source:\*\*\s*[\"']?([^\"'\n]+)[\"']?"
    result.presentation_refs = [match.group(1).strip() for match in re.finditer(source_pattern, content, re.IGNORECASE)]

    for match in re.finditer(r"`([a-z0-9\-]{15,})`", content):
        if match.group(1) not in result.presentation_refs:
            result.presentation_refs.append(match.group(1))

    result.timestamps = [match.group(1).strip() for match in re.finditer(r"\*\*Timestamp:\*\*\s*([^\n]+)", content)]
    if not result.timestamps:
        result.timestamps = re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", content)[:5]

    result.github_links = re.findall(r"https?://github\.com/[^\s\)>\]\"]+", content)
    result.arxiv_links = re.findall(r"https?://arxiv\.org/[^\s\)>\]\"]+", content)
    result.paper_refs = [f"arXiv:{pid}" for pid in re.findall(r"arXiv[:\s]*(\d{4}\.\d{4,5})", content, re.IGNORECASE)]
    all_urls = re.findall(r"https?://[^\s\)>\]\"]+", content)
    result.web_links = [url for url in all_urls if "github.com" not in url and "arxiv.org" not in url]
    result.cross_refs = [match.group(2) for match in re.finditer(r"\[([^\]]+)\]\((\d{2}_[^\)]+\.md)\)", content)]

    result.has_summary = bool(re.search(r"^#+\s*(Executive\s+)?Summary", content, re.MULTILINE | re.IGNORECASE))
    result.has_key_findings = bool(re.search(r"^#+\s*Key\s+(Findings|Takeaways)", content, re.MULTILINE | re.IGNORECASE))
    result.has_references_section = bool(re.search(r"^#+\s*References", content, re.MULTILINE | re.IGNORECASE))
    result.has_cross_references_section = bool(re.search(r"^#+\s*Cross[-\s]?References", content, re.MULTILINE | re.IGNORECASE))

    if not result.presentation_refs:
        result.issues.append("No presentation references found")
    if not result.timestamps:
        result.issues.append("No timestamps found")
    if not (result.github_links or result.arxiv_links):
        result.issues.append("No GitHub or arXiv links")
    if not result.cross_refs:
        result.issues.append("No cross-references to other docs")
    if not result.has_summary:
        result.issues.append("Missing Summary section")
    if not result.has_key_findings:
        result.issues.append("Missing Key Findings/Takeaways section")
    if not result.has_references_section:
        result.issues.append("Missing References section")
    return result


def generate_qa_report(results: list[QAResult]) -> str:
    total_score = sum(result.score for result in results)
    average = total_score / len(results) if results else 0.0
    passed = [result for result in results if result.score >= 70]
    needs_work = [result for result in results if 50 <= result.score < 70]
    failing = [result for result in results if result.score < 50]

    lines = [
        "# QA Report",
        "",
        "## Summary",
        "",
        f"- Documents analyzed: {len(results)}",
        f"- Average score: {average:.1f}/100",
        f"- Passing (>=70): {len(passed)}",
        f"- Needs work (50-69): {len(needs_work)}",
        f"- Failing (<50): {len(failing)}",
        "",
        "## Document Scores",
        "",
        "| Document | Score | Lines | Pres Refs | Timestamps | GitHub | arXiv | Cross-Refs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (-item.score, item.filename)):
        lines.append(
            "| "
            f"{result.filename} | {result.score} | {result.line_count} | "
            f"{len(result.presentation_refs)} | {len(result.timestamps)} | "
            f"{len(result.github_links)} | {len(result.arxiv_links)} | {len(result.cross_refs)} |"
        )

    lines.extend(["", "## Issues by Document", ""])
    for result in sorted(results, key=lambda item: (item.score, item.filename)):
        if not result.issues:
            continue
        lines.append(f"### {result.filename} (Score: {result.score})")
        lines.extend(f"- {issue}" for issue in result.issues)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def step_qa(inputs: QAInputs) -> QAOutputs:
    inputs.analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = inputs.analysis_dir / "qa_report.md"
    if not inputs.overwrite_existing and report_path.exists():
        return QAOutputs(report_path=report_path, per_doc_scores={}, digest=sha256_file(report_path))

    docs = [
        path
        for path in sorted(inputs.analysis_dir.glob("*.md"))
        if not path.name.startswith(("NeurIPS_Analysis_Plan", "coverage_report", "qa_report", "pilot_"))
    ]
    results = [analyze_document(path) for path in docs]
    report_path.write_text(generate_qa_report(results), encoding="utf-8")
    return QAOutputs(
        report_path=report_path,
        per_doc_scores={result.filename: result.score for result in results},
        digest=sha256_file(report_path),
    )
