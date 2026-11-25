#!/usr/bin/env python3
"""
Phase 4: Quality Assurance Script for NeurIPS 2025 Analysis Documents

Checks each document for:
1. Presentations referenced with timestamps
2. External resources (GitHub, arXiv, docs)
3. Cross-references to related topics
4. Document structure (summary, key findings, etc.)
5. Formatting quality
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set
from datetime import datetime
import json


@dataclass
class QAResult:
    """QA results for a single document."""
    filename: str
    line_count: int = 0
    word_count: int = 0

    # Presentations and timestamps
    presentation_refs: List[str] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)

    # External resources
    github_links: List[str] = field(default_factory=list)
    arxiv_links: List[str] = field(default_factory=list)
    paper_refs: List[str] = field(default_factory=list)
    web_links: List[str] = field(default_factory=list)

    # Cross-references
    cross_refs: List[str] = field(default_factory=list)

    # Structure checks
    has_summary: bool = False
    has_key_findings: bool = False
    has_references_section: bool = False
    has_cross_references_section: bool = False

    # Issues found
    issues: List[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Calculate QA score (0-100)."""
        score = 0

        # Presentations with timestamps (30 points)
        if len(self.presentation_refs) > 0:
            score += 15
        if len(self.timestamps) > 0:
            score += 15

        # External resources (20 points)
        if len(self.github_links) > 0 or len(self.arxiv_links) > 0:
            score += 10
        if len(self.paper_refs) > 0 or len(self.web_links) > 0:
            score += 10

        # Cross-references (15 points)
        if len(self.cross_refs) > 0:
            score += 15

        # Structure (35 points)
        if self.has_summary:
            score += 10
        if self.has_key_findings:
            score += 10
        if self.has_references_section:
            score += 10
        if self.has_cross_references_section:
            score += 5

        return score


def analyze_document(filepath: Path) -> QAResult:
    """Analyze a single document for QA criteria."""
    with open(filepath) as f:
        content = f.read()

    lines = content.split('\n')
    words = content.split()

    result = QAResult(
        filename=filepath.name,
        line_count=len(lines),
        word_count=len(words)
    )

    # Check for presentations/sources
    source_pattern = r'\*\*Source:\*\*\s*["\']?([^"\'\n]+)["\']?'
    for match in re.finditer(source_pattern, content, re.IGNORECASE):
        result.presentation_refs.append(match.group(1).strip())

    # Also check for backtick presentation references
    backtick_pattern = r'`([a-z0-9\-]{15,})`'
    for match in re.finditer(backtick_pattern, content):
        if match.group(1) not in result.presentation_refs:
            result.presentation_refs.append(match.group(1))

    # Check for timestamps
    timestamp_pattern = r'\*\*Timestamp:\*\*\s*([^\n]+)'
    for match in re.finditer(timestamp_pattern, content):
        result.timestamps.append(match.group(1).strip())

    # Also check inline timestamps like HH:MM:SS
    time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'
    times = re.findall(time_pattern, content)
    if times and not result.timestamps:
        result.timestamps = times[:5]  # Sample

    # Check for GitHub links
    github_pattern = r'https?://github\.com/[^\s\)>\]"]+'
    result.github_links = re.findall(github_pattern, content)

    # Check for arXiv links
    arxiv_pattern = r'https?://arxiv\.org/[^\s\)>\]"]+'
    result.arxiv_links = re.findall(arxiv_pattern, content)

    # Check for paper references (arXiv IDs)
    arxiv_id_pattern = r'arXiv[:\s]*(\d{4}\.\d{4,5})'
    paper_ids = re.findall(arxiv_id_pattern, content, re.IGNORECASE)
    result.paper_refs = [f"arXiv:{pid}" for pid in paper_ids]

    # Check for other web links
    url_pattern = r'https?://[^\s\)>\]"]+'
    all_urls = re.findall(url_pattern, content)
    result.web_links = [u for u in all_urls if 'github.com' not in u and 'arxiv.org' not in u]

    # Check for cross-references to other analysis docs
    cross_ref_pattern = r'\[([^\]]+)\]\((\d{2}_[^\)]+\.md)\)'
    for match in re.finditer(cross_ref_pattern, content):
        result.cross_refs.append(match.group(2))

    # Check document structure
    result.has_summary = bool(re.search(r'^#+\s*(Executive\s+)?Summary', content, re.MULTILINE | re.IGNORECASE))
    result.has_key_findings = bool(re.search(r'^#+\s*Key\s+(Findings|Takeaways)', content, re.MULTILINE | re.IGNORECASE))
    result.has_references_section = bool(re.search(r'^#+\s*References', content, re.MULTILINE | re.IGNORECASE))
    result.has_cross_references_section = bool(re.search(r'^#+\s*Cross[-\s]?References', content, re.MULTILINE | re.IGNORECASE))

    # Identify issues
    if len(result.presentation_refs) == 0:
        result.issues.append("❌ No presentation references found")
    if len(result.timestamps) == 0:
        result.issues.append("⚠️ No timestamps found")
    if len(result.github_links) == 0 and len(result.arxiv_links) == 0:
        result.issues.append("⚠️ No GitHub or arXiv links")
    if len(result.cross_refs) == 0:
        result.issues.append("⚠️ No cross-references to other docs")
    if not result.has_summary:
        result.issues.append("⚠️ Missing Summary section")
    if not result.has_key_findings:
        result.issues.append("⚠️ Missing Key Findings/Takeaways section")
    if not result.has_references_section:
        result.issues.append("⚠️ Missing References section")

    return result


def generate_qa_report(results: List[QAResult]) -> str:
    """Generate QA report."""
    report = []
    report.append("# Phase 4: Quality Assurance Report\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append("## Summary\n")

    total_score = sum(r.score for r in results)
    avg_score = total_score / len(results)

    passed = [r for r in results if r.score >= 70]
    needs_work = [r for r in results if 50 <= r.score < 70]
    failing = [r for r in results if r.score < 50]

    report.append(f"- **Documents Analyzed:** {len(results)}")
    report.append(f"- **Average Score:** {avg_score:.1f}/100")
    report.append(f"- **Passing (≥70):** {len(passed)}")
    report.append(f"- **Needs Work (50-69):** {len(needs_work)}")
    report.append(f"- **Failing (<50):** {len(failing)}\n")

    report.append("---\n")
    report.append("## Document Scores\n")
    report.append("| Document | Score | Lines | Pres Refs | Timestamps | GitHub | arXiv | Cross-Refs |")
    report.append("|----------|-------|-------|-----------|------------|--------|-------|------------|")

    for r in sorted(results, key=lambda x: x.score, reverse=True):
        status = "✅" if r.score >= 70 else "⚠️" if r.score >= 50 else "❌"
        report.append(f"| {status} {r.filename} | {r.score} | {r.line_count} | {len(r.presentation_refs)} | {len(r.timestamps)} | {len(r.github_links)} | {len(r.arxiv_links)} | {len(r.cross_refs)} |")

    report.append("\n---\n")
    report.append("## Issues by Document\n")

    for r in sorted(results, key=lambda x: x.score):
        if r.issues:
            report.append(f"\n### {r.filename} (Score: {r.score})\n")
            for issue in r.issues:
                report.append(f"- {issue}")

    report.append("\n---\n")
    report.append("## QA Criteria\n")
    report.append("| Criterion | Points | Description |")
    report.append("|-----------|--------|-------------|")
    report.append("| Presentation refs | 15 | References to source presentations |")
    report.append("| Timestamps | 15 | Timestamp citations for quotes |")
    report.append("| GitHub/arXiv | 10 | External code/paper links |")
    report.append("| Other resources | 10 | Web links, documentation |")
    report.append("| Cross-references | 15 | Links to related analysis docs |")
    report.append("| Summary section | 10 | Executive summary present |")
    report.append("| Key findings | 10 | Key takeaways section |")
    report.append("| References section | 10 | References section present |")
    report.append("| Cross-refs section | 5 | Cross-references section |")

    return "\n".join(report)


def main():
    analysis_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025/analysis")

    # Get all analysis docs (exclude meta files)
    docs = [f for f in analysis_dir.glob("*.md")
            if not f.name.startswith("NeurIPS_Analysis_Plan")
            and not f.name.startswith("coverage_report")
            and not f.name.startswith("pilot_")
            and not f.name.startswith("qa_report")]

    print(f"Analyzing {len(docs)} documents...\n")

    results = []
    for doc in sorted(docs):
        result = analyze_document(doc)
        results.append(result)
        status = "✅" if result.score >= 70 else "⚠️" if result.score >= 50 else "❌"
        print(f"{status} {result.filename}: {result.score}/100 ({len(result.presentation_refs)} refs, {len(result.timestamps)} timestamps)")

    # Generate and save report
    report = generate_qa_report(results)
    report_path = analysis_dir / "qa_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"QA Report saved to: {report_path}")
    print(f"Average score: {sum(r.score for r in results)/len(results):.1f}/100")

    # Print summary of issues
    all_issues = []
    for r in results:
        for issue in r.issues:
            all_issues.append((r.filename, issue))

    if all_issues:
        print(f"\nTop issues to address:")
        issue_counts = {}
        for _, issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {count}x {issue}")


if __name__ == "__main__":
    main()
