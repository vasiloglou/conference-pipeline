"""Reusable deep narrative analysis and provenance helpers.

Public topic reports are deep analyst documents. Raw extraction, chunk,
research, and matched-paper evidence belongs in provenance sidecars and the
citation manifest so downstream gates stay machine-checkable without making
human-facing reports read like audit ledgers. Deck-focused summaries live in
``analysis/slide_briefs/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from conference_pipeline import workflow
from conference_pipeline.model_governance import default_model
from conference_pipeline.provider_receipts import openai_chat_completions_create
from conference_pipeline.workflow_runs import write_json_atomic

DEFAULT_NARRATIVE_STATUS = "production-final-v1.63"
DEFAULT_ANALYSIS_PROFILE = "deep-narrative-v1"
# Registry-backed deep-narrative default (gpt-5.6-sol as of the 2026-07-15
# model-refresh sweep; runs with registry-configured high reasoning effort).
DEFAULT_DEEP_NARRATIVE_MODEL = default_model("analysis.deep_narrative_synthesis")
DEFAULT_DEEP_NARRATIVE_MIN_WORDS = 450
DEFAULT_DEEP_NARRATIVE_MIN_SUBSECTIONS = 3
DEFAULT_DEEP_NARRATIVE_MIN_REFERENCES = 3
DEFAULT_DEEP_NARRATIVE_MAX_ATTEMPTS = 3
DEFAULT_DEEP_NARRATIVE_TIMEOUT_SECONDS = 180.0
DEFAULT_MIN_CHUNK_CITATIONS_PER_TOPIC = 3
DEFAULT_MATCHED_PAPER_TOP_K = 20
NARRATIVE_SOURCE_PACK_DIRNAME = "narrative_source_packs"

# Structured source-metadata keys that must survive compact_chunk_evidence()'s
# context-budget truncation (Phase 10.1 Defect 1). Compaction still discards
# everything else on the raw chunk -- only these enumerated keys, each with an
# explicit per-key bound, cross into the compacted narrative-pack entry.
COMPACT_METADATA_KEYS = (
    "paper_title",
    "title",
    "talk_title",
    "session_title",
    "workshop_title",
    "authors",
)
MAX_COMPACT_AUTHORS = 12
MAX_COMPACT_TITLE_CHARS = 300
SLIDE_BRIEF_DIRNAME = "slide_briefs"

_SAFE_TOPIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_NARRATIVE_FORBIDDEN_HEADINGS = {
    "Source Basis",
    "Cited extraction record ids",
    "Key Findings (Evidence-Backed)",
    "Verbatim transcript/OCR evidence",
    "Matched Paper Evidence",
    "Source Gaps",
}

_EXTRACTION_CITATION_RE = re.compile(r"<!--\s*extraction:([^>\s]+)\s*-->")
_VECTOR_STORE_CITATION_RE = re.compile(r"<!--\s*chunk:([^>\s]+)\s*-->")
_QUERY_RESEARCH_CITATION_RE = re.compile(r"<!--\s*research:([^>\s]+)\s*-->")
_MATCHED_PAPER_CITATION_RE = re.compile(r"<!--\s*matched-paper:([^>\s]+)\s*-->")


@dataclass(frozen=True)
class NarrativeOutputResult:
    status: str
    report_paths: tuple[Path, ...]
    provenance_paths: tuple[Path, ...]
    provenance_json_paths: tuple[Path, ...]
    citation_manifest_path: Path | None
    citation_summaries: tuple[dict[str, Any], ...]
    slide_brief_paths: tuple[Path, ...] = ()
    blockers: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DeepNarrativeSynthesisResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    request_count: int


@contextmanager
def _narrative_call_timeout(seconds: float):
    """Bound provider calls even when the SDK-level timeout does not return."""

    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"deep narrative synthesis timed out after {seconds:g}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class OpenAIDeepNarrativeClient:
    """Reusable OpenAI client for production deep narrative synthesis."""

    task_id = "analysis.deep_narrative_synthesis"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.model = model or default_model(self.task_id)
        self.api_key = api_key
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else DEFAULT_DEEP_NARRATIVE_TIMEOUT_SECONDS
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    @classmethod
    def from_env(cls) -> "OpenAIDeepNarrativeClient | None":
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            return None
        raw_timeout = os.environ.get("NARRATIVE_ANALYSIS_TIMEOUT_SECONDS")
        timeout_seconds = None
        if raw_timeout:
            try:
                timeout_seconds = max(1.0, float(raw_timeout))
            except ValueError:
                timeout_seconds = DEFAULT_DEEP_NARRATIVE_TIMEOUT_SECONDS
        return cls(
            model=os.environ.get("NARRATIVE_ANALYSIS_MODEL") or None,
            timeout_seconds=timeout_seconds,
        )

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> DeepNarrativeSynthesisResult:  # pragma: no cover - provider call
        del kind
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai_not_available") from exc
        import os

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("missing_openai_api_key")
        client = OpenAI(api_key=api_key, timeout=self.timeout_seconds)
        with _narrative_call_timeout(self.timeout_seconds):
            response = openai_chat_completions_create(
                client,
                caller=__file__,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.request_count += 1
        return DeepNarrativeSynthesisResult(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            request_count=self.request_count,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_topic_id_for_path(topic_id: str) -> str:
    normalized = str(topic_id or "").strip()
    if not _SAFE_TOPIC_ID_RE.fullmatch(normalized):
        raise ValueError(f"invalid narrative topic_id for path: {normalized!r}")
    return normalized


def _compact_metadata_value(chunk: Mapping[str, Any], key: str) -> Any:
    """Resolve a structured metadata key, falling back to source_context.

    Plan 10.1-02 (same wave, different files) puts ``authors`` on the
    top-level chunk payload; this must not assume that plan has already
    merged, so it also checks ``chunk["source_context"]`` for the same key.
    """

    value = chunk.get(key)
    if value not in (None, "", [], {}):
        return value
    source_context = chunk.get("source_context")
    if isinstance(source_context, Mapping):
        return source_context.get(key)
    return None


def compact_chunk_evidence(
    chunks: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    # Cross-reference (Phase 10.1-10 Finding A): ``scripts/run_cvpr2026_production.py``
    # defines its OWN, separate compaction function, ``_analysis_compact_chunk``,
    # which is what actually builds the on-disk narrative source packs for
    # CVPR2026 -- this function is NOT called on that path. It reuses this
    # module's ``COMPACT_METADATA_KEYS``/``MAX_COMPACT_AUTHORS``/
    # ``MAX_COMPACT_TITLE_CHARS`` constants but reimplements the preservation
    # loop locally (its bare-key names and quote-length bound differ from
    # this function's). If you change what metadata survives compaction here,
    # change it there too.
    compact: list[dict[str, Any]] = []
    for chunk in chunks[: max(0, int(limit))]:
        quote = " ".join(
            str(chunk.get("verbatim_quote") or chunk.get("quote") or "").split()
        )
        entry: dict[str, Any] = {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "extraction_record_id": str(chunk.get("extraction_record_id") or ""),
            "source_artifact_path": str(chunk.get("source_artifact_path") or ""),
            "quote": quote[:600],
        }
        for key in COMPACT_METADATA_KEYS:
            raw_value = _compact_metadata_value(chunk, key)
            if key == "authors":
                authors = [
                    str(author).strip()
                    for author in (raw_value if isinstance(raw_value, list) else [])
                    if str(author).strip()
                ]
                if authors:
                    entry["authors"] = authors[:MAX_COMPACT_AUTHORS]
                continue
            title_value = str(raw_value or "").strip()
            if title_value:
                entry[key] = title_value[:MAX_COMPACT_TITLE_CHARS]
        compact.append(entry)
    return compact


def compact_matched_papers(
    matched_papers: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_MATCHED_PAPER_TOP_K,
) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for paper in matched_papers[: max(0, int(limit))]:
        compact.append(
            {
                "paper_id": str(paper.get("paper_id") or ""),
                "title": str(paper.get("title") or "Untitled matched paper"),
                "url": str(paper.get("url") or ""),
                "evidence_snippet": " ".join(
                    str(paper.get("evidence_snippet") or "").split()
                )[:700],
                "source_research_path": str(paper.get("source_research_path") or ""),
            }
        )
    return compact


def evidence_pairing_issues(source_pack: Mapping[str, Any]) -> list[dict[str, str]]:
    """Detect brittle paper-to-chunk joins that only rely on list position."""

    chunks = list(source_pack.get("chunks") or [])
    chunk_ids = {str(chunk.get("chunk_id") or "") for chunk in chunks if chunk.get("chunk_id")}
    issues: list[dict[str, str]] = []
    for index, paper in enumerate(source_pack.get("matched_papers") or []):
        if not isinstance(paper, Mapping):
            continue
        title = str(paper.get("title") or paper.get("paper_id") or f"paper-{index}")
        explicit_chunk_id = str(
            paper.get("chunk_id")
            or paper.get("source_chunk_id")
            or paper.get("evidence_chunk_id")
            or ""
        )
        index_pair = (
            paper.get("chunk_index")
            if "chunk_index" in paper
            else paper.get("source_chunk_index")
            if "source_chunk_index" in paper
            else paper.get("paired_chunk_index")
        )
        evidence_snippet = str(paper.get("evidence_snippet") or "").strip()
        if explicit_chunk_id and explicit_chunk_id not in chunk_ids:
            issues.append(
                {
                    "paper": title,
                    "reason": "declared_chunk_id_missing_from_source_pack",
                    "chunk_id": explicit_chunk_id,
                }
            )
        if explicit_chunk_id and not evidence_snippet:
            issues.append(
                {
                    "paper": title,
                    "reason": "paper_chunk_pairing_missing_paper_evidence_snippet",
                    "chunk_id": explicit_chunk_id,
                }
            )
        if index_pair is not None and not evidence_snippet:
            issues.append(
                {
                    "paper": title,
                    "reason": "index_only_paper_chunk_pairing",
                    "chunk_index": str(index_pair),
                }
            )
    return issues


def _reconciliation_text_from_source_pack(source_pack: Mapping[str, Any]) -> str:
    reconciliation = source_pack.get("topic_reconciliation")
    if not isinstance(reconciliation, Mapping):
        return ""
    return str(reconciliation.get("reconciliation_text") or "")


def source_pack_is_sufficient(
    source_pack: Mapping[str, Any],
    *,
    min_chunks: int = DEFAULT_MIN_CHUNK_CITATIONS_PER_TOPIC,
) -> bool:
    return (
        bool(source_pack.get("finding_text"))
        and len(source_pack.get("chunks") or []) >= min_chunks
        and bool(source_pack.get("research_id"))
        and not evidence_pairing_issues(source_pack)
    )


def blocked_narrative_topic_report(
    source_pack: Mapping[str, Any],
    *,
    min_chunks: int = DEFAULT_MIN_CHUNK_CITATIONS_PER_TOPIC,
    reason: str = "insufficient_source_pack",
) -> str:
    title = str(source_pack.get("title") or source_pack.get("topic_id") or "Conference Topic")
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    chunk_count = len(source_pack.get("chunks") or [])
    research_id = str(source_pack.get("research_id") or "")
    has_findings = bool(source_pack.get("finding_text"))
    lines = [
        f"# {title}",
        "",
        "Status: blocked",
        "",
        "## Blocked Receipt",
        "",
        (
            f"`{topic_id}` was not regenerated as a public narrative because the "
            "analysis source pack or narrative synthesis path does not meet the "
            "deep evidence contract."
        ),
        "",
        "## Missing Evidence",
        "",
        f"- Block reason: `{reason}`.",
        f"- Finding text present: `{has_findings}`.",
        f"- Vector/source chunks present: `{chunk_count}` (minimum `{min_chunks}`).",
        f"- Research receipt id present: `{bool(research_id)}`.",
        "",
        "Rerun upstream extraction, vector-store, query-research, paper-matching, and deep narrative synthesis before retrying analysis.",
    ]
    return "\n".join(lines) + "\n"


def build_narrative_prompts(source_pack: Mapping[str, Any]) -> tuple[str, str]:
    title = str(source_pack.get("title") or "Conference Topic")
    source_json = json.dumps(
        {
            "topic_id": source_pack.get("topic_id"),
            "title": title,
            "research_id": source_pack.get("research_id"),
            "research_path": source_pack.get("research_path"),
            "extraction_record_ids": (source_pack.get("extraction_record_ids") or [])[:30],
            "chunks": source_pack.get("chunks") or [],
            "llm_full_read_evidence": source_pack.get("llm_full_read_evidence") or {},
            "topic_reconciliation": source_pack.get("topic_reconciliation") or {},
            "matched_papers": source_pack.get("matched_papers") or [],
        },
        indent=2,
        sort_keys=True,
    )
    paper_texts = source_pack.get("paper_matching_texts") or {}
    system_prompt = (
        "You write deep conference analysis documents for technical audiences. "
        "Use the supplied local source material to synthesize a detailed analyst report, "
        "not a slide brief and not an evidence ledger. Do not invent papers, IDs, "
        "quotes, metrics, URLs, or paths."
    )
    user_prompt = (
        f"Write a rigorous, publication-quality deep narrative analysis for this conference topic: {title}.\n\n"
        "Required markdown shape:\n"
        f"# {title}\n"
        f"Status: {source_pack.get('status') or DEFAULT_NARRATIVE_STATUS}\n\n"
        "## Executive Summary\n"
        "3-5 paragraphs with the core thesis, what changed, and why it matters.\n\n"
        "## Key Findings\n"
        "Bulleted source-backed findings with paper, benchmark, or artifact names when available.\n\n"
        "## Detailed Analysis\n"
        "Include at least 4 `###` subsections. Each subsection should explain concrete methods, "
        "results, limits, benchmarks, and the field-level interpretation. Avoid generic boilerplate.\n\n"
        "## Slide Brief\n"
        "A compact deck-facing summary with 3-5 slide seeds. This is a summary, not the whole report.\n\n"
        "## References And Sources\n"
        "List the main paper titles, local source paths, research IDs, extraction IDs, and web/project resources if present.\n\n"
        "## Cross-References\n"
        "Name related conference topics or analysis docs when visible in the source pack.\n\n"
        "## Recommendations\n"
        "Actionable recommendations for follow-up reading, deck emphasis, or research/product implications.\n\n"
        "## Questions For Further Investigation\n"
        "Open questions, source gaps, or verification work still needed.\n\n"
        "Do not create top-level sections named Source Basis, Cited extraction record ids, "
        "Key Findings (Evidence-Backed), Verbatim transcript/OCR evidence, "
        "Matched Paper Evidence, or Source Gaps. Those belong in provenance sidecars.\n\n"
        "## Structured source pack\n\n"
        f"{source_json}\n\n"
        "## Synthesized findings\n\n"
        f"{source_pack.get('finding_text') or ''}\n\n"
        "## Topic evidence reconciliation\n\n"
        f"{_reconciliation_text_from_source_pack(source_pack)}\n\n"
        "## Paper matching GPT-Researcher findings\n\n"
        f"{paper_texts.get('paper_gpt_researcher_findings') or ''}\n\n"
        "## Artifact GPT-Researcher findings\n\n"
        f"{paper_texts.get('artifact_gpt_researcher_findings') or ''}\n"
    )
    return system_prompt, user_prompt


def _heading_block(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        flags=re.MULTILINE,
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _heading_present(text: str, heading_pattern: str) -> bool:
    return bool(re.search(heading_pattern, text or "", flags=re.MULTILINE | re.IGNORECASE))


def _count_reference_items(text: str) -> int:
    refs = (
        _heading_block(text, "## References And Sources")
        or _heading_block(text, "## References")
        or _heading_block(text, "## Sources")
    )
    return len(re.findall(r"^\s*(?:[-*]|\d+\.)\s+", refs, flags=re.MULTILINE))


def _has_repeated_boilerplate(text: str) -> bool:
    counts: dict[str, int] = {}
    for raw in (text or "").splitlines():
        line = " ".join(raw.strip().split())
        if len(line) < 64:
            continue
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= 3:
            return True
    return False


def validate_narrative_topic_report(text: str) -> dict[str, Any]:
    headings = re.findall(r"^##\s+(.+?)\s*$", text or "", flags=re.MULTILINE)
    forbidden = [heading for heading in headings if heading in _NARRATIVE_FORBIDDEN_HEADINGS]
    major_theme_count = len(
        re.findall(r"^##\s+Major Theme\s+\d+:", text or "", flags=re.MULTILINE)
    )
    detailed = _heading_block(text or "", "## Detailed Analysis")
    detailed_subsection_count = len(re.findall(r"^###\s+", detailed, flags=re.MULTILINE))
    word_count = len(re.findall(r"\b[\w'-]+\b", text or ""))
    reference_item_count = _count_reference_items(text or "")
    has_repeated_boilerplate = _has_repeated_boilerplate(text or "")
    checks: dict[str, Any] = {
        "has_executive_summary": "## Executive Summary" in (text or ""),
        "has_key_findings": "## Key Findings" in (text or ""),
        "has_detailed_analysis": "## Detailed Analysis" in (text or ""),
        "detailed_subsection_count": detailed_subsection_count,
        "has_slide_brief": "## Slide Brief" in (text or ""),
        "has_references_and_sources": _heading_present(
            text or "",
            r"^##\s+(References And Sources|References|Sources)\s*$",
        ),
        "reference_item_count": reference_item_count,
        "has_cross_references": "## Cross-References" in (text or ""),
        "has_recommendations": "## Recommendations" in (text or ""),
        "has_questions": _heading_present(
            text or "",
            r"^##\s+Questions For Further Investigation\s*$",
        ),
        "word_count": word_count,
        "has_repeated_boilerplate": has_repeated_boilerplate,
        "major_theme_count": major_theme_count,
        "has_why_this_matters": "Why This Matters" in (text or ""),
        "has_slide_seeds": "Slide Seeds" in (text or ""),
        "has_concise_references": bool(
            re.search(r"^##\s+Concise References\s*$", text or "", flags=re.MULTILINE)
        ),
        "forbidden_headings": forbidden,
        "starts_with_source_basis": bool(
            re.search(r"^# .+?\n\nStatus:.*?\n\n## Source Basis", text or "", re.S)
        ),
    }
    checks["status"] = (
        "pass"
        if checks["has_executive_summary"]
        and checks["has_key_findings"]
        and checks["has_detailed_analysis"]
        and detailed_subsection_count >= DEFAULT_DEEP_NARRATIVE_MIN_SUBSECTIONS
        and checks["has_slide_brief"]
        and checks["has_references_and_sources"]
        and reference_item_count >= DEFAULT_DEEP_NARRATIVE_MIN_REFERENCES
        and checks["has_cross_references"]
        and checks["has_recommendations"]
        and checks["has_questions"]
        and word_count >= DEFAULT_DEEP_NARRATIVE_MIN_WORDS
        and not has_repeated_boilerplate
        and not forbidden
        and not checks["starts_with_source_basis"]
        else "fail"
    )
    return checks


def clean_narrative_topic_report(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.split(r"\n+If you want,?\s+", cleaned, maxsplit=1)[0].rstrip()
    return cleaned + "\n"


def _validation_retry_prompt(validation: Mapping[str, Any]) -> str:
    failed = [
        key
        for key, value in validation.items()
        if key.startswith("has_") and value is False
    ]
    return (
        "\n\n## Validation repair instructions\n\n"
        "The previous draft failed the production validator. Write a complete "
        "replacement report, not a note about the failure. The replacement must "
        "include these exact top-level sections: Executive Summary, Key Findings, "
        "Detailed Analysis, Slide Brief, References And Sources, Cross-References, "
        "Recommendations, and Questions For Further Investigation. Detailed "
        "Analysis must contain at least four `###` subsections; References And "
        "Sources must contain at least three bullet or numbered items; the report "
        "must be at least 450 words; and it must not include blocked/status "
        "failure language. Current validator result:\n\n"
        f"{json.dumps(validation, indent=2, sort_keys=True)}\n\n"
        f"Missing boolean checks: {', '.join(failed) if failed else 'see validator JSON'}.\n"
    )


def _theme_title_from_paper(index: int, paper: Mapping[str, str], fallback: str) -> str:
    title = str(paper.get("title") or "").strip()
    if not title:
        return fallback
    cleaned = re.sub(r"\s+", " ", title).strip(" .")
    if len(cleaned) > 76:
        cleaned = cleaned[:73].rstrip() + "..."
    return cleaned.title() if cleaned.isupper() else cleaned


def deterministic_narrative_topic_report(source_pack: Mapping[str, Any]) -> str:
    """Provider-free deep preview used by tests, mocks, and offline review.

    Production callers should pass ``allow_deterministic_fallback=False`` to
    :func:`render_narrative_topic_report`.
    """
    title = str(source_pack.get("title") or source_pack.get("topic_id") or "Conference Topic")
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    chunks = source_pack.get("chunks") or []
    papers = source_pack.get("matched_papers") or []
    extraction_ids = source_pack.get("extraction_record_ids") or []
    research_id = str(source_pack.get("research_id") or "")
    status = str(source_pack.get("status") or DEFAULT_NARRATIVE_STATUS)
    finding_text = " ".join(str(source_pack.get("finding_text") or "").split())
    first_quotes = [
        str(chunk.get("quote") or chunk.get("verbatim_quote") or "")
        for chunk in chunks
        if chunk.get("quote") or chunk.get("verbatim_quote")
    ]
    paper_names = [str(paper.get("title") or "Untitled matched paper") for paper in papers[:6]]
    if not paper_names:
        paper_names = [title]
    reference_paths = sorted(
        {
            str(chunk.get("source_artifact_path") or "")
            for chunk in chunks
            if chunk.get("source_artifact_path")
        }
    )
    source_families = sorted(
        {
            (path.split("/", 1)[0] if "/" in path else path)
            for path in reference_paths
            if path
        }
    )

    lines = [
        f"# {title}",
        "",
        f"Status: {status}",
        "",
        "## Executive Summary",
        "",
        (
            f"The `{topic_id}` source pack points to a technical area where individual "
            "papers matter, but the larger story comes from how methods, evaluation "
            "pressure, datasets, and deployment constraints reinforce each other. "
            "The strongest report should therefore preserve paper-level specificity "
            "while also explaining the cross-paper pattern."
        ),
        "",
        (
            f"The synthesized findings emphasize: {finding_text[:900] or 'the supplied research findings are present but require deeper synthesis.'} "
            "This offline preview intentionally stays conservative: it names only "
            "papers and source paths present in the source pack and leaves richer "
            "interpretation to the governed production LLM synthesis path."
        ),
        "",
        "## Key Findings",
        "",
    ]

    for index, name in enumerate(paper_names[:5], start=1):
        lines.append(
            f"- `{name}` is one named anchor for the topic; production synthesis should explain its method, evidence, and limits instead of using it as a catalog entry."
        )
    lines.extend(
        [
            f"- The source pack contains `{len(chunks)}` compact evidence chunks and `{len(extraction_ids)}` extraction ids, which is enough for a traceable deep draft but not a substitute for interpretation.",
            f"- Source-family coverage: {', '.join(source_families) if source_families else 'not recorded in compact source pack'}.",
            "",
            "## Detailed Analysis",
            "",
        ]
    )

    section_specs = [
        ("Paper-Level Architecture Claims", "paper titles, abstracts, and matched-paper snippets"),
        ("Evidence From Local Artifacts", "transcripts, PDFs, source metadata, and vector chunks"),
        ("Benchmarks, Failure Modes, And Evaluation Pressure", "datasets, limitations, and measurement concerns"),
        ("Implications For Future Conference Outputs", "deck, podcast, and synthesis decisions"),
    ]
    paper_anchor_summary = (
        ", ".join(f"`{name}`" for name in paper_names[:3])
        if paper_names
        else "`not recorded in compact source pack`"
    )
    for index, (heading, evidence_focus) in enumerate(section_specs, start=1):
        quote = first_quotes[index - 1] if index - 1 < len(first_quotes) else ""
        source_path = reference_paths[index - 1] if index - 1 < len(reference_paths) else ""
        lines.extend(
            [
                f"### {heading}",
                "",
                (
                    f"This subsection uses {evidence_focus} to keep the `{topic_id}` story "
                    f"grounded. Representative paper anchors include {paper_anchor_summary}. "
                    "The production analysis should connect these anchors through explicit "
                    "paper evidence or research receipts, identify what is technically new, "
                    "and explain why the claim matters beyond a single citation."
                ),
                "",
                (
                    f"Evidence basis: `{source_path or source_pack.get('research_path') or 'local source pack'}`. "
                    f"Representative excerpt: \"{quote[:420] or 'not recorded in compact source pack'}\" "
                    "The excerpt is treated as local evidence, not as a license to "
                    "attach unrelated paper titles to adjacent transcript chunks."
                ),
                "",
                (
                    f"Interpretation {index}: the useful analysis move is to describe the method, "
                    "the evaluation context, and the remaining uncertainty together. "
                    "That is the distinction between a deep report and the previous "
                    "slide-brief contract."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Slide Brief",
            "",
            "- Lead with the cross-paper thesis, not with a table of titles.",
            "- Use one slide for named methods and one slide for evaluation pressure.",
            "- Reserve provenance-heavy details for speaker notes or appendix material.",
            "",
            "## References And Sources",
            "",
            f"- Research receipt: `{research_id or 'not-recorded'}`.",
            f"- Research findings: `{source_pack.get('research_path') or 'not-recorded'}`.",
            "- Matched papers: "
            + (", ".join(f"`{name}`" for name in paper_names[:6]) if paper_names else "not recorded")
            + ".",
            "- Representative extraction ids: "
            + (", ".join(f"`{item}`" for item in extraction_ids[:10]) if extraction_ids else "not recorded")
            + ".",
            "- Representative source paths: "
            + (", ".join(f"`{item}`" for item in reference_paths[:6]) if reference_paths else "not recorded")
            + ".",
            "",
            "## Cross-References",
            "",
            "- Compare this topic with adjacent analysis documents that share papers, benchmarks, model families, or evaluation constraints.",
            "- Use provenance sidecars to recover exact extraction ids and vector chunks for any cross-topic claim.",
            "",
            "## Recommendations",
            "",
            "- Re-run production synthesis with the governed deep narrative LLM before publishing or building decks.",
            "- Promote paper-level sections with concrete mechanisms, metrics, and failure modes over generic theme prose.",
            "- Keep deck summaries downstream of the deep report through `analysis/slide_briefs/`.",
            "",
            "## Questions For Further Investigation",
            "",
            "- Which named findings have external project pages, code repositories, or updated paper versions that should enrich the public report?",
            "- Which local transcript or metadata snippets are too noisy to trust without a paper-text or web-source cross-check?",
            "- Which benchmark claims should be compared across topics before final synthesis?",
        ]
    )
    return "\n".join(lines) + "\n"


def _anchor_papers_from_narrative_text(markdown: str, *, limit: int = 6) -> list[str]:
    """Recover anchor-paper titles from a deep report when no sidecar exists."""

    anchors: list[str] = []
    in_references = False
    in_main_papers = False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_references = stripped == "## References And Sources"
            in_main_papers = False
            continue
        if not in_references:
            continue
        if "Main paper / artifact titles" in stripped:
            in_main_papers = True
            continue
        if in_main_papers and stripped.startswith("- **") and "Presentation ID:" not in stripped:
            in_main_papers = False
        if not in_main_papers or "Presentation ID:" not in stripped:
            continue
        match = re.match(r"^[-*]\s+\*\*(?P<title>[^*]{8,180})\*\*", stripped)
        if not match:
            continue
        title = " ".join(match.group("title").split())
        if title not in anchors:
            anchors.append(title)
        if len(anchors) >= limit:
            break
    return anchors


def render_slide_brief(
    source_pack: Mapping[str, Any],
    *,
    narrative_text: str | None = None,
) -> str:
    title = str(source_pack.get("title") or source_pack.get("topic_id") or "Conference Topic")
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    papers = [str(paper.get("title") or "") for paper in (source_pack.get("matched_papers") or [])]
    papers = [paper for paper in papers if paper]
    if not papers and narrative_text:
        papers = _anchor_papers_from_narrative_text(narrative_text)
    key_findings = _heading_block(narrative_text or "", "## Key Findings")
    findings = [
        re.sub(r"^\s*[-*]\s+", "", line).strip()
        for line in key_findings.splitlines()
        if re.match(r"^\s*[-*]\s+", line)
    ][:4]
    if not findings:
        findings = [
            "Explain the cross-paper thesis in one sentence.",
            "Name the concrete methods or systems that make the topic worth presenting.",
            "Surface the benchmark or failure-mode pressure shaping the claims.",
        ]
    lines = [
        f"# {title} Slide Brief",
        "",
        "Status: slide-brief-v1.63",
        "",
        f"Source deep analysis: `analysis/{topic_id}.md`",
        "",
        "## Key Points",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Slide Seeds",
            "",
            "- Thesis slide: what changed and why it matters now.",
            "- Evidence slide: named papers, local source evidence, and benchmark pressure.",
            "- Implications slide: what this changes for models, data, tools, or evaluation.",
            "",
            "## Anchor Papers",
            "",
        ]
    )
    if papers:
        lines.extend(f"- `{paper}`" for paper in papers[:6])
    else:
        lines.append("- Not recorded in source pack.")
    return "\n".join(lines) + "\n"


def render_narrative_topic_report(
    source_pack: Mapping[str, Any],
    *,
    llm_client: Any | None = None,
    min_chunks: int = DEFAULT_MIN_CHUNK_CITATIONS_PER_TOPIC,
    allow_deterministic_fallback: bool = True,
    max_llm_attempts: int = DEFAULT_DEEP_NARRATIVE_MAX_ATTEMPTS,
) -> str:
    pairing_issues = evidence_pairing_issues(source_pack)
    if pairing_issues:
        return blocked_narrative_topic_report(
            source_pack,
            min_chunks=min_chunks,
            reason="invalid_evidence_pairing",
        )
    if not source_pack_is_sufficient(source_pack, min_chunks=min_chunks):
        return blocked_narrative_topic_report(source_pack, min_chunks=min_chunks)
    failure_reason = "missing_llm_client"
    if llm_client is not None:
        system_prompt, user_prompt = build_narrative_prompts(source_pack)
        attempts = max(1, int(max_llm_attempts))
        for attempt in range(attempts):
            try:
                result = llm_client.synthesize(
                    kind="narrative_analysis",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                text = clean_narrative_topic_report(result.text)
                validation = validate_narrative_topic_report(text)
                if validation.get("status") == "pass":
                    return text
                failure_reason = "llm_output_failed_deep_validation"
                if attempt + 1 < attempts:
                    user_prompt = user_prompt + _validation_retry_prompt(validation)
            except Exception:
                failure_reason = "llm_synthesis_failed"
                if attempt + 1 >= attempts:
                    break
    if not allow_deterministic_fallback:
        return blocked_narrative_topic_report(
            source_pack,
            min_chunks=min_chunks,
            reason=failure_reason,
        )
    return clean_narrative_topic_report(deterministic_narrative_topic_report(source_pack))


def render_topic_provenance_json(
    *,
    topic_id: str,
    markdown_path: str,
    provenance_path: str,
    run_id: str,
    extraction_record_ids: list[str],
    vector_chunks: list[dict[str, Any]],
    research_id: str,
    matched_papers: list[dict[str, Any]],
    llm_full_read_evidence: dict[str, Any] | None = None,
    topic_reconciliation: dict[str, Any] | None = None,
    slide_brief_path: str | None = None,
    analysis_profile: str = DEFAULT_ANALYSIS_PROFILE,
    narrative_model: str | None = None,
    deep_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_extraction_ids = {str(item) for item in (extraction_record_ids or []) if str(item)}
    all_extraction_ids.update(
        str(chunk.get("extraction_record_id") or "")
        for chunk in vector_chunks
        if chunk.get("extraction_record_id")
    )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "topic_id": topic_id,
        "markdown_path": markdown_path,
        "deep_analysis_path": markdown_path,
        "slide_brief_path": slide_brief_path,
        "analysis_profile": analysis_profile,
        "narrative_model": narrative_model,
        "deep_validation": deep_validation or {},
        "provenance_path": provenance_path,
        "extraction_record_ids": sorted(all_extraction_ids),
        "vector_store_chunk_ids": sorted(
            {str(chunk.get("chunk_id") or "") for chunk in vector_chunks if chunk.get("chunk_id")}
        ),
        "research_ids": [research_id] if research_id else [],
        "matched_paper_ids": sorted(
            {
                str(paper.get("paper_id") or "")
                for paper in matched_papers
                if isinstance(paper, dict) and paper.get("paper_id")
            }
        ),
        "chunk_evidence": compact_chunk_evidence(vector_chunks, limit=20),
        "matched_papers": compact_matched_papers(matched_papers, limit=20),
        "llm_full_read_evidence": llm_full_read_evidence or {},
        "topic_reconciliation": topic_reconciliation or {},
    }


def source_pack_content_sha256(source_pack: Mapping[str, Any]) -> str:
    """Digest of everything a narrative is rendered from; the pack's own build time is excluded."""

    content = {key: value for key, value in source_pack.items() if key != "generated_at"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _reuse_existing_narrative_if_current(
    *,
    report_path: Path,
    provenance_json_path: Path,
    source_pack: Mapping[str, Any],
    run_id: str,
) -> str | None:
    """Return an existing passing report when its sidecar matches the source pack.

    A sidecar that recorded its pack digest must match it exactly; older sidecars
    fall back to the research-id and chunk-id comparison.
    """

    if not report_path.exists() or not provenance_json_path.exists():
        return None
    try:
        narrative = report_path.read_text(encoding="utf-8")
        validation = validate_narrative_topic_report(narrative)
        provenance = json.loads(provenance_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if validation.get("status") != "pass":
        return None
    if not isinstance(provenance, Mapping):
        return None
    if provenance.get("run_id") != run_id:
        return None
    if str(provenance.get("topic_id") or "") != str(source_pack.get("topic_id") or ""):
        return None
    recorded_pack_sha256 = str(provenance.get("source_pack_sha256") or "")
    if recorded_pack_sha256 and recorded_pack_sha256 != source_pack_content_sha256(source_pack):
        return None
    research_id = str(source_pack.get("research_id") or "")
    if research_id:
        existing_research_ids = {
            str(item) for item in (provenance.get("research_ids") or []) if str(item)
        }
        if research_id not in existing_research_ids:
            return None
    expected_chunk_ids = {
        str(chunk.get("chunk_id") or "")
        for chunk in (source_pack.get("chunks") or [])
        if isinstance(chunk, Mapping) and chunk.get("chunk_id")
    }
    existing_chunk_ids = {
        str(item) for item in (provenance.get("vector_store_chunk_ids") or []) if str(item)
    }
    if expected_chunk_ids and expected_chunk_ids != existing_chunk_ids:
        return None
    return clean_narrative_topic_report(narrative)


def narrative_report_is_current(
    workspace_root: Path,
    *,
    run_id: str,
    source_pack: Mapping[str, Any],
    narrative_model: str,
    analysis_profile: str,
) -> bool:
    """Whether the topic's existing report can stand in for a fresh synthesis.

    True only when the report passes deep validation and its provenance
    sidecar records this run, this exact source-pack digest, this narrative
    model and this analysis profile -- the inputs a paid re-render would use.
    A sidecar that never recorded its pack digest is not current.
    """

    try:
        topic_id = _validate_topic_id_for_path(str(source_pack.get("topic_id") or ""))
    except ValueError:
        return False
    analysis_root = Path(workspace_root) / "analysis"
    report_path = analysis_root / f"{topic_id}.md"
    provenance_json_path = analysis_root / "provenance" / f"{topic_id}.json"
    if (
        _reuse_existing_narrative_if_current(
            report_path=report_path,
            provenance_json_path=provenance_json_path,
            source_pack=source_pack,
            run_id=run_id,
        )
        is None
    ):
        return False
    try:
        provenance = json.loads(provenance_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(provenance, Mapping)
        and provenance.get("source_pack_sha256") == source_pack_content_sha256(source_pack)
        and provenance.get("narrative_model") == narrative_model
        and provenance.get("analysis_profile") == analysis_profile
    )


def render_provenance_markdown(
    source_pack: Mapping[str, Any],
    *,
    run_id: str,
    markdown_path: str,
    provenance_json_path: str | None = None,
) -> str:
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    title = str(source_pack.get("title") or topic_id)
    chunks = list(source_pack.get("chunks") or [])
    extraction_ids = [
        str(item)
        for item in (source_pack.get("extraction_record_ids") or [])
        if str(item)
    ]
    for chunk in chunks:
        erid = str(chunk.get("extraction_record_id") or "")
        if erid and erid not in extraction_ids:
            extraction_ids.append(erid)
    research_id = str(source_pack.get("research_id") or "")
    matched_papers = list(source_pack.get("matched_papers") or [])
    topic_reconciliation = (
        source_pack.get("topic_reconciliation")
        if isinstance(source_pack.get("topic_reconciliation"), dict)
        else {}
    )
    lines = [
        f"# {title} Provenance",
        "",
        "Status: provenance-sidecar",
        "",
        "## Source Basis",
        "",
        f"- Public narrative: `{markdown_path}`",
        f"- Run id: `{run_id}`",
        f"- Topic id: `{topic_id}`",
    ]
    if provenance_json_path:
        lines.append(f"- Structured provenance: `{provenance_json_path}`")
    if source_pack.get("research_path"):
        lines.append(f"- Research findings: `{source_pack.get('research_path')}`")
    if research_id:
        lines.append(f"- Research id: `{research_id}` <!-- research:{research_id} -->")
    if topic_reconciliation:
        lines.append(
            "- Topic reconciliation: "
            f"`{topic_reconciliation.get('markdown_path') or 'not-recorded'}`"
        )

    lines.extend(["", "## Extraction Evidence", ""])
    if extraction_ids:
        for aid in sorted(set(extraction_ids)):
            lines.append(f"- `{aid}` <!-- extraction:{aid} -->")
    else:
        lines.append("- No extraction ids recorded.")

    lines.extend(["", "## Vector Chunk Evidence", ""])
    if chunks:
        for index, chunk in enumerate(chunks, start=1):
            cid = str(chunk.get("chunk_id") or "")
            erid = str(chunk.get("extraction_record_id") or "")
            quote = " ".join(str(chunk.get("quote") or chunk.get("verbatim_quote") or "").split())
            source_path = str(chunk.get("source_artifact_path") or "")
            lines.extend(
                [
                    f"{index}. `{cid or 'unknown-chunk'}` <!-- chunk:{cid or 'unknown'} -->",
                    f"   - Extraction record id: `{erid or 'not-recorded'}`",
                    f"   - Source artifact path: `{source_path or 'not-recorded'}`",
                    f"   - Quote: \"{quote[:360] or 'not-recorded'}\"",
                ]
            )
    else:
        lines.append("- No vector chunks recorded.")

    lines.extend(["", "## Matched Paper Evidence", ""])
    if matched_papers:
        for index, paper in enumerate(matched_papers[:DEFAULT_MATCHED_PAPER_TOP_K], start=1):
            paper_id = str(paper.get("paper_id") or "")
            title_text = str(paper.get("title") or "Untitled matched paper")
            snippet = " ".join(str(paper.get("evidence_snippet") or "").split())[:360]
            lines.extend(
                [
                    f"{index}. `{paper_id or 'unknown-paper'}`: {title_text}",
                    f"   - Evidence snippet: {snippet or 'not-recorded'}",
                    f"   <!-- matched-paper:{paper_id or 'unknown'} -->",
                ]
            )
    else:
        lines.append("- No matched-paper output recorded.")

    lines.extend(["", "## Source Gaps", ""])
    gaps = []
    if not extraction_ids:
        gaps.append("No extraction ids recorded.")
    if not chunks:
        gaps.append("No vector chunks recorded.")
    if not research_id:
        gaps.append("No research id recorded.")
    if not matched_papers:
        gaps.append("No matched-paper output recorded.")
    if not topic_reconciliation:
        gaps.append("No topic reconciliation output recorded.")
    if gaps:
        lines.extend(f"- {gap}" for gap in gaps)
    else:
        lines.append("- All required provenance channels are populated.")
    return "\n".join(lines) + "\n"


def citation_summary_from_source_pack(
    source_pack: Mapping[str, Any],
    *,
    markdown_path: str,
    slide_brief_path: str | None = None,
    provenance_path: str,
    provenance_json_path: str,
    analysis_profile: str = DEFAULT_ANALYSIS_PROFILE,
    narrative_model: str | None = None,
    deep_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunks = list(source_pack.get("chunks") or [])
    extraction_ids = {
        str(item)
        for item in (source_pack.get("extraction_record_ids") or [])
        if str(item)
    }
    extraction_ids.update(
        str(chunk.get("extraction_record_id") or "")
        for chunk in chunks
        if chunk.get("extraction_record_id")
    )
    research_id = str(source_pack.get("research_id") or "")
    matched_papers = list(source_pack.get("matched_papers") or [])
    topic_reconciliation = (
        source_pack.get("topic_reconciliation")
        if isinstance(source_pack.get("topic_reconciliation"), dict)
        else {}
    )
    return {
        "topic_id": str(source_pack.get("topic_id") or Path(markdown_path).stem),
        "markdown_path": markdown_path,
        "deep_analysis_path": markdown_path,
        "slide_brief_path": slide_brief_path,
        "analysis_profile": analysis_profile,
        "narrative_model": narrative_model,
        "deep_validation": deep_validation or {},
        "provenance_path": provenance_path,
        "provenance_json_path": provenance_json_path,
        "extraction_record_ids": sorted(extraction_ids),
        "vector_store_chunk_ids": sorted(
            {str(chunk.get("chunk_id") or "") for chunk in chunks if chunk.get("chunk_id")}
        ),
        "research_ids": [research_id] if research_id else [],
        "matched_paper_ids": sorted(
            {
                str(paper.get("paper_id") or "")
                for paper in matched_papers
                if isinstance(paper, dict) and paper.get("paper_id")
            }
        ),
        "topic_reconciliation": topic_reconciliation or {},
    }


def emit_analysis_citation_manifest(
    workspace_root: Path,
    run_id: str,
    per_topic_citations: list[dict[str, Any]],
    *,
    evidence_chain_version: str = "v1",
) -> Path:
    analysis_dir = workspace_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "evidence_chain_version": evidence_chain_version,
        "topics": [
            {
                "topic_id": entry.get("topic_id"),
                "markdown_path": entry.get("markdown_path"),
                "deep_analysis_path": entry.get("deep_analysis_path") or entry.get("markdown_path"),
                "slide_brief_path": entry.get("slide_brief_path"),
                "analysis_profile": entry.get("analysis_profile") or DEFAULT_ANALYSIS_PROFILE,
                "narrative_model": entry.get("narrative_model"),
                "deep_validation": entry.get("deep_validation") or {},
                "provenance_path": entry.get("provenance_path"),
                "provenance_json_path": entry.get("provenance_json_path"),
                "extraction_record_ids": sorted(set(entry.get("extraction_record_ids") or [])),
                "vector_store_chunk_ids": sorted(set(entry.get("vector_store_chunk_ids") or [])),
                "research_ids": sorted(set(entry.get("research_ids") or [])),
                "matched_paper_ids": sorted(set(entry.get("matched_paper_ids") or [])),
            }
            for entry in per_topic_citations
        ],
    }
    out_path = analysis_dir / "analysis_citation_manifest.json"
    write_json_atomic(out_path, payload)
    return out_path


def load_analysis_citation_manifest_topics(workspace_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = workspace_root / "analysis" / "analysis_citation_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in topics:
        if not isinstance(entry, dict):
            continue
        topic_id = str(entry.get("topic_id") or "")
        if topic_id:
            out[topic_id] = entry
    return out


def citation_entry_for_report(
    citation_topics: Mapping[str, dict[str, Any]],
    report_rel: str,
) -> dict[str, Any] | None:
    normalized = report_rel.replace("\\", "/")
    for entry in citation_topics.values():
        if str(entry.get("markdown_path") or "").replace("\\", "/") == normalized:
            return entry
    return citation_topics.get(Path(normalized).stem)


def citation_ids_from_manifest(
    workspace_root: Path,
    report_paths: list[str],
    field: str,
) -> set[str]:
    citation_topics = load_analysis_citation_manifest_topics(workspace_root)
    if not citation_topics:
        return set()
    cited: set[str] = set()
    for relative in report_paths:
        entry = citation_entry_for_report(citation_topics, relative)
        if not entry:
            continue
        cited.update(str(item) for item in entry.get(field) or [] if item)
    return cited


def _collect_marker_citations(
    workspace_root: Path,
    report_paths: list[str],
    pattern: re.Pattern[str],
) -> set[str]:
    cited: set[str] = set()
    for relative in report_paths:
        path = workspace_root / relative
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        cited.update(pattern.findall(text))
    return cited


def collect_extraction_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    cited = _collect_marker_citations(workspace_root, report_paths, _EXTRACTION_CITATION_RE)
    cited.update(citation_ids_from_manifest(workspace_root, report_paths, "extraction_record_ids"))
    return cited


def collect_vector_store_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    cited = _collect_marker_citations(workspace_root, report_paths, _VECTOR_STORE_CITATION_RE)
    cited.update(citation_ids_from_manifest(workspace_root, report_paths, "vector_store_chunk_ids"))
    return cited


def collect_query_research_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    cited = _collect_marker_citations(workspace_root, report_paths, _QUERY_RESEARCH_CITATION_RE)
    cited.update(citation_ids_from_manifest(workspace_root, report_paths, "research_ids"))
    return cited


def collect_matched_paper_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    cited = _collect_marker_citations(workspace_root, report_paths, _MATCHED_PAPER_CITATION_RE)
    cited.update(citation_ids_from_manifest(workspace_root, report_paths, "matched_paper_ids"))
    return cited


def load_narrative_source_packs(
    analysis_dir: Path,
    *,
    source_pack_dir: Path | None = None,
) -> list[dict[str, Any]]:
    source_root = source_pack_dir or analysis_dir / NARRATIVE_SOURCE_PACK_DIRNAME
    if not source_root.exists():
        return []
    packs: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        pack = dict(raw)
        pack.setdefault("topic_id", path.stem)
        pack.setdefault("title", path.stem.replace("_", " ").title())
        pack["chunks"] = compact_chunk_evidence(list(pack.get("chunks") or []), limit=20)
        pack["matched_papers"] = compact_matched_papers(
            list(pack.get("matched_papers") or []),
            limit=20,
        )
        packs.append(pack)
    return packs


def render_narrative_outputs_from_source_packs(
    workspace_root: Path,
    *,
    run_id: str,
    analysis_dir: Path | None = None,
    source_pack_dir: Path | None = None,
    llm_client: Any | None = None,
    evidence_chain_version: str = "v1",
    allow_deterministic_fallback: bool = True,
    narrative_model: str | None = None,
    analysis_profile: str = DEFAULT_ANALYSIS_PROFILE,
    reuse_existing_pass_reports: bool = True,
    carry_forward_topic_ids: frozenset[str] | None = None,
) -> NarrativeOutputResult:
    """Render every topic's narrative from its source pack.

    Topics in ``carry_forward_topic_ids`` are never re-rendered: their existing
    report and sidecars are kept untouched when they are still current for the
    pack, and the topic blocks otherwise. This lets a paid pass re-render only
    the topics whose packs were rebuilt.
    """

    analysis_root = analysis_dir or workspace_root / "analysis"
    packs = load_narrative_source_packs(analysis_root, source_pack_dir=source_pack_dir)
    if not packs:
        return NarrativeOutputResult(
            status="skipped",
            report_paths=(),
            provenance_paths=(),
            provenance_json_paths=(),
            citation_manifest_path=None,
            citation_summaries=(),
        )

    analysis_root.mkdir(parents=True, exist_ok=True)
    provenance_dir = analysis_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    slide_brief_dir = analysis_root / SLIDE_BRIEF_DIRNAME
    slide_brief_dir.mkdir(parents=True, exist_ok=True)

    report_paths: list[Path] = []
    provenance_paths: list[Path] = []
    provenance_json_paths: list[Path] = []
    slide_brief_paths: list[Path] = []
    citation_summaries: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for pack in packs:
        raw_topic_id = str(pack.get("topic_id") or "")
        if not raw_topic_id:
            blockers.append(
                {
                    "code": "narrative_source_pack_missing_topic_id",
                    "retryable": True,
                }
            )
            continue
        topic_id = _validate_topic_id_for_path(raw_topic_id)
        markdown_rel = f"analysis/{topic_id}.md"
        slide_brief_rel = f"analysis/{SLIDE_BRIEF_DIRNAME}/{topic_id}.md"
        provenance_rel = f"analysis/provenance/{topic_id}.md"
        provenance_json_rel = f"analysis/provenance/{topic_id}.json"
        workspace_root_resolved = workspace_root.resolve()
        report_path = workflow._resolve_path_inside_workspace(
            workspace_root_resolved,
            markdown_rel,
            label="narrative report path",
        )
        provenance_path = workflow._resolve_path_inside_workspace(
            workspace_root_resolved,
            provenance_rel,
            label="narrative provenance path",
        )
        slide_brief_path = workflow._resolve_path_inside_workspace(
            workspace_root_resolved,
            slide_brief_rel,
            label="narrative slide brief path",
        )
        provenance_json_path = workflow._resolve_path_inside_workspace(
            workspace_root_resolved,
            provenance_json_rel,
            label="narrative provenance json path",
        )

        if carry_forward_topic_ids is not None and topic_id in carry_forward_topic_ids:
            carried = _reuse_existing_narrative_if_current(
                report_path=report_path,
                provenance_json_path=provenance_json_path,
                source_pack=pack,
                run_id=run_id,
            )
            if carried is None:
                blockers.append(
                    {
                        "code": "narrative_carry_forward_not_current",
                        "retryable": True,
                        "topic_id": topic_id,
                        "evidence": markdown_rel,
                    }
                )
                continue
            deep_validation = validate_narrative_topic_report(carried)
            report_paths.append(report_path)
            provenance_paths.append(provenance_path)
            provenance_json_paths.append(provenance_json_path)
            slide_brief_paths.append(slide_brief_path)
            citation_summaries.append(
                citation_summary_from_source_pack(
                    pack,
                    markdown_path=markdown_rel,
                    slide_brief_path=slide_brief_rel,
                    provenance_path=provenance_rel,
                    provenance_json_path=provenance_json_rel,
                    analysis_profile=analysis_profile,
                    narrative_model=narrative_model,
                    deep_validation=deep_validation,
                )
            )
            continue

        narrative = None
        if reuse_existing_pass_reports:
            narrative = _reuse_existing_narrative_if_current(
                report_path=report_path,
                provenance_json_path=provenance_json_path,
                source_pack=pack,
                run_id=run_id,
            )
        if narrative is None:
            narrative = render_narrative_topic_report(
                pack,
                llm_client=llm_client,
                allow_deterministic_fallback=allow_deterministic_fallback,
            )
        deep_validation = validate_narrative_topic_report(narrative)
        report_path.write_text(narrative, encoding="utf-8")
        slide_brief = render_slide_brief(pack, narrative_text=narrative)
        slide_brief_path.write_text(slide_brief, encoding="utf-8")
        provenance = render_provenance_markdown(
            pack,
            run_id=run_id,
            markdown_path=markdown_rel,
            provenance_json_path=provenance_json_rel,
        )
        provenance_path.write_text(provenance, encoding="utf-8")
        research_id = str(pack.get("research_id") or "")
        chunks = list(pack.get("chunks") or [])
        matched_papers = list(pack.get("matched_papers") or [])
        llm_full_read_evidence = pack.get("llm_full_read_evidence")
        if not isinstance(llm_full_read_evidence, dict):
            llm_full_read_evidence = {}
        topic_reconciliation = pack.get("topic_reconciliation")
        if not isinstance(topic_reconciliation, dict):
            topic_reconciliation = {}
        extraction_ids = [
            str(item)
            for item in (pack.get("extraction_record_ids") or [])
            if str(item)
        ]
        provenance_json = render_topic_provenance_json(
            topic_id=topic_id,
            markdown_path=markdown_rel,
            provenance_path=provenance_rel,
            run_id=run_id,
            extraction_record_ids=extraction_ids,
            vector_chunks=chunks,
            research_id=research_id,
            matched_papers=matched_papers,
            llm_full_read_evidence=llm_full_read_evidence,
            topic_reconciliation=topic_reconciliation,
            slide_brief_path=slide_brief_rel,
            analysis_profile=analysis_profile,
            narrative_model=narrative_model,
            deep_validation=deep_validation,
        )
        provenance_json["source_pack_sha256"] = source_pack_content_sha256(pack)
        write_json_atomic(provenance_json_path, provenance_json)

        report_paths.append(report_path)
        provenance_paths.append(provenance_path)
        provenance_json_paths.append(provenance_json_path)
        slide_brief_paths.append(slide_brief_path)
        citation_summaries.append(
            citation_summary_from_source_pack(
                pack,
                markdown_path=markdown_rel,
                slide_brief_path=slide_brief_rel,
                provenance_path=provenance_rel,
                provenance_json_path=provenance_json_rel,
                analysis_profile=analysis_profile,
                narrative_model=narrative_model,
                deep_validation=deep_validation,
            )
        )
        if deep_validation.get("status") != "pass":
            blockers.append(
                {
                    "code": "narrative_source_pack_insufficient",
                    "retryable": True,
                    "topic_id": topic_id,
                    "evidence": markdown_rel,
                }
            )

    citation_manifest_path = emit_analysis_citation_manifest(
        workspace_root,
        run_id,
        citation_summaries,
        evidence_chain_version=evidence_chain_version,
    )
    return NarrativeOutputResult(
        status="blocked" if blockers else "succeeded",
        report_paths=tuple(report_paths),
        provenance_paths=tuple(provenance_paths),
        provenance_json_paths=tuple(provenance_json_paths),
        citation_manifest_path=citation_manifest_path,
        citation_summaries=tuple(citation_summaries),
        slide_brief_paths=tuple(slide_brief_paths),
        blockers=tuple(blockers),
    )
