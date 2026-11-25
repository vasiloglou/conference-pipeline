"""v1.91 artifact-grounded paper matching.

Canonical flow:

artifact-store GPT-Researcher findings -> configured frontier keypoints ->
paper-store GPT-Researcher -> returned paper titles.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

import yaml

from conference_pipeline.iclr2025_live import utc_now
from conference_pipeline.model_governance import default_model, default_provider
from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline._security.loopback import is_loopback_url

PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N = 20
# Reasoning-model local runtimes (e.g. mlx_lm serving Qwen3.6's thinking
# variant) emit chain-of-thought tokens inside the same completion-token
# budget as the final answer. Leaving max_tokens unset falls back to the
# local server's own low default (observed: 512), which the model
# consistently exhausts on reasoning before ever emitting an answer,
# yielding an empty completion. Mirrors the DEFAULT_MAX_OUTPUT_TOKENS
# convention already used for this exact local model in
# analysis/topic_reconciliation.py, analysis/topic_reference_repair.py,
# and analysis/exhaustive_summaries.py.
PAPER_MATCHING_DEFAULT_RESEARCH_MAX_OUTPUT_TOKENS = 16_000
MATCHED_PAPERS_SUPERSEDED_INFIX = ".superseded-"
# Artifact findings are truncated to this many characters in the keypoint
# prompt. costs.estimator.PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP sizes the
# keypoint call from the same value (a test keeps the two equal).
KEYPOINT_FINDINGS_CHAR_CAP = 24_000
# Provider-receipt identity for paper matching. Without a workspace-rooted
# receipt context the receipts fell back to Path.cwd() and landed in the
# repository-root storage/provider-calls/matcher/, outside every
# conference's cost totals.
PAPER_MATCHING_STAGE_ID = "paper-matching"
# Joins the keypoint call's receipts to costs/task_specs.yaml in the rollup.
PAPER_MATCHING_KEYPOINT_TASK_ID = "paper_matching_keypoint_extraction"


@contextlib.contextmanager
def paper_matching_receipt_scope(
    workspace_root: Path,
    run_id: str,
    *,
    task_id: str | None = None,
) -> Iterator[None]:
    """Route provider receipts written inside to this conference workspace.

    An enclosing context's run, stage, caller, and task identity are kept
    (a runner may already scope the stage); the workspace root is always the
    conference being matched.
    """

    from conference_pipeline.provider_receipts import (
        current_provider_receipt_context,
        provider_receipt_context,
    )

    current = current_provider_receipt_context()
    with provider_receipt_context(
        workspace_root=Path(workspace_root),
        run_id=current.run_id or run_id,
        stage_id=current.stage_id or PAPER_MATCHING_STAGE_ID,
        caller=current.caller,
        task_id=task_id or current.task_id,
    ):
        yield


def retain_superseded_matched_papers(path: Path, previous: bytes | None) -> Path | None:
    """Keep the prior matched list beside the new one when a re-run changed it.

    Matching is an LLM judgment that shifts between runs; consumers may rank a
    paper an earlier run matched after every current match instead of losing it.
    """

    if previous is None or not path.exists():
        return None
    try:
        old_papers = json.loads(previous.decode("utf-8")).get("papers")
        new_papers = json.loads(path.read_text(encoding="utf-8")).get("papers")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        old_papers, new_papers = None, []
    if old_papers == new_papers:
        return None
    stamp = utc_now().replace("-", "").replace(":", "")
    retained = path.with_name(f"{path.stem}{MATCHED_PAPERS_SUPERSEDED_INFIX}{stamp}{path.suffix}")
    retained.write_bytes(previous)
    return retained


class PaperMatchingError(RuntimeError):
    """Fail-closed boundary error for v1.91 paper matching."""


class KeypointExtractor(Protocol):
    """Extract topic keypoints from artifact-store GPT-Researcher findings."""

    def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:
        ...


class PaperResearcher(Protocol):
    """Subset of the repository Researcher protocol needed for matching."""

    def run(self, query: str, *, context_path: Path | None = None) -> Any:
        ...


class ResearchRunner(Protocol):
    """Store-grounded research runner boundary used by the matching flow."""

    def research(
        self,
        *,
        topic_id: str,
        query: str,
        output_path: Path,
    ) -> str:
        ...


@dataclass(frozen=True)
class PaperMatchingConfig:
    """Runtime config for the artifact-grounded paper matching stage."""

    enabled: bool = True
    keypoint_provider: str = field(
        default_factory=lambda: default_provider("research.paper_matching_keypoints")
    )
    keypoint_model: str = field(
        default_factory=lambda: default_model("research.paper_matching_keypoints")
    )
    keypoint_source: str = "artifact_gpt_researcher_findings"
    paper_research_store: str = "papers"
    artifact_research_store: str = "artifacts"
    output_top_n: int = PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N
    research_provider: str = field(
        default_factory=lambda: default_provider("research.gpt_researcher_smart")
    )
    research_model: str = field(
        default_factory=lambda: default_model("research.gpt_researcher_smart")
    )
    research_max_output_tokens: int = PAPER_MATCHING_DEFAULT_RESEARCH_MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class MatchedPaper:
    rank: int
    title: str
    paper_id: str = ""
    url: str = ""
    evidence_snippet: str = ""
    source: str = "paper_store_gpt_researcher"
    store_kind: str = "papers"
    keypoint_trace: tuple[str, ...] = field(default_factory=tuple)
    source_research_path: str = ""


@dataclass(frozen=True)
class PaperMatchResult:
    topic_id: str
    run_id: str
    artifact_research_path: Path
    keypoints_path: Path
    paper_research_path: Path
    matched_papers_path: Path
    ranked_papers_path: Path | None
    matched_papers: tuple[MatchedPaper, ...] = field(default_factory=tuple)


class HeuristicKeypointExtractor:
    """Deterministic extractor for explicit tests, fixtures, and dry-run callers."""

    def __init__(self, *, max_keypoints: int = 8) -> None:
        self.max_keypoints = max(1, int(max_keypoints))

    def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:  # noqa: ARG002
        candidates: list[str] = []
        for raw_line in artifact_findings.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^[-*]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)
            if len(line) < 16:
                continue
            candidates.append(line[:240])
            if len(candidates) >= self.max_keypoints:
                break
        if candidates:
            return _dedupe_strings(candidates)[: self.max_keypoints]
        fallback = artifact_findings.strip().replace("\n", " ")
        return [fallback[:240]] if fallback else []


class OpenAIKeypointExtractor:
    """Frontier-model keypoint extractor configured by workflow.paper_matching."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_keypoints: int = 8,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_keypoints = max(1, int(max_keypoints))
        if not self.api_key:
            raise PaperMatchingError("missing_openai_api_key")

    def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PaperMatchingError("openai_not_available") from None  # noqa: B904
        from conference_pipeline.provider_receipts import openai_chat_completions_create

        prompt = (
            "Extract the most important topic keypoints from artifact-store "
            "GPT-Researcher findings. These keypoints will be used to query "
            "the paper vector store for papers that match what the oral/session "
            "artifacts discuss. Return strict JSON: "
            '{"keypoints": ["...", "..."]}.\n\n'
            f"Topic: {topic_id}\n\n"
            f"Artifact findings:\n{artifact_findings[:KEYPOINT_FINDINGS_CHAR_CAP]}"
        )
        try:
            response = openai_chat_completions_create(
                OpenAI(api_key=self.api_key),
                caller=__file__,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract concise, evidence-grounded research "
                            "keypoints for downstream paper matching."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary
            raise PaperMatchingError("openai_keypoint_extraction_failed") from exc
        content = response.choices[0].message.content or ""
        keypoints = _extract_keypoints_from_model_text(content)
        if not keypoints:
            raise PaperMatchingError("empty_keypoints")
        return keypoints[: self.max_keypoints]


class LocalKeypointExtractor:
    """Local OpenAI-compatible keypoint extractor for artifact-grounded matching."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        max_keypoints: int = 8,
        _client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("PAPER_MATCHING_LOCAL_LLM_BASE_URL")
            or os.environ.get("GPTR_LOCAL_LLM_BASE_URL")
            or "http://127.0.0.1:8080/v1"
        ).rstrip("/")
        if not is_loopback_url(self.base_url):
            raise PaperMatchingError("local_keypoint_non_loopback_base_url")
        self.max_keypoints = max(1, int(max_keypoints))
        self._client = _client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PaperMatchingError("openai_not_available_for_local_keypoints") from None
        return OpenAI(base_url=self.base_url, api_key="local-only")

    def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:
        from conference_pipeline.provider_receipts import local_openai_chat_completions_create

        prompt = (
            "Extract the most important topic keypoints from artifact-store "
            "GPT-Researcher findings. These keypoints will be used to query "
            "the paper vector store for papers that match what the oral/session "
            "artifacts discuss. Return strict JSON: "
            '{"keypoints": ["...", "..."]}.\n\n'
            f"Topic: {topic_id}\n\n"
            f"Artifact findings:\n{artifact_findings[:KEYPOINT_FINDINGS_CHAR_CAP]}"
        )
        try:
            response = local_openai_chat_completions_create(
                self._get_client(),
                caller=__file__,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract concise, evidence-grounded research "
                            "keypoints for downstream paper matching."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary
            raise PaperMatchingError("local_keypoint_extraction_failed") from exc
        content = response.choices[0].message.content or ""
        keypoints = _extract_keypoints_from_model_text(content)
        if not keypoints:
            raise PaperMatchingError("empty_keypoints")
        return keypoints[: self.max_keypoints]


class StoreBackedResearchRunner:
    """Run research with retrieved chunks from one Chroma store and no web fallback."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        run_id: str,
        store_kind: str,
        retriever: Any,
        researcher: PaperResearcher,
        k: int = 12,
        filter_by_topic: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.run_id = run_id
        self.store_kind = store_kind
        self.retriever = retriever
        self.researcher = researcher
        self.k = max(1, int(k))
        self.filter_by_topic = bool(filter_by_topic)

    def research(
        self,
        *,
        topic_id: str,
        query: str,
        output_path: Path,
    ) -> str:
        retrieval_topic_id = topic_id if self.filter_by_topic else ""
        chunks = self.retriever.retrieve(
            topic_id=retrieval_topic_id,
            query=query,
            k=self.k,
        )
        if not chunks:
            raise PaperMatchingError(f"{self.store_kind}_store_retrieval_empty")
        prompt = self._build_prompt(topic_id=topic_id, query=query, chunks=chunks)
        output = _run_researcher(self.researcher, prompt, output_path.parent)
        if not output.strip():
            raise PaperMatchingError(f"{self.store_kind}_store_research_empty")
        _write_text(output_path, output.rstrip() + "\n")
        return output

    def _build_prompt(self, *, topic_id: str, query: str, chunks: list[Any]) -> str:
        chunk_blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source_path = str(getattr(chunk, "source_artifact_path", "") or "")
            chunk_id = str(getattr(chunk, "chunk_id", "") or f"chunk-{index}")
            text = str(getattr(chunk, "text", "") or "")
            chunk_blocks.append(
                f"[{index}] chunk_id={chunk_id}\nsource={source_path}\n{text[:4000]}"
            )
        return (
            f"Use only the {self.store_kind} vector store evidence below. "
            "Do not use web search or Tavily fallback.\n\n"
            f"Topic: {topic_id}\n\n"
            f"Research query:\n{query}\n\n"
            "Retrieved store evidence:\n"
            + "\n\n".join(chunk_blocks)
        )


class DirectResearchRunner:
    """Explicit test/compatibility runner that bypasses retrieval by injection."""

    def __init__(self, researcher: PaperResearcher, *, context_path: Path | None = None) -> None:
        self.researcher = researcher
        self.context_path = context_path

    def research(
        self,
        *,
        topic_id: str,
        query: str,
        output_path: Path,
    ) -> str:  # noqa: ARG002
        output = _run_researcher(self.researcher, query, self.context_path or output_path.parent)
        _write_text(output_path, output.rstrip() + "\n")
        return output


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _topic_label_from_id(topic_id: str) -> str:
    label = re.sub(r"^\d+[_-]*", "", str(topic_id or "").strip())
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label or str(topic_id or "topic").strip() or "topic"


def _artifact_findings_report_no_topic_evidence(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").lower())
    no_evidence_markers = (
        "no presentations or transcripts",
        "not present in the provided artifacts",
        "not present in the provided artifact",
        "not present in the provided evidence",
        "no analysis of claims",
        "cannot be generated from the given evidence",
    )
    return any(marker in normalized for marker in no_evidence_markers)


def _topic_fallback_keypoints(topic_id: str, *, max_keypoints: int) -> list[str]:
    label = _topic_label_from_id(topic_id)
    candidates = [
        label,
        f"{label} methods and technical mechanisms",
        f"{label} datasets, benchmarks, and evaluation",
        f"{label} applications and empirical findings",
        f"{label} limitations and open questions",
    ]
    return _dedupe_strings(candidates)[: max(1, int(max_keypoints))]


def _fallback_keypoints_for_empty_extraction(
    *,
    topic_id: str,
    artifact_findings: str,
    max_keypoints: int,
) -> tuple[list[str], str]:
    if _artifact_findings_report_no_topic_evidence(artifact_findings):
        return (
            _topic_fallback_keypoints(topic_id, max_keypoints=max_keypoints),
            "artifact_store_reported_no_topic_evidence",
        )
    keypoints = HeuristicKeypointExtractor(max_keypoints=max_keypoints).extract_keypoints(
        topic_id=topic_id,
        artifact_findings=artifact_findings,
    )
    return keypoints, "frontier_keypoint_extraction_empty_heuristic_fallback"


def _json_candidates(text: str) -> list[Any]:
    candidates: list[str] = [text]
    candidates.extend(
        match.group(1)
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    )
    parsed: list[Any] = []
    for candidate in candidates:
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return parsed


def _extract_keypoints_from_model_text(text: str) -> list[str]:
    for parsed in _json_candidates(text):
        if isinstance(parsed, dict) and isinstance(parsed.get("keypoints"), list):
            return _dedupe_strings(
                [str(item).strip() for item in parsed["keypoints"] if str(item).strip()]
            )
        if isinstance(parsed, list):
            return _dedupe_strings([str(item).strip() for item in parsed if str(item).strip()])
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if len(line) >= 16:
            lines.append(line)
    return _dedupe_strings(lines)


def _papers_from_json(value: Any) -> list[MatchedPaper]:
    rows: list[Any] = []
    if isinstance(value, dict):
        for key in ("papers", "matched_papers", "results", "paper_matches"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
    elif isinstance(value, list):
        rows = value
    papers: list[MatchedPaper] = []
    for index, item in enumerate(rows, start=1):
        if isinstance(item, str):
            title = item.strip()
            paper_id = url = evidence = ""
        elif isinstance(item, dict):
            title = str(
                item.get("title")
                or item.get("paper_title")
                or item.get("name")
                or ""
            ).strip()
            paper_id = str(item.get("paper_id") or item.get("id") or "").strip()
            raw_url = str(item.get("url") or item.get("paper_url") or item.get("pdf_url") or "").strip()
            url = _canonical_paper_url(raw_url, paper_id=paper_id)
            evidence = str(
                item.get("evidence_snippet")
                or item.get("snippet")
                or item.get("evidence")
                or item.get("reason")
                or ""
            ).strip()
        else:
            continue
        if title:
            papers.append(
                MatchedPaper(
                    rank=index,
                    title=title,
                    paper_id=paper_id,
                    url=url,
                    evidence_snippet=evidence,
                )
            )
    return papers


def _openreview_url_id(url: str) -> str:
    parsed = urlsplit(url)
    if "openreview.net" not in parsed.netloc.lower():
        return ""
    if parsed.path.startswith("/forum"):
        return (parse_qs(parsed.query).get("id") or [""])[0].strip()
    if parsed.path.startswith("/pdf"):
        path_id = parsed.path.removeprefix("/pdf/").removesuffix(".pdf").strip("/")
        if path_id:
            return path_id
        return (parse_qs(parsed.query).get("id") or [""])[0].strip()
    return ""


def _looks_like_openreview_note_id(value: str, *, allow_long: bool) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if len(candidate) > (40 if allow_long else 24):
        return False
    if re.match(r"^(ICLR\d{4}_|NeurIPS\d{4}_|CVPR\d{4}_)", candidate, re.I):
        return False
    if candidate.lower() in {"paper", "paper_url", "pdf", "url", "openreview"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{6,40}", candidate))


def _canonical_paper_url(url: str, *, paper_id: str = "") -> str:
    """Return a safe paper URL or blank when model output cannot be trusted.

    Paper matching asks an LLM/researcher to emit paper URLs, but those URLs are
    downstream slide hyperlinks. Keep only http(s) URLs, and for OpenReview
    require consistency with the extracted paper id when one is present. This
    prevents a model from pairing a valid-looking but unrelated OpenReview PDF id
    with a different paper.
    """

    normalized = url.strip()
    normalized_paper_id = paper_id.strip()
    if normalized and not re.match(r"https?://", normalized):
        normalized = ""
    if normalized:
        parsed = urlsplit(normalized)
        if "openreview.net" in parsed.netloc.lower():
            url_id = _openreview_url_id(normalized)
            if not _looks_like_openreview_note_id(url_id, allow_long=True):
                return ""
            if normalized_paper_id and url_id != normalized_paper_id:
                return ""
        return normalized
    if _looks_like_openreview_note_id(normalized_paper_id, allow_long=False):
        return f"https://openreview.net/forum?id={normalized_paper_id}"
    return ""


def _paper_from_line(line: str, rank: int) -> MatchedPaper | None:
    stripped = line.strip()
    if not stripped:
        return None
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
    if re.match(r"^#+\s+", stripped):
        stripped = re.sub(r"^#+\s+", "", stripped)
    title_match = re.search(
        r"(?:paper\s+title|title|paper)\s*[:\-]\s*(.+)$",
        stripped,
        re.IGNORECASE,
    )
    title = title_match.group(1).strip() if title_match else ""
    url = ""
    if not title:
        link_match = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", stripped)
        if link_match:
            title = link_match.group(1).strip()
            url = link_match.group(2).strip()
    if not title and stripped.lower().startswith("matched paper"):
        title = stripped.split(":", 1)[-1].strip()
    title = re.sub(r"\s+[-–—]\s+.*$", "", title).strip()
    title = title.strip('"`* ')
    if not title or len(title) < 4:
        return None
    return MatchedPaper(rank=rank, title=title, url=url)


def extract_papers_from_research(
    research_text: str, *, output_top_n: int = 20
) -> list[MatchedPaper]:
    """Extract returned paper titles from paper-store GPT-Researcher output."""
    papers: list[MatchedPaper] = []
    for parsed in _json_candidates(research_text):
        papers.extend(_papers_from_json(parsed))
    if not papers:
        for line in research_text.splitlines():
            parsed = _paper_from_line(line, len(papers) + 1)
            if parsed is not None:
                papers.append(parsed)
    deduped: list[MatchedPaper] = []
    seen: set[str] = set()
    for paper in papers:
        key = paper.title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            MatchedPaper(
                rank=len(deduped) + 1,
                title=paper.title,
                paper_id=paper.paper_id,
                url=paper.url,
                evidence_snippet=paper.evidence_snippet,
                source=paper.source,
                store_kind=paper.store_kind,
                keypoint_trace=paper.keypoint_trace,
                source_research_path=paper.source_research_path,
            )
        )
        if len(deduped) >= output_top_n:
            break
    return deduped


def build_artifact_research_prompt(*, topic_id: str) -> str:
    return (
        "Use the artifact vector store only. Analyze what the oral/session "
        "artifacts discuss for this topic. Focus on claims, recurring themes, "
        "technical mechanisms, evaluation concerns, and presentation emphasis. "
        "Do not select papers in this step.\n\n"
        f"Topic: {topic_id}"
    )


def build_paper_research_prompt(
    *, topic_id: str, keypoints: list[str], output_top_n: int
) -> str:
    keypoint_block = "\n".join(f"- {keypoint}" for keypoint in keypoints)
    return (
        "Use the paper vector store only. Find papers that match what the "
        "oral/session artifacts discuss. Do not rank by generic popularity, "
        "citation count, or podcast frequency. The retrieved store evidence is "
        "the candidate set; choose the best matching candidates from it and "
        "never return an empty `papers` list when candidates are present. If "
        "evidence is sparse, choose the closest candidates by topic, title, "
        "and abstract text.\n\n"
        f"Topic: {topic_id}\n"
        "Artifact-derived keypoints:\n"
        f"{keypoint_block}\n\n"
        f"Return up to {output_top_n} papers as strict compact JSON only. "
        "Do not include prose, markdown, code fences, or extra fields. Use a "
        "top-level `papers` list. Each paper object must contain exactly "
        "`title`, `paper_id`, `url`, and `evidence_snippet`; keep each "
        "`evidence_snippet` under 160 characters. Extract `title` from the "
        "paper title text in the evidence, and extract `paper_id` from the "
        "OpenReview id in the source or title prefix. Only include a `url` "
        "when it is explicitly present in the retrieved evidence and matches "
        "that `paper_id`; otherwise leave `url` blank."
    )


def _run_researcher(researcher: PaperResearcher, query: str, context_path: Path) -> str:
    result = researcher.run(query, context_path=context_path)
    if asyncio.iscoroutine(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return str(asyncio.run(result))
        else:
            loop = asyncio.new_event_loop()
            try:
                return str(loop.run_until_complete(result))
            finally:
                loop.close()
    return str(result)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_keypoints(
    path: Path,
    *,
    topic_id: str,
    run_id: str,
    config: PaperMatchingConfig,
    artifact_research_path: Path,
    keypoints: list[str],
    fallback_reason: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "topic_id": topic_id,
        "run_id": run_id,
        "keypoint_source": config.keypoint_source,
        "keypoint_provider": config.keypoint_provider,
        "keypoint_model": config.keypoint_model,
        "artifact_research_path": str(artifact_research_path),
        "created_at": utc_now(),
        "keypoints": keypoints,
    }
    if fallback_reason:
        payload["keypoint_fallback_reason"] = fallback_reason
    write_json_atomic(path, payload)


def _write_matched_papers(
    path: Path,
    *,
    topic_id: str,
    run_id: str,
    config: PaperMatchingConfig,
    artifact_research_path: Path,
    keypoints_path: Path,
    paper_research_path: Path,
    papers: list[MatchedPaper],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "run_id": run_id,
            "matching_basis": "artifact_keypoints_to_paper_store_gpt_researcher",
            "keypoint_source": config.keypoint_source,
            "keypoint_provider": config.keypoint_provider,
            "keypoint_model": config.keypoint_model,
            "artifact_research_store": config.artifact_research_store,
            "paper_research_store": config.paper_research_store,
            "artifact_research_path": str(artifact_research_path),
            "keypoints_path": str(keypoints_path),
            "paper_research_path": str(paper_research_path),
            "source_research_paths": {
                "artifact": str(artifact_research_path),
                "paper": str(paper_research_path),
            },
            "created_at": utc_now(),
            "papers": [asdict(paper) for paper in papers],
        },
    )


def _write_ranked_compat(path: Path, papers: list[MatchedPaper]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "source": "matched_papers.json",
            "compatibility": "legacy_ranked_papers_for_neurips2025_deck_scripts",
            "papers": [
                {
                    "rank": paper.rank,
                    "title": paper.title,
                    "paper_id": paper.paper_id,
                    "url": paper.url,
                    "pdf_url": paper.url,
                    "score": round(max(0.0, 1.0 - ((paper.rank - 1) * 0.01)), 4),
                    "why_selected": paper.evidence_snippet
                    or "Returned by paper-store GPT-Researcher from artifact-derived keypoints.",
                    "source": paper.source,
                    "store_kind": paper.store_kind,
                    "keypoint_trace": list(paper.keypoint_trace),
                    "source_research_path": paper.source_research_path,
                }
                for paper in papers
            ],
        },
    )


def _positive_int(value: Any, *, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _load_workflow_config_data(workspace_root: Path) -> dict[str, Any]:
    config_path = Path(workspace_root) / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def configured_paper_matching_output_top_n(workspace_root: Path) -> int:
    """Return the reusable paper-matching output cap from conference config."""

    payload = _load_workflow_config_data(workspace_root)
    workflow_data = payload.get("workflow") if isinstance(payload, dict) else {}
    matching_data = workflow_data.get("paper_matching") if isinstance(workflow_data, dict) else {}
    value = (
        matching_data.get("output_top_n")
        if isinstance(matching_data, dict)
        else PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N
    )
    return _positive_int(value, fallback=PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N)


def configured_deck_max_papers(workspace_root: Path, *, fallback: int) -> int:
    """Return workflow.deck.max_papers without allowing invalid low/empty values."""

    payload = _load_workflow_config_data(workspace_root)
    workflow_data = payload.get("workflow") if isinstance(payload, dict) else {}
    deck_data = workflow_data.get("deck") if isinstance(workflow_data, dict) else {}
    if not isinstance(deck_data, dict) or "max_papers" not in deck_data:
        return _positive_int(fallback, fallback=PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N)
    return _positive_int(deck_data.get("max_papers"), fallback=fallback)


def verify_paper_matching_caps(
    workspace_root: Path,
    *,
    paper_matching_output_top_n: int | None = None,
    analysis_matched_paper_top_k: int | None = None,
    deck_max_papers: int | None = None,
) -> dict[str, Any]:
    """Verify downstream paper caps do not undercut paper-matching output."""

    configured = configured_paper_matching_output_top_n(workspace_root)
    observed = {
        "workflow.paper_matching.output_top_n": configured,
        "paper_matching.output_top_n": _positive_int(
            paper_matching_output_top_n
            if paper_matching_output_top_n is not None
            else configured,
            fallback=configured,
        ),
        "analysis.matched_paper_top_k": _positive_int(
            analysis_matched_paper_top_k
            if analysis_matched_paper_top_k is not None
            else configured,
            fallback=configured,
        ),
        "workflow.deck.max_papers": _positive_int(
            deck_max_papers
            if deck_max_papers is not None
            else configured_deck_max_papers(workspace_root, fallback=configured),
            fallback=configured,
        ),
    }
    blockers: list[dict[str, Any]] = []
    for cap_name, value in observed.items():
        if cap_name == "workflow.paper_matching.output_top_n":
            continue
        if value < configured:
            blockers.append(
                {
                    "code": "paper_matching_cap_below_configured_output_top_n",
                    "retryable": False,
                    "cap_name": cap_name,
                    "observed": value,
                    "minimum": configured,
                    "reason": (
                        "Paper matching, analysis ingestion, and deck paper caps must not "
                        "silently undercut workflow.paper_matching.output_top_n."
                    ),
                }
            )
    return {
        "schema_version": 1,
        "status": "pass" if not blockers else "blocked",
        "configured_source_of_truth": "workflow.paper_matching.output_top_n",
        "observed_caps": observed,
        "blockers": blockers,
    }


def load_paper_matching_config(
    workspace_root: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> PaperMatchingConfig:
    """Load workflow.paper_matching from a conference config with defaults."""
    workspace_root = Path(workspace_root)
    data = _load_workflow_config_data(workspace_root)
    workflow_data = data.get("workflow") if isinstance(data.get("workflow"), dict) else {}
    matching_data = (
        workflow_data.get("paper_matching")
        if isinstance(workflow_data.get("paper_matching"), dict)
        else {}
    )
    research_data = workflow_data.get("research") if isinstance(workflow_data.get("research"), dict) else {}
    values = {
        "enabled": bool(matching_data.get("enabled", True)),
        "keypoint_provider": str(
            matching_data.get(
                "keypoint_provider",
                default_provider("research.paper_matching_keypoints"),
            )
        ),
        "keypoint_model": str(
            matching_data.get(
                "keypoint_model",
                default_model("research.paper_matching_keypoints"),
            )
        ),
        "keypoint_source": str(
            matching_data.get("keypoint_source", "artifact_gpt_researcher_findings")
        ),
        "paper_research_store": str(matching_data.get("paper_research_store", "papers")),
        "artifact_research_store": str(matching_data.get("artifact_research_store", "artifacts")),
        "output_top_n": configured_paper_matching_output_top_n(workspace_root),
        "research_provider": str(
            matching_data.get(
                "research_provider",
                research_data.get("provider")
                or research_data.get("report_provider")
                or default_provider("research.gpt_researcher_smart"),
            )
        ),
        "research_model": str(
            matching_data.get(
                "research_model",
                research_data.get("model")
                or research_data.get("report_model")
                or default_model("research.gpt_researcher_smart"),
            )
        ),
        "research_max_output_tokens": matching_data.get(
            "research_max_output_tokens",
            research_data.get("max_output_tokens")
            or PAPER_MATCHING_DEFAULT_RESEARCH_MAX_OUTPUT_TOKENS,
        ),
    }
    env_overrides = {
        "keypoint_provider": os.environ.get("PAPER_MATCHING_KEYPOINT_PROVIDER"),
        "keypoint_model": os.environ.get("PAPER_MATCHING_KEYPOINT_MODEL"),
        "research_provider": os.environ.get("PAPER_MATCHING_RESEARCH_PROVIDER"),
        "research_model": os.environ.get("PAPER_MATCHING_RESEARCH_MODEL"),
        "research_max_output_tokens": os.environ.get(
            "PAPER_MATCHING_RESEARCH_MAX_OUTPUT_TOKENS"
        ),
    }
    for key, value in env_overrides.items():
        if value not in (None, ""):
            values[key] = str(value)
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
    values["output_top_n"] = _positive_int(
        values["output_top_n"],
        fallback=PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N,
    )
    values["research_max_output_tokens"] = _positive_int(
        values["research_max_output_tokens"],
        fallback=PAPER_MATCHING_DEFAULT_RESEARCH_MAX_OUTPUT_TOKENS,
    )
    return PaperMatchingConfig(**values)


def build_keypoint_extractor(config: PaperMatchingConfig) -> KeypointExtractor:
    provider = config.keypoint_provider.strip().lower()
    if provider == "openai":
        return OpenAIKeypointExtractor(
            model=config.keypoint_model,
            max_keypoints=config.output_top_n,
        )
    if provider in {"local", "mlx", "ollama", "local-openai-compatible"}:
        return LocalKeypointExtractor(
            model=config.keypoint_model,
            max_keypoints=config.output_top_n,
        )
    if provider in {"fake", "heuristic"}:
        return HeuristicKeypointExtractor(max_keypoints=config.output_top_n)
    raise PaperMatchingError(f"unsupported_keypoint_provider:{config.keypoint_provider}")


def build_store_research_runner(
    *,
    workspace_root: Path,
    run_id: str,
    store_kind: str,
    config: PaperMatchingConfig,
    filter_by_topic: bool = False,
) -> StoreBackedResearchRunner:
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        build_researcher,
    )

    return StoreBackedResearchRunner(
        workspace_root=workspace_root,
        run_id=run_id,
        store_kind=store_kind,
        retriever=ChromaVectorStoreRetriever(
            workspace_root,
            run_id=run_id,
            store_kind=store_kind,
        ),
        researcher=build_researcher(
            {
                "provider": config.research_provider,
                "model": config.research_model,
                "max_tokens": config.research_max_output_tokens,
            }
        ),
        filter_by_topic=filter_by_topic,
    )


def _decorate_papers(
    papers: list[MatchedPaper],
    *,
    keypoints: list[str],
    paper_research_path: Path,
    store_kind: str,
) -> list[MatchedPaper]:
    return [
        replace(
            paper,
            rank=index,
            keypoint_trace=tuple(keypoints),
            store_kind=store_kind,
            source_research_path=str(paper_research_path),
        )
        for index, paper in enumerate(papers, start=1)
    ]


def run_paper_matching(
    *,
    workspace_root: Path,
    run_id: str,
    topic_id: str,
    **kwargs: Any,
) -> PaperMatchResult:
    """Run v1.91 paper matching for one topic, with workspace-rooted receipts."""
    with paper_matching_receipt_scope(Path(workspace_root), run_id):
        return _run_paper_matching_in_scope(
            workspace_root=workspace_root,
            run_id=run_id,
            topic_id=topic_id,
            **kwargs,
        )


def _run_paper_matching_in_scope(
    *,
    workspace_root: Path,
    run_id: str,
    topic_id: str,
    artifact_findings_path: Path | None = None,
    output_dir: Path | None = None,
    decks_dir: Path | None = None,
    config: PaperMatchingConfig | None = None,
    keypoint_extractor: KeypointExtractor | None = None,
    artifact_research_runner: ResearchRunner | None = None,
    paper_research_runner: ResearchRunner | None = None,
    paper_researcher: PaperResearcher | None = None,
    output_top_n: int | None = None,
    write_compat_ranked: bool = True,
) -> PaperMatchResult:
    workspace_root = Path(workspace_root)
    config = config or load_paper_matching_config(workspace_root)
    if output_top_n is not None:
        config = replace(config, output_top_n=int(output_top_n))
    if not config.enabled:
        raise PaperMatchingError("paper_matching_disabled")
    if output_dir is None:
        output_dir = workspace_root / "storage" / "paper-matching" / run_id / topic_id
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_research_path = output_dir / "artifact_gpt_researcher_findings.md"
    if artifact_findings_path is not None:
        artifact_findings_path = Path(artifact_findings_path)
        artifact_findings = artifact_findings_path.read_text(encoding="utf-8")
        if artifact_findings_path.resolve() != artifact_research_path.resolve():
            _write_text(artifact_research_path, artifact_findings.rstrip() + "\n")
    else:
        runner = artifact_research_runner or build_store_research_runner(
            workspace_root=workspace_root,
            run_id=run_id,
            store_kind=config.artifact_research_store,
            config=config,
        )
        artifact_findings = runner.research(
            topic_id=topic_id,
            query=build_artifact_research_prompt(topic_id=topic_id),
            output_path=artifact_research_path,
        )

    extractor = keypoint_extractor or build_keypoint_extractor(config)
    keypoint_fallback_reason: str | None = None
    try:
        with paper_matching_receipt_scope(
            workspace_root, run_id, task_id=PAPER_MATCHING_KEYPOINT_TASK_ID
        ):
            keypoints = extractor.extract_keypoints(
                topic_id=topic_id,
                artifact_findings=artifact_findings,
            )
    except PaperMatchingError as exc:
        if str(exc) != "empty_keypoints":
            raise
        keypoints, keypoint_fallback_reason = _fallback_keypoints_for_empty_extraction(
            topic_id=topic_id,
            artifact_findings=artifact_findings,
            max_keypoints=config.output_top_n,
        )
        if not keypoints:
            raise
        fallback_label = (
            "topic_fallback"
            if keypoint_fallback_reason == "artifact_store_reported_no_topic_evidence"
            else "heuristic_fallback"
        )
        config = replace(
            config,
            keypoint_provider=f"{config.keypoint_provider}+{fallback_label}",
        )
    keypoints = _dedupe_strings(keypoints)
    if not keypoints:
        raise PaperMatchingError("empty_keypoints")

    keypoints_path = output_dir / "keypoints.json"
    _write_keypoints(
        keypoints_path,
        topic_id=topic_id,
        run_id=run_id,
        config=config,
        artifact_research_path=artifact_research_path,
        keypoints=keypoints,
        fallback_reason=keypoint_fallback_reason,
    )

    paper_research_path = output_dir / "paper_gpt_researcher_findings.md"
    prompt = build_paper_research_prompt(
        topic_id=topic_id,
        keypoints=keypoints,
        output_top_n=config.output_top_n,
    )
    if paper_research_runner is None and paper_researcher is not None:
        paper_research_runner = DirectResearchRunner(
            paper_researcher,
            context_path=workspace_root
            / "storage"
            / "vector-store"
            / run_id
            / "stores"
            / config.paper_research_store,
        )
    runner = paper_research_runner or build_store_research_runner(
        workspace_root=workspace_root,
        run_id=run_id,
        store_kind=config.paper_research_store,
        config=config,
    )
    paper_research = runner.research(
        topic_id=topic_id,
        query=prompt,
        output_path=paper_research_path,
    )
    papers = extract_papers_from_research(
        paper_research,
        output_top_n=max(1, int(config.output_top_n)),
    )
    if not papers:
        raise PaperMatchingError("paper_store_research_returned_no_titles")
    papers = _decorate_papers(
        papers,
        keypoints=keypoints,
        paper_research_path=paper_research_path,
        store_kind=config.paper_research_store,
    )
    matched_papers_path = output_dir / "matched_papers.json"
    previous_matched = matched_papers_path.read_bytes() if matched_papers_path.exists() else None
    _write_matched_papers(
        matched_papers_path,
        topic_id=topic_id,
        run_id=run_id,
        config=config,
        artifact_research_path=artifact_research_path,
        keypoints_path=keypoints_path,
        paper_research_path=paper_research_path,
        papers=papers,
    )
    retain_superseded_matched_papers(matched_papers_path, previous_matched)
    ranked_path: Path | None = None
    if write_compat_ranked:
        ranked_root = decks_dir or workspace_root / "decks" / topic_id
        ranked_root.mkdir(parents=True, exist_ok=True)
        ranked_path = ranked_root / "ranked_papers.json"
        _write_ranked_compat(ranked_path, papers)
    return PaperMatchResult(
        topic_id=topic_id,
        run_id=run_id,
        artifact_research_path=artifact_research_path,
        keypoints_path=keypoints_path,
        paper_research_path=paper_research_path,
        matched_papers_path=matched_papers_path,
        ranked_papers_path=ranked_path,
        matched_papers=tuple(papers),
    )


def _discover_topic_ids(workspace_root: Path) -> list[str]:
    for path in (
        workspace_root / "analysis" / "analysis_manifest.json",
        workspace_root / "analysis" / "iclr2025_taxonomy.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        topics = payload.get("topics") if isinstance(payload, dict) else None
        if isinstance(topics, list):
            out: list[str] = []
            for topic in topics:
                if isinstance(topic, dict):
                    value = topic.get("id") or topic.get("topic_id") or topic.get("slug")
                else:
                    value = topic
                if isinstance(value, str) and value.strip():
                    out.append(value.strip())
            if out:
                return out
    return []


def run_paper_matching_stage(
    *,
    workspace_root: Path,
    run_id: str,
    topic_id: str | None = None,
    config: PaperMatchingConfig | None = None,
) -> dict[str, Any]:
    """Workflow-stage entrypoint for v1.91 paper matching."""
    workspace_root = Path(workspace_root)
    config = config or load_paper_matching_config(workspace_root)
    topics = [topic_id] if topic_id else _discover_topic_ids(workspace_root)
    if not topics:
        raise PaperMatchingError("paper_matching_topic_required")
    results = [
        run_paper_matching(
            workspace_root=workspace_root,
            run_id=run_id,
            topic_id=topic,
            config=config,
        )
        for topic in topics
    ]
    artifacts: list[dict[str, object]] = []
    for result in results:
        artifacts.extend(
            [
                {
                    "kind": "paper_matching_artifact_research",
                    "topic_id": result.topic_id,
                    "path": result.artifact_research_path.relative_to(workspace_root).as_posix(),
                },
                {
                    "kind": "paper_matching_keypoints",
                    "topic_id": result.topic_id,
                    "path": result.keypoints_path.relative_to(workspace_root).as_posix(),
                },
                {
                    "kind": "paper_matching_paper_research",
                    "topic_id": result.topic_id,
                    "path": result.paper_research_path.relative_to(workspace_root).as_posix(),
                },
                {
                    "kind": "matched_papers",
                    "topic_id": result.topic_id,
                    "path": result.matched_papers_path.relative_to(workspace_root).as_posix(),
                },
            ]
        )
        if result.ranked_papers_path is not None:
            artifacts.append(
                {
                    "kind": "legacy_ranked_papers_compat",
                    "topic_id": result.topic_id,
                    "path": result.ranked_papers_path.relative_to(workspace_root).as_posix(),
                }
            )
    return {
        "status": "succeeded",
        "run_id": run_id,
        "topics": [result.topic_id for result in results],
        "artifacts": artifacts,
    }
