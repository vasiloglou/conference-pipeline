"""Topic-level evidence reconciliation before final narrative synthesis.

The final public topic report should not make the slide generator reconcile
parallel analysis products for the first time. This module performs the bounded
pre-slide reconciliation step: for each topic, combine the exhaustive summary
packs, dual deep-dive GPT-Researcher reports, and current narrative source pack
into a compact analyst brief that can be injected back into the source pack.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from conference_pipeline.analysis.topic_reference_validation import (
    SOURCE_ID_TOKEN_RE,
    TopicReferenceLedger,
    build_reference_ledger,
    canonicalize_reference_id,
    format_reference_ledger_prompt_section,
    normalize_text,
    validate_reconciliation_references,
)
from conference_pipeline.receipt_lineage import stamp_substep_receipt, unbind_substep_receipt
from conference_pipeline.research.deep_dive_lanes import DEFAULT_QWEN36_MODEL
from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.workflow_runs import write_json_atomic

OWNER_TOPIC_RECONCILIATION = "analysis.topic_reconciliation"
RECEIPT_TYPE_TOPIC_RECONCILIATION = "conference_pipeline.topic_reconciliation"
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_INPUT_TOKEN_CEILING = 220_000
DEFAULT_MAX_OUTPUT_TOKENS = 16_000


class TopicReconciliationError(RuntimeError):
    """Raised when topic reconciliation cannot safely continue."""


class AsyncResearcherLike(Protocol):
    provider_receipt_paths: list[str]
    model: str

    async def run(self, query: str, *, context_path: Path | None = None) -> str: ...

    def usage(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReconciliationInput:
    topic_id: str
    title: str
    source_pack_path: Path
    source_pack: dict[str, Any]
    paper_summary_path: Path | None
    transcript_summary_path: Path | None
    deep_dive_paths: tuple[Path, ...]
    input_digest_sha256: str
    input_token_estimate: int
    prompt: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pack_input_sha256(source_pack: Mapping[str, Any]) -> str:
    payload = dict(source_pack)
    payload.pop("topic_reconciliation", None)
    payload.pop("generated_at", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicReconciliationError(f"json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise TopicReconciliationError(f"json_not_object:{path}")
    return payload


def estimate_tokens(text: str) -> int:
    """Fast token estimate using tiktoken when available.

    Qwen tokenization is close enough for budget gating at these input sizes;
    exact local-tokenizer counts are unnecessary unless the prompt approaches
    the configured context limit.
    """

    try:
        import tiktoken  # type: ignore[import-not-found]

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:  # pragma: no cover - optional dependency fallback.
        return max(1, len(text) // 4)


def _topic_prefix(topic_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", topic_id).strip("_")
    return normalized or "topic"


def deep_dive_id_prefix(topic_id: str) -> str:
    """Mirror the selected deep-dive filename prefix convention."""

    return _topic_prefix(topic_id)[:32].strip("._") or "topic"


def _deep_dive_markdown_paths(workspace: Path, topic_id: str) -> tuple[Path, ...]:
    root = workspace / "analysis" / "deep_dive_topics"
    if not root.exists():
        return ()
    prefixes = {topic_id, deep_dive_id_prefix(topic_id)}
    paths: set[Path] = set()
    for prefix in prefixes:
        paths.update(root.glob(f"{prefix}__dd[0-9][0-9]_*__global.md"))
        paths.update(root.glob(f"{prefix}__dd[0-9][0-9]_*__heuristic.md"))
    return tuple(sorted(paths))


def _canonicalize_or_keep(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    return canonicalize_reference_id(raw) or raw


def _canonicalized_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(chunk)
    for key in ("chunk_id", "extraction_record_id"):
        if key in compact:
            compact[key] = _canonicalize_or_keep(compact.get(key))
    return compact


def _canonicalized_selected_item(item: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(item)
    if "unit_id" in compact:
        compact["unit_id"] = _canonicalize_or_keep(compact.get("unit_id"))
    return compact


def _compact_source_pack(source_pack: Mapping[str, Any]) -> dict[str, Any]:
    """Build the source-pack JSON embedded in the reconciliation/repair prompts.

    Every id-shaped field (chunk_id, extraction_record_id, unit_id, the
    top-level extraction_record_ids list) is canonicalized to the same
    paper_abstract:/speech_transcript: reference-id space the Reference Ledger
    section uses, BEFORE it reaches the model. The raw corpus artifact-id
    prefixes (paper:/workshop_paper:/event_media_caption:) are internal
    bookkeeping identifiers, not citable ids -- leaking them here previously let
    the model copy a raw id verbatim into its citations (the prompt explicitly
    said to "preserve ... ids ... when visible"), which the fixed-width
    SOURCE_ID_TOKEN_RE citation matcher then partially truncates on slash
    characters (workshop-paper native ids contain slashes) into garbage tokens
    that never resolve against the ledger, producing spurious unknown-reference
    blockers unrelated to any real hallucinated citation.
    """
    llm_evidence = source_pack.get("llm_full_read_evidence")
    if not isinstance(llm_evidence, Mapping):
        llm_evidence = {}
    return {
        "topic_id": source_pack.get("topic_id"),
        "title": source_pack.get("title"),
        "research_id": source_pack.get("research_id"),
        "research_path": source_pack.get("research_path"),
        "finding_text": source_pack.get("finding_text"),
        "chunks": [
            _canonicalized_chunk(chunk)
            for chunk in list(source_pack.get("chunks") or [])[:12]
            if isinstance(chunk, Mapping)
        ],
        "llm_full_read_evidence": {
            "status": llm_evidence.get("status"),
            "unit_count": llm_evidence.get("unit_count"),
            "input_kind_counts": llm_evidence.get("input_kind_counts"),
            "selected_items": [
                _canonicalized_selected_item(item)
                for item in list(llm_evidence.get("selected_items") or [])[:24]
                if isinstance(item, Mapping)
            ],
        },
        "matched_papers": list(source_pack.get("matched_papers") or [])[:20],
        "extraction_record_ids": [
            _canonicalize_or_keep(reference_id)
            for reference_id in list(source_pack.get("extraction_record_ids") or [])[:40]
        ],
    }


def build_reconciliation_prompt(
    *,
    topic_id: str,
    title: str,
    source_pack: Mapping[str, Any],
    paper_summary_text: str,
    transcript_summary_text: str,
    deep_dive_texts: Mapping[str, str],
    reference_ledger_text: str = "",
) -> str:
    source_pack_json = json.dumps(
        _compact_source_pack(source_pack),
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    )
    deep_dive_blocks = []
    for label, text in sorted(deep_dive_texts.items()):
        deep_dive_blocks.append(f"## Deep Dive Report: {label}\n\n{text.strip()}")
    deep_dive_section = "\n\n".join(deep_dive_blocks)
    return (
        "You are reconciling all available analysis evidence for one conference "
        "topic before final public narrative synthesis. Produce a compact but "
        "substantive analyst reconciliation brief. Use only supplied evidence. "
        "Do not invent papers, quotes, speaker names, metrics, URLs, or IDs.\n\n"
        f"Topic id: {topic_id}\n"
        f"Topic title: {title}\n\n"
        "Output markdown with this exact shape:\n"
        "# Topic Evidence Reconciliation\n"
        "## Consensus Thesis\n"
        "## Evidence Map\n"
        "### Papers\n"
        "### Transcripts And Talks\n"
        "### Deep-Dive Research\n"
        "## Agreements Across Sources\n"
        "## Tensions And Disagreements\n"
        "## Strongest Slide-Worthy Claims\n"
        "## Evidence To Preserve In Final Narrative\n"
        "## Caveats And Source Gaps\n"
        "## Recommended Final Narrative Emphasis\n\n"
        "Reconciliation rules:\n"
        "- Treat exhaustive summaries as broad coverage evidence.\n"
        "- Treat global and heuristic deep-dive reports as complementary research lenses.\n"
        "- Explicitly call out where deep dives agree or disagree with exhaustive summaries.\n"
        "- Prefer named methods, benchmarks, datasets, systems, talks, and paper titles when present.\n"
        "- Preserve source paths and short quotes when they are visible.\n"
        "- Cite ONLY the canonical ids listed in the Reference Ledger section above (paper_abstract:.../"
        "speech_transcript:...). Never cite a raw source-pack id directly -- the chunk_id, "
        "extraction_record_id, and unit_id fields shown in the source pack JSON below are internal "
        "bookkeeping identifiers, not citable reference ids. Do not attach a title to an id unless that "
        "title is in the same ledger row.\n"
        "- Keep it compact enough to feed into a later GPT-5.6 Luna final narrative prompt.\n\n"
        f"{reference_ledger_text.strip()}\n\n"
        "## Current Narrative Source Pack\n\n"
        f"{source_pack_json}\n\n"
        "## Exhaustive Paper Summary\n\n"
        f"{paper_summary_text.strip() or 'Missing.'}\n\n"
        "## Exhaustive Transcript Summary\n\n"
        f"{transcript_summary_text.strip() or 'Missing.'}\n\n"
        "## Dual Deep-Dive Reports\n\n"
        f"{deep_dive_section or 'Missing.'}\n"
    )


def collect_reconciliation_input(
    workspace: Path,
    *,
    topic_id: str,
    input_token_ceiling: int = DEFAULT_INPUT_TOKEN_CEILING,
) -> ReconciliationInput:
    workspace = workspace.resolve()
    source_pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    if not source_pack_path.exists():
        raise TopicReconciliationError(f"source_pack_missing:{topic_id}")
    source_pack = _load_json_object(source_pack_path)
    title = str(source_pack.get("title") or topic_id)
    paper_summary_path = workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.md"
    transcript_summary_path = workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__transcripts.md"
    if not paper_summary_path.exists():
        paper_summary_path = None
    if not transcript_summary_path.exists():
        transcript_summary_path = None
    deep_dive_paths = _deep_dive_markdown_paths(workspace, topic_id)
    deep_dive_texts = {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in deep_dive_paths
    }
    reference_ledger_text = format_reference_ledger_prompt_section(
        build_reference_ledger(workspace, topic_id)
    )
    prompt = build_reconciliation_prompt(
        topic_id=topic_id,
        title=title,
        source_pack=source_pack,
        paper_summary_text=_read_text(paper_summary_path),
        transcript_summary_text=_read_text(transcript_summary_path),
        deep_dive_texts=deep_dive_texts,
        reference_ledger_text=reference_ledger_text,
    )
    token_estimate = estimate_tokens(prompt)
    if token_estimate > input_token_ceiling:
        raise TopicReconciliationError(
            f"topic_reconciliation_input_too_large:{topic_id}:{token_estimate}>{input_token_ceiling}"
        )
    digest_parts = {
        "source_pack": source_pack_input_sha256(source_pack),
        "paper_summary": sha256_file(paper_summary_path) if paper_summary_path else None,
        "transcript_summary": (
            sha256_file(transcript_summary_path) if transcript_summary_path else None
        ),
        "deep_dives": {
            workspace_relpath(workspace, path): sha256_file(path)
            for path in deep_dive_paths
        },
        "prompt_sha256": sha256_text(prompt),
    }
    input_digest = hashlib.sha256(
        json.dumps(digest_parts, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ReconciliationInput(
        topic_id=topic_id,
        title=title,
        source_pack_path=source_pack_path,
        source_pack=source_pack,
        paper_summary_path=paper_summary_path,
        transcript_summary_path=transcript_summary_path,
        deep_dive_paths=deep_dive_paths,
        input_digest_sha256=input_digest,
        input_token_estimate=token_estimate,
        prompt=prompt,
    )


def _deterministic_reconciliation_text(inputs: ReconciliationInput) -> str:
    return (
        "# Topic Evidence Reconciliation\n\n"
        "## Consensus Thesis\n\n"
        f"Dry-run reconciliation for `{inputs.topic_id}` confirms that the topic "
        "has bounded report-level material suitable for local Qwen reconciliation.\n\n"
        "## Evidence Map\n\n"
        "### Papers\n\n"
        f"- Paper summary present: `{bool(inputs.paper_summary_path)}`.\n\n"
        "### Transcripts And Talks\n\n"
        f"- Transcript summary present: `{bool(inputs.transcript_summary_path)}`.\n\n"
        "### Deep-Dive Research\n\n"
        f"- Deep-dive report count: `{len(inputs.deep_dive_paths)}`.\n\n"
        "## Agreements Across Sources\n\n"
        "- Dry-run mode does not synthesize claims.\n\n"
        "## Tensions And Disagreements\n\n"
        "- Dry-run mode preserves reconciliation as a live-local task.\n\n"
        "## Strongest Slide-Worthy Claims\n\n"
        "- Run live reconciliation to select slide-worthy claims.\n\n"
        "## Evidence To Preserve In Final Narrative\n\n"
        f"- Source pack: `{workspace_relpath(inputs.source_pack_path.parent.parent.parent, inputs.source_pack_path)}`.\n\n"
        "## Caveats And Source Gaps\n\n"
        "- This is a provider-free planning artifact.\n\n"
        "## Recommended Final Narrative Emphasis\n\n"
        "- Rerun with local Qwen 3.6 before GPT-5.6 Luna final narrative synthesis.\n"
    )


async def _synthesize_reconciliation(
    *,
    inputs: ReconciliationInput,
    researcher: AsyncResearcherLike,
) -> str:
    return await researcher.run(inputs.prompt)


def _topic_ids_from_source_packs(workspace: Path) -> list[str]:
    root = workspace / "analysis" / "narrative_source_packs"
    return [path.stem for path in sorted(root.glob("*.json"))]


def _output_paths(workspace: Path, topic_id: str) -> tuple[Path, Path]:
    root = workspace / "analysis" / "topic_reconciliations"
    return root / f"{topic_id}.md", root / f"{topic_id}.json"


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07D-topic-reconciliation.json"


def _enrich_source_pack(
    *,
    workspace: Path,
    inputs: ReconciliationInput,
    markdown_path: Path,
    sidecar_path: Path,
    text: str,
    model: str,
    generated_at: str,
) -> dict[str, Any]:
    source_pack = dict(inputs.source_pack)
    reconciliation = {
        "status": "succeeded",
        "owner_id": OWNER_TOPIC_RECONCILIATION,
        "generated_at": generated_at,
        "model": model,
        "input_digest_sha256": inputs.input_digest_sha256,
        "input_token_estimate": inputs.input_token_estimate,
        "markdown_path": workspace_relpath(workspace, markdown_path),
        "json_path": workspace_relpath(workspace, sidecar_path),
        "markdown_sha256": sha256_text(text),
        "deep_dive_report_count": len(inputs.deep_dive_paths),
        "paper_summary_path": (
            workspace_relpath(workspace, inputs.paper_summary_path)
            if inputs.paper_summary_path
            else None
        ),
        "transcript_summary_path": (
            workspace_relpath(workspace, inputs.transcript_summary_path)
            if inputs.transcript_summary_path
            else None
        ),
        "deep_dive_paths": [
            workspace_relpath(workspace, path) for path in inputs.deep_dive_paths
        ],
        "reconciliation_text": text,
    }
    source_pack["topic_reconciliation"] = reconciliation
    write_json_atomic(inputs.source_pack_path, source_pack)
    return reconciliation


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def _line_unknown_ids(line: str, *, ledger: TopicReferenceLedger) -> set[str]:
    """Canonical ids cited on one line that the ledger cannot bind.

    Mirrors validate_reconciliation_references' own per-token
    canonicalization exactly (same regex, same observed-raw-prefix scoping),
    so this guard can never strip a citation the validator would have
    accepted.
    """

    unknown: set[str] = set()
    for match in SOURCE_ID_TOKEN_RE.finditer(line):
        raw_id = match.group("id")
        raw_prefix = raw_id.split(":", 1)[0] if ":" in raw_id else ""
        if raw_prefix in ("paper_abstract", "speech_transcript") or (
            raw_prefix and raw_prefix in ledger.observed_raw_prefixes
        ):
            reference_id = canonicalize_reference_id(raw_id) or raw_id
        else:
            reference_id = raw_id
        if reference_id not in ledger.entries:
            unknown.add(reference_id)
    return unknown


def strip_unledgered_citations(
    text: str, *, ledger: TopicReferenceLedger
) -> tuple[str, list[dict[str, Any]]]:
    """Drop citations the deterministic ledger cannot bind.

    The reconciliation prompt already instructs the model to cite only
    ledger ids, and it complies for the overwhelming majority of citations.
    The residue is not over-reach but *fabrication*: the deep-dive reports
    quote transcripts without carrying their ledger ids, so the model
    synthesizes a plausible-looking id for the quote it wants to keep --
    inventing separators (``lane__video`` where the corpus uses
    ``lane:video``) and sometimes saying so outright ("inferred from deep
    dive"). A prompt cannot fix that; only the ledger knows which ids exist.

    An evidence reconciliation brief must not carry a claim whose source
    cannot be verified, so a list item resting on an unbindable id is
    dropped whole rather than left as an unattributed quote. Every removal
    is returned for the receipt: model output is never discarded silently.
    """

    result = validate_reconciliation_references(
        text=text, ledger=ledger, require_citations=False
    )
    if not result.unknown_reference_ids:
        return text, []

    kept: list[str] = []
    removed: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        unknown = _line_unknown_ids(line, ledger=ledger)
        if not unknown:
            kept.append(line)
            continue
        if _LIST_ITEM_RE.match(line):
            removed.append(
                {
                    "line": lineno,
                    "reference_ids": sorted(unknown),
                    "action": "dropped_list_item",
                    "text": line.strip()[:240],
                }
            )
            continue
        # Prose (a heading, a paragraph): excising the sentence would change
        # meaning, so drop only the citation token and let the surrounding
        # claim stand on whatever else supports it.
        cleaned = line
        for match in reversed(list(SOURCE_ID_TOKEN_RE.finditer(line))):
            raw_id = match.group("id")
            raw_prefix = raw_id.split(":", 1)[0] if ":" in raw_id else ""
            if raw_prefix in ("paper_abstract", "speech_transcript") or (
                raw_prefix and raw_prefix in ledger.observed_raw_prefixes
            ):
                reference_id = canonicalize_reference_id(raw_id) or raw_id
            else:
                reference_id = raw_id
            if reference_id in unknown:
                cleaned = cleaned[: match.start()] + cleaned[match.end() :]
        cleaned = re.sub(r"``+", "", cleaned)
        removed.append(
            {
                "line": lineno,
                "reference_ids": sorted(unknown),
                "action": "stripped_citation_token",
                "text": line.strip()[:240],
            }
        )
        kept.append(cleaned)
    return "\n".join(kept) + ("\n" if text.endswith("\n") else ""), removed


def _sub_outside_code_spans(pattern: str, line: str) -> str:
    """Delete ``pattern`` matches from ``line`` except inside `backtick` spans."""

    parts = re.split(r"(`[^`]*`)", line)
    return "".join(
        part if part.startswith("`") and part.endswith("`") and len(part) >= 2
        else re.sub(pattern, "", part, flags=re.IGNORECASE)
        for part in parts
    )


def strip_foreign_title_aliases(
    text: str, *, ledger: TopicReferenceLedger
) -> tuple[str, list[dict[str, Any]]]:
    """Remove labels that belong to a different ledger entry than the one cited.

    Distinct from :func:`strip_unledgered_citations`: here the *citation is
    valid* and only its label is wrong -- the model tags a correctly cited
    transcript with a short name belonging to another work (observed live:
    ``speech_transcript:AqneAlYzEAs:segment:00000 (PSIVG)``). Dropping the
    line would discard a good citation, so only the offending label is
    excised and the claim plus its reference stay intact.

    Parenthesised labels are removed whole; a bare alias is removed as a
    phrase. Every edit is returned for the receipt.
    """

    result = validate_reconciliation_references(
        text=text, ledger=ledger, require_citations=False
    )
    if not result.title_mismatch_errors:
        return text, []

    by_line: dict[int, set[str]] = {}
    for issue in result.title_mismatch_errors:
        lineno = issue.get("line")
        if not isinstance(lineno, int):
            continue
        aliases = issue.get("foreign_title_aliases") or {}
        by_line.setdefault(lineno, set()).update(str(a) for a in aliases)

    lines = text.splitlines()
    removed: list[dict[str, Any]] = []
    for lineno, aliases in sorted(by_line.items()):
        if not 1 <= lineno <= len(lines):
            continue
        original = lines[lineno - 1]
        cleaned = original
        for alias in sorted(aliases, key=len, reverse=True):
            if not alias:
                continue
            # Prefer excising the whole parenthetical that carries the label,
            # which is how the model actually writes it.
            def _drop_parenthetical(match: re.Match[str]) -> str:
                return "" if alias in normalize_text(match.group(1)) else match.group(0)

            candidate = re.sub(r"\(([^()]*)\)", _drop_parenthetical, cleaned)
            if candidate != cleaned:
                cleaned = candidate
                continue
            # Otherwise remove the alias phrase itself, tolerating the
            # original casing and internal punctuation -- but never inside a
            # backticked id token: an alias that happens to be a segment of a
            # cited id (live CVPR 2026: "workshop" inside
            # `paper_abstract:thecvf.com_CVPR_2026_Workshop_EDGE_...`) must
            # not turn a valid citation into an unknown one.
            pattern = r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(
                re.escape(word) for word in alias.split()
            ) + r"(?![A-Za-z0-9])"
            cleaned = _sub_outside_code_spans(pattern, cleaned)
        # Tidy the gap the excision left without touching the leading indent:
        # collapsing that would re-level the markdown list the line sits in.
        indent = original[: len(original) - len(original.lstrip())]
        body = re.sub(r"[ \t]{2,}", " ", cleaned[len(indent) :])
        body = re.sub(r"\s+([;,.)])", r"\1", body)
        cleaned = (indent + body).rstrip()
        if cleaned == original:
            continue
        lines[lineno - 1] = cleaned
        removed.append(
            {
                "line": lineno,
                "foreign_title_aliases": sorted(aliases),
                "action": "stripped_foreign_title_alias",
                "text": original.strip()[:240],
            }
        )
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), removed


def _write_incremental_reconciliation_receipt(
    *,
    workspace: Path,
    run_id: str,
    mode: str,
    model: str,
    input_token_ceiling: int,
    max_output_tokens: int,
    topic_ids: list[str],
    outputs: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    enrich_source_packs: bool,
    researcher: AsyncResearcherLike | None,
    provider_receipts_before: set[str],
    generated_at: str,
    started: float,
    receipt_path: Path,
) -> None:
    """Write a partial-progress 07D receipt after each topic iteration.

    Mirrors the final receipt shape built at the end of
    :func:`run_topic_reconciliation`, but reports ``in_progress`` while topics
    remain so a mid-run kill leaves accurate evidence instead of a stale
    all-blocked snapshot from a prior attempt.
    """

    remaining = len(topic_ids) - (len(outputs) + len(blockers))
    provider_receipts_after = set(getattr(researcher, "provider_receipt_paths", []) or [])
    usage = researcher.usage() if researcher is not None else {}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_RECONCILIATION,
        "owner_id": OWNER_TOPIC_RECONCILIATION,
        "status": "in_progress" if remaining > 0 else ("blocked" if blockers else "succeeded"),
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "input_token_ceiling": input_token_ceiling,
        "max_output_tokens": max_output_tokens,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "output_paths": [item["markdown_path"] for item in outputs],
        "sidecar_paths": [
            workspace_relpath(workspace, _output_paths(workspace, str(item["topic_id"]))[1])
            for item in outputs
        ],
        "source_packs_enriched": enrich_source_packs and not blockers,
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": False,
        "local_provider_receipt_paths": sorted(provider_receipts_after - provider_receipts_before),
        "usage": usage,
        "blockers": list(blockers),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    write_json_atomic(receipt_path, receipt)


def run_topic_reconciliation(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    model: str = DEFAULT_QWEN36_MODEL,
    topics: list[str] | None = None,
    input_token_ceiling: int = DEFAULT_INPUT_TOKEN_CEILING,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    resume: bool = False,
    enrich_source_packs: bool = True,
    researcher: AsyncResearcherLike | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "topic-reconciliation")
    generated_at = utc_now()
    topic_ids = topics or _topic_ids_from_source_packs(workspace)
    if not topic_ids:
        raise TopicReconciliationError("narrative_source_packs_missing")
    out_root = workspace / "analysis" / "topic_reconciliations"
    out_root.mkdir(parents=True, exist_ok=True)
    receipt_path = _receipt_path(workspace, run_id)
    if mode == "live" and researcher is None:
        researcher = LocalLLMResearcher(smart_model=model, fast_model=model, max_tokens=max_output_tokens)

    outputs: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    provider_receipts_before = set(getattr(researcher, "provider_receipt_paths", []) or [])
    for topic_id in topic_ids:
        try:
            inputs = collect_reconciliation_input(
                workspace,
                topic_id=topic_id,
                input_token_ceiling=input_token_ceiling,
            )
            markdown_path, sidecar_path = _output_paths(workspace, topic_id)
            existing_sidecar = None
            if resume and sidecar_path.exists():
                existing_sidecar = _load_json_object(sidecar_path)
            if (
                existing_sidecar
                and existing_sidecar.get("input_digest_sha256") == inputs.input_digest_sha256
                and existing_sidecar.get("status") == "succeeded"
                and markdown_path.exists()
                # A dry run's deterministic placeholder is never a live result.
                and (mode != "live" or existing_sidecar.get("mode") == "live")
            ):
                text = markdown_path.read_text(encoding="utf-8", errors="replace")
                reused = True
            else:
                if mode == "live":
                    if researcher is None:
                        raise TopicReconciliationError("local_researcher_missing")
                    text = asyncio.run(
                        _synthesize_reconciliation(inputs=inputs, researcher=researcher)
                    )
                else:
                    text = _deterministic_reconciliation_text(inputs)
                reused = False
                markdown_path.write_text(text, encoding="utf-8")
            sidecar = {
                "schema_version": SCHEMA_VERSION,
                "owner_id": OWNER_TOPIC_RECONCILIATION,
                "status": "succeeded",
                "generated_at": generated_at,
                "run_id": run_id,
                "mode": mode,
                "topic_id": topic_id,
                "title": inputs.title,
                "model": model,
                "input_digest_sha256": inputs.input_digest_sha256,
                "input_token_estimate": inputs.input_token_estimate,
                "output_token_estimate": estimate_tokens(text),
                "markdown_path": workspace_relpath(workspace, markdown_path),
                "markdown_sha256": sha256_text(text),
                "source_pack_path": workspace_relpath(workspace, inputs.source_pack_path),
                "paper_summary_path": (
                    workspace_relpath(workspace, inputs.paper_summary_path)
                    if inputs.paper_summary_path
                    else None
                ),
                "transcript_summary_path": (
                    workspace_relpath(workspace, inputs.transcript_summary_path)
                    if inputs.transcript_summary_path
                    else None
                ),
                "deep_dive_paths": [
                    workspace_relpath(workspace, path) for path in inputs.deep_dive_paths
                ],
                "deep_dive_report_count": len(inputs.deep_dive_paths),
                "reused": reused,
            }
            write_json_atomic(sidecar_path, sidecar)
            if enrich_source_packs:
                reconciliation = _enrich_source_pack(
                    workspace=workspace,
                    inputs=inputs,
                    markdown_path=markdown_path,
                    sidecar_path=sidecar_path,
                    text=text,
                    model=model,
                    generated_at=generated_at,
                )
                sidecar["source_pack_enriched"] = True
                sidecar["source_pack_topic_reconciliation_sha256"] = reconciliation[
                    "markdown_sha256"
                ]
                write_json_atomic(sidecar_path, sidecar)
            outputs.append(sidecar)
        except Exception as exc:  # noqa: BLE001 - receipt boundary.
            blockers.append(
                {
                    "code": str(exc).split(":", 1)[0] or "topic_reconciliation_failed",
                    "topic_id": topic_id,
                    "reason": str(exc),
                    "retryable": True,
                }
            )
        finally:
            # Fix option 5: write an incremental 07D receipt after every topic so
            # a mid-run kill leaves accurate partial-progress evidence instead of
            # a stale prior-attempt snapshot. The final post-loop write below
            # remains authoritative and unchanged in shape for a clean run.
            _write_incremental_reconciliation_receipt(
                workspace=workspace,
                run_id=run_id,
                mode=mode,
                model=model,
                input_token_ceiling=input_token_ceiling,
                max_output_tokens=max_output_tokens,
                topic_ids=topic_ids,
                outputs=outputs,
                blockers=blockers,
                enrich_source_packs=enrich_source_packs,
                researcher=researcher,
                provider_receipts_before=provider_receipts_before,
                generated_at=generated_at,
                started=started,
                receipt_path=receipt_path,
            )

    provider_receipts_after = set(getattr(researcher, "provider_receipt_paths", []) or [])
    usage = researcher.usage() if researcher is not None else {}
    status = "blocked" if blockers else "succeeded"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_RECONCILIATION,
        "owner_id": OWNER_TOPIC_RECONCILIATION,
        "status": status,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "input_token_ceiling": input_token_ceiling,
        "max_output_tokens": max_output_tokens,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "output_paths": [item["markdown_path"] for item in outputs],
        "sidecar_paths": [
            workspace_relpath(workspace, _output_paths(workspace, str(item["topic_id"]))[1])
            for item in outputs
        ],
        "source_packs_enriched": enrich_source_packs and not blockers,
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": False,
        "local_provider_receipt_paths": sorted(provider_receipts_after - provider_receipts_before),
        "usage": usage,
        "blockers": blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    stamp_substep_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        substep_id="topic-reconciliation",
        complete=set(topic_ids) >= set(_topic_ids_from_source_packs(workspace)),
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile per-topic analysis evidence before final narrative synthesis."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--input-token-ceiling", type=int, default=DEFAULT_INPUT_TOKEN_CEILING)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-enrich-source-packs", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_topic_reconciliation(
        workspace=args.workspace,
        run_id=args.run_id,
        mode=args.mode,
        model=args.model,
        topics=args.topics,
        input_token_ceiling=args.input_token_ceiling,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
        enrich_source_packs=not args.no_enrich_source_packs,
    )
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "topic_count": receipt.get("topic_count"),
                    "completed_topic_count": receipt.get("completed_topic_count"),
                    "receipt_path": receipt.get("receipt_path"),
                    "output_paths": receipt.get("output_paths"),
                    "blockers": receipt.get("blockers"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if receipt.get("status") == "succeeded" else 1


__all__ = [
    "DEFAULT_INPUT_TOKEN_CEILING",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_QWEN36_MODEL",
    "OWNER_TOPIC_RECONCILIATION",
    "TopicReconciliationError",
    "build_reconciliation_prompt",
    "collect_reconciliation_input",
    "estimate_tokens",
    "run_main",
    "run_topic_reconciliation",
]
