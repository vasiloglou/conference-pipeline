"""Keyword prefilter packs for deep-dive GPT-Researcher runs.

The base vector stores stay topic-agnostic; topic ids are derived late and can
change. This helper builds a deterministic evidence pack from stable chunk
metadata plus an explicitly recorded keyword list so a deep-dive run can compare
the normal global GPT-Researcher lane with a narrower heuristic lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_STORES: tuple[str, ...] = ("papers", "artifacts")
_WORD_RE = re.compile(r"\s+")


class HeuristicPrefilterError(ValueError):
    """Raised when a heuristic evidence pack cannot be constructed."""


@dataclass(frozen=True)
class PrefilteredEvidence:
    """One selected chunk for the heuristic deep-dive lane."""

    chunk_id: str
    store: str
    source_artifact_id: str
    source_family_id: str
    source_path: str
    source_url: str
    content_kind: str
    conference_scope: str
    text: str
    score: float
    matched_terms: tuple[str, ...]
    source_context: dict[str, Any]
    locator: dict[str, Any]


def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword/phrase while preserving punctuation-bearing terms."""

    return _WORD_RE.sub(" ", str(keyword).strip().casefold())


def normalized_keywords(keywords: Iterable[str]) -> tuple[str, ...]:
    """Return sorted unique normalized keywords, rejecting empty input."""

    normalized = sorted({normalize_keyword(keyword) for keyword in keywords if normalize_keyword(keyword)})
    if not normalized:
        raise HeuristicPrefilterError("empty_keyword_list")
    return tuple(normalized)


def keyword_digest(keywords: Iterable[str]) -> str:
    """Stable digest of the normalized prefilter keyword contract."""

    payload = json.dumps(list(normalized_keywords(keywords)), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_prefiltered_evidence(
    *,
    workspace: str | Path,
    run_id: str,
    keywords: Iterable[str],
    stores: Sequence[str] = DEFAULT_STORES,
    limit: int = 32,
    max_per_source_artifact: int = 3,
    min_score: float = 1.0,
) -> list[PrefilteredEvidence]:
    """Select ranked chunk evidence from sidecar chunks using explicit keywords."""

    terms = normalized_keywords(keywords)
    if limit < 1:
        raise HeuristicPrefilterError("limit_must_be_positive")
    if max_per_source_artifact < 1:
        raise HeuristicPrefilterError("max_per_source_artifact_must_be_positive")

    candidates: list[PrefilteredEvidence] = []
    for store, record in _iter_chunk_records(Path(workspace), run_id, stores):
        evidence = _score_record(record, store=store, keywords=terms)
        if evidence is not None and evidence.score >= min_score:
            candidates.append(evidence)

    candidates.sort(key=lambda item: (-item.score, item.store, item.source_artifact_id, item.chunk_id))

    selected: list[PrefilteredEvidence] = []
    per_source: Counter[str] = Counter()
    for evidence in candidates:
        source_key = evidence.source_artifact_id or evidence.chunk_id
        if per_source[source_key] >= max_per_source_artifact:
            continue
        selected.append(evidence)
        per_source[source_key] += 1
        if len(selected) >= limit:
            break
    return selected


def build_prefiltered_evidence_pack(
    *,
    workspace: str | Path,
    run_id: str,
    deep_dive_id: str,
    question: str,
    keywords: Iterable[str],
    keyword_seed_source: str,
    stores: Sequence[str] = DEFAULT_STORES,
    limit: int = 32,
    max_per_source_artifact: int = 3,
    min_score: float = 1.0,
) -> dict[str, Any]:
    """Build the JSON-serializable evidence pack consumed by the heuristic lane."""

    terms = normalized_keywords(keywords)
    evidence = select_prefiltered_evidence(
        workspace=workspace,
        run_id=run_id,
        keywords=terms,
        stores=stores,
        limit=limit,
        max_per_source_artifact=max_per_source_artifact,
        min_score=min_score,
    )
    selected_ids = [item.chunk_id for item in evidence]
    return {
        "schema_version": 1,
        "workflow": "workflow.research.heuristic_prefiltered_gpt_researcher",
        "retrieval_scope": "heuristic_keyword_prefilter",
        "deep_dive_id": deep_dive_id,
        "question": question,
        "run_id": run_id,
        "keyword_seed_source": keyword_seed_source,
        "keywords": list(terms),
        "prefilter_terms_sha256": keyword_digest(terms),
        "stores_searched": list(stores),
        "source_family_scope": sorted({item.source_family_id for item in evidence if item.source_family_id}),
        "candidate_count": len(evidence),
        "selected_evidence_ids": selected_ids,
        "invalid_citation_ids": [],
        "evidence": [asdict(item) for item in evidence],
    }


def write_prefiltered_evidence_pack(pack: dict[str, Any], path: str | Path) -> Path:
    """Write an evidence pack and return the resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pack, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _iter_chunk_records(
    workspace: Path,
    run_id: str,
    stores: Sequence[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    vector_root = workspace / "storage" / "vector-store" / run_id / "stores"
    for store in stores:
        chunks_path = vector_root / store / "chunks.jsonl"
        if not chunks_path.exists():
            raise HeuristicPrefilterError(f"missing_chunks_jsonl:{store}")
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HeuristicPrefilterError(
                        f"invalid_chunks_jsonl:{store}:{line_number}"
                    ) from exc
                if isinstance(record, dict):
                    yield store, record


def _score_record(
    record: dict[str, Any],
    *,
    store: str,
    keywords: tuple[str, ...],
) -> PrefilteredEvidence | None:
    text = str(record.get("text") or "")
    metadata_text = _metadata_search_text(record)
    haystack = normalize_keyword(f"{metadata_text}\n{text}")
    matched: list[str] = []
    score = 0.0
    for term in keywords:
        count = haystack.count(term)
        if count <= 0:
            continue
        matched.append(term)
        score += count * max(1.0, len(term.split()) * 1.5)
    if not matched:
        return None

    source_context = record.get("source_context") if isinstance(record.get("source_context"), dict) else {}
    locator = record.get("locator") if isinstance(record.get("locator"), dict) else {}
    source_url = str(source_context.get("source_url") or source_context.get("final_url") or "")
    source_artifact_id = str(record.get("artifact_id") or locator.get("source_artifact_id") or record.get("chunk_id") or "")
    return PrefilteredEvidence(
        chunk_id=str(record.get("chunk_id") or ""),
        store=store,
        source_artifact_id=source_artifact_id,
        source_family_id=str(record.get("source_family_id") or source_context.get("source_family_id") or ""),
        source_path=str(record.get("source_path") or ""),
        source_url=source_url,
        content_kind=str(record.get("content_kind") or source_context.get("content_kind") or ""),
        conference_scope=str(record.get("conference_scope") or source_context.get("conference_scope") or ""),
        text=text,
        score=round(score, 4),
        matched_terms=tuple(matched),
        source_context=dict(source_context),
        locator=dict(locator),
    )


def _metadata_search_text(record: dict[str, Any]) -> str:
    source_context = record.get("source_context") if isinstance(record.get("source_context"), dict) else {}
    pieces: list[str] = []
    for key in (
        "title",
        "paper_title",
        "talk_title",
        "talk_speaker",
        "workshop_title",
        "session_title",
        "paper_id",
        "slideslive_id",
        "source_url",
        "final_url",
    ):
        value = source_context.get(key) or record.get(key)
        if value:
            pieces.append(str(value))
    return "\n".join(pieces)


def _read_keywords(values: Sequence[str], keywords_file: str | None) -> tuple[str, ...]:
    keywords = list(values)
    if keywords_file:
        payload = json.loads(Path(keywords_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            file_keywords = payload.get("keywords") or payload.get("terms") or []
        else:
            file_keywords = payload
        if not isinstance(file_keywords, list):
            raise HeuristicPrefilterError("keywords_file_must_contain_list")
        keywords.extend(str(item) for item in file_keywords)
    return normalized_keywords(keywords)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deep-dive-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--keywords-file")
    parser.add_argument("--keyword-seed-source", required=True)
    parser.add_argument("--store", action="append", dest="store", default=[])
    parser.add_argument(
        "--stores",
        dest="stores_csv",
        help="Comma-separated store list; alias for repeated --store.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-per-source-artifact", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--output-evidence-json", required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    keywords = _read_keywords(args.keyword, args.keywords_file)
    stores_input = list(args.store)
    if args.stores_csv:
        stores_input.extend(part.strip() for part in str(args.stores_csv).split(","))
    stores = tuple(store for store in stores_input if store) if stores_input else DEFAULT_STORES
    pack = build_prefiltered_evidence_pack(
        workspace=args.workspace,
        run_id=args.run_id,
        deep_dive_id=args.deep_dive_id,
        question=args.question,
        keywords=keywords,
        keyword_seed_source=args.keyword_seed_source,
        stores=stores,
        limit=args.limit,
        max_per_source_artifact=args.max_per_source_artifact,
        min_score=args.min_score,
    )
    output = write_prefiltered_evidence_pack(pack, args.output_evidence_json)
    if args.summary:
        print(
            json.dumps(
                {
                    "status": "succeeded",
                    "output_evidence_json": str(output),
                    "candidate_count": pack["candidate_count"],
                    "prefilter_terms_sha256": pack["prefilter_terms_sha256"],
                    "stores_searched": pack["stores_searched"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
