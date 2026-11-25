"""Shared, order-independent narrative source-pack chunk selection.

**Finding D** (Phase 10.1.1, PROD-15): the per-topic narrative source-pack
scan over a conference's ``papers`` vector-store ``chunks.jsonl`` sidecar
historically depended on the *physical file order* of
``main_conference``-scoped versus ``workshop``-scoped chunks, not on
topical relevance. Every conference runner (CVPR2026, ICLR2026, ICML2026)
carried its own triplicated copy of ``build_analysis_source_packs``, each
scanning the ``papers`` store as a single per-topic bucket capped at
``ANALYSIS_SOURCE_PACK_MAX_CHUNKS`` and stopping once that cap filled. On
the live CVPR2026 papers store, ``workshop``-scoped chunks occupy indices
0-7180 and ``main_conference``-scoped chunks start at index 7181 (measured
in
``.planning/workstreams/cvpr-2026/phases/10.1.1-source-pack-family-stratification/10.1.1-FAMILY-VOLUMES.md``)
-- so a cap-then-stop scan could fill an entire topic's quota from workshop
matches alone, thousands of chunk-positions before ever reaching a
main_conference match, regardless of how many main-conference sources
actually existed.

This module extends the existing per-store independence pattern (see the
comment above ``per_store_selected`` in the CVPR2026 runner) one level
deeper for the ``papers`` store: instead of one shared bucket per topic, it
maintains one sub-bucket *per conference-scope family* per topic, each with
its own per-source de-duplication set, so no family can starve another out
of physical file position alone.

Composition is a **floor with backfill, never a hard reservation**: each
topic's papers-store quota starts with a minimum guarantee of
``main_conference_floor`` main-conference chunks, then fills with workshop
chunks up to ``paper_target``, then backfills any still-unfilled
papers-store slots from whichever family has unused chunks left, then fills
remaining slots from every other store, then backfills any still-leftover
slots from the papers-store families again. A hard reservation would leave
slots permanently empty for a topic with no workshop matches (or vice
versa) and would break the existing caption-starvation contract test
(``test_analysis_source_packs_are_not_paper_starved_by_captions``), whose
fixture carries no ``conference_scope`` at all and therefore resolves
entirely to ``main_conference`` via the fail-open rule below.

Runner call sites: plans 03 (ICLR2026), 04 (ICML2026/CVPR2026 wiring) of
Phase 10.1.1-source-pack-family-stratification replace each runner's own
``build_analysis_source_packs`` selection body with a call to
``select_topic_source_chunks``, keeping each runner's own serialization
(source packs, query plans) unchanged -- this module performs no file
writes.

**Phase 10.1.1.1 (PROD-15): content-derived main-conference ranking.**
Order-independence alone was not sufficient: on the live CVPR2026 workspace,
the ``main_conference`` family's admission test was a boolean ``any(...)``
keyword match, and generic computer-vision vocabulary matched nearly every
topic -- so among the ~4,068 trivially-matching main-conference papers per
topic, the first five distinct papers in physical file order always won,
identically, across all 12 topic decks (measured ranks 1, 17, 34; union 5 /
intersection 5 across every deck, including plainly unrelated ones such as
``12_imaging_society_arts``). File position had merely moved from
deciding *which family* filled a topic's quota (Finding D, above) to
deciding *which paper within the family* did.

The fix replaces that boolean admission test, for the ``main_conference``
family only, with a content-derived score: priority membership from the
already-paid per-topic ``matched_papers.json`` matching signal (joined by
the bare sha256 hash, present on the raw chunk as ``source_context.paper_id``
and on the compact chunk as ``extraction_record_id``), then that signal's
own ``rank`` field, then the *count* of distinct topic keywords matched
(not a boolean "any hit"). Eight of the twelve CVPR2026 topics have fewer
joinable priority candidates than the main-conference floor, so the
distinct-keyword-hit-count backfill is not an edge case -- it is required
for most decks. File position survives only as the score tuple's final
element, a determinism tiebreak, never the primary decider.

This is a correctness and attribution fix: it changes *which* main-
conference papers a topic selects, never *how many* (the budget constants
``DEFAULT_MAX_CHUNKS_PER_TOPIC``/``DEFAULT_PAPER_TARGET``/
``DEFAULT_MAIN_CONFERENCE_FLOOR`` are unchanged). It does not, by itself,
change ledger eligibility or the named-people ceiling: ``contribution_note``
eligibility is derived from already-paid narrative markdown that predates
this fix and is structurally decoupled from which papers are selected (see
``.planning/workstreams/cvpr-2026/phases/10.1.1.1-topical-main-conference-source-selection/10.1.1.1-RESEARCH.md``,
Q3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from conference_pipeline.indexing.corpus_vector_store import iter_chunk_jsonl

# Per-topic chunk cap. Unchanged from the pre-fix per-store scan: raising
# this masks order-dependence behind a wider net rather than removing it,
# and would still starve whichever family sorts last in the file.
DEFAULT_MAX_CHUNKS_PER_TOPIC = 8

# Papers-majority target per topic when the papers store has matches in
# both families -- reused verbatim from
# ``scripts/run_cvpr2026_production.py``'s ``ANALYSIS_SOURCE_PACK_PAPER_TARGET``.
DEFAULT_PAPER_TARGET = 5

# Minimum main_conference representation guaranteed per topic, justified in
# 10.1.1-FAMILY-VOLUMES.md against the live-measured minimum distinct
# main_conference source count (4068, vastly above this floor). Reuses the
# existing ANALYSIS_SOURCE_PACK_PAPER_TARGET value rather than inventing a
# new budget number.
DEFAULT_MAIN_CONFERENCE_FLOOR = 5

# The two conference_scope families derive_conference_scope() stamps at
# index time (src/conference_pipeline/indexing/corpus_vector_store.py).
CONFERENCE_SCOPE_FAMILIES: tuple[str, ...] = ("main_conference", "workshop")

# The only store name that receives per-family sub-bucketing. Every other
# store keeps the pre-existing single-bucket-per-topic behavior.
STRATIFIED_STORE_NAME = "papers"

# Internal sentinel family key for stores that are not stratified -- kept
# out of CONFERENCE_SCOPE_FAMILIES so it never leaks into family_counts.
_UNSTRATIFIED_FAMILY = "_unstratified"

# Bounded candidate-pool size for the main_conference family's
# content-derived ranking (Phase 10.1.1.1, PROD-15). Exceeds both measured
# ceilings so every joinable priority candidate a topic can have, plus ample
# keyword-backfill headroom, fits without materializing the store:
# DEFAULT_MAX_CHUNKS_PER_TOPIC (8, the composed-pack cap) and the
# matched_papers.json compaction limit ``DEFAULT_MATCHED_PAPER_TOP_K`` (20,
# in ``conference_pipeline.analysis.narrative``) both fit comfortably inside
# 32, with headroom left for keyword-only backfill candidates.
DEFAULT_RANKED_POOL_SIZE = 32

CompactChunkFn = Callable[[Mapping[str, Any]], dict[str, Any]]


def chunk_conference_scope(chunk: Mapping[str, Any]) -> str:
    """Read the conference_scope family a chunk belongs to.

    Reads the top-level ``conference_scope`` key, falls back to
    ``chunk["source_context"]["conference_scope"]``, and fails open to
    ``"main_conference"`` for any missing or unrecognized value. This does
    NOT re-derive the classification from paths or source families --
    ``derive_conference_scope`` already computed it once at index time, and
    re-deriving here risks disagreeing with the authoritative value that
    paper-matching and the heuristic prefilter already consume.
    """

    value = chunk.get("conference_scope")
    if not value:
        source_context = chunk.get("source_context")
        if isinstance(source_context, Mapping):
            value = source_context.get("conference_scope")
    text = str(value or "")
    if text in CONFERENCE_SCOPE_FAMILIES:
        return text
    return "main_conference"


# Tokens that match nearly every chunk and so cannot rank one: English
# function words plus the vocabulary every conference chunk shares. The
# conference's own name, slug and year are added per workspace by
# ``conference_keyword_terms`` -- never hardcoded here.
GENERIC_KEYWORD_TERMS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of", "on",
        "or", "the", "to", "via", "with", "we", "our", "this", "that", "these", "those", "new", "using",
        "based", "towards", "toward", "approach", "method", "methods", "model", "models", "learning",
        "conference", "workshop", "workshops", "paper", "papers", "talk", "talks", "session", "sessions",
        "keynote", "tutorial", "poster", "oral", "et", "al",
    }
)
MIN_KEYWORD_LENGTH = 3


def conference_keyword_terms(workspace: Path) -> frozenset[str]:
    """The workspace conference's own name, slug and year as lowercase tokens."""

    from conference_pipeline.conference_identity import ConferenceIdentityError, load_conference_identity

    try:
        identity = load_conference_identity(workspace)
    except ConferenceIdentityError:
        return frozenset({workspace.name.lower()})
    terms = {identity.slug.lower(), identity.name.lower()}
    for text in (identity.name, identity.full_name, identity.slug):
        terms.update(token for token in re.split(r"[^a-z0-9]+", text.lower()) if token)
    if identity.year:
        terms.add(str(identity.year))
    return frozenset(term for term in terms if term)


def informative_keywords(
    keywords: Sequence[object], *, ignored_terms: Iterable[str] = ()
) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` -- lowercase keywords that can discriminate a chunk.

    A keyword is dropped when it is a generic term, an ignored conference
    term, or shorter than ``MIN_KEYWORD_LENGTH``. A multi-word keyword is
    kept whenever any of its words is informative.
    """

    ignored = {str(term).lower() for term in ignored_terms} | GENERIC_KEYWORD_TERMS
    kept: list[str] = []
    dropped: list[str] = []
    for raw in keywords:
        keyword = str(raw or "").strip().lower()
        if not keyword:
            continue
        words = [word for word in re.split(r"[^a-z0-9+#.-]+", keyword) if word]
        informative = [word for word in words if len(word) >= MIN_KEYWORD_LENGTH and word not in ignored]
        if keyword in ignored or not informative:
            dropped.append(keyword)
        else:
            kept.append(keyword)
    return kept, dropped


def identical_selection_groups(selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[list[str]]:
    """Topic ids grouped by identical non-empty chunk-id sets (groups of two or more)."""

    by_key: dict[tuple[str, ...], list[str]] = {}
    for topic_id, chunks in selected.items():
        key = tuple(sorted(str(chunk.get("chunk_id") or "") for chunk in chunks if chunk.get("chunk_id")))
        if key:
            by_key.setdefault(key, []).append(str(topic_id))
    return sorted(group for group in by_key.values() if len(group) > 1)


@dataclass(frozen=True)
class SourcePackSelection:
    """Result of a single ``select_topic_source_chunks`` call.

    ``selected``: topic id -> composed, capped, deduplicated chunk list.
    ``fallback``: the first ``max_chunks_per_topic`` non-empty-quote chunks
      seen across all stores, in scan order -- used by callers when a
      topic's own selection is thinner than their minimum-chunk floor.
    ``family_counts``: topic id -> conference_scope family -> count of
      chunks from that family actually included in ``selected`` (papers
      store only; every other store contributes to composition but is not
      family-attributed).
    """

    selected: dict[str, list[dict[str, Any]]]
    fallback: list[dict[str, Any]]
    family_counts: dict[str, dict[str, int]]
    # topic id -> the keywords its spec carried that were too generic to
    # discriminate (stopwords, the conference's own name) and were ignored.
    dropped_keywords: dict[str, list[str]] = field(default_factory=dict)
    # Groups of topic ids whose selected chunk sets are identical. Every
    # ICML 2026 narrative pack held the same eight keynote chunks because the
    # keyword lists shared filler terms; a caller must treat a non-empty
    # list as a selection failure, never as twelve topics that agree.
    identical_selection_groups: list[list[str]] = field(default_factory=list)


def _consider_ranked_pool(
    pool: list[tuple[tuple[int, int, int, int], dict[str, Any]]],
    cap: int,
    score: tuple[int, int, int, int],
    chunk: dict[str, Any],
) -> None:
    """Bounded top-K insertion: keep at most ``cap`` entries in ``pool``,
    keyed by ``score`` (lower sorts first). Once full, evict the current
    worst-scoring entry only when the incoming entry scores strictly
    better. This keeps per-topic per-family memory bounded at ``cap``
    without ever fully materializing or sorting the underlying store."""

    entry = (score, chunk)
    if len(pool) < cap:
        pool.append(entry)
        return
    worst_index = max(range(len(pool)), key=lambda i: pool[i][0])
    if score < pool[worst_index][0]:
        pool[worst_index] = entry


def select_topic_source_chunks(
    *,
    store_chunk_paths: Sequence[tuple[str, Path]],
    topic_specs: Sequence[Mapping[str, Any]],
    compact_chunk: CompactChunkFn,
    max_chunks_per_topic: int = DEFAULT_MAX_CHUNKS_PER_TOPIC,
    paper_target: int = DEFAULT_PAPER_TARGET,
    main_conference_floor: int = DEFAULT_MAIN_CONFERENCE_FLOOR,
    stratified_store: str = STRATIFIED_STORE_NAME,
    priority_source_ranks: Mapping[str, Mapping[str, int]] | None = None,
    ranked_pool_size: int = DEFAULT_RANKED_POOL_SIZE,
    ignored_keyword_terms: Iterable[str] = (),
) -> SourcePackSelection:
    """Select and compose per-topic narrative source-pack chunks.

    Pure function of (store chunk paths, topic specs, a compaction
    callable): performs no file writes, so every runner keeps its own
    on-disk source-pack/query-plan serialization unchanged. Iterates each
    store's ``chunks.jsonl`` sidecar in file order via ``iter_chunk_jsonl``.

    Non-stratified stores keep the pre-existing early-exit contract
    unchanged: a per-store early exit fires once every topic's bucket in
    that store has reached ``max_chunks_per_topic``, so the sidecar is
    never fully scanned when the caps are already satisfied.

    The stratified (``papers``) store's ``main_conference`` family (Phase
    10.1.1.1, PROD-15) does NOT early-exit -- it is scanned in full. A
    bounded top-K eviction pool cannot early-exit without silently
    discarding a better-scoring later candidate (priority membership, rank,
    or keyword-hit count could all still improve after the cap-sized
    prefix), which would simply reintroduce file-order dependence under a
    new name. Memory stays bounded regardless: at most ``ranked_pool_size``
    ``(score, chunk)`` entries per topic are ever held for that family, so
    the module still never sorts or fully materializes the sidecar itself
    -- only the small per-topic candidate pool is sorted, once, after that
    store's scan completes.
    """

    specs_by_id: dict[str, Mapping[str, Any]] = {str(spec["id"]): spec for spec in topic_specs}
    topic_ids = list(specs_by_id)
    fallback: list[dict[str, Any]] = []
    ignored_terms = frozenset(str(term).lower() for term in ignored_keyword_terms)
    keywords_by_topic: dict[str, list[str]] = {}
    dropped_keywords: dict[str, list[str]] = {}
    for topic_id, spec in specs_by_id.items():
        kept, dropped = informative_keywords(spec.get("keywords") or [], ignored_terms=ignored_terms)
        keywords_by_topic[topic_id] = kept
        if dropped:
            dropped_keywords[topic_id] = dropped

    # topic_id -> extraction_record_id ("paper:<hash>"-shaped) -> 1-based
    # rank (lower is better). Normalized once so a missing/None argument, an
    # empty dict, or a per-topic empty dict are all the same silent no-op.
    priority_by_topic: dict[str, dict[str, int]] = {
        str(topic_id): {str(source_id): int(rank) for source_id, rank in (ranks or {}).items()}
        for topic_id, ranks in (priority_source_ranks or {}).items()
    }

    # store_name -> family -> topic_id -> selected compact chunks (in scan order)
    per_store_family_selected: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    # store_name -> family -> topic_id -> seen extraction_record_ids (per-source de-dup)
    per_store_family_seen: dict[str, dict[str, dict[str, set[str]]]] = {}
    per_store_families: dict[str, tuple[str, ...]] = {}
    # store_name -> topic_id -> bounded ranked pool of (score, chunk) for the
    # main_conference family only, within stratified stores. Every other
    # family/store keeps the plain first-found list above; see module
    # docstring (Phase 10.1.1.1, PROD-15) for why main_conference alone needs
    # a content-derived pool instead of a position-derived cap.
    per_store_main_conference_pool: dict[
        str, dict[str, list[tuple[tuple[int, int, int, int], dict[str, Any]]]]
    ] = {}

    for store_name, _ in store_chunk_paths:
        families = (
            CONFERENCE_SCOPE_FAMILIES
            if store_name == stratified_store
            else (_UNSTRATIFIED_FAMILY,)
        )
        per_store_families[store_name] = families
        per_store_family_selected[store_name] = {
            family: {topic_id: [] for topic_id in topic_ids} for family in families
        }
        per_store_family_seen[store_name] = {
            family: {topic_id: set() for topic_id in topic_ids} for family in families
        }
        per_store_main_conference_pool[store_name] = {topic_id: [] for topic_id in topic_ids}

    # Monotonically increasing counter over every chunk seen across every
    # store, in scan order -- used ONLY as the final determinism tiebreak in
    # the main_conference score tuple, never as the primary selection key.
    scan_index = 0

    for store_name, chunks_path in store_chunk_paths:
        is_stratified = store_name == stratified_store
        families = per_store_families[store_name]
        family_selected = per_store_family_selected[store_name]
        family_seen = per_store_family_seen[store_name]
        main_conference_pool = per_store_main_conference_pool[store_name]

        for chunk in iter_chunk_jsonl(chunks_path):
            compact = compact_chunk(chunk)
            quote = compact.get("quote") or ""
            if quote and len(fallback) < max_chunks_per_topic:
                fallback.append(compact)
            text_lower = str(quote).lower()
            source_id = str(compact.get("extraction_record_id") or "")
            chunk_scan_index = scan_index
            scan_index += 1

            # Classify from the RAW chunk before compaction -- the compact
            # representation does not carry conference_scope at all, so
            # classifying after compaction would silently classify
            # everything as main_conference.
            family = chunk_conference_scope(chunk) if is_stratified else _UNSTRATIFIED_FAMILY
            bucket_by_topic = family_selected[family]
            seen_by_topic = family_seen[family]

            for topic_id, spec in specs_by_id.items():
                keywords = keywords_by_topic[topic_id]

                if is_stratified and family == "main_conference":
                    # Content-derived ranking (Phase 10.1.1.1, PROD-15):
                    # priority membership from the already-paid per-topic
                    # matching signal, then that signal's own rank, then
                    # distinct-keyword-hit count -- file position (scan
                    # index) is the last-resort determinism tiebreak only.
                    seen_ids = seen_by_topic[topic_id]
                    if source_id and source_id in seen_ids:
                        continue
                    keyword_hits = sum(
                        1 for keyword in keywords if keyword and keyword in text_lower
                    )
                    topic_priority_ranks = priority_by_topic.get(topic_id, {})
                    is_priority = bool(source_id) and source_id in topic_priority_ranks
                    if not (keyword_hits > 0 or is_priority):
                        continue
                    priority_rank = topic_priority_ranks.get(source_id, 0) if is_priority else 0
                    score = (
                        0 if is_priority else 1,
                        priority_rank,
                        -keyword_hits,
                        chunk_scan_index,
                    )
                    _consider_ranked_pool(
                        main_conference_pool[topic_id], ranked_pool_size, score, compact
                    )
                    if source_id:
                        seen_ids.add(source_id)
                    continue

                # workshop family and every _UNSTRATIFIED_FAMILY store keep
                # today's exact behavior: boolean keyword match, first-found
                # append, capped at max_chunks_per_topic.
                bucket = bucket_by_topic[topic_id]
                if len(bucket) >= max_chunks_per_topic:
                    continue
                if source_id and source_id in seen_by_topic[topic_id]:
                    continue
                if any(keyword and keyword in text_lower for keyword in keywords):
                    bucket.append(compact)
                    if source_id:
                        seen_by_topic[topic_id].add(source_id)

            # The stratified store must be scanned in full: a bounded
            # top-K eviction pool cannot early-exit without silently
            # discarding a better-scoring later candidate, which would be
            # file-order dependence in new clothing. Non-stratified stores
            # keep today's exact early-exit behavior.
            if not is_stratified and all(
                len(family_selected[family][topic_id]) >= max_chunks_per_topic
                for family in families
                for topic_id in topic_ids
            ):
                break

        if is_stratified:
            main_bucket_by_topic = family_selected["main_conference"]
            for topic_id in topic_ids:
                ranked = sorted(main_conference_pool[topic_id], key=lambda entry: entry[0])
                main_bucket_by_topic[topic_id] = [ranked_chunk for _, ranked_chunk in ranked]

    stratified_selected = per_store_family_selected.get(stratified_store, {})
    main_hits_by_topic = stratified_selected.get("main_conference", {})
    workshop_hits_by_topic = stratified_selected.get("workshop", {})
    other_store_order = [name for name, _ in store_chunk_paths if name != stratified_store]

    selected: dict[str, list[dict[str, Any]]] = {}
    family_counts: dict[str, dict[str, int]] = {}

    for topic_id in topic_ids:
        composed: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        counts = {family: 0 for family in CONFERENCE_SCOPE_FAMILIES}

        def take(bucket: list[dict[str, Any]], cap: int, family: str | None = None) -> None:
            effective_cap = min(cap, max_chunks_per_topic)
            for chunk in bucket:
                if len(composed) >= effective_cap:
                    break
                chunk_id = str(chunk.get("chunk_id") or "")
                if chunk_id and chunk_id in seen_ids:
                    continue
                if chunk_id:
                    seen_ids.add(chunk_id)
                composed.append(chunk)
                if family is not None:
                    counts[family] += 1

        main_hits = main_hits_by_topic.get(topic_id, [])
        workshop_hits = workshop_hits_by_topic.get(topic_id, [])

        # Step 1: guarantee the main-conference floor.
        take(main_hits, main_conference_floor, family="main_conference")
        # Step 2: fill with workshop chunks up to the papers-store target.
        take(workshop_hits, paper_target, family="workshop")
        # Step 3: floor-with-backfill -- if the papers-store target is still
        # unmet (e.g. no workshop matches), backfill from whichever
        # papers-store family still has unused chunks. Never a hard
        # reservation: a topic with no workshop matches still reaches the
        # full papers_target from main_conference alone.
        take(main_hits, paper_target, family="main_conference")
        take(workshop_hits, paper_target, family="workshop")
        # Step 4: fill remaining slots from every other store, in the
        # caller-supplied store order (mirrors the pre-fix papers-then-
        # artifacts composition order).
        for other_store_name in other_store_order:
            other_bucket = (
                per_store_family_selected[other_store_name]
                .get(_UNSTRATIFIED_FAMILY, {})
                .get(topic_id, [])
            )
            take(other_bucket, max_chunks_per_topic)
        # Step 5: backfill any still-leftover slots from the papers-store
        # families again (e.g. the other stores fell short), so a slot is
        # never left empty because of an unfillable reservation.
        take(main_hits, max_chunks_per_topic, family="main_conference")
        take(workshop_hits, max_chunks_per_topic, family="workshop")

        selected[topic_id] = composed
        family_counts[topic_id] = counts

    return SourcePackSelection(
        selected=selected,
        fallback=fallback,
        family_counts=family_counts,
        dropped_keywords=dropped_keywords,
        identical_selection_groups=identical_selection_groups(selected),
    )
