"""Conference-general virtual program item-page classification and coverage gates.

Default-include contract: every ``/virtual/<year>/...`` event item page is an
acquisition target unless a structural (navigation/index) segment explicitly
excludes it. Acquisition code must never allowlist event families — the ICML
2026 production run silently dropped all 10 tutorials and every invited talk
because its classifier only recognized ``/poster/``, ``/paper/``, and
``/oral/`` item paths.

Media-family coverage is fail-closed: each media-bearing program family that
the workspace declares (config ``urls`` keys / source-intake families) must
yield at least one discovered item page, or the acquisition stage must emit a
blocker naming the family. A zero-item family is evidence of a discovery gap
(JS-rendered index, bare-ID links, a new URL shape) — never permission to
skip. Excluding a family requires an explicit, receipt-recorded operator
decision.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple
from urllib.parse import urlsplit

# Structural virtual-site path segments that are navigation or index surfaces,
# not event item pages. This is deliberately a DENYLIST: unknown segments are
# treated as item pages so new program families are included by default.
NON_ITEM_SEGMENTS = frozenset(
    {
        "events",
        "eventlistwithbios",
        "calendar",
        "day",
        "loc",
        "api",
        "static",
        "search",
        "papers",
        "awards_detail",
        "affinity_events",
    }
)

# Default mapping from declared program-family ids (workspace config ``urls``
# keys / source-intake family ids) to the item-page path slugs that satisfy
# their coverage. ``invited_talks`` has no index URL on most virtual sites —
# its items are discovered from the calendar — so callers should include it
# whenever the conference program advertises keynotes.
DEFAULT_MEDIA_BEARING_FAMILY_SLUGS: Mapping[str, tuple[str, ...]] = {
    "tutorials": ("tutorial",),
    "orals": ("oral",),
    "workshops": ("workshop",),
    "expo": ("expo", "expo-talk", "expo-demonstration", "expo-workshop"),
    "invited_talks": ("invited-talk", "keynote", "invited-speaker"),
}


class VirtualItemPage(NamedTuple):
    """A classified per-event page on a conference virtual site."""

    family: str | None  # path slug such as "tutorial"; None for bare-ID links
    item_id: str


def _virtual_path_re(year: int) -> re.Pattern[str]:
    return re.compile(
        rf"^/virtual/{year}/"
        rf"(?:(?P<family>[a-z0-9_-]+)/(?P<typed_id>[a-zA-Z0-9_-]+)"
        rf"|(?P<bare_id>\d+))/?$"
    )


def classify_virtual_item_page(url: str, *, year: int) -> VirtualItemPage | None:
    """Classify ``url`` as a virtual-site event item page, or return None.

    Recognizes both typed item paths (``/virtual/<year>/<family>/<id>``, any
    family slug not in :data:`NON_ITEM_SEGMENTS`) and bare-ID item paths
    (``/virtual/<year>/<digits>``) such as the ICML 2026 tutorials index links.
    """

    try:
        path = urlsplit(url).path
    except ValueError:
        return None
    match = _virtual_path_re(year).match(path.lower())
    if match is None:
        return None
    bare_id = match.group("bare_id")
    if bare_id is not None:
        return VirtualItemPage(family=None, item_id=bare_id)
    family = match.group("family")
    if family in NON_ITEM_SEGMENTS:
        return None
    return VirtualItemPage(family=family, item_id=match.group("typed_id"))


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def classify_item_page_title_family(
    html_text: str,
    *,
    declared_families: Mapping[str, tuple[str, ...]] = DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
) -> str | None:
    """Resolve a cached item page's family from its ``<title>`` text.

    Fallback for bare-ID item pages an index links without a typed path
    segment (ICML 2026 linked its 10 tutorials and 23 expo talks/demos/
    workshops as bare ``/virtual/2026/<id>`` URLs from the calendar page, not
    from the tutorials/expo index pages, so ``source_family_id`` attribution
    fails). The longest matching slug phrase wins so a compound title like
    "ICML Expo Workshop ..." resolves to ``expo`` rather than the plain
    ``workshops`` family.
    """

    match = _TITLE_RE.search(html_text)
    if match is None:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    best_family: str | None = None
    best_length = 0
    for family_id, slugs in declared_families.items():
        for slug in slugs:
            phrase = slug.replace("-", " ")
            if phrase in title and len(phrase) > best_length:
                best_family = family_id
                best_length = len(phrase)
    return best_family


def item_page_title(html_text: str) -> str | None:
    """The cached item page's ``<title>`` text, whitespace-collapsed, or ``None``."""

    match = _TITLE_RE.search(html_text)
    if match is None:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or None


# Item-page ``<title>`` phrases that identify an event KIND (the per-item slug
# the download record carries as ``item_kind``), for pages an index links by
# bare id so no typed path segment names the kind. The longest matching phrase
# wins: "ICML Expo Talk Panel" is an expo item, not a session, and "Invited
# Talk: ..." is a keynote, not a plain talk. Ordered by kind so ties are stable.
ITEM_KIND_TITLE_PHRASES: tuple[tuple[str, str], ...] = (
    ("tutorial", "tutorial"),
    ("expo", "expo"),
    ("expo", "expo talk"),
    ("expo", "expo workshop"),
    ("expo", "expo demonstration"),
    ("invited-talk", "invited talk"),
    ("invited-talk", "keynote"),
    ("invited-talk", "invited speaker"),
    ("town-hall", "town hall"),
    ("affinity-event", "affinity event"),
    ("affinity-event", "affinity workshop"),
    ("test-of-time", "test of time"),
    ("workshop", "workshop"),
    ("oral", "oral"),
    ("session", "session"),
    ("poster", "poster"),
)


def classify_item_page_title_kind(
    html_text: str,
    *,
    phrases: tuple[tuple[str, str], ...] = ITEM_KIND_TITLE_PHRASES,
) -> str | None:
    """Resolve a cached item page's event kind slug from its ``<title>``.

    Companion to :func:`classify_item_page_title_family`, which answers the
    coverage question ("which declared family does this page satisfy"); this
    answers the labelling question ("what kind of event is this recording"),
    so an acquisition record for a bare-ID page carries ``item_kind`` and the
    corpus index, talk maps, and the title spine can tell a tutorial from an
    oral session. Returns ``None`` when no phrase matches.
    """

    title = item_page_title(html_text)
    if title is None:
        return None
    lowered = title.lower()
    best_kind: str | None = None
    best_length = 0
    for kind, phrase in phrases:
        if phrase in lowered and len(phrase) > best_length:
            best_kind = kind
            best_length = len(phrase)
    return best_kind


def media_family_coverage(
    discovered_targets: Iterable[Mapping[str, Any]],
    *,
    year: int,
    declared_families: Mapping[str, tuple[str, ...]],
    url_field: str = "target_url",
    source_family_field: str = "source_family_id",
    resolve_bare_id_family: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Count discovered item pages per declared media-bearing program family.

    A target counts toward a family when its URL is a typed item page whose
    slug belongs to the family. Bare-ID item pages (no family segment in the
    path) are resolved via ``resolve_bare_id_family`` when given — callers
    should back it with cached item-page content and
    :func:`classify_item_page_title_family`, since ``source_family_id``
    alone is unreliable: an index page (e.g. a conference calendar) commonly
    links items belonging to other families. Only as a last resort, when no
    resolver is given or it returns nothing, does ``source_family_id`` itself
    get tried as a family id.
    """

    slug_to_families: dict[str, list[str]] = {}
    for family_id, slugs in declared_families.items():
        for slug in slugs:
            slug_to_families.setdefault(slug, []).append(family_id)

    counts: dict[str, int] = {family_id: 0 for family_id in declared_families}
    unattributed_bare_ids = 0
    for target in discovered_targets:
        url = str(target.get(url_field) or "")
        item = classify_virtual_item_page(url, year=year)
        if item is None:
            continue
        if item.family is not None:
            for family_id in slug_to_families.get(item.family, ()):
                counts[family_id] += 1
            continue
        resolved = resolve_bare_id_family(url) if resolve_bare_id_family else None
        if resolved is None:
            resolved = str(target.get(source_family_field) or "")
        if resolved in counts:
            counts[resolved] += 1
        else:
            unattributed_bare_ids += 1

    return {
        "year": year,
        "families": {
            family_id: {
                "item_page_count": counts[family_id],
                "slugs": list(declared_families[family_id]),
            }
            for family_id in declared_families
        },
        "zero_item_families": sorted(
            family_id for family_id, count in counts.items() if count == 0
        ),
        "unattributed_bare_id_count": unattributed_bare_ids,
    }


def media_family_coverage_blockers(
    coverage: Mapping[str, Any],
    *,
    excluded_families: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Fail-closed blockers for declared families with zero discovered items.

    ``excluded_families`` must reflect an explicit, receipt-recorded operator
    exclusion decision; excluded families are skipped but remain visible in
    the coverage payload.
    """

    excluded = {str(item) for item in excluded_families}
    blockers: list[dict[str, Any]] = []
    for family_id in coverage.get("zero_item_families", ()):
        if family_id in excluded:
            continue
        blockers.append(
            {
                "code": "program_family_media_items_missing",
                "family_id": family_id,
                "retryable": True,
                "expected": (
                    "at least one discovered item page for this declared "
                    "media-bearing program family"
                ),
                "observed": "zero item pages discovered for this family",
                "next_action": (
                    "Inspect the family's index/source page for JS-rendered or "
                    "bare-ID item links and extend discovery, or record an "
                    "explicit operator exclusion receipt for this family. Do "
                    "not proceed to transcription/analysis while this family "
                    "is silently missing."
                ),
            }
        )
    return blockers
