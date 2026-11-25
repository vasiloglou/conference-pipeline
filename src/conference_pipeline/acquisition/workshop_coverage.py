"""Workshop paper coverage: schedule vs OpenReview vs proceedings vs other hosts.

Answers, for every workshop on a conference's official schedule, where its
papers are and whether the corpus has them. Deterministic, offline (except
the optional OpenReview group-metadata fetch), and conference-neutral.

Inputs
------
* schedule: the official program's workshop list (id, title, website). Field
  names differ per conference; callers pass them explicitly.
* OpenReview workshop groups with full title + website
  (``fetch_openreview_workshop_groups``). Short venue names alone mis-match:
  organizers rename workshops between hosts and split them into sub-tracks.
* the OpenReview workshop inventory and download receipt (paper counts).
* proceedings workshops (``proceedings_sources``).
* optional per-schedule-id paper counts from other registered sources
  (workshop websites, arXiv).

Matching rules (learned on CVPR 2026, where naive short-name matching gave 40
"orphan" proceedings workshops instead of 31):

1. OpenReview sub-tracks (``X_Proceedings_Track``, ``X_Non-Archival_Track``,
   ``X-AT``/``X-NA``) are grouped under base name ``X`` even when no plain
   ``X`` group exists.
2. schedule <-> OpenReview group: website match first, then the base short
   name appearing in the schedule website URL, then normalized title
   similarity; greedy one-to-one by score, threshold ``TITLE_THRESHOLD``.
3. schedule <-> proceedings workshop: title similarity (one-to-one); then a
   proceedings short name equal to a matched OpenReview base name; then
   additional near-identical listings (a host may list one workshop twice).
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from conference_pipeline.acquisition.proceedings_sources import ProceedingsWorkshop

COVERAGE_RECEIPT_TYPE = "conference_pipeline.workshop_paper_coverage"
TITLE_THRESHOLD = 0.8
DUPLICATE_LISTING_THRESHOLD = 0.9
# Hosting providers whose hostname alone does not identify one workshop.
SHARED_HOSTS = ("github.io", "sites.google.com", "codabench.org", "openreview.net", "google.com", "notion.site")

CATEGORY_ORDER = (
    "openreview_and_proceedings",
    "openreview_only",
    "proceedings_only",
    "other_source_only",
    "openreview_titles_only",
    "openreview_venue_empty",
    "no_source",
)

_TITLE_STOPWORDS = re.compile(
    r"\b(the|workshop|workshops|on|in|conjunction|with|and|for|of|a|an|at|\d+(st|nd|rd|th)|first|second|third|"
    r"fourth|fifth|sixth|seventh|eighth|ninth|tenth|ieee|cvf|international|annual|edition|track|archival|non|"
    r"proceedings?|proceeding|19\d\d|20\d\d)\b"
)
_TRACK_SUFFIX = re.compile(
    r"([_-](non[-_ ]?)?(archival|proceedings?|proceeding|extended[-_]abstracts?)([-_]track)?|-at|-na)$", re.I
)


@dataclass(frozen=True)
class ScheduleWorkshop:
    schedule_id: str
    title: str
    website: str | None


def load_schedule(
    path: Path,
    *,
    id_field: str = "schedule_id",
    title_field: str = "title",
    website_field: str = "website",
    list_key: str = "workshops",
) -> list[ScheduleWorkshop]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get(list_key) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"schedule file has no '{list_key}' list: {path}")
    schedule: list[ScheduleWorkshop] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get(id_field) or not row.get(title_field):
            continue
        website = row.get(website_field)
        schedule.append(ScheduleWorkshop(str(row[id_field]), str(row[title_field]), str(website) if website else None))
    return schedule


# --------------------------------------------------------------------------- normalization


def title_key(title: str | None, *, extra_stopwords: Iterable[str] = ()) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())
    text = _TITLE_STOPWORDS.sub(" ", text)
    for word in extra_stopwords:
        text = re.sub(rf"\b{re.escape(word.lower())}\b", " ", text)
    return " ".join(text.split())


def title_similarity(a: str | None, b: str | None, *, extra_stopwords: Iterable[str] = ()) -> float:
    return difflib.SequenceMatcher(
        None, title_key(a, extra_stopwords=extra_stopwords), title_key(b, extra_stopwords=extra_stopwords)
    ).ratio()


def url_key(url: str | None) -> str:
    if not url:
        return ""
    found = re.search(r"https?://\S+", url)
    parts = urlsplit((found.group(0) if found else url).strip().lower())
    return (parts.netloc.removeprefix("www.") + parts.path).rstrip("/")


def websites_match(a: str | None, b: str | None) -> bool:
    ka, kb = url_key(a), url_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka.startswith(kb + "/") or kb.startswith(ka + "/"):
        return True
    host_a, host_b = ka.split("/")[0], kb.split("/")[0]
    return host_a == host_b and not any(shared in host_a for shared in SHARED_HOSTS)


def openreview_base_name(short_name: str) -> str:
    return _TRACK_SUFFIX.sub("", short_name)


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^\d+(st|nd|rd|th)", "", name.lower()))


# --------------------------------------------------------------------------- OpenReview groups


def fetch_openreview_workshop_groups(
    parent_prefix: str, get_json: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Direct child groups of ``parent_prefix`` with title/subtitle/website.

    ``get_json(path, params)`` performs an (authenticated) API GET, e.g.
    ``("groups", {"prefix": ..., "limit": 1000})``.
    """
    prefix = parent_prefix.rstrip("/") + "/"
    payload = get_json("groups", {"prefix": prefix, "limit": 1000})
    depth = prefix.count("/")
    groups: dict[str, dict[str, Any]] = {}
    for group in payload.get("groups", []) or []:
        group_id = str(group.get("id") or "")
        if not group_id.startswith(prefix) or group_id.count("/") != depth:
            continue
        content = group.get("content") or {}

        def value(key: str) -> Any:
            item = content.get(key)
            return item.get("value") if isinstance(item, Mapping) else item

        groups[group_id] = {"title": value("title"), "subtitle": value("subtitle"), "website": value("website")}
    return groups


# --------------------------------------------------------------------------- coverage


def _greedy(pairs: list[tuple[float, str, str, str]], threshold: float) -> dict[str, tuple[str, str, float]]:
    pairs.sort(key=lambda item: -item[0])
    assigned: dict[str, tuple[str, str, float]] = {}
    used: set[str] = set()
    for score, via, left, right in pairs:
        if score < threshold:
            break
        if left in assigned or right in used:
            continue
        assigned[left] = (right, via, round(min(score, 1.0), 3))
        used.add(right)
    return assigned


def build_coverage(
    *,
    schedule: list[ScheduleWorkshop],
    openreview_groups: Mapping[str, Mapping[str, Any]],
    openreview_inventory: Mapping[str, Any] | None,
    openreview_download_receipt: Mapping[str, Any] | None,
    proceedings: list[ProceedingsWorkshop],
    other_source_papers: Mapping[str, Mapping[str, int]] | None = None,
    extra_title_stopwords: Iterable[str] = (),
) -> dict[str, Any]:
    """Match schedule workshops to paper sources; see module docstring for the rules.

    ``other_source_papers`` maps schedule_id -> {source_id: papers_available}.
    ``extra_title_stopwords`` are conference-identity words (e.g. the
    conference acronym and year) that should not drive title similarity.
    """
    stop = tuple(extra_title_stopwords)
    inventory_rows = {
        str(row.get("venueid")): row
        for row in (openreview_inventory or {}).get("workshops", []) or []
        if isinstance(row, Mapping)
    }
    downloaded = {
        str(row.get("venueid")): row
        for row in (openreview_download_receipt or {}).get("per_workshop", []) or []
        if isinstance(row, Mapping)
    }
    venue_ids = set(openreview_groups) | set(inventory_rows)
    groups: dict[str, list[str]] = defaultdict(list)
    for venue_id in sorted(venue_ids):
        groups[openreview_base_name(venue_id.rsplit("/", 1)[-1])].append(venue_id)

    pairs: list[tuple[float, str, str, str]] = []
    for workshop in schedule:
        site = url_key(workshop.website)
        for base, members in groups.items():
            titles = [openreview_groups.get(m, {}).get("title") for m in members]
            score = max((title_similarity(workshop.title, t, extra_stopwords=stop) for t in titles if t), default=0.0)
            via = "title"
            if any(websites_match(workshop.website, openreview_groups.get(m, {}).get("website")) for m in members):
                score, via = 2.0, "website"
            elif len(base) >= 3 and site and re.search(r"(^|[^a-z0-9])" + re.escape(base.lower()) + r"([^a-z]|$)", site):
                score, via = max(score, 0.95), "short_name_in_website"
            pairs.append((score, via, workshop.schedule_id, base))
    openreview_match = _greedy(pairs, TITLE_THRESHOLD)
    schedule_by_base = {base: sid for sid, (base, _, _) in openreview_match.items()}

    by_key = {f"{p.source_id}:{p.short_name}": p for p in proceedings}
    title_pairs = [
        (title_similarity(w.title, p.title, extra_stopwords=stop), "title", w.schedule_id, key)
        for w in schedule
        for key, p in by_key.items()
    ]
    proceedings_match: dict[str, list[tuple[str, str, float | None]]] = defaultdict(list)
    for sid, (key, via, score) in _greedy(list(title_pairs), TITLE_THRESHOLD).items():
        proceedings_match[sid].append((key, via, score))
    used = {key for matches in proceedings_match.values() for key, _, _ in matches}
    for key, p in by_key.items():
        if key in used:
            continue
        for base, sid in schedule_by_base.items():
            if _compact(p.short_name) == _compact(base):
                proceedings_match[sid].append((key, "short_name_via_openreview", None))
                used.add(key)
                break
    for score, _, sid, key in sorted(title_pairs, key=lambda item: -item[0]):
        if key in used or score < DUPLICATE_LISTING_THRESHOLD:
            continue
        proceedings_match[sid].append((key, "duplicate_listing", round(score, 3)))
        used.add(key)

    rows: list[dict[str, Any]] = []
    for workshop in schedule:
        match = openreview_match.get(workshop.schedule_id)
        members = groups[match[0]] if match else []
        accepted = sum(int(inventory_rows.get(m, {}).get("paper_count") or 0) for m in members)
        pdfs = sum(int(downloaded.get(m, {}).get("available") or 0) for m in members)
        matched_proceedings = [by_key[key] for key, _, _ in proceedings_match.get(workshop.schedule_id, [])]
        listed = sum(p.papers_listed for p in matched_proceedings)
        others = dict((other_source_papers or {}).get(workshop.schedule_id, {}))
        other_total = sum(others.values())
        if pdfs and listed:
            category = "openreview_and_proceedings"
        elif pdfs:
            category = "openreview_only"
        elif listed:
            category = "proceedings_only"
        elif other_total:
            category = "other_source_only"
        elif match and accepted:
            category = "openreview_titles_only"
        elif match:
            category = "openreview_venue_empty"
        else:
            category = "no_source"
        rows.append(
            {
                **asdict(workshop),
                "category": category,
                "openreview": {
                    "base_name": match[0],
                    "matched_via": match[1],
                    "score": match[2],
                    "venues": members,
                    "papers_accepted": accepted,
                    "pdfs_downloaded": pdfs,
                }
                if match
                else None,
                "proceedings": [
                    {
                        "source_id": by_key[key].source_id,
                        "short_name": by_key[key].short_name,
                        "title": by_key[key].title,
                        "matched_via": via,
                        "score": score,
                        "papers_listed": by_key[key].papers_listed,
                        "papers_new": by_key[key].papers_available,
                    }
                    for key, via, score in proceedings_match.get(workshop.schedule_id, [])
                ],
                "other_sources": others,
                "has_papers": category in CATEGORY_ORDER[:4],
            }
        )

    matched_bases = {base for base, _, _ in openreview_match.values()}
    counts = {category: sum(1 for r in rows if r["category"] == category) for category in CATEGORY_ORDER}
    return {
        "schema_version": 1,
        "receipt_type": COVERAGE_RECEIPT_TYPE,
        "schedule_workshop_count": len(schedule),
        "workshops_with_papers": sum(1 for r in rows if r["has_papers"]),
        "category_counts": counts,
        "paper_totals": {
            "openreview_pdfs": sum((r["openreview"] or {}).get("pdfs_downloaded", 0) for r in rows),
            "proceedings_listed": sum(p["papers_listed"] for r in rows for p in r["proceedings"]),
            "proceedings_new": sum(p["papers_new"] for r in rows for p in r["proceedings"]),
            "other_sources": sum(sum(r["other_sources"].values()) for r in rows),
        },
        "proceedings_without_openreview": [
            {"schedule_id": r["schedule_id"], "title": r["title"], "proceedings": r["proceedings"]}
            for r in rows
            if r["proceedings"] and not r["openreview"]
        ],
        "unmatched_openreview_groups": sorted(base for base in groups if base not in matched_bases),
        "unmatched_proceedings": sorted(key for key in by_key if key not in used),
        "workshops": rows,
    }


def render_markdown(coverage: Mapping[str, Any], *, heading: str = "Workshop paper coverage") -> str:
    lines = [f"# {heading}", ""]
    lines.append(
        f"{coverage['workshops_with_papers']} of {coverage['schedule_workshop_count']} scheduled workshops have papers."
    )
    lines += ["", "| Category | Workshops |", "|---|---|"]
    for category, count in coverage["category_counts"].items():
        lines.append(f"| {category} | {count} |")
    totals = coverage["paper_totals"]
    lines += [
        "",
        f"Papers: {totals['openreview_pdfs']} OpenReview PDFs, {totals['proceedings_listed']} proceedings-listed "
        f"({totals['proceedings_new']} new), {totals['other_sources']} from other registered sources.",
        "",
        "## Workshops without papers",
        "",
        "| Schedule id | Category | OpenReview venue | Title | Website |",
        "|---|---|---|---|---|",
    ]
    for row in coverage["workshops"]:
        if row["has_papers"]:
            continue
        base = (row["openreview"] or {}).get("base_name") or "-"
        lines.append(f"| {row['schedule_id']} | {row['category']} | {base} | {row['title']} | {row['website'] or '-'} |")
    lines += ["", "## Proceedings workshops without an OpenReview venue", "", "| Papers | Source:short name | Title |", "|---|---|---|"]
    for item in sorted(coverage["proceedings_without_openreview"], key=lambda i: -sum(p["papers_listed"] for p in i["proceedings"])):
        names = ", ".join(f"{p['source_id']}:{p['short_name']}" for p in item["proceedings"])
        lines.append(f"| {sum(p['papers_listed'] for p in item['proceedings'])} | {names} | {item['title']} |")
    lines += [
        "",
        f"Unmatched OpenReview groups: {', '.join(coverage['unmatched_openreview_groups']) or 'none'}",
        f"Unmatched proceedings listings: {', '.join(coverage['unmatched_proceedings']) or 'none'}",
        "",
    ]
    return "\n".join(lines)
