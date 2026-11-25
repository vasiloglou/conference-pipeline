"""Resolve paper titles to open copies and publication venues (conference-neutral).

Given titles (and optional authors) of workshop papers whose PDFs are not on
OpenReview or a proceedings host, look each one up on pluggable backends:

* ``arxiv`` (default, free): arXiv id, first-version date, and the author
  ``comment`` / ``journal_ref`` that usually names the venue ("Accepted to the
  X Workshop at ..."). The title is searched as a phrase first and, when that
  finds no match, by its content words; a title with no match costs two
  requests;
* ``crossref`` (default, free): DOI-registered versions -- journal articles,
  IEEE/ACM proceedings, book chapters -- with container title and year;
* ``openalex`` (opt-in, needs ``OPENALEX_API_KEY``): all indexed locations.
  OpenAlex meters anonymous use against a small daily budget ("Insufficient
  budget ... resets at midnight UTC"); a CVPR 2026 run exhausted it after ~100
  titles, so it is never a default and never auto-funded;
* ``semantic_scholar`` (opt-in, needs ``SEMANTIC_SCHOLAR_API_KEY``): the
  shared anonymous pool returns 429 at any real volume.

A hit requires normalized-title similarity >= ``MATCH_THRESHOLD``; nothing is
inferred from a weaker match. ``classify`` then reports whether an open PDF
exists and which venues other than arXiv and the conference itself the paper
appears in, so a caller can tell a workshop paper apart from a paper
re-presented from another event. No user identity is sent to any backend.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

MATCH_THRESHOLD = 0.9
USER_AGENT = "conference-pipeline-paper-title-resolver/1.0"
DEFAULT_BACKENDS = ("arxiv", "crossref")
BACKEND_KEY_ENV = {"openalex": "OPENALEX_API_KEY", "semantic_scholar": "SEMANTIC_SCHOLAR_API_KEY"}
BACKEND_DELAY_SECONDS = {"arxiv": 3.1, "crossref": 1.0, "openalex": 0.35, "semantic_scholar": 1.1}
# Cooldowns before a refusing backend is disabled for the run: arXiv throttles
# bursts (HTTP 406/503) for a minute or two; an exhausted daily budget does not recover.
REFUSAL_COOLDOWN_SECONDS = (60.0, 120.0, 240.0)
# arXiv's title index holds no stop words (Lucene's default English set), so a
# term such as ``ti:for`` or ``ti:and`` matches no paper and one of them in an
# AND query returns nothing. Until 2026-11 every title word longer than two
# letters was ANDed: a CVPR 2026 run found 0 of the 331 titles whose query held
# one of these words, and 70 of the other 140. Checked on 2026-11-13: each word
# of the set longer than two letters returns no result as a single ``ti:`` term,
# while other function words (from, via, without, between, when, ...) do and so
# stay in the query.
ARXIV_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these they this"
    " to was will with".split()
)
ARXIV_MAX_QUERY_WORDS = 12
# Backends whose lookup can send a second request. They take the caller's
# ``sleep`` so the pause between their own requests is paced like the rest.
MULTI_REQUEST_BACKENDS = frozenset({"arxiv"})
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

Fetch = Callable[[str], bytes]


def normalize(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def similarity(a: str | None, b: str | None) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def urllib_fetch(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 40.0,
    tries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (500, 502, 503) and attempt < tries - 1:
                sleep(5.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < tries - 1:
                sleep(3.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


class BackendUnavailable(RuntimeError):
    """A backend refused service (rate limit, exhausted budget, missing key)."""


def _fetch_or_unavailable(fetch: Fetch, url: str, backend: str) -> bytes:
    try:
        return fetch(url)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 402, 403, 406, 429, 503):
            raise BackendUnavailable(f"{backend}: HTTP {exc.code}") from exc
        raise


def lookup_crossref(title: str, *, fetch: Fetch = urllib_fetch) -> dict[str, Any] | None:
    query = urllib.parse.urlencode(
        {"query.title": title[:300], "rows": 5, "select": "title,container-title,DOI,type,published,event,author"}
    )
    payload = json.loads(_fetch_or_unavailable(fetch, f"https://api.crossref.org/works?{query}", "crossref"))
    best: tuple[float, Mapping[str, Any]] | None = None
    for item in (payload.get("message") or {}).get("items", []) or []:
        item_title = " ".join(item.get("title") or [])
        score = similarity(title, item_title)
        if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
            best = (score, item)
    if best is None:
        return None
    score, item = best
    parts = ((item.get("published") or {}).get("date-parts") or [[None]])[0]
    return {
        "match_score": round(score, 3),
        "doi": item.get("DOI"),
        "title": " ".join(item.get("title") or []),
        "type": item.get("type"),
        "container_title": " ".join(item.get("container-title") or []) or None,
        "event": (item.get("event") or {}).get("name"),
        "year": parts[0] if parts else None,
        "authors": [
            " ".join(part for part in (author.get("given"), author.get("family")) if part)
            for author in item.get("author") or []
            if isinstance(author, Mapping) and (author.get("family") or author.get("given"))
        ],
    }


def lookup_semantic_scholar(title: str, *, fetch: Fetch = urllib_fetch) -> dict[str, Any] | None:
    fields = "title,year,venue,publicationVenue,externalIds,openAccessPdf,publicationDate"
    query = urllib.parse.urlencode({"query": title[:300], "fields": fields})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search/match?{query}"
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key and fetch is urllib_fetch:
        fetch = lambda u: urllib_fetch(u, headers={"x-api-key": api_key})  # noqa: E731
    try:
        payload = json.loads(_fetch_or_unavailable(fetch, url, "semantic_scholar"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # no match
            return None
        raise
    paper = (payload.get("data") or [None])[0]
    if not paper or similarity(title, paper.get("title")) < MATCH_THRESHOLD:
        return None
    ids = paper.get("externalIds") or {}
    return {
        "match_score": round(similarity(title, paper.get("title")), 3),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "venue": paper.get("venue") or ((paper.get("publicationVenue") or {}).get("name")),
        "arxiv_id": ids.get("ArXiv"),
        "doi": ids.get("DOI"),
        "open_pdf_url": (paper.get("openAccessPdf") or {}).get("url"),
    }


def lookup_openalex(title: str, *, fetch: Fetch = urllib_fetch) -> dict[str, Any] | None:
    params = {"search": re.sub(r"[:,;]", " ", title)[:300], "per-page": 5}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    query = urllib.parse.urlencode(params)
    payload = json.loads(_fetch_or_unavailable(fetch, f"https://api.openalex.org/works?{query}", "openalex"))
    best: tuple[float, Mapping[str, Any]] | None = None
    for work in payload.get("results", []) or []:
        score = similarity(title, work.get("title"))
        if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
            best = (score, work)
    if best is None:
        return None
    score, work = best
    return {
        "match_score": round(score, 3),
        "openalex_id": work.get("id"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "date": work.get("publication_date"),
        "type": work.get("type"),
        "doi": work.get("doi"),
        "primary_source": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "locations": [
            {
                "source": ((loc.get("source") or {}).get("display_name")),
                "landing_page_url": loc.get("landing_page_url"),
                "pdf_url": loc.get("pdf_url"),
                "version": loc.get("version"),
            }
            for loc in work.get("locations") or []
        ],
        "oa_url": (work.get("open_access") or {}).get("oa_url"),
    }


def arxiv_title_queries(title: str) -> list[str]:
    """arXiv ``search_query`` expressions for one title, in the order to try them.

    First the whole title as a phrase, which arXiv analyzes the same way it
    indexed its own titles. Then the title's content words ANDed together,
    which also finds a copy whose title differs by a word or two. The second
    form must leave stop words out (see ``ARXIV_STOP_WORDS``).
    """
    if not normalize(title):
        return []
    # Inside a quoted phrase only the double quote and the backslash are query syntax.
    queries = ['ti:"{}"'.format(" ".join(re.sub(r'["\\]', " ", title).split()))]
    words = [w for w in normalize(title).split() if len(w) > 2 and w not in ARXIV_STOP_WORDS]
    if words:
        queries.append(" AND ".join(f"ti:{w}" for w in words[:ARXIV_MAX_QUERY_WORDS]))
    return queries


def lookup_arxiv(
    title: str, *, fetch: Fetch = urllib_fetch, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any] | None:
    """Best arXiv entry whose title matches ``title``, or ``None``.

    Tries each ``arxiv_title_queries`` expression until an entry reaches
    ``MATCH_THRESHOLD``. arXiv allows one request every three seconds, so every
    request after the first waits ``BACKEND_DELAY_SECONDS["arxiv"]`` through
    ``sleep``; the caller paces the gap between lookups.
    """
    best: tuple[float, Any, str] | None = None
    for position, search_query in enumerate(arxiv_title_queries(title)):
        if position:
            sleep(BACKEND_DELAY_SECONDS["arxiv"])
        query = urllib.parse.urlencode({"search_query": search_query, "max_results": 5})
        root = ET.fromstring(_fetch_or_unavailable(fetch, f"https://export.arxiv.org/api/query?{query}", "arxiv"))
        for entry in root.findall(f"{_ATOM}entry"):
            entry_title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            score = similarity(title, entry_title)
            if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
                best = (score, entry, entry_title)
        if best is not None:
            break
    if best is None:
        return None
    score, entry, entry_title = best
    arxiv_id = (entry.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1]
    return {
        "match_score": round(score, 3),
        "arxiv_id": arxiv_id,
        "title": entry_title,
        "published": entry.findtext(f"{_ATOM}published"),
        "comment": " ".join((entry.findtext(f"{_ARXIV}comment") or "").split()) or None,
        "journal_ref": entry.findtext(f"{_ARXIV}journal_ref"),
        "authors": [
            " ".join((author.findtext(f"{_ATOM}name") or "").split())
            for author in entry.findall(f"{_ATOM}author")
            if (author.findtext(f"{_ATOM}name") or "").strip()
        ],
        "pdf_url": f"https://export.arxiv.org/pdf/{re.sub(r'v[0-9]+$', '', arxiv_id)}" if arxiv_id else None,
    }


def classify(record: Mapping[str, Any], *, home_venue_patterns: Iterable[str] = ()) -> dict[str, Any]:
    """Summarize one resolved record across whichever backends ran.

    ``home_venue_patterns`` are case-insensitive regexes naming the conference
    itself (e.g. its acronym); venues matching them, and arXiv, are not
    "other" venues.
    """
    patterns = [re.compile(p, re.I) for p in home_venue_patterns]
    openalex = record.get("openalex") or {}
    arxiv = record.get("arxiv") or {}
    crossref = record.get("crossref") or {}
    s2 = record.get("semantic_scholar") or {}
    arxiv_id = arxiv.get("arxiv_id") or s2.get("arxiv_id")
    open_pdf = arxiv.get("pdf_url")
    other_venues: list[str] = []

    def add_venue(name: str | None) -> None:
        if name and "arxiv" not in name.lower() and not any(p.search(name) for p in patterns) and name not in other_venues:
            other_venues.append(name)

    for loc in openalex.get("locations") or []:
        source = loc.get("source") or ""
        url = (loc.get("landing_page_url") or "").lower()
        if "arxiv" in source.lower() or "arxiv.org" in url:
            if not arxiv_id and "/abs/" in url:
                arxiv_id = url.rsplit("/abs/", 1)[-1]
            open_pdf = open_pdf or loc.get("pdf_url")
            continue
        add_venue(source)
    add_venue(crossref.get("container_title") or crossref.get("event"))
    add_venue(s2.get("venue"))
    if not open_pdf and arxiv_id:
        open_pdf = f"https://export.arxiv.org/pdf/{re.sub(r'v[0-9]+$', '', arxiv_id)}"
    open_pdf = open_pdf or s2.get("open_pdf_url") or openalex.get("oa_url")
    venue_text = " ".join(filter(None, [arxiv.get("comment"), arxiv.get("journal_ref")]))
    years = [openalex.get("year"), crossref.get("year"), s2.get("year"), _year(arxiv.get("published"))]
    return {
        "found": bool(openalex or arxiv or crossref or s2),
        "arxiv_id": arxiv_id,
        "open_pdf_url": open_pdf,
        "other_venues": other_venues,
        "doi": crossref.get("doi") or s2.get("doi") or openalex.get("doi"),
        "arxiv_venue_note": venue_text or None,
        "earliest_year": min((y for y in years if y), default=None),
    }


def _year(value: str | None) -> int | None:
    match = re.match(r"(\d{4})", value or "")
    return int(match.group(1)) if match else None


BACKENDS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "arxiv": lookup_arxiv,
    "crossref": lookup_crossref,
    "openalex": lookup_openalex,
    "semantic_scholar": lookup_semantic_scholar,
}


def check_backends(backends: Iterable[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Validate backend names; opt-in backends require their API key in the environment."""
    env = os.environ if env is None else env
    chosen = list(backends)
    for name in chosen:
        if name not in BACKENDS:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
        key = BACKEND_KEY_ENV.get(name)
        if key and not env.get(key):
            raise ValueError(f"backend {name!r} needs {key} (anonymous use is rate-limited or budget-metered)")
    return chosen


def resolve_titles(
    items: list[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
    fetch: Fetch = urllib_fetch,
    sleep: Callable[[float], None] = time.sleep,
    backends: Iterable[str] = DEFAULT_BACKENDS,
    home_venue_patterns: Iterable[str] = (),
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Resolve ``items`` ({key, title, ...}); resumable through ``output_path``.

    A backend that refuses service (``BackendUnavailable``) is retried after
    ``REFUSAL_COOLDOWN_SECONDS`` cooldowns; if it still refuses it is disabled
    for the rest of the run and recorded on each later record as skipped
    (rerun later to fill those in), instead of retrying into a hang.
    """
    chosen = list(backends)
    done: dict[str, dict[str, Any]] = {}
    if output_path and output_path.exists():
        try:
            done = {
                r["key"]: r
                for r in json.loads(output_path.read_text(encoding="utf-8"))
                if "error" not in r and not r.get("backends_skipped") and set(chosen) <= set(r.get("backends_run") or [])
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            done = {}
    patterns = list(home_venue_patterns)
    disabled: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        key = str(item["key"])
        if key in done:
            results.append(done[key])
            continue
        record: dict[str, Any] = dict(item)
        record["backends_run"] = []
        try:
            for name in chosen:
                if name in disabled:
                    record.setdefault("backends_skipped", {})[name] = disabled[name]
                    continue
                pacing = {"sleep": sleep} if name in MULTI_REQUEST_BACKENDS else {}
                for attempt, cooldown in enumerate((*REFUSAL_COOLDOWN_SECONDS, None)):
                    try:
                        record[name] = BACKENDS[name](str(item["title"]), fetch=fetch, **pacing)
                        record["backends_run"].append(name)
                        break
                    except BackendUnavailable as exc:
                        if cooldown is None:
                            disabled[name] = str(exc)
                            record.setdefault("backends_skipped", {})[name] = str(exc)
                        else:
                            sleep(cooldown)
                sleep(BACKEND_DELAY_SECONDS.get(name, 1.0))
            record["resolution"] = classify(record, home_venue_patterns=patterns)
        except Exception as exc:  # noqa: BLE001 - recorded; rerun resumes
            record["error"] = f"{exc.__class__.__name__}: {exc}"
        results.append(record)
        if output_path and (index % 20 == 0 or index == len(items)):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if progress:
            progress(index, len(items))
    return results


def summarize(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    resolved = [r for r in results if r.get("resolution")]
    return {
        "titles": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "found": sum(1 for r in resolved if r["resolution"]["found"]),
        "open_pdf": sum(1 for r in resolved if r["resolution"]["open_pdf_url"]),
        "arxiv": sum(1 for r in resolved if r["resolution"]["arxiv_id"]),
        "with_other_venues": sum(1 for r in resolved if r["resolution"]["other_venues"]),
        "backends_skipped": sorted({name for r in results for name in (r.get("backends_skipped") or {})}),
    }
