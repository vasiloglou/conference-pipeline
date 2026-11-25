"""Offline coverage for the conference-workshop-paper-discovery platform pieces."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from conference_pipeline.acquisition import workshop_coverage as wc
from conference_pipeline.acquisition import workshop_paper_sources as sources
from conference_pipeline.acquisition import workshop_site_papers as site
from conference_pipeline.acquisition.proceedings_sources import ProceedingsWorkshop, load_proceedings_workshops
from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid
from conference_pipeline.indexing import paper_title_resolver as resolver


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _download_receipt(status: str = "succeeded", per_workshop=None) -> dict:
    return {
        "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
        "status": status,
        "output_dir": "storage/workshops/papers",
        "per_workshop": per_workshop or [],
    }


# --------------------------------------------------------------------------- registry


def test_registry_registers_pins_and_blocks_drift(tmp_path: Path) -> None:
    ws = tmp_path / "conf"
    receipt = _write(ws / "storage/workshops/a_receipt.json", _download_receipt())
    assert sources.load_registered_sources(ws) == ([], [])

    sources.register_source(workspace=ws, source_id="a", receipt_path="storage/workshops/a_receipt.json")
    loaded, blockers = sources.load_registered_sources(ws)
    assert blockers == [] and [s["source_id"] for s in loaded] == ["a"]

    _write(receipt, {**_download_receipt(), "extra": 1})
    loaded, blockers = sources.load_registered_sources(ws)
    assert loaded == [] and blockers[0]["code"] == "workshop_paper_source_receipt_digest_mismatch"

    sources.register_source(workspace=ws, source_id="a", receipt_path=receipt)  # re-pin, absolute path accepted
    # a caller holding <workspace>/<relative> must not double-prefix the workspace
    import os

    cwd = Path.cwd()
    try:
        os.chdir(ws.parent)
        sources.register_source(workspace=ws, source_id="a", receipt_path=Path(ws.name) / "storage/workshops/a_receipt.json")
    finally:
        os.chdir(cwd)
    assert json.loads(sources.registry_path(ws).read_text())["sources"][0]["receipt_path"] == "storage/workshops/a_receipt.json"
    loaded, blockers = sources.load_registered_sources(ws)
    assert blockers == [] and len(json.loads(sources.registry_path(ws).read_text())["sources"]) == 1


def test_registry_admits_partial_but_refuses_blocked_receipt(tmp_path: Path) -> None:
    ws = tmp_path / "conf"
    # partial is normal for site sources: some listed papers have no public PDF
    _write(ws / "partial.json", _download_receipt(status="partial"))
    sources.register_source(workspace=ws, source_id="site", receipt_path="partial.json")
    loaded, blockers = sources.load_registered_sources(ws)
    assert blockers == [] and [s["source_id"] for s in loaded] == ["site"]

    _write(ws / "blocked.json", _download_receipt(status="blocked"))
    with pytest.raises(ValueError):
        sources.register_source(workspace=ws, source_id="x", receipt_path="blocked.json")


# --------------------------------------------------------------------------- coverage


def test_openreview_group_fetch_keeps_direct_children_only() -> None:
    payload = {
        "groups": [
            {"id": "org/Conf/2099/Workshop/ABC", "content": {"title": {"value": "ABC Workshop"}, "website": {"value": "https://abc.org"}}},
            {"id": "org/Conf/2099/Workshop/ABC/Authors", "content": {}},
            {"id": "org/Conf/2099/Workshop_Proposals/X", "content": {}},
        ]
    }
    groups = wc.fetch_openreview_workshop_groups("org/Conf/2099/Workshop", lambda path, params: payload)
    assert groups == {"org/Conf/2099/Workshop/ABC": {"title": "ABC Workshop", "subtitle": None, "website": "https://abc.org"}}


def test_base_name_strips_track_suffixes() -> None:
    assert wc.openreview_base_name("DriveX_Non-Archival_Track") == "DriveX"
    assert wc.openreview_base_name("HOW_Proceedings_Track") == "HOW"
    assert wc.openreview_base_name("AUTOPILOT-AT") == "AUTOPILOT"
    assert wc.openreview_base_name("DataMFM_Non-archival") == "DataMFM"
    assert wc.openreview_base_name("MetaFood") == "MetaFood"


def test_build_coverage_matching_rules_and_categories() -> None:
    V = "org/Conf/2099/Workshop/"
    schedule = [
        wc.ScheduleWorkshop("1", "Foundation Models for Autonomous Driving", "https://drivex.github.io/2099"),
        wc.ScheduleWorkshop("2", "12th Workshop on Medical Computer Vision", "https://sites.google.com/view/mcv"),
        wc.ScheduleWorkshop("3", "Omnidirectional Computer Vision", "https://sites.google.com/view/omni"),
        wc.ScheduleWorkshop("4", "Efficient and On-Device Generation", "https://edge-ws.github.io/"),
        wc.ScheduleWorkshop("5", "Workshop on Any-to-any Multimodal Learning", "https://a2a-mml.github.io/"),
        wc.ScheduleWorkshop("6", "Image Restoration Challenges", "https://ntire.org"),
        wc.ScheduleWorkshop("7", "Bitter Lessons", None),
    ]
    groups = {
        V + "DriveX_Archival_Track": {"title": "[Archival] DriveX - Foundation Models for V2X", "website": "https://drivex.github.io/2099"},
        V + "DriveX_Non-Archival_Track": {"title": "[Non-Archival] DriveX", "website": "https://drivex.github.io/2099/"},
        V + "OmniCV": {"title": "6th Omnidirectional Computer Vision Workshop", "website": "https://sites.google.com/view/omni"},
        V + "EDGE": {"title": "3rd Workshop on Efficient and On-Device Generation (EDGE)", "website": "https://edge-ws.github.io"},
        V + "A2A-MML": {"title": "Any-to-Any Multimodal Learning Workshop", "website": "https://other-host.org"},
    }
    inventory = {"workshops": [{"venueid": V + "OmniCV", "paper_count": 5}, {"venueid": V + "EDGE", "paper_count": 8}]}
    receipt = {"per_workshop": [{"venueid": V + "EDGE", "available": 8}]}
    proceedings = [
        ProceedingsWorkshop("host", "DriveX", "Foundation Models for V2X-Based Cooperative Driving", None, 11, 11),
        ProceedingsWorkshop("host", "EDGE", "3rd Workshop on Efficient and On-Device Generation (EDGE)", None, 8, 0),
        ProceedingsWorkshop("host", "EGDE", "3rd Workshop on Efficient and On-Device Generation (EDGE)", None, 1, 1),
        ProceedingsWorkshop("host", "NTIRE", "Image Restoration Challenges", None, 106, 106),
    ]
    cov = wc.build_coverage(
        schedule=schedule, openreview_groups=groups, openreview_inventory=inventory,
        openreview_download_receipt=receipt, proceedings=proceedings,
        other_source_papers={"7": {"workshop_site": 3}},
    )
    rows = {r["schedule_id"]: r for r in cov["workshops"]}
    # sub-tracks without a base group are grouped; website match
    assert rows["1"]["openreview"]["base_name"] == "DriveX" and len(rows["1"]["openreview"]["venues"]) == 2
    # proceedings short name joins through the OpenReview base name when titles differ
    assert [p["short_name"] for p in rows["1"]["proceedings"]] == ["DriveX"]
    # a shared host (sites.google.com) with a different path is not a website match
    assert rows["2"]["openreview"] is None and rows["2"]["category"] == "no_source"
    assert rows["3"]["openreview"]["base_name"] == "OmniCV" and rows["3"]["category"] == "openreview_titles_only"
    # duplicate listings of one workshop on a proceedings host both attach
    assert sorted(p["short_name"] for p in rows["4"]["proceedings"]) == ["EDGE", "EGDE"]
    assert rows["4"]["category"] == "openreview_and_proceedings"
    assert rows["5"]["openreview"]["matched_via"] in {"short_name_in_website", "title"}
    assert rows["5"]["category"] == "openreview_venue_empty"
    assert rows["6"]["category"] == "proceedings_only" and cov["proceedings_without_openreview"][0]["schedule_id"] == "6"
    assert rows["7"]["category"] == "other_source_only" and rows["7"]["has_papers"]
    assert cov["unmatched_proceedings"] == [] and cov["workshops_with_papers"] == 4
    assert "Workshops without papers" in wc.render_markdown(cov)


def test_load_proceedings_workshops_normalizes_cvf_receipt(tmp_path: Path) -> None:
    receipt = {"per_workshop": [{"cvf_short_name": "EV", "title": "EarthVision", "paper_count_listed": 3, "available": 2}]}
    workshops, loaded = load_proceedings_workshops(tmp_path, receipts={"cvf_openaccess": receipt})
    assert loaded == ["cvf_openaccess"]
    assert workshops[0].short_name == "EV" and workshops[0].papers_listed == 3 and workshops[0].papers_available == 2


# --------------------------------------------------------------------------- title resolver


OPENALEX_HIT = {
    "results": [
        {
            "id": "https://openalex.org/W1", "title": "Sparse Widgets for Vision", "publication_year": 2025,
            "type": "article",
            "primary_location": {"source": {"display_name": "Pattern Recognition Letters"}},
            "locations": [
                {"source": {"display_name": "Pattern Recognition Letters"}, "landing_page_url": "https://doi.org/x"},
                {"source": {"display_name": "arXiv (Cornell University)"}, "landing_page_url": "https://arxiv.org/abs/2501.00001", "pdf_url": "https://arxiv.org/pdf/2501.00001"},
                {"source": {"display_name": "Conf 2099 Workshops"}, "landing_page_url": "https://conf.org"},
            ],
        },
        {"id": "https://openalex.org/W2", "title": "Something Else Entirely"},
    ]
}
ARXIV_FEED = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2601.12345v2</id><title>Dense Gadgets
for Robots</title><published>2026-01-05T00:00:00Z</published><arxiv:comment>Accepted to the XYZ Workshop at Conf 2099</arxiv:comment></entry></feed>"""


CROSSREF_HIT = {"message": {"items": [
    {"title": ["Sparse Widgets for Vision"], "container-title": ["IEEE Transactions on Widgets"], "DOI": "10.1/tw.1",
     "type": "journal-article", "published": {"date-parts": [[2024, 5]]}},
    {"title": ["Unrelated"], "container-title": ["X"]},
]}}


def _resolver_fetch(url: str) -> bytes:
    if "openalex" in url:
        return json.dumps(OPENALEX_HIT if "Sparse" in url else {"results": []}).encode()
    if "crossref" in url:
        return json.dumps(CROSSREF_HIT if "Sparse" in url else {"message": {"items": []}}).encode()
    return ARXIV_FEED if "gadgets" in url.lower() else b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


ITEMS = [
    {"key": "a", "title": "Sparse Widgets for Vision"},
    {"key": "b", "title": "Dense Gadgets for Robots"},
    {"key": "c", "title": "Nothing Matches This"},
]


def test_resolver_default_backends_find_open_copies_and_other_venues(tmp_path: Path) -> None:
    out = tmp_path / "res.json"
    calls: list[str] = []
    results = resolver.resolve_titles(
        ITEMS, output_path=out, fetch=lambda u: calls.append(u) or _resolver_fetch(u), sleep=lambda s: None,
        home_venue_patterns=[r"\bConf\b"],
    )
    assert not any("openalex" in u or "semanticscholar" in u for u in calls)  # opt-in only
    by_key = {r["key"]: r["resolution"] for r in results}
    assert by_key["a"]["other_venues"] == ["IEEE Transactions on Widgets"] and by_key["a"]["doi"] == "10.1/tw.1"
    assert by_key["a"]["earliest_year"] == 2024
    assert by_key["b"]["open_pdf_url"] == "https://export.arxiv.org/pdf/2601.12345"
    assert "XYZ Workshop" in by_key["b"]["arxiv_venue_note"]
    assert by_key["c"]["found"] is False
    assert resolver.summarize(results)["found"] == 2

    calls.clear()
    resolver.resolve_titles(ITEMS, output_path=out, fetch=lambda u: calls.append(u) or b"", sleep=lambda s: None)
    assert calls == []  # fully resumed from the output file


def test_resolver_openalex_locations_and_home_venue_filter() -> None:
    record = {"openalex": resolver.lookup_openalex("Sparse Widgets for Vision", fetch=_resolver_fetch)}
    resolution = resolver.classify(record, home_venue_patterns=[r"\bConf\b"])
    assert resolution["other_venues"] == ["Pattern Recognition Letters"]
    assert resolution["arxiv_id"] == "2501.00001" and resolution["open_pdf_url"] == "https://arxiv.org/pdf/2501.00001"


def test_resolver_recovers_after_a_throttle_cooldown() -> None:
    state = {"refusals": 1}

    def fetch(url: str) -> bytes:
        if "arxiv" in url and state["refusals"]:
            state["refusals"] -= 1
            raise urllib.error.HTTPError(url, 406, "Not Acceptable", None, None)
        return _resolver_fetch(url)

    results = resolver.resolve_titles(ITEMS[1:2], fetch=fetch, sleep=lambda s: None)
    assert "backends_skipped" not in results[0] and results[0]["resolution"]["arxiv_id"] == "2601.12345v2"


def test_resolver_disables_refusing_backend_instead_of_hanging(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        if "crossref" in url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
        return _resolver_fetch(url)

    sleeps: list[float] = []
    results = resolver.resolve_titles(ITEMS, fetch=fetch, sleep=sleeps.append)
    assert all(r["backends_skipped"] == {"crossref": "crossref: HTTP 429"} for r in results)
    assert sum(1 for s in sleeps if s in resolver.REFUSAL_COOLDOWN_SECONDS) == len(resolver.REFUSAL_COOLDOWN_SECONDS)
    assert results[1]["resolution"]["arxiv_id"] == "2601.12345v2"
    assert resolver.summarize(results)["backends_skipped"] == ["crossref"]
    with pytest.raises(ValueError):
        resolver.check_backends(["openalex"], env={})
    assert resolver.check_backends(["openalex"], env={"OPENALEX_API_KEY": "k"}) == ["openalex"]


# --------------------------------------------------------------------------- arXiv title queries

# Three papers that are on arXiv under exactly these titles (ids as listed on
# 2026-11-13) and that the resolver reported missing while it ANDed every title
# word, stop words included. Every other entry in these tests is invented.
GUIDE_TITLE = "A Guide to Misinformation Detection Data and Evaluation"
SET_SEQUENCE_TITLE = "A Set-Sequence Model for Time Series"
SUBNETWORKS_TITLE = "Finding Stable Subnetworks at Initialization with Dataset Distillation"
STOP_WORD_TITLES = {"2411.05060": GUIDE_TITLE, "2505.11243": SET_SEQUENCE_TITLE, "2503.17905": SUBNETWORKS_TITLE}
ARXIV_DELAY = resolver.BACKEND_DELAY_SECONDS["arxiv"]


class FakeArxiv:
    """arXiv's title search as the resolver sees it; ``events`` logs requests and pauses in order.

    Like the real index it holds no stop words: a ``ti:and``, ``ti:for``,
    ``ti:with`` or ``ti:the`` term matches no title, so a query that ANDs one
    in returns an empty feed. A phrase query matches a title that contains
    the phrase.
    """

    UNINDEXED = frozenset({"and", "for", "with", "the"})

    def __init__(self, titles: dict[str, str], *, refuse_request_numbers: tuple[int, ...] = ()) -> None:
        self.titles = titles
        self.refuse_request_numbers = set(refuse_request_numbers)
        self.events: list[tuple[str, object]] = []

    @property
    def queries(self) -> list[str]:
        return [str(value) for kind, value in self.events if kind == "request"]

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", seconds))

    def fetch(self, url: str) -> bytes:
        parts = urllib.parse.urlsplit(url)
        assert (parts.netloc, parts.path) == ("export.arxiv.org", "/api/query")
        params = urllib.parse.parse_qs(parts.query)
        assert params["max_results"] == ["5"]
        (query,) = params["search_query"]
        self.events.append(("request", query))
        if len(self.queries) in self.refuse_request_numbers:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
        entries = "".join(
            f"<entry><id>http://arxiv.org/abs/{arxiv_id}v1</id><title>{title}</title>"
            "<published>2025-02-03T00:00:00Z</published><author><name>Ada Author</name></author></entry>"
            for arxiv_id, title in self.titles.items()
            if self._matches(query, title)
        )
        return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'.encode()

    def _matches(self, query: str, title: str) -> bool:
        indexed = resolver.normalize(title)
        if query.startswith('ti:"'):
            assert query.endswith('"') and query.count('"') == 2 and "\\" not in query
            return f" {resolver.normalize(query[4:-1])} " in f" {indexed} "
        terms = query.split(" AND ")
        assert all(re.fullmatch(r"ti:[a-z0-9]+", term) for term in terms), query
        words = set(indexed.split()) - self.UNINDEXED
        return all(term[3:] in words for term in terms)

    def paced(self) -> bool:
        """True when a pause of at least the arXiv delay separates every two requests."""
        waited: float | None = None
        for kind, value in self.events:
            if kind == "sleep":
                waited = None if waited is None else waited + float(value)
            elif waited is not None and waited < ARXIV_DELAY:
                return False
            else:
                waited = 0.0
        return True


def test_arxiv_lookup_finds_stop_word_titles_with_one_phrase_query() -> None:
    for arxiv_id, title in STOP_WORD_TITLES.items():
        arxiv = FakeArxiv(STOP_WORD_TITLES)
        hit = resolver.lookup_arxiv(title, fetch=arxiv.fetch, sleep=arxiv.sleep)
        assert hit is not None and hit["match_score"] == 1.0 and hit["title"] == title
        assert hit["arxiv_id"] == f"{arxiv_id}v1" and hit["pdf_url"] == f"https://export.arxiv.org/pdf/{arxiv_id}"
        assert hit["authors"] == ["Ada Author"] and hit["published"] == "2025-02-03T00:00:00Z"
        assert arxiv.events == [("request", f'ti:"{title}"')]  # found at once: no second request, no pause


def test_arxiv_lookup_falls_back_to_content_words_after_a_pause() -> None:
    # The arXiv copy carries one more word, so the phrase is not in its title.
    listed = "Finding Stable Subnetworks at Random Initialization with Dataset Distillation"
    title = SUBNETWORKS_TITLE
    arxiv = FakeArxiv({"0000.00021": listed})
    hit = resolver.lookup_arxiv(title, fetch=arxiv.fetch, sleep=arxiv.sleep)
    assert hit is not None and hit["title"] == listed and hit["arxiv_id"] == "0000.00021v1"
    assert resolver.MATCH_THRESHOLD <= hit["match_score"] < 1.0
    assert arxiv.events == [
        ("request", f'ti:"{title}"'),
        ("sleep", ARXIV_DELAY),
        ("request", "ti:finding AND ti:stable AND ti:subnetworks AND ti:initialization AND ti:dataset AND ti:distillation"),
    ]


def test_arxiv_lookup_never_sends_a_stop_word_term() -> None:
    arxiv = FakeArxiv({})  # nothing listed, so every lookup sends both of its queries
    titles = [*STOP_WORD_TITLES.values(), "The Bitter Lesson for Vision and Language with Agents", "This Is Not the End"]
    for title in titles:
        assert resolver.lookup_arxiv(title, fetch=arxiv.fetch, sleep=arxiv.sleep) is None
    assert len(arxiv.queries) == 2 * len(titles)
    for query in arxiv.queries:
        assert not re.search(r"\bti:(and|for|with|the|this|not)\b", query), query
    assert arxiv.queries[-1] == "ti:end"
    assert arxiv.queries[1] == "ti:guide AND ti:misinformation AND ti:detection AND ti:data AND ti:evaluation"


def test_arxiv_lookup_returns_none_after_both_queries_and_the_pause_between_them() -> None:
    arxiv = FakeArxiv(STOP_WORD_TITLES)
    assert resolver.lookup_arxiv("Sparse Widgets for Vision", fetch=arxiv.fetch, sleep=arxiv.sleep) is None
    assert arxiv.events == [
        ("request", 'ti:"Sparse Widgets for Vision"'),
        ("sleep", ARXIV_DELAY),
        ("request", "ti:sparse AND ti:widgets AND ti:vision"),
    ]
    # A near miss stays a miss: the acceptance rule is still similarity >= MATCH_THRESHOLD.
    near = FakeArxiv({"0000.00031": "A Set-Sequence Model for Multivariate Financial Time Series"})
    assert resolver.lookup_arxiv(SET_SEQUENCE_TITLE, fetch=near.fetch, sleep=near.sleep) is None
    assert len(near.queries) == 2


def test_arxiv_title_queries_phrase_and_content_word_forms() -> None:
    queries = resolver.arxiv_title_queries
    # Double quotes and backslashes would end or escape the phrase; everything else is left to arXiv's analyzer.
    assert queries('The "Best"  \\alpha-Model:\n a (Re)Study?')[0] == 'ti:"The Best alpha-Model: a (Re)Study?"'
    # The 12-word cap counts content words, so stop words no longer use it up.
    long_title = " ".join(f"the word{n:02d} for" for n in range(15))
    assert queries(long_title)[1] == " AND ".join(f"ti:word{n:02d}" for n in range(12))
    # No content word: only the phrase is searched. Nothing to search: no request at all.
    assert queries("To Be or Not to Be") == ['ti:"To Be or Not to Be"']
    assert queries("") == [] and queries(" ?! ") == []
    arxiv = FakeArxiv(STOP_WORD_TITLES)
    assert resolver.lookup_arxiv("To Be or Not to Be", fetch=arxiv.fetch, sleep=arxiv.sleep) is None
    assert resolver.lookup_arxiv(" ?! ", fetch=arxiv.fetch, sleep=arxiv.sleep) is None
    assert arxiv.events == [("request", 'ti:"To Be or Not to Be"')]


def test_resolve_titles_paces_every_arxiv_request() -> None:
    listed = {
        "2411.05060": GUIDE_TITLE,
        "2505.11243": SET_SEQUENCE_TITLE,
        "0000.00021": "Finding Stable Subnetworks at Random Initialization with Dataset Distillation",
    }
    items = [
        {"key": "phrase", "title": GUIDE_TITLE},
        {"key": "words", "title": SUBNETWORKS_TITLE},
        {"key": "none", "title": "Nothing Matches This"},
        {"key": "phrase-2", "title": SET_SEQUENCE_TITLE},
    ]
    arxiv = FakeArxiv(listed)
    results = resolver.resolve_titles(items, fetch=arxiv.fetch, sleep=arxiv.sleep, backends=("arxiv",))
    found = {r["key"]: r["resolution"]["arxiv_id"] for r in results}
    assert found == {"phrase": "2411.05060v1", "words": "0000.00021v1", "none": None, "phrase-2": "2505.11243v1"}
    assert results[0]["resolution"]["open_pdf_url"] == "https://export.arxiv.org/pdf/2411.05060"
    assert all(r["backends_run"] == ["arxiv"] and "error" not in r for r in results)
    # 1 + 2 + 2 + 1 requests. A pause taken on the real clock instead of the
    # injected sleep would show here as two requests in a row.
    assert [kind for kind, _ in arxiv.events] == ["request", "sleep"] * 6
    assert all(value == ARXIV_DELAY for kind, value in arxiv.events if kind == "sleep")
    assert arxiv.paced()

    # A refusal of the second request is retried after the cooldown, and the retry is paced too.
    throttled = FakeArxiv(listed, refuse_request_numbers=(2,))
    results = resolver.resolve_titles(items[1:2], fetch=throttled.fetch, sleep=throttled.sleep, backends=("arxiv",))
    assert results[0]["resolution"]["arxiv_id"] == "0000.00021v1" and "backends_skipped" not in results[0]
    assert [kind for kind, _ in throttled.events] == ["request", "sleep", "request", "sleep"] * 2
    assert throttled.events[3] == ("sleep", resolver.REFUSAL_COOLDOWN_SECONDS[0]) and throttled.paced()


def test_research_dir_skips_a_resolutions_list_written_beside_it(tmp_path: Path) -> None:
    # The discovery skill writes --output into --research-dir, so the second
    # run finds its own resolutions list there and must not read it as research.
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "resolve_paper_titles.py"
    spec = importlib.util.spec_from_file_location("resolve_paper_titles_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    research = tmp_path / "gap-research"
    _write(research / "ws_a.json", {"workshop": "WS A", "schedule_id": "s1",
                                    "papers": [{"title": "An Invented Title", "authors": ["A. Author"]}]})
    _write(research / "title_resolutions.json", [{"key": "WS A:abc", "title": "An Invented Title", "resolution": {}}])
    _write(research / "notes.json", {"workshop": "WS B"})

    items = module.items_from_research_dir(research)
    assert [(item["workshop"], item["title"], item["authors"]) for item in items] == [
        ("WS A", "An Invented Title", ["A. Author"])
    ]


# --------------------------------------------------------------------------- workshop site papers


def test_normalize_link_kinds() -> None:
    assert site.normalize_link("https://arxiv.org/abs/2601.00001v2") == ("https://export.arxiv.org/pdf/2601.00001v2", "arxiv")
    assert site.normalize_link("https://drive.google.com/file/d/FILE123/view?usp=sharing") == (
        "https://drive.google.com/uc?export=download&id=FILE123", "google_drive")
    assert site.normalize_link("https://openreview.net/forum?id=abc") == ("https://openreview.net/pdf?id=abc", "openreview")
    assert site.normalize_link(None) == (None, "none")
    assert site.normalize_link("https://ws.org/papers/2.pdf") == ("https://ws.org/papers/2.pdf", "direct")
    assert site.normalize_link("https://ws.org/papers/on\u2011sensor.pdf") == ("https://ws.org/papers/on%E2%80%91sensor.pdf", "direct")


def _site_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "conf"
    venue = "org/Conf/2099/Workshop/A"
    _write(ws / "storage/workshops/papers" / slugify_venueid(venue) / "receipts/n1.json",
           {"status": "succeeded", "title": "Already On OpenReview", "note_id": "n1", "authors": ["Ann One"]})
    _write(ws / "storage/workshops/download_receipt.json", _download_receipt(per_workshop=[{"venueid": venue, "status": "succeeded"}]))
    _write(ws / "research/ws.json", {
        "workshop": "WS", "schedule_id": "42", "classification": "papers_published_on_site", "source_url": "https://ws.org",
        "papers": [
            {"title": "Direct PDF Paper", "authors": ["B"], "link": "https://ws.org/p1.pdf"},
            {"title": "Private OpenReview Paper", "authors": ["C"], "link": "https://openreview.net/pdf?id=priv"},
            {"title": "HTML Only Paper", "authors": ["D"], "link": "https://ws.org/p3.html"},
            {"title": "No Link But On ArXiv", "authors": ["E"], "link": None},
            {"title": "No Link Anywhere", "authors": ["F"], "link": None},
            {"title": "Already on OpenReview", "authors": ["Ann One"], "link": "https://ws.org/dup.pdf"},
        ],
    })
    _write(ws / "resolutions.json", [
        {"key": "x", "title": "No Link But On ArXiv", "resolution": {"open_pdf_url": "https://arxiv.org/abs/2601.00002"}},
    ])
    return ws


def _site_fetch(calls: list[str]):
    pages = {
        "https://ws.org/p1.pdf": b"%PDF-1.7 direct",
        "https://ws.org/p3.html": b"<html>paper</html>",
        "https://export.arxiv.org/pdf/2601.00002": b"%PDF-1.5 arxiv",
    }

    def fetch(url: str):
        calls.append(url)
        if "openreview" in url:
            raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)
        return 200, "application/pdf", pages[url]

    return fetch


def test_site_papers_uses_curl_for_arxiv_only(tmp_path: Path) -> None:
    ws = _site_workspace(tmp_path)
    seen: list[str] = []

    def arxiv_curl(url: str):
        seen.append(url)
        return 200, "application/pdf", b"%PDF-1.5 curl"

    import conference_pipeline.acquisition.workshop_site_papers as mod

    original = mod.curl_fetch
    mod.curl_fetch = arxiv_curl
    try:
        receipt = mod.run(
            mod.SiteRunOptions(
                workspace=ws, research_dir=ws / "research", resolutions_path=ws / "resolutions.json",
                rate_limit_rps=100.0, arxiv_rate_limit_rps=100.0,
            ),
            fetch=_site_fetch([]),
        )
    finally:
        mod.curl_fetch = original
    # curl is used for the arXiv candidate only; site links keep the normal fetcher
    assert seen == ["https://export.arxiv.org/pdf/2601.00002"]
    assert receipt["per_workshop"][0]["downloaded"] == 2


def test_site_papers_verify_download_register_and_resume(tmp_path: Path) -> None:
    ws = _site_workspace(tmp_path)
    options = site.SiteRunOptions(
        workspace=ws, research_dir=ws / "research", resolutions_path=ws / "resolutions.json",
        rate_limit_rps=100.0, arxiv_via_curl=False,
    )

    calls: list[str] = []
    verified = site.run(site.SiteRunOptions(**{**options.__dict__, "verify_only": True}), fetch=_site_fetch(calls))
    assert verified["status"] == "verified" and verified["totals"]["succeeded"] == 2
    assert not (ws / "storage/workshops/papers" / slugify_venueid("workshop-site/42")).exists()

    receipt = site.run(options, fetch=_site_fetch([]))
    assert receipt["status"] == "succeeded"
    entry = receipt["per_workshop"][0]
    assert entry["downloaded"] == 2 and len(entry["duplicates"]) == 1
    reasons = {u["title"]: u["reason"] for u in entry["unavailable"]}
    assert reasons == {
        "Private OpenReview Paper": "not_public",
        "HTML Only Paper": "response_not_pdf",
        "No Link Anywhere": "no_link",
    }
    paper_receipts = list((ws / "storage/workshops/papers" / slugify_venueid(entry["venueid"]) / "receipts").glob("*.json"))
    kinds = sorted(json.loads(p.read_text())["link_kind"] for p in paper_receipts)
    assert kinds == ["direct", "title_resolution"]

    receipt_path = _write(ws / site.DEFAULT_RECEIPT_RELATIVE, receipt)
    sources.register_source(workspace=ws, source_id=site.SOURCE_ID, receipt_path=receipt_path)
    registered, blockers = sources.load_registered_sources(ws)
    assert blockers == [] and site.other_source_papers(registered) == {"42": {"workshop_site": 2}}

    calls = []
    again = site.run(options, fetch=_site_fetch(calls))
    assert again["totals"]["resume_skipped"] == 2 and "https://ws.org/p1.pdf" not in calls


def test_site_papers_keep_the_submission_system_venue_and_note_id(tmp_path: Path) -> None:
    """Metadata-only venues: a PDF found elsewhere joins back to its listing."""
    ws = _site_workspace(tmp_path)
    for path in (ws / "research").glob("*.json"):
        path.unlink()
    _write(ws / "research/meta.json", {
        "workshop": "Meta", "schedule_id": "Meta", "venueid": "Z/Workshop/Meta",
        "papers": [{"title": "No Link But On ArXiv", "authors": ["E"], "link": None, "note_id": "abc123"}],
    })
    receipt = site.run(
        site.SiteRunOptions(
            workspace=ws, research_dir=ws / "research", resolutions_path=ws / "resolutions.json",
            rate_limit_rps=100.0, arxiv_via_curl=False,
        ),
        fetch=_site_fetch([]),
    )
    entry = receipt["per_workshop"][0]
    assert entry["venueid"] == "Z/Workshop/Meta" and entry["downloaded"] == 1
    # Files stay in the site-source directory; the receipt names the real venue and note.
    receipts = list((ws / "storage/workshops/papers" / slugify_venueid("workshop-site/Meta") / "receipts").glob("*.json"))
    paper = json.loads(receipts[0].read_text())
    assert (paper["venueid"], paper["note_id"]) == ("Z/Workshop/Meta", "abc123")
    assert paper["site_paper_id"].startswith("site_")


def test_known_papers_resolve_every_output_dir_form(tmp_path: Path) -> None:
    """Absolute, workspace-relative and repo-root-relative output_dir all find receipts."""
    from conference_pipeline.acquisition import cvf_openaccess_workshops as shared

    ws = tmp_path / "conferences" / "zeta"
    _write(ws / "storage/workshops/papers" / slugify_venueid("Z/Workshop/A") / "receipts/n1.json",
           {"note_id": "n1", "title": "Known Paper", "authors": ["Ann One"], "status": "succeeded"})
    for output_dir in (
        str(ws / "storage/workshops/papers"),
        "storage/workshops/papers",
        "conferences/zeta/storage/workshops/papers",
    ):
        receipt = {"output_dir": output_dir, "per_workshop": [{"venueid": "Z/Workshop/A"}]}
        known = shared.load_known_papers(ws, [receipt])
        assert known.find("Known Paper", ("Ann One",)) is not None, output_dir
