"""Tests for conference-general program item-page classification and coverage gates."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.program_coverage import (  # noqa: E402
    DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
    VirtualItemPage,
    classify_item_page_title_family,
    classify_virtual_item_page,
    media_family_coverage,
    media_family_coverage_blockers,
)


class TestClassifyVirtualItemPage:
    def test_typed_poster_page(self) -> None:
        assert classify_virtual_item_page(
            "https://icml.cc/virtual/2026/poster/64074", year=2026
        ) == VirtualItemPage(family="poster", item_id="64074")

    def test_typed_tutorial_page_is_item_by_default(self) -> None:
        assert classify_virtual_item_page(
            "https://icml.cc/virtual/2026/tutorial/75374", year=2026
        ) == VirtualItemPage(family="tutorial", item_id="75374")

    def test_typed_invited_talk_page(self) -> None:
        assert classify_virtual_item_page(
            "https://icml.cc/virtual/2026/invited-talk/67258", year=2026
        ) == VirtualItemPage(family="invited-talk", item_id="67258")

    def test_unknown_future_family_is_included_by_default(self) -> None:
        item = classify_virtual_item_page(
            "https://icml.cc/virtual/2026/robot-demo/123", year=2026
        )
        assert item is not None and item.family == "robot-demo"

    def test_bare_id_item_page(self) -> None:
        # ICML 2026 tutorials index links items as /virtual/2026/<digits>.
        assert classify_virtual_item_page(
            "https://icml.cc/virtual/2026/75365", year=2026
        ) == VirtualItemPage(family=None, item_id="75365")

    def test_structural_pages_are_not_items(self) -> None:
        for url in (
            "https://icml.cc/virtual/2026/events/tutorial",
            "https://icml.cc/virtual/2026/eventlistwithbios/invited%20talk",
            "https://icml.cc/virtual/2026/calendar",
            "https://icml.cc/virtual/2026/day/2026-07-14",
            "https://icml.cc/virtual/2026/papers.html",
        ):
            assert classify_virtual_item_page(url, year=2026) is None, url

    def test_wrong_year_is_not_matched(self) -> None:
        assert (
            classify_virtual_item_page(
                "https://icml.cc/virtual/2025/poster/64074", year=2026
            )
            is None
        )

    def test_query_and_fragment_do_not_break_classification(self) -> None:
        assert classify_virtual_item_page(
            "https://icml.cc/virtual/2026/tutorial/75374?presentation=1#top",
            year=2026,
        ) == VirtualItemPage(family="tutorial", item_id="75374")


class TestMediaFamilyCoverage:
    def _targets(self) -> list[dict[str, str]]:
        return [
            {
                "target_url": "https://icml.cc/virtual/2026/poster/1",
                "source_family_id": "icml-calendar",
            },
            {
                "target_url": "https://icml.cc/virtual/2026/oral/2",
                "source_family_id": "icml-calendar",
            },
            {
                "target_url": "https://icml.cc/virtual/2026/invited-talk/3",
                "source_family_id": "icml-calendar",
            },
            # Bare-ID tutorial links discovered from the tutorials index page.
            {
                "target_url": "https://icml.cc/virtual/2026/75365",
                "source_family_id": "tutorials",
            },
            {
                "target_url": "https://icml.cc/virtual/2026/75366",
                "source_family_id": "tutorials",
            },
            # Non-item link noise.
            {
                "target_url": "https://icml.cc/virtual/2026/events/workshop",
                "source_family_id": "workshops",
            },
        ]

    def test_counts_typed_and_bare_id_items(self) -> None:
        coverage = media_family_coverage(
            self._targets(),
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
        )
        families = coverage["families"]
        assert families["orals"]["item_page_count"] == 1
        assert families["invited_talks"]["item_page_count"] == 1
        assert families["tutorials"]["item_page_count"] == 2
        assert families["workshops"]["item_page_count"] == 0
        assert coverage["zero_item_families"] == ["expo", "workshops"]

    def test_blockers_fail_closed_for_zero_item_families(self) -> None:
        coverage = media_family_coverage(
            self._targets(),
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
        )
        blockers = media_family_coverage_blockers(coverage)
        assert {b["family_id"] for b in blockers} == {"expo", "workshops"}
        assert all(
            b["code"] == "program_family_media_items_missing" for b in blockers
        )

    def test_explicit_exclusion_suppresses_blocker_but_not_coverage(self) -> None:
        coverage = media_family_coverage(
            self._targets(),
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
        )
        blockers = media_family_coverage_blockers(
            coverage, excluded_families=["expo", "workshops"]
        )
        assert blockers == []
        assert coverage["zero_item_families"] == ["expo", "workshops"]

    def test_icml2026_regression_tutorials_never_silently_zero(self) -> None:
        """Pin the exact ICML 2026 failure: poster/oral-only discovery must block."""

        poster_only = [
            {
                "target_url": f"https://icml.cc/virtual/2026/poster/{i}",
                "source_family_id": "icml-calendar",
            }
            for i in range(5)
        ]
        coverage = media_family_coverage(
            poster_only,
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
        )
        blockers = media_family_coverage_blockers(coverage)
        assert "tutorials" in {b["family_id"] for b in blockers}
        assert "invited_talks" in {b["family_id"] for b in blockers}


class TestClassifyItemPageTitleFamily:
    def test_tutorial_title(self) -> None:
        html = "<html><head><title>ICML Tutorial Proving Theorems with Lean</title></head></html>"
        assert classify_item_page_title_family(html) == "tutorials"

    def test_expo_workshop_title_beats_plain_workshop_slug(self) -> None:
        html = "<title>ICML Expo Workshop From Digital Agents to Physical Intelligence</title>"
        assert classify_item_page_title_family(html) == "expo"

    def test_expo_demonstration_title(self) -> None:
        html = "<title>ICML Expo Demonstration Fara1.5: Computer Use Agents</title>"
        assert classify_item_page_title_family(html) == "expo"

    def test_expo_talk_panel_title(self) -> None:
        html = "<title>ICML Expo Talk Panel Scaling Deep Learning in Financial Markets</title>"
        assert classify_item_page_title_family(html) == "expo"

    def test_plain_workshop_title(self) -> None:
        html = "<title>ICML Workshop on AI for Science</title>"
        assert classify_item_page_title_family(html) == "workshops"

    def test_invited_talk_title(self) -> None:
        html = "<title>ICML Invited Talk Towards AI Agents In the Real World</title>"
        assert classify_item_page_title_family(html) == "invited_talks"

    def test_no_title_returns_none(self) -> None:
        assert classify_item_page_title_family("<html><body>no title here</body></html>") is None

    def test_unmatched_title_returns_none(self) -> None:
        html = "<title>ICML Poster Some Paper Title</title>"
        assert classify_item_page_title_family(html) is None


class TestMediaFamilyCoverageWithResolver:
    def test_icml2026_calendar_linked_bare_ids_resolve_via_title(self) -> None:
        """Pin the exact bug: bare-ID tutorials/expo linked from icml-calendar,
        not from their own family's index page, so source_family_id alone
        (icml-calendar) cannot attribute them without a content-based resolver."""

        targets = [
            {"target_url": "https://icml.cc/virtual/2026/75365", "source_family_id": "icml-calendar"},
            {"target_url": "https://icml.cc/virtual/2026/75708", "source_family_id": "icml-calendar"},
            {"target_url": "https://icml.cc/virtual/2026/oral/2", "source_family_id": "icml-calendar"},
        ]
        titles = {
            "https://icml.cc/virtual/2026/75365": "tutorials",
            "https://icml.cc/virtual/2026/75708": "expo",
        }

        coverage_without_resolver = media_family_coverage(
            targets, year=2026, declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS
        )
        assert coverage_without_resolver["families"]["tutorials"]["item_page_count"] == 0
        assert coverage_without_resolver["families"]["expo"]["item_page_count"] == 0
        assert coverage_without_resolver["unattributed_bare_id_count"] == 2

        coverage_with_resolver = media_family_coverage(
            targets,
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
            resolve_bare_id_family=lambda url: titles.get(url),
        )
        assert coverage_with_resolver["families"]["tutorials"]["item_page_count"] == 1
        assert coverage_with_resolver["families"]["expo"]["item_page_count"] == 1
        assert coverage_with_resolver["unattributed_bare_id_count"] == 0
        blockers = media_family_coverage_blockers(coverage_with_resolver)
        assert {b["family_id"] for b in blockers} == {"invited_talks", "workshops"}

    def test_resolver_returning_none_falls_back_to_source_family_id(self) -> None:
        targets = [
            {"target_url": "https://icml.cc/virtual/2026/99999", "source_family_id": "tutorials"},
        ]
        coverage = media_family_coverage(
            targets,
            year=2026,
            declared_families=DEFAULT_MEDIA_BEARING_FAMILY_SLUGS,
            resolve_bare_id_family=lambda url: None,
        )
        assert coverage["families"]["tutorials"]["item_page_count"] == 1


# ---------------------------------------------------------------------------
# Item kind from the page title (bare-ID pages carry no kind in their URL)
# ---------------------------------------------------------------------------


def test_classify_item_page_title_kind_resolves_bare_id_pages() -> None:
    from conference_pipeline.program_coverage import classify_item_page_title_kind, item_page_title

    html = "<html><head><title>ICML  Tutorial:\n Math Copilots</title></head></html>"
    assert item_page_title(html) == "ICML Tutorial: Math Copilots"
    assert classify_item_page_title_kind(html) == "tutorial"
    assert classify_item_page_title_kind("<title>ICML Expo Talk Panel: Frontier Labs</title>") == "expo"
    assert classify_item_page_title_kind("<title>ICML Expo Workshop: Tooling</title>") == "expo"
    assert classify_item_page_title_kind("<title>Invited Talk: Open Science</title>") == "invited-talk"
    assert classify_item_page_title_kind("<title>Oral Session 1A: Reasoning</title>") == "session"
    assert classify_item_page_title_kind("<title>ICML Town Hall</title>") == "town-hall"
    assert classify_item_page_title_kind("<title>Just a page</title>") is None
    assert classify_item_page_title_kind("<html>no title</html>") is None
