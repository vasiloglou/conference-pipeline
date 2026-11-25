"""Contracts for CVPR's r5 category-complete, caption-only event-media lane."""

from __future__ import annotations

import copy

import pytest

from conference_pipeline.acquisition.cvpr_event_media import (
    EVENT_CATEGORY_COUNTS,
    apply_caption_outcomes,
    apply_event_page_outcome,
    apply_event_page_outcomes,
    apply_r4_poster_outcomes,
    build_event_media_inventory,
    canonical_event_media_inventory_digest,
    event_page_youtube_candidates,
    merge_caption_candidates,
    resume_event_media_inventory,
    validate_event_media_inventory,
    validate_r4_poster_reuse,
)
from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    parse_virtual_site_boundary,
)


FIXTURE_VIRTUAL_URL = "https://cvpr.thecvf.com/virtual/2026"
FIXTURE_BOUNDARY = parse_virtual_site_boundary(FIXTURE_VIRTUAL_URL)


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = (
        ("poster", "Poster", EVENT_CATEGORY_COUNTS["poster"]),
        ("invited-talk", "Keynote", EVENT_CATEGORY_COUNTS["keynote"]),
        ("tutorial", "Tutorial", EVENT_CATEGORY_COUNTS["tutorial"]),
        ("workshop", "Workshop", EVENT_CATEGORY_COUNTS["workshop"]),
        ("session", "Oral Session", EVENT_CATEGORY_COUNTS["oral"]),
        ("session", "Poster Session", EVENT_CATEGORY_COUNTS["session"]),
        ("demonstration", "Demo", 5),
        ("events", "Event", 7),
        ("meeting", "Meeting", 1),
        ("mentorship", "Mentorship", 2),
        ("reception", "Reception", 1),
        ("remarks", "Remarks", 1),
    )
    number = 0
    for route_family, title_prefix, count in specs:
        for _ in range(count):
            number += 1
            rows.append(
                {
                    "source_record_id": f"cvpr2026:source:program-schedule:{number:04d}",
                    "family_id": "program-schedule",
                    "title": f"{title_prefix} {number}",
                    "aliases": [f"{FIXTURE_VIRTUAL_URL}/{route_family}/{number}"],
                }
            )
    return rows


def test_event_inventory_has_the_fixed_category_census_and_is_deterministic() -> None:
    first = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    second = build_event_media_inventory(
        source_records=list(reversed(_records())), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )

    assert first["counts"]["event_categories"] == {
        "poster": 5_023,
        "keynote": 3,
        "tutorial": 19,
        "workshop": 153,
        "oral": 24,
        "session": 19,
        "main_event": 17,
    }
    assert first["counts"]["event_route_count"] == 5_258
    assert first["counts"]["oral_session_route_count"] == 24
    assert first["event_media_inventory_sha256"] == second["event_media_inventory_sha256"]
    assert canonical_event_media_inventory_digest(first) == first["event_media_inventory_sha256"]
    assert all(row["audio_downloaded"] is False for row in first["events"])
    assert all(row["video_downloaded"] is False for row in first["events"])
    validate_event_media_inventory(first, boundary=FIXTURE_BOUNDARY)


def test_event_inventory_rejects_duplicate_routes_and_unclassifiable_sessions() -> None:
    duplicate = _records()
    duplicate[1]["aliases"] = list(duplicate[0]["aliases"])
    with pytest.raises(ValueError, match="duplicate"):
        build_event_media_inventory(
            source_records=duplicate,
            source_inventory_sha256="a" * 64,
            boundary=FIXTURE_BOUNDARY,
        )

    missing_title = _records()
    next(row for row in reversed(missing_title) if "/session/" in row["aliases"][0])["title"] = ""
    with pytest.raises(ValueError, match="session"):
        build_event_media_inventory(
            source_records=missing_title,
            source_inventory_sha256="a" * 64,
            boundary=FIXTURE_BOUNDARY,
        )


def test_coverage_claim_requires_one_terminal_outcome_per_route() -> None:
    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    with pytest.raises(ValueError, match="not terminal"):
        validate_event_media_inventory(inventory, boundary=FIXTURE_BOUNDARY, require_terminal=True)

    inventory = apply_event_page_outcomes(
        inventory,
        boundary=FIXTURE_BOUNDARY,
        outcomes=[
            {
                "source_url": event["source_url"],
                "status": "direct_youtube_recovered" if index == 0 else "no_admissible_official_youtube_link",
                "video_ids": ["dQw4w9WgXcQ"] if index == 0 else [],
            }
            for index, event in enumerate(inventory["events"])
        ],
    )

    validate_event_media_inventory(inventory, boundary=FIXTURE_BOUNDARY, require_terminal=True)
    assert inventory["coverage"]["keynote"]["status"] == "complete"
    assert inventory["coverage"]["tutorial"]["status"] == "complete"
    assert inventory["caption_candidates"] == [
        {
            "audio_downloaded": False,
            "caption": None,
            "caption_status": "pending",
            "provenance": ["r5_official_event_page"],
            "source_event_urls": [inventory["events"][0]["source_url"]],
            "video_downloaded": False,
            "video_id": "dQw4w9WgXcQ",
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
    ]


def test_caption_candidate_merge_deduplicates_r4_and_r5_video_ids() -> None:
    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    first_url = inventory["events"][0]["source_url"]
    inventory = apply_event_page_outcome(
        inventory,
        boundary=FIXTURE_BOUNDARY,
        source_url=first_url,
        status="direct_youtube_recovered",
        video_ids=["dQw4w9WgXcQ"],
    )
    merged = merge_caption_candidates(
        inventory,
        boundary=FIXTURE_BOUNDARY,
        r4_media_records=[
            {
                "video_id": "dQw4w9WgXcQ",
                "parent_request_key": "r4-poster",
                "media_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            }
        ],
    )

    assert len(merged["caption_candidates"]) == 1
    row = merged["caption_candidates"][0]
    assert row["provenance"] == ["r4_registration_media", "r5_official_event_page"]
    assert row["r4_parent_request_keys"] == ["r4-poster"]
    assert row["audio_downloaded"] is False
    assert row["video_downloaded"] is False

    invalid = copy.deepcopy(merged)
    invalid["caption_candidates"][0]["audio_downloaded"] = True
    invalid["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(invalid)
    with pytest.raises(ValueError, match="media"):
        validate_event_media_inventory(invalid, boundary=FIXTURE_BOUNDARY)


def test_r4_poster_reuse_is_exact_and_leaves_only_the_omitted_pages_pending() -> None:
    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    posters = [event for event in inventory["events"] if event["event_category"] == "poster"]
    requests = []
    for index, event in enumerate(posters[:2_565]):
        requests.append(
            {
                "request_key": f"r4-{index}",
                "source_url": event["source_url"],
                "result_status": "direct_youtube_recovered" if index == 0 else "no_admissible_official_youtube_link",
                "auth_status": "authenticated",
                "http_status": 200,
            }
        )
    reused = apply_r4_poster_outcomes(
        inventory,
        boundary=FIXTURE_BOUNDARY,
        r4_requests=requests,
        r4_media_records=[{"parent_request_key": "r4-0", "video_id": "dQw4w9WgXcQ"}],
        r4_overlay_sha256="b" * 64,
    )

    assert reused["counts"]["r4_reused_poster_route_count"] == 2_565
    assert reused["coverage"]["poster"]["terminal_route_count"] == 2_565
    assert reused["coverage"]["poster"]["status"] == "incomplete"
    assert len([row for row in reused["events"] if row["page_status"] == "pending"]) == 2_693
    validate_r4_poster_reuse(
        reused,
        boundary=FIXTURE_BOUNDARY,
        r4_requests=requests,
        r4_media_records=[{"parent_request_key": "r4-0", "video_id": "dQw4w9WgXcQ"}],
    )

    substituted = copy.deepcopy(reused)
    next(
        row
        for row in substituted["events"]
        if row["outcome_origin"] == "r4_registration_media" and not row["video_ids"]
    )["page_status"] = "official_video_not_published"
    substituted["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(substituted)
    validate_event_media_inventory(substituted, boundary=FIXTURE_BOUNDARY)
    with pytest.raises(ValueError, match="immutable overlay"):
        validate_r4_poster_reuse(
            substituted,
            boundary=FIXTURE_BOUNDARY,
            r4_requests=requests,
            r4_media_records=[{"parent_request_key": "r4-0", "video_id": "dQw4w9WgXcQ"}],
        )


def test_available_caption_must_name_its_exact_candidate_video_id() -> None:
    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    inventory = apply_event_page_outcome(
        inventory,
        boundary=FIXTURE_BOUNDARY,
        source_url=inventory["events"][0]["source_url"],
        status="direct_youtube_recovered",
        video_ids=["dQw4w9WgXcQ"],
    )
    inventory["caption_candidates"][0]["caption_status"] = "youtube_caption_available"
    inventory["caption_candidates"][0]["caption"] = {
        "local_path": "storage/download-corpus/event-media/captions/ZZZZZZZZZZZ.en.vtt",
        "mime": "text/vtt",
        "byte_count": 8,
        "sha256": "c" * 64,
        "transcript_source": "youtube_caption_auto",
        "language": "en",
    }
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)

    with pytest.raises(ValueError, match="available caption"):
        validate_event_media_inventory(inventory, boundary=FIXTURE_BOUNDARY)


@pytest.mark.parametrize(
    "video_id",
    ("AbcToken123", "KeyAbc12345", "secret12345", "authABC1234", "bearer12345"),
)
def test_event_page_parser_extracts_valid_marker_substring_youtube_ids(video_id: str) -> None:
    video_ids, unsupported = event_page_youtube_candidates(
        page_html=f'<iframe src="https://www.youtube.com/embed/{video_id}"></iframe>',
        parent_url=f"{FIXTURE_VIRTUAL_URL}/poster/1",
        boundary=FIXTURE_BOUNDARY,
    )

    assert video_ids == [video_id]
    assert unsupported is False


def test_event_media_resume_reuses_only_compatible_r5_page_and_caption_outcomes() -> None:
    fresh = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    posters = [event for event in fresh["events"] if event["event_category"] == "poster"]
    fresh = apply_r4_poster_outcomes(
        fresh,
        boundary=FIXTURE_BOUNDARY,
        r4_requests=[
            {
                "request_key": f"r4-{index}",
                "source_url": event["source_url"],
                "result_status": "no_admissible_official_youtube_link",
                "auth_status": "authenticated",
                "http_status": 200,
            }
            for index, event in enumerate(posters[:2_565])
        ],
        r4_media_records=[],
        r4_overlay_sha256="b" * 64,
    )
    pending_url = next(row["source_url"] for row in fresh["events"] if row["page_status"] == "pending")
    prior = apply_event_page_outcome(
        fresh,
        boundary=FIXTURE_BOUNDARY,
        source_url=pending_url,
        status="direct_youtube_recovered",
        video_ids=["dQw4w9WgXcQ"],
        auth_status="authenticated",
        http_status=200,
    )
    prior = apply_caption_outcomes(
        prior,
        boundary=FIXTURE_BOUNDARY,
        outcomes=[
            {
                "video_id": "dQw4w9WgXcQ",
                "caption_status": "youtube_caption_unavailable",
                "caption": None,
            }
        ],
    )

    resumed = resume_event_media_inventory(
        fresh_inventory=fresh,
        prior_inventory=prior,
        reusable_caption_outcomes=[
            {
                "video_id": "dQw4w9WgXcQ",
                "caption_status": "youtube_caption_unavailable",
                "caption": None,
            }
        ],
        boundary=FIXTURE_BOUNDARY,
    )

    restored = next(row for row in resumed["events"] if row["source_url"] == pending_url)
    assert restored["page_status"] == "direct_youtube_recovered"
    assert resumed["caption_candidates"][0]["caption_status"] == "youtube_caption_unavailable"
    assert resumed["counts"]["r4_reused_poster_route_count"] == 2_565

    incompatible = copy.deepcopy(prior)
    incompatible["source_lineage"]["r4_registration_media_overlay_sha256"] = "c" * 64
    incompatible["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(incompatible)
    with pytest.raises(ValueError, match="lineage"):
        resume_event_media_inventory(
            fresh_inventory=fresh,
            prior_inventory=incompatible,
            reusable_caption_outcomes=[],
            boundary=FIXTURE_BOUNDARY,
        )


@pytest.mark.parametrize(
    "virtual_url",
    (
        "http://cvpr.thecvf.com/virtual/2026",
        "https://user:password@cvpr.thecvf.com/virtual/2026",
        "https://cvpr.thecvf.com:443/virtual/2026",
        "https://cvpr.thecvf.com/virtual/2026?query=1",
        "https://cvpr.thecvf.com/virtual/2026#fragment",
        "https://cvpr.thecvf.com/virtual/%2F2026",
        "https://cvpr.thecvf.com/virtual/../2026",
        "https://cvpr.thecvf.com",
    ),
)
def test_virtual_site_boundary_rejects_malformed_config(virtual_url: str) -> None:
    with pytest.raises(ConferenceIdentityError):
        parse_virtual_site_boundary(virtual_url)


def test_virtual_site_boundary_is_explicit_and_preserves_r5_canonical_digest() -> None:
    assert FIXTURE_BOUNDARY.origin == "https://cvpr.thecvf.com"
    assert FIXTURE_BOUNDARY.root_segments == ("virtual", "2026")

    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    expected_digest = inventory["event_media_inventory_sha256"]
    assert expected_digest == "1d51faa5fca51ee5d4336b192a812db4bd6e6ac07e349f9fba7d1392e2689fd7"
    assert canonical_event_media_inventory_digest(inventory) == expected_digest
    with pytest.raises(TypeError):
        validate_event_media_inventory(inventory)
    validate_event_media_inventory(inventory, boundary=FIXTURE_BOUNDARY)


@pytest.mark.parametrize(
    "unsafe_url",
    (
        "http://cvpr.thecvf.com/virtual/2026/poster/1",
        "https://foreign.example/virtual/2026/poster/1",
        "https://cvpr.thecvf.com/other/2026/poster/1",
        "https://cvpr.thecvf.com/virtual/2025/poster/1",
        "https://user@cvpr.thecvf.com/virtual/2026/poster/1",
        "https://cvpr.thecvf.com/virtual/2026/poster/1?token=fixture",
        "https://cvpr.thecvf.com/virtual/2026/poster/1#fragment",
        "https://cvpr.thecvf.com/virtual/2026/poster/%2F1",
        "https://cvpr.thecvf.com/virtual/2026/poster/../1",
    ),
)
def test_event_inventory_rejects_routes_outside_configured_boundary(unsafe_url: str) -> None:
    inventory = build_event_media_inventory(
        source_records=_records(), source_inventory_sha256="a" * 64, boundary=FIXTURE_BOUNDARY
    )
    invalid = copy.deepcopy(inventory)
    invalid["events"][0]["source_url"] = unsafe_url
    invalid["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(invalid)

    with pytest.raises(ValueError, match="provenance"):
        validate_event_media_inventory(invalid, boundary=FIXTURE_BOUNDARY)
