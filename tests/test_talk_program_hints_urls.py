"""Event-page URL parsing for talk program hints: every typed slug and bare IDs."""

from __future__ import annotations

from conference_pipeline.talk_program_hints import parse_event_url


def test_parse_event_url_accepts_hyphenated_and_bare_id_pages() -> None:
    assert parse_event_url("https://icml.cc/virtual/2026/invited-talk/67258") == ("invited", "67258")
    assert parse_event_url("https://icml.cc/virtual/2026/tutorial/45001") == ("tutorial", "45001")
    assert parse_event_url("https://icml.cc/virtual/2026/affinity-event/300") == ("affinity", "300")
    assert parse_event_url("https://icml.cc/virtual/2026/town-hall/1") == ("town-hall", "1")
    assert parse_event_url("https://icml.cc/virtual/2026/expo/22") == ("expo", "22")
    assert parse_event_url("https://cvpr.thecvf.com/virtual/2026/workshops/123") == ("workshop", "123")
    # Bare-ID pages carry no category in the URL; the caller resolves it from the page title.
    assert parse_event_url("https://icml.cc/virtual/2026/75372") == ("item", "2026-75372")
    assert parse_event_url("https://icml.cc/virtual/2026/75372?x=1") == ("item", "2026-75372")
    assert parse_event_url("https://icml.cc/virtual/2026/calendar") == (None, None)
    assert parse_event_url(None) == (None, None)
