"""Identity keys and the merge (v6.0-Public-Release Phase 5, T2)."""

from __future__ import annotations

import pytest

from conference_pipeline.indexing.paper_identity import (
    IdentityEntry,
    PaperIdentityError,
    identity_key,
    is_stable,
    key_from_source,
    merge_identities,
)


@pytest.mark.parametrize(
    "kind,raw,key",
    [
        ("openreview", " AbCdEf123 ", "openreview:AbCdEf123"),
        ("arxiv", "2501.00001v3", "arxiv:2501.00001"),
        ("arxiv", "arXiv:2501.00001", "arxiv:2501.00001"),
        ("arxiv", "2501.00001", "arxiv:2501.00001"),
        ("site", "site_abc", "site:site_abc"),
        ("proceedings", "abc123", "proceedings:abc123"),
    ],
)
def test_identity_keys(kind, raw, key):
    assert identity_key(kind, raw) == key


@pytest.mark.parametrize("kind,raw", [("doi", "10.1/x"), ("openreview", " "), ("arxiv", "v2")])
def test_bad_identities_refuse(kind, raw):
    with pytest.raises(PaperIdentityError):
        identity_key(kind, raw)


def test_source_field_keys():
    assert key_from_source("arxiv:2501.00001v1") == "arxiv:2501.00001"
    assert key_from_source("biorxiv:10.1101/x") is None
    assert key_from_source("") is None


def test_stability():
    assert is_stable("openreview:x") and is_stable("arxiv:1") and is_stable("proceedings:p")
    assert not is_stable("site:s")


def test_merge_is_transitive_through_shared_keys():
    entries = [
        IdentityEntry(keys=("openreview:A",)),
        IdentityEntry(keys=("openreview:A", "arxiv:1")),
        IdentityEntry(keys=("arxiv:1",)),
        IdentityEntry(keys=("openreview:B",)),
    ]
    keys = merge_identities(entries).paper_keys
    assert keys[0] == keys[1] == keys[2] == "openreview:A"  # the OpenReview id is preferred (IF9)
    assert keys[3] == "openreview:B"


def test_ambiguous_title_leaves_a_site_record_alone():
    title = "A Title Long Enough To Be A Key"
    entries = [
        IdentityEntry(keys=("openreview:A",), title=title),
        IdentityEntry(keys=("openreview:B",), title=title),
        IdentityEntry(keys=("site:s",), title=title),
    ]
    result = merge_identities(entries)
    assert result.paper_keys[2] == "site:s"
    assert result.title_joins == ()
    assert result.title_collisions[0]["paper_keys"] == ["openreview:A", "openreview:B"]


def test_site_records_with_one_title_and_no_stable_paper_are_one_paper():
    title = "Another Title Long Enough For A Key"
    entries = [IdentityEntry(keys=("site:a",), title=title), IdentityEntry(keys=("site:b",), title=title)]
    result = merge_identities(entries)
    assert result.paper_keys[0] == result.paper_keys[1]


def test_short_titles_never_join():
    entries = [IdentityEntry(keys=("openreview:A",), title="Short"), IdentityEntry(keys=("site:s",), title="short")]
    assert merge_identities(entries).paper_keys[1] == "site:s"


def test_r1_f1_a_site_record_never_joins_two_stable_papers():
    """F1: a site record carrying two titles, each matching a different stable paper, joins neither."""
    x, y = "A First Title Long Enough For A Key", "A Second Title Long Enough For A Key"
    entries = [
        IdentityEntry(keys=("openreview:A",), title=x),
        IdentityEntry(keys=("openreview:B",), title=y),
        IdentityEntry(keys=("site:S",), title=x),
        IdentityEntry(keys=("site:S",), title=y),
    ]
    result = merge_identities(entries)
    assert result.paper_keys[0] == "openreview:A"
    assert result.paper_keys[1] == "openreview:B"
    assert result.paper_keys[2] == result.paper_keys[3] == "site:S"
    assert result.title_joins == ()
