"""Stable paper identity: identity keys and the merge of records into papers.

v6.0-Public-Release Phase 5 (spec ``notes/phase-5-spec.md`` section 6.4, D5). A record
carries one or more identity keys:

- ``openreview:<forum id>``, ``arxiv:<id without version>``, ``proceedings:<id>`` are
  *stable*: they name one paper wherever it appears;
- ``site:<id>`` (a workshop site's or a schedule's own id) is not.

Records that share any key are one paper. A record whose keys are all non-stable may also
join a paper by normalized title (at least 20 characters); two records that each carry a
stable key never merge on title alone, and a shared title between two stable papers is
reported as a collision instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from conference_pipeline.acquisition.proceedings_paper_sources import normalize_title

STABLE_KINDS = frozenset({"openreview", "arxiv", "proceedings"})
ID_KINDS = STABLE_KINDS | {"site"}
MIN_TITLE_KEY_CHARS = 20
_ARXIV_VERSION = re.compile(r"v\d+$")
_ARXIV_PREFIX = re.compile(r"^(?:arxiv:|arXiv:)", re.IGNORECASE)


class PaperIdentityError(ValueError):
    """An identity of an unknown kind, or an empty id."""


def identity_key(kind: str, raw: str) -> str:
    """The canonical key for one id, e.g. ``identity_key("arxiv", "2501.00001v2")``."""
    if kind not in ID_KINDS:
        raise PaperIdentityError(f"unknown identity kind: {kind}")
    value = str(raw or "").strip()
    if kind == "arxiv":
        value = _ARXIV_VERSION.sub("", _ARXIV_PREFIX.sub("", value))
    if not value:
        raise PaperIdentityError(f"empty {kind} id")
    return f"{kind}:{value}"


def key_from_source(source: str) -> str | None:
    """A key from a ``<kind>:<id>`` source field (``arxiv:2501.00001v1``), if it names one."""
    kind, sep, raw = str(source or "").partition(":")
    if not sep or kind not in ID_KINDS or not raw.strip():
        return None
    return identity_key(kind, raw)


def is_stable(key: str) -> bool:
    return key.split(":", 1)[0] in STABLE_KINDS


def title_key(title: str | None) -> str | None:
    normalized = normalize_title(title)
    return normalized if len(normalized) >= MIN_TITLE_KEY_CHARS else None


@dataclass(frozen=True)
class IdentityEntry:
    keys: tuple[str, ...]
    title: str | None = None


@dataclass(frozen=True)
class MergeResult:
    paper_keys: tuple[str, ...]  # per entry, in input order
    title_joins: tuple[Mapping[str, str], ...] = ()
    title_collisions: tuple[Mapping[str, object], ...] = field(default_factory=tuple)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _groups(uf: _UnionFind, size: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index in range(size):
        groups.setdefault(uf.find(index), []).append(index)
    return groups


# A paper's key prefers the id a workspace's own corpus uses (OpenReview), then
# proceedings, then arXiv, then a site id; the smallest within a kind (IF9).
KEY_PREFERENCE = ("openreview", "proceedings", "arxiv", "site")


def _group_key(entries: Sequence[IdentityEntry], members: Iterable[int]) -> str:
    keys = {key for index in members for key in entries[index].keys}
    return min(keys, key=lambda key: (KEY_PREFERENCE.index(key.split(":", 1)[0]), key))


def merge_identities(entries: Sequence[IdentityEntry]) -> MergeResult:
    """Assign one ``paper_key`` per entry: shared keys first, then site-only title joins."""
    for entry in entries:
        if not entry.keys:
            raise PaperIdentityError("an identity entry needs at least one key")
    uf = _UnionFind(len(entries))
    first_by_key: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for key in entry.keys:
            if key in first_by_key:
                uf.union(index, first_by_key[key])
            else:
                first_by_key[key] = index

    # Title evidence, per group: stable groups by title, and each site-only group's titles.
    groups = _groups(uf, len(entries))
    stable_by_title: dict[str, list[int]] = {}
    site_titles: dict[int, list[str]] = {}
    for root, members in sorted(groups.items()):
        has_stable = any(is_stable(key) for index in members for key in entries[index].keys)
        titles = sorted({t for index in members if (t := title_key(entries[index].title))})
        if has_stable:
            for title in titles:
                stable_by_title.setdefault(title, []).append(root)
        elif titles:
            site_titles[root] = titles

    joins: list[dict[str, str]] = []
    unmatched: dict[str, list[int]] = {}
    for root, titles in sorted(site_titles.items()):
        # Every stable paper any of this group's titles names. A site group joins only when
        # they all name one paper; two would glue distinct papers together (review F1).
        targets = sorted({target for title in titles for target in stable_by_title.get(title, ())})
        if len(targets) > 1:
            continue  # ambiguous: the site record stays its own paper
        if not targets:
            for title in titles:
                unmatched.setdefault(title, []).append(root)
            continue
        joins.append({"from": _group_key(entries, groups[root]), "to": _group_key(entries, groups[targets[0]])})
        uf.union(root, targets[0])
    # Site-only records with one title and no stable paper are one paper among themselves.
    for title, roots in sorted(unmatched.items()):
        for other in roots[1:]:
            if uf.find(other) != uf.find(roots[0]):
                joins.append({"from": _group_key(entries, groups[other]), "to": _group_key(entries, groups[roots[0]])})
                uf.union(other, roots[0])

    collisions = [
        {"title": title, "paper_keys": sorted({_group_key(entries, groups[root]) for root in roots})}
        for title, roots in sorted(stable_by_title.items())
        if len(set(roots)) > 1
    ]

    final = _groups(uf, len(entries))
    key_by_root = {root: _group_key(entries, members) for root, members in final.items()}
    paper_keys = tuple(key_by_root[uf.find(index)] for index in range(len(entries)))
    joins.sort(key=lambda join: (join["to"], join["from"]))
    return MergeResult(paper_keys=paper_keys, title_joins=tuple(joins), title_collisions=tuple(collisions))
