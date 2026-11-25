"""Provider limits for each publish surface, checked before anything goes live.

v6.0 Phase 7.1 (spec section 6.2). One table holds every limit a surface enforces on a
file, a site or a folder; ``release check`` and ``release publish`` both read it, so a
file over a provider's limit is refused locally instead of failing half-way through a
live push. Each row names the provider page it was read from (2026-11-21).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MiB = 1024 * 1024
GiB = 1024 * MiB


@dataclass(frozen=True)
class SurfaceLimit:
    surface: str
    max_file_bytes: int
    max_files: int | None
    max_entries_per_dir: int | None
    source: str


LIMITS: tuple[SurfaceLimit, ...] = (
    # "GitHub blocks files larger than 100 MiB."
    SurfaceLimit(
        "github",
        100 * MiB,
        None,
        None,
        "https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github",
    ),
    # "The maximum file size for a single Cloudflare Pages site asset is 25 MiB." and
    # "up to 20,000 files on the Free plan".
    SurfaceLimit("pages", 25 * MiB, 20_000, None, "https://developers.cloudflare.com/pages/platform/limits"),
    # "500GB is the hard limit for a single file size"; "cannot exceed 10k files per folder".
    SurfaceLimit("hf", 500 * 1000**3, None, 10_000, "https://huggingface.co/docs/hub/storage-limits"),
    # One PUT uploads at most 5 MiB less than 5 GiB; the uploader sends each object in one PUT.
    SurfaceLimit("r2", 5 * GiB - 5 * MiB, None, None, "https://developers.cloudflare.com/r2/platform/limits/"),
)


def limit_for(surface: str) -> SurfaceLimit:
    for row in LIMITS:
        if row.surface == surface:
            return row
    raise KeyError(f"no limits for surface {surface!r}")


def _walk(root: Path) -> tuple[list[tuple[str, int]], dict[str, int]]:
    files: list[tuple[str, int]] = []
    entries: dict[str, int] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        relative_dir = Path(directory).relative_to(root).as_posix() or "."
        entries[relative_dir] = len(dirnames) + len(filenames)
        for name in sorted(filenames):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            files.append((relative, path.lstat().st_size))
    return files, entries


def check_tree(surface: str, root: Path) -> list[dict[str, str]]:
    """Every way the tree at ``root`` breaks ``surface``'s limits, as release-check findings."""
    files, entries = _walk(root)
    return check_files(surface, files, entries)


def check_paths(surface: str, root: Path, relatives: list[str]) -> list[dict[str, str]]:
    """The same check over a chosen file list under ``root`` (an upload that is not a whole tree)."""
    return check_sized(surface, [(relative, (root / relative).lstat().st_size) for relative in relatives])


def check_sized(surface: str, files: list[tuple[str, int]]) -> list[dict[str, str]]:
    """The same check over a planned remote tree given as (path, size) pairs.

    For an upload that adds files of its own, such as the HF card (pr-49-r1 F5).
    """
    files = sorted(dict(files).items())
    children: dict[str, set[str]] = {}
    for relative, _size in files:
        parts = relative.split("/")
        for depth in range(len(parts)):
            parent = "/".join(parts[:depth]) or "."
            children.setdefault(parent, set()).add(parts[depth])
    entries = {parent: len(names) for parent, names in children.items()}
    return check_files(surface, files, entries)


def check_files(
    surface: str, files: list[tuple[str, int]], entries: dict[str, int]
) -> list[dict[str, str]]:
    limit = limit_for(surface)
    findings: list[dict[str, str]] = []
    for relative, size in files:
        if size > limit.max_file_bytes:
            findings.append(
                {
                    "check": "surface_limit",
                    "path": relative,
                    "detail": f"{surface}: {size} bytes is over the {limit.max_file_bytes}-byte file limit",
                }
            )
    if limit.max_files is not None and len(files) > limit.max_files:
        findings.append(
            {
                "check": "surface_limit",
                "path": ".",
                "detail": f"{surface}: {len(files)} files is over the {limit.max_files}-file limit",
            }
        )
    if limit.max_entries_per_dir is not None:
        for relative_dir, count in sorted(entries.items()):
            if count > limit.max_entries_per_dir:
                findings.append(
                    {
                        "check": "surface_limit",
                        "path": relative_dir,
                        "detail": f"{surface}: {count} entries is over the {limit.max_entries_per_dir}-entry folder limit",
                    }
                )
    return findings
