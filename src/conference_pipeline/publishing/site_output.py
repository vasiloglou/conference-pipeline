"""Output helpers shared by the public site and the hub (v6.0 Phase 6 D14; Phase 4 open
question 4): atomic writes, output checksums, the safe staged swap of an output directory,
and the last check that nothing private reached a public file.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from conference_pipeline._security.private_markers import (
    has_publishing_marker,
    has_publishing_marker_bytes,
)
from conference_pipeline.publishing.public_index import sha256_file


class StaticSiteError(RuntimeError):
    """Raised when static site generation fails closed."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StaticSiteError(f"JSON object expected: {path}")
    return payload


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def output_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file()
    }


def check_output_dir(output_dir: Path, bundle_roots: Sequence[Path]) -> Path:
    """The output directory, refused when replacing it would delete a bundle or a home.

    A directory inside a bundle is allowed: the release orchestrator writes ``<bundle>/site``,
    which the site never copies from.
    """
    resolved = output_dir.resolve()
    unsafe = {Path.cwd().resolve(), Path.home().resolve(), Path("/")}
    for bundle_root in bundle_roots:
        root = bundle_root.resolve()
        if resolved == root or resolved in root.parents:
            raise StaticSiteError(f"refusing unsafe public site output directory: {resolved}")
    if resolved in unsafe:
        raise StaticSiteError(f"refusing unsafe public site output directory: {resolved}")
    return resolved


@contextmanager
def staged_output(output_dir: Path, bundle_roots: Sequence[Path]) -> Iterator[Path]:
    """Build into a sibling staging directory, then swap it into place (phase-4-r1 F8).

    On any failure the staging directory is removed and ``output_dir`` is as it was before
    the build: a previous good site stays whole, and a partial one is never left behind.
    """
    final = check_output_dir(output_dir, bundle_roots)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{os.getpid()}"
    old = final.parent / f".{final.name}.old-{os.getpid()}"
    for leftover in (staging, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    staging.mkdir()
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # The swap itself can fail between its two renames: put the previous output back
    # (phase-4-r2 F5).
    moved_old = False
    try:
        if final.exists():
            final.rename(old)
            moved_old = True
        staging.rename(final)
    except BaseException:
        if moved_old and not final.exists():
            old.rename(final)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def assert_site_output_public_safe(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if has_publishing_marker(rel):
            raise StaticSiteError(f"private marker found in public site path: {rel}")
        if not path.is_file():
            continue
        if has_publishing_marker_bytes(path.read_bytes()):
            raise StaticSiteError(f"private marker found in public site file: {rel}")


_check_output_dir = check_output_dir  # the name Phase 4's callers used
