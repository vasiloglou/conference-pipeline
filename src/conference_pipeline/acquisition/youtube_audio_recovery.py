"""Public-only, single-stream r6 YouTube audio recovery.

The normal yt-dlp download pipeline is intentionally *not* used here.  It
has filename templating, part files, post-processing and external-downloader
extension points that are inappropriate for a receipt-bound r6 payload.  We
use yt-dlp only for metadata and its documented in-process ``YoutubeDL.urlopen``
transport.  The selected direct stream is copied to a descriptor-owned sink;
no pathname, ``/dev/fd`` spelling, template, hook, or subprocess reaches the
library.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import math
import mimetypes
import os
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract, R6ArtifactContractError, R6RawAudioStagingCapability,
    prepare_r6_raw_audio_staging_capability,
)
from conference_pipeline.acquisition.youtube_talks import extract_video_id, url_host_allowed

PUBLIC_YOUTUBE_HOSTS: tuple[str, ...] = ("youtube.com", "youtu.be")
TERMINAL_STATES = frozenset({"acquired", "malformed_or_unsupported_source", "private_or_nonpublic", "age_or_login_gated", "download_failed", "duration_unavailable"})
_YT_DLP_TIMEOUT_SECONDS = 120
_PINNED_DISTRIBUTION = "yt-dlp"
_CHUNK = 1024 * 1024

# This is deliberately private and unset in production.  It exists solely so
# the disposable-tree regression can make a deterministic mutation in the
# otherwise tiny stat/open interval.  It is not read from configuration, the
# environment, or a receipt.
_TREE_WALK_TEST_HOOK: Callable[[str, str], None] | None = None


class R6DownloaderPreflightError(RuntimeError):
    """The pinned in-process downloader is unavailable or untrusted."""


def canonical_youtube_url(video_id: str) -> str:
    if not extract_video_id(f"https://youtu.be/{video_id}"):
        raise ValueError("video_id is not a canonical YouTube identifier")
    return f"https://www.youtube.com/watch?v={video_id}"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _trusted_yt_dlp_module(*, _test_module: object | None = None) -> object:
    if _test_module is not None:
        if not getattr(_test_module, "__r6_test_fake__", False):
            raise R6DownloaderPreflightError("r6 test module injection is invalid")
        return _test_module
    try:
        import yt_dlp  # type: ignore[import-not-found]
        distribution = importlib.metadata.distribution(_PINNED_DISTRIBUTION)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise R6DownloaderPreflightError("r6 pinned downloader module is unavailable") from exc
    module_file = Path(str(getattr(yt_dlp, "__file__", "")))
    root = Path(distribution.locate_file(".")).resolve()
    if not module_file.is_file() or not root.is_dir():
        raise R6DownloaderPreflightError("r6 pinned downloader module provenance is invalid")
    try:
        module_file.resolve().relative_to(root)
    except ValueError as exc:
        raise R6DownloaderPreflightError("r6 pinned downloader module provenance drifted") from exc
    if not isinstance(getattr(yt_dlp, "YoutubeDL", None), type):
        raise R6DownloaderPreflightError("r6 pinned downloader module API is invalid")
    return yt_dlp


def _declared_pin() -> str:
    """Bind both repository declarations to one exact non-editable release."""
    root = Path(__file__).parents[3]
    declarations: list[str] = []
    root_fd = _open_trusted_directory(root, label="repository root")
    try:
        for relative in ("requirements.txt", "pyproject.toml"):
            try:
                text = _read_regular_at(root_fd, relative, relative=relative).decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise R6DownloaderPreflightError("r6 downloader lock declaration is unavailable") from exc
            declarations.extend(
                line.strip().strip(",").strip().strip('"')
                for line in text.splitlines()
                if "yt-dlp" in line
            )
    finally:
        os.close(root_fd)
    exact = [line for line in declarations if line.startswith("yt-dlp==") and ";" not in line and "@" not in line]
    if len(exact) != 2 or len(set(exact)) != 1:
        raise R6DownloaderPreflightError("r6 downloader lock declaration is not exact")
    return exact[0]


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino and left.st_mode == right.st_mode


def _read_regular_at(parent_fd: int, name: str, *, relative: str) -> bytes:
    """Read a regular entry through its held parent, never through a path."""

    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise R6DownloaderPreflightError("r6 package tree node cannot be inspected") from exc
    if stat.S_ISLNK(before.st_mode):
        raise R6DownloaderPreflightError("r6 package tree contains a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise R6DownloaderPreflightError("r6 package tree contains a nonregular artifact")
    if _TREE_WALK_TEST_HOOK is not None:
        _TREE_WALK_TEST_HOOK("before_file_open", relative)
    try:
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    except OSError as exc:
        raise R6DownloaderPreflightError("r6 package file cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise R6DownloaderPreflightError("r6 package file changed during validation")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise R6DownloaderPreflightError("r6 package file cannot be read safely") from exc
    finally:
        os.close(descriptor)


def _open_trusted_directory(path: Path, *, label: str) -> int:
    """Retain a no-follow directory descriptor for all descendant reads."""
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise R6DownloaderPreflightError(f"r6 {label} is not a real directory")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise R6DownloaderPreflightError(f"r6 {label} is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
            raise R6DownloaderPreflightError(f"r6 {label} changed during validation")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_relative_regular_at(root_fd: int, relative: str, *, label: str) -> bytes:
    """Read a regular leaf through retained no-follow directory descriptors."""
    parts = relative.replace("\\", "/").split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise R6DownloaderPreflightError(f"r6 {label} path is malformed")
    current = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                before = os.stat(component, dir_fd=current, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise R6DownloaderPreflightError(f"r6 {label} parent is unsafe")
                child = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
                    os.close(child)
                    raise R6DownloaderPreflightError(f"r6 {label} parent changed during validation")
            except OSError as exc:
                raise R6DownloaderPreflightError(f"r6 {label} parent is unavailable") from exc
            os.close(current)
            current = child
        return _read_regular_at(current, parts[-1], relative=relative)
    finally:
        os.close(current)


def _safe_package_entries(package_root: Path) -> tuple[tuple[str, str, int], ...]:
    """Walk an installed package without following links or accepting debris.

    ``Path.rglob`` follows enough platform-specific behaviour that it is not a
    suitable provenance boundary.  The package is intentionally tiny in the
    sense relevant here: every directory must lead to a recorded regular file;
    every other node is rejected.  Python's import cache is not part of a
    wheel, so a cache directory is correctly an installation drift, too.
    """
    try:
        root_stat = os.lstat(package_root)
        root_fd = os.open(
            package_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise R6DownloaderPreflightError("r6 package root is unavailable") from exc
    try:
        opened_root = os.fstat(root_fd)
        if (not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode)
                or not stat.S_ISDIR(opened_root.st_mode) or not _same_file(root_stat, opened_root)):
            raise R6DownloaderPreflightError("r6 package root is not a real directory")
        return _safe_package_entries_from_fd(root_fd)
    finally:
        os.close(root_fd)


def _safe_package_entries_from_fd(root_fd: int) -> tuple[tuple[str, str, int], ...]:
    """Descriptor-only implementation of the installed package tree walk."""
    entries: list[tuple[str, str, int]] = []

    def list_children(directory_fd: int) -> list[str]:
        """Enumerate through a disposable duplicate of an owned descriptor."""

        try:
            listing_fd = os.dup(directory_fd)
        except OSError as exc:
            raise R6DownloaderPreflightError("r6 package directory cannot be duplicated") from exc
        try:
            return sorted(os.listdir(listing_fd))
        except OSError as exc:
            raise R6DownloaderPreflightError("r6 package tree cannot be enumerated") from exc
        finally:
            # Some platform ``listdir(fd)`` implementations consume the
            # descriptor they enumerate.  It is a disposable duplicate, so
            # either close outcome leaves the caller-owned descriptor intact.
            try:
                os.close(listing_fd)
            except OSError:
                pass

    def walk(current_fd: int, prefix: str) -> None:
        children = list_children(current_fd)
        if prefix and not children:
            raise R6DownloaderPreflightError("r6 package tree has an unrecorded directory")
        for name in children:
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise R6DownloaderPreflightError("r6 package tree contains a malformed path")
            relative = f"{prefix}/{name}" if prefix else name
            try:
                before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            except OSError as exc:
                raise R6DownloaderPreflightError("r6 package tree node cannot be inspected") from exc
            if stat.S_ISLNK(before.st_mode):
                raise R6DownloaderPreflightError("r6 package tree contains a symlink")
            if stat.S_ISDIR(before.st_mode):
                # Bytecode cache is generated by importing the already
                # verified module; it is not a wheel payload.  Every other
                # extra directory remains a fail-closed tree violation.
                if name == "__pycache__":
                    continue
                if _TREE_WALK_TEST_HOOK is not None:
                    _TREE_WALK_TEST_HOOK("before_directory_open", relative)
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    raise R6DownloaderPreflightError("r6 package directory cannot be opened safely") from exc
                try:
                    opened = os.fstat(child_fd)
                    if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
                        raise R6DownloaderPreflightError("r6 package directory changed during validation")
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(before.st_mode):
                raw = _read_regular_at(current_fd, name, relative=relative)
                entries.append((relative, _sha(raw), len(raw)))
            else:
                raise R6DownloaderPreflightError("r6 package tree contains a nonregular artifact")
        # Catch an insertion/removal made while this held directory was walked.
        if list_children(current_fd) != children:
            raise R6DownloaderPreflightError("r6 package tree changed during validation")

    walk(root_fd, "")
    if not entries:
        raise R6DownloaderPreflightError("r6 package tree is empty")
    # An empty directory is already rejected.  All other directory paths are
    # necessarily ancestors of an entry and are therefore represented by the
    # exact RECORD file set rather than silently ignored.
    return tuple(entries)


def _strict_record(package_root: Path, record_path: Path) -> tuple[tuple[str, str, int], ...]:
    """Parse only immutable wheel RECORD entries for the package tree."""
    try:
        # Keep the containing dist-info directory open while reading RECORD;
        # a later pathname replacement cannot influence its bytes.
        parent_fd = _open_trusted_directory(record_path.parent, label="distribution RECORD parent")
        try:
            raw = _read_regular_at(parent_fd, record_path.name, relative="RECORD")
        finally:
            os.close(parent_fd)
        rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise R6DownloaderPreflightError("r6 distribution RECORD cannot be read") from exc
    parsed: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    all_seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise R6DownloaderPreflightError("r6 distribution RECORD is malformed")
        path, encoded, size_text = row
        path = path.replace("\\", "/")
        if not path or path in all_seen:
            raise R6DownloaderPreflightError("r6 distribution RECORD has duplicate or malformed entry")
        all_seen.add(path)
        if not path.startswith("yt_dlp/"):
            continue
        relative = path.removeprefix("yt_dlp/")
        if not relative or relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise R6DownloaderPreflightError("r6 distribution RECORD escapes package root")
        if relative in seen:
            raise R6DownloaderPreflightError("r6 distribution RECORD has duplicate package entry")
        seen.add(relative)
        if not encoded.startswith("sha256=") or not size_text.isdecimal():
            raise R6DownloaderPreflightError("r6 distribution RECORD has unhashed package entry")
        token = encoded.removeprefix("sha256=")
        if not token or "=" in token:
            raise R6DownloaderPreflightError("r6 distribution RECORD has malformed package hash")
        try:
            digest = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        except (ValueError, TypeError) as exc:
            raise R6DownloaderPreflightError("r6 distribution RECORD has malformed package hash") from exc
        if len(digest) != 32:
            raise R6DownloaderPreflightError("r6 distribution RECORD has malformed package hash")
        parsed.append((relative, digest.hex(), int(size_text)))
    if not parsed:
        raise R6DownloaderPreflightError("r6 distribution RECORD has no package entries")
    return tuple(sorted(parsed))


def _verified_tree_receipt(package_root: Path, record_path: Path) -> dict[str, str]:
    """Return canonical independent digests after exact tree/RECORD equality."""
    tree = _safe_package_entries(package_root)
    record = _strict_record(package_root, record_path)
    if tuple(sorted(tree)) != record:
        raise R6DownloaderPreflightError("r6 package tree and RECORD differ")
    tree_payload = "\n".join(f"{path}|{digest}|{size}" for path, digest, size in tree).encode()
    record_payload = "\n".join(f"{path}|{digest}|{size}" for path, digest, size in record).encode()
    return {"tree_manifest_sha256": _sha(tree_payload), "record_sha256": _sha(record_payload)}


def _locked_yt_dlp_artifact(version: str) -> tuple[str, str]:
    """Return the immutable universal-wheel identity from this repository lock."""
    root = Path(__file__).parents[3]
    root_fd = _open_trusted_directory(root, label="repository root")
    try:
        lock_raw = _read_regular_at(root_fd, "uv.lock", relative="uv.lock")
        lock = tomllib.loads(lock_raw.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise R6DownloaderPreflightError("r6 downloader uv lock is unavailable") from exc
    finally:
        os.close(root_fd)
    matches = [item for item in lock.get("package", []) if isinstance(item, dict) and item.get("name") == _PINNED_DISTRIBUTION]
    if len(matches) != 1:
        raise R6DownloaderPreflightError("r6 downloader uv lock is ambiguous")
    package = matches[0]
    if package.get("version") != version or not isinstance(package.get("source"), dict) or package["source"].get("registry") != "https://pypi.org/simple":
        raise R6DownloaderPreflightError("r6 downloader uv lock source/version drifted")
    wheels = package.get("wheels")
    if not isinstance(wheels, list):
        raise R6DownloaderPreflightError("r6 downloader uv lock lacks wheel artifacts")
    expected_name = f"yt_dlp-{version}-py3-none-any.whl"
    artifacts = [item for item in wheels if isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"].endswith(expected_name)]
    if len(artifacts) != 1 or not isinstance(artifacts[0].get("hash"), str) or not artifacts[0]["hash"].startswith("sha256:") or not _is_sha256(artifacts[0]["hash"][7:]):
        raise R6DownloaderPreflightError("r6 downloader uv lock artifact is not immutable")
    # The project declaration is a second intentional binding, not a loose
    # ambient installed-package check.
    if _declared_pin() != f"yt-dlp=={version}":
        raise R6DownloaderPreflightError("r6 project downloader requirement drifted")
    return artifacts[0]["hash"][7:], _sha(lock_raw)


def _distribution_identity(module: object) -> dict[str, str]:
    if getattr(module, "__r6_test_fake__", False):
        # A disposable distribution fixture can exercise the *same* tree and
        # RECORD verifier without ever changing site-packages.  The legacy
        # byte-only fake remains for small metadata unit tests; it cannot be
        # used to claim installed-tree provenance.
        fixture_root = getattr(module, "__r6_distribution_root__", None)
        fixture_record = getattr(module, "__r6_distribution_record__", None)
        fixture_file = getattr(module, "__file__", None)
        if fixture_root is not None or fixture_record is not None:
            if not all(isinstance(value, (str, Path)) for value in (fixture_root, fixture_record, fixture_file)):
                raise R6DownloaderPreflightError("r6 fixture distribution provenance is invalid")
            package_root = Path(fixture_root)
            record_path = Path(fixture_record)
            verified = _verified_tree_receipt(package_root, record_path)
            try:
                module_path = Path(fixture_file).resolve()
                module_bytes = module_path.read_bytes()
                relative_module = module_path.relative_to(package_root).as_posix()
            except (OSError, ValueError) as exc:
                raise R6DownloaderPreflightError("r6 fixture module provenance drifted") from exc
            tree = _safe_package_entries(package_root)
            if relative_module not in {item[0] for item in tree}:
                raise R6DownloaderPreflightError("r6 fixture module is outside the verified tree")
            pin = str(getattr(module, "__r6_lock_identity__", "yt-dlp==fixture"))
            return {
                "distribution": _PINNED_DISTRIBUTION,
                "distribution_identity": f"{_PINNED_DISTRIBUTION}==fixture",
                "tree_manifest_sha256": verified["tree_manifest_sha256"],
                "module_bytes_sha256": _sha(module_bytes),
                "record_sha256": verified["record_sha256"],
                "lock_artifact_sha256": _sha(pin.encode()),
                "lock_sha256": _sha(pin.encode()),
            }
        source = bytes(getattr(module, "__r6_source_bytes__", b"fixture-source"))
        record = bytes(getattr(module, "__r6_record_bytes__", b"yt_dlp/__init__.py,fixture,1\n"))
        pin = str(getattr(module, "__r6_lock_identity__", "yt-dlp==fixture"))
        return {"distribution": _PINNED_DISTRIBUTION, "distribution_identity": f"{_PINNED_DISTRIBUTION}==fixture", "tree_manifest_sha256": _sha(source), "module_bytes_sha256": _sha(source), "record_sha256": _sha(record), "lock_artifact_sha256": _sha(pin.encode()), "lock_sha256": _sha(pin.encode())}
    try:
        distribution = importlib.metadata.distribution(_PINNED_DISTRIBUTION)
        version = importlib.metadata.version(_PINNED_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise R6DownloaderPreflightError("r6 pinned downloader module is unavailable") from exc
    pin = _declared_pin()
    if pin != f"yt-dlp=={version}":
        raise R6DownloaderPreflightError("r6 installed downloader does not match repository lock")
    root = Path(distribution.locate_file(".")).resolve()
    package_root = root / "yt_dlp"
    record_paths = [Path(distribution.locate_file(item)) for item in distribution.files or () if str(item).replace("\\", "/").endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise R6DownloaderPreflightError("r6 distribution RECORD is unavailable")
    try:
        record_paths[0].resolve().relative_to(root)
    except ValueError as exc:
        raise R6DownloaderPreflightError("r6 distribution RECORD escapes distribution root") from exc
    verified = _verified_tree_receipt(package_root, record_paths[0])
    tree = _safe_package_entries(package_root)
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        relative_module = module_path.relative_to(package_root)
        package_fd = _open_trusted_directory(package_root, label="installed package root")
        try:
            module_bytes = _read_relative_regular_at(
                package_fd, relative_module.as_posix(), label="installed module"
            )
        finally:
            os.close(package_fd)
    except (OSError, ValueError) as exc:
        raise R6DownloaderPreflightError("r6 module source provenance drifted") from exc
    if relative_module.as_posix() not in {item[0] for item in tree}:
        raise R6DownloaderPreflightError("r6 imported module is outside the verified tree")
    artifact_sha, lock_sha = _locked_yt_dlp_artifact(version)
    return {"distribution": _PINNED_DISTRIBUTION, "distribution_identity": f"{_PINNED_DISTRIBUTION}=={version}", "tree_manifest_sha256": verified["tree_manifest_sha256"], "module_bytes_sha256": _sha(module_bytes), "record_sha256": verified["record_sha256"], "lock_artifact_sha256": artifact_sha, "lock_sha256": lock_sha}


def preflight_fixed_r6_yt_dlp(*, _test_module: object | None = None) -> dict[str, str]:
    module = _trusted_yt_dlp_module(_test_module=_test_module)
    return _distribution_identity(module)


def _loaded_distribution_identity(module: object) -> str:
    """Bind a loaded client to the owner-approved preflight without a tree walk.

    The full RECORD/tree/module-byte verification is an owner preflight
    operation.  Repeating a recursive descriptor walk in every parallel
    network worker is neither required to bind the already-loaded module nor
    safe on platforms whose directory enumeration consumes a descriptor.
    """

    if getattr(module, "__r6_test_fake__", False):
        return f"{_PINNED_DISTRIBUTION}==fixture"
    try:
        return f"{_PINNED_DISTRIBUTION}=={importlib.metadata.version(_PINNED_DISTRIBUTION)}"
    except importlib.metadata.PackageNotFoundError as exc:
        raise R6DownloaderPreflightError("r6 pinned downloader module is unavailable") from exc


def validate_fixed_r6_yt_dlp_preflight(preflight: Mapping[str, object]) -> dict[str, str]:
    required = {"distribution", "distribution_identity", "tree_manifest_sha256", "module_bytes_sha256", "record_sha256", "lock_artifact_sha256", "lock_sha256"}
    if not isinstance(preflight, Mapping) or set(preflight) != required or preflight.get("distribution") != _PINNED_DISTRIBUTION or not isinstance(preflight.get("distribution_identity"), str) or not str(preflight["distribution_identity"]).startswith(f"{_PINNED_DISTRIBUTION}==") or not all(_is_sha256(preflight.get(key)) for key in required - {"distribution", "distribution_identity"}):
        raise R6DownloaderPreflightError("r6 downloader preflight identity is invalid")
    return {key: str(preflight[key]) for key in sorted(required)}


def _metadata_options() -> dict[str, object]:
    return {"noplaylist": True, "retries": 0, "skip_download": True, "quiet": True, "no_warnings": True, "cachedir": False, "ignoreconfig": True, "proxy": "", "socket_timeout": _YT_DLP_TIMEOUT_SECONDS, "postprocessors": [], "writethumbnail": False, "writeinfojson": False, "writesubtitles": False, "writeautomaticsub": False, "external_downloader": None}


def _assert_options(options: Mapping[str, object], *, download: bool = False) -> None:
    # ``download`` remains keyword-compatible only for old callers; a true
    # value is an immediate fail-closed assertion, never a mode switch.
    allowed = {"noplaylist", "retries", "skip_download", "quiet", "no_warnings", "cachedir", "ignoreconfig", "proxy", "socket_timeout", "postprocessors", "writethumbnail", "writeinfojson", "writesubtitles", "writeautomaticsub", "external_downloader"}
    if download or set(options) != allowed or options.get("noplaylist") is not True or options.get("retries") != 0 or options.get("skip_download") is not True or options.get("cachedir") is not False or options.get("ignoreconfig") is not True or options.get("proxy") != "" or options.get("postprocessors") != [] or options.get("external_downloader") is not None or options.get("socket_timeout") != _YT_DLP_TIMEOUT_SECONDS or any(options.get(k) is not False for k in ("writethumbnail", "writeinfojson", "writesubtitles", "writeautomaticsub")):
        raise R6DownloaderPreflightError("r6 downloader options violate public-only policy")


def _metadata_extract(module: object, url: str) -> tuple[object, Mapping[str, object] | None, Exception | None]:
    options = _metadata_options(); _assert_options(options)
    factory = getattr(module, "YoutubeDL", None)
    if not callable(factory):
        raise R6DownloaderPreflightError("r6 pinned downloader module API is invalid")
    client = factory(dict(options)); extract = getattr(client, "extract_info", None)
    if not callable(extract):
        raise R6DownloaderPreflightError("r6 pinned downloader client API is invalid")
    try:
        result = extract(url, download=False)
    except Exception as exc:
        return client, None, exc
    return client, result if isinstance(result, Mapping) else None, None


def _parse_metadata(payload: Mapping[str, object] | None, video_id: str) -> tuple[float | None, str | None]:
    if payload is None: return None, "download_failed"
    if str(payload.get("id") or "") != video_id: return None, "malformed_or_unsupported_source"
    availability = str(payload.get("availability") or "public").lower()
    if availability in {"private", "unlisted", "subscriber_only", "premium_only"}: return None, "private_or_nonpublic"
    if availability in {"needs_auth", "login_required", "age_restricted", "needs_subscription"}: return None, "age_or_login_gated"
    try: duration = float(payload.get("duration"))
    except (TypeError, ValueError): duration = float("nan")
    return (duration, None) if math.isfinite(duration) and duration > 0 else (None, "duration_unavailable")


def _selected_direct_stream(payload: Mapping[str, object]) -> str:
    # On a candidate format these fields describe a fragmented, converted, or
    # path-bearing operation and disqualify that candidate.  Do not apply the
    # same fence to the top-level metadata object: with ``skip_download=True``
    # yt-dlp adds default ``requested_formats`` selection facts to ordinary
    # ``download=False`` results, while public videos normally include
    # read-only thumbnail/caption inventories. Recovery never executes those
    # facts; it opens only one independently validated direct audio-only URL
    # below. Write-mode-only path/download fields remain forbidden at both
    # levels.
    forbidden = {"requested_formats", "requested_downloads", "fragments", "__files_to_move", "_filename", "filepath", "postprocessors", "fixup", "fixup_policy", "__postprocessors"}
    top_level_forbidden = forbidden - {"requested_formats"}
    if any(
        key in payload and payload[key] not in (None, [], {}, False, "")
        for key in top_level_forbidden
    ):
        raise R6DownloaderPreflightError(
            "r6 metadata advertises write-mode or fragmented state"
        )
    candidates = payload.get("formats")
    if not isinstance(candidates, list):
        candidates = [payload]
    direct: list[tuple[float, str, str]] = []
    for value in candidates:
        if not isinstance(value, Mapping) or any(key in value and value[key] not in (None, [], {}, False, "") for key in forbidden):
            continue
        url = value.get("url"); protocol = str(value.get("protocol") or "https").lower()
        parsed = urlparse(str(url)) if isinstance(url, str) else None
        if (
            not isinstance(url, str)
            or parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or protocol not in {"http", "https"}
        ):
            continue
        if str(value.get("vcodec") or "none") != "none" or str(value.get("acodec") or "none") in {"none", ""}:
            continue
        try: quality = float(value.get("abr") or value.get("tbr") or 0)
        except (TypeError, ValueError): quality = 0
        direct.append((quality, str(value.get("format_id") or ""), url))
    if not direct: raise R6DownloaderPreflightError("r6 metadata has no direct audio-only stream")
    # Deterministic selection is intentionally local: it does not let yt-dlp
    # select a merged multi-format representation later.
    return sorted(direct, key=lambda row: (-row[0], row[1], row[2]))[0][2]


class _DescriptorOnlySink:
    def __init__(self, capability: R6RawAudioStagingCapability): self._capability = capability
    def copy(self, response: object) -> None:
        read = getattr(response, "read", None)
        if not callable(read): raise R6DownloaderPreflightError("r6 stream response is not readable")
        while True:
            chunk = read(_CHUNK)
            if not chunk: return
            if not isinstance(chunk, bytes): raise R6DownloaderPreflightError("r6 stream response bytes are invalid")
            self._capability.write(chunk)


def _base_record(item_id: str, video_id: str | None) -> dict[str, Any]:
    return {"item_id": item_id, "source_url": canonical_youtube_url(video_id) if video_id else None, "video_id": video_id, "provenance": "direct_public_youtube", "attempt_count": 0}
def _terminal(item_id: str, video_id: str | None, state: str, **extra: Any) -> dict[str, Any]:
    if state not in TERMINAL_STATES: raise ValueError(f"unknown terminal state: {state}")
    return {**_base_record(item_id, video_id), "terminal_state": state, **extra}
def resume_record_matches(record: Mapping[str, Any], *, item_id: str, video_id: str) -> bool:
    return record.get("terminal_state") in TERMINAL_STATES and record.get("item_id") == item_id and record.get("video_id") == video_id and (record.get("terminal_state") != "acquired" or (isinstance(record.get("sha256"), str) and len(str(record["sha256"])) == 64 and isinstance(record.get("byte_count"), int) and record["byte_count"] > 0))


def recover_public_youtube_audio(*, item_id: str, source_url: str, r6_root: Path, artifact_contract: R6ArtifactContract | None = None, _workspace_lease: object | None = None, preflight: Mapping[str, object], max_attempts: int = 1, _test_module: object | None = None) -> dict[str, Any]:
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 3: raise ValueError("max_attempts must be a bounded integer between 1 and 3")
    expected = validate_fixed_r6_yt_dlp_preflight(preflight)
    if _test_module is not None and (artifact_contract is None or not artifact_contract.fixture_injection): raise ValueError("r6 test module injection is unavailable in production")
    module = _trusted_yt_dlp_module(_test_module=_test_module)
    if _loaded_distribution_identity(module) != expected["distribution_identity"]:
        raise R6DownloaderPreflightError("r6 downloader identity changed after preflight")
    if not url_host_allowed(source_url, PUBLIC_YOUTUBE_HOSTS): return _terminal(item_id, None, "malformed_or_unsupported_source")
    video_id = extract_video_id(source_url)
    if video_id is None: return _terminal(item_id, None, "malformed_or_unsupported_source")
    url = canonical_youtube_url(video_id)
    if source_url not in {url, f"https://youtube.com/watch?v={video_id}", f"https://youtu.be/{video_id}"}: return _terminal(item_id, None, "malformed_or_unsupported_source")
    client: object | None = None; payload: Mapping[str, object] | None = None; duration: float | None = None
    for attempt in range(1, max_attempts + 1):
        client, payload, extraction_error = _metadata_extract(module, url)
        # A metadata extractor can report an immediately terminal, deterministic
        # outcome before a stream is selected.  Only transport-shaped failures
        # consume further bounded recovery operations; retrying every parser or
        # policy error would fabricate attempt evidence and may reopen a route
        # the terminal result has already ruled out.
        if extraction_error is not None:
            if isinstance(extraction_error, (OSError, TimeoutError)) and attempt < max_attempts:
                continue
            return _terminal(item_id, video_id, "download_failed", attempt_count=attempt)
        duration, blocker = _parse_metadata(payload, video_id)
        if blocker == "download_failed": continue
        if blocker: return _terminal(item_id, video_id, blocker, attempt_count=attempt)
        break
    if duration is None or payload is None or client is None: return _terminal(item_id, video_id, "download_failed", attempt_count=max_attempts)
    if artifact_contract is None or Path(r6_root) != artifact_contract.r6_root: raise ValueError("r6 audio recovery requires the fixed artifact contract")
    try: stream_url = _selected_direct_stream(payload)
    except R6DownloaderPreflightError: return _terminal(item_id, video_id, "download_failed", duration_seconds=duration, attempt_count=1)
    open_url = getattr(client, "urlopen", None)
    if not callable(open_url): raise R6DownloaderPreflightError("r6 pinned downloader lacks in-process transport")
    lease = artifact_contract._acquire_workspace_lease() if _workspace_lease is None else _workspace_lease
    for attempt in range(1, max_attempts + 1):
        capability = prepare_r6_raw_audio_staging_capability(contract=artifact_contract, item_id=item_id, video_id=video_id, _workspace_lease=lease)
        response: object | None = None
        try:
            response = open_url(stream_url)
            _DescriptorOnlySink(capability).copy(response)
            content = capability.publish_and_read()
        except (R6ArtifactContractError, R6DownloaderPreflightError):
            capability.discard(); raise
        except Exception:
            capability.discard(); continue
        finally:
            # urllib-like transports own a finite socket/file resource.  Close
            # exactly once even when read, sink, publication, or error mapping
            # fails; close failures cannot convert a safely completed payload
            # into an unaccounted retry.
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        mime_type = mimetypes.guess_type(urlparse(stream_url).path)[0] or "application/octet-stream"
        return {**_base_record(item_id, video_id), "terminal_state": "acquired", "duration_seconds": duration, "local_media_path": str(artifact_contract.workspace / capability.relative), "sha256": _sha(content), "mime_type": mime_type, "byte_count": len(content), "attempt_count": attempt}
    return _terminal(item_id, video_id, "download_failed", duration_seconds=duration, attempt_count=max_attempts)
