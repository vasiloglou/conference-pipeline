"""Which bundle files go to which surface, and the public URLs a release states.

v6.0 Phase 7.1 (spec sections 6.1, 6.5 and 6.7). The bundle keeps one copy of every
file. The GitHub data repository gets the small documents, never ``site/`` or
``index/``. The HF dataset gets everything but the site, the staged repository and the
two index files no reader needs (a pickle, and the ``.npy`` the Parquet replaces). The
site is the reading surface.

The URLs come from the workspace's ``release:`` block and are written into
``RELEASE.json`` when the bundle is built, so publishing rewrites nothing afterwards
(restamping changes the bundle's tree digest; spec section 1).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# Top-level bundle folders the GitHub data repository never copies.
GITHUB_EXCLUDED_TOP = ("public-repo", "site", "index")
# Bundle paths the HF dataset never uploads.
HF_EXCLUDED_TOP = ("public-repo", "site")
# The pickle, the npy the Parquet replaces, and the index and dataset package default
# receipts, which are run records and not part of the release (phase-7.1-r2 F5, F6).
HF_EXCLUDED_FILES = (
    "index/bm25_index.pkl",
    "index/embeddings.npy",
    "index/index_build_receipt.json",
    "index/semantic_query_proof.json",  # a run record of the index build (phase-7.1-r3 F3)
    "metadata/dataset_package_receipt.json",
)

# Files the HF upload adds beside the bundle's own: the generated card and licence, and the
# .gitattributes the Hub keeps. They count towards its limits (pr-49-r1 F5).
HF_GENERATED_FILES = ("README.md", "LICENSE", ".gitattributes")

URL_FIELDS = ("public_repo_url", "public_site_url", "hf_dataset_url", "cdn_base_url")

_TAG_BASE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def valid_tag(name: Any) -> bool:
    """A bundle id usable as a one-component git tag (pr-49-r1 F7).

    Word characters, "-" and "." from a letter or digit, plus git check-ref-format's rules
    that bite on those characters: no "..", and no ending in "." or ".lock".
    """
    return (
        isinstance(name, str)
        and bool(_TAG_BASE.fullmatch(name))  # whole string: "$" admits a final newline (pr-49-r2 F3)
        and ".." not in name
        and not name.endswith(".")
        and not name.endswith(".lock")
    )

# owner/name: the owner as GitHub and HF allow it; the name never "." or ".." and never
# starting with a dot (phase-7.1-r1 F9).
_REPO = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/(?!\.)[A-Za-z0-9._-]{1,100}$")
_PROJECT = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,56}[a-z0-9])?$")
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_URL = re.compile(rf"^https://{_LABEL}(?:\.{_LABEL})+(/[A-Za-z0-9._~/%-]*)?$")


class ReleaseTargetError(ValueError):
    """A ``release:`` target is not a name the URL can be built from."""


def _hub_accepts(repo_id: str) -> bool:
    """The Hub's own id rules (phase-7.1-r2 F7): its validator when installed, else the same
    rules written out (no "--" or "..", no ".git" end, a name of at most 96 characters that
    neither starts nor ends with "-" or ".")."""
    try:
        from huggingface_hub.utils import validate_repo_id
    except ImportError:
        # The SDK's own pattern: each part starts and ends with a word character
        # (namespace included; phase-7.1-r3 F4) and holds only word characters, "-" and ".".
        namespace, name = repo_id.split("/", 1)
        part = re.compile(r"^\w(?:[\w.-]*\w)?$")
        return (
            "--" not in repo_id and ".." not in repo_id and not repo_id.endswith(".git")
            and len(name) <= 96 and bool(part.match(namespace)) and bool(part.match(name))
        )
    try:
        validate_repo_id(repo_id)
    except Exception:
        return False
    return True


def hf_upload_paths(bundle_root: Path, relatives: Iterable[str]) -> list[str]:
    """The bundle-relative files the HF dataset carries, in a stable order."""
    chosen = []
    for relative in relatives:
        top = relative.split("/", 1)[0]
        if top in HF_EXCLUDED_TOP or relative in HF_EXCLUDED_FILES:
            continue
        chosen.append(relative)
    return sorted(chosen)


def github_excluded(relative: str) -> bool:
    return relative.split("/", 1)[0] in GITHUB_EXCLUDED_TOP


def _text(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ReleaseTargetError(f"release: {key} must be a string")
    return value.strip()


def workspace_release_settings(conference_dir: Path) -> dict[str, Any]:
    """The workspace ``config.yaml`` ``release:`` block, or an empty mapping."""
    import yaml

    path = conference_dir / "config.yaml"
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseTargetError("config.yaml is not readable YAML") from exc
    block = payload.get("release") if isinstance(payload, dict) else None
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ReleaseTargetError("config.yaml release: must be a mapping")
    return block


def stamped_urls(settings: Mapping[str, Any]) -> dict[str, str]:
    """The public URLs a release states, derived from its ``release:`` targets.

    Only the targets that are set produce a URL, so a workspace that names none states
    none (as before Phase 7.1).
    """
    urls: dict[str, str] = {}
    repo = _text(settings, "github_repo")
    if repo is not None:
        if not _REPO.match(repo) or repo.endswith(".git") or ".." in repo:
            raise ReleaseTargetError(f"release: github_repo {repo!r} is not owner/name")
        urls["public_repo_url"] = f"https://github.com/{repo}"
    site = _text(settings, "public_site_url")
    project = _text(settings, "pages_project")
    # Every target is checked, whether or not it forms the URL: publish deploys to the
    # project even when a custom domain names the site (phase-7.1-r1 F9).
    if project is not None and not _PROJECT.match(project):
        raise ReleaseTargetError(f"release: pages_project {project!r} is not a Pages project name")
    if site is not None:
        if not _URL.match(site):
            raise ReleaseTargetError(f"release: public_site_url {site!r} is not an https URL")
        urls["public_site_url"] = site.rstrip("/")
    elif project is not None:
        urls["public_site_url"] = f"https://{project}.pages.dev"
    hf = _text(settings, "hf_repo_id")
    if hf is not None:
        if not _REPO.match(hf) or not _hub_accepts(hf):
            raise ReleaseTargetError(f"release: hf_repo_id {hf!r} is not a Hugging Face repo id")
        urls["hf_dataset_url"] = f"https://huggingface.co/datasets/{hf}"
    cdn = _text(settings, "cdn_base_url")
    if cdn is not None:
        if not _URL.match(cdn):
            raise ReleaseTargetError(f"release: cdn_base_url {cdn!r} is not an https URL")
        urls["cdn_base_url"] = cdn.rstrip("/")
    previous = _text(settings, "previous_version_doi")
    if previous is not None:
        if not re.match(r"^10\.\d{4,9}/[^\s]+$", previous):
            raise ReleaseTargetError(f"release: previous_version_doi {previous!r} is not a DOI")
        urls["previous_version_doi"] = previous
    return urls


def hf_file_url(hf_dataset_url: str, bundle_id: str, relative: str) -> str:
    """The tag-pinned download URL of one bundle file on the HF dataset."""
    return f"{hf_dataset_url.rstrip('/')}/resolve/{bundle_id}/{relative}"
