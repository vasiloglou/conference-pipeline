"""Single source of truth for per-conference identity.

Reusable platform code must never hardcode a conference name, year, domain, or
program URL. Every such value lives in the workspace's ``config.yaml`` and is
read through :func:`load_conference_identity` so the same code path serves
NeurIPS, ICLR, ICML, or any future conference without edits.

``config.yaml`` shape (only ``conference.name`` is strictly required)::

    conference:
      slug: "icml2026"
      name: "ICML 2026"
      full_name: "International Conference on Machine Learning 2026"
      year: 2026
    urls:
      base: "https://icml.cc/Conferences/2026"
      virtual: "https://icml.cc/virtual/2026"
      workshops: "https://icml.cc/virtual/2026/events/workshop"
      ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


class ConferenceIdentityError(RuntimeError):
    """Raised when a workspace does not declare a usable conference identity."""


@dataclass(frozen=True)
class VirtualSiteBoundary:
    """Strict, configured origin and virtual-root policy for route admission.

    This value deliberately contains no conference-derived fallback.  Callers
    must obtain a virtual URL from their workspace configuration and parse it
    before admitting a route into a conference-specific receipt or inventory.
    """

    scheme: str
    host: str
    root_segments: tuple[str, ...]

    @property
    def origin(self) -> str:
        """Return the normalized origin without credentials or a port."""

        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}"

    def canonical_url(self, *suffix_segments: str) -> str:
        """Build a URL from the configured root and already-validated suffix."""

        if any(not isinstance(segment, str) or not segment for segment in suffix_segments):
            raise ConferenceIdentityError("virtual-site URL suffix must contain nonempty segments")
        return f"{self.origin}/{'/'.join((*self.root_segments, *suffix_segments))}"


def _strict_url_path_segments(path: str) -> tuple[str, ...]:
    """Return path segments only when no normalization can change membership."""

    if not path.startswith("/"):
        raise ConferenceIdentityError("conference virtual URL must have an absolute path")
    raw_segments = path.split("/")
    if raw_segments[0] != "":
        raise ConferenceIdentityError("conference virtual URL path is malformed")
    if raw_segments[-1] == "":
        raw_segments = raw_segments[:-1]
    if not raw_segments[1:]:
        raise ConferenceIdentityError("conference virtual URL must include a nonempty path")

    segments: list[str] = []
    for segment in raw_segments[1:]:
        decoded = unquote(segment)
        if (
            not segment
            or segment != decoded
            or decoded in {".", ".."}
            or "/" in decoded
            or "\\" in decoded
        ):
            raise ConferenceIdentityError("conference virtual URL path is not safely normalized")
        segments.append(segment)
    return tuple(segments)


def parse_virtual_site_boundary(virtual_url: str) -> VirtualSiteBoundary:
    """Parse a configured virtual URL into a strict, reusable route boundary.

    The policy accepts exactly an HTTPS origin without credentials, a port,
    query, fragment, or ambiguous path encoding.  It preserves a normalized
    host and virtual-root path without baking a conference host, year, or
    taxonomy into reusable code.
    """

    if not isinstance(virtual_url, str) or not virtual_url.strip():
        raise ConferenceIdentityError("conference virtual URL is required")
    try:
        parsed = urlsplit(virtual_url)
        port = parsed.port
    except ValueError as error:
        raise ConferenceIdentityError("conference virtual URL has invalid authority") from error
    host = (parsed.hostname or "").rstrip(".").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConferenceIdentityError("conference virtual URL has unsafe authority")
    return VirtualSiteBoundary(
        scheme="https",
        host=host,
        root_segments=_strict_url_path_segments(parsed.path),
    )


@dataclass(frozen=True)
class ConferenceIdentity:
    """Conference identity resolved from a workspace ``config.yaml``."""

    slug: str
    name: str
    full_name: str
    year: int | None
    base_url: str
    virtual_url: str
    site_domain: str
    urls: dict[str, str] = field(default_factory=dict)

    def url(self, *keys: str, default: str = "") -> str:
        """Return the first present ``urls.<key>`` among ``keys`` (else default)."""
        for key in keys:
            value = str(self.urls.get(key) or "").strip()
            if value:
                return value
        return default

    def site_root(self) -> str:
        """Return ``scheme://host`` for the conference site (for relative hrefs)."""
        for candidate in (self.base_url, self.virtual_url, *self.urls.values()):
            parts = urlsplit(str(candidate or ""))
            if parts.scheme and parts.netloc:
                return f"{parts.scheme}://{parts.netloc}"
        if self.site_domain:
            return f"https://{self.site_domain}"
        return ""


def _slug_from_workspace(workspace: Path, declared: object) -> str:
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return workspace.resolve().name


def _domain_from_urls(urls: dict[str, str], base_url: str, virtual_url: str) -> str:
    for candidate in (virtual_url, base_url, *urls.values()):
        host = urlsplit(str(candidate or "")).hostname or ""
        if host:
            return host.removeprefix("www.").lower()
    return ""


def _coerce_year(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=32)
def _load_cached(config_path_str: str, mtime_ns: int) -> ConferenceIdentity:
    config_path = Path(config_path_str)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConferenceIdentityError(f"conference config is not a mapping: {config_path}")
    conference = payload.get("conference")
    conference = conference if isinstance(conference, dict) else {}
    name = str(conference.get("name") or "").strip()
    if not name:
        raise ConferenceIdentityError(
            f"{config_path} must declare conference.name (e.g. 'ICML 2026')"  # conference-literal-ok: error-message example, not emitted output
        )
    raw_urls = payload.get("urls")
    urls = {
        str(key): str(value).strip()
        for key, value in (raw_urls.items() if isinstance(raw_urls, dict) else [])
        if isinstance(value, str) and str(value).strip()
    }
    base_url = urls.get("base", "")
    virtual_url = urls.get("virtual", "")
    workspace = config_path.parent
    return ConferenceIdentity(
        slug=_slug_from_workspace(workspace, conference.get("slug")),
        name=name,
        full_name=str(conference.get("full_name") or name).strip(),
        year=_coerce_year(conference.get("year")),
        base_url=base_url,
        virtual_url=virtual_url,
        site_domain=_domain_from_urls(urls, base_url, virtual_url),
        urls=urls,
    )


def load_conference_identity(workspace: str | Path) -> ConferenceIdentity:
    """Load the conference identity for ``workspace`` (a conference dir).

    ``workspace`` may be the conference directory itself or a path that contains
    ``config.yaml``. Raises :class:`ConferenceIdentityError` when no readable
    ``config.yaml`` with ``conference.name`` is found.
    """
    root = Path(workspace)
    config_path = root if root.name == "config.yaml" else root / "config.yaml"
    if not config_path.is_file():
        raise ConferenceIdentityError(
            f"conference config not found: {config_path}; reusable code must read "
            "conference identity from the workspace config, not hardcode it"
        )
    return _load_cached(str(config_path.resolve()), config_path.stat().st_mtime_ns)


def conference_name(workspace: str | Path, *, default: str | None = None) -> str:
    """Convenience: conference display name, with an optional non-raising default.

    When ``default`` is provided and the config is missing/unreadable, return the
    default instead of raising — for best-effort labels where a hard failure is
    worse than a graceful fallback. Without ``default``, propagate the error.
    """
    try:
        return load_conference_identity(workspace).name
    except ConferenceIdentityError:
        if default is not None:
            return default
        raise
