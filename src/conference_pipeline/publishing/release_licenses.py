"""Licenses a release declares, read from its own RELEASE.json.

Each export family carries a ``license`` in the conference's public export
manifest. The exporter copies them into ``RELEASE.json`` as
``family_licenses`` (family id -> SPDX id), and every file that states a
license for the release (``metadata/``, ``public-repo/LICENSE`` and its README
and manifest) takes it from there. No license is assumed: a release without
``family_licenses`` is refused.
"""

from __future__ import annotations

from typing import Any, Mapping


class ReleaseLicenseError(ValueError):
    """The release does not say, usably, which license each family carries."""


MIT_TEXT = (
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files (the \"Software\"), to deal\n"
    "in the Software without restriction, including without limitation the rights\n"
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    "copies of the Software, and to permit persons to whom the Software is\n"
    "furnished to do so, subject to the following conditions:\n\n"
    "The above copyright notice and this permission notice shall be included in all\n"
    "copies or substantial portions of the Software.\n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
    "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
    "SOFTWARE.\n"
)


def family_licenses(release: Mapping[str, Any]) -> dict[str, str]:
    """Return ``{family id: license}`` from a RELEASE.json payload, or refuse."""
    raw = release.get("family_licenses")
    if not isinstance(raw, dict) or not raw:
        raise ReleaseLicenseError("RELEASE.json has no family_licenses")
    licenses: dict[str, str] = {}
    for family, license_id in raw.items():
        if not isinstance(family, str) or not family or not isinstance(license_id, str) or not license_id.strip():
            raise ReleaseLicenseError(f"RELEASE.json family_licenses has an unusable entry for {family!r}")
        licenses[family] = license_id.strip()
    included = release.get("included_families")
    if isinstance(included, list) and sorted(included) != sorted(licenses):
        raise ReleaseLicenseError("RELEASE.json family_licenses does not name exactly the included families")
    return dict(sorted(licenses.items()))


def license_groups(release: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """Each distinct license with the families it covers, both sorted."""
    groups: dict[str, list[str]] = {}
    for family, license_id in family_licenses(release).items():
        groups.setdefault(license_id, []).append(family)
    return sorted((license_id, sorted(families)) for license_id, families in groups.items())


def license_url(license_id: str) -> str:
    return f"https://spdx.org/licenses/{license_id}.html"


def license_summary(release: Mapping[str, Any]) -> str:
    """One line, e.g. ``MIT (decks, podcasts); CC-BY-4.0 (reader_catalog)``."""
    return "; ".join(f"{license_id} ({', '.join(families)})" for license_id, families in license_groups(release))


def license_text(release: Mapping[str, Any]) -> str:
    """The text of a release's LICENSE file, one section per license."""
    groups = license_groups(release)
    parts = ["This release is licensed per artifact family, as RELEASE.json records under family_licenses.\n"]
    for license_id, families in groups:
        parts.append(f"\n## {license_id}\n\nFamilies: {', '.join(families)}.\n")
        if license_id == "MIT":
            parts.append("\nCopyright (c) the Conference Pipeline Team\n\n" + MIT_TEXT)
        else:
            parts.append(f"\nLicense text: {license_url(license_id)}\n")
    return "".join(parts)
