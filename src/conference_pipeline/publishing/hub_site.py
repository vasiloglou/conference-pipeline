"""Build the cross-conference hub from several release bundles.

v6.0-Public-Release Phase 4 (spec ``notes/phase-4-spec.md`` section 6.7). The hub reads
only bundles (through ``reader_bundle``, so every file is checked against its bundle's
``CHECKSUMS.json``). It lists every release with its DOI, dataset, site and repository, and
merges the catalogs under ``<release id>/<paper id>`` with cross-conference search.

Rows a bundle excludes (``catalog/excluded_ids.json``) and catalog files outside its
``CHECKSUMS.json`` are never read; ``hub_manifest.json`` counts both. The build fails if
the hub's row count is not the sum of the rows read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline.publishing import site_templates as tpl
from conference_pipeline.publishing.search_shards import write_search_shards
from conference_pipeline.publishing.reader_bundle import (
    ReaderBundle,
    ReaderBundleError,
    load_reader_bundle,
    semantic_block,
)
from conference_pipeline.publishing.static_site import (
    StaticSiteError,
    assert_site_output_public_safe,
    output_checksums,
    search_body,
    staged_output,
    write_json,
    write_text,
)

HUB_NAV = (("index.html", "Conferences and search"), ("catalog.html", "All papers"))


class HubSiteError(RuntimeError):
    """The hub cannot be built from these bundles."""


@dataclass(frozen=True)
class HubSiteResult:
    output_dir: Path
    manifest_path: Path
    rows_total: int


def _qualified(bundle: ReaderBundle, row: dict[str, Any]) -> dict[str, Any]:
    paper_id = str(row.get("paper_id"))
    url = row.get("official_url")
    return {
        "id": f"{bundle.bundle_id}/{paper_id}",
        "paper_id": paper_id,
        "release_id": bundle.bundle_id,
        "conference": bundle.conference_name,
        "kind": "paper",
        "title": row.get("title"),
        "text": " ".join(str(x) for x in (", ".join(row.get("authors") or []), row.get("venue"), row.get("workshop")) if x),
        "track": row.get("track"),
        "href": url if isinstance(url, str) and url.startswith("https://") else None,
    }


def _release_item(bundle: ReaderBundle, rows: int) -> str:
    links = []
    for key, label in (("public_site_url", "Site"), ("zenodo_doi", "DOI"), ("hf_dataset_url", "Dataset"), ("public_repo_url", "Data repository")):
        value = bundle.release.get(key)
        if value:
            href = value if str(value).startswith("https://") else f"https://doi.org/{value}"
            links.append(f'<a href="{escape(href)}">{label}: {escape(bundle.conference_name)}</a>')
    return (f'<li data-release="{escape(bundle.bundle_id)}"><p><strong>{escape(bundle.conference_name)}</strong> '
            f'<span class="meta">{escape(bundle.bundle_id)} · {rows:,} papers</span></p><p>{" · ".join(links)}</p></li>')


def build_hub_site(bundle_roots: Sequence[Path], output_dir: Path, *, hub_title: str, code_repo_url: str | None = None) -> HubSiteResult:
    if not bundle_roots:
        raise HubSiteError("the hub needs at least one bundle")
    try:
        bundles = [load_reader_bundle(Path(root)) for root in bundle_roots]
    except ReaderBundleError as exc:
        raise HubSiteError(str(exc)) from exc
    ids = [b.bundle_id for b in bundles]
    if len(set(ids)) != len(ids):
        raise HubSiteError("two bundles share a release id")

    rows: list[dict[str, Any]] = []
    releases: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        qualified = [_qualified(bundle, row) for row in bundle.catalog]
        rows.extend(qualified)
        releases[bundle.bundle_id] = {
            "conference": bundle.conference_name,
            "rows_read": len(qualified),
            "rows_excluded": bundle.rows_excluded,
            "files_ignored_not_in_checksums": len(bundle.ignored_not_in_checksums),
        }
    rows_total = sum(r["rows_read"] for r in releases.values())
    if rows_total != len(rows) or len({r["id"] for r in rows}) != len(rows):
        raise HubSiteError("hub coverage does not equal the sum of the bundles' public rows")

    # The hub searches catalog rows. A bundle's browser vectors index its text chunks, not
    # its catalog rows, so no bundle carries vectors the hub could search: the hub is full
    # text only, whatever the bundles' models (spec IF1). semantic_models is recorded so a
    # later bundle format with per-row vectors can be checked against it.
    semantic_models = sorted({(b["model"], b["dim"]) for b in map(semantic_block, bundles) if b})
    modes = ["text"]

    # Staged and swapped in only when complete and marker-clean, as the site is (F3, F8).
    try:
        with staged_output(Path(output_dir), [b.root for b in bundles]) as staging:
            _write_hub(staging, HubContent(bundles, releases, rows, rows_total, modes, semantic_models, hub_title, code_repo_url))
    except StaticSiteError as exc:
        raise HubSiteError(str(exc)) from exc
    final = Path(output_dir).resolve()
    return HubSiteResult(final, final / "hub_manifest.json", rows_total)


@dataclass(frozen=True)
class HubContent:
    """What the hub pages are written from (v6.0 Phase 6 D14; Phase 4 open question 5)."""

    bundles: list[ReaderBundle]
    releases: dict[str, dict[str, Any]]
    rows: list[dict[str, Any]]
    rows_total: int
    modes: list[str]
    semantic_models: list
    hub_title: str
    code_repo_url: str | None


def _catalog_item(row: dict[str, Any], *, repeated: bool) -> str:
    # A title several rows share (one paper at two workshops, at two conferences, or in two
    # releases of one) is named apart by its conference, release, track and id (6b; review
    # phase-6b-r1 F7).
    return tpl.paper_item(
        item_id=str(row["id"]), title=str(row["title"]), href=row["href"],
        meta=f'{escape(row["conference"])} · {escape(row["release_id"])}',
        distinguish=[row["conference"], row["release_id"], row.get("track"), row["paper_id"]], repeated=repeated,
    )


def _write_hub(output_dir: Path, content: HubContent) -> None:
    bundles, releases, rows, rows_total = content.bundles, content.releases, content.rows, content.rows_total
    modes, semantic_models = content.modes, content.semantic_models
    hub_title, code_repo_url = content.hub_title, content.code_repo_url
    write_text(output_dir / "assets" / "site.css", tpl.CSS)
    write_text(output_dir / "assets" / "search.js", tpl.SEARCH_JS)
    release_list = "".join(_release_item(b, releases[b.bundle_id]["rows_read"]) for b in bundles)
    code = f'<p><a href="{escape(code_repo_url)}">Source code of the pipeline</a></p>' if code_repo_url else ""
    body = (f'<h2>Conferences</h2><ul class="topics">{release_list}</ul>{code}' + search_body(modes))
    write_text(output_dir / "index.html", tpl.page(title=hub_title, heading=hub_title, body=body, site_name=hub_title,
                                                   nav=HUB_NAV, scripts=("assets/search.js",)))
    repeated = tpl.repeated_titles([r["title"] for r in rows])
    items = "".join(_catalog_item(r, repeated=tpl.title_key(r["title"]) in repeated) for r in rows)
    write_text(output_dir / "catalog.html", tpl.page(title=f"All papers: {hub_title}", heading="All papers", site_name=hub_title,
                                                     nav=HUB_NAV, body=f'<p>{rows_total:,} papers across {len(bundles)} releases.</p><ul class="papers">{items}</ul>'))
    write_json(output_dir / "catalog.json", rows)
    write_search_shards(rows, output_dir)  # sharded, under the Pages file limit (Phase 7.1)
    write_json(output_dir / "search-config.json", {"modes": modes, "semantic": None})
    manifest = {
        "schema_version": 1,
        "hub_title": hub_title,
        "releases": releases,
        "rows_total": rows_total,
        "search_modes": modes,
        "bundle_semantic_models": [{"model": m, "dim": d} for m, d in semantic_models],
    }  # no clock: the same bundles give the same hub bytes (6b review r3 F8)
    manifest["output_checksums"] = output_checksums(output_dir)
    manifest_path = output_dir / "hub_manifest.json"
    write_json(manifest_path, manifest)
    assert_site_output_public_safe(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cross-conference hub from release bundles")
    parser.add_argument("--bundle-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Conference intelligence releases")
    parser.add_argument("--code-repo-url")
    args = parser.parse_args(argv)
    try:
        result = build_hub_site(args.bundle_root, args.output_dir, hub_title=args.title, code_repo_url=args.code_repo_url)
    except HubSiteError as exc:
        print(f"error: {exc}")
        return 2
    print(f"Hub written: {result.output_dir} ({result.rows_total} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
