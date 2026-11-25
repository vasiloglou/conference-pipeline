"""Generate the reader-facing public site of one release bundle.

v6.0-Public-Release Phase 4 (spec ``notes/phase-4-spec.md`` sections 6.2-6.5). From the
bundle alone, it writes:

- ``index.html``: the release, its statistics, the topics in manifest order;
- ``topics/<id>.html``: deck, videos, podcast, narrative and the papers the topic cites;
- ``papers.html``: every public catalog row, linked to its official copy;
- ``data.html``: every file in ``CHECKSUMS.json``, the only page that links control files;
- ``search.html``: full-text search, plus semantic search when the bundle has a browser
  index.

Every linked bundle file is copied under ``files/`` after its sha256 is checked against
``CHECKSUMS.json``; videos stay on the CDN their sidecars name. Nothing under
``public-repo/`` or ``site/`` is ever copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from conference_pipeline._security.private_markers import (
    has_publishing_marker,
)
from conference_pipeline.publishing import site_templates as tpl
from conference_pipeline.publishing.reader_bundle import (
    ReaderBundle,
    ReaderBundleError,
    Topic,
    load_reader_bundle,
    semantic_block,
)
from conference_pipeline.publishing.reader_families import talk_display_name
from conference_pipeline.publishing.search_shards import write_search_shards, write_vector_shards
from conference_pipeline.publishing.site_checks import render_markdown
from conference_pipeline.publishing.site_output import (  # noqa: F401 -- re-exported for Phase 4's callers
    StaticSiteError,
    _check_output_dir,
    assert_site_output_public_safe,
    check_output_dir,
    output_checksums,
    read_json,
    staged_output,
    utc_now,
    write_json,
    write_text,
)

NEVER_COPIED = ("public-repo/", "site/", "index/")
EXCLUDED_IDS = "catalog/excluded_ids.json"
PUBLIC_CATALOG = "papers.jsonl"  # the public catalog rows, written by the site (F5)
BROWSER_QUERY_PROMPT = "task: search result | query: "  # public_index._GEMMA_QUERY_PROMPT
ABSENT = '<p class="absent">{what}: not in this release.</p>'


@dataclass(frozen=True)
class PublicSiteConfig:
    bundle_root: Path
    output_dir: Path
    receipt_path: Path | None = None


@dataclass(frozen=True)
class PublicSiteResult:
    output_dir: Path
    manifest_path: Path
    receipt_path: Path
    output_checksums: dict[str, str]


# --- copying ------------------------------------------------------------------------


def _copyable(rel: str) -> bool:
    if rel.startswith("catalog/") and (rel.endswith(".jsonl") or rel == EXCLUDED_IDS):
        # The raw catalog may hold excluded rows, and the exclusion list names them. The
        # site publishes the public rows it renders as papers.jsonl instead (F5).
        return False
    return not rel.startswith(NEVER_COPIED) and "/.git/" not in f"/{rel}"


def _copy_bundle_files(bundle: ReaderBundle, output_dir: Path) -> set[str]:
    copied: set[str] = set()
    for rel in sorted(bundle.checksums):
        if not _copyable(rel) or rel.endswith(".cdn.json"):
            continue
        source = bundle.verified(rel)
        target = output_dir / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.add(rel)
    return copied


def _file(rel: str, depth: int) -> str:
    return "../" * depth + "files/" + rel


# --- pages --------------------------------------------------------------------------


def _has_line(topic: Topic) -> str:
    parts = [
        "slides" if topic.deck_html or topic.deck_pdf or topic.deck_pptx or topic.deck_text else None,
        f"{len(topic.videos)} video(s)" if topic.videos else None,
        "podcast" if topic.podcast else None,
        "narrative" if topic.narrative else None,
        f"{len(topic.papers)} paper(s)" if topic.papers else None,
    ]
    present = [p for p in parts if p]
    return ", ".join(present) if present else "no reader content in this release"


def _citation(bundle: ReaderBundle) -> str:
    doi = bundle.release.get("zenodo_doi")
    name = bundle.conference_name or bundle.bundle_id
    return f"{name}: public release {bundle.bundle_id}." + (f" DOI {doi}." if doi else "")


LIMITATIONS_MD = "limitations/limitations.md"


def _talks_section(talks: Any) -> str:
    """The release's talks per kind (Phase 5.1's talks block, display names from Phase 6)."""
    if not isinstance(talks, dict) or not talks.get("by_kind"):
        return ""
    rows = "".join(
        f'<tr><th scope="row">{escape(str(row.get("display") or talk_display_name(kind)))}</th><td>{int(row.get("listed", 0)):,}</td>'
        f'<td>{int(row.get("admitted", 0)):,}</td><td>{int(row.get("excluded", 0)):,}</td></tr>'
        for kind, row in sorted(talks["by_kind"].items())
    )
    excluded = ", ".join(f"{escape(str(reason))} {int(count)}" for reason, count in sorted((talks.get("excluded") or {}).items()))
    return (
        f'<h2>Talks</h2><p>{int(talks.get("admitted", 0)):,} of {int(talks.get("listed", 0)):,} recorded talks are in the corpus'
        + (f" (not included: {excluded})" if excluded else "") + ".</p>"
        '<table class="talks"><thead><tr><th scope="col">Kind</th><th scope="col">Listed</th>'
        f'<th scope="col">In the corpus</th><th scope="col">Not included</th></tr></thead><tbody>{rows}</tbody></table>'
    )


def _index_body(bundle: ReaderBundle) -> str:
    facts = [f"<li>Release {escape(bundle.bundle_id)}</li>"]
    stats = bundle.statistics or {}
    total = (stats.get("papers") or {}).get("total")
    if total is not None:
        label = (stats.get("corpus_version") or {}).get("label")
        facts.append(f"<li>{int(total):,} papers in the catalog" + (f" (statistics as of {escape(label)})" if label else "") + "</li>")
    if stats.get("d11_note"):
        facts.append(f'<li class="note">{escape(str(stats["d11_note"]))}</li>')
    if bundle.listed(LIMITATIONS_MD):
        facts.append('<li><a href="limitations.html">Limitations of this release</a></li>')
    links = [("zenodo_doi", "DOI"), ("hf_dataset_url", "Dataset"), ("public_repo_url", "Data repository")]
    for key, label in links:
        value = bundle.release.get(key)
        if value:
            href = value if str(value).startswith("https://") else f"https://doi.org/{value}"
            facts.append(f'<li><a href="{escape(href)}">{label}</a></li>')
    topics = "".join(
        f'<li data-topic="{escape(t.topic_id)}"><a href="topics/{escape(t.topic_id)}.html">{escape(t.title)}</a>'
        f'<p class="has">{escape(_has_line(t))}</p></li>'
        for t in bundle.topics
    ) or "<li>This release has no topics.</li>"
    executive = "".join(
        f'<li><a href="{_file(t.deck_html, 0)}">{escape(t.title)}</a></li>' for t in bundle.executive if t.deck_html
    )
    citation = escape(_citation(bundle))
    return (
        f'<ul class="facts">{"".join(facts)}</ul>'
        + _talks_section(stats.get("talks"))
        + f'<h2>Topics</h2><ul class="topics">{topics}</ul>'
        + (f'<h2>Executive summaries</h2><ul class="topics">{executive}</ul>' if executive else "")
        + f'<h2>Cite this release</h2><div class="citation"><p id="citation">{citation}</p>'
        '<button type="button" data-copy-citation>Copy citation</button><p id="citation-status" class="status" aria-live="polite"></p></div>'
    )


_RELATIVE_LINK = re.compile(r'<a href="(?!https://|#)[^"]*"[^>]*>(?P<text>.*?)</a>', re.S)


def _unlink_workspace_paths(html: str) -> str:
    """A narrative's relative links name workspace files (analysis drafts, cache paths) that
    no release carries; they become their text, as relink drops an unmapped Drive link (6b)."""
    return _RELATIVE_LINK.sub(lambda m: m.group("text"), html)


def _paper_list(rows: list[dict[str, Any]]) -> str:
    repeated = tpl.repeated_titles([row.get("title") or row.get("paper_id") for row in rows])
    items = "".join(_paper_item(r, repeated=tpl.title_key(r.get("title") or r.get("paper_id")) in repeated) for r in rows)
    return f'<ul class="papers">{items}</ul>'


def _paper_item(row: dict[str, Any], *, repeated: bool = False) -> str:
    # A title several rows share (a paper and its workshop version) is named apart by its
    # track and id, so a links list can tell them apart (6b, NeurIPS 2025).
    meta = " · ".join(escape(str(x)) for x in (", ".join(row.get("authors") or []), row.get("track"), row.get("workshop") or row.get("venue")) if x)
    return tpl.paper_item(
        item_id=str(row.get("paper_id")), title=str(row.get("title") or row.get("paper_id")), href=row.get("official_url"),
        meta=meta, distinguish=[row.get("track"), row.get("paper_id")], repeated=repeated,
    )


def _topic_body(bundle: ReaderBundle, topic: Topic, papers_by_id: dict[str, dict[str, Any]]) -> str:
    deck = []
    if topic.deck_html:
        deck.append(f'<p><a href="{_file(topic.deck_html, 1)}">Open the slides (HTML)</a></p>')
    if topic.deck_pdf:
        deck.append(f'<p><a href="{_file(topic.deck_pdf, 1)}">Download the slides (PDF)</a></p>')
    else:
        deck.append(ABSENT.format(what="Slides as PDF"))
    if topic.deck_text:
        deck.append(f'<p><a href="{_file(topic.deck_text, 1)}">Read the slides (text)</a></p>')
    if topic.deck_pptx:
        deck.append(f'<p><a href="{_file(topic.deck_pptx, 1)}">Download the slides (PowerPoint)</a></p>')
    videos = "".join(
        f'<h3>{escape(v.label)}</h3><video controls preload="none"><source src="{escape(v.url)}" type="{escape(v.mime_type)}"></video>'
        f'<p><a href="{escape(v.url)}">Download: {escape(v.label)}</a></p>'
        for v in topic.videos
    ) or ABSENT.format(what="Video")
    if topic.podcast:
        src = topic.podcast if topic.podcast.startswith("https://") else _file(topic.podcast, 1)
        podcast = f'<audio controls preload="none" src="{escape(src)}"></audio><p><a href="{escape(src)}">Download the podcast</a></p>'
    else:
        podcast = ABSENT.format(what="Podcast")
    narrative = (
        f'<article class="narrative">{_unlink_workspace_paths(render_markdown(bundle.read_text(topic.narrative), heading_offset=2,
                                                       reserved={bundle.conference_name or bundle.bundle_id: "../index.html"}))}</article>'
        if topic.narrative else ABSENT.format(what="Narrative")
    )
    rows = [papers_by_id[p] for p in topic.papers if p in papers_by_id]
    papers = _paper_list(rows) if rows else ABSENT.format(what="Cited papers")
    return (
        f'<section data-item="deck"><h2>Slides</h2>{"".join(deck)}</section>'
        f'<section data-item="videos"><h2>Videos</h2>{videos}</section>'
        f'<section data-item="podcast"><h2>Podcast</h2>{podcast}</section>'
        f'<section data-item="narrative"><h2>Narrative</h2>{narrative}</section>'
        f'<section data-item="papers"><h2>Papers this topic cites</h2>{papers}</section>'
    )


def _papers_body(bundle: ReaderBundle) -> str:
    """The catalog page; a release with no catalog says so rather than "0 papers" (IF8)."""
    if not bundle.catalog_files:
        return '<p class="absent">Papers: not in this release.</p>'
    count = len(bundle.catalog)
    return (f'<p>{count:,} {"paper" if count == 1 else "papers"}. Each title links its official copy. '
            f'<a href="{PUBLIC_CATALOG}">Download the catalog (JSON Lines)</a>.</p>'
            + _paper_list(bundle.catalog))


def _data_body(bundle: ReaderBundle, copied: set[str]) -> str:
    rows = []
    for rel, entry in sorted(bundle.checksums.items()):
        if not _copyable(rel):
            continue
        name = f'<a href="files/{escape(rel)}">{escape(rel)}</a>' if rel in copied else escape(rel)
        rows.append(f"<tr><td>{name}</td><td>{int(entry.get('size_bytes') or 0):,}</td><td><code>{escape(str(entry.get('sha256') or ''))}</code></td></tr>")
    return (
        "<p>Every file in this release, with its size and sha256. Control files (release metadata, "
        "checksums, manifests) live here, apart from the reading pages.</p>"
        f'<table><thead><tr><th scope="col">File</th><th scope="col">Bytes</th><th scope="col">sha256</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def search_body(modes: list[str]) -> str:
    if not modes:  # nothing to search: no index and no catalog (IF8)
        return '<p class="absent">Search: not in this release.</p>'
    labels = {"text": "Full text", "semantic": "Semantic", "both": "Both"}
    buttons = "".join(
        f'<button type="button" data-mode="{m}" aria-pressed="{"true" if i == 0 else "false"}">{labels[m]}</button>'
        for i, m in enumerate(modes)
    )
    note = (
        "<p>This release has full-text search and semantic search.</p>" if "semantic" in modes
        else "<p>This release has full-text search only.</p>"
    )
    return (
        f'<section class="search-panel" aria-labelledby="search-heading"><h2 id="search-heading">Search this release</h2>{note}'
        '<form id="search-form" role="search"><label for="q">Search terms</label><br>'
        '<input id="q" name="q" type="search" autocomplete="off"> <button type="submit">Search</button>'
        f'<div class="mode" role="group" aria-label="Search mode">{buttons}</div></form>'
        '<p id="search-status" class="status" aria-live="polite"></p><ul id="results" class="results"></ul></section>'
    )


# --- search data --------------------------------------------------------------------


def _chunk_kind(source_path: str) -> tuple[str, str | None]:
    if source_path.startswith("narratives/"):
        return "narrative", f"topics/{Path(source_path).stem}.html"
    if source_path.startswith("audio/"):
        return "podcast", None
    if source_path.startswith("decks/"):
        return "deck", None
    if source_path.startswith("transcripts/"):
        return "talk", None
    return "file", None


def _search_rows(bundle: ReaderBundle, copied: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    chunk_rows: dict[str, int] = {}
    if bundle.listed("index/corpus_chunks.jsonl"):
        for line in bundle.read_text("index/corpus_chunks.jsonl").split("\n"):
            if not line.strip():
                continue
            chunk = json.loads(line)
            text, source = str(chunk.get("text") or ""), str(chunk.get("source_path") or "")
            # Only text of a file the site publishes is searchable: a chunk of the raw
            # catalog or the exclusion list would publish excluded rows (phase-4-r2 F3).
            if source not in copied or has_publishing_marker(text) or has_publishing_marker(source):
                continue
            kind, href = _chunk_kind(source)
            if href is None and source in copied:
                href = f"files/{source}"
            chunk_rows[str(chunk.get("chunk_id"))] = len(rows)
            rows.append({"id": chunk.get("chunk_id"), "kind": kind, "title": source, "source_path": source, "text": text, "href": href})
    for row in bundle.catalog:
        url = row.get("official_url")
        rows.append({
            "id": row.get("paper_id"), "kind": "paper", "title": row.get("title"), "source_path": "catalog/papers.jsonl",
            "text": " ".join(str(x) for x in (", ".join(row.get("authors") or []), row.get("venue"), row.get("workshop")) if x),
            "href": url if isinstance(url, str) and url.startswith("https://") else None,
        })
    return rows, chunk_rows


def _semantic_config(bundle: ReaderBundle, chunk_rows: dict[str, int], output_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The page's semantic search config, or None and why it is not offered."""
    block = semantic_block(bundle)
    if block is None:
        return None, None
    order_rel = block["order"]
    order_ids = bundle.read_json(order_rel) if bundle.listed(order_rel) else list(chunk_rows)
    source = bundle.verified(block["blob"])
    # A checksummed blob can still be the wrong shape for its manifest (F7).
    if not isinstance(order_ids, list) or source.stat().st_size != 4 * block["dim"] * len(order_ids):
        return None, "the browser index blob does not match its manifest's dim and row count"
    if not any(chunk_rows.get(str(cid)) is not None for cid in order_ids):
        return None, "no indexed chunk is in this site's search rows"
    # The float blob is published on HF; the site carries int8 shards under the Pages
    # file limit (v6.0 Phase 7.1, spec 6.4, D8).
    vectors = np.fromfile(source, dtype="<f4").reshape(len(order_ids), block["dim"])
    shards = write_vector_shards(vectors, output_dir)
    return {
        "model": block["model"], "dim": block["dim"], "digest": block["digest"], "dtype": "q8",
        "vectors": shards, "query_prefix": BROWSER_QUERY_PROMPT,
        "order": [chunk_rows.get(str(cid)) for cid in order_ids],
    }, None


# --- build --------------------------------------------------------------------------


def build_public_site(config: PublicSiteConfig) -> PublicSiteResult:
    try:
        bundle = load_reader_bundle(config.bundle_root)
    except ReaderBundleError as exc:
        raise StaticSiteError(str(exc)) from exc
    final = _check_output_dir(config.output_dir, [bundle.root])
    receipt_path = (config.receipt_path or final.parent / "public_site_receipt.json").resolve()
    if receipt_path == final or final in receipt_path.parents:
        raise StaticSiteError("public site receipt path must not be inside the deployable site directory")
    # Built in a staging directory and swapped in only when every check passed: a refused
    # build leaves no partial site, and a previous site stays whole (IF7, F8). The receipt
    # is written beside its destination before the swap, so an unwritable receipt fails
    # the build while the previous site is still in place (phase-4-r2 F6); the final
    # os.replace is a rename within that one directory.
    receipt_tmp = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    try:
        with staged_output(final, [bundle.root]) as staging:
            _build(bundle, staging)
            checksums = output_checksums(staging)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_tmp.write_text(json.dumps({
                "schema_version": 1,
                "receipt_type": "conference_pipeline.public_site",
                "status": "succeeded",
                "exit_code": 0,
                "bundle_id": bundle.bundle_id,
                "output_dir": str(final),
                "output_checksums": checksums,
                "generated_at": utc_now(),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except ReaderBundleError as exc:
        receipt_tmp.unlink(missing_ok=True)
        raise StaticSiteError(str(exc)) from exc
    except BaseException:
        receipt_tmp.unlink(missing_ok=True)
        raise
    os.replace(receipt_tmp, receipt_path)
    return PublicSiteResult(final, final / "site_manifest.json", receipt_path, checksums)


def _build(bundle: ReaderBundle, output_dir: Path) -> None:
    site_name = bundle.conference_name or bundle.bundle_id
    copied = _copy_bundle_files(bundle, output_dir)
    papers_by_id = {str(r.get("paper_id")): r for r in bundle.catalog}
    write_text(output_dir / "assets" / "site.css", tpl.CSS)
    write_text(output_dir / "assets" / "site.js", tpl.SITE_JS)
    write_text(output_dir / "assets" / "search.js", tpl.SEARCH_JS)
    pages = {
        "index.html": tpl.page(title=f"{site_name}: public release", heading=site_name, body=_index_body(bundle),
                               site_name=site_name, scripts=("assets/site.js",)),
        "papers.html": tpl.page(title=f"Papers: {site_name}", heading="Papers", site_name=site_name,
                                body=_papers_body(bundle)),
        "data.html": tpl.page(title=f"Data files: {site_name}", heading="Data files", site_name=site_name, body=_data_body(bundle, copied)),
    }
    if bundle.listed(LIMITATIONS_MD):
        pages["limitations.html"] = tpl.page(
            title=f"Limitations: {site_name}", heading="Limitations", site_name=site_name,
            body=f'<article class="narrative">{render_markdown(bundle.read_text(LIMITATIONS_MD), heading_offset=1)}</article>')
    for topic in bundle.topics:
        pages[f"topics/{topic.topic_id}.html"] = tpl.page(
            title=f"{topic.title}: {site_name}", heading=topic.title, depth=1, site_name=site_name,
            body=_topic_body(bundle, topic, papers_by_id))
    rows, chunk_rows = _search_rows(bundle, copied)
    semantic, semantic_reason = _semantic_config(bundle, chunk_rows, output_dir) if rows else (None, None)
    modes = (["text", "semantic", "both"] if semantic else ["text"]) if rows else []
    pages["search.html"] = tpl.page(title=f"Search: {site_name}", heading="Search", site_name=site_name,
                                    body=search_body(modes), scripts=("assets/search.js",) if modes else ())
    for rel, text in pages.items():
        write_text(output_dir / rel, text)
    if bundle.catalog_files:
        write_text(output_dir / PUBLIC_CATALOG, "".join(json.dumps(r, sort_keys=True) + "\n" for r in bundle.catalog))
    # The rows are a sharded index, so no file is over the Pages limit (Phase 7.1, 6.3).
    write_search_shards(rows, output_dir)
    write_json(output_dir / "search-config.json", {"modes": modes, "semantic": semantic})

    semantic_status: dict[str, Any] = {"status": "available" if semantic else "not_in_this_release",
                                       "model": semantic["model"] if semantic else None, "dim": semantic["dim"] if semantic else None}
    if semantic_reason:
        semantic_status["reason"] = semantic_reason
    manifest = {
        "schema_version": "2.0.0",
        "bundle_id": bundle.bundle_id,
        "conference_name": site_name,
        "source_sha": bundle.release.get("source_sha"),
        "public_export_manifest_sha256": bundle.release.get("export_manifest_sha256"),
        "pages": sorted(pages),
        "topics": [t.topic_id for t in bundle.topics],
        "catalog_rows": len(bundle.catalog),
        "search_modes": modes,
        "semantic_search": semantic_status,
        "files_copied": len(copied),
        "generated_at": utc_now(),
    }
    manifest["output_checksums"] = output_checksums(output_dir)
    write_json(output_dir / "site_manifest.json", manifest)
    assert_site_output_public_safe(output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a public release static site")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_public_site(PublicSiteConfig(args.bundle_root, args.output_dir, args.receipt_path))
    except StaticSiteError as exc:
        print(f"error: {exc}")
        return 2
    print(f"Public site written: {result.output_dir}")
    print(f"Receipt: {result.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
