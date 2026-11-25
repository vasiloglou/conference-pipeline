"""Page shell, stylesheet and scripts for the public site and the hub (v6.0 Phase 4, spec 6.2).

Kept as Python strings rather than package-data files so the wheel needs no data-file
configuration (spec IF2). Every page shares one shell: a skip link, a header nav, one
``<main id="main">`` and a footer. Pages pass their body HTML already escaped.
"""

from __future__ import annotations

from html import escape

TRANSFORMERS_VERSION = "3.7.5"

NAV = (
    ("index.html", "Topics"),
    ("papers.html", "Papers"),
    ("search.html", "Search"),
    ("data.html", "Data files"),
)


def page(*, title: str, heading: str, body: str, depth: int = 0, site_name: str, nav: tuple = NAV, scripts: tuple[str, ...] = ()) -> str:
    """A full page. ``depth`` is how many directories below the site root it sits."""
    up = "../" * depth
    links = "".join(f'<li><a href="{up}{href}">{escape(label)}</a></li>' for href, label in nav)
    tags = "".join(f'<script type="module" src="{up}{src}"></script>' for src in scripts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="stylesheet" href="{up}assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
<p class="site-name"><a href="{up}index.html">{escape(site_name)}</a></p>
<nav aria-label="Site"><ul>{links}</ul></nav>
</header>
<main id="main">
<h1>{escape(heading)}</h1>
{body}
</main>
<footer><p>Public release. Paper links go to each paper's official copy; papers are not redistributed here.</p></footer>
{tags}
</body>
</html>
"""


CSS = """
:root { --ink: #1d2733; --muted: #536271; --line: #d8dee5; --accent: #1f5f99; --bg: #ffffff; --panel: #f5f7f9; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
a { color: var(--accent); }
a:focus, button:focus, input:focus, select:focus { outline: 3px solid #f2b705; outline-offset: 2px; }
.skip { position: absolute; left: -999px; top: 0; padding: 8px 12px; background: var(--ink); color: #fff; }
.skip:focus { left: 8px; z-index: 10; }
.topbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 24px; padding: 12px 32px; border-bottom: 1px solid var(--line); }
.site-name { margin: 0; font-weight: 600; }
nav ul { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 0; padding: 0; list-style: none; }
main { max-width: 960px; margin: 0 auto; padding: 24px 32px 48px; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 12px; overflow-wrap: anywhere; }
h2 { font-size: 1.35rem; margin: 28px 0 8px; overflow-wrap: anywhere; }
h3 { font-size: 1.1rem; margin: 20px 0 6px; overflow-wrap: anywhere; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 24px; color: var(--muted); margin: 0 0 16px; padding: 0; list-style: none; }
.topics { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.topics li { border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; }
.has { color: var(--muted); font-size: 0.92rem; margin: 4px 0 0; }
.absent { color: var(--muted); font-style: italic; }
video, audio { width: 100%; max-width: 100%; margin: 8px 0; }
.papers { list-style: none; padding: 0; margin: 0; }
.papers li { padding: 10px 0; border-bottom: 1px solid var(--line); }
.meta { color: var(--muted); font-size: 0.92rem; }
.narrative { border-left: 3px solid var(--line); padding-left: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; overflow-wrap: anywhere; }
main p, main li, main code, main pre, main blockquote, main dd { overflow-wrap: anywhere; }
main pre { white-space: pre-wrap; }
.search-panel { background: var(--panel); padding: 16px; border-radius: 6px; }
.search-panel input[type=search] { width: 100%; max-width: 560px; padding: 8px 10px; font-size: 1rem; }
.mode { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.mode button, .citation button { padding: 6px 12px; border: 1px solid var(--line); border-radius: 4px; background: #fff; font-size: 0.95rem; cursor: pointer; }
.mode button[aria-pressed=true] { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.status { color: var(--muted); min-height: 1.5em; }
.results { list-style: none; padding: 0; }
.results li { padding: 10px 0; border-bottom: 1px solid var(--line); }
[hidden] { display: none !important; }
@media (max-width: 767px) {
  .topbar { padding: 12px 16px; }
  main { padding: 16px 16px 40px; }
  h1 { font-size: 1.5rem; }
}
"""


SITE_JS = """
// Shared page behaviour: the citation copy button.
const button = document.querySelector('[data-copy-citation]');
if (button) {
  button.addEventListener('click', async () => {
    const text = document.getElementById('citation').textContent.trim();
    const status = document.getElementById('citation-status');
    try { await navigator.clipboard.writeText(text); status.textContent = 'Citation copied.'; }
    catch (error) { status.textContent = 'Copy failed; select the citation text instead.'; }
  });
}
"""


def title_key(title: object) -> str:
    """A title as a reader sees it: spacing collapsed, so near-copies count as repeats."""
    return " ".join(str(title).split())


def repeated_titles(titles: list[object]) -> set[str]:
    """The keys of titles more than one row shares."""
    counts: dict[str, int] = {}
    for title in titles:
        counts[title_key(title)] = counts.get(title_key(title), 0) + 1
    return {key for key, count in counts.items() if count > 1}


def paper_item(*, item_id: str, title: str, href: object, meta: str, distinguish: list[object], repeated: bool) -> str:
    """One paper in a links list, for a conference site and the hub alike (v6.0 Phase 7.1, D7).

    A title several rows share (a paper and its workshop version, one paper at two
    conferences) is named apart for screen readers by the fields in ``distinguish``.
    ``meta`` is already escaped.
    """
    shown = escape(title)
    named = f"{title} ({', '.join(str(part) for part in distinguish)})"
    label = f' aria-label="{escape(named, quote=True)}"' if repeated else ""
    head = (
        f'<a href="{escape(href)}"{label}>{shown}</a>'
        if isinstance(href, str) and href.startswith("https://")
        else f'{shown} <span class="absent">(no public copy known)</span>'
    )
    return f'<li data-paper-id="{escape(item_id)}"><p>{head}</p><p class="meta">{meta}</p></li>'


SEARCH_JS = """
// Client-side search over the sharded index in search/ (search_shards.py). Full text is
// BM25 with the constants and summation order of site_checks.bm25_rank: a query fetches
// only its terms' buckets and the doc blocks of the hits it shows. Semantic search (when
// the release has browser vectors) embeds the query with the model in search-config.json
// and scores the int8 vector shards.
const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@__TRANSFORMERS_VERSION__";
const form = document.getElementById('search-form');
const input = document.getElementById('q');
const results = document.getElementById('results');
const status = document.getElementById('search-status');
let mode = 'text';
let meta = null, config = null, extractor = null, vectors = null;
const buckets = new Map(), blocks = new Map();

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
function tokenize(text) { return (String(text || '').toLowerCase().match(/[a-z0-9]+/g)) || []; }
function fnv1a(term) {
  let h = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(term)) { h ^= byte; h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
async function getJson(path) { const r = await fetch(path); if (!r.ok) throw new Error(path); return r.json(); }

async function load() {
  if (meta) return;
  status.textContent = 'Loading the search index...';
  [config, meta] = await Promise.all([getJson('search-config.json'), getJson('search/meta.json')]);
  status.textContent = '';
}

async function bucket(term) {
  const n = fnv1a(term) & (meta.buckets - 1);
  if (!buckets.has(n)) buckets.set(n, getJson(`search/terms/${n}.json`));
  return (await buckets.get(n))[term] || [];
}

async function fullTextRanked(query) {
  const terms = tokenize(query), n = meta.n;
  if (!terms.length || !n) return [];
  const avgdl = meta.total_length / n || 1, k1 = meta.k1, b = meta.b, tf = new Map();
  for (const term of new Set(terms)) {
    const flat = await bucket(term), m = new Map();
    for (let i = 0; i < flat.length; i += 2) m.set(flat[i], flat[i + 1]);
    tf.set(term, m);
  }
  const candidates = [...new Set([...tf.values()].flatMap((m) => [...m.keys()]))].sort((x, y) => x - y), out = [];
  for (const row of candidates) {
    let score = 0;
    for (const term of terms) {
      const f = tf.get(term).get(row) || 0;
      if (!f) continue;
      const d = tf.get(term).size;
      const idf = Math.log(1 + (n - d + 0.5) / (d + 0.5));
      score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * meta.lengths[row] / avgdl));
    }
    if (score > 0) out.push([score, row]);
  }
  return out.sort((x, y) => y[0] - x[0]).map(([, row]) => row);
}

async function rowsFor(indices) {
  const size = meta.doc_block;
  return Promise.all(indices.map(async (i) => {
    const n = Math.floor(i / size);
    if (!blocks.has(n)) blocks.set(n, getJson(`search/docs/${n}.json`));
    return (await blocks.get(n))[i % size];
  }));
}

async function ensureModel() {
  if (extractor) return extractor;
  const { pipeline } = await import(TRANSFORMERS_CDN);
  extractor = await pipeline('feature-extraction', config.semantic.model, { dtype: config.semantic.dtype || 'q8' });
  return extractor;
}

async function loadVectors() {
  if (vectors) return vectors;
  const block = config.semantic.vectors, dim = block.dim, step = 4 + dim;
  const parts = await Promise.all(block.shards.map(async (name) => new DataView(await (await fetch(name)).arrayBuffer())));
  vectors = [];
  for (const view of parts) for (let offset = 0; offset + step <= view.byteLength; offset += step) {
    vectors.push([view.getFloat32(offset, true), new Int8Array(view.buffer, view.byteOffset + offset + 4, dim)]);
  }
  return vectors;
}

async function semanticSearch(query, k = 20) {
  const model = await ensureModel();
  const rows = await loadVectors();
  const output = await model(`${config.semantic.query_prefix || ''}${query}`, { pooling: 'mean', normalize: true });
  const q = output.data, dim = config.semantic.dim, scores = [];
  rows.forEach(([scale, codes], row) => {
    // A vector whose chunk the site dropped (a marker, say) maps to null: skip it before
    // ranking, so it neither renders nor takes one of the k places.
    const target = config.semantic.order[row];
    if (target === null || target === undefined) return;
    let s = 0;
    for (let j = 0; j < dim; j++) s += q[j] * codes[j];
    scores.push([s * scale, target]);
  });
  return scores.sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);
}

function rrf(...lists) {
  const score = new Map();
  for (const list of lists) list.forEach((i, rank) => score.set(i, (score.get(i) || 0) + 1 / (60 + rank)));
  return [...score.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
}

async function renderRanked(indices) {
  const shown = await rowsFor(indices.slice(0, 50));
  results.innerHTML = shown.map((row) => {
    const label = [row.kind, row.conference, row.release_id].filter(Boolean).map(escapeHtml).join(' · ');
    const title = escapeHtml(row.title || row.source_path || row.id);
    const link = row.href ? `<a href="${escapeHtml(row.href)}">${title}</a>` : title;
    return `<li><p>${link}</p><p class="meta">${label}</p></li>`;
  }).join('');
  status.textContent = indices.length ? `${indices.length} result(s).` : 'No results.';
}

for (const b of document.querySelectorAll('.mode button[data-mode]')) {
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    for (const other of document.querySelectorAll('.mode button[data-mode]')) other.setAttribute('aria-pressed', String(other === b));
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  await load();
  const query = input.value.trim();
  if (!query) { results.innerHTML = ''; status.textContent = ''; return; }
  const textRanked = await fullTextRanked(query);
  if (mode === 'text') return renderRanked(textRanked);
  try {
    status.textContent = 'Loading the semantic model (first use only)...';
    const semantic = await semanticSearch(query);
    await renderRanked(mode === 'semantic' ? semantic : rrf(textRanked, semantic));
  } catch (error) {
    status.textContent = 'Semantic search failed to load; showing full-text results. Search again to retry.';
    await renderRanked(textRanked);
  }
});
""".replace("__TRANSFORMERS_VERSION__", TRANSFORMERS_VERSION)
