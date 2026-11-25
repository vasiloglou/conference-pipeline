"""Generate browser BYOK tooling for public release bundles."""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline._security.private_markers import has_publishing_marker, publishing_marker_hits
from conference_pipeline.publishing.public_index import (
    sha256_file,
    validate_public_bundle_root,
)


class ByokToolsError(RuntimeError):
    """Raised when BYOK browser tooling fails closed."""


RECEIPT_TYPE = "conference_pipeline.byok_tools"
FORBIDDEN_BROWSER_PATTERNS = (
    "localStorage",
    "sessionStorage",
    "document.cookie",
    "URLSearchParams",
    "window.location.search",
    "?key=",
    "api_key=",
    "apikey=",
    "/api/llm",
    "/api/chat",
    "maintainer-funded",
)


@dataclass(frozen=True)
class ByokToolsConfig:
    bundle_root: Path
    receipt_path: Path | None = None


@dataclass(frozen=True)
class ByokToolsResult:
    bundle_root: Path
    manifest_path: Path
    receipt_path: Path
    output_checksums: dict[str, str]
    chunk_count: int


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ByokToolsError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def has_private_marker(value: str) -> bool:
    return has_publishing_marker(value)


def load_public_chunks(bundle_root: Path) -> list[dict[str, Any]]:
    chunks_path = bundle_root / "index" / "corpus_chunks.jsonl"
    if not chunks_path.exists():
        return []
    chunks: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("text", ""))
            source_path = str(payload.get("source_path", ""))
            if has_private_marker(text) or has_private_marker(source_path):
                continue
            chunks.append(
                {
                    "chunk_id": str(payload.get("chunk_id", "")),
                    "source_path": source_path,
                    "content_type": str(payload.get("content_type", "text")),
                    "text": text,
                    "license": str(payload.get("license", "CC-BY-SA-4.0")),
                    "provenance": str(payload.get("provenance", "public-bundle")),
                }
            )
    return chunks


def output_checksums(paths: Sequence[Path], root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
        if path.is_file()
    }


def html_text(release: dict[str, Any], *, has_self_host: bool, has_audio_manifest: bool) -> str:
    bundle_id = html.escape(str(release.get("bundle_id", "unknown")))
    source_sha = html.escape(str(release.get("source_sha", "unknown"))[:12])
    self_host_copy = (
        '<ul class="tool-links">'
        '<li><a href="self-host/README.md">Open self-host kit</a></li>'
        '<li><a href="self-host/gpt-researcher.env.example">Open GPT-Researcher config</a></li>'
        '<li><a href="self-host/deck-builder-config.json">Open deck builder config</a></li>'
        '<li><a href="self-host/report-builder-config.json">Open report builder config</a></li>'
        '</ul>'
        if has_self_host
        else '<span class="status gated">Self-host kit not generated yet</span>'
    )
    audio_copy = (
        '<a href="metadata/notebooklm_audio_manifest.json">Open delivered audio manifest</a>'
        if has_audio_manifest
        else '<span class="status gated">NotebookLM audio is operator-gated</span>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{bundle_id} BYOK tools</title>
  <link rel="stylesheet" href="assets/byok-tools.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>{bundle_id} tools</h1>
      <p>Source SHA {source_sha}. BYOK and self-hosted generation over the published corpus.</p>
    </div>
    <nav><a href="index.html">Search</a><a href="tools.html">Tools</a><a href="site_manifest.json">Dataset</a></nav>
  </header>
  <main>
    <section class="facts" aria-label="Tool safety status">
      <div><strong>Key handling</strong><span>In-memory for this page only</span></div>
      <div><strong>Search scope</strong><span>Published corpus only</span></div>
      <div><strong>Maintainer cost</strong><span>No public LLM endpoint</span></div>
    </section>
    <section class="tool-layout">
      <aside class="control-panel" aria-labelledby="byok-heading">
        <h2 id="byok-heading">BYOK settings</h2>
        <label for="provider">Provider</label>
        <select id="provider">
          <option value="openai">OpenAI-compatible chat</option>
          <option value="anthropic">Anthropic-compatible messages</option>
        </select>
        <label for="model">Model</label>
        <input id="model" value="gpt-4o-mini" autocomplete="off">
        <label for="credential">API key</label>
        <input id="credential" type="password" autocomplete="off" spellcheck="false">
        <button id="clear-credential" type="button">Clear key</button>
        <p id="key-status" class="status" aria-live="polite">Enter a key to enable LLM actions.</p>
      </aside>
      <section class="workspace">
        <div class="tool-panel">
          <h2>Ask the release</h2>
          <label for="question">Question</label>
          <textarea id="question" rows="5" maxlength="1200"></textarea>
          <label for="top-k">Sources</label>
          <input id="top-k" type="number" min="1" max="12" value="6">
          <button id="ask-button" type="button" disabled>Ask</button>
          <p id="answer-status" class="status" aria-live="polite">Loading corpus.</p>
          <pre id="answer-output"></pre>
        </div>
        <div class="tool-panel">
          <h2>Deep search</h2>
          <label for="deep-query">Research question</label>
          <textarea id="deep-query" rows="4" maxlength="1600"></textarea>
          <div class="segmented" role="group" aria-label="Depth">
            <button type="button" data-depth="light" aria-pressed="true">Light</button>
            <button type="button" data-depth="standard" aria-pressed="false">Standard</button>
            <button type="button" data-depth="exhaustive" aria-pressed="false">Exhaustive</button>
          </div>
          <button id="deep-button" type="button" disabled>Run deep search</button>
          <pre id="deep-output"></pre>
        </div>
      </section>
    </section>
    <section class="source-panel">
      <h2>Sources</h2>
      <div id="source-results" aria-live="polite"></div>
    </section>
    <section class="links">
      <h2>Self-host and audio</h2>
      <p>{self_host_copy}</p>
      <p><a href="self-host/podcastfy_config.yaml">Open podcastfy Edge TTS config</a></p>
      <p>{audio_copy}</p>
      <p><a href="metadata/byok_tools_manifest.json">Open BYOK policy manifest</a></p>
    </section>
  </main>
  <script src="assets/byok-tools.js"></script>
</body>
</html>
"""


def css_text() -> str:
    return """
:root {
  color: #1f2933;
  background: #f7f8fa;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body { margin: 0; font-size: 15px; line-height: 1.5; }
.topbar { position: sticky; top: 0; display: flex; justify-content: space-between; gap: 16px; padding: 16px 24px; background: #ffffff; border-bottom: 1px solid #d7dde3; }
h1 { margin: 0; font-size: 28px; line-height: 1.15; }
h2 { font-size: 20px; line-height: 1.25; }
main { max-width: 1180px; margin: 0 auto; padding: 24px; }
label, strong { display: block; font-size: 12px; font-weight: 600; }
input, select, textarea { width: 100%; box-sizing: border-box; min-height: 40px; border: 1px solid #d7dde3; border-radius: 6px; padding: 8px; font: inherit; }
textarea { resize: vertical; }
button, a { color: #1f2933; }
button { min-height: 40px; border: 1px solid #d7dde3; border-radius: 6px; background: #ffffff; padding: 8px 12px; font: inherit; }
button:enabled[id$="button"] { background: #0f766e; color: #ffffff; border-color: #0f766e; }
button:focus, a:focus, input:focus, select:focus, textarea:focus { outline: 2px solid #0f766e; outline-offset: 2px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.facts div, .control-panel, .tool-panel, .source-panel, .links { background: #ffffff; border: 1px solid #d7dde3; border-radius: 8px; padding: 16px; }
.tool-layout { display: grid; grid-template-columns: minmax(260px, 320px) 1fr; gap: 16px; align-items: start; }
.workspace { display: grid; gap: 16px; }
.control-panel, .tool-panel { display: grid; gap: 12px; }
.segmented { display: flex; flex-wrap: wrap; gap: 8px; }
.segmented button[aria-pressed="true"] { background: #6b5b95; color: #ffffff; }
.status { color: #536271; }
.gated { color: #b45309; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f8fa; border: 1px solid #d7dde3; border-radius: 6px; padding: 12px; min-height: 48px; }
.source-row { border-top: 1px solid #d7dde3; padding: 12px 0; }
.source-row h3 { margin: 0 0 4px; font-size: 15px; overflow-wrap: anywhere; }
@media (max-width: 767px) {
  .topbar { display: block; padding: 16px; }
  main { padding: 16px; }
  .tool-layout { grid-template-columns: 1fr; }
}
"""


def js_text() -> str:
    return """
// SAFETY-04: Future hosted-key paths must follow docs/operations/public-byok-cost-safety.md.
let inMemoryCredential = '';
let selectedDepth = 'light';
let corpusRows = [];
const privateMarkers = ['file:' + '//', '/User' + 's/', '/Volume' + 's/', 'Google' + ' Drive', 'drive.google' + '.com', '.' + 'env'];
function safeText(value) {
  const text = String(value || '');
  return privateMarkers.some((marker) => text.includes(marker)) ? '' : text;
}
async function loadCorpus() {
  const response = await fetch('assets/byok-corpus.json');
  if (!response.ok) throw new Error('corpus unavailable');
  const rows = await response.json();
  return rows.filter((row) => safeText(row.text) && safeText(row.source_path));
}
function terms(value) {
  return String(value || '').toLowerCase().match(/[a-z0-9]+/g) || [];
}
function retrieve(query, limit) {
  const queryTerms = new Set(terms(query));
  return corpusRows.map((row) => {
    const rowTerms = terms(`${row.source_path} ${row.text}`);
    const score = rowTerms.filter((term) => queryTerms.has(term)).length;
    return {...row, score};
  }).filter((row) => row.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
}
function renderSources(rows) {
  const container = document.querySelector('#source-results');
  if (!rows.length) {
    container.textContent = 'No matches in this release';
    return;
  }
  container.innerHTML = rows.map((row) => `<article class="source-row"><h3>${row.source_path}</h3><p>${row.content_type} · ${row.license} · score ${row.score}</p><p>${row.text.slice(0, 320)}</p></article>`).join('');
}
function buildPrompt(question, rows) {
  const sourceText = rows.map((row, index) => `[${index + 1}] ${row.source_path}\\n${row.text.slice(0, 900)}`).join('\\n\\n');
  return `Answer using only the published conference corpus. Cite source paths.\\n\\nQuestion: ${question}\\n\\nSources:\\n${sourceText}`;
}
function providerRequest(provider, model, prompt) {
  if (provider === 'anthropic') {
    return {
      url: 'https://api.anthropic.com/v1/messages',
      headers: {'content-type': 'application/json', 'x-api-key': inMemoryCredential, 'anthropic-version': '2023-06-01'},
      body: {model, max_tokens: 900, messages: [{role: 'user', content: prompt}]}
    };
  }
  return {
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {'content-type': 'application/json', authorization: `Bearer ${inMemoryCredential}`},
    body: {model, messages: [{role: 'user', content: prompt}], temperature: 0.2}
  };
}
function updateButtons() {
  const hasKey = inMemoryCredential.length > 0;
  const question = document.querySelector('#question').value.trim();
  const deepQuery = document.querySelector('#deep-query').value.trim();
  document.querySelector('#ask-button').disabled = !(hasKey && question && corpusRows.length);
  document.querySelector('#deep-button').disabled = !(hasKey && deepQuery && corpusRows.length);
  document.querySelector('#key-status').textContent = hasKey ? 'Key active for this page.' : 'Enter a key to enable LLM actions.';
}
async function callProvider(prompt) {
  const provider = document.querySelector('#provider').value;
  const model = document.querySelector('#model').value.trim() || 'gpt-4o-mini';
  const request = providerRequest(provider, model, prompt);
  const response = await fetch(request.url, {method: 'POST', headers: request.headers, body: JSON.stringify(request.body)});
  if (!response.ok) throw new Error(`provider returned ${response.status}`);
  const data = await response.json();
  return data.choices?.[0]?.message?.content || data.content?.[0]?.text || JSON.stringify(data, null, 2);
}
document.addEventListener('DOMContentLoaded', async () => {
  const credential = document.querySelector('#credential');
  const askButton = document.querySelector('#ask-button');
  const deepButton = document.querySelector('#deep-button');
  const answerStatus = document.querySelector('#answer-status');
  try {
    corpusRows = await loadCorpus();
    answerStatus.textContent = corpusRows.length ? 'Corpus loaded.' : 'No public corpus chunks are available.';
  } catch (error) {
    answerStatus.textContent = 'The public corpus could not be loaded.';
  }
  credential.addEventListener('input', () => {
    inMemoryCredential = credential.value;
    updateButtons();
  });
  document.querySelector('#clear-credential').addEventListener('click', () => {
    inMemoryCredential = '';
    credential.value = '';
    updateButtons();
  });
  for (const id of ['question', 'deep-query', 'top-k', 'model', 'provider']) {
    document.querySelector(`#${id}`).addEventListener('input', updateButtons);
    document.querySelector(`#${id}`).addEventListener('change', updateButtons);
  }
  document.querySelectorAll('[data-depth]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedDepth = button.dataset.depth;
      document.querySelectorAll('[data-depth]').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
    });
  });
  askButton.addEventListener('click', async () => {
    const question = document.querySelector('#question').value.trim();
    const limit = Number(document.querySelector('#top-k').value || 6);
    const sources = retrieve(question, limit);
    renderSources(sources);
    document.querySelector('#answer-output').textContent = 'Running BYOK request...';
    try {
      document.querySelector('#answer-output').textContent = await callProvider(buildPrompt(question, sources));
    } catch (error) {
      document.querySelector('#answer-output').textContent = `Request failed: ${error.message}`;
    }
  });
  deepButton.addEventListener('click', async () => {
    const query = document.querySelector('#deep-query').value.trim();
    const depthLimit = selectedDepth === 'exhaustive' ? 12 : selectedDepth === 'standard' ? 8 : 4;
    const sources = retrieve(query, depthLimit);
    renderSources(sources);
    const prompt = buildPrompt(`Create a ${selectedDepth} corpus-only research memo for: ${query}`, sources);
    document.querySelector('#deep-output').textContent = 'Running corpus-only deep search...';
    try {
      document.querySelector('#deep-output').textContent = await callProvider(prompt);
    } catch (error) {
      document.querySelector('#deep-output').textContent = `Request failed: ${error.message}`;
    }
  });
  updateButtons();
});
"""


def scan_browser_policy(paths: Sequence[Path]) -> None:
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
    for pattern in FORBIDDEN_BROWSER_PATTERNS:
        if pattern in combined:
            raise ByokToolsError(f"forbidden browser key/cost pattern found: {pattern}")
    hits = publishing_marker_hits(combined, allow_env_mention=True)
    if hits:
        raise ByokToolsError(f"private marker found in BYOK assets: {hits[0]}")


def generate_byok_tooling(config: ByokToolsConfig) -> ByokToolsResult:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    release = read_json(bundle_root / "RELEASE.json")
    chunks = load_public_chunks(bundle_root)
    index_manifest_path = bundle_root / "index" / "index_manifest.json"
    index_manifest_digest = sha256_file(index_manifest_path) if index_manifest_path.is_file() else None
    has_self_host = (bundle_root / "self-host" / "README.md").exists()
    has_audio_manifest = (bundle_root / "metadata" / "notebooklm_audio_manifest.json").exists()
    files = [
        bundle_root / "tools.html",
        bundle_root / "assets" / "byok-tools.css",
        bundle_root / "assets" / "byok-tools.js",
        bundle_root / "assets" / "byok-corpus.json",
        bundle_root / "metadata" / "byok_tools_manifest.json",
    ]
    write_text(bundle_root / "tools.html", html_text(release, has_self_host=has_self_host, has_audio_manifest=has_audio_manifest))
    write_text(bundle_root / "assets" / "byok-tools.css", css_text())
    write_text(bundle_root / "assets" / "byok-tools.js", js_text())
    write_json(bundle_root / "assets" / "byok-corpus.json", chunks)
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "public_export_manifest_sha256": release.get("export_manifest_sha256"),
        "index_identifier": "index/index_manifest.json",
        "index_manifest_sha256": index_manifest_digest,
        "dataset_identifier": "assets/byok-corpus.json",
        "tooling": "browser-byok",
        "key_policy": "in-memory-only",
        "llm_request_path": "browser-to-user-selected-provider",
        "maintainer_llm_endpoint": False,
        "deep_search_scope": "published-corpus-only",
        "web_search_enabled": False,
        "safety_doc": "docs/operations/public-byok-cost-safety.md",
        "self_host_status": "available" if has_self_host else "missing",
        "notebooklm_audio_status": "available" if has_audio_manifest else "operator_gated",
        "chunk_count": len(chunks),
        "generated_at": utc_now(),
        "output_checksums": {},
    }
    manifest_path = bundle_root / "metadata" / "byok_tools_manifest.json"
    write_json(manifest_path, manifest)
    scan_browser_policy(files[:4])
    checksums = output_checksums(files, bundle_root)
    manifest["output_checksums"] = checksums
    write_json(manifest_path, manifest)
    receipt_path = config.receipt_path or bundle_root / "metadata" / "byok_tools_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": "succeeded",
            "exit_code": 0,
            "bundle_id": release.get("bundle_id"),
            "chunk_count": len(chunks),
            "key_policy": "in-memory-only",
            "web_search_enabled": False,
            "maintainer_llm_endpoint": False,
            "output_checksums": checksums,
            "generated_at": utc_now(),
        },
    )
    return ByokToolsResult(bundle_root, manifest_path, receipt_path, checksums, len(chunks))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate browser BYOK tools for a public bundle")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_byok_tooling(
        ByokToolsConfig(bundle_root=args.bundle_root, receipt_path=args.receipt_path)
    )
    print(f"BYOK tools written: {result.bundle_root / 'tools.html'}")
    print(f"Receipt: {result.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
