"""Characterization harness for the thin-orchestrator lift (wave W1).

W1 pins the *current* behaviour of every runner stage the plan intends to lift
(``.planning/proposals/thin-conference-orchestrators-20260930.md`` sections 4
and 5). A characterization test

1. materializes a synthetic fixture workspace from ``tests/fixtures/orchestrators``
   into ``tmp_path/conferences/<conference>`` (the stage-owner registry scopes
   Stages 2-5 by ``conferences/<slug>`` identity, so the directory name selects
   the conference's own runtime contract; the content stays synthetic),
2. imports the runner module in-process from its path (never runs its CLI),
3. drives one stage entry point with every network, provider, LLM, embedding
   and subprocess seam replaced by a fake (:func:`offline_guard` fails the test
   if a real one is reached), and
4. snapshots the receipt and outputs with :func:`stable_snapshot` and compares
   against ``tests/golden/orchestrators/<conference>/<name>.json``.

Why a stable snapshot on top of :func:`tests.orchestrator_golden.snapshot_receipt`:
the runners embed the absolute workspace path and run timestamps in the JSON
outputs they hash (``sha256_file``), so the *recorded* digests differ between
two temp directories even when the stage behaves identically. The stable
snapshot therefore

* maps every recorded sha that equals the raw or canonical digest of a
  workspace file to ``<sha256:relpath>`` / ``<canonical:relpath>``, and every
  digest that equals a receipt's recorded ``output_digest`` /
  ``lineage_digest`` / ``content_digest`` to ``<output_digest:relpath>``, ...
* pins each output file by a *normalized content* digest (roots and ISO
  timestamps replaced, run stamps dropped) instead of its raw bytes, and
* recomputes ``output_digest`` and ``lineage_digest`` over those normalized
  identities (``normalized_output_digest`` / ``normalized_lineage_digest``),
  alongside the harness's ``*_consistent`` flags computed on the raw receipt.

The registry sha (``script_registry_sha256``) is masked: lifts are
registry-neutral by contract, and an unrelated registry edit must not read as
a stage behaviour change. ``required_script_ids`` and ``stage_contract`` stay
pinned, so a change of owner or contract outcome still shows up.

Set ``ORCHESTRATOR_GOLDEN_UPDATE=1`` to (re)record golden files.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.error import HTTPError

import pytest

from conference_pipeline.receipt_lineage import (
    OUTPUT_VOLATILE_KEYS,
    canonical_json_digest,
    canonical_output_sha256,
    compute_output_digest,
    content_digest,
    lineage_digest,
    sha256_file,
    workspace_relpath,
)
from conference_pipeline.research.egress_guard import loopback_only
from conference_pipeline.stage_script_registry import registry_sha256
from tests.orchestrator_golden import (
    FIXTURE_ROOT,
    TIMESTAMP_PLACEHOLDER,
    VOLATILE_KEYS,
    assert_matches_golden,
    load_fixture_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden" / "orchestrators"
RUNNER_PATHS = {
    "icml2026": REPO_ROOT / "scripts" / "run_icml2026_production.py",
    "cvpr2026": REPO_ROOT / "scripts" / "run_cvpr2026_production.py",
}
REGISTRY_PLACEHOLDER = "<REGISTRY_SHA>"
MASK_PLACEHOLDER = "<MASKED>"
DERIVED_DIGEST_PLACEHOLDER = "<DERIVED_DIGEST>"
FROZEN_NOW = "2031-01-02T03:04:05Z"

# Every env var a runner could turn into a live credential or provider switch.
PROVIDER_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "OPENREVIEW_USERNAME",
    "OPENREVIEW_PASSWORD",
    "OPENREVIEW_TOKEN",
    "ICML_COOKIES_FILE",
    "ICML2026_PROVIDER_COST_APPROVED",
    "CVPR2026_PROVIDER_COST_APPROVED",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_LENGTH",
    "OLLAMA_BASE_URL",
    "EXTRACTION_LOCAL_LLM_BASE_URL",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

_ISO_IN_TEXT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")
# Receipts quote digests shortened to 12 hex chars in human-readable evidence.
_HEX12_RE = re.compile(r"\b[0-9a-f]{12}\b")
# Wall-clock throughput and file-stat values some owners persist (progress
# files, chunk manifests); they vary per run and carry no stage behaviour.
RUNTIME_METRIC_KEYS = frozenset(
    {
        "chunks_per_minute",
        "eta_seconds",
        "text_file_mtime_ns",
        "mtime_ns",
        "mtime",
        "pid",
    }
)
_RECEIPT_DROP_KEYS = frozenset(VOLATILE_KEYS | RUNTIME_METRIC_KEYS)
_OUTPUT_DROP_KEYS = frozenset(VOLATILE_KEYS | OUTPUT_VOLATILE_KEYS | RUNTIME_METRIC_KEYS)


# --------------------------------------------------------------------------
# Runner loading and workspaces
# --------------------------------------------------------------------------

_RUNNER_CACHE: dict[str, ModuleType] = {}


def load_runner(conference: str) -> ModuleType:
    """Import a production runner from its path (in-process, once per session)."""

    cached = _RUNNER_CACHE.get(conference)
    if cached is not None:
        return cached
    path = RUNNER_PATHS[conference]
    name = f"orchestrator_w1_{conference}_runner"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _RUNNER_CACHE[conference] = module
    return module


def materialize_conference_workspace(fixture: str, tmp_path: Path, conference: str) -> Path:
    """Copy a fixture workspace to ``tmp_path/conferences/<conference>``.

    Tiny PDFs declared in ``fixture.json`` are generated as in W0's
    :func:`tests.orchestrator_golden.materialize_fixture_workspace`. A minimal
    storage manifest is added when the fixture has none (every runner's
    preflight requires one).
    """

    target = Path(tmp_path) / "conferences" / conference
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE_ROOT / fixture, target)
    from tests.orchestrator_golden import _MINIMAL_PDF  # noqa: PLC0415 - W0 private constant

    for relative in load_fixture_manifest(fixture).get("generated_pdfs") or ():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_MINIMAL_PDF.format(name=Path(relative).stem).encode("ascii"))
    manifest = target / "storage" / "storage_manifest.json"
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"schema_version": 1, "artifact_families": []}) + "\n", encoding="utf-8")
    return target


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Offline guard: no network, no provider, no subprocess, no real credential
# --------------------------------------------------------------------------


class ForbiddenCallError(AssertionError):
    """A real provider, network or subprocess seam was reached in a W1 test."""


ForbiddenCall = ForbiddenCallError


def forbidden(label: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise ForbiddenCallError(f"W1 characterization must not reach {label}")

    return _raise


class _ForbiddenClass:
    label = "a provider client"

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForbiddenCallError(f"W1 characterization must not construct {self.label}")


def _forbidden_class(label: str) -> type:
    return type(f"Forbidden_{re.sub(r'[^A-Za-z0-9]', '_', label)}", (_ForbiddenClass,), {"label": label})


# Names a runner module imports that construct a paid/remote client.
RUNNER_PROVIDER_SEAMS = (
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "OpenAIDeepNarrativeClient",
    "build_embedding_client",
    "ChromaEmbeddingAdapter",
    "load_openreview_auth",
)


@contextlib.contextmanager
def offline_guard(monkeypatch: pytest.MonkeyPatch, *runners: ModuleType) -> Iterator[dict[str, Any]]:
    """Block egress, poison provider constructors and subprocesses, scrub env.

    Tests re-install fakes for the seams they need *after* entering the guard.
    On exit the egress report must show no blocked destination (a runner that
    swallowed an ``EgressBlockedError`` into a receipt would otherwise pass).
    """

    for name in PROVIDER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for module_name, attr in (
        ("openai", "OpenAI"),
        ("openai", "AsyncOpenAI"),
        ("anthropic", "Anthropic"),
        ("anthropic", "AsyncAnthropic"),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            with contextlib.suppress(Exception):
                module = importlib.import_module(module_name)
        if module is not None and hasattr(module, attr):
            monkeypatch.setattr(module, attr, _forbidden_class(f"{module_name}.{attr}"))
    with contextlib.suppress(Exception):
        from google import genai  # noqa: PLC0415

        monkeypatch.setattr(genai, "Client", _forbidden_class("google.genai.Client"))
    with contextlib.suppress(Exception):
        from conference_pipeline.indexing import embeddings  # noqa: PLC0415

        monkeypatch.setattr(embeddings, "build_embedding_client", forbidden("indexing.embeddings.build_embedding_client"))
    for runner in runners:
        for attr in RUNNER_PROVIDER_SEAMS:
            if hasattr(runner, attr):
                value = getattr(runner, attr)
                replacement: Any = (
                    _forbidden_class(f"runner.{attr}") if isinstance(value, type) else forbidden(f"runner.{attr}")
                )
                monkeypatch.setattr(runner, attr, replacement)
        if hasattr(runner, "urlopen"):
            monkeypatch.setattr(runner, "urlopen", forbidden("runner.urlopen (install a FakeWeb)"))
    monkeypatch.setattr(subprocess, "run", forbidden("subprocess.run"))
    monkeypatch.setattr(subprocess, "Popen", _forbidden_class("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "check_output", forbidden("subprocess.check_output"))
    monkeypatch.setattr(subprocess, "check_call", forbidden("subprocess.check_call"))
    state: dict[str, Any] = {}
    with loopback_only() as report:
        state["egress"] = report
        yield state
    assert not report.blocked, f"non-loopback egress attempted: {report.blocked}"


# --------------------------------------------------------------------------
# Fake HTTP transport for runners that call urllib.request.urlopen
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, url: str, body: bytes, *, status: int = 200, content_type: str = "text/html") -> None:
        self._url = url
        self._stream = io.BytesIO(body)
        self.status = status
        self.code = status
        headers = Message()
        headers["content-type"] = content_type
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size if size is not None else -1)

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stream.close()


class FakeWeb:
    """``urlopen`` stand-in serving canned bodies by exact URL (query ignored optionally).

    Unknown URLs raise ``HTTPError(404)``; the fake never reaches a socket.
    ``calls`` records every requested URL in order (thread-safe).
    """

    def __init__(self, routes: Mapping[str, tuple[int, str, bytes]] | None = None) -> None:
        self.routes: dict[str, tuple[int, str, bytes]] = dict(routes or {})
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def add(self, url: str, body: bytes | str, *, status: int = 200, content_type: str = "text/html") -> None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.routes[url] = (status, content_type, data)

    def __call__(self, request: Any, timeout: float | None = None, **_kwargs: Any) -> FakeResponse:
        url = request.full_url if hasattr(request, "full_url") else str(request)
        with self._lock:
            self.calls.append(url)
        route = self.routes.get(url)
        if route is None:
            route = self.routes.get(url.split("?", 1)[0])
        if route is None or route[0] >= 400:
            status = route[0] if route else 404
            body = route[2] if route else b"<html><title>Missing</title><body>missing</body></html>"
            headers = Message()
            headers["content-type"] = (route[1] if route else "text/html")
            raise HTTPError(url, status, "fake", headers, io.BytesIO(body))
        status, content_type, body = route
        return FakeResponse(url, body, status=status, content_type=content_type)


# --------------------------------------------------------------------------
# Stable snapshots
# --------------------------------------------------------------------------


def _root_variants(root: Path) -> list[str]:
    variants = {str(root), str(Path(root).resolve())}
    for item in list(variants):
        if item.startswith("/private/"):
            variants.add(item[len("/private") :])
    return sorted(variants, key=len, reverse=True)


class DigestIndex:
    """Maps digests found in a workspace to stable, path-based tokens."""

    def __init__(
        self,
        workspace: Path,
        *,
        skip_dirs: Iterable[str] = (),
        literals: Mapping[str, str] | None = None,
        json_digest_fns: Mapping[str, Callable[[Any], Any]] | None = None,
        derived_digest_keys: Iterable[str] = (),
    ) -> None:
        self.workspace = Path(workspace).resolve()
        # Keys whose value is a digest of an in-memory payload that embedded a
        # path-dependent digest and no longer exists on disk: kept as a
        # placeholder only when no file/receipt token matches.
        self.derived_digest_keys = frozenset(derived_digest_keys)
        self.tokens: dict[str, str] = {}
        # Exact strings (e.g. a registry-governed model id) -> placeholder, so
        # a catalog upgrade does not read as a stage behaviour change.
        self.literals: list[tuple[str, str]] = sorted(
            ((key, value) for key, value in (literals or {}).items() if key), key=lambda item: len(item[0]), reverse=True
        )
        registry = registry_sha256()
        self.tokens[registry] = REGISTRY_PLACEHOLDER
        skip = tuple(skip_dirs)
        files = sorted(
            path
            for path in self.workspace.rglob("*")
            if path.is_file() and not any(part in skip for part in path.relative_to(self.workspace).parts)
        )
        # Receipt-level digests first (they are the most specific identity).
        for path in files:
            rel = path.relative_to(self.workspace).as_posix()
            if path.suffix == ".json":
                with contextlib.suppress(Exception):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, Mapping):
                        # Self-declared content digests (e.g. the analysis
                        # manifest's content_sha256) cover path-bearing fields.
                        self._add(payload.get("content_sha256"), f"<content_sha256:{rel}>")
                        for label, fn in (json_digest_fns or {}).items():
                            with contextlib.suppress(Exception):
                                self._add(fn(payload), f"<{label}:{rel}>")
                        self._add(payload.get("output_digest"), f"<output_digest:{rel}>")
                        lineage = payload.get("lineage")
                        if isinstance(lineage, Mapping):
                            self._add(lineage.get("lineage_digest"), f"<lineage_digest:{rel}>")
                        self._add(content_digest(payload), f"<content_digest:{rel}>")
        for path in files:
            rel = path.relative_to(self.workspace).as_posix()
            self._add(sha256_file(path), f"<sha256:{rel}>")
            self._add(canonical_output_sha256(path), f"<canonical:{rel}>")
        for directory in sorted(p for p in self.workspace.rglob("*") if p.is_dir()):
            rel = directory.relative_to(self.workspace).as_posix()
            if any(part in skip for part in directory.relative_to(self.workspace).parts):
                continue
            from conference_pipeline.receipt_lineage import directory_sha256  # noqa: PLC0415

            self._add(directory_sha256(directory), f"<dir:{rel}>")

    def _add(self, digest: Any, token: str) -> None:
        if isinstance(digest, str) and _HEX64_RE.fullmatch(digest) and digest not in self.tokens:
            self.tokens[digest] = token

    def replace(self, text: str) -> str:
        if text in self.tokens:
            return self.tokens[text]
        text = _HEX64_RE.sub(lambda match: self.tokens.get(match.group(0), match.group(0)), text)
        if _HEX12_RE.search(text):
            prefixes = self._prefixes()
            text = _HEX12_RE.sub(lambda match: prefixes.get(match.group(0), match.group(0)), text)
        return text

    def _prefixes(self) -> dict[str, str]:
        cached = getattr(self, "_prefix_cache", None)
        if cached is None:
            cached = {}
            ambiguous: set[str] = set()
            for digest, token in self.tokens.items():
                prefix = digest[:12]
                if prefix in cached and cached[prefix] != token:
                    ambiguous.add(prefix)
                cached.setdefault(prefix, token)
            for prefix in ambiguous:
                cached.pop(prefix, None)
            self._prefix_cache = cached
        return cached


def _mask(payload: Any, dotted_paths: Iterable[str]) -> Any:
    payload = json.loads(json.dumps(payload, default=str))
    for dotted in dotted_paths:
        parts = dotted.split(".")
        _mask_walk(payload, parts)
    return payload


def _mask_walk(node: Any, parts: list[str]) -> None:
    if not parts:
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
        for child in list(children):
            _mask_walk(child, rest)
        return
    if isinstance(node, dict) and head in node:
        if rest:
            _mask_walk(node[head], rest)
        else:
            node[head] = MASK_PLACEHOLDER


def normalize_payload(
    payload: Any,
    *,
    workspace: Path,
    index: DigestIndex,
    extra_roots: Iterable[Path] = (),
    drop_keys: Iterable[str] = _RECEIPT_DROP_KEYS,
) -> Any:
    """Roots -> placeholders, timestamps -> placeholder, digests -> tokens."""

    replacements = [(variant, "<WORKSPACE>") for variant in _root_variants(workspace)]
    replacements += [(variant, "<REPO>") for variant in _root_variants(REPO_ROOT)]
    for index_root, root in enumerate(extra_roots):
        replacements += [(variant, f"<ROOT{index_root}>") for variant in _root_variants(root)]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    drop = frozenset(drop_keys)

    derived = index.derived_digest_keys

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {
                str(key): (
                    DERIVED_DIGEST_PLACEHOLDER
                    if str(key) in derived
                    and isinstance(value, str)
                    and _HEX64_RE.fullmatch(value)
                    and value not in index.tokens
                    else walk(value)
                )
                for key, value in node.items()
                if str(key) not in drop
            }
        if isinstance(node, (list, tuple)):
            return [walk(value) for value in node]
        if isinstance(node, str):
            return normalize_text(node, replacements=replacements, index=index)
        return node

    return walk(payload)


def normalize_text(text: str, *, replacements: list[tuple[str, str]], index: DigestIndex) -> str:
    for literal, placeholder in index.literals:
        if literal in text:
            text = text.replace(literal, placeholder)
    for root, placeholder in replacements:
        if root and root in text:
            text = text.replace(root, placeholder)
    if _ISO_IN_TEXT_RE.fullmatch(text):
        return TIMESTAMP_PLACEHOLDER
    text = _ISO_IN_TEXT_RE.sub(TIMESTAMP_PLACEHOLDER, text)
    return index.replace(text)


def normalized_file_digest(
    path: Path, *, workspace: Path, index: DigestIndex, skip_dirs: Iterable[str] = ()
) -> Any:
    """Digest of a file's content with roots, timestamps and digests normalized."""

    path = Path(path)
    skip = tuple(skip_dirs)
    if path.is_dir():
        return {
            child.relative_to(path).as_posix(): normalized_file_digest(child, workspace=workspace, index=index)
            for child in sorted(path.rglob("*"))
            if child.is_file() and not any(part in skip for part in child.relative_to(path).parts)
        }
    if not path.is_file():
        return None
    raw = path.read_bytes()
    replacements = [(variant, "<WORKSPACE>") for variant in _root_variants(workspace)]
    replacements += [(variant, "<REPO>") for variant in _root_variants(REPO_ROOT)]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    if path.suffix in {".json", ".jsonl"}:
        try:
            if path.suffix == ".json":
                payload: Any = json.loads(raw.decode("utf-8"))
            else:
                payload = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if payload is not None:
            normalized = normalize_payload(payload, workspace=workspace, index=index, drop_keys=_OUTPUT_DROP_KEYS)
            return "json:" + canonical_json_digest(normalized)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "bytes:" + hashlib.sha256(raw).hexdigest()
    normalized_text = normalize_text(text, replacements=replacements, index=index)
    return "text:" + hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def stable_snapshot(
    receipt: Mapping[str, Any],
    *,
    workspace: Path,
    extra_output_paths: Iterable[str | Path] = (),
    mask: Iterable[str] = (),
    extra_roots: Iterable[Path] = (),
    skip_dirs: Iterable[str] = (),
    literals: Mapping[str, str] | None = None,
    json_digest_fns: Mapping[str, Callable[[Any], Any]] | None = None,
    derived_digest_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Snapshot a stage receipt so two runs in two temp dirs compare equal.

    ``skip_dirs`` names directory components (e.g. ``chroma``) whose files are
    neither indexed for digest tokens nor pinned as output files.
    ``literals`` maps exact strings (a registry-governed model id) to
    placeholders everywhere in the receipt and the output files.
    ``json_digest_fns`` are extra ``payload -> digest`` functions applied to
    every workspace JSON file, for owners that record a content digest of a
    path-bearing payload (tokenized as ``<label:relpath>``).
    ``derived_digest_keys`` name keys whose untokenizable 64-hex value is a
    digest of a transient path-bearing payload (``<DERIVED_DIGEST>``).
    """

    workspace = Path(workspace).resolve()
    skip_dirs = tuple(skip_dirs)
    raw = json.loads(json.dumps(dict(receipt), default=str))
    index = DigestIndex(
        workspace,
        skip_dirs=skip_dirs,
        literals=literals,
        json_digest_fns=json_digest_fns,
        derived_digest_keys=derived_digest_keys,
    )
    lineage = raw.get("lineage") if isinstance(raw.get("lineage"), Mapping) else None
    artifacts = [item for item in raw.get("output_artifacts") or () if isinstance(item, Mapping)]
    components = (lineage or {}).get("output_digest_components") or None
    recorded_output = raw.get("output_digest")
    masked = _mask(raw, mask)
    normalized = normalize_payload(masked, workspace=workspace, index=index, extra_roots=extra_roots)
    norm_artifacts = [item for item in normalized.get("output_artifacts") or () if isinstance(item, Mapping)]
    norm_lineage = normalized.get("lineage") if isinstance(normalized.get("lineage"), Mapping) else None

    output_paths: list[str] = []
    for item in artifacts:
        if isinstance(item.get("path"), str) and item["path"]:
            output_paths.append(item["path"])
    output_paths.extend(str(path) for path in extra_output_paths)
    output_files: dict[str, Any] = {}
    for rawpath in output_paths:
        candidate = Path(rawpath)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        output_files[workspace_relpath(workspace, candidate)] = normalized_file_digest(
            candidate, workspace=workspace, index=index, skip_dirs=skip_dirs
        )

    return {
        "receipt": normalized,
        "lineage_digest_consistent": None if lineage is None else lineage.get("lineage_digest") == lineage_digest(lineage),
        "normalized_lineage_digest": None if norm_lineage is None else lineage_digest(norm_lineage),
        "output_digest_consistent": (
            None
            if recorded_output is None
            else recorded_output == compute_output_digest(artifacts, components=components)
        ),
        "normalized_output_digest": (
            None
            if recorded_output is None
            else compute_output_digest(norm_artifacts, components=(norm_lineage or {}).get("output_digest_components") or None)
        ),
        "output_files": dict(sorted(output_files.items())),
    }


def golden_path(conference: str, name: str) -> Path:
    return GOLDEN_ROOT / conference / f"{name}.json"


def assert_golden(conference: str, name: str, snapshot: Mapping[str, Any]) -> None:
    assert_matches_golden(snapshot, golden_path(conference, name))


def snapshot_payload(payload: Any, *, workspace: Path, mask: Iterable[str] = ()) -> Any:
    """Normalize a non-receipt payload (a helper's return value) for a golden."""

    workspace = Path(workspace).resolve()
    index = DigestIndex(workspace)
    return normalize_payload(_mask(payload, mask), workspace=workspace, index=index)


__all__ = [
    "FROZEN_NOW",
    "FakeResponse",
    "FakeWeb",
    "ForbiddenCall",
    "ForbiddenCallError",
    "GOLDEN_ROOT",
    "REPO_ROOT",
    "assert_golden",
    "file_sha256",
    "forbidden",
    "golden_path",
    "load_runner",
    "materialize_conference_workspace",
    "offline_guard",
    "read_json",
    "snapshot_payload",
    "stable_snapshot",
    "write_json",
]
