"""Package self-host public tooling configs for public release bundles."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline._security.private_markers import publishing_marker_hits
from conference_pipeline.publishing.public_index import (
    sha256_file,
    validate_public_bundle_root,
)


class SelfHostKitError(RuntimeError):
    """Raised when self-host kit generation fails closed."""


RECEIPT_TYPE = "conference_pipeline.self_host_tools"


@dataclass(frozen=True)
class SelfHostKitConfig:
    bundle_root: Path
    receipt_path: Path | None = None


@dataclass(frozen=True)
class SelfHostKitResult:
    kit_dir: Path
    manifest_path: Path
    receipt_path: Path
    output_checksums: dict[str, str]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelfHostKitError(f"JSON object expected: {path}")
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


def readme_text(release: dict[str, Any]) -> str:
    bundle_id = release.get("bundle_id", "public-release")
    return f"""# Self-host tools for {bundle_id}

This kit runs against the downloaded public release bundle and its prebuilt
index. It does not use maintainer provider credentials.

## Scope

- Deep research: local corpus only, web search disabled.
- Deck builder: reuse `scripts/generate_slides_agentic.py` from the private
  platform checkout, pointed at this public bundle.
- Report builder: reuse GPT-Researcher local-source mode over this bundle.
- Audio: use `podcastfy_config.yaml` with Microsoft Edge TTS.
- NotebookLM: optional operator/self-host path described in
  `../metadata/notebooklm_audio_manifest.json` when present.

## Commands

```bash
export DOC_PATH=/absolute/path/to/{bundle_id}
export LLM_PROVIDER_KEY=replace-with-your-own-key
python scripts/query_gpt_researcher.py --report-source local --doc-path "$DOC_PATH"
python scripts/generate_slides_agentic.py --config self-host/deck-builder-config.json
python scripts/query_gpt_researcher_hybrid.py --config self-host/report-builder-config.json
podcastfy --config self-host/podcastfy_config.yaml
```

The corpus snapshot is this public bundle plus `index/`. Do not point these
configs at private Drive mounts or unpublished workspace directories.
"""


def env_example_text() -> str:
    return """# Public self-host GPT-Researcher defaults
DOC_PATH=..
report_source=local
WEB_SEARCH_ENABLED=false
LLM_PROVIDER_KEY=replace-with-your-own-key
LOCAL_MODEL_ENDPOINT=http://localhost:11434
"""


def deck_config(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "tool": "generate_slides_agentic",
        "bundle_id": release.get("bundle_id"),
        "input_bundle": "..",
        "index_dir": "../index",
        "llm": {"mode": "byok-or-self-hosted", "key_env": "LLM_PROVIDER_KEY"},
        "outputs": {"format": ["html", "pdf"], "directory": "./outputs/decks"},
    }


def report_config(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "tool": "gpt-researcher-local-report",
        "bundle_id": release.get("bundle_id"),
        "doc_path": "..",
        "report_source": "local",
        "WEB_SEARCH_ENABLED": False,
        "index_dir": "../index",
        "llm": {"mode": "byok-or-self-hosted", "key_env": "LLM_PROVIDER_KEY"},
        "outputs": {"format": ["markdown"], "directory": "./outputs/reports"},
    }


def podcastfy_config_text(release: dict[str, Any]) -> str:
    bundle_id = release.get("bundle_id", "public-release")
    return f"""# podcastfy public audio config for {bundle_id}
source:
  path: ..
  include:
    - analysis/**/*.md
    - decks/**/*.md
    - index/corpus_chunks.jsonl
llm:
  mode: local_or_user_supplied
tts:
  provider: edge
  voice: en-US-JennyNeural
  api_key_required: false
output:
  directory: ./outputs/audio
  format: mp3
"""


def docker_compose_text() -> str:
    return """services:
  public-research-tools:
    image: python:3.11-slim
    working_dir: /workspace
    volumes:
      - ..:/workspace/public-bundle:ro
      - ./outputs:/workspace/outputs
    environment:
      DOC_PATH: /workspace/public-bundle
      WEB_SEARCH_ENABLED: "false"
      report_source: local
      LLM_PROVIDER_KEY: ${LLM_PROVIDER_KEY:-}
    command: python -c "print('Mount the public bundle and run the documented scripts from your platform checkout.')"
"""


def assert_no_private_markers(paths: Sequence[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = publishing_marker_hits(text, allow_env_mention=True)
        if hits:
            raise SelfHostKitError(f"private marker found in self-host kit: {hits[0]}")


def package_self_host_tools(config: SelfHostKitConfig) -> SelfHostKitResult:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    release = read_json(bundle_root / "RELEASE.json")
    index_manifest_path = bundle_root / "index" / "index_manifest.json"
    index_manifest_digest = sha256_file(index_manifest_path) if index_manifest_path.is_file() else None
    kit_dir = bundle_root / "self-host"
    files = [
        kit_dir / "README.md",
        kit_dir / "gpt-researcher.env.example",
        kit_dir / "deck-builder-config.json",
        kit_dir / "report-builder-config.json",
        kit_dir / "podcastfy_config.yaml",
        kit_dir / "docker-compose.yml",
        kit_dir / "self_host_manifest.json",
    ]
    write_text(kit_dir / "README.md", readme_text(release))
    write_text(kit_dir / "gpt-researcher.env.example", env_example_text())
    write_json(kit_dir / "deck-builder-config.json", deck_config(release))
    write_json(kit_dir / "report-builder-config.json", report_config(release))
    write_text(kit_dir / "podcastfy_config.yaml", podcastfy_config_text(release))
    write_text(kit_dir / "docker-compose.yml", docker_compose_text())
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "public_export_manifest_sha256": release.get("export_manifest_sha256"),
        "index_identifier": "index/index_manifest.json",
        "index_manifest_sha256": index_manifest_digest,
        "dataset_identifier": "self-host/",
        "corpus_snapshot": "phase06-bundle-plus-phase07-index",
        "web_search_enabled": False,
        "report_source": "local",
        "credential_model": "byok-or-self-hosted",
        "files": [path.relative_to(bundle_root).as_posix() for path in files[:-1]],
        "generated_at": utc_now(),
        "output_checksums": {},
    }
    manifest_path = kit_dir / "self_host_manifest.json"
    write_json(manifest_path, manifest)
    assert_no_private_markers(files)
    checksums = {
        path.relative_to(bundle_root).as_posix(): sha256_file(path)
        for path in files
        if path.is_file()
    }
    manifest["output_checksums"] = checksums
    write_json(manifest_path, manifest)
    receipt_path = config.receipt_path or kit_dir / "self_host_tools_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": "succeeded",
            "exit_code": 0,
            "bundle_id": release.get("bundle_id"),
            "kit_dir": str(kit_dir),
            "web_search_enabled": False,
            "report_source": "local",
            "output_checksums": checksums,
            "generated_at": utc_now(),
        },
    )
    return SelfHostKitResult(kit_dir, manifest_path, receipt_path, checksums)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package public self-host tools")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = package_self_host_tools(
        SelfHostKitConfig(bundle_root=args.bundle_root, receipt_path=args.receipt_path)
    )
    print(f"Self-host kit written: {result.kit_dir}")
    print(f"Receipt: {result.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
