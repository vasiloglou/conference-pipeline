from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.source_pack import COUNT_KEYS, build_deck_source_pack


REPO_ROOT = Path(__file__).resolve().parent.parent


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_source_pack_workspace(
    tmp_path: Path,
    *,
    include_malformed: bool = False,
) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    extraction_root = workspace / "storage" / "extractions" / "demo-run"
    topic_id = "02_model_architectures"
    artifact_id = f"{topic_id}:Demo1234"
    source_artifact_id = "openreview:Demo1234"
    items: list[dict[str, Any]] = [
        {
            "artifact_id": artifact_id,
            "confidence": 0.91,
            "extraction_method": "fixture",
            "source_artifact_digest": "digest-demo",
            "source_artifact_id": source_artifact_id,
            "source_artifact_path": "storage/transcription-ocr/text-extraction/openreview/Demo1234.txt",
        }
    ]
    if include_malformed:
        items.append(
            {
                "confidence": 0.12,
                "source_artifact_path": "storage/transcription-ocr/text-extraction/openreview/Missing123.txt",
            }
        )

    write_json(
        extraction_root / "by-topic" / f"{topic_id}.json",
        {
            "artifact_count": len(items),
            "items": items,
            "schema_version": 1,
            "topic_id": topic_id,
        },
    )
    write_json(
        extraction_root / "by-artifact" / "02_model_architectures_Demo1234.json",
        {
            "artifact_id": artifact_id,
            "confidence": 0.91,
            "evidence": [
                {
                    "claim": "Demo method improves the result.",
                    "quote": "Figure 2 shows the architecture diagram.",
                }
            ],
            "quality_metadata": {"input_kind": "paper_pdf"},
            "source_artifact_id": source_artifact_id,
            "source_artifact_path": "storage/transcription-ocr/text-extraction/openreview/Demo1234.txt",
        },
    )
    (extraction_root / "review").mkdir(parents=True, exist_ok=True)
    (extraction_root / "review" / f"{topic_id}-dereferenced.md").write_text(
        "# Model Architectures\n\nA finding with a figure and a limitation.\n",
        encoding="utf-8",
    )
    ocr_path = workspace / "storage" / "transcription-ocr" / "text-extraction" / "openreview" / "Demo1234.txt"
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_path.write_text(
        "Demo1234 paper text with an equation, a method, and a chart.",
        encoding="utf-8",
    )
    return workspace


def test_source_pack_builds_from_topic_extraction_fixture(tmp_path: Path) -> None:
    workspace = make_source_pack_workspace(tmp_path)

    receipt = build_deck_source_pack(
        workspace,
        "02_model_architectures",
        "demo-run",
    )

    deck_dir = workspace / "decks" / "02_model_architectures"
    assert (deck_dir / "extracted_content.json").is_file()
    assert (deck_dir / "source_attributions.json").is_file()
    assert (deck_dir / "source_pack_ranked_papers.json").is_file()
    assert (deck_dir / "source_pack_receipt.json").is_file()
    assert receipt["extraction_run_id"] == "demo-run"
    assert receipt["topic_id"] == "02_model_architectures"
    assert set(receipt["counts"]) == set(COUNT_KEYS)
    assert all("path" in entry and "sha256" in entry for entry in receipt["inputs"])

    attributions = json.loads((deck_dir / "source_attributions.json").read_text(encoding="utf-8"))
    assert attributions["sources"][0]["ocr_paths"] == [
        "storage/transcription-ocr/text-extraction/openreview/Demo1234.txt"
    ]


def test_source_pack_records_skipped_records_and_input_hashes(tmp_path: Path) -> None:
    workspace = make_source_pack_workspace(tmp_path, include_malformed=True)

    receipt = build_deck_source_pack(
        workspace,
        "02_model_architectures",
        "demo-run",
    )

    assert all("path" in entry and "sha256" in entry for entry in receipt["inputs"])
    assert any(
        "missing artifact_id" in skipped["reason"]
        for skipped in receipt["skipped_records"]
    )


def test_source_pack_dry_run_rejects_output_dir_outside_workspace(tmp_path: Path) -> None:
    workspace = make_source_pack_workspace(tmp_path)
    outside = tmp_path / "outside"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/build_deck_source_pack.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "02_model_architectures",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "02_model_architectures",
            "--extraction-run-id",
            "demo-run",
            "--output-dir",
            str(outside),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--output-dir must stay inside the workspace" in result.stderr


def test_resolve_inside_admits_drive_backed_symlink(tmp_path: Path) -> None:
    """A deliberate workspace-owned symlink (e.g. the Drive-backed
    storage/transcription-ocr on-demand leaf) whose real target sits outside
    the repo must be admitted by _resolve_inside, not rejected -- otherwise
    source-pack building fails closed the moment transcription-ocr is a Drive
    symlink. A textually-outside absolute path is still rejected (see the CLI
    security test)."""
    import sys as _sys

    if _sys.platform == "win32":
        return
    from conference_pipeline.presentation.source_pack import _resolve_inside

    workspace = tmp_path / "conferences" / "demo"
    (workspace / "storage").mkdir(parents=True)
    external = tmp_path / "drive-backed" / "transcription-ocr"
    external.mkdir(parents=True)
    (external / "artifact.json").write_text("{}", encoding="utf-8")
    (workspace / "storage" / "transcription-ocr").symlink_to(external, target_is_directory=True)

    resolved = _resolve_inside(workspace, "storage/transcription-ocr", label="transcription OCR")
    assert (resolved / "artifact.json").read_text(encoding="utf-8") == "{}"


def test_resolve_inside_still_rejects_absolute_outside_path(tmp_path: Path) -> None:
    from conference_pipeline.presentation.source_pack import _resolve_inside

    workspace = tmp_path / "conferences" / "demo"
    workspace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        _resolve_inside(workspace, outside, label="output_dir")
    except ValueError as exc:
        assert "must stay inside the workspace" in str(exc)
    else:
        raise AssertionError("expected a textually-outside absolute path to be rejected")
