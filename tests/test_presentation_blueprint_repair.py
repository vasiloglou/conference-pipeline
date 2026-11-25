from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_HASH_MISMATCH,
    select_latest_approved_blueprint,
    validate_blueprint_review_file,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _blueprint_copy(tmp_path: Path) -> Path:
    path = tmp_path / "deck" / "presentation_blueprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _approved_review(blueprint: Path, path: Path) -> Path:
    payload = json.loads((FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    payload["blueprint_sha256"] = hashlib.sha256(blueprint.read_bytes()).hexdigest()
    return _write(path, payload)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/presentations/repair_presentation_blueprint.py", *args],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_full_repair_creates_new_revision_and_repair_of(tmp_path: Path) -> None:
    blueprint = _blueprint_copy(tmp_path)
    review = _approved_review(blueprint, tmp_path / "deck" / "blueprint_review.json")
    output = tmp_path / "deck" / "presentation_blueprint.repair.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--review-json",
            str(review),
            "--output-json",
            str(output),
            "--repair-mode",
            "full",
            "--repair-notes",
            "Tighten source evidence and story structure.",
        ]
    )

    assert result.returncode == 0, result.stderr
    original = json.loads(blueprint.read_text(encoding="utf-8"))
    repaired = json.loads(output.read_text(encoding="utf-8"))
    assert repaired["blueprint_id"] != original["blueprint_id"]
    assert repaired["repair_of"] == original["blueprint_id"]
    assert repaired["source_plan_repair_of"] is None
    assert repaired["revision"] == original["revision"] + 1
    assert repaired["repair_notes"] == "Tighten source evidence and story structure."


def test_source_plan_only_repair_tracks_source_plan_repair_of(tmp_path: Path) -> None:
    blueprint = _blueprint_copy(tmp_path)
    review = _approved_review(blueprint, tmp_path / "deck" / "blueprint_review.json")
    output = tmp_path / "deck" / "presentation_blueprint.source-repair.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--review-json",
            str(review),
            "--output-json",
            str(output),
            "--repair-mode",
            "source-plan-only",
            "--repair-notes",
            "Repair source labels only.",
        ]
    )

    assert result.returncode == 0, result.stderr
    original = json.loads(blueprint.read_text(encoding="utf-8"))
    repaired = json.loads(output.read_text(encoding="utf-8"))
    assert repaired["repair_of"] is None
    assert repaired["source_plan_repair_of"] == original["blueprint_id"]
    assert repaired["revision"] == original["revision"] + 1


def test_original_approval_token_does_not_approve_repaired_blueprint(tmp_path: Path) -> None:
    blueprint = _blueprint_copy(tmp_path)
    original_review = _approved_review(blueprint, tmp_path / "deck" / "blueprint_review.json")
    output = tmp_path / "deck" / "presentation_blueprint.repair.json"
    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--review-json",
            str(original_review),
            "--output-json",
            str(output),
            "--repair-mode",
            "full",
            "--repair-notes",
            "Repair.",
        ]
    )
    assert result.returncode == 0, result.stderr

    validation = validate_blueprint_review_file(output, original_review)

    assert BLUEPRINT_HASH_MISMATCH in {record.code for record in validation.errors}


def test_latest_approved_blueprint_ignores_unapproved_repairs(tmp_path: Path) -> None:
    blueprint = _blueprint_copy(tmp_path)
    review = _approved_review(blueprint, tmp_path / "deck" / "blueprint_review.json")
    repaired = tmp_path / "deck" / "presentation_blueprint.repair.json"
    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--review-json",
            str(review),
            "--output-json",
            str(repaired),
            "--repair-mode",
            "full",
            "--repair-notes",
            "Repair.",
        ]
    )
    assert result.returncode == 0, result.stderr

    selected = select_latest_approved_blueprint([(blueprint, review), (repaired, tmp_path / "deck" / "missing_review.json")])

    assert selected is not None
    assert selected.blueprint_path == blueprint
    assert selected.review_path == review


def _generation_receipt(blueprint: Path, path: Path) -> Path:
    return _write(
        path,
        {
            "blueprint_sha256": hashlib.sha256(blueprint.read_bytes()).hexdigest(),
            "mode": "live",
            "production_evidence": True,
            "provider_receipt_paths": ["provider-calls/editorial_blueprint/call.json"],
            "receipt_type": "conference_pipeline.presentation.blueprint_generation",
        },
    )


def _in_place_slide_repair(tmp_path: Path, patch: dict[str, object]) -> tuple[Path, Path, subprocess.CompletedProcess[str]]:
    blueprint = _blueprint_copy(tmp_path)
    review = _approved_review(blueprint, tmp_path / "deck" / "blueprint_review.json")
    receipt = _generation_receipt(blueprint, tmp_path / "deck" / "presentation_blueprint_receipt.json")
    patch_path = _write(tmp_path / "slide_patch.json", patch)
    result = _run(
        [
            "--blueprint", str(blueprint),
            "--review-json", str(review),
            "--output-json", str(blueprint),
            "--repair-mode", "full",
            "--repair-notes", "Drop an attribution the slide never binds.",
            "--slide-patch", str(patch_path),
            "--blueprint-receipt", str(receipt),
            "--receipt-output", str(receipt),
        ]
    )
    return blueprint, receipt, result


def _first_slide_id() -> str:
    payload = json.loads((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    return str(payload["slides"][0]["slide_id"])


def test_in_place_slide_repair_rebinds_receipt_through_archived_parent(tmp_path: Path) -> None:
    from conference_pipeline.presentation.blueprint import validate_blueprint_receipt_file

    slide_id = _first_slide_id()
    blueprint, receipt, result = _in_place_slide_repair(
        tmp_path, {slide_id: {"narration_beat": "A rewritten beat that names no unbound work."}}
    )

    assert result.returncode == 0, result.stderr
    archived_blueprint = blueprint.with_name("presentation_blueprint.rev1.json")
    archived_receipt = receipt.with_name("presentation_blueprint_receipt.rev1.json")
    assert archived_blueprint.is_file() and archived_receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["blueprint_sha256"] == hashlib.sha256(blueprint.read_bytes()).hexdigest()
    assert payload["repair"]["parent_blueprint_sha256"] == hashlib.sha256(archived_blueprint.read_bytes()).hexdigest()
    assert payload["repair"]["changed_paths"] == ["$.slides[0].narration_beat"]
    assert payload["repair"]["provider_calls_made"] is False
    assert not validate_blueprint_receipt_file(blueprint, receipt, require_provider_evidence=True).errors


def test_repair_receipt_fails_when_archived_parent_receipt_is_altered(tmp_path: Path) -> None:
    from conference_pipeline.presentation.blueprint import BLUEPRINT_RECEIPT_INVALID, validate_blueprint_receipt_file

    blueprint, receipt, result = _in_place_slide_repair(tmp_path, {_first_slide_id(): {"narration_beat": "Rewritten."}})
    assert result.returncode == 0, result.stderr
    archived_receipt = receipt.with_name("presentation_blueprint_receipt.rev1.json")
    archived_receipt.write_text(archived_receipt.read_text(encoding="utf-8").replace("live", "mock"), encoding="utf-8")

    codes = {record.code for record in validate_blueprint_receipt_file(blueprint, receipt).errors}

    assert BLUEPRINT_RECEIPT_INVALID in codes


def test_slide_patch_refuses_source_binding_fields(tmp_path: Path) -> None:
    blueprint, _receipt, result = _in_place_slide_repair(tmp_path, {_first_slide_id(): {"source_plan": []}})

    assert result.returncode == 2
    assert "may not change" in result.stderr
    assert not blueprint.with_name("presentation_blueprint.rev1.json").exists()
