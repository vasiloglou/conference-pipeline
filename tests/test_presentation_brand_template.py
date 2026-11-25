from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from conference_pipeline.presentation.brand_template import (
    BrandAssetRef,
    BrandTemplateFinding,
    BrandTemplateLock,
    validate_brand_template_lock,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "presentations" / "phase16"
BAD_FIXTURE = FIXTURES / "bad_brand_topic4.html"
GOOD_FIXTURE = FIXTURES / "good_relationalai_brand.html"

REFERENCE_NAME = "relationalai-neurips2025-reference"

EXPECTED_BLOCKER_CODES = {
    "fake_inline_title_logo_svg",
    "text_only_rai_brand_bug",
    "template_metadata_missing",
    "template_sha256_missing",
    "brand_asset_hash_missing",
    "scaffold_prose_present",
    "repeated_author_credit_on_content_slide",
}


def test_brand_lock_rejects_bad_topic4_brand_patterns() -> None:
    html_content = BAD_FIXTURE.read_text(encoding="utf-8")
    lock = validate_brand_template_lock(
        html_content=html_content,
        assets_sha256="",
        reference_name=REFERENCE_NAME,
    )
    assert lock.status == "fail"
    blocker_codes = {f.code for f in lock.findings if f.severity == "blocker"}
    assert blocker_codes == EXPECTED_BLOCKER_CODES, (
        f"Expected blocker codes {EXPECTED_BLOCKER_CODES}, got {blocker_codes}"
    )


def test_brand_lock_accepts_hash_bound_relationalai_reference() -> None:
    html_content = GOOD_FIXTURE.read_text(encoding="utf-8")
    lock = validate_brand_template_lock(
        html_content=html_content,
        assets_sha256="abc123placeholder",
        reference_name=REFERENCE_NAME,
    )
    blocker_findings = [f for f in lock.findings if f.severity == "blocker"]
    assert blocker_findings == [], (
        f"Unexpected blockers for good fixture: {[f.code for f in blocker_findings]}"
    )
    assert lock.status == "pass"


def test_brand_lock_records_required_geometry() -> None:
    html_content = GOOD_FIXTURE.read_text(encoding="utf-8")
    lock = validate_brand_template_lock(
        html_content=html_content,
        assets_sha256="abc123placeholder",
        reference_name=REFERENCE_NAME,
    )
    assert lock.renderer_geometry["title_logo"]["width"] == 309
    assert lock.renderer_geometry["title_logo"]["height"] == 48
    assert lock.renderer_geometry["title_logo"]["top_css_px"] == 26
    assert lock.renderer_geometry["title_logo"]["left_css_px"] == 77
    assert lock.renderer_geometry["decorative_art"]["width"] == 520
    assert lock.renderer_geometry["decorative_art"]["height"] == 530
    assert lock.renderer_geometry["content_brand_bug"]["width"] == 54
    assert lock.renderer_geometry["content_brand_bug"]["height"] == 40


def test_lock_brand_template_cli_writes_markdown_and_json(tmp_path: pathlib.Path) -> None:
    deck_dir = tmp_path / "decks" / "01_reference"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation.html").write_text(
        GOOD_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    output_md = tmp_path / "01-BRAND-REFERENCE-LOCK.md"
    output_json = tmp_path / "01-BRAND-REFERENCE-LOCK.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/lock_brand_template.py",
            "--workspace", str(tmp_path),
            "--deck-dir", str(deck_dir),
            "--output-md", str(output_md),
            "--output-json", str(output_json),
            "--expected-blockers", "",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert output_md.exists(), f"Markdown not written. stderr: {result.stderr}"
    assert output_json.exists(), f"JSON not written. stderr: {result.stderr}"
    receipt = json.loads(output_json.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 1
    assert receipt["receipt_type"] == "conference_pipeline.presentation.brand_template_lock"
    assert "status" in receipt
    assert "renderer_geometry" in receipt
    assert "findings" in receipt
    assert "blocker_count" in receipt


def test_topic4_canary_runner_rejects_non_topic4_scope(tmp_path: pathlib.Path) -> None:
    wrong_deck = tmp_path / "01_reasoning_post_training"
    wrong_deck.mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/run_topic4_canary.py",
            "--deck-dir", str(wrong_deck),
            "--phase-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (
        f"Expected exit 2 for non-Topic-4 deck, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )


def test_topic4_canary_runner_accepts_topic4_deck(tmp_path: pathlib.Path) -> None:
    topic4_deck = tmp_path / "conferences" / "iclr2026" / "decks" / "04_safety_alignment_privacy"
    topic4_deck.mkdir(parents=True)
    phase_dir = tmp_path / "empty_phase"
    phase_dir.mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/run_topic4_canary.py",
            "--deck-dir", str(topic4_deck),
            "--phase-dir", str(phase_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Expected exit 0 for Topic-4 deck with --dry-run, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )


@pytest.mark.xfail(reason="skills created in Plan 02")
def test_skill_files_exist() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    required_skills = [
        repo_root / ".codex" / "skills" / "conference-brand-template-lock" / "SKILL.md",
        repo_root / ".codex" / "skills" / "conference-editorial-storyboard" / "SKILL.md",
        repo_root / ".codex" / "skills" / "conference-rough-cut-review" / "SKILL.md",
        repo_root / ".codex" / "skills" / "conference-narration-review" / "SKILL.md",
        repo_root / ".codex" / "skills" / "conference-narration-fix" / "SKILL.md",
        repo_root / ".claude" / "skills" / "conference-presentation-blueprint" / "SKILL.md",
        repo_root / ".claude" / "skills" / "conference-figure-design-brief" / "SKILL.md",
    ]
    missing = [str(p) for p in required_skills if not p.exists()]
    assert missing == [], f"Missing skill files: {missing}"


@pytest.mark.xfail(reason="registry updated in Plan 04")
def test_phase01_brand_template_and_canary_owner_stage_bindings() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    registry_path = (
        repo_root / "src" / "conference_pipeline" / "stage_owner_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    required_keys = [
        "presentation.brand_template_lock",
        "presentation.editorial_source_board",
        "presentation.storyboard_generation",
        "presentation.storyboard_review",
        "presentation.rough_cut_review_packet",
        "presentation.rough_cut_review",
    ]
    missing = [k for k in required_keys if k not in registry]
    assert missing == [], f"Missing registry keys: {missing}"
