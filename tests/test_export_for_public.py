from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from conference_pipeline.publishing import GateClosed

REPO_ROOT = Path(__file__).parent.parent
REAL_MANIFEST = REPO_ROOT / "conferences" / "neurips2025" / "storage" / "public_export_manifest.json"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "public_export"


def load_export_module():
    path = REPO_ROOT / "scripts" / "export_for_public.py"
    spec = importlib.util.spec_from_file_location("export_for_public", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_stub(paths: list[Path]) -> list[dict]:
    return []


def _dirty_stub(paths: list[Path]) -> list[dict]:
    return [{"RuleID": "test-secret", "File": str(paths[0]) if paths else "none"}]


def _manifest(conference: str = "neurips2025", *, release_tag: str = "neurips2025-test") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "conference": conference,
        "release": {
            "source_sha": "abcdef1234567890",
            "release_tag": release_tag,
        },
        "families": [
            {
                "id": "presentations",
                "private_path": "decks",
                "public_path": "exports/public-release/{bundle}/decks/",
                "storage_manifest_id": "presentations",
                "redistribution_status": "allowed",
                "license": "CC-BY-SA-4.0",
                "provenance": "pipeline-generated",
                "format_contract": {"delivery": "inline-if-small", "max_inline_size_mb": 1},
                "checksum_required": True,
            },
            {
                "id": "podcasts",
                "private_path": "audio",
                "public_path": "exports/public-release/{bundle}/audio/",
                "storage_manifest_id": "podcasts",
                "redistribution_status": "allowed",
                "license": "CC-BY-SA-4.0",
                "provenance": "pipeline-generated",
                "large_media_policy": {"delivery": "cdn-only"},
                "cdn_base_url": "https://cdn.example.test/neurips2025",
                "checksum_required": True,
            },
            {
                "id": "transcripts",
                "private_path": "analysis/transcripts",
                "public_path": "exports/public-release/{bundle}/transcripts/",
                "storage_manifest_id": "transcripts",
                "redistribution_status": "caution",
                "license": "CC-BY-NC-ND-4.0",
                "provenance": "talk-derived",
                "checksum_required": True,
                "public_export_notes": "Operator acknowledgement required.",
            },
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_workspace(tmp_path: Path, *, manifest: dict[str, Any] | None = None) -> tuple[Path, Path]:
    conference_dir = tmp_path / "conferences" / "neurips2025"
    (conference_dir / "decks" / "topic-b").mkdir(parents=True)
    (conference_dir / "decks" / "topic-a").mkdir(parents=True)
    (conference_dir / "audio").mkdir(parents=True)
    (conference_dir / "analysis" / "transcripts").mkdir(parents=True)

    (conference_dir / "decks" / "topic-b" / "slides.md").write_text("slides b\n", encoding="utf-8")
    (conference_dir / "decks" / "topic-a" / "slides.md").write_text("slides a\n", encoding="utf-8")
    (conference_dir / "audio" / "episode.mp3").write_bytes(b"fake audio bytes")
    (conference_dir / "analysis" / "transcripts" / "talk.txt").write_text("transcript\n", encoding="utf-8")

    manifest_path = conference_dir / "storage" / "public_export_manifest.json"
    _write_json(manifest_path, manifest or _manifest())
    return conference_dir, manifest_path


def make_config(module, conference_dir: Path, manifest_path: Path, **overrides: Any):
    kwargs = {
        "conference_dir": conference_dir,
        "export_manifest": manifest_path,
        "output_dir": conference_dir / "exports" / "public-release",
        "bundle_id": None,
        "dry_run": True,
        "receipt_path": conference_dir / "storage" / "workflow-runs" / "receipt.json",
        "reviewed_dry_run_receipt": None,
        "ack_caution_families": ("transcripts",),
        "source_sha": None,
    }
    kwargs.update(overrides)
    return module.ExportConfig(**kwargs)


def make_reviewed_write_plan(module, conference_dir: Path, manifest_path: Path, **overrides: Any):
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    kwargs = {
        "dry_run": False,
        "reviewed_dry_run_receipt": dry_plan.config.receipt_path,
    }
    kwargs.update(overrides)
    return module.build_export_plan(make_config(module, conference_dir, manifest_path, **kwargs))


def copy_matching_reviewed_receipt(
    module,
    conference_dir: Path,
    manifest_path: Path,
    destination: Path,
) -> Path:
    reviewed = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / "matching-reviewed-source.json"
    )
    dry_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            receipt_path=reviewed,
        )
    )
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reviewed, destination)
    return destination


def materialize_staging(module, plan, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    module.materialize_items(plan, root)
    module.write_json_atomic(root / "RELEASE.json", module.release_payload(plan))
    module.write_checksums(root)
    return root


def make_changed_rebuild_plan(module, tmp_path: Path):
    conference_dir, manifest_path = make_workspace(tmp_path)
    first_plan = make_reviewed_write_plan(
        module,
        conference_dir,
        manifest_path,
        receipt_path=(
            conference_dir / "storage" / "workflow-runs" / "first-write.json"
        ),
    )
    module.execute_export_plan(first_plan, _scanner=_clean_stub)
    prior_digest = module.bundle_tree_sha256(first_plan.bundle_root)
    prior_bytes = {
        path.relative_to(first_plan.bundle_root).as_posix(): path.read_bytes()
        for path in module._portable_bundle_files(first_plan.bundle_root)
    }

    (conference_dir / "decks" / "topic-a" / "slides.md").write_text(
        "changed rebuild bytes\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release"]["source_sha"] = "fedcba0987654321"
    _write_json(manifest_path, manifest)
    reviewed_path = (
        conference_dir / "storage" / "workflow-runs" / "changed-review.json"
    )
    dry_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            receipt_path=reviewed_path,
        )
    )
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    rebuild_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=(
                conference_dir / "storage" / "workflow-runs" / "rebuild.json"
            ),
            reviewed_dry_run_receipt=reviewed_path,
        )
    )
    return rebuild_plan, prior_digest, prior_bytes


def assert_publication_debris_absent(plan) -> None:
    assert not any(plan.config.output_dir.glob(f".{plan.bundle_id}.staging-*"))
    assert not any(plan.config.output_dir.glob(f".{plan.bundle_id}.backup-*"))


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def classify_export_scan_call(plan, paths: list[Path]) -> tuple[str, Path | None]:
    normalized = [path.absolute() for path in paths]
    if normalized and all(_is_below(path, plan.bundle_root) for path in normalized):
        return "final", plan.bundle_root

    staging_roots: set[Path] = set()
    for path in normalized:
        try:
            relative = path.relative_to(plan.config.output_dir.absolute())
        except ValueError:
            continue
        if relative.parts and relative.parts[0].startswith(
            f".{plan.bundle_id}.staging-"
        ):
            staging_roots.add(plan.config.output_dir.absolute() / relative.parts[0])
    if len(staging_roots) == 1:
        staging_root = next(iter(staging_roots))
        if normalized and all(_is_below(path, staging_root) for path in normalized):
            return "staging", staging_root

    assert not any(_is_below(path, plan.bundle_root) for path in normalized)
    assert not staging_roots
    return "source", None


def assert_required_produced_scan_paths(
    plan,
    root: Path,
    paths: list[Path],
) -> set[str]:
    relative_paths = {
        path.absolute().relative_to(root.absolute()).as_posix() for path in paths
    }
    inline_paths = {item.release_path.as_posix() for item in plan.inline_items}
    sidecar_paths = {item.release_path.as_posix() for item in plan.cdn_items}

    assert "RELEASE.json" in relative_paths
    assert "CHECKSUMS.json" in relative_paths
    assert inline_paths
    assert inline_paths <= relative_paths
    assert sidecar_paths
    assert {
        path for path in relative_paths if path.endswith(".cdn.json")
    } == sidecar_paths
    return relative_paths


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_neurips_manifest_redaction_rules_are_the_ruled_ones():
    """Phase 6b, D25: figures named not copied, home paths and markers redacted in the deck
    text; marker redaction on the reader text and transcripts; no rule anywhere else."""
    data = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    kinds = {family["id"]: [rule["kind"] for rule in family.get("redaction_rules") or []] for family in data["families"]}
    assert kinds == {
        "reader_catalog": [],
        "reader_text": ["marker_text"],
        "transcripts": ["marker_text"],
        "presentation_markdown": ["figure_links", "absolute_path", "marker_text"],
        "presentation_manifest": [],
    }


def test_build_export_plan_reads_public_export_manifest_only(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    assert not (conference_dir / "storage" / "storage_manifest.json").exists()
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    assert plan.manifest_path == manifest_path.resolve()
    assert "presentations" in plan.included_families


def test_build_export_plan_is_deterministically_sorted(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    config = make_config(module, conference_dir, manifest_path)
    first = module.build_export_plan(config)
    second = module.build_export_plan(config)
    assert [item.release_path.as_posix() for item in first.items] == sorted(
        item.release_path.as_posix() for item in first.items
    )
    assert [item.release_path.as_posix() for item in first.items] == [
        item.release_path.as_posix() for item in second.items
    ]


def test_bundle_id_uses_cli_or_release_identity(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    explicit = module.build_export_plan(
        make_config(module, conference_dir, manifest_path, bundle_id="explicit-bundle")
    )
    from_manifest = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    assert explicit.bundle_id == "explicit-bundle"
    assert from_manifest.bundle_id == "neurips2025-test"


def test_dry_run_reports_families_exclusions_redactions_and_cdn_stubs(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    report = module.dry_run_export(plan, _scanner=_clean_stub)
    receipt = json.loads(plan.config.receipt_path.read_text(encoding="utf-8"))
    assert "Included families:" in report
    assert "Excluded artifacts:" in report
    assert "Redacted files: 0" in report
    assert "CDN stubs:" in report
    assert receipt["dry_run"] is True
    assert receipt["exit_code"] == 0
    assert receipt["redacted_files"] == []
    assert receipt["cdn_stubs"]


def test_dry_run_does_not_write_bundle_files(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(plan, _scanner=_clean_stub)
    assert not plan.bundle_root.exists()


def test_write_mode_materializes_release_relative_layout(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    bundle = module.execute_export_plan(plan, _scanner=_clean_stub)
    assert (bundle / "decks" / "topic-a" / "slides.md").exists()
    assert not (bundle / "exports" / "public-release").exists()


def test_write_mode_rejects_path_traversal(tmp_path: Path):
    module = load_export_module()
    manifest = _manifest()
    manifest["families"][0]["private_path"] = "../outside"
    conference_dir, manifest_path = make_workspace(tmp_path, manifest=manifest)
    with pytest.raises(module.PublicExportBlocked):
        module.build_export_plan(make_config(module, conference_dir, manifest_path))


def test_manifest_source_root_allows_workspace_symlink_without_exposing_target(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    external_audio = tmp_path / "drive-audio"
    external_audio.mkdir()
    (external_audio / "episode.mp3").write_bytes(b"drive audio bytes")
    shutil.rmtree(conference_dir / "audio")
    try:
        (conference_dir / "audio").symlink_to(external_audio, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    podcast_items = [item for item in plan.items if item.family_id == "podcasts"]

    assert [item.release_path.as_posix() for item in podcast_items] == ["audio/episode.mp3.cdn.json"]
    assert podcast_items[0].source_path.as_posix().startswith((conference_dir / "audio").as_posix())
    assert not podcast_items[0].source_path.as_posix().startswith(external_audio.resolve().as_posix())


def test_required_family_missing_source_writes_blocked_receipt(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    shutil.rmtree(conference_dir / "audio")
    config = make_config(module, conference_dir, manifest_path)

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    receipt = json.loads(config.receipt_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert receipt["status"] == "blocked"
    assert receipt["failure_stage"] == "build_export_plan"
    assert receipt["secret_values_recorded"] is False
    assert receipt["blocker_receipt"] == {
        "family_id": "podcasts",
        "path": "audio",
        "reason": "missing_required_source",
    }


def test_selected_broken_symlink_writes_blocked_receipt(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    broken_link = conference_dir / "decks" / "topic-a" / "selected-broken.md"
    try:
        broken_link.symlink_to(conference_dir / "missing" / "selected.md")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    config = make_config(module, conference_dir, manifest_path)

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    receipt = json.loads(config.receipt_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert receipt["status"] == "blocked"
    assert receipt["failure_stage"] == "build_export_plan"
    assert receipt["blocker_receipt"] == {
        "family_id": "presentations",
        "path": "decks/topic-a/selected-broken.md",
        "reason": "selected_broken_symlink",
    }
    assert "missing/selected.md" not in config.receipt_path.read_text(encoding="utf-8")


def test_excluded_broken_symlink_does_not_expand_scope(tmp_path: Path):
    module = load_export_module()
    manifest = _manifest()
    manifest["families"][0]["include_globs"] = ["**/*.md"]
    conference_dir, manifest_path = make_workspace(tmp_path, manifest=manifest)
    broken_link = conference_dir / "decks" / "topic-a" / "excluded.bin"
    try:
        broken_link.symlink_to(conference_dir / "missing" / "excluded.bin")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))

    assert all("excluded.bin" not in item.release_path.as_posix() for item in plan.items)
    assert any(
        artifact.path == "topic-a/excluded.bin"
        and artifact.reason == "not_matched_by_include_globs"
        for artifact in plan.excluded_artifacts
    )


def test_cdn_only_family_writes_sidecars_not_bytes(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    bundle = module.execute_export_plan(plan, _scanner=_clean_stub)
    assert (bundle / "audio" / "episode.mp3.cdn.json").exists()
    assert not (bundle / "audio" / "episode.mp3").exists()


def test_cdn_sidecar_records_real_source_checksum_and_size(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    source = conference_dir / "audio" / "episode.mp3"
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    bundle = module.execute_export_plan(plan, _scanner=_clean_stub)
    sidecar = json.loads((bundle / "audio" / "episode.mp3.cdn.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == module.sha256_file(source)
    assert sidecar["size_bytes"] == source.stat().st_size
    assert sidecar["mime_type"] == "audio/mpeg"


def test_cdn_source_mutation_after_plan_blocks_publication(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)

    (conference_dir / "audio" / "episode.mp3").write_bytes(
        b"mutated after the reviewed write plan"
    )

    with pytest.raises(
        module.PublicExportBlocked,
        match="cdn_source_identity_mismatch",
    ):
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert not plan.bundle_root.exists()
    assert not any(plan.config.output_dir.glob(f".{plan.bundle_id}.backup-*"))


def test_cdn_symlink_target_swap_after_plan_blocks_publication(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    source = conference_dir / "audio" / "episode.mp3"
    first_target = tmp_path / "first-target.mp3"
    second_target = tmp_path / "second-target.mp3"
    first_target.write_bytes(b"first reviewed target")
    second_target.write_bytes(b"second swapped target")
    source.unlink()
    try:
        source.symlink_to(first_target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)

    source.unlink()
    source.symlink_to(second_target)

    with pytest.raises(
        module.PublicExportBlocked,
        match="cdn_source_identity_mismatch",
    ) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert not plan.bundle_root.exists()
    serialized = json.dumps(exc_info.value.receipt, sort_keys=True)
    assert str(first_target) not in serialized
    assert str(second_target) not in serialized


def test_cdn_upload_source_contracts_are_deterministic(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    first_plan = module.build_export_plan(
        make_config(module, conference_dir, manifest_path)
    )

    first_rows = module.cdn_upload_source_contracts(first_plan)
    first_evidence = module.verify_cdn_source_identity(first_plan)
    repeated_evidence = module.verify_cdn_source_identity(first_plan)
    item = first_plan.cdn_items[0]

    assert first_rows == [
        {
            "family_id": item.family_id,
            "path": item.release_path.as_posix(),
            "object_key": item.cdn_object_key,
            "source_path": "audio/episode.mp3",
            "source_sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
    ]
    assert first_evidence == repeated_evidence
    assert first_evidence["cdn_upload_sources"] == first_rows
    assert len(first_evidence["cdn_upload_sources_sha256"]) == 64
    assert str(conference_dir) not in json.dumps(first_evidence, sort_keys=True)

    (conference_dir / "audio" / "episode.mp3").write_bytes(
        b"different reviewed source bytes"
    )
    changed_plan = module.build_export_plan(
        make_config(module, conference_dir, manifest_path)
    )
    changed_evidence = module.verify_cdn_source_identity(changed_plan)

    assert (
        changed_evidence["cdn_upload_sources_sha256"]
        != first_evidence["cdn_upload_sources_sha256"]
    )


def test_write_mode_writes_release_and_checksums(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    bundle = module.execute_export_plan(plan, _scanner=_clean_stub)
    assert (bundle / "RELEASE.json").exists()
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    assert "RELEASE.json" in checksums
    assert "audio/episode.mp3.cdn.json" in checksums


def test_materialized_bundle_verifier_records_complete_evidence(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    staging = materialize_staging(module, plan, tmp_path / "verified-staging")

    evidence = module.verify_materialized_bundle(plan, staging, _scanner=_clean_stub)
    sidecar = json.loads((staging / "audio" / "episode.mp3.cdn.json").read_text(encoding="utf-8"))

    assert evidence == {
        "status": "passed",
        "artifact_count": len(plan.items) + 2,
        "exact_artifact_set": True,
        "checksums_verified": True,
        "portable_regular_files": True,
        "cdn_sidecars_verified": True,
        "manifest_source_identity_verified": True,
        "safety_gates_verified": True,
        "bundle_tree_sha256": module.bundle_tree_sha256(staging),
    }
    assert sidecar["object_key"] == plan.cdn_items[0].cdn_object_key


@pytest.mark.parametrize("mutation", ["extra", "missing", "tampered", "checksum"])
def test_materialized_bundle_verifier_rejects_extra_missing_and_tampered_files(
    tmp_path: Path,
    mutation: str,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    staging = materialize_staging(module, plan, tmp_path / "mutated-staging")

    if mutation == "extra":
        (staging / "extra.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "missing":
        (staging / "decks" / "topic-a" / "slides.md").unlink()
    elif mutation == "tampered":
        (staging / "decks" / "topic-a" / "slides.md").write_text("tampered\n", encoding="utf-8")
    else:
        checksums = json.loads((staging / "CHECKSUMS.json").read_text(encoding="utf-8"))
        checksums["decks/topic-a/slides.md"]["size_bytes"] += 1
        _write_json(staging / "CHECKSUMS.json", checksums)

    with pytest.raises(module.PublicExportBlocked):
        module.verify_materialized_bundle(plan, staging, _scanner=_clean_stub)


def test_materialized_bundle_verifier_rejects_output_symlink(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    staging = materialize_staging(module, plan, tmp_path / "symlink-staging")
    target = staging / "decks" / "topic-a" / "slides.md"
    target.unlink()
    try:
        target.symlink_to(conference_dir / "decks" / "topic-a" / "slides.md")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    with pytest.raises(module.PublicExportBlocked, match="symlink"):
        module.verify_materialized_bundle(plan, staging, _scanner=_clean_stub)


def test_bundle_tree_sha256_rejects_non_regular_nodes(tmp_path: Path):
    module = load_export_module()
    root = tmp_path / "tree"
    root.mkdir()
    (root / "regular.txt").write_text("regular\n", encoding="utf-8")
    try:
        (root / "linked.txt").symlink_to(root / "regular.txt")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    with pytest.raises(module.PublicExportBlocked, match="symlink"):
        module.bundle_tree_sha256(root)


def test_failed_first_publish_leaves_no_final_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    parent = tmp_path / "public-release"
    parent.mkdir()
    staging = parent / ".bundle.staging-test"
    staging.mkdir()
    (staging / "new.txt").write_text("new\n", encoding="utf-8")
    final = parent / "bundle"
    real_replace = module.os.replace

    def fail_publish(source, destination):
        if Path(source) == staging and Path(destination) == final:
            raise OSError("injected first publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_publish)

    with pytest.raises(module.PublicExportBlocked, match="atomic_publication_failed"):
        module.publish_verified_staging(staging, final)

    assert not final.exists()
    assert not staging.exists()
    assert not any(parent.glob(".bundle.backup-*"))


def test_failed_rebuild_restores_prior_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    parent = tmp_path / "public-release"
    parent.mkdir()
    final = parent / "bundle"
    final.mkdir()
    (final / "old.txt").write_text("old bytes\n", encoding="utf-8")
    prior_digest = module.bundle_tree_sha256(final)
    staging = parent / ".bundle.staging-test"
    staging.mkdir()
    (staging / "new.txt").write_text("new bytes\n", encoding="utf-8")
    real_replace = module.os.replace

    def fail_replacement(source, destination):
        if Path(source) == staging and Path(destination) == final:
            raise OSError("injected rebuild publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_replacement)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.publish_verified_staging(staging, final)

    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert module.bundle_tree_sha256(final) == prior_digest
    assert (final / "old.txt").read_text(encoding="utf-8") == "old bytes\n"
    assert not staging.exists()
    assert not any(parent.glob(".bundle.backup-*"))


def test_final_verification_failure_rolls_back_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    real_verify = module.verify_materialized_bundle

    def fail_final(candidate_plan, root, *, _scanner=None):
        if root == plan.bundle_root:
            raise module.PublicExportBlocked("injected_final_verification")
        return real_verify(candidate_plan, root, _scanner=_scanner)

    monkeypatch.setattr(module, "verify_materialized_bundle", fail_final)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "final_verification"
    )
    assert not plan.bundle_root.exists()
    assert_publication_debris_absent(plan)


def test_final_verification_failure_restores_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    plan, prior_digest, prior_bytes = make_changed_rebuild_plan(module, tmp_path)
    real_verify = module.verify_materialized_bundle

    def fail_final(candidate_plan, root, *, _scanner=None):
        if root == plan.bundle_root:
            raise module.PublicExportBlocked("injected_final_verification")
        return real_verify(candidate_plan, root, _scanner=_scanner)

    monkeypatch.setattr(module, "verify_materialized_bundle", fail_final)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    restored_bytes = {
        path.relative_to(plan.bundle_root).as_posix(): path.read_bytes()
        for path in module._portable_bundle_files(plan.bundle_root)
    }
    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "final_verification"
    )
    assert module.bundle_tree_sha256(plan.bundle_root) == prior_digest
    assert restored_bytes == prior_bytes
    assert_publication_debris_absent(plan)


def test_success_receipt_failure_rolls_back_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)

    def fail_receipt(*_args, **_kwargs):
        raise OSError("injected success receipt failure")

    monkeypatch.setattr(module, "write_export_receipt", fail_receipt)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "success_receipt_persistence"
    )
    assert not plan.bundle_root.exists()
    assert_publication_debris_absent(plan)


def test_success_receipt_failure_restores_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    plan, prior_digest, prior_bytes = make_changed_rebuild_plan(module, tmp_path)

    def fail_receipt(*_args, **_kwargs):
        backup_paths = list(
            plan.config.output_dir.glob(f".{plan.bundle_id}.backup-*")
        )
        assert len(backup_paths) == 1
        raise OSError("injected success receipt failure")

    monkeypatch.setattr(module, "write_export_receipt", fail_receipt)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    restored_bytes = {
        path.relative_to(plan.bundle_root).as_posix(): path.read_bytes()
        for path in module._portable_bundle_files(plan.bundle_root)
    }
    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "success_receipt_persistence"
    )
    assert module.bundle_tree_sha256(plan.bundle_root) == prior_digest
    assert restored_bytes == prior_bytes
    assert_publication_debris_absent(plan)


def test_success_receipt_persists_before_rebuild_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    plan, _prior_digest, _prior_bytes = make_changed_rebuild_plan(module, tmp_path)
    real_write = module.write_export_receipt
    observed_backup = False

    def observe_receipt(*args, **kwargs):
        nonlocal observed_backup
        backup_paths = list(
            plan.config.output_dir.glob(f".{plan.bundle_id}.backup-*")
        )
        assert len(backup_paths) == 1
        observed_backup = True
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "write_export_receipt", observe_receipt)

    module.execute_export_plan(plan, _scanner=_clean_stub)

    receipt = json.loads(plan.config.receipt_path.read_text(encoding="utf-8"))
    assert observed_backup is True
    assert receipt["publication"]["rollback_performed"] is False
    assert (
        receipt["publication"]["rollback_authority_retained_through_receipt"]
        is True
    )
    assert receipt["publication"]["failure_point"] is None
    assert_publication_debris_absent(plan)


def test_partial_backup_cleanup_keeps_verified_final_and_never_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    plan, prior_digest, prior_bytes = make_changed_rebuild_plan(module, tmp_path)
    original_remove = module._remove_publication_path
    cleanup_calls: list[Path] = []

    def partially_remove_backup(path: Path) -> None:
        if path.name.startswith(f".{plan.bundle_id}.backup-"):
            cleanup_calls.append(path)
            (path / "RELEASE.json").unlink()
            raise OSError(
                "TOKEN_SENTINEL=cleanup-secret /Volumes/Google Drive/private-backup"
            )
        original_remove(path)

    monkeypatch.setattr(module, "_remove_publication_path", partially_remove_backup)

    exit_code, _message = module.run_export(plan.config, _scanner=_clean_stub)

    receipt_text = plan.config.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    backup_paths = list(
        plan.config.output_dir.glob(f".{plan.bundle_id}.backup-*")
    )
    current_bytes = {
        path.relative_to(plan.bundle_root).as_posix(): path.read_bytes()
        for path in module._portable_bundle_files(plan.bundle_root)
    }

    assert exit_code == 1
    assert len(cleanup_calls) == 1
    assert len(backup_paths) == 1
    assert not (backup_paths[0] / "RELEASE.json").exists()
    assert (plan.bundle_root / "RELEASE.json").is_file()
    assert (plan.bundle_root / "CHECKSUMS.json").is_file()
    verification = module.verify_materialized_bundle(
        plan,
        plan.bundle_root,
        _scanner=_clean_stub,
    )
    assert verification["exact_artifact_set"] is True
    assert module.bundle_tree_sha256(plan.bundle_root) != prior_digest
    assert current_bytes != prior_bytes
    assert receipt["publication"]["rollback_performed"] is False
    assert receipt["publication"]["cleanup_pending"] is True
    assert receipt["publication"]["failure_point"] == "publication_backup_cleanup"
    assert receipt["secret_values_recorded"] is False
    assert "cleanup-secret" not in receipt_text
    assert "Google Drive" not in receipt_text


def test_execute_publish_failure_preserves_rollback_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    plan, prior_digest, prior_bytes = make_changed_rebuild_plan(module, tmp_path)
    real_replace = module.os.replace

    def fail_staging_publish(source, destination, *args, **kwargs):
        if (
            Path(source).name.startswith(f".{plan.bundle_id}.staging-")
            and Path(destination) == plan.bundle_root
        ):
            raise OSError("injected staging publish failure")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "replace", fail_staging_publish)

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    restored_bytes = {
        path.relative_to(plan.bundle_root).as_posix(): path.read_bytes()
        for path in module._portable_bundle_files(plan.bundle_root)
    }
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "staging_to_final"
    )
    assert exc_info.value.receipt["publication"]["rollback_performed"] is True
    assert module.bundle_tree_sha256(plan.bundle_root) == prior_digest
    assert restored_bytes == prior_bytes
    assert_publication_debris_absent(plan)


def test_write_receipt_records_manifest_digest_source_sha_and_exit_code(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "write.json"
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path, receipt_path=receipt_path)
    module.execute_export_plan(plan, _scanner=_clean_stub)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_sha"] == "abcdef1234567890"
    assert receipt["export_manifest_sha256"] == module.sha256_file(manifest_path)
    assert receipt["exit_code"] == 0
    assert receipt["planned_artifact_count"] == len(plan.items)
    assert receipt["actual_artifact_count"] > 0
    assert receipt["license_gating"]["caution_families"] == ["transcripts"]
    assert receipt["cdn_upload_sources"] == module.cdn_upload_source_contracts(plan)
    assert len(receipt["cdn_upload_sources_sha256"]) == 64


def test_release_files_are_deterministic_across_repeated_runs(tmp_path: Path):
    module = load_export_module()
    c1, m1 = make_workspace(tmp_path / "one")
    c2, m2 = make_workspace(tmp_path / "two")
    p1 = make_reviewed_write_plan(module, c1, m1)
    p2 = make_reviewed_write_plan(module, c2, m2)
    b1 = module.execute_export_plan(p1, _scanner=_clean_stub)
    b2 = module.execute_export_plan(p2, _scanner=_clean_stub)
    assert (b1 / "RELEASE.json").read_bytes() == (b2 / "RELEASE.json").read_bytes()
    assert (b1 / "CHECKSUMS.json").read_bytes() == (b2 / "CHECKSUMS.json").read_bytes()


def test_existing_bundle_refuses_overwrite_by_default(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path, dry_run=False))
    plan.bundle_root.mkdir(parents=True)
    (plan.bundle_root / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(module.PublicExportBlocked, match=module.ERROR_REVIEWED_RECEIPT):
        module.execute_export_plan(plan, _scanner=_clean_stub)


def test_write_mode_requires_reviewed_dry_run_receipt_before_first_write(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path, dry_run=False))

    with pytest.raises(module.PublicExportBlocked, match=module.ERROR_REVIEWED_RECEIPT):
        module.execute_export_plan(plan, _scanner=_clean_stub)


def test_reviewed_dry_run_receipt_allows_matching_overwrite(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    first_receipt = conference_dir / "storage" / "workflow-runs" / "first-write.json"
    first_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=first_receipt,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    module.execute_export_plan(first_plan, _scanner=_clean_stub)
    write_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=conference_dir / "storage" / "workflow-runs" / "rebuild.json",
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    bundle = module.execute_export_plan(write_plan, _scanner=_clean_stub)
    assert (bundle / "RELEASE.json").exists()


def test_direct_exporter_rejects_external_reviewed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    external = copy_matching_reviewed_receipt(
        module,
        conference_dir,
        manifest_path,
        tmp_path / "external-reviewed.json",
    )
    plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=external,
        )
    )
    monkeypatch.setattr(
        module,
        "load_receipt",
        lambda _path: pytest.fail("external reviewed receipt must not be read"),
    )

    with pytest.raises(module.PublicExportBlocked, match="unsafe_receipt_path") as raised:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert raised.value.receipt == {
        "label": "reviewed_dry_run_receipt",
        "reason": "unsafe_receipt_path",
        "reason_code": "outside_workflow_runs",
    }
    assert not plan.bundle_root.exists()


def test_run_export_rejects_external_reviewed_receipt_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    external = copy_matching_reviewed_receipt(
        module,
        conference_dir,
        manifest_path,
        tmp_path / "external-reviewed.json",
    )
    external_bytes = external.read_bytes()
    config = make_config(
        module,
        conference_dir,
        manifest_path,
        dry_run=False,
        receipt_path=None,
        reviewed_dry_run_receipt=external,
    )
    monkeypatch.setattr(
        module,
        "load_receipt",
        lambda _path: pytest.fail("external reviewed receipt must not be read"),
    )

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    safe_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / "public-export-neurips2025-test-write-receipt.json"
    )
    receipt = json.loads(safe_receipt.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert not (config.output_dir / "neurips2025-test").exists()
    assert external.read_bytes() == external_bytes
    assert receipt["blocker_receipt"] == {
        "label": "reviewed_dry_run_receipt",
        "reason": "unsafe_receipt_path",
        "reason_code": "outside_workflow_runs",
    }
    assert receipt["secret_values_recorded"] is False
    assert str(external) not in safe_receipt.read_text(encoding="utf-8")


def test_direct_exporter_rejects_symlinked_reviewed_receipt_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    external = copy_matching_reviewed_receipt(
        module,
        conference_dir,
        manifest_path,
        tmp_path / "external-reviewed.json",
    )
    linked = (
        conference_dir / "storage" / "workflow-runs" / "linked-reviewed.json"
    )
    try:
        linked.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=linked,
        )
    )
    monkeypatch.setattr(
        module,
        "load_receipt",
        lambda _path: pytest.fail("symlinked reviewed receipt must not be read"),
    )

    with pytest.raises(module.PublicExportBlocked, match="unsafe_receipt_path") as raised:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert raised.value.receipt["reason_code"] == "symlinked_receipt_component"
    assert not plan.bundle_root.exists()


def test_direct_exporter_rejects_symlinked_reviewed_receipt_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path / "workspace")
    external_dir = tmp_path / "external-receipts"
    external = copy_matching_reviewed_receipt(
        module,
        conference_dir,
        manifest_path,
        external_dir / "reviewed.json",
    )
    linked_dir = conference_dir / "storage" / "workflow-runs" / "linked"
    try:
        linked_dir.symlink_to(external_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=linked_dir / external.name,
        )
    )
    monkeypatch.setattr(
        module,
        "load_receipt",
        lambda _path: pytest.fail("symlinked reviewed receipt must not be read"),
    )

    with pytest.raises(module.PublicExportBlocked, match="unsafe_receipt_path") as raised:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert raised.value.receipt["reason_code"] == "symlinked_receipt_component"
    assert not plan.bundle_root.exists()


def test_write_receipt_records_normalized_confined_reviewed_path(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    reviewed = (
        conference_dir / "storage" / "workflow-runs" / "relative-reviewed.json"
    )
    dry_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            receipt_path=reviewed,
        )
    )
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    relative_reviewed = reviewed.relative_to(tmp_path)
    write_receipt = (
        conference_dir / "storage" / "workflow-runs" / "normalized-write.json"
    )
    plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=write_receipt,
            reviewed_dry_run_receipt=relative_reviewed,
        )
    )

    module.execute_export_plan(plan, _scanner=_clean_stub)

    receipt = json.loads(write_receipt.read_text(encoding="utf-8"))
    assert receipt["reviewed_dry_run_receipt_path"] == str(reviewed.absolute())
    assert receipt["planned_artifacts"] == module.planned_artifacts(plan)
    assert receipt["cdn_stubs"] == module.cdn_stub_artifacts(plan)


def test_reviewed_receipt_object_key_drift_blocks_write(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    receipt = json.loads(dry_plan.config.receipt_path.read_text(encoding="utf-8"))
    receipt["cdn_stubs"][0]["object_key"] += "-drift"
    _write_json(dry_plan.config.receipt_path, receipt)
    write_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )

    with pytest.raises(module.PublicExportBlocked, match="cdn_stubs"):
        module.execute_export_plan(write_plan, _scanner=_clean_stub)


def test_released_at_freezes_bundle(tmp_path: Path):
    module = load_export_module()
    manifest = _manifest()
    manifest["release"]["released_at"] = "2026-07-09T00:00:00Z"
    conference_dir, manifest_path = make_workspace(tmp_path, manifest=manifest)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)

    with pytest.raises(module.PublicExportBlocked, match="immutable") as exc_info:
        module.execute_export_plan(plan, _scanner=_clean_stub)

    assert exc_info.value.receipt["publication"]["published_version_evidence"]
    assert not plan.bundle_root.exists()


def test_existing_release_released_at_freezes_bundle(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    reviewed_path = conference_dir / "storage" / "workflow-runs" / "receipt.json"
    plan = make_reviewed_write_plan(
        module,
        conference_dir,
        manifest_path,
        receipt_path=conference_dir / "storage" / "workflow-runs" / "first-write.json",
    )
    module.execute_export_plan(plan, _scanner=_clean_stub)
    release_path = plan.bundle_root / "RELEASE.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["release"]["released_at"] = "2026-07-09T00:00:00Z"
    _write_json(release_path, release)
    rebuild = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=reviewed_path,
        )
    )

    with pytest.raises(module.PublicExportBlocked, match="immutable"):
        module.execute_export_plan(rebuild, _scanner=_clean_stub)


def test_successful_live_receipt_freezes_bundle(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    live_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / f"github-{plan.bundle_id}-live-receipt.json"
    )
    _write_json(
        live_receipt,
        {
            "receipt_type": "conference_pipeline.provider_success",
            "status": "succeeded",
            "exit_code": 0,
                "live": True,
                "dry_run": False,
                "bundle_id": plan.bundle_id,
                "public_url": "https://github.example.test/iclr2026-data",
                "commit_sha": "abc123",
            },
        )

    evidence = module.published_bundle_evidence(plan)

    assert len(evidence) == 1
    with pytest.raises(module.PublicExportBlocked, match="immutable"):
        module.execute_export_plan(plan, _scanner=_clean_stub)
    assert not plan.bundle_root.exists()


def test_false_live_filename_and_planning_only_cdn_receipt_do_not_freeze(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    receipts = conference_dir / "storage" / "workflow-runs"
    _write_json(
        receipts / "false-live-receipt.json",
        {
            "receipt_type": "conference_pipeline.provider_success",
            "status": "succeeded",
            "exit_code": 0,
            "live": False,
            "dry_run": False,
            "bundle_id": plan.bundle_id,
            "public_url": "https://example.test/data",
            "commit_sha": "abc123",
        },
    )
    _write_json(
        receipts / "planning-only-cdn-receipt.json",
        {
            "receipt_type": "conference_pipeline.public_cdn_upload",
            "status": "succeeded",
            "exit_code": 0,
            "live": True,
            "dry_run": False,
            "bundle_id": plan.bundle_id,
            "planned_upload_count": 1,
            "planned_uploads": [{"object_key": "audio.mp3", "sha256": "abc", "size_bytes": 1}],
            "uploaded_count": 0,
            "uploaded_objects": [],
        },
    )
    assert module.published_bundle_evidence(plan) == []


def test_failed_dry_run_and_unrelated_live_receipts_do_not_freeze_bundle(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    receipts = conference_dir / "storage" / "workflow-runs"
    ignored = [
        ("github-neurips2025-test-live-receipt.json", {"status": "failed", "exit_code": 1, "live": True}),
        ("hf-neurips2025-test-live-receipt.json", {"status": "succeeded", "exit_code": 0, "dry_run": True}),
        ("github-other-bundle-live-receipt.json", {"status": "succeeded", "exit_code": 0, "live": True}),
    ]
    for name, payload in ignored:
        _write_json(receipts / name, {"receipt_type": "conference_pipeline.provider_success", **payload})

    assert module.published_bundle_evidence(plan) == []
    assert module.execute_export_plan(plan, _scanner=_clean_stub).exists()


def test_unpublished_rebuild_records_equal_tree_digests(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    first_receipt = conference_dir / "storage" / "workflow-runs" / "first-write.json"
    first = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=first_receipt,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    module.execute_export_plan(first, _scanner=_clean_stub)
    first_checksums = (first.bundle_root / "CHECKSUMS.json").read_bytes()
    rebuild_receipt = conference_dir / "storage" / "workflow-runs" / "rebuild.json"
    rebuild = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=rebuild_receipt,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )

    module.execute_export_plan(rebuild, _scanner=_clean_stub)
    receipt = json.loads(rebuild_receipt.read_text(encoding="utf-8"))

    assert receipt["reviewed_dry_run_receipt_path"] == str(dry_plan.config.receipt_path)
    assert receipt["reproducibility"] == {
        "source_sha": rebuild.source_sha,
        "export_manifest_sha256": rebuild.manifest_sha256,
        "prior_bundle_tree_sha256": receipt["bundle_verification"]["bundle_tree_sha256"],
        "current_bundle_tree_sha256": receipt["bundle_verification"]["bundle_tree_sha256"],
        "byte_identical": True,
    }
    assert (rebuild.bundle_root / "CHECKSUMS.json").read_bytes() == first_checksums


def test_unpublished_rebuild_blocks_tree_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    first = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=conference_dir / "storage" / "workflow-runs" / "first-write.json",
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    module.execute_export_plan(first, _scanner=_clean_stub)
    prior_digest = module.bundle_tree_sha256(first.bundle_root)
    rebuild = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            receipt_path=conference_dir / "storage" / "workflow-runs" / "mismatch.json",
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    real_digest = module.bundle_tree_sha256

    def mismatched_digest(root: Path) -> str:
        digest = real_digest(root)
        if root != first.bundle_root:
            return "0" * 64
        return digest

    monkeypatch.setattr(module, "bundle_tree_sha256", mismatched_digest)

    with pytest.raises(module.PublicExportBlocked, match="reproducibility"):
        module.execute_export_plan(rebuild, _scanner=_clean_stub)

    assert module.sha256_file(first.bundle_root / "CHECKSUMS.json")
    assert real_digest(first.bundle_root) == prior_digest


def test_mismatched_reviewed_receipt_blocks_overwrite(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path, bundle_id="other"))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    write_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            bundle_id="current",
            dry_run=False,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )
    write_plan.bundle_root.mkdir(parents=True)
    (write_plan.bundle_root / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(module.PublicExportBlocked, match=module.ERROR_REVIEWED_RECEIPT):
        module.execute_export_plan(write_plan, _scanner=_clean_stub)


def test_reviewed_receipt_blocks_when_planned_artifact_changes(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    (conference_dir / "decks" / "topic-a" / "slides.md").write_text("changed slides\n", encoding="utf-8")
    write_plan = module.build_export_plan(
        make_config(
            module,
            conference_dir,
            manifest_path,
            dry_run=False,
            reviewed_dry_run_receipt=dry_plan.config.receipt_path,
        )
    )

    with pytest.raises(module.PublicExportBlocked, match="planned_artifacts"):
        module.execute_export_plan(write_plan, _scanner=_clean_stub)


def test_write_mode_rejects_placeholder_cdn_base_url(tmp_path: Path):
    module = load_export_module()
    manifest = _manifest()
    manifest["families"][1]["cdn_base_url"] = "https://pub-placeholder.example.invalid/neurips2025"
    conference_dir, manifest_path = make_workspace(tmp_path, manifest=manifest)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)

    with pytest.raises(module.PublicExportBlocked, match=module.ERROR_PLACEHOLDER_CDN):
        module.execute_export_plan(plan, _scanner=_clean_stub)


def test_cdn_targets_reject_malformed_or_bundleless_urls(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    family = next(entry for entry in plan.families if entry.family_id == "podcasts")
    media_release_path = Path("audio/episode.mp3")
    canonical_url = (
        "https://cdn.example.test/neurips2025/"
        "neurips2025-test/audio/episode.mp3"
    )
    cdn_item = plan.cdn_items[0]

    assert cdn_item.cdn_url == canonical_url
    assert cdn_item.cdn_object_key == (
        "neurips2025/neurips2025-test/audio/episode.mp3"
    )
    module.dry_run_export(plan, _scanner=_clean_stub)
    receipt = json.loads(plan.config.receipt_path.read_text(encoding="utf-8"))
    assert receipt["cdn_stubs"][0]["object_key"] == cdn_item.cdn_object_key

    assert module.validate_cdn_target(
        family,
        plan.bundle_id,
        media_release_path,
        canonical_url,
    ) == "neurips2025/neurips2025-test/audio/episode.mp3"

    invalid_urls = [
        canonical_url.replace("https://", "http://", 1),
        canonical_url.replace("https://", "https://user:secret@", 1),
        f"{canonical_url}?download=1",
        f"{canonical_url}#fragment",
        "https://cdn.example.test/neurips2025/audio/episode.mp3",
        f"{canonical_url}/neurips2025-test",
    ]
    for invalid_url in invalid_urls:
        with pytest.raises(module.PublicExportBlocked):
            module.validate_cdn_target(
                family,
                plan.bundle_id,
                media_release_path,
                invalid_url,
            )


def test_duplicate_release_path_fails_closed(tmp_path: Path):
    module = load_export_module()
    manifest = _manifest()
    duplicate = dict(manifest["families"][0])
    duplicate["id"] = "duplicate_presentations"
    duplicate["storage_manifest_id"] = "duplicate_presentations"
    manifest["families"].append(duplicate)
    conference_dir, manifest_path = make_workspace(tmp_path, manifest=manifest)

    with pytest.raises(module.PublicExportBlocked, match="duplicate_release_path") as exc_info:
        module.build_export_plan(make_config(module, conference_dir, manifest_path))

    assert exc_info.value.receipt["path"] == "decks/topic-a/slides.md"


def test_secret_like_nested_release_path_fails_closed(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    nested_secret_dir = conference_dir / "decks" / "topic-a" / ".env"
    nested_secret_dir.mkdir()
    (nested_secret_dir / "slides.md").write_text("must not export\n", encoding="utf-8")

    with pytest.raises(module.PublicExportBlocked, match="secret_like_release_path") as exc_info:
        module.build_export_plan(make_config(module, conference_dir, manifest_path))

    assert exc_info.value.receipt == {
        "family_id": "presentations",
        "path": "decks/topic-a/.env/slides.md",
        "reason": "secret_like_release_path",
    }


def test_plan_construction_failure_writes_sanitized_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    config = make_config(module, conference_dir, manifest_path)

    def fail_plan(_config):
        raise OSError("API_TOKEN=never-record-me /Volumes/Google Drive/private")

    monkeypatch.setattr(module, "build_export_plan", fail_plan)

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    receipt_text = config.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert exit_code == 1
    assert receipt["status"] == "failed"
    assert receipt["failure_stage"] == "build_export_plan"
    assert receipt["secret_values_recorded"] is False
    assert receipt["blocker_receipt"] == {"reason": "OSError"}
    assert "never-record-me" not in receipt_text
    assert "Google Drive/private" not in receipt_text


def test_failed_receipt_sanitizes_write_filesystem_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    dry_plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(dry_plan, _scanner=_clean_stub)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "failed-write.json"
    config = make_config(
        module,
        conference_dir,
        manifest_path,
        dry_run=False,
        receipt_path=receipt_path,
        reviewed_dry_run_receipt=dry_plan.config.receipt_path,
    )

    def fail_write(_plan, *, _scanner=None):
        raise OSError("PASSWORD=never-record-me /Volumes/Google Drive/private")

    monkeypatch.setattr(module, "execute_export_plan", fail_write)

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert exit_code == 1
    assert receipt["status"] == "failed"
    assert receipt["failure_stage"] == "execute_export_plan"
    assert receipt["secret_values_recorded"] is False
    assert receipt["blocker_receipt"] == {"reason": "OSError"}
    assert "never-record-me" not in receipt_text
    assert "Google Drive/private" not in receipt_text


@pytest.mark.parametrize(
    "target_kind",
    [
        "credentials",
        "manifest",
        "selected_source",
        "release_metadata",
        "checksums",
        "bundle_descendant",
        "unrelated_project_file",
    ],
)
def test_export_receipt_paths_are_confined_to_workflow_runs(
    tmp_path: Path,
    target_kind: str,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    bundle_root = (
        conference_dir
        / "exports"
        / "public-release"
        / "neurips2025-test"
    )
    targets = {
        "credentials": tmp_path / ".env",
        "manifest": manifest_path,
        "selected_source": conference_dir / "decks" / "topic-a" / "slides.md",
        "release_metadata": bundle_root / "RELEASE.json",
        "checksums": bundle_root / "CHECKSUMS.json",
        "bundle_descendant": bundle_root / "decks" / "topic-a" / "slides.md",
        "unrelated_project_file": tmp_path / "unrelated.json",
    }
    target = targets[target_kind]
    target.parent.mkdir(parents=True, exist_ok=True)
    sentinel = f"sentinel:{target_kind}\n".encode()
    target.write_bytes(sentinel)
    config = make_config(
        module,
        conference_dir,
        manifest_path,
        bundle_id="neurips2025-test",
        receipt_path=target,
    )

    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    assert exit_code == 1
    assert target.read_bytes() == sentinel
    safe_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / "public-export-neurips2025-test-dry-run-receipt.json"
    )
    receipt = json.loads(safe_receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["blocker_receipt"] == {
        "label": "receipt_path",
        "reason": "unsafe_receipt_path",
        "reason_code": "outside_workflow_runs",
    }
    assert receipt["secret_values_recorded"] is False


def test_export_receipt_path_accepts_default_and_nested_workflow_receipts(
    tmp_path: Path,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    default_config = make_config(
        module,
        conference_dir,
        manifest_path,
        bundle_id="neurips2025-test",
        receipt_path=None,
    )

    default_exit, _message = module.run_export(default_config, _scanner=_clean_stub)

    default_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / "public-export-neurips2025-test-dry-run-receipt.json"
    )
    assert default_exit == 0
    assert default_receipt.is_file() and not default_receipt.is_symlink()

    nested_receipt = (
        conference_dir
        / "storage"
        / "workflow-runs"
        / "nested"
        / "proof"
        / "receipt.json"
    )
    nested_config = make_config(
        module,
        conference_dir,
        manifest_path,
        bundle_id="neurips2025-test",
        receipt_path=nested_receipt.relative_to(tmp_path),
    )

    nested_exit, _message = module.run_export(nested_config, _scanner=_clean_stub)

    assert nested_exit == 0
    assert nested_receipt.is_file() and not nested_receipt.is_symlink()


@pytest.mark.parametrize("symlink_kind", ["destination", "ancestor", "workflow_root"])
def test_export_receipt_path_rejects_symlinked_destination_or_ancestor(
    tmp_path: Path,
    symlink_kind: str,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    receipt_root = conference_dir / "storage" / "workflow-runs"
    receipt_root.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / f"{symlink_kind}-sentinel.json"
    sentinel_bytes = f"sentinel:{symlink_kind}\n".encode()
    sentinel.write_bytes(sentinel_bytes)
    safe_receipt = receipt_root / "public-export-neurips2025-test-dry-run-receipt.json"

    try:
        if symlink_kind == "destination":
            receipt_path = receipt_root / "linked-receipt.json"
            receipt_path.symlink_to(sentinel)
        elif symlink_kind == "ancestor":
            external = tmp_path / "external-receipts"
            external.mkdir()
            linked_dir = receipt_root / "linked-dir"
            linked_dir.symlink_to(external, target_is_directory=True)
            receipt_path = linked_dir / "receipt.json"
        else:
            receipt_root.rmdir()
            external = tmp_path / "external-workflow-runs"
            external.mkdir()
            receipt_root.symlink_to(external, target_is_directory=True)
            receipt_path = receipt_root / "receipt.json"
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    config = make_config(
        module,
        conference_dir,
        manifest_path,
        bundle_id="neurips2025-test",
        receipt_path=receipt_path,
    )
    exit_code, _message = module.run_export(config, _scanner=_clean_stub)

    assert exit_code == 1
    assert sentinel.read_bytes() == sentinel_bytes
    if symlink_kind == "destination":
        assert receipt_path.is_symlink()
    else:
        assert not receipt_path.exists()
    if symlink_kind != "workflow_root":
        receipt = json.loads(safe_receipt.read_text(encoding="utf-8"))
        assert receipt["blocker_receipt"]["reason_code"] == "symlinked_receipt_component"


def test_export_receipt_temp_symlink_cannot_redirect_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_export_module()
    from conference_pipeline.publishing import receipt_paths

    receipt_dir = tmp_path / "workflow-runs"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "receipt.json"
    sentinel = tmp_path / "credential-sentinel.json"
    sentinel_bytes = b"credential sentinel bytes\n"
    sentinel.write_bytes(sentinel_bytes)
    collision = receipt_dir / ".receipt.json.collision.tmp"
    try:
        collision.symlink_to(sentinel)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    tokens = iter(["collision", "fresh"])
    monkeypatch.setattr(receipt_paths.secrets, "token_hex", lambda _size: next(tokens))

    module.write_json_atomic(receipt_path, {"status": "succeeded"})

    assert sentinel.read_bytes() == sentinel_bytes
    assert collision.is_symlink()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "status": "succeeded"
    }
    assert receipt_path.is_file() and not receipt_path.is_symlink()
    assert sorted(path.name for path in receipt_dir.iterdir()) == [
        ".receipt.json.collision.tmp",
        "receipt.json",
    ]


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("poisoned_raw_talk_video_manifest.json", "raw_talk_video"),
        ("poisoned_slide_image_manifest.json", "slide_images"),
        ("poisoned_slide_ocr_manifest.json", "slide_ocr"),
        ("poisoned_full_pdf_manifest.json", "full_pdfs"),
    ],
)
def test_prohibited_public_export_fixtures_fail_closed(fixture_name: str, expected: str):
    module = load_export_module()
    with pytest.raises(GateClosed) as exc_info:
        module.load_export_manifest(FIXTURE_DIR / fixture_name)
    findings = exc_info.value.receipt["findings"]
    assert any(finding.get("family_id") == expected for finding in findings)


def test_raw_talk_video_fixture_fails_closed():
    test_prohibited_public_export_fixtures_fail_closed("poisoned_raw_talk_video_manifest.json", "raw_talk_video")


def test_slide_image_fixture_fails_closed():
    test_prohibited_public_export_fixtures_fail_closed("poisoned_slide_image_manifest.json", "slide_images")


def test_slide_ocr_fixture_fails_closed():
    test_prohibited_public_export_fixtures_fail_closed("poisoned_slide_ocr_manifest.json", "slide_ocr")


def test_full_pdf_fixture_fails_closed():
    test_prohibited_public_export_fixtures_fail_closed("poisoned_full_pdf_manifest.json", "full_pdfs")


def test_transcripts_caution_requires_ack(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(
        make_config(module, conference_dir, manifest_path, ack_caution_families=())
    )
    with pytest.raises(module.PublicExportBlocked, match=module.ERROR_CAUTION_ACK):
        module.dry_run_export(plan, _scanner=_clean_stub)


def test_transcripts_caution_ack_allows_run_and_is_recorded(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(plan, _scanner=_clean_stub)
    receipt = json.loads(plan.config.receipt_path.read_text(encoding="utf-8"))
    assert receipt["license_gating"]["caution_families"] == ["transcripts"]
    assert receipt["license_gating"]["acknowledged_caution_families"] == ["transcripts"]


def test_staging_produced_byte_scan_records_metadata_and_artifacts(
    tmp_path: Path,
):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    calls: list[dict[str, Any]] = []

    def staging_only_scanner(paths: list[Path]) -> list[dict]:
        kind, root = classify_export_scan_call(plan, paths)
        dirty = kind == "staging"
        if root is not None:
            assert_required_produced_scan_paths(plan, root, paths)
        calls.append({"kind": kind, "root": root, "dirty": dirty})
        if dirty:
            return [{"RuleID": "staging-produced-byte", "File": "generated"}]
        return []

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=staging_only_scanner)

    assert calls[0] == {"kind": "source", "root": None, "dirty": False}
    assert [call["kind"] for call in calls] == ["source", "staging"]
    assert sum(call["dirty"] for call in calls) == 1
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "staging_verification"
    )
    assert not plan.bundle_root.exists()
    assert_publication_debris_absent(plan)


@pytest.mark.parametrize("rebuild", [False, True])
def test_final_produced_byte_scan_passes_staging_then_rolls_back(
    tmp_path: Path,
    rebuild: bool,
):
    module = load_export_module()
    if rebuild:
        plan, prior_digest, prior_bytes = make_changed_rebuild_plan(
            module,
            tmp_path,
        )
    else:
        conference_dir, manifest_path = make_workspace(tmp_path)
        plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
        prior_digest = None
        prior_bytes = None
    calls: list[dict[str, Any]] = []
    staging_seen = False

    def final_only_scanner(paths: list[Path]) -> list[dict]:
        nonlocal staging_seen
        kind, root = classify_export_scan_call(plan, paths)
        if root is not None:
            relative_paths = assert_required_produced_scan_paths(plan, root, paths)
        else:
            relative_paths = set()
        if kind == "staging":
            staging_seen = True
        dirty = kind == "final" and staging_seen
        calls.append(
            {
                "kind": kind,
                "root": root,
                "dirty": dirty,
                "relative_paths": relative_paths,
            }
        )
        if dirty:
            return [{"RuleID": "final-produced-byte", "File": "generated"}]
        return []

    with pytest.raises(module.PublicExportBlocked) as exc_info:
        module.execute_export_plan(plan, _scanner=final_only_scanner)

    assert any(call["kind"] == "source" for call in calls)
    staging_calls = [call for call in calls if call["kind"] == "staging"]
    assert len(staging_calls) == 1
    assert staging_calls[0]["dirty"] is False
    assert calls[-1]["kind"] == "final"
    assert calls[-1]["root"] == plan.bundle_root
    assert calls[-1]["dirty"] is True
    assert exc_info.value.receipt["publication"]["failure_point"] == (
        "final_verification"
    )
    assert exc_info.value.receipt["publication"]["rollback_performed"] is True

    if rebuild:
        final_calls = [call for call in calls if call["kind"] == "final"]
        assert len(final_calls) == 2
        assert final_calls[0]["dirty"] is False
        restored_bytes = {
            path.relative_to(plan.bundle_root).as_posix(): path.read_bytes()
            for path in module._portable_bundle_files(plan.bundle_root)
        }
        assert module.bundle_tree_sha256(plan.bundle_root) == prior_digest
        assert restored_bytes == prior_bytes
    else:
        assert not plan.bundle_root.exists()
    assert_publication_debris_absent(plan)


def test_blocked_write_leaves_no_partial_bundle(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    with pytest.raises(module.PublicExportBlocked):
        module.execute_export_plan(plan, _scanner=_dirty_stub)
    assert not plan.bundle_root.exists()
    assert not any(plan.config.output_dir.glob(f".{plan.bundle_id}.staging-*"))


def test_neurips_dry_run_proof_reports_expected_fields(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = module.build_export_plan(make_config(module, conference_dir, manifest_path))
    module.dry_run_export(plan, _scanner=_clean_stub)
    receipt = json.loads(plan.config.receipt_path.read_text(encoding="utf-8"))
    assert receipt["planned_artifacts"]
    assert receipt["redacted_files"] == []
    assert receipt["excluded_artifacts"] == []
    assert receipt["cdn_stubs"]
    assert receipt["dry_run"] is True


def test_neurips_write_proof_emits_release_checksums_and_receipt(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    receipt_path = conference_dir / "storage" / "workflow-runs" / "write-proof.json"
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path, receipt_path=receipt_path)
    bundle = module.execute_export_plan(plan, _scanner=_clean_stub)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (bundle / "RELEASE.json").exists()
    assert (bundle / "CHECKSUMS.json").exists()
    assert receipt["source_sha"] == "abcdef1234567890"
    assert receipt["export_manifest_sha256"] == module.sha256_file(manifest_path)
    assert receipt["exit_code"] == 0


def test_f9_release_json_carries_each_family_license_and_the_check_holds_it(tmp_path: Path):
    module = load_export_module()
    conference_dir, manifest_path = make_workspace(tmp_path)
    plan = make_reviewed_write_plan(module, conference_dir, manifest_path)
    staging = materialize_staging(module, plan, tmp_path / "license-staging")

    release = json.loads((staging / "RELEASE.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert release["family_licenses"] == {family["id"]: family["license"] for family in manifest["families"]}
    module.verify_materialized_bundle(plan, staging, _scanner=_clean_stub)

    release["family_licenses"] = {family: "MIT" for family in release["family_licenses"]}
    _write_json(staging / "RELEASE.json", release)
    module.write_checksums(staging)
    with pytest.raises(module.PublicExportBlocked, match="family_licenses"):
        module.verify_materialized_bundle(plan, staging, _scanner=_clean_stub)
