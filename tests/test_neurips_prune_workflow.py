from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_prune_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "prune_conference_workspace.py"
    spec = importlib.util.spec_from_file_location("prune_conference_workspace", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_config(
    conference_dir: Path,
    *,
    manifest_path: str = "storage/storage_manifest.json",
    support_material_root: str = "storage",
) -> None:
    write_text(
        conference_dir / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            f"  manifest_path: {json.dumps(manifest_path)}\n"
            "  support_material:\n"
            f"    storage: {json.dumps(support_material_root)}\n"
        ),
    )


def make_workspace(
    tmp_path: Path,
    *,
    manifest_path: str = "storage/storage_manifest.json",
    support_material_root: str = "storage",
) -> Path:
    conference_dir = tmp_path / "conferences" / "demo2026"
    conference_dir.mkdir(parents=True)

    write_config(
        conference_dir,
        manifest_path=manifest_path,
        support_material_root=support_material_root,
    )
    write_text(conference_dir / "cache" / "vector-store" / "cache.bin", "cache-data")
    write_text(conference_dir / "exports" / "deep-dive" / "bundle.txt", "bundle-data")
    write_text(conference_dir / "audio" / "episode.txt", "podcast-data")
    write_text(conference_dir / "README.md", "demo")

    manifest = {
        "schema_version": 1,
        "workspace": "conferences/demo2026",
        "inventory_exemptions": [],
        "artifact_families": [
            {
                "id": "config_yaml",
                "path": "config.yaml",
                "storage_class": "git",
                "rebuildable": False,
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "vector_store",
                "path": "cache/vector-store",
                "storage_class": "cache",
                "rebuildable": True,
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "export_for_deep_dive",
                "path": "exports/deep-dive",
                "storage_class": "backup",
                "rebuildable": True,
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "podcasts",
                "path": "audio",
                "storage_class": "public",
                "rebuildable": True,
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "presentation_payloads",
                "path_glob": "decks/**/*.{html,png,svg,pdf}",
                "storage_class": "public",
                "rebuildable": True,
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "readme_md",
                "path": "README.md",
                "storage_class": "git",
                "rebuildable": False,
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    }
    inventory = {
        "schema_version": 1,
        "artifact_families": [
            {"path": "cache/vector-store", "size_bytes": 10},
            {"path": "exports/deep-dive", "size_bytes": 11},
            {"path": "audio", "size_bytes": 12},
        ],
    }
    write_text(
        conference_dir / manifest_path,
        json.dumps(manifest, indent=2) + "\n",
    )
    write_text(
        conference_dir / support_material_root / "inventory.json",
        json.dumps(inventory, indent=2) + "\n",
    )
    return conference_dir


def make_remote_copy(conference_dir: Path, relative_path: str, remote_root: Path) -> None:
    source = conference_dir / relative_path
    destination = remote_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for file_path in source.rglob("*"):
            if file_path.is_file():
                write_text(destination / file_path.relative_to(source), file_path.read_text(encoding="utf-8"))
    else:
        write_text(destination, source.read_text(encoding="utf-8"))


def test_build_candidates_uses_manifest_rebuildable_and_storage_class(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)

    candidates = module.build_candidates(conference_dir)
    candidate_ids = {candidate.family_id for candidate in candidates}

    assert candidate_ids == {"vector_store", "export_for_deep_dive", "podcasts"}
    tiers = {candidate.family_id: candidate.tier for candidate in candidates}
    assert tiers["vector_store"] == "cache"
    assert tiers["export_for_deep_dive"] == "remote-verified"
    assert tiers["podcasts"] == "remote-verified"


def test_build_candidates_uses_custom_manifest_path_from_config(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path, manifest_path="custom/manifest.json")

    candidates = module.build_candidates(conference_dir)

    assert {candidate.family_id for candidate in candidates} == {
        "vector_store",
        "export_for_deep_dive",
        "podcasts",
    }
    assert module.resolve_manifest_path(conference_dir) == (
        conference_dir / "custom" / "manifest.json"
    ).resolve()


def test_build_candidates_uses_custom_support_material_root_for_inventory(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(
        tmp_path,
        manifest_path="contracts/manifest.json",
        support_material_root="support-material",
    )

    candidates = {
        candidate.family_id: candidate
        for candidate in module.build_candidates(conference_dir)
    }

    assert candidates["vector_store"].size_bytes == 10
    assert candidates["export_for_deep_dive"].size_bytes == 11
    assert module.resolve_inventory_path(conference_dir) == (
        conference_dir / "support-material" / "inventory.json"
    ).resolve()
    assert module.resolve_verification_dir(conference_dir, None) == (
        conference_dir / "support-material" / "prune-verification"
    ).resolve()


def test_support_material_pointer_rejects_parent_traversal(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    write_config(conference_dir, support_material_root="../outside")

    with pytest.raises(
        ValueError,
        match=r"storage\.support_material\.storage must stay inside the conference workspace",
    ):
        module.resolve_inventory_path(conference_dir)


def test_verify_restore_writes_content_aware_evidence(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    verification_dir = conference_dir / "storage" / "prune-verification"
    remote_root = tmp_path / "backup-root"
    make_remote_copy(conference_dir, "exports/deep-dive", remote_root)

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "export_for_deep_dive"
    )

    evidence = module.verify_restore(
        candidate,
        remote_root=remote_root,
        profile="backup",
        verification_dir=verification_dir,
        sample_size=2,
    )

    assert evidence["status"] == "passed"
    assert evidence["comparison_method"] == "sha256 + file-size + file-count"
    assert evidence["remote_contains_local_floor"] is True
    assert (verification_dir / "backup" / "export_for_deep_dive.json").exists()
    assert evidence["sample_checks"][0]["matches"] is True


def test_verify_restore_uses_support_material_root_for_default_evidence_dir(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(
        tmp_path,
        manifest_path="ops/manifest.json",
        support_material_root="ops-support",
    )
    remote_root = tmp_path / "backup-root"
    make_remote_copy(conference_dir, "exports/deep-dive", remote_root)

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "export_for_deep_dive"
    )
    verification_dir = module.resolve_verification_dir(conference_dir, None)

    evidence = module.verify_restore(
        candidate,
        remote_root=remote_root,
        profile="backup",
        verification_dir=verification_dir,
        sample_size=2,
    )

    assert evidence["status"] == "passed"
    assert verification_dir == (conference_dir / "ops-support" / "prune-verification").resolve()
    assert (
        verification_dir / "backup" / "export_for_deep_dive.json"
    ).exists()


def test_verify_restore_allows_remote_superset_when_sample_hashes_match(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    verification_dir = conference_dir / "storage" / "prune-verification"
    remote_root = tmp_path / "public-root"
    make_remote_copy(conference_dir, "audio", remote_root)
    write_text(remote_root / "audio" / "extra.txt", "remote-extra")

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "podcasts"
    )

    evidence = module.verify_restore(
        candidate,
        remote_root=remote_root,
        profile="public",
        verification_dir=verification_dir,
        sample_size=2,
    )

    assert evidence["status"] == "passed"
    assert evidence["remote_file_count"] > evidence["local_file_count"]
    assert evidence["remote_contains_local_floor"] is True


def test_verify_restore_cli_refuses_mixed_selection_without_tier_narrowing(tmp_path: Path, capsys):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    remote_root = tmp_path / "backup-root"
    make_remote_copy(conference_dir, "exports/deep-dive", remote_root)
    make_remote_copy(conference_dir, "audio", remote_root)

    exit_code = module.main(
        [
            "verify-restore",
            "--conference-dir",
            str(conference_dir),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Refusing verify-restore for ineligible families:" in captured.err
    assert "vector_store" in captured.err
    assert "--tier remote-verified" in captured.err
    assert not (conference_dir / "storage" / "prune-verification" / "backup").exists()


def test_verify_restore_cli_refuses_wrong_profile_selection(tmp_path: Path, capsys):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    remote_root = tmp_path / "public-root"
    remote_root.mkdir(parents=True, exist_ok=True)

    exit_code = module.main(
        [
            "verify-restore",
            "--conference-dir",
            str(conference_dir),
            "--profile",
            "public",
            "--family",
            "export_for_deep_dive",
            "--remote-path",
            str(remote_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Refusing verify-restore for ineligible families:" in captured.err
    assert "export_for_deep_dive" in captured.err
    assert "--tier remote-verified" in captured.err
    assert not (conference_dir / "storage" / "prune-verification" / "public").exists()


def test_verify_restore_cli_succeeds_for_eligible_remote_verified_family(tmp_path: Path, capsys):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    remote_root = tmp_path / "backup-root"
    make_remote_copy(conference_dir, "exports/deep-dive", remote_root)

    exit_code = module.main(
        [
            "verify-restore",
            "--conference-dir",
            str(conference_dir),
            "--profile",
            "backup",
            "--tier",
            "remote-verified",
            "--family",
            "export_for_deep_dive",
            "--remote-path",
            str(remote_root),
            "--sample-size",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload == [
        {
            "family_id": "export_for_deep_dive",
            "path": "exports/deep-dive",
            "profile": "backup",
            "verified_at": payload[0]["verified_at"],
            "status": "passed",
            "comparison_method": "sha256 + file-size + file-count",
            "local_path": payload[0]["local_path"],
            "remote_path": payload[0]["remote_path"],
            "local_file_count": 1,
            "remote_file_count": 1,
            "local_size_bytes": 11,
            "remote_size_bytes": 11,
            "remote_contains_local_floor": True,
            "sample_checks": payload[0]["sample_checks"],
        }
    ]
    evidence_path = conference_dir / "storage" / "prune-verification" / "backup" / "export_for_deep_dive.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["family_id"] == "export_for_deep_dive"


def test_execute_quarantine_refuses_remote_verified_family_without_verify_restore(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    quarantine_dir = conference_dir / ".quarantine"
    verification_dir = conference_dir / "storage" / "prune-verification"

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "export_for_deep_dive"
    )

    with pytest.raises(RuntimeError, match="Refusing prune without recoverability verification:"):
        module.execute_quarantine(
            conference_dir=conference_dir,
            quarantine_dir=quarantine_dir,
            candidates=[candidate],
            verification_dir=verification_dir,
            cooldown_days=7,
        )


def test_execute_quarantine_moves_cache_family_and_writes_receipt_json(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    quarantine_dir = conference_dir / ".quarantine"
    verification_dir = conference_dir / "storage" / "prune-verification"

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "vector_store"
    )

    receipt_path = module.execute_quarantine(
        conference_dir=conference_dir,
        quarantine_dir=quarantine_dir,
        candidates=[candidate],
        verification_dir=verification_dir,
        cooldown_days=3,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_path.name == "receipt.json"
    assert receipt["cooldown_days"] == 3
    assert not (conference_dir / "cache" / "vector-store").exists()
    assert Path(receipt["items"][0]["quarantine_path"]).exists()


def test_purge_requires_age_and_keeps_quarantine_separate_from_execute_quarantine(tmp_path: Path):
    module = load_prune_module()
    conference_dir = make_workspace(tmp_path)
    quarantine_dir = conference_dir / ".quarantine"
    verification_dir = conference_dir / "storage" / "prune-verification"

    candidate = next(
        candidate for candidate in module.build_candidates(conference_dir)
        if candidate.family_id == "vector_store"
    )
    receipt_path = module.execute_quarantine(
        conference_dir=conference_dir,
        quarantine_dir=quarantine_dir,
        candidates=[candidate],
        verification_dir=verification_dir,
        cooldown_days=0,
    )

    early = module.purge_quarantine(
        conference_dir=conference_dir,
        quarantine_dir=quarantine_dir,
        verification_dir=verification_dir,
        batch_ids=[],
        min_age_hours=999,
    )
    assert early == []
    assert receipt_path.exists()

    purged = module.purge_quarantine(
        conference_dir=conference_dir,
        quarantine_dir=quarantine_dir,
        verification_dir=verification_dir,
        batch_ids=[],
        min_age_hours=0,
    )
    assert len(purged) == 1
    assert not receipt_path.parent.exists()
