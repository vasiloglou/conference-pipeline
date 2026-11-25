#!/usr/bin/env python3
"""Recoverability-first prune helper for conference workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_MANIFEST_PATH = Path("storage/storage_manifest.json")
DEFAULT_SUPPORT_MATERIAL_ROOT = Path("storage")
DEFAULT_INVENTORY_PATH = Path("storage/inventory.json")
DEFAULT_VERIFICATION_DIR = Path("storage/prune-verification")
DEFAULT_COOLDOWN_DAYS = 7
DEFAULT_MIN_AGE_HOURS = 24
DEFAULT_SAMPLE_SIZE = 3
RECEIPT_FILENAME = "receipt.json"
ERROR_RESTORE_PREFIX = "Refusing prune without recoverability verification:"
VERIFY_RESTORE_INELIGIBLE_PREFIX = "Refusing verify-restore for ineligible families:"
PROFILE_ENV_VARS = {
    "public": "GDRIVE_PUBLIC_PATH",
}


@dataclass(frozen=True)
class ManifestFamily:
    family_id: str
    relative_path: Path
    storage_class: str
    rebuildable: bool
    sync_profiles: dict[str, bool]


@dataclass(frozen=True)
class PruneCandidate:
    family_id: str
    relative_path: Path
    absolute_path: Path
    storage_class: str
    rebuildable: bool
    tier: str
    required_profiles: tuple[str, ...]
    size_bytes: int | None
    exists: bool


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_z(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read conference config: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Conference config is not a mapping: {path}")
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, verify, quarantine, and purge rebuildable conference workspace artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="List prune candidates from the manifest contract.")
    add_common_candidate_filters(report)
    report.add_argument("--conference-dir", type=Path, required=True)
    report.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")

    verify = subparsers.add_parser(
        "verify-restore",
        help="Verify mounted-drive restore integrity for one profile before quarantine.",
    )
    add_common_candidate_filters(verify)
    verify.add_argument("--conference-dir", type=Path, required=True)
    verify.add_argument("--profile", choices=("public", "backup"), required=True)
    verify.add_argument("--remote-path", type=Path, help="Mounted conference leaf to verify against.")
    verify.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    verify.add_argument("--verification-dir", type=Path)

    quarantine = subparsers.add_parser(
        "execute-quarantine",
        help="Move eligible families into quarantine after restore verification passes.",
    )
    add_common_candidate_filters(quarantine)
    quarantine.add_argument("--conference-dir", type=Path, required=True)
    quarantine.add_argument("--quarantine-dir", type=Path, required=True)
    quarantine.add_argument("--cooldown-days", type=int, default=DEFAULT_COOLDOWN_DAYS)
    quarantine.add_argument("--verification-dir", type=Path)

    purge = subparsers.add_parser(
        "purge",
        help="Permanently remove expired quarantine batches without touching live paths.",
    )
    purge.add_argument("--conference-dir", type=Path, required=True)
    purge.add_argument("--quarantine-dir", type=Path, required=True)
    purge.add_argument("--batch-id", action="append", default=[], help="Restrict purge to one batch id.")
    purge.add_argument("--min-age-hours", type=int, default=DEFAULT_MIN_AGE_HOURS)
    purge.add_argument("--verification-dir", type=Path)

    return parser.parse_args(argv)


def add_common_candidate_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tier",
        choices=("cache", "remote-verified"),
        help="Restrict to one prune tier.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Restrict to one or more manifest family ids.",
    )


def resolve_conference_dir(path: Path) -> Path:
    conference_dir = path.expanduser().resolve()
    if not conference_dir.exists():
        raise ValueError(f"Conference directory does not exist: {conference_dir}")
    return conference_dir


def resolve_config_path(conference_dir: Path) -> Path:
    config_path = (conference_dir / "config.yaml").resolve()
    if not config_path.is_file():
        raise ValueError(f"Could not read conference config: {config_path}")
    return config_path


def load_storage_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_file(config_path)
    storage_config = config.get("storage", {})
    if storage_config is None:
        storage_config = {}
    if not isinstance(storage_config, dict):
        raise ValueError(f"Invalid storage config in {config_path}")
    return storage_config


def safe_relative_path(raw: str, *, label: str) -> Path:
    candidate = Path(raw.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay inside the conference workspace")
    return candidate


def resolve_workspace_path(
    conference_dir: Path,
    relative_path: Path,
    *,
    label: str,
) -> Path:
    workspace_root = conference_dir.resolve()
    resolved = (conference_dir / relative_path).resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"{label} must stay inside the conference workspace")
    return resolved


def resolve_support_material_root(conference_dir: Path, config_path: Path | None = None) -> Path:
    resolved_config = config_path if config_path is not None else resolve_config_path(conference_dir)
    storage_config = load_storage_config(resolved_config)
    support_material = storage_config.get("support_material", {})
    if support_material is None:
        support_material = {}
    if not isinstance(support_material, dict):
        raise ValueError(f"Invalid support_material config in {resolved_config}")

    support_value = support_material.get("storage")
    if isinstance(support_value, str) and support_value.strip():
        support_root = safe_relative_path(
            support_value,
            label="storage.support_material.storage",
        )
    else:
        support_root = DEFAULT_SUPPORT_MATERIAL_ROOT
    return resolve_workspace_path(
        conference_dir,
        support_root,
        label="storage.support_material.storage",
    )


def resolve_manifest_path(conference_dir: Path, config_path: Path | None = None) -> Path:
    resolved_config = config_path if config_path is not None else resolve_config_path(conference_dir)
    storage_config = load_storage_config(resolved_config)

    manifest_value = storage_config.get("manifest_path")
    if isinstance(manifest_value, str) and manifest_value.strip():
        manifest_relative_path = safe_relative_path(
            manifest_value,
            label="storage.manifest_path",
        )
    else:
        manifest_relative_path = DEFAULT_MANIFEST_PATH

    manifest_path = resolve_workspace_path(
        conference_dir,
        manifest_relative_path,
        label="storage.manifest_path",
    )
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Manifest is not a file: {manifest_path}")
    return manifest_path


def resolve_inventory_path(conference_dir: Path, config_path: Path | None = None) -> Path:
    support_material_root = resolve_support_material_root(conference_dir, config_path)
    inventory_path = support_material_root / "inventory.json"
    if not inventory_path.exists():
        raise ValueError(f"Inventory not found: {inventory_path}")
    if not inventory_path.is_file():
        raise ValueError(f"Inventory is not a file: {inventory_path}")
    return inventory_path


def resolve_verification_dir(
    conference_dir: Path,
    override: Path | None,
    config_path: Path | None = None,
) -> Path:
    if override is not None:
        base = override
    else:
        support_material_root = resolve_support_material_root(conference_dir, config_path)
        base = support_material_root / "prune-verification"
    return base.expanduser().resolve()


def resolve_remote_path(profile: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()

    if profile == "backup":
        raise ValueError(
            f"{ERROR_RESTORE_PREFIX} missing explicit --remote-path for profile {profile}"
        )

    env_var = PROFILE_ENV_VARS.get(profile)
    if env_var is None:
        raise ValueError(f"{ERROR_RESTORE_PREFIX} missing remote path for profile {profile}")
    env_value = os.environ.get(env_var, "").strip()
    if not env_value:
        raise ValueError(
            f"{ERROR_RESTORE_PREFIX} missing remote path for profile {profile} "
            f"({env_var})"
        )
    return Path(env_value).expanduser().resolve()


def load_manifest_families(manifest_path: Path) -> list[ManifestFamily]:
    data = load_json(manifest_path)
    raw_families = data.get("artifact_families", [])
    if not isinstance(raw_families, list):
        raise ValueError(f"Malformed manifest at {manifest_path}")

    families: list[ManifestFamily] = []
    for raw_family in raw_families:
        if not isinstance(raw_family, dict):
            raise ValueError(f"Malformed manifest entry at {manifest_path}")
        raw_path = raw_family.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        sync_profiles = raw_family.get("sync_profiles", {})
        if not isinstance(sync_profiles, dict):
            raise ValueError(f"Malformed sync profile entry in {manifest_path}")
        families.append(
            ManifestFamily(
                family_id=str(raw_family["id"]),
                relative_path=Path(raw_path),
                storage_class=str(raw_family["storage_class"]),
                rebuildable=bool(raw_family["rebuildable"]),
                sync_profiles={
                    "public": bool(sync_profiles.get("public", False)),
                    "backup": bool(sync_profiles.get("backup", False)),
                },
            )
        )
    return families


def load_inventory_sizes(inventory_path: Path) -> dict[str, int]:
    inventory = load_json(inventory_path)
    sizes: dict[str, int] = {}
    for family in inventory.get("artifact_families", []):
        if isinstance(family, dict) and isinstance(family.get("path"), str):
            sizes[str(family["path"])] = int(family.get("size_bytes", 0))
    return sizes


def determine_tier(family: ManifestFamily) -> str:
    if family.storage_class == "cache":
        return "cache"
    return "remote-verified"


def required_profiles(family: ManifestFamily) -> tuple[str, ...]:
    if family.storage_class == "cache":
        return ()
    ordered = []
    if family.sync_profiles.get("backup"):
        ordered.append("backup")
    if family.sync_profiles.get("public"):
        ordered.append("public")
    return tuple(ordered)


def build_candidates(conference_dir: Path) -> list[PruneCandidate]:
    config_path = resolve_config_path(conference_dir)
    manifest_path = resolve_manifest_path(conference_dir, config_path)
    inventory_path = resolve_inventory_path(conference_dir, config_path)
    families = load_manifest_families(manifest_path)
    inventory_sizes = load_inventory_sizes(inventory_path)

    candidates: list[PruneCandidate] = []
    for family in families:
        if family.storage_class == "git" or not family.rebuildable:
            continue

        absolute_path = conference_dir / family.relative_path
        candidates.append(
            PruneCandidate(
                family_id=family.family_id,
                relative_path=family.relative_path,
                absolute_path=absolute_path,
                storage_class=family.storage_class,
                rebuildable=family.rebuildable,
                tier=determine_tier(family),
                required_profiles=required_profiles(family),
                size_bytes=inventory_sizes.get(family.relative_path.as_posix()),
                exists=absolute_path.exists(),
            )
        )
    return candidates


def filter_candidates(
    candidates: list[PruneCandidate],
    family_ids: Sequence[str],
    tier: str | None,
) -> list[PruneCandidate]:
    selected = candidates
    if family_ids:
        wanted = set(family_ids)
        selected = [candidate for candidate in selected if candidate.family_id in wanted]
    if tier is not None:
        selected = [candidate for candidate in selected if candidate.tier == tier]
    return selected


def select_verify_restore_candidates(
    candidates: Sequence[PruneCandidate],
    profile: str,
) -> list[PruneCandidate]:
    eligible: list[PruneCandidate] = []
    ineligible: list[PruneCandidate] = []

    for candidate in candidates:
        if candidate.tier == "remote-verified" and profile in candidate.required_profiles:
            eligible.append(candidate)
        else:
            ineligible.append(candidate)

    if ineligible:
        family_ids = ", ".join(candidate.family_id for candidate in ineligible)
        raise ValueError(
            f"{VERIFY_RESTORE_INELIGIBLE_PREFIX} {family_ids}; "
            "use --tier remote-verified or explicit --family filters."
        )
    return eligible


def describe_candidates(candidates: Iterable[PruneCandidate]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "family_id": candidate.family_id,
                "path": candidate.relative_path.as_posix(),
                "storage_class": candidate.storage_class,
                "tier": candidate.tier,
                "required_profiles": list(candidate.required_profiles),
                "size_bytes": candidate.size_bytes,
                "exists": candidate.exists,
            }
        )
    return rows


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(file_path for file_path in path.rglob("*") if file_path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_path(path: Path) -> dict[str, Any]:
    files = iter_files(path)
    file_count = len(files)
    size_bytes = sum(file_path.stat().st_size for file_path in files)
    return {
        "file_count": file_count,
        "size_bytes": size_bytes,
        "sample_files": [file_path for file_path in files[:DEFAULT_SAMPLE_SIZE]],
    }


def sample_relative_paths(path: Path, sample_size: int) -> list[str]:
    sample_files = iter_files(path)[:sample_size]
    if path.is_file():
        return [path.name]
    return [file_path.relative_to(path).as_posix() for file_path in sample_files]


def compare_sample_files(local_path: Path, remote_path: Path, sample_size: int) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for relative_name in sample_relative_paths(local_path, sample_size):
        local_file = local_path if local_path.is_file() else local_path / relative_name
        remote_file = remote_path if remote_path.is_file() else remote_path / relative_name
        matches = remote_file.exists()
        comparison = {
            "relative_path": relative_name,
            "local_size_bytes": local_file.stat().st_size if local_file.exists() else None,
            "remote_size_bytes": remote_file.stat().st_size if remote_file.exists() else None,
            "local_sha256": sha256_file(local_file) if local_file.exists() else None,
            "remote_sha256": sha256_file(remote_file) if remote_file.exists() else None,
        }
        comparison["matches"] = matches and (
            comparison["local_size_bytes"] == comparison["remote_size_bytes"]
            and comparison["local_sha256"] == comparison["remote_sha256"]
        )
        comparisons.append(comparison)
    return comparisons


def evidence_path(verification_dir: Path, profile: str, family_id: str) -> Path:
    return verification_dir / profile / f"{family_id}.json"


def verify_restore(
    candidate: PruneCandidate,
    remote_root: Path,
    profile: str,
    verification_dir: Path,
    sample_size: int,
) -> dict[str, Any]:
    local_path = candidate.absolute_path
    remote_path = remote_root / candidate.relative_path

    if not local_path.exists():
        raise ValueError(f"Local family path does not exist: {local_path}")
    if not remote_path.exists():
        raise RuntimeError(
            f"{ERROR_RESTORE_PREFIX} remote family path missing for {candidate.family_id}: {remote_path}"
        )

    local_summary = summarize_path(local_path)
    remote_summary = summarize_path(remote_path)
    sample_checks = compare_sample_files(local_path, remote_path, sample_size)
    remote_contains_local_floor = (
        remote_summary["file_count"] >= local_summary["file_count"]
        and remote_summary["size_bytes"] >= local_summary["size_bytes"]
    )

    status = remote_contains_local_floor and all(check["matches"] for check in sample_checks)

    evidence = {
        "family_id": candidate.family_id,
        "path": candidate.relative_path.as_posix(),
        "profile": profile,
        "verified_at": isoformat_z(utc_now()),
        "status": "passed" if status else "failed",
        "comparison_method": "sha256 + file-size + file-count",
        "local_path": str(local_path),
        "remote_path": str(remote_path),
        "local_file_count": local_summary["file_count"],
        "remote_file_count": remote_summary["file_count"],
        "local_size_bytes": local_summary["size_bytes"],
        "remote_size_bytes": remote_summary["size_bytes"],
        "remote_contains_local_floor": remote_contains_local_floor,
        "sample_checks": sample_checks,
    }

    target = evidence_path(verification_dir, profile, candidate.family_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def load_verification_evidence(
    verification_dir: Path,
    profile: str,
    family_id: str,
) -> dict[str, Any] | None:
    target = evidence_path(verification_dir, profile, family_id)
    if not target.exists():
        return None
    return load_json(target)


def ensure_recoverability(candidate: PruneCandidate, verification_dir: Path) -> list[str]:
    if candidate.tier == "cache":
        return []

    verified_profiles: list[str] = []
    for profile in candidate.required_profiles:
        evidence = load_verification_evidence(verification_dir, profile, candidate.family_id)
        if evidence and evidence.get("status") == "passed":
            verified_profiles.append(profile)

    if not verified_profiles:
        profiles = ", ".join(candidate.required_profiles) or "no eligible profiles"
        raise RuntimeError(
            f"{ERROR_RESTORE_PREFIX} {candidate.family_id} needs a passing verify-restore record for {profiles}."
        )
    return verified_profiles


def execute_quarantine(
    conference_dir: Path,
    quarantine_dir: Path,
    candidates: Sequence[PruneCandidate],
    verification_dir: Path,
    cooldown_days: int,
) -> Path:
    if not candidates:
        raise ValueError("No prune candidates selected for quarantine.")

    now = utc_now()
    batch_id = now.strftime("%Y%m%dT%H%M%SZ")
    batch_dir = quarantine_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    moved_items: list[dict[str, Any]] = []
    cooldown_expires_at = now + timedelta(days=cooldown_days)

    for candidate in candidates:
        if not candidate.absolute_path.exists():
            continue
        verified_profiles = ensure_recoverability(candidate, verification_dir)
        destination = batch_dir / candidate.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate.absolute_path), str(destination))
        moved_items.append(
            {
                "family_id": candidate.family_id,
                "tier": candidate.tier,
                "storage_class": candidate.storage_class,
                "source_path": str(candidate.absolute_path),
                "quarantine_path": str(destination),
                "verified_profiles": verified_profiles,
            }
        )

    receipt = {
        "batch_id": batch_id,
        "conference_dir": str(conference_dir),
        "created_at": isoformat_z(now),
        "cooldown_days": cooldown_days,
        "cooldown_expires_at": isoformat_z(cooldown_expires_at),
        "items": moved_items,
    }
    receipt_path = batch_dir / RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def purge_quarantine(
    conference_dir: Path,
    quarantine_dir: Path,
    verification_dir: Path,
    batch_ids: Sequence[str],
    min_age_hours: int,
) -> list[dict[str, Any]]:
    purge_records: list[dict[str, Any]] = []
    now = utc_now()
    age_floor = timedelta(hours=min_age_hours)

    for receipt_path in sorted(quarantine_dir.rglob(RECEIPT_FILENAME)):
        receipt = load_json(receipt_path)
        batch_id = str(receipt["batch_id"])
        if batch_ids and batch_id not in set(batch_ids):
            continue

        created_at = datetime.fromisoformat(str(receipt["created_at"]).replace("Z", "+00:00"))
        cooldown_expires_at = datetime.fromisoformat(
            str(receipt["cooldown_expires_at"]).replace("Z", "+00:00")
        )
        if now - created_at < age_floor or now < cooldown_expires_at:
            continue

        batch_dir = receipt_path.parent
        purge_record = {
            "batch_id": batch_id,
            "purged_at": isoformat_z(now),
            "batch_dir": str(batch_dir),
            "item_count": len(receipt.get("items", [])),
            "min_age_hours": min_age_hours,
        }
        purge_records.append(purge_record)

        purge_log = verification_dir / "purges" / f"{batch_id}.json"
        purge_log.parent.mkdir(parents=True, exist_ok=True)
        purge_log.write_text(json.dumps(purge_record, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(batch_dir)

    return purge_records


def render_report(candidates: Sequence[PruneCandidate]) -> str:
    lines = [
        "Prune candidates",
        "================",
    ]
    for candidate in candidates:
        size = str(candidate.size_bytes) if candidate.size_bytes is not None else "unknown"
        profiles = ", ".join(candidate.required_profiles) if candidate.required_profiles else "-"
        lines.append(
            f"- {candidate.family_id}: {candidate.relative_path} "
            f"[tier={candidate.tier}, class={candidate.storage_class}, profiles={profiles}, size_bytes={size}]"
        )
    if len(lines) == 2:
        lines.append("- none")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    conference_dir = resolve_conference_dir(args.conference_dir)
    config_path = resolve_config_path(conference_dir)
    verification_dir = resolve_verification_dir(
        conference_dir,
        getattr(args, "verification_dir", None),
        config_path,
    )
    candidates = build_candidates(conference_dir)
    selected = filter_candidates(candidates, getattr(args, "family", []), getattr(args, "tier", None))

    if args.command == "report":
        rows = describe_candidates(selected)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(render_report(selected))
        return 0

    if args.command == "verify-restore":
        try:
            selected = select_verify_restore_candidates(selected, args.profile)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        remote_root = resolve_remote_path(args.profile, args.remote_path)
        results = [
            verify_restore(
                candidate,
                remote_root=remote_root,
                profile=args.profile,
                verification_dir=verification_dir,
                sample_size=args.sample_size,
            )
            for candidate in selected
        ]
        failures = [result for result in results if result["status"] != "passed"]
        print(json.dumps(results, indent=2))
        return 1 if failures else 0

    if args.command == "execute-quarantine":
        receipt_path = execute_quarantine(
            conference_dir=conference_dir,
            quarantine_dir=args.quarantine_dir.expanduser().resolve(),
            candidates=selected,
            verification_dir=verification_dir,
            cooldown_days=args.cooldown_days,
        )
        print(f"Wrote quarantine receipt: {receipt_path}")
        return 0

    if args.command == "purge":
        purge_records = purge_quarantine(
            conference_dir=conference_dir,
            quarantine_dir=args.quarantine_dir.expanduser().resolve(),
            verification_dir=verification_dir,
            batch_ids=args.batch_id,
            min_age_hours=args.min_age_hours,
        )
        print(json.dumps(purge_records, indent=2))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
