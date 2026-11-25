#!/usr/bin/env python3
"""Create a versioned repair of a Presentation Blueprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.blueprint import (  # noqa: E402
    drop_unauthored_sections as drop_unauthored,
    require_approved_blueprint,
    validate_blueprint_payload,
)
from conference_pipeline.presentation.blueprint_markdown import blueprint_stats, render_blueprint_markdown  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402

# Wording a repair may change on one slide without touching what it cites.
SLIDE_PATCH_FIELDS = frozenset(
    {"narration_beat", "narration_beats", "visual_narration_hook", "visible_text", "claims", "title", "intent"}
)
REPAIR_METADATA_KEYS = frozenset(
    {"revision", "blueprint_id", "repair_notes", "repaired_at", "repair_of", "source_plan_repair_of", "provenance"}
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--review-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--repair-mode", required=True, choices=("full", "source-plan-only"))
    parser.add_argument("--repair-notes", required=True)
    parser.add_argument("--source-plan-patch", type=Path)
    parser.add_argument(
        "--slide-patch",
        type=Path,
        help=(
            "JSON object {slide_id: {field: value}} replacing wording fields on named slides "
            f"({', '.join(sorted(SLIDE_PATCH_FIELDS))}); source bindings go through --source-plan-patch."
        ),
    )
    parser.add_argument(
        "--blueprint-receipt",
        type=Path,
        help="Generation receipt of --blueprint; with --receipt-output, re-bound to the repaired blueprint.",
    )
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument(
        "--drop-unauthored-sections",
        action="store_true",
        help=(
            "remove sections that have no slides and clusters bound to no slide, recording each "
            "under dropped_sections. For a blueprint BLOCKED on cluster binding, so the input is "
            "not required to be approved; the repaired blueprint must validate instead."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.blueprint_receipt is None) != (args.receipt_output is None):
        print("--blueprint-receipt and --receipt-output must be given together", file=sys.stderr)
        return 2
    try:
        if not args.drop_unauthored_sections:
            require_approved_blueprint(args.blueprint, args.review_json, receipt_path=args.blueprint_receipt)
        repaired = repair_blueprint(
            args.blueprint,
            repair_mode=args.repair_mode,
            repair_notes=args.repair_notes,
            source_plan_patch=args.source_plan_patch,
            slide_patch=args.slide_patch,
            drop_unauthored_sections=args.drop_unauthored_sections,
        )
        if args.drop_unauthored_sections:
            errors = validate_blueprint_payload(repaired).errors
            if errors:
                raise ValueError(
                    "repaired blueprint still fails validation: "
                    + "; ".join(f"{error.code} at {error.path}" for error in errors[:5])
                )
    except Exception as exc:  # noqa: BLE001 - CLI should report validation errors clearly.
        print(str(exc), file=sys.stderr)
        return 2

    blueprint_path = args.blueprint.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve() if args.output_md else output_json.with_suffix(".md")
    original = json.loads(blueprint_path.read_text(encoding="utf-8"))
    parent_sha256 = _sha256_file(blueprint_path)
    archived: dict[str, str] = {}
    if output_json == blueprint_path:
        archived["blueprint"] = str(_archive_revision(blueprint_path, int(original.get("revision") or 1)))
    receipt_output: Path | None = None
    if args.blueprint_receipt is not None:
        receipt_path = args.blueprint_receipt.expanduser().resolve()
        receipt_output = args.receipt_output.expanduser().resolve()
        parent_receipt = receipt_path
        if receipt_output == receipt_path:
            parent_receipt = _archive_revision(receipt_path, int(original.get("revision") or 1))
            archived["receipt"] = str(parent_receipt)
    write_json_atomic(output_json, repaired)
    if receipt_output is not None:
        write_json_atomic(
            receipt_output,
            repair_receipt(
                parent_receipt,
                repaired=repaired,
                repaired_sha256=_sha256_file(output_json),
                parent_blueprint_sha256=parent_sha256,
                changed_paths=changed_paths(original, repaired),
            ),
        )
    _write_repair_markdown(output_md, repaired)
    print(
        json.dumps(
            {
                "status": "repaired",
                "blueprint_id": repaired["blueprint_id"],
                "revision": repaired["revision"],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "receipt_output": str(receipt_output) if receipt_output else None,
                "archived": archived,
                "changed_paths": changed_paths(original, repaired),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def repair_blueprint(
    blueprint_path: Path,
    *,
    repair_mode: str,
    repair_notes: str,
    source_plan_patch: Path | None = None,
    slide_patch: Path | None = None,
    drop_unauthored_sections: bool = False,
) -> dict[str, Any]:
    original = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    old_id = str(original["blueprint_id"])
    old_revision = int(original.get("revision") or 1)
    repaired = dict(original)
    repaired["revision"] = old_revision + 1
    repaired["blueprint_id"] = _new_blueprint_id(old_id, repaired["revision"], repair_mode, repair_notes)
    repaired["repair_notes"] = repair_notes
    repaired["repaired_at"] = _utc_now()
    if repair_mode == "full":
        repaired["repair_of"] = old_id
        repaired["source_plan_repair_of"] = None
    elif repair_mode == "source-plan-only":
        repaired["repair_of"] = None
        repaired["source_plan_repair_of"] = old_id
    else:
        raise ValueError(f"unknown repair mode: {repair_mode}")
    if source_plan_patch is not None:
        patch = json.loads(Path(source_plan_patch).read_text(encoding="utf-8"))
        if not isinstance(patch, dict):
            raise ValueError("--source-plan-patch must contain a JSON object")
        repaired.update(patch)
    if slide_patch is not None:
        repaired["slides"] = _patched_slides(repaired.get("slides") or [], json.loads(Path(slide_patch).read_text(encoding="utf-8")))
    if drop_unauthored_sections:
        repaired = json.loads(json.dumps(repaired))
        if not drop_unauthored(repaired, reasons=None):
            raise ValueError("--drop-unauthored-sections: every section has slides and every cluster is bound")
    provenance = dict(repaired.get("provenance") or {})
    provenance["repair"] = {
        "repair_of": old_id if repair_mode == "full" else None,
        "source_plan_repair_of": old_id if repair_mode == "source-plan-only" else None,
        "repair_mode": repair_mode,
        "repair_notes": repair_notes,
    }
    repaired["provenance"] = provenance
    return repaired


def _patched_slides(slides: list[Any], patch: Any) -> list[Any]:
    if not isinstance(patch, dict) or not patch:
        raise ValueError("--slide-patch must contain a non-empty JSON object keyed by slide_id")
    by_id = {slide.get("slide_id"): index for index, slide in enumerate(slides) if isinstance(slide, dict)}
    patched = list(slides)
    for slide_id, fields in patch.items():
        if slide_id not in by_id:
            raise ValueError(f"--slide-patch names unknown slide_id {slide_id!r}")
        if not isinstance(fields, dict) or not fields:
            raise ValueError(f"--slide-patch entry for {slide_id!r} must be a non-empty object")
        refused = sorted(set(fields) - SLIDE_PATCH_FIELDS)
        if refused:
            raise ValueError(f"--slide-patch may not change {refused} on {slide_id!r}; allowed: {sorted(SLIDE_PATCH_FIELDS)}")
        patched[by_id[slide_id]] = {**patched[by_id[slide_id]], **fields}
    return patched


def changed_paths(original: Any, repaired: Any, path: str = "$") -> list[str]:
    """JSON paths whose content differs, ignoring the repair's own bookkeeping."""

    if isinstance(original, dict) and isinstance(repaired, dict):
        found: list[str] = []
        for key in sorted(set(original) | set(repaired)):
            if path == "$" and key in REPAIR_METADATA_KEYS:
                continue
            found.extend(changed_paths(original.get(key), repaired.get(key), f"{path}.{key}"))
        return found
    if isinstance(original, list) and isinstance(repaired, list) and len(original) == len(repaired):
        return [p for i, (a, b) in enumerate(zip(original, repaired)) for p in changed_paths(a, b, f"{path}[{i}]")]
    return [] if original == repaired else [path]


def repair_receipt(
    parent_receipt: Path,
    *,
    repaired: dict[str, Any],
    repaired_sha256: str,
    parent_blueprint_sha256: str,
    changed_paths: list[str],
) -> dict[str, Any]:
    """Carry the parent's generation evidence forward, bound to the repaired blueprint.

    The repair itself makes no provider call; ``repair`` names the archived parent
    receipt so the validator can check the chain back to the generated blueprint.
    """

    parent = json.loads(parent_receipt.read_text(encoding="utf-8"))
    if str(parent.get("blueprint_sha256") or "") != parent_blueprint_sha256:
        raise ValueError("--blueprint-receipt is not bound to --blueprint")
    receipt = dict(parent)
    receipt["blueprint_sha256"] = repaired_sha256
    receipt["repair"] = {
        "changed_paths": changed_paths,
        "parent_blueprint_id": parent.get("blueprint_id") or repaired.get("repair_of") or repaired.get("source_plan_repair_of"),
        "parent_blueprint_sha256": parent_blueprint_sha256,
        "parent_receipt_path": str(parent_receipt),
        "parent_receipt_sha256": _sha256_file(parent_receipt),
        "provider_calls_made": False,
        "repair_mode": repaired["provenance"]["repair"]["repair_mode"],
        "repair_notes": repaired.get("repair_notes"),
        "repaired_at": repaired.get("repaired_at"),
        "revision": repaired.get("revision"),
    }
    return receipt


def _archive_revision(path: Path, revision: int) -> Path:
    target = path.with_name(f"{path.stem}.rev{revision}{path.suffix}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite archived revision {target}")
    target.write_bytes(path.read_bytes())
    return target


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_blueprint_id(old_id: str, revision: int, repair_mode: str, repair_notes: str) -> str:
    digest = hashlib.sha256(f"{old_id}:{revision}:{repair_mode}:{repair_notes}".encode("utf-8")).hexdigest()[:8]
    return f"{old_id}-rev{revision}-{digest}"


def _write_repair_markdown(path: Path, repaired: dict[str, Any]) -> None:
    lines = [
        "# Presentation Blueprint Repair",
        "",
        f"- Blueprint ID: `{repaired.get('blueprint_id')}`",
        f"- Revision: `{repaired.get('revision')}`",
        f"- Repair of: `{repaired.get('repair_of')}`",
        f"- Source-plan repair of: `{repaired.get('source_plan_repair_of')}`",
        f"- Notes: {repaired.get('repair_notes')}",
        "",
    ]
    review_view = render_blueprint_markdown(
        repaired,
        topic_title=str(repaired.get("audience_title") or repaired.get("presentation_id") or ""),
        stats=blueprint_stats(repaired),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" + review_view, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
