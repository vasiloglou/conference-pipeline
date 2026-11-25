#!/usr/bin/env python3
"""Record an operator's outcome for a figure the critic left open, and apply a patch.

The adversarial critic stops after three rounds; what is still open is the
operator's to decide (docs/operations/figure-fidelity-policy.md, item 9).
This is the sanctioned way to act on that decision without leaving the deck's
step ledger stale:

  accept   record that the figure stands, with the reason
  patch    apply named edits to the spec (validated and compiled here, no
           model call), then resume the runner so the figure is recompiled,
           re-screenshotted and re-judged
  replace  swap the whole spec for another the operator supplies (an honest
           archetype for the same claim), validated and compiled the same way;
           with --text-layout, the replacement is built from the deck contract
           slide (presentation_deck.json): a `key_points` figure carrying only
           the slide's title, first claim and visible text, verbatim

Every outcome is appended to <figure>.patch.json beside the spec, with the
operator, the reason, the ops and the before/after digests, and written into
the critic sidecar as ``operator_resolution`` so receipts and review pages
stop counting the figure as open.

Patch ops (JSON list)::

    [{"op": "set",    "path": "content.nodes[2].label", "value": "Observation"},
     {"op": "delete", "path": "content.cells[3]"},
     {"op": "append", "path": "content.edges", "value": {"from": "a", "to": "b"}}]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.figures.compiler import compile_figure  # noqa: E402
from conference_pipeline.figures.errors import FigureCompilerError  # noqa: E402
from conference_pipeline.figures.spec import load_spec, validate_spec  # noqa: E402
from conference_pipeline.figures.text_layout import key_points_spec  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402

OUTCOMES = ("accept", "patch", "replace")
_STEP_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _walk(root: Any, path: str) -> tuple[Any, str | int]:
    """The container and final key a dotted/indexed path names."""

    steps: list[str | int] = []
    for key, index in _STEP_RE.findall(path):
        steps.append(int(index) if index else key)
    if not steps:
        raise ValueError(f"empty path {path!r}")
    node = root
    for step in steps[:-1]:
        node = node[step]
    return node, steps[-1]


def apply_ops(spec: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    patched = copy.deepcopy(spec)
    for op in ops:
        kind = str(op.get("op") or "")
        container, key = _walk(patched, str(op.get("path") or ""))
        if kind == "set":
            container[key] = op.get("value")
        elif kind == "delete":
            del container[key]
        elif kind == "append":
            target = container[key]
            if not isinstance(target, list):
                raise ValueError(f"append target {op.get('path')!r} is not a list")
            target.append(op.get("value"))
        else:
            raise ValueError(f"unknown op {kind!r}; use set, delete or append")
    return patched


def check_spec(payload: dict[str, Any]) -> list[str]:
    """Validation errors that stop a patch; a budget-only overrun that draws passes."""

    try:
        spec = load_spec(payload, strict=False)
    except FigureCompilerError as error:
        return [str(error)]
    errors = validate_spec(spec, mode="compile")
    if errors:
        return errors
    try:
        compile_figure(payload)
    except FigureCompilerError as error:
        return [str(error)]
    return []


def resolve_figure(deck_dir: Path, slide_id: str) -> tuple[Path, str]:
    spec_dir = deck_dir / "figure_specs"
    matches = sorted(p for p in spec_dir.glob("*-figure.json") if p.name.startswith(slide_id))
    if len(matches) != 1:
        raise SystemExit(f"{slide_id!r} matches {len(matches)} specs under {spec_dir}: {[m.name for m in matches]}")
    return matches[0], matches[0].name[: -len(".json")]


DECK_CONTRACT = "presentation_deck.json"


def contract_slide(deck_dir: Path, slide_id: str) -> dict[str, Any]:
    """The deck contract's slide with exactly this id."""

    contract_path = deck_dir / DECK_CONTRACT
    if not contract_path.is_file():
        raise SystemExit(f"--text-layout needs the deck contract {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    matches = [slide for slide in contract.get("slides") or []
               if isinstance(slide, dict) and str(slide.get("slide_id") or "") == slide_id]
    if len(matches) != 1:
        raise SystemExit(f"{slide_id!r} matches {len(matches)} slides in {contract_path}")
    return matches[0]


def record(deck_dir: Path, figure_id: str, entry: dict[str, Any]) -> None:
    spec_dir = deck_dir / "figure_specs"
    log_path = spec_dir / f"{figure_id}.patch.json"
    history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    history.append(entry)
    write_json_atomic(log_path, history)
    critic_path = spec_dir / f"{figure_id}.critic.json"
    if critic_path.is_file():
        critique = json.loads(critic_path.read_text(encoding="utf-8"))
        critique["operator_resolution"] = {k: entry[k] for k in ("outcome", "by", "reason", "at")}
        write_json_atomic(critic_path, critique)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--slide-id", required=True, help="slide id or unique prefix")
    parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    parser.add_argument("--by", required=True, help="operator recorded on the decision")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--ops", type=Path, help="patch: JSON file with the op list")
    parser.add_argument("--spec", type=Path, help="replace: JSON file with the whole new spec")
    parser.add_argument("--text-layout", action="store_true",
                        help="replace: build the new spec from the slide's own text (key_points)")
    parser.add_argument("--dry-run", action="store_true", help="validate and compile; write nothing")
    args = parser.parse_args(argv)
    if args.text_layout and args.outcome != "replace":
        parser.error("--text-layout applies to --outcome replace only")

    spec_path, figure_id = resolve_figure(args.deck_dir, args.slide_id)
    before = json.loads(spec_path.read_text(encoding="utf-8"))
    entry: dict[str, Any] = {
        "at": _utc_now(), "by": args.by, "outcome": args.outcome, "reason": args.reason,
        "before_sha256": _sha256(before),
    }
    after = before
    if args.outcome == "patch":
        if args.ops is None:
            parser.error("--outcome patch needs --ops")
        ops = json.loads(args.ops.read_text(encoding="utf-8"))
        after = apply_ops(before, ops)
        entry["ops"] = ops
    elif args.outcome == "replace":
        if args.text_layout == (args.spec is not None):
            parser.error("--outcome replace needs exactly one of --spec or --text-layout")
        if args.text_layout:
            slide_id = str(before.get("slide_id") or "")
            try:
                after = key_points_spec(contract_slide(args.deck_dir, slide_id),
                                        figure_id=str(before.get("figure_id") or figure_id),
                                        section=str(before.get("section") or ""))
            except FigureCompilerError as error:
                print(f"refused; no text layout for {slide_id}: {error}", file=sys.stderr)
                return 1
            entry["text_layout"] = True
        else:
            after = json.loads(args.spec.read_text(encoding="utf-8"))
        # Identity is the slide's, never the replacement file's.
        for key in ("spec_version", "figure_id", "slide_id", "section"):
            if key in before:
                after[key] = before[key]
    if args.outcome != "accept":
        errors = check_spec(after)
        if errors:
            print("refused; the figure would not validate or draw:\n- " + "\n- ".join(errors[:6]), file=sys.stderr)
            return 1
        entry["after_sha256"] = _sha256(after)
    if args.dry_run:
        print(json.dumps({"figure_id": figure_id, "would_record": entry}, indent=2))
        return 0
    if args.outcome != "accept":
        write_json_atomic(spec_path, after)
    record(args.deck_dir, figure_id, entry)
    print(json.dumps({"figure_id": figure_id, "recorded": entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
