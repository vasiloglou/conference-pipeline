"""Build the ontology inputs of one workspace (v6.1 Phase 2, spec 6.8).

roster -> analysis citations -> splits -> leakage report -> catalog shards (each checked
by ``gnd validate``) -> optional competency-question file. Ignored outputs go to
``<workspace>/storage/ontology/<run-id>/`` (written once per run id, R12); the committed
``<workspace>/ontology/splits.json`` holds ids only and is never silently replaced.

Exit codes: 0 built; 1 a ``*_problems`` check, an unreadable analysis product, a failed
``gnd validate`` or a differing committed split; 3 an input refused; 11 the run id exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import catalog, leakage, questions, splits, work_families
from conference_pipeline.ontology.gnd_tool import CommandFamily, run_gnd
from conference_pipeline.ontology.work_families import OntologyInputError, canonical_bytes

EXIT_OK, EXIT_PROBLEM, EXIT_INPUT, EXIT_EXISTS = 0, 1, 3, 11
RUN_ROOT = Path("storage/ontology")
SPLITS_PATH = Path("ontology/splits.json")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--full-read-dir", type=Path)
    p.add_argument("--salt", required=True, help="fixed split salt, recorded in splits.json")
    p.add_argument("--dev-per-topic", type=int, default=30)
    p.add_argument("--test-per-topic", type=int, default=30)
    p.add_argument("--dev-per-workshop", type=int, default=2)
    p.add_argument("--test-per-workshop", type=int, default=2)
    p.add_argument("--stratum-rule", choices=splits.STRATUM_RULES, default="rarest")
    p.add_argument("--replace-splits", action="store_true", help="overwrite a differing committed splits.json")
    p.add_argument("--questions-dir", type=Path)
    p.add_argument("--question-edits", type=Path)
    p.add_argument("--questions-out", type=Path)
    p.add_argument("--skip-gnd", action="store_true", help="do not run gnd validate on the catalog shards")
    return p


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    workspace = args.workspace.resolve()
    out = workspace / RUN_ROOT / args.run_id
    if out.exists():
        return EXIT_EXISTS, {"error": f"run id exists: {out}"}
    summary: dict[str, Any] = {"run_id": args.run_id, "problems": []}
    try:
        roster = work_families.build_roster(workspace, run_id=args.run_id, full_read_dir=args.full_read_dir,
                                            workspace_label=args.workspace.as_posix())
        inventory_ids = work_families.inventory_artifact_ids(workspace)
    except OntologyInputError as exc:
        return EXIT_INPUT, {"error": str(exc)}
    summary["problems"] += work_families.roster_problems(roster, inventory_ids)

    rows = leakage.scan_products(workspace / "analysis", work_families.artifact_index(roster))
    cited = leakage.cited_set(rows)
    split_doc = splits.assign_splits(
        roster, cited, salt=args.salt, dev_per_topic_stratum=args.dev_per_topic,
        test_per_topic_stratum=args.test_per_topic, dev_per_other_stratum=args.dev_per_workshop,
        test_per_other_stratum=args.test_per_workshop, stratum_rule=args.stratum_rule)
    summary["problems"] += splits.split_problems(split_doc, roster)
    report = leakage.leakage_report(rows, split_doc)
    if report["counts"]["unreadable"]:
        summary["problems"].append(f"{report['counts']['unreadable']} analysis products are unreadable")
    if report["counts"]["contaminated"]:
        summary["problems"].append(f"{report['counts']['contaminated']} analysis products cite held-out works")

    try:
        index = catalog.build_catalog(roster, out / "catalog")
    except catalog.CatalogTooLargeError as exc:
        return EXIT_INPUT, {"error": str(exc)}
    checks = []
    for shard in index["shards"]:
        model = out / "catalog" / shard["model"]
        if args.skip_gnd:
            checks.append({"shard": shard["shard"], "gnd_validate": "not-run"})
            continue
        result = run_gnd(["validate", str(model)], family=CommandFamily.CHECK, cwd=model.parent)
        checks.append({"shard": shard["shard"], "gnd_validate": result.exit_code, "meaning": result.meaning})
        if result.exit_code != 0:
            summary["problems"].append(f"gnd validate {shard['shard']} exited {result.exit_code}")
    summary["catalog"] = {"shards": len(index["shards"]), "works": index["works"], "checks": checks}

    _write(out / "roster.json", roster)
    _write(out / "leakage.json", report)
    splits_path = workspace / SPLITS_PATH
    new_bytes = canonical_bytes(split_doc)
    if splits_path.exists() and splits_path.read_bytes() != new_bytes and not args.replace_splits:
        summary["problems"].append(f"{SPLITS_PATH} differs from this run's splits (pass --replace-splits)")
        _write(out / "splits.candidate.json", split_doc)
    else:
        _write(splits_path, split_doc)
    summary["splits_sha256"] = hashlib.sha256(new_bytes).hexdigest()

    if args.questions_dir is not None:
        try:
            qdoc = questions.build_questions(args.questions_dir, args.question_edits)
        except (OSError, ValueError) as exc:
            return EXIT_INPUT, {"error": f"questions: {exc}"}
        core = questions.core_ids_from_set(args.questions_dir / "core-set.md") \
            if (args.questions_dir / "core-set.md").exists() else []
        summary["problems"] += questions.question_problems(qdoc, core)
        if args.questions_out is not None:
            _write(args.questions_out, qdoc)
        summary["questions"] = qdoc["counts"]

    summary.update({"roster": roster["counts"], "splits": split_doc["counts"], "leakage": report["counts"]})
    _write(out / "summary.json", summary)
    return (EXIT_PROBLEM if summary["problems"] else EXIT_OK), summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code, summary = run(args)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
