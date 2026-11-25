"""The gold lane's command line (v6.1 Phase 3): select, estimate, label, queue, build.

``select`` writes the committed selection manifest; ``estimate`` prices the labelling run
from measured prompt sizes before any call; ``label`` runs both labellers (``--producer
saved`` replays with zero calls, ``--producer live`` calls the governed tasks under a
``--budget-cap-usd`` that halts the run); ``queue`` writes the owner's adjudication queue;
``build`` turns labels and rulings into gold files and the two split pairs.

Exit codes: 0 done; 1 a ``*_problems`` check failed; 3 an input refused; 4 the spend cap
halted labelling (the works not reached are listed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gold, gold_adjudicate, gold_label, gold_links, gold_select

EXIT_OK, EXIT_PROBLEM, EXIT_INPUT, EXIT_BUDGET = 0, 1, 3, 4
# Output tokens assumed per call for the estimate (the run records the real counts).
ASSUMED_OUTPUT_TOKENS = {"extract": 2500, "check": 800}
CHARS_PER_TOKEN = 4


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _picks(selection: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(split, area, family) for every pick, in a fixed order."""
    return [(split, area, p["family"]) for split, areas in sorted(selection["splits"].items())
            for area, block in sorted(areas.items()) for p in block["picks"]]


def cmd_select(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    splits = _load(workspace / "ontology/splits.json")
    roster = _load(args.roster)
    questions = _load(workspace / "ontology/competency-questions.json")
    try:
        areas = gold_select.load_workshop_areas(workspace / "ontology/workshop-areas.json")
    except gold_select.SelectionInputError as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_INPUT
    pool = [f for name in ("dev", "sealed_test") for g in splits["splits"][name] for f in splits["groups"][g]]
    texts = gold_select.load_texts(workspace, roster, pool)
    selection = gold_select.select_gold(splits, roster, questions, texts, per_area=args.per_area, salt=args.salt,
                                        workshop_areas=areas)
    problems = gold_select.selection_problems(selection, splits)
    selection["feasibility"] = gold_select.feasibility(selection, questions)
    _write(workspace / "ontology/gold-selection.json", selection)
    print(json.dumps({"counts": selection["counts"], "feasibility": selection["feasibility"], "problems": problems}))
    return EXIT_PROBLEM if problems else EXIT_OK


def _texts(workspace: Path, roster_path: Path, selection: dict[str, Any]) -> dict[str, str]:
    return gold_select.load_texts(workspace, _load(roster_path), [f for _, _, f in _picks(selection)])


def cmd_estimate(args: argparse.Namespace) -> int:
    from conference_pipeline.costs.estimator import estimate_text_model_cost
    from conference_pipeline.model_governance.registry import load_registry

    workspace = args.workspace.resolve()
    selection = _load(workspace / "ontology/gold-selection.json")
    texts = _texts(workspace, args.roster, selection)
    registry = load_registry()
    totals: dict[str, dict[str, float]] = {}
    for _, _, family in _picks(selection):
        text = texts.get(family, "")
        items = gold_label.passages(text)
        extract = len(gold_label.extract_prompt(text, items)) // CHARS_PER_TOKEN
        check = extract + 1200  # the other labeller's claims ride along
        for role in gold_label.ROLES:
            row = totals.setdefault(role, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            row["input_tokens"] += extract + check
            row["output_tokens"] += ASSUMED_OUTPUT_TOKENS["extract"] + ASSUMED_OUTPUT_TOKENS["check"]
            row["calls"] += 2
    usd = 0.0
    for role, row in totals.items():
        entry = registry.get(gold_label.TASKS[role])
        row["provider"], row["model"] = entry.provider, entry.model
        row["usd"] = round(estimate_text_model_cost(entry.model, int(row["input_tokens"]), int(row["output_tokens"]),
                                                    provider=entry.provider), 2)
        usd += row["usd"]
    print(json.dumps({"works": len(_picks(selection)), "by_role": totals, "usd": round(usd, 2),
                      "assumed_output_tokens": ASSUMED_OUTPUT_TOKENS}, indent=1, sort_keys=True))
    return EXIT_OK


def cmd_label(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    selection = _load(workspace / "ontology/gold-selection.json")
    texts = _texts(workspace, args.roster, selection)
    if args.producer == "live":
        if args.budget_cap_usd is None:
            print(json.dumps({"error": "a live run needs --budget-cap-usd (R4)"}))
            return EXIT_INPUT
        producer: Any = gold_label.LiveProducer(save_root=args.responses, budget_cap_usd=args.budget_cap_usd)
    else:
        producer = gold_label.SavedProducer(args.responses)
    done, not_reached = [], []
    for split, area, family in _picks(selection):
        target = args.out / split / f"{gold._doc_id(family)}.labels.json"
        if target.exists():
            done.append(family)
            continue
        try:
            labels = gold_label.label_work(family, texts.get(family, ""), producer)
        except gold_label.BudgetExceededError:
            not_reached.append(family)
            continue
        _write(target, {**labels, "split": split, "area": area})
        done.append(family)
    link_summary = {}
    if not not_reached:
        all_labels = _all_labels(args.out)
        for split in sorted({lab["split"] for lab in all_labels}):
            target = args.out / f"links-{split}.json"
            if target.exists():
                continue
            pairs = gold_links.candidates([lab for lab in all_labels if lab["split"] == split])
            try:
                result = gold_links.label_links(split, pairs, texts, producer)
            except gold_label.BudgetExceededError:
                not_reached.append(f"links:{split}")
                continue
            _write(target, result)
            link_summary[split] = len(pairs)
    summary = {"labelled": len(done), "not_reached": not_reached, "calls": producer.calls, "link_pairs": link_summary,
               "spent_usd": round(getattr(producer, "spent_usd", 0.0), 4)}
    _write(args.out / "label-summary.json", summary)
    print(json.dumps(summary))
    return EXIT_BUDGET if not_reached else EXIT_OK


def _all_labels(labels_dir: Path) -> list[dict[str, Any]]:
    return [_load(p) for p in sorted(labels_dir.rglob("*.labels.json"))]


def cmd_queue(args: argparse.Namespace) -> int:
    labels = _all_labels(args.labels)
    items = gold_adjudicate.queue(labels, salt=args.salt)
    for path in sorted(args.labels.glob("links-*.json")):
        items += gold_links.link_queue(_load(path), salt=args.salt)
    _write(args.out, {"schema": "ontology-adjudication-queue/1", "salt": args.salt, "items": items,
                      "agreement": gold_adjudicate.agreement_report(labels)})
    print(json.dumps({"queued": len(items), "works": len(labels)}))
    return EXIT_OK


def cmd_build(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    rulings = _load(args.rulings).get("rulings", {})
    problems = gold_adjudicate.ruling_problems(rulings)
    if problems:
        print(json.dumps({"problems": problems}))
        return EXIT_PROBLEM
    roster = _load(args.roster)
    selection = _load(workspace / "ontology/gold-selection.json")
    texts = _texts(workspace, args.roster, selection)
    artifacts = {f["family_key"]: f for f in roster["families"]}
    by_split: dict[str, list[tuple[dict[str, Any], bytes, str]]] = {"dev": [], "sealed_test": []}
    # r1 F2: every label's split is the split its work was selected for, and every pick is labelled.
    selected = {family: split for split, _, family in _picks(selection)}
    all_labels = _all_labels(args.labels)
    labelled = {lab["work"] for lab in all_labels}
    for lab in all_labels:
        if selected.get(lab["work"]) != lab.get("split"):
            problems.append(f"{lab['work']} is labelled for {lab.get('split')} but selected for "
                            f"{selected.get(lab['work'], 'no split')}")
    problems += [f"{family} is selected for {split} but has no labels" for family, split in sorted(selected.items())
                 if family not in labelled]
    if problems:
        print(json.dumps({"problems": problems[:50], "count": len(problems)}))
        return EXIT_PROBLEM
    for labels in all_labels:
        family = labels["work"]
        paper = next((a for a in artifacts[family]["artifacts"]), "")
        try:
            doc = gold_adjudicate.gold_for_work(labels, rulings, salt=args.salt, split=labels["split"],
                                                document={"artifact": paper})
        except gold_adjudicate.UnruledItemsError as exc:
            problems.append(str(exc))
            continue
        text = texts.get(family, "")
        problems += [f"{family}: {p}" for p in gold.gold_problems(doc, text)]
        by_split[labels["split"]].append((doc, text.encode("utf-8"), labels["area"]))
    if problems:
        print(json.dumps({"problems": problems[:50], "count": len(problems)}))
        return EXIT_PROBLEM
    digests = {}
    for split, golds in by_split.items():
        link_path = args.labels / f"links-{split}.json"
        if not link_path.is_file() and gold_links.candidates([lab for lab in all_labels if lab["split"] == split]):
            # r2 F3: the approved link lane must finish before a split's gold is built
            print(json.dumps({"problems": [f"{split}: the link lane has candidate pairs but no links-{split}.json; "
                                           "rerun label to finish it"]}))
            return EXIT_PROBLEM
        try:
            links = gold_links.build_links(_load(link_path), rulings, salt=args.salt) if link_path.is_file() else {
                "schema": gold.LINKS_SCHEMA, "split": split, "identity_pairs": [], "name_pairs": [], "chains": [],
                "reuse_cases": []}
        except gold_links.UnruledLinksError as exc:
            print(json.dumps({"problems": [str(exc)]}))
            return EXIT_PROBLEM
        works = {doc["work"]: split for doc, _, _ in golds}
        link_problems = gold.links_problems(links, works)
        if link_problems:
            print(json.dumps({"problems": link_problems[:50]}))
            return EXIT_PROBLEM
        digests[split] = gold.build_split_pair(args.out / split, split, golds, links)
    pair = gold.pair_problems(args.out / "dev", args.out / "sealed_test")
    _write(args.out / "digests.json", digests)
    print(json.dumps({"digests": digests, "pair_problems": pair}))
    return EXIT_PROBLEM if pair else EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "estimate", "label", "build"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", type=Path, required=True)
        p.add_argument("--roster", type=Path, required=True)
    sub.choices["select"].add_argument("--salt", required=True)
    sub.choices["select"].add_argument("--per-area", type=int, default=9)
    label = sub.choices["label"]
    label.add_argument("--out", type=Path, required=True)
    label.add_argument("--responses", type=Path, required=True)
    label.add_argument("--producer", choices=("saved", "live"), default="saved")
    label.add_argument("--budget-cap-usd", type=float)
    queue = sub.add_parser("queue")
    queue.add_argument("--labels", type=Path, required=True)
    queue.add_argument("--salt", required=True)
    queue.add_argument("--out", type=Path, required=True)
    build = sub.choices["build"]
    build.add_argument("--labels", type=Path, required=True)
    build.add_argument("--rulings", type=Path, required=True)
    build.add_argument("--salt", required=True)
    build.add_argument("--out", type=Path, required=True)
    return parser


COMMANDS = {"select": cmd_select, "estimate": cmd_estimate, "label": cmd_label, "queue": cmd_queue, "build": cmd_build}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
