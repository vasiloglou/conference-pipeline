"""The model-pilot command line (v6.1 Phase 5): sample, estimate, run, score, report.

``sample`` writes the committed dev and growth sample manifests; ``estimate`` prices one arm
for one candidate from the prompts it would send (no call); ``run`` runs one candidate over the
sample under a spend cap (``--producer live``) or from saved replies (``--producer saved``,
zero calls), stopping a Modal app in a ``finally`` when ``--modal-app`` is given; ``score``
scores a run; ``report`` puts the scores of every candidate in one Markdown table.

Exit codes: 0 done; 3 an input refused; 4 the spend cap halted the run (the works not reached
are listed in ``arm.json``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import pilot, pilot_score

EXIT_OK, EXIT_INPUT, EXIT_BUDGET = 0, 3, 4
SEED = "seed-v1.orm.yaml"
# Assumed output tokens per call for the estimate (spec §3.3); the run records the real counts.
ASSUMED_OUTPUT = {"broad": 3000, "targeted": 1500, "review": 1500, "growth": 8000}
MODAL_TOKENS_PER_S = 50.0  # 012/013: bf16 Qwen3.8-27B on one H100, one stream, thinking off
READING_MAX_TOKENS, GROWTH_MAX_TOKENS = 8000, 16000


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_text() -> str:
    return resources.files("conference_pipeline.ontology.vocabulary").joinpath(SEED).read_text(encoding="utf-8")


def fact_types() -> list[str]:
    import yaml

    return [f["id"] for f in yaml.safe_load(seed_text())["fact_types"]]


def call_options(arm: str, provider: str, model: str) -> dict[str, Any]:
    """The arm's fixed settings (spec §3.2): reading arms think as little as the provider allows."""
    options: dict[str, Any] = {"provider": provider, "model": model}
    if arm == "growth":
        return {**options, "max_tokens": GROWTH_MAX_TOKENS}
    options.update(max_tokens=READING_MAX_TOKENS, timeout=900.0)
    if provider in ("anthropic", "modal", "local", "openrouter"):
        options["thinking"] = False
    elif provider == "gemini":
        options["reasoning_effort"] = "minimal"
    elif provider == "openai":
        options["reasoning_effort"] = "low"
    return options


def _inputs(args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, Any]:
    if sample["kind"] == "dev":
        texts, golds, links = pilot.load_dev(args.dev_pair, sample)
        return {"texts": texts, "golds": golds, "links": links, "fact_types": fact_types()}
    from conference_pipeline.ontology import gold_select, growth

    roster, splits = _load(args.roster), _load(args.splits)
    commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip() or "unknown"
    return {"seed_raw": seed_text().encode("utf-8"), "seed_artifact": f"src/conference_pipeline/ontology/vocabulary/{SEED}",
            "seed_commit": commit, "blocked": growth.split_blocker(splits),
            "text_of": lambda key: gold_select.load_texts(args.workspace, roster, [key]).get(key, ""),
            "requested_at": args.requested_at, "strategy": "pilot"}


def _prompts(arm: str, inputs: dict[str, Any], sample: dict[str, Any]) -> list[str]:
    from conference_pipeline.ontology import growth, reading

    out = []
    for w in sample["works"]:
        work = w["work"]
        if arm in ("broad", "targeted"):
            text = inputs["texts"][work]
            out += [reading.extract_prompt(text, g, inputs["fact_types"], (), "assertions") for g in pilot.units(arm, text)]
        elif arm == "review":
            facts, by_id = pilot.gold_as_facts(inputs["golds"][work], inputs["texts"][work])
            shown = [{k: v for k, v in f.items() if not k.startswith("_")} for f in facts]
            out.append(reading.review_prompt(inputs["texts"][work], shown, by_id))
        else:
            from conference_pipeline.ontology.seed_package import unattended_seed

            root = unattended_seed(inputs["seed_raw"], label=pilot.SEED_LABEL, artifact=inputs["seed_artifact"],
                                   commit=inputs["seed_commit"])
            out.append(growth.growth_prompt(root, inputs["text_of"](work), work))
    return out


def cmd_sample(args: argparse.Namespace) -> int:
    if args.kind == "dev":
        manifest = pilot.select_sample(_load(args.dev_pair / "corpus.json"), salt=args.salt, n=args.n)
    else:
        from conference_pipeline.ontology import gold_select

        roster = _load(args.roster)
        manifest = pilot.growth_sample(_load(args.splits), roster,
                                       lambda k: gold_select.load_texts(args.workspace, roster, [k]).get(k, ""),
                                       salt=args.salt, n=args.n)
    _write(args.out, manifest)
    print(json.dumps({"works": len(manifest["works"]), "sha256": manifest["sha256"]}))
    return EXIT_OK


def cmd_estimate(args: argparse.Namespace) -> int:
    from conference_pipeline.costs.estimator import estimate_text_model_cost
    from conference_pipeline.ontology.producer import gpu_second_price

    sample = _load(args.sample)
    prompts = _prompts(args.arm, _inputs(args, sample), sample)
    in_tokens = sum(len(p.encode("utf-8")) // 4 for p in prompts)
    out_tokens = ASSUMED_OUTPUT[args.arm] * len(prompts)
    price = gpu_second_price(args.provider, args.model)
    usd = (price * (out_tokens / MODAL_TOKENS_PER_S) * pilot_score.GPU_SURCHARGE if price
           else float(estimate_text_model_cost(args.model, in_tokens, out_tokens, provider=args.provider)))
    print(json.dumps({"arm": args.arm, "provider": args.provider, "model": args.model, "calls": len(prompts),
                      "input_tokens": in_tokens, "assumed_output_tokens": out_tokens, "estimate_usd": round(usd, 4),
                      "basis": "GPU seconds at 50 tok/s x 1.05, no cold start" if price else "catalog tokens"}))
    return EXIT_OK


def _stopper(args: argparse.Namespace):
    if not args.modal_app:
        return None

    def stop() -> dict[str, Any]:
        done = subprocess.run([args.modal_bin, "app", "stop", args.modal_app, "--yes"], capture_output=True, text=True,
                              check=False)
        return {"app": args.modal_app, "exit": done.returncode, "stderr": done.stderr[-300:]}
    return stop


def cmd_run(args: argparse.Namespace) -> int:
    from conference_pipeline.ontology.producer import OntologyProducer, SavedProducer
    from conference_pipeline.provider_receipts import provider_receipt_context

    sample = _load(args.sample)
    inputs = _inputs(args, sample)
    role = pilot.ROLES[args.arm]
    if args.producer == "saved":
        producer: Any = SavedProducer(args.responses)
    else:
        producer = OntologyProducer(save_root=args.out / "responses", budget_cap_usd=args.cap,
                                    tasks={role: pilot.TASKS[args.arm]}, caller=f"ontology.pilot.{args.arm}",
                                    role_options={role: call_options(args.arm, args.provider, args.model)})
    with provider_receipt_context(workspace_root=args.workspace, run_id=args.run_id, stage_id="v6.1-phase-5-pilot",
                                  caller=f"ontology.pilot.{args.arm}", task_id=pilot.TASKS[args.arm]):
        summary = pilot.run_arm(args.arm, sample, args.out, producer=producer, inputs=inputs,
                                stop_app=_stopper(args))
    summary.update(provider=args.provider, model=args.model, cap_usd=args.cap, producer=args.producer,
                   options=call_options(args.arm, args.provider, args.model))
    _write(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "spent_usd", "calls", "works_not_reached")}))
    return EXIT_BUDGET if summary["status"] == "budget-limited" else EXIT_OK


def _tokens(run: Path) -> tuple[int, int]:
    ins = outs = 0
    for path in (run / "responses").rglob("*.json") if (run / "responses").is_dir() else []:
        if ".failed-" in path.name or ".superseded-" in path.name:
            continue
        saved = _load(path)
        ins += int(saved.get("input_tokens") or 0)
        outs += int(saved.get("output_tokens") or 0)
    return ins, outs


def cmd_score(args: argparse.Namespace) -> int:
    sample, arm = _load(args.sample), _load(args.run / "arm.json")
    summary = _load(args.run / "summary.json")
    if arm["summary"]["sample_sha256"] != sample.get("sha256") or sample.get("sha256") != pilot.digest(sample):
        print(json.dumps({"status": "refused", "reason": "run was not made from this sample"}))
        return EXIT_INPUT
    name = summary["arm"]
    out: dict[str, Any] = {"schema": pilot_score.SCORE_SCHEMA, "arm": name, "provider": summary.get("provider"),
                           "model": summary.get("model"), "status": summary["status"],
                           "works": len(summary["works_done"]), "sample_sha256": sample["sha256"]}
    if name in ("broad", "targeted", "review"):
        texts, golds, links = pilot.load_dev(args.dev_pair, sample)
    if name in ("broad", "targeted"):
        out["reading"] = pilot_score.score_reading(arm["results"], golds, links)
        run = pilot.seed_run(args.run / "verify", args.seed)
        verdicts = [pilot.verify_claims(run, w, texts[w], r["claims"], executing_model=str(summary.get("model")))
                    for w, r in sorted(arm["results"].items())]
        meanings: dict[str, int] = {}
        for v in verdicts:
            meanings[v["verify_meaning"]] = meanings.get(v["verify_meaning"], 0) + 1
        out["verify"] = {"sources": len(verdicts), "exit_0": sum(v["verify_exit"] == 0 for v in verdicts),
                         "meanings": meanings, "rows": sum(v["rows"] for v in verdicts),
                         "unanchored_rows": sum(v["unanchored"] for v in verdicts)}
    elif name == "review":
        out["review"] = pilot_score.score_review(arm["results"])
    else:
        out["growth"] = pilot_score.score_growth(list(arm["results"].values()))
    ins, outs = _tokens(args.run)
    out["cost"] = pilot_score.cost(summary["spent_usd"], str(summary.get("provider")), len(summary["works_done"]),
                                   summary["wall_seconds"], input_tokens=ins, output_tokens=outs)
    _write(args.out, out)
    print(json.dumps({"arm": name, "model": out["model"], **out["cost"]}))
    return EXIT_OK


def _pct(r: dict[str, int] | None) -> str:
    return "–" if not r or not r["denominator"] else f"{r['numerator']}/{r['denominator']} ({100 * r['numerator'] / r['denominator']:.0f} %)"


def cmd_report(args: argparse.Namespace) -> int:
    scores = sorted((_load(p) for p in args.scores), key=lambda s: (pilot.ARMS.index(s["arm"]), s["model"] or ""))
    lines = ["# Phase 5 pilot report", "", "Proposals for routing are the owner's to rule on (P4).", ""]
    for arm in pilot.ARMS:
        rows = [s for s in scores if s["arm"] == arm]
        if not rows:
            continue
        lines += [f"## {arm}", ""]
        if arm in ("broad", "targeted"):
            lines += ["| Model | Status | Recall | Precision | Polarity | Conditions | Verify exit 0 | USD/work | Works/h |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for s in rows:
                r = s["reading"]
                lines.append(f"| {s['provider']}/{s['model']} | {s['status']} | {_pct(r['recall'])} | {_pct(r['precision'])} | "
                             f"{_pct(r['element_agreement']['polarity'])} | {_pct(r['element_agreement']['conditions'])} | "
                             f"{s['verify']['exit_0']}/{s['verify']['sources']} | {s['cost']['usd_per_work']} | "
                             f"{s['cost']['works_per_hour']} |")
        elif arm == "review":
            lines += ["| Model | Status | Balanced accuracy | Supported kept | Decoys passed | Missing | USD/work |",
                      "|---|---|---|---|---|---|---|"]
            for s in rows:
                r = s["review"]
                lines.append(f"| {s['provider']}/{s['model']} | {s['status']} | {r['balanced_accuracy']} | "
                             f"{_pct(r['true_positive_rate'])} | {_pct(r['decoys_judged_supported'])} | "
                             f"{r['missing_verdicts']} | {s['cost']['usd_per_work']} |")
        else:
            lines += ["| Model | Status | Applied rounds | Rejected by stage | Added elements | USD/round |",
                      "|---|---|---|---|---|---|"]
            for s in rows:
                g = s["growth"]
                lines.append(f"| {s['provider']}/{s['model']} | {s['status']} | {g['applied']}/{g['rounds']} | "
                             f"{g['rejected_by_stage']} | {g['added_elements']} | {s['cost']['usd_per_work']} |")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"scores": len(scores), "out": str(args.out)}))
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pilot_run.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def data(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workspace", type=Path, required=True, help="the conference workspace")
        p.add_argument("--dev-pair", type=Path, help="the dev gold pair directory (reading and review arms)")
        p.add_argument("--roster", type=Path, help="work-family roster (growth)")
        p.add_argument("--splits", type=Path, help="splits.json (growth)")
        p.add_argument("--requested-at", default="2026-11-21T00:00:00+00:00")

    s = sub.add_parser("sample")
    data(s)
    s.add_argument("--kind", choices=("dev", "growth"), required=True)
    s.add_argument("--salt", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--out", type=Path, required=True)
    for name in ("estimate", "run"):
        p = sub.add_parser(name)
        data(p)
        p.add_argument("--sample", type=Path, required=True)
        p.add_argument("--arm", choices=pilot.ARMS, required=True)
        p.add_argument("--provider", required=True)
        p.add_argument("--model", required=True)
        if name == "run":
            p.add_argument("--cap", type=float, required=True, help="this run's share of the owner's cap (USD)")
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--run-id", required=True)
            p.add_argument("--producer", choices=("live", "saved"), default="live")
            p.add_argument("--responses", type=Path, help="saved replies (--producer saved)")
            p.add_argument("--modal-app", help="stop this Modal app when the run ends, on every path")
            p.add_argument("--modal-bin", default="modal")
    p = sub.add_parser("score")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dev-pair", type=Path)
    p.add_argument("--seed", type=Path, default=Path(str(resources.files("conference_pipeline.ontology.vocabulary")
                                                         .joinpath(SEED))))
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"sample": cmd_sample, "estimate": cmd_estimate, "run": cmd_run, "score": cmd_score,
                "report": cmd_report}[args.command](args)
    except pilot.PilotError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}))
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
