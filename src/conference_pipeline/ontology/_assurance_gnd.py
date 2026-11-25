"""Runs inside GND's own interpreter (``gnd python -I``); never imported by the host.

Scores the checking view and the extraction view with GND 0.4.0's packaged
``assurance_report.score`` under the floors it is given, then merges the two into one
report: semantic support from the extraction view (the fidelity route reads the original
extraction), every other stage from the checking view, and freshness over both views'
evidence bindings through GND's own ``_stage``. Prints one JSON object.
"""

import json
import sys
from pathlib import Path

from gnd.examples.literature import assurance_report as ar

FROM_EXTRACTION = {"semantic-support"}
_RUN_ROUTE = ar.run_route


ANCHOR_MODEL = {}


def _unwrapped_run_route(gnd, route, paths, shown):
    """GND 0.4.0 wraps ``check --format json`` in an envelope (``reference_binding`` +
    ``report``) when the model binds reference populations; the packaged scorer reads the
    unwrapped report. Unwrap it, keeping the binding (Phase 3 finding)."""
    if route == "R2" and ANCHOR_MODEL.get("path"):
        # GND 0.4.0 refuses --anchors on a model with reference populations ("reference
        # sidecar combinations are unsupported"); anchors are checked against the same root
        # without them. The sidecar binds the vocabulary digest, which excludes them.
        paths = {**paths, "model": ANCHOR_MODEL["path"]}
    result = _RUN_ROUTE(gnd, route, paths, shown)
    payload = result.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("report"), dict) and "reference_binding" in payload:
        result = dict(result)
        result["payload"] = {**payload["report"], "reference_binding": payload["reference_binding"]}
    return result


ar.run_route = _unwrapped_run_route


def _stale_units(stage):
    return set((stage.get("disagreements") or {}).get("units") or [])


def _freshness(checking, extraction, sources):
    """Fresh = bound and current in the view that owns the binding (spec 6.7)."""
    owned = [("checking", f"{s}:anchors", s, "anchors") for s in sources] \
        + [("extraction", f"{s}:extraction", s, "extraction") for s in sources] \
        + [("checking", "composition", "run", "composition")]
    present, fresh, stale, refused = 0, 0, [], 0
    for view, unit, source, kind in owned:
        stage = checking if view == "checking" else extraction
        units = _stale_units(stage)
        absent = f"{unit}:absent" in units
        if absent:
            stale.append(f"{view}:{unit}:absent")
            continue
        present += 1
        if f"{unit}:refused" in units:
            refused += 1
            stale.append(f"{view}:{unit}:refused")
        elif unit in units or any(u.startswith(f"{unit}:") for u in units) or (
                kind == "composition" and "run:composition" in units):
            stale.append(f"{view}:{unit}")
        else:
            fresh += 1
    return ar._stage("freshness", metrics={"precision": (fresh, present), "recall": (fresh, len(owned))},
                     floors=ar.FLOORS["freshness"], route_present=present > 0, refused=refused,
                     sample_size=len(owned), disagreements={"count": len(stale), "units": stale},
                     routes=(checking.get("routes") or []) + (extraction.get("routes") or []))


def main(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ANCHOR_MODEL["path"] = config.get("anchor_model")
    ar.FLOORS = {stage: {m: tuple(v) for m, v in metrics.items()} for stage, metrics in config["floors"].items()}
    reports = {}
    for view in ("checking", "extraction"):
        paths = config["views"][view]
        reports[view] = ar.score(Path(paths["corpus_root"]), Path(paths["answers"]), Path(paths["run"]))
    for view, report in reports.items():
        if report.get("exit_code") == 3:
            return {"schema": ar.SCHEMA, "exit_code": 3, "reason": f"{view}:{report.get('reason')}",
                    "stages": [], "failed_stages": [], "views": {v: r.get("exit_code") for v, r in reports.items()}}
    by_view = {v: {s["stage"]: s for s in r["stages"]} for v, r in reports.items()}
    stages = []
    for name in ar.STAGES:
        if name == "freshness":
            stage = _freshness(by_view["checking"][name], by_view["extraction"][name], config["sources"])
            stage["root"] = "both"
        elif name == "derivation" and not config.get("checking_has_derivations", True):
            stage = {"stage": name, "active": False, "owner": None, "status": "not-applicable",
                     "reason": "the checking root declares no derived fact type", "failed_floors": [],
                     "unit": ar.UNITS[name], "root": "checking"}
        else:
            view = "extraction" if name in FROM_EXTRACTION else "checking"
            stage = dict(by_view[view][name])
            stage["root"] = view
        stages.append(stage)
    failed = [s["stage"] for s in stages if s["active"] and s["status"] != "pass"]
    checking = reports["checking"]
    return {
        "schema": ar.SCHEMA, "exit_code": 1 if failed else 0, "reason": None,
        "inputs": {"checking": checking.get("inputs"), "extraction": reports["extraction"].get("inputs")},
        "labels": checking.get("labels"), "stages": stages, "failed_stages": failed,
        "not_measured": [{"stage": s["stage"], "owner": s["owner"]} for s in stages if s["status"] == "not-measured"],
        "unscored_labels": checking.get("unscored_labels"),
        "views": {v: {"exit_code": r["exit_code"], "failed_stages": r["failed_stages"]} for v, r in reports.items()},
    }


if __name__ == "__main__":
    try:
        result = main(sys.argv[1])
    except Exception as error:  # totality: the host always gets one JSON object
        result = {"schema": ar.SCHEMA, "exit_code": 3, "reason": f"adapter:internal:{type(error).__name__}",
                  "stages": [], "failed_stages": []}
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
