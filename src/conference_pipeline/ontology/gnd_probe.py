"""Offline capability probe for the pinned GND version (v6.1 Phase 1).

Runs every GND capability the v6.1 plan uses (plan §2.6) on GND's own packaged
examples, each with a positive and a negative row, and writes a
``gnd-capability-probe/1`` receipt. No model provider is called: the reading
examples use GND's saved producer responses, and credential-named variables are
scrubbed from every subprocess (``gnd_tool.scrubbed_env``).

Exits: 0 every row as expected and every capability covered both ways; 1 a row
off its expectation or a coverage hole; 3 GND or its source checkout unusable;
11 ``--out`` already exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily, GndResult, GndUnavailable

SCHEMA = "gnd-capability-probe/1"
GND_COMMIT = "648cc6e07fc6ea0e4d8791a2a33a0d005f4e106a"  # tag v0.4.0 (tag object a6ad5b53...)
ARCHIVE_PATHS = ("tools/", "tests/support/", "tests/test_li1_assurance_report.py", "src/gnd/examples/literature/")
LITERATURE_IN_ARCHIVE = "src/gnd/examples/literature"

# Capability ids of plan §2.6 (v6.1-ICML-2026-Ontology plan r5); each needs a + and a - row.
CAPABILITIES: tuple[str, ...] = (
    "shared-vocabulary", "roots-rebind", "contextual-assertions", "judgment-admission",
    "multipass-reading", "reading-coverage", "anchors", "identity-resolution", "name-diagnostics",
    "reference-catalog", "lineage", "lineage-dates", "semantic-revision", "support-verdicts",
    "witness-replay", "why-not", "assurance-report", "interchange", "discover", "rule-mining",
    "pyrel-export",
)

EXIT_OK, EXIT_ROW_FAILED, EXIT_REFUSED, EXIT_OUT_EXISTS = 0, 1, 3, 11
MODULE_MISSING = -1002  # observed_exit for a row whose example module is absent
KIND_GND, KIND_EXAMPLE, KIND_PYTHON = "gnd", "example", "python"


@dataclass
class Context:
    """Paths every row may use. ``real`` is False under a fake runner: no setup runs."""

    work: Path
    gnd_root: Path  # site-packages/gnd of the installed tool
    archive: Path
    real: bool = True
    reports: dict[str, Any] = field(default_factory=dict)

    @property
    def lit(self) -> Path:
        return self.gnd_root / "examples" / "literature"

    def p(self, *parts: str) -> str:
        return str((self.work / Path(*parts)).resolve())


Check = Callable[[GndResult, Context], str | None]  # None = fine, else the problem


@dataclass(frozen=True)
class Row:
    id: str
    capabilities: tuple[str, ...]
    polarity: str  # "+" or "-"
    family: CommandFamily
    expected_exit: int
    kind: str  # "gnd" | "example" | "python"
    args: Callable[[Context], list[str]]
    prepare: Callable[[Context], None] | None = None
    check: Check | None = None
    env: Callable[[Context], Mapping[str, str]] | None = None
    reason: str | None = None  # text GND must print (stdout or stderr) for this outcome (r1 F2)
    outputs: tuple[str, ...] = ()  # files under work/ the row must produce; digested (r1 F5)


# --- setup helpers (run only for real runs) -------------------------------------------


def _copy(ctx: Context, src: Path | str, dst: str) -> Path:
    target = ctx.work / dst
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    for root, dirs, files in os.walk(target):
        for name in dirs + files:
            path = Path(root) / name
            path.chmod(path.stat().st_mode | 0o200)
    return target


def _flip_first_string_char(path: Path) -> None:
    """Change one character inside the first JSON string value: a semantic edit, not whitespace."""
    data = bytearray(path.read_bytes())
    start = data.find(b'": "')
    start = start + 4 if start >= 0 else data.find(b'"', 10) + 1
    data[start] = ord("X") if data[start] != ord("X") else ord("Y")
    path.write_bytes(bytes(data))


def _json_out(result: GndResult) -> dict[str, Any] | None:
    return gnd_tool.parse_json_stdout(result)


def _expect_json(predicate: Callable[[dict[str, Any]], str | None]) -> Check:
    def check(result: GndResult, ctx: Context) -> str | None:
        payload = _json_out(result)
        if payload is None:
            return "stdout is not a JSON object"
        return predicate(payload)

    return check


def _remember(name: str, then: Check | None = None) -> Check:
    def check(result: GndResult, ctx: Context) -> str | None:
        ctx.reports[name] = _json_out(result)
        return then(result, ctx) if then else None

    return check


def _staffing(ctx: Context) -> None:
    _copy(ctx, ctx.gnd_root / "examples" / "staffing", "staffing")


def _quantity(ctx: Context, name: str) -> Path:
    return _copy(ctx, ctx.lit / "quantity_fixture", name)


def _prep_unwitnessed(ctx: Context) -> None:
    root = _quantity(ctx, "q-unwitnessed")
    pop = json.loads((root / "time.population.json").read_text())
    pop["facts"]["time_step"][0]["coefficient"] = 7
    raw = (json.dumps(pop, indent=2) + "\n").encode()
    (root / "time.population.json").write_bytes(raw)
    anchors = json.loads((root / "time.anchors.json").read_text())
    anchors["anchors"][0]["fact_key"]["roles"]["coefficient"] = 7
    anchors["population_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "time.anchors.json").write_text(json.dumps(anchors))


def _prep_stale_span(ctx: Context) -> None:
    root = _quantity(ctx, "q-stale")
    anchors = json.loads((root / "time.anchors.json").read_text())
    anchors["anchors"][0]["span"] = {"start": 24, "end": 28}  # the "2 fs" occurrence, excerpt digest unchanged
    (root / "time.anchors.json").write_text(json.dumps(anchors))


def _prep_names_limit(ctx: Context) -> None:
    cases = json.loads((ctx.lit / "diagnostics" / "cases.json").read_text())
    base = cases["occurrences"][0]
    cases["occurrences"] = [dict(copy.deepcopy(base), id=f"{base['id']}-{i}") for i in range(501)]
    (ctx.work / "names-501.json").write_text(json.dumps(cases))


def _prep_tamper_admitted(ctx: Context) -> None:
    root = _copy(ctx, ctx.work / "lineage-run", "lineage-run-tampered")
    target = sorted((root / "admissions").glob("*/admitted.population.json"))[0]
    _flip_first_string_char(target)


M7_SCRIPT = """
import json, pathlib, sys
import test_li1_assurance_report as t
out = pathlib.Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
run = t.build_case("M7-false-merge", out)
report = t._score(run)
try:
    t.assert_matches_expectation(report, "M7-false-merge")
    matches = True
except AssertionError:
    matches = False
print(json.dumps({"matches": matches, "exit_code": report["exit_code"], "failed_stages": report["failed_stages"]}))
sys.exit(report["exit_code"])
"""


def _has_call_counts(payload: dict[str, Any]) -> str | None:
    calls = payload.get("producer_calls")
    if not isinstance(calls, dict) or not calls or not all(isinstance(v, int) for v in calls.values()):
        return f"no producer_calls counts in the reading report: {calls!r}"
    return None


def _coverage_states(payload: dict[str, Any]) -> list[str]:
    return [row.get("state") for row in payload.get("coverage", [])]


def _dates_pinn(payload: dict[str, Any]) -> str | None:
    """The written advice names exactly one advisory, the PINN edge, for its stated reason."""
    advisories = [(e.get("id"), e.get("reason")) for e in payload.get("edges", []) if e.get("outcome") == "advisory"]
    expected = [("pinn", "parent-published-later")]
    return None if advisories == expected else f"advisories {advisories}, expected exactly {expected}"


def _dates_controls(payload: dict[str, Any]) -> str | None:
    passed = {e.get("id") for e in payload.get("edges", []) if e.get("outcome") == "pass"}
    return None if {"new-to-old", "same-year"} <= passed else f"control edges passing: {sorted(passed)}"


def _read_json_file(path: str) -> Callable[[GndResult, Context], str | None]:
    def check(result: GndResult, ctx: Context) -> str | None:
        try:
            payload = json.loads(Path(path.format(work=ctx.work)).read_text())
        except (OSError, ValueError) as exc:
            return f"cannot read {path}: {exc}"
        ctx.reports[path] = payload
        return None

    return check


def _rows() -> list[Row]:
    lit = lambda c, *p: str(c.lit.joinpath(*p))  # noqa: E731
    return [
        Row("depth", ("contextual-assertions",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.depth", "--out", c.p("depth-run")]),
        Row("reading-run", ("multipass-reading",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.reading", "run", "--out", c.p("r")], outputs=("r/run.json",),
            check=_remember("reading-run", _expect_json(_has_call_counts))),
        Row("reading-replay", ("multipass-reading",), "+", CommandFamily.REPLAY, 0, KIND_EXAMPLE,
            lambda c: ["literature.reading", "replay", c.p("r")], check=_remember(
                "reading-replay", _expect_json(lambda d: None if not any(d.get("producer_calls", {}).values())
                                               else f"replay made producer calls {d.get('producer_calls')}"))),
        Row("reading-replay-stale", ("multipass-reading",), "-", CommandFamily.REPLAY, 3, KIND_EXAMPLE,
            lambda c: ["literature.reading", "replay", c.p("r-tampered")],
            prepare=lambda c: _flip_first_string_char(_copy(c, c.work / "r", "r-tampered") / "payload-2.json")),
        Row("reading-budget-0", ("reading-coverage",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.reading", "run", "--out", c.p("r0"), "--budget-bytes", "0"],
            check=_expect_json(lambda d: None if d.get("stop") == "budget-exhausted" else f"stop={d.get('stop')}")),
        Row("reading-unavailable", ("reading-coverage",), "-", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.reading", "run", "--out", c.p("ru"), "--scenario", "unavailable"],
            check=_expect_json(lambda d: None if "unavailable" in _coverage_states(d)
                               else f"no unavailable disposition in {_coverage_states(d)}")),
        Row("interop-run", ("roots-rebind", "reference-catalog", "identity-resolution", "witness-replay"), "+",
            CommandFamily.OTHER, 0, KIND_EXAMPLE, lambda c: ["literature.interop", "run", "--out", c.p("i"), "--dates"]),
        Row("interop-verify", ("roots-rebind",), "+", CommandFamily.REPLAY, 0, KIND_EXAMPLE,
            lambda c: ["literature.interop", "verify", c.p("i")]),
        Row("interop-verify-tampered", ("roots-rebind",), "-", CommandFamily.REPLAY, 3, KIND_EXAMPLE,
            lambda c: ["literature.interop", "verify", c.p("i-tampered")],
            prepare=lambda c: _flip_first_string_char(
                _copy(c, c.work / "i", "i-tampered") / "sources" / "survey" / "population.json")),
        Row("compose", ("reference-catalog", "identity-resolution"), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: ["compose", c.p("i", "checking.orm.json"), c.p("i", "sources"),
                       "--out", c.p("compose-ok.population.json"), "--record", c.p("compose-ok.record.json")]),
        Row("compose-stale-catalog", ("reference-catalog",), "-", CommandFamily.OTHER, 3, KIND_GND,
            lambda c: ["compose", c.p("i-catalog", "checking.orm.json"), c.p("i-catalog", "sources"),
                       "--out", c.p("c1.population.json"), "--record", c.p("c1.record.json")],
            prepare=lambda c: _flip_first_string_char(
                _copy(c, c.work / "i", "i-catalog") / "reference" / "catalog.json")),
        Row("compose-bad-resolution", ("identity-resolution",), "-", CommandFamily.OTHER, 3, KIND_GND,
            lambda c: ["compose", c.p("i-resolution", "checking.orm.json"), c.p("i-resolution", "sources"),
                       "--out", c.p("c2.population.json"), "--record", c.p("c2.record.json")],
            prepare=lambda c: _set_resolution_digest(_copy(c, c.work / "i", "i-resolution"))),
        Row("validate-includes", ("shared-vocabulary",), "+", CommandFamily.CHECK, 0, KIND_GND,
            lambda c: ["validate", c.p("i", "extraction.orm.json"), "--format", "json"]),
        Row("validate-includes-stale", ("shared-vocabulary",), "-", CommandFamily.CHECK, 3, KIND_GND,
            lambda c: ["validate", c.p("i-vocab", "extraction.orm.json"), "--format", "json"],
            prepare=lambda c: _flip_first_string_char(
                _copy(c, c.work / "i", "i-vocab") / "vocabulary" / "literature.orm.json")),
        Row("assertion-admission", ("judgment-admission", "contextual-assertions"), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.assertion_admission", *_admission_args(c, "asr")],
            prepare=lambda c: _copy(c, c.lit / "assertions", "asr")),
        Row("assertion-admission-stale", ("judgment-admission", "contextual-assertions"), "-",
            CommandFamily.OTHER, 3, KIND_EXAMPLE, lambda c: ["literature.assertion_admission", *_admission_args(c, "asr-stale")],
            prepare=lambda c: _append(_copy(c, c.lit / "assertions", "asr-stale") / "evidence" / "passages.txt")),
        Row("assurance-reference", ("assurance-report",), "+", CommandFamily.ASSURANCE, 0, KIND_EXAMPLE,
            lambda c: ["literature.assurance_report", lit(c), lit(c, "answers", "answer-key.json"),
                       lit(c, "runs", "reference")]),
        Row("assurance-false-merge", ("assurance-report", "identity-resolution"), "-", CommandFamily.ASSURANCE, 1, KIND_PYTHON,
            lambda c: ["-c", M7_SCRIPT, c.p("m7")],
            env=lambda c: {**os.environ, "PYTHONPATH": str(c.archive / "tests")},
            check=_expect_json(lambda d: None if d.get("matches") is True
                               else f"report differs from expected/M7-false-merge.expected.json: {d}")),
        Row("lineage-assemble", ("lineage",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.lineage_run", "assemble", lit(c), lit(c, "lineage", "model.orm.yaml"),
                       lit(c, "lineage", "development-input.json"), c.p("lineage-run")]),
        Row("lineage-score", ("lineage",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.lineage_run", "score", lit(c, "lineage", "model.orm.yaml"), c.p("lineage-run"),
                       lit(c, "answers", "answer-key.json")]),
        Row("lineage-score-tampered", ("lineage",), "-", CommandFamily.OTHER, 3, KIND_EXAMPLE,
            lambda c: ["literature.lineage_run", "score", lit(c, "lineage", "model.orm.yaml"),
                       c.p("lineage-run-tampered"), lit(c, "answers", "answer-key.json")],
            prepare=_prep_tamper_admitted),
        Row("interop-dates", ("lineage-dates",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.interop", "dates", c.p("i"), "--out", c.p("advice.json")],
            check=lambda r, c: _file_check(c, "advice.json", _dates_controls)),
        Row("interop-dates-pinn", ("lineage-dates",), "-", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.interop", "dates", c.p("i"), "--out", c.p("advice-2.json")],
            check=lambda r, c: _file_check(c, "advice-2.json", _dates_pinn)),
        Row("name-diagnostics", ("name-diagnostics",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.name_diagnostics", lit(c, "diagnostics", "cases.json")]),
        Row("name-diagnostics-limit", ("name-diagnostics",), "-", CommandFamily.OTHER, 3, KIND_EXAMPLE,
            lambda c: ["literature.name_diagnostics", c.p("names-501.json")], prepare=_prep_names_limit,
            check=_expect_json(lambda d: None if d.get("reason") == "input:occurrence-limit" else f"reason={d.get('reason')}")),
        Row("revision-run", ("semantic-revision",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.revision", "run", c.p("r"), "--out", c.p("revision-run")]),
        Row("revision-replay", ("semantic-revision",), "+", CommandFamily.REPLAY, 0, KIND_EXAMPLE,
            lambda c: ["literature.revision", "replay", c.p("revision-run")]),
        Row("revision-replay-stale", ("semantic-revision",), "-", CommandFamily.REPLAY, 3, KIND_EXAMPLE,
            lambda c: ["literature.revision", "replay", c.p("revision-tampered")],
            prepare=lambda c: _flip_first_string_char(
                _copy(c, c.work / "revision-run", "revision-tampered") / "after" / "claims.json")),
        Row("verdict-supported", ("support-verdicts",), "+", CommandFamily.VERDICT, 0, KIND_EXAMPLE,
            lambda c: ["literature.verdict", "evaluate", c.p("revision-run"), "--case", "both-approved"]),
        Row("verdict-stale-edition", ("support-verdicts",), "-", CommandFamily.VERDICT, 3, KIND_EXAMPLE,
            lambda c: ["literature.verdict", "evaluate", c.p("revision-run"), "--case", "both-approved",
                       "--edition", "revised"]),
        Row("verdict-rejected", ("support-verdicts",), "-", CommandFamily.VERDICT, 2, KIND_EXAMPLE,
            lambda c: ["literature.verdict", "evaluate", c.p("revision-run"), "--case", "reject-both"]),
        Row("verdict-run", ("support-verdicts",), "+", CommandFamily.OTHER, 0, KIND_EXAMPLE,
            lambda c: ["literature.verdict", "run", c.p("revision-run"), "--out", c.p("verdict-run")]),
        Row("verdict-replay", ("support-verdicts",), "+", CommandFamily.REPLAY, 0, KIND_EXAMPLE,
            lambda c: ["literature.verdict", "replay", c.p("revision-run"), c.p("verdict-run")]),
        Row("witness-time", ("witness-replay",), "+", CommandFamily.WITNESS, 0, KIND_GND,
            lambda c: ["witness", c.p("q", "time.orm.yaml"), c.p("q", "time.population.json"),
                       "--targets", c.p("q", "time.targets.json"), "--raw", "--out", c.p("q", "time.witness.json")],
            prepare=lambda c: _quantity(c, "q")),
        Row("replay-time", ("witness-replay",), "+", CommandFamily.WITNESS, 0, KIND_GND,
            lambda c: ["replay", c.p("q", "time.orm.yaml"), c.p("q", "time.population.json"),
                       c.p("q", "time.witness.json"), "--raw"]),
        Row("witness-fractional", ("witness-replay",), "-", CommandFamily.WITNESS, 2, KIND_GND,
            lambda c: ["witness", lit(c, "depth_fixture", "decimal.orm.json"),
                       lit(c, "depth_fixture", "decimal.population.json"),
                       "--targets", lit(c, "depth_fixture", "decimal.targets.json")],
            check=_expect_json(lambda d: None if any(r.get("reason") == "fragment-unsupported"
                                                     for r in d.get("results", [])) else "no fragment-unsupported result")),
        Row("replay-tampered", ("witness-replay",), "-", CommandFamily.WITNESS, 1, KIND_GND,
            lambda c: ["replay", c.p("q", "time.orm.yaml"), c.p("q", "time.population.json"),
                       c.p("q", "tampered.witness.json"), "--raw"],
            prepare=lambda c: _tamper_certificate(c.work / "q")),
        Row("anchors-ok", ("anchors",), "+", CommandFamily.CHECK, 0, KIND_GND,
            lambda c: _anchor_args(c, "q")),
        Row("anchors-unwitnessed", ("anchors",), "-", CommandFamily.CHECK, 1, KIND_GND,
            lambda c: _anchor_args(c, "q-unwitnessed"), prepare=_prep_unwitnessed),
        Row("anchors-stale-binding", ("anchors",), "-", CommandFamily.CHECK, 3, KIND_GND,
            lambda c: _anchor_args(c, "q-stale"), prepare=_prep_stale_span),
        Row("why-not", ("why-not",), "+", CommandFamily.WHY_NOT, 0, KIND_GND,
            lambda c: ["why-not", "--fact-type", "lineage_step", c.p("lineage-run", "model.orm.yaml"),
                       c.p("lineage-run", "composed.population.json")],
            check=_expect_json(lambda d: None if d.get("status") in ("derived", "blocked") else f"status={d.get('status')}")),
        Row("why-not-asserted", ("why-not",), "-", CommandFamily.WHY_NOT, 3, KIND_GND,
            lambda c: ["why-not", "--fact-type", "assertion_statement", c.p("lineage-run", "model.orm.yaml"),
                       c.p("lineage-run", "composed.population.json")]),
        Row("ossie", ("interchange",), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: ["export", "ossie", lit(c, "assertions", "model.orm.yaml"), "--out", c.p("ossie.yaml"),
                       "--loss-report", c.p("ossie-loss.json")]),
        Row("ossie-include-root", ("interchange",), "-", CommandFamily.OTHER, 3, KIND_GND,
            lambda c: ["export", "ossie", c.p("i", "checking.orm.json"), "--out", c.p("ossie-2.yaml"), "--strict"]),
        Row("discover", ("discover",), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: _discover_args(c, "discover.json", "discover-report.json")),
        Row("discover-out-exists", ("discover",), "-", CommandFamily.OTHER, 11, KIND_GND,
            lambda c: _discover_args(c, "discover.json", "discover-report-2.json")),
        Row("amie-export", ("rule-mining",), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: ["export", "ilp", lit(c, "model.orm.yaml"), "--population",
                       lit(c, "runs", "reference", "composed", "population.json"), "--target", "amie",
                       "--out", c.p("amie")]),
        Row("mined-rules-admitted", ("rule-mining",), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: _mined_args(c, "amie-2", "mined-admitted.json"), prepare=_staffing),
        Row("mined-rules-refused", ("rule-mining",), "-", CommandFamily.OTHER, 3, KIND_GND,
            lambda c: _mined_args(c, "amie", "mined-refused.json")),
        Row("pyrel-lqp", ("pyrel-export",), "+", CommandFamily.OTHER, 0, KIND_GND,
            lambda c: ["export", "pyrel", "--rules", "lqp", "--out", c.p("rules-lqp.py"),
                       str(c.gnd_root / "examples" / "rule_check" / "model.orm.yaml")]),
        Row("pyrel-unsupported", ("pyrel-export",), "-", CommandFamily.OTHER, 3, KIND_GND,
            lambda c: ["export", "pyrel", "--rules", "lqp", "--out", c.p("rules-lqp-2.py"), lit(c, "model.orm.yaml")]),
    ]


# The text GND prints for each negative outcome, taken from the installed 0.4.0 (r1 F2):
# a negative row passes only when its exit AND this text match, so an unrelated refusal
# with the same exit fails it.
REASONS: Mapping[str, str] = {
    "reading-replay-stale": "stale retained reading output",
    "interop-verify-tampered": "rebind differs at survey/population.json",
    "compose-stale-catalog": "reference raw hash does not match declared sha256",
    "compose-bad-resolution": "resolution digest differs from the manifest",
    "validate-includes-stale": "include:hash-mismatch",
    "assertion-admission-stale": "judgments:refused",
    "lineage-score-tampered": "input:admission-binding",
    "name-diagnostics-limit": "input:occurrence-limit",
    "revision-replay-stale": "release dependencies are stale",
    "verdict-stale-edition": "revalidation-required",
    "verdict-rejected": '"decision":"unsupported"',
    "witness-fractional": "fragment-unsupported",
    "replay-tampered": "proof-invalid",
    "anchors-unwitnessed": "Source anchor fail for time_step",
    "anchors-stale-binding": "anchor:binding-mismatch: source span or excerpt differs",
    "why-not-asserted": "input-invalid",
    "ossie-include-root": "include:unsupported-route: export.ossie",
    "discover-out-exists": "exists",
    "mined-rules-refused": "does-not-hold",
    "pyrel-unsupported": "is a later phase of the rules programs",
    "reading-unavailable": '"state":"unavailable"',
    "assurance-false-merge": '"matches": true',
}

# Files each row must produce, relative to work/ (r1 F5).
OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "depth": ("depth-run/report.json",),
    "interop-run": ("i/interop.json", "i/composition.json"),
    "compose": ("compose-ok.population.json", "compose-ok.record.json"),
    "assertion-admission": ("asr/trial-admission/admission.json", "asr/trial-admission/admitted.population.json"),
    "lineage-assemble": ("lineage-run/composed.population.json", "lineage-run/composition.json"),
    "interop-dates": ("advice.json",),
    "interop-dates-pinn": ("advice-2.json",),
    "revision-run": ("revision-run/release.json",),
    "witness-time": ("q/time.witness.json",),
    "ossie": ("ossie.yaml", "ossie-loss.json"),
    "discover": ("discover.json", "discover-report.json"),
    "amie-export": ("amie/amie/facts.tsv", "amie/amie/relations.json"),
    "mined-rules-admitted": ("mined-admitted.json",),
    "pyrel-lqp": ("rules-lqp.py",),
}


def build_rows() -> list[Row]:
    """The probe rows, in execution order (later rows use earlier rows' outputs)."""
    rows = []
    for row in _rows():
        if row.polarity == "-" and row.id not in REASONS and row.check is None:
            raise ValueError(f"negative row {row.id} has neither a reason text nor a check")
        rows.append(replace(row, reason=REASONS.get(row.id, row.reason),
                            outputs=OUTPUTS.get(row.id, row.outputs)))
    return rows


def _set_resolution_digest(root: Path) -> None:
    manifest_path = root / "sources" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolution"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))


def _append(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b" ")


def _admission_args(ctx: Context, folder: str) -> list[str]:
    root = ctx.work / folder
    return [str(root / "model.orm.yaml"), str(root / "raw.population.json"), str(root / "policy.json"),
            str(root / "receipts.json"), str(root / "evidence"), "--out", str((root / "trial-admission").resolve())]


def _anchor_args(ctx: Context, folder: str) -> list[str]:
    return ["check", ctx.p(folder, "time.orm.yaml"), ctx.p(folder, "time.population.json"),
            "--document", ctx.p(folder, "source.txt"), "--anchors", ctx.p(folder, "time.anchors.json"),
            "--format", "json"]


def _discover_args(ctx: Context, out: str, report: str) -> list[str]:
    return ["discover", str(ctx.lit / "model.orm.yaml"), str(ctx.lit / "runs" / "reference" / "composed" / "population.json"),
            "--out", ctx.p(out), "--report", ctx.p(report), "--actor", "v6.1-phase-1-probe",
            "--requested-at", "2026-11-18T00:00:00+00:00"]


def _mined_args(ctx: Context, run: str, out: str) -> list[str]:
    root = ctx.work / "staffing"
    return ["import", "mined-rules", str(root / "model.orm.yaml"), str(root / "learners" / run / "rules.txt"),
            "--population", str(root / "population.yaml"), "--reported", str(root / "learners" / run / "reported.json"),
            "--target", "amie", "--out", ctx.p(out), "--format", "json"]


def _tamper_certificate(root: Path) -> None:
    """Change a value inside the first proof node; the binding (digests) stays as written (r1 F1)."""
    cert = json.loads((root / "time.witness.json").read_text())
    cell = cert["nodes"][0]["tuple"][0]
    cell["value"] = cell["value"] + 5 if isinstance(cell["value"], int) else f"{cell['value']}-tampered"
    (root / "tampered.witness.json").write_text(json.dumps(cert))


def _file_check(ctx: Context, name: str, predicate: Callable[[dict[str, Any]], str | None]) -> str | None:
    try:
        payload = json.loads((ctx.work / name).read_text())
    except (OSError, ValueError) as exc:
        return f"cannot read {name}: {exc}"
    return predicate(payload)


# --- coverage, source archive, receipt ------------------------------------------------


def coverage(rows: Iterable[Row], capabilities: Sequence[str] = CAPABILITIES) -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, list[str]]] = {cap: {"positive": [], "negative": []} for cap in capabilities}
    for row in rows:
        for cap in row.capabilities:
            table.setdefault(cap, {"positive": [], "negative": []})
            table[cap]["positive" if row.polarity == "+" else "negative"].append(row.id)
    return table


def coverage_holes(table: Mapping[str, Mapping[str, list[str]]]) -> list[str]:
    return sorted(f"{cap}:{side}" for cap, sides in table.items() for side in ("positive", "negative") if not sides[side])


def prepare_archive(source: Path, out: Path, installed_literature: Path, *, env: Mapping[str, str] | None = None) -> str | None:
    """Archive GND_COMMIT's helper paths into ``out``; return a refusal reason or None."""
    probe = subprocess.run(["git", "-C", str(source), "cat-file", "-e", f"{GND_COMMIT}^{{commit}}"],
                           capture_output=True, check=False)
    if probe.returncode != 0:
        return f"commit {GND_COMMIT} is not in {source}"
    archive = subprocess.run(["git", "-C", str(source), "archive", "--format=tar", GND_COMMIT, *ARCHIVE_PATHS],
                             capture_output=True, check=False)
    if archive.returncode != 0:
        return f"git archive failed: {archive.stderr.decode(errors='replace')[:200]}"
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(out, filter="data")
    archived = out / LITERATURE_IN_ARCHIVE
    for path in sorted(p for p in archived.rglob("*") if p.is_file()):
        rel = path.relative_to(archived)
        installed = installed_literature / rel
        if not installed.is_file() or installed.read_bytes() != path.read_bytes():
            return f"installed GND differs from commit {GND_COMMIT[:8]} at {rel}"
    pytest_check = gnd_tool.run_gnd_python(["-c", "import pytest"], family=CommandFamily.OTHER, cwd=out, env=env)
    if pytest_check.exit_code != 0:
        return "pytest is not importable by the gnd interpreter (install with --with pytest==9.1.1)"
    return None


def _installed_gnd_root(env: Mapping[str, str] | None) -> Path | None:
    result = gnd_tool.run_gnd_python(["-c", "import gnd, os; print(os.path.dirname(gnd.__file__))"],
                                     family=CommandFamily.OTHER, cwd=Path.cwd(), env=env)
    if result.exit_code != 0:
        return None
    return Path(result.stdout.strip())


Runner = Callable[[Row, list[str], Context], GndResult]


def real_runner(row: Row, argv: list[str], ctx: Context) -> GndResult:
    env = row.env(ctx) if row.env else None
    if row.kind == KIND_GND:
        return gnd_tool.run_gnd(argv, family=row.family, cwd=ctx.work, env=env)
    if row.kind == KIND_EXAMPLE:
        return gnd_tool.run_gnd_example(argv[0], argv[1:], family=row.family, cwd=ctx.work, env=env)
    return gnd_tool.run_gnd_python(argv, family=row.family, cwd=ctx.archive, env=env)


def _redact_home(text: str, ctx: Context | None = None) -> str:
    """Replace the probe's own roots and the home directory, so a receipt names no private path."""
    if ctx is not None:
        for root, label in ((ctx.work, "<work>"), (ctx.archive, "<gnd-src>"), (ctx.gnd_root, "<gnd>")):
            text = text.replace(str(root), label)
    home = str(Path.home())
    return text.replace(home, "~") if home and home != "/" else text


def _digests(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


REVISION_REPLAY_CALLS = "not-reported-by-gnd"  # its report has no producer_calls; it replays by digest


def execute(rows: Sequence[Row], runner: Runner, ctx: Context, *, force_prepare: bool = False,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        problem = None
        result = None
        argv: list[str] = []
        try:
            if (ctx.real or force_prepare) and row.prepare:
                row.prepare(ctx)
            argv = row.args(ctx)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            problem = f"setup failed: {type(exc).__name__}: {exc}"
        else:
            result = runner(row, argv, ctx)
        outputs: dict[str, str] = {}
        if result is not None:
            if result.exit_code != 0 and "No module named 'gnd.examples" in result.stderr:
                problem = "module missing"
            elif result.exit_code != row.expected_exit:
                problem = f"exit {result.exit_code}, expected {row.expected_exit}"
            elif row.reason and row.reason not in result.stdout + result.stderr:
                problem = f"exit {result.exit_code} as expected, but GND did not say {row.reason!r}"
            else:
                missing = [name for name in row.outputs if not (ctx.work / name).is_file()]
                if missing:
                    problem = f"outputs not written: {missing}"
                else:
                    outputs = {name: hashlib.sha256((ctx.work / name).read_bytes()).hexdigest() for name in row.outputs}
                    problem = row.check(result, ctx) if row.check is not None else None
        results.append({
            "id": row.id, "capabilities": list(row.capabilities), "polarity": row.polarity,
            "family": str(row.family), "argv": [_redact_home(a, ctx) for a in argv if "\n" not in a],
            "expected_exit": row.expected_exit, "reason_required": row.reason,
            "observed_exit": MODULE_MISSING if problem == "module missing" else (result.exit_code if result else None),
            "meaning": result.meaning if result else None,
            "problem": _redact_home(problem, ctx) if problem else None, "ok": problem is None,
            "outputs": outputs,
            "stderr_tail": _redact_home(result.stderr[-300:], ctx) if result and problem else "",
        })
    reading = (ctx.reports.get("reading-run") or {}).get("producer_calls")
    replay = (ctx.reports.get("reading-replay") or {}).get("producer_calls")
    calls = {"paid_provider_calls": 0,
             "saved_adapter_calls": {"reading-run": reading} if reading is not None else {},
             "replay_producer_calls": {**({"reading-replay": replay} if replay is not None else {}),
                                       "revision-replay": REVISION_REPLAY_CALLS}}
    return results, calls


def run_probe(out: Path, gnd_source: Path, *, runner: Runner | None = None, env: Mapping[str, str] | None = None,
              rows: Sequence[Row] | None = None, gnd_root: Path | None = None,
              skip_archive: bool = False, force_prepare: bool = False) -> int:
    if out.exists():
        print(f"--out {out} exists; nothing written")
        return EXIT_OUT_EXISTS
    rows = list(rows) if rows is not None else build_rows()
    real = runner is None
    if real:
        try:
            version = gnd_tool.require_gnd(env=env)
        except GndUnavailable as exc:
            print(f"refused: {exc}")
            return EXIT_REFUSED
        gnd_root = _installed_gnd_root(env)
        if gnd_root is None:
            print("refused: cannot locate the installed gnd package")
            return EXIT_REFUSED
        gnd_python_path = _redact_home(str(gnd_tool.gnd_python(env=env)))
    else:
        version = gnd_tool.GND_VERSION
        gnd_root = gnd_root or out / "fake-gnd"
        gnd_python_path = "<fake runner>"
    work = (out / "work").resolve() if out.is_absolute() else (Path.cwd() / out / "work").resolve()
    archive = work.parent / "gnd-src"
    if not skip_archive:
        refusal = prepare_archive(gnd_source, archive, gnd_root / "examples" / "literature", env=env)
        if refusal:
            print(f"refused: {refusal}")
            return EXIT_REFUSED
    work.mkdir(parents=True, exist_ok=True)
    ctx = Context(work=work, gnd_root=gnd_root, archive=archive, real=real)
    results, calls = execute(rows, runner or real_runner, ctx, force_prepare=force_prepare)
    table = coverage(rows)
    holes = coverage_holes(table)
    receipt = {
        "schema": SCHEMA, "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "gnd_version": version,
        # gnd_root: the installed package, home shortened to ~
        "gnd_commit": GND_COMMIT, "gnd_root": _redact_home(str(gnd_root)), "gnd_python": gnd_python_path,
        **calls,
        "rows": results, "coverage": table, "coverage_holes": holes,
    }
    replay_calls = calls["replay_producer_calls"].get("reading-replay")
    receipt["ok"] = (all(r["ok"] for r in results) and not holes
                     and (not real or (replay_calls is not None and not any(replay_calls.values()))))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": receipt["ok"], "rows": len(results), "failed": [r["id"] for r in results if not r["ok"]],
                      "coverage_holes": holes}))
    return EXIT_OK if receipt["ok"] else EXIT_ROW_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="new directory for the receipt and work files")
    parser.add_argument("--gnd-source", required=True, type=Path, help="a gnd-public checkout holding the pinned commit")
    args = parser.parse_args(argv)
    return run_probe(args.out, args.gnd_source)
