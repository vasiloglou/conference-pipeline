#!/usr/bin/env python3
"""Fail-closed audit for hardcoded conference-specific values in reusable code.

The platform must serve any conference (NeurIPS, ICLR, ICML, ...) from the same
code, reading identity from each workspace's ``config.yaml`` and generated
manifests — never from literals baked into ``src/conference_pipeline/`` or the
shared ``scripts/``. This audit greps the reusable surface for conference names,
years, and program domains that appear in *functional* string context and exits
non-zero when any survive, so a milestone cannot close with a portability
regression.

Run:
    python scripts/audit_conference_hardcoding.py            # human report
    python scripts/audit_conference_hardcoding.py --json     # machine report

Suppressing an intentional literal: append ``# conference-literal-ok`` to the
line (e.g. a guarded historical fallback, the fixed RelationalAI brand lock, or
an argparse help example). Whole conference-specific entrypoints are excluded by
path (see ALLOWLIST_PATH_SUBSTRINGS).

Fix pattern for a real finding: read the value from the workspace instead —
``from conference_pipeline.conference_identity import load_conference_identity,
conference_name`` for name/year/urls, ``analysis/analysis_manifest.json`` for
topics, ``decks/<id>/presentation_frame.json`` for audience framing.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The automated fail-closed gate covers the reusable platform LIBRARY, which is
# imported everywhere and must serve any conference. Entry-point and generator
# SCRIPTS (scripts/**) are a mix of active generators, conference-specific
# entrypoints, and historical one-off tools; the skill drives a broader manual
# sweep over those using the same patterns (see the SKILL.md rubric). Keeping the
# automated gate scoped to src/ makes it unambiguous and low-noise.
SCAN_DIRS = ("src/conference_pipeline",)

# Individual scripts that ARE reusable multi-conference platform code despite
# living under scripts/. This is the executive/digest chain -- the path that
# built ICLR 2026's, ICML 2026's and CVPR 2026's executive decks -- plus the
# render gates it feeds. A conference literal in one of these is a portability
# defect exactly as it is in src/.
#
# Added 2026-09-15 after build_digest_deck_plan.py was found carrying a
# hardcoded table of ICLR 2026 topic slugs: every other conference matched none
# of them, every topic fell through un-grouped, and the executive index slide
# silently rendered in a layout built for a third as many items. The audit's
# old src/-only scope could not see the file at all.
#
# Scoped to a named list rather than all of scripts/presentations deliberately.
# That directory also holds ~20 NeurIPS-2025-era generators that each carry a
# copy-pasted 20-slug topic table (524 findings). Those are real instances of
# the same defect class and are tracked as debt in
# docs/reference/pipeline-capabilities.md, but they are legacy defaults on a
# path no current conference runs, and folding them in now would mean shipping
# a gate that fails by 524 and gets ignored. This list is the code that runs
# today, and it is enforced.
SCAN_FILES = (
    "scripts/presentations/build_executive_summary.py",
    "scripts/presentations/generate_executive_topic_digests.py",
    "scripts/presentations/build_digest_deck_plan.py",
    "scripts/presentations/add_bigpicture_diagrams.py",
    "scripts/presentations/generate_executive_overview_decks.py",
    "scripts/presentations/build_executive_videos.py",
    "scripts/presentations/generate_executive_narration.py",
    "scripts/presentations/generate_executive_dialogue.py",
    "scripts/presentations/add_bookend_slides.py",
    "scripts/presentations/inspect_rendered_slides.py",
    "scripts/presentations/inspect_rendered_slides_llm.py",
    "scripts/presentations/validate_video_sync.py",
)

# Path substrings that are legitimately conference-specific or out of scope.
# Per-conference entrypoints may name their conference; they should still read
# specifics from config, but a literal in their own name/title is not a platform
# portability defect. iclr2025_live's ICLR-2025 acquisition crawl is ICLR-only
# and not on the multi-conference path (new conferences use their own runners).
ALLOWLIST_PATH_SUBSTRINGS = (
    "/tests/",
    "test_",
    "run_iclr2025",
    "run_iclr2026_production",
    "run_icml2026_production",
    "run_neurips",
    "supervise_icml2026",
    "run_phase26",
    "ai4physics",
    "iclr2025_live.py",  # ICLR-2025 acquisition module; ICLR-only, not multi-conf path
    "audit_conference_hardcoding.py",  # this file names conferences in docs
    # CVPR-2026-only acquisition/execution modules (verified: imported only by
    # scripts/run_cvpr2026_production.py and each other, never by another
    # conference's runner) -- same "single-conference module living under
    # src/conference_pipeline/" shape as iclr2025_live.py above.
    "acquisition/cvpr_r6_stage_admission.py",
    "acquisition/cvpr_r6_media_recovery.py",
    "acquisition/cvpr_event_media.py",
    "acquisition/cvpr_transcription_ocr.py",
    "acquisition/cvpr_corpus_download.py",
    "acquisition/_cvpr_r6_media_owner_impl.py",
    "acquisition/cvpr_registration_media.py",
    "acquisition/cvpr_program_spine.py",
    "acquisition/cvpr_r6_artifact_contract.py",
    "acquisition/cvpr_r6_checkpoint_migration.py",
    "acquisition/cvpr_r6_material_index.py",
    "r6_transcription_ocr_execution.py",  # CVPR-2026-only; same
)

SUPPRESS_MARKER = "conference-literal-ok"

# Functional conference literals: a conference name/year, a program domain, or a
# year-pinned virtual path. Kept deliberately high-signal to avoid noise.
PATTERNS = (
    re.compile(r"(ICLR|ICML|NeurIPS|CVPR)\s*20\d\d"),
    re.compile(r"\b(iclr|icml|neurips)\.cc\b"),
    re.compile(r"\bcvpr\.thecvf\.com\b"),
    re.compile(r"/virtual/20\d\d"),
    re.compile(r"NeurIPS-quality"),
)

# A topic slug ("01_reasoning_post_training"). Topic ids are defined per
# conference in analysis_manifest.json, so a literal one in reusable code is a
# table that silently matches nothing for every other conference -- the exact
# shape of the 2026-09-15 executive-index defect.
#
# Enforced on SCAN_FILES only, not repo-wide. Repo-wide it currently reports 55
# more findings in src/ (analysis/constants.py, analysis/steps/gap_fill.py and
# workflow.py each carry a legacy NeurIPS topic table, plus the deliberate
# NeurIPS proof-contract defaults in smoke_test.py and presentation_proof.py).
# Those are real and tracked as debt, but they are a separate cleanup from this
# fix; scoping the check to the chain that every conference runs today makes it
# a gate that passes and therefore one that stays on.
CHAIN_ONLY_PATTERNS = (re.compile(r"\b\d{2}_[a-z0-9]+(?:_[a-z0-9]+)+\b"),)

def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Ids of Constant string nodes that are module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        base = REPO_ROOT / rel
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    for rel in SCAN_FILES:
        path = REPO_ROOT / rel
        if path.is_file():
            files.append(path)
        else:
            # A renamed or deleted module must not silently drop out of scope.
            raise SystemExit(
                f"audit scope error: SCAN_FILES names {rel}, which does not exist. "
                "Update SCAN_FILES if the module moved."
            )
    return files


def _allowlisted(path: Path) -> bool:
    posix = path.as_posix()
    return any(sub in posix for sub in ALLOWLIST_PATH_SUBSTRINGS)


def _match_literal(value: str, *, chain_file: bool) -> str | None:
    patterns = PATTERNS + (CHAIN_ONLY_PATTERNS if chain_file else ())
    for pattern in patterns:
        m = pattern.search(value)
        if m:
            return m.group(0)
    return None


def _is_chain_file(path: Path) -> bool:
    """True for the reusable executive-chain modules named in SCAN_FILES."""
    return any(path == (REPO_ROOT / rel) for rel in SCAN_FILES)


# Pinned run counts: a stage-contract module that asserts a specific run's
# numbers (5,258 routes, 350 captions, 2,002 unavailable) turns one run's
# facts into a rule the codebase enforces forever. On CVPR 2026 that pin is
# what pushed later transcript recoveries into side directories that nothing
# consumed. Contracts verify structure and lineage; counts are reported by
# receipts and derived from digest-bound upstreams, never typed into src/.
RUN_COUNT_MODULE_MARKER = "contract"
RUN_COUNT_MIN = 100
RUN_COUNT_SUPPRESS_MARKER = "run-count-literal-ok"


def _is_contract_module(path: Path) -> bool:
    return RUN_COUNT_MODULE_MARKER in path.name


# The same rule for registry DATA: a workspace binding's runtime_contract must
# not `equals`-pin a run count on the transcript-admission path. Scoped to the
# stages that gate what the corpus indexes; the historical r4/r5/r6
# acquisition bindings under download-corpus and transcription-ocr-r6 still
# carry 12 such pins (program_item_count, event_route_count, ...), tracked as
# debt rather than failed here because they bind completed, digest-verified
# runs and have no appendable successor yet.
STAGE_OWNER_REGISTRY = REPO_ROOT / "src/conference_pipeline/stage_owner_registry.json"
RUN_COUNT_FREE_STAGE_IDS = ("transcription-ocr", "corpus-inventory-index")


def _registry_count_pins() -> list[dict[str, object]]:
    try:
        registry = json.loads(STAGE_OWNER_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[dict[str, object]] = []
    bindings = registry.get("stage_bindings") if isinstance(registry, dict) else None
    if not isinstance(bindings, dict):
        return findings

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            value = node.get("equals")
            if isinstance(value, int) and not isinstance(value, bool) and value >= RUN_COUNT_MIN:
                findings.append(
                    {
                        "file": STAGE_OWNER_REGISTRY.relative_to(REPO_ROOT).as_posix(),
                        "line": 0,
                        "match": f"pinned run count {value}",
                        "text": f"{where}: {node.get('path')} equals {value}",
                    }
                )
            for key, child in node.items():
                walk(child, f"{where}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{where}[{index}]")

    for stage_id in RUN_COUNT_FREE_STAGE_IDS:
        walk(bindings.get(stage_id), f"stage_bindings/{stage_id}")
    return findings


def audit() -> list[dict[str, object]]:
    findings: list[dict[str, object]] = _registry_count_pins()
    for path in _iter_python_files():
        if _allowlisted(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        source_lines = source.splitlines()
        chain_file = _is_chain_file(path)
        docstring_ids = _docstring_constant_ids(tree)
        contract_module = _is_contract_module(path)
        for node in ast.walk(tree):
            if (
                contract_module
                and isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and node.value >= RUN_COUNT_MIN
            ):
                lineno = getattr(node, "lineno", 0)
                line_text = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
                if RUN_COUNT_SUPPRESS_MARKER in line_text or SUPPRESS_MARKER in line_text:
                    continue
                findings.append(
                    {
                        "file": path.relative_to(REPO_ROOT).as_posix(),
                        "line": lineno,
                        "match": f"pinned run count {node.value}",
                        "text": line_text.strip()[:200],
                    }
                )
                continue
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docstring_ids:
                continue
            match = _match_literal(node.value, chain_file=chain_file)
            if not match:
                continue
            lineno = getattr(node, "lineno", 0)
            line_text = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
            # Inline suppression for intentional literals (guarded historical
            # fallback, brand lock, argparse help example). Scanned across the
            # literal's FULL line span: a multi-line or implicitly-concatenated
            # string reports its first line, but the marker is written on the
            # line the reader sees the literal on, which may be a later one.
            end_lineno = getattr(node, "end_lineno", lineno) or lineno
            span = source_lines[max(0, lineno - 1) : min(len(source_lines), end_lineno)]
            if any(SUPPRESS_MARKER in text for text in span):
                continue
            findings.append(
                {
                    "file": path.relative_to(REPO_ROOT).as_posix(),
                    "line": lineno,
                    "match": match,
                    "text": line_text.strip()[:200],
                }
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    findings = audit()
    if args.json:
        print(json.dumps({"status": "fail" if findings else "pass", "findings": findings}, indent=2))
    else:
        if not findings:
            print("PASS: no hardcoded conference literals in reusable code.")
        else:
            print(f"FAIL: {len(findings)} hardcoded conference literal(s) in reusable code:\n")
            for f in findings:
                print(f"  {f['file']}:{f['line']}  [{f['match']}]")
                print(f"      {f['text']}")
            print(
                "\nFix: read the value from the workspace config/manifest/frame via "
                "conference_pipeline.conference_identity (name/year/urls), "
                "analysis_manifest.json (topics), or presentation_frame.json "
                "(audience framing). If a literal is genuinely intentional (guarded "
                f"historical fallback, brand lock, example), append '# {SUPPRESS_MARKER}'."
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
