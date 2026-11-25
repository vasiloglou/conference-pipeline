#!/usr/bin/env python3
"""Compile a deck's figure specs into SVGs and figure manifests.

Deterministic and free: no provider call, no network. Given a directory of
figure specs (or one spec file), it compiles each to SVG, writes a manifest per
figure, checks the deck's variety quotas, and writes one receipt.

    python scripts/presentations/compile_figures.py --deck-dir conferences/<c>/decks/<id>
    python scripts/presentations/compile_figures.py --spec path/to/spec.json --out figure.svg
    python scripts/presentations/compile_figures.py --gallery out/dir   # every archetype
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from conference_pipeline.figures import compile_figure  # noqa: E402
from conference_pipeline.figures.compiler import COMPILER_VERSION  # noqa: E402
from conference_pipeline.figures.errors import FigureCompilerError  # noqa: E402
from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402
from conference_pipeline.figures.ledger import build_ledger, load_ledger, verify_spec_grounding  # noqa: E402
from conference_pipeline.figures.selection import deck_variety_report  # noqa: E402
from conference_pipeline.figures.errors import FigureSpecError  # noqa: E402
from conference_pipeline.figures.spec import load_spec, structure_findings  # noqa: E402

SPEC_DIR_NAME = "figure_specs"
OUTPUT_DIR_NAME = "assets/compiled"
RECEIPT_NAME = "figure_compile_receipt.json"
LEDGER_NAME = "numeric_claims_ledger.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _cached_figure(
    payload: dict[str, Any], svg_path: Path, manifest_path: Path, deck_dir: Path
) -> dict[str, Any] | None:
    """The previous compile of this exact spec, or ``None`` to compile it again.

    A hit needs three things to agree: the manifest's spec hash matches the hash
    of the spec we were handed, the manifest was written by this compiler
    version, and the SVG on disk still hashes to what the manifest recorded. Any
    disagreement -- an edited spec, an upgraded compiler, a truncated or
    hand-edited SVG -- recompiles rather than trusting the stale file.
    """

    if not (svg_path.is_file() and manifest_path.is_file()):
        return None
    try:
        spec = load_spec(payload, strict=False)
    except FigureSpecError:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("spec_hash") or "") != spec.spec_hash():
        return None
    if str(manifest.get("compiler_version") or "") != COMPILER_VERSION:
        return None
    recorded_sha = str(manifest.get("svg_sha256") or "")
    actual_sha = hashlib.sha256(svg_path.read_bytes()).hexdigest()
    if not recorded_sha or recorded_sha != actual_sha:
        return None
    grounding = manifest.get("grounding")
    grounding = grounding if isinstance(grounding, dict) else {}
    layout = manifest.get("layout")
    layout = layout if isinstance(layout, dict) else {}
    return {
        "figure_id": str(manifest.get("figure_id") or svg_path.stem),
        "slide_id": str(manifest.get("slide_id") or ""),
        "section": spec.section,
        "archetype": str(manifest.get("archetype") or ""),
        "family": str(manifest.get("family") or ""),
        "grounding_kind": str(grounding.get("kind") or ""),
        "ledger_rows": [str(row) for row in (grounding.get("ledger_rows") or [])],
        "svg_path": str(svg_path.relative_to(deck_dir)),
        "manifest_path": str(manifest_path.relative_to(deck_dir)),
        "svg_sha256": actual_sha,
        "spec_hash": spec.spec_hash(),
        "layout_attempt": int(layout.get("attempt") or 0),
        "entities_missing": [str(item) for item in (manifest.get("entities_missing") or [])],
        "claim": str(manifest.get("claim") or ""),
        "cached": True,
    }


def _section_can_show_a_number(ledger, cited_rows, claim: str) -> bool:
    """Does this slide give its section a number a CHART could carry?

    Two conditions, and the deck fails both in different places.

    The row has to be plottable. Half the rows on both CVPR decks have a value
    and a unit and no metric -- the span says a number happened, not what
    quantity it names -- so charting one means inventing its axis label, which
    is the exact failure the ledger exists to prevent.

    And the slide has to have CITED it. Text matching is how the ledger offers
    candidates to an authoring prompt, which is a job it does well and says so:
    "a single shared word is enough to OFFER a claim... not enough to ASSERT
    that a section has measured claims of its own". Raising the bar to three
    shared words did not change its nature. One row about hierarchical
    diffusion's memory was matched into three unrelated sections -- editing,
    video time, physics -- and each was told to draw a chart of a number that
    belonged to none of them. A citation is authored, deliberate and auditable;
    an overlap score is a hint.
    """

    if ledger is None:
        return bool(cited_rows)
    return any(
        (row := ledger.get(str(row_id))) is not None and row.plottable
        for row_id in cited_rows or ()
    )


def compile_deck(
    deck_dir: Path, *, spec_dir: Path | None = None, output_dir: Path | None = None,
    vision_deck: bool = False, ledger_path: Path | None = None,
    allow_variety_findings: bool = False, use_cache: bool = True,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    """Compile every spec in a deck; return the receipt payload."""

    deck_dir = deck_dir.resolve()
    specs_root = spec_dir or (deck_dir / SPEC_DIR_NAME)
    out_root = output_dir or (deck_dir / OUTPUT_DIR_NAME)
    blockers: list[dict[str, Any]] = []
    # Historical generations leave old slide specs beside the active ones.
    # The authoring receipt identifies the current deck's exact set; globbing
    # every JSON compiled stale figures and could block a clean current deck.
    authoring_receipt = deck_dir / "figure_spec_receipt.json"
    if spec_dir is None and authoring_receipt.is_file():
        try:
            authored = _load_json(authoring_receipt)
            rows = authored.get("specs")
            if not isinstance(rows, list):
                raise ValueError("specs must be a list")
            if authored.get("status") != "pass":
                blockers.append({
                    "code": "figure_spec_authoring_not_pass",
                    "status": str(authored.get("status") or "missing"),
                })
            deck_contract = _load_json(deck_dir / "presentation_deck.json")
            expected_ids = {
                str(slide.get("slide_id") or "")
                for slide in deck_contract.get("slides", [])
                if isinstance(slide, dict)
                and str(slide.get("slide_kind") or "").strip().lower() not in {"events", "resources"}
            }
            listed_ids = {
                str(row.get("slide_id") or "") for row in rows if isinstance(row, dict)
            }
            if expected_ids != listed_ids:
                blockers.append({
                    "code": "figure_spec_receipt_incomplete",
                    "missing_slide_ids": sorted(expected_ids - listed_ids),
                    "extra_slide_ids": sorted(listed_ids - expected_ids),
                })
            spec_files = []
            seen: set[Path] = set()
            for row in rows:
                relative = str(row.get("spec_path") or "") if isinstance(row, dict) else ""
                path = (deck_dir / relative).resolve()
                if not relative or not path.is_relative_to(specs_root.resolve()):
                    blockers.append({"code": "figure_spec_receipt_path_invalid", "spec_path": relative})
                elif not path.is_file():
                    blockers.append({"code": "figure_spec_receipt_file_missing", "spec_path": relative})
                elif path not in seen:
                    seen.add(path)
                    spec_files.append(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            spec_files = []
            blockers.append({"code": "figure_spec_receipt_unreadable", "detail": str(error)[:300]})
    else:
        # Custom spec directories and standalone compiler calls keep the
        # historical glob behavior when no authoring receipt is available.
        spec_files = (
            sorted(path for path in specs_root.glob("*.json") if not path.name.endswith(".critic.json"))
            if specs_root.is_dir() else []
        )

    ledger = None
    resolved_ledger = ledger_path or (deck_dir / LEDGER_NAME)
    if resolved_ledger.is_file():
        ledger = load_ledger(resolved_ledger)

    compiled: list[dict[str, Any]] = []
    archetypes: list[str] = []
    cache_hits = 0
    sections: list[str] = []
    grounded_sections: set[str] = set()

    for spec_path in spec_files:
        payload = _load_json(spec_path)
        figure_id = str(payload.get("figure_id") or spec_path.stem)
        # A --dry-run placeholder is a structurally valid spec, so it compiles
        # to a real-looking figure and passes every gate. Refuse it here rather
        # than let a preflight artifact ship as a slide.
        if not allow_placeholders and "placeholder spec (dry run)" in str(
            (payload.get("grounding") or {}).get("note") or ""
        ):
            blockers.append({
                "code": "figure_spec_is_a_dry_run_placeholder",
                "figure_id": figure_id,
                "spec_path": str(spec_path.relative_to(deck_dir)),
                "detail": "this spec was written by --dry-run and was never authored; re-run spec generation for this slide",
            })
            continue
        if ledger is not None:
            missing_rows = verify_spec_grounding(payload, ledger)
            if missing_rows:
                blockers.append({
                    "code": "figure_spec_ledger_row_missing",
                    "figure_id": figure_id,
                    "spec_path": str(spec_path.relative_to(deck_dir)),
                    "missing_rows": missing_rows,
                    "detail": "the spec cites numeric-claims ledger rows that the deck's ledger does not hold",
                })
                continue

        # Cache key is the spec hash, as the design spec says: the same spec and
        # the same compiler version always produce byte-identical SVG, so
        # recompiling an unchanged figure can only reproduce the same bytes.
        svg_path = out_root / f"{figure_id}.svg"
        manifest_path = out_root / f"{figure_id}.manifest.json"
        cached = _cached_figure(payload, svg_path, manifest_path, deck_dir) if use_cache else None
        if cached is not None:
            cache_hits += 1
            archetypes.append(str(cached.get("archetype") or ""))
            section = str(cached.get("section") or "").strip() or "(unsectioned)"
            sections.append(section)
            if _section_can_show_a_number(ledger, cached.get("ledger_rows"),
                                          str(cached.get("claim") or "")):
                grounded_sections.add(section)
            if cached.get("entities_missing"):
                blockers.append({
                    "code": "figure_required_entity_missing",
                    "figure_id": figure_id,
                    "missing": cached["entities_missing"],
                    "detail": "required entities are absent from the rendered figure text",
                })
            compiled.append(cached)
            continue

        # Structure before geometry: a spec whose shape misstates its claim is
        # not a layout problem, and saying so with the rule's own code tells a
        # reader which slide needs re-authoring and why.
        lint_findings = structure_findings(load_spec(payload, strict=False))
        if lint_findings:
            blockers.append({
                "code": "figure_spec_lint",
                "figure_id": figure_id,
                "spec_path": str(spec_path.relative_to(deck_dir)),
                "findings": lint_findings,
                "detail": "re-author this slide; the compiler cannot restructure content",
            })
            continue

        try:
            figure = compile_figure(payload)
        except FigureCompilerError as error:
            blockers.append({
                "code": type(error).__name__,
                "figure_id": figure_id,
                "spec_path": str(spec_path.relative_to(deck_dir)),
                "detail": str(error),
                "findings": list(getattr(error, "findings", [])),
            })
            continue
        figure.write(svg_path, manifest_path)
        archetypes.append(figure.spec.archetype)
        section = (figure.spec.section or "").strip() or "(unsectioned)"
        sections.append(section)
        # A section counts as grounded when the deck's ledger actually holds
        # measured claims for one of its slides -- either cited by this figure,
        # or findable from the claim text. Sections with nothing measured to
        # show are not held to the per-section quota.
        if _section_can_show_a_number(ledger, figure.spec.grounding.ledger_rows,
                                      figure.spec.claim):
            grounded_sections.add(section)
        compiled.append({
            "figure_id": figure_id,
            "slide_id": figure.spec.slide_id,
            "section": figure.spec.section,
            "archetype": figure.spec.archetype,
            "family": figure.spec.family,
            "grounding_kind": figure.spec.grounding.kind,
            "ledger_rows": list(figure.spec.grounding.ledger_rows),
            "svg_path": str(svg_path.relative_to(deck_dir)),
            "manifest_path": str(manifest_path.relative_to(deck_dir)),
            "svg_sha256": figure.svg_sha256,
            "spec_hash": figure.spec.spec_hash(),
            "layout_attempt": figure.attempt,
            "entities_missing": figure.manifest.entities_missing,
            "claim": figure.spec.claim,
            "cached": False,
        })
        if figure.manifest.entities_missing:
            blockers.append({
                "code": "figure_required_entity_missing",
                "figure_id": figure_id,
                "missing": figure.manifest.entities_missing,
                "detail": "required entities are absent from the rendered figure text",
            })

    variety = deck_variety_report(archetypes, vision_deck=vision_deck,
                                  sections=sections, grounded_sections=grounded_sections)
    if variety.findings and not allow_variety_findings:
        # The canary shipped 27 box-and-arrow slides and nobody noticed until a
        # reviewer said so, which is what a non-blocking quota buys. Failing
        # closed here makes monotony a build failure with a named escape.
        blockers.append({
            "code": "figure_deck_variety_quota",
            "detail": "; ".join(variety.findings),
            "escape": "--allow-variety-findings records the mix without blocking",
        })
    return {
        "receipt_type": "conference_pipeline.figures.compile",
        "compiler_version": COMPILER_VERSION,
        "deck_dir": str(deck_dir),
        "spec_dir": str(specs_root),
        "output_dir": str(out_root),
        "spec_count": len(spec_files),
        "compiled_count": len(compiled),
        "cache_hit_count": cache_hits,
        "status": "pass" if compiled and not blockers else "blocked",
        "figures": compiled,
        "variety": variety.to_dict(),
        "numeric_ledger": {
            "path": str(resolved_ledger) if ledger is not None else "",
            "claim_count": len(ledger.claims) if ledger is not None else 0,
        },
        "blockers": blockers,
    }


def write_render_contracts(
    deck_dir: Path, receipt: dict[str, Any], *, presentation_id: str = ""
) -> dict[str, str]:
    """Emit visual_plan.json and asset_inventory.json from the compiled figures.

    These are the two contracts ``generate_render_spec.py`` reads to find a
    slide's visual, so writing them is what makes a compiled figure reach the
    deck. The shape matches what the model-drawn path emits; the fields that
    only make sense for generated CODE (prompt/response hashes, model route)
    are replaced by the compiler's own provenance: the spec hash, the archetype
    and the manifest.
    """

    workspace_relative = f"decks/{deck_dir.name}"
    slides: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for row in receipt.get("figures") or []:
        asset_path = f"{workspace_relative}/{row['svg_path']}"
        manifest_path = f"{workspace_relative}/{row['manifest_path']}"
        figure = {
            "asset_paths": [asset_path],
            "figure_id": row["figure_id"],
            "archetype": row["archetype"],
            "family": row["family"],
            "grounding_kind": row["grounding_kind"],
            "ledger_rows": row["ledger_rows"],
            "manifest_path": manifest_path,
            "spec_hash": row["spec_hash"],
            "renderer_family": "figure_compiler_svg",
        }
        slides.append({"slide_id": row["slide_id"], "figures": [figure]})
        assets.append({
            "slide_id": row["slide_id"],
            "visual_id": row["figure_id"],
            "path": asset_path,
            "sha256": row["svg_sha256"],
            "renderer_family": "figure_compiler_svg",
            "generation_route": "figure_compiler",
            "archetype": row["archetype"],
            "grounding_kind": row["grounding_kind"],
            "ledger_rows": row["ledger_rows"],
            "manifest_path": manifest_path,
            "spec_hash": row["spec_hash"],
            "layout_attempt": row["layout_attempt"],
        })

    visual_plan = {
        "schema_version": 1,
        "presentation_id": presentation_id or deck_dir.name,
        "slides": slides,
    }
    asset_inventory = {
        "schema_version": 1,
        "presentation_id": presentation_id or deck_dir.name,
        "generator": "scripts/presentations/compile_figures.py",
        "generation_route": "figure_compiler",
        "compiler_version": receipt.get("compiler_version"),
        "status": receipt.get("status"),
        "semantic_review_required": False,
        "generated_assets": assets,
    }
    # The render-spec step also expects the four evidence artifacts the drawsvg
    # route produces. Their compiler-route equivalents are written here and say
    # what is actually true of a compiled figure: no code was authored, the
    # geometry gate passed by construction, and entity coverage comes from the
    # figure manifest rather than from scraping SVG text.
    figure_specs = {
        "schema_version": 1,
        "presentation_id": presentation_id or deck_dir.name,
        "generation_route": "figure_compiler",
        "figures": [
            {
                "figure_id": row["figure_id"],
                "slide_id": row["slide_id"],
                "archetype": row["archetype"],
                "spec_hash": row["spec_hash"],
                "asset_path": f"{workspace_relative}/{row['svg_path']}",
                "sha256": row["svg_sha256"],
            }
            for row in receipt.get("figures") or []
        ],
    }
    model_receipt = {
        "schema_version": 1,
        # The render-spec gate reads `status` here, and the compiler route did
        # not write one: these four review artifacts were designed for the
        # drawsvg route, where a model authored drawing CODE that had to be
        # reviewed. The compiler answers the same questions from the compile
        # itself, so it must answer them in the shape the gate reads, or a deck
        # that compiles cleanly can never be rendered.
        "status": receipt.get("status"),
        "generation_route": "figure_compiler",
        "compiler_version": receipt.get("compiler_version"),
        "model_in_layout_path": False,
        "note": (
            "Layout is deterministic: the model authored declarative specs (see "
            "figure_spec_receipt.json) and never chose a coordinate, colour, font "
            "or marker. The same spec always compiles to the same SVG."
        ),
        "spec_authoring_receipt": "figure_spec_receipt.json",
    }
    code_review = {
        "schema_version": 1,
        "status": receipt.get("status"),
        "generation_route": "figure_compiler",
        "note": "no drawing code was authored, so there is none to review",
        "figures": [
            {"figure_id": row["figure_id"], "layout_attempt": row["layout_attempt"], "status": "pass"}
            for row in receipt.get("figures") or []
        ],
        "findings": [],
    }
    svg_validation = {
        "schema_version": 1,
        "status": receipt.get("status"),
        "generation_route": "figure_compiler",
        "note": (
            "every figure passed the measured geometry gate before it was written; "
            "a figure that could not is a hard error, never a warning"
        ),
        "assets_validated": len(assets),
        "findings": [],
    }
    semantic_review = {
        "schema_version": 1,
        "presentation_id": presentation_id or deck_dir.name,
        "status": receipt.get("status"),
        "generation_route": "figure_compiler",
        "asset_inventory": f"{workspace_relative}/asset_inventory.json",
        "generated_dir": f"{workspace_relative}/{OUTPUT_DIR_NAME}",
        "assets_reviewed": len(assets),
        "reviewed_assets": [row["figure_id"] for row in receipt.get("figures") or []],
        "note": (
            "entity coverage is read from each figure's manifest, which records the "
            "text runs actually rendered, instead of being scraped back out of the SVG"
        ),
        # The gate reads `semantic_review` as an object and asks it one
        # question. The per-figure rows it used to hold moved to `figures`,
        # where they are still the evidence for the answer.
        "semantic_review": {
            "render_spec_generation_allowed": not any(
                row["entities_missing"] for row in receipt.get("figures") or []
            ),
            "figures_reviewed": len(receipt.get("figures") or []),
            "figures_with_missing_entities": sum(
                1 for row in receipt.get("figures") or [] if row["entities_missing"]
            ),
        },
        "figures": [
            {
                "figure_id": row["figure_id"],
                "slide_id": row["slide_id"],
                "archetype": row["archetype"],
                "entities_missing": row["entities_missing"],
                "status": "pass" if not row["entities_missing"] else "fail",
            }
            for row in receipt.get("figures") or []
        ],
        "findings": [
            f"{row['figure_id']}: required entities absent from the figure: "
            + ", ".join(row["entities_missing"])
            for row in receipt.get("figures") or []
            if row["entities_missing"]
        ],
    }

    written: dict[str, str] = {}
    for name, payload in (
        ("visual_plan.json", visual_plan),
        ("asset_inventory.json", asset_inventory),
        ("figure_specs.json", figure_specs),
        ("figure_generation_model_receipt.json", model_receipt),
        ("figure_code_review_report.json", code_review),
        ("figure_svg_validation_report.json", svg_validation),
        ("figure_semantic_review.json", semantic_review),
    ):
        path = deck_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name.replace(".json", "")] = str(path)
    return written


def build_gallery(out_dir: Path) -> dict[str, Any]:
    """Compile the worked example for every archetype (the visual regression set)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, payload in EXAMPLES.items():
        try:
            figure = compile_figure(payload)
        except FigureCompilerError as error:
            failures.append({"archetype": name, "detail": str(error),
                             "findings": list(getattr(error, "findings", []))})
            continue
        figure.write(out_dir / f"{name}.svg", out_dir / f"{name}.manifest.json")
        rows.append({"archetype": name, "family": figure.spec.family,
                     "layout_attempt": figure.attempt, "svg_sha256": figure.svg_sha256})
    return {"compiler_version": COMPILER_VERSION, "archetypes": len(EXAMPLES),
            "compiled": rows, "failures": failures,
            "status": "pass" if not failures else "blocked"}


def build_deck_ledger(deck_dir: Path, *, topic_id: str = "") -> dict[str, Any]:
    """Extract the deck's numeric-claims ledger from its bound sources."""

    source_ledger_path = deck_dir / "speaker_author_work_ledger.json"
    source_ledger = _load_json(source_ledger_path) if source_ledger_path.is_file() else {}
    ledger = build_ledger(topic_id=topic_id or deck_dir.name, source_ledger=source_ledger)
    ledger.write(deck_dir / LEDGER_NAME)
    return ledger.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, help="deck directory holding figure_specs/")
    parser.add_argument("--spec", type=Path, help="compile a single spec file")
    parser.add_argument("--out", type=Path, help="output SVG path for --spec")
    parser.add_argument("--spec-dir", type=Path, help="override the spec directory")
    parser.add_argument("--output-dir", type=Path, help="override the SVG output directory")
    parser.add_argument("--gallery", type=Path, help="compile one example per archetype into this directory")
    parser.add_argument("--build-ledger", action="store_true",
                        help="extract the deck's numeric-claims ledger before compiling")
    parser.add_argument("--allow-variety-findings", action="store_true",
                        help="record the deck's figure mix without blocking on the quotas")
    parser.add_argument("--write-render-contracts", action="store_true",
                        help="also write visual_plan.json and asset_inventory.json for the renderer")
    parser.add_argument("--allow-placeholder-specs", action="store_true",
                        help="compile --dry-run placeholder specs (wiring proof only, never a real deck)")
    parser.add_argument("--no-cache", action="store_true",
                        help="recompile every figure even when its spec hash is unchanged")
    parser.add_argument("--vision-deck", action="store_true",
                        help="apply the visual-domain quota for a computer-vision deck")
    parser.add_argument("--receipt", type=Path, help="where to write the receipt (default: deck dir)")
    parser.add_argument("--json", action="store_true", help="print the receipt as JSON")
    args = parser.parse_args(argv)

    if args.gallery:
        receipt = build_gallery(args.gallery)
        print(json.dumps(receipt, indent=2) if args.json
              else f"gallery: {len(receipt['compiled'])}/{receipt['archetypes']} archetypes -> {args.gallery}")
        return 0 if receipt["status"] == "pass" else 1

    if args.spec:
        payload = _load_json(args.spec)
        try:
            figure = compile_figure(payload)
        except FigureCompilerError as error:
            print(f"figure compile failed: {error}", file=sys.stderr)
            for finding in getattr(error, "findings", []):
                print(f"  {finding}", file=sys.stderr)
            return 1
        out = args.out or args.spec.with_suffix(".svg")
        figure.write(out, out.with_suffix(".manifest.json"))
        print(f"compiled {figure.spec.archetype} -> {out}")
        return 0

    if not args.deck_dir:
        parser.error("one of --deck-dir, --spec or --gallery is required")

    if args.build_ledger:
        ledger = build_deck_ledger(args.deck_dir)
        print(f"numeric-claims ledger: {ledger['claim_count']} claims from {ledger['sources_scanned']} sources")

    receipt = compile_deck(
        args.deck_dir, spec_dir=args.spec_dir, output_dir=args.output_dir,
        vision_deck=args.vision_deck, allow_variety_findings=args.allow_variety_findings,
        use_cache=not args.no_cache, allow_placeholders=args.allow_placeholder_specs,
    )
    if args.write_render_contracts and receipt["status"] == "pass":
        receipt["render_contracts"] = write_render_contracts(args.deck_dir, receipt)
    receipt_path = args.receipt or (args.deck_dir / RECEIPT_NAME)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, indent=2))
    else:
        print(f"{receipt['compiled_count']}/{receipt['spec_count']} figures compiled "
              f"({receipt['status']}); receipt {receipt_path}")
        for blocker in receipt["blockers"][:6]:
            print(f"  blocker {blocker['code']}: {blocker.get('figure_id', '')} {blocker.get('detail', '')[:100]}")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
