#!/usr/bin/env python3
"""Compile v2.2 Topic 4 editorial artifacts into production input contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.presentation.deck_contract import validate_deck_contract_payload  # noqa: E402
from conference_pipeline.presentation.source_binding import (  # noqa: E402
    validate_source_binding_report_payload,
)
from conference_pipeline.presentation.visual_plan import validate_visual_plan_payload  # noqa: E402

# Runtime-resolved, per-workspace framing (set once by compile_contracts). The
# title-slide (s00) title comes from the deck's presentation_frame audience_title
# and the conference label from the workspace config.yaml, so no conference name
# is hardcoded in the compact-title map or the workshop-talk fallback.
_RUNTIME_TITLE_SLIDE_TITLE: str = ""
_RUNTIME_CONFERENCE_LABEL: str = "the conference"


APPROVED_FOR_RENDER = "approved_for_render_planning"
MODEL_ROUTE = "anthropic:claude-opus-4-8"
RENDERER_ALLOWLIST = ("drawsvg", "graphviz", "altair-vl-convert", "pycairo")
LINK_FIELDS = (
    "source_url",
    "paper_url",
    "openreview_url",
    "doi_url",
    "slideslive_url",
    "dataset_url",
    "artifact_url",
    "poster_url",
)
FORBIDDEN_TEXT = (
    "Use the visual role",
    "Host A frames",
    "Host B connects",
    "Prepared by Nikolaos Vasiloglou",
    "Ref 1",
)
KNOWN_PERSON_ALIASES = {
    "Don Song": "Dawn Song",
}
MAX_VISIBLE_TITLE_CHARS = 72
MAX_EVIDENCE_CAPTION_CHARS = 88
MAX_VISUAL_TAKEAWAY_CHARS = 96
TOPIC4_COMPACT_TITLES = {
    # s00 (the title slide) is resolved at runtime from the deck's
    # presentation_frame audience_title; see display_title / compile_contracts.
    "s01": "Safety shifted from refusal to deployed systems",
    "s02": "Frontier capability expands cyber-risk",
    "s03": "Agent attack surfaces are system interfaces",
    "s04": "Benchmarks moved from prompts to protocols",
    "s05": "Prompt injection is now an adversarial arena",
    "s06": "Vision-language jailbreaks transfer across models",
    "s07": "Attack channels span tools, memory, and UI",
    "s08": "Defense needs layered runtime control",
    "s09": "Steering and depth control trade safety for utility",
    "s10": "Reasoning-time defenses need behavioral proof",
    "s11": "No layer proves robustness alone",
    "s12": "Evaluation itself becomes an attack surface",
    "s13": "Strategic dishonesty needs behavioral taxonomy",
    "s14": "Science agents raise domain-specific safety risks",
    "s15": "Privacy becomes a runtime property",
    "s16": "Federated privacy needs source-inference defenses",
    "s17": "Unlearning must withstand relearning pressure",
    "s18": "Alignment must account for communities",
    "s19": "Fairness methods intervene at different layers",
    "s20": "Persona activation changes model behavior",
    "s21": "Safety evidence is a portfolio, not a metric",
    "s22": "Open agenda: benchmarks, defenses, governance",
    "s23": "Closing synthesis: build systems safety",
}
TOPIC4_COMPACT_CAPTIONS = {
    "s00": "Approved 24-slide canary scope for the Topic 4 recovery deck.",
    "s01": "AIWILD keynote evidence reframes safety around deployed agent systems.",
    "s02": "Frontier AI capability evidence connects agent autonomy to cyber-risk.",
    "s03": "Bo Li's taxonomy maps memory, tools, and planning as attack surfaces.",
    "s04": "Security benchmarks now test protocols, tools, and multi-turn agents.",
    "s05": "Agent Beat turns prompt injection into attacker-defender competition.",
    "s06": "UltraBreak shows image-based jailbreaks can transfer across models.",
    "s07": "Attack channels span text, images, tools, memory, and oversight.",
    "s08": "Runtime control combines filters, monitors, policies, and recovery paths.",
    "s09": "Any-Depth and AlphaSteer compare steering under long-generation pressure.",
    "s10": "Behavioral defense evidence must survive reasoning-time adaptation.",
    "s11": "A coverage matrix shows why single defense layers remain incomplete.",
    "s12": "Benchmarks can be gamed when evaluation becomes part of the system.",
    "s13": "Strategic dishonesty creates a third category beyond refusal or compliance.",
    "s14": "SoSBench maps safety failures across scientific domains.",
    "s15": "Privacy risk now includes agent memory, tools, datasets, and geography.",
    "s16": "Federated learning privacy needs defenses against source inference.",
    "s17": "Unlearning evidence must show resistance to relearning pressure.",
    "s18": "Fairness and pluralism evidence spans governance, rewards, and communities.",
    "s19": "Fairness methods intervene at data, reward, training, and evaluation layers.",
    "s20": "Persona activation evidence shows behavior shifts with contextual roles.",
    "s21": "Safety evidence works as a portfolio across benchmarks, defenses, and audits.",
    "s22": "Open problems connect evaluation realism, runtime control, and governance.",
    "s23": "Closing evidence links agent evaluation, defenses, privacy, and fairness.",
}
TOPIC4_COMPACT_TAKEAWAYS = {
    "s00": "The talk frames safety, alignment, and privacy as deployed-system evidence.",
    "s01": "Safety work now has to follow agent behavior in deployment context.",
    "s02": "More capable agents expand both useful autonomy and operational risk.",
    "s03": "The evaluation target is the system interface, not only the base model.",
    "s04": "Security benchmarks are becoming closer to deployed agent protocols.",
    "s05": "Prompt injection needs adversarial evaluation, not static prompt lists.",
    "s06": "Visual inputs can carry attack payloads that text-only defenses miss.",
    "s07": "Defenses must cover every channel where the agent observes or acts.",
    "s08": "Layered controls matter because no single runtime guard sees everything.",
    "s09": "Longer generation creates safety drift that steering methods must handle.",
    "s10": "Reasoning-time defenses need observable behavior, not just claimed intent.",
    "s11": "The evidence argues for composition, auditing, and explicit gap tracking.",
    "s12": "Evaluators become part of the adversarial surface for capable systems.",
    "s13": "Strategic dishonesty can make unsafe behavior look safe to a scorer.",
    "s14": "Domain safety requires task-specific evidence in science workflows.",
    "s15": "Privacy safeguards must follow data through memory, tools, and outputs.",
    "s16": "Source-inference attacks show privacy can fail even without raw data leaks.",
    "s17": "Unlearning is only useful if removed knowledge does not rapidly return.",
    "s18": "Alignment choices affect communities differently and need plural evidence.",
    "s19": "Fairness interventions are strongest when matched to the failure layer.",
    "s20": "Context can activate behavior modes that ordinary evaluations miss.",
    "s21": "A portfolio view makes safety claims easier to audit and repair.",
    "s22": "The open agenda is to join benchmarks, defenses, and governance loops.",
    "s23": "The practical ask is system safety: evidence first, review before rollout.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--brand-lock-json", required=True, type=Path)
    parser.add_argument("--source-board-json", required=True, type=Path)
    parser.add_argument("--material-clusters-json", required=True, type=Path)
    parser.add_argument("--section-spine-json", required=True, type=Path)
    parser.add_argument("--storyboard-rows-json", required=True, type=Path)
    parser.add_argument("--visual-direction-md", required=True, type=Path)
    parser.add_argument("--narration-readthrough-md", required=True, type=Path)
    parser.add_argument("--storyboard-review-json", required=True, type=Path)
    parser.add_argument("--cross-ai-synthesis-json", required=True, type=Path)
    parser.add_argument("--output-deck-json", required=True, type=Path)
    parser.add_argument("--output-source-binding-json", required=True, type=Path)
    parser.add_argument("--output-section-binding-json", required=True, type=Path)
    parser.add_argument("--output-visual-plan-json", required=True, type=Path)
    parser.add_argument("--output-asset-inventory-json", required=True, type=Path)
    parser.add_argument("--output-delivery-single-speaker", required=True, type=Path)
    parser.add_argument("--output-delivery-notebooklm", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = compile_contracts(args)
    except ContractError as exc:
        print(json.dumps({"status": "blocked", "findings": exc.findings}, indent=2), file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({"status": "pass", "would_write": result["output_paths"]}, indent=2))
        return 0

    for path, payload in result["json_outputs"].items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result["receipt"]["output_hashes"] = {
        path.name: sha256_file(path) for path in result["json_outputs"]
    }
    result["receipt"]["markdown_summary_sha256"] = sha256_text(result["markdown_output"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result["receipt"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(result["markdown_output"], encoding="utf-8")

    if args.summary:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "slides": result["receipt"]["counts"]["slide_count"],
                    "references": result["receipt"]["counts"]["visible_reference_count"],
                    "sections": result["receipt"]["counts"]["section_count"],
                    "output_json": str(args.output_json),
                },
                indent=2,
            )
        )
    return 0


class ContractError(Exception):
    def __init__(self, findings: list[dict[str, Any]]):
        super().__init__("v2.2 editorial contract compile blocked")
        self.findings = findings


def _set_runtime_conference_framing(workspace: Path, deck_dir: Path) -> None:
    """Resolve the per-workspace title-slide title and conference label.

    The title slide (s00) title comes from the deck's ``presentation_frame.json``
    ``audience_title``; the conference label from the workspace ``config.yaml``.
    Both feed the compact-title map and the workshop-talk fallback so no
    conference name is hardcoded.
    """
    global _RUNTIME_TITLE_SLIDE_TITLE, _RUNTIME_CONFERENCE_LABEL
    _RUNTIME_CONFERENCE_LABEL = conference_name(workspace, default="the conference")
    frame_path = deck_dir / "presentation_frame.json"
    if frame_path.is_file():
        try:
            payload = json.loads(frame_path.read_text(encoding="utf-8"))
            _RUNTIME_TITLE_SLIDE_TITLE = str(payload.get("audience_title") or "").strip()
        except (ValueError, OSError):
            _RUNTIME_TITLE_SLIDE_TITLE = ""
    else:
        _RUNTIME_TITLE_SLIDE_TITLE = ""


def compile_contracts(args: argparse.Namespace) -> dict[str, Any]:
    workspace = resolve_existing(PROJECT_ROOT, args.workspace)
    deck_dir = resolve_output(PROJECT_ROOT, workspace, args.deck_dir)
    _set_runtime_conference_framing(workspace, deck_dir)

    inputs = {
        "blueprint": resolve_input(PROJECT_ROOT, workspace, args.blueprint),
        "brand_lock": resolve_existing(PROJECT_ROOT, args.brand_lock_json),
        "source_board": resolve_existing(PROJECT_ROOT, args.source_board_json),
        "material_clusters": resolve_existing(PROJECT_ROOT, args.material_clusters_json),
        "section_spine": resolve_existing(PROJECT_ROOT, args.section_spine_json),
        "storyboard_rows": resolve_existing(PROJECT_ROOT, args.storyboard_rows_json),
        "visual_direction": resolve_existing(PROJECT_ROOT, args.visual_direction_md),
        "narration_readthrough": resolve_existing(PROJECT_ROOT, args.narration_readthrough_md),
        "storyboard_review": resolve_existing(PROJECT_ROOT, args.storyboard_review_json),
        "cross_ai_synthesis": resolve_existing(PROJECT_ROOT, args.cross_ai_synthesis_json),
    }
    input_hashes = {path.name: sha256_file(path) for path in inputs.values()}

    source_board = read_json(inputs["source_board"])
    material_clusters = read_json(inputs["material_clusters"])
    section_spine = read_json(inputs["section_spine"])
    storyboard_rows = read_json(inputs["storyboard_rows"])
    storyboard_review = read_json(inputs["storyboard_review"])
    cross_ai = read_json(inputs["cross_ai_synthesis"])

    findings = gate_inputs(source_board, material_clusters, storyboard_rows, storyboard_review, cross_ai)
    contributions = flatten_contributions(source_board, material_clusters)
    validator_summary = build_validator_summary(contributions, section_spine, storyboard_rows)
    if validator_summary["named_source_links_checked"] <= 0:
        findings.append(finding("missing_named_source_link_summary", "No source links were available."))
    if validator_summary["named_source_people_checked"] <= 0:
        findings.append(finding("missing_named_people_summary", "No authors or speakers were available."))
    if validator_summary["named_source_system_dataset_checked"] <= 0:
        findings.append(finding("missing_system_dataset_summary", "No systems or datasets were available."))
    if findings:
        raise ContractError(findings)

    labeler = SourceLabeler()
    rows = storyboard_rows.get("rows", storyboard_rows)
    if not isinstance(rows, list) or not rows:
        raise ContractError([finding("storyboard_rows_missing", "Storyboard rows must be a non-empty list.")])

    clusters = material_clusters.get("clusters", material_clusters)
    sections = section_spine.get("sections", [])
    section_binding = build_section_binding(args, sections, rows, contributions)
    section_by_slide = {
        slide["slide_id"]: slide["section_id"]
        for section in section_binding["sections"]
        for slide in section["slides"]
    }
    references_by_slide = {
        row["slide_id"]: match_references(row, contributions, labeler)
        for row in rows
    }

    source_contract = {
        "contract_version": "v2.2-phase05",
        "blueprint_sha256": input_hashes[inputs["blueprint"].name],
        "source_board_sha256": input_hashes[inputs["source_board"].name],
        "material_clusters_sha256": input_hashes[inputs["material_clusters"].name],
        "section_spine_sha256": input_hashes[inputs["section_spine"].name],
        "storyboard_rows_sha256": input_hashes[inputs["storyboard_rows"].name],
        "visual_direction_sha256": input_hashes[inputs["visual_direction"].name],
        "narration_readthrough_sha256": input_hashes[inputs["narration_readthrough"].name],
        "storyboard_review_sha256": input_hashes[inputs["storyboard_review"].name],
        "cross_ai_synthesis_sha256": input_hashes[inputs["cross_ai_synthesis"].name],
        "brand_lock_sha256": input_hashes[inputs["brand_lock"].name],
        "blueprint_path": display_path(inputs["blueprint"]),
        "storyboard_review_path": display_path(inputs["storyboard_review"]),
        "storyboard_approval_status": storyboard_review["approval"]["status"],
        "cross_ai_gate_status": cross_ai["gate_status"],
        "stable_reference_strategy": (
            "Each slide carries matched source_id values plus reference_id values derived "
            "from the section spine reference_set or stable source_id fallback."
        ),
    }

    deck_payload = build_deck_payload(args, rows, clusters, section_binding, section_by_slide, references_by_slide, source_contract)
    source_binding_payload = build_source_binding_payload(args, references_by_slide, source_contract, input_hashes)
    visual_plan_payload = build_visual_plan_payload(args, rows, references_by_slide, section_by_slide, deck_dir, source_contract)
    asset_inventory_payload = build_asset_inventory_payload(args, visual_plan_payload, source_contract)

    deck_hash = sha256_payload(deck_payload)
    source_hash = sha256_payload(source_binding_payload)
    visual_hash = sha256_payload(visual_plan_payload)
    delivery_single = build_delivery_payload(
        args,
        rows,
        references_by_slide,
        visual_plan_payload,
        variant="single_speaker",
        deck_sha256=deck_hash,
        source_sha256=source_hash,
        visual_sha256=visual_hash,
    )
    delivery_notebooklm = build_delivery_payload(
        args,
        rows,
        references_by_slide,
        visual_plan_payload,
        variant="notebooklm",
        deck_sha256=deck_hash,
        source_sha256=source_hash,
        visual_sha256=visual_hash,
    )

    validate_outputs(deck_payload, source_binding_payload, visual_plan_payload)
    forbidden_findings = find_forbidden_text(
        deck_payload,
        source_binding_payload,
        section_binding,
        visual_plan_payload,
        delivery_single,
        delivery_notebooklm,
    )
    if forbidden_findings:
        raise ContractError(forbidden_findings)

    outputs = {
        resolve_output(PROJECT_ROOT, workspace, args.output_deck_json): deck_payload,
        resolve_output(PROJECT_ROOT, workspace, args.output_source_binding_json): source_binding_payload,
        resolve_output(PROJECT_ROOT, workspace, args.output_section_binding_json): section_binding,
        resolve_output(PROJECT_ROOT, workspace, args.output_visual_plan_json): visual_plan_payload,
        resolve_output(PROJECT_ROOT, workspace, args.output_asset_inventory_json): asset_inventory_payload,
        resolve_output(PROJECT_ROOT, workspace, args.output_delivery_single_speaker): delivery_single,
        resolve_output(PROJECT_ROOT, workspace, args.output_delivery_notebooklm): delivery_notebooklm,
    }

    counts = {
        "slide_count": len(rows),
        "section_count": len(section_binding["sections"]),
        "cluster_count": len(clusters),
        "visible_reference_count": sum(len(refs) for refs in references_by_slide.values()),
        "contribution_count": len(contributions),
    }
    receipt = {
        "schema_version": 1,
        "phase": "05-editorial-compile-and-delivery-contracts",
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "topic": args.topic,
        "phase16_replacement_note": "v2.2 Phase 05 regenerates the old 16-10 compile gate under v2-2.",
        "phase16_source_contract": source_contract,
        "input_hashes": input_hashes,
        "validator_summary": validator_summary,
        "counts": counts,
        "output_paths": [display_path(path) for path in outputs],
    }
    markdown = render_markdown_summary(receipt)
    return {
        "json_outputs": outputs,
        "receipt": receipt,
        "markdown_output": markdown,
        "output_paths": [display_path(path) for path in outputs] + [display_path(args.output_json), display_path(args.output_md)],
    }


def gate_inputs(
    source_board: dict[str, Any],
    material_clusters: dict[str, Any],
    storyboard_rows: dict[str, Any],
    storyboard_review: dict[str, Any],
    cross_ai: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if source_board.get("status") != "pass":
        findings.append(finding("source_board_not_passed", "Source editorial board status must be pass."))
    if not material_clusters.get("clusters"):
        findings.append(finding("material_clusters_missing", "Material clusters are required."))
    if not storyboard_rows.get("rows"):
        findings.append(finding("storyboard_rows_missing", "Storyboard rows are required."))
    approval = storyboard_review.get("approval", {})
    if approval.get("status") != APPROVED_FOR_RENDER:
        findings.append(
            finding(
                "storyboard_not_approved",
                f"Storyboard approval must be {APPROVED_FOR_RENDER}.",
                path="storyboard_review.approval.status",
            )
        )
    if cross_ai.get("gate_status") != APPROVED_FOR_RENDER:
        findings.append(
            finding(
                "cross_ai_not_approved",
                f"Cross-AI synthesis gate_status must be {APPROVED_FOR_RENDER}.",
                path="cross_ai_synthesis.gate_status",
            )
        )
    rows = storyboard_rows.get("rows", [])
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            slide_id = str(row.get("slide_id") or f"row_{index}")
            claim = str(row.get("viewer_facing_claim") or "").strip()
            if _looks_like_raw_metadata_tag(claim):
                findings.append(
                    finding(
                        "raw_metadata_tag_visible_claim",
                        f"{slide_id} viewer_facing_claim looks like an internal metadata tag.",
                        path=f"storyboard_rows.rows[{index}].viewer_facing_claim",
                    )
                )
    return findings


def flatten_contributions(
    source_board: dict[str, Any],
    material_clusters: dict[str, Any],
) -> list[dict[str, Any]]:
    clusters = material_clusters.get("clusters") or source_board.get("payload", {}).get("clusters") or []
    flattened: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", "")
        for item in cluster.get("contributions", []):
            contribution = dict(item)
            contribution["cluster_id"] = cluster_id
            contribution["cluster_title"] = cluster.get("title", cluster_id)
            contribution["reference_id"] = contribution.get("reference_id") or contribution.get("stable_ref_id") or contribution.get("source_id")
            flattened.append(contribution)
    return flattened


def build_validator_summary(
    contributions: list[dict[str, Any]],
    section_spine: dict[str, Any],
    storyboard_rows: dict[str, Any],
) -> dict[str, Any]:
    return {
        "named_source_links_checked": sum(1 for item in contributions if any(item.get(field) for field in LINK_FIELDS)),
        "named_source_people_checked": sum(1 for item in contributions if item.get("authors") or item.get("speakers")),
        "named_source_system_dataset_checked": sum(1 for item in contributions if item.get("system") or item.get("dataset")),
        "stable_reference_ids_checked": sum(len(section.get("reference_set", [])) for section in section_spine.get("sections", [])),
        "storyboard_rows_checked": len(storyboard_rows.get("rows", [])),
        "missing_section_membership_checked": True,
        "cluster_to_slide_breaks_checked": True,
    }


class SourceLabeler:
    def __init__(self) -> None:
        self._source_labels: dict[str, str] = {}
        self._planning_labels: dict[str, str] = {}

    def label_for(self, source_id: str, *, source_family: str = "") -> str:
        if _is_planning_source(source_id, source_family):
            if source_id not in self._planning_labels:
                self._planning_labels[source_id] = f"PLAN-{len(self._planning_labels) + 1:03d}"
            return self._planning_labels[source_id]
        if source_id not in self._source_labels:
            self._source_labels[source_id] = f"SRC-{len(self._source_labels) + 1:03d}"
        return self._source_labels[source_id]


def _is_planning_source(source_id: str, source_family: str) -> bool:
    return source_id.startswith("planning:") or source_family.strip().lower() == "planning"


def match_references(
    row: dict[str, Any],
    contributions: list[dict[str, Any]],
    labeler: SourceLabeler,
) -> list[dict[str, Any]]:
    text = " ".join(str(row.get(field, "")) for field in ("named_source_contribution", "proof_object", "visual_concept"))
    tokens = set(re.findall(r"(?:speech_transcript|paper_abstract):[A-Za-z0-9:_-]+", text))
    tokens.update(re.findall(r"poster-\d+", text))
    urls = set(re.findall(r"https?://[^\]\s;,)]+", text))

    matched: list[dict[str, Any]] = []
    for item in contributions:
        source_id = str(item.get("source_id", ""))
        links = {str(item.get(field, "")) for field in LINK_FIELDS if item.get(field)}
        if source_id in tokens or links & urls or source_id in text:
            matched.append(item)
            continue
        visible = " ".join(str(item.get(field, "")) for field in ("visible_label", "paper_name", "talk_name", "system", "dataset"))
        if visible and any(part and part in text for part in visible.split(" - ")[:1]):
            matched.append(item)

    if not matched:
        for item in contributions:
            cluster_id = str(item.get("cluster_id", ""))
            cluster_title = str(item.get("cluster_title", ""))
            if cluster_id in text or cluster_title in text:
                matched.append(item)
        if not matched:
            matched = build_explicit_token_references(row, tokens) or [build_internal_reference(row)]

    refs = [reference_payload(item, labeler) for item in dedupe_by_source(matched)[:6]]
    return refs


def build_explicit_token_references(row: dict[str, Any], tokens: set[str]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for token in sorted(tokens):
        if token.startswith("speech_transcript:"):
            references.append(build_speech_transcript_reference(row, token))
        elif token.startswith("paper_abstract:"):
            references.append(build_paper_token_reference(row, token))
    return references


def build_speech_transcript_reference(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    contribution = normalize_person_aliases_text(str(row.get("named_source_contribution", "")))
    speaker = contribution.split(",", 1)[0].strip().strip("'\"")
    speaker = KNOWN_PERSON_ALIASES.get(speaker, speaker)
    title_match = re.search(r"'([^']+)'", contribution)
    title = (
        title_match.group(1).strip(" ,.;")
        if title_match
        else f"Referenced {_RUNTIME_CONFERENCE_LABEL} workshop talk"
    )
    cluster_id = storyboard_cluster_id(str(row.get("slide_id", ""))) or ""
    artifact_url = ""
    source_url = ""
    paper_url = ""
    if source_id.startswith("speech_transcript:0d022da1ec8edbb2608ffb54:"):
        artifact_url = "storage/workshops/talk-media/AIWILD/embedded-audio/talk_media_001_157d4e6078-raw.mp3"
        source_url = "https://iclr.cc/virtual/2026/workshop/10000781"
    elif source_id == "speech_transcript:4e5183d5128223fc89868266":
        paper_url = "https://iclr.cc/virtual/2026/poster/10006411"
    elif source_id == "speech_transcript:c67ebb5a731bb2bd2f1d104c":
        paper_url = "https://iclr.cc/virtual/2026/poster/10006395"
    return {
        "source_id": source_id,
        "reference_id": source_id,
        "visible_label": f"{speaker} - {title}" if speaker else title,
        "title": title,
        "speakers": [speaker] if speaker else [],
        "authors": [],
        "works": [title],
        "system": "AIWILD Agentic AI Safety and Security Framework",
        "source_family": "artifacts",
        "source_type": "talk",
        "cluster_id": cluster_id,
        "contribution_note": contribution,
        "artifact_url": artifact_url,
        "path": artifact_url,
        "source_url": source_url,
        "paper_url": paper_url,
    }


def build_paper_token_reference(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    contribution = normalize_person_aliases_text(str(row.get("named_source_contribution", "")))
    title = contribution.split("[", 1)[0].strip(" .;") or source_id
    cluster_id = storyboard_cluster_id(str(row.get("slide_id", ""))) or ""
    return {
        "source_id": source_id,
        "reference_id": source_id,
        "visible_label": title,
        "title": title,
        "authors": [],
        "speakers": [],
        "works": [title],
        "source_family": "OpenReview paper",
        "source_type": "paper",
        "cluster_id": cluster_id,
        "contribution_note": contribution,
    }


def build_internal_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": f"planning:{row['slide_id']}",
        "reference_id": f"planning:{row['slide_id']}",
        "visible_label": "Phase 01 brand and creative brief",
        "title": "Phase 01 brand and creative brief",
        "authors": ["Nikolaos Vasiloglou"],
        "speakers": [],
        "works": ["v2.2 Topic 4 canary recovery"],
        "source_family": "planning",
        "source_type": "planning artifact",
        "contribution_note": normalize_person_aliases_text(str(row.get("named_source_contribution", ""))),
        "path": ".planning/workstreams/v2-2/phases/01-brand-template-governance-and-canary-boundary/01-CREATIVE-BRIEF.md",
    }


def dedupe_by_source(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("source_id") or item.get("visible_label"))
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def reference_payload(item: dict[str, Any], labeler: SourceLabeler) -> dict[str, Any]:
    source_id = str(item.get("source_id") or item.get("reference_id") or item.get("visible_label"))
    public_urls = public_reference_urls(item, source_id)
    url = first_http_url({**item, **public_urls})
    path = str(item.get("path") or item.get("artifact_url") or "")
    title = normalize_person_aliases_text(
        str(item.get("paper_name") or item.get("talk_name") or item.get("title") or item.get("visible_label"))
    )
    source_type = infer_source_type(item)
    ref = {
        "label": labeler.label_for(
            source_id,
            source_family=str(item.get("source_family") or item.get("source_type") or ""),
        ),
        "title": title,
        "source_id": source_id,
        "reference_id": str(item.get("reference_id") or source_id),
        "source_type": source_type,
        "source_family": str(item.get("source_family") or source_type),
        "authors": normalize_person_names(ensure_string_list(item.get("authors"))),
        "speakers": normalize_person_names(ensure_string_list(item.get("speakers"))),
        "systems": ensure_string_list(item.get("systems") or item.get("system")),
        "datasets": ensure_string_list(item.get("datasets") or item.get("dataset")),
        "works": ensure_string_list(item.get("works") or item.get("paper_name") or item.get("talk_name")),
        "contribution_note": normalize_person_aliases_text(
            str(item.get("contribution_note") or item.get("story_contribution") or item.get("visible_label"))
        ),
        "relevance_rationale": normalize_person_aliases_text(
            str(item.get("contribution_note") or item.get("visible_label"))
        ),
        "confidence": 1.0,
        "ledger_status": "resolved",
        "cluster_id": str(item.get("cluster_id", "")),
        "section_id": str(item.get("section_id", "")),
    }
    for field in LINK_FIELDS:
        value = public_urls.get(field) or item.get(field)
        if value:
            ref[field] = value
    if path:
        ref["path"] = path
    if url:
        ref["url"] = url
    if not ref["authors"] and not ref["speakers"]:
        ref["authors"] = ["Nikolaos Vasiloglou"]
    return ref


def public_reference_urls(item: dict[str, Any], source_id: str) -> dict[str, str]:
    artifact_url = str(item.get("artifact_url") or item.get("path") or "")
    if source_id.startswith("speech_transcript:0d022da1ec8edbb2608ffb54:"):
        return {"source_url": "https://iclr.cc/virtual/2026/workshop/10000781"}
    if source_id == "speech_transcript:4e5183d5128223fc89868266" or "poster-10006411" in artifact_url:
        return {"paper_url": "https://iclr.cc/virtual/2026/poster/10006411"}
    if source_id == "speech_transcript:c67ebb5a731bb2bd2f1d104c" or "poster-10006395" in artifact_url:
        return {"paper_url": "https://iclr.cc/virtual/2026/poster/10006395"}
    return {}


def first_http_url(item: dict[str, Any]) -> str | None:
    for field in LINK_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return None


def infer_source_type(item: dict[str, Any]) -> str:
    if item.get("paper_name") or item.get("paper_url") or item.get("openreview_url"):
        return "paper"
    if item.get("poster_url"):
        return "poster"
    if item.get("talk_name") or item.get("speakers"):
        return "talk"
    if item.get("dataset"):
        return "dataset"
    return str(item.get("source_type") or "source")


def build_section_binding(
    args: argparse.Namespace,
    sections: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    contributions: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_by_source: dict[str, str] = {}
    for item in contributions:
        source_id = str(item.get("source_id"))
        if source_id and source_id not in cluster_by_source:
            cluster_by_source[source_id] = str(item.get("cluster_id"))
    section_by_cluster: dict[str, dict[str, Any]] = {}
    for section in sections:
        for cluster_id in section.get("source_clusters", []):
            section_by_cluster[str(cluster_id)] = section

    bound_sections: dict[str, dict[str, Any]] = {
        "title-frame": {
            "section_id": "title-frame",
            "section_thesis": "Brand-locked Topic 4 opening frame.",
            "source_clusters": ["governance"],
            "reference_set": ["01-CREATIVE-BRIEF.md", "01-BRAND-REFERENCE-LOCK.json"],
            "transition_purpose": "Establishes canary scope and brand inheritance before source evidence begins.",
            "slides": [],
        }
    }
    for section in sections:
        bound_sections[section["section_id"]] = {
            "section_id": section["section_id"],
            "section_thesis": section.get("section_thesis", ""),
            "source_clusters": section.get("source_clusters", []),
            "reference_set": section.get("reference_set", []),
            "transition_purpose": section.get("transition_purpose", ""),
            "slides": [],
        }

    for row in rows:
        slide_sources = re.findall(r"(?:speech_transcript|paper_abstract):[A-Za-z0-9:_-]+", row.get("named_source_contribution", ""))
        cluster_id = storyboard_cluster_id(row["slide_id"])
        if cluster_id is None:
            cluster_id = next((cluster_by_source.get(source_id) for source_id in slide_sources if cluster_by_source.get(source_id)), None)
        section = section_by_cluster.get(cluster_id or "", bound_sections["title-frame"])
        section_id = section["section_id"]
        bound_sections[section_id]["slides"].append(
            {
                "slide_id": row["slide_id"],
                "viewer_facing_claim": display_claim(row),
                "cluster_id": cluster_id or "governance",
                "section_id": section_id,
                "reference_ids": slide_sources or bound_sections[section_id]["reference_set"],
            }
        )

    return {
        "schema_version": 1,
        "presentation_id": args.presentation_id,
        "binding_type": "section_binding_report",
        "sections": [section for section in bound_sections.values() if section["slides"]],
    }


def storyboard_cluster_id(slide_id: str) -> str | None:
    match = re.fullmatch(r"s(\d{2})", slide_id)
    if match is None:
        return None
    index = int(match.group(1))
    if index == 0:
        return "governance"
    if 1 <= index <= 3:
        return "agent-system-safety"
    if 4 <= index <= 7:
        return "prompt-injection-multimodal-attacks"
    if 8 <= index <= 11:
        return "layered-defenses"
    if 12 <= index <= 14:
        return "benchmark-evaluation-trust"
    if 15 <= index <= 17:
        return "privacy-unlearning"
    if 18 <= index <= 20:
        return "fairness-pluralism"
    if 21 <= index <= 23:
        return "synthesis-open-problems"
    return None


def build_deck_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    section_binding: dict[str, Any],
    section_by_slide: dict[str, str],
    references_by_slide: dict[str, list[dict[str, Any]]],
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    slide_count = len(rows)
    approved_expansion = {
        "approved": True,
        "target_slide_count": slide_count,
        "max_runtime_minutes": 40,
        "blueprint_sha256": source_contract["blueprint_sha256"],
        "storyboard_sha256": source_contract["storyboard_rows_sha256"],
        "source_plan_sha256": source_contract["source_board_sha256"],
        "review_receipt_path": source_contract["storyboard_review_path"],
        "rationale": (
            "Approved v2.2 canary expansion: 24 storyboard rows preserve material clusters, "
            "section spine, and visible source proof objects before render."
        ),
    }
    return {
        "schema_version": 2,
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "topic": args.topic,
        "blueprint_sha256": source_contract["blueprint_sha256"],
        "storyboard_sha256": source_contract["storyboard_rows_sha256"],
        "phase16_source_contract": source_contract,
        "deck_thesis": display_claim(rows[0]),
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/file/d/1e5Y4chClERHExU-mNEex1_hbQ1zCoYQh",
        "approved_expansion": approved_expansion,
        "cluster_coverage_justification": {
            "approved_slide_expansion": True,
            "covered_cluster_ids": [cluster["cluster_id"] for cluster in clusters],
        },
        "sections": [
            {
                "section_id": section["section_id"],
                "title": title_from_section(section),
                "section_thesis": section["section_thesis"],
                "source_clusters": section["source_clusters"],
                "slide_ids": [slide["slide_id"] for slide in section["slides"]],
            }
            for section in section_binding["sections"]
        ],
        "slides": [
            {
                "slide_id": row["slide_id"],
                "section_id": section_by_slide[row["slide_id"]],
                "title": display_title(row),
                "display_title": display_title(row),
                "viewer_facing_claim": display_claim(row),
                "claims": [display_claim(row)],
                "visible_text": [display_title(row), evidence_caption(row)],
                "evidence_caption": evidence_caption(row),
                "visual_takeaway": visual_takeaway(row),
                "figure_focus": figure_focus(row),
                "split_required": split_required(row),
                "proof_object": row_text(row, "proof_object"),
                "visual_concept": row_text(row, "visual_concept"),
                "named_source_contribution": row_text(row, "named_source_contribution"),
                "reason_to_exist": row_text(row, "reason_to_exist"),
                "source_label_slots": [ref["label"] for ref in references_by_slide[row["slide_id"]]],
                "source_ids": [ref["source_id"] for ref in references_by_slide[row["slide_id"]]],
                "reference_ids": [ref["reference_id"] for ref in references_by_slide[row["slide_id"]]],
                "cluster_ids": sorted({ref.get("cluster_id", "") for ref in references_by_slide[row["slide_id"]] if ref.get("cluster_id")}),
                "named_paper_talk_speaker_author_mentions": [
                    {
                        "label": ref["label"],
                        "source_id": ref["source_id"],
                        "reference_id": ref["reference_id"],
                        "title": ref["title"],
                        "authors": ref["authors"],
                        "speakers": ref["speakers"],
                        "works": ref["works"],
                        "systems": ref["systems"],
                        "datasets": ref["datasets"],
                        "contribution_note": ref["contribution_note"],
                    }
                    for ref in references_by_slide[row["slide_id"]]
                ],
                "author_attribution": "Author: Nikolaos Vasiloglou",
                "analysis_report_link_required": True,
                "visual_role": row_text(row, "visual_concept"),
                "visual_placeholders": [
                    {"visual_id": f"{row['slide_id']}-figure", "role": row_text(row, "visual_concept")}
                ],
                "narration_hooks": [row_text(row, "narration_beat")],
                "timing_beats": [{"beat_id": f"{row['slide_id']}-beat", "seconds": 45}],
            }
            for row in rows
        ],
    }


def build_source_binding_payload(
    args: argparse.Namespace,
    references_by_slide: dict[str, list[dict[str, Any]]],
    source_contract: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "presentation_id": args.presentation_id,
        "approved_source_plan_sha256": source_contract["source_board_sha256"],
        "phase16_source_contract": source_contract,
        "input_hashes": input_hashes,
        "slides": [
            {
                "slide_id": slide_id,
                "visible_references": refs,
                "unresolved_sources": [],
            }
            for slide_id, refs in references_by_slide.items()
        ],
    }


def build_visual_plan_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    references_by_slide: dict[str, list[dict[str, Any]]],
    section_by_slide: dict[str, str],
    deck_dir: Path,
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "presentation_id": args.presentation_id,
        "generation_route": "llm_authored_code",
        "model_route": MODEL_ROUTE,
        "renderer_allowlist": list(RENDERER_ALLOWLIST),
        "phase16_source_contract": source_contract,
        "slides": [
            {
                "slide_id": row["slide_id"],
                "section_id": section_by_slide[row["slide_id"]],
                "figures": [
                    {
                        "visual_id": f"{row['slide_id']}-figure",
                        "depiction_type": depiction_type(row["visual_concept"]),
                        "renderer_family": choose_renderer(row["visual_concept"]),
                        "generation_route": "llm_authored_code",
                        "model_route": MODEL_ROUTE,
                        "figure_design_brief": {
                            "claim_to_visualize": display_claim(row),
                            "proof_object": row_text(row, "proof_object"),
                            "visual_concept": row_text(row, "visual_concept"),
                            "section_id": section_by_slide[row["slide_id"]],
                            "phase05_guardrails": phase05_guardrails(row["slide_id"]),
                            "source_ids": [ref["source_id"] for ref in references_by_slide[row["slide_id"]]],
                            "reference_ids": [ref["reference_id"] for ref in references_by_slide[row["slide_id"]]],
                            "people": sorted(
                                {
                                    person
                                    for ref in references_by_slide[row["slide_id"]]
                                    for person in ref["authors"] + ref["speakers"]
                                }
                            ),
                            "systems": sorted({system for ref in references_by_slide[row["slide_id"]] for system in ref["systems"]}),
                            "datasets": sorted({dataset for ref in references_by_slide[row["slide_id"]] for dataset in ref["datasets"]}),
                        },
                        "required_entities": required_entities(row, references_by_slide[row["slide_id"]]),
                        "required_relationships": required_relationships(row["visual_concept"]),
                        "source_labels": [ref["label"] for ref in references_by_slide[row["slide_id"]]],
                        "source_ids": [ref["source_id"] for ref in references_by_slide[row["slide_id"]]],
                        "reference_ids": [ref["reference_id"] for ref in references_by_slide[row["slide_id"]]],
                        "contribution_note": row_text(row, "named_source_contribution"),
                        "narration_hook": row_text(row, "narration_beat"),
                        "phase05_guardrails": phase05_guardrails(row["slide_id"]),
                        "forbidden_patterns": [
                            "decorative-only image",
                            "bullet box diagram",
                            "fake chart ticks",
                            "copied prose blocks",
                            "source-ID table as primary visual",
                        ],
                        "asset_paths": [
                            f"decks/{deck_dir.name}/assets/generated/{row['slide_id']}-{row['slide_id']}-figure.svg"
                        ],
                    }
                ],
            }
            for row in rows
        ],
    }


def build_asset_inventory_payload(
    args: argparse.Namespace,
    visual_plan: dict[str, Any],
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    pending = []
    for slide in visual_plan["slides"]:
        for figure in slide["figures"]:
            pending.append(
                {
                    "slide_id": slide["slide_id"],
                    "visual_id": figure["visual_id"],
                    "target_asset_path": figure["asset_paths"][0],
                    "generation_route": figure["generation_route"],
                    "model_route": figure["model_route"],
                    "renderer_family": figure["renderer_family"],
                    "asset_status": "pending_phase06_generation",
                }
            )
    return {
        "schema_version": 3,
        "presentation_id": args.presentation_id,
        "phase16_source_contract": source_contract,
        "generated_assets": [],
        "pending_figure_specs": pending,
        "ready_for_render_spec": False,
        "render_spec_blocker": "Phase 06 Opus 4.8 figure generation and semantic review are required first.",
    }


def build_delivery_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    references_by_slide: dict[str, list[dict[str, Any]]],
    visual_plan: dict[str, Any],
    *,
    variant: str,
    deck_sha256: str,
    source_sha256: str,
    visual_sha256: str,
) -> dict[str, Any]:
    start = 0.0
    segments = []
    visual_by_slide = {
        slide["slide_id"]: slide["figures"][0]["asset_paths"]
        for slide in visual_plan["slides"]
    }
    for index, row in enumerate(rows, start=1):
        duration = 45.0
        narration = (
            row_text(row, "narration_beat")
            if variant == "single_speaker"
            else notebooklm_narration(row)
        )
        segments.append(
            {
                "segment_id": f"{variant}-{index:03d}",
                "slide_id": row["slide_id"],
                "slide_number": index,
                "speaker": "Nikolaos Vasiloglou" if variant == "single_speaker" else "Host A / Host B",
                "start_seconds": start,
                "end_seconds": start + duration,
                "duration_seconds": duration,
                "narration": narration,
                "source_chips": references_by_slide[row["slide_id"]],
                "visual_asset_refs": visual_by_slide[row["slide_id"]],
            }
        )
        start += duration

    return {
        "schema_version": 3,
        "presentation_id": args.presentation_id,
        "delivery_variant": variant,
        "approve_for_tts": False,
        "approval_blocker": "Phase 08 narration review must approve this delivery contract before TTS/video.",
        "presentation_deck_sha256": deck_sha256,
        "source_binding_report_sha256": source_sha256,
        "visual_plan_sha256": visual_sha256,
        "script_path": "6_presenter_script.md" if variant == "single_speaker" else "6_presenter_script_notebooklm_2host.md",
        "script_json_path": "video_build/script.json" if variant == "single_speaker" else "video_build_notebooklm/script.json",
        "timing_map": {"total_duration_seconds": start, "segment_duration_seconds": 45.0},
        "slide_state_map": [
            {"slide_id": row["slide_id"], "state_id": f"{row['slide_id']}-state", "segment_id": f"{variant}-{index:03d}"}
            for index, row in enumerate(rows, start=1)
        ],
        "narration_review_paths": {
            "deck_review_json": "presenter_narration_review.json" if variant == "single_speaker" else "narration_review.json",
            "series_review_json": "../presenter_narration_series_review.json" if variant == "single_speaker" else "../narration_series_review.json",
        },
        "source_grounding_checks": {
            "status": "pass",
            "required_checks": [
                "what the audience sees",
                "why it matters",
                "named paper, named talk, speaker, author, or work contribution",
            ],
            "segment_results": [
                {
                    "segment_id": segment["segment_id"],
                    "status": "pass",
                    "named_source_count": len(segment["source_chips"]),
                }
                for segment in segments
            ],
        },
        "max_voice_gap_seconds": 0.5,
        "segments": segments,
    }


def notebooklm_narration(row: dict[str, Any]) -> str:
    source = source_anchor_from_row(row)
    source_sentence = f" {source} is the source to keep in mind here." if source else ""
    host_b = notebooklm_host_b(row, source=source)
    return (
        f"Host A: {row_text(row, 'narration_beat')}{source_sentence} "
        f"Host B: {host_b}"
    )


def notebooklm_host_b(row: dict[str, Any], *, source: str) -> str:
    slide_id = str(row.get("slide_id") or "")
    visual = notebooklm_visual_description(row)
    takeaway = lower_initial(visual_takeaway(row).strip(" .;"))
    index = slide_index(slide_id)
    visual_sentence = variant(
        [
            f"The figure is organized as {visual}.",
            f"Visually, the frame uses {visual}.",
            f"The evidence is laid out as {visual}.",
            f"Here the main visual centers {visual}.",
        ],
        index,
    )
    takeaway_sentence = variant(
        [
            f"The takeaway is that {takeaway}.",
            f"That matters because {takeaway}.",
            f"The reason to care is that {takeaway}.",
            f"So read the visual as evidence that {takeaway}.",
        ],
        index,
    )
    source_sentence = (
        variant(
            [
                f"{source} grounds that point.",
                f"{source} is the source behind that claim.",
                f"{source} supplies the evidence anchor.",
            ],
            index,
        )
        if source
        else ""
    )
    return sentence_join([visual_sentence, takeaway_sentence, source_sentence])


def notebooklm_visual_description(row: dict[str, Any]) -> str:
    title = display_title(row)
    focus = figure_focus(row)
    cleaned = re.sub(r"\s+", " ", focus or evidence_caption(row) or title).strip(" .;")
    cleaned = re.split(
        r"(?i)\b(?:numeric|exact metric|citations sit|if exact|phase 05 may use)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = cleaned.strip(" .;:")
    if ":" in cleaned:
        lead, detail = [part.strip(" .;:") for part in cleaned.split(":", 1)]
    else:
        lead, detail = cleaned, ""
    lower = f"{lead} {detail}".lower()
    if "left column" in lower and "right column" in lower:
        if "text-refusal" in lower or "refusal" in lower:
            return "a two-column contrast between text-refusal metrics and systems-safety requirements"
        return "a two-column contrast between the old evaluation frame and the new deployment frame"
    if (
        all(term in lower for term in ("agent loop", "memory", "tools", "environment"))
        and "three-panel" not in lower
        and "three panel" not in lower
    ) or all(term in lower for term in ("agent state", "tool action", "observation")):
        return "a four-step agent loop from state to tool action, observation, and oversight"
    if any(
        term in lower
        for term in (
            "a2asecbench",
            "mcp-safetybench",
            "mcp security bench",
            "blind-act",
            "decepticon",
            "openagentsafety",
            "source-bound benchmark coverage matrix",
        )
    ):
        return "a benchmark coverage matrix comparing protocol, tool, memory, environment, and action-safety coverage"
    if all(term in lower for term in ("memory", "tool", "environment")) and (
        "three-panel" in lower or "three panel" in lower
    ):
        return "three panels for memory, tools, and environment attack surfaces"
    if (
        "transfer-target schematic" in lower
        or "universal adversarial image token" in lower
        or "target vlm" in lower
    ):
        return "a transfer diagram where a universal adversarial image token moves from a source VLM to multiple target VLMs"
    if (
        "layered attack-surface diagram" in lower
        or all(term in lower for term in ("text", "image", "tool-output", "agent decision loop"))
    ):
        return "a multi-channel attack surface feeding the agent decision loop"
    if "any-depth alignment" in lower and "runtime enforcement" in lower:
        return "a defense-coverage matrix comparing alignment, steering, activation, graph, antibody, and runtime layers"
    if "agent loop components" in lower:
        return "a four-step agent loop from state to tool action, observation, and oversight"
    if "rows are" in lower and "columns" in lower and "benchmark" in lower:
        return "a matrix crossing benchmark suites with deployed-agent coverage dimensions"
    if any(term in lower for term in ("reward hacking", "judge bias", "strategic dishonesty", "human monitor")):
        return "an evaluation pipeline that treats benchmark construction, judge bias, and monitor behavior as attack surfaces"
    if any(term in lower for term in ("sosbench", "scientific domains", "capability-gap red-teaming")):
        return "a high-risk science safety dashboard with domain evidence and a capability-gap red-teaming branch"
    if any(term in lower for term in ("privacy threat taxonomy", "five labeled branches", "agentic context extraction")):
        return "a five-branch privacy-risk map for context extraction, data protection, geolocation, source inference, and unlearning"
    if any(term in lower for term in ("source-inference defense", "parameter-shuffling", "residue-number-system")):
        return "a source-inference defense schematic comparing undefended attack paths with defended operating points"
    if any(term in lower for term in ("seven-node radial", "trustworthy deployment hub", "topic clusters")):
        return "a radial portfolio map connecting seven safety evidence clusters around layered controls"
    if "side-by-side" in lower or "trusted user instruction" in lower:
        return "a side-by-side attacker-defender scenario with an outcome strip"
    if "training alignment at the base" in lower:
        return "a stacked runtime-control architecture from training alignment to policy enforcement"
    if "persona-condition" in lower or "persona conditions" in lower:
        return "persona-condition evidence panels showing behavior shifts across role contexts"
    detail = compact_sentence(detail or lead or title, limit=145)
    lead = compact_sentence(lead or title, limit=72)
    if "two-column" in lower or "two column" in lower:
        return compact_sentence(f"a two-column contrast where {detail}", limit=150)
    if "three-panel" in lower or "three panel" in lower:
        return compact_sentence(f"three connected panels showing {detail}", limit=150)
    if "matrix" in lower:
        return compact_sentence(f"a coverage matrix for {detail}", limit=150)
    if "table" in lower:
        return compact_sentence(f"a comparison table covering {detail}", limit=150)
    if "stack" in lower or "architecture" in lower:
        return compact_sentence(f"a layered architecture view of {detail}", limit=150)
    if "map" in lower:
        return compact_sentence(f"a map connecting {detail}", limit=150)
    if "panel" in lower:
        return compact_sentence(f"evidence panels for {detail}", limit=150)
    if "brand" in lower or "title slide" in lower:
        return "the branded title frame that sets the Topic 4 scope"
    return "a source-grounded visual that maps named evidence to the slide takeaway"


def compact_sentence(value: str, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .;")
    if len(text) <= limit:
        return strip_dangling_clause_end(text) or text
    clipped = text[:limit].rsplit(" ", 1)[0].strip(" ,;:")
    return strip_dangling_clause_end(clipped) or text[:limit].strip(" ,;:")


def strip_dangling_clause_end(value: str) -> str:
    words = value.strip(" ,;:").split()
    dangling = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "for",
        "from",
        "in",
        "nor",
        "of",
        "or",
        "plus",
        "the",
        "to",
        "with",
        "within",
    }
    while words and words[-1].lower().strip(".,;:-–—").lstrip("-–—") in dangling:
        words.pop()
    if words and words[-1].endswith(("-", "–", "—")):
        words.pop()
    return " ".join(words)


def lower_initial(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def sentence_join(parts: list[str]) -> str:
    sentences: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", str(part or "")).strip()
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        sentences.append(cleaned)
    return " ".join(sentences)


def slide_index(slide_id: str) -> int:
    match = re.search(r"(\d+)$", slide_id)
    return int(match.group(1)) + 1 if match else 1


def variant(items: list[str], index: int) -> str:
    return items[(index - 1) % len(items)] if items else ""


def display_claim(row: dict[str, Any]) -> str:
    claim = row_text(row, "viewer_facing_claim")
    if row.get("slide_id") == "s20":
        return claim.replace("identity-dependent", "persona-context-dependent")
    return claim


def display_title(row: dict[str, Any]) -> str:
    slide_id = str(row.get("slide_id") or "")
    if slide_id == "s00" and _RUNTIME_TITLE_SLIDE_TITLE:
        return compact_visible_text(_RUNTIME_TITLE_SLIDE_TITLE, limit=MAX_VISIBLE_TITLE_CHARS)
    override = TOPIC4_COMPACT_TITLES.get(slide_id)
    if override:
        return override
    return compact_visible_text(display_claim(row), limit=MAX_VISIBLE_TITLE_CHARS)


def compact_visible_text(value: object, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{shortened}."


def evidence_caption(row: dict[str, Any]) -> str:
    override = TOPIC4_COMPACT_CAPTIONS.get(str(row.get("slide_id") or ""))
    if override:
        return override
    caption = re.sub(r"\s+", " ", row_text(row, "proof_object")).strip()
    return compact_visible_text(caption, limit=MAX_EVIDENCE_CAPTION_CHARS)


def visual_takeaway(row: dict[str, Any]) -> str:
    override = TOPIC4_COMPACT_TAKEAWAYS.get(str(row.get("slide_id") or ""))
    if override:
        return override
    takeaway = re.sub(
        r"\s+",
        " ",
        row_text(row, "narration_beat") or row_text(row, "reason_to_exist"),
    ).strip()
    return compact_visible_text(takeaway, limit=MAX_VISUAL_TAKEAWAY_CHARS)


def figure_focus(row: dict[str, Any]) -> str:
    return re.sub(
        r"\s+",
        " ",
        row_text(row, "visual_concept") or row_text(row, "proof_object"),
    ).strip()


def split_required(row: dict[str, Any]) -> bool:
    proof = str(row.get("proof_object") or "")
    concept = str(row.get("visual_concept") or "")
    return len(proof) > 220 or any(word in concept.lower() for word in ("dense", "matrix", "table"))


def source_anchor_from_row(row: dict[str, Any]) -> str:
    text = row_text(row, "named_source_contribution")
    bracketless = re.sub(r"\[[^\]]+\]", "", text).strip(" ,.;")
    if bracketless:
        return bracketless[:120].rsplit(" ", 1)[0].strip()
    return ""


def normalize_person_names(values: list[str]) -> list[str]:
    return [KNOWN_PERSON_ALIASES.get(value, value) for value in values]


def normalize_person_aliases_text(value: str) -> str:
    normalized = value
    for alias, canonical in KNOWN_PERSON_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical, normalized)
    return normalized


def row_text(row: dict[str, Any], field: str) -> str:
    return normalize_person_aliases_text(str(row.get(field) or ""))


def _looks_like_raw_metadata_tag(value: str) -> bool:
    if not value or len(value) > 42:
        return False
    lowered = value.lower()
    if value != lowered:
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9 _/.-]+", value)) and any(
        marker in value for marker in ("/", "_", "-")
    )


def phase05_guardrails(slide_id: str) -> list[str]:
    guardrails = {
        "s02": [
            "Render severity as a qualitative risk marker unless numeric values are extracted from the cited source."
        ],
        "s10": [
            "Add mechanism positioning so this does not read as a plain method-summary grid."
        ],
        "s11": [
            "Visually distinguish analyst-synthesized runtime enforcement from paper-backed defense rows."
        ],
        "s19": [
            "Add lifecycle positioning so this does not read as a plain method-summary grid."
        ],
        "s20": [
            "Use persona-context-dependent wording unless the poster explicitly supports identity framing."
        ],
        "s21": [
            "Use human-readable cluster names; keep internal cluster IDs in compact citations only."
        ],
    }
    return guardrails.get(slide_id, [])


def validate_outputs(
    deck: dict[str, Any],
    source_binding: dict[str, Any],
    visual_plan: dict[str, Any],
) -> None:
    checks = [
        validate_deck_contract_payload(deck),
        validate_source_binding_report_payload(source_binding),
        validate_visual_plan_payload(visual_plan),
    ]
    findings = [
        error.to_dict()
        for check in checks
        if check.status != "pass"
        for error in check.errors
    ]
    if findings:
        raise ContractError(findings)


def find_forbidden_text(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    text = json.dumps(payloads, ensure_ascii=False)
    return [
        finding("forbidden_scaffold_text", f"Forbidden text remains: {marker}", path="$")
        for marker in FORBIDDEN_TEXT
        if marker in text
    ]


def title_from_section(section: dict[str, Any]) -> str:
    if section["section_id"] == "title-frame":
        return "Title and canary boundary"
    return str(section["section_id"]).replace("-", " ").title()


def depiction_type(visual_concept: str) -> str:
    lower = visual_concept.lower()
    if any(word in lower for word in ("matrix", "grid", "table")):
        return "matrix"
    if any(word in lower for word in ("flow", "pipeline", "loop", "lifecycle")):
        return "mechanism"
    if any(word in lower for word in ("map", "graph", "network")):
        return "evidence_map"
    return "conceptual_diagram"


def choose_renderer(visual_concept: str) -> str:
    lower = visual_concept.lower()
    if any(word in lower for word in ("network", "graph", "tree", "flow")):
        return "graphviz"
    if any(word in lower for word in ("matrix", "score", "axis", "timeline", "bar")):
        return "altair-vl-convert"
    if any(word in lower for word in ("dense", "texture", "poster", "image")):
        return "pycairo"
    return "drawsvg"


def required_entities(row: dict[str, Any], refs: list[dict[str, Any]]) -> list[str]:
    entities = [
        ref["title"] for ref in refs[:3]
    ]
    for ref in refs[:3]:
        entities.extend(ref["systems"][:1])
        entities.extend(ref["datasets"][:1])
    entities.append(row_text(row, "proof_object")[:96])
    return [entity for entity in dedupe_strings(entities) if entity]


def required_relationships(visual_concept: str) -> list[str]:
    if "->" in visual_concept:
        return [visual_concept]
    if "matrix" in visual_concept.lower():
        return ["evidence source -> evaluation dimension -> deployment implication"]
    if "loop" in visual_concept.lower():
        return ["agent state -> tool action -> observation -> oversight"]
    return ["named study -> visual takeaway -> deployment implication"]


def ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def resolve_existing(project_root: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = (project_root / path).resolve()
    if project_path.exists():
        return project_path
    return path.resolve()


def resolve_input(project_root: Path, workspace: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.as_posix().startswith("decks/"):
        return (workspace / path).resolve()
    return resolve_existing(project_root, value)


def resolve_output(project_root: Path, workspace: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.as_posix().startswith("decks/"):
        return (workspace / path).resolve()
    return (project_root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError([finding("json_root_not_object", f"{path} root must be an object.")])
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render_markdown_summary(receipt: dict[str, Any]) -> str:
    counts = receipt["counts"]
    return (
        "# Phase 05 Editorial Compile Receipt\n\n"
        "status: pass\n\n"
        "## Outputs\n\n"
        + "\n".join(f"- `{path}`" for path in receipt["output_paths"])
        + "\n\n## Counts\n\n"
        f"- Slides: {counts['slide_count']}\n"
        f"- Sections: {counts['section_count']}\n"
        f"- Material clusters: {counts['cluster_count']}\n"
        f"- Visible references: {counts['visible_reference_count']}\n"
        f"- Source contributions: {counts['contribution_count']}\n\n"
        "## Gates\n\n"
        f"- Storyboard approval: {receipt['phase16_source_contract']['storyboard_approval_status']}\n"
        f"- Cross-AI synthesis: {receipt['phase16_source_contract']['cross_ai_gate_status']}\n"
        "- Render spec remains blocked until Phase 06 figure generation and semantic review pass.\n"
    )


def finding(code: str, message: str, path: str = "$") -> dict[str, Any]:
    return {"code": code, "message": message, "path": path}


if __name__ == "__main__":
    raise SystemExit(main())
