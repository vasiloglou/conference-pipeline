#!/usr/bin/env python3
"""Deterministic editorial gate for presentation decks before TTS/video.

The gate catches production blockers that are cheap to detect before paid TTS:
duration budget overruns, excessive zoom/detail beats, obvious redundant slide
claims, generic/repetitive figure patterns, and important figures that the
narration never names.

``--narration-only`` runs the gate on decks produced by the compiled
(``presentation_deck.json`` / ``render_spec.json``) route, which has no legacy
slide markdown and no ``editorial_storyboard.json``: only the narration rules
(duration budget, zoom density, figure entities named in narration) apply, and
the slide count comes from ``--render-spec``. ``--advisory`` records the
verdict without failing the process, for a chain that wants the report on the
record before paying for TTS but leaves the budget call to the operator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SLIDE_RE = re.compile(
    r"(?ms)^## Slide (?P<num>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^## Slide \d+:|\Z)"
)
SCRIPT_HEADER_RE = re.compile(
    r"(?m)^### \[(?P<kind>SLIDE|ZOOM)\s+(?P<num>\d+)\]\s*(?P<title>.+)$"
)
NOTEBOOK_TURN_RE = re.compile(r"(?m)^-\s+\*\*[^*]+:\*\*\s*(?P<text>.+)$")
MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)")
REFERENCE_LINE_RE = re.compile(r"(?im)^-\s+References?:\s*(?P<refs>.+)$")
MARKDOWN_LINK_LABEL_RE = re.compile(r"\[([^\]]+)\]\(")
WORD_RE = re.compile(r"\b[\w'’-]+\b")

DEFAULT_MAX_DURATION_SECONDS = 40 * 60
DEFAULT_TARGET_DURATION_SECONDS = 38 * 60
BLUEPRINT_DURATION_GATE_OWNER = "Phase 14"
FINAL_RENDER_DURATION_GATE_OWNER = "Phase 13 final rendered"
DEFAULT_TOPIC_MAX_ZOOM_RATIO = 0.35
DEFAULT_TOPIC_MAX_ZOOM_COUNT = 16
DEFAULT_TOPIC_MAX_SLIDES = 45
SINGLE_PRESENTER_WPM = 150
NOTEBOOKLM_WPM = 165

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}
FIGURE_ROLE_TERMS = {
    "architecture",
    "benchmark",
    "comparison",
    "evidence",
    "experiment",
    "failure",
    "map",
    "mechanism",
    "molecule",
    "structure",
    "system",
    "workflow",
}
GENERIC_FIGURE_TERMS = (
    "diagram",
    "showing",
    "flowchart",
    "block diagram",
    "generic",
    "illustrating",
    "visual",
)
GENERIC_ALT_RE = re.compile(r"^generated figure for slide \d+$", re.I)
ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+-]{3,}")
STORYBOARD_REQUIRED_CLUSTER_KEYS = (
    "claim",
    "representative_papers",
    "retained_slides",
    "figure_roles",
    "narration_beats",
)
STORYBOARD_RATIONALE_KEYS = (
    "rationale",
    "rationales",
    "retained_rationale",
    "merged_rationale",
    "dropped_rationale",
    "retained_slide_rationale",
    "merged_slide_rationale",
    "dropped_slide_rationale",
)


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    message: str
    deck_id: str | None = None
    value: Any = None
    threshold: Any = None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalized_terms(text: str) -> list[str]:
    return [
        term
        for term in (match.group(0).lower() for match in WORD_RE.finditer(text))
        if len(term) > 2 and term not in STOPWORDS
    ]


def parse_slides(slides_md: Path) -> list[dict[str, Any]]:
    if not slides_md.exists():
        return []
    text = slides_md.read_text(encoding="utf-8")
    slides: list[dict[str, Any]] = []
    for match in SLIDE_RE.finditer(text):
        body = match.group("body")
        images = [
            {"alt": image.group("alt").strip(), "target": image.group("target").strip()}
            for image in MARKDOWN_IMAGE_RE.finditer(body)
        ]
        refs: list[str] = []
        ref_match = REFERENCE_LINE_RE.search(body)
        if ref_match:
            refs = [normalize_space(label) for label in MARKDOWN_LINK_LABEL_RE.findall(ref_match.group("refs"))]
        bullets = [
            normalize_space(line.lstrip("- ").strip())
            for line in body.splitlines()
            if line.strip().startswith("- ") and "References:" not in line
        ]
        figure_need = ""
        need_match = re.search(r"(?m)^FIGURE_NEED:\s*(.+)$", body)
        if need_match:
            figure_need = normalize_space(need_match.group(1))
        slides.append(
            {
                "num": int(match.group("num")),
                "title": normalize_space(match.group("title")),
                "body": body,
                "bullets": bullets,
                "images": images,
                "refs": refs,
                "figure_need": figure_need,
            }
        )
    return slides


def _named_block(block: str, name: str) -> str:
    pattern = re.compile(
        rf"(?ms)^\*\*{re.escape(name)}:?\*\*\s*(.*?)(?=^\*\*[A-Z][A-Z ]*:?\*\*|^---|\Z)"
    )
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def parse_presenter_script(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"main": {}, "zoom": {}, "zoom_count": 0, "word_count": 0, "text": ""}
    text = path.read_text(encoding="utf-8")
    headers = list(SCRIPT_HEADER_RE.finditer(text))
    main: dict[int, str] = defaultdict(str)
    zoom: dict[int, str] = defaultdict(str)
    all_spoken: list[str] = []
    for index, match in enumerate(headers):
        start = match.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[start:end]
        kind = match.group("kind")
        raw_num = int(match.group("num"))
        title = match.group("title")
        is_zoom = kind == "ZOOM" or raw_num >= 100000 or "figure detail" in title.lower()
        num = raw_num - 100000 if raw_num >= 100000 else raw_num
        if "DIALOGUE:" in block:
            spoken = " ".join(turn.group("text") for turn in NOTEBOOK_TURN_RE.finditer(block))
        else:
            spoken = _named_block(block, "SAY")
            if not spoken and kind == "SLIDE":
                spoken = block
        spoken = normalize_space(re.sub(r"^[>\-\s]+", "", spoken))
        if not spoken:
            continue
        all_spoken.append(spoken)
        if is_zoom:
            zoom[num] += " " + spoken
        else:
            main[num] += " " + spoken
    combined = normalize_space(" ".join(all_spoken))
    return {
        "main": {key: normalize_space(value) for key, value in main.items()},
        "zoom": {key: normalize_space(value) for key, value in zoom.items()},
        "zoom_count": len(zoom),
        "word_count": len(words(combined)),
        "text": combined,
    }


def _duration_from_json(path: Path | None) -> float | None:
    if not path or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates: list[float] = []
    if isinstance(payload, dict):
        for key in ("duration_seconds", "total_duration_seconds", "audio_duration_seconds"):
            value = payload.get(key)
            if isinstance(value, int | float) and value > 0:
                candidates.append(float(value))
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("duration_seconds", "total_duration_seconds", "audio_duration_seconds"):
                value = metadata.get(key)
                if isinstance(value, int | float) and value > 0:
                    candidates.append(float(value))
        segments = payload.get("segments")
        if isinstance(segments, list):
            segment_total = 0.0
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                value = segment.get("duration_seconds") or segment.get("duration_s")
                if isinstance(value, int | float) and value > 0:
                    segment_total += float(value)
            if segment_total > 0:
                candidates.append(segment_total)
    return max(candidates) if candidates else None


def _render_spec_slide_count(path: Path | None) -> int:
    """Slide count of a compiled deck's render spec (0 when absent/unreadable)."""

    if not path or not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    slides = payload.get("slides") if isinstance(payload, dict) else None
    return len(slides) if isinstance(slides, list) else 0


def _estimated_duration_seconds(script: dict[str, Any], script_json: Path | None, *, wpm: int) -> float:
    from_json = _duration_from_json(script_json)
    if from_json is not None:
        return from_json
    return float(script["word_count"]) / max(1, wpm) * 60.0


def _status(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "failed"
    if any(finding.severity == "warn" for finding in findings):
        return "warning"
    return "passed"


def _slide_signature(slide: dict[str, Any]) -> str:
    source = " ".join(slide["bullets"][:3]) or slide["title"]
    terms = normalized_terms(source)
    return " ".join(terms[:14])


def _figure_spec_rows(asset_inventory: Path | None) -> list[dict[str, Any]]:
    if not asset_inventory or not asset_inventory.exists():
        return []
    try:
        payload = json.loads(asset_inventory.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("figure_specs", [])
    return [row for row in rows if isinstance(row, dict)]


def _non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return value is not None


def _storyboard_payload(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _storyboard_incomplete_clusters(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return [{"cluster_index": None, "missing": ["valid JSON object"]}]
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        return [{"cluster_index": None, "missing": ["non-empty clusters"]}]

    incomplete: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict):
            incomplete.append({"cluster_index": index, "missing": ["cluster object"]})
            continue
        missing = [
            key for key in STORYBOARD_REQUIRED_CLUSTER_KEYS if not _non_empty(cluster.get(key))
        ]
        if not any(_non_empty(cluster.get(key)) for key in STORYBOARD_RATIONALE_KEYS):
            missing.append("retained/merged/dropped rationale")
        if missing:
            incomplete.append({"cluster_index": index, "missing": missing})
    return incomplete


def _entity_tokens(spec: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for value in spec.get("entities", []) or []:
        for token in ENTITY_TOKEN_RE.findall(str(value)):
            normalized = token.lower()
            if normalized not in STOPWORDS:
                tokens.add(normalized)
    return tokens


def analyze_deck(
    *,
    deck_dir: Path,
    slides_md: Path | None = None,
    presenter_script_md: Path | None = None,
    notebooklm_script_md: Path | None = None,
    presenter_script_json: Path | None = None,
    notebooklm_script_json: Path | None = None,
    asset_inventory: Path | None = None,
    storyboard: Path | None = None,
    deck_kind: str = "topic",
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    target_duration_seconds: int = DEFAULT_TARGET_DURATION_SECONDS,
    max_zoom_ratio: float = DEFAULT_TOPIC_MAX_ZOOM_RATIO,
    max_zoom_count: int = DEFAULT_TOPIC_MAX_ZOOM_COUNT,
    max_topic_slides: int = DEFAULT_TOPIC_MAX_SLIDES,
    narration_only: bool = False,
    render_spec: Path | None = None,
) -> dict[str, Any]:
    deck_dir = deck_dir.resolve()
    slides_md = slides_md or deck_dir / "5_draft_slides_with_images.md"
    presenter_script_md = presenter_script_md or deck_dir / "6_presenter_script.md"
    notebooklm_script_md = notebooklm_script_md or deck_dir / "6_presenter_script_notebooklm_2host.md"
    presenter_script_json = presenter_script_json or deck_dir / "video_build" / "script.json"
    notebooklm_script_json = notebooklm_script_json or deck_dir / "video_build_notebooklm" / "script.json"
    asset_inventory = asset_inventory or deck_dir / "asset_inventory.json"
    storyboard = storyboard or deck_dir / "editorial_storyboard.json"

    findings: list[Finding] = []
    slides = [] if narration_only else parse_slides(slides_md)
    presenter = parse_presenter_script(presenter_script_md)
    notebook = parse_presenter_script(notebooklm_script_md)
    presenter_duration = _estimated_duration_seconds(
        presenter, presenter_script_json, wpm=SINGLE_PRESENTER_WPM
    )
    notebook_duration = _estimated_duration_seconds(
        notebook, notebooklm_script_json, wpm=NOTEBOOKLM_WPM
    )
    main_slide_count = len(slides)
    if narration_only:
        main_slide_count = _render_spec_slide_count(render_spec) or len(
            set(presenter["main"]) | set(notebook["main"])
        )
    zoom_count = max(int(presenter["zoom_count"]), int(notebook["zoom_count"]))
    zoom_ratio = zoom_count / main_slide_count if main_slide_count else 0.0
    is_topic = deck_kind == "topic"
    # The storyboard is a legacy-markdown-route artifact; the compiled route
    # has none, so its rules are skipped in narration-only mode.
    storyboard_rules = is_topic and not narration_only
    storyboard_payload = _storyboard_payload(storyboard)
    storyboard_incomplete_clusters = (
        _storyboard_incomplete_clusters(storyboard_payload)
        if storyboard_rules and storyboard.exists()
        else []
    )

    if storyboard_rules and not storyboard.exists():
        findings.append(
            Finding(
                rule="editorial_storyboard_missing",
                severity="fail",
                message="Topic deck is missing the required editorial_storyboard.json compression artifact.",
                deck_id=deck_dir.name,
                value=str(storyboard),
                threshold="present",
            )
        )
    elif storyboard_rules and storyboard_payload is None:
        findings.append(
            Finding(
                rule="editorial_storyboard_invalid_json",
                severity="fail",
                message="Topic deck editorial_storyboard.json must be a valid JSON object.",
                deck_id=deck_dir.name,
                value=str(storyboard),
                threshold="valid JSON object",
            )
        )
    elif storyboard_rules and storyboard_incomplete_clusters:
        findings.append(
            Finding(
                rule="editorial_storyboard_incomplete",
                severity="fail",
                message="Topic deck storyboard clusters must include representative papers, slide rationale, figure roles, and narration beats.",
                deck_id=deck_dir.name,
                value=storyboard_incomplete_clusters[:10],
                threshold={
                    "cluster_keys": STORYBOARD_REQUIRED_CLUSTER_KEYS,
                    "rationale": "retained/merged/dropped rationale",
                },
            )
        )

    for label, duration in (
        ("single_presenter", presenter_duration),
        ("notebooklm", notebook_duration),
    ):
        if duration > max_duration_seconds:
            findings.append(
                Finding(
                    rule=f"{label}_duration_over_budget",
                    severity="fail",
                    message=f"{label} narration estimate exceeds the hard 40-minute budget.",
                    deck_id=deck_dir.name,
                    value=round(duration, 2),
                    threshold=max_duration_seconds,
                )
            )
        elif is_topic and duration > target_duration_seconds:
            findings.append(
                Finding(
                    rule=f"{label}_duration_over_target",
                    severity="warn",
                    message=f"{label} narration estimate is above the 34-38 minute working target.",
                    deck_id=deck_dir.name,
                    value=round(duration, 2),
                    threshold=target_duration_seconds,
                )
            )

    if is_topic and main_slide_count > max_topic_slides:
        findings.append(
            Finding(
                rule="topic_slide_count_over_editorial_budget",
                severity="warn",
                message="Topic deck slide count is high for a <=40 minute talk; compression should justify retained slides.",
                deck_id=deck_dir.name,
                value=main_slide_count,
                threshold=max_topic_slides,
            )
        )

    if is_topic and zoom_count:
        if zoom_ratio > max_zoom_ratio or zoom_count > max_zoom_count:
            findings.append(
                Finding(
                    rule="zoom_detail_density_over_budget",
                    severity="fail",
                    message="Too many separate figure-detail/zoom narration beats; most visuals should be integrated into parent slide narration.",
                    deck_id=deck_dir.name,
                    value={"zoom_count": zoom_count, "zoom_ratio": round(zoom_ratio, 4)},
                    threshold={"max_zoom_count": max_zoom_count, "max_zoom_ratio": max_zoom_ratio},
                )
            )

    signature_counts = Counter(
        signature for signature in (_slide_signature(slide) for slide in slides) if signature
    )
    duplicate_signatures = {
        signature: count for signature, count in signature_counts.items() if count >= 3
    }
    if duplicate_signatures:
        findings.append(
            Finding(
                rule="redundant_slide_claim_clusters",
                severity="fail",
                message="Three or more slides share the same normalized claim signature; merge similar papers/slides.",
                deck_id=deck_dir.name,
                value=duplicate_signatures,
                threshold="<3 repeats per claim signature",
            )
        )

    ref_sets = Counter(tuple(sorted(slide["refs"])) for slide in slides if len(slide["refs"]) >= 2)
    repeated_ref_sets = {" | ".join(refs): count for refs, count in ref_sets.items() if count >= 4}
    if repeated_ref_sets:
        findings.append(
            Finding(
                rule="repeated_reference_cluster_without_compression",
                severity="warn",
                message="The same paper set appears across many slides; storyboard should justify keeping them separate.",
                deck_id=deck_dir.name,
                value=repeated_ref_sets,
                threshold="<4 repeated reference-set slides",
            )
        )

    images = [image for slide in slides for image in slide["images"]]
    generic_alts = sum(1 for image in images if GENERIC_ALT_RE.match(image["alt"]))
    figure_needs = [slide["figure_need"] for slide in slides if slide["figure_need"]]
    generic_needs = sum(
        1
        for need in figure_needs
        if any(term in need.lower() for term in GENERIC_FIGURE_TERMS)
        and not any(role in need.lower() for role in FIGURE_ROLE_TERMS)
    )
    if images and generic_alts / len(images) > 0.5:
        findings.append(
            Finding(
                rule="generic_figure_alt_text_repetition",
                severity="fail",
                message="Most figure references use generic generated alt text instead of paper-specific visual descriptions.",
                deck_id=deck_dir.name,
                value={"generic_alts": generic_alts, "image_count": len(images)},
                threshold="<=50%",
            )
        )
    if figure_needs and generic_needs / len(figure_needs) > 0.45:
        findings.append(
            Finding(
                rule="generic_figure_need_repetition",
                severity="fail",
                message="Many FIGURE_NEED entries request generic diagrams instead of paper-specific visual roles.",
                deck_id=deck_dir.name,
                value={"generic_figure_needs": generic_needs, "figure_need_count": len(figure_needs)},
                threshold="<=45%",
            )
        )

    specs = _figure_spec_rows(asset_inventory)
    missing_mentions: list[dict[str, Any]] = []
    combined_by_slide: dict[int, str] = {}
    narration_slide_numbers: list[int] = (
        sorted(
            set(presenter["main"]) | set(presenter["zoom"]) | set(notebook["main"]) | set(notebook["zoom"])
        )
        if narration_only
        else [slide["num"] for slide in slides]
    )
    for num in narration_slide_numbers:
        combined_by_slide[num] = normalize_space(
            " ".join(
                [
                    presenter["main"].get(num, ""),
                    presenter["zoom"].get(num, ""),
                    notebook["main"].get(num, ""),
                    notebook["zoom"].get(num, ""),
                ]
            )
        ).lower()
    for spec in specs:
        slide_number = spec.get("slide_number")
        if not isinstance(slide_number, int):
            continue
        entity_tokens = _entity_tokens(spec)
        if not entity_tokens:
            continue
        narration_text = combined_by_slide.get(slide_number, "")
        if not narration_text:
            missing_mentions.append(
                {"slide_number": slide_number, "entities": sorted(entity_tokens)[:5]}
            )
            continue
        if not any(token in narration_text for token in entity_tokens):
            missing_mentions.append(
                {"slide_number": slide_number, "entities": sorted(entity_tokens)[:5]}
            )
    if missing_mentions:
        findings.append(
            Finding(
                rule="central_figure_entities_missing_from_narration",
                severity="fail",
                message="Important figure entities from asset_inventory are not named in the slide narration.",
                deck_id=deck_dir.name,
                value=missing_mentions[:10],
                threshold=0,
            )
        )

    metrics = {
        "deck_id": deck_dir.name,
        "deck_kind": deck_kind,
        "narration_only": narration_only,
        "render_spec_path": str(render_spec) if render_spec else None,
        "slide_count": main_slide_count,
        "image_count": len(images),
        "figure_spec_count": len(specs),
        "zoom_count": zoom_count,
        "zoom_ratio": round(zoom_ratio, 4),
        "presenter_word_count": presenter["word_count"],
        "notebooklm_word_count": notebook["word_count"],
        "presenter_estimated_duration_seconds": round(presenter_duration, 2),
        "notebooklm_estimated_duration_seconds": round(notebook_duration, 2),
        "max_duration_seconds": max_duration_seconds,
        "target_duration_seconds": target_duration_seconds,
        "blueprint_duration_gate_owner": BLUEPRINT_DURATION_GATE_OWNER,
        "final_render_duration_gate_owner": FINAL_RENDER_DURATION_GATE_OWNER,
        "storyboard_path": str(storyboard),
        "storyboard_present": storyboard.exists(),
        "storyboard_cluster_count": (
            len(storyboard_payload.get("clusters", []))
            if isinstance(storyboard_payload, dict)
            and isinstance(storyboard_payload.get("clusters"), list)
            else 0
        ),
        "storyboard_incomplete_cluster_count": len(storyboard_incomplete_clusters),
        "duplicate_claim_signature_count": len(duplicate_signatures),
        "generic_figure_alt_count": generic_alts,
        "generic_figure_need_count": generic_needs,
        "central_figure_missing_mention_count": len(missing_mentions),
    }
    findings_payload = [asdict(finding) for finding in findings]
    return {
        "status": _status(findings),
        "metrics": metrics,
        "findings": findings_payload,
        "finding_counts": dict(Counter(finding["severity"] for finding in findings_payload)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--slides-md", type=Path)
    parser.add_argument("--presenter-script-md", type=Path)
    parser.add_argument("--notebooklm-script-md", type=Path)
    parser.add_argument("--presenter-script-json", type=Path)
    parser.add_argument("--notebooklm-script-json", type=Path)
    parser.add_argument("--asset-inventory", type=Path)
    parser.add_argument("--storyboard", type=Path)
    parser.add_argument(
        "--deck-kind",
        default="topic",
        choices=("topic", "executive_short", "executive_long", "executive"),
    )
    parser.add_argument("--max-duration-seconds", type=int, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--target-duration-seconds", type=int, default=DEFAULT_TARGET_DURATION_SECONDS)
    parser.add_argument("--max-zoom-ratio", type=float, default=DEFAULT_TOPIC_MAX_ZOOM_RATIO)
    parser.add_argument("--max-zoom-count", type=int, default=DEFAULT_TOPIC_MAX_ZOOM_COUNT)
    parser.add_argument("--max-topic-slides", type=int, default=DEFAULT_TOPIC_MAX_SLIDES)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--render-spec",
        type=Path,
        help="Compiled deck render spec; supplies the slide count in --narration-only mode.",
    )
    parser.add_argument(
        "--narration-only",
        action="store_true",
        help="Apply only the narration rules (no legacy slide markdown or storyboard required).",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Record the verdict in the report but always exit 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_deck(
        deck_dir=args.deck_dir,
        slides_md=args.slides_md,
        presenter_script_md=args.presenter_script_md,
        notebooklm_script_md=args.notebooklm_script_md,
        presenter_script_json=args.presenter_script_json,
        notebooklm_script_json=args.notebooklm_script_json,
        asset_inventory=args.asset_inventory,
        storyboard=args.storyboard,
        deck_kind=args.deck_kind,
        max_duration_seconds=args.max_duration_seconds,
        target_duration_seconds=args.target_duration_seconds,
        max_zoom_ratio=args.max_zoom_ratio,
        max_zoom_count=args.max_zoom_count,
        max_topic_slides=args.max_topic_slides,
        narration_only=args.narration_only,
        render_spec=args.render_spec,
    )
    report["advisory"] = bool(args.advisory)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(
        f"presentation editorial gate: {report['status']} "
        f"({report['finding_counts'].get('fail', 0)} fail, "
        f"{report['finding_counts'].get('warn', 0)} warn)",
        file=sys.stderr,
    )
    if args.advisory:
        return 0
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
