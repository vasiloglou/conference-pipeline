"""Deterministic coverage-gap fill step."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from conference_pipeline.analysis._digest import sha256_file

# DRIFT-GUARD: this dict must remain byte-identical to
# scripts/fill_coverage_gaps.py::TOPIC_CORRECTIONS per D-11 (scripts not
# modified). A test enforces equality.
TOPIC_CORRECTIONS = {
    "imageomics-discovering-biological-knowledge-from-images-using-ai": "09_genai_in_science",
    "building-an-ai-ecosystem-for-multiscale-biological-discovery": "09_genai_in_science",
    "cure-bench-competition-on-reasoning-models-for-drug-decision-making-in": "09_genai_in_science",
    "qoq-med-building-multimodal-clinical-foundation-models-with-domain-aware-grpo": "09_genai_in_science",
    "2nd-workshop-on-multi-modal-foundation-models-and-large-language-models-for": "09_genai_in_science",
    "fair-universe-handling-uncertainties-and-distribution-shifts-for-precision": "09_genai_in_science",
    "differentiable-learning-of-combinatorial-algorithms-from-theory-to-practice": "02_genai_architectures",
    "recent-developments-in-geometric-machine-learning-foundations-models-and-more": "02_genai_architectures",
    "adjoint-schrödinger-bridge-sampler": "02_genai_architectures",
    "gnnxemplar-exemplars-to-explanations-natural-language-rules-for-global-gnn": "02_genai_architectures",
    "dynamical-decoupling-of-generalization-and-overfitting-in-large-two-layer": "02_genai_architectures",
    "rag4gfm-bridging-knowledge-gaps-in-graph-foundation-models-through-graph": "02_genai_architectures",
    "auto-compressing-networks": "17_llm_scaling_laws_efficiency",
    "in-search-of-adam-s-secret-sauce": "17_llm_scaling_laws_efficiency",
    "real-time-hyper-personalized-generative-ai-should-be-regulated-to-prevent-the": "11_ethical_problems",
    "toward-general-full-autonomy-open-research-challenges-in-scalable-self-driving": "13_industry_insights",
    "ai-assistants-in-the-wild-agents-adaptation-and-memory-augmented-deployment": "18_agentic_ai_autonomous_systems",
    "the-2025-google-code-golf-championship": "08_language_models",
    "multimodal-data-foundation-at-industry-scale": "12_commercial_applications",
    "telling-stories-at-scale-multimodal-ml-in-the-global-media-landscape": "12_commercial_applications",
}


@dataclass(frozen=True)
class GapFillInputs:
    mapping_path: Path
    analysis_dir: Path
    presentations_dir: Path
    topic_corrections: dict[str, str] | None = None
    overwrite_existing: bool = True


@dataclass(frozen=True)
class GapFillOutputs:
    docs_modified: tuple[Path, ...]
    corrected_mapping_path: Path
    digest: str


def _load_mapping(mapping_path: Path) -> list[dict[str, object]]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    mappings = data.get("mappings") if isinstance(data, dict) else None
    return list(mappings) if isinstance(mappings, list) else []


def _extract_snippet(presentation_dir: Path, max_words: int = 50) -> str:
    transcript_file = presentation_dir / "transcripts" / "full_transcript.txt"
    if not transcript_file.exists():
        return ""
    try:
        content = transcript_file.read_text(encoding="utf-8")
    except OSError:
        return ""

    substantive_lines: list[str] = []
    for line in content.splitlines()[5:]:
        stripped = line.strip()
        if len(stripped) > 50 and not any(
            marker in stripped.lower()
            for marker in ("welcome", "thank you", "good morning", "good afternoon", "hello everyone")
        ):
            substantive_lines.append(stripped)
            if len(" ".join(substantive_lines).split()) > max_words:
                break
    if not substantive_lines:
        return ""
    return " ".join(" ".join(substantive_lines).split()[:max_words]) + "..."


def _format_duration(minutes: object) -> str:
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        value = 0
    if value == 0:
        return "N/A"
    if value >= 60:
        return f"{value / 60:.1f} hours"
    return f"{int(value)} min"


def _topic_section(presentations: list[dict[str, object]], presentations_dir: Path) -> str:
    lines = [
        "",
        "---",
        "",
        "## Additional Presentations (Phase 6 Coverage)",
        "",
        "The following presentations were identified as relevant to this topic:",
        "",
    ]
    sorted_presentations = sorted(
        presentations,
        key=lambda item: -float(item.get("duration_minutes") or 0),
    )
    for presentation in sorted_presentations:
        folder = str(presentation.get("folder") or "")
        title = str(presentation.get("title") or folder)
        lines.extend(
            [
                f"### `{folder}`",
                "",
                f"**Title:** {title}",
                f"**Duration:** {_format_duration(presentation.get('duration_minutes'))}",
                "",
            ]
        )
        snippet = _extract_snippet(presentations_dir / folder)
        if snippet:
            lines.extend([f"> {snippet}", ""])
    return "\n".join(lines).rstrip() + "\n\n"


def _update_topic_document(topic_file: Path, new_section: str) -> bool:
    content = topic_file.read_text(encoding="utf-8") if topic_file.exists() else ""
    if "## Additional Presentations (Phase 6 Coverage)" in content:
        return False
    insertion_points = [
        "## Cross-References",
        "## External Resources",
        "## References",
        "---\n\n**Document Version:",
        "---\n\n**Generated:",
    ]
    for marker in insertion_points:
        if marker in content:
            content = content.replace(marker, new_section + marker, 1)
            break
    else:
        content = content.rstrip() + new_section
    topic_file.write_text(content.rstrip() + "\n", encoding="utf-8")
    return True


def step_gap_fill(inputs: GapFillInputs) -> GapFillOutputs:
    inputs.analysis_dir.mkdir(parents=True, exist_ok=True)
    corrected_mapping_path = inputs.analysis_dir / "presentation_topic_mapping_corrected.json"
    if not inputs.overwrite_existing and corrected_mapping_path.exists():
        return GapFillOutputs(
            docs_modified=(),
            corrected_mapping_path=corrected_mapping_path,
            digest=sha256_file(corrected_mapping_path),
        )

    corrections = (
        inputs.topic_corrections if inputs.topic_corrections is not None else dict(TOPIC_CORRECTIONS)
    )
    mappings = _load_mapping(inputs.mapping_path)
    for mapping in mappings:
        folder = str(mapping.get("folder") or "")
        if folder in corrections:
            mapping["topic"] = corrections[folder]

    topic_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for mapping in mappings:
        topic = str(mapping.get("topic") or "")
        if topic:
            topic_groups[topic].append(mapping)

    modified: list[Path] = []
    for topic, presentations in sorted(topic_groups.items()):
        topic_file = inputs.analysis_dir / f"{topic}.md"
        if not topic_file.exists():
            continue
        if _update_topic_document(topic_file, _topic_section(presentations, inputs.presentations_dir)):
            modified.append(topic_file)

    corrected_mapping_path.write_text(
        json.dumps(
            {
                "total_presentations": len(mappings),
                "topic_counts": {topic: len(items) for topic, items in sorted(topic_groups.items())},
                "mappings": mappings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return GapFillOutputs(
        docs_modified=tuple(modified),
        corrected_mapping_path=corrected_mapping_path,
        digest=sha256_file(corrected_mapping_path),
    )
