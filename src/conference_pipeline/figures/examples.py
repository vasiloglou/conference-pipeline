"""One worked spec per archetype.

These are the compiler's test corpus and the spec author's few-shot examples,
so they are written from the real CVPR 2026 canary slides rather than invented:
every archetype below names the slide it would have served. Numbers appear only
where the deck's approved narration beats carry a measured number, and those
specs cite ledger rows.
"""

from __future__ import annotations

from typing import Any

_V = "figure-spec/1"


def _spec(archetype: str, **kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "spec_version": _V,
        "figure_id": f"example-{archetype}",
        "slide_id": kwargs.pop("slide_id", "sl_example"),
        "archetype": archetype,
        "grounding": kwargs.pop("grounding", {"kind": "schematic"}),
    }
    payload.update(kwargs)
    return payload


EXAMPLES: dict[str, dict[str, Any]] = {
    # -- Family A: topology ---------------------------------------------------
    "flow": _spec(
        "flow", slide_id="sl_14_contrastive-training",
        title="Concept-aware contrastive training",
        claim="Hard negatives are mined per concept before the contrastive loss sees the batch.",
        takeaway="Mine hard negatives per concept",
        emphasis={"highlight": ["loss"]},
        required_entities=["C2-CLIP"],
        content={
            "nodes": [
                {"id": "corpus", "label": "Image-text corpus", "kind": "data"},
                {"id": "sample", "label": "Concept-aware batch sampling", "kind": "process",
                 "sublabel": "C2-CLIP"},
                {"id": "encoder", "label": "Vision encoder", "kind": "model"},
                {"id": "loss", "label": "Contrastive loss", "kind": "process"},
                {"id": "emb", "label": "Concept-separated embedding", "kind": "output"},
            ],
            "edges": [
                {"from": "corpus", "to": "sample", "label": "per concept"},
                {"from": "sample", "to": "encoder"},
                {"from": "encoder", "to": "loss", "label": "hard negatives"},
                {"from": "loss", "to": "emb"},
            ],
        },
    ),
    "loop": _spec(
        "loop", slide_id="sl_16_self-play",
        title="VisPlay self-play loop",
        claim="A proposer and a solver improve each other without new human labels.",
        takeaway="Self-play replaces new human labels",
        emphasis={"highlight": ["reward"]},
        content={
            "centre_label": "No new human labels",
            "nodes": [
                {"id": "proposer", "label": "Proposer", "kind": "model", "edge_label": "asks"},
                {"id": "solver", "label": "Solver", "kind": "model", "edge_label": "answers"},
                {"id": "reward", "label": "Reward model", "kind": "process", "edge_label": "scores"},
                {"id": "update", "label": "Policy update", "kind": "process", "edge_label": "updates"},
            ],
        },
    ),
    "nested_architecture": _spec(
        "nested_architecture", slide_id="sl_18_frozen-backbone",
        title="Adapting a frozen vision-language model",
        claim="A small surrogate adapts the encoder while the language model stays frozen.",
        takeaway="Adapt the encoder, freeze the rest",
        content={
            "groups": [
                {"id": "lvlm", "label": "Frozen LVLM", "members": ["vision", "llm"]},
                {"id": "adapt", "label": "Trained surrogate", "members": ["surrogate", "head"]},
            ],
            "nodes": [
                {"id": "vision", "label": "Vision encoder", "kind": "frozen"},
                {"id": "llm", "label": "Language model", "kind": "frozen"},
                {"id": "surrogate", "label": "Surrogate adapter", "kind": "adapted"},
                {"id": "head", "label": "Task head", "kind": "adapted"},
            ],
            "edges": [{"from": "vision", "to": "surrogate", "label": "features"},
                      {"from": "surrogate", "to": "head"}],
        },
    ),
    "layered_stack": _spec(
        "layered_stack", slide_id="sl_19_token-homogenisation",
        title="Visual tokens homogenise with depth",
        claim="Visual token representations converge as they pass through the language model's layers.",
        takeaway="Deeper layers, less visual variety",
        emphasis={"highlight": ["late"]},
        content={
            "rail_label": "token diversity",
            "layers": [
                {"id": "early", "label": "Early layers", "note": "tokens still carry distinct regions"},
                {"id": "mid", "label": "Middle layers", "note": "region identity begins to blur"},
                {"id": "late", "label": "Late layers", "note": "tokens are near-interchangeable"},
            ],
        },
    ),
    "branching_paths": _spec(
        "branching_paths", slide_id="sl_06_illusions",
        title="Recall or perception?",
        claim="On a modified illusion the model can answer from memory instead of from the pixels.",
        takeaway="Modified illusions separate recall from sight",
        emphasis={"highlight": ["perceive"]},
        content={
            "source": {"id": "image", "label": "Modified illusion image", "kind": "input"},
            "branches": [
                {"id": "recall", "label": "Recall path",
                 "steps": [{"id": "memory", "label": "Matches a known illusion", "kind": "process"},
                           {"id": "canned", "label": "Textbook answer", "kind": "output"}]},
                {"id": "perceive", "label": "Perception path",
                 "steps": [{"id": "look", "label": "Reads the actual pixels", "kind": "process"},
                           {"id": "correct", "label": "Answer matches the image", "kind": "output"}]},
            ],
        },
    ),
    "hierarchy_tree": _spec(
        "hierarchy_tree", slide_id="sl_07_ava-bench",
        title="AVA-Bench atomic abilities",
        claim="One aggregate score hides fourteen separately measured visual abilities.",
        takeaway="Aggregate scores hide atomic abilities",
        content={
            "root": {"id": "bench", "label": "AVA-Bench", "kind": "model"},
            "children": [
                {"id": "loc", "label": "Localisation", "kind": "process",
                 "children": [{"id": "count", "label": "Counting"}, {"id": "rel", "label": "Spatial relations"}]},
                {"id": "rec", "label": "Recognition", "kind": "process",
                 "children": [{"id": "ocr", "label": "Text reading"}, {"id": "attr", "label": "Attributes"}]},
                {"id": "reason", "label": "Reasoning", "kind": "process",
                 "children": [{"id": "cmp", "label": "Comparison"}]},
            ],
        },
    ),
    "swimlane_sequence": _spec(
        "swimlane_sequence", slide_id="sl_17_tool-use",
        title="Tool-integrated visual reasoning",
        claim="The model crops and re-queries the image instead of answering from one glance.",
        takeaway="The model looks twice, on purpose",
        content={
            "actors": [
                {"id": "model", "label": "Vision-language model"},
                {"id": "tools", "label": "Crop and zoom tools"},
                {"id": "image", "label": "Source image"},
            ],
            "steps": [
                {"id": "s1", "from": "model", "to": "tools", "label": "request a crop"},
                {"id": "s2", "from": "tools", "to": "image", "label": "read region"},
                {"id": "s3", "from": "image", "to": "model", "label": "zoomed evidence", "kind": "feedback"},
                {"id": "s4", "from": "model", "to": "model", "label": "answer with the region cited"},
            ],
        },
    ),
    "hub_spoke": _spec(
        "hub_spoke", slide_id="sl_01_thesis",
        title="Where multimodal foundation models are going",
        claim="Four threads carry this year's multimodal work.",
        takeaway="Four threads, one shift",
        emphasis={"highlight": ["open"]},
        content={
            "hub": {"id": "thesis", "label": "Multimodal foundation models", "kind": "model"},
            "spokes": [
                {"id": "open", "label": "Open recipes", "kind": "process"},
                {"id": "percept", "label": "Perception failures", "kind": "process"},
                {"id": "reason", "label": "Reasoning depth", "kind": "process"},
                {"id": "effic", "label": "Efficiency", "kind": "process"},
            ],
        },
    ),
    # -- Family B: comparison -------------------------------------------------
    "feature_matrix": _spec(
        "feature_matrix", slide_id="sl_03_open-recipes",
        title="What 'open' actually covers",
        claim="Only one of these releases opens weights, data and the training recipe together.",
        takeaway="Open weights is not an open recipe",
        emphasis={"highlight": ["OpenMMReasoner"]},
        required_entities=["OpenMMReasoner", "Molmo2"],
        content={
            "rows": [{"label": "OpenMMReasoner"}, {"label": "Molmo2"}, {"label": "Closed checkpoint"}],
            "cols": [{"label": "Open weights"}, {"label": "Open data"}, {"label": "Open recipe"},
                     {"label": "Not distilled"}],
            "cells": [
                {"row": "OpenMMReasoner", "col": "Open weights", "value": "yes"},
                {"row": "OpenMMReasoner", "col": "Open data", "value": "yes"},
                {"row": "OpenMMReasoner", "col": "Open recipe", "value": "yes"},
                {"row": "OpenMMReasoner", "col": "Not distilled", "value": "yes"},
                {"row": "Molmo2", "col": "Open weights", "value": "yes"},
                {"row": "Molmo2", "col": "Open data", "value": "yes"},
                {"row": "Molmo2", "col": "Open recipe", "value": "partial"},
                {"row": "Molmo2", "col": "Not distilled", "value": "yes"},
                {"row": "Closed checkpoint", "col": "Open weights", "value": "no"},
                {"row": "Closed checkpoint", "col": "Open data", "value": "no"},
                {"row": "Closed checkpoint", "col": "Open recipe", "value": "no"},
                {"row": "Closed checkpoint", "col": "Not distilled", "value": "no"},
            ],
        },
    ),
    "contrast_pair": _spec(
        "contrast_pair", slide_id="sl_10_temporal",
        title="Per-frame versus joint temporal editing",
        claim="Editing each frame alone produces flicker that joint temporal modelling removes.",
        takeaway="Per-frame edits flicker; joint edits hold",
        emphasis={"highlight": ["joint"]},
        content={
            "left": {"id": "perframe", "title": "Per-frame enhancement",
                     "items": ["Each frame is corrected alone",
                               "Identity drifts between frames",
                               "Flicker appears in motion"]},
            "right": {"id": "joint", "title": "Joint temporal modelling",
                      "items": ["Frames are corrected together",
                                "Identity is shared across time",
                                "Motion stays coherent"]},
        },
    ),
    "quadrant_map": _spec(
        "quadrant_map", slide_id="sl_09_reasoning-depth",
        grounding={"kind": "data", "ledger_rows": ["nc-0901", "nc-0902", "nc-0903", "nc-0904"]},
        title="Width versus depth in visual reasoning",
        claim="Most failures come from too little exploration, not from shallow chains.",
        takeaway="Explore wider before reasoning deeper",
        emphasis={"highlight": ["agentic"]},
        content={
            "x_axis": {"low": "narrow exploration", "high": "wide exploration"},
            "y_axis": {"low": "shallow chain", "high": "deep chain"},
            "quadrant_labels": [
                {"x": "low", "y": "high", "label": "Deep but blinkered"},
                {"x": "high", "y": "high", "label": "Wide exploration wins"},
            ],
            "points": [
                {"id": "single", "label": "Single-pass answer", "x": 0.18, "y": 0.2,
                 "basis": "Single-pass models answer without exploring alternatives."},
                {"id": "cot", "label": "Long chain of thought", "x": 0.28, "y": 0.82,
                 "basis": "Chain-of-thought deepens one line of reasoning without widening search."},
                {"id": "agentic", "label": "Tool-integrated search", "x": 0.8, "y": 0.72,
                 "basis": "Tool-integrated agents explore many candidate regions before answering."},
                {"id": "retrieval", "label": "Retrieval shortcut", "x": 0.62, "y": 0.24,
                 "basis": "Retrieval widens the candidate set but skips the reasoning chain."},
            ],
        },
    ),
    "ranked_list": _spec(
        "ranked_list", slide_id="sl_26_most-referenced",
        title="Most-referenced works in this deck",
        claim="Four works carry most of this deck's evidence.",
        takeaway="Four works carry the evidence",
        content={
            "items": [
                {"id": "hawk", "label": "HAWK visual token pruning", "note": "5 slides"},
                {"id": "c2clip", "label": "C2-CLIP concept separation", "note": "4 slides"},
                {"id": "ava", "label": "AVA-Bench atomic abilities", "note": "3 slides"},
                {"id": "visplay", "label": "VisPlay self-play", "note": "2 slides"},
            ],
        },
    ),
    "decision_tree": _spec(
        "decision_tree", slide_id="sl_13_release-gate",
        title="Which grounding check before release",
        claim="The evidence a deployment needs depends on how sensitive its domain is.",
        takeaway="Sensitive domains need grounded evidence",
        content={
            "root": {"id": "q", "label": "Is the domain safety-sensitive?", "kind": "decision"},
            "children": [
                {"id": "yes", "label": "Yes: require grounded evidence", "kind": "process",
                 "children": [{"id": "audit", "label": "Per-domain audit", "kind": "output"}]},
                {"id": "no", "label": "No: aggregate benchmark suffices", "kind": "process",
                 "children": [{"id": "ship", "label": "Ship with monitoring", "kind": "output"}]},
            ],
        },
    ),
    "venn_overlap": _spec(
        "venn_overlap", slide_id="sl_02_what-a-vlm-is",
        title="What a vision-language model shares",
        claim="A vision-language model is where visual features and language modelling meet.",
        takeaway="Vision and language share one space",
        content={
            "overlap_label": "Shared embedding space",
            "sets": [{"label": "Visual features"}, {"label": "Language modelling"}],
        },
    ),
    # -- Family C: quantitative -----------------------------------------------
    "bar_compare": _spec(
        "bar_compare", slide_id="sl_19_speculative-decoding",
        title="Speedup by efficiency method",
        claim="Speculative decoding gives the largest measured speedup of the three.",
        takeaway="Speculative decoding leads on speed",
        grounding={"kind": "data", "ledger_rows": ["nc-02-0104", "nc-02-0105", "nc-02-0106"]},
        emphasis={"highlight": ["spec"]},
        content={
            "metric": {"label": "End-to-end speedup", "unit": "x"},
            "bars": [
                {"id": "spec", "label": "Speculative decoding", "value": 5.0, "source": "SRC-021"},
                {"id": "quant", "label": "SegQuant quantisation", "value": 1.49, "source": "SRC-018"},
                {"id": "prune", "label": "HAWK token pruning", "value": 1.34, "source": "SRC-014"},
            ],
        },
    ),
    "paired_delta": _spec(
        "paired_delta", slide_id="sl_20_pruning",
        title="Pruning visual tokens without paying for it",
        claim="HAWK prunes most visual tokens while keeping almost all of the accuracy.",
        takeaway="80% fewer tokens, 96% of the accuracy",
        grounding={"kind": "data", "ledger_rows": ["nc-01-0417", "nc-01-0418"]},
        emphasis={"highlight": ["hawk"]},
        required_entities=["HAWK"],
        content={
            "metric": {"label": "Accuracy retained", "unit": "%"},
            "series": [
                {"id": "hawk", "label": "HAWK", "before": 100.0, "after": 96.0,
                 "delta_label": "96.0% at 19.8% tokens", "source": "SRC-014"},
                {"id": "transprune", "label": "TransPrune", "before": 100.0, "after": 97.1,
                 "delta_label": "97.1% at 34% tokens", "source": "SRC-013"},
            ],
        },
    ),
    "stat_callout": _spec(
        "stat_callout", slide_id="sl_21_watermarking",
        title="Watermark recovery under motion",
        claim="The watermark survives motion and paraphrase attacks in most trials.",
        takeaway="Watermarks survive most attacks",
        grounding={"kind": "data", "ledger_rows": ["nc-02-0311"]},
        content={
            "items": [
                {"id": "recovery", "value": 84.96, "unit": "%", "label": "watermark recovery under motion",
                 "comparator": "measured across the attack suite", "source": "SRC-031"},
                {"id": "tokens", "value": 24.4, "unit": "%", "label": "fewer tokens with EVATok",
                 "source": "SRC-029"},
            ],
        },
    ),
    "line_curve": _spec(
        "line_curve", slide_id="sl_05_sampling-steps",
        title="Quality against sampling steps",
        claim="Measured quality saturates well before the maximum step count.",
        takeaway="Quality saturates before the step budget",
        grounding={"kind": "data", "ledger_rows": ["nc-02-0088"]},
        content={
            "x": {"label": "Sampling steps"},
            "y": {"label": "Quality score"},
            "series": [
                {"id": "flow", "label": "Flow matching", "source": "SRC-009",
                 "points": [{"x": 1, "y": 62}, {"x": 2, "y": 74}, {"x": 4, "y": 81},
                            {"x": 8, "y": 84}, {"x": 16, "y": 85}]},
                {"id": "diffusion", "label": "Diffusion baseline", "source": "SRC-008",
                 "points": [{"x": 1, "y": 31}, {"x": 2, "y": 48}, {"x": 4, "y": 66},
                            {"x": 8, "y": 78}, {"x": 16, "y": 84}]},
            ],
        },
    ),
    "scatter_frontier": _spec(
        "scatter_frontier", slide_id="sl_18_speed-quality",
        title="Speed against quality",
        claim="Three methods sit on the speed-quality frontier; the rest are dominated.",
        takeaway="Three methods hold the frontier",
        grounding={"kind": "data", "ledger_rows": ["nc-02-0104", "nc-02-0110"]},
        content={
            "x": {"label": "Speedup", "unit": "x"},
            "y": {"label": "Quality retained", "unit": "%"},
            "frontier": True,
            "points": [
                {"id": "segquant", "label": "SegQuant", "x": 1.49, "y": 98.2, "on_frontier": True},
                {"id": "hawk", "label": "HAWK", "x": 1.34, "y": 96.0, "on_frontier": False},
                {"id": "spec", "label": "Speculative decoding", "x": 5.0, "y": 92.0, "on_frontier": True},
                {"id": "naive", "label": "Naive downsampling", "x": 2.1, "y": 71.0, "on_frontier": False},
            ],
        },
    ),
    "radar_profile": _spec(
        "radar_profile", slide_id="sl_07_ability-profile",
        title="Per-ability profile",
        claim="The aggregate score hides an uneven ability profile.",
        takeaway="One score, five different abilities",
        grounding={"kind": "data", "ledger_rows": ["nc-01-0203"]},
        content={
            "axes": [{"label": "Counting"}, {"label": "Spatial relations"}, {"label": "Text reading"},
                     {"label": "Attributes"}, {"label": "Comparison"}],
            "series": [
                {"id": "model", "label": "Evaluated model", "values": [78, 41, 88, 72, 55],
                 "source": "SRC-007"},
            ],
        },
    ),
    "distribution_breakdown": _spec(
        "distribution_breakdown", slide_id="sl_09_failure-types",
        title="Where visual reasoning fails",
        claim="More than half of the failures come from insufficient exploration.",
        takeaway="Exploration causes most failures",
        grounding={"kind": "data", "ledger_rows": ["nc-01-0311"]},
        emphasis={"highlight": ["explore"]},
        content={
            "total_label": "Failure cases by cause",
            "parts": [
                {"id": "explore", "label": "Insufficient exploration", "value": 55, "source": "SRC-011"},
                {"id": "backtrack", "label": "No backtracking", "value": 27, "source": "SRC-011"},
                {"id": "percept", "label": "Perception error", "value": 18, "source": "SRC-011"},
            ],
        },
    ),
    "heatmap_grid": _spec(
        "heatmap_grid", slide_id="sl_13_grounding-by-domain",
        title="Grounding accuracy by domain",
        claim="Grounding accuracy varies more across domains than across models.",
        takeaway="Domain matters more than model",
        grounding={"kind": "data", "ledger_rows": ["nc-01-0502"]},
        content={
            "rows": [{"label": "Qwen 3VL"}, {"label": "Gemini class"}],
            "cols": [{"label": "GUI"}, {"label": "Documents"}, {"label": "Natural scenes"}],
            "cells": [
                {"row": "Qwen 3VL", "col": "GUI", "value": 71.4, "unit": "%"},
                {"row": "Qwen 3VL", "col": "Documents", "value": 88.1, "unit": "%"},
                {"row": "Qwen 3VL", "col": "Natural scenes", "value": 92.6, "unit": "%"},
                {"row": "Gemini class", "col": "GUI", "value": 74.9, "unit": "%"},
                {"row": "Gemini class", "col": "Documents", "value": 90.2, "unit": "%"},
                {"row": "Gemini class", "col": "Natural scenes", "value": 93.1, "unit": "%"},
            ],
        },
    ),
    "timeline_data": _spec(
        "timeline_data", slide_id="sl_15_clip-lineage",
        title="From CLIP to concept separation",
        claim="Each step in this lineage fixed a specific weakness of the one before it.",
        takeaway="Each step fixed the last one's blind spot",
        content={
            "events": [
                {"id": "clip", "when": "baseline", "label": "CLIP contrastive pretraining"},
                {"id": "siglip", "when": "next", "label": "SigLIP pairwise sigmoid loss"},
                {"id": "hardneg", "when": "then", "label": "Hard-negative mining"},
                {"id": "c2clip", "when": "now", "label": "C2-CLIP concept separation"},
            ],
        },
    ),
    "magnitude_ladder": _spec(
        "magnitude_ladder", slide_id="sl_24_small-models",
        title="Parameter count is not the story",
        claim="A 7B medical model matches reasoners an order of magnitude larger.",
        takeaway="7B matches models ten times larger",
        grounding={"kind": "data", "ledger_rows": ["nc-01-0611"]},
        emphasis={"highlight": ["octomed"]},
        content={
            "metric": {"label": "Parameters", "unit": "B"},
            "bars": [
                {"id": "octomed", "label": "OctoMed", "value": 7, "source": "SRC-024"},
                {"id": "mid", "label": "General reasoner", "value": 70, "source": "SRC-025"},
                {"id": "large", "label": "Frontier reasoner", "value": 400, "source": "SRC-026"},
            ],
        },
    ),
    # -- Family D: schematic shapes -------------------------------------------
    "schematic_curve": _spec(
        "schematic_curve", slide_id="sl_08_calibration",
        title="Confidence should fall as the image degrades",
        claim="A calibrated model loses confidence as blur rises; the observed model does not.",
        takeaway="Confidence stays flat as blur rises",
        emphasis={"highlight": ["observed"]},
        content={
            "x": {"label": "Blur applied to the image"},
            "y": {"label": "Reported confidence"},
            "curves": [
                {"id": "ideal", "label": "Calibrated behaviour", "shape": "monotone_down", "style": "ideal"},
                {"id": "observed", "label": "Observed behaviour", "shape": "flat", "style": "observed",
                 "regimes": [{"from": 0.55, "to": 1.0, "label": "image no longer legible"}]},
            ],
            "annotations": [{"at": 0.55, "text": "humans stop answering here"}],
        },
    ),
    "trajectory_on_manifold": _spec(
        "trajectory_on_manifold", slide_id="sl_02_manifold",
        title="Staying on the data manifold",
        claim="Tangent updates keep the sample on the manifold; unconstrained steps leave it.",
        takeaway="Tangent steps stay on the manifold",
        emphasis={"highlight": ["tangent"]},
        content={
            "manifold": {"label": "Data manifold"},
            "trajectories": [
                {"id": "tangent", "label": "Tangent-space update", "on_manifold": True},
                {"id": "free", "label": "Unconstrained gradient step", "on_manifold": False},
            ],
            "vectors": [{"at": 0.45, "kind": "normal", "label": "off-manifold component"}],
        },
    ),
    "landscape_regions": _spec(
        "landscape_regions", slide_id="sl_13_where-it-applies",
        title="Where evidence sensitivity matters",
        claim="Evidence sensitivity matters most where stakes are high and the domain is unfamiliar.",
        takeaway="High stakes plus novelty needs evidence",
        content={
            "x_axis": {"label": "Domain familiarity"},
            "y_axis": {"label": "Stakes"},
            "regions": [
                {"id": "clinic", "label": "Clinical imaging", "x": 0.25, "y": 0.8, "extent": 0.2},
                {"id": "gui", "label": "GUI agents", "x": 0.55, "y": 0.55, "extent": 0.18},
                {"id": "photo", "label": "Consumer photo editing", "x": 0.82, "y": 0.2, "extent": 0.16},
            ],
        },
    ),
    "tug_of_war": _spec(
        "tug_of_war", slide_id="sl_10_priors-vs-pixels",
        title="Language priors against pixel evidence",
        claim="On coarse questions language priors win; fine-grained questions need the pixels.",
        takeaway="Coarse questions let priors win",
        content={
            "outcome": "Answer on a coarse question",
            "left": {"id": "priors", "label": "Language priors", "weight": 0.68,
                     "note": "Plausible answers without looking"},
            "right": {"id": "pixels", "label": "Pixel evidence", "weight": 0.32,
                      "note": "Only consulted when the question forces it"},
        },
    ),
    # -- Family E: visual-domain ----------------------------------------------
    "image_transform_strip": _spec(
        "image_transform_strip", slide_id="sl_08_blur-strip",
        title="The same question, a degraded image",
        claim="The same question is asked of a clean and a heavily blurred image.",
        takeaway="Same question, unusable image",
        emphasis={"highlight": ["blurred"]},
        content={
            "panels": [
                {"id": "clean", "glyph": "photo", "caption": "Original frame"},
                {"id": "blurred", "glyph": "blurred_photo", "params": {"sigma": 4},
                 "caption": "Heavy blur applied", "via": "gaussian blur"},
                {"id": "answer", "glyph": "text_bubble", "caption": "Same confident answer",
                 "via": "same prompt"},
            ],
        },
    ),
    "video_frame_strip": _spec(
        "video_frame_strip", slide_id="sl_10_flicker",
        title="Where per-frame editing flickers",
        claim="Independent per-frame edits disagree between neighbouring frames.",
        takeaway="Neighbouring frames disagree",
        content={
            "time_label": "frame order",
            "highlight": ["f2", "f3"],
            "frames": [
                {"id": "f1", "caption": "frame 1"},
                {"id": "f2", "caption": "frame 2"},
                {"id": "f3", "caption": "frame 3"},
                {"id": "f4", "caption": "frame 4"},
            ],
            "links": [{"from": 1, "to": 2, "label": "identity shifts"}],
        },
    ),
    "spatial_3d_schematic": _spec(
        "spatial_3d_schematic", slide_id="sl_25_3d-scene",
        title="Language-referenced 3-D scenes",
        claim="A language query selects a region of a reconstructed 3-D scene.",
        takeaway="Language selects a region in 3-D",
        emphasis={"highlight": ["target"]},
        content={
            "primitives": [
                {"id": "camera", "kind": "camera", "label": "Capture camera",
                 "position": {"col": -2.4, "row": 0, "level": 0.4}},
                {"id": "splats", "kind": "splats", "label": "Gaussian splats",
                 "position": {"col": 0, "row": 0, "level": 0.5}},
                {"id": "target", "kind": "box", "label": "Referred object",
                 "position": {"col": 1.6, "row": -0.8, "level": 0}},
            ],
        },
    ),
    "token_grid_overlay": _spec(
        "token_grid_overlay", slide_id="sl_20_token-grid",
        title="Which visual tokens survive pruning",
        claim="Most patches are dropped before the language model ever sees them.",
        takeaway="Most patches never reach the model",
        grounding={"kind": "hybrid", "ledger_rows": ["nc-01-0417"]},
        content={
            "grid": {"rows": 5, "cols": 8, "kept": [9, 10, 11, 17, 18, 19, 26, 27, 34],
                     "caption": "kept patches, in accent"},
            "notes": [
                {"id": "pruned", "value": "80.2%", "label": "of visual tokens pruned"},
                {"id": "kept", "value": "96.0%", "label": "of accuracy retained"},
            ],
        },
    ),
    "modality_fusion": _spec(
        "modality_fusion", slide_id="sl_22_fusion",
        title="Higher-order fusion with a missing modality",
        claim="The fusion still produces an answer when one modality is missing.",
        takeaway="Fusion survives a missing modality",
        content={
            "modalities": [
                {"id": "rgb", "glyph": "photo", "label": "RGB frames"},
                {"id": "event", "glyph": "event_stream", "label": "Event stream"},
                {"id": "depth", "glyph": "depth_map", "label": "Depth", "missing": True},
            ],
            "fusion": {"label": "Higher-order fusion", "output": {"label": "Robust scene estimate"}},
        },
    ),
    "evidence_crop": _spec(
        "evidence_crop", slide_id="sl_12_licensed-figure",
        title="A licensed figure from the source",
        claim="The published figure shows the failure case directly.",
        takeaway="The source figure shows the failure",
        grounding={"kind": "evidence_crop", "license": "CC BY 4.0"},
        content={"crop": {"asset": "assets/licensed/fig3_failure_case.png",
                          "caption": "Figure 3 from the cited paper"}},
    ),
    # -- Family F: composite ---------------------------------------------------
    "composite": _spec(
        "composite", slide_id="sl_20_composite",
        title="Pruning: the mechanism and the result",
        claim="Pruning drops most patches, and the accuracy barely moves.",
        takeaway="Fewer patches, nearly all the accuracy",
        grounding={"kind": "hybrid", "ledger_rows": ["nc-01-0417"]},
        content={
            "layout": "row",
            "panels": [
                {"title": "Mechanism", "weight": 1.0, "spec": {
                    "spec_version": _V, "figure_id": "example-composite-a", "slide_id": "sl_20_composite",
                    "archetype": "flow", "claim": "Patches are scored, then dropped.",
                    "takeaway": "Score, then drop", "grounding": {"kind": "schematic"},
                    "content": {
                        "nodes": [{"id": "patches", "label": "Patches", "kind": "data"},
                                  {"id": "score", "label": "Importance score", "kind": "process"},
                                  {"id": "keep", "label": "Kept tokens", "kind": "output"}],
                        "edges": [{"from": "patches", "to": "score"}, {"from": "score", "to": "keep"}],
                    }}},
                {"title": "Result", "weight": 1.0, "spec": {
                    "spec_version": _V, "figure_id": "example-composite-b", "slide_id": "sl_20_composite",
                    "archetype": "stat_callout", "claim": "Accuracy barely moves.",
                    "takeaway": "96% retained",
                    "grounding": {"kind": "data", "ledger_rows": ["nc-01-0418"]},
                    "content": {"items": [
                        {"id": "kept", "value": 96.0, "unit": "%", "label": "accuracy retained",
                         "source": "SRC-014"}]}}},
            ],
        },
    ),
    # -- Family G: text -------------------------------------------------------
    # ICML 2026 deck 01, sl_03: title, first claim and visible text verbatim.
    # The operator's outcome when no picture carries the slide honestly; it
    # adds nothing, so it has no takeaway the slide does not state.
    "key_points": _spec(
        "key_points", slide_id="sl_03_tools-carry-the-science-rosetta-and-simula",
        title="Tools carry the science: Rosetta and simulators, not chat",
        claim=("In Agent Rosetta and agentic epidemiological modeling, the LLM orchestrates "
               "while physics-based design software or a calibrated simulator supplies the "
               "scientific constraint."),
        takeaway="",
        content={
            "points": [
                {"text": "LLM plans; the domain tool evaluates"},
                {"text": "Agent Rosetta handles non-canonical residues"},
                {"text": "Agents synthesize, calibrate, and refine simulators"},
                {"text": "Tool use is not experimental validation"},
            ],
        },
    ),
}


def example_for(archetype: str) -> dict[str, Any]:
    return EXAMPLES[archetype]
