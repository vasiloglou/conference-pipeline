"""Per-family layout engines.

A layout takes a validated spec and a plot rectangle and draws into an
``SvgDoc``. It may not read the network, call a model, or use randomness: the
same spec must always produce the same marks.
"""

from __future__ import annotations

from collections.abc import Callable

from conference_pipeline.figures.layouts import (
    comparison,
    composite,
    quantitative,
    schematic,
    text,
    topology,
    visual,
)

LayoutFn = Callable[..., None]

LAYOUTS: dict[str, LayoutFn] = {
    # Family A
    "flow": topology.draw_flow,
    "loop": topology.draw_loop,
    "nested_architecture": topology.draw_nested_architecture,
    "layered_stack": topology.draw_layered_stack,
    "branching_paths": topology.draw_branching_paths,
    "hierarchy_tree": topology.draw_hierarchy_tree,
    "swimlane_sequence": topology.draw_swimlane_sequence,
    "hub_spoke": topology.draw_hub_spoke,
    # Family B
    "feature_matrix": comparison.draw_feature_matrix,
    "contrast_pair": comparison.draw_contrast_pair,
    "quadrant_map": comparison.draw_quadrant_map,
    "ranked_list": comparison.draw_ranked_list,
    "decision_tree": comparison.draw_decision_tree,
    "venn_overlap": comparison.draw_venn_overlap,
    # Family C
    "bar_compare": quantitative.draw_bar_compare,
    "paired_delta": quantitative.draw_paired_delta,
    "stat_callout": quantitative.draw_stat_callout,
    "line_curve": quantitative.draw_line_curve,
    "scatter_frontier": quantitative.draw_scatter_frontier,
    "radar_profile": quantitative.draw_radar_profile,
    "distribution_breakdown": quantitative.draw_distribution_breakdown,
    "heatmap_grid": quantitative.draw_heatmap_grid,
    "timeline_data": quantitative.draw_timeline_data,
    "magnitude_ladder": quantitative.draw_magnitude_ladder,
    # Family D
    "schematic_curve": schematic.draw_schematic_curve,
    "trajectory_on_manifold": schematic.draw_trajectory_on_manifold,
    "landscape_regions": schematic.draw_landscape_regions,
    "tug_of_war": schematic.draw_tug_of_war,
    # Family E
    "image_transform_strip": visual.draw_image_transform_strip,
    "video_frame_strip": visual.draw_video_frame_strip,
    "spatial_3d_schematic": visual.draw_spatial_3d_schematic,
    "token_grid_overlay": visual.draw_token_grid_overlay,
    "modality_fusion": visual.draw_modality_fusion,
    "evidence_crop": visual.draw_evidence_crop,
    # Family F
    "composite": composite.draw_composite,
    # Family G
    "key_points": text.draw_key_points,
}

__all__ = ["LAYOUTS", "LayoutFn"]
