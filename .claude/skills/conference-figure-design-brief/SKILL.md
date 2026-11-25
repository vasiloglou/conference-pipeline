---
name: conference-figure-design-brief
description: Create source-grounded figure design briefs from an approved Presentation Blueprint so SVGs depict mechanisms, architectures, experiments, comparisons, or evidence rather than bullet boxes.
---

# Conference Figure Design Brief

Repo canonical skills live under .claude/skills; local mirrored skills under ~/.codex/skills are sync targets, not the only source. `~/.claude/skills` is also a local sync target when present.

Use this skill before generating programmatic SVGs. The figure design brief is
the bridge between the approved Presentation Blueprint and rendering code.

## Phase 01 Prerequisites (Topic 4 Canary)

Before any Topic 4 rebuild, require:

- Brand template lock: `01-BRAND-REFERENCE-LOCK.json` status is `pass` or expected blockers are explicitly recorded.
- Editorial source board: `01-SOURCE-EDITORIAL-BOARD.json` exists.
- Storyboard review: `01-STORYBOARD-REVIEW.json` has `approval.status: approved_for_render_planning`.
- Cross-AI creative/storyboard synthesis: `01-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json` has `gate_status: approved_for_render_planning`.
- Visual direction: `01-VISUAL-DIRECTION.md` exists.
- Narration readthrough: `01-NARRATION-READTHROUGH.md` exists.
- Rough-cut review: `01-ROUGH-CUT-VIDEO-REVIEW.md` exists.
- Cross-AI rough-cut synthesis exists with no unresolved critical or major findings.
- Explicit human approval receipt exists for Topic 4.

## Required Contract

- Consume `presentation_blueprint.json`, `blueprint_review.json`, source labels, ranked papers, slide claims, and narration beats.
- Write a `figure_design_brief` for every generated visual.
- Choose a visual role: mechanism, architecture, experiment, benchmark, comparison, evidence map, molecule/structure, system workflow, or failure mode.
- Write the `visual_concept` with its depiction vocabulary prefix (`evidence_chart:`,
  `tradeoff_map:`, `comparison_matrix:`, `architecture:`, `mechanism_pipeline:`,
  `timeline:`, `failure_mode:`); the figure generator reads that prefix as the
  `depiction_type` and picks the drawing archetype from it.
- Vary the visual form across the deck. The generator caps runs of box-and-arrow
  figures (`MAX_CONSECUTIVE_BOX_AND_ARROW`) and rotates a brief-compatible archetype
  (timeline, layered stack, before/after split, evidence chart, tradeoff quadrant,
  comparison matrix, ladder) into the third consecutive one; a plan whose briefs are
  all `mechanism_pipeline:` will still be rotated, so prefer to name the better form in
  the brief yourself whenever the evidence has numbers, ordered steps, layers, or a
  failing-vs-fixed contrast.
- Every rendered SVG must pass the measured geometry gate
  (`conference_pipeline.presentation.svg_geometry`): no label crossing its box border,
  no arrow that fails to reach a shape border, no connector through text, no text
  outside the canvas. The generator blocks on these at codegen time and
  `validate_svgs.py` reports them as `geometry_*` warnings; the vision QC stage
  remains the pixel-level authority.
- Require specific entities, relationships, labels, and source traces.
- Use advanced renderers when appropriate; Matplotlib and Plotly are for quantitative charts, not generic diagrams.
- Record renderer family and model governance in `asset_inventory.json` and semantic review receipts.

## Premium Generated Image Option

For executive-summary decks, the visual plan may call for a small set of
premium generated images when a synthesized editorial image communicates the
conference lesson better than a code-generated SVG. A reasonable target is
roughly eight high-quality images for the long executive deck and a smaller
curated subset for the short deck, selected from the highest-value points
rather than one image per slide.

Generated images must still be source-grounded. The brief must name the lesson,
the transcript quote or source evidence it visualizes, the intended metaphor or
scene, prohibited generic tropes, and the review criteria. Prefer a multi-model
review loop using high-end image-capable LLMs such as Gemini, GPT-5-class
models, and Claude, recording model/provider provenance, prompts, outputs, and
review findings. These images supplement the programmatic figure lane; they do
not justify decorative or ungrounded art.

## Rejected Evidence

- Copied bullet text in boxes.
- Repeated generic "diagram showing" templates.
- Placeholder charts with fake ticks.
- Figure narration that never mentions the figure's actual entities or relationship.
- Decorative figures that the blueprint did not request.
- Generic generated images without quote/source bindings or multi-review
  evidence.

No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.

## Figure fidelity policy (2026-10-01)

Follow `docs/operations/figure-fidelity-policy.md`. In a brief:

- List 2-6 components, each a system, method, benchmark, work, person or stage
  that THIS slide's bullets, narration or cited sources name. Never invent a
  baseline, comparator, axis, stage or row the sources do not describe.
- Choose `comparison_matrix` only when two or more cited sources report the
  same properties; `tradeoff_map` only when measured numbers place the points.
- The brief is a list of candidates for the spec author; `figures.grounding`
  drops what the slide does not name and records it. Expect that, do not fight it.
