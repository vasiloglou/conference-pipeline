---
name: conference-editorial-storyboard
description: Build Topic 4 source board, material clusters, section spine, storyboard, visual direction, and v2.2 expansion evidence before render planning.
---

# Conference Editorial Storyboard

Use this skill to build the editorial foundation for a Topic 4 presentation
before any render planning begins. All storyboard work must be grounded in
named sources (papers, authors, speakers) — never generic reference labels.

For executive-summary variants, the deck is DERIVED FROM EACH TOPIC'S NARRATION
by an LLM (Opus `claude-opus-4-8`), not hand-authored from a small lesson list.
Rebuild it with the single entry point
`scripts/presentations/build_executive_summary.py --scope short` (chains the Opus
topic-digest pass → `build_digest_deck_plan.py` adapter → deck contracts →
Reveal.js HTML → PDF/frame export; add `--regenerate-digests` to re-run Opus).
The short deck structure is: a title slide carrying the cross-topic CENTRAL
MESSAGE; a source-grounded 12-TOPIC INDEX (one Opus one-liner per topic) as
slide 2 — this REPLACES any analysis-derived "important talks" bookend; ONE
EVIDENCE SLIDE PER TOPIC (named papers/speakers + an evidence→decision visual);
then closing OPEN-QUESTIONS and PREDICTIONS synthesis slides; then a
FURTHER-READING bookend. The long deck expands to multiple slides per topic from
the same digests. Slide 2 is the topic index, never a hallucinated talk list;
event/resource bookends may only surface program entries that carry a verifiable
`iclr.cc` URL (enforced by the `synthesize_key_events.py` program-anchor gate and
the `select_opening_events` URL guard). The deck cannot use beginning/end
resource indexes as a substitute for synthesis. Each topic slide must name the
learned point, the transcript/paper evidence behind it (named sources), and the
decision implication. Duration targets remain ~900 s (short) / ~1800 s (long).
Duration is a delivery/render contract, not a proxy for slide count: executive
storyboards must require downstream duration proof of about 900 seconds for the
short deck and about 1800 seconds for the long deck before they are considered
ready for production. Duration proof must include narration text length and, for
production review, a sampled TTS/audio extrapolation; a timing JSON that merely
sums to the target is not enough.
Executive-summary visual work is source-evidence-first. Prefer extracted
paper/talk figures, source-derived charts, and deterministic diagrams that make
the claim inspectable. GenAI raster images are optional premium supplements,
not a quota: use them only when they make the evidence clearer than a source
figure or sober programmatic diagram, and back any used image with provenance
and review evidence. Briefs alone are planning evidence, not production
evidence.

## Required Contract

- Run `build_editorial_source_board.py` to produce `01-SOURCE-EDITORIAL-BOARD.json` with named paper/author/speaker contributions.
- Run `build_storyboard.py` to produce `01-STORYBOARD.json` with one row per slide.
- Run `review_storyboard.py` to produce `01-STORYBOARD-REVIEW.json`.
- For Topic 4 v2.2, compile the approved source board/storyboard into `presentation_deck.json`, `source_binding_report.json`, `visual_plan.json`, `delivery_single_speaker.json`, and `delivery_notebooklm.json` before any figure generation or rendering.
- If the deck exceeds 14 slides, require an explicit `approved_expansion` record with `approved: true`, `target_slide_count`, `max_runtime_minutes`, separate `blueprint_sha256`, `storyboard_sha256`, `source_plan_sha256`, `review_receipt_path`, and `rationale`.
- Every storyboard row must include all required fields:
  - `slide_id` — unique slide identifier
  - `viewer_facing_claim` — the claim the viewer takes away
  - `proof_object` — the visual element that proves the claim (diagram, chart, table, etc.)
  - `visual_concept` — the visual mechanism or layout concept
  - `named_source_contribution` — specific paper, author, or speaker being cited
  - `narration_beat` — the narration moment that corresponds to this slide
  - `reason_to_exist` — why this slide is in the deck (cannot be "summarizes the topic")
- Each compiled slide must separate short on-slide `visible_text` from narration and include `evidence_caption`, `visual_takeaway`, `figure_focus`, and `split_required`.
- Order the storyboard to TEACH a broad, mixed-expertise audience that cannot be assumed to have background in every subfield: assume no prior knowledge of the topic's subfield, build from fundamentals, motivate why each idea matters before how it works, and require every specialist term, method, acronym, and benchmark to be introduced in plain language before a later `narration_beat` relies on it.
- Executive-summary body slides must use evidence-to-decision proof objects:
  extracted source figures, source-derived charts, deterministic diagrams, or
  carefully reviewed optional premium images. Generic bullet/card placeholders
  and decorative art are not sufficient.
- If executive-summary builds use GenAI images, they must write
  `premium_image_receipt.json` with source-bound prompts, provenance, and
  Gemini/GPT-5/Claude/Qwen review requirements. There is no mandatory GenAI
  image count when source figures or programmatic evidence visuals are stronger.
- Keep `SRC-*` source IDs in machine-readable contracts and review packets; visible/audio copy should use natural source names unless a source ID is genuinely useful on screen.
- Every source label must reference a specific named paper, author, or speaker — not a generic reference ID.
- `source_url`, `paper_url`, or `openreview_url` must be present for each named source.

## Rejected Evidence

- Generic source labels such as `Ref 1`, `Paper 2`, `Source A` — must name the actual paper or author.
- Source-ID tables used as the primary visual for a slide — these are metadata, not proof objects.
- Missing named paper, author, or speaker references — every claim needs a named source.
- Missing `source_url`, `paper_url`, or `openreview_url` for each named source contribution.
- `01-STORYBOARD-REVIEW.json` with `approval.status` other than `approved_for_render_planning`.
- Storyboard rows missing any of the 7 required fields.
- Bulk storyboard expansion not grounded in the approved source board.
- Any >14-slide deck without the explicit `approved_expansion` record.
- Conflated blueprint/storyboard/source-plan digest fields; each digest must be separate.
- Full speaker notes, narration paragraphs, raw metadata tags, or source ledgers in `visible_text`.
- Displayed or spoken person names that do not match source records or approved aliases.
- Executive-summary storyboards that repeat topic/resource indexes without a
  quote-backed lesson spine.
- Executive-summary lesson rows without transcript evidence, a named source
  anchor, and a decision implication.
- Executive-summary decks NOT derived from the per-topic Opus digests
  (`executive_topic_digests.json` via `build_executive_summary.py`): hand-picking
  a small lesson subset instead of covering all 12 topics, or using an
  analysis-derived "important talks" bookend instead of the source-grounded
  12-topic index, is rejected.
- Executive-summary plans that claim the 15/30-minute requirement from slide
  count alone, without narration word-count proof and TTS sample extrapolation.
- Executive-summary body visuals that hide evidence in metadata while showing
  only generic bullet cards on the slide.
- Executive-summary visuals that are flashy, metaphorical, or generic but do
  not make a concrete evidence claim inspectable.
- Executive-summary builds that claim premium GenAI images were used but stop
  at `premium_image_briefs.json` without actual raster assets and
  `premium_image_receipt.json`.

## Verification

- `01-SOURCE-EDITORIAL-BOARD.json` exists.
- `01-STORYBOARD.json` exists with one row per slide, each containing all 7 required fields.
- `01-STORYBOARD-REVIEW.json` exists with `approval.status: approved_for_render_planning`.
- Compiled deck/source/visual/delivery contracts have matching slide IDs and slide counts.
- Executive-summary compiled contracts include `duration_contract.json`,
  `narration_duration_verification.json`, and a TTS sample/extrapolation receipt
  matching the short and long targets before approval.
- Executive-summary compiled contracts include screenshot-reviewable body
  slides with visible lesson/evidence/source/decision fields and
  `asset_inventory.json` / `figure_specs.json` proving each body visual is an
  inspectable source figure, source-derived chart, deterministic diagram, or
  reviewed optional premium image.
- Any approved expansion is digest-bound and current to the blueprint, storyboard, and source plan.
- No bulk presentation regeneration before approved `04_safety_alignment_privacy` canary.
