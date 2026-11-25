---
name: conference-presentation-blueprint
description: Generate a source-grounded Presentation Blueprint before conference slide markdown, including thesis, telescopic sections, source plan, visual roles, narration beats, conclusion, and duration budget.
---

# Conference Presentation Blueprint

Repo canonical skills live under .claude/skills; local mirrored skills under ~/.codex/skills are sync targets, not the only source. `~/.claude/skills` is also a local sync target when present.

Use this skill when creating or regenerating conference presentation decks. Do
not start from a target slide count or a compressed slide brief. Start from the
full analysis document, conference corpus, source attributions, and paper
matching evidence.

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

- Require or create `presentation_frame.json` before blueprint generation. It
  must distinguish `retrieval_title` from audience-facing `audience_title`,
  include `field_thesis`, at least three broad `general_questions`,
  `source_analysis_path`, and `opening_anchor_policy`, and match the requested
  deck/topic ids.
- Produce `presentation_blueprint.json`, `presentation_blueprint.md`, and `presentation_blueprint_receipt.json`.
- Use `scripts/presentations/generate_editorial_blueprint.py` through registry task `presentation.blueprint_generation`, passing `--presentation-frame decks/<id>/presentation_frame.json`.
- Include thesis, audience promise, telescopic section map, subtopic clusters, source/paper plan, slide intents, visual roles, narration beats, conclusion, and duration budget.
- Use the frame's `audience_title` as the public deck title; keep the original topic/title words as retrieval metadata.
- Make the opening slides answer broad field-level questions before introducing specialist paper mechanisms.
- Write the spine and narration beats to TEACH a broad, mixed-expertise audience,
  assuming no prior knowledge of the topic's subfield: this program covers a wide
  spectrum of work and no listener can have background on everything. Order sections
  to build from fundamentals, motivate why each idea matters before how it works, and
  require every specialist term, method, acronym, and benchmark to be introduced in
  plain language before the spine relies on it. Narration beats must be pedagogical,
  not written for a subfield expert.
- Record passing `presentation_frame_validation` in the generated blueprint.
- Bind every slide intent to approved source labels and source paths before draft markdown exists.
- Plan for <=38 min before TTS; treat >40 min as a hard downstream blocker.
- Record the governed model/provider posture and receipt paths without secrets.

## Executive Overview Blueprint Addendum

For executive-summary decks, do not let the opening and closing bookends become
the story. Resource links are supporting navigation, not the main narrative.
Use source selection to build a candidate pool of up to 24 distilled lessons
from the conference, roughly two per topic when all 12 topics are in scope.
That pool is for selection and coverage; it is not the final deck shape. The
operator-locked final structure is: the short deck carries around 8 executive
points and targets about 15 minutes; the long deck carries the same 8-point
spine plus 3 finer-grained expansions, around 11 points total, and targets about
30 minutes. The long deck is a finer-grained version of the short deck, not a
separate topic index. Each selected point should state what was learned, why it
matters, and what action or research judgment follows.

Every executive lesson needs evidence: prefer short transcript quotes from
talks, keynotes, workshops, panels, or oral sessions; use tightly grounded
paraphrases only when a direct quote is unavailable or too long. The slide plan
must record the quote/source binding that backs the lesson. Repeated resource
indexes at both the beginning and end are a blocker unless the surrounding
slides clearly deliver new synthesis.

Plan the short deck as the crisp decision version and the long deck as the
finer-grained evidence version. Both must share the same central spine and
should read as leadership synthesis grounded in conference voices.

## Rejected Evidence

- A slide-count expansion from `analysis/slide_briefs/<topic_id>.md` alone.
- A deck outline without source/paper bindings.
- Using the topic-strategy retrieval title as the public title without an approved `presentation_frame.json`.
- Opening with highly technical paper mechanisms before field-level questions.
- Mock/dry-run blueprint artifacts as production completion evidence.
- Bulk presentation regeneration before approved canary review.
- Executive-summary outlines that mostly repeat topic/resource indexes instead
  of quote-backed conference lessons.
- Executive-summary outlines that treat the up-to-24 candidate pool as the
  final deck structure instead of selecting about 8 short-deck points and 3
  long-deck expansions.
- Long-deck outlines that diverge from the short-deck spine rather than
  deepening it.

No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.
