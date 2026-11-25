---
name: conference-narration-review
description: Review Topic 4 narration and delivery contracts for visual alignment, source grounding, attribution, and tone before TTS.
---

# Conference Narration Review

Use this skill before submitting narration scripts to TTS. Every narration claim
that references a visual element must map to `render_spec.json`,
`source_binding_report.json`, and the rendered frame for that slide.

## Required Contract

- Read `render_spec.json`, `source_binding_report.json`, `delivery_single_speaker.json`, and `delivery_notebooklm.json`.
- Read both narration scripts: single-presenter (`6_presenter_script.md`) and NotebookLM-style two-host (`6_presenter_script_notebooklm_2host.md`).
- For every narration claim that names a visual element — `the diagram shows`, `the grid`, `the ladder`, `the shield`, `the chart`, `the table`, or any specific visual — confirm that:
  - The visual element appears in `figure_focus`, `visual_takeaway`, `evidence_caption`, or the corresponding figure spec for that slide.
  - The visual element is visible in the rendered frame for that slide.
- Each NotebookLM segment must say what the audience sees, why it matters, and which human-readable source/person/work supports it.
- Confirm the narration is pedagogical and assumes no prior knowledge of the subfield: every technical term, method name, acronym, and benchmark is defined in plain language at first mention, ideas are motivated before mechanisms, and intuition or analogy precedes formalism. Treat any spoken passage that assumes the audience already knows this subfield's vocabulary or prior results, or that uses an undefined technical term, as a fail-level finding.
- For executive-summary decks, each content narration beat must say what the
  conference taught, cite or tightly paraphrase transcript-backed evidence, and
  land a decision implication. Slide 2/final resource narration should be short
  support context, not the body of the story.
- For executive-summary decks, read `duration_contract.json` when present and
  verify `delivery_single_speaker.json` and `delivery_notebooklm.json` total
  about 900 seconds for the short deck and about 1800 seconds for the long deck.
  Missing or mismatched duration proof is a pre-TTS blocker.
- Keep `SRC-*` IDs mandatory in delivery JSON metadata and review context, but do not require them to be spoken in audio.
- Validate all displayed or spoken person names against source records or approved aliases before TTS.
- Write `narration_review.json` with per-slide findings and a top-level `approve_for_tts` boolean.
- A narration script with zero unresolved findings receives `approve_for_tts: true`.

## Rejected Evidence

- Repeated `Prepared by Nikolaos Vasiloglou` on any content slide narration.
- Mechanical two-host phrases `Host A frames` / `Host B connects` — must be natural dialogue.
- Hardcoded Topic 4 host text or formulaic source phrase banks instead of generation from `render_spec.json`, `source_binding_report.json`, and visual hooks.
- Visual references in narration that are absent from render/delivery contracts.
- Visual references in narration that are absent from the rendered slide frame.
- `approve_for_tts: false` in `narration_review.json` used as a TTS submission.
- Narration that never names the specific entities shown in the visual (generic "the figure shows" with no entity reference).
- Source-grounding drift: claims without metadata-backed sources in `source_binding_report.json`.
  A slide flagged `"aside": true` in `engaging_narration_raw.json` may open with one or two
  sentences of general-knowledge colour (a well-known episode from the field, an everyday
  story) that is NOT source-backed; that is allowed and is not drift, provided the aside is
  framed as the host's own remark, is not attributed to any paper/speaker/conference, and
  carries no invented statistic, date, quote, or name. An aside attributed to a source, or
  two asides on consecutive slides, is a finding.
- Locator references: narration that names a source by where it was presented instead of by
  its work title -- "this poster", "the poster shows", "poster 63306", a session/slot number,
  `talk_media_...`, a transcript hash. A poster IS a paper; the audience needs the paper's
  name (or the speaker's) and the slide's chip must link the paper (and the poster page).
  `generate_engaging_narration.locator_references` lists the offending lines.
- Monotone hook rhythm: the same hook kind (`hook_kind` in `engaging_narration_raw.json`) on
  consecutive content slides, or every slide's hook drawn only from the paper text.
- Attribution drift such as `Don Song` for `Dawn Song`, or any unapproved displayed/spoken person alias.
- Executive-summary narration that repeats source/resource indexes instead of
  delivering quote-backed lessons.
- Executive-summary narration submitted for TTS without duration proof matching
  the 15-minute short and 30-minute long delivery targets.
- NotebookLM executive-summary dialogue that repeats "source trail" phrasing
  rather than having one host test the evidence or implication of the lesson.

## Verification

- `narration_review.json` exists at the deck level.
- `narration_review.json` has `approve_for_tts: true`.
- Zero fail-level findings in `narration_review.json`.
- Delivery JSON retains `SRC-*` metadata while audio copy uses natural source names.
- Executive-summary delivery totals match the duration contract before TTS.
- No bulk presentation regeneration before approved `04_safety_alignment_privacy` canary.

## Review iteration cap (operator rule, 2026-09-26)

- At most **3 review iterations** per deck per gate: review -> fix -> review -> fix -> review. After the third review, stop: apply no further fixes and run no further reviews, even if new findings appear.
- New findings raised by a later iteration count against the same cap; they do not reset it.
- Each fix pass addresses only the blocking findings of the review before it, with the smallest change the source supports.
- After the cap, report the remaining findings to the operator, who decides whether to accept the deck as is or to fix named items without another review. Never start a fourth review on your own.
