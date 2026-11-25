---
name: conference-presentation-blueprint-review
description: Review, approve, or route repairs for a conference Presentation Blueprint before draft slides, figures, narration, TTS, PDFs, videos, release, or proof.
---

# Conference Presentation Blueprint Review

Repo canonical skills live under .claude/skills; local mirrored skills under ~/.codex/skills are sync targets, not the only source. `~/.claude/skills` is also a local sync target when present.

Use this skill after `conference-presentation-blueprint` and before
`draft_generation`. The review is an approval token, not commentary.

## Required Contract

- Run `scripts/presentations/review_presentation_blueprint.py` through registry task `presentation.blueprint_review`.
- Write `blueprint_review.json` and `blueprint_review.md`.
- Approve draft generation only when `approved_for_draft_generation: true` and the current `blueprint_sha256` are present.
- Record gate type: deterministic, gate type: LLM-judge, and gate type: human evidence.
- Confirm `presentation_frame.json` was consumed, deck/topic ids match, the public title comes from `audience_title`, and `presentation_frame_validation.status` is `pass`.
- Confirm the opening section answers broad field-level questions before specialist paper mechanisms dominate.
- Confirm the spine and narration beats are pedagogical and assume no prior subfield knowledge: specialist terms, methods, acronyms, and benchmarks are introduced in plain language before use, and ideas are motivated before mechanisms. Flag any beat written for a subfield expert or that leans on undefined jargon.
- Confirm source-plan coverage, telescopic section structure, conclusion, visual roles, narration beats, and duration budget.
- If blocked, use `scripts/presentations/repair_presentation_blueprint.py` through `presentation.repair_presentation_blueprint`, then rerun review.

## Downstream Rule

`scripts/presentations/generate_slides_agent.py --from-blueprint` is the only
production draft path after this review. TTS, PDF export, video rendering,
release, proof, and bulk rollout remain blocked when the review is missing,
stale, or not approved.

No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.

## Review iteration cap (operator rule, 2026-09-26)

- At most **3 review iterations** per deck per gate: review -> fix -> review -> fix -> review. After the third review, stop: apply no further fixes and run no further reviews, even if new findings appear.
- New findings raised by a later iteration count against the same cap; they do not reset it.
- Each fix pass addresses only the blocking findings of the review before it, with the smallest change the source supports.
- After the cap, report the remaining findings to the operator, who decides whether to accept the deck as is or to fix named items without another review. Never start a fourth review on your own.
