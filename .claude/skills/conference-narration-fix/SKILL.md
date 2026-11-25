---
name: conference-narration-fix
description: Repair Topic 4 narration scripts and delivery contracts that fail pre-TTS review gates.
---

# Conference Narration Fix

Use this skill when `narration_review.json` has `approve_for_tts: false` or
contains any fail-level findings. Read the findings, fix each one in the
delivery contracts or their source artifacts, and re-run narration review to
confirm the deck is clear before TTS submission.

## Required Contract

- Read `narration_review.json` and identify all fail-level and warn-level findings.
- For each unresolved finding:
  - If a visual reference is missing from the rendered frame: revise the narration to reference only what is actually visible, or escalate to add the visual to the slide.
  - If a mechanical two-host phrase is present: rewrite the exchange to be natural dialogue.
  - If a passage assumes prior subfield knowledge or uses an undefined technical term, method, acronym, or benchmark: rewrite it to define the term in plain language at first mention and lead with intuition or why-it-matters before the detail. Do not delete the substance — teach it (a host's "wait, why?" turn is the natural place).
  - If `Prepared by Nikolaos Vasiloglou` appears on a content slide narration: remove it.
  - If a visual entity is unnamed: name the specific entity shown.
  - If a segment lacks a human-readable source/person/work: repair `delivery_notebooklm.json` from `render_spec.json` and `source_binding_report.json`.
  - If a claim has no source metadata: fix `source_binding_report.json` or remove the claim; do not patch around missing evidence in prose.
  - If a displayed/spoken name is not a source record or approved alias: correct the upstream source metadata, then regenerate affected delivery text.
  - If an executive-summary script repeats resource indexes, source-backbone
    narration, or generic "where to read next" language as content: rebuild the
    beat as a quote-backed lesson with claim, transcript evidence, named source
    anchor, and decision implication.
  - If an executive-summary `duration_contract.json` is missing or delivery
    totals miss the 15-minute short or 30-minute long target: repair the
    upstream timing contract and regenerate affected delivery scripts.
- After all fixes are applied, re-run `conference-narration-review` skill to produce a new `narration_review.json`.
- Confirm the new `narration_review.json` shows `approve_for_tts: true`, 0 fail findings, 0 warn findings.

## Rejected Evidence

- Same patterns as `conference-narration-review`:
  - Repeated `Prepared by Nikolaos Vasiloglou` on content slides.
  - Mechanical `Host A frames` / `Host B connects` two-host phrases.
  - Visual references absent from storyboard proof objects.
  - Visual references absent from rendered slide frames.
  - Generic "the figure shows" with no named entity.
- New prose that drops `SRC-*` IDs from delivery metadata.
- Fixes that make audio sound natural while leaving stale source-binding or render-spec drift.
- A fix attempt that does not re-run narration review before claiming the gate is clear.
- `narration_review.json` with any remaining fail or warn findings presented as a passing receipt.
- Executive-summary narration fixes that make the wording smoother while
  preserving the repeated beginning/end resource-index structure.
- Executive-summary fixes that leave stale or absent duration proof while
  claiming the short/long timing requirement has been met.

## Verification

- `narration_review.json` at deck level shows `approve_for_tts: true`.
- `narration_review.json` shows 0 fail findings and 0 warn findings after fix.
- Delivery contracts, scripts, and rendered frames are current to the same `render_spec.json` digest.
- Executive-summary delivery totals match the 900-second short or 1800-second
  long duration contract after repair.
- No bulk presentation regeneration before approved `04_safety_alignment_privacy` canary.

## Review iteration cap (operator rule, 2026-09-26)

- At most **3 review iterations** per deck per gate: review -> fix -> review -> fix -> review. After the third review, stop: apply no further fixes and run no further reviews, even if new findings appear.
- New findings raised by a later iteration count against the same cap; they do not reset it.
- Each fix pass addresses only the blocking findings of the review before it, with the smallest change the source supports.
- After the cap, report the remaining findings to the operator, who decides whether to accept the deck as is or to fix named items without another review. Never start a fourth review on your own.
