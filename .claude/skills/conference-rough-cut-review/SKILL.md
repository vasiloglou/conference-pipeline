---
name: conference-rough-cut-review
description: Run the two fail-closed produced-artifact gates for any conference deck - deck acceptance on rendered slides BEFORE video, then rough-cut review of the finished videos.
---

# Conference Rough-Cut Review

Two gates, in this order, for any conference workspace. Running them in the
wrong order is expensive and was the single costliest mistake of the CVPR 2026
production run.

## Gate A: deck acceptance, BEFORE any TTS or video rendering

Run the multi-model produced-deck review as soon as the rendered screenshots,
PDF and visual-QC report exist, and BLOCK TTS and video rendering until it
passes. It needs the blueprint, blueprint review, deck contract, render spec,
source binding, visual-QC report and the slide screenshots. **It does not need
a video, an audio segment, or a delivery contract**, so there is no reason to
pay for those first.

**Why this ordering is mandatory.** On CVPR 2026 the review ran only at the end
of the delivery/video substage. Twenty-four videos were rendered across two and
a half hours, and the review then failed every deck on figure defects the
reviewers had confirmed in the screenshots. Every one of those videos was
invalidated by the figure fix that followed. A ten-minute gate on one deck
would have caught it before the first video was built. Treat a deck as
unaccepted until Gate A passes, and never let an expensive step run ahead of a
cheap gate that can refuse its input.

**Canary the gate itself.** Before running Gate A across a full deck set, run
it on ONE deck and read the per-reviewer statuses. A reviewer that is
unavailable for an environment reason (an untrusted working directory, a model
the account cannot use) silently degrades the ensemble to fewer voices while
still reporting a verdict, and that has happened in production.

## Gate B: rough-cut review, AFTER video production

Use this gate once both video formats exist, to validate what only a finished
video can show: audio/visual sync, frame health, transcript quality and
duration contracts. Gate B never re-litigates figure quality -- Gate A owns
that, on the screenshots, before the videos exist.

## Re-encode or re-record: deciding what a deck change actually costs

When a deck changes after its videos were built, decide by what changed in the
delivery contract, not by what changed on screen:

- **Narration text unchanged** (the delivery contract's
  `presentation_deck_sha256` still matches `presentation_deck.json`, the
  segment count still matches the slide count, and the figures' text nodes are
  unchanged): the recorded audio is still correct. Re-render the frames and
  re-encode. TTS segments are cached by a fingerprint of (text, model, voice),
  so re-running the video build reuses every segment and costs nothing in
  provider spend. Prove it by diffing the `<text>` nodes of the old and new
  figures, not by assuming: compile the same specs with the previous code
  revision and compare.
- **Narration text changed at all**: the audio no longer matches the script.
  Redo the narration review, the TTS and the video, in that order. Do not
  splice new audio into an old timeline.

A purely geometric figure change -- a connector re-route, a glyph swap, a
crossing mark -- is a re-encode. A changed label, a changed slide count, a
changed deck contract is a re-record.

## Required Contract

- Gate A: run the multi-model produced-deck review on screenshots, PDF and the
  visual-QC report BEFORE any TTS or video rendering, and block both until it passes.
- Gate A must record a per-reviewer status. An ensemble that ran fewer reviewers
  than its declared minimum is `blocked`, never `passed`, and an environment-caused
  reviewer failure must be repaired rather than accepted as "unavailable".
- Gate B: run the `presentation.produced_artifact_review` gate again after both
  presenter/NotebookLM videos are produced, for sync, frame health and duration only.
- Provide PDF screenshot for every slide.
- Provide `render_contact_sheet.png` showing all slide thumbnails.
- Provide both MP4 files: single-presenter narration video and NotebookLM-style two-host video.
- Provide both transcript files: single-presenter transcript and two-host transcript.
- Provide review inputs: `presentation_blueprint.json`, blueprint review, `presentation_deck.json`, `render_spec.json`, `source_binding_report.json`, delivery contract, video receipt, visual QC report, and contact sheet.
- Write the produced review report documenting findings across guideline adherence, technical-audience pleasantness, production readiness, aesthetic pleasantness, clarity/congestion, and artifact-contract adherence.
- Run cross-AI rough-cut or produced-artifact review synthesis and confirm no unresolved critical, major, or blocker findings.
- The review receipt must be current to artifact digests and must have no blockers before Phase 13 canary or bulk rollout can pass.
- Each narration beat must map to a visible frame — no narration reference to a visual element absent from the corresponding slide.
- Executive-summary rough cuts must prove that the produced deck is a
  quote-backed synthesis, not a resource portal: slide 2 and final resources
  are support bookends, while the body carries around 8 executive points in the
  short deck and the same 8-point spine plus 3 finer-grained expansions in the
  long deck, all with transcript evidence and decision implications.
- Executive-summary rough cuts must include current duration proof: the short
  deck totals about 900 seconds and the long deck totals about 1800 seconds in
  delivery contracts and rendered video planning, and the narration itself must
  pass word-count verification plus sampled TTS/audio extrapolation.
- Executive-summary review must inspect actual rendered screenshots or frame
  contact sheets before approval. JSON-only checks cannot prove visual quality.
- Executive-summary rough cuts must be source-evidence-first: body visuals
  should be extracted source figures, source-derived charts, deterministic
  diagrams, or carefully reviewed optional premium images. Decorative or
  metaphor-only generated art fails even if it is polished.
- If premium generated images are used, require `premium_image_receipt.json`
  and review them as evidence-adjacent editorial artifacts: they must be
  source-aligned, non-generic, visually clear, and free of misleading technical
  details before approval.

## Rejected Evidence

- Missing slide screenshots for any slide in the deck.
- Black frames or visually desynced frames in either MP4 file.
- Mechanical two-host dialogue patterns such as `Host A frames` / `Host B connects` — dialogue must feel natural and conversational.
- Narration/visual mismatch: narration references `the diagram`, `the grid`, `the ladder`, `the shield`, or any visual element not present on the slide shown in that frame.
- Any `needs_revision` receipt from a prior review cycle that was not addressed before this rough-cut review.
- Scaffold prose in either video transcript.
- Cross-AI synthesis with unresolved critical, major, or blocker findings.
- Any reviewer unavailable, `failed`, or missing from the expected reviewer set.
- Contract drift between blueprint, approved expansion, deck, render spec, source binding, delivery segments, video receipts, visual QC, or contact sheet digests.
- Video duration mismatch, missing video receipt, stale visual QC, unreadable evidence visuals, or legibility blockers.
- Produced review report absent, incomplete, stale, or not digest-bound to the artifacts under review.
- Executive-summary rough cuts where source/index slides are visually stronger
  than the lesson slides, or where the closing lands on a link index instead of
  a decision frame.
- Generated executive images without provenance, source-fit review, and a
  recorded pass/fail quality decision.
- Generated executive images that look impressive but do not make the slide's
  evidence claim inspectable.
- Executive-summary rough cuts that satisfy the requested point count but miss
  the 15-minute or 30-minute narration-duration contract, word-count proof, or
  sampled TTS extrapolation.
- Executive-summary rough cuts whose body slides are generic bullet/card
  diagrams, flashy metaphor art, or abstract raster images rather than
  evidence-to-decision synthesis visuals.
- Executive-summary rough cuts that use GenAI raster images but lack a
  provenance manifest and source-fit review.
- Executive-summary rough cuts approved without screenshot/contact-sheet
  inspection evidence.

## Verification

- Produced review report exists with review findings for slides, scripts, visuals, and both video formats.
- Cross-AI rough-cut or produced-artifact synthesis exists with no unresolved critical, major, or blocker findings.
- Both MP4 files play without black or desynced frames.
- Both transcript files are present.
- `render_contact_sheet.png` exists and shows all slides.
- Review receipt digests match current produced artifacts.
- Executive-summary duration contracts match the produced delivery/video
  planning totals, and `narration_duration_verification.json` plus sampled TTS
  extrapolation stay above target.
- Executive-summary screenshot/contact-sheet review confirms the body slides
  visibly carry lesson, evidence, source, and decision implication, and that
  the visuals are inspectable evidence objects rather than decorative art.
- If GenAI images are used, `premium_image_receipt.json` proves source-bound
  prompts, provenance, and reviewer decisions.
- Gate A passed on the rendered screenshots before any TTS or video spend, with
  every declared reviewer either returning a verdict or its environment failure fixed.
- When a deck changed after its videos existed, the receipt records which of
  re-encode or re-record was chosen and the evidence for it (delivery-contract
  digest match plus a figure text-node diff).
- No bulk presentation regeneration before the workspace's own approved canary deck.

## Review iteration cap (operator rule, 2026-09-26)

- At most **3 review iterations** per deck per gate: review -> fix -> review -> fix -> review. After the third review, stop: apply no further fixes and run no further reviews, even if new findings appear.
- New findings raised by a later iteration count against the same cap; they do not reset it.
- Each fix pass addresses only the blocking findings of the review before it, with the smallest change the source supports.
- After the cap, report the remaining findings to the operator, who decides whether to accept the deck as is or to fix named items without another review. Never start a fourth review on your own.

## Figure fidelity policy (2026-10-01)

Gate A procedure for the reviewing agent, before asking the operator:

1. Open `<deck>/figure_review_page.html` (written by the runner at the canary
   stop) AND view every slide screenshot under
   `assets/generated/qc/screenshots/` yourself; the judges' text is not a
   substitute for looking.
2. For each figure still open, or wrong on inspection: patch or replace it
   with `scripts/presentations/patch_figure_spec.py` (`--by` is you, with the
   reason); accept only visual-only defects with the meaning intact; escalate
   what you cannot verify against the sources.
3. Resume the deck through the runner; look again at every slide you changed.
   At most two agent passes; then hand over with every finding decided.
4. Only then ask the operator to approve and record
   `record_human_review_receipt.py` with the operator's name.

Gate A reads `<deck>/figure_review_page.html`, written by the runner at the
canary stop. Every figure finding listed there as still open must get an
operator outcome through `scripts/presentations/patch_figure_spec.py` (accept,
patch, replace), then the deck is resumed through the runner. This is
enforced in code, not by this text: `record_human_review_receipt.py` refuses
`--decision approved_for_bulk_rollout` and the runner refuses
`--accept-presentation-review-findings` while
`conference_pipeline.presentation.figure_findings.undecided_figure_findings`
is non-empty. No fourth automated critic round. See
`docs/operations/figure-fidelity-policy.md`.
