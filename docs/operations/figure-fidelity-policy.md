# Figure fidelity policy

Adopted 2026-10-01 after the ICML 2026 canary investigation
(`conferences/icml2026/storage/workflow-runs/icml2026-v4.0-live-production-20260711/figure-fidelity-investigation-20261001.md`).
It applies to every conference from its next figure generation; nothing is
regenerated retroactively unless the operator asks.

## What a figure may say

1. **The compiler draws; it never decides.** `conference_pipeline.figures`
   lays a figure out from a spec deterministically. Every fact on a slide's
   figure entered through the spec, so fidelity is enforced on the spec and
   on the brief it was written from.
2. **A component must be named by the slide.** The blueprint's figure brief is
   a list of candidates, not an order. `figures.grounding` applies one rule at
   both ends: a component is kept only if the slide's own text, narration or
   cited sources name it (phrase, name token, or content words). Items of a
   comparison must each be named; a mechanism figure may carry a generic stage
   label, but comparator foils ("hand-designed baseline") and ungrounded twins
   of a named component are dropped and recorded (`dropped_components` on the
   brief, `brief_components_dropped` on the spec receipt row).
3. **Numbers and positions come from measured rows.** A figure that places,
   plots or ranks (quadrant_map, scatter_frontier, bar_compare, line_curve and
   kin) is data-grounded: one numeric-claims ledger row per element. With no
   measured rows the archetype is demoted to a form that carries no positions
   (landscape_regions, feature_matrix, flow). Schematic curves and region
   extents may not imply measured values.
4. **A matrix cell has a source or says so.** `yes`/`no`/`partial` only when
   a cited source states the property for that row; otherwise `not_reported`,
   drawn as a dash. A matrix with one real row is not a comparison.
5. **Nothing is invented to fill a budget.** Briefs ask for 2-6 components,
   never a quota; the spec author carries fewer items rather than add one.

Figure headings name the mechanism or measurement drawn rather than repeat
the slide title. `generate_figure_specs.py` checks this during new authoring,
cached-spec reuse, critic revisions, and compiler repairs. A compilable figure
cannot bypass the check through the layout-budget fallback. Text-layout
archetypes are exempt because they deliberately carry the slide's own words.
On resume, an existing duplicate heading re-enters bounded authoring; valid
cached specs remain reusable without another provider call.

## How a figure is checked

6. **Adversarial critic, always on, three rounds.** Every spec is read by the
   critic against the slide's approved sources and numeric rows before it is
   compiled. Issues are graded major/minor; only a major issue buys a
   re-author, and the revision is read again, up to `MAX_CRITIC_ROUNDS = 3`.
   A revision that fails layout or validation gets the refusal fed back once
   before it is discarded. The critic's worst case must fit its cap or the
   deck halts before spending; the cap is never raised automatically.
7. **Round three is the end of automation.** Whatever is still open is
   reported (`critic.unresolved_major` in the spec receipt, and the review
   page beside the rendered slide). No fourth automated round, ever.
8. **The vision judge sees the current figures.** The drawn SVGs are inputs
   of the screenshot step, so a re-authored figure is re-screenshotted and
   re-judged; a judgement on stale screenshots is a lineage defect.

## What the reviewing agent does, then what the operator decides

9. **The agent reviews before the operator does.** After round three and the
   vision judge, the reviewing agent (Claude, in the session running the
   deck) opens `figure_review_page.html` and looks at every rendered slide
   screenshot itself -- not only the judges' text -- for all canary decks and
   for every patched or re-authored slide in a bulk run. For each figure it:
   - *patches* what the mechanics could not fix, through
     `scripts/presentations/patch_figure_spec.py` (deterministic validate and
     recompile, no model call), attributing itself as `--by` and stating the
     reason; a replacement archetype when the original cannot carry the claim;
     or, when no picture carries the slide honestly, the slide's text layout
     (`--outcome replace --text-layout`): a `key_points` figure built from the
     deck contract slide that typesets only its title, the first sentence of
     its first claim and its visible text, verbatim. It adds nothing, passes
     the grounding rule by construction, and is left out of the deck variety
     quotas;
   - *accepts* only a visual-only defect whose meaning is intact, with the
     reason recorded;
   - *escalates* to the operator anything it cannot decide (a claim it cannot
     verify against the sources, a slide text that is itself wrong), named on
     the hand-over.
   Then it resumes the deck through the runner so the patched figures are
   recompiled, re-screenshotted and re-judged, and repeats once if that pass
   surfaced new findings. Two agent passes at most; what remains is escalated.
10. **The operator approves last**, on the review page, with every finding
    decided and the agent's outcomes visible beside each slide. A deck is not
    approved while any finding is undecided (`record_human_review_receipt.py`
    and the runner enforce this).
11. **Repairs go through the runner.** Targeted re-authoring uses
    `--figure-slide-ids`; hand edits use the patch script. Editing spec files
    directly leaves the step ledger stale and is not done.

## What is guaranteed, and what is not

Guaranteed by code, for every conference that runs the shared presentation
chain (`conference_pipeline.production.presentation_chain`):

- every figure's brief is grounded and every spec is read by the critic
  against the slide's sources before it is compiled (no flag turns it off);
- a critic finding is either repaired within three rounds or reported as
  undecided in the spec receipt and on the deck's `figure_review_page.html`;
- no deck is approved (`record_human_review_receipt.py`) and no failed
  review is operator-accepted in the runner while any finding is undecided
  (`conference_pipeline.presentation.figure_findings`);
- every outcome is attributed and digest-logged (`figure_specs/*.patch.json`).

Not guaranteed: that the critic finds every semantic error. It is a model
reading a spec against sources, and it has false negatives. "Zero semantic
faults" therefore means zero *undecided* findings plus an operator who read
the review page; it does not mean the figures are proven correct.

## What is measured

12. Per deck, the spec receipt reports: critic calls, figures re-authored,
    open major findings after three rounds, revisions saved by fed-back
    refusals, and brief components dropped. The rollout stop rule stays at a
    first-pass accept rate of 90% or better; below it, the fix goes to the
    brief or the archetype, not to more rounds.
