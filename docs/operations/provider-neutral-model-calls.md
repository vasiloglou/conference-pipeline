# Provider-neutral model calls

A governed task in `src/conference_pipeline/model_governance/registry.yaml` names
which provider/model pairs may do it. Until 2026-10-03 a pair could only be used
if the stage's own code happened to build that provider's SDK client: visual QC
required OpenAI, blueprint, figure-spec and narration stages required Anthropic.
`conference_pipeline.model_governance.llm_client.complete` removes that coupling.

One call shape (system, prompt or messages, optional images, token budget) goes
to OpenAI, OpenRouter, Anthropic or Gemini. Every call is receipted through
`provider_receipts`, and the selection is still checked by `validate_selection`,
so a model the registry has not allowed for the task is refused whatever the
transport. Defaults are unchanged: with nothing set, every stage uses the model
the registry records.

## Choosing a model per stage

A model alone is enough (`openrouter/moonshotai/kimi-k3` resolves to OpenRouter);
a provider alone takes the registry's model for that provider.

| Stage | Registry task | Select with |
|---|---|---|
| Rendered-slide visual QC | `presentation.visual_qc` | `--provider/--model` on `inspect_rendered_slides_llm.py`, or `PRESENTATION_VISUAL_QC_PROVIDER` / `_MODEL` |
| Gate A review panel | `presentation.presentation_review` | `--reviewers` on `review_presentation.py`, or `PRESENTATION_REVIEW_REVIEWERS` |
| Figure quality panel | `presentation.figure_quality_panel` | `FIGURE_PANEL_JUDGES` |
| Figure-spec authoring and critic | `presentation.figure_spec_authoring`, `presentation.figure_spec_critic` | `--model/--provider`, `--critic-model/--critic-provider`, or `FIGURE_SPEC_MODEL` / `_PROVIDER` / `_CRITIC_MODEL` / `_CRITIC_PROVIDER` |
| Editorial blueprint (compiler route) | `presentation.blueprint_generation` | `--provider/--model`, or `EDITORIAL_BLUEPRINT_PROVIDER` / `_MODEL` |
| Presentation blueprint (`generate_presentation_blueprint.py`) | `presentation.blueprint_generation` | `--provider/--model` |
| Engaging narration | `presentation.engaging_narration` | `ENGAGING_NARRATION_PROVIDER` / `_MODEL` |
| Executive topic digests | `presentation.executive_topic_digest` | `--model` |

To change a stage's default for everyone, edit its `model_ref` in the registry.
Registry edits change every receipt digest: batch them before re-running stages,
never mid-run.

## Gate A and the figure panel

* **Gate A members** are CLI tools by default (Claude Code, Codex, Gemini), which
  open the screenshot files themselves. A model the review task allows can join
  as an **API reviewer**: it is sent every screenshot as an image (downscaled to
  `PRESENTATION_REVIEW_API_IMAGE_MAX_DIM`, default 1600 px). A deck with more
  than `PRESENTATION_REVIEW_API_MAX_IMAGES` (default 100) screenshots blocks that
  reviewer rather than being truncated. Example:
  `PRESENTATION_REVIEW_REVIEWERS=claude_code_opus,codex_5_5,openrouter/moonshotai/kimi-k3`.
  An unknown or unallowed token is an error, never a smaller panel. The
  quorum (`PRESENTATION_REVIEW_MIN_REVIEWERS`, default 2) counts API members.
* **Figure panel judges** default to Gemini, Claude and a local Qwen. Any allowed
  model can be seated instead (`FIGURE_PANEL_JUDGES=gemini,claude,openrouter/moonshotai/kimi-k3`);
  a panel that does not seat Qwen never needs the local MLX server. The majority
  rule is a strict majority of the judges seated (two of three, three of four), so
  a tie fails.

## Provider differences the client handles

* `thinking=False` asks a model not to spend its budget on hidden reasoning:
  Anthropic `thinking: disabled`, OpenRouter `reasoning.enabled=false`. Whether a
  given OpenRouter model honours it is that model's behaviour, not verified here.
* `temperature` is not sent to OpenRouter models (some upstreams fix it) nor to
  Claude generations that reject it.
* A truncated or refused reply is normalised to `LLMResult.stop_reason`
  (`max_tokens`, `refusal`) whatever the provider calls it, so callers' existing
  refusal-fallback and larger-budget logic works for every provider.
* `stream=True` matters only for Anthropic at very large `max_tokens`.

## Status of Kimi K3

K3 is allowed (`role: candidate`) for the tasks above. It was measured as a
*reviewer* on 2026-10-03 with `scripts/presentations/trial_review_model.py`
(method below; evidence for ICML 2026 in its context packet). It has **not** been
measured as a generator beyond the one-section blueprint comparison of
2026-10-01 and the 07A summaries.

| Role | Result | Verdict |
|---|---|---|
| Visual-QC reviewer, planted defects | Blatant: 36/36 (same as `gpt-6-sol`). Mild (about a quarter strength): 31/36 vs 30/36. Both near-blind to a 3.5% edge clip (2/6 vs 1/6). | Equal detection to `gpt-6-sol` |
| Visual-QC reviewer, recorded canary decks (52 slides) | Found real text defects `gpt-6-sol` passed (leaked `\texttt`, a raw markdown link, labels cut mid-title, a dangling `&`). Also inflated severity (11 blockers vs 1), made two false positives, and missed the one connector-clutter blocker. | A more sensitive second opinion, not a replacement gate |
| Gate A reviewer (deck 01, K3 alone) | Valid structured review in 114 s for $0.20; failed the deck on a stale narration (3 of 25 slide ids match) and stale deck-contract hashes, all verified. | Works as an API member |
| Figure-panel judge (mild set) | 20/36 detected, 0/6 on edge clip, one answer cut off by the 4000-token cap, $0.028 per call (three times a visual-QC call). | Not recommended as configured |

Method: *replay* re-judges recorded slides and compares with the baseline report;
*seeded* plants known defects in slides the baseline passed (six kinds, at two
strengths) and scores detection against 6 clean controls. The seeded defects are
synthetic, so detection of them is a floor, not proof of quality; the replay is
judged by looking at the renders. Every call is receipted under its own run id and
the trial halts at a spend cap.

## Built into the pipeline (2026-10-03)

The trial's findings are now part of how a deck is built, in four places. Each has a
cost and a reason; none changes a default for a conference that does not opt in.

| What | Where | Cost | Blocks? |
|---|---|---|---|
| **Fix the cause.** A text macro (`\texttt{X}`) is unwrapped, not left to leak its name; a link keeps its text and loses its URL; a source label never ends on a connector; the event kind is the pill's tag, not part of the name. Text already built wrong is repaired at display time. | `presentation/display_text.py`, used by the ledger builder, the chip labeller and the bookend slides | $0 | n/a |
| **Catch what slips through, free.** A static lint over the deck's HTML: leaked LaTeX macro, raw Markdown link or marker, label cut on a connector, template placeholder (errors); figure heading that repeats the slide title (warning). | `presentation/slide_text_lint.py`, run inside `inspect_rendered_slides` before the paid vision pass | $0 | errors block, warnings are recorded |
| **Don't author it.** A figure whose title only repeats its slide's title is sent back for one cheap re-author. | `figures/lint.py::title_repeats_slide_title`, applied in `generate_figure_specs.py` | one re-author | yes, for that slide |
| **A second opinion, per conference.** Gate A can include more reviewers, and a second vision model can run beside the primary judge for recall. | `presentation_review` in the conference's `config.yaml` | about $0.20 per deck (Gate A member), about $0.39 per deck (advisory visual QC), measured | Gate A member: as any member. Advisory QC: never |

On the two ICML canary decks the lint alone finds every text defect the vision model
found (three `textttFlashSchNet` slides, two labels cut on `&`, one raw link) with no
false positive among its six errors, so most of the second vision pass's added value
is available for nothing; what remains is judgement the lint cannot make.

Checked end to end: re-rendering the two canary decks from their existing render specs
with the fixed code (to a throwaway file; no deck file was changed) clears all six lint
errors, and the lint is silent on the result.

### Configuring a conference

```yaml
presentation_review:
  reviewers: [claude_code_opus, codex_5_5, gemini, openrouter/moonshotai/kimi-k3]
  advisory_visual_qc:
    model: openrouter/moonshotai/kimi-k3
```

Absent, a conference keeps the three CLI reviewers and no advisory pass. Precedence for
the panel: `--reviewers`, then `PRESENTATION_REVIEW_REVIEWERS`, then this section. A model
the registry does not allow is an error when used, never a smaller panel. The advisory
pass writes `assets/generated/visual_llm_qc_advisory_report.json` and a compact
`visual_llm_qc_advisory_summary.json`, which Gate A receives as hints to verify on the
render; it is recorded on the deck and cannot fail a stage. It runs in the per-deck
builder (`visual_qc_advisory` stage) and in the production chain's topic-deck path; the
executive-deck path is not covered (the trial did not test it).

### What the integration deliberately does not do

* It does not add an owner script to the stage-owner registry: editing that registry
  changes `script_registry_sha256` and invalidates every earlier receipt. The lint runs
  inside the existing `inspect_rendered_slides` step, and the advisory pass reuses the
  existing `inspect_rendered_slides_llm.py` owner script.
* It does not make K3 the only gate or a figure-panel judge; see below.

## Using K3 well

* As a **reviewer, add it, do not swap it in alone**: its severity is inflated
  (style nits marked blocker), so treat its findings as triage input or calibrate
  severity before it can hard-fail a deck.
* Prefer the free lint for what a string can show; use K3 for judgement the lint cannot
  make (an unlabelled branch, a stale narration, a figure that says the wrong thing).

## Not yet converted

Stages that still build a provider SDK client directly, mostly the legacy
(non-compiler) route: `add_bigpicture_diagrams.py`, `add_long_diagrams.py`,
`add_teaching_anecdotes.py`, `enrich_diagram_axes.py`, `enrich_hyperlinks_agent.py`,
`enrich_slides_with_papers.py`, `extract_figures_agent.py`, `sanity_check_slides.py`,
`reflow_slides.py`, `generate_programmatic_figures.py`, `generate_executive_narration.py`,
`generate_executive_dialogue.py`, `run_figure_trial.py` and `replay_figure_trial.py`.
They still work as before; moving one onto `complete` is the pattern in
`generate_figure_specs.py::_call_model`.

The receipt-boundary guard (`tests/test_provider_receipt_static_guard.py`) still
reports four OpenRouter call sites that predate the wrapper
(`openrouter_chat_completions_create`): `exhaustive_summaries.py`,
`local_summary_benchmark.py`, `compare_summary_models.py`,
`compare_section_models.py`.
