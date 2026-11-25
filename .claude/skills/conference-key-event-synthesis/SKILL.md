---
name: conference-key-event-synthesis
description: Select top conference-program events for conference presentation bookends by merging deterministic deck/program evidence with GPT-researcher talk/transcript findings.
---

# Conference Key Event Synthesis

Use this skill when choosing, auditing, or repairing the "Relevant events" slide
for conference presentation decks, especially when workshops, keynotes, oral
talks, panels, invited talks, or transcript-backed workshop talks may be more
important than the most-cited papers.

## Workflow

1. Generate deterministic top-5 candidates and GPT-researcher prompts:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/presentations/synthesize_key_events.py \
     --workspace conferences/iclr2026 \
     --topic all \
     --top-n 5
   ```

   When rerunning after operator feedback or after prior event overrides have
   already been written, add `--ignore-required-events` and use a larger
   `--top-n` for replacement search. This prevents stale slide-2 choices from
   biasing the new deck-resonance audit.

2. For each topic that needs a second opinion, open:

   ```text
   conferences/iclr2026/decks/<topic>/key_event_synthesis/gpt_researcher_prompt.md
   ```

   Run that prompt through the approved GPT-researcher lane only when budget
   allows. Prefer local/loopback GPT-researcher-style synthesis when available;
   before any live provider command, source the repo credential file required by
   `AGENTS.md` and reconcile provider receipts against the active spend cap.

3. Save GPT-researcher output as JSON or Markdown, then merge it:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/presentations/synthesize_key_events.py \
     --workspace conferences/iclr2026 \
     --topic <topic> \
     --top-n 5 \
     --gpt-researcher-result conferences/iclr2026/decks/<topic>/key_event_synthesis/gpt_researcher_result.json
   ```

4. Review the synthesis and deck-resonance audit:

   ```text
   conferences/iclr2026/decks/<topic>/key_event_synthesis/key_event_synthesis.json
   ```

   The output must state whether GPT-researcher evidence was actually loaded or
   only staged. Do not claim GPT-researcher agreement when the status is
   `staged_no_gpt_researcher_result`.

   Also inspect each selected event's `deck_resonance` block. The slide-2 event
   list must be in harmony with the deck's actual thesis, section sequence,
   source chips, and claims. Broad topic overlap is not enough.

   - `deck_resonance.action = keep`: event has slide/source support and may
     remain in the candidate list.
   - `deck_resonance.action = review_or_replace`: do not promote this event
     without explicit editorial approval or a documented replacement search.
   - `deck_resonance_gate.weak_top_key_events` must be empty before rebuilding
     narration or videos.

5. After human/editorial approval, convert the accepted top events into
   `<deck>/bookend_event_overrides.json` `required_events`, rerun
   `add_bookend_slides.py`, then rebuild PDFs before any narration or video
   regeneration.

## Selection Standard

- Search the full available conference program cache, not only deck-local source
  chips.
- Mine deck-local `research_cluster_plan.json` event annotations as well as
  `render_spec.json`, program HTML, workshop inventory, and GPT-researcher
  outputs. Transcript-backed labels such as `workshop keynote:` and
  `workshop talk:` are first-class event evidence.
- Treat workshops, keynotes, invited talks, oral sessions, spotlights, panels,
  tutorials, competitions, and transcript-backed workshop talks as event
  candidates.
- Promote conference-program salience in the ranking. Keynotes, workshops,
  competitions, oral sessions/talks, invited talks, spotlights, and panels
  should outrank paper-like source mentions when they also have topic and
  deck-story support.
- Do not let coverage repair demote a higher-priority event kind, such as a
  keynote, merely to add a lower-priority workshop for another strand.
- Use transcript substance to decide importance: concrete argument, framework,
  benchmark, controversy, or field-level framing beats generic schedule
  presence.
- Require event/deck resonance: each event should map to supporting deck slides
  or source chips, or be replaced with an event that better explains the deck's
  story.
- For broad topic decks, require coverage balance across the deck's major
  strands. Do not let the densest source-chip cluster crowd out a visible
  workshop/keynote/oral event for another section-level strand, such as the
  materials workshop inside the AI-for-science deck.
- Keep papers on the paper rail unless the program/transcript evidence makes
  the item an actual event.
- Render only audience-facing event details on slide 2: `Keynote`, `Workshop
  keynote`, `Workshop talk`, `Oral session`, `Oral talk`, `Invited talk`,
  `Panel`, `Workshop`, `Competition`, and similar labels. Never show internal
  pipeline phrases such as `Conference program candidate` or `Workshop in
  acquired ICLR program inventory`.
