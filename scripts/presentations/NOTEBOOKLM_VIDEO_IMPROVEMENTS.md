# NotebookLM-Style Video Pipeline: Research & Improvement Plan

## Current State

The `build_notebooklm_style_presenter_video.py` script generates 2-host dialogue videos from presenter scripts. It works but uses a minimal dialogue generation prompt that lacks the polish and natural feel of Google's NotebookLM or its best open-source replications.

### Current Pipeline Flow

```
6_presenter_script.md ──► Gemini 2.5 Pro (per-slide dialogue transform)
                              │
                              ▼
                      script.json (2-host segments)
                              │
                              ▼
                      Gemini TTS (dual voice: Puck / Kore)
                              │
                              ▼
                      audio_segments/*.mp3
                              │
presentation.html ──► Playwright screenshots ──► slides/*.png
                              │                        │
                              ▼                        ▼
                      ffmpeg concat ──────────► presentation_video_notebooklm.mp4
```

### Current Dialogue Prompt (lines 732-786)

A bare-bones per-slide transformer:
- "Transform this presenter content into a natural 2-host deep-dive dialogue"
- Generic "Host A" / "Host B" with no persona definitions
- No verbal fillers, no warmth, no natural speech patterns
- No analysis or brainstorming step before generating dialogue
- No "aha moments" or listener engagement hooks
- Strictly JSON output with rigid turn count + char limits

---

## Reference Implementations

### Google NotebookLM (reverse-engineered)

Source: [Reverse-engineered prompt analysis](https://baoyu.io/blog/google-notebooklm-system-prompt-en)

Key design principles:
- **Dual voices**: Enthusiastic storyteller + calm analyst
- **"Aha moments"**: Focus on making content actionable and surprising
- **Strict source adherence**: All facts from provided materials only
- **Structure**: Warm intro → alternating voices → practical relevance → recap → reflective question
- **Listener-facing**: Direct address ("you"), as if talking to someone
- **~5 minute target**: Tight, efficient coverage

### Open-NotebookLM (gabrielchua/open-notebooklm)

Source: [GitHub](https://github.com/gabrielchua/open-notebooklm) / [prompts.py](https://github.com/gabrielchua/open-notebooklm/blob/main/prompts.py)

Key design principles:
- **"World-class podcast producer"** role assignment
- **Multi-step pipeline**: Analyze → brainstorm creative angles → craft dialogue → weave insights → ensure authenticity
- **Defined personas**: Jane (host/interviewer) + guest (expert)
- **Natural speech**: Verbal fillers ("um," "well," "you know"), natural interruptions
- **100 chars per dialogue line** (tight for TTS pacing)
- **PG-rated**, no marketing content
- **Host always initiates** and interviews the guest

### Podcastfy

Source: [GitHub](https://github.com/souzatharsis/podcastfy)

Key design principles:
- Two-host with defined roles: expert + curious beginner
- Natural interruptions and back-and-forth
- Configurable tone, language, and length (short/medium/long)
- Multilingual support

---

## Identified Problems

### 1. Flat, Generic Dialogue

**Problem**: Hosts have no personality. Output sounds like two text-to-speech bots reading facts at each other.

**Solution**: Define rich personas with distinct roles:
- **Host A ("Alex")**: The curious explainer — sets up topics with hooks, asks "why does this matter?", uses accessible language, expresses genuine surprise
- **Host B ("Jordan")**: The analytical expert — provides depth, challenges assumptions, connects dots across slides, delivers the "aha moment"

### 2. No Verbal Naturalness

**Problem**: Every turn is a perfectly formed sentence. Real podcasts have fillers, false starts, and reactive language.

**Solution**: Instruct the model to include:
- Brief reactive interjections ("Right.", "Exactly.", "Oh, that's interesting.")
- Occasional verbal bridges ("So here's the thing...", "And what really stood out to me...")
- Natural turn-taking cues (one host finishing the other's thought)

### 3. No Brainstorming / Analysis Step

**Problem**: The model jumps straight from raw SAY/CONTEXT to JSON dialogue. No "thinking" about what's interesting.

**Solution**: Add a two-phase prompt:
1. **Phase 1 (internal scratchpad)**: "What's the most interesting angle on this slide? What would surprise the listener? What's the 'so what'?"
2. **Phase 2 (dialogue)**: Generate the dialogue informed by the scratchpad

### 4. Image/Figure Zoom Not Integrated

**Problem**: When a slide has a figure (SVG), the video currently handles zoom companion slides visually (showing main + zoom screenshots), but the dialogue doesn't describe or reference the figure content. The hosts talk about abstract concepts while a chart is on screen.

**Solution**: Feed figure metadata into the dialogue prompt. The presenter script's CONTEXT block often describes figures, but we should also:
- Parse `5_draft_slides_with_images.md` for `![caption](path)` references per slide
- Include figure captions in the dialogue prompt
- Add explicit instruction: "When a figure is on screen, one host should briefly describe what the audience is seeing"

### 5. Script Generated Before HTML (Ordering Problem)

**Problem**: Currently the presenter script (`6_presenter_script.md`) is generated at Step 8, before the HTML presentation is built (Step 7). The NotebookLM video script should ideally be generated *after* the HTML + PDF exist, so:
- We know exactly which slides have zoom companions
- We can reference rendered slide content
- The slide mapping JSON is available

**Solution**: Step 10 (Video Generation) already runs after Step 7 (HTML build). The dialogue transformation happens inside `build_notebooklm_style_presenter_video.py`, which already takes `--slides-html` as input. We should enhance it to:
1. Read `presentation_slide_mapping.json` to understand which markdown slides map to which HTML sections
2. Read `5_draft_slides_with_images.md` to extract figure captions per slide
3. Pass figure descriptions into the per-slide dialogue prompt
4. For slides with zoom companions, instruct hosts to call out the visual: "If you look at the chart..."

### 6. Model Selection

**Problem**: The script defaults to `gemini-2.5-pro` for dialogue script generation. Gemini 3 Pro is now available and better suited for this task.

The pipeline has two distinct model calls — the prompt improvements only affect step 1:

| Step | Purpose | Current Model | Recommended Model |
|------|---------|---------------|-------------------|
| **1. Script generation** (LLM) | Transform presenter notes → 2-host dialogue JSON | `gemini-2.5-pro` | `gemini-3-pro-preview` |
| **2. Audio synthesis** (TTS) | Read dialogue text aloud with two voices | `gemini-2.5-pro-preview-tts` | `gemini-2.5-pro-preview-tts` (no change) |

All prompt improvements (personas, verbal fillers, figure awareness, brainstorming) apply to **step 1 only**. Step 2 (TTS) is pure text-to-speech with no prompt — it just reads whatever text the LLM generated.

**Available models for script generation (as of Feb 2026):**

| Model ID | Notes |
|----------|-------|
| `gemini-3.1-pro-preview` | Best reasoning and instruction following. Use this for base/high script generation. |
| `gemini-3.1-flash-lite-preview` | Fast, cost-effective alternative for draft scripts and ranking-style passes. |
| `gemini-2.5-pro` | Historical default. Stable but weaker at complex persona instructions. |

**Recommendation**: Use `gemini-3.1-pro-preview` for script generation.
Use `gemini-2.5-flash-preview-tts` for base video narration and
`gemini-2.5-pro-preview-tts` only for high-quality narration.

---

## Proposed Improvements

### Option A: Enhanced Prompt (Minimal Code Changes)

Replace the `slide_prompt()` function's prompt text with a richer version inspired by Open-NotebookLM and the reverse-engineered Google prompt. No structural code changes.

**Changes:**
- Rich persona definitions for both hosts
- Multi-step prompting (scratchpad → dialogue)
- Natural speech instructions (fillers, reactive language)
- Figure-aware instructions when a slide has visuals
- "Aha moment" targeting

**Effort**: Small (prompt text only)
**Risk**: Low
**Impact**: Significant improvement in dialogue quality

### Option B: Figure-Aware Pipeline (Medium Code Changes)

Everything in Option A, plus:
- Parse `5_draft_slides_with_images.md` to extract figure captions per slide number
- Pass figure descriptions into the dialogue prompt
- Use `presentation_slide_mapping.json` for accurate slide-to-section mapping (replacing the current CSS class-based approach)
- Add instruction for hosts to reference on-screen visuals

**Changes:**
- New function: `extract_figure_captions(slides_md_path) → dict[int, list[str]]`
- New function: `load_slide_mapping(html_path) → dict[int, int]`
- Modified `slide_prompt()` to include figure descriptions
- Modified `build_logical_to_rendered_slide_map()` to prefer the JSON mapping

**Effort**: Medium
**Risk**: Low
**Impact**: Videos feel more connected to the visual content on screen

### Option C: Full Podcast Pipeline (Larger Refactor)

Everything in Options A + B, plus:
- Two-phase generation: first generate a full-presentation outline/arc, then per-slide dialogue (so hosts can foreshadow and reference earlier slides)
- Intro and outro generation (warm opening, closing summary with key takeaways)
- Cross-slide awareness: each slide prompt includes brief context about what was covered on the previous 2 slides
- Support for `gemini-3-pro-preview` as default model
- Configurable podcast "style" presets:
  - `deep-dive`: Technical, detailed (current default)
  - `executive`: High-level, business implications focus
  - `casual`: Light, curious, more verbal fillers

**Changes:**
- New two-pass architecture: outline pass → per-slide pass
- Intro/outro slide handling
- Style preset system
- Previous-slide context window
- Model upgrade

**Effort**: Large
**Risk**: Medium (more complex, harder to debug)
**Impact**: Professional podcast quality, significant uplift

---

## Implementation Priority

**Recommended order:**

1. **Option A first** — Biggest bang for least effort. Swap the prompt, test on executive summary.
2. **Option B next** — Makes figures feel integrated instead of ignored.
3. **Option C later** — Only if the podcast quality gap is still noticeable after A + B.

---

## Zoom Companion Slide Handling

### Current Behavior

The script already handles zoom companions via `--slide-sync-mode auto`:
1. Parses all `<section>` tags from `presentation.html`
2. Identifies sections with class `slide-figure-zoom` as zoom companions
3. Groups each non-zoom section with its following zoom companion(s)
4. Splits narration time proportionally across main + zoom screenshots

### What's Missing

The dialogue has no awareness that a figure is being shown. Hosts discuss abstract concepts while a detailed chart fills the screen.

### Proposed Fix

1. Extract figure captions from `5_draft_slides_with_images.md` per slide
2. Include in the prompt: "This slide displays a figure: [caption]. One host should briefly describe what the audience is seeing."
3. The existing zoom screenshot timing is fine — we just need the dialogue to reference the visual

---

## Model Upgrade Path

### Dialogue Generation

```python
# Current
DEFAULT_GEN_MODEL = "gemini-2.5-pro"

# Recommended upgrade
DEFAULT_GEN_MODEL = "gemini-3-pro-preview"
```

Gemini 3 Pro has significantly improved reasoning and instruction following, which should produce more natural dialogue with fewer retries and better adherence to the persona/style instructions.

### TTS

```python
# Current
DEFAULT_AUDIO_MODEL = "gemini-2.5-pro-preview-tts"

# No Gemini 3 TTS model available yet — keep as-is
```

The TTS model is still `gemini-2.5-pro-preview-tts` — there's no Gemini 3 TTS model listed in the API docs yet. Keep the current default.

---

## Pipeline Ordering Clarification

The current pipeline order is correct for video generation:

```
Step 7:  Build presentation.html (+ presentation_slide_mapping.json)
Step 8:  Generate 6_presenter_script.md
Step 10: Generate NotebookLM video (reads both HTML + presenter script)
```

The video script is generated *inside* Step 10, not at Step 8. Step 10 already has access to:
- `presentation.html` (for slide screenshots + section parsing)
- `presentation_slide_mapping.json` (for accurate slide mapping)
- `6_presenter_script.md` (source dialogue content)
- `5_draft_slides_with_images.md` (could be read for figure captions — not yet implemented)

The pipeline ordering is correct. The enhancement needed is for Step 10 to actually *use* the slide mapping JSON and figure captions, rather than relying solely on CSS class parsing.
