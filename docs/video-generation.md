# Video Generation

This document covers the single-presenter and NotebookLM-style video generation
pipelines synced to Reveal.js slide presentations.

## Overview

The single-presenter pipeline takes `6_presenter_script.md` and writes
`video_build/script.json` for one presenter voice. The NotebookLM pipeline then
uses the same source script to produce a two-host dialogue in
`6_presenter_script_notebooklm_2host.md` and
`video_build_notebooklm/script.json`.

**Single-presenter script:** `scripts/presentations/build_presenter_video.py`

**NotebookLM script:** `scripts/presentations/build_notebooklm_style_presenter_video.py`

## Prerequisites

| Dependency | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Environment variable for Gemini API access |
| `ffmpeg` | Audio/video assembly |
| `npx playwright` | Slide screenshot capture |
| `google-genai` | Python SDK for Gemini (dialogue + TTS) |
| `python-dotenv` | Environment loading |

Install Python deps:
```bash
pip install google-genai python-dotenv
npx playwright install chromium
```

## Provider Cost Estimate

Dialogue generation and TTS are paid provider work; Playwright screenshots and
`ffmpeg` assembly are local-only `$0 API cost` work. Estimate video costs before
approval with:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope video-generation \
  --scenario base
```

The command reads `src/conference_pipeline/costs/pricing_catalog.json` and
`src/conference_pipeline/costs/task_specs.yaml`; final invoices should be
reconciled from Gemini/OpenAI receipts.

## Pipeline Stages

The video build script runs the media stages sequentially. Production runs
should add the narration lint/review gates before TTS so weak scripts fail
before paid audio/video work.

### 0. Single-Presenter Script JSON And Review

Before rendering single-presenter TTS, generate or refresh only the TTS JSON:

```bash
python3 scripts/presentations/build_presenter_video.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script.md \
  --slides-html conferences/<slug>/decks/<deck>/presentation.html \
  --slide-mapping conferences/<slug>/decks/<deck>/presentation_slide_mapping.json \
  --build-dir conferences/<slug>/decks/<deck>/video_build \
  --output conferences/<slug>/decks/<deck>/presentation_video.mp4 \
  --stop-after-script-json
```

This mode writes `video_build/script.json` and exits before audio, screenshots,
or video assembly. The JSON contract is strict: segment text may contain spoken
SAY text and clean TRANSITION text only. It must not contain `**CONTEXT**`,
`**REFS**`, `not spoken`, `[arXiv]`, source metadata, or stage directions such
as `Open with...`.

Run the single-presenter review gate before TTS:

```bash
python3 scripts/presentations/lint_presenter_narration.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script.md \
  --script-json conferences/<slug>/decks/<deck>/video_build/script.json \
  --deck-kind topic \
  --conference-label "ICLR 2026" \
  --report-out conferences/<slug>/decks/<deck>/presenter_narration_lint_report.json

python3 scripts/presentations/review_presenter_narration.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script.md \
  --script-json conferences/<slug>/decks/<deck>/video_build/script.json \
  --lint-report conferences/<slug>/decks/<deck>/presenter_narration_lint_report.json \
  --deck-kind topic \
  --review-out conferences/<slug>/decks/<deck>/presenter_narration_review.md \
  --json-out conferences/<slug>/decks/<deck>/presenter_narration_review.json
```

For series-level single-presenter review, reuse the cross-deck checker with the
single-presenter script name:

```bash
python3 scripts/presentations/review_narration_series.py \
  --decks-dir conferences/<slug>/decks \
  --script-name 6_presenter_script.md \
  --conference-label "ICLR 2026" \
  --report-out conferences/<slug>/decks/presenter_narration_series_review.json \
  --markdown-out conferences/<slug>/decks/presenter_narration_series_review.md
```

Single-presenter TTS is blocked unless every deck has
`presenter_narration_review.json` with `approve_for_tts=true` and the
single-presenter series review has no unaccepted failure. For executive
summary decks, the review also rejects narration that sounds assembled from
bullets, especially repeated stitch phrases such as `That connects directly
to`, `The operational test is`, or `That sets up`; the source should read like
authored synthesis prose. Topic closings must not contain audible grammar joins
such as `The next step is We need...`, and long executive summaries should use
enough transition variety that the same bridge form does not recur every few
slides.

### 1. Parse Presenter Script

Reads `6_presenter_script.md` and extracts per-slide blocks containing:
- **SAY** — What the presenter says (main content)
- **CONTEXT** — Background knowledge bullets
- **TRANSITION** — How to bridge to the next slide
- **FIGURE** — Description of the visual on screen (if any)

Block values may be either inline or on following lines. For example, both
`**TRANSITION:** "Next idea..."` and a multi-line `**TRANSITION:**` block are
valid and must be preserved into the two-host artifact.

### 2. Dialogue Transformation (Gemini 2.5 Pro)

Each slide's single-host content is transformed into a multi-turn dialogue between two hosts using the selected prompt style. The dialogue model generates JSON with alternating speaker turns.

**Prompt styles** (selectable via `--prompt-style`):

| Style | Description |
|---|---|
| `podcastfy` | **Default.** Warm narrative lead + analytical partner pattern, paced questions, concrete stakes, and explicit anti-repetition/question-ending guidance. |
| `google` | Reverse-engineered Google NotebookLM style. Enthusiastic guide + analytical voice, hook→explore→relevance→recap structure. |
| `open-notebooklm` | Based on gabrielchua/open-notebooklm. Host interviews guest, verbal fillers, brainstorm-then-refine, shorter lines. |
| `default` | Custom prompt with explicit anti-repetition rules. |

Zoom companion slides (figure deep-dives) are handled separately with a focused 2-turn, 200-character-max prompt.

**Caching:** Generated dialogues are cached per-slide in `build_dir/`. Re-runs skip already-generated slides unless `--force-transform` is used. When `--skip-transform` reuses an existing `video_build_notebooklm/script.json`, the human-readable `6_presenter_script_notebooklm_2host.md` artifact is regenerated from that JSON and validated for actual host dialogue markers.

### 3. Narration Quality Gate

Run deterministic lint first:

```bash
python3 scripts/presentations/lint_narration_script.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script_notebooklm_2host.md \
  --source-script-md conferences/<slug>/decks/<deck>/6_presenter_script.md \
  --deck-kind topic \
  --conference-label "ICLR 2026" \
  --report-out conferences/<slug>/decks/<deck>/narration_lint_report.json
```

Then run the editorial review gate:

```bash
python3 scripts/presentations/review_narration_script.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script_notebooklm_2host.md \
  --source-script-md conferences/<slug>/decks/<deck>/6_presenter_script.md \
  --lint-report conferences/<slug>/decks/<deck>/narration_lint_report.json \
  --deck-kind topic \
  --review-out conferences/<slug>/decks/<deck>/narration_review.md \
  --json-out conferences/<slug>/decks/<deck>/narration_review.json
```

The lint gate checks measurable defects: missing host markers, stale `Host A/B`
stage directions, dropped transitions, missing executive bridges,
truncated/hanging fragments, repeated `Yes, and`, repeated
`Exactly.`/`Precisely.` openers, repeated "pay closest attention" questions,
question density, slide-ending question hooks, exclamation density, and
conference-label repetition. The review gate turns those signals into
`approve_for_tts`, accuracy blockers, listener blockers, 1-5 quality scores,
and a revision brief.

Passing means the script is not only valid. It should be accurate, paced,
specific, pleasant, and natural without becoming mechanical. Scripts that fail
review should be revised or regenerated before TTS.

After every deck-level script review passes, run the series-level narration
review. This catches cross-deck repetition that a per-script check cannot see,
such as the same figure-detail chart narration being reused across topic decks:

```bash
python3 scripts/presentations/review_narration_series.py \
  --decks-dir conferences/<slug>/decks \
  --conference-label "ICLR 2026" \
  --report-out conferences/<slug>/decks/narration_series_review.json \
  --markdown-out conferences/<slug>/decks/narration_series_review.md
```

The series review compares shared figure-detail slide IDs, normalized figure
titles, and repeated technical phrases across all decks. Failed series review
should be fixed before TTS unless the operator explicitly accepts the remaining
series-level issue, for example when a shared audio segment is intentional.

When the series review finds cross-deck phrase repetition, run the series fixer
and then rerun both deck-level and series-level reviews:

```bash
python3 scripts/presentations/fix_narration_series.py \
  --decks-dir conferences/<slug>/decks \
  --report-out conferences/<slug>/decks/narration_series_fix_report.json
```

The series fixer only edits figure-detail narration and synchronizes
`6_presenter_script_notebooklm_2host.md` with
`video_build_notebooklm/script.json` across the deck set.

When the issue is a localized script defect rather than a full regeneration
problem, use the fix helper and rerun review:

```bash
python3 scripts/presentations/fix_narration_script.py \
  --script-md conferences/<slug>/decks/<deck>/6_presenter_script_notebooklm_2host.md \
  --script-json conferences/<slug>/decks/<deck>/video_build_notebooklm/script.json \
  --deck-kind topic \
  --conference-label "ICLR 2026" \
  --conference-label-limit 12 \
  --replacement-json narration_replacements.json \
  --report-out conferences/<slug>/decks/<deck>/narration_fix_report.json
```

The fix helper updates both the human-readable Markdown and the TTS-driving
`script.json`. It can also thin repeated conference-label mentions so the audio
keeps the conference frame without sounding like every slide is re-announcing
the event. Do not render TTS from a stale JSON after editing Markdown.

### 4. TTS Audio Synthesis (Gemini TTS)

Each dialogue turn is synthesized to audio using Gemini's Chirp 3 HD voices. Long turns are chunked (default 360 chars max per TTS call) and concatenated.

**Default voices:**

| Host | Voice | Character |
|---|---|---|
| Iapetus | Iapetus | Primary host / summarizer |
| Erinome | Erinome | Secondary host / questioner |

All 30 Gemini Chirp 3 HD voices are supported. See [Gemini TTS docs](https://ai.google.dev/gemini-api/docs/text-to-speech) for the full list.

**Caching:** Audio segments are cached per-turn in `build_dir/audio/`. Re-runs skip existing segments unless `--force-transform` is used.

### 5. Slide Screenshot Capture (Playwright)

Screenshots are captured from the Reveal.js HTML presentation using Playwright. The `?hidecontrols` query parameter is automatically appended to suppress navigation arrows in screenshots.

**Capture modes:**
- `parallel` (default) — Multiple Playwright processes, fastest
- `batch` — Single browser session (requires local Node playwright module)
- `single` — Sequential, slowest but most reliable

**Caching:** Screenshots are cached in `build_dir/slides/`. Use `--skip-slides` to reuse.

### 6. Video Assembly (ffmpeg)

All audio segments and slide images are assembled into a final MP4:
1. Per-slide audio segments are concatenated into slide groups
2. Each slide image is displayed for the duration of its audio
3. A narration audio track is exported separately, then the slide video + narration are combined
4. Optional speaking rate adjustment is applied via ffmpeg `atempo` filter

## Usage

### Basic (single topic)

```bash
python3 scripts/presentations/build_notebooklm_style_presenter_video.py \
  --script-md conferences/neurips2025/decks/01_tools_and_open_source/6_presenter_script.md \
  --slides-html conferences/neurips2025/decks/01_tools_and_open_source/presentation.html \
  --build-dir conferences/neurips2025/decks/01_tools_and_open_source/video_build_notebooklm \
  --output conferences/neurips2025/decks/01_tools_and_open_source/presentation_video_notebooklm.mp4
```

### With speaking rate adjustment

```bash
python3 scripts/presentations/build_notebooklm_style_presenter_video.py \
  --script-md ... --slides-html ... --build-dir ... --output ... \
  --speaking-rate 1.15
```

### Partial re-run (reuse cached dialogue + audio)

```bash
python3 scripts/presentations/build_notebooklm_style_presenter_video.py \
  --script-md ... --slides-html ... --build-dir ... --output ... \
  --skip-transform --skip-audio
```

### Narration-only generation

Use this mode when you want the two-host narration artifacts for review but do
not want TTS, slide capture, or video assembly to start:

```bash
python3 scripts/presentations/build_notebooklm_style_presenter_video.py \
  --script-md ... --slides-html ... --build-dir ... \
  --podcast-script-out .../6_presenter_script_notebooklm_2host.md \
  --stop-after-transform
```

This writes or refreshes `6_presenter_script_notebooklm_2host.md` and
`video_build_notebooklm/script.json`, then exits before any audio provider calls.

### Debug first N slides

```bash
python3 scripts/presentations/build_notebooklm_style_presenter_video.py \
  --script-md ... --slides-html ... --build-dir ... --output ... \
  --max-slides 20
```

### Using the orchestrator (Execute_Presentation_Plan.md)

From the presentation pipeline orchestrator, request Step 10:
```
Run step 10 for topic 01
```

## CLI Reference

### Input/Output

| Flag | Default | Description |
|---|---|---|
| `--script-md` | Master presentation script | Presenter script markdown path |
| `--slides-html` | Master presentation HTML | Reveal.js HTML path |
| `--build-dir` | `video_build_notebooklm/` | Working directory for intermediates |
| `--output` | `presentation_video_notebooklm.mp4` | Output MP4 path |
| `--podcast-script-out` | None | Export generated 2-host markdown script |

### Models & Voices

| Flag | Default | Description |
|---|---|---|
| `--generation-model` | `gemini-2.5-pro` | Gemini model for dialogue transformation |
| `--audio-model` | `gemini-2.5-pro-preview-tts` | Gemini model for TTS |
| `--host-a` | `Iapetus` | Primary host display name |
| `--host-b` | `Erinome` | Secondary host display name |
| `--voice-a` | `Iapetus` | TTS voice for host A |
| `--voice-b` | `Erinome` | TTS voice for host B |

### Dialogue Controls

| Flag | Default | Description |
|---|---|---|
| `--prompt-style` | `podcastfy` | Prompt style: `podcastfy`, `google`, `open-notebooklm`, `default` |
| `--turns-per-slide` | `2` | Number of dialogue turns per slide |
| `--max-turn-chars` | `420` | Max characters per dialogue turn |
| `--max-context-items` | `6` | Max CONTEXT bullets used per slide |

### TTS Controls

| Flag | Default | Description |
|---|---|---|
| `--max-tts-chars` | `360` | Max characters per TTS chunk |
| `--max-tts-retries` | `8` | Retry attempts per TTS chunk |
| `--speaking-rate` | `1.0` | Playback speed multiplier (via ffmpeg atempo) |

### Parallelism

| Flag | Default | Description |
|---|---|---|
| `--dialogue-workers` | `4` | Parallel dialogue generation workers |
| `--tts-workers` | `6` | Parallel TTS synthesis workers |
| `--duration-workers` | `8` | Parallel ffprobe duration lookup workers |
| `--slide-workers` | `4` | Parallel Playwright screenshot workers |

### Caching & Skip Flags

| Flag | Description |
|---|---|
| `--skip-transform` | Reuse cached `script.json` dialogue |
| `--stop-after-transform` | Write or refresh two-host narration artifacts, then exit before TTS/video |
| `--skip-audio` | Reuse cached audio segments |
| `--skip-slides` | Reuse cached slide screenshots |
| `--force-transform` | Regenerate dialogue even if cached |

### Slide Capture

| Flag | Default | Description |
|---|---|---|
| `--slide-capture-mode` | `parallel` | Capture strategy: `parallel`, `batch`, `single` |
| `--viewport` | `1280,720` | Screenshot viewport dimensions |
| `--wait-ms` | `1200` | Wait time per screenshot for rendering |
| `--slide-sync-mode` | `auto` | Slide mapping: `auto` (handles zoom companions), `linear` |
| `--zoom-slide-class` | `slide-figure-zoom` | CSS class marking zoom companion slides |

### Duration Matching

| Flag | Default | Description |
|---|---|---|
| `--duration-reference-script-json` | None | Reference script.json for per-slide character budgets |
| `--char-budget-tolerance` | `0.12` | Allowed deviation from character target (+/- 12%) |
| `--char-budget-scale` | `1.0` | Scale factor for character budgets |

### Debug

| Flag | Description |
|---|---|
| `--max-slides` | Cap on number of slides to process |
| `--max-generation-retries` | Retry attempts per slide dialogue (default: 5) |

## Output Files

| File | Description |
|---|---|
| `presentation_video_notebooklm.mp4` | Final synced video |
| `6_presenter_script_notebooklm_2host.md` | Generated 2-host dialogue script (markdown); must contain the configured host markers and must not contain `Host A:`/`Host B:` stage directions |
| `video_build_notebooklm/script.json` | Full dialogue with timing data |
| `narration_lint_report.json` | Deterministic narration lint metrics and fail/warn findings |
| `narration_review.md` / `narration_review.json` | Editorial approval gate with TTS decision and revision brief |
| `video_build_notebooklm/audio/` | Cached TTS audio segments |
| `video_build_notebooklm/slides/` | Cached slide screenshots |
| `video_build_notebooklm/concat/` | Per-slide ffmpeg concat files |

## Troubleshooting

### TTS fails with "Failed to synthesize audio"
- Verify `GOOGLE_API_KEY` is set and has Gemini API access
- Check that the voice names are valid Gemini Chirp 3 HD voices
- Reduce `--tts-workers` if hitting rate limits

### Screenshots missing navigation arrows (expected)
- The `?hidecontrols` query parameter is appended automatically
- The HTML must include the `hideControls` JS check (rebuild HTML with `md_to_revealjs.py` if missing)

### Pipeline stuck on dialogue generation
- Check for rate limits; reduce `--dialogue-workers`
- Use `--max-generation-retries 2` to fail faster on problematic slides
- Re-run — cached slides will be skipped automatically

### Video too long/short
- Adjust `--speaking-rate` (1.15 = 15% faster, 0.9 = 10% slower)
- Use `--duration-reference-script-json` with a known-good script.json for duration matching
- Adjust `--max-turn-chars` to control dialogue verbosity
