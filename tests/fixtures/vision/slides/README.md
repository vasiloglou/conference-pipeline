# Vision Fixture Pack — `tests/fixtures/vision/slides/`

This directory holds the v1.61 Phase 69 slide-image fixture pack used to
exercise the vision benchmark harness. The directory is empty in source
control: the actual slide images and ground-truth annotations are
authored manually (or sourced from a permissively licensed conference
deck) and added per release.

## Required slide categories

The harness expects at least one fixture per category below. Each
fixture is a pair `<id>.png` (or `.jpg`/`.webp`) plus `<id>.gt.json`.
PNG is preferred because it preserves text rendering quality.

| Category id | Description | Primary task it scores |
|---|---|---|
| `text_heavy` | Mostly title + bullet text, minimal graphics. | `vision.slide_analysis`, `ocr.slide_image_parse` |
| `figure_bearing` | Slide with one or more named figures (Fig. 1, Fig. 2). | `vision.figure_caption`, `vision.slide_analysis` |
| `chart_line` | Line chart with axis labels and units. | `vision.chart_summary` |
| `chart_bar` | Bar chart with category labels and units. | `vision.chart_summary` |
| `chart_scatter` | Scatter plot with axis labels and a fitted line. | `vision.chart_summary` |
| `code_slide` | Slide showing a code snippet (preserves indentation). | `vision.slide_analysis` |
| `equation_slide` | Slide showing one or more equations. | `vision.slide_analysis` |
| `table_slide` | Slide with a tabular layout. | `vision.slide_analysis`, `ocr.slide_image_parse` |
| `mixed` | Slide combining text + figure + chart + code. | All vision tasks |

## Ground-truth annotation schema (`<id>.gt.json`)

```jsonc
{
  "fixture_id": "chart_bar_001",
  "category": "chart_bar",
  "expected_text": "...full text reading order, used for OCR/text recall...",
  "expected_salient_claims": [
    "claim 1 verbatim",
    "claim 2 verbatim"
  ],
  "expected_figure_captions": [
    "Figure 1 — caption verbatim with citation/provenance preserved"
  ],
  "expected_chart_summary": "Axes: <x-axis-label> vs <y-axis-label>. Trend: ...",
  "expected_headings": ["Title text"],
  "license": "CC-BY-4.0",
  "source_url": "https://..."
}
```

All annotations must preserve provenance (paper titles, citations, figure
numbers, session names) verbatim where present, per the v1.61
governance contract.

## How fixtures get exercised

1. The benchmark harness iterates registered candidates from
   `registry.yaml` `local_candidates`.
2. For each candidate × fixture, the harness runs the model and computes
   `vision_scoring.VisionScores`.
3. Per-fixture and aggregate scores are persisted to
   `storage/workflow-runs/<run-id>/vision_benchmark/<task>/<model>.json`.
4. `conference-pipeline promote-vision-candidate --task <id> --model <id>
   --receipt <path>` flips the registry status to `approved` if all
   thresholds pass.

## Status (v1.61)

- ✅ Scoring rubric (`benchmarks/vision_scoring.py`) implemented.
- ✅ Promotion CLI (`benchmarks/promote_vision.py` +
  `conference-pipeline promote-vision-candidate`) implemented.
- ⏳ **Fixture images + ground-truth JSON: not yet authored.** This is the
  one piece of v1.61 Phase 69 that requires manual content (drawing
  slides or sourcing a permissively licensed deck) and therefore lands
  in a follow-up content-only commit.
- ⏳ Canonical Qwen3-VL-8B-4bit passing receipt: requires real Apple
  Silicon hardware to run the harness once fixtures land.
