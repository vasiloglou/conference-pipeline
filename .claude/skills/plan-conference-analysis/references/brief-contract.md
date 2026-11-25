# Brief Contract

Require a PRD-style brief with these fields before canonical writes:

- conference identity
- source URLs and auth notes
- target outputs (including `podcast_notebooklm: true|false` — per-deck Google NotebookLM Audio Overviews need a Google account with NotebookLM Pro, the `notebooklm` extra, and a `notebooklm login --browser-cookies chrome` session)
- proof mode
- topic strategy
- provider preferences
- provider-cost approval posture and budget constraints
- content-fidelity posture for exhaustive extraction, slide/visual OCR,
  programmatic figure/SVG quality loops, and source traceability ID preservation
- owner-command posture for existing implementations, missing implementations,
  and deprecated presentation paper-figure scripts
- storage/share targets
- acceptance criteria
- platform version
- conference name
- workshop coverage posture (required whenever the brief's target outputs
  include workshops, or workshops are not declared `out_of_scope` in the
  event-family inventory)

## Workshop Coverage Posture

Generated planning must record one of:

- `all_openreview_with_pdf` — discover every workshop venue under the
  conference's OpenReview parent group; download PDFs for venues classified
  `openreview_with_pdf`; record explicit decisions for venues classified
  `openreview_metadata_only` or `no_openreview_record`.
- `all_workshops_best_effort` — same as the above plus scaffold/execute
  workshop-site downloaders for `openreview_metadata_only` and
  `no_openreview_record` venues so external PDFs are also acquired.
- `selective` — an explicit allow-list of workshop venue IDs; every other
  workshop must be recorded as `excluded` with rationale.
- `out_of_scope` — workshops are excluded from the conference scope; the
  exclusion must be recorded with `scope`, `reason`, `decision context`,
  `source note`, and `downstream effect`.

Per-workshop decisions must record one of `approved`, `metadata_only_acceptable`,
`external_site_required`, or `excluded`. Workshops in the `openreview_metadata_only`
and `no_openreview_record` classes default to `external_site_required` unless
the brief explicitly chooses otherwise; the analysis scaffold writes an
external-site downloader scaffold and tracker entry for each such workshop.

If a field is missing, ask targeted questions. Store auth notes as labels and
operator instructions only; never copy cookie values, API keys, provider tokens,
or `.env` contents into generated artifacts.

When paid provider work is expected, require the handoff to name the
full-pipeline estimate command:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/<slug> \
  --scope full \
  --scenario base \
  --json
```

When presentations, slides, figures, posters, video, or exhaustive reading are
expected, require the handoff to state whether `extracted_content.json` must be
exhaustive, which visual assets require model-assisted OCR/descriptions, how
`FIGURE_NEED` slides get content-aware, slide-specific programmatic
matplotlib/Plotly/svgwrite SVG figures or explicit per-slide waivers, how
template selection is tied to slide intent, how generic placeholder/scaffold
SVGs are rejected during rendered inspection, how NeurIPS-style hyperlink and
SVG density targets are configured and validated, how rendered screenshots are
passed through governed `presentation.visual_qc` vision-LLM review, and how
ordinary source artifact IDs stay unredacted while secrets stay redacted.
Paper-extracted/cropped figures are supplemental only and must not satisfy
`presentation.svg_figure_generation`.
