# Roadmap Backbone

Every generated conference roadmap must cover:

- source intake
- workshop discovery & corpus freeze
- workspace/storage
- provider cost gate
- acquisition proof
- transcription/OCR
- analysis
- content fidelity gates
- QA
- podcast (conference overview TTS lane and, when in scope, per-deck NotebookLM Audio Overviews)
- presentation production
- validation
- proof/closeout

## Workshop Discovery & Corpus Freeze

This area is required whenever the brief's `workshop_coverage_posture` is any
value other than `out_of_scope`. The phase must:

- Call `scripts/discover_openreview_workshops.py --venue-prefix
  <openreview_workshop_parent_venue>` and persist
  `storage/workshops/inventory.json` with a per-workshop record (venueid,
  paper_count, tiers, pdf_availability_class, sub_track_parent).
- Call `scripts/download_openreview_workshop_papers.py --inventory ...
  --rate-limit-rps 1.4` for venues classified `openreview_with_pdf`,
  emitting one per-paper receipt per accepted note.
- For CVF-published conferences (CVPR, ICCV, WACV), call
  `scripts/download_cvf_openaccess_workshop_papers.py --workspace
  conferences/<slug>` (dry-run first) to download archival workshop papers from
  the CVF Open Access workshop proceedings (`urls.openaccess_workshops`),
  skipping papers already acquired from OpenReview. Only venues with no CVF
  listing, and non-archival tracks, fall through to the stub below.
- For `openreview_metadata_only` and `no_openreview_record` venues, emit an
  `external_site_required` scaffold under
  `scripts/conference/<slug>/workshop_site_downloaders/<workshop_slug>.py`
  and register the stub in `storage/workshops/workshop_site_tracker.json`.
- Receipt boundary: the discovery inventory digest is stable across reruns,
  every `openreview_with_pdf` venueid has a download summary, and every
  metadata-only/no-record venueid has either a completed external-site
  downloader receipt or an explicit decision (`metadata_only_acceptable`,
  `excluded`).

## Acquisition Proof

When a conference site exposes media through authenticated item pages, the plan
must distinguish cached page evidence from local media evidence. Successful
`item_page_auth_probe` HTML can justify focused embedded media recovery, but it
does not satisfy transcription/media completion until audio/video artifacts or
unreachable-source blockers exist. For ICLR-style SlidesLive embeds, name
`acquisition.slideslive_media_recovery` or a conference-local equivalent before
transcription/OCR, write `storage/download-corpus/slideslive_media_recovery.json`,
and preserve the no-redownload boundary for source pages, paper PDFs, workshop
PDFs, poster images, and unrelated media families.

If slide/visual/OCR evidence is in scope, SlidesLive slide images require their
own acquisition and cleanup chain: `acquisition.slideslive_slide_download`,
`acquisition.slide_reference_detection`, and `acquisition.slide_cleanup`.
Audio-only recovery, paper PDFs, and poster thumbnails are rejected evidence for
SlidesLive slides; duplicate and generic/banner slides are removed from
OCR/indexing via non-destructive `skip_ocr` markers.

Workshop talk media acquisition must carry a conference-local embedded host
allowlist. For ICLR production, use `--embedded-media-host slideslive.com` and
record `allowed_hosts`; do not treat YouTube, Vimeo, Zoom, social/profile, or
unrelated embeds as ICLR workshop presentation media unless an explicit future
phase widens the allowlist.

## Token-Bounded Indexing And Talk-Aware Transcripts

Production handoff must include `PG-CORPUS-TOKEN-INDEXING` before query
planning. The Stage 6 receipt and index digest must record
`token_budget_policy_digest`, `embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`,
`CORPUS_VECTOR_EMBED_BATCH_SIZE`, and `vector_embeddings_written: true`, with
embedded chunk counts matching sidecar chunk counts. Word-count-only sidecars,
readiness-only probes, partial Chroma rows, and historical exhaustive
extraction output without registry receipt are rejected evidence.

Workshop transcripts must first prove talk-media acquisition or waiver when
workshops are in scope: require
`storage/workshops/talk_media_acquisition_receipt.json` or
`storage/workshops/workshop_talk_media_scope_waiver.json` before Stage 5/6. A
zero-count workshop speech audit describes current inventory only; it is not a
waiver.

Workshop transcripts must include `PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY` and run
`transcription_ocr.workshop_talk_boundary_discovery` when workshop speech is in
scope and no vetted `talks.json` exists. The governed discovery owner is
`scripts/discover_workshop_talks.py`; default to Gemini 2.5 Pro
(`gemini-2.5-pro`), write `*_talks.json`, emit 5-minute talk-context batches,
and record `provider_receipt_paths` plus
`cost_task_id=workshop_talk_boundary_discovery`. Then run
`transcription_ocr.workshop_talk_segmentation`. If neither `talks.json` nor
governed discovery is available, record
`talk_segmentation_provider_not_configured`. audio-only recovery is rejected
evidence for talk-aware transcripts, and indexed transcript chunks preserve
`workshop_id`, `workshop_title`, `talk_id`, `talk_title`, and `talk_speaker`.

## Official Workflow Mapping

Every generated ROADMAP.md must render ## Official Workflow Mapping before generated phase entries.

Each generated phase must include:

- Goal
- Requirements
- Success Criteria
- Canonical References

Plans with analysis, presentation, slide, figure, or exhaustive-reading outputs
must add explicit content-fidelity success criteria for extracted-content
provenance, slide/visual OCR descriptions, figure/SVG render-inspection repair
loops, governed vision-LLM visual QC over rendered screenshots, and source
traceability ID preservation.

Keep roadmap entries phase-level. Downstream GSD discuss and plan phases own
task-level decomposition.
