# Talk admission: case stories

This corpus is the approval instrument for v6.0-Public-Release Phase 5.1 (spec
`.planning/milestones/v6.0-Public-Release/notes/phase-5.1-spec.md`, D10). Each case is a
story: a starting workspace, what runs, and the question. The answers are **not** here.
They are in `EXPECTED.json`, kept apart so the stories can be read blind.

Two harnesses build each case in a fresh `tmp_path`:
- `tests/test_talk_admission.py` covers C1–C9: `bind_talk_sources`, then
  `admit_paper_sets`;
- `tests/test_talk_coverage_guard.py` covers C10–C17: `scripts/audit_talk_coverage.py`.

No network and no real workspace are involved. Event URLs use `https://example.org/`.

## The shapes the stories use

- **A deep-dive presentation** is one file under
  `exports/deep-dive/openai_claude/presentations/`, listed in
  `exports/deep-dive/manifest.json`. It holds:
  - a fenced JSON metadata block: `id`, `title`, `track`, `neurips_url`, `slideslive_id`;
  - a `## Audio Transcript` section, followed by the other sections the real files carry.

  The metadata spellings are the export's own. `track` becomes the kind.
- **A v1.5 receipt** is `storage/transcription-ocr/v15_transcription_ocr_receipts.json`.
  - Its `media_artifacts` entries have `artifact_class` `speech_transcript`, `status`
    `transcribed`, a `provider_model`, and a `transcript_path` under
    `storage/transcription-ocr/media/<item_id>/transcript.txt`.
  - Each item's `kind` and `source_url` come from the event inventory
    `storage/acquisition/v15_source_inventory.json`.
- **The talk config** is `storage/transcription-ocr/talk_sources.json`. It names one lane
  per source: lane `talks-a` for the deep-dive export, lane `talks-b` for the v1.5
  receipt. Named exclusions go in `storage/transcription-ocr/talk_exclusions.json`.
- **A paper base**, where a case needs one, is the P5 shape: an OpenReview-acquisition
  file with two main papers, declared in `storage/papers/late_paper_lanes.json`, with no
  late lanes.
- **The guard's admitted set** is `storage/corpus/corpus_inventory.json`. Its
  `speech_transcript` records name a `transcript_path`, a `source_path`, an
  `indexed_text_path`, or a `source:<path>` provenance entry.

## Binding and admission (C1–C9)

- **C1, one deep-dive talk.** One presentation: event `1001`, an `oral` titled "Scaling
  Laws for Sparse Experts", SlidesLive id `5001`, with a transcript. Bind. What record does
  lane `talks-a` hold, where is its transcript written, and what does it carry as kind,
  title, URLs and provenance?
- **C2, one v1.5 talk.** A receipt with one transcribed event, `oral-2001`, and one
  `poster_asset` entry that is not a transcript. The event inventory says `oral-2001` is
  an `oral` at its URL. Bind. What record does lane `talks-b` hold, which file does it
  point at, and is any transcript copied?
- **C3, a named exclusion inside a shared recording.** Two presentations share SlidesLive
  id `5003`:
  - event `3001`, an `expo_talk_panel`, whose transcript is a YouTube-boilerplate loop;
  - event `3002`, an `oral`.

  `talk_exclusions.json` names `3001` with reason `asr_hallucination`. Bind. Which event
  keeps the recording, and what becomes of `3001`?
- **C4, two empty transcripts.** Both sources are declared:
  - presentation `4001`, a `workshop` with no SlidesLive id, has no `## Audio Transcript`
    section;
  - the v1.5 transcript of `poster-4002` is one newline.

  Bind. How many records and gaps does each lane hold, and with what reasons?
- **C5, one recording, three events.** Presentations `5101`, `5102` and `5103`, all
  `oral`, share SlidesLive id `5100`, and each has its own transcript.
  1. Bind with `--dry-run`: what is reported, and is any file written?
  2. Bind for real: which event keeps the recording, which URLs and events does the kept
     record list, and what are the other two?
- **C6, bind twice.** C5's workspace, bound twice. Do the lane receipt, the registry and
  the transcript file change between the runs?
- **C7, admission with talks.** A paper base (two main papers), and both sources:
  - **deep-dive:**
    - `7101`, an `oral` (SlidesLive `9101`);
    - `7102`, an `expo_talk_panel` (`9102`), named in `talk_exclusions.json` as
      `asr_hallucination`;
    - `7103`, a `workshop` with no SlidesLive id and no transcript section;
    - `7104`, a `keynote`, and `7105`, a `test_of_time`, both on SlidesLive `9104`;
  - **v1.5:**
    - `poster-7201`, a `poster`;
    - `oral-7202`, an `oral`;
    - `poster-7203`, a `poster` whose transcript is one newline.

  Bind, then admit. What is the `talks` block of `corpus_statistics.json`, overall and per
  kind? How many talk records and gap records are in the inventory? Do the paper tallies
  differ from admitting the same paper base with no lanes?
- **C8, P5's corpus replayed.** Every case of `tests/fixtures/paper_admission/` is admitted
  again with this phase's code. These workspaces have no transcript lane registry. Are the
  three output files byte-identical to what P5's code wrote, and does a `talks` key
  appear anywhere?
- **C9, admission twice, with an older lane.** A paper base with one main paper. The lane
  registry lists two lanes, in this order:
  - `legacy-captions`, a hand-built lane receipt in the CVPR shape: one record,
    `yt-abc`, with no `item_kind`;
  - lane `talks-a`, bound from one deep-dive `oral`, `9001`.

  Admit twice. Are the three outputs byte-identical? How is the `legacy-captions` record
  admitted, and under which kind is it counted?

## The guard (C10–C17)

Each case is a workspace and one guard run. The question is always the exit code and what
the report says.

- **C10, all admitted.** A v1.5 receipt with one transcribed event, `poster-1`, whose file
  exists. The inventory has a `speech_transcript` record with that `transcript_path`.
- **C11, one v1.5 transcript left out.** The receipt names three transcribed posters:
  - `poster-1`, admitted;
  - `poster-2`, whose file exists but which has no inventory record;
  - `poster-3`, whose file does not exist.

  The event inventory gives each the kind `poster`.
- **C12, one stage-inventory transcript left out (ICML shape).**
  `storage/transcription-ocr/transcription_ocr_inventory.json` lists two paths in
  `output_inventory.speech_transcript_outputs`; both files exist. The inventory admits
  the first only.
- **C12b, the caption shape (CVPR shape).** `storage/transcription-ocr/conf/inventory.json`
  lists three `caption_transcripts` entries, all files present:
  - `v1`: admitted by a record whose `transcript_path` is `v1`'s `caption_path`;
  - `v3`: admitted by a record whose `transcript_path` is `v3`'s `output_path`;
  - `v2`: no record matches it.
- **C13, a lane record missing from the inventory.** A registered lane receipt has records
  `r1` and `r2` (both transcript files exist) and one coverage gap, `g1`, with reason
  `silent_audio`. The inventory admits `r1` only.
- **C14, a deep-dive file left out.** The manifest lists three presentations:
  - `f1`, whose file exists and which an inventory record names as `source:` provenance;
  - `f2`, whose file exists and which no record names;
  - `f3`, whose file does not exist.
- **C15, exclusions account for talks.** Two talks are on disk, and the inventory records
  that name them both carry an `index_exclusion`:
  - v1.5 transcript `t1`, named by a record's `transcript_path`, excluded as
    `empty_transcript`;
  - deep-dive file `d1`, named by an audit-only gap record's `source_path`, excluded as
    `duplicate_recording`.
- **C16, no inventory, and nothing at all.**
  - (a) A v1.5 transcript is on disk, and there is no `storage/corpus/corpus_inventory.json`.
  - (b) A workspace holding only an empty `storage/`.
- **C17, unreadable input.**
  - (a) The v1.5 receipt is not valid JSON.
  - (b) `--workspace` names a directory that does not exist.
