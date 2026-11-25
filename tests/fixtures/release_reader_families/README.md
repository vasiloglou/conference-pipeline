# Release reader families: case stories

This corpus is the approval instrument for v6.0-Public-Release Phase 6, part 6a (spec
`.planning/milestones/v6.0-Public-Release/notes/phase-6-spec.md`, D18). Each case is a
story: a starting workspace, what runs, and the question. The answers are **not** here.
They are in `EXPECTED.json`, kept apart so the stories can be read blind.

The harness is `tests/test_release_reader_families.py`. It builds each case in a fresh
`tmp_path`, under a project root `proj/` with one workspace, `proj/conferences/conf/`.
No network is used, no real workspace is read, and nothing is published.

## The base workspace, W0

**Reader sources.** `storage/release/reader_sources.json` declares where the reader content
comes from:
- the catalog adapter and its file;
- the statistics source;
- the topic reference ledger and validation folders;
- the narratives;
- the limitations source;
- an optional D11 note.

**Corpus inventory** (`storage/corpus/corpus_inventory.json`, in the Phase 5 shape), with
these paper records:

| Record | `paper_key` | Track | Title | Authors | Kept? |
|---|---|---|---|---|---|
| P1 | `openreview:AAAAAAAAA1` | main | "Sparse Mixtures for Long Context Reasoning" | Ada One, Ben Two | yes |
| P1b | `openreview:AAAAAAAAA1` | main | same paper, a duplicate copy | — | no (`duplicate_copy`) |
| P2 | `arxiv:2501.00001` | workshop | "Neural Operators for Ocean Forecasting" | Cy Three | yes |
| P3 | `site:conf-77` | main | "A Site-Only Paper" | (none) | yes |
| P4 | `openreview:RRRRRRRRR1` | workshop | "A Rejected Submission" | Di Four | no (`rejected_submission`) |

It also holds one admitted `speech_transcript` talk record, which is not a paper.

**Topics.** One topic, `01_alpha`:
- a ledger whose entries are paper `AAAAAAAAA1` and talk `talk-xyz`;
- a validation file whose `valid_reference_ids` are `paper_abstract:AAAAAAAAA1`,
  `speech:talk-xyz` and `paper_abstract:ZZZZZZZZZ9`. That last paper is not in the
  catalog.

**Narrative.** `analysis/01_alpha.md` holds a heading and one paragraph.

## Catalog (R1–R5)

- **R1, the corpus-inventory adapter.** Build the reader families from W0.
  - Which rows are in `catalog/papers.jsonl`, in what order, and what is each row's
    `official_url`?
  - Which topics does each row carry?
  - What is in `catalog/topic_papers.json`, and how many topic references could not be
    resolved to a catalog row?
- **R2, a site paper with its own link.** P3's record gains a field
  `official_url: "https://example.org/papers/77"`. What is P3's `official_url` now? And
  what is it in R1, where the record has none?
- **R3, the accepted-papers adapter.** Another workspace declares the
  `openreview_accepted_papers` adapter over a file with one paper:
  - id `BBBBBBBBB1`, title "Calibrated Uncertainty in Diffusion Planners";
  - author Ed Five, venue "Conf 2026 Poster";
  - `forum_url` `https://openreview.net/forum?id=BBBBBBBBB1`.

  What row is written?
- **R4, a named exclusion.** In W0, P4 carries `index_exclusion: rejected_submission`. Is
  its row in the catalog? What does `catalog/excluded_ids.json` hold?
- **R5, a row that fails the schema.** P2's title is made empty.
  - What does the build do, and with what exit from the CLI?
  - Is any reader file written?

## Statistics (S1–S3)

- **S1, from `corpus_statistics.json`.** W0 adds a Phase 5.1 statistics file:
  - papers 3: main 2, workshop 1;
  - corpus version `conf-corpus-0123456789ab`;
  - talks: `oral` 2 listed, 2 admitted; `expo_talk_panel` 2 listed, 1 admitted, 1
    excluded as `asr_hallucination`; `mystery_kind` 1 listed, 1 admitted.

  What is in `catalog/statistics.json`? Specifically: the talks per kind, with each display
  name, and does every kind satisfy listed = admitted + excluded?
- **S2, no statistics file.** The same workspace without `corpus_statistics.json`. Its
  inventory instead holds Stage 6-style records:
  - two `paper_text` records with titles and authors;
  - one `workshop_paper_text` record;
  - one excluded `paper_text` record;
  - three `speech_transcript` records: two `poster` (one admitted, one excluded
    `speechless_recording`) and one `oral` (admitted).

  What statistics are computed, and what is the corpus version label?
- **S3, the D11 note.** `reader_sources.json` names a `d11_note` text. Where does it appear
  in `catalog/statistics.json`? Is it absent when no note is declared?

## Limitations (L1–L4)

The project root holds a close record, `.planning/milestones/Conf-CONTEXT.md`, with
numbered lines.

- **L1, every quote matches.** The limitations source has two items, `lim-1` (an accepted
  finding) and `lim-2` (a refused support label). Each quote sits verbatim at the lines it
  cites. What do `limitations/limitations.json` and `limitations.md` contain?
- **L2, a quote not at its lines.** `lim-2` cites lines that hold other text. What does the
  build do?
- **L3, redaction.** `lim-1`'s summary names the approver "Jane Approver", listed in its
  `redact`, and includes an email address. What does the emitted text show for each?
- **L4, the row-by-row check.** The source lists the close record's accepted line ranges.
  - (a) Every range is covered by an item: what does `check_limitations` exit with?
  - (b) One range is covered by no item: what does it exit with, and what does it name?

## Redaction in the bundle copy (X1–X6)

A manifest family with `redaction_rules` exports a small folder holding:
- a JSON receipt with `publication.target`;
- a text file with an absolute path inside the workspace and a home path outside it;
- an HTML deck with two Drive links: one the link map resolves to P1, one it does not;
- a PDF whose bytes contain `drive.google.com`.

- **X1, `json_field`.** What does the bundle copy hold at `/publication/target`?
- **X2, `absolute_path`.** What do the two paths become?
- **X3, `relink`.** What does each Drive link become?
- **X4, a binary.** Is the PDF rewritten? Where is it reported?
- **X5, the receipt and checksums.** What does the write receipt's `redacted_files` list?
  Is each file's `CHECKSUMS.json` digest that of the redacted bytes?
- **X6, the workspace originals.** Are the four source files unchanged?

## Manifest, check, site and hub (M1, M2, C1, H1, H2, E1)

- **M1, the schema.**
  - A manifest whose `release` has `conference_name: "Conf 2026"`: does it validate?
  - One with `release.foo`: does it?
- **M2, `release init --refresh`.** An existing manifest promotes family F1, and its
  `excluded_families` lacks storage family F3. After `--refresh`:
  - is F3 listed, and with what reason?
  - is F1 unchanged?
  - is anything promoted?
- **C1, `release check`.** A bundle text file contains `drive.google.com`. What does
  `release check` exit with, and what does it name?
- **H1, the site.** A fixture bundle carries `catalog/statistics.json` (S1's talks, the
  corpus label and the S3 note) and `limitations/limitations.md`. Does the index page show
  the talks with their display names, the label and the note? Is there a Limitations page?
- **H2, the hub.** Bundle `alpha` lists `openreview:RRRRRRRRR1` in
  `catalog/excluded_ids.json`, and its `papers.jsonl` still holds that row. Does the hub
  catalog contain it?
- **E1, end to end.** A TEMPLATE-style workspace whose manifest exports the staged reader
  family, with `release.conference_name`. Run `build_reader_families`, then `release plan`
  and `release build --bundle-only`. Which reader files are in the bundle? What is
  `RELEASE.json` `release.conference_name`?
