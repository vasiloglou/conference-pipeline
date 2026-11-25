# `conference-pipeline release`: case stories

This corpus is the approval instrument for v6.0-Public-Release Phase 3 (spec
`.planning/milestones/v6.0-Public-Release/notes/phase-3-spec.md`, D6). Each case is a story:
a starting state, a command and the question it asks. The expected exit code and
effects are **not** here. They are in `EXPECTED.json`, kept apart so the stories can be
read blind.

The harness is `tests/test_release_cli.py`. It builds each case in a fresh `tmp_path`,
never in `conferences/TEMPLATE/` itself, and it never calls a live provider. Where a
case needs a command to "run", the harness injects a fake runner that records the
command lines and returns success.

## The starting workspaces

- **W0, a fresh workspace.** A project root with `conferences/sampleconf2027/` copied
  from `conferences/TEMPLATE/`. Its `config.yaml` has the slug `sampleconf2027`. It has
  one deck file, `decks/01_sample/slides.md`, holding the line `public slides`. The
  project root is a git repository with one commit, so a source SHA resolves. It has no
  `storage/public_export_manifest.json`.
- **W1, drafted.** W0 after `release init sampleconf2027`.
- **W2, promoted.** W1 after the operator promotes one family. The
  `presentation_markdown` row moves from `excluded_families` into `families` as
  `{id: presentation_markdown, private_path: decks, public_path:
  exports/public-release/{bundle}/decks/, storage_manifest_id: presentation_markdown,
  redistribution_status: allowed, license: CC-BY-SA-4.0, provenance: pipeline-generated}`.
- **W3, planned.** W2 after `release plan sampleconf2027`.
- **W4, built.** W3 after `release build sampleconf2027`.
- **W2-live.** W2 plus a planted live receipt,
  `storage/workflow-runs/github-sampleconf2027-v1.0.0-live-receipt.json`, recording a
  public repository push of bundle `sampleconf2027-v1.0.0`. It stands for a bundle that
  was already published.

The isolated directory `D` is a path under `tmp_path`, outside the workspace, that does
not exist yet.

## init

- **C1.** In W0, run `release init sampleconf2027`. What happens to the workspace, and
  what does the drafted manifest contain?
- **C2.** In W1, run `release init sampleconf2027` a second time. Does the operator's
  manifest survive?
- **C3.** In W0 with `storage/storage_manifest.json` deleted, run
  `release init sampleconf2027`. Can a manifest be drafted with no storage manifest?

## plan

- **C4.** In W1 (nothing promoted yet), run `release plan sampleconf2027`. Can a release
  be planned from a draft that publishes nothing?
- **C6.** In W2, run `release plan sampleconf2027`. What does a dry run leave behind?

## build

- **C5.** In W2 (promoted, but never planned), run `release build sampleconf2027`. Can the
  bundle be written without the dry run first?
- **C7.** In W3, run `release build sampleconf2027`. Is the bundle written, and does the
  digest its receipt records match the bytes on disk?

## check

- **C8.** In W4, run `release check sampleconf2027`. Is the freshly built bundle clean?
- **C9.** In W4, append a line naming a file in the home directory of a user called
  `alice` (a Linux-style home path, `notes.txt` inside it) to the bundle's copy of
  `decks/01_sample/slides.md`, then run `release check sampleconf2027`. Does the check
  catch the home path, and what does it print? (The harness builds that path at run
  time, so this corpus stores no home path of its own.)
- **C10.** In W4, change one byte of the bundle's copy of `decks/01_sample/slides.md`
  (the source deck is untouched), then run `release check sampleconf2027`. Does the
  check notice?

## publish

- **C11.** In W3, run `release publish sampleconf2027 --surface github` with no
  `--approved-by`. Does anything run, and is `.env` read? (The harness plants a `.env`
  that raises if opened, and a runner that records every call.)
- **C12.** As C11, but with `--approved-by "  "` (whitespace only).
- **C13.** In W4 *(amended 2026-11-15, IF1; was: W3)*, with targets passed for GitHub, run
  `release publish sampleconf2027 --approved-by alice --surface github` with the fake
  runner. Which commands run, in what order, and what does the top-level receipt say
  about the approval?
- **C14.** In W3, run `release publish sampleconf2027 --approved-by alice` with no
  `--surface`. Does anything run?

## isolated builds (the reproduction path)

- **C15.** In W2-live, run `release plan sampleconf2027 --isolated D`, then
  `release build sampleconf2027 --isolated D`. Can an already-published bundle be
  rebuilt for comparison, and what changes inside the workspace?
- **C16.** In W2-live after `release plan sampleconf2027`, run
  `release build sampleconf2027` without `--isolated`. Can a published bundle be
  overwritten in place?
- **C17.** In W2-live after `release plan sampleconf2027 --isolated D`, run
  `release build sampleconf2027 --isolated D --expect-tree-sha256` followed by 64 zeros.
  What does a digest mismatch look like?
- **C19.** In W2-live, run `release build sampleconf2027 --isolated D` with no
  `D/plan-report.json`. In a second variant, `D/plan-report.json` exists but names a
  different export manifest digest. Can an isolated build skip its dry run?
- **C20.** In W2, run `release plan sampleconf2027 --isolated D`, where (a) `D` is
  `conferences/sampleconf2027/tmp-iso` inside the workspace, or (b) `D` is a directory
  that already holds a file. Is the isolated directory allowed to be either?

## configuration

- **C18.** Two parts.
  - (a) In W4 *(amended 2026-11-15, IF1; was: W3)*, `config.yaml` gets a `release:` block with `github_repo:
    example-org/sampleconf2027-data`, and no target flag is passed. Run
    `release publish sampleconf2027 --approved-by alice --surface github` with the fake
    runner. Where does the GitHub target come from?
  - (b) Run `scripts/push_public_repo.py` in dry-run mode over W4's bundle without
    `--repo-name`. What repository name does it stage?

Case numbers follow the spec (§8); C4 comes before C6 because C4 is the refusal on the
draft.
