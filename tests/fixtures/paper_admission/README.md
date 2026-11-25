# Paper admission: case stories

This corpus is the approval instrument for v6.0-Public-Release Phase 5 (spec
`.planning/milestones/v6.0-Public-Release/notes/phase-5-spec.md`, D9). Each case is a
story: a starting workspace, what is added, and the question. The expected numbers are
**not** here. They are in `EXPECTED.json`, kept apart so the stories can be read blind.

The harness is `tests/test_paper_admission.py`. It builds each case in a fresh
`tmp_path` and runs `admit_paper_sets` on it. No network and no real workspace are
involved.

## The base workspace, F0

An existing corpus in the OpenReview-acquisition format, `storage/acquisition/papers.json`,
declared as the workspace's existing-corpus adapter. It holds four papers, each with an
abstract only (no full text):

| Record | Stable id | Track | Title |
|---|---|---|---|
| P1 | OpenReview forum `AAAAAAAAA1` | main | "Sparse Mixtures for Long Context Reasoning" |
| P2 | OpenReview forum `AAAAAAAAA2` | main | "Calibrated Uncertainty in Diffusion Planners" |
| P3 | OpenReview forum `AAAAAAAAA3` | main | "Graph Priors for Molecular Generation Tasks" |
| W1 | OpenReview forum `BBBBBBBBB1` | workshop | "Benchmarks for Scientific Agents in the Lab" |

There is an empty lane registry, `storage/papers/late_paper_lanes.json`, and no exclusion
file. Each case adds one or more lanes (a unit file plus its registry entry) or an
exclusion. A "full text" in these stories is a text unit of at least 1,500 characters; a
"short text" is shorter.

## Cases

- **C1, a genuinely new paper.** Add a workshop-text lane with one full-text unit for
  OpenReview id `NNNNNNNNN1`, titled "Active Learning for Protein Stability Screens",
  which F0 does not hold. How many papers does the corpus hold, and which paper is new?
- **C2, an existing paper gains text.** Add a main-text lane with one full-text unit for
  P1's forum id `AAAAAAAAA1`. Does the paper count change? Which paper gained text?
- **C3, one paper through two lanes.** Add two workshop lanes, each with one full-text unit for the
  same new paper, "Neural Operators for Ocean Forecasting". The first lane knows it as
  arXiv `2501.00001v1`, the second as arXiv `2501.00001v2`. Neither id is in F0. How many
  papers are added, and how many duplicate copies are merged?
- **C4, a site-only copy of a known paper.** Add a workshop-site lane with one full-text
  unit whose only identity is a site id, `site_abc123`. Its title is P2's title with
  different capitalization and punctuation ("Calibrated uncertainty in diffusion
  planners."). Is it a new paper, or a copy of P2? Is the join reported?
- **C4b, two stable ids with one title.** Add a workshop-text lane with one full-text unit
  for OpenReview id `NNNNNNNNN2`, whose title is exactly P3's. Are these one paper or two?
- **C5, a named exclusion.** Add an exclusion file naming W1 (`BBBBBBBBB1`) with reason
  `rejected_submission`. Is W1 still in the inventory? Is it counted?
- **C6, an existing paper with no late counterpart.** Add the C2 lane only (text for P1).
  What happens to P3's record?
- **C7, a short text.** Add a main-text lane with one short-text unit (under 1,500
  characters) for P2's forum id. Is it an artifact? Does it count as P2 gaining full
  text?
- **C8, admission twice.** Take the workspace of C3, admit, and admit again. Do the
  inventory, the identity report and the statistics change?
- **C9, a dry run.** Take the workspace of C1 and admit with `dry_run=True`. What is
  written, and what is reported?
- **C10, the statistics.** Take the workspace that combines C1, C2, C3 and C5. Do the
  statistics equal a recount from the inventory, and what do they say about the corpus
  before admission?
