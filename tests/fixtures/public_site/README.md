# Public site and hub: case stories

This corpus is the approval instrument for v6.0-Public-Release Phase 4 (spec
`.planning/milestones/v6.0-Public-Release/notes/phase-4-spec.md`, D7). Each case is a
story: a starting bundle (or two), what is built, and the question. The expected results
are **not** here. They are in `EXPECTED.json`, kept apart so the stories can be read
blind.

The harnesses are `tests/test_public_site.py` and `tests/test_hub_site.py`. They build
each bundle in a fresh `tmp_path`, with the layout of the real ICLR 2026 v1.0.0 bundle
(D6). Nothing touches the network: video URLs are `https://cdn.example.test/…`, and
paper links are `https://openreview.net/forum?id=…` strings that are never fetched.

## The two fixture bundles

Both live at `conferences/<slug>/exports/public-release/<bundle id>/`, so they pass the
public-bundle check. Each holds `RELEASE.json`, `CHECKSUMS.json` (every file, with sha256
and size), `decks/control/presentation_manifest.json` and the reader content below.

- **`alpha`**, bundle `alphaconf-v1.0.0`, conference "Alpha Conference 2026".
  - **Topic `01_reasoning`, "Reasoning at test time":**
    - `presentation.html` and `presentation.pdf`;
    - two video sidecars (`presentation_video.mp4.cdn.json`,
      `presentation_video_notebooklm.mp4.cdn.json`);
    - a podcast `audio/01_reasoning/podcast.mp3`;
    - a narrative `narratives/01_reasoning.md`, which contains the unusual word
      "quokkalike" once.
  - **Topic `02_agents`, "Agents and tools":** only `presentation.html`. No `slides.md`,
    no video, no podcast, no narrative.
  - **Catalog.** `catalog/papers.jsonl` has 4 rows:
    - `openreview:A1` through `openreview:A3`, each with an `official_url`;
    - `site:s1`, with `official_url: null`.
  - **Topic papers.** `catalog/topic_papers.json` gives `01_reasoning` the papers A1 and
    A2.
  - **Index.** A search index with a browser-embedding block: model `fixture-embed-model`,
    dim 8, and a blob listed in `CHECKSUMS.json`.
- **`beta`**, bundle `betaconf-v1.0.0`, conference "Beta Conference 2025".
  - **Topic `01_vision`:** deck HTML only.
  - **Catalog.** `catalog/papers.jsonl` has 3 rows: `openreview:B1`, `openreview:B2` and
    `openreview:B3`.
  - **Exclusions.** `catalog/excluded_ids.json` lists `openreview:B3`.
  - **A stray file.** `catalog/extra_papers.jsonl` holds one row, `openreview:B9`. It is
    **not** in `CHECKSUMS.json`.
  - **Index.** No browser-embedding block (full text only).

Exactly two titles contain the word "diffusion": `alpha`'s A2 and `beta`'s B1. The
excluded B3 has "diffusion" in its title too, so the query also tests the exclusion.
"diffusion" is the hub's acceptance query.

## Cases

- **C1, not a bundle.** Build the site from the workspace root
  `conferences/alphaconf/`, not the bundle. Does the build go ahead? Is anything written?
- **C2, what each topic has.** Build `alpha`'s site. What does each topic page show for
  deck, videos, podcast and narrative? What does `02_agents` say about what it lacks?
- **C3, a paper with no public copy.** On `alpha`'s catalog page, how is `site:s1` shown?
- **C4, the links.** Run the link check over `alpha`'s built site. Does every relative
  link name a file in the site? Is any `public-repo/`, `.git` or control file reachable
  outside the data-files page?
- **C5, full text.** Search `alpha`'s site for "quokkalike" in full-text mode. What comes
  first?
- **C6, semantic shown when it works.** Which search modes does `alpha`'s search page
  show? Which model and dimension does its search configuration name?
- **C7, semantic hidden when it cannot work.** Which modes does `beta`'s search page
  show, and what does the page say?
- **C8, a tampered file.** After `alpha` is written, change one byte of
  `narratives/01_reasoning.md` without updating `CHECKSUMS.json`, then build the site.
  What happens?
- **C9, the hub's coverage.** Build the hub over `alpha` and `beta`. How many catalog
  rows does it hold?
- **C10, the acceptance query.** Search the hub for "diffusion". Which results come
  back, and how is each labelled?
- **C11, what the hub must not show.** Is `openreview:B3` (excluded) or `openreview:B9`
  (in a file outside `CHECKSUMS.json`) anywhere in the hub? How does the hub manifest
  account for them?
- **C12, mixed search capabilities.** `alpha` has semantic vectors and `beta` has none.
  Which search modes does the hub show?
