---
name: conference-workshop-paper-discovery
description: Find where every scheduled workshop's papers live (OpenReview, proceedings hosts such as CVF Open Access, workshop websites, arXiv), measure coverage against the official schedule, research the gaps, verify links, and download recovered papers with corpus-admissible receipts. Use after the OpenReview workshop download for any conference, or when asked which workshops have no papers, where missing workshop papers are, or to recover them.
---

# Conference Workshop Paper Discovery

OpenReview is rarely the complete record of a conference's workshop papers.
On CVPR 2026 it held PDFs for 35 of 153 scheduled workshops. Following this
workflow raised that to 107 workshops and roughly tripled the workshop corpus,
and it listed where the rest of the gap sits. Nothing here is specific to one
conference. The schedule, OpenReview venue and proceedings hosts all come from
the workspace (`conferences/<slug>/config.yaml` and its storage files).

This repository copy is canonical; mirrors under `~/.claude/skills` and
`~/.codex/skills` are sync targets.

## When To Use

- After `acquisition.workshop_paper_download` (OpenReview) for any conference
  whose workshop coverage posture is not `out_of_scope`.
- Before counting `no_openreview_record` / `openreview_metadata_only` workshops
  as a coverage boundary or writing external-site downloader stubs.
- When asked "which workshops have no papers", "where are their papers", or to
  recover missing workshop papers.

## Rules (learned the hard way)

1. **The official schedule is the spine.** Coverage is measured per scheduled
   workshop, not per OpenReview venue. Many workshops have no venue at all.
2. **Match on full titles and websites, never short names alone.** Short-name
   matching gave 40 "orphan" proceedings workshops where the real number was
   31. Sub-tracks (`X_Proceedings_Track`, `X-NA`, …) group under `X` even when
   no plain `X` venue exists.
3. **Research output is a claim; only a fetched `%PDF-` is evidence.** A
   workshop page linked OpenReview "PDFs" that returned HTTP 403 ("does not
   have permission") to an authenticated account. Such papers are
   `unavailable: not_public`, never "downloadable".
4. **Duplicates are title OR (shared authors AND similar title).** Camera-ready
   titles change ("[UNI]101" → "CSU101"). Every downloader skips papers that
   the base OpenReview download or an earlier registered source already has.
5. **Every source registers its download receipt** in
   `storage/workshops/workshop_paper_sources.json`. Corpus stages load that
   registry instead of a hardcoded path per source. Re-running a downloader
   re-pins the digest; a drifted receipt blocks admission.
6. **Workshops with nothing new are `no_new_papers`, not `succeeded`.** A
   `succeeded` entry promises a receipts directory; the CVPR Stage 6 rebuild
   blocked on 16 workshops that broke that promise.
7. **Free lookups by default, keyed ones opt-in.** Proceedings hosts, arXiv
   and Crossref need no credentials. OpenAlex now meters anonymous use
   against a daily budget: on CVPR 2026 it ran out after ~100 titles with
   "Insufficient budget … resets at midnight UTC". Semantic Scholar returns 429
   without a key. Both are enabled only with `OPENALEX_API_KEY` or
   `SEMANTIC_SCHOLAR_API_KEY`; never add funds without operator approval. Only
   OpenReview uses `OPENREVIEW_USERNAME`/`OPENREVIEW_PASSWORD` from the
   repo-root `.env` (names only in receipts). Respect rate limits: proceedings
   host ≤1 rps, arXiv 1 request per 3 s, Crossref ~1 rps. A backend that
   refuses service is disabled for the rest of the run, not retried into a
   hang.

## Workflow

All commands run from the repository root with `PYTHONPATH=src .venv/bin/python`.
Paths are workspace-relative unless shown otherwise.

### 1. Schedule spine (conference-specific input)

Produce a JSON file with a `workshops` list carrying an id, title and website
per scheduled workshop, from the conference's official program. The field
names are yours; pass them to step 4. (CVPR 2026 used
`scripts/discover_cvpr2026_workshop_sites.py` →
`storage/workshops/workshop_external_sites.json`.)

### 2. OpenReview sources

Run the existing owners (`scripts/discover_openreview_workshops.py`,
`scripts/download_openreview_workshop_papers.py`), then save each workshop
group's full title and website:

```
scripts/fetch_openreview_workshop_groups.py --workspace conferences/<slug>
```

→ `storage/workshops/openreview_workshop_groups.json` (venue prefix from
`urls.openreview_workshop_parent`).

### 3. Proceedings sources

For each proceedings host registered in
`conference_pipeline.acquisition.proceedings_sources.SOURCES` that applies to
the conference, run its downloader, dry run first. Today that is CVF Open
Access (CVPR, ICCV, WACV; `urls.openaccess_workshops`):

```
scripts/download_cvf_openaccess_workshop_papers.py --workspace conferences/<slug> --dry-run
scripts/download_cvf_openaccess_workshop_papers.py --workspace conferences/<slug>
```

A new host (ECVA, ACL Anthology, PMLR, …) needs a `ProceedingsSource` entry, a
downloader writing the shared receipt schema, a stage-owner registry entry
and tests. Do not scrape a new host ad hoc.

### 4. Coverage audit

```
scripts/audit_workshop_paper_coverage.py --workspace conferences/<slug> \
  --schedule <schedule.json> --id-field <id key> --title-field <title key> --website-field <website key>
```

→ `storage/workshops/workshop_paper_coverage.{json,md}`. Categories:
`openreview_and_proceedings`, `openreview_only`, `proceedings_only`,
`other_source_only`, `openreview_titles_only`, `openreview_venue_empty`,
`no_source`. Review `unmatched_openreview_groups`, `unmatched_proceedings` and
any low `score` matches before trusting the numbers.

### 5. Gap research (LLM agent)

For workshops without papers, run web research agents in parallel, about 15
workshops per agent, following
[references/gap-research.md](references/gap-research.md). Each agent writes one
JSON file per workshop into a research directory (for example
`storage/workshops/gap-research/`) and returns a classification table. Agents
browse and write JSON only. They never download papers.

### 6. Title resolution

```
scripts/resolve_paper_titles.py --research-dir <research dir> \
  --output storage/workshops/gap-research/title_resolutions.json --workspace conferences/<slug>
```

This looks up every title on arXiv and Crossref (add
`--backends arxiv,crossref,openalex` only with an API key). It records an open
PDF when one exists (usually arXiv), DOI-registered versions at other venues,
and arXiv venue notes. The same script accepts `--input` with any
`{key,title}` list, for example to check whether proceedings-only papers were
published elsewhere. It is resumable: rerun with the same `--output`.

### 7. Verify, then download

```
scripts/download_workshop_site_papers.py --workspace conferences/<slug> \
  --research-dir <research dir> --resolutions <title_resolutions.json> --verify-only
scripts/download_workshop_site_papers.py --workspace conferences/<slug> \
  --research-dir <research dir> --resolutions <title_resolutions.json>
```

The verify pass writes `workshop_site_download_receipt.verify.json`. The
download pass writes per-paper receipts under
`storage/workshops/papers/workshop-site_<schedule_id>/` and registers source
`workshop_site`. Report `unavailable` reasons (`not_public`,
`response_not_pdf`, `no_link`) as real gaps, not failures.

### 8. Re-audit and hand off

Re-run step 4; recovered papers now count under `other_source_only` or the
proceedings categories. Rebuild the corpus-inventory stage so registered
sources are admitted. For CVPR 2026 that is `--stage corpus-inventory-index`,
which is local embedding with no provider cost. Downstream analysis re-runs are
paid and need operator approval.

## Outputs

| Artifact | Owner |
|---|---|
| `storage/workshops/openreview_workshop_groups.json` | `acquisition.openreview_workshop_groups` |
| `storage/workshops/cvf_openaccess_download_receipt.json` | `acquisition.cvf_openaccess_workshop_paper_download` |
| `storage/workshops/workshop_paper_coverage.{json,md}` | `acquisition.workshop_paper_coverage_audit` |
| `<research dir>/*.json`, `title_resolutions.json` | gap research + `indexing.paper_title_resolution` |
| `storage/workshops/workshop_site_download_receipt.json` | `acquisition.workshop_site_paper_download` |
| `storage/workshops/workshop_paper_sources.json` | `acquisition.workshop_paper_sources` registry |

PDFs live under `storage/workshops/papers/` (gitignored); receipts and
coverage files are tracked.

## Completion Checklist

- [ ] Coverage audit run on the final state; category counts reported with the
      list of workshops still without papers.
- [ ] Every "downloadable" claim backed by a verify or download receipt.
- [ ] Registered sources load without blockers
      (`load_registered_sources` / Stage 6 `workshop_paper_sources` binding).
- [ ] `python scripts/audit_conference_hardcoding.py` passes.

## Main-Proceedings Tracks Outside The Main Listing

Not every missing paper is a workshop paper. Compare the program's poster
list with the indexed papers; a large unmatched block usually means a second
proceedings track (CVPR 2026: 942 Findings papers at
`openaccess.thecvf.com/CVPR2026_findings`). Download it with
`scripts/download_cvf_proceedings_papers.py` so it is admitted as
main-conference papers with `venue_track`, not as workshop papers.

