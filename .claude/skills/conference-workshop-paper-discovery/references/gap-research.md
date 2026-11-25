# Workshop Gap Research (agent brief)

Use this for the workshops the coverage audit marks `openreview_titles_only`,
`openreview_venue_empty` or `no_source`. Split them into batches of about 15
and run one read-only web research agent per batch, in parallel. Paste the
brief below with the batch list, filling in the placeholders.

## Batch input per workshop

`schedule_id`, title, website (from the schedule), and the empty or
titles-only OpenReview venue if the audit matched one.

## Agent brief

> Read-only web research on `<Conference>` workshops. Do not download papers
> or modify anything except the JSON files named below.
>
> These workshops are on the official `<Conference>` schedule, but we have no
> papers for them from OpenReview or the proceedings hosts
> (`<proceedings hosts>`). For each workshop, find out whether it accepted
> papers or extended abstracts, and where they are published.
>
> 1. Fetch the workshop website. Look for Accepted Papers, Papers, Posters,
>    Program, Call for Papers, Proceedings and Challenge pages, and follow
>    in-site links.
> 2. Search the web for `"<name> <Conference> accepted papers"` and
>    `"<acronym> <Conference> workshop"`, plus arXiv when relevant. Check
>    other OpenReview groups, CMT, Google Drive or Sheets, and GitHub.
> 3. For lists longer than ~15, parse raw HTML (`curl -sL -A "Mozilla/5.0"`
>    plus a parser) instead of a summarizing fetch, which drops items. For
>    Google Sheets use `/export?format=csv&gid=<gid>`. For JS-rendered pages
>    find the data file the page loads. Skip HTML-commented-out lists.
> 4. Write `<research dir>/<short>.json`:
>    `{"workshop": "<short>", "schedule_id": "<id>", "classification": "papers_published_on_site|papers_listed_no_pdfs|call_for_papers_but_no_accepted_list_found|no_papers|unclear", "source_url": "<page>", "stated_total": <int|null>, "papers": [{"title": "<verbatim>", "authors": ["..."], "link": "<pdf|arXiv|drive|openreview|null>"}], "notes": "<format (non-archival, extended abstracts, talks/challenge only), partial lists, unreadable pages>"}`
> 5. Copy titles verbatim. Never invent titles, counts or URLs. When you
>    cannot verify something, say so in `notes` and classify `unclear`.
>
> Reply with a compact table (short | classification | papers | where |
> evidence URL) and 2–3 observations, under ~700 words.

## After the agents return

- Treat every "has PDFs" statement as a claim. Step 7 (`--verify-only`) is the
  evidence.
- Confirm the file count equals the batch size, and that `stated_total`
  matches `len(papers)` or `notes` explains why not.
- `no_papers` needs the page's own statement (talks, panel or challenge only;
  "no paper submission"). Without one, use `unclear`.
