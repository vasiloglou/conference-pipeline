---
name: conference-youtube-talk-harvest
description: Harvest conference talks (keynotes, orals, posters) from public YouTube against a program spine when official recordings are paywalled or absent. Bidirectional program↔YouTube matching, transcript-first via captions, receipts with explicit gaps. Use when asked to collect conference talk videos or transcripts from YouTube for any conference.
---

# Conference YouTube Talk Harvest

Harvest publicly available conference talk videos and their transcripts from
YouTube when the official conference recordings are paywalled, missing, or out
of scope. Conference-agnostic: the conference label, program spine, channels,
queries, and host allowlist are all inputs — nothing here is specific to any
one conference.

## When To Use

- Official conference session/keynote recordings sit behind a registration
  paywall and the milestone runs a no-official-recordings version.
- A conference (or workshop) posts author-uploaded talk videos, keynote
  recordings, or session streams on YouTube.
- Analysis needs talk transcripts and the videos are captioned, so paying for
  ASR or downloading audio is unnecessary.

## Owner Command

The registry owner is `acquisition.youtube_talk_harvest`
(`conference_pipeline.acquisition.youtube_talks`,
`scripts/harvest_youtube_talks.py`; usage docs
`docs/reference/stage-owner-audit.md#acquisition-youtube-talk-harvest`):

```
PYTHONPATH=src .venv/bin/python scripts/harvest_youtube_talks.py \
  --workspace conferences/<slug> --run-id <run-id> \
  --conference-label "<Conference YYYY>" \
  --program-spine <spine.json> \
  --source <youtube-channel-or-playlist-url> \
  --broad-query "<Conference YYYY> keynote" \
  --tavily --resume --summary
```

Explicit video list with the Gemini fallback:

```
set -a; source .env; set +a
PYTHONPATH=src .venv/bin/python scripts/harvest_youtube_talks.py \
  --workspace conferences/<slug> --run-id <run-id> \
  --video-list <videos.json> --output-dir <dir> \
  --gemini-fallback --gemini-budget-usd 10 --resume --summary
```

Requires yt-dlp (project `media` extra). The Tavily lane needs
`TAVILY_API_KEY` in the environment (source the repo-root `.env`); the key is
never written to receipts.

## Workflow

1. **Build the program spine.** Extract the talk list from the conference
   schedule/program surfaces (keynote/invited, oral, session, poster families)
   into a JSON list of `{title, speakers, family, forum_id, session}` rows.
   Reuse the title-spine families the topic-strategy stage already collects
   when available.
2. **Direction A — harvest → match.** Enumerate the conference's YouTube
   channels/playlists (`--source`, repeatable) and broad queries
   (`--broad-query`), then match harvested videos back to the spine by
   normalized title with speaker-surname boosts.
3. **Direction B — program → search.** For every still-unmatched item, run
   per-item yt-dlp search and, when enabled, the Tavily web-search lane
   restricted to the host allowlist. Priority families first: keynotes/invited
   and orals are searched before posters.
4. **Transcript tiers.** For each matched talk:
   - **Tier 1 — YouTube captions (free).** yt-dlp pulls manual subtitles before
     auto-generated ones so `transcript_source` (`youtube_caption_manual` |
     `youtube_caption_auto`) is honest provenance. No audio or video is
     downloaded. A "confirm you're not a bot" answer is an IP-level block, not a
     missing caption: the tier stops for the rest of the run (more requests
     only extend the block) and the receipt records `youtube_lane.blocked`.
   - **Tier 2 — Gemini on the YouTube URL (paid, opt-in).**
     `--gemini-fallback --gemini-budget-usd <usd>` (needs `GOOGLE_API_KEY`
     from `.env`) sends the public URL to the `transcription.youtube_url`
     registry model. Google fetches the video, so the local block does not
     apply. Gemini transcribes the speech (it does not read the caption
     track); provenance is `gemini_youtube_url_transcription`. Recordings are
     split into windows (`--gemini-window-seconds`, default 900 — 30-minute
     windows silently dropped speech) at `--gemini-fps` 0.02, the duration
     comes from the watch page (taken-down or private videos are recorded as
     `video_unavailable`), the last window stops short of the listed end, a
     truncated or sparse window (far fewer output tokens per second than
     speech) is re-asked in halves, every call is receipted, and the run halts
     before a window would cross the budget — never raise the budget without
     operator approval.
   - Anything still untranscribed (private video, unknown duration, Gemini
     off or failed) is marked `asr_fallback_required` for
     `transcription.session_speech` as an explicit per-talk decision.

   For an explicit list of videos (no program matching), use
   `--video-list <json>` with `{video_id, title, channel, duration_seconds}`
   rows; it checkpoints after every video, so `--resume` continues after a
   block or budget halt.
5. **Reconcile and receipt.** The inventory
   (`storage/youtube-talks/talk_inventory.json`) records per-talk
   `youtube_video_id`, match direction, method, and confidence; the receipt
   (`storage/youtube-talks/harvest_receipt.json`) records `allowed_hosts`,
   counts, the Tavily lane state, caption failures, and the transcript-first
   boundaries (`audio_downloaded: false`, `paywalled_sources_touched: false`).

## Gap Semantics (fail-honest)

- **Keynote/invited and oral misses are `must_investigate: true` gaps** and
  keep the receipt at `status: partial` — these families are expected to have
  videos, so escalate (widen sources, rerun Direction B, check the official
  program page for the exact upload title) before accepting the gap.
- Poster/author-upload misses are ordinary explicit gaps: authors in some
  regions skip the upload. Never silently omit an unmatched program item.

## Multi-Talk Recordings

One video is not always one talk. Workshop streams, oral sessions, and
tutorials are often uploaded as a single multi-hour recording. The harvest
records them whole and matches them to their event page; splitting is a
separate, governed step after binding:

1. Bind the harvest (next section). The lane transcript keeps timestamps.
2. `scripts/discover_harvest_talk_boundaries.py --estimate` prices windowed
   boundary discovery for every admitted recording of 45 minutes or more in
   the corpus inventory (base captions and every lane; caption files too) and
   records an eligibility decision for each. Get operator approval for that
   spend; the task budget cap is a halt, never raised automatically.
3. Run with `--provider-cost-approved --resume`, spot-check with
   `--report-only`, then re-bind harvest lanes with `--attach-talk-maps`.
   Base-caption and r-lane maps need no re-binding: Stage 6 attaches any map
   whose transcript -> map -> annotation digest chain verifies.

Caption-timed recordings split at cue resolution. Model-transcribed recordings
only have window timing, so their boundaries are approximate (synthetic cues
weighted by characters, flagged `synthetic_timestamp`). Speaker names are never
invented; unclear stretches become unattributed segments. Short named talks
(lightning papers) are kept, never folded into the previous talk. A workshop
is stamped on talks only when the event page names it; a fuzzy title match is
an unverified hint. A recording whose talk map fails validation keeps one
whole-recording talk and is not attached. Talk maps keep the raw per-window
model candidates, so a merge-rule change can be re-applied without new calls.

Auto captions scroll: each cue repeats the tail of the previous one. Raw
harvest transcripts therefore read every phrase two or three times. Never
index or prompt on the raw `text`; the lane binder and the corpus indexer
collapse the overlap (only between cues that abut in time).

## Talks Given In Another Language

The corpus is searched in English. For a recording spoken in Japanese,
Chinese, Korean, or another language, re-transcribe it into English with
`scripts/harvest_youtube_talks.py --video-list <list> --gemini-fallback
--gemini-only --gemini-output-language en` (same governed
`transcription.youtube_url` task and budget), then bind that harvest with
`build_youtube_harvest_evidence_lanes.py --supersede-existing` so its lane
replaces the original-language transcripts of the same videos (base inventory
records still win). The original sidecars stay on disk. Windows that loop are
re-asked in halves automatically; interleaved `[NO SPEECH]` markers are
stripped. Split long ones with the talk-discovery steps above.

## Model Transcripts Can Be Invented

Gemini URL transcription can return fluent speech that was never said: on
CVPR 2026, seven silent poster clips all "transcribed" as the same breakfast
recipe. After binding, read the `suspect_model_transcripts` the binder lists
(short model transcripts sharing no word with their title) and record rejects
in `storage/transcription-ocr/evidence-lanes/transcript_exclusions.json`; the
binder turns them into coverage gaps.

## Binding Into The Corpus

A harvest directory reaches the index only through transcript evidence lanes:

```bash
.venv/bin/python scripts/build_youtube_harvest_evidence_lanes.py \
  --workspace conferences/<slug> --harvest-dir storage/youtube-talks/<run> --dry-run
```

Drop `--dry-run` to bind. Then run the conference's corpus-inventory-index
stage; indexing is incremental, so only the new transcripts are embedded.
Declare the harvest directory families in `storage/storage_manifest.json`
(inventories and receipts git-class, transcripts and captions backup-class)
and regenerate the workspace `.gitignore` before committing anything from it.

## Boundaries

- Public YouTube surfaces only; never route around a conference paywall.
- Host allowlist is explicit (`--allowed-host`, default `youtube.com` +
  `youtu.be`); receipts record `allowed_hosts`. Videos on other hosts are
  dropped, not followed.
- `--pilot-limit` is a development-only spine cap, stamped into the receipt;
  live production corpus commands must not use item-count caps.
- Live runs in a production milestone stay behind the corpus-freeze and
  provider-cost gates of the generated production plan; this skill does not
  bypass them.

## Testing

`tests/test_youtube_talk_harvest.py` covers matching, priority ordering,
caption provenance, Tavily fallback, resume, host filtering, and receipt
boundaries with a mocked yt-dlp runner; `tests/test_youtube_transcripts.py`
covers the bot-gate stop, tier order, Gemini windowing/splitting/retries, the
budget halt, video-list resume, and the CLI flag contract — run both after any
change to the modules or script.
