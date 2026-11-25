#!/usr/bin/env python3
"""Harvest conference talks from YouTube against a program spine.

Conference-general owner command for ``acquisition.youtube_talk_harvest``.
Reads a program-spine JSON (keynotes, orals, posters, ... — each row with
``title``, optional ``speakers``, ``family``, ``forum_id``), then runs the
bidirectional match:

- Direction A: enumerate ``--source`` channels/playlists and ``--broad-query``
  searches, match harvested videos back to the spine.
- Direction B: per-item yt-dlp search for unmatched items, keynote/invited
  and oral families first. Paid Tavily recovery is intentionally unavailable
  from this generic CLI: the owning production runner must first bind an exact
  priority-gap frontier, cost authority, and provider preflight.

Transcript-first: captions are pulled with yt-dlp (manual before auto for
provenance); no audio or video is downloaded. A YouTube "confirm you're not a
bot" gate stops the caption tier for the rest of the run instead of being
recorded as a missing caption. Talks without captions are marked
``asr_fallback_required`` for the transcription stage, unless the paid Gemini
tier is enabled with ``--gemini-fallback --gemini-budget-usd <usd>``: Gemini
then transcribes the public YouTube URL server-side (model from the
``transcription.youtube_url`` registry task, windowed, every call receipted,
halting before the budget is crossed).

``--video-list <json>`` skips program matching and runs the same transcript
tiers over an explicit list of ``{video_id, title, channel, duration_seconds}``
rows, checkpointing after every video so ``--resume`` continues where a block
or budget halt stopped it.

Secrets: this CLI never reads a Tavily key. Receipts record ``allowed_hosts``
and sanitized URLs. A runner-owned paid lane uses the canonical provider
receipt context so secret-bearing request fields are redacted before writes.

Example:

    PYTHONPATH=src .venv/bin/python scripts/harvest_youtube_talks.py \
      --workspace conferences/<slug> --run-id <run-id> \
      --conference-label "<Conference YYYY>" \
      --program-spine <spine.json> \
      --source https://www.youtube.com/@ComputerVisionFoundation \
      --broad-query "<Conference YYYY> keynote" \
      --resume --summary

    # Transcripts for an explicit video list, Gemini for what YouTube won't give:
    set -a; source .env; set +a
    PYTHONPATH=src .venv/bin/python scripts/harvest_youtube_talks.py \
      --workspace conferences/<slug> --run-id <run-id> \
      --video-list <videos.json> --output-dir <dir> \
      --gemini-fallback --gemini-budget-usd 10 --resume --summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:  # allow running both as ``python scripts/...`` and via package import
    from conference_pipeline.acquisition.youtube_talks import (
        DEFAULT_ALLOWED_HOSTS,
        DEFAULT_MATCH_THRESHOLD,
        DEFAULT_PER_ITEM_MAX_RESULTS,
        HarvestConfig,
        YouTubeTalkHarvestError,
        harvest_youtube_talks,
        load_program_spine,
    )
    from conference_pipeline.acquisition.youtube_transcripts import (
        DEFAULT_GEMINI_FPS,
        DEFAULT_WINDOW_SECONDS,
        GeminiBudgetExhausted,
        GeminiTranscriber,
        VideoListConfig,
        load_video_list,
        transcribe_video_list,
    )
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from conference_pipeline.acquisition.youtube_talks import (
        DEFAULT_ALLOWED_HOSTS,
        DEFAULT_MATCH_THRESHOLD,
        DEFAULT_PER_ITEM_MAX_RESULTS,
        HarvestConfig,
        YouTubeTalkHarvestError,
        harvest_youtube_talks,
        load_program_spine,
    )
    from conference_pipeline.acquisition.youtube_transcripts import (
        DEFAULT_GEMINI_FPS,
        DEFAULT_WINDOW_SECONDS,
        GeminiBudgetExhausted,
        GeminiTranscriber,
        VideoListConfig,
        load_video_list,
        transcribe_video_list,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--conference-label",
        default=None,
        help="Public conference label, e.g. 'CVPR 2026' (used in search queries "
        "and stripped from titles during matching). Required with --program-spine.",
    )
    parser.add_argument(
        "--program-spine",
        type=Path,
        default=None,
        help="JSON list (or {'items': [...]}) of program rows with title, "
        "speakers, family (keynote/invited/oral/...), forum_id.",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        default=None,
        help="JSON list (or {'videos': [...]}) of {video_id, title, channel, "
        "duration_seconds} rows to transcribe without program matching.",
    )
    parser.add_argument(
        "--gemini-fallback",
        action="store_true",
        help="Enable the paid Gemini tier for videos whose YouTube captions are "
        "missing or bot-gated (needs GOOGLE_API_KEY and --gemini-budget-usd).",
    )
    parser.add_argument(
        "--gemini-budget-usd",
        type=float,
        default=None,
        help="Run budget for the Gemini tier; the run halts before crossing it. "
        "Must not exceed the transcription.youtube_url registry cap.",
    )
    parser.add_argument(
        "--gemini-window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS
    )
    parser.add_argument("--gemini-fps", type=float, default=DEFAULT_GEMINI_FPS)
    parser.add_argument(
        "--gemini-output-language",
        choices=["en"],
        default=None,
        help="Transcribe into English (for talks given in another language). "
        "Default: verbatim in the language spoken.",
    )
    parser.add_argument(
        "--gemini-only",
        action="store_true",
        help="Video-list mode: skip the free caption tier and transcribe every "
        "video with Gemini. Nothing contacts YouTube, so no bot gate applies.",
    )
    parser.add_argument(
        "--gemini-concurrency",
        type=int,
        default=1,
        help="Videos transcribed in parallel (requires --gemini-only; the "
        "caption tier stays serial so it can be paced).",
    )
    parser.add_argument(
        "--min-sleep-seconds",
        type=float,
        default=15.0,
        help="Video-list mode: minimum jittered pause between YouTube caption requests.",
    )
    parser.add_argument("--max-sleep-seconds", type=float, default=30.0)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="YouTube channel or playlist URL to enumerate (repeatable).",
    )
    parser.add_argument(
        "--broad-query",
        action="append",
        default=[],
        help="Broad Direction-A search query (repeatable).",
    )
    parser.add_argument(
        "--per-item-search",
        dest="per_item_search",
        action="store_true",
        default=True,
        help="Run Direction-B per-item searches for unmatched items (default).",
    )
    parser.add_argument(
        "--no-per-item-search", dest="per_item_search", action="store_false"
    )
    parser.add_argument(
        "--per-item-max-results",
        type=int,
        default=DEFAULT_PER_ITEM_MAX_RESULTS,
    )
    parser.add_argument(
        "--tavily",
        action="store_true",
        help="Enable the Tavily web-search lane for still-unmatched items "
        "(requires TAVILY_API_KEY in the environment).",
    )
    parser.add_argument(
        "--transcripts",
        dest="transcripts",
        action="store_true",
        default=True,
        help="Pull YouTube captions for matched talks (default; transcript-first).",
    )
    parser.add_argument("--no-transcripts", dest="transcripts", action="store_false")
    parser.add_argument(
        "--transcript-lang",
        action="append",
        default=[],
        help="Caption language code (repeatable; default: en).",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Embedded-media host allowlist entry (repeatable; default: "
        "youtube.com, youtu.be).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <workspace>/storage/youtube-talks).",
    )
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument(
        "--pilot-limit",
        type=int,
        default=None,
        help="EXPLICIT development cap on spine size (highest-priority items "
        "kept). Recorded in the receipt; live production corpus commands "
        "must not use item-count caps.",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Reject Tavily, pilot caps, and non-YouTube allowlist entries for a "
        "named public-only production lane. Paid escalation is a separate "
        "governed operation.",
    )
    parser.add_argument("--summary", action="store_true")
    return parser


def public_only_argument_errors(args: argparse.Namespace) -> list[str]:
    """Return strict public-lane violations without changing generic defaults."""

    errors: list[str] = []
    requested_hosts = tuple(args.allowed_host) or DEFAULT_ALLOWED_HOSTS
    if tuple(requested_hosts) != DEFAULT_ALLOWED_HOSTS:
        errors.append("--public-only requires allowed hosts exactly youtube.com,youtu.be")
    if args.pilot_limit is not None:
        errors.append("--public-only rejects --pilot-limit")
    if args.tavily:
        errors.append("--public-only rejects --tavily; use the governed paid-search lane")
    if getattr(args, "gemini_fallback", False) or getattr(args, "gemini_only", False):
        errors.append("--public-only rejects --gemini-fallback; it is a paid provider tier")
    return errors


def gemini_argument_errors(args: argparse.Namespace, environ: dict[str, str] | None = None) -> list[str]:
    """Validate the paid Gemini tier flags before any provider is constructed."""

    if not args.gemini_fallback:
        return (
            ["--gemini-only requires --gemini-fallback"] if getattr(args, "gemini_only", False) else []
        )
    env = os.environ if environ is None else environ
    errors: list[str] = []
    if args.gemini_budget_usd is None or args.gemini_budget_usd <= 0:
        errors.append("--gemini-fallback requires a positive --gemini-budget-usd")
    if not (env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")):
        errors.append("--gemini-fallback requires GOOGLE_API_KEY (source the repository .env)")
    if args.gemini_concurrency < 1:
        errors.append("--gemini-concurrency must be at least 1")
    if args.gemini_concurrency > 1 and not args.gemini_only:
        errors.append("--gemini-concurrency > 1 requires --gemini-only")
    if args.gemini_window_seconds < 300:
        errors.append("--gemini-window-seconds must be at least 300")
    if not 0 < args.gemini_fps <= 1:
        errors.append("--gemini-fps must be in (0, 1]")
    return errors


def build_gemini_transcriber(args: argparse.Namespace) -> "GeminiTranscriber | None":
    if not args.gemini_fallback:
        return None
    return GeminiTranscriber.from_registry(
        api_key=os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        budget_usd=float(args.gemini_budget_usd),
        workspace=args.workspace,
        run_id=args.run_id,
        window_seconds=args.gemini_window_seconds,
        fps=args.gemini_fps,
        output_language=getattr(args, "gemini_output_language", None),
    )


def run_video_list(args: argparse.Namespace) -> int:
    videos = load_video_list(args.video_list)
    try:
        gemini = build_gemini_transcriber(args)
    except GeminiBudgetExhausted as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    config = VideoListConfig(
        workspace=args.workspace,
        videos=videos,
        output_dir=args.output_dir or (args.workspace / "storage" / "youtube-talks"),
        run_id=args.run_id,
        langs=tuple(args.transcript_lang) or ("en",),
        resume=args.resume,
        min_sleep_seconds=args.min_sleep_seconds,
        max_sleep_seconds=args.max_sleep_seconds,
        gemini=gemini,
        gemini_only=args.gemini_only,
        concurrency=args.gemini_concurrency,
    )
    receipt = transcribe_video_list(config)
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt_path": receipt["receipt_path"],
                    "inventory_path": receipt["inventory_path"],
                    "counts": receipt["counts"],
                    "youtube_lane": receipt["youtube_lane"],
                    "gemini_lane": receipt["gemini_lane"],
                    "halted_reason": receipt["halted_reason"],
                },
                indent=2,
            )
        )
    return 0 if receipt["status"] in {"succeeded", "partial"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.public_only:
        for error in public_only_argument_errors(args):
            parser.error(error)
    if args.tavily:
        parser.error(
            "--tavily cannot dispatch from the generic harvester; use the "
            "CVPR production runner's governed priority-gap lane"
        )
    for error in gemini_argument_errors(args):
        parser.error(error)
    if (args.program_spine is None) == (args.video_list is None):
        parser.error("exactly one of --program-spine or --video-list is required")
    if args.program_spine is not None and not args.conference_label:
        parser.error("--program-spine requires --conference-label")
    if not args.workspace.is_dir():
        print(f"ERROR: workspace {args.workspace} is not a directory", file=sys.stderr)
        return 2
    if args.gemini_only and args.video_list is None:
        parser.error("--gemini-only is only available in --video-list mode")
    if args.video_list is not None:
        if not args.video_list.is_file():
            print(f"ERROR: video list {args.video_list} not found", file=sys.stderr)
            return 2
        return run_video_list(args)
    if not args.program_spine.is_file():
        print(
            f"ERROR: program spine {args.program_spine} not found", file=sys.stderr
        )
        return 2
    config = HarvestConfig(
        workspace=args.workspace,
        conference_label=args.conference_label,
        program_spine=load_program_spine(args.program_spine),
        sources=args.source,
        broad_queries=args.broad_query,
        allowed_hosts=tuple(args.allowed_host) or DEFAULT_ALLOWED_HOSTS,
        per_item_search=args.per_item_search,
        per_item_max_results=args.per_item_max_results,
        tavily_api_key=None,
        transcripts_enabled=args.transcripts,
        transcript_langs=tuple(args.transcript_lang) or ("en",),
        match_threshold=args.match_threshold,
        output_dir=args.output_dir,
        receipt_path=args.receipt_path,
        resume=args.resume,
        rate_limit_seconds=args.rate_limit_seconds,
        pilot_limit=args.pilot_limit,
        run_id=args.run_id,
    )
    try:
        config.gemini_transcriber = build_gemini_transcriber(args)
        receipt = harvest_youtube_talks(config)
    except GeminiBudgetExhausted as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except YouTubeTalkHarvestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt_path": receipt["receipt_path"],
                    "inventory_path": receipt["inventory_path"],
                    "counts": receipt["counts"],
                    "tavily_lane": receipt["tavily_lane"],
                    "youtube_lane": receipt["youtube_lane"],
                    "gemini_lane": receipt["gemini_lane"],
                },
                indent=2,
            )
        )
    return 0 if receipt["status"] in {"succeeded", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
