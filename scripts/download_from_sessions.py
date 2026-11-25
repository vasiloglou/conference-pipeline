#!/usr/bin/env python3
"""Download assets directly from chosen sessions in sessions.json.

This script reads sessions.json (where chosen='Y') and downloads assets
(audio + slides) for each session that has a SlidesLive embed.

Supports automatic resume - if interrupted, re-running will skip already
completed sessions and continue from where it left off.

Usage:
    python download_from_sessions.py
    python download_from_sessions.py --limit 5
    python download_from_sessions.py --sessions-file sessions.json
    python download_from_sessions.py --retry-errors  # Retry failed sessions
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession
from conference_pipeline.config import load_settings
from conference_pipeline.models import Presentation, PresentationStatus
from conference_pipeline.pipeline import NeuripsPipeline
from conference_pipeline.presentation_parser import PresentationPageParser
from conference_pipeline.storage import StorageBackend
from src.utils.logging import setup_logging


async def main(
    sessions_file: Path,
    limit: int | None = None,
    retry_errors: bool = False,
):
    """Download assets from chosen sessions.

    Supports resuming from interruptions - already completed sessions are skipped.
    """

    settings = load_settings()

    # Setup logging
    log_file = settings.storage_root / "logs" / "session_asset_download.log"
    setup_logging(level="INFO", log_file=log_file)

    print(f"Logs will be saved to: {log_file}")
    print()
    print(f"Downloading assets from sessions for NeurIPS {settings.year}")
    print(f"Sessions file: {sessions_file}")
    print(f"Storage root: {settings.storage_root}")
    print()

    # Load sessions
    with open(sessions_file, 'r') as f:
        sessions_data = json.load(f)

    # Filter to chosen sessions
    chosen = [s for s in sessions_data.get('sessions', []) if s.get('chosen') == 'Y']
    print(f"Found {len(chosen)} chosen sessions")

    # All session types can have SlidesLive embeds
    sessions_with_videos = chosen

    # Breakdown by type
    by_type = {}
    for s in sessions_with_videos:
        stype = s['session_type']
        by_type[stype] = by_type.get(stype, 0) + 1

    print("\nBreakdown by type:")
    for stype, count in sorted(by_type.items()):
        print(f"  {stype}: {count}")
    print()

    # Apply limit
    if limit:
        sessions_with_videos = sessions_with_videos[:limit]
        print(f"Limiting to first {limit} sessions")
        print()

    # Initialize storage
    storage = StorageBackend(settings)

    # Initialize browser session
    print("Initializing browser session...")
    browser_session = NeuripsBrowserSession(settings)
    await browser_session.initialize()

    # Login
    if settings.neurips_email and settings.neurips_password:
        print("Logging in to NeurIPS...")
        success = await browser_session.login()
        if success:
            print("✓ Login successful!")
        else:
            print("⚠ Login reported failure - continuing anyway")

    http_session = await NeuripsSession.create(settings)

    try:
        # Initialize pipeline
        pipeline = NeuripsPipeline(settings, http_session, browser_session)
        await pipeline.initialize()

        # Check for already completed presentations (resume support)
        completed_ids = set()
        error_ids = set()
        for pres in storage.load_all_presentations():
            if pres.status in [
                PresentationStatus.ASSETS_DOWNLOADED,
                PresentationStatus.TRANSCRIBED,
                PresentationStatus.SLIDES_EXTRACTED,
                PresentationStatus.OCR_DONE,
                PresentationStatus.REFERENCES_EXTRACTED,
                PresentationStatus.PAPERS_DOWNLOADED,
            ]:
                completed_ids.add(pres.id)
            elif pres.status == PresentationStatus.ERROR:
                error_ids.add(pres.id)

        # If retry_errors is set, don't skip error sessions
        if retry_errors:
            print(f"Retry mode: will retry {len(error_ids)} previously failed sessions")
        else:
            completed_ids.update(error_ids)

        # Filter out already completed sessions
        pending_sessions = [s for s in sessions_with_videos if s['id'] not in completed_ids]
        skipped = len(sessions_with_videos) - len(pending_sessions)

        print("\nDownloading assets...")
        print("=" * 80)

        if skipped > 0:
            print(f"Skipping {skipped} already completed sessions (resume mode)")
            print(f"Processing {len(pending_sessions)} remaining sessions")

        if not pending_sessions:
            print("All sessions already completed!")

        # Process each session
        for i, sess_dict in enumerate(pending_sessions, 1):
            print(f"\n[{i}/{len(pending_sessions)}] {sess_dict['title'][:70]}...")
            print(f"  Type: {sess_dict['session_type']}")
            print(f"  URL: {sess_dict['url']}")

            # Check if presentation exists but incomplete
            existing = storage.load_presentation(sess_dict['id'])
            if existing:
                pres = existing
                print(f"  Resuming from status: {pres.status.value}")
                # When retrying errors, start from the canonical session URL in sessions.json.
                # The stored `pres.neurips_url` may point to a nested poster page (oral → poster)
                # that does not contain the `/session/<id>` link with the actual SlidesLive embed.
                if retry_errors and sess_dict.get("url") and pres.neurips_url != sess_dict["url"]:
                    print(f"  Resetting URL for retry: {pres.neurips_url} -> {sess_dict['url']}")
                    pres.neurips_url = sess_dict["url"]
            else:
                # Create a Presentation object from the session
                pres = Presentation(
                    id=sess_dict['id'],
                    title=sess_dict['title'],
                    neurips_url=sess_dict['url'],
                    track=sess_dict['session_type'],
                    session_title=sess_dict.get('title'),
                    status=PresentationStatus.DISCOVERED,
                )

            try:
                # Enrich to get SlidesLive ID if we don't have it
                if not pres.slideslive_id:
                    print(f"  Checking for video embed...")
                    parser = PresentationPageParser(http_session, settings, browser_session)
                    pres = await parser.enrich_presentation(pres)

                if pres.slideslive_id:
                    print(f"  ✓ Found SlidesLive ID: {pres.slideslive_id}")

                    # Download assets
                    print(f"  Downloading assets...")
                    pres = await pipeline.download_assets_for_presentation(pres)

                    pres.status = PresentationStatus.ASSETS_DOWNLOADED
                    print(f"  ✓ Assets downloaded")
                else:
                    print(f"  ✗ No video embed found")
                    pres.status = PresentationStatus.ERROR
                    pres.error_message = "No SlidesLive embed found"

                # Save
                storage.save_presentation(pres)

            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Error: {error_msg}")
                pres.status = PresentationStatus.ERROR
                pres.error_message = error_msg
                storage.save_presentation(pres)

            # Rate limiting
            await asyncio.sleep(settings.request_delay_seconds)

        print("\n" + "=" * 80)
        print("✓ Download completed!")

        # Final stats
        all_presentations = storage.load_all_presentations()
        completed = len([p for p in all_presentations if p.status == PresentationStatus.ASSETS_DOWNLOADED])
        error_presentations = [p for p in all_presentations if p.status == PresentationStatus.ERROR]

        print(f"\nFinal statistics:")
        print(f"  Completed: {completed}")
        print(f"  Errors: {len(error_presentations)}")

        # Report ALL failed sessions (not just from this run)
        if error_presentations:
            print(f"\n" + "=" * 80)
            print(f"All failed sessions ({len(error_presentations)}):")
            print("=" * 80)
            for pres in error_presentations:
                print(f"\n  {pres.title[:70] if pres.title else pres.id}...")
                print(f"    ID: {pres.id}")
                print(f"    Track: {pres.track or 'unknown'}")
                print(f"    URL: {pres.neurips_url or 'N/A'}")
                print(f"    Error: {pres.error_message or 'Unknown error'}")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")

    finally:
        await browser_session.close()
        await http_session.close()
        if pipeline.http_client:
            await pipeline.http_client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download assets from chosen sessions"
    )
    parser.add_argument(
        "--sessions-file",
        type=Path,
        default=Path("sessions.json"),
        help="Path to sessions.json file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of sessions to process (for testing)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry sessions that previously failed with errors",
    )

    args = parser.parse_args()

    asyncio.run(main(
        sessions_file=args.sessions_file,
        limit=args.limit,
        retry_errors=args.retry_errors,
    ))
