#!/usr/bin/env python3
"""Discover NeurIPS sessions and output to JSON for manual filtering.

This script discovers sessions by type (tutorials, workshops, keynotes, etc.)
and outputs them to a JSON file. You should then review the file and set
the 'chosen' field to 'Y' or 'N' for each session.

Usage:
    python discover_sessions.py                    # Uses year from .env or defaults to 2025
    python discover_sessions.py --output my_sessions.json
    python discover_sessions.py --year 2024        # Override to use 2024 (already available)

Note: NeurIPS 2025 won't be available until December 2025.
      Use --year 2024 to test with existing conference data.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conference_pipeline.config import load_settings
from conference_pipeline.models import SessionType, TARGET_SESSION_TYPES
from conference_pipeline.session_scraper import SessionScraper
from conference_pipeline.auth import NeuripsSession, NeuripsBrowserSession
from src.utils.logging import setup_logging


async def main(
    output_path: Path,
    year: int | None = None,
    use_browser: bool = True,
):
    """Discover sessions and save to JSON.

    Args:
        output_path: Output JSON file path.
        year: Conference year (overrides settings).
        use_browser: Whether to use browser for scraping.
    """
    setup_logging(level="INFO")

    # Load settings
    settings = load_settings()
    if year:
        settings.year = year
        settings.virtual_base_url = f"https://neurips.cc/virtual/{year}"
        settings.storage_root = Path(f"./data/neurips_{year}")

    print(f"Discovering sessions for NeurIPS {settings.year}")
    print(f"Virtual site: {settings.virtual_base_url}")
    print(f"Output file: {output_path}")
    print()

    print("Target session types:")
    for st in TARGET_SESSION_TYPES:
        print(f"  - {st.value}")
    print()

    # Create sessions
    browser_session = None
    if use_browser:
        print("Initializing browser session...")
        browser_session = NeuripsBrowserSession(settings)
        await browser_session.initialize()

        # Try to login if credentials available
        if settings.neurips_email and settings.neurips_password:
            print("Logging in to NeurIPS...")
            success = await browser_session.login()
            if success:
                print("Login successful!")
            else:
                print("Login failed - continuing without authentication")

    http_session = await NeuripsSession.create(settings)

    try:
        # Create scraper
        scraper = SessionScraper(http_session, settings, browser_session)

        # Discover sessions
        print("\nDiscovering sessions...")
        sessions = await scraper.discover_sessions(TARGET_SESSION_TYPES)

        # Sort by type and title
        sessions.sort(key=lambda s: (s.session_type.value, s.title))

        # Save to JSON
        await scraper.save_sessions_json(sessions, output_path)

        # Print summary
        print(f"\nDiscovered {len(sessions)} sessions:")
        by_type: dict[str, int] = {}
        for s in sessions:
            by_type[s.session_type.value] = by_type.get(s.session_type.value, 0) + 1

        for session_type, count in sorted(by_type.items()):
            print(f"  {session_type}: {count}")

        print(f"\nSessions saved to: {output_path}")
        print("\nNext steps:")
        print(f"  1. Open {output_path} in a text editor")
        print("  2. For each session, set 'chosen' to 'Y' or 'N'")
        print("  3. Run the pipeline with the filtered sessions file")

    finally:
        if browser_session:
            await browser_session.close()
        await http_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover NeurIPS sessions for manual filtering"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("sessions.json"),
        help="Output JSON file (default: sessions.json)",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Conference year (default: from settings)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't use browser for scraping (faster but may miss content)",
    )

    args = parser.parse_args()

    asyncio.run(main(
        output_path=args.output,
        year=args.year,
        use_browser=not args.no_browser,
    ))
