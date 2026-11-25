#!/usr/bin/env python3
"""Export a logged-in browser's cookies for one domain to a Netscape cookie file.

Conference virtual sites (icml.cc, iclr.cc, neurips.cc) and other providers
gate some pages/media behind an attendee login. This helper reuses an existing
browser session: it reads cookies straight from the local browser profile via
yt-dlp's cookie backend, keeps only the cookies whose domain matches
``--domain``, and writes them in Netscape ``cookies.txt`` format for the
pipeline's cookie-file consumers (e.g. ``ICML_COOKIES_FILE``).

No password is entered here and no full cookie jar is written: only the
requested domain's cookies land in the output file. Point the relevant
``*_COOKIES_FILE`` env var at the output path.

Example:
    python scripts/export_browser_cookies.py \
        --browser chrome --domain icml.cc \
        --output ~/.secrets/icml2026_cookies.txt

Then add ``ICML_COOKIES_FILE=~/.secrets/icml2026_cookies.txt`` to ``.env``.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from http.cookiejar import Cookie
from pathlib import Path

SUPPORTED_BROWSERS = (
    "chrome",
    "chromium",
    "brave",
    "edge",
    "firefox",
    "safari",
    "opera",
    "vivaldi",
)


def _domain_matches(cookie_domain: str, target: str) -> bool:
    cd = cookie_domain.lstrip(".").lower()
    t = target.lstrip(".").lower()
    return cd == t or cd.endswith("." + t)


def export_cookies(browser: str, domain: str, output: Path) -> dict[str, object]:
    try:
        from yt_dlp.cookies import YoutubeDLCookieJar, extract_cookies_from_browser
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "yt-dlp is required for cookie export; install it into the venv "
            "(`.venv/bin/python -m pip install yt-dlp`)."
        ) from exc

    full_jar = extract_cookies_from_browser(browser)
    out_jar = YoutubeDLCookieJar()
    kept_names: list[str] = []
    for cookie in full_jar:
        if _domain_matches(cookie.domain, domain):
            out_jar.set_cookie(cookie)
            kept_names.append(cookie.name)

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    # ignore_discard/ignore_expires so session cookies are written too.
    out_jar.save(str(output), ignore_discard=True, ignore_expires=True)
    # Restrict to owner read/write; this file authenticates as the user.
    output.chmod(stat.S_IRUSR | stat.S_IWUSR)

    return {
        "browser": browser,
        "domain": domain,
        "output": str(output),
        "cookie_count": len(kept_names),
        "cookie_names": sorted(set(kept_names)),
        "secret_values_recorded": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one domain's browser cookies to a Netscape cookie file."
    )
    parser.add_argument(
        "--browser",
        default="chrome",
        choices=SUPPORTED_BROWSERS,
        help="Browser profile to read cookies from (default: chrome).",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Cookie domain to keep, e.g. icml.cc (subdomains included).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the Netscape cookie file (keep it outside the repo).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_cookies(args.browser, args.domain, args.output)
    if result["cookie_count"] == 0:
        print(
            f"no {args.domain} cookies found in {args.browser}; is the site "
            "logged in in that browser profile?",
            file=sys.stderr,
        )
        return 1
    # Names only; never print cookie values.
    print(
        f"exported {result['cookie_count']} {args.domain} cookies "
        f"({', '.join(result['cookie_names'])}) to {result['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
