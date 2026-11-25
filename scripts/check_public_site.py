#!/usr/bin/env python3
"""Check a built public site or hub: links, accessibility lint, mobile render (v6.0 Phase 4).

    python scripts/check_public_site.py <site dir> [--no-render]

Exit codes: 0 every check clean (a skipped render is reported, never counted clean);
1 a finding; 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.publishing.site_checks import check_links, lint_accessibility, render_mobile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--no-render", action="store_true", help="skip the Playwright mobile render")
    args = parser.parse_args(argv)
    if not args.site_dir.is_dir():
        print("error: not a directory", file=sys.stderr)
        return 2
    code = 0
    checks = [("links", check_links), ("accessibility", lint_accessibility)]
    if not args.no_render:
        checks.append(("mobile render", render_mobile))
    for name, check in checks:
        result = check(args.site_dir)
        if result.skipped:
            print(f"{name}: SKIPPED ({result.skipped})")
            continue
        print(f"{name}: {len(result.findings)} finding(s)")
        for finding in result.findings[:50]:
            print(f"  {finding}")
        code = max(code, result.exit_code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
