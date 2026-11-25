#!/usr/bin/env python3
"""Fetch OpenReview workshop group metadata (full title, subtitle, website).

The workshop inventory only records venue ids (short names). Coverage
matching against the official schedule needs each group's full title and
website: short names mis-match renamed workshops and sub-tracks. Reads the
parent venue from ``urls.openreview_workshop_parent`` in the workspace
``config.yaml`` unless ``--venue-prefix`` is given. Authenticated with
OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD from the environment (source the
repo-root .env); only env var names are recorded.

    PYTHONPATH=src .venv/bin/python scripts/fetch_openreview_workshop_groups.py --workspace conferences/<slug>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.acquisition.workshop_coverage import fetch_openreview_workshop_groups  # noqa: E402
from conference_pipeline.conference_identity import load_conference_identity  # noqa: E402
from conference_pipeline.openreview_auth import AUTH_MODES, load_openreview_auth  # noqa: E402

USER_AGENT = "conference-pipeline/openreview-workshop-groups"
DEFAULT_OUTPUT = Path("storage/workshops/openreview_workshop_groups.json")


def venue_prefix_from_config(workspace: Path) -> str:
    url = load_conference_identity(workspace).url("openreview_workshop_parent")
    prefix = (urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("id") or [""])[0]
    if not prefix:
        raise ValueError("set urls.openreview_workshop_parent in config.yaml or pass --venue-prefix")
    return prefix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--venue-prefix", default=None)
    parser.add_argument("--output", type=Path, default=None, help=f"Default: <workspace>/{DEFAULT_OUTPUT}")
    parser.add_argument("--openreview-auth", choices=AUTH_MODES, default="required")
    args = parser.parse_args(argv)
    try:
        prefix = args.venue_prefix or venue_prefix_from_config(args.workspace)
        auth = load_openreview_auth(USER_AGENT, mode=args.openreview_auth)
    except Exception as exc:  # noqa: BLE001 - clean, secret-free failure
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def get_json(path: str, params: dict) -> dict:
        url = f"https://api2.openreview.net/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=auth.headers({"User-Agent": USER_AGENT}))
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    groups = fetch_openreview_workshop_groups(prefix, get_json)
    output = args.output or args.workspace / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.openreview_workshop_groups",
        "venue_prefix": prefix,
        "group_count": len(groups),
        "groups": groups,
        **auth.receipt_fields(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"openreview_workshop_groups: {len(groups)} groups -> {output}")
    return 0 if groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
