#!/usr/bin/env python3
"""Audit workshop paper coverage: official schedule vs every paper source.

For each workshop on the official schedule, reports which sources hold its
papers (OpenReview, proceedings hosts such as CVF Open Access, registered
workshop-site/arXiv sources) and which workshops have none. Writes
``storage/workshops/workshop_paper_coverage.json`` and ``.md``. Offline.

The schedule file is conference-specific; name its fields:

    PYTHONPATH=src .venv/bin/python scripts/audit_workshop_paper_coverage.py \\
        --workspace conferences/<slug> --schedule conferences/<slug>/storage/workshops/<schedule>.json \\
        --id-field <id key> --title-field title --website-field <website key>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.acquisition import proceedings_sources, workshop_coverage, workshop_site_papers  # noqa: E402
from conference_pipeline.acquisition.workshop_paper_sources import load_registered_sources  # noqa: E402
from conference_pipeline.conference_identity import load_conference_identity  # noqa: E402

OUTPUT_JSON = Path("storage/workshops/workshop_paper_coverage.json")
GROUPS = Path("storage/workshops/openreview_workshop_groups.json")
INVENTORY = Path("storage/workshops/inventory.json")
OPENREVIEW_RECEIPT = Path("storage/workshops/download_receipt.json")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--id-field", default="schedule_id")
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--website-field", default="website")
    parser.add_argument("--list-key", default="workshops")
    parser.add_argument("--groups", type=Path, default=None, help=f"Default: <workspace>/{GROUPS}")
    args = parser.parse_args(argv)
    ws = args.workspace

    groups_payload = _read(args.groups or ws / GROUPS)
    if groups_payload is None:
        print("missing OpenReview group metadata; run scripts/fetch_openreview_workshop_groups.py first", file=sys.stderr)
        return 2
    identity = load_conference_identity(ws)
    stopwords = [t for t in re.findall(r"[A-Za-z]+", identity.name) if len(t) > 1]
    schedule = workshop_coverage.load_schedule(
        args.schedule, id_field=args.id_field, title_field=args.title_field,
        website_field=args.website_field, list_key=args.list_key,
    )
    proceedings, proceedings_ids = proceedings_sources.load_proceedings_workshops(ws)
    registered, registry_blockers = load_registered_sources(ws)
    site_like = [s for s in registered if s["source_id"] not in proceedings_sources.SOURCES]
    coverage = workshop_coverage.build_coverage(
        schedule=schedule,
        openreview_groups=groups_payload.get("groups", groups_payload),
        openreview_inventory=_read(ws / INVENTORY),
        openreview_download_receipt=_read(ws / OPENREVIEW_RECEIPT),
        proceedings=proceedings,
        other_source_papers=workshop_site_papers.other_source_papers(site_like),
        extra_title_stopwords=stopwords,
    )
    coverage.update(
        {
            "conference": identity.name,
            "schedule_path": str(args.schedule),
            "proceedings_sources": proceedings_ids,
            "registered_sources": [s["source_id"] for s in registered],
            "registry_blockers": registry_blockers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    out = ws / OUTPUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(
        workshop_coverage.render_markdown(coverage, heading=f"{identity.name} workshop paper coverage"), encoding="utf-8"
    )
    print(
        f"workshop_paper_coverage: {coverage['workshops_with_papers']}/{coverage['schedule_workshop_count']} with papers; "
        f"{coverage['category_counts']}; proceedings_without_openreview={len(coverage['proceedings_without_openreview'])} -> {out}"
    )
    return 1 if registry_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
