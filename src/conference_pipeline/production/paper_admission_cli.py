"""``conference-pipeline papers admit <slug>``: the CLI over ``production.paper_admission``.

Exit codes: 0 admitted (or a dry run reported); 2 refused (no workspace, no lane
registry, a malformed lane or adapter input).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from conference_pipeline.indexing.paper_identity import PaperIdentityError
from conference_pipeline.indexing.paper_lanes import PaperLaneError
from conference_pipeline.production.paper_admission import admit_paper_sets

papers_app = typer.Typer(help="Admit paper sets added after a workspace's original corpus.")


def _project_root(project_root: Path | None) -> Path:
    return (project_root or Path(__file__).resolve().parents[3]).resolve()


def admit(slug: str, *, project_root: Path | None = None, dry_run: bool = False, echo=print) -> int:
    root = _project_root(project_root)
    if not slug or "/" in slug or slug.startswith("."):
        echo("refused: the workspace slug must be a plain conferences/<slug> name")
        return 2
    workspace = root / "conferences" / slug
    if not workspace.is_dir():
        echo(f"refused: no workspace conferences/{slug}")
        return 2
    try:
        result = admit_paper_sets(workspace, dry_run=dry_run)
    except (PaperLaneError, PaperIdentityError, OSError, json.JSONDecodeError) as exc:
        echo(f"refused: {exc}")
        return 2
    echo(
        json.dumps(
            {
                "papers_total": result.papers_total,
                "by_track": dict(result.by_track),
                "new_papers": len(result.new_paper_keys),
                "text_gained": len(result.text_gained_keys),
                "duplicate_copies_merged": result.duplicate_copies_merged,
                "artifacts_total": result.artifacts_total,
                "full_text_before": result.full_text_before,
                "full_text_after": result.full_text_after,
                "title_joins": len(result.title_joins),
                "title_collisions": len(result.title_collisions),
                "exclusions": len(result.exclusions),
                "written": result.written,
                "corpus_records_sha256": result.corpus_records_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


@papers_app.command("admit")
def admit_command(
    slug: str = typer.Argument(..., help="The workspace, conferences/<slug>."),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the tallies; write nothing."),
) -> None:
    raise typer.Exit(admit(slug, project_root=project_root, dry_run=dry_run))
