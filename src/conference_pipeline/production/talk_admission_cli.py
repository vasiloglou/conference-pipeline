"""``conference-pipeline talks bind <slug>``: the CLI over ``indexing.talk_source_lanes``.

Exit codes: 0 bound (or a dry run reported); 2 refused (a bad slug, no workspace, no
talk sources, a malformed or missing source file).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from conference_pipeline.indexing.talk_source_lanes import PaperLaneError, bind_talk_sources

talks_app = typer.Typer(help="Bind a workspace's existing talk transcripts into evidence lanes.")


def _project_root(project_root: Path | None) -> Path:
    return (project_root or Path(__file__).resolve().parents[3]).resolve()


def bind(slug: str, *, project_root: Path | None = None, dry_run: bool = False, echo=print) -> int:
    root = _project_root(project_root)
    if not slug or "/" in slug or slug.startswith("."):
        echo("refused: the workspace slug must be a plain conferences/<slug> name")
        return 2
    workspace = root / "conferences" / slug
    if not workspace.is_dir():
        echo(f"refused: no workspace conferences/{slug}")
        return 2
    try:
        result = bind_talk_sources(workspace, dry_run=dry_run)
    except (PaperLaneError, OSError, json.JSONDecodeError) as exc:
        echo(f"refused: {exc}")
        return 2
    echo(
        json.dumps(
            {"lanes": {k: dict(v) for k, v in result.lanes.items()}, "written": result.written,
             "changed_lanes": list(result.changed_lanes)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


@talks_app.command("bind")
def bind_command(
    slug: str = typer.Argument(..., help="The workspace, conferences/<slug>."),
    project_root: Path | None = typer.Option(None, "--project-root", help="The checkout that holds conferences/<slug>/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the lanes; write nothing."),
) -> None:
    raise typer.Exit(bind(slug, project_root=project_root, dry_run=dry_run))
