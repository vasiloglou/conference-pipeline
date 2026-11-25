from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conference_pipeline.slideslive_recovery import (
    discover_cached_slideslive_assets,
    extract_iclr_slideslive_page_asset,
    sanitize_url,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "recover_iclr2026_slideslive_media.py"
SECRET = "SECRET_TOKEN_VALUE_SHOULD_NOT_SURVIVE"


def test_extract_iclr_slideslive_page_asset_sanitizes_embed_tokens() -> None:
    html = f"""
    <html><body>
      <iframe src="https://slideslive.com/embed/presentation/39056835?embed_init_token={SECRET}&user_uuid=abc"></iframe>
    </body></html>
    """

    asset = extract_iclr_slideslive_page_asset(
        page_url=f"https://iclr.cc/virtual/2026/poster/10008853?token={SECRET}",
        html_text=html,
        page_artifact_path="cache/acquisition-downloads/item-page-probes/poster.html",
        source_family_id="posters",
    )

    assert asset.item_id == "poster-10008853"
    assert asset.kind == "poster"
    assert asset.slideslive_id == "39056835"
    assert asset.slideslive_url == "https://slideslive.com/39056835"
    assert SECRET not in json.dumps(asset.__dict__)
    assert sanitize_url(f"https://slideslive.com/embed/39056835?embed_init_token={SECRET}") == (
        "https://slideslive.com/embed/39056835"
    )


def test_discover_cached_slideslive_assets_reads_only_item_page_probes(tmp_path: Path) -> None:
    workspace = tmp_path / "iclr2026"
    page = workspace / "cache" / "acquisition-downloads" / "item-page-probes" / "poster.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        """
        <div id="presentation-embed-39056835"></div>
        <script>new SlidesLiveEmbed("presentation-embed-39056835", {presentationId: "39056835"});</script>
        """,
        encoding="utf-8",
    )
    ignored = workspace / "cache" / "acquisition-downloads" / "source-pages" / "ignored.html"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("presentationId: '99999999'", encoding="utf-8")

    inventory = {
        "local_artifacts": [
            {
                "artifact_kind": "item_page_auth_probe",
                "status": "succeeded",
                "source_family_id": "posters",
                "source_url": "https://iclr.cc/virtual/2026/poster/10008853",
                "local_path": page.relative_to(workspace).as_posix(),
            },
            {
                "artifact_kind": "source_page",
                "status": "succeeded",
                "source_family_id": "conference",
                "source_url": "https://iclr.cc/Conferences/2026",
                "local_path": ignored.relative_to(workspace).as_posix(),
            },
        ]
    }

    assets = discover_cached_slideslive_assets(workspace, inventory)

    assert [asset.slideslive_id for asset in assets] == ["39056835"]
    assert assets[0].page_artifact_path.endswith("item-page-probes/poster.html")


def test_recovery_script_merges_only_slideslive_audio_with_fake_ytdlp(tmp_path: Path) -> None:
    workspace = tmp_path / "iclr2026"
    page = workspace / "cache" / "acquisition-downloads" / "item-page-probes" / "poster.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        """
        <div id="presentation-embed-39056835"></div>
        <script>new SlidesLiveEmbed("presentation-embed-39056835", {presentationId: "39056835"});</script>
        """,
        encoding="utf-8",
    )
    inventory_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.iclr2026_download_corpus",
                "workspace": str(workspace),
                "run_id": "test-run",
                "stage_id": "download-corpus",
                "status": "succeeded",
                "local_artifact_count": 1,
                "direct_download_artifact_count": 0,
                "media_artifact_count": 0,
                "paper_artifact_count": 0,
                "local_artifacts": [
                    {
                        "artifact_kind": "item_page_auth_probe",
                        "status": "succeeded",
                        "source_family_id": "posters",
                        "source_url": f"https://iclr.cc/virtual/2026/poster/10008853?token={SECRET}",
                        "local_path": page.relative_to(workspace).as_posix(),
                    }
                ],
                "direct_download_artifacts": [],
                "blockers": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ytdlp = fake_bin / "yt-dlp"
    fake_ytdlp.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
template = args[args.index("--output") + 1]
path = pathlib.Path(template.replace("%(ext)s", "mp3"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(b"fake mp3 bytes")
""",
        encoding="utf-8",
    )
    fake_ytdlp.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--run-id",
            "test-run",
            "--resume",
            "--worker-count",
            "1",
            "--no-env-file",
            "--summary",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"summary": summary, "inventory": inventory})

    assert summary["status"] == "succeeded"
    assert summary["slideslive_embed_count"] == 1
    assert summary["downloaded_audio_count"] == 1
    assert summary["boundaries"]["paper_pdfs_redownloaded"] is False
    assert inventory["media_artifact_count"] == 1
    assert len(inventory["direct_download_artifacts"]) == 1
    assert inventory["direct_download_artifacts"][0]["slideslive_id"] == "39056835"
    assert inventory["direct_download_artifacts"][0]["local_path"].endswith("raw.mp3")
    assert SECRET not in serialized


def test_resolve_workspace_member_allows_workspace_symlink(tmp_path: Path) -> None:
    from conference_pipeline.slideslive_recovery import resolve_workspace_member

    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-store"
    external_root.mkdir()
    real_file = external_root / "audio.mp3"
    real_file.write_bytes(b"fake-audio")
    linked_dir = workspace / "storage" / "download-corpus"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved = resolve_workspace_member(workspace, "storage/download-corpus/audio.mp3")
    assert resolved is not None
    assert resolved.read_bytes() == b"fake-audio"


def test_resolve_workspace_member_still_blocks_dotdot_escape(tmp_path: Path) -> None:
    from conference_pipeline.slideslive_recovery import resolve_workspace_member

    workspace = tmp_path / "conference"
    workspace.mkdir()
    (tmp_path / "outside.mp3").write_bytes(b"secret")

    resolved = resolve_workspace_member(workspace, "../outside.mp3")
    assert resolved is None


def test_item_identity_accepts_any_typed_slug_and_resolves_bare_ids_from_the_title() -> None:
    """The ICML 2026 recovery lost every tutorial, expo, session and affinity
    recording's label because the path pattern only knew eight ICLR-era
    slugs. Any typed slug is a kind now, and a bare-ID page resolves its kind
    from the cached page ``<title>``."""

    from conference_pipeline.slideslive_recovery import item_identity_from_url, media_artifact_record

    assert item_identity_from_url("https://icml.cc/virtual/2026/tutorial/45001") == ("tutorial-45001", "tutorial")
    assert item_identity_from_url("https://icml.cc/virtual/2026/session/61377") == ("session-61377", "session")
    assert item_identity_from_url("https://icml.cc/virtual/2026/affinity-event/300") == (
        "affinity-event-300",
        "affinity-event",
    )
    assert item_identity_from_url("https://icml.cc/virtual/2026/75372") == ("virtual-2026-75372", None)
    assert item_identity_from_url("https://icml.cc/virtual/2026/75372?x=1") == ("virtual-2026-75372", None)
    # Navigation surfaces are never items.
    assert item_identity_from_url("https://icml.cc/virtual/2026/calendar/day1")[1] is None

    tutorial = extract_iclr_slideslive_page_asset(
        page_url="https://icml.cc/virtual/2026/75372",
        html_text="<title>ICML Tutorial: Math Copilots</title><iframe src='https://slideslive.com/embed/presentation/39056900'>",
        page_artifact_path="cache/acquisition-downloads/item-page-probes/75372.html",
        source_family_id="icml-calendar",
    )
    assert tutorial.item_id == "virtual-2026-75372"
    assert tutorial.kind == "tutorial"
    assert tutorial.page_title == "ICML Tutorial: Math Copilots"
    assert tutorial.slideslive_id == "39056900"

    record = media_artifact_record(
        Path("/tmp/ws"),
        path=Path(__file__),
        asset=tutorial,
        retrieval_status="downloaded",
    )
    assert record["item_kind"] == "tutorial"
    assert record["source_title"] == "ICML Tutorial: Math Copilots"
