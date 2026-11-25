from __future__ import annotations
import pytest

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_workspace_gitignore.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("workspace_gitignore", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def write_manifest(workspace: Path, families: list[dict[str, object]]) -> None:
    manifest_path = workspace / "storage" / "storage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace": workspace.relative_to(workspace.parents[1]).as_posix(),
                "artifact_families": families,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def family(
    family_id: str,
    path: str,
    storage_class: str,
    *,
    exceptions: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": family_id,
        "path": path,
        "purpose": f"{family_id} test family",
        "producer": "pytest",
        "storage_class": storage_class,
        "share_policy": "none" if storage_class in {"git", "cache"} else "private",
        "rebuildable": storage_class != "git",
        "sync_profiles": {"public": storage_class == "public", "backup": storage_class == "backup"},
    }
    if exceptions:
        payload["gitignore_exceptions"] = exceptions
    return payload


def test_generator_renders_storage_class_policy_and_exceptions(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    workspace = tmp_path / "conferences" / "demo2027"
    write_manifest(
        workspace,
        [
            family("readme", "README.md", "git"),
            family("storage", "storage", "git"),
            family("presentations", "decks", "public", exceptions=["presentation_manifest.json"]),
            family("cache", "cache/vector-store", "cache"),
        ],
    )

    report = module.build_report(workspace=workspace)
    generated = report["generated_gitignore"]

    assert report["status"] == "pass"
    assert report["diagnostic"]["status"] == "pass"
    assert report["diagnostic"]["checked_inputs"]["family_count"] == 4
    assert report["diagnostic"]["next_actions"]
    assert "!/README.md" in generated
    assert "!/storage/**" in generated
    assert "/decks/*" in generated
    assert "!/decks/presentation_manifest.json" in generated
    assert "/cache/vector-store/*" in generated


def test_generator_expands_brace_globs_into_gitignore_patterns(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    payload_dir = workspace / "decks" / "topic-a"
    payload_dir.mkdir(parents=True)
    (payload_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
    (payload_dir / "presentation_video.mp4").write_bytes(b"video")
    (payload_dir / "presentation_slide_mapping.json").write_text("{}", encoding="utf-8")
    (payload_dir / "6_presenter_script.md").write_text("script", encoding="utf-8")
    (workspace / "decks" / "presentation_manifest.json").write_text("{}", encoding="utf-8")
    write_manifest(
        workspace,
        [
            family("presentation_lane", "decks", "git"),
            family("presentation_manifest", "decks/presentation_manifest.json", "git"),
            {
                "id": "presentation_payloads",
                "path_glob": "decks/**/*.{html,mp4,json}",
                "purpose": "Generated deck payloads.",
                "producer": "pytest",
                "storage_class": "public",
                "share_policy": "curated",
                "rebuildable": True,
                "sync_profiles": {"public": True, "backup": True},
            },
        ],
    )

    report = module.build_report(workspace=workspace)
    module.write_workspace_gitignore(workspace, report["generated_gitignore"])
    checked = module.build_report(workspace=workspace, check_git=True)
    generated = report["generated_gitignore"]

    assert "/decks/**/*.html" in generated
    assert "/decks/**/*.mp4" in generated
    assert "/decks/**/*.json" in generated
    assert "!/decks/presentation_manifest.json" in generated
    assert "/decks/**/*.{html,mp4,json}" not in generated
    assert checked["status"] == "pass"
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/decks/topic-a/presentation_video.mp4",
        ).stdout.strip()
        == "conferences/demo2027/decks/topic-a/presentation_video.mp4"
    )
    markdown_probe = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "conferences/demo2027/decks/topic-a/6_presenter_script.md",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert markdown_probe.returncode != 0
    manifest_probe = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "conferences/demo2027/decks/presentation_manifest.json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert manifest_probe.returncode != 0


def test_generator_keeps_os_metadata_ignored_after_workspace_unignores(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    (workspace / "decks").mkdir(parents=True)
    (workspace / "decks" / ".DS_Store").write_text("metadata", encoding="utf-8")
    (workspace / "scripts" / "__pycache__").mkdir(parents=True)
    (workspace / "scripts" / "__pycache__" / "tool.cpython-312.pyc").write_text(
        "bytecode", encoding="utf-8"
    )
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            family("presentation_lane", "decks", "git"),
            family("scripts", "scripts", "git"),
        ],
    )

    report = module.build_report(workspace=workspace)
    module.write_workspace_gitignore(workspace, report["generated_gitignore"])
    checked = module.build_report(workspace=workspace, check_git=True)

    assert ".DS_Store" in report["generated_gitignore"]
    assert "__pycache__/" in report["generated_gitignore"]
    assert checked["status"] == "pass"
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/decks/.DS_Store",
        ).stdout.strip()
        == "conferences/demo2027/decks/.DS_Store"
    )
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/scripts/__pycache__/tool.cpython-312.pyc",
        ).stdout.strip()
        == "conferences/demo2027/scripts/__pycache__/tool.cpython-312.pyc"
    )


def test_generator_ignores_ignored_family_root_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    workspace.mkdir(parents=True)
    external_leaf = tmp_path / "external-drive" / "audio"
    external_leaf.mkdir(parents=True)
    # A Google Drive backup leaf is materialized on disk as a symlink at the
    # ignored family's root path, not as a real directory of files.
    (workspace / "audio").symlink_to(external_leaf)
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            family("audio", "audio", "backup"),
        ],
    )

    report = module.build_report(workspace=workspace)
    module.write_workspace_gitignore(workspace, report["generated_gitignore"])

    # The family root itself (the symlink) is ignored, not just its contents.
    assert "/audio" in report["generated_gitignore"]
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/audio",
        ).stdout.strip()
        == "conferences/demo2027/audio"
    )


def test_check_blocks_tracked_public_payload(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    (workspace / "decks").mkdir(parents=True)
    (workspace / "decks" / "payload.svg").write_text("<svg />\n", encoding="utf-8")
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            family("presentations", "decks", "public", exceptions=["presentation_manifest.json"]),
        ],
    )
    module.write_workspace_gitignore(
        workspace,
        module.build_report(workspace=workspace)["generated_gitignore"],
    )
    run_git(tmp_path, "add", "-f", "conferences/demo2027/decks/payload.svg")

    report = module.build_report(workspace=workspace, check_git=True)

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"drive_or_cache_family_file_tracked_in_git"}
    assert "Move or untrack" in report["blockers"][0]["next_action"]
    assert report["blockers"][0]["expected"]
    assert report["blockers"][0]["observed"]
    assert report["diagnostic"]["next_actions"]


def test_check_passes_for_literal_git_file_tied_in_depth_with_ignored_glob(
    tmp_path: Path, monkeypatch
) -> None:
    """A literal (non-wildcard) committed exception file must win over an
    equal-*depth* ignored glob family, even though both patterns split into
    the same number of `/`-separated segments. Regression for the REND-02
    ai4physics-talk `presentation.html` exception, where `family_specificity`
    previously broke ties by segment count alone and silently misclassified
    an already-committable literal path as still belonging to the ignored
    glob family once it was actually tracked in git.
    """
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    deck_dir = workspace / "decks" / "topic-a"
    deck_dir.mkdir(parents=True)
    (deck_dir / "keep.html").write_text("<html></html>", encoding="utf-8")
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            {
                "id": "deck_renders",
                "path_glob": "decks/**/*.{html,pdf}",
                "purpose": "Generated deck renders.",
                "producer": "pytest",
                "storage_class": "public",
                "share_policy": "curated",
                "rebuildable": True,
                "sync_profiles": {"public": True, "backup": True},
            },
            family("committed_render", "decks/topic-a/keep.html", "git"),
        ],
    )
    module.write_workspace_gitignore(
        workspace,
        module.build_report(workspace=workspace)["generated_gitignore"],
    )
    keep_html_ignore_probe = subprocess.run(
        ["git", "check-ignore", "conferences/demo2027/decks/topic-a/keep.html"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert keep_html_ignore_probe.returncode != 0
    run_git(tmp_path, "add", "conferences/demo2027/decks/topic-a/keep.html")

    report = module.build_report(workspace=workspace, check_git=True)

    assert report["status"] == "pass"
    assert report["blockers"] == []


def test_check_blocks_oversized_git_owned_payload(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "MAX_GIT_BLOB_BYTES", 1024)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    inventory = workspace / "storage" / "download-corpus" / "download_corpus.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("x" * 2048, encoding="utf-8")
    write_manifest(workspace, [family("storage", "storage", "git")])
    module.write_workspace_gitignore(
        workspace,
        module.build_report(workspace=workspace)["generated_gitignore"],
    )
    run_git(
        tmp_path,
        "add",
        "-f",
        "conferences/demo2027/storage/download-corpus/download_corpus.json",
    )

    report = module.build_report(workspace=workspace, check_git=True)

    assert report["status"] == "blocked"
    blockers = {blocker["code"]: blocker for blocker in report["blockers"]}
    assert "git_owned_file_exceeds_size_limit" in blockers
    assert blockers["git_owned_file_exceeds_size_limit"]["byte_count"] > 1024
    assert "untrack" in blockers["git_owned_file_exceeds_size_limit"]["next_action"]


def test_write_adds_gitignore_entry_for_oversized_new_file(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "MAX_GIT_BLOB_BYTES", 1024)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    oversized = workspace / "storage" / "generated-large.bin"
    oversized.parent.mkdir(parents=True)
    oversized.write_text("x" * 2048, encoding="utf-8")
    write_manifest(workspace, [family("storage", "storage", "git")])

    report = module.build_report(workspace=workspace)
    module.write_workspace_gitignore(workspace, report["generated_gitignore"])
    checked = module.build_report(workspace=workspace, check_git=True)

    assert "/storage/generated-large.bin" in report["generated_gitignore"]
    assert report["oversized_local_gitignore_paths"] == ["storage/generated-large.bin"]
    assert checked["status"] == "pass"
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/storage/generated-large.bin",
        ).stdout.strip()
        == "conferences/demo2027/storage/generated-large.bin"
    )


def test_oversized_file_owned_by_a_deeper_ignored_glob_family_gets_no_local_entry(
    tmp_path: Path, monkeypatch
) -> None:
    """A whole-tree literal git family must not hide a deeper ignored glob owner.

    CVPR 2026 regression: a 102 MB Findings PDF under
    ``storage/papers/proceedings/*/{*.pdf,text}`` (backup) was listed as an
    explicit oversized local entry because the ``storage`` git family outranked
    the glob family, so the managed block differed between checkouts.
    """

    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "MAX_GIT_BLOB_BYTES", 1024)
    run_git(tmp_path, "init")
    workspace = tmp_path / "conferences" / "demo2027"
    pdf = workspace / "storage" / "papers" / "proceedings" / "track_a" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"x" * 2048)
    glob_family = {
        "id": "proceedings_paper_payloads",
        "path_glob": "storage/papers/proceedings/*/{*.pdf,text}",
        "purpose": "proceedings payloads",
        "producer": "pytest",
        "storage_class": "backup",
    }
    write_manifest(workspace, [family("storage", "storage", "git"), glob_family])

    report = module.build_report(workspace=workspace)
    module.write_workspace_gitignore(workspace, report["generated_gitignore"])
    checked = module.build_report(workspace=workspace, check_git=True)

    assert report["oversized_local_gitignore_paths"] == []
    assert "paper.pdf" not in report["generated_gitignore"]
    assert checked["status"] == "pass", checked["blockers"]
    assert (
        run_git(
            tmp_path,
            "check-ignore",
            "--no-index",
            "conferences/demo2027/storage/papers/proceedings/track_a/paper.pdf",
        ).stdout.strip()
        == "conferences/demo2027/storage/papers/proceedings/track_a/paper.pdf"
    )


@pytest.mark.repo_data("conferences/iclr2026/storage/storage_manifest.json")
def test_iclr2026_download_corpus_inventory_is_not_git_owned() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "iclr2026" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    assert families["download_corpus_inventory"]["path"] == (
        "storage/download-corpus/download_corpus.json"
    )
    assert families["download_corpus_inventory"]["storage_class"] == "backup"
    assert families["download_corpus_inventory"]["sync_profiles"] == {
        "public": False,
        "backup": True,
    }
    tracked = run_git(
        REPO_ROOT,
        "ls-files",
        "--",
        "conferences/iclr2026/storage/download-corpus/download_corpus.json",
    ).stdout.strip()
    assert tracked == ""


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr2026_source_discovery_inventory_is_trackable_and_raw_snapshots_are_ignored() -> None:
    module = load_generator_module()
    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    manifest = json.loads(
        (workspace / "storage" / "storage_manifest.json").read_text(encoding="utf-8")
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    inventory = families["source_discovery_inventory"]
    raw_snapshots = families["source_discovery_raw_snapshots"]
    assert inventory["path"] == "storage/source-discovery/source_discovery_inventory.json"
    assert inventory["storage_class"] == "git"
    assert inventory["share_policy"] == "private"
    assert inventory["sync_profiles"] == {"public": False, "backup": True}
    assert raw_snapshots["path"] == "cache/source-discovery"
    assert raw_snapshots["storage_class"] == "cache"
    assert raw_snapshots["sync_profiles"] == {"public": False, "backup": False}

    report = module.build_report(workspace=workspace, check_git=True)
    assert report["status"] == "pass"
    assert not module.git_check_ignore(
        workspace / "storage/source-discovery/source_discovery_inventory.json"
    )
    assert module.git_check_ignore(workspace / "cache/source-discovery/family.json")
    assert not module.git_check_ignore(workspace / "cache/source-discovery-other/family.json")


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr2026_download_corpus_payloads_are_private_backup_only() -> None:
    module = load_generator_module()
    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    manifest = json.loads(
        (workspace / "storage" / "storage_manifest.json").read_text(encoding="utf-8")
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    assert families["download_corpus_ledger"]["storage_class"] == "backup"
    assert families["download_corpus_artifacts"]["storage_class"] == "backup"
    assert families["download_corpus_youtube_captions"]["storage_class"] == "backup"
    assert families["download_corpus_focused_recovery"]["storage_class"] == "backup"
    assert families["download_corpus_worker_logs"]["storage_class"] == "cache"

    report = module.build_report(workspace=workspace, check_git=True)
    assert report["status"] == "pass", report["blockers"]
    assert module.git_check_ignore(workspace / "storage/download-corpus/download_corpus.json")
    assert module.git_check_ignore(workspace / "storage/download-corpus/artifacts/paper.pdf")
    assert module.git_check_ignore(workspace / "storage/download-corpus/youtube/captions/talk.vtt")
    assert module.git_check_ignore(workspace / "storage/download-corpus/focused-recovery/result.json")
    assert module.git_check_ignore(workspace / "storage/download-corpus/logs/worker.log")
    assert not module.git_check_ignore(
        workspace / "storage/download-corpus/download_corpus_report.md"
    )
    assert module.git_check_ignore(workspace / "cache/source-discovery/family.json")


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr2026_registration_media_overlay_and_session_seams_are_not_git_payloads() -> None:
    module = load_generator_module()
    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    manifest = json.loads(
        (workspace / "storage" / "storage_manifest.json").read_text(encoding="utf-8")
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    assert families["registration_media_overlay_ledger"]["storage_class"] == "backup"
    assert families["registration_media_captions"]["storage_class"] == "backup"
    assert families["registration_media_artifacts"]["storage_class"] == "backup"
    for family_id in (
        "registration_media_raw_page_cache",
        "registration_media_worker_logs",
        "registration_media_diagnostics",
        "registration_media_session_cache",
    ):
        assert families[family_id]["storage_class"] == "cache"

    report = module.build_report(workspace=workspace, check_git=True)
    assert report["status"] == "pass", report["blockers"]
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/registration_media.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/captions/talk.vtt"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/artifacts/talk.mp4"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/raw-pages/event.html"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/logs/worker.log"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/diagnostics/retry.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/session/cookies.txt"
    )
    assert not module.git_check_ignore(
        workspace / "storage/download-corpus/registration-media/registration_media_report.md"
    )


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr2026_event_media_payloads_and_session_seams_are_not_git_payloads() -> None:
    module = load_generator_module()
    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    manifest = json.loads(
        (workspace / "storage" / "storage_manifest.json").read_text(encoding="utf-8")
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    assert families["event_media_inventory"]["storage_class"] == "backup"
    assert families["event_media_captions"]["storage_class"] == "backup"
    for family_id in (
        "event_media_checkpoint_cache",
        "event_media_raw_page_cache",
        "event_media_worker_logs",
        "event_media_session_cache",
    ):
        assert families[family_id]["storage_class"] == "cache"

    report = module.build_report(workspace=workspace, check_git=True)
    assert report["status"] == "pass", report["blockers"]
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/captions/talk.vtt"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/event_media.checkpoint.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/raw-pages/event.html"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/logs/worker.log"
    )
    assert module.git_check_ignore(
        workspace / "storage/download-corpus/event-media/session/cookies.txt"
    )


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr2026_transcription_ocr_payloads_are_private_backup_only() -> None:
    module = load_generator_module()
    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    manifest = json.loads(
        (workspace / "storage" / "storage_manifest.json").read_text(encoding="utf-8")
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    assert families["transcription_ocr_inventory"]["storage_class"] == "backup"
    assert families["transcription_ocr_report"]["storage_class"] == "git"
    for family_id in (
        "transcription_ocr_caption_sidecars",
        "transcription_ocr_paper_text",
        "transcription_ocr_speech_text",
        "transcription_ocr_workshop_talks",
        "transcription_ocr_scope_waivers",
        "transcription_ocr_worker_logs",
    ):
        assert families[family_id]["storage_class"] in {"backup", "cache"}

    report = module.build_report(workspace=workspace, check_git=True)
    assert report["status"] == "pass", report["blockers"]
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/inventory.json"
    )
    assert not module.git_check_ignore(
        workspace / "storage/transcription-ocr/transcription_ocr_report.md"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/caption-transcripts/talk.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/paper-text/paper.txt"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/speech-text/talk.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/workshop-talks/talks.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/scope-waivers/waiver.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/inventory.checkpoint.json"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/raw-inputs/event.html"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/logs/worker.log"
    )
    assert module.git_check_ignore(
        workspace / "storage/transcription-ocr/cvpr2026/session/cookies.txt"
    )
    for retry_control in (
        "stage-05-transcription-ocr.pre-repair-blocked.json",
        "stage-05-transcription-ocr.retry-authorization.json",
        "stage-05-transcription-ocr.retry-attempt.json",
        "stage-05-transcription-ocr.receipt-finalization.json",
    ):
        assert module.git_check_ignore(
            workspace
            / "storage/workflow-runs/cvpr2026-live-production-r5-event-media"
            / retry_control
        )


@pytest.mark.repo_data("conferences/iclr2026/storage/storage_manifest.json")
def test_iclr2026_generated_artifact_families_are_not_git_owned() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "iclr2026" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = {family["id"]: family for family in manifest["artifact_families"]}

    expected = {
        "workspace_provider_call_receipts": ("storage/provider-calls", None, "backup"),
        "workflow_run_provider_call_receipts": (
            None,
            "storage/workflow-runs/*/provider-calls/**",
            "backup",
        ),
        "workspace_logs": ("storage/logs", None, "cache"),
        "paper_matching_outputs": ("storage/paper-matching", None, "backup"),
        "presentation_redesign_snapshots": (
            "storage/presentation-redesign",
            None,
            "backup",
        ),
        "quality_audit_payloads": ("storage/quality-audits", None, "backup"),
        "presentation_orchestrator_logs": (
            None,
            "decks/*/orchestrator_logs/**",
            "cache",
        ),
        "presentation_generated_figure_code": (
            None,
            "decks/*/assets/generated/**/code/**",
            "cache",
        ),
    }
    for family_id, (path, path_glob, storage_class) in expected.items():
        family_payload = families[family_id]
        assert family_payload.get("path") == path
        assert family_payload.get("path_glob") == path_glob
        assert family_payload["storage_class"] == storage_class


def test_starter_workspace_gitignore_policy_checks_pass() -> None:
    # The running interpreter, not a checkout-local .venv: the check needs no venv of its
    # own (review phase-7-r1 F5).
    for workspace in ("conferences/TEMPLATE",):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_workspace_gitignore.py",
                "--workspace",
                workspace,
                "--check",
                "--json",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == "pass"
        assert payload["diagnostic"]["summary"].startswith(
            "Workspace .gitignore matches storage manifest"
        )


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_existing_neurips_families_have_public_export_field() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = manifest["artifact_families"]
    for f in families:
        assert "public_export" in f.get("sync_profiles", {}), (
            f"Family {f['id']!r} is missing sync_profiles.public_export"
        )
    family_map = {f["id"]: f for f in families}
    for pilot_id in ("presentations", "podcasts"):
        assert family_map[pilot_id]["sync_profiles"]["public_export"] is True, (
            f"Pilot family {pilot_id!r} should have public_export: true"
        )
    # Phase 6b: v1.1.0's reader families export from release_staging (its bundle copy is
    # the public one); release_sources holds operator inputs and is never exported.
    assert family_map["release_staging"]["sync_profiles"]["public_export"] is True
    non_exported = [
        f["id"] for f in families
        if f["id"] not in ("presentations", "podcasts", "public_index_staging", "release_staging")
    ]
    for fid in non_exported:
        assert family_map[fid]["sync_profiles"]["public_export"] is False, (
            f"Non-pilot family {fid!r} should have public_export: false"
        )


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_export_bundle_family_is_backup_class() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = {f["id"]: f for f in manifest["artifact_families"]}
    assert "export_bundle" in families, "export_bundle family missing from manifest"
    eb = families["export_bundle"]
    assert eb["storage_class"] == "backup"
    assert eb["sync_profiles"]["backup"] is True
    assert eb["sync_profiles"]["public"] is False
    assert eb["sync_profiles"]["public_export"] is False
    assert eb["path"] == "exports/public-release"


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_public_index_staging_family_is_cache_class() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = {f["id"]: f for f in manifest["artifact_families"]}
    assert "public_index_staging" in families, "public_index_staging family missing from manifest"
    pis = families["public_index_staging"]
    assert pis["storage_class"] == "cache"
    assert pis["sync_profiles"]["backup"] is False
    assert pis["sync_profiles"]["public"] is False
    assert pis["sync_profiles"]["public_export"] is True
    assert pis["path"] == "cache/public-index"
    assert "public_export_notes" in pis


def test_gitignore_covers_exports_public_release(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    workspace = tmp_path / "conferences" / "neurips2025"
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            family("export_bundle", "exports/public-release", "backup"),
        ],
    )
    report = module.build_report(workspace=workspace)
    generated = report["generated_gitignore"]
    assert report["status"] == "pass"
    assert "/exports/public-release/*" in generated


def test_gitignore_covers_cache_public_index(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    workspace = tmp_path / "conferences" / "neurips2025"
    write_manifest(
        workspace,
        [
            family("storage", "storage", "git"),
            family("public_index_staging", "cache/public-index", "cache"),
        ],
    )
    report = module.build_report(workspace=workspace)
    generated = report["generated_gitignore"]
    assert report["status"] == "pass"
    assert "/cache/public-index/*" in generated


def test_existing_family_gitignore_policy_unchanged(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    workspace = tmp_path / "conferences" / "neurips2025"
    write_manifest(
        workspace,
        [
            family("readme_md", "README.md", "git"),
            family("storage", "storage", "git"),
            family("presentations", "decks", "public", exceptions=["presentation_manifest.json"]),
            family("analysis", "analysis", "backup"),
            family("vector_store", "cache/vector-store", "cache"),
            family("export_bundle", "exports/public-release", "backup"),
            family("public_index_staging", "cache/public-index", "cache"),
        ],
    )
    report = module.build_report(workspace=workspace)
    generated = report["generated_gitignore"]
    assert report["status"] == "pass"
    assert "!/README.md" in generated
    assert "!/storage/**" in generated
    assert "/decks/*" in generated
    assert "!/decks/presentation_manifest.json" in generated
    assert "/analysis/*" in generated
    assert "/cache/vector-store/*" in generated
    assert "/exports/public-release/*" in generated
    assert "/cache/public-index/*" in generated


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_neurips_tracked_samples_remain_small_textual_controls() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tracked_samples = [
        sample
        for family_payload in manifest["artifact_families"]
        for sample in family_payload.get("tracked_samples", [])
    ]

    assert tracked_samples
    for sample in tracked_samples:
        path = REPO_ROOT / "conferences" / "neurips2025" / sample
        assert path.exists(), sample
        assert path.suffix in {".md", ".json", ".txt"}, sample


@pytest.mark.repo_data("conferences/neurips2025/.gitignore")
def test_every_release_workspace_ignores_its_written_bundles() -> None:
    """6b: a written release bundle (exports/public-release/<bundle>) is generated, thousands of
    files, and published only through its upload. ICML, CVPR and ICLR 2025 left it untracked
    and not ignored, so one `git add -A` would commit it. Every workspace that builds releases
    declares a backup-class family covering it, and its .gitignore ignores it."""
    checked = []
    for manifest_path in sorted((REPO_ROOT / "conferences").glob("*/storage/storage_manifest.json")):
        families = json.loads(manifest_path.read_text(encoding="utf-8"))["artifact_families"]
        if not any(f["id"] == "release_sources" for f in families):
            continue
        workspace = manifest_path.parents[1]
        covering = [f for f in families if f.get("path") in ("exports", "exports/public-release")]
        assert covering and all(f["storage_class"] == "backup" for f in covering), workspace.name
        probe = f"conferences/{workspace.name}/exports/public-release/x-v1.0.0/RELEASE.json"
        ignored = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPO_ROOT, check=False)
        assert ignored.returncode == 0, f"{workspace.name}: {probe} is not ignored"
        checked.append(workspace.name)
    assert {"neurips2025", "iclr2025", "iclr2026", "icml2026", "cvpr2026"} <= set(checked)
