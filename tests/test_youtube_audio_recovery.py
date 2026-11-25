"""Offline contracts for the in-process r6 YouTube adapter."""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeYoutubeDL:
    def __init__(self, options, module):
        self.options = options
        self.module = module
        module.options.append(dict(options))

    def extract_info(self, url, *, download):
        self.module.calls.append((url, download))
        outcome = self.module.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def urlopen(self, url):
        self.module.stream_urls.append(url)
        if self.module.urlopen_hook is not None:
            return self.module.urlopen_hook(url)
        return io.BytesIO(self.module.stream_bytes)


def _fake(*outcomes):
    module = SimpleNamespace(__r6_test_fake__=True, __r6_source_bytes__=b"fixture-source", __r6_record_bytes__=b"yt_dlp/__init__.py,fixture,1\n", __r6_lock_identity__="yt-dlp==fixture", outcomes=list(outcomes), calls=[], options=[], stream_urls=[], stream_bytes=b"fixture-audio", urlopen_hook=None)
    module.YoutubeDL = lambda options: _FakeYoutubeDL(options, module)
    return module


def _preflight(module):
    from conference_pipeline.acquisition.youtube_audio_recovery import preflight_fixed_r6_yt_dlp
    return preflight_fixed_r6_yt_dlp(_test_module=module)


def _contract(tmp_path):
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    return contract


def _recorded_package(tmp_path: Path) -> tuple[Path, Path]:
    """A wheel-shaped, disposable package tree (never site-packages)."""
    package = tmp_path / "yt_dlp"
    package.mkdir()
    body = b"__version__ = 'fixture'\n"
    (package / "__init__.py").write_bytes(body)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
    record = tmp_path / "yt_dlp-0.dist-info" / "RECORD"
    record.parent.mkdir()
    record.write_text(f"yt_dlp/__init__.py,sha256={encoded},{len(body)}\n", encoding="utf-8")
    return package, record


def test_tree_record_verifier_rejects_unrecorded_nodes_and_record_mutations(tmp_path):
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        R6DownloaderPreflightError,
        _verified_tree_receipt,
    )
    package, record = _recorded_package(tmp_path)
    receipt = _verified_tree_receipt(package, record)
    assert set(receipt) == {"tree_manifest_sha256", "record_sha256"}
    (package / "injected.py").write_text("x", encoding="utf-8")
    with pytest.raises(R6DownloaderPreflightError, match="tree and RECORD"):
        _verified_tree_receipt(package, record)
    (package / "injected.py").unlink()
    record.write_text("yt_dlp/__init__.py,,1\n", encoding="utf-8")
    with pytest.raises(R6DownloaderPreflightError, match="unhashed"):
        _verified_tree_receipt(package, record)


def test_tree_record_verifier_rejects_empty_directories_and_symlinks(tmp_path):
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        R6DownloaderPreflightError,
        _verified_tree_receipt,
    )
    package, record = _recorded_package(tmp_path)
    (package / "unexpected").mkdir()
    with pytest.raises(R6DownloaderPreflightError, match="directory"):
        _verified_tree_receipt(package, record)
    (package / "unexpected").rmdir()
    (package / "link").symlink_to(package / "__init__.py")
    with pytest.raises(R6DownloaderPreflightError, match="symlink"):
        _verified_tree_receipt(package, record)


def test_descriptor_tree_walk_keeps_its_owned_directory_descriptors(tmp_path, monkeypatch):
    """A platform may consume ``listdir(fd)``; the walker must use a duplicate."""

    import conference_pipeline.acquisition.youtube_audio_recovery as recovery

    package, _record = _recorded_package(tmp_path)
    original_listdir = os.listdir

    def consuming_listdir(path):
        result = original_listdir(path)
        if isinstance(path, int):
            try:
                os.close(path)
            except OSError:
                pass
        return result

    monkeypatch.setattr(recovery.os, "listdir", consuming_listdir)
    entries = recovery._safe_package_entries(package)

    assert entries == (("__init__.py", hashlib.sha256(b"__version__ = 'fixture'\n").hexdigest(), 24),)


def test_descriptor_tree_walk_rejects_a_post_stat_directory_swap(tmp_path, monkeypatch):
    """The provenance walk holds parents; it never descends through a path race."""
    import conference_pipeline.acquisition.youtube_audio_recovery as recovery
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        R6DownloaderPreflightError,
        _verified_tree_receipt,
    )

    package, record = _recorded_package(tmp_path)
    child = package / "nested"
    child.mkdir()
    body = b"nested source\n"
    (child / "mod.py").write_bytes(body)
    root_body = (package / "__init__.py").read_bytes()
    def record_line(path, payload):
        return f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')},{len(payload)}\n"
    record.write_text(
        record_line("yt_dlp/__init__.py", root_body) + record_line("yt_dlp/nested/mod.py", body),
        encoding="utf-8",
    )

    def swap(stage, relative):
        if stage == "before_directory_open" and relative == "nested":
            child.rename(package / "original")
            child.symlink_to(tmp_path / "escape", target_is_directory=True)

    monkeypatch.setattr(recovery, "_TREE_WALK_TEST_HOOK", swap)
    with pytest.raises(R6DownloaderPreflightError, match="directory|symlink"):
        _verified_tree_receipt(package, record)


def test_inprocess_preflight_and_recovery_never_launch_a_child(tmp_path, monkeypatch):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake({"id": "abcDEF12345", "duration": 42.5, "availability": "public", "formats": [{"url": "https://media.example.invalid/a.mp3", "protocol": "https", "vcodec": "none", "acodec": "mp3", "format_id": "a"}]})
    contract = _contract(tmp_path)
    def poison(*args, **kwargs):
        raise AssertionError("a child executable must never be reached")
    monkeypatch.setattr(subprocess, "run", poison)
    monkeypatch.setattr(subprocess, "Popen", poison)
    record = recover_public_youtube_audio(item_id="item-1", source_url="https://youtu.be/abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module, max_attempts=2)
    assert record["terminal_state"] == "acquired"
    assert record["sha256"] == hashlib.sha256(b"fixture-audio").hexdigest()
    assert module.calls == [("https://www.youtube.com/watch?v=abcDEF12345", False)]
    assert module.options[0]["skip_download"] is True and "outtmpl" not in module.options[0]
    assert module.stream_urls == ["https://media.example.invalid/a.mp3"]


def test_recovery_uses_the_owner_approved_preflight_without_rewalking_the_tree(tmp_path, monkeypatch):
    """Each worker binds the loaded module, not a second fragile tree traversal."""

    import conference_pipeline.acquisition.youtube_audio_recovery as recovery
    module = _fake(
        {
            "id": "abcDEF12345",
            "duration": 42.5,
            "availability": "public",
            "formats": [{"url": "https://media.example.invalid/a.mp3", "protocol": "https", "vcodec": "none", "acodec": "mp3", "format_id": "a"}],
        }
    )
    preflight = _preflight(module)
    monkeypatch.setattr(
        recovery,
        "_distribution_identity",
        lambda _module: (_ for _ in ()).throw(AssertionError("worker must not rewalk package tree")),
    )

    record = recovery.recover_public_youtube_audio(
        item_id="item",
        source_url="https://youtu.be/abcDEF12345",
        r6_root=(contract := _contract(tmp_path)).r6_root,
        artifact_contract=contract,
        preflight=preflight,
        _test_module=module,
    )

    assert record["terminal_state"] == "acquired"


@pytest.mark.parametrize("url", ["https://example.invalid/x", "https://www.youtube.com/results?search_query=x", "https://www.youtube.com/playlist?list=x", "https://www.youtube.com/watch?v=abcDEF12345&list=x"])
def test_noncanonical_or_playlist_sources_never_create_client(tmp_path, url):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake()
    contract = _contract(tmp_path)
    record = recover_public_youtube_audio(item_id="item", source_url=url, r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module)
    assert record["terminal_state"] == "malformed_or_unsupported_source"
    assert module.calls == []


@pytest.mark.parametrize(("payload", "state"), [({"id": "abcDEF12345", "availability": "private"}, "private_or_nonpublic"), ({"id": "abcDEF12345", "availability": "needs_auth"}, "age_or_login_gated"), ({"id": "wrong", "duration": 1}, "malformed_or_unsupported_source"), ({"id": "abcDEF12345"}, "duration_unavailable"), (RuntimeError("bad"), "download_failed")])
def test_metadata_terminal_mapping(tmp_path, payload, state):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake(payload)
    contract = _contract(tmp_path)
    record = recover_public_youtube_audio(item_id="item", source_url="https://youtube.com/watch?v=abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module)
    assert record["terminal_state"] == state


def test_metadata_extraction_failure_is_one_terminal_operation_before_stream_staging(tmp_path):
    """A deterministic metadata failure is evidence, not three invented retries."""
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio

    module = _fake(RuntimeError("metadata unavailable"))
    contract = _contract(tmp_path)
    result = recover_public_youtube_audio(
        item_id="item", source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root, artifact_contract=contract,
        preflight=_preflight(module), _test_module=module, max_attempts=3,
    )

    assert result["terminal_state"] == "download_failed"
    assert result["attempt_count"] == 1
    assert module.calls == [("https://www.youtube.com/watch?v=abcDEF12345", False)]
    assert module.stream_urls == []
    assert not contract.audio_root.exists()


def test_transient_metadata_transport_failures_record_the_bounded_actual_count(tmp_path):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio

    module = _fake(TimeoutError("one"), TimeoutError("two"), TimeoutError("three"))
    contract = _contract(tmp_path)
    result = recover_public_youtube_audio(
        item_id="item", source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root, artifact_contract=contract,
        preflight=_preflight(module), _test_module=module, max_attempts=3,
    )

    assert result["terminal_state"] == "download_failed"
    assert result["attempt_count"] == 3
    assert len(module.calls) == 3


def test_download_retry_discards_failed_staging(tmp_path):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake({"id": "abcDEF12345", "duration": 2, "formats": [{"url": "https://media.example.invalid/a.mp3", "protocol": "https", "vcodec": "none", "acodec": "mp3"}]})
    attempts = [RuntimeError("bad")]
    def fail_once(_url):
        if attempts:
            raise attempts.pop()
        return io.BytesIO(module.stream_bytes)
    module.urlopen_hook = fail_once
    contract = _contract(tmp_path)
    record = recover_public_youtube_audio(item_id="item", source_url="https://youtube.com/watch?v=abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module, max_attempts=2)
    assert record["terminal_state"] == "acquired" and record["attempt_count"] == 2
    assert not list(contract.audio_root.glob(".*.staged.mp3"))


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_test_module_is_rejected_for_production_contract(tmp_path):
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake({"id": "abcDEF12345", "duration": 2})
    with pytest.raises(ValueError, match="unavailable in production"):
        recover_public_youtube_audio(item_id="item", source_url="https://youtu.be/abcDEF12345", r6_root=Path("x"), artifact_contract=R6ArtifactContract.production(), preflight=_preflight(module), _test_module=module)


def test_option_fence_denies_auth_hooks_sidecars_and_non_descriptor_output():
    from conference_pipeline.acquisition.youtube_audio_recovery import _assert_options, _metadata_options
    options = _metadata_options()
    options["cookiefile"] = "x"
    with pytest.raises(Exception, match="policy"):
        _assert_options(options, download=False)
    with pytest.raises(Exception, match="policy"):
        _assert_options(options, download=True)


@pytest.mark.parametrize("bad", [
    {"requested_formats": [{"url": "https://x"}]},
    {"fragments": [{"url": "https://x"}]},
    {"_filename": "evil.%(ext)s"},
])
def test_metadata_fanout_is_rejected_before_stream_or_staging(tmp_path, bad):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    payload = {
        "id": "abcDEF12345",
        "duration": 2,
        "formats": [
            {
                "url": "https://media.example.invalid/a.mp3",
                "protocol": "https",
                "vcodec": "none",
                "acodec": "mp3",
                **bad,
            }
        ],
    }
    module = _fake(payload)
    contract = _contract(tmp_path)
    result = recover_public_youtube_audio(item_id="item", source_url="https://youtu.be/abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module)
    assert result["terminal_state"] == "download_failed"
    assert module.stream_urls == []
    assert not contract.audio_root.exists()


def test_informational_metadata_does_not_block_the_selected_audio_stream(tmp_path):
    """yt-dlp's read-only metadata inventories are not download sidecars.

    Public YouTube metadata normally includes thumbnails, subtitles, and
    automatic-caption listings even when every corresponding write option is
    disabled.  Their presence must not make the direct audio-only stream
    unreachable.
    """

    from conference_pipeline.acquisition.youtube_audio_recovery import (
        recover_public_youtube_audio,
    )

    stream_url = "https://media.example.invalid/a.m4a"
    module = _fake(
        {
            "id": "abcDEF12345",
            "duration": 2,
            "thumbnails": [{"url": "https://images.example.invalid/thumb.jpg"}],
            "subtitles": {"en": [{"url": "https://captions.example.invalid/en.vtt"}]},
            "automatic_captions": {
                "en": [{"url": "https://captions.example.invalid/auto-en.vtt"}]
            },
            # ``extract_info(download=False)`` may still describe yt-dlp's
            # default selection. Recovery never executes that selection; it
            # opens only the independently validated audio-only format below.
            "requested_formats": [{"format_id": "video"}, {"format_id": "audio"}],
            "formats": [
                {
                    "url": stream_url,
                    "protocol": "https",
                    "vcodec": "none",
                    "acodec": "mp4a.40.2",
                    "format_id": "140",
                }
            ],
        }
    )
    contract = _contract(tmp_path)

    result = recover_public_youtube_audio(
        item_id="item",
        source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root,
        artifact_contract=contract,
        preflight=_preflight(module),
        _test_module=module,
    )

    assert result["terminal_state"] == "acquired"
    assert module.stream_urls == [stream_url]


@pytest.mark.parametrize(
    "bad",
    [
        {"requested_downloads": [{"format_id": "write-mode-selection"}]},
        {"filepath": "/unexpected/yt-dlp-output.webm"},
        {"__files_to_move": {"source": "destination"}},
    ],
)
def test_top_level_write_mode_state_is_rejected(tmp_path, bad):
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        recover_public_youtube_audio,
    )

    payload = {
        "id": "abcDEF12345",
        "duration": 2,
        "formats": [
            {
                "url": "https://media.example.invalid/a.m4a",
                "protocol": "https",
                "vcodec": "none",
                "acodec": "mp4a.40.2",
            }
        ],
        **bad,
    }
    module = _fake(payload)
    contract = _contract(tmp_path)

    result = recover_public_youtube_audio(
        item_id="item",
        source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root,
        artifact_contract=contract,
        preflight=_preflight(module),
        _test_module=module,
    )

    assert result["terminal_state"] == "download_failed"
    assert module.stream_urls == []


@pytest.mark.parametrize(
    "stream_url",
    ["https:///missing-host.m4a", "https://user:secret@media.example.invalid/a.m4a"],
)
def test_direct_stream_requires_a_host_without_embedded_credentials(tmp_path, stream_url):
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        recover_public_youtube_audio,
    )

    module = _fake(
        {
            "id": "abcDEF12345",
            "duration": 2,
            "formats": [
                {
                    "url": stream_url,
                    "protocol": "https",
                    "vcodec": "none",
                    "acodec": "mp4a.40.2",
                }
            ],
        }
    )
    contract = _contract(tmp_path)

    result = recover_public_youtube_audio(
        item_id="item",
        source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root,
        artifact_contract=contract,
        preflight=_preflight(module),
        _test_module=module,
    )

    assert result["terminal_state"] == "download_failed"
    assert module.stream_urls == []


def test_recovery_rejects_a_preflight_with_a_mismatched_loaded_identity(tmp_path):
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio, R6DownloaderPreflightError
    module = _fake({"id": "abcDEF12345", "duration": 2})
    preflight = _preflight(module)
    preflight["distribution_identity"] = "yt-dlp==different"
    contract = _contract(tmp_path)
    with pytest.raises(R6DownloaderPreflightError, match="identity changed"):
        recover_public_youtube_audio(item_id="item", source_url="https://youtu.be/abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=preflight, _test_module=module)


def test_disposable_distribution_drift_rejects_at_the_next_owner_preflight(tmp_path):
    """Tree/RECORD verification is a pre-dispatch owner operation."""
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        R6DownloaderPreflightError,
        preflight_fixed_r6_yt_dlp,
    )
    package, record = _recorded_package(tmp_path)
    module = _fake({"id": "abcDEF12345", "duration": 2})
    module.__file__ = str(package / "__init__.py")
    module.__r6_distribution_root__ = package
    module.__r6_distribution_record__ = record
    preflight = preflight_fixed_r6_yt_dlp(_test_module=module)
    (package / "injected.py").write_bytes(b"drift")
    with pytest.raises(R6DownloaderPreflightError, match="tree and RECORD|provenance"):
        preflight_fixed_r6_yt_dlp(_test_module=module)
    assert module.calls == []


def test_actual_client_uses_only_metadata_and_inprocess_transport(tmp_path, monkeypatch):
    """The real client class uses the unmodified recovery-time provenance path."""
    import yt_dlp
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        preflight_fixed_r6_yt_dlp, recover_public_youtube_audio,
    )
    def poison(*_args, **_kwargs):
        raise AssertionError("child/postprocessor lane is forbidden")
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, poison)
    for name in ("system", "execv", "execve", "execvp", "execvpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, poison)
    # yt-dlp's postprocessor constructors are process-capable even if a
    # metadata-only path ought never to instantiate them.
    import yt_dlp.postprocessor as postprocessor
    for name in ("FFmpegPostProcessor", "FFmpegMergerPP", "FFmpegFixupM3u8PP", "FFmpegVideoRemuxerPP", "FFmpegVideoConvertorPP"):
        if hasattr(postprocessor, name):
            monkeypatch.setattr(postprocessor, name, poison)
    metadata = {"id": "abcDEF12345", "duration": 2, "formats": [{"url": "https://media.example.invalid/a.mp3", "protocol": "https", "vcodec": "none", "acodec": "mp3", "format_id": "a"}]}
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", lambda _self, _url, *, download=False: metadata)
    monkeypatch.setattr(yt_dlp.YoutubeDL, "urlopen", lambda _self, _url: io.BytesIO(b"actual-client-audio"))
    contract = _contract(tmp_path)
    result = recover_public_youtube_audio(
        item_id="item", source_url="https://youtu.be/abcDEF12345",
        r6_root=contract.r6_root, artifact_contract=contract,
        # This is the actual installed module/distribution/RECORD/lock path;
        # only its in-memory transport methods are replaced.
        preflight=preflight_fixed_r6_yt_dlp(),
    )
    assert result["terminal_state"] == "acquired"
    assert result["sha256"] == hashlib.sha256(b"actual-client-audio").hexdigest()


@pytest.mark.parametrize("failure", ["read", "write", "publish"])
def test_stream_response_is_closed_once_on_every_failure(tmp_path, monkeypatch, failure):
    """The in-process transport must not leak a response on any sink path."""
    from conference_pipeline.acquisition.youtube_audio_recovery import recover_public_youtube_audio
    module = _fake({"id": "abcDEF12345", "duration": 2, "formats": [{"url": "https://media.example.invalid/a.mp3", "protocol": "https", "vcodec": "none", "acodec": "mp3"}]})
    class Response:
        closes = 0
        def read(self, _size):
            if failure == "read": raise RuntimeError("read")
            return b"audio" if not hasattr(self, "done") else b""
        def close(self): self.closes += 1
    response = Response()
    # Avoid an infinite successful read in the non-read cases.
    original_read = response.read
    def once(size):
        if failure == "read": return original_read(size)
        if getattr(response, "done", False): return b""
        response.done = True
        return b"audio"
    response.read = once  # type: ignore[method-assign]
    module.urlopen_hook = lambda _url: response
    contract = _contract(tmp_path)
    if failure == "write":
        from conference_pipeline.acquisition import youtube_audio_recovery as mod
        monkeypatch.setattr(mod._DescriptorOnlySink, "copy", lambda _self, _response: (_ for _ in ()).throw(RuntimeError("write")))
    if failure == "publish":
        from conference_pipeline.acquisition import cvpr_r6_artifact_contract as contract_mod
        monkeypatch.setattr(contract_mod.R6RawAudioStagingCapability, "publish_and_read", lambda _self: (_ for _ in ()).throw(RuntimeError("publish")))
    result = recover_public_youtube_audio(item_id="item", source_url="https://youtu.be/abcDEF12345", r6_root=contract.r6_root, artifact_contract=contract, preflight=_preflight(module), _test_module=module)
    assert result["terminal_state"] == "download_failed"
    assert response.closes == 1
