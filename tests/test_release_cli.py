"""The `conference-pipeline release` corpus (v6.0-Public-Release Phase 3).

Each test is one case of ``tests/fixtures/release_cli/README.md``; its exit code comes
from the answer key ``EXPECTED.json``. Cases run in ``tmp_path`` copies of
``conferences/TEMPLATE``; no case calls a live provider. Where a case needs live
commands to "run", a fake runner records them and writes the child receipt each would.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "release_cli"
CASES = json.loads((CORPUS / "EXPECTED.json").read_text(encoding="utf-8"))["cases"]
SLUG = "sampleconf2027"
BUNDLE = f"{SLUG}-v1.0.0"
DECK = Path("decks") / "01_sample" / "slides.md"


def expected_exit(case: str) -> int:
    return CASES[case]["exit"]


def cli():
    return importlib.import_module("conference_pipeline.publishing.release_cli")


def orchestrator():
    return importlib.import_module("conference_pipeline.publishing.public_release")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _no_secrets(paths: list[Path]) -> list[dict]:
    return []


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.conf = root / "conferences" / SLUG
        self.runs = self.conf / "storage" / "workflow-runs"
        self.manifest = self.conf / "storage" / "public_export_manifest.json"
        self.bundle = self.conf / "exports" / "public-release" / BUNDLE
        self.out: list[str] = []

    def echo(self, line: str) -> None:
        self.out.append(line)

    def kw(self) -> dict[str, Any]:
        return {"project_root": self.root, "echo": self.echo, "err": self.echo}

    def snapshot(self) -> dict[str, str]:
        return {
            path.relative_to(self.conf).as_posix(): (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
            )
            for path in sorted(self.conf.rglob("*"))
        }

    def text(self) -> str:
        return "\n".join(self.out)


@pytest.fixture
def w0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """W0: a fresh TEMPLATE copy with one deck file, in a one-commit git repository."""
    root = tmp_path / "project"
    ws = Workspace(root)
    shutil.copytree(REPO_ROOT / "conferences" / "TEMPLATE", ws.conf)
    config = ws.conf / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace('slug: "replace-me"', f'slug: "{SLUG}"'), encoding="utf-8")
    (ws.conf / DECK).parent.mkdir(parents=True)
    (ws.conf / DECK).write_text("public slides\n", encoding="utf-8")
    (ws.runs).mkdir(parents=True, exist_ok=True)
    (root / "scripts").symlink_to(REPO_ROOT / "scripts")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))
    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    return ws


def draft(ws: Workspace) -> Workspace:
    assert cli().release_init(SLUG, **ws.kw()) == 0, ws.text()
    return ws


def promote(ws: Workspace) -> Workspace:
    manifest = json.loads(ws.manifest.read_text(encoding="utf-8"))
    manifest["excluded_families"] = [
        row for row in manifest["excluded_families"] if row["storage_manifest_id"] != "presentation_markdown"
    ]
    manifest["families"] = [
        {
            "id": "presentation_markdown",
            "private_path": "decks",
            "public_path": "exports/public-release/{bundle}/decks/",
            "storage_manifest_id": "presentation_markdown",
            "redistribution_status": "allowed",
            "license": "CC-BY-SA-4.0",
            "provenance": "pipeline-generated",
        }
    ]
    ws.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ws


def plan(ws: Workspace) -> Workspace:
    assert cli().release_plan(SLUG, **ws.kw()) == 0, ws.text()
    return ws


def build(ws: Workspace) -> Workspace:
    assert cli().release_build(SLUG, mock=True, **ws.kw()) == 0, ws.text()
    return ws


def plant_live_receipt(ws: Workspace) -> None:
    payload = {
        "receipt_type": "conference_pipeline.public_repo_push",
        "status": "succeeded",
        "exit_code": 0,
        "dry_run": False,
        "live": True,
        "bundle_id": BUNDLE,
        "commit_sha": "0" * 40,
        "public_repo_url": "https://example.invalid/sampleconf2027-data",
    }
    (ws.runs / f"github-{BUNDLE}-live-receipt.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


_STAGE_BY_RECEIPT_PREFIX = (
    ("github-", "live_github"),
    ("cloudflare-pages-", "live_pages"),
    ("hf-", "live_hf"),
    ("zenodo-", "live_zenodo"),
    ("public-url-liveness-", "check_liveness"),
    ("public-release-metadata-agreement-", "check_metadata"),
    ("public-deployment-preflight-", "preflight_live"),
)


class FakeRunner:
    """Records each child command and writes the receipt a successful child would."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.argvs: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        module = orchestrator()
        receipt = Path(argv[argv.index("--receipt-path") + 1])
        stage = next(stage for prefix, stage in _STAGE_BY_RECEIPT_PREFIX if receipt.name.startswith(prefix))
        self.stages.append(stage)
        self.argvs.append([str(part) for part in argv])
        stage_enum = module.PublicReleaseStage(stage)
        payload: dict[str, Any] = {
            "receipt_type": module._CHILD_RECEIPT_TYPES[stage_enum],
            "status": "succeeded",
            "exit_code": 0,
            "bundle_id": BUNDLE,
            "conference": SLUG,
        }
        for field in module._CHILD_REQUIRED_FIELDS.get(stage_enum, ()):
            payload.setdefault(field, "fixture")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


# --- init --------------------------------------------------------------------------


def test_c1_init_drafts_every_family_excluded(w0: Workspace) -> None:
    code = cli().release_init(SLUG, **w0.kw())
    assert code == expected_exit("C1")
    manifest = json.loads(w0.manifest.read_text(encoding="utf-8"))
    storage = json.loads((w0.conf / "storage" / "storage_manifest.json").read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w0.root, capture_output=True, text=True).stdout.strip()
    assert manifest["conference"] == SLUG
    assert manifest["release"]["source_sha"] == head
    assert manifest["families"] == []
    ids = [row["storage_manifest_id"] for row in manifest["excluded_families"]]
    assert ids == [family["id"] for family in storage["artifact_families"]]
    assert len(ids) == 29
    for row in manifest["excluded_families"]:
        assert set(row) == {"storage_manifest_id", "path", "storage_class", "share_policy", "reason"}
        assert row["reason"] == "default: not reviewed for publication"
    assert not (w0.conf / "exports").exists()


def test_c1_schema_version_is_the_current_one(w0: Workspace) -> None:
    """C1's schema_version: 1.0.0 (amended by the owner, IF3)."""
    draft(w0)
    manifest = json.loads(w0.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"


def test_c2_init_never_overwrites(w0: Workspace) -> None:
    draft(w0)
    before = w0.manifest.read_bytes()
    code = cli().release_init(SLUG, **w0.kw())
    assert code == expected_exit("C2")
    assert w0.manifest.read_bytes() == before
    assert "--force-draft-path" in w0.text()


def test_c3_init_needs_a_storage_manifest(w0: Workspace) -> None:
    (w0.conf / "storage" / "storage_manifest.json").unlink()
    code = cli().release_init(SLUG, **w0.kw())
    assert code == expected_exit("C3")
    assert not w0.manifest.exists()


# --- plan --------------------------------------------------------------------------


def test_c4_plan_refuses_an_unpromoted_draft(w0: Workspace) -> None:
    draft(w0)
    code = cli().release_plan(SLUG, **w0.kw())
    assert code == expected_exit("C4")
    assert "no_family_promoted" in w0.text()
    assert not (w0.conf / "exports").exists()


def test_c6_plan_writes_receipts_and_no_bundle(w0: Workspace) -> None:
    promote(draft(w0))
    code = cli().release_plan(SLUG, **w0.kw())
    assert code == expected_exit("C6"), w0.text()
    assert (w0.runs / f"public-export-{BUNDLE}-validation-receipt.json").is_file()
    dry = json.loads((w0.runs / f"public-export-{BUNDLE}-dry-run-receipt.json").read_text(encoding="utf-8"))
    assert dry["status"] == "succeeded" and dry["dry_run"] is True
    assert not (w0.conf / "exports").exists()


# --- build -------------------------------------------------------------------------


def test_c5_build_refuses_without_the_dry_run(w0: Workspace) -> None:
    promote(draft(w0))
    code = cli().release_build(SLUG, mock=True, **w0.kw())
    assert code == expected_exit("C5")
    assert not (w0.conf / "exports" / "public-release" / BUNDLE).exists()


def test_c7_build_writes_a_verifiable_bundle(w0: Workspace) -> None:
    plan(promote(draft(w0)))
    code = cli().release_build(SLUG, mock=True, **w0.kw())
    assert code == expected_exit("C7"), w0.text()
    for name in ("RELEASE.json", "CHECKSUMS.json", DECK.as_posix()):
        assert (w0.bundle / name).is_file(), name
    receipt = json.loads((w0.runs / f"public-export-{BUNDLE}-write-receipt.json").read_text(encoding="utf-8"))
    export = orchestrator()._load_export_script(REPO_ROOT)
    files = [row["path"] for row in receipt["actual_artifacts"]]
    assert {"RELEASE.json", "CHECKSUMS.json"} <= set(files)
    assert export.tree_sha256_for(w0.bundle, files) == receipt["bundle_verification"]["bundle_tree_sha256"]
    stages = []
    for command in ("build", "build-repo"):
        top = json.loads((w0.runs / f"public-release-{BUNDLE}-{command}-receipt.json").read_text(encoding="utf-8"))
        stages += [row["stage"] for row in top["command_results"]]
    assert stages == ["export_write", "build_index", "generate_site", "package_dataset", "stage_repo"]


# --- check -------------------------------------------------------------------------


@pytest.fixture
def w4(w0: Workspace) -> Workspace:
    return build(plan(promote(draft(w0))))


def test_c8_check_passes_a_fresh_bundle(w4: Workspace) -> None:
    code = cli().release_check(SLUG, secret_scanner=_no_secrets, **w4.kw())
    assert code == expected_exit("C8"), w4.text()
    assert "finding:" not in w4.text()


def test_c9_check_catches_and_redacts_a_home_path(w4: Workspace) -> None:
    planted = "/" + "home" + "/alice/notes.txt"  # built at run time; see the corpus README
    with (w4.bundle / DECK).open("a", encoding="utf-8") as handle:
        handle.write(f"see {planted}\n")
    code = cli().release_check(SLUG, secret_scanner=_no_secrets, **w4.kw())
    assert code == expected_exit("C9")
    lines = [line for line in w4.out if "private_marker" in line]
    assert len(lines) == 1 and "home_path" in lines[0] and DECK.as_posix() in lines[0]
    assert "<redacted:home_path>" in lines[0]
    assert planted not in w4.text() and "alice" not in w4.text()


def test_c10_check_notices_an_altered_byte(w4: Workspace) -> None:
    path = w4.bundle / DECK
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))
    code = cli().release_check(SLUG, secret_scanner=_no_secrets, **w4.kw())
    assert code == expected_exit("C10")
    assert any(DECK.as_posix() in line for line in w4.out if line.startswith("finding:"))


@pytest.mark.parametrize("damage", ["no_family_licenses", "stale_license_file"])
def test_6b_check_refuses_a_bundle_whose_licenses_are_not_its_families(w4: Workspace, damage: str) -> None:
    """An older exporter wrote no family_licenses and a fixed LICENSE; every other check passed it."""
    release_path = w4.bundle / "RELEASE.json"
    if damage == "no_family_licenses":
        release = json.loads(release_path.read_text(encoding="utf-8"))
        release.pop("family_licenses", None)
        release_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    else:
        license_path = w4.bundle / "public-repo" / "LICENSE"
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_text("This release uses a split license model.\n", encoding="utf-8")
    code = cli().release_check(SLUG, secret_scanner=_no_secrets, **w4.kw())
    assert code != 0
    assert any("licenses" in line for line in w4.out if line.startswith("finding:")), w4.text()


# --- publish -----------------------------------------------------------------------


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    opened: list[Path] = []

    def refuse(path: Path, environ: Any):
        opened.append(path)
        raise AssertionError(".env must not be read")

    monkeypatch.setattr(orchestrator(), "_load_env_file", refuse)
    return opened


@pytest.mark.parametrize("case,approver", [("C11", None), ("C12", "  ")])
def test_c11_c12_publish_needs_a_named_approver(w0: Workspace, no_env: list[Path], case: str, approver: str | None) -> None:
    plan(promote(draft(w0)))
    runner = FakeRunner()
    code = cli().release_publish(SLUG, approved_by=approver, surfaces=("github",), runner=runner, **w0.kw())
    assert code == expected_exit(case)
    assert "owner_approval_missing" in w0.text()
    assert runner.stages == [] and no_env == []


def test_c13_publish_runs_the_live_stage_then_verification(w4: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """C13, from W4 (IF1) with the IF2 order (both amended by the owner)."""
    monkeypatch.setattr(orchestrator(), "_load_env_file", lambda path, environ: ({}, ()))
    runner = FakeRunner()
    code = cli().release_publish(
        SLUG,
        approved_by="alice",
        surfaces=("github",),
        github_repo="example-org/sampleconf2027-data",
        runner=runner,
        **w4.kw(),
    )
    assert code == expected_exit("C13"), w4.text()
    assert runner.stages == ["live_github", "check_liveness", "check_metadata"]
    top = json.loads((w4.runs / f"public-release-{BUNDLE}-publish-receipt.json").read_text(encoding="utf-8"))
    assert top["owner_approval"]["approved_by"] == "alice"
    assert top["owner_approval"]["surfaces"] == ["github"]
    assert top["secret_values_recorded"] is False


def test_c14_publish_needs_a_surface(w0: Workspace, no_env: list[Path]) -> None:
    plan(promote(draft(w0)))
    runner = FakeRunner()
    code = cli().release_publish(SLUG, approved_by="alice", surfaces=(), runner=runner, **w0.kw())
    assert code == expected_exit("C14")
    assert "--surface" in w0.text()
    assert runner.stages == []


# --- isolated builds ---------------------------------------------------------------


@pytest.fixture
def w2_live(w0: Workspace) -> Workspace:
    promote(draft(w0))
    plant_live_receipt(w0)
    return w0


def test_c15_isolated_rebuild_leaves_the_workspace_untouched(w2_live: Workspace, tmp_path: Path) -> None:
    iso = tmp_path / "iso"
    before = w2_live.snapshot()
    assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == expected_exit("C15"), w2_live.text()
    code = cli().release_build(SLUG, isolated=iso, **w2_live.kw())
    assert code == expected_exit("C15"), w2_live.text()
    for name in ("RELEASE.json", "CHECKSUMS.json", DECK.as_posix()):
        assert (iso / "bundle" / name).is_file(), name
    assert w2_live.snapshot() == before
    for report in ("plan-report.json", "build-report.json"):
        text = (iso / report).read_text(encoding="utf-8")
        assert str(tmp_path) not in text and str(w2_live.root) not in text


def test_c16_published_bundle_is_not_overwritten(w2_live: Workspace) -> None:
    plan(w2_live)
    code = cli().release_build(SLUG, mock=True, **w2_live.kw())
    assert code == expected_exit("C16")
    assert "public export bundle is immutable after publication" in w2_live.text()
    assert not w2_live.bundle.exists()


def test_c17_isolated_digest_mismatch_lists_paths(w2_live: Workspace, tmp_path: Path) -> None:
    iso = tmp_path / "iso"
    assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == 0, w2_live.text()
    code = cli().release_build(SLUG, isolated=iso, expect_tree_sha256="0" * 64, **w2_live.kw())
    assert code == expected_exit("C17")
    report = json.loads((iso / "build-report.json").read_text(encoding="utf-8"))
    assert report["expected_tree_sha256"] == "0" * 64
    assert report["bundle_tree_sha256"] != "0" * 64
    assert report["tree_sha256_matches"] is False


def test_c17_differing_paths_are_sorted_against_a_reference(w2_live: Workspace, tmp_path: Path) -> None:
    receipt = w2_live.runs / f"public-export-{BUNDLE}-write-receipt.json"
    receipt.write_text(
        json.dumps({"actual_artifacts": [{"path": "zz.md", "sha256": "1", "size_bytes": 1}, {"path": "aa.md", "sha256": "1", "size_bytes": 1}]}),
        encoding="utf-8",
    )
    iso = tmp_path / "iso"
    assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == 0, w2_live.text()
    cli().release_build(SLUG, isolated=iso, expect_tree_sha256="0" * 64, **w2_live.kw())
    report = json.loads((iso / "build-report.json").read_text(encoding="utf-8"))
    assert report["differing_paths"] == sorted(report["differing_paths"])
    assert {"aa.md", "zz.md", "RELEASE.json"} <= set(report["differing_paths"])


def test_c19_isolated_build_needs_its_dry_run(w2_live: Workspace, tmp_path: Path) -> None:
    iso = tmp_path / "iso"
    code = cli().release_build(SLUG, isolated=iso, **w2_live.kw())
    assert code == expected_exit("C19")
    assert not (iso / "bundle").exists()
    iso2 = tmp_path / "iso2"
    assert cli().release_plan(SLUG, isolated=iso2, **w2_live.kw()) == 0
    report = json.loads((iso2 / "plan-report.json").read_text(encoding="utf-8"))
    report["export_manifest_sha256"] = "f" * 64
    (iso2 / "plan-report.json").write_text(json.dumps(report), encoding="utf-8")
    code = cli().release_build(SLUG, isolated=iso2, **w2_live.kw())
    assert code == expected_exit("C19")
    assert not (iso2 / "bundle").exists()
    assert "plan-report.json" in w2_live.text()


@pytest.mark.parametrize("variant", ["inside", "non_empty"])
def test_c20_isolated_directory_must_be_outside_and_empty(w0: Workspace, tmp_path: Path, variant: str) -> None:
    promote(draft(w0))
    if variant == "inside":
        iso = w0.conf / "tmp-iso"
    else:
        iso = tmp_path / "iso"
        iso.mkdir()
        (iso / "keep.txt").write_text("x\n", encoding="utf-8")
    before = w0.snapshot()
    code = cli().release_plan(SLUG, isolated=iso, **w0.kw())
    assert code == expected_exit("C20")
    assert w0.snapshot() == before
    assert not (iso / "plan-report.json").exists()
    if variant == "non_empty":
        assert sorted(p.name for p in iso.iterdir()) == ["keep.txt"]


# --- configuration -----------------------------------------------------------------


def test_c18a_publish_reads_targets_from_config(w4: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """C18 (a), from W4 (IF1, amended by the owner)."""
    monkeypatch.setattr(orchestrator(), "_load_env_file", lambda path, environ: ({}, ()))
    config = w4.conf / "config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8") + '\nrelease:\n  github_repo: "example-org/sampleconf2027-data"\n',
        encoding="utf-8",
    )
    runner = FakeRunner()
    code = cli().release_publish(SLUG, approved_by="alice", surfaces=("github",), runner=runner, **w4.kw())
    assert code == expected_exit("C18"), w4.text()
    live = runner.argvs[runner.stages.index("live_github")]
    assert live[live.index("--repo-name") + 1] == "sampleconf2027-data"


def test_c18b_repo_name_defaults_to_the_workspace_slug(w4: Workspace, tmp_path: Path) -> None:
    from conference_pipeline.publishing.public_repo import PublicRepoConfig, stage_public_repo

    target = tmp_path / "repo"
    result = stage_public_repo(
        PublicRepoConfig(
            bundle_root=w4.bundle,
            target_dir=target,
            dry_run=True,
            receipt_path=tmp_path / "repo-receipt.json",
            secret_scanner=_no_secrets,
        )
    )
    receipt = json.loads((tmp_path / "repo-receipt.json").read_text(encoding="utf-8"))
    assert receipt["repo_name"] == "sampleconf2027-data"
    assert "neurips2025" not in json.dumps(receipt)
    assert result is not None


# --- wiring ------------------------------------------------------------------------


def test_release_group_is_on_the_cli() -> None:
    from typer.testing import CliRunner

    from conference_pipeline.cli import app

    result = CliRunner().invoke(app, ["release", "--help"])
    assert result.exit_code == 0
    for command in ("init", "plan", "build", "check", "publish"):
        assert command in result.output


def test_template_end_to_end(w0: Workspace) -> None:
    """T2: init, promote, plan, build (mock), check -- exit 0 at every step."""
    module = cli()
    assert module.release_init(SLUG, **w0.kw()) == 0, w0.text()
    promote(w0)
    assert module.release_plan(SLUG, **w0.kw()) == 0, w0.text()
    assert module.release_build(SLUG, mock=True, **w0.kw()) == 0, w0.text()
    assert module.release_check(SLUG, secret_scanner=_no_secrets, **w0.kw()) == 0, w0.text()


# --- successors (implementation and review regressions, not owner-approved cases) ----


def test_s1_restaging_never_nests_the_bundles_own_repo(w4: Workspace, tmp_path: Path) -> None:
    """IF7: staging into another directory must not copy <bundle>/public-repo/ into it."""
    from conference_pipeline.publishing.public_repo import PublicRepoConfig, stage_public_repo

    assert (w4.bundle / "public-repo").is_dir()
    stage_public_repo(
        PublicRepoConfig(
            bundle_root=w4.bundle,
            target_dir=tmp_path / "elsewhere",
            dry_run=True,
            receipt_path=tmp_path / "elsewhere-receipt.json",
            secret_scanner=_no_secrets,
        )
    )
    assert not (tmp_path / "elsewhere" / "public-repo").exists()


def test_t4_two_isolated_builds_are_byte_identical(w2_live: Workspace, tmp_path: Path) -> None:
    """T4 / CP3's second clause: a second materialization is byte-identical."""
    digests = []
    for name in ("iso-a", "iso-b"):
        iso = tmp_path / name
        assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == 0, w2_live.text()
        assert cli().release_build(SLUG, isolated=iso, **w2_live.kw()) == 0, w2_live.text()
        digests.append(json.loads((iso / "build-report.json").read_text(encoding="utf-8"))["bundle_tree_sha256"])
    assert digests[0] == digests[1]
    files_a = sorted(p.relative_to(tmp_path / "iso-a" / "bundle") for p in (tmp_path / "iso-a" / "bundle").rglob("*"))
    files_b = sorted(p.relative_to(tmp_path / "iso-b" / "bundle") for p in (tmp_path / "iso-b" / "bundle").rglob("*"))
    assert files_a == files_b
    for rel in files_a:
        a, b = tmp_path / "iso-a" / "bundle" / rel, tmp_path / "iso-b" / "bundle" / rel
        if a.is_file():
            assert a.read_bytes() == b.read_bytes(), rel


def test_s3_project_root_option_reaches_every_command(w0: Workspace) -> None:
    """IF10: the Typer commands accept --project-root and act on that checkout."""
    from typer.testing import CliRunner

    from conference_pipeline.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["release", "init", SLUG, "--project-root", str(w0.root)])
    assert result.exit_code == 0, result.output
    assert w0.manifest.is_file()
    for command in ("plan", "build", "check", "publish"):
        help_text = runner.invoke(app, ["release", command, "--help"]).output
        assert "--project-root" in help_text, command


# --- review phase-3-r1 regressions -----------------------------------------------------


def test_r1_f1_isolated_report_never_follows_a_symlink(w2_live: Workspace, tmp_path: Path) -> None:
    """F1: a build-report.json symlink into the workspace must not be written through."""
    iso = tmp_path / "iso"
    assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == 0, w2_live.text()
    victim = w2_live.runs / f"github-{BUNDLE}-live-receipt.json"
    before = victim.read_bytes()
    (iso / "build-report.json").symlink_to(victim)
    cli().release_build(SLUG, isolated=iso, **w2_live.kw())
    assert victim.read_bytes() == before


def test_r1_f3_check_scans_large_text_files(w4: Workspace) -> None:
    """F3: a text file over the scanner's 2 MiB cap is still scanned by check."""
    planted = "/" + "home" + "/alice/notes.txt"
    big = w4.bundle / "decks" / "big.md"
    big.write_text(("public line\n" * 200_000) + f"see {planted}\n", encoding="utf-8")
    code = cli().release_check(SLUG, secret_scanner=_no_secrets, **w4.kw())
    assert code == 1
    assert any("private_marker" in line and "big.md" in line for line in w4.out)
    assert planted not in w4.text()


def test_r1_f4_isolated_build_keeps_caution_acknowledgements(w0: Workspace, tmp_path: Path) -> None:
    """F4: --ack-caution-family reaches the isolated build, not only the isolated plan."""
    promote(draft(w0))
    manifest = json.loads(w0.manifest.read_text(encoding="utf-8"))
    manifest["families"][0]["redistribution_status"] = "caution"
    w0.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    iso = tmp_path / "iso"
    ack = ("presentation_markdown",)
    assert cli().release_plan(SLUG, isolated=iso, ack_caution_families=ack, **w0.kw()) == 0, w0.text()
    assert cli().release_build(SLUG, isolated=iso, ack_caution_families=ack, **w0.kw()) == 0, w0.text()


def test_r1_f5_check_json_redacts_every_field(w4: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F5: an operator marker in the bundle id never reaches the JSON report."""
    marker = "OperatorSecretName"
    monkeypatch.setenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", marker)
    out = tmp_path / "check.json"
    cli().release_check(SLUG, bundle_id=f"{marker}-v9", json_path=out, secret_scanner=_no_secrets, **w4.kw())
    if out.exists():
        assert marker not in out.read_text(encoding="utf-8")
    assert marker not in w4.text()


def test_r1_f7_isolated_report_always_states_its_path_comparison(w2_live: Workspace, tmp_path: Path) -> None:
    """F7: without a write receipt the report still says whether paths could be compared."""
    iso = tmp_path / "iso"
    assert cli().release_plan(SLUG, isolated=iso, **w2_live.kw()) == 0
    cli().release_build(SLUG, isolated=iso, expect_tree_sha256="0" * 64, **w2_live.kw())
    report = json.loads((iso / "build-report.json").read_text(encoding="utf-8"))
    assert "differing_paths" in report
    assert report["differing_paths"] is None
    assert report["reference_receipt"] is None


# --- v6.0 Phase 7.1 ------------------------------------------------------------------


def _with_targets(ws: Workspace) -> Workspace:
    config = ws.conf / "config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\nrelease:\n  github_repo: "example-org/sampleconf2027-data"\n  pages_project: "sampleconf2027-data"\n'
        '  hf_repo_id: "example-org/sampleconf2027-data"\n  cdn_base_url: "https://cdn.example.test/conferences/sampleconf2027"\n',
        encoding="utf-8",
    )
    _git(ws.root, "add", "-A")
    _git(ws.root, "commit", "-q", "-m", "targets")
    return ws


def test_t13_build_stamps_the_urls_before_the_digest_and_check_passes(w0: Workspace) -> None:
    ws = build(plan(promote(draft(_with_targets(w0)))))
    release = json.loads((ws.bundle / "RELEASE.json").read_text(encoding="utf-8"))
    assert release["public_repo_url"] == "https://github.com/example-org/sampleconf2027-data"
    assert release["public_site_url"] == "https://sampleconf2027-data.pages.dev"
    assert release["hf_dataset_url"] == "https://huggingface.co/datasets/example-org/sampleconf2027-data"
    assert release["cdn_base_url"] == "https://cdn.example.test/conferences/sampleconf2027"
    receipts = {p.name: p.read_bytes() for p in ws.runs.glob("*.json")}
    assert cli().release_check(SLUG, env={}, secret_scanner=_no_secrets, **ws.kw()) == 0, ws.text()
    # Checking reads receipts and the bundle; it rewrites neither (R9).
    assert {p.name: p.read_bytes() for p in ws.runs.glob("*.json")} == receipts


def test_t13_a_bad_target_refuses_the_build(w0: Workspace) -> None:
    config = w0.conf / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8") + '\nrelease:\n  github_repo: "not a repo"\n', encoding="utf-8")
    _git(w0.root, "add", "-A")
    _git(w0.root, "commit", "-q", "-m", "bad target")
    promote(draft(w0))
    assert cli().release_plan(SLUG, **w0.kw()) != 0
    assert "github_repo" in w0.text()


def test_t1_the_staged_repository_holds_no_site_or_index(w4: Workspace) -> None:
    repo = w4.bundle / "public-repo"
    assert (repo / "RELEASE.json").is_file() and (repo / DECK).is_file()
    assert (w4.bundle / "index").is_dir() and (w4.bundle / "site").is_dir()
    assert not (repo / "index").exists() and not (repo / "site").exists()
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "Hugging Face dataset" in readme and "`index/` contains" not in readme


def test_t3_check_reports_a_file_over_a_surface_limit(w4: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    import conference_pipeline.publishing.surface_limits as limits

    monkeypatch.setattr(limits, "LIMITS", tuple(
        limits.SurfaceLimit(row.surface, 64 if row.surface == "pages" else row.max_file_bytes, row.max_files,
                            row.max_entries_per_dir, row.source) for row in limits.LIMITS))
    code = cli().release_check(SLUG, env={}, secret_scanner=_no_secrets, **w4.kw())
    assert code == 1
    assert any(line.startswith("finding: surface_limit site/") for line in w4.out)
