"""Report-only orchestrator share audit on synthetic fixture orchestrators."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conference_pipeline import orchestrator_share_audit as audit
from tests.orchestrator_golden import FIXTURE_NAMES, FIXTURE_REPO, FIXTURE_ROOT, load_fixture_manifest

AUDIT_MODULE = Path(audit.__file__)


@pytest.fixture(scope="module")
def report() -> dict:
    return audit.run_audit(repo_root=FIXTURE_REPO)


def _orchestrator(report: dict, fixture: str) -> dict:
    target = load_fixture_manifest(fixture)["orchestrator"].split("repo/", 1)[1]
    matches = [item for item in report["orchestrators"] if item["path"] == target]
    assert len(matches) == 1, [item["path"] for item in report["orchestrators"]]
    return matches[0]


def _stages(item: dict) -> dict[str, dict]:
    return {stage["stage_id"]: stage for stage in item["stages"]}


def _units(item: dict) -> dict[str, dict]:
    return {unit["name"]: unit for unit in item["units"]}


def test_discovery_finds_runner_scripts_only(report: dict) -> None:
    assert report["discovery"] == "discovered"
    assert [item["path"] for item in report["orchestrators"]] == [
        "scripts/run_newconf2032_production.py",
        "scripts/run_quokka2031_production.py",
        "scripts/run_wombat2031_production.py",
    ]
    assert report["mode"] == "report-only"
    assert report["status"] == "pass"
    assert report["diagnostic"]["tool"] == "orchestrator_share_audit"
    assert report["diagnostic"]["blocker_count"] == 0


def test_registry_production_runner_ids_extend_discovery(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "drafts").mkdir()
    (repo / "scripts" / "run_alpha2040_production.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (repo / "drafts" / "next_orchestrator.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "scripts": {
                    "alpha2040.production_runner": {"path": "scripts/run_alpha2040_production.py"},
                    "beta2041.production_runner": {"path": "drafts/next_orchestrator.py"},
                    "beta2041.other_tool": {"path": "drafts/ignored.py"},
                }
            }
        ),
        encoding="utf-8",
    )

    found = audit.discover_orchestrators(repo, registry)

    assert [path.relative_to(repo.resolve()).as_posix() for path in found] == [
        "drafts/next_orchestrator.py",
        "scripts/run_alpha2040_production.py",
    ]


def test_minimal_new_conference_is_all_shared_with_no_hooks(report: dict) -> None:
    item = _orchestrator(report, "minimal_new_conference")
    metrics = item["metrics"]
    stages = _stages(item)

    assert list(stages) == ["download-corpus", "analysis", "release", "proof"]
    assert [stages[stage]["order"] for stage in stages] == [4, 7, 11, 12]
    assert {stage["status"] for stage in stages.values()} == {"shared"}
    assert metrics["shared_stage_ratio"] == 1.0
    assert item["hooks"] == []
    assert metrics["declared_hook_lines"] == 0
    assert metrics["dead_lines"] == 0
    assert metrics["shadow_lines"] == 0
    assert metrics["clone_lines"] == 0
    assert stages["analysis"]["shared_components"] == ["conference_pipeline.production.analysis_stage"]


def test_icml_shaped_runner_reports_local_stages_hooks_dead_and_shadows(report: dict) -> None:
    item = _orchestrator(report, "icml_shaped")
    stages = _stages(item)
    units = _units(item)
    metrics = item["metrics"]

    assert [hook["hook_id"] for hook in item["hooks"]] == ["download-corpus.auth", "download-corpus.acquisition_lane"]
    assert sorted(stages["download-corpus"]["hooks"]) == load_fixture_manifest("icml_shaped")["expected_hooks"]
    # Hooks are declared, but the stage still runs local logic.
    assert stages["download-corpus"]["status"] == "local"
    assert stages["download-corpus"]["local_entry_units"] == ["run_download_corpus"]
    # The hook decorator's module is not a shared component the stage adapts to.
    assert stages["download-corpus"]["shared_components"] == []
    assert stages["runner-preflight"]["status"] == "local"
    assert stages["analysis"]["status"] == "local"
    assert stages["release"]["status"] == "shared"
    assert stages["proof"]["shared_components"] == ["conference_pipeline.production.proof_stage"]
    assert metrics["shared_stage_count"] == 2 and metrics["stage_count"] == 5

    # Every stage has an explicit branch, so the not-implemented fallback is dead.
    assert item["fallback_branches_dead"] is True
    dead = sorted(name for name, unit in units.items() if unit["classification"] == "dead")
    assert dead == ["legacy_presentation_summary", "run_unavailable_stage"]
    assert metrics["dead_lines"] == sum(units[name]["lines"] for name in dead)

    assert units["sha256_file"]["shadow_of"]["function"] == "sha256_file"
    assert units["sha256_file"]["shadow_of"]["similarity"] == 1.0
    assert units["classify_speech_lane"]["shadow_of"]["path"] == "conference_pipeline/receipt_io.py"
    assert 0.8 <= units["classify_speech_lane"]["shadow_of"]["similarity"] < 1.0
    assert units["run_release"]["classification"] == "adapter"
    assert units["build_stage_receipt"]["group"] == "shared-infra"
    assert units["paper_priority_key"]["group"] == "download-corpus"
    assert units["openreview_bearer_auth"]["classification"] == "hook"


def test_cvpr_shaped_runner_counts_annotated_acquisition_modules_as_shared(report: dict) -> None:
    item = _orchestrator(report, "cvpr_shaped")
    stages = _stages(item)
    units = _units(item)

    download = stages["download-corpus"]
    assert download["status"] == "shared+hooks"
    assert download["hooks"] == ["download-corpus.acquisition_lane"]
    assert "conference_pipeline.acquisition.youtube_channel_harvest" in download["shared_components"]
    assert "conference_pipeline.production.download_corpus_stage" in download["shared_components"]
    assert download["local_lines"] == 0

    assert stages["source-discovery"]["status"] == "conference-local"
    assert stages["source-discovery"]["conference_local_components"] == [
        "conference_pipeline.acquisition.wombat_program_spine"
    ]
    assert item["conference_local_components"] == ["conference_pipeline.acquisition.wombat_program_spine"]
    assert stages["transcription-ocr"]["status"] == "local"
    # A long keyword signature does not stop a one-call wrapper being an adapter.
    assert units["run_release_stage"]["lines"] > 8
    assert units["run_release_stage"]["classification"] == "adapter"
    assert stages["release"]["status"] == "shared"

    assert item["extra_commands"] == ["prepare_preflight_estimate"]
    assert units["prepare_preflight_estimate"]["group"] == "cli-extra"
    assert units["dispatch_stage"]["classification"] == "dead"
    assert units["build_dispatch_fingerprint"]["classification"] == "dead"

    annotated = report["shared_component_rules"]["annotated_modules_referenced"]
    assert annotated == [
        {"module": "conference_pipeline.acquisition.youtube_channel_harvest", "recommended_for": ["WOMBAT"]}
    ]


def test_cross_runner_clones_normalize_conference_tokens(report: dict) -> None:
    icml = _units(_orchestrator(report, "icml_shaped"))
    cvpr = _units(_orchestrator(report, "cvpr_shaped"))

    # The two receipt builders differ only by their conference names.
    assert icml["build_stage_receipt"]["clone_of"] == {
        "orchestrator": "scripts/run_wombat2031_production.py",
        "unit": "build_stage_receipt",
        "similarity": 1.0,
    }
    assert cvpr["build_stage_receipt"]["clone_of"]["orchestrator"] == "scripts/run_quokka2031_production.py"
    clones = _orchestrator(report, "icml_shaped")["metrics"]["clone_lines_by_orchestrator"]
    assert clones["scripts/run_wombat2031_production.py"] >= icml["build_stage_receipt"]["lines"]


def test_token_regex_is_derived_from_runner_file_names() -> None:
    pattern = audit.conference_token_regex(["quokka2031", "wombat2031"])
    assert pattern is not None
    assert pattern.sub("CONF", "QUOKKA 2031 / wombat_2031 / quokka31 / Quokka") == "CONF / CONF / CONF / CONF"
    assert audit.runner_slug(Path("scripts/run_quokka2031_production.py")) == "quokka2031"
    assert audit.runner_slug(Path("scripts/build_workspace_notes.py")) is None
    assert audit.conference_token_regex([None]) is None


def test_explicit_orchestrator_and_missing_path_warning() -> None:
    result = audit.run_audit(
        repo_root=FIXTURE_REPO,
        orchestrators=[Path("scripts/run_newconf2032_production.py"), Path("scripts/missing.py")],
        include_units=False,
    )
    assert result["discovery"] == "explicit"
    assert [item["path"] for item in result["orchestrators"]] == ["scripts/run_newconf2032_production.py"]
    assert "units" not in result["orchestrators"][0]
    assert [warning["code"] for warning in result["warnings"]] == ["orchestrator_share_audit_path_missing"]
    assert result["status"] == "pass"


def test_cli_writes_json_and_markdown_and_always_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    json_out = tmp_path / "share.json"
    markdown_out = tmp_path / "share.md"

    code = audit.main(
        [
            "--repo-root",
            str(FIXTURE_REPO),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--orchestrator",
            "scripts/does_not_exist.py",
        ]
    )

    assert code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["orchestrators"] == []
    assert payload["warnings"]
    markdown = markdown_out.read_text(encoding="utf-8")
    assert markdown.startswith("# Orchestrator share audit (report-only)")
    assert capsys.readouterr().out.strip().startswith("# Orchestrator share audit")

    code = audit.main(["--repo-root", str(FIXTURE_REPO), "--format", "json", "--no-units"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert len(printed["orchestrators"]) == 3
    table = audit.render_markdown(printed)
    assert "| `scripts/run_newconf2032_production.py` |" in table
    assert "| download-corpus | shared+hooks |" in table


def test_audit_module_carries_no_conference_literals() -> None:
    source = AUDIT_MODULE.read_text(encoding="utf-8").lower()
    for token in ("neurips", "iclr", "icml", "cvpr"):
        assert re.search(rf"\b{token}", source) is None, token


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_orchestrator_is_audited(report: dict, name: str) -> None:
    item = _orchestrator(report, name)
    assert item["runner_slug"] == load_fixture_manifest(name)["slug"]
    assert (FIXTURE_ROOT / load_fixture_manifest(name)["orchestrator"]).is_file()
    hooks = sorted(hook["hook_id"] for hook in item["hooks"])
    assert hooks == load_fixture_manifest(name)["expected_hooks"]


DELEGATING_ORCHESTRATOR = '''"""Synthetic orchestrator that delegates CLI/dispatch to the shared orchestrator."""

from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production import release_stage as production_release_stage
from conference_pipeline.production import stage_table as production_stage_table
from conference_pipeline.production.hooks import ConferenceHooks, ExtraCommand, conference_hook

STAGE_NEXT_ACTIONS = {
    "download-corpus": "repair download",
    "release": "repair release",
}
STAGES = production_stage_table.from_registry(next_actions=STAGE_NEXT_ACTIONS)


def run_download_corpus_locally(workspace, run_id):
    total = 0
    for index in range(10):
        total += index
    return {"status": "succeeded", "total": total}


def dispatch_download_corpus(args, stage):
    return production_orchestrator.StageRun.from_outcome(run_download_corpus_locally(args.workspace, args.run_id))


def dispatch_release(args, stage):
    return production_orchestrator.StageRun.from_outcome(production_release_stage.run_release_stage(args))


def prepare_estimate(args):
    return {"estimate": 0}


@conference_hook("cli.extra_cli_args")
def add_cli_args(parser):
    parser.add_argument("--prepare-estimate", action="store_true")


def orchestrator_spec():
    return production_orchestrator.OrchestratorSpec(
        display_name="Delegate 2033",
        description="Run Delegate 2033 stages.",
        stages=STAGES,
        handlers={
            "download-corpus": dispatch_download_corpus,
            "release": dispatch_release,
        },
        hooks=ConferenceHooks(
            extra_cli_args=add_cli_args,
            extra_commands={
                "prepare_estimate": ExtraCommand(run=lambda args: prepare_estimate(args), failure_label="cannot prepare"),
            },
        ),
    )


def parse_args(argv=None):
    return production_orchestrator.parse_args(argv, orchestrator_spec())


def main(argv=None):
    return production_orchestrator.run(argv, orchestrator_spec(), args=parse_args(argv))
'''


def test_delegating_orchestrator_routes_stages_through_its_dispatch_table(tmp_path: Path) -> None:
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO / "src", repo / "src")
    production = repo / "src" / "conference_pipeline" / "production"
    for module in ("orchestrator", "stage_table"):
        (production / f"{module}.py").write_text(f'"""Synthetic {module}."""\n', encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_delegate2033_production.py").write_text(DELEGATING_ORCHESTRATOR, encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "stage_bindings": {
                    "download-corpus": {"stage_order": 4, "receipt_filename": "stage-04-download-corpus.json"},
                    "release": {"stage_order": 11, "receipt_filename": "stage-11-release.json"},
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit.run_audit(repo_root=repo, registry_path=registry)
    [item] = report["orchestrators"]
    stages = _stages(item)
    units = _units(item)

    # Stage ids come from the next-action table, orders from the registry.
    assert list(stages) == ["download-corpus", "release"]
    assert [stages[stage]["order"] for stage in stages] == [4, 11]
    assert all(stage["routed"] for stage in stages.values())
    # The dispatch adapter is looked through: the local entry is what it calls.
    assert stages["download-corpus"]["status"] == "local"
    assert stages["download-corpus"]["local_entry_units"] == ["run_download_corpus_locally"]
    assert units["run_download_corpus_locally"]["group"] == "download-corpus"
    assert stages["release"]["status"] == "shared"
    assert stages["release"]["shared_components"] == ["conference_pipeline.production.release_stage"]
    # The shared orchestrator modules route stages; they are not shared stages.
    for stage in stages.values():
        assert "conference_pipeline.production.orchestrator" not in stage["shared_components"]
    assert units["dispatch_release"]["classification"] == "adapter"
    # Extra commands come from the extra_commands table.
    assert item["extra_commands"] == ["prepare_estimate"]
    assert units["prepare_estimate"]["group"] == "cli-extra"
    # The spec is declarative: its tables do not make it a local unit.
    assert units["orchestrator_spec"]["classification"] == "adapter"
    assert units["parse_args"]["classification"] != "dead"
    assert [hook["hook_id"] for hook in item["hooks"]] == ["cli.extra_cli_args"]
