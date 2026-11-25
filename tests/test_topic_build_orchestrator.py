"""Tests for the per-figure acceptance cache and the topic build orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def figures_module():
    return _load_module("scripts/presentations/generate_programmatic_figures.py", "gpf_test")


@pytest.fixture(scope="module")
def orchestrator_module():
    return _load_module("scripts/presentations/run_topic_presentation_build.py", "orch_test")


_PASSING_CODE = '''import drawsvg as draw


def build_svg(output_path: str) -> None:
    d = draw.Drawing(1280, 720)
    d.append(draw.Rectangle(0, 0, 1280, 720, fill="#1a1a2e"))
    d.append(draw.Text("Hybrid Reinforcement", 26, 300, 200, fill="#a8d8ea"))
    d.append(draw.Text("Tao, Kulikov, Weston", 20, 300, 232, fill="#cccccc"))
    d.append(draw.Text("Adaptive compute", 26, 300, 420, fill="#e0e0e0"))
    d.append(draw.Text("Hybrid Reinforcement -> Adaptive compute", 22, 300, 560, fill="#51cf66"))
    d.save_svg(output_path)
'''


def _install_drawsvg_stub() -> None:
    if "drawsvg" in sys.modules:
        return

    class _Rectangle:
        def __init__(self, x, y, width, height, **attrs):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.attrs = attrs

        def to_svg(self) -> str:
            fill = self.attrs.get("fill", "none")
            return (
                f'<rect x="{self.x}" y="{self.y}" width="{self.width}" '
                f'height="{self.height}" fill="{fill}" />'
            )

    class _Text:
        def __init__(self, text, font_size, x, y, **attrs):
            self.text = text
            self.font_size = font_size
            self.x = x
            self.y = y
            self.attrs = attrs

        def to_svg(self) -> str:
            fill = self.attrs.get("fill", "#ffffff")
            return (
                f'<text x="{self.x}" y="{self.y}" font-size="{self.font_size}" '
                f'fill="{fill}">{self.text}</text>'
            )

    class _Drawing:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.elements = []

        def append(self, element):
            self.elements.append(element)

        def save_svg(self, output_path: str) -> None:
            body = "\n".join(element.to_svg() for element in self.elements)
            metadata = (
                "<defs><style>"
                "text{font-family:Inter,Arial,sans-serif;font-weight:600;}"
                "rect{shape-rendering:geometricPrecision;}"
                "</style></defs>\n"
                "<desc>Offline drawsvg test double for Phase 16 figure reuse. "
                "The generated SVG includes production-sized metadata so the "
                "execution guard exercises the same success path as a real "
                "drawsvg renderer without requiring the optional package.</desc>\n"
            )
            Path(output_path).write_text(
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{self.width}" height="{self.height}" '
                    f'viewBox="0 0 {self.width} {self.height}">\n'
                    f"{metadata}"
                    f"{body}\n</svg>\n"
                ),
                encoding="utf-8",
            )

    sys.modules["drawsvg"] = types.SimpleNamespace(
        Drawing=_Drawing,
        Rectangle=_Rectangle,
        Text=_Text,
    )


def _make_row(figures_module, tmp_path: Path) -> dict:
    asset_path = tmp_path / "fig.svg"
    code_path = tmp_path / "fig.py"
    return {
        "slide_id": "sl_x",
        "section_id": "S1",
        "cluster_ids": [],
        "visual_id": "sl_x-visual",
        "title": "Adaptive compute",
        "claim": "Adaptive compute beats uniform",
        "depiction_type": "mechanism",
        "visual_concept": "mechanism",
        "visual_role": "Left-to-right mechanism flow",
        "composition_budget": {
            "max_visible_text_labels": 26,
            "max_entity_nodes": 2,
            "max_relationship_edges": 1,
            "dominant_flow_direction": "left_to_right",
        },
        "proof_object": "benchmark deltas",
        "narration_hook": "hook",
        "contribution_note": "note",
        "source_labels": ["SRC-002", "SRC-004"],
        "source_ids": ["paper_abstract:abc", "paper_abstract:def"],
        "source_chip_bindings": ["Hybrid Reinforcement — Tao, Kulikov, Weston", "Strategic Scaling — Vu et al."],
        "reference_ids": [],
        "required_entities": ["Hybrid Reinforcement", "Adaptive compute"],
        "required_relationships": ["Hybrid Reinforcement -> Adaptive compute"],
        "forbidden_patterns": ["bullet-box"],
        "people": [],
        "systems": [],
        "datasets": [],
        "source_refs": [],
        "asset_path": asset_path,
        "code_path": code_path,
        "relative_asset": "assets/generated/phase16/fig.svg",
        "relative_code": "assets/generated/phase16/code/fig.py",
        "original_figure": {"visual_id": "sl_x-visual"},
    }


def _accept_and_cache(figures_module, row, tmp_path: Path) -> dict:
    _install_drawsvg_stub()
    row["code_path"].write_text(_PASSING_CODE, encoding="utf-8")
    error = figures_module._execute_phase16_generated_code(row["code_path"], row["asset_path"])
    assert not error, error
    prompt = figures_module._phase16_codegen_prompt(
        row=row, research_text="research", feedback=""
    )
    return {
        str(row["visual_id"]): {
            "prompt_sha256": figures_module._sha256_text(prompt),
            "response_sha256": "resp",
            "code_sha256": figures_module._sha256_file(row["code_path"]),
            "svg_sha256": figures_module._sha256_file(row["asset_path"]),
        }
    }


def test_reuse_candidate_accepts_unchanged_passing_figure(figures_module, tmp_path):
    row = _make_row(figures_module, tmp_path)
    cache = _accept_and_cache(figures_module, row, tmp_path)
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()
    result = figures_module._phase16_reuse_candidate(
        row=row, args=args, research_text="research", cache=cache
    )
    assert result is not None
    assert result["reused"] is True
    assert result["model_calls"] == []
    assert result["figure_spec"]["svg_sha256"] == cache["sl_x-visual"]["svg_sha256"]


def test_reuse_candidate_rejects_changed_prompt(figures_module, tmp_path):
    row = _make_row(figures_module, tmp_path)
    cache = _accept_and_cache(figures_module, row, tmp_path)
    row["required_relationships"] = ["Hybrid Reinforcement -> something new"]
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()
    assert (
        figures_module._phase16_reuse_candidate(
            row=row, args=args, research_text="research", cache=cache
        )
        is None
    )


def test_reuse_candidate_does_not_let_pixel_verdict_override_changed_prompt(
    figures_module, tmp_path
):
    row = _make_row(figures_module, tmp_path)
    cache = _accept_and_cache(figures_module, row, tmp_path)
    row["visual_concept"] = "new left-to-right flow with stricter connector lanes"
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()

    assert (
        figures_module._phase16_reuse_candidate(
            row=row,
            args=args,
            research_text="research",
            cache=cache,
            qc_verdicts={
                str(row["slide_id"]): {
                    "status": "pass",
                    "svg_sha256": cache["sl_x-visual"]["svg_sha256"],
                }
            },
        )
        is None
    )


def test_reuse_candidate_rejects_modified_svg(figures_module, tmp_path):
    row = _make_row(figures_module, tmp_path)
    cache = _accept_and_cache(figures_module, row, tmp_path)
    row["asset_path"].write_text(
        row["asset_path"].read_text(encoding="utf-8") + "<!-- tampered -->",
        encoding="utf-8",
    )
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()
    assert (
        figures_module._phase16_reuse_candidate(
            row=row, args=args, research_text="research", cache=cache
        )
        is None
    )


def test_reuse_candidate_disk_reuses_passing_artifacts_without_cache_entry(figures_module, tmp_path):
    # A receipted prior run left passing code+SVG on disk; acceptance is decided
    # by the same check suite, so the figure reuses even without a cache entry.
    row = _make_row(figures_module, tmp_path)
    _accept_and_cache(figures_module, row, tmp_path)
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()
    result = figures_module._phase16_reuse_candidate(
        row=row, args=args, research_text="research", cache={}
    )
    assert result is not None and result["reused"] is True


def test_reuse_candidate_rejects_missing_files(figures_module, tmp_path):
    row = _make_row(figures_module, tmp_path)
    args = type("Args", (), {"workspace": tmp_path, "generated_dir": tmp_path})()
    assert (
        figures_module._phase16_reuse_candidate(
            row=row, args=args, research_text="research", cache={}
        )
        is None
    )


def test_orchestrator_state_roundtrip_and_force_from(orchestrator_module, tmp_path):
    workspace = tmp_path / "conf"
    (workspace / "decks").mkdir(parents=True)
    args = orchestrator_module.parse_args(
        [
            "--workspace", str(workspace),
            "--deck-id", "99_demo",
            "--final-report-url", "https://example.com/report",
        ]
    )
    ctx = orchestrator_module.BuildContext(args)
    assert ctx.stage_status("blueprint") == "pending"
    ctx.mark("blueprint", "done")
    ctx.mark("deck_contract", "done")

    # A fresh context reads the persisted state back.
    ctx2 = orchestrator_module.BuildContext(args)
    assert ctx2.stage_status("blueprint") == "done"
    assert ctx2.stage_status("deck_contract") == "done"

    # --force-from rewinds the named stage and everything after it.
    rc = orchestrator_module.main(
        [
            "--workspace", str(workspace),
            "--deck-id", "99_demo",
            "--final-report-url", "https://example.com/report",
            "--force-from", "blueprint",
            "--dry-run",
        ]
    )
    assert rc == 0
    ctx3 = orchestrator_module.BuildContext(args)
    assert ctx3.stage_status("blueprint") == "stale"
    assert ctx3.stage_status("deck_contract") == "stale"


def test_orchestrator_dry_run_lists_all_stages(orchestrator_module, tmp_path, capsys):
    workspace = tmp_path / "conf"
    (workspace / "decks").mkdir(parents=True)
    rc = orchestrator_module.main(
        [
            "--workspace", str(workspace),
            "--deck-id", "99_demo",
            "--final-report-url", "https://example.com/report",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    for stage, _ in orchestrator_module.STAGES:
        if stage == "produced_review":
            assert stage not in out  # opt-in only
        else:
            assert stage in out


def test_orchestrator_rejects_legacy_blueprint_mode(orchestrator_module, tmp_path):
    with pytest.raises(SystemExit):
        orchestrator_module.parse_args(
            [
                "--workspace", str(tmp_path),
                "--deck-id", "99_demo",
                "--final-report-url", "https://example.com/report",
                "--blueprint-mode", "source-ledger",
            ]
        )


def test_blueprint_stage_requires_presentation_frame(orchestrator_module, tmp_path):
    deck_dir = tmp_path / "decks" / "99_demo"
    deck_dir.mkdir(parents=True)

    class FakeContext:
        deck_id = "99_demo"
        topic_title = "Demo Topic"

        class args:
            blueprint_mode = "editorial"
            final_report_url = "https://example.com/report"

        def ids(self):
            return []

        def rel(self, name: str) -> str:
            return f"decks/99_demo/{name}"

        def run(self, stage: str, cmd: list[str]) -> None:
            raise AssertionError("blueprint command should not run without a frame")

    fake_context = FakeContext()
    fake_context.deck_dir = deck_dir
    with pytest.raises(FileNotFoundError, match="presentation_frame.json not found"):
        orchestrator_module.stage_blueprint(fake_context)


def test_slides_draft_uses_blueprint_audience_title(orchestrator_module, tmp_path):
    deck_dir = tmp_path / "decks" / "05_generative_models"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation_blueprint.json").write_text(
        json.dumps({"audience_title": "Generative AI after sample quality"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    class FakeContext:
        def __init__(self) -> None:
            self.deck_dir = deck_dir
            self.deck_rel = "decks/05_generative_models"
            self.topic_title = "Generative modeling: diffusion, flow, and controllability"

        def ids(self):
            return [
                "--workspace",
                str(tmp_path),
                "--presentation-id",
                "05_generative_models",
                "--presentation-type",
                "topic-deep-dive",
                "--topic",
                "05_generative_models",
            ]

        def rel(self, name: str) -> str:
            return f"{self.deck_rel}/{name}"

        def run(self, stage: str, cmd: list[str]) -> None:
            captured[stage] = cmd
            (self.deck_dir / "slides.md").write_text("# draft\n", encoding="utf-8")

    orchestrator_module.stage_slides_draft(FakeContext())

    cmd = captured["slides_draft"]
    assert cmd[cmd.index("--display-title") + 1] == "Generative AI after sample quality"


def test_sync_plan_prefers_frame_contract_family_over_payload_glob(tmp_path):
    sync_module = _load_module("scripts/sync_to_gdrive.py", "sync_test")
    workspace = tmp_path / "conferences" / "demo2027"
    deck_dir = workspace / "decks" / "05_generative_models"
    deck_dir.mkdir(parents=True)
    (workspace / "storage").mkdir()
    (deck_dir / "presentation_frame.json").write_text("{}", encoding="utf-8")
    (deck_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
    (workspace / "config.yaml").write_text(
        "storage:\n  manifest_path: storage/storage_manifest.json\n",
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        json.dumps(
            {
                "artifact_families": [
                    {
                        "id": "presentation_frame_contracts",
                        "path_glob": "decks/**/presentation_frame.json",
                        "storage_class": "git",
                        "sync_profiles": {"public": True, "backup": True},
                    },
                    {
                        "id": "presentation_payloads",
                        "path_glob": "decks/**/*.{html,json}",
                        "storage_class": "public",
                        "sync_profiles": {"public": True, "backup": True},
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = sync_module.SyncConfig(
        conference_dir=workspace,
        config_path=workspace / "config.yaml",
        profile="backup",
        remote_path=tmp_path / "drive" / "Demo 2027-raw",
        dry_run=True,
        delete=False,
        quiet=True,
        reviewed_dry_run=False,
        init_only=False,
        reviewed_dry_run_receipt=None,
    )

    plan = sync_module.build_sync_plan(config)
    frame_items = [
        item
        for item in plan.sync_items
        if item.relative_path.as_posix() == "decks/05_generative_models/presentation_frame.json"
    ]

    assert [item.family_id for item in frame_items] == ["presentation_frame_contracts"]
    assert "presentation_payloads" in plan.included_families


def test_orchestrator_stage_order_matches_runbook(orchestrator_module):
    names = [name for name, _ in orchestrator_module.STAGES]
    # contract-critical orderings proven on the Topic-1 build
    assert names.index("blueprint_review") < names.index("deck_contract")
    assert names.index("source_binding") < names.index("figures_pass1")
    assert names.index("figures_pass2") < names.index("validate_svgs")
    assert names.index("delivery_bootstrap") < names.index("render_spec")
    assert names.index("render_spec") < names.index("bookends")
    assert names.index("bookends") < names.index("render_html")
    assert names.index("presenter_script") < names.index("engaging_narration")
    assert names.index("reconcile_hashes") < names.index("narration_gates")
    assert names.index("narration_gates") < names.index("build_videos")
    assert names.index("build_videos") < names.index("video_sync")


def test_default_args_enable_parallelism(orchestrator_module):
    args = orchestrator_module.parse_args(
        ["--workspace", "ws", "--deck-id", "07_x", "--final-report-url", "https://e/r"]
    )
    assert args.figure_workers == 4  # figures parallel by default
    assert args.sequential_videos is False  # videos concurrent by default


def test_run_parallel_runs_concurrently_and_propagates_failure(orchestrator_module, tmp_path):
    workspace = tmp_path / "conf"
    (workspace / "decks").mkdir(parents=True)
    args = orchestrator_module.parse_args(
        ["--workspace", str(workspace), "--deck-id", "07_x", "--final-report-url", "https://e/r"]
    )
    ctx = orchestrator_module.BuildContext(args)

    # Two sleepers that each take ~0.5s: concurrent wall-time must be < their sum.
    import sys as _sys
    import time

    sleeper = ["-c", "import time; time.sleep(0.5)"]
    start = time.monotonic()
    ctx.run_parallel("twin", [("a", [_sys.executable, *sleeper]), ("b", [_sys.executable, *sleeper])])
    elapsed = time.monotonic() - start
    assert elapsed < 0.9  # ran in parallel, not 1.0s+ serial

    # A failing command surfaces as a stage error.
    with pytest.raises(RuntimeError, match="failed command"):
        ctx.run_parallel(
            "twin",
            [
                ("ok", [_sys.executable, "-c", "pass"]),
                ("bad", [_sys.executable, "-c", "import sys; sys.exit(3)"]),
            ],
        )
