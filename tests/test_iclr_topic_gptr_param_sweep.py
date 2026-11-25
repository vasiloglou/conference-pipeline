from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_iclr_topic_gptr_param_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("topic2_gptr_param_sweep", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _controls(sweep, **overrides):
    values = {
        "topic": "02_model_architectures",
        "source_run_id": "v151-iclr2025-topic2-local-analysis",
        "vector_store_run_id": "v151-iclr2025-topic2-local-analysis",
        "embedding_provider": "mlx",
        "embedding_model": "mlx-community/bge-m3-mlx-8bit",
        "embedding_dim": None,
        "llm_runtime_kind": "mlx",
        "llm_model": "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
        "fast_model": "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
        "local_base_url": "http://127.0.0.1:8081/v1",
        "temperature": 0.0,
        "server_max_tokens": 8192,
        "web_search_enabled": False,
        "total_words": 3000,
        "retrieval_k": 18,
        "candidate_multiplier": 3,
        "rrf_k": 60,
        "bm25_weight": None,
        "vector_weight": None,
        "paper_matching_retrieval_k": 18,
    }
    values.update(overrides)
    return sweep.SweepControls(**values)


def test_default_matrix_matches_experiment_plan() -> None:
    sweep = _load_module()

    observed = [
        (
            variant.variant_id,
            variant.max_iterations,
            variant.breadth,
            variant.depth,
            variant.max_subtopics,
        )
        for variant in sweep.default_variants()
    ]

    assert observed == [
        ("i3-b1-d1-s3", 3, 1, 1, 3),
        ("i5-b1-d1-s3", 5, 1, 1, 3),
        ("i3-b3-d1-s3", 3, 3, 1, 3),
        ("i3-b1-d2-s3", 3, 1, 2, 3),
        ("i3-b1-d1-s6", 3, 1, 1, 6),
        ("i5-b3-d2-s6", 5, 3, 2, 6),
        ("i5-b4-d3-s8", 5, 4, 3, 8),
    ]


def test_config_generation_uses_uppercase_gpt_researcher_keys() -> None:
    sweep = _load_module()
    variant = sweep.default_variants()[0]
    controls = _controls(sweep)

    config = sweep.config_for_variant(variant, controls)

    assert config["MAX_ITERATIONS"] == 3
    assert config["DEEP_RESEARCH_BREADTH"] == 1
    assert config["DEEP_RESEARCH_DEPTH"] == 1
    assert config["DEEP_RESEARCH_CONCURRENCY"] == 2
    assert config["MAX_SUBTOPICS"] == 3
    assert config["TOTAL_WORDS"] == 3000
    assert config["SERVER_MAX_TOKENS"] == 8192
    assert config["SMART_TOKEN_LIMIT"] == 8192
    assert config["STRATEGIC_TOKEN_LIMIT"] == 8192
    assert config["FAST_TOKEN_LIMIT"] == 4096
    assert config["TEMPERATURE"] == 0.0
    assert config["RETRIEVAL_K"] == 18
    assert config["CANDIDATE_MULTIPLIER"] == 3
    assert config["RRF_K"] == 60
    assert config["MAX_SEARCH_RESULTS_PER_QUERY"] == 18
    assert config["PAPER_MATCHING_RETRIEVAL_K"] == 18
    assert config["WEB_SEARCH_ENABLED"] is False
    assert config["LOCAL_LLM_BASE_URL"] == "http://127.0.0.1:8081/v1"
    assert config["vector_store_run_id"] == "v151-iclr2025-topic2-local-analysis"


def test_local_endpoint_label_uses_runtime_kind() -> None:
    sweep = _load_module()

    assert (
        sweep.local_llm_endpoint_label(_controls(sweep, llm_runtime_kind="mlx"))
        == "mlx://mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    )
    assert (
        sweep.local_llm_endpoint_label(_controls(sweep, llm_runtime_kind="ollama"))
        == "ollama://mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    )

    with pytest.raises(sweep.SweepError, match="llm_runtime_kind"):
        sweep.local_llm_endpoint_label(_controls(sweep, llm_runtime_kind="remote"))


def test_local_only_guard_rejects_non_loopback_base_url() -> None:
    sweep = _load_module()

    with pytest.raises(sweep.SweepError, match="non-loopback"):
        sweep.ensure_loopback_base_url("https://api.openai.com/v1")

    assert (
        sweep.ensure_loopback_base_url("http://localhost:8081/v1")
        == "http://localhost:8081/v1"
    )


def test_build_controls_rejects_unknown_runtime_kind(tmp_path: Path) -> None:
    sweep = _load_module()
    args = sweep.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--llm-runtime-kind",
            "remote",
        ]
    )

    with pytest.raises(sweep.SweepError, match="llm_runtime_kind"):
        sweep.build_controls(args)


def test_output_path_guard_stays_under_experiment_root(tmp_path: Path) -> None:
    sweep = _load_module()
    root = tmp_path / "storage" / "experiments" / "topic2-gptr-param-sweep"
    root.mkdir(parents=True)

    safe = sweep.safe_experiment_child(root, "variants", "i3-b1-d1-s3")
    assert safe == (root / "variants" / "i3-b1-d1-s3").resolve()

    with pytest.raises(sweep.SweepError, match="outside experiment root"):
        sweep.ensure_under_experiment_root(root.parent / "escape", root)


def test_live_variant_fails_closed_without_bound_receipts(tmp_path: Path) -> None:
    sweep = _load_module()
    workspace = tmp_path / "iclr2025"
    root = workspace / "storage" / "experiments" / "topic2-gptr-param-sweep"

    with pytest.raises(sweep.SweepError, match="missing required exhaustive"):
        sweep.run_variant(
            workspace_root=workspace,
            root=root,
            controls=_controls(sweep),
            variant=sweep.default_variants()[0],
            skip_paper_matching=True,
            max_evidence_chunks=1,
        )


def test_dry_run_creates_configs_without_touching_production_analysis(tmp_path: Path) -> None:
    sweep = _load_module()
    workspace = tmp_path / "iclr2025"
    production_report = workspace / "analysis" / "02_model_architectures.md"
    before_local = (
        workspace
        / "storage"
        / "experiments"
        / "topic2-local-analysis"
        / "02_model_architectures.before-local.md"
    )
    taxonomy = workspace / "analysis" / "iclr2025_taxonomy.json"
    production_report.parent.mkdir(parents=True, exist_ok=True)
    production_report.write_text("# Production Report\n\nDo not touch.\n", encoding="utf-8")
    before_local.parent.mkdir(parents=True, exist_ok=True)
    before_local.write_text("# Before Local\n\nBaseline.\n", encoding="utf-8")
    _write_json(
        taxonomy,
        {
            "topic_families": [
                {
                    "id": "02_model_architectures",
                    "title": "Model Architectures",
                    "source_families": ["accepted-papers"],
                }
            ]
        },
    )
    original_report = production_report.read_text(encoding="utf-8")

    exit_code = sweep.main(
        [
            "--workspace",
            str(workspace),
            "--source-run-id",
            "v151-iclr2025-topic2-local-analysis",
            "--topic",
            "02_model_architectures",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert production_report.read_text(encoding="utf-8") == original_report

    root = workspace / "storage" / "experiments" / "topic2-gptr-param-sweep"
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert len(plan["variants"]) == 7

    for variant in sweep.default_variants():
        config_path = root / "variants" / variant.variant_id / "config.json"
        dry_run_path = root / "variants" / variant.variant_id / "dry_run.json"
        assert config_path.exists()
        assert dry_run_path.exists()
        dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
        assert dry_run["provider_calls"] == 0
        assert dry_run["writes_production_analysis"] is False

    assert (root / "comparison.md").exists()
    assert (root / "comparison.json").exists()
