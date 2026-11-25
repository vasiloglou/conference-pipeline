"""Wiring tests for ICML2026's build_analysis_source_packs delegation.

Phase 10.1.1-04 (PROD-15, Finding D on the sibling runners): ICML2026's
``build_analysis_source_packs`` previously ran its own single concatenated
per-topic bucket scan over the papers/artifacts vector-store chunk sidecars,
with no per-source de-duplication and no family (main_conference/workshop)
stratification at all -- strictly older than even CVPR2026's pre-fix
per-store scan. This module proves the runner now delegates chunk SELECTION
to the shared, order-independent, family-stratified
``select_topic_source_chunks`` (Phase 10.1.1-02) without importing any other
conference's serialization.

There is no ``tests/test_icml2026_production_runner.py`` in this repository
(ICML2026's milestone is currently reopened for a separate recovery
workstream); this module is deliberately narrow -- it covers only the
source-pack selection wiring, not a full runner test suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_icml2026_production.py"


def load_runner_module():
    """Load run_icml2026_production.py by file path (mirrors the sibling
    runner test modules' load_runner_module pattern)."""

    spec = importlib.util.spec_from_file_location("icml2026_production_runner_test", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_module_exposes_shared_source_pack_selector() -> None:
    """Import-surface proof: the runner imports select_topic_source_chunks
    from the shared platform module rather than carrying its own scan."""

    runner = load_runner_module()
    assert hasattr(runner, "select_topic_source_chunks"), (
        "ICML2026 runner must delegate to the shared select_topic_source_chunks"
    )
    assert (
        runner.select_topic_source_chunks.__module__
        == "conference_pipeline.analysis.source_pack_selection"
    )


def test_analysis_source_packs_order_independence_main_conference_representation(
    tmp_path: Path,
) -> None:
    """Regression: the papers store's file order must not determine whether
    main-conference chunks are represented in a topic's source pack.

    Workshop-scoped chunks are written FIRST and alone supply enough distinct
    artifact_ids to exhaust ANALYSIS_SOURCE_PACK_MAX_CHUNKS, so this test
    cannot pass by the main-conference chunks merely happening to fit before
    the cap fills."""

    runner = load_runner_module()
    workspace = tmp_path / "icml2026"
    run_id = "test-analysis-source-pack-order-independence"
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    paper_store.mkdir(parents=True)

    workshop_chunks = [
        {
            "chunk_id": f"paper:workshop-{index}:0",
            "artifact_id": f"workshop_paper_abstract:workshop-{index}",
            "source_path": f"https://icml.cc/virtual/2026/workshop-paper-{index}",
            "text": f"Efficiency workshop evidence, distinct source {index}.",
            "conference_scope": "workshop",
        }
        for index in range(8)
    ]
    main_conference_chunks = [
        {
            "chunk_id": f"paper:main-{index}:0",
            "artifact_id": f"paper_abstract:main-{index}",
            "source_path": f"https://icml.cc/virtual/2026/paper-{index}",
            "text": f"Efficiency main-conference evidence, distinct source {index}.",
            "conference_scope": "main_conference",
        }
        for index in range(5)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in workshop_chunks + main_conference_chunks:
            handle.write(json.dumps(chunk) + "\n")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.workspace_relpath(workspace, paper_store / "chunks.jsonl")
            },
        }
    }
    (vector_root / "index_digest.json").write_text(
        json.dumps(index_digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    topic_specs = (
        {
            "id": "01_efficiency_systems",
            "title": "Efficiency and scalable systems",
            "keywords": ("efficiency",),
        },
    )

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_efficiency_systems.json")
        .read_text(encoding="utf-8")
    )
    chunk_ids = [chunk["chunk_id"] for chunk in source_pack["chunks"]]
    main_ids = [cid for cid in chunk_ids if cid.startswith("paper:main-")]
    workshop_ids = [cid for cid in chunk_ids if cid.startswith("paper:workshop-")]

    assert len(main_ids) >= 5, (
        f"expected at least 5 main-conference chunks despite workshop chunks "
        f"occupying every earlier file position, got {chunk_ids}"
    )
    assert workshop_ids, "workshop chunks should still be represented, not excluded entirely"
    assert len(chunk_ids) == runner.ANALYSIS_SOURCE_PACK_MAX_CHUNKS
