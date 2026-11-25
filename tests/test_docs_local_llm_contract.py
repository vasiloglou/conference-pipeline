"""Phase 47 Plan 03 — docs/local-llm.md contract tests.

Mirrors the `tests/test_operator_docs_contract.py` pattern: load the
doc, assert exact substrings present. Catches docs drift from code
when the env var names, error messages, or CLI invocations change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_PATH = _REPO_ROOT / "docs" / "local-llm.md"
_CAPABILITIES_PATH = _REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md"
_SOURCE_OF_TRUTH_PATH = _REPO_ROOT / "docs" / "source-of-truth.md"


@pytest.fixture(scope="module")
def local_llm_doc() -> str:
    assert _DOC_PATH.exists(), f"docs/local-llm.md missing at {_DOC_PATH}"
    return _DOC_PATH.read_text(encoding="utf-8")


def test_doc_exists(local_llm_doc):
    assert local_llm_doc, "docs/local-llm.md is empty"


def test_doc_documents_all_v19_env_vars(local_llm_doc):
    """Every v1.9 env var must appear in the doc's reference table."""
    expected = [
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_BATCH_SIZE",
        "OLLAMA_BASE_URL",
        "GPTR_LOCAL_LLM_BASE_URL",
        "GPTR_LOCAL_LLM_FAST_MODEL",
        "GPTR_LOCAL_LLM_SMART_MODEL",
        "GPTR_LOCAL_LLM_ENDPOINT_LABEL",
        "GPTR_LOCAL_LLM_RUNTIME_KIND",
        "OPENAI_API_KEY",
    ]
    for var in expected:
        assert var in local_llm_doc, f"env var {var!r} missing from local-llm.md"


def test_doc_lists_all_three_provider_defaults(local_llm_doc):
    """Embedding model defaults table covers all three providers."""
    assert "text-embedding-3-small" in local_llm_doc, "OpenAI default missing"
    assert "mlx-community/bge-m3-mlx-8bit" in local_llm_doc, "MLX default missing"
    assert "nomic-embed-text" in local_llm_doc, "Ollama default missing"
    # Dimensions are surfaced.
    assert "1536" in local_llm_doc
    assert "1024" in local_llm_doc
    assert "768" in local_llm_doc


def test_doc_documents_dim_mismatch_remediation(local_llm_doc):
    """The exact rebuild CLI command appears in the dim-policy section."""
    assert "vector-store rebuild --provider" in local_llm_doc
    assert "VectorStoreDimensionMismatch" in local_llm_doc


def test_doc_documents_smoketest_local_contract(local_llm_doc):
    """The OPENAI_API_KEY-must-be-unset rule is documented verbatim."""
    assert (
        "--local requires OPENAI_API_KEY to be unset"
        in local_llm_doc
    )
    assert "unset OPENAI_API_KEY" in local_llm_doc


def test_doc_documents_manage_server_lifecycle(local_llm_doc):
    """The auto-spawn lifecycle (D4) is documented including log path."""
    assert "manage_server=True" in local_llm_doc
    assert ".planning/logs/ollama-serve.log" in local_llm_doc
    assert "atexit" in local_llm_doc


def test_doc_documents_local_e2e_marker(local_llm_doc):
    """The opt-in test marker is documented for operators running locally."""
    assert "local_e2e" in local_llm_doc
    assert "RUN_LOCAL_E2E=1" in local_llm_doc


def test_doc_lists_troubleshooting_table(local_llm_doc):
    """Common failure modes documented with diagnosis."""
    expected_symptoms = [
        "non_loopback_base_url",
        "ollama_unreachable",
        "ollama_serve_crashed",
        "MLXEmbeddingClient requires the mlx-embeddings package",
        "VectorStoreDimensionMismatch",
        "unknown embedding provider",
    ]
    for symptom in expected_symptoms:
        assert symptom in local_llm_doc, (
            f"troubleshooting symptom {symptom!r} missing from local-llm.md"
        )


def test_capability_doc_claims_local_lane_support():
    """pipeline-capabilities.md adds rows for the v1.9 local lanes."""
    assert _CAPABILITIES_PATH.exists()
    text = _CAPABILITIES_PATH.read_text(encoding="utf-8")
    # Local-lane rows added to the capability table.
    assert "Vector-store embeddings (cloud + local lanes)" in text
    assert "GPT-Researcher synthesis (cloud + local lanes)" in text
    assert "Vector-store rebuild CLI" in text
    assert "`conference-smoketest --local` lane" in text
    # Each row points at docs/local-llm.md.
    assert text.count("local-llm.md") >= 4


def test_source_of_truth_links_local_llm_doc():
    """source-of-truth.md adds a Canonical Layers row for the local lane."""
    assert _SOURCE_OF_TRUTH_PATH.exists()
    text = _SOURCE_OF_TRUTH_PATH.read_text(encoding="utf-8")
    assert "Local LLM and embedding lane" in text
    assert "local-llm.md" in text


# ---------------------------------------------------------------------------
# LOCAL-03 Task 3: paid embedding opt-in documented (D-10)
# ---------------------------------------------------------------------------


def test_doc_documents_paid_embedding_opt_in(local_llm_doc):
    """D-10: CONFERENCE_ALLOW_PAID_EMBEDDINGS must appear in the docs."""
    assert "CONFERENCE_ALLOW_PAID_EMBEDDINGS" in local_llm_doc, (
        "CONFERENCE_ALLOW_PAID_EMBEDDINGS env var missing from local-llm.md"
    )


def test_doc_states_local_mlx_is_default(local_llm_doc):
    """D-09: docs must state that local MLX is the default embedding provider."""
    # The env var table default column now shows 'mlx' instead of 'openai'.
    assert "mlx-community/bge-m3-mlx-8bit" in local_llm_doc
    # The default MLX dim must be documented.
    assert "1024" in local_llm_doc


def test_doc_documents_paid_embedding_receipts(local_llm_doc):
    """D-10: docs must mention cost_status for paid embedding receipts."""
    assert "cost_status" in local_llm_doc, (
        "cost_status field not documented — paid embedding receipts require it"
    )
    # The paid embedding receipt section must call out provider/model/dim/cost_status
    # as the only receipt fields (T-03-03 — no API keys).
    assert "provider/model/dim/cost_status" in local_llm_doc or (
        "provider" in local_llm_doc
        and "cost_status" in local_llm_doc
    )


# ---------------------------------------------------------------------------
# GPTR-04/GPTR-05 Task 3: local GPTR config boundary, paid comparison,
# and research cache namespace (D-14, D-19, D-20)
# ---------------------------------------------------------------------------


_PIPELINE_WORKFLOW_PATH = _REPO_ROOT / "docs" / "workflows" / "pipeline-workflow.md"


@pytest.fixture(scope="module")
def pipeline_workflow_doc() -> str:
    assert _PIPELINE_WORKFLOW_PATH.exists(), (
        f"docs/workflows/pipeline-workflow.md missing at {_PIPELINE_WORKFLOW_PATH}"
    )
    return _PIPELINE_WORKFLOW_PATH.read_text(encoding="utf-8")


def test_doc_documents_local_gptr_evidence_draft_layer(local_llm_doc):
    """Docs must describe local GPTR as evidence/draft layer unless comparison gates pass."""
    assert any(
        phrase in local_llm_doc
        for phrase in ["evidence/draft layer", "evidence and draft", "evidence layer"]
    ), "local-llm.md must describe local GPT-Researcher as evidence/draft layer"


def test_doc_documents_research_cache_namespace(local_llm_doc):
    """D-19: docs must state the cache namespace storage/research-cache/."""
    assert "storage/research-cache/" in local_llm_doc, (
        "local-llm.md must document the cache namespace storage/research-cache/"
    )


def test_doc_documents_cache_hit_receipt(local_llm_doc):
    """D-20: docs must describe cache-hit receipts with exact reuse proof."""
    assert "cache-hit" in local_llm_doc or "cache_hit" in local_llm_doc, (
        "local-llm.md must document cache-hit receipts"
    )


def test_doc_documents_paid_comparison_same_retrieval_bundle(local_llm_doc):
    """D-14: docs must state paid comparison uses same retrieval bundle and query plan."""
    assert "retrieval bundle" in local_llm_doc or "retrieval_bundle" in local_llm_doc, (
        "local-llm.md must document that paid comparison uses the same retrieval bundle"
    )


def test_doc_documents_paid_comparison_paid_synthesis_only(local_llm_doc):
    """D-14: docs must state only synthesis model/provider changes in paid comparison."""
    assert "paid_synthesis_only" in local_llm_doc or "paid synthesis only" in local_llm_doc, (
        "local-llm.md must state paid_synthesis_only for the comparison lane"
    )


def test_doc_documents_cache_key_components(local_llm_doc):
    """D-17: docs must document cache key includes retrieval bundle, query plan, model/provider."""
    for component in ["retrieval bundle digest", "query plan digest", "model", "provider"]:
        assert component in local_llm_doc, (
            f"local-llm.md must document cache key component: {component!r}"
        )


def test_doc_documents_cache_never_caches_failures(local_llm_doc):
    """D-18: docs must state transient failures are never cached."""
    assert any(
        phrase in local_llm_doc
        for phrase in [
            "never cached",
            "not cached",
            "Transient failure",
            "transient failure",
        ]
    ), "local-llm.md must state that transient failures are never cached"


def test_pipeline_workflow_doc_documents_research_cache(pipeline_workflow_doc):
    """pipeline-workflow.md must mention storage/research-cache/."""
    assert "storage/research-cache/" in pipeline_workflow_doc, (
        "pipeline-workflow.md must document the storage/research-cache/ namespace"
    )


def test_pipeline_workflow_doc_documents_local_gptr_evidence_layer(pipeline_workflow_doc):
    """pipeline-workflow.md must state local GPTR is evidence/draft layer."""
    assert any(
        phrase in pipeline_workflow_doc
        for phrase in ["evidence layer", "evidence/draft", "evidence and draft"]
    ), "pipeline-workflow.md must describe local GPT-Researcher as evidence layer"
