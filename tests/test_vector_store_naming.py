"""Phase 45 Plan 01 — vector_store_naming module unit tests.

Covers v1.9 requirements COMPAT-01 (suffixed collection name) and
COMPAT-02 (dim-mismatch fail-stop with rebuild remediation message).
GOVERNANCE-01 / 02 are covered by the broader integration tests in
test_enhanced_indexing_vector_store.py once Plan 45-02 lands.
"""

from __future__ import annotations

import pytest

from conference_pipeline.indexing.vector_store_naming import (
    VectorStoreDimensionMismatch,
    _slug,
    derive_collection_name,
    guard_dimension,
)


# ---------------------------------------------------------------------------
# derive_collection_name — happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,model,dim,expected",
    [
        (
            "openai",
            "text-embedding-3-small",
            1536,
            "neurips_presentations__openai_text-embedding-3-small_1536",
        ),
        (
            "mlx",
            "Qwen/Qwen3-Embedding-0.6B",
            1024,
            "neurips_presentations__mlx_qwen-qwen3-embedding-0-6b_1024",
        ),
        (
            "ollama",
            "nomic-embed-text",
            768,
            "neurips_presentations__ollama_nomic-embed-text_768",
        ),
        (
            "fake",
            "text-embedding-3-small",
            8,
            "neurips_presentations__fake_text-embedding-3-small_8",
        ),
    ],
)
def test_derive_collection_name_known_providers(provider, model, dim, expected):
    """The three v1.9 providers + fake produce stable, predictable names."""
    actual = derive_collection_name(
        "neurips_presentations", provider, model, dim
    )
    assert actual == expected
    assert 3 <= len(actual) <= 63


def test_derive_collection_name_normalizes_special_chars():
    """``/``, ``.``, uppercase all collapse to lowercase + hyphens."""
    name = derive_collection_name(
        "base", "mlx", "Org.Sub/Model.v1.5-Beta", 1024
    )
    assert "/" not in name
    assert "." not in name
    assert name.lower() == name


def test_derive_collection_name_truncates_long_model_names():
    """Long model names are slug-truncated; prefix + dim preserved verbatim."""
    long_model = "a" * 80
    name = derive_collection_name("neurips_presentations", "mlx", long_model, 1024)
    assert len(name) <= 63
    assert name.startswith("neurips_presentations__mlx_")
    assert name.endswith("_1024")


def test_derive_collection_name_raises_on_zero_dim():
    with pytest.raises(ValueError, match="must be positive int"):
        derive_collection_name("base", "fake", "model", 0)


def test_derive_collection_name_raises_on_negative_dim():
    with pytest.raises(ValueError, match="must be positive int"):
        derive_collection_name("base", "fake", "model", -1)


def test_derive_collection_name_raises_on_empty_base():
    with pytest.raises(ValueError, match="'base' must be non-empty"):
        derive_collection_name("", "fake", "model", 1536)


def test_derive_collection_name_raises_on_empty_provider():
    with pytest.raises(ValueError, match="'provider' must be non-empty"):
        derive_collection_name("base", "", "model", 1536)


# ---------------------------------------------------------------------------
# _slug — sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("simple", "simple"),
        ("UPPERCASE", "uppercase"),
        ("with spaces", "with-spaces"),
        ("Org/Model.v2-beta", "org-model-v2-beta"),
        ("---leading-trailing---", "leading-trailing"),
        ("a!@#$%^&*()b", "a-b"),
    ],
)
def test_slug_lowercases_and_collapses(model, expected):
    assert _slug(model) == expected


def test_slug_truncates_to_max_len():
    s = _slug("a" * 100, max_len=20)
    assert len(s) == 20


def test_slug_strips_trailing_hyphen_after_truncation():
    """If truncation cuts mid-special-char run, the trailing hyphen is stripped."""
    s = _slug("abc!@#xyz", max_len=4)  # would be "abc-" pre-strip
    assert not s.endswith("-")


def test_slug_raises_on_non_str():
    with pytest.raises(ValueError, match="requires a str"):
        _slug(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# guard_dimension — fail-stop
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, *, name: str, metadata: dict | None) -> None:
        self.name = name
        self.metadata = metadata


def test_guard_dimension_passes_when_metadata_absent():
    """Legacy v1.0–v1.8 collections (no metadata.dim) pass through unchanged."""
    collection = _FakeCollection(name="legacy_neurips_presentations", metadata=None)
    # Should not raise.
    guard_dimension(
        collection,
        active_provider="mlx",
        active_model="Qwen/Qwen3-Embedding-0.6B",
        active_dim=1024,
    )


def test_guard_dimension_passes_when_metadata_lacks_dim_key():
    collection = _FakeCollection(
        name="x", metadata={"provider": "openai", "model": "old"}
    )
    guard_dimension(
        collection,
        active_provider="mlx",
        active_model="Qwen/Qwen3-Embedding-0.6B",
        active_dim=1024,
    )


def test_guard_dimension_passes_when_dims_match():
    collection = _FakeCollection(
        name="neurips_presentations__mlx_qwen3-embedding-0-6b_1024",
        metadata={"dim": 1024},
    )
    guard_dimension(
        collection,
        active_provider="mlx",
        active_model="Qwen/Qwen3-Embedding-0.6B",
        active_dim=1024,
    )


def test_guard_dimension_raises_with_remediation_message():
    """Mismatch fails fast with a message that names the rebuild CLI."""
    collection = _FakeCollection(
        name="neurips_presentations__openai_text-embedding-3-small_1536",
        metadata={"dim": 1536},
    )
    with pytest.raises(VectorStoreDimensionMismatch) as exc_info:
        guard_dimension(
            collection,
            active_provider="mlx",
            active_model="Qwen/Qwen3-Embedding-0.6B",
            active_dim=1024,
        )
    msg = str(exc_info.value)
    # Remediation: the exact CLI invocation.
    assert "vector-store rebuild --provider mlx" in msg
    # Both dims explicit.
    assert "dim 1536" in msg
    assert "dim 1024" in msg
    # Active model identified.
    assert "Qwen/Qwen3-Embedding-0.6B" in msg


def test_dimension_mismatch_carries_attributes():
    """Caller can inspect the exception to render its own remediation UI."""
    collection = _FakeCollection(
        name="some_collection", metadata={"dim": 1536}
    )
    with pytest.raises(VectorStoreDimensionMismatch) as exc_info:
        guard_dimension(
            collection,
            active_provider="ollama",
            active_model="nomic-embed-text",
            active_dim=768,
        )
    exc = exc_info.value
    assert exc.collection == "some_collection"
    assert exc.persisted_dim == 1536
    assert exc.active_dim == 768
    assert exc.active_provider == "ollama"
    assert exc.active_model == "nomic-embed-text"


# ---------------------------------------------------------------------------
# VectorStoreInputs dataclass extension (Plan 45-01)
# ---------------------------------------------------------------------------


def test_vector_store_inputs_extended_with_collection_base_default():
    """Phase 45: VectorStoreInputs.collection_base defaults to the v1.0 root."""
    from pathlib import Path
    from conference_pipeline.analysis.steps.vector_store_setup import (
        VectorStoreInputs,
    )

    inputs = VectorStoreInputs(
        workspace_root=Path("/tmp/ws"),
        papers_file=Path("/tmp/p.json"),
        presentations_dir=Path("/tmp/pres"),
        output_dir=Path("/tmp/out"),
    )
    assert inputs.collection_base == "neurips_presentations"
    assert inputs.runtime_profile is None


def test_vector_store_inputs_accepts_custom_collection_base():
    from pathlib import Path
    from conference_pipeline.analysis.steps.vector_store_setup import (
        VectorStoreInputs,
    )

    inputs = VectorStoreInputs(
        workspace_root=Path("/tmp/ws"),
        papers_file=Path("/tmp/p.json"),
        presentations_dir=Path("/tmp/pres"),
        output_dir=Path("/tmp/out"),
        collection_base="iclr2025_presentations",
    )
    assert inputs.collection_base == "iclr2025_presentations"


# ---------------------------------------------------------------------------
# Plan 45-02: VectorStoreSetupReceipt
# ---------------------------------------------------------------------------


def test_vector_store_setup_receipt_dataclass_shape():
    from conference_pipeline.provider_receipts import (
        VECTOR_STORE_SETUP_RECEIPT_MARKER,
        VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        VectorStoreSetupReceipt,
    )

    receipt = VectorStoreSetupReceipt(
        schema_version=VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        receipt_type=VECTOR_STORE_SETUP_RECEIPT_MARKER,
        provider="mlx",
        model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        collection_name="neurips_presentations__mlx_qwen-qwen3-embedding-0-6b_1024",
        chunk_count=42,
        request_count=1,
        cost_usd=0.0,
        endpoint_label="mlx://Qwen/Qwen3-Embedding-0.6B",
        duration_s=0.5,
        outcome="ok",
        timestamp="2026-05-01T12:00:00Z",
        run_id="run-abc",
        stage_id="vector-store-setup",
    )
    payload = receipt.to_dict()
    assert payload["receipt_type"] == VECTOR_STORE_SETUP_RECEIPT_MARKER
    assert payload["provider"] == "mlx"
    assert payload["embedding_dim"] == 1024
    assert payload["chunk_count"] == 42
    assert payload["cost_usd"] == 0.0


def test_write_vector_store_setup_receipt_with_run_id(tmp_path):
    from pathlib import Path
    from conference_pipeline.provider_receipts import (
        VECTOR_STORE_SETUP_RECEIPT_MARKER,
        VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        VectorStoreSetupReceipt,
        write_vector_store_setup_receipt,
    )

    receipt = VectorStoreSetupReceipt(
        schema_version=VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        receipt_type=VECTOR_STORE_SETUP_RECEIPT_MARKER,
        provider="ollama",
        model="nomic-embed-text",
        embedding_dim=768,
        collection_name="neurips_presentations__ollama_nomic-embed-text_768",
        chunk_count=5,
        request_count=5,
        cost_usd=0.0,
        endpoint_label="http://localhost:11434",
        duration_s=0.1,
        outcome="ok",
        timestamp="2026-05-01T12:34:56Z",
        run_id="run-xyz",
        stage_id="vector-store-setup",
    )
    path = write_vector_store_setup_receipt(receipt, tmp_path)
    expected_dir = (
        tmp_path / "storage" / "workflow-runs" / "run-xyz"
        / "vector-store-receipts" / "vector-store-setup"
    )
    assert path.parent == expected_dir
    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert payload["provider"] == "ollama"
    assert payload["collection_name"].startswith("neurips_presentations__ollama_")


def test_write_vector_store_setup_receipt_without_run_id(tmp_path):
    from conference_pipeline.provider_receipts import (
        VECTOR_STORE_SETUP_RECEIPT_MARKER,
        VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        VectorStoreSetupReceipt,
        write_vector_store_setup_receipt,
    )

    receipt = VectorStoreSetupReceipt(
        schema_version=VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
        receipt_type=VECTOR_STORE_SETUP_RECEIPT_MARKER,
        provider="fake",
        model="text-embedding-3-small",
        embedding_dim=8,
        collection_name="x__fake_text-embedding-3-small_8",
        chunk_count=0,
        request_count=0,
        cost_usd=0.0,
        endpoint_label="fake",
        duration_s=0.0,
        outcome="ok",
        timestamp="2026-05-01T00:00:00Z",
        run_id=None,
        stage_id=None,
    )
    path = write_vector_store_setup_receipt(receipt, tmp_path)
    # No run_id → top-level vector-store-receipts dir.
    assert "workflow-runs" not in path.parts
    assert "vector-store-receipts" in path.parts
    assert path.exists()


# ---------------------------------------------------------------------------
# Plan 45-02: walker collection_name awareness
# ---------------------------------------------------------------------------


def test_walker_collection_name_includes_provider_model_dim():
    """Walker derives collection name from the active embedding client.

    Verifies the constant rename _COLLECTION_NAME → _COLLECTION_BASE +
    derive_collection_name composition. We don't run the full
    walk_and_index here (too heavy for a unit test); we just check the
    module-level constant is gone and the new pieces are wired.
    """
    from conference_pipeline.indexing import walker as _walker

    # _COLLECTION_BASE is the new constant; _COLLECTION_NAME removed.
    assert _walker._COLLECTION_BASE == "v15_iclr2025_production_extractions"
    assert not hasattr(_walker, "_COLLECTION_NAME"), (
        "_COLLECTION_NAME should be removed in Phase 45 Plan 02"
    )
    # derive_collection_name is imported.
    assert hasattr(_walker, "derive_collection_name")
