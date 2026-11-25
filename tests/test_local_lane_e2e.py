"""Phase 47 Plan 02 — local-lane end-to-end exercise.

Uses Fake clients (project convention; no real MLX/Ollama required) to
drive the v1.9 local lane through:

1. ``build_embedding_client`` factory dispatch.
2. ``MLXEmbeddingClient.embed_texts`` with workspace → writes
   ``LocalEmbeddingReceipt``.
3. ``derive_collection_name`` → suffixed Chroma collection name.
4. ``build_researcher`` factory dispatch.
5. ``LocalLLMResearcher.run`` with a Fake OpenAI-compatible client →
   returns synthesis content.

This is a *structural* end-to-end: it asserts the local lane is
plumbed correctly through factories and adapters with no OpenAI key
in the environment. Real-MLX/real-Ollama exercise lives in operator
runs of ``conference-smoketest --local`` (CLI tests in Plan 47-01).

Marked ``@pytest.mark.local_e2e``; default CI excludes via the
conftest hook. Run locally with ``RUN_LOCAL_E2E=1 pytest -m local_e2e``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.local_e2e


def test_local_lane_end_to_end_with_fakes(tmp_path, monkeypatch):
    """Drive: factory → MLX embed (with workspace receipt) → researcher synth."""
    # 1. Operator must NOT have OPENAI_API_KEY for the local-only assertion.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # 2. Build an MLX embedding client via the factory with an injected
    #    fake loader (no mlx-embeddings install required).
    from conference_pipeline.indexing.embeddings import build_embedding_client

    class _FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": np.zeros((len(texts), 4), dtype=int)}

    class _FakeModel:
        def __call__(self, tokenized):
            n = int(tokenized["input_ids"].shape[0])
            return np.tile(np.arange(1024, dtype=float) / 1000.0, (n, 1))

    def _fake_loader(model_name):  # noqa: ARG001
        return _FakeModel(), _FakeTokenizer()

    embedding_client = build_embedding_client(
        {
            "provider": "mlx",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "_loader": _fake_loader,
        }
    )

    # Wire workspace + run_id so the embed call writes a receipt.
    embedding_client._workspace = tmp_path
    embedding_client._run_id = "e2e-run-001"
    embedding_client._stage_id = "vector-store-setup"

    # 3. Drive embeddings — assert receipt landed under the expected path.
    vectors = embedding_client.embed_texts(
        ["chunk one alpha", "chunk two beta", "chunk three gamma"]
    )
    assert len(vectors) == 3
    assert all(len(v) == 1024 for v in vectors)
    receipt_dir = (
        tmp_path / "storage" / "workflow-runs" / "e2e-run-001"
        / "local-embedding-receipts" / "vector-store-setup"
    )
    receipts = list(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["provider"] == "mlx"
    assert payload["input_count"] == 3
    assert payload["cost_usd"] == 0.0
    assert payload["endpoint_label"].startswith("mlx://")

    # 4. Verify derived collection name is what Phase 45 produces.
    from conference_pipeline.indexing.vector_store_naming import (
        derive_collection_name,
    )
    collection_name = derive_collection_name(
        "neurips_presentations", "mlx", "Qwen/Qwen3-Embedding-0.6B", 1024
    )
    assert collection_name == "neurips_presentations__mlx_qwen-qwen3-embedding-0-6b_1024"

    # 5. Build a LocalLLMResearcher via factory with a Fake OpenAI client.
    from conference_pipeline.research import build_researcher

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self):
            self.choices = [_FakeChoice("local-lane synthesis report")]
            self.usage = type(
                "U", (), {"prompt_tokens": 50, "completion_tokens": 200}
            )()

    class _FakeCompletions:
        def create(self, **_kwargs):
            return _FakeResponse()

    fake_openai_client = type(
        "C",
        (),
        {"chat": type("Ch", (), {"completions": _FakeCompletions()})()},
    )()

    researcher = build_researcher(
        {
            "provider": "local",
            "base_url": "http://127.0.0.1:8080/v1",
            "_client": fake_openai_client,
        }
    )
    result = asyncio.run(researcher.run("what is the main contribution?"))
    assert result == "local-lane synthesis report"

    # 6. Telemetry parity check — researcher.usage() returns OpenAIResearcher shape.
    usage = researcher.usage()
    assert set(usage.keys()) == {
        "input_tokens",
        "output_tokens",
        "request_count",
        "estimated_cost_usd",
        "provider_receipt_paths",
        "temperature",
        "max_tokens",
    }
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] == 200
    assert usage["request_count"] == 1


def test_local_e2e_marker_requires_run_local_e2e_env():
    """Sanity: when this test runs, RUN_LOCAL_E2E must be set (else it'd be skipped)."""
    assert os.environ.get("RUN_LOCAL_E2E") == "1"


def test_local_lane_retrieval_restored(tmp_path, monkeypatch):
    """Phase 48 INT-BLOCKER-01 closure (chain level): walker writes a
    suffixed Chroma collection with ChromaEmbeddingAdapter registered;
    retriever opens the SAME collection with the SAME adapter; ``retrieve``
    returns non-empty chunks; researcher synthesizes on non-empty context.
    Replays the broken-on-main flow from v1.9-MILESTONE-AUDIT.md.

    This is the CHAIN-level proof. Unit-level proof for both lanes lives
    in tests/test_researcher_collection_name.py::test_walker_retriever_round_trip.
    """
    # Test isolation: prevent .env from re-injecting OPENAI_API_KEY
    # (mirrors commit 168f4dac).
    monkeypatch.setattr(
        "dotenv.load_dotenv", lambda *a, **kw: True, raising=False
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from tests._indexing_fixtures import _build_indexing_workspace

    bundle = _build_indexing_workspace(tmp_path)
    workspace = bundle["workspace"]
    run_id = bundle["run_id"]

    # 1. Build an MLX embedding client via the factory + the same fake
    #    loader pattern used in test_local_lane_end_to_end_with_fakes
    #    (defined inline so this test does not depend on cross-test
    #    state).
    from conference_pipeline.indexing.embeddings import (
        build_embedding_client,
    )

    class _FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": np.zeros((len(texts), 4), dtype=int)}

    class _FakeModel:
        def __call__(self, tokenized):
            n = int(tokenized["input_ids"].shape[0])
            return np.tile(np.arange(1024, dtype=float) / 1000.0, (n, 1))

    def _fake_loader(model_name):  # noqa: ARG001
        return _FakeModel(), _FakeTokenizer()

    embedding_client = build_embedding_client(
        {
            "provider": "mlx",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "_loader": _fake_loader,
        }
    )

    # 2. Walk + index: writes the suffixed Chroma collection with
    #    ChromaEmbeddingAdapter registered (D-14 write side).
    from conference_pipeline.indexing.walker import walk_and_index

    result = walk_and_index(
        workspace,
        run_id=run_id,
        embedding_client=embedding_client,
        force_rebuild=True,
    )
    assert result.indexed_count > 0
    expected = result.collection_name
    assert expected.startswith(
        "v15_iclr2025_production_extractions__mlx_"
    ), (
        f"expected suffixed mlx collection name; got {expected!r}"
    )

    # 3. Build the retriever via constructor injection (D-16 production
    #    path — what iclr2025_live.py:run_query_planning_gpt_researcher
    #    now does).
    from conference_pipeline.research import ChromaVectorStoreRetriever

    retriever = ChromaVectorStoreRetriever(
        workspace, run_id=run_id, embedding_client=embedding_client
    )

    # 4. Retrieve — non-empty chunks → INT-BLOCKER-01 closed.
    chunks = retriever.retrieve(
        topic_id="02_model_architectures",
        query="transformer architectures",
        k=5,
    )
    assert len(chunks) > 0, (
        "INT-BLOCKER-01 NOT CLOSED: retriever returned 0 chunks "
        "from the suffixed mlx collection."
    )

    # 4b. Negative-control: retriever opened the SUFFIXED name, not a
    #     legacy hardcoded one. If a future change regresses this, the
    #     assertion fails immediately.
    assert retriever._collection is not None
    assert retriever._collection.name == expected, (
        f"retriever opened {retriever._collection.name!r}, "
        f"expected {expected!r}"
    )

    # 5. Build LocalLLMResearcher via factory with a fake OpenAI client
    #    (mirrors test_local_lane_end_to_end_with_fakes).
    from conference_pipeline.research import build_researcher

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self):
            self.choices = [_FakeChoice("local-lane synthesis on non-empty context")]
            self.usage = type(
                "U", (), {"prompt_tokens": 50, "completion_tokens": 200}
            )()

    class _FakeCompletions:
        def create(self, **_kwargs):
            return _FakeResponse()

    fake_openai_client = type(
        "C",
        (),
        {"chat": type("Ch", (), {"completions": _FakeCompletions()})()},
    )()

    researcher = build_researcher(
        {
            "provider": "local",
            "base_url": "http://127.0.0.1:8080/v1",
            "_client": fake_openai_client,
        }
    )

    # 6. Drive synthesis on the actual retrieved chunks.
    synthesis_input = (
        "Topic: model architectures\n\n"
        + "\n\n".join(c.text for c in chunks)
    )
    body = asyncio.run(researcher.run(synthesis_input))
    assert isinstance(body, str)
    assert body, "non-empty synthesis on non-empty context"
