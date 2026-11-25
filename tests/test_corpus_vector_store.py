from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_embedding_policy_digest_ignores_runtime_batch_size() -> None:
    from conference_pipeline.indexing.corpus_vector_store import (
        embedding_policy_digest,
        embedding_policy_payload,
    )

    class FixtureClient:
        provider = "fake"
        model = "fixture-embedding"
        embedding_dim = 8
        max_length = None

    config = {"provider": "fake", "model": "fixture-embedding", "embedding_dim": 8}
    first = embedding_policy_payload(config, FixtureClient(), 8)
    second = embedding_policy_payload(config, FixtureClient(), 4096)

    assert first == second
    assert embedding_policy_digest(first) == embedding_policy_digest(second)


def test_embedding_sidecars_resume_without_reembedding(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import build_store_embedding_sidecars

    class CountingEmbeddingClient:
        def __init__(self) -> None:
            self.calls = 0

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    workspace = tmp_path / "workspace"
    store_dir = workspace / "storage" / "vector-store" / "test-run" / "stores" / "papers"
    chunks_path = store_dir / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": "papers:000001",
            "artifact_id": "paper-1",
            "source_path": "papers/one.pdf",
            "source_sha256": "source-1",
            "text_path": "text/one.txt",
            "text_sha256": "text-1",
            "source_family_id": "paper_text",
            "store": "papers",
            "text": "first reusable embedding chunk",
        },
        {
            "chunk_id": "papers:000002",
            "artifact_id": "paper-2",
            "source_path": "papers/two.pdf",
            "source_sha256": "source-2",
            "text_path": "text/two.txt",
            "text_sha256": "text-2",
            "source_family_id": "paper_text",
            "store": "papers",
            "text": "second reusable embedding chunk",
        },
    ]
    chunks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    store = {
        "path": str(store_dir.relative_to(workspace)),
        "chunks_path": str(chunks_path.relative_to(workspace)),
        "chunks_sha256": _sha256_file(chunks_path),
        "chunk_count": len(rows),
    }
    client = CountingEmbeddingClient()

    first = build_store_embedding_sidecars(
        workspace=workspace,
        run_id="test-run",
        store_name="papers",
        store=store,
        store_dir=store_dir,
        embedding_client=client,
        provider="fake",
        model="fixture-embedding",
        embedding_dim=4,
        embedding_policy_digest_value="embedding-policy",
        token_budget_policy_digest="token-policy",
        materialization_batch_size=1,
        sidecar_shard_size=1,
    )
    calls_after_first = client.calls
    second = build_store_embedding_sidecars(
        workspace=workspace,
        run_id="test-run",
        store_name="papers",
        store=store,
        store_dir=store_dir,
        embedding_client=client,
        provider="fake",
        model="fixture-embedding",
        embedding_dim=4,
        embedding_policy_digest_value="embedding-policy",
        token_budget_policy_digest="token-policy",
        materialization_batch_size=2,
        sidecar_shard_size=1,
    )

    assert first["embedded_now_count"] == len(rows)
    assert first["embedding_sidecar_resume_count"] == 0
    assert second["embedded_now_count"] == 0
    assert second["embedding_sidecar_resume_count"] == len(rows)
    assert client.calls == calls_after_first


def test_embedding_sidecars_use_array_fast_path_and_write_progress(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import build_store_embedding_sidecars

    class ArrayEmbeddingClient:
        def __init__(self) -> None:
            self.array_calls: list[list[str]] = []
            self.list_calls = 0

        def embed_texts_array(self, texts: list[str]) -> np.ndarray:
            self.array_calls.append(list(texts))
            return np.asarray(
                [[float(len(text)), float(index), 1.0, 0.0] for index, text in enumerate(texts)],
                dtype=np.float32,
            )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
            self.list_calls += 1
            raise AssertionError("sidecar writer should prefer embed_texts_array")

    workspace = tmp_path / "workspace"
    store_dir = workspace / "storage" / "vector-store" / "test-run" / "stores" / "papers"
    chunks_path = store_dir / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": f"papers:{index:06d}",
            "artifact_id": f"paper-{index}",
            "source_path": f"papers/{index}.pdf",
            "source_sha256": f"source-{index}",
            "text_path": f"text/{index}.txt",
            "text_sha256": f"text-{index}",
            "source_family_id": "paper_text",
            "store": "papers",
            "text": f"array fast path reusable embedding chunk {index}",
        }
        for index in range(3)
    ]
    chunks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    store = {
        "path": str(store_dir.relative_to(workspace)),
        "chunks_path": str(chunks_path.relative_to(workspace)),
        "chunks_sha256": _sha256_file(chunks_path),
        "chunk_count": len(rows),
    }
    client = ArrayEmbeddingClient()

    manifest = build_store_embedding_sidecars(
        workspace=workspace,
        run_id="test-run",
        store_name="papers",
        store=store,
        store_dir=store_dir,
        embedding_client=client,
        provider="fake",
        model="fixture-embedding",
        embedding_dim=4,
        embedding_policy_digest_value="embedding-policy",
        token_budget_policy_digest="token-policy",
        materialization_batch_size=2,
        bucket_count=1,
    )

    assert [len(call) for call in client.array_calls] == [2, 1]
    assert client.list_calls == 0
    assert manifest["embedding_array_fast_path_used"] is True
    progress_path = workspace / str(manifest["embedding_sidecar_progress_path"])
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "succeeded"
    assert progress["completed_chunk_count"] == len(rows)
    assert progress["embedding_array_fast_path_used"] is True
    shard_path = workspace / str(manifest["shards"][0]["path"])
    with np.load(shard_path, allow_pickle=False) as loaded:
        embeddings = loaded["embeddings"]
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (3, 4)


def test_corpus_sidecars_prefer_transcript_text_over_asr_receipt_json(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import build_corpus_index_sidecars

    workspace = tmp_path / "workspace"
    transcript_path = workspace / "storage" / "transcription-ocr" / "speech-transcripts" / "talk.txt"
    receipt_path = transcript_path.with_suffix(".json")
    audio_path = workspace / "cache" / "media" / "talk.mp3"
    corpus_inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        "This main conference talk discusses latent reasoning, agentic systems, "
        "and efficient inference over real transcript text. " * 10,
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(
            {
                "receipt_type": "conference_pipeline.speech_transcript",
                "cost_task_id": "asr_transcription",
                "transcript_path": str(transcript_path.relative_to(workspace)),
                "text_character_count": transcript_path.stat().st_size,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"audio")
    corpus_inventory_path.write_text("{}\n", encoding="utf-8")

    digest = build_corpus_index_sidecars(
        workspace=workspace,
        run_id="test-run",
        artifact_records=[
            {
                "artifact_id": "speech_transcript:main",
                "source_family_id": "speech_transcript",
                "source_path": str(audio_path.relative_to(workspace)),
                "source_sha256": "audio-sha",
                "transcript_path": str(transcript_path.relative_to(workspace)),
                "transcript_json_path": str(receipt_path.relative_to(workspace)),
                # This reproduces the stale ICLR bug: indexed_text_path pointed
                # at the ASR receipt, but the reusable asset must still choose
                # the actual transcript text.
                "indexed_text_path": str(receipt_path.relative_to(workspace)),
                "indexed_text_kind": "transcript_json",
                "indexable_text": True,
                "metadata": {
                    "slideslive_id": "39000000",
                    "session_title": "Main Session",
                    "source_title": "Main Talk",
                },
                "provenance": {
                    "download_artifact": {
                        "source_url": "https://slideslive.com/39000000",
                        "slideslive_id": "39000000",
                        "session_title": "Main Session",
                        "source_title": "Main Talk",
                    }
                },
            }
        ],
        corpus_inventory_path=corpus_inventory_path,
        corpus_inventory_sha256="corpus-sha",
        embedding_readiness={
            "status": "available",
            "embedding_status": "chunk_sidecars_written_local_embedding_client_ready",
            "embedding_policy": {"provider": "fake"},
            "embedding_policy_digest": "embedding-policy-sha",
            "embedding_max_length": None,
            "materialization_batch_size": 8,
        },
    )

    artifact_chunks = (
        workspace
        / "storage"
        / "vector-store"
        / "test-run"
        / "stores"
        / "artifacts"
        / "chunks.jsonl"
    )
    rows = [json.loads(line) for line in artifact_chunks.read_text(encoding="utf-8").splitlines()]

    assert digest["indexed_text_kind_counts"] == {"transcript_text": 1}
    assert rows
    assert rows[0]["text_path"].endswith("talk.txt")
    assert rows[0]["indexed_text_kind"] == "transcript_text"
    assert "real transcript text" in rows[0]["text"]
    assert "conference_pipeline.speech_transcript" not in rows[0]["text"]
    assert rows[0]["source_context"]["session_title"] == "Main Session"
    assert rows[0]["source_context"]["source_url"] == "https://slideslive.com/39000000"


def test_corpus_sidecars_reject_snippet_only_first_240_char_input(tmp_path: Path) -> None:
    """A first-240-character snippet-only record must never become a
    vectorized production chunk (done gate 6). ``build_corpus_index_sidecars``
    only ever reads real chunkable text via ``read_indexable_source_text``
    (``text_path`` / ``indexed_text_path`` / ``transcript_path``); a record
    that supplies only a truncated ``evidence.snippet`` and no such path is
    skipped entirely, proving the planning-stage snippet fallback in
    ``chunks_for_record`` is unreachable from the production Stage 6 sidecar
    builder.
    """

    from conference_pipeline.indexing.corpus_vector_store import build_corpus_index_sidecars

    workspace = tmp_path / "workspace"
    corpus_inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    corpus_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_inventory_path.write_text("{}\n", encoding="utf-8")

    snippet_text = (
        "Truncated planning-stage evidence snippet describing the paper contribution. " * 10
    )[:240]
    assert len(snippet_text) == 240

    digest = build_corpus_index_sidecars(
        workspace=workspace,
        run_id="test-run",
        artifact_records=[
            {
                "artifact_id": "paper_text:snippet-only",
                "source_family_id": "paper_text",
                "source_path": "papers/snippet-only.pdf",
                "source_sha256": "snippet-sha",
                "indexable_text": True,
                # Deliberately no text_path / indexed_text_path / transcript_path
                # -- only a first-240-character evidence snippet, the
                # planning-stage artifact shape that is not corpus evidence.
                "evidence": {"snippet": snippet_text},
                "paper_title": "Snippet Only Fixture Paper",
            }
        ],
        corpus_inventory_path=corpus_inventory_path,
        corpus_inventory_sha256="corpus-sha",
        embedding_readiness={
            "status": "available",
            "embedding_status": "chunk_sidecars_written_local_embedding_client_ready",
            "embedding_policy": {"provider": "fake"},
            "embedding_policy_digest": "embedding-policy-sha",
            "embedding_max_length": None,
            "materialization_batch_size": 8,
        },
    )

    papers_chunks = (
        workspace / "storage" / "vector-store" / "test-run" / "stores" / "papers" / "chunks.jsonl"
    )
    assert digest["stores"]["papers"]["chunk_count"] == 0
    assert papers_chunks.exists()
    assert papers_chunks.stat().st_size == 0
    assert snippet_text not in papers_chunks.read_text(encoding="utf-8")


def test_corpus_chroma_metadata_exposes_stable_source_fields(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import chunk_chroma_metadata

    metadata = chunk_chroma_metadata(
        {
            "artifact_id": "speech_transcript:workshop",
            "source_path": "storage/workshops/talk-media/demo.mp3",
            "source_sha256": "source-sha",
            "text_path": "storage/transcription-ocr/workshop/annotated.json",
            "text_sha256": "text-sha",
            "strategy": "transcript_120s_segments",
            "chunk_index": 0,
            "chunk_count": 1,
            "source_family_id": "speech_transcript",
            "store": "artifacts",
            "source_context": {
                "content_kind": "transcript",
                "conference_scope": "workshop",
                "workshop_id": "ICLR.cc/2026/Workshop/Demo",
                "workshop_title": "Demo Workshop",
                "talk_id": "talk_001",
                "talk_title": "Demo Talk",
                "talk_speaker": "Dr Demo",
                "source_url": "https://slideslive.com/39000001",
            },
            "text": "Workshop: Demo Workshop\nTalk: Demo Talk\n\nUseful transcript text.",
        },
        workspace=tmp_path,
        run_id="test-run",
        provider="fake",
        model="fake-embedding",
        embedding_dim=8,
        embedding_digest="embedding-digest",
        embedding_policy_digest="embedding-policy",
        token_budget_policy_digest="token-policy",
        embedding_max_length=None,
        materialization_batch_size=8,
    )

    assert metadata["topic_ids"] == ""
    assert metadata["content_kind"] == "transcript"
    assert metadata["conference_scope"] == "workshop"
    assert metadata["workshop_id"] == "ICLR.cc/2026/Workshop/Demo"
    assert metadata["talk_title"] == "Demo Talk"
    assert metadata["source_url"] == "https://slideslive.com/39000001"


def test_resolve_workspace_member_allows_workspace_symlink(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import resolve_workspace_member

    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-vector-store"
    external_root.mkdir()
    real_file = external_root / "stores" / "papers" / "chunks.jsonl"
    real_file.parent.mkdir(parents=True)
    real_file.write_text("{}\n", encoding="utf-8")
    linked_dir = workspace / "storage" / "vector-store"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved, error = resolve_workspace_member(
        workspace, "storage/vector-store/stores/papers/chunks.jsonl"
    )
    assert error is None
    assert resolved is not None
    assert resolved.read_text(encoding="utf-8") == "{}\n"


def test_resolve_workspace_member_still_blocks_dotdot_escape(tmp_path: Path) -> None:
    from conference_pipeline.indexing.corpus_vector_store import resolve_workspace_member

    workspace = tmp_path / "conference"
    workspace.mkdir()
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")

    resolved, error = resolve_workspace_member(workspace, "../outside.json")
    assert resolved is None
    assert error == "path_outside_workspace"


def _write_minimal_reusable_digest_fixture(workspace: Path, run_id: str):
    """Build a self-consistent index_digest.json + per-store chunks/metadata
    files, returning (index_digest_path, corpus_inventory_sha256, embedding_
    readiness) so tests can flip vector_embeddings_written and observe reuse
    decisions.
    """
    import hashlib

    from conference_pipeline.indexing.corpus_vector_store import STORE_NAMES

    vector_root = workspace / "storage" / "vector-store" / run_id
    stores = {}
    for store_name in STORE_NAMES:
        store_dir = vector_root / "stores" / store_name
        store_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = store_dir / "chunks.jsonl"
        chunks_path.write_text("", encoding="utf-8")
        metadata_path = store_dir / "metadata.json"
        metadata_path.write_text("{}\n", encoding="utf-8")
        stores[store_name] = {
            "path": str(store_dir.relative_to(workspace)),
            "chunks_path": str(chunks_path.relative_to(workspace)),
            "chunks_sha256": _sha256_file(chunks_path),
            "metadata_path": str(metadata_path.relative_to(workspace)),
            "chunk_count": 0,
        }
    corpus_inventory_sha256 = hashlib.sha256(b"fixture-corpus").hexdigest()
    from conference_pipeline.indexing.token_policy import (
        default_token_budget_policy,
        policy_metadata,
    )

    token_policy_digest = str(policy_metadata(default_token_budget_policy())["policy_digest"])
    embedding_readiness = {"embedding_policy_digest": "embedding-policy-digest"}
    digest = {
        "receipt_type": "conference_pipeline.corpus_index_digest",
        "corpus_inventory_sha256": corpus_inventory_sha256,
        "token_budget_policy_digest": token_policy_digest,
        "embedding_policy_digest": "embedding-policy-digest",
        "stores": stores,
    }
    index_digest_path = vector_root / "index_digest.json"
    return digest, index_digest_path, corpus_inventory_sha256, embedding_readiness


def test_load_reusable_corpus_index_sidecars_rejects_incomplete_prior_attempt(
    tmp_path: Path,
) -> None:
    """A digest from an interrupted run (embeddings never materialized) must

    not be treated as reusable -- otherwise every subsequent resume silently
    skips re-chunking forever while the real vector store never matches.
    """
    from conference_pipeline.indexing.corpus_vector_store import (
        load_reusable_corpus_index_sidecars,
    )

    workspace = tmp_path / "workspace"
    digest, index_digest_path, corpus_inventory_sha256, embedding_readiness = (
        _write_minimal_reusable_digest_fixture(workspace, "test-run")
    )
    # No "vector_embeddings_written" key at all -- exactly what a crash
    # between build_corpus_index_sidecars and a successful materialize_
    # corpus_vector_embeddings leaves on disk.
    index_digest_path.write_text(json.dumps(digest), encoding="utf-8")

    result = load_reusable_corpus_index_sidecars(
        workspace=workspace,
        run_id="test-run",
        corpus_inventory_sha256=corpus_inventory_sha256,
        embedding_readiness=embedding_readiness,
    )
    assert result is None


def test_load_reusable_corpus_index_sidecars_accepts_completed_prior_attempt(
    tmp_path: Path,
) -> None:
    from conference_pipeline.indexing.corpus_vector_store import (
        load_reusable_corpus_index_sidecars,
    )

    workspace = tmp_path / "workspace"
    digest, index_digest_path, corpus_inventory_sha256, embedding_readiness = (
        _write_minimal_reusable_digest_fixture(workspace, "test-run")
    )
    digest["vector_embeddings_written"] = True
    index_digest_path.write_text(json.dumps(digest), encoding="utf-8")

    result = load_reusable_corpus_index_sidecars(
        workspace=workspace,
        run_id="test-run",
        corpus_inventory_sha256=corpus_inventory_sha256,
        embedding_readiness=embedding_readiness,
    )
    assert result is not None
    assert result["sidecar_resume_reused"] is True


def test_compact_record_fields_preserves_authors_list_native_type() -> None:
    from conference_pipeline.indexing.corpus_vector_store import _compact_record_fields

    out = _compact_record_fields({"authors": ["A B", "C D"]})

    assert out["authors"] == ["A B", "C D"]
    assert isinstance(out["authors"], list)


def test_compact_record_fields_treats_empty_authors_list_as_absent() -> None:
    from conference_pipeline.indexing.corpus_vector_store import _compact_record_fields

    out = _compact_record_fields({"authors": [], "paper_title": "Some Paper"})

    assert "authors" not in out
    assert out["paper_title"] == "Some Paper"


def test_stable_source_context_carries_authors_list() -> None:
    from conference_pipeline.indexing.corpus_vector_store import stable_source_context

    context = stable_source_context(
        {
            "source_family_id": "paper_text",
            "authors": ["A B"],
        }
    )

    assert context["authors"] == ["A B"]


def test_chunk_record_for_artifact_promotes_authors_list_to_top_level() -> None:
    from conference_pipeline.indexing.corpus_vector_store import chunk_record_for_artifact

    record = {
        "artifact_id": "paper:fixture",
        "source_family_id": "paper_text",
        "paper_title": "Fixture Paper",
        "authors": ["A B", "C D"],
    }

    payload = chunk_record_for_artifact(record)

    assert payload["authors"] == ["A B", "C D"]


def test_chunk_record_for_artifact_omits_empty_authors_list() -> None:
    from conference_pipeline.indexing.corpus_vector_store import chunk_record_for_artifact

    record = {
        "artifact_id": "paper:fixture-empty-authors",
        "source_family_id": "paper_text",
        "paper_title": "Fixture Paper",
        "authors": [],
    }

    payload = chunk_record_for_artifact(record)

    assert "authors" not in payload


def _chroma_metadata_for_authors(authors: object) -> dict[str, object]:
    from conference_pipeline.indexing.corpus_vector_store import chunk_chroma_metadata

    return chunk_chroma_metadata(
        {
            "artifact_id": "paper:fixture",
            "source_path": "papers/fixture.pdf",
            "source_sha256": "source-sha",
            "text_path": "text/fixture.txt",
            "text_sha256": "text-sha",
            "strategy": "paper_text_chunk",
            "chunk_index": 0,
            "chunk_count": 1,
            "source_family_id": "paper_text",
            "store": "papers",
            "source_context": {
                "content_kind": "paper",
                "conference_scope": "main_conference",
                "paper_title": "Fixture Paper",
                "authors": authors,
            },
            "text": "Fixture paper body text.",
        },
        workspace=Path("/tmp/workspace-fixture"),
        run_id="test-run",
        provider="fake",
        model="fake-embedding",
        embedding_dim=8,
        embedding_digest="embedding-digest",
        embedding_policy_digest="embedding-policy",
        token_budget_policy_digest="token-policy",
        embedding_max_length=None,
        materialization_batch_size=8,
    )


def test_chunk_chroma_metadata_joins_authors_list_into_bounded_string() -> None:
    metadata = _chroma_metadata_for_authors(["A B", "C D"])

    assert metadata["authors"] == "A B; C D"
    assert type(metadata["authors"]) is str
    assert "[" not in metadata["authors"]
    assert "]" not in metadata["authors"]


def test_chunk_chroma_metadata_truncates_pathological_author_roster() -> None:
    from conference_pipeline.indexing.corpus_vector_store import CHROMA_AUTHORS_MAX_CHARS

    authors = [f"Author Number {index:03d}" for index in range(200)]
    metadata = _chroma_metadata_for_authors(authors)

    assert len(metadata["authors"]) <= CHROMA_AUTHORS_MAX_CHARS
    assert metadata["authors"].endswith("…")


def test_chunk_chroma_metadata_authors_empty_when_absent() -> None:
    metadata = _chroma_metadata_for_authors(None)

    assert metadata["authors"] == ""


def test_chunk_chroma_metadata_source_context_json_roundtrips_authors_list() -> None:
    metadata = _chroma_metadata_for_authors(["A B", "C D"])

    round_tripped = json.loads(metadata["source_context_json"])
    assert round_tripped["authors"] == ["A B", "C D"]


def test_authors_survive_all_three_allowlist_locations_in_one_chain() -> None:
    """Regression guard for RESEARCH Pitfall 2: fails if any ONE of

    SOURCE_CONTEXT_FIELDS, CHROMA_STABLE_METADATA_FIELDS, or
    chunk_record_for_artifact()'s copy-loop tuple loses the ``authors`` key.
    """
    from conference_pipeline.indexing.corpus_vector_store import (
        chunk_chroma_metadata,
        chunk_record_for_artifact,
        stable_source_context,
    )

    record = {
        "artifact_id": "paper:three-allowlists-fixture",
        "source_family_id": "paper_text",
        "paper_title": "Three Allowlists Fixture Paper",
        "authors": ["A B", "C D"],
    }

    # Location 1: SOURCE_CONTEXT_FIELDS -> stable_source_context()
    source_context = stable_source_context(record)
    assert source_context["authors"] == ["A B", "C D"]

    # Location 2: chunk_record_for_artifact()'s copy-loop
    chunk_payload = chunk_record_for_artifact(record)
    assert chunk_payload["authors"] == ["A B", "C D"]

    # Location 3: CHROMA_STABLE_METADATA_FIELDS -> chunk_chroma_metadata()
    metadata = chunk_chroma_metadata(
        {
            **chunk_payload,
            "chunk_index": 0,
            "chunk_count": 1,
            "strategy": "paper_text_chunk",
            "text": "Fixture body text.",
        },
        workspace=Path("/tmp/workspace-fixture"),
        run_id="test-run",
        provider="fake",
        model="fake-embedding",
        embedding_dim=8,
        embedding_digest="embedding-digest",
        embedding_policy_digest="embedding-policy",
        token_budget_policy_digest="token-policy",
        embedding_max_length=None,
        materialization_batch_size=8,
    )
    assert metadata["authors"] == "A B; C D"


# Real record key sets copied verbatim (values stripped to short placeholders)
# from conferences/cvpr2026/storage/corpus/corpus_inventory.json this session, so
# the authors-propagation regression proves the fix works for BOTH paper
# families, not a workshop-only special case (CONTEXT.md's platform-generality
# decision). Derived from:
#   paper_text -> artifact_id "paper:0016253a99ab5149ba1bf1d34aa6c865ae8763c4ab87dca60d527c19cbcfd7eb"
#   workshop_paper_text -> artifact_id "workshop_paper:thecvf.com/CVPR/2026/Workshop/3D4S:1FdAbS4FoQ"
REAL_SHAPE_RECORD_FIXTURES = {
    "paper_text": {
        "artifact_id": "paper:fixture-stem",
        "indexable_text": True,
        "retention_class": "paper_text",
        "source_family_id": "paper_text",
        "source_path": "storage/download-corpus/artifacts/paper_pdf/fixture-stem.pdf",
        "source_sha256": "fixture-source-sha",
        "storage_lane": "storage/corpus/paper-text",
        "text_path": "storage/corpus/paper-text/fixture-stem.txt",
        "authors": ["A B", "C D"],
    },
    "workshop_paper_text": {
        "artifact_id": "workshop_paper:example.org/Workshop/Fixture:fixture-stem",
        "indexable_text": True,
        "paper_title": "Fixture Workshop Paper",
        "retention_class": "workshop_paper_text",
        "source_family_id": "workshop_paper_text",
        "source_path": "storage/workshops/papers/example-org_workshop_fixture/fixture-stem.pdf",
        "source_sha256": "fixture-source-sha",
        "storage_lane": "storage/workshops/papers",
        "text_path": "storage/workshops/papers/example-org_workshop_fixture/text/fixture-stem.txt",
        "workshop_id": "example.org/Workshop/Fixture",
        "authors": ["A B", "C D"],
    },
}


@pytest.mark.parametrize("family", ["paper_text", "workshop_paper_text"])
def test_authors_propagate_for_both_real_paper_family_shapes(family: str) -> None:
    """Proves the authors-allowlist fix is corpus-wide, not workshop-only.

    Uses the exact key set of a real ``paper_text``/``workshop_paper_text``
    record (values stripped to placeholders) so this is not a hand-invented
    fixture shape.
    """
    from conference_pipeline.indexing.corpus_vector_store import (
        chunk_chroma_metadata,
        chunk_record_for_artifact,
        stable_source_context,
    )

    record = REAL_SHAPE_RECORD_FIXTURES[family]

    source_context = stable_source_context(record)
    assert source_context["authors"] == ["A B", "C D"]

    chunk_payload = chunk_record_for_artifact(record)
    assert chunk_payload["authors"] == ["A B", "C D"]

    metadata = chunk_chroma_metadata(
        {
            **chunk_payload,
            "chunk_index": 0,
            "chunk_count": 1,
            "strategy": "paper_text_chunk",
            "text": "Fixture body text.",
        },
        workspace=Path("/tmp/workspace-fixture"),
        run_id="test-run",
        provider="fake",
        model="fake-embedding",
        embedding_dim=8,
        embedding_digest="embedding-digest",
        embedding_policy_digest="embedding-policy",
        token_budget_policy_digest="token-policy",
        embedding_max_length=None,
        materialization_batch_size=8,
    )
    assert metadata["authors"] == "A B; C D"


def test_bulk_load_store_chroma_deletes_stale_ids_no_longer_in_corpus(tmp_path: Path) -> None:
    """collection.upsert only adds/updates; a chunk id that drops out of the

    corpus on a later run (re-chunked boundaries, a source removed) must be
    deleted from the persisted collection, not left behind forever inflating
    the count past what the current corpus actually contains.
    """
    pytest.importorskip("chromadb")
    from conference_pipeline.indexing.corpus_vector_store import (
        DEFAULT_COLLECTION_BASE,
        build_store_embedding_sidecars,
        bulk_load_store_chroma_from_embedding_sidecars,
    )

    class FixedEmbeddingClient:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    workspace = tmp_path / "workspace"
    store_dir = workspace / "storage" / "vector-store" / "test-run" / "stores" / "papers"
    chunks_path = store_dir / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_chunks(rows: list[dict]) -> None:
        chunks_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _materialize(rows: list[dict]) -> dict:
        _write_chunks(rows)
        store = {
            "path": str(store_dir.relative_to(workspace)),
            "chunks_path": str(chunks_path.relative_to(workspace)),
            "chunks_sha256": _sha256_file(chunks_path),
            "chunk_count": len(rows),
        }
        sidecars = build_store_embedding_sidecars(
            workspace=workspace,
            run_id="test-run",
            store_name="papers",
            store=store,
            store_dir=store_dir,
            embedding_client=FixedEmbeddingClient(),
            provider="fake",
            model="fixture-embedding",
            embedding_dim=4,
            embedding_policy_digest_value="embedding-policy",
            token_budget_policy_digest="token-policy",
            materialization_batch_size=2,
            sidecar_shard_size=2,
        )
        return bulk_load_store_chroma_from_embedding_sidecars(
            workspace=workspace,
            run_id="test-run",
            store_name="papers",
            store=store,
            store_dir=store_dir,
            provider="fake",
            model="fixture-embedding",
            embedding_dim=4,
            embedding_policy={},
            embedding_policy_digest_value="embedding-policy",
            token_budget_policy_digest="token-policy",
            materialization_batch_size=2,
            chroma_insert_batch_size=100,
            embedding_sidecars=sidecars,
            corpus_inventory_sha256="fixture-sha",
            vector_store_dir=Path("storage") / "vector-store",
            collection_base=DEFAULT_COLLECTION_BASE,
            sanitize_fn=lambda value: value,
        )

    def _row(chunk_id: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "artifact_id": chunk_id,
            "source_path": f"papers/{chunk_id}.pdf",
            "source_sha256": f"source-{chunk_id}",
            "text_path": f"text/{chunk_id}.txt",
            "text_sha256": f"text-{chunk_id}",
            "source_family_id": "paper_text",
            "store": "papers",
            "text": f"chunk text for {chunk_id}",
        }

    first = _materialize([_row("A"), _row("B")])
    assert first["embedded_chunk_count"] == 2
    assert first["embedding_stale_deleted_count"] == 0

    # Simulate a re-chunk that drops A and B in favor of a differently-keyed C.
    second = _materialize([_row("C")])
    assert second["embedded_chunk_count"] == 1
    assert second["embedding_stale_deleted_count"] == 2

    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(store_dir / "chroma"))
    collection = chroma_client.get_collection(second["collection_name"])
    assert collection.count() == 1
    assert collection.get(include=[])["ids"] == ["C"]
