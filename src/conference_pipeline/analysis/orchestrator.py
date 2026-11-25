"""Orchestrator and injected runners for the composable analysis stage."""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from conference_pipeline import workflow
from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.constants import NEURIPS_TOPIC_IDS
from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    conference_name,
    load_conference_identity,
)
from conference_pipeline.analysis.narrative import (
    OpenAIDeepNarrativeClient,
    render_narrative_outputs_from_source_packs,
)
from conference_pipeline.analysis.steps import (
    CoverageInputs,
    GPTResearcherInputs,
    GapFillInputs,
    QAInputs,
    TraceabilityInputs,
    VectorStoreInputs,
    step_coverage,
    step_gap_fill,
    step_gpt_researcher_hybrid,
    step_qa,
    step_traceability,
    step_vector_store_setup,
)


@dataclass(frozen=True)
class AnalyzeStageResult:
    manifest_path: Path
    manifest_sha256: str
    manifest_content_sha256: str
    sub_step_digests: dict[str, str]


class StepRunner(Protocol):
    """Injection seam for live-vs-test step execution."""

    embedding_client: Any
    researcher: Any
    vector_store: Any


@dataclass
class LiveStepRunner:
    env_file: Path | None = None
    vector_store_dir: Path | None = None
    workspace_root: Path | None = None

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        *,
        workspace_root: Path | None = None,
    ) -> "LiveStepRunner":
        return cls(env_file=env_file, workspace_root=workspace_root)

    @property
    def embedding_client(self) -> Any:
        return OpenAIEmbeddingClient()

    @property
    def researcher(self) -> Any:
        return GPTResearcherAdapter()

    @property
    def vector_store(self) -> Any:
        vector_store_dir = self.vector_store_dir
        if vector_store_dir is None:
            if self.workspace_root is None:
                raise ValueError(
                    "LiveStepRunner needs workspace_root or vector_store_dir to "
                    "locate this conference's vector store"
                )
            # Resolve to THIS workspace's vector store, not a hardcoded conference.
            vector_store_dir = Path(self.workspace_root) / "vector_store"
        return ChromaVectorStore(
            vector_store_dir / "chroma_db",
            collection_name=_presentations_collection_name(self.workspace_root),
        )


@dataclass
class OpenAIEmbeddingClient:
    model_name: str = "text-embedding-3-small"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from chromadb.utils import embedding_functions  # type: ignore[import-not-found]

        function = embedding_functions.OpenAIEmbeddingFunction(model_name=self.model_name)
        return function(texts)


@dataclass
class GPTResearcherAdapter:
    report_type: str = "research_report"

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        from gpt_researcher import GPTResearcher  # type: ignore[import-not-found]

        researcher = GPTResearcher(query=query, report_type=self.report_type)
        if context_path is not None:
            researcher.context = [f"Local vector store path: {context_path}"]
        return await researcher.write_report()


def _presentations_collection_name(workspace_root: Path | None) -> str:
    """Derive the presentations collection name from the workspace conference slug.

    Falls back to the workspace directory name (or the historical
    ``neurips_presentations`` default when no workspace is available) so reusable
    code never hardcodes a single conference's collection.
    """
    if workspace_root is None:
        return "neurips_presentations"
    try:
        slug = load_conference_identity(workspace_root).slug
    except ConferenceIdentityError:
        slug = Path(workspace_root).resolve().name
    return f"{slug}_presentations"


@dataclass
class ChromaVectorStore:
    chroma_db_path: Path
    # Default kept for backward-compat; live callers derive the collection from
    # the workspace conference slug via ``_presentations_collection_name``.
    collection_name: str = "neurips_presentations"

    def query(self, text: str, *, top_k: int) -> list[dict[str, Any]]:
        import chromadb  # type: ignore[import-not-found]

        client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        collection = client.get_or_create_collection(self.collection_name)
        results = collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0] if isinstance(results, dict) else []
        metadatas = results.get("metadatas", [[]])[0] if isinstance(results, dict) else []
        distances = results.get("distances", [[]])[0] if isinstance(results, dict) else []
        for document, metadata, distance in zip(documents, metadatas, distances):
            output.append(
                {
                    "document": document,
                    "metadata": metadata or {},
                    "score": 1 - distance if distance is not None else 0.0,
                }
            )
        return output


@dataclass
class FakeEmbeddingClient:
    dim: int = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


@dataclass
class CannedResearcher:
    by_query: dict[str, str]

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        return self.by_query.get(query, self.by_query.get("default", "# Canned\n"))


@dataclass
class CannedVectorStore:
    hits_by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def query(self, text: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.hits_by_query.get(text, [])[:top_k]


@dataclass
class MockStepRunner:
    embedding_client: FakeEmbeddingClient
    researcher: CannedResearcher
    vector_store: CannedVectorStore

    @classmethod
    def from_fixture(cls, fixture_dir: Path) -> "MockStepRunner":
        canned_path = fixture_dir / "mini_gpt_researcher" / "canned_responses.json"
        payload = json.loads(canned_path.read_text(encoding="utf-8"))
        by_query = payload.get("by_query", {})
        if not isinstance(by_query, dict):
            by_query = {}
        return cls(
            embedding_client=FakeEmbeddingClient(),
            researcher=CannedResearcher({str(k): str(v) for k, v in by_query.items()}),
            vector_store=CannedVectorStore(),
        )


def run_analyze_stage(
    workspace: Any,
    *,
    run_id: str | None = None,
    env_file: Path | None = None,
    storage_root: Path | None = None,
    step_runner: StepRunner | None = None,
    overwrite_existing: bool = False,
) -> AnalyzeStageResult | dict[str, Any]:
    _ = storage_root
    workspace_root = workspace.root
    runner = step_runner or LiveStepRunner.from_env(env_file, workspace_root=workspace_root)
    conference_label = conference_name(workspace_root, default="the conference")

    analysis_dir = workflow._resolve_path_inside_workspace(
        workspace_root,
        "analysis",
        label="analysis directory",
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir = workflow._resolve_path_inside_workspace(
        workspace_root,
        "vector_store",
        label="vector_store directory",
    )
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    presentations_dir = workflow._resolve_path_inside_workspace(
        workspace_root,
        "presentations",
        label="presentations directory",
    )
    presentations_dir.mkdir(parents=True, exist_ok=True)
    papers_file = workspace_root / "papers.json"

    vs_out = step_vector_store_setup(
        VectorStoreInputs(
            workspace_root=workspace_root,
            papers_file=papers_file,
            presentations_dir=presentations_dir,
            output_dir=vector_store_dir,
        ),
        embedding_client=runner.embedding_client,
    )
    topics = tuple(_discover_topics(analysis_dir) or _fallback_topic_ids(workspace_root))
    gp_out = step_gpt_researcher_hybrid(
        GPTResearcherInputs(
            workspace_root=workspace_root,
            vector_store_dir=vs_out.chroma_db_path.parent,
            output_dir=analysis_dir,
            topics=topics,
            overwrite_existing=overwrite_existing,
            conference_label=conference_label,
        ),
        researcher=runner.researcher,
    )
    narrative_client = OpenAIDeepNarrativeClient.from_env()
    narrative_out = render_narrative_outputs_from_source_packs(
        workspace_root,
        run_id=run_id or "workflow-analyze",
        analysis_dir=analysis_dir,
        evidence_chain_version="generic-narrative-v1",
        llm_client=narrative_client,
        allow_deterministic_fallback=narrative_client is None,
        narrative_model=getattr(narrative_client, "model", None),
    )
    citations_by_topic = {
        str(entry.get("topic_id")): {
            "extraction_record_ids": len(entry.get("extraction_record_ids") or []),
            "vector_store_chunk_ids": len(entry.get("vector_store_chunk_ids") or []),
            "research_ids": len(entry.get("research_ids") or []),
        }
        for entry in narrative_out.citation_summaries
        if entry.get("topic_id")
    }
    provenance_by_topic = {
        str(entry.get("topic_id")): {
            "provenance_path": entry.get("provenance_path"),
            "provenance_json_path": entry.get("provenance_json_path"),
        }
        for entry in narrative_out.citation_summaries
        if entry.get("topic_id")
    }
    if narrative_out.status == "blocked":
        from conference_pipeline.analysis.manifest import emit_analysis_manifest

        manifest_kwargs: dict[str, Any] = {
            "generator": f"conference_pipeline.analysis.emit_manifest@{_short_commit()}",
        }
        if citations_by_topic:
            manifest_kwargs["citations_by_topic"] = citations_by_topic
        if provenance_by_topic:
            manifest_kwargs["provenance_by_topic"] = provenance_by_topic
        narrative_metadata_by_topic = {
            str(entry.get("topic_id")): {
                "deep_analysis_path": entry.get("deep_analysis_path") or entry.get("markdown_path"),
                "slide_brief_path": entry.get("slide_brief_path"),
                "analysis_profile": entry.get("analysis_profile"),
                "narrative_model": entry.get("narrative_model"),
                "deep_validation": entry.get("deep_validation") or {},
            }
            for entry in narrative_out.citation_summaries
            if entry.get("topic_id")
        }
        if narrative_metadata_by_topic:
            manifest_kwargs["narrative_metadata_by_topic"] = narrative_metadata_by_topic
        manifest_path, manifest_sha256, content_sha256 = emit_analysis_manifest(
            workspace,
            **manifest_kwargs,
        )
        return {
            "status": "blocked",
            "reason": "narrative_source_pack_insufficient",
            "blockers": list(narrative_out.blockers),
            "artifacts": [
                {
                    "kind": "analysis_manifest",
                    "path": _relative_to_workspace(manifest_path, workspace_root),
                    "manifest_sha256": manifest_sha256,
                    "manifest_content_sha256": content_sha256,
                },
                *(
                    [
                        {
                            "kind": "analysis_citation_manifest",
                            "path": _relative_to_workspace(
                                narrative_out.citation_manifest_path,
                                workspace_root,
                            ),
                        }
                    ]
                    if narrative_out.citation_manifest_path is not None
                    else []
                ),
            ],
        }
    narrative_metadata_by_topic = {
        str(entry.get("topic_id")): {
            "deep_analysis_path": entry.get("deep_analysis_path") or entry.get("markdown_path"),
            "slide_brief_path": entry.get("slide_brief_path"),
            "analysis_profile": entry.get("analysis_profile"),
            "narrative_model": entry.get("narrative_model"),
            "deep_validation": entry.get("deep_validation") or {},
        }
        for entry in narrative_out.citation_summaries
        if entry.get("topic_id")
    }
    cov_out = step_coverage(
        CoverageInputs(
            analysis_dir=analysis_dir,
            presentations_dir=presentations_dir,
            overwrite_existing=overwrite_existing,
        )
    )
    gap_out = step_gap_fill(
        GapFillInputs(
            mapping_path=cov_out.mapping_path,
            analysis_dir=analysis_dir,
            presentations_dir=presentations_dir,
            topic_corrections=None,
            overwrite_existing=overwrite_existing,
        )
    )
    qa_out = step_qa(QAInputs(analysis_dir=analysis_dir, overwrite_existing=overwrite_existing))
    tr_out = step_traceability(
        TraceabilityInputs(
            analysis_dir=analysis_dir,
            presentations_root=presentations_dir,
            output_dir=analysis_dir / "traceability_addenda",
            vector_store_dir=vs_out.chroma_db_path.parent,
            use_gpt_researcher=True,
            vector_top_k=5,
            overwrite_existing=overwrite_existing,
        ),
        researcher=runner.researcher,
        vector_store=runner.vector_store,
    )

    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    manifest_kwargs = {
        "generator": f"conference_pipeline.analysis.emit_manifest@{_short_commit()}",
    }
    if citations_by_topic:
        manifest_kwargs["citations_by_topic"] = citations_by_topic
    if provenance_by_topic:
        manifest_kwargs["provenance_by_topic"] = provenance_by_topic
    if narrative_metadata_by_topic:
        manifest_kwargs["narrative_metadata_by_topic"] = narrative_metadata_by_topic
    manifest_path, manifest_sha256, content_sha256 = emit_analysis_manifest(workspace, **manifest_kwargs)
    sub_step_digests = {
        "vector_store_setup": vs_out.digest,
        "gpt_researcher_hybrid": gp_out.digest,
        "coverage": cov_out.digest,
        "gap_fill": gap_out.digest,
        "qa": qa_out.digest,
        "traceability": tr_out.digest,
    }
    if narrative_out.status != "skipped":
        sub_step_digests["narrative_provenance"] = _combined_existing_path_digest(
            tuple(narrative_out.report_paths)
            + tuple(narrative_out.provenance_paths)
            + tuple(narrative_out.provenance_json_paths)
            + (
                (narrative_out.citation_manifest_path,)
                if narrative_out.citation_manifest_path is not None
                else ()
            )
        )
    return AnalyzeStageResult(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest_content_sha256=content_sha256,
        sub_step_digests=sub_step_digests,
    )


def _discover_topics(analysis_dir: Path) -> list[str]:
    return [
        path.stem
        for path in sorted(analysis_dir.glob("[0-9][0-9]_*.md"))
        if path.name not in {"coverage_report.md", "qa_report.md"}
    ]


def _manifest_topic_ids(workspace_root: Path) -> list[str]:
    """Read topic ids from the workspace ``analysis/analysis_manifest.json``."""
    manifest_path = Path(workspace_root) / "analysis" / "analysis_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        return []
    ids: list[str] = []
    for entry in topics:
        if isinstance(entry, dict):
            topic_id = str(entry.get("id") or "").strip()
            if topic_id:
                ids.append(topic_id)
    return ids


def _fallback_topic_ids(workspace_root: Path) -> list[str]:
    """Derive the analysis topic list for THIS workspace when no reports exist yet.

    Prefers the workspace analysis manifest; only falls back to the NeurIPS
    constant for the neurips2025 workspace, otherwise raises so a misconfigured
    workspace never silently inherits NeurIPS topics.
    """
    manifest_topics = _manifest_topic_ids(workspace_root)
    if manifest_topics:
        return manifest_topics
    try:
        slug = load_conference_identity(workspace_root).slug
    except ConferenceIdentityError:
        slug = Path(workspace_root).resolve().name
    if slug == "neurips2025":
        return list(NEURIPS_TOPIC_IDS)
    raise RuntimeError(
        "analysis topic discovery failed: no analysis/NN_*.md reports and no "
        f"analysis/analysis_manifest.json topics for workspace {slug!r}; "
        "run topic strategy / narrative stages before analysis, or populate the "
        "analysis manifest"
    )


def _short_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workflow.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "dev"
    return completed.stdout.strip() or "dev"


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _combined_existing_path_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.exists()):
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()
