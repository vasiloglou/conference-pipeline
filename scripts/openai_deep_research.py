"""
OpenAI Deep Research using the OpenAI Deep Research API

This script uses the OpenAI Responses API with deep-research models to perform
multi-step research. It can:
1) Run with web search only
2) Upload local presentation documents to OpenAI file search (all text or top-k)
3) Run in two phases (web-only + local-only)

Requirements:
    pip install openai chromadb

Environment Variables Required:
    OPENAI_API_KEY - Your OpenAI API key
"""

import argparse
import asyncio
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from conference_pipeline.model_governance.registry import default_model
from conference_pipeline.provider_receipts import (
    infer_conference_workspace,
    openai_responses_create,
)

# Load environment variables from .env file if present
load_dotenv()

# Configuration
BASE_DIR = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
INPUT_FILE = BASE_DIR / "analysis" / "20_small_language_models.md"
OUTPUT_DIR = BASE_DIR / "experimental_videos" / "20_small_language_models" / "openai-deep-research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_OUTPUT_FILE = OUTPUT_DIR / "openai_research_result.md"
SOURCES_FILE = OUTPUT_DIR / "research_sources.md"

# Configure OpenAI Deep Research model and local retrieval
DEFAULT_MODEL = default_model("research.deep_research")
MODEL = os.environ.get("OPENAI_DEEP_RESEARCH_MODEL", DEFAULT_MODEL)
MAX_RETRIES = int(os.environ.get("OPENAI_DEEP_RESEARCH_MAX_RETRIES", "3"))
POLL_INTERVAL_SECONDS = int(os.environ.get("OPENAI_DEEP_RESEARCH_POLL_INTERVAL", "10"))
MAX_POLL_SECONDS = int(os.environ.get("OPENAI_DEEP_RESEARCH_MAX_POLL_SECONDS", "1800"))
DEFAULT_LOCAL_DOCS_DIR = BASE_DIR / "data" / "presentations"
LOCAL_DOCS_DIR = Path(os.environ.get("OPENAI_DEEP_RESEARCH_LOCAL_DOCS_DIR", DEFAULT_LOCAL_DOCS_DIR))
LOCAL_TOP_K = int(os.environ.get("OPENAI_DEEP_RESEARCH_LOCAL_TOP_K", "12"))
LOCAL_MODE = os.environ.get("OPENAI_DEEP_RESEARCH_LOCAL_MODE", "all")
VECTOR_STORE_BATCH_SIZE = int(os.environ.get("OPENAI_DEEP_RESEARCH_VECTOR_BATCH", "100"))
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".srt",
    ".vtt",
}
INCLUDE_INPUT_FILE_IN_LOCAL = os.environ.get("OPENAI_DEEP_RESEARCH_INCLUDE_INPUT", "true").lower() in (
    "1",
    "true",
    "yes",
)
CLEANUP_OPENAI_FILES = os.environ.get("OPENAI_DEEP_RESEARCH_CLEANUP", "true").lower() in (
    "1",
    "true",
    "yes",
)


def setup_environment():
    """Verify required API keys are available."""

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Missing required API key: OPENAI_API_KEY\n"
            "Set it as an environment variable or in a .env file."
        )


def build_system_message(use_web: bool, use_local: bool) -> str:
    """Provide consistent structure and expectations for the report."""
    context_note = ""
    if use_local and not use_web:
        context_note = (
            "Use only the provided local documents. Do not introduce facts that "
            "are not supported by those documents."
        )
    elif use_local and use_web:
        context_note = (
            "Use both web sources and the provided local documents. Prioritize local "
            "documents for NeurIPS-specific details when available."
        )
    return (
        "You are a professional research analyst. Produce a structured, "
        "citation-rich report with clear headings, bullet points where helpful, "
        "and a concise executive summary. Use only verifiable sources. Include "
        "inline citations and avoid unsupported claims. "
        f"{context_note}".strip()
    )


def extract_report_and_citations(response):
    """Pull the final report text and citation annotations from the response."""
    for item in reversed(response.output):
        content = getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text:
                annotations = getattr(part, "annotations", []) or []
                return text, annotations
    raise ValueError("No report text found in response output.")


def with_retries(fn, action: str):
    """Retry transient OpenAI API errors with simple exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
            if attempt >= MAX_RETRIES:
                raise
            wait_seconds = min(2 ** attempt, 20)
            print(f"⚠️  {action} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            print(f"   Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def wait_for_completion(client: OpenAI, response_id: str):
    """Poll a background response until completion or timeout."""
    start_time = time.time()
    while True:
        response = with_retries(
            lambda: client.responses.retrieve(response_id),
            action="Polling response status",
        )
        status = getattr(response, "status", None)
        if not status or status == "completed":
            return response
        if status in ("failed", "cancelled"):
            error = getattr(response, "error", None)
            raise RuntimeError(f"Deep research failed (status={status}): {error}")

        elapsed = int(time.time() - start_time)
        if elapsed > MAX_POLL_SECONDS:
            raise TimeoutError(
                f"Deep research timed out after {MAX_POLL_SECONDS} seconds."
            )
        print(f"⏳ Status: {status} (elapsed {elapsed}s). Polling again in {POLL_INTERVAL_SECONDS}s...")
        time.sleep(POLL_INTERVAL_SECONDS)


def wait_for_vector_store_ready(client: OpenAI, vector_store_id: str):
    """Poll a vector store until it is ready for use."""
    start_time = time.time()
    while True:
        vector_store = with_retries(
            lambda: client.vector_stores.retrieve(vector_store_id),
            action="Polling vector store status",
        )
        status = getattr(vector_store, "status", None)
        if status == "completed":
            return vector_store
        if status == "expired":
            raise RuntimeError("Vector store expired before completing indexing.")

        elapsed = int(time.time() - start_time)
        if elapsed > MAX_POLL_SECONDS:
            raise TimeoutError(
                f"Vector store indexing timed out after {MAX_POLL_SECONDS} seconds."
            )
        print(
            f"⏳ Vector store status: {status} (elapsed {elapsed}s). "
            f"Polling again in {POLL_INTERVAL_SECONDS}s..."
        )
        time.sleep(POLL_INTERVAL_SECONDS)


def is_text_file(path: Path) -> bool:
    """Return True if the file looks like text content worth uploading."""
    return path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS


def collect_all_text_files(docs_dir: Path):
    """Collect all text-like files under the presentations directory."""
    if not docs_dir.exists():
        print(f"⚠️  Local docs directory not found at {docs_dir}; skipping local documents.")
        return []

    return [path for path in docs_dir.rglob("*") if is_text_file(path)]


def select_local_doc_paths(query: str, docs_dir: Path, top_k: int):
    """Select relevant local documents using the existing ChromaDB vector store."""
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
    except ImportError:
        print("⚠️  chromadb not installed; skipping local document selection.")
        return []

    project_root = Path(__file__).parent.parent
    vector_store_dir = project_root / "conferences" / "neurips2025" / "vector_store"
    chroma_dir = vector_store_dir / "chroma_db"
    if not chroma_dir.exists():
        print(f"⚠️  Vector store not found at {chroma_dir}; skipping local documents.")
        return []

    if not docs_dir.exists():
        print(f"⚠️  Local docs directory not found at {docs_dir}; skipping local documents.")
        return []

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    collection = client.get_collection(
        name="neurips_presentations",
        embedding_function=openai_ef
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    presentation_ids = []
    for metadata in results.get("metadatas", [[]])[0]:
        pres_id = metadata.get("presentation_id") if metadata else None
        if pres_id and pres_id not in presentation_ids:
            presentation_ids.append(pres_id)

    doc_paths = []
    for pres_id in presentation_ids:
        pres_dir = docs_dir / pres_id
        if pres_dir.exists() and pres_dir.is_dir():
            doc_paths.extend([path for path in pres_dir.rglob("*") if is_text_file(path)])
        else:
            txt_path = docs_dir / f"{pres_id}.txt"
            if txt_path.exists():
                doc_paths.append(txt_path)

    if INCLUDE_INPUT_FILE_IN_LOCAL and INPUT_FILE.exists():
        doc_paths.append(INPUT_FILE)

    deduped = []
    seen = set()
    for path in doc_paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)

    return deduped


def upload_local_documents(client: OpenAI, doc_paths):
    """Upload local documents to OpenAI and create a vector store."""
    if not doc_paths:
        return None, []

    file_ids = []
    try:
        for path in doc_paths:
            with open(path, "rb") as handle:
                file_obj = with_retries(
                    lambda: client.files.create(file=handle, purpose="assistants"),
                    action=f"Uploading {path.name}",
                )
                file_ids.append(file_obj.id)

        vector_store = with_retries(
            lambda: client.vector_stores.create(
                name="neurips-local-docs",
            ),
            action="Creating vector store",
        )

        for i in range(0, len(file_ids), VECTOR_STORE_BATCH_SIZE):
            batch_ids = file_ids[i:i + VECTOR_STORE_BATCH_SIZE]
            with_retries(
                lambda: client.vector_stores.file_batches.create_and_poll(
                    vector_store_id=vector_store.id,
                    file_ids=batch_ids,
                ),
                action=f"Attaching file batch {i // VECTOR_STORE_BATCH_SIZE + 1}",
            )

        wait_for_vector_store_ready(client, vector_store.id)
        return vector_store.id, file_ids
    finally:
        pass


def cleanup_openai_files(client: OpenAI, vector_store_id: str, file_ids):
    """Remove uploaded files and vector store from OpenAI."""
    if not CLEANUP_OPENAI_FILES:
        return

    if vector_store_id:
        try:
            with_retries(
                lambda: client.vector_stores.delete(vector_store_id),
                action=f"Deleting vector store {vector_store_id}",
            )
        except Exception as exc:
            print(f"⚠️  Failed to delete vector store {vector_store_id}: {exc}")

    for file_id in file_ids:
        try:
            with_retries(
                lambda: client.files.delete(file_id),
                action=f"Deleting file {file_id}",
            )
        except APIStatusError as exc:
            if getattr(exc, "status_code", None) == 404:
                continue
            print(f"⚠️  Failed to delete file {file_id}: {exc}")
        except Exception as exc:
            print(f"⚠️  Failed to delete file {file_id}: {exc}")


def summarize_doc_set(doc_paths):
    """Print a quick summary of the local document set."""
    total_bytes = 0
    for path in doc_paths:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    size_mb = total_bytes / (1024 * 1024)
    print(f"✓ Selected {len(doc_paths)} local documents (~{size_mb:.1f} MB)")


async def run_deep_research(query: str, use_web: bool, vector_store_id: str | None) -> dict:
    """
    Run OpenAI Deep Research on a query.

    Args:
        query: The research query/topic
        use_web: Whether to enable web search
        vector_store_id: Optional vector store id for file_search

    Returns:
        Dictionary with report and sources
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_message = build_system_message(use_web=use_web, use_local=bool(vector_store_id))

    tools = []
    if use_web:
        tools.append({"type": "web_search_preview"})
    if vector_store_id:
        tools.append({"type": "file_search", "vector_store_ids": [vector_store_id]})

    print(f"\n{'='*60}")
    print("Starting Deep Research with OpenAI Deep Research API")
    print(f"{'='*60}")
    print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    response = openai_responses_create(client, caller=__file__,
        model=MODEL,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            },
        ],
        reasoning={"summary": "auto"},
        tools=tools,
        background=True,
    )

    status = getattr(response, "status", None)
    if status and status != "completed":
        print(f"Background task started (response_id={response.id}).")
        response = wait_for_completion(client, response.id)

    report_text, annotations = extract_report_and_citations(response)

    return {
        "report": report_text,
        "sources": annotations,
        "response_id": response.id,
        "model": MODEL,
    }


def save_results(results: dict, report_file: Path, sources_file: Path):
    """Save research results to files."""
    # Save main report
    with open(report_file, "w") as f:
        f.write(results["report"])
    print(f"\nReport saved to: {report_file}")

    # Save sources separately for easy reference
    with open(sources_file, "w") as f:
        f.write("# Research Sources\n\n")
        f.write(f"Model: {results.get('model')}\n")
        f.write(f"Response ID: {results.get('response_id')}\n")
        f.write(f"Total citations: {len(results['sources'])}\n\n")
        for i, source in enumerate(results["sources"], 1):
            title = getattr(source, "title", "Untitled")
            url = getattr(source, "url", "N/A")
            start = getattr(source, "start_index", "N/A")
            end = getattr(source, "end_index", "N/A")
            f.write(f"{i}. {title}\n")
            f.write(f"   URL: {url}\n")
            f.write(f"   Location: chars {start}-{end}\n")
    print(f"Sources saved to: {sources_file}")


def build_research_query(document_path: Path) -> str:
    """Build a research query from the input document."""
    with open(document_path, "r") as f:
        document_content = f.read()

    # Create a focused research query based on the document
    query = f"""
Conduct comprehensive research on Small Language Models (SLMs) and their applications,
based on the following research context from NeurIPS 2025:

{document_content}

Focus areas for deep research:
1. Find and verify specific papers mentioned - locate actual NeurIPS 2025 and arXiv papers
2. Identify key techniques: speculative decoding, knowledge distillation, quantization, pruning
3. Find recent (2024-2025) breakthrough papers on SLMs that may be missing
4. Research real-world deployment case studies and benchmarks
5. Investigate the efficiency vs capability tradeoffs in current SLM architectures

Provide verified citations with URLs for all papers and sources found.
"""
    return query


async def main():
    """Main entry point for deep research."""
    parser = argparse.ArgumentParser(description="OpenAI Deep Research runner")
    parser.add_argument("--two-phase", action="store_true", help="Run web-only and local-only phases")
    parser.add_argument("--no-web", action="store_true", help="Disable web search")
    parser.add_argument("--no-local", action="store_true", help="Disable local document upload")
    parser.add_argument("--local-top-k", type=int, default=LOCAL_TOP_K)
    parser.add_argument(
        "--local-mode",
        choices=("all", "topk"),
        default=LOCAL_MODE,
        help="Select all text files or top-k relevant presentations.",
    )
    parser.add_argument("--local-docs-dir", type=Path, default=LOCAL_DOCS_DIR)
    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --local-docs-dir to avoid falling back to Path.cwd() (repo root);
    # this tool's default paths are a legacy experimental directory that is
    # not under conferences/<slug>, in which case no workspace is inferred.
    inferred_workspace = infer_conference_workspace(args.local_docs_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "openai_deep_research")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    print("=" * 60)
    print("OpenAI Deep Research (via OpenAI Deep Research API)")
    print("=" * 60)

    # 1. Setup environment
    try:
        setup_environment()
        print("✓ Environment configured")
        print(f"  - MODEL: {MODEL}")
    except ValueError as e:
        print(f"✗ Environment setup failed:\n{e}")
        return

    # 2. Check input file
    if not INPUT_FILE.exists():
        print(f"✗ Input file not found: {INPUT_FILE}")
        return
    print(f"✓ Input file: {INPUT_FILE}")

    # 3. Build research query
    query = build_research_query(INPUT_FILE)
    print(f"✓ Research query built ({len(query)} chars)")

    use_web = not args.no_web
    use_local = not args.no_local

    if not use_web and not use_local:
        print("✗ Both web and local sources are disabled. Nothing to run.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vector_store_id = None
    uploaded_file_ids = []

    try:
        if args.two_phase:
            web_results = None
            local_results = None
            if use_web:
                print("\n[Phase A] Web-only research")
                web_results = await run_deep_research(query=query, use_web=True, vector_store_id=None)
                web_report_file = OUTPUT_DIR / "openai_research_result_web.md"
                web_sources_file = OUTPUT_DIR / "research_sources_web.md"
                save_results(web_results, web_report_file, web_sources_file)

            if use_local:
                print("\n[Phase B] Local-only research")
                if args.local_mode == "all":
                    doc_paths = collect_all_text_files(args.local_docs_dir)
                else:
                    doc_paths = select_local_doc_paths(query, args.local_docs_dir, args.local_top_k)
                if doc_paths:
                    summarize_doc_set(doc_paths)
                    vector_store_id, uploaded_file_ids = upload_local_documents(client, doc_paths)
                    local_results = await run_deep_research(
                        query=query,
                        use_web=False,
                        vector_store_id=vector_store_id,
                    )
                    local_report_file = OUTPUT_DIR / "openai_research_result_local.md"
                    local_sources_file = OUTPUT_DIR / "research_sources_local.md"
                    save_results(local_results, local_report_file, local_sources_file)
                else:
                    print("✗ No local documents selected; skipping local phase.")
            else:
                print("\n[Phase B] Local-only research disabled.")

            if web_results and local_results:
                with open(RESEARCH_OUTPUT_FILE, "w") as f:
                    f.write("# Web Research Report\n\n")
                    f.write(web_results["report"])
                    f.write("\n\n# Local Research Report\n\n")
                    f.write(local_results["report"])

                with open(SOURCES_FILE, "w") as f:
                    f.write("# Research Sources (Two-Phase)\n\n")
                    f.write("## Web Sources\n\n")
                    f.write(f"Model: {web_results.get('model')}\n")
                    f.write(f"Response ID: {web_results.get('response_id')}\n")
                    f.write(f"Total citations: {len(web_results['sources'])}\n\n")
                    for i, source in enumerate(web_results["sources"], 1):
                        title = getattr(source, "title", "Untitled")
                        url = getattr(source, "url", "N/A")
                        start = getattr(source, "start_index", "N/A")
                        end = getattr(source, "end_index", "N/A")
                        f.write(f"{i}. {title}\n")
                        f.write(f"   URL: {url}\n")
                        f.write(f"   Location: chars {start}-{end}\n")

                    f.write("\n## Local Sources\n\n")
                    f.write(f"Model: {local_results.get('model')}\n")
                    f.write(f"Response ID: {local_results.get('response_id')}\n")
                    f.write(f"Total citations: {len(local_results['sources'])}\n\n")
                    for i, source in enumerate(local_results["sources"], 1):
                        title = getattr(source, "title", "Untitled")
                        url = getattr(source, "url", "N/A")
                        start = getattr(source, "start_index", "N/A")
                        end = getattr(source, "end_index", "N/A")
                        f.write(f"{i}. {title}\n")
                        f.write(f"   URL: {url}\n")
                        f.write(f"   Location: chars {start}-{end}\n")
            elif web_results:
                save_results(web_results, RESEARCH_OUTPUT_FILE, SOURCES_FILE)
            elif local_results:
                save_results(local_results, RESEARCH_OUTPUT_FILE, SOURCES_FILE)
            else:
                print("✗ No results produced in two-phase mode.")
        else:
            if use_local:
                if args.local_mode == "all":
                    doc_paths = collect_all_text_files(args.local_docs_dir)
                else:
                    doc_paths = select_local_doc_paths(query, args.local_docs_dir, args.local_top_k)
                if doc_paths:
                    summarize_doc_set(doc_paths)
                    vector_store_id, uploaded_file_ids = upload_local_documents(client, doc_paths)
                else:
                    print("⚠️  No local documents selected; continuing with web-only.")
                    use_local = False

            print("\n[Phase] Combined research")
            results = await run_deep_research(
                query=query,
                use_web=use_web,
                vector_store_id=vector_store_id if use_local else None,
            )
            save_results(results, RESEARCH_OUTPUT_FILE, SOURCES_FILE)
    except Exception as e:
        print(f"✗ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        if uploaded_file_ids or vector_store_id:
            cleanup_openai_files(client, vector_store_id, uploaded_file_ids)

    # Summary
    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)
    if RESEARCH_OUTPUT_FILE.exists():
        report_text = RESEARCH_OUTPUT_FILE.read_text()
        print(f"Report length: {len(report_text)} characters")
    if SOURCES_FILE.exists():
        print(f"Sources file: {SOURCES_FILE}")
    print("\nOutput files:")
    print(f"  - Report: {RESEARCH_OUTPUT_FILE}")
    print(f"  - Sources: {SOURCES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
