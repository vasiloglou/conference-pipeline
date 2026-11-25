#!/usr/bin/env python3
"""
Query Persistent Vector Store

Query the persistent ChromaDB vector store created from conference presentations.
This script can be run multiple times without re-indexing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from conference_pipeline.model_governance import default_model
from conference_pipeline.provider_receipts import (
    openai_chat_completions_create,
)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("Error: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai not installed. Run: pip install openai")
    sys.exit(1)


def load_env_from_environment():
    """Return the API keys set in the environment.

    Keys are never read from a shell profile: source the repository's .env first
    (set -a; source .env; set +a).
    """
    return {key: value for key, value in os.environ.items() if key.endswith("_API_KEY")}


def setup_environment():
    """Setup environment variables"""
    env_vars = load_env_from_environment()

    if 'OPENAI_API_KEY' in env_vars:
        os.environ['OPENAI_API_KEY'] = env_vars['OPENAI_API_KEY']
        return True
    else:
        print("✗ OPENAI_API_KEY not set in the environment (source the repository .env)")
        return False


def query_vector_store(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    Query the persistent vector store.

    Args:
        query: The search query
        n_results: Number of results to return

    Returns:
        List of relevant document chunks with metadata
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    vector_store_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'
    chroma_dir = vector_store_dir / 'chroma_db'

    if not chroma_dir.exists():
        print(f"Error: Vector store not found at {chroma_dir}")
        print("Please run: .venv/bin/python scripts/setup_vector_store.py")
        sys.exit(1)

    # Load metadata
    metadata_file = vector_store_dir / 'vector_store_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"Vector store info:")
            print(f"  - {metadata['total_presentations']} presentations")
            print(f"  - {metadata['total_chunks']} document chunks")
            print(f"  - Model: {metadata['embedding_model']}")
            print()

    # Connect to persistent ChromaDB
    print(f"Connecting to vector store at: {chroma_dir}")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    # Setup embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-embedding-3-small"
    )

    # Get collection
    collection = client.get_collection(
        name="neurips_presentations",
        embedding_function=openai_ef
    )

    print(f"Querying: {query}")
    print(f"Retrieving top {n_results} results...")
    print()

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Format results
    formatted_results = []
    if results['documents']:
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'rank': i + 1,
                'document': doc,
                'metadata': metadata,
                'similarity_score': 1 - distance  # Convert distance to similarity
            })

    return formatted_results


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer using OpenAI based on retrieved context.

    Args:
        query: The user's question
        context_chunks: Retrieved document chunks from vector store

    Returns:
        Generated answer
    """
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Build context from top chunks
    context = ""
    sources = set()

    for chunk in context_chunks[:5]:  # Use top 5 chunks
        context += f"\n--- Excerpt from {chunk['metadata']['title']} ---\n"
        context += chunk['document']
        context += "\n\n"
        sources.add((chunk['metadata']['title'], chunk['metadata'].get('url', 'N/A')))

    # Create prompt
    system_prompt = """You are an AI assistant that answers questions about NeurIPS 2025 conference presentations.
Use the provided context from presentation transcripts and slides to answer questions accurately.
If you cannot answer based on the context, say so clearly.
Cite specific presentations when relevant."""

    user_prompt = f"""Context from NeurIPS 2025 presentations:

{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include specific details from the presentations when relevant."""

    # Generate response
    model = default_model("vector_store.query_answer")
    print(f"Generating answer using {model}...")
    response = openai_chat_completions_create(client, caller=__file__,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    answer = response.choices[0].message.content

    # Add sources
    sources_text = "\n\n## Sources:\n\n"
    for title, url in sorted(sources):
        sources_text += f"- **{title}**\n"
        if url != 'N/A':
            sources_text += f"  {url}\n"

    return answer + sources_text


def main():
    """Main execution function"""
    # D-QUICK-260715-rz1: this legacy tool is hardcoded to the neurips2025
    # workspace; establish a workspace-rooted receipt context so standalone
    # invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault(
        "PROVIDER_RECEIPT_WORKSPACE",
        str(Path(__file__).parent.parent / "conferences" / "neurips2025"),
    )
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "query_vector_store")

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    # Get query from command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Usage: python query_vector_store.py <your question>")
        print()
        print("Example queries:")
        print("  - What are the key innovations in autoregressive models?")
        print("  - How are diffusion models being applied?")
        print("  - What presentations discuss scaling laws?")
        print()
        sys.exit(1)

    print("=" * 60)
    print("Querying NeurIPS 2025 Vector Store")
    print("=" * 60)
    print()

    # Query vector store
    results = query_vector_store(query, n_results=10)

    if not results:
        print("No results found.")
        sys.exit(0)

    # Generate answer
    answer = generate_answer(query, results)

    # Display answer
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print()
    print(answer)
    print()

    # Save result
    output_dir = Path(__file__).parent.parent / "conferences" / "neurips2025" / "vector_store" / "queries"
    output_dir.mkdir(exist_ok=True)

    filename = query[:50].replace(" ", "_").replace("?", "").lower() + ".md"
    output_file = output_dir / filename

    with open(output_file, 'w') as f:
        f.write(f"# Query\n\n{query}\n\n")
        f.write(f"---\n\n")
        f.write(answer)
        f.write("\n\n---\n\n## Retrieved Context\n\n")
        for result in results[:5]:
            f.write(f"### {result['metadata']['title']} (Similarity: {result['similarity_score']:.3f})\n\n")
            f.write(f"{result['document'][:500]}...\n\n")

    print(f"Query saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
