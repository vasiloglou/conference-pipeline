#!/usr/bin/env python3
"""
Configure GPT Researcher to use the persistent ChromaDB vector store

This script creates the configuration needed for GPT Researcher to use
the persistent vector store, combining the best of both approaches.
"""

import os
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conference_pipeline.model_governance.registry import default_model_for_provider


def load_env_from_environment():
    """Return the API keys set in the environment.

    Keys are never read from a shell profile: source the repository's .env first
    (set -a; source .env; set +a).
    """
    return {key: value for key, value in os.environ.items() if key.endswith("_API_KEY")}


def create_hybrid_config():
    """
    Create a configuration that combines GPT Researcher with ChromaDB vector store.
    """
    project_root = Path(__file__).parent.parent
    vector_store_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'
    chroma_dir = vector_store_dir / 'chroma_db'

    # Check if vector store exists
    if not chroma_dir.exists():
        print("Error: ChromaDB vector store not found!")
        print(f"Expected location: {chroma_dir}")
        print()
        print("Please run first:")
        print("  .venv/bin/python scripts/setup_vector_store.py")
        return None

    # Create config directory
    config_dir = vector_store_dir / 'gpt_researcher_config'
    config_dir.mkdir(exist_ok=True)

    # Create config file for GPT Researcher
    config = {
        "retriever": "custom",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "llm_provider": "openai",
        "fast_llm_model": default_model_for_provider("research.gpt_researcher_fast", "openai"),
        "smart_llm_model": default_model_for_provider("research.gpt_researcher_smart", "openai"),
        "strategic_llm_model": default_model_for_provider("research.gpt_researcher_smart", "openai"),
        "temperature": 0.3,
        "max_search_results": 10,
        "max_iterations": 3,
        "browse_chunk_max_length": 8000,
        "summary_token_limit": 1000,
        "total_words": 1000,
        "report_format": "markdown",
        "max_subtopics": 8,
        "chroma_db_path": str(chroma_dir),
        "collection_name": "neurips_presentations"
    }

    config_path = config_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created GPT Researcher config at: {config_path}")

    return config_path


def create_custom_retriever():
    """
    Create a custom retriever class for GPT Researcher that uses ChromaDB.
    """
    project_root = Path(__file__).parent.parent
    vector_store_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'

    retriever_dir = vector_store_dir / 'gpt_researcher_config'
    retriever_dir.mkdir(exist_ok=True)

    retriever_code = '''"""
Custom ChromaDB retriever for GPT Researcher
"""

import os
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("chromadb not installed. Run: pip install chromadb")


class ChromaDBRetriever:
    """
    Custom retriever that uses our persistent ChromaDB vector store.
    """

    def __init__(self, chroma_db_path: str, collection_name: str = "neurips_presentations"):
        """
        Initialize the ChromaDB retriever.

        Args:
            chroma_db_path: Path to the ChromaDB database
            collection_name: Name of the collection to use
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Setup embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-3-small"
        )

        # Get collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.openai_ef
        )

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant documents.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with content and metadata
        """
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results
        )

        # Format results for GPT Researcher
        formatted_results = []

        if results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'href': metadata.get('url', ''),
                    'title': metadata.get('title', 'Unknown'),
                    'body': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance
                })

        return formatted_results

    async def get_source_url(self, query: str, max_results: int = 10) -> List[str]:
        """
        Get source URLs for a query.

        Args:
            query: Search query
            max_results: Maximum number of URLs to return

        Returns:
            List of source URLs
        """
        results = await self.search(query, max_results)
        return [result.get('href', '') for result in results if result.get('href')]


async def search(query: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Main search function called by GPT Researcher.

    Args:
        query: Search query
        **kwargs: Additional arguments (chroma_db_path, collection_name, max_results)

    Returns:
        List of search results
    """
    # Get config from kwargs or use defaults
    chroma_db_path = kwargs.get('chroma_db_path')
    collection_name = kwargs.get('collection_name', 'neurips_presentations')
    max_results = kwargs.get('max_results', 10)

    if not chroma_db_path:
        # Try to find it relative to this file
        script_dir = Path(__file__).parent.parent
        chroma_db_path = str(script_dir / 'chroma_db')

    # Create retriever
    retriever = ChromaDBRetriever(chroma_db_path, collection_name)

    # Perform search
    results = await retriever.search(query, max_results)

    return results
'''

    retriever_path = retriever_dir / 'chroma_retriever.py'
    with open(retriever_path, 'w') as f:
        f.write(retriever_code)

    print(f"✓ Created custom ChromaDB retriever at: {retriever_path}")

    return retriever_path


def main():
    """Main execution function"""
    print("=" * 60)
    print("GPT Researcher + ChromaDB Integration Setup")
    print("=" * 60)
    print()

    # Load environment
    env_vars = load_env_from_environment()

    if 'OPENAI_API_KEY' in env_vars:
        os.environ['OPENAI_API_KEY'] = env_vars['OPENAI_API_KEY']
        print("✓ Loaded OPENAI_API_KEY")
    else:
        print("✗ OPENAI_API_KEY not set in the environment (source the repository .env)")

    if 'TAVILY_API_KEY' in env_vars:
        os.environ['TAVILY_API_KEY'] = env_vars['TAVILY_API_KEY']
        print("✓ Loaded TAVILY_API_KEY")
    else:
        print("✗ TAVILY_API_KEY not set in the environment (source the repository .env)")

    print()

    # Create configuration
    print("Step 1: Creating GPT Researcher configuration...")
    config_path = create_hybrid_config()

    if not config_path:
        return

    print()

    # Create custom retriever
    print("Step 2: Creating custom ChromaDB retriever...")
    retriever_path = create_custom_retriever()
    print()

    # Summary
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("GPT Researcher is now configured to use the persistent")
    print("ChromaDB vector store. You get:")
    print("  ✓ Persistent embeddings (no re-indexing)")
    print("  ✓ GPT Researcher's advanced features")
    print("  ✓ Comprehensive report generation")
    print("  ✓ Multi-source synthesis")
    print()
    print("To query, use:")
    print("  .venv/bin/python scripts/query_gpt_researcher_hybrid.py \"Your question\"")
    print()


if __name__ == '__main__':
    main()
