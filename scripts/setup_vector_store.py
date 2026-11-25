#!/usr/bin/env python3
"""
Setup Persistent Vector Store for Conference Data

This script creates a persistent ChromaDB vector database from all presentations
that can be reused across multiple query sessions without re-indexing.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("Error: chromadb not installed. Run: pip install chromadb")
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
        print("✓ Loaded OPENAI_API_KEY")
    else:
        print("✗ OPENAI_API_KEY not set in the environment (source the repository .env)")

    if 'TAVILY_API_KEY' in env_vars:
        os.environ['TAVILY_API_KEY'] = env_vars['TAVILY_API_KEY']
        print("✓ Loaded TAVILY_API_KEY")
    else:
        print("✗ TAVILY_API_KEY not set in the environment (source the repository .env)")


def load_presentations(presentations_dir: Path) -> List[Dict[str, Any]]:
    """Load all presentation data"""
    presentations = []

    if not presentations_dir.exists():
        print(f"Presentations directory not found: {presentations_dir}")
        return presentations

    for pres_dir in sorted(presentations_dir.iterdir()):
        if not pres_dir.is_dir():
            continue

        pres_data = {
            'id': pres_dir.name,
            'path': str(pres_dir),
            'meta': {},
            'transcript': '',
            'slides': []
        }

        # Load meta.json
        meta_file = pres_dir / 'meta.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                pres_data['meta'] = json.load(f)

        # Load transcript
        transcript_file = pres_dir / 'transcripts' / 'transcript.txt'
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                pres_data['transcript'] = f.read()

        # Load OCR data from slides
        ocr_dir = pres_dir / 'ocr'
        if ocr_dir.exists():
            for ocr_file in sorted(ocr_dir.glob('slide_*.json')):
                try:
                    with open(ocr_file, 'r') as f:
                        slide_data = json.load(f)
                        pres_data['slides'].append(slide_data)
                except json.JSONDecodeError:
                    continue

        presentations.append(pres_data)

    return presentations


def create_document_chunks(presentation: Dict[str, Any], chunk_size: int = 2000) -> List[Dict[str, str]]:
    """
    Create chunks from a presentation for embedding.

    Each chunk contains a portion of content with metadata.
    """
    chunks = []
    meta = presentation['meta']
    pres_id = presentation['id']
    title = meta.get('title', pres_id)

    # Create metadata string
    metadata_str = f"Title: {title}\n"
    metadata_str += f"Track: {meta.get('track', 'N/A')}\n"
    metadata_str += f"NeurIPS URL: {meta.get('neurips_url', 'N/A')}\n\n"

    # Chunk 1: Metadata + beginning of transcript
    if presentation['transcript']:
        transcript = presentation['transcript']

        # Create chunks from transcript
        words = transcript.split()
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunk_text = metadata_str + " ".join(current_chunk)
                chunks.append({
                    'id': f"{pres_id}_transcript_{len(chunks)}",
                    'text': chunk_text,
                    'metadata': {
                        'presentation_id': pres_id,
                        'title': title,
                        'type': 'transcript',
                        'track': meta.get('track', ''),
                        'url': meta.get('neurips_url', '')
                    }
                })
                current_chunk = []
                current_size = 0

        # Add remaining chunk
        if current_chunk:
            chunk_text = metadata_str + " ".join(current_chunk)
            chunks.append({
                'id': f"{pres_id}_transcript_{len(chunks)}",
                'text': chunk_text,
                'metadata': {
                    'presentation_id': pres_id,
                    'title': title,
                    'type': 'transcript',
                    'track': meta.get('track', ''),
                    'url': meta.get('neurips_url', '')
                }
            })

    # Create chunks from slides
    for slide in presentation['slides']:
        slide_text_data = slide.get('slide_text', {})
        slide_idx = slide.get('slide_index', 0)

        slide_content = metadata_str
        slide_content += f"Slide {slide_idx}\n"

        if slide_text_data.get('title'):
            slide_content += f"Title: {slide_text_data['title']}\n\n"

        if slide_text_data.get('equations'):
            slide_content += "Equations:\n"
            for eq in slide_text_data['equations']:
                slide_content += f"  {eq}\n"
            slide_content += "\n"

        if slide_text_data.get('bullet_points'):
            for bullet in slide_text_data['bullet_points']:
                slide_content += f"• {bullet}\n"
            slide_content += "\n"

        if slide_text_data.get('other_text'):
            slide_content += f"{slide_text_data['other_text']}\n\n"

        if slide.get('key_concepts'):
            slide_content += "Key Concepts:\n"
            for concept in slide['key_concepts']:
                slide_content += f"• {concept.get('concept', '')}: {concept.get('explanation', '')}\n"
            slide_content += "\n"

        if len(slide_content) > len(metadata_str) + 20:  # Only add if has content
            chunks.append({
                'id': f"{pres_id}_slide_{slide_idx}",
                'text': slide_content,
                'metadata': {
                    'presentation_id': pres_id,
                    'title': title,
                    'type': 'slide',
                    'slide_index': slide_idx,
                    'track': meta.get('track', ''),
                    'url': meta.get('neurips_url', '')
                }
            })

    return chunks


def create_vector_store(presentations: List[Dict[str, Any]], output_dir: Path):
    """
    Create a persistent ChromaDB vector store from presentations.
    """
    print("Creating ChromaDB client...")

    # Setup persistent ChromaDB
    chroma_dir = output_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Create embedding function using OpenAI
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-embedding-3-small"
    )

    # Get or create collection
    collection_name = "neurips_presentations"

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"description": "NeurIPS 2025 presentations with transcripts and slides"}
    )

    print(f"Created collection: {collection_name}")

    # Process presentations and add to vector store
    total_chunks = 0

    for i, pres in enumerate(presentations, 1):
        print(f"Processing [{i}/{len(presentations)}]: {pres['id']}")

        chunks = create_document_chunks(pres)

        if chunks:
            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]

            # Add to collection in batches
            batch_size = 100
            for j in range(0, len(chunks), batch_size):
                batch_ids = ids[j:j+batch_size]
                batch_docs = documents[j:j+batch_size]
                batch_meta = metadatas[j:j+batch_size]

                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )

            total_chunks += len(chunks)

    print(f"\n✓ Created vector store with {total_chunks} chunks")
    print(f"  Location: {chroma_dir}")

    # Save metadata
    metadata = {
        'collection_name': collection_name,
        'total_presentations': len(presentations),
        'total_chunks': total_chunks,
        'embedding_model': 'text-embedding-3-small',
        'chroma_db_path': str(chroma_dir)
    }

    metadata_file = output_dir / 'vector_store_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_file}")

    return collection


def main():
    """Main execution function"""
    print("=" * 60)
    print("Vector Store Setup for Conference Pipeline")
    print("=" * 60)
    print()

    # Setup environment
    print("Step 1: Loading environment variables...")
    setup_environment()
    print()

    # Define paths
    project_root = Path(__file__).parent.parent
    presentations_dir = project_root / 'conferences' / 'neurips2025' / 'data' / 'presentations'
    output_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'

    # Load presentations
    print("Step 2: Loading presentation data...")
    presentations = load_presentations(presentations_dir)
    print(f"✓ Loaded {len(presentations)} presentations")
    print()

    if not presentations:
        print("No presentations found. Exiting.")
        return

    # Create vector store
    print("Step 3: Creating persistent vector store...")
    print("(This will take a few minutes - creating embeddings...)")
    create_vector_store(presentations, output_dir)
    print()

    # Print summary
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print(f"Vector store location: {output_dir / 'chroma_db'}")
    print()
    print("The vector store is now persistent and can be reused")
    print("across multiple sessions without re-indexing!")
    print()
    print("To query the vector store, use:")
    print("  .venv/bin/python scripts/query_vector_store.py \"Your question\"")
    print()


if __name__ == '__main__':
    main()
