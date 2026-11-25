#!/usr/bin/env python3
"""
Enhanced Vector Store Setup with Smart Chunking and Entity Tracking

Pinned reference strategy: this historical NeurIPS 2025 script documents the
chunking approach promoted into conference_pipeline.indexing.chunking. It is
not the canonical production owner command for future conferences; production
workflows should call the reusable indexing module via the
enhanced-indexing-vector-store stage.

This enhanced version includes:
1. Papers indexing (title + abstract together)
2. Smart transcript chunking using segments.json (preserves timestamps)
3. Enhanced slides chunking
4. Entity and topic tracking for trend analysis
5. Comprehensive metadata
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
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


class EntityTracker:
    """Track entities and topics encountered during indexing"""

    def __init__(self):
        self.entities = defaultdict(lambda: defaultdict(list))
        # entity_type -> entity_name -> [sources]

    def add_entity(self, entity_type: str, entity_name: str, source: str):
        """Add an entity with its source"""
        if source not in self.entities[entity_type][entity_name]:
            self.entities[entity_type][entity_name].append(source)

    def extract_entities_from_text(self, text: str, source: str):
        """Extract entities from text using patterns"""

        # Tools/Frameworks (common patterns)
        tools_patterns = [
            r'\b(PyTorch|TensorFlow|JAX|Hugging\s*Face|LangChain|LlamaIndex|ChromaDB|FAISS|Pinecone)\b',
            r'\b(GPT-\d+|BERT|T5|LLaMA|Mistral|Claude|Gemini|PaLM)\b',
            r'\b(Stable\s*Diffusion|DALL-E|Midjourney|ControlNet)\b',
            r'\b(Transformer|Attention|LSTM|GRU|CNN)\b',
        ]

        for pattern in tools_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.add_entity('tool', match.group(0), source)

        # Models/Architectures
        model_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:Former|Net|GAN|VAE|Transformer))\b',
            r'\b(Vision\s*Transformer|ViT|CLIP|BLIP)\b',
        ]

        for pattern in model_patterns:
            for match in re.finditer(pattern, text):
                self.add_entity('model', match.group(0), source)

        # Methods/Techniques
        method_patterns = [
            r'\b(fine-tuning|prompt\s*engineering|few-shot|zero-shot|chain-of-thought)\b',
            r'\b(RLHF|DPO|PPO|reinforcement\s*learning)\b',
            r'\b(distillation|quantization|pruning|compression)\b',
        ]

        for pattern in method_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.add_entity('method', match.group(0), source)

        # Topics
        topic_patterns = [
            r'\b(multimodal|vision-language|text-to-image|image-to-text)\b',
            r'\b(reasoning|planning|agent|autonomous)\b',
            r'\b(fairness|bias|safety|alignment|ethics)\b',
            r'\b(scaling\s*law|emergence|capability)\b',
        ]

        for pattern in topic_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.add_entity('topic', match.group(0), source)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked entities"""
        stats = {}

        for entity_type, entities in self.entities.items():
            stats[entity_type] = {
                'total_unique': len(entities),
                'top_20': sorted(
                    [(name, len(sources)) for name, sources in entities.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            }

        return stats

    def save(self, output_path: Path):
        """Save entity tracking data"""
        data = {
            'entities': dict(self.entities),
            'statistics': self.get_statistics()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def load_papers(papers_file: Path, entity_tracker: EntityTracker) -> List[Dict[str, Any]]:
    """
    Load papers from papers.json

    Returns list of paper chunks (title + abstract together)
    """
    if not papers_file.exists():
        print(f"Papers file not found: {papers_file}")
        return []

    with open(papers_file, 'r') as f:
        data = json.load(f)

    papers_data = data.get('papers', [])
    print(f"✓ Found {len(papers_data)} papers")

    chunks = []

    for paper in papers_data:
        paper_id = paper.get('id', 'unknown')
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        authors = paper.get('authors', [])
        keywords = paper.get('keywords', [])
        venue = paper.get('venue', '')
        track = paper.get('track', '')

        # Create single chunk: title + abstract
        content = f"Title: {title}\n\nAbstract: {abstract}"

        # Extract entities
        entity_tracker.extract_entities_from_text(
            content,
            f"paper_{paper_id}"
        )

        # Track keywords as topics
        for keyword in keywords:
            entity_tracker.add_entity('topic', keyword, f"paper_{paper_id}")

        chunks.append({
            'id': f"paper_{paper_id}",
            'text': content,
            'metadata': {
                'type': 'paper',
                'paper_id': paper_id,
                'title': title,
                'authors': ', '.join(authors) if authors else '',
                'venue': venue,
                'track': track,
                'keywords': ', '.join(keywords) if keywords else '',
                'url': paper.get('openreview_url', paper.get('pdf_url', ''))
            }
        })

    return chunks


def load_presentations(presentations_dir: Path, entity_tracker: EntityTracker) -> List[Dict[str, Any]]:
    """Load all presentation data with enhanced metadata"""
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
            'segments': [],
            'slides': []
        }

        # Load meta.json
        meta_file = pres_dir / 'meta.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                pres_data['meta'] = json.load(f)

        # Load segments.json (better than raw transcript)
        segments_file = pres_dir / 'transcripts' / 'segments.json'
        if segments_file.exists():
            with open(segments_file, 'r') as f:
                pres_data['segments'] = json.load(f)

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


def create_transcript_chunks(
    presentation: Dict[str, Any],
    entity_tracker: EntityTracker,
    chunk_duration: float = 120.0  # 2 minutes
) -> List[Dict[str, Any]]:
    """
    Create chunks from transcript segments, preserving timestamps.

    Groups segments into ~2-minute chunks to preserve semantic continuity.
    """
    chunks = []
    segments = presentation.get('segments', [])

    if not segments:
        return chunks

    meta = presentation['meta']
    pres_id = presentation['id']
    title = meta.get('title', pres_id)

    current_chunk = []
    chunk_start = None
    chunk_end = None
    chunk_text_parts = []

    for segment in segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '')

        if chunk_start is None:
            chunk_start = start

        # Check if adding this segment would exceed chunk duration
        if chunk_start is not None and (end - chunk_start) > chunk_duration:
            # Save current chunk
            if chunk_text_parts:
                chunk_text = " ".join(chunk_text_parts)

                # Extract entities
                entity_tracker.extract_entities_from_text(
                    chunk_text,
                    f"presentation_{pres_id}"
                )

                chunks.append({
                    'id': f"{pres_id}_transcript_{len(chunks)}",
                    'text': f"Presentation: {title}\nTimestamp: {format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}\n\n{chunk_text}",
                    'metadata': {
                        'presentation_id': pres_id,
                        'title': title,
                        'type': 'transcript',
                        'track': meta.get('track', ''),
                        'url': meta.get('neurips_url', ''),
                        'timestamp_start': chunk_start,
                        'timestamp_end': chunk_end,
                        'timestamp_range': f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}"
                    }
                })

            # Start new chunk
            chunk_start = start
            chunk_text_parts = [text]
        else:
            chunk_text_parts.append(text)

        chunk_end = end

    # Add remaining chunk
    if chunk_text_parts:
        chunk_text = " ".join(chunk_text_parts)

        entity_tracker.extract_entities_from_text(
            chunk_text,
            f"presentation_{pres_id}"
        )

        chunks.append({
            'id': f"{pres_id}_transcript_{len(chunks)}",
            'text': f"Presentation: {title}\nTimestamp: {format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}\n\n{chunk_text}",
            'metadata': {
                'presentation_id': pres_id,
                'title': title,
                'type': 'transcript',
                'track': meta.get('track', ''),
                'url': meta.get('neurips_url', ''),
                'timestamp_start': chunk_start,
                'timestamp_end': chunk_end,
                'timestamp_range': f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}"
            }
        })

    return chunks


def create_slide_chunks(
    presentation: Dict[str, Any],
    entity_tracker: EntityTracker
) -> List[Dict[str, Any]]:
    """
    Create chunks from slides (one chunk per slide).

    Ensures slide title and content stay together.
    """
    chunks = []
    meta = presentation['meta']
    pres_id = presentation['id']
    title = meta.get('title', pres_id)

    for slide in presentation['slides']:
        slide_text_data = slide.get('slide_text', {})
        slide_idx = slide.get('slide_index', 0)
        timestamp = slide.get('timestamp_sec', None)

        # Build slide content
        slide_content_parts = []

        if slide_text_data.get('title'):
            slide_content_parts.append(f"Slide Title: {slide_text_data['title']}")

        if slide_text_data.get('equations'):
            slide_content_parts.append("Equations:")
            for eq in slide_text_data['equations']:
                slide_content_parts.append(f"  {eq}")

        if slide_text_data.get('bullet_points'):
            slide_content_parts.append("Content:")
            for bullet in slide_text_data['bullet_points']:
                slide_content_parts.append(f"• {bullet}")

        if slide_text_data.get('other_text'):
            other_text = slide_text_data['other_text']
            if isinstance(other_text, list):
                # Sometimes other_text is a list of strings
                slide_content_parts.extend(str(item) for item in other_text if item)
            else:
                slide_content_parts.append(str(other_text))

        if slide.get('key_concepts'):
            slide_content_parts.append("Key Concepts:")
            for concept in slide['key_concepts']:
                slide_content_parts.append(
                    f"• {concept.get('concept', '')}: {concept.get('explanation', '')}"
                )

        # Include visual_elements for better searchability (e.g., MLX, tool logos, diagrams)
        if slide.get('visual_elements'):
            visual_parts = []
            for ve in slide['visual_elements']:
                ve_type = ve.get('type', '')
                ve_desc = ve.get('description', '')
                ve_data = ve.get('data', '')
                ve_labels = ve.get('labels', [])

                if ve_desc:
                    visual_parts.append(ve_desc)
                if ve_data:
                    visual_parts.append(ve_data)
                if ve_labels:
                    # Labels often contain tool names like "MLX", "LM Studio", etc.
                    visual_parts.append(f"Labels: {', '.join(ve_labels)}")

            if visual_parts:
                slide_content_parts.append("Visual Elements:")
                for vp in visual_parts:
                    slide_content_parts.append(f"• {vp}")

        if not slide_content_parts:
            continue

        slide_content = "\n".join(slide_content_parts)

        # Extract entities
        entity_tracker.extract_entities_from_text(
            slide_content,
            f"presentation_{pres_id}_slide_{slide_idx}"
        )

        # Full text for embedding
        full_text = f"Presentation: {title}\nSlide {slide_idx}"
        if timestamp is not None:
            full_text += f" (Timestamp: {format_timestamp(timestamp)})"
        full_text += f"\n\n{slide_content}"

        chunks.append({
            'id': f"{pres_id}_slide_{slide_idx}",
            'text': full_text,
            'metadata': {
                'presentation_id': pres_id,
                'title': title,
                'type': 'slide',
                'slide_index': slide_idx,
                'track': meta.get('track', ''),
                'url': meta.get('neurips_url', ''),
                'timestamp': timestamp,
                'timestamp_formatted': format_timestamp(timestamp) if timestamp else None
            }
        })

    return chunks


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    if seconds is None:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metadata for ChromaDB - remove None values and convert to valid types.
    ChromaDB only accepts: bool, int, float, str
    """
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue  # Skip None values
        elif isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
        else:
            # Convert other types to string
            cleaned[key] = str(value)
    return cleaned


def create_vector_store(
    papers: List[Dict[str, Any]],
    presentations: List[Dict[str, Any]],
    entity_tracker: EntityTracker,
    output_dir: Path
):
    """
    Create a persistent ChromaDB vector store from all sources.
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
        metadata={"description": "NeurIPS 2025: papers, presentations, transcripts, and slides"}
    )

    print(f"Created collection: {collection_name}")

    total_chunks = 0

    # Index papers
    print(f"\nIndexing {len(papers)} papers...")
    if papers:
        ids = [chunk['id'] for chunk in papers]
        documents = [chunk['text'] for chunk in papers]
        metadatas = [clean_metadata(chunk['metadata']) for chunk in papers]

        # Add in batches
        batch_size = 100
        for j in range(0, len(papers), batch_size):
            batch_ids = ids[j:j+batch_size]
            batch_docs = documents[j:j+batch_size]
            batch_meta = metadatas[j:j+batch_size]

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
            print(f"  Indexed papers {j+1}-{min(j+batch_size, len(papers))}/{len(papers)}")

        total_chunks += len(papers)

    # Index presentations
    print(f"\nIndexing {len(presentations)} presentations...")
    for i, pres in enumerate(presentations, 1):
        print(f"Processing [{i}/{len(presentations)}]: {pres['id']}")

        # Create chunks from transcripts
        transcript_chunks = create_transcript_chunks(pres, entity_tracker)

        # Create chunks from slides
        slide_chunks = create_slide_chunks(pres, entity_tracker)

        all_chunks = transcript_chunks + slide_chunks

        if all_chunks:
            ids = [chunk['id'] for chunk in all_chunks]
            documents = [chunk['text'] for chunk in all_chunks]
            metadatas = [clean_metadata(chunk['metadata']) for chunk in all_chunks]

            # Add to collection in batches
            batch_size = 100
            for j in range(0, len(all_chunks), batch_size):
                batch_ids = ids[j:j+batch_size]
                batch_docs = documents[j:j+batch_size]
                batch_meta = metadatas[j:j+batch_size]

                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )

            total_chunks += len(all_chunks)
            print(f"  Added {len(transcript_chunks)} transcript chunks, {len(slide_chunks)} slide chunks")

    print(f"\n✓ Created vector store with {total_chunks} chunks")
    print(f"  Location: {chroma_dir}")

    # Save metadata
    metadata = {
        'collection_name': collection_name,
        'total_papers': len(papers),
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
    print("Enhanced Vector Store Setup")
    print("=" * 60)
    print()

    # Setup environment
    print("Step 1: Loading environment variables...")
    setup_environment()
    print()

    # Define paths
    project_root = Path(__file__).parent.parent
    papers_file = project_root / 'conferences' / 'neurips2025' / 'data' / 'papers.json'
    presentations_dir = project_root / 'conferences' / 'neurips2025' / 'data' / 'presentations'
    output_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'

    # Initialize entity tracker
    entity_tracker = EntityTracker()

    # Load papers
    print("Step 2: Loading papers...")
    papers = load_papers(papers_file, entity_tracker)
    print()

    # Load presentations
    print("Step 3: Loading presentations...")
    presentations = load_presentations(presentations_dir, entity_tracker)
    print(f"✓ Loaded {len(presentations)} presentations")
    print()

    if not papers and not presentations:
        print("No data found. Exiting.")
        return

    # Create vector store
    print("Step 4: Creating persistent vector store...")
    print("(This will take 10-15 minutes - creating embeddings...)")
    create_vector_store(papers, presentations, entity_tracker, output_dir)
    print()

    # Save entity tracking data
    print("Step 5: Saving entity tracking data...")
    entity_file = output_dir / 'entity_topic_index.json'
    entity_tracker.save(entity_file)

    stats = entity_tracker.get_statistics()
    print(f"✓ Tracked entities:")
    for entity_type, data in stats.items():
        print(f"  - {entity_type}: {data['total_unique']} unique entities")

    print(f"✓ Saved entity index to {entity_file}")
    print()

    # Print summary
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print(f"Vector store location: {output_dir / 'chroma_db'}")
    print(f"Entity index: {entity_file}")
    print()
    print("The enhanced vector store includes:")
    print(f"  - {len(papers)} papers (title + abstract)")
    print(f"  - {len(presentations)} presentations (transcripts + slides)")
    print("  - Entity and topic tracking for trend analysis")
    print("  - Timestamp preservation in transcripts")
    print()


if __name__ == '__main__':
    main()
