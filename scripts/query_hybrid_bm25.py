#!/usr/bin/env python3
"""
True Hybrid Search: BM25 (keyword) + Vector (semantic) with Reciprocal Rank Fusion

This script solves the exact-match problem where semantic search misses short terms
like "MLX", "Q#", "WarpDrive" because they get embedded to similar vector spaces
as unrelated terms.

The solution combines:
1. BM25 (keyword search) - exact term matching using TF-IDF
2. Vector search (semantic) - concept/meaning matching using embeddings
3. Reciprocal Rank Fusion (RRF) - combines rankings from both methods

Usage:
    python query_hybrid_bm25.py "MLX"                  # Finds exact "MLX" matches
    python query_hybrid_bm25.py "Apple on-device ML"   # Semantic concept matching
    python query_hybrid_bm25.py --rebuild              # Rebuild BM25 index
"""

import os
import sys
import json
import pickle
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_env():
    """Load environment variables"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    if line.startswith('export '):
                        line = line[7:]
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")


class HybridSearcher:
    """
    Combines BM25 keyword search with ChromaDB vector search using Reciprocal Rank Fusion.
    """

    def __init__(self, chroma_dir: Path, bm25_cache_dir: Path):
        self.chroma_dir = chroma_dir
        self.bm25_cache_dir = bm25_cache_dir
        self.bm25_cache_dir.mkdir(exist_ok=True)

        self.bm25_index_path = self.bm25_cache_dir / "bm25_index.pkl"
        self.docs_cache_path = self.bm25_cache_dir / "docs_cache.pkl"

        self.bm25 = None
        self.doc_ids = None
        self.doc_contents = None
        self.doc_metadata = None

        # ChromaDB
        self.collection = None

    def _get_chromadb_collection(self):
        """Get or initialize ChromaDB collection"""
        if self.collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                self.collection = client.get_collection("neurips_presentations")
            except Exception as e:
                print(f"Error connecting to ChromaDB: {e}")
                raise
        return self.collection

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Lowercase, split on non-alphanumeric, filter short tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return [t for t in tokens if len(t) >= 2]

    def build_bm25_index(self, force_rebuild: bool = False):
        """Build BM25 index from all ChromaDB documents"""
        from rank_bm25 import BM25Okapi

        # Check if cache exists and is recent
        if not force_rebuild and self.bm25_index_path.exists() and self.docs_cache_path.exists():
            cache_age = datetime.now().timestamp() - self.bm25_index_path.stat().st_mtime
            if cache_age < 86400:  # Less than 1 day old
                print(f"Loading cached BM25 index (age: {cache_age/3600:.1f} hours)...")
                self._load_cache()
                return

        print("Building BM25 index from ChromaDB...")
        collection = self._get_chromadb_collection()

        # Get ALL documents from ChromaDB
        # ChromaDB limits to 10k per query, so we need to paginate
        all_docs = []
        all_ids = []
        all_metadata = []

        # Get total count
        count = collection.count()
        print(f"Total documents in ChromaDB: {count}")

        batch_size = 5000
        offset = 0

        while offset < count:
            print(f"  Fetching {offset}-{min(offset + batch_size, count)}...")
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )

            if not results['ids']:
                break

            all_ids.extend(results['ids'])
            all_docs.extend(results['documents'])
            all_metadata.extend(results['metadatas'])

            offset += batch_size

        print(f"Loaded {len(all_docs)} documents")

        # Tokenize all documents
        print("Tokenizing documents for BM25...")
        tokenized_docs = [self._tokenize(doc) for doc in all_docs]

        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_ids = all_ids
        self.doc_contents = all_docs
        self.doc_metadata = all_metadata

        # Cache the index
        print("Caching BM25 index...")
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'doc_ids': self.doc_ids,
            }, f)

        with open(self.docs_cache_path, 'wb') as f:
            pickle.dump({
                'doc_contents': self.doc_contents,
                'doc_metadata': self.doc_metadata,
            }, f)

        print(f"BM25 index built and cached ({len(all_docs)} documents)")

    def _load_cache(self):
        """Load cached BM25 index"""
        with open(self.bm25_index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.doc_ids = data['doc_ids']

        with open(self.docs_cache_path, 'rb') as f:
            data = pickle.load(f)
            self.doc_contents = data['doc_contents']
            self.doc_metadata = data['doc_metadata']

    def search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[str, float, str, dict]]:
        """
        Search using BM25 (keyword matching).

        Returns list of (doc_id, score, content, metadata)
        """
        if self.bm25 is None:
            self.build_bm25_index()

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's any match
                results.append((
                    self.doc_ids[idx],
                    scores[idx],
                    self.doc_contents[idx],
                    self.doc_metadata[idx]
                ))

        return results

    def search_vector(self, query: str, top_k: int = 20) -> List[Tuple[str, float, str, dict]]:
        """
        Search using ChromaDB vector embeddings (semantic matching).

        Returns list of (doc_id, score, content, metadata)
        """
        collection = self._get_chromadb_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Convert distance to similarity score (1 - distance)
                distance = results['distances'][0][i]
                similarity = 1 - distance

                output.append((
                    doc_id,
                    similarity,
                    results['documents'][0][i],
                    results['metadatas'][0][i]
                ))

        return output

    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        """
        Automatically determine BM25/Vector weights based on query characteristics.

        Short queries with exact terms (like "MLX", "OpenHands") benefit from higher BM25 weight.
        Longer conceptual queries benefit from higher vector weight.

        Returns:
            Tuple of (bm25_weight, vector_weight)
        """
        tokens = self._tokenize(query)
        word_count = len(tokens)
        avg_word_len = sum(len(t) for t in tokens) / max(1, word_count)

        # Short exact queries (1-2 words, often technical terms)
        if word_count <= 2:
            return (0.6, 0.4)  # Favor BM25 for exact matches

        # Medium queries (3-4 words)
        elif word_count <= 4:
            return (0.5, 0.5)  # Balanced

        # Long conceptual queries (5+ words)
        else:
            return (0.3, 0.7)  # Favor vector for semantic matching

    def search_hybrid(
        self,
        query: str,
        top_k: int = 15,
        bm25_weight: float = None,
        vector_weight: float = None,
        rrf_k: int = 60,
        auto_weights: bool = True
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and vector search using Reciprocal Rank Fusion.

        Args:
            query: Search query
            top_k: Number of final results to return
            bm25_weight: Weight for BM25 scores in RRF (None = auto-detect)
            vector_weight: Weight for vector scores in RRF (None = auto-detect)
            rrf_k: RRF constant (higher = smoother fusion)
            auto_weights: If True and weights not specified, auto-detect based on query

        Returns:
            List of dicts with doc_id, content, metadata, scores
        """
        # Determine weights
        if bm25_weight is None or vector_weight is None:
            if auto_weights:
                bm25_weight, vector_weight = self._get_adaptive_weights(query)
            else:
                bm25_weight, vector_weight = 0.5, 0.5  # Default to balanced

        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        vector_results = self.search_vector(query, top_k=top_k * 2)

        # Build ranking maps
        bm25_ranks = {doc_id: rank for rank, (doc_id, _, _, _) in enumerate(bm25_results)}
        vector_ranks = {doc_id: rank for rank, (doc_id, _, _, _) in enumerate(vector_results)}

        # Collect all unique doc_ids
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # Build content/metadata maps
        content_map = {}
        metadata_map = {}
        bm25_score_map = {}
        vector_score_map = {}

        for doc_id, score, content, metadata in bm25_results:
            content_map[doc_id] = content
            metadata_map[doc_id] = metadata
            bm25_score_map[doc_id] = score

        for doc_id, score, content, metadata in vector_results:
            content_map[doc_id] = content
            metadata_map[doc_id] = metadata
            vector_score_map[doc_id] = score

        # Calculate RRF scores
        # Key fix: Only add RRF contribution from methods that actually found the document
        # Previously, documents found by only one method got a poor default score from the
        # other method, which unfairly penalized exact keyword matches not in vector results
        rrf_scores = {}
        for doc_id in all_doc_ids:
            in_bm25 = doc_id in bm25_ranks
            in_vector = doc_id in vector_ranks

            rrf_score = 0.0

            # RRF formula: weight * 1 / (k + rank)
            if in_bm25:
                bm25_rank = bm25_ranks[doc_id]
                rrf_score += bm25_weight * (1 / (rrf_k + bm25_rank))

            if in_vector:
                vector_rank = vector_ranks[doc_id]
                rrf_score += vector_weight * (1 / (rrf_k + vector_rank))

            rrf_scores[doc_id] = rrf_score

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build final results
        results = []
        for doc_id, rrf_score in sorted_docs:
            results.append({
                'doc_id': doc_id,
                'content': content_map.get(doc_id, ''),
                'metadata': metadata_map.get(doc_id, {}),
                'rrf_score': rrf_score,
                'bm25_score': bm25_score_map.get(doc_id, 0),
                'vector_score': vector_score_map.get(doc_id, 0),
                'bm25_rank': bm25_ranks.get(doc_id, -1),
                'vector_rank': vector_ranks.get(doc_id, -1),
                'matched_by': self._get_match_source(doc_id, bm25_ranks, vector_ranks)
            })

        return results

    def _get_match_source(self, doc_id: str, bm25_ranks: dict, vector_ranks: dict) -> str:
        """Determine which method(s) found this document"""
        in_bm25 = doc_id in bm25_ranks
        in_vector = doc_id in vector_ranks

        if in_bm25 and in_vector:
            return "both"
        elif in_bm25:
            return "bm25_only"
        else:
            return "vector_only"


def format_result(result: Dict, index: int) -> str:
    """Format a single result for display"""
    meta = result['metadata']

    lines = [
        f"\n### {index}. {meta.get('title', 'Unknown')}",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Presentation ID | `{meta.get('presentation_id', 'N/A')}` |",
        f"| Chunk Type | {meta.get('chunk_type', 'N/A')} |",
        f"| Timestamp | {meta.get('timestamp', meta.get('timestamp_start', 'N/A'))} |",
        f"| Slide | {meta.get('slide_number', 'N/A')} |",
        f"| URL | {meta.get('url', 'N/A')} |",
        f"| **Matched By** | **{result['matched_by'].upper()}** |",
        f"| BM25 Score | {result['bm25_score']:.4f} (rank: {result['bm25_rank'] if result['bm25_rank'] >= 0 else 'N/A'}) |",
        f"| Vector Score | {result['vector_score']:.4f} (rank: {result['vector_rank'] if result['vector_rank'] >= 0 else 'N/A'}) |",
        f"| **RRF Score** | **{result['rrf_score']:.6f}** |",
        f"",
        f"**Excerpt:**",
        f"> {result['content'][:400]}..." if len(result['content']) > 400 else f"> {result['content']}",
        f""
    ]

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Hybrid Search: BM25 + Vector with Reciprocal Rank Fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Search for exact term (BM25 shines)
    python query_hybrid_bm25.py "MLX"

    # Search for concept (vector shines)
    python query_hybrid_bm25.py "Apple on-device machine learning framework"

    # Compare BM25 vs vector results
    python query_hybrid_bm25.py "WarpDrive" --compare

    # Rebuild BM25 index after re-indexing ChromaDB
    python query_hybrid_bm25.py --rebuild
        """
    )

    parser.add_argument('query', nargs='*', help='Search query')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild BM25 index')
    parser.add_argument('--compare', action='store_true', help='Show BM25 vs Vector comparison')
    parser.add_argument('--top-k', type=int, default=15, help='Number of results (default: 15)')
    parser.add_argument('--bm25-weight', type=float, default=None, help='BM25 weight (default: auto-detect based on query)')
    parser.add_argument('--vector-weight', type=float, default=None, help='Vector weight (default: auto-detect based on query)')
    parser.add_argument('--no-auto-weights', action='store_true', help='Disable auto weight detection, use 0.5/0.5')

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store' / 'chroma_db'
    bm25_cache_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store' / 'bm25_cache'

    load_env()

    # Initialize searcher
    searcher = HybridSearcher(chroma_dir, bm25_cache_dir)

    # Handle rebuild
    if args.rebuild:
        print("Forcing BM25 index rebuild...")
        searcher.build_bm25_index(force_rebuild=True)
        if not args.query:
            print("Done!")
            return

    # Get query
    if not args.query:
        parser.print_help()
        return

    query = " ".join(args.query)

    # Determine weights to use
    if args.bm25_weight is not None and args.vector_weight is not None:
        # User specified both weights
        bm25_weight, vector_weight = args.bm25_weight, args.vector_weight
        weight_source = "manual"
    elif args.no_auto_weights:
        # User disabled auto-detection
        bm25_weight, vector_weight = 0.5, 0.5
        weight_source = "default (0.5/0.5)"
    else:
        # Auto-detect based on query
        bm25_weight, vector_weight = searcher._get_adaptive_weights(query)
        weight_source = "auto"

    print("=" * 70)
    print("HYBRID SEARCH: BM25 (Keyword) + Vector (Semantic)")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"Weights: BM25={bm25_weight}, Vector={vector_weight} ({weight_source})")
    print(f"Top-K: {args.top_k}")
    print()

    # Compare mode - show separate results
    if args.compare:
        print("-" * 70)
        print("BM25 RESULTS (Keyword Matching)")
        print("-" * 70)

        bm25_results = searcher.search_bm25(query, top_k=10)
        if bm25_results:
            for i, (doc_id, score, content, meta) in enumerate(bm25_results, 1):
                print(f"\n{i}. [{score:.4f}] {meta.get('title', 'Unknown')}")
                print(f"   Slide: {meta.get('slide_number', 'N/A')}, ID: {meta.get('presentation_id', 'N/A')}")
                print(f"   Preview: {content[:150]}...")
        else:
            print("   No BM25 matches found")

        print()
        print("-" * 70)
        print("VECTOR RESULTS (Semantic Matching)")
        print("-" * 70)

        vector_results = searcher.search_vector(query, top_k=10)
        if vector_results:
            for i, (doc_id, score, content, meta) in enumerate(vector_results, 1):
                print(f"\n{i}. [{score:.4f}] {meta.get('title', 'Unknown')}")
                print(f"   Slide: {meta.get('slide_number', 'N/A')}, ID: {meta.get('presentation_id', 'N/A')}")
                print(f"   Preview: {content[:150]}...")
        else:
            print("   No vector matches found")

        print()

    # Hybrid search
    print("-" * 70)
    print("HYBRID RESULTS (RRF Fusion)")
    print("-" * 70)

    results = searcher.search_hybrid(
        query,
        top_k=args.top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        auto_weights=False  # We already computed weights above
    )

    if results:
        # Summary stats
        both_count = len([r for r in results if r['matched_by'] == 'both'])
        bm25_only = len([r for r in results if r['matched_by'] == 'bm25_only'])
        vector_only = len([r for r in results if r['matched_by'] == 'vector_only'])

        print(f"\nMatched by: BOTH={both_count}, BM25_ONLY={bm25_only}, VECTOR_ONLY={vector_only}")

        for i, result in enumerate(results, 1):
            print(format_result(result, i))
    else:
        print("No results found")

    # Save results
    output_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store' / 'reports'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query[:40].replace(" ", "_").replace("?", "").replace("/", "_").lower()
    output_file = output_dir / f"hybrid_{timestamp}_{safe_query}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'weights': {'bm25': bm25_weight, 'vector': vector_weight, 'source': weight_source},
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
