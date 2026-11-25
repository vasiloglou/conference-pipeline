#!/usr/bin/env python3
"""Update podcast ranked_papers.json files with correct paper sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_podcast_sources(
    sources_path: Path,
    podcasts_dir: Path,
    dry_run: bool = False
) -> dict:
    """Update podcast files with corrected paper sources."""

    # Load found papers with correct sources
    with open(sources_path) as f:
        data = json.load(f)

    # Group by topic
    by_topic = {}
    for paper in data['found']:
        topic = paper.get('topic_slug', 'unknown')
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(paper)

    results = {
        'updated_topics': [],
        'updated_papers': 0,
        'skipped_topics': [],
        'errors': []
    }

    # Process each topic
    for topic_slug, papers in by_topic.items():
        topic_dir = podcasts_dir / topic_slug
        ranked_papers_path = topic_dir / "paper_ranks" / "ranked_papers.json"

        if not ranked_papers_path.exists():
            print(f"[SKIP] {topic_slug}: ranked_papers.json not found")
            results['skipped_topics'].append(topic_slug)
            continue

        print(f"\n[{topic_slug}] Processing {len(papers)} papers...")

        try:
            # Load existing ranked papers
            with open(ranked_papers_path) as f:
                ranked_data = json.load(f)

            papers_list = ranked_data.get('papers', [])
            updates_made = 0

            # Create lookup by title (normalized)
            def normalize_title(t):
                return t.lower().strip()[:50]

            corrections = {}
            for p in papers:
                key = normalize_title(p['expected_title'])
                corrections[key] = {
                    'new_url': p.get('new_page_url', ''),
                    'new_pdf_url': p.get('new_pdf_url', ''),
                    'new_title': p.get('new_title', ''),
                    'source': p.get('new_source', ''),
                    'confidence': p.get('match_confidence', 'unknown')
                }

            # Update papers
            for ranked_paper in papers_list:
                title = ranked_paper.get('title', '')
                key = normalize_title(title)

                if key in corrections:
                    correction = corrections[key]

                    # Store original values
                    if 'original_url' not in ranked_paper:
                        ranked_paper['original_url'] = ranked_paper.get('url', '')
                        ranked_paper['original_pdf_url'] = ranked_paper.get('pdf_url', '')

                    # Update with new sources
                    if correction['new_url']:
                        ranked_paper['url'] = correction['new_url']
                    if correction['new_pdf_url']:
                        ranked_paper['pdf_url'] = correction['new_pdf_url']

                    ranked_paper['source_corrected'] = True
                    ranked_paper['correction_source'] = correction['source']
                    ranked_paper['correction_confidence'] = correction['confidence']

                    print(f"  Updated: {title[:50]}...")
                    print(f"    -> {correction['new_url'][:60] if correction['new_url'] else 'no URL'}...")
                    updates_made += 1

            if updates_made > 0:
                if not dry_run:
                    # Save updated file
                    with open(ranked_papers_path, 'w') as f:
                        json.dump(ranked_data, f, indent=2)
                    print(f"  Saved {updates_made} updates to {ranked_papers_path}")
                else:
                    print(f"  [DRY RUN] Would save {updates_made} updates")

                results['updated_topics'].append(topic_slug)
                results['updated_papers'] += updates_made

        except Exception as e:
            print(f"  ERROR: {e}")
            results['errors'].append({
                'topic': topic_slug,
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Topics updated: {len(results['updated_topics'])}")
    print(f"Papers updated: {results['updated_papers']}")
    print(f"Topics skipped: {len(results['skipped_topics'])}")
    print(f"Errors: {len(results['errors'])}")

    if dry_run:
        print("\n[DRY RUN] No files were modified")

    return results


def main():
    parser = argparse.ArgumentParser(description="Update podcast sources")
    parser.add_argument(
        "sources_path",
        type=Path,
        help="Path to paper_sources.json"
    )
    parser.add_argument(
        "--podcasts-dir",
        type=Path,
        default=Path("conferences/neurips2025/audio"),
        help="Path to podcasts directory"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be updated without modifying files"
    )
    args = parser.parse_args()

    update_podcast_sources(args.sources_path, args.podcasts_dir, args.dry_run)


if __name__ == "__main__":
    main()
