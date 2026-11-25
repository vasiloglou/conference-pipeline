#!/usr/bin/env python3
"""Update markdown files with corrected paper sources."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def update_markdown_sources(
    sources_path: Path,
    md_dir: Path,
    dry_run: bool = False
) -> dict:
    """Update markdown files with corrected paper sources."""

    # Load found papers with correct sources
    with open(sources_path) as f:
        data = json.load(f)

    # Build lookup by normalized title
    def normalize_title(t):
        # Extract key words and acronyms
        return t.lower().strip()[:60]

    corrections = {}
    for paper in data['found']:
        key = normalize_title(paper['expected_title'])
        corrections[key] = {
            'new_url': paper.get('new_page_url', ''),
            'new_pdf_url': paper.get('new_pdf_url', ''),
            'new_title': paper.get('new_title', ''),
            'source': paper.get('new_source', ''),
            'confidence': paper.get('match_confidence', 'unknown'),
            'expected_title': paper['expected_title']
        }

    results = {
        'files_updated': [],
        'papers_updated': 0,
        'files_unchanged': [],
        'errors': []
    }

    # Find all markdown files
    md_files = list(md_dir.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files to check", flush=True)

    for md_file in md_files:
        try:
            content = md_file.read_text()
            original_content = content
            updates_in_file = 0

            # Look for paper titles and URLs
            # Pattern: ## N. Title or - **Title** or ### Title
            title_patterns = [
                r'^## \d+\.\s+(.+)$',  # ## 1. Title
                r'^### (.+)$',  # ### Title
                r'^\*\*(.+?)\*\*',  # **Title**
            ]

            # Find URLs that might need updating
            url_pattern = r'- \*\*URL:\*\*\s*(https?://[^\s]+)'

            lines = content.split('\n')
            current_title = None

            for i, line in enumerate(lines):
                # Check for title
                for pattern in title_patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        current_title = match.group(1).strip()
                        break

                # Check for URL line
                url_match = re.match(url_pattern, line)
                if url_match and current_title:
                    old_url = url_match.group(1)

                    # Check if this title needs correction
                    key = normalize_title(current_title)
                    if key in corrections:
                        correction = corrections[key]
                        new_url = correction['new_url']

                        if new_url and old_url != new_url:
                            # Check if it's an openalex search URL (definitely wrong)
                            if 'openalex.org/works?search=' in old_url or correction['confidence'] in ['high', 'medium']:
                                new_line = f"- **URL:** {new_url}"
                                lines[i] = new_line
                                updates_in_file += 1
                                print(f"  [{md_file.name}] Updated: {current_title[:40]}...", flush=True)
                                print(f"    Old: {old_url[:60]}...", flush=True)
                                print(f"    New: {new_url[:60]}...", flush=True)

            if updates_in_file > 0:
                new_content = '\n'.join(lines)
                if not dry_run:
                    md_file.write_text(new_content)
                results['files_updated'].append(str(md_file))
                results['papers_updated'] += updates_in_file
            else:
                results['files_unchanged'].append(str(md_file))

        except Exception as e:
            print(f"  ERROR in {md_file}: {e}", flush=True)
            results['errors'].append({
                'file': str(md_file),
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files updated: {len(results['files_updated'])}")
    print(f"Papers updated: {results['papers_updated']}")
    print(f"Files unchanged: {len(results['files_unchanged'])}")
    print(f"Errors: {len(results['errors'])}")

    if dry_run:
        print("\n[DRY RUN] No files were modified")

    return results


def main():
    parser = argparse.ArgumentParser(description="Update markdown files with correct sources")
    parser.add_argument(
        "sources_path",
        type=Path,
        help="Path to paper_sources.json"
    )
    parser.add_argument(
        "md_dir",
        type=Path,
        help="Directory containing markdown files to update"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be updated without modifying files"
    )
    args = parser.parse_args()

    update_markdown_sources(args.sources_path, args.md_dir, args.dry_run)


if __name__ == "__main__":
    main()
