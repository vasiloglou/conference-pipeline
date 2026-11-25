#!/usr/bin/env python3
"""Normalize newly added PDFs and replace incomplete downloads."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


def slugify(title: str, max_len: int = 80) -> str:
    """Convert title to slug for filename."""
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug[:max_len].rstrip('-')


def get_file_hash(path: Path) -> str:
    """Get short hash of file content."""
    content = path.read_bytes()[:8192]
    return hashlib.sha256(content).hexdigest()[:8]


def normalize_new_pdfs(
    papers_dir: Path,
    dry_run: bool = False
) -> dict:
    """Find and normalize new PDFs, replacing incomplete ones."""

    results = {
        'replaced': [],
        'normalized': [],
        'skipped': [],
        'removed_duplicates': []
    }

    # Define mappings of new PDFs to their targets
    # Format: (new_pdf_pattern, target_topic_rank, expected_title, arxiv_id)
    pdf_mappings = [
        {
            'new_file': '34_More_effort_is_needed_to_pr.pdf',
            'target_pattern': '11_ethical_problems__rank24__more-effort',
            'expected_title': 'More Effort Is Needed to Protect Pedestrian Privacy in the Era of AI',
            'arxiv_id': None,  # OpenReview paper
        },
        {
            'new_file': '2505.23839v1.pdf',
            'target_pattern': '16_neurips_awards__rank19__attacks-against-dna',
            'expected_title': 'Attacks Against DNA Language Models with Pathogenic Guidance',
            'arxiv_id': '2505.23839',
        },
        {
            'new_file': 'Two Heads Are Better Than One_ Improving Search Effectiveness Through LLM-Generated Query Variants.pdf',
            'target_pattern': '20_small_language_models__rank19__two-heads',
            'expected_title': 'Two Heads Are Better Than One: Improving Search Effectiveness Through LLM-Generated Query Variants',
            'arxiv_id': None,
        },
        {
            'new_file': '2506.05340v1.pdf',
            'target_pattern': '02_genai_architectures__rank04__grafting',
            'expected_title': 'Grafting: Efficient Architecture Exploration via Diffusion Transformer',
            'arxiv_id': '2506.05340',
        },
        {
            'new_file': '16696_Antidistillation_Samplin.pdf',
            'target_pattern': '20_small_language_models__rank32__antidistillation',
            'expected_title': 'Antidistillation Sampling: Preventing Unauthorized Knowledge Transfer',
            'arxiv_id': None,
        },
    ]

    for mapping in pdf_mappings:
        new_file = papers_dir / mapping['new_file']
        if not new_file.exists():
            print(f"[SKIP] New file not found: {mapping['new_file']}")
            results['skipped'].append({
                'file': mapping['new_file'],
                'reason': 'not_found'
            })
            continue

        new_size = new_file.stat().st_size
        print(f"\n[PROCESSING] {mapping['new_file']} ({new_size:,} bytes)")

        # Find target file(s)
        target_pattern = mapping['target_pattern']
        target_files = list(papers_dir.glob(f"{target_pattern}*.pdf"))

        if not target_files:
            print(f"  -> No target found matching: {target_pattern}")
            # Create new normalized name
            topic_rank = target_pattern.split('__')[0] + '__' + target_pattern.split('__')[1]
            slug = slugify(mapping['expected_title'])
            file_hash = get_file_hash(new_file)
            new_name = f"{topic_rank}__{slug}__{file_hash}.pdf"
            target_path = papers_dir / new_name

            if not dry_run:
                shutil.copy(new_file, target_path)
                new_file.unlink()
            print(f"  -> Normalized to: {new_name}")
            results['normalized'].append({
                'from': str(new_file),
                'to': str(target_path),
                'title': mapping['expected_title']
            })
            continue

        # Find smallest target (likely incomplete)
        target_files.sort(key=lambda f: f.stat().st_size)
        smallest_target = target_files[0]
        smallest_size = smallest_target.stat().st_size

        print(f"  -> Found {len(target_files)} target(s), smallest: {smallest_size:,} bytes")

        # Replace if new file is significantly larger
        if new_size > smallest_size * 2:
            # Generate proper name
            topic_rank = smallest_target.name.split('__')[0] + '__' + smallest_target.name.split('__')[1]
            slug = slugify(mapping['expected_title'])
            file_hash = get_file_hash(new_file)
            new_name = f"{topic_rank}__{slug}__{file_hash}.pdf"
            target_path = papers_dir / new_name

            # Find files to remove before actually removing
            files_to_remove = [f for f in target_files if f.stat().st_size < new_size / 2]

            if not dry_run:
                # Remove all incomplete targets
                for old_target in files_to_remove:
                    print(f"  -> Removing incomplete: {old_target.name}")
                    old_target.unlink()
                    results['removed_duplicates'].append(str(old_target))

                # Copy and normalize new file
                shutil.copy(new_file, target_path)
                new_file.unlink()

            print(f"  -> Replaced with: {new_name}")
            results['replaced'].append({
                'from': str(new_file),
                'to': str(target_path),
                'replaced_files': [str(f) for f in files_to_remove],
                'title': mapping['expected_title'],
                'arxiv_id': mapping.get('arxiv_id')
            })
        else:
            print(f"  -> Target already has good version, removing duplicate")
            if not dry_run:
                new_file.unlink()
            results['skipped'].append({
                'file': mapping['new_file'],
                'reason': 'target_already_complete'
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Replaced: {len(results['replaced'])}")
    print(f"Normalized: {len(results['normalized'])}")
    print(f"Removed duplicates: {len(results['removed_duplicates'])}")
    print(f"Skipped: {len(results['skipped'])}")

    if dry_run:
        print("\n[DRY RUN] No files were modified")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Normalize new PDFs")
    parser.add_argument(
        "papers_dir",
        type=Path,
        help="Directory containing papers"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done"
    )
    args = parser.parse_args()

    normalize_new_pdfs(args.papers_dir, args.dry_run)


if __name__ == "__main__":
    main()
