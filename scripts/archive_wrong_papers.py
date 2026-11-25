#!/usr/bin/env python3
"""Archive wrong papers to a separate folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def archive_wrong_papers(
    report_path: Path,
    papers_dir: Path,
    archive_name: str = "archive_wrong",
    dry_run: bool = False
) -> dict:
    """Move wrong papers to archive folder."""

    with open(report_path) as f:
        data = json.load(f)

    archive_dir = papers_dir / archive_name

    results = {
        'moved': [],
        'kept': [],
        'not_found': [],
        'errors': []
    }

    # Get papers to archive (bad or warn status)
    to_archive = []
    to_keep = []

    for record in data['records']:
        status = record['status']
        file_path = Path(record['file_path'])

        # Check if it's been confirmed wrong
        confirmed_wrong = record.get('confirmed_wrong', False)

        if status == 'bad' or confirmed_wrong:
            to_archive.append(record)
        elif status == 'warn':
            # Keep warn papers but note them
            to_keep.append(record)
        else:
            to_keep.append(record)

    print(f"Papers to archive: {len(to_archive)}")
    print(f"Papers to keep: {len(to_keep)}")

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"Archive directory: {archive_dir}")

    # Move wrong papers
    for record in to_archive:
        file_path = Path(record['file_path'])

        if not file_path.exists():
            results['not_found'].append({
                'file': str(file_path),
                'expected_title': record.get('expected_title', 'unknown')
            })
            continue

        # Determine destination path (preserve topic subfolder structure)
        rel_path = file_path.relative_to(papers_dir) if file_path.is_relative_to(papers_dir) else file_path.name
        dest_path = archive_dir / rel_path

        try:
            if dry_run:
                print(f"  [DRY RUN] Would move: {file_path.name}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dest_path))
                print(f"  Moved: {file_path.name}")

            results['moved'].append({
                'from': str(file_path),
                'to': str(dest_path),
                'expected_title': record.get('expected_title', 'unknown'),
                'status': record['status']
            })
        except Exception as e:
            print(f"  ERROR moving {file_path.name}: {e}")
            results['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })

    # Record kept papers
    for record in to_keep:
        file_path = Path(record['file_path'])
        if file_path.exists():
            results['kept'].append({
                'file': str(file_path),
                'expected_title': record.get('expected_title', 'unknown'),
                'status': record['status']
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Moved to archive: {len(results['moved'])}")
    print(f"Kept in place: {len(results['kept'])}")
    print(f"Not found: {len(results['not_found'])}")
    print(f"Errors: {len(results['errors'])}")

    if dry_run:
        print("\n[DRY RUN] No files were moved")

    # Save results
    results_path = papers_dir / "archive_results.json"
    if not dry_run:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Archive wrong papers")
    parser.add_argument(
        "report_path",
        type=Path,
        help="Path to title_sanity_report_updated.json"
    )
    parser.add_argument(
        "papers_dir",
        type=Path,
        help="Directory containing papers"
    )
    parser.add_argument(
        "--archive-name",
        default="archive_wrong",
        help="Name of archive folder"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be moved without actually moving"
    )
    args = parser.parse_args()

    archive_wrong_papers(
        args.report_path,
        args.papers_dir,
        args.archive_name,
        args.dry_run
    )


if __name__ == "__main__":
    main()
