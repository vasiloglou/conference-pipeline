#!/usr/bin/env python3
"""Re-download papers from found sources."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def download_paper(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a paper from URL to output path."""
    try:
        # Handle arXiv PDF URLs
        if 'arxiv.org/abs/' in url:
            url = url.replace('/abs/', '/pdf/') + '.pdf'
        elif 'arxiv.org' in url and not url.endswith('.pdf'):
            url = url + '.pdf'

        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; paper-downloader)'}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()

            # Verify it's a PDF
            if not content.startswith(b'%PDF'):
                print(f"  Warning: Response doesn't look like a PDF")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(content)

            return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def redownload_papers(
    sources_path: Path,
    output_dir: Path,
    limit: int = None,
    dry_run: bool = False
) -> dict:
    """Re-download papers from found sources."""

    with open(sources_path) as f:
        data = json.load(f)

    found_papers = data.get('found', [])
    if limit:
        found_papers = found_papers[:limit]

    print(f"Re-downloading {len(found_papers)} papers...", flush=True)

    results = {
        'downloaded': [],
        'failed': [],
        'skipped': []
    }

    for i, paper in enumerate(found_papers):
        expected_title = paper['expected_title']
        pdf_url = paper.get('new_pdf_url')
        page_url = paper.get('new_page_url')
        old_path = Path(paper['file_path'])

        print(f"[{i+1}/{len(found_papers)}] {expected_title[:50]}...", flush=True)

        if not pdf_url:
            # Try to construct PDF URL from page URL
            if page_url:
                if 'arxiv.org/abs/' in page_url:
                    pdf_url = page_url.replace('/abs/', '/pdf/') + '.pdf'
                elif 'openreview.net/forum' in page_url:
                    forum_id = page_url.split('id=')[-1]
                    pdf_url = f"https://openreview.net/pdf?id={forum_id}"
            else:
                print(f"  -> No PDF URL available, skipping", flush=True)
                results['skipped'].append({
                    **paper,
                    'reason': 'no_pdf_url'
                })
                continue

        # Determine output path (replace old file)
        output_path = old_path

        if dry_run:
            print(f"  -> Would download from: {pdf_url}", flush=True)
            print(f"  -> Would save to: {output_path}", flush=True)
            results['downloaded'].append({
                **paper,
                'new_file_path': str(output_path),
                'dry_run': True
            })
            continue

        # Backup old file
        backup_path = old_path.with_suffix('.pdf.bak')
        if old_path.exists():
            old_path.rename(backup_path)

        # Download new file
        print(f"  -> Downloading from: {pdf_url[:60]}...", flush=True)
        success = download_paper(pdf_url, output_path)

        if success:
            print(f"  -> Downloaded successfully", flush=True)
            # Remove backup
            if backup_path.exists():
                backup_path.unlink()
            results['downloaded'].append({
                **paper,
                'new_file_path': str(output_path)
            })
        else:
            print(f"  -> Download failed, restoring backup", flush=True)
            # Restore backup
            if backup_path.exists():
                backup_path.rename(old_path)
            results['failed'].append({
                **paper,
                'reason': 'download_failed',
                'attempted_url': pdf_url
            })

        time.sleep(1)  # Rate limiting

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Downloaded: {len(results['downloaded'])}", flush=True)
    print(f"Failed: {len(results['failed'])}", flush=True)
    print(f"Skipped: {len(results['skipped'])}", flush=True)

    # Save results
    results_path = sources_path.parent / "redownload_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Re-download papers from found sources")
    parser.add_argument(
        "sources_path",
        type=Path,
        help="Path to paper_sources.json"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for downloaded papers (default: use original paths)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of papers to download"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    args = parser.parse_args()

    redownload_papers(
        args.sources_path,
        args.output_dir,
        args.limit,
        args.dry_run
    )


if __name__ == "__main__":
    main()
