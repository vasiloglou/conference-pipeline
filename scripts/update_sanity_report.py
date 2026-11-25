#!/usr/bin/env python3
"""Update the title sanity report with verification results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_sanity_report(
    original_report: Path,
    verification_results: Path,
    output_path: Path
) -> dict:
    """Update sanity report with verification corrections."""

    # Load original report
    with open(original_report) as f:
        report = json.load(f)

    # Load verification results
    with open(verification_results) as f:
        verification = json.load(f)

    # Create lookup for actually correct papers
    actually_correct = {
        r['file_path']: r for r in verification.get('actually_correct', [])
    }

    # Create lookup for needs_redownload papers
    needs_redownload = {
        r['file_path']: r for r in verification.get('needs_redownload', [])
    }

    # Update records
    updates = {
        'corrected_to_ok': 0,
        'confirmed_bad': 0,
        'unchanged': 0
    }

    for record in report['records']:
        file_path = record['file_path']

        if file_path in actually_correct:
            # This paper was marked bad but is actually correct
            verification_info = actually_correct[file_path]['verification']
            record['original_status'] = record['status']
            record['status'] = 'ok'
            record['verification_corrected'] = True
            record['verification'] = {
                'confidence': verification_info.get('confidence', 'unknown'),
                'actual_title': verification_info.get('actual_title', ''),
                'reason': verification_info.get('reason', '')
            }
            updates['corrected_to_ok'] += 1

        elif file_path in needs_redownload:
            # This paper is confirmed as wrong
            verification_info = needs_redownload[file_path]
            record['original_status'] = record['status']
            record['confirmed_wrong'] = True
            record['actual_paper'] = verification_info.get('actual_paper', '')
            record['verification'] = verification_info.get('verification', {})
            updates['confirmed_bad'] += 1

        else:
            updates['unchanged'] += 1

    # Update summary
    ok_count = sum(1 for r in report['records'] if r['status'] == 'ok')
    warn_count = sum(1 for r in report['records'] if r['status'] == 'warn')
    bad_count = sum(1 for r in report['records'] if r['status'] == 'bad')

    report['summary'] = {
        'total': len(report['records']),
        'ok': ok_count,
        'warn': warn_count,
        'bad': bad_count,
        'verification_updates': updates
    }

    # Save updated report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Updated report saved to: {output_path}")
    print(f"Summary:")
    print(f"  - Corrected to OK: {updates['corrected_to_ok']}")
    print(f"  - Confirmed bad: {updates['confirmed_bad']}")
    print(f"  - Unchanged: {updates['unchanged']}")
    print(f"  - Final counts: {ok_count} ok, {warn_count} warn, {bad_count} bad")

    return report


def main():
    parser = argparse.ArgumentParser(description="Update sanity report with verification results")
    parser.add_argument(
        "original_report",
        type=Path,
        help="Path to original title_sanity_report.json"
    )
    parser.add_argument(
        "verification_results",
        type=Path,
        help="Path to verification_full.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for updated report"
    )
    args = parser.parse_args()

    output_path = args.output or args.original_report.parent / "title_sanity_report_updated.json"
    update_sanity_report(args.original_report, args.verification_results, output_path)


if __name__ == "__main__":
    main()
