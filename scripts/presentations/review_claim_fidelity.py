#!/usr/bin/env python3
"""Fail-closed, one-pass source fidelity review after figures or narration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.claim_fidelity_review import TASK_ID, review_claim_fidelity
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--deck-dir', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['figures', 'narration'])
    parser.add_argument('--report-output', required=True, type=Path)
    args = parser.parse_args(argv)
    # The previous report for this phase lets unchanged, settled slides keep their verdict.
    prior = None
    if args.report_output.is_file():
        try:
            prior = json.loads(args.report_output.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            prior = None
    if isinstance(prior, dict) and prior.get('error'):
        prior = prior.get('last_verdict')
    try:
        with provider_receipt_context(workspace_root=args.workspace, stage_id='claim_fidelity_review', task_id=TASK_ID, caller=__file__):
            report = review_claim_fidelity(deck_dir=args.deck_dir, phase=args.phase, prior=prior)
    except Exception as exc:
        report = {'status': 'failed', 'task_id': TASK_ID, 'phase': args.phase,
                  'error': f'{type(exc).__name__}: {exc}', 'findings': [],
                  # An execution failure judged nothing; keep the last real verdict for the next run.
                  **({'last_verdict': prior} if isinstance(prior, dict) and not prior.get('error') else
                     {'last_verdict': prior['last_verdict']} if isinstance(prior, dict) and prior.get('last_verdict') else {})}
    write_json_atomic(args.report_output, report)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
