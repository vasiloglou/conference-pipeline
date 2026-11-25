#!/usr/bin/env python3
"""Fail-closed claim fidelity for the executive decks, one governed call per topic.

A topic whose claims and evidence are unchanged since a passing review is reused,
so a rebuild that only touches links or layout spends nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.claim_fidelity_review import TASK_ID  # noqa: E402
from conference_pipeline.presentation.executive_claim_fidelity import (  # noqa: E402
    review_topic, topic_claims, topic_evidence)
from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402


def _input_hashes(topic: dict, deck_dir: Path) -> dict:
    _, _, hashes = topic_evidence(deck_dir)
    claims = json.dumps(topic_claims(topic), sort_keys=True, ensure_ascii=False)
    return hashes | {'claims': hashlib.sha256(claims.encode()).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--digests', default='decks/executive_topic_digests.json')
    parser.add_argument('--report-output', default='decks/executive_claim_fidelity_report.json')
    parser.add_argument('--topics', default='', help='comma-separated topic ids (default: all)')
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    digests = json.loads((workspace / args.digests).read_text(encoding='utf-8'))
    output = workspace / args.report_output
    prior = {}
    if output.is_file():
        prior = {row.get('topic_id'): row for row in json.loads(output.read_text(encoding='utf-8')).get('topics', [])}
    wanted = {item.strip() for item in args.topics.split(',') if item.strip()}
    rows = []
    for topic in digests.get('topics') or []:
        topic_id = str(topic.get('topic_id'))
        deck_dir = workspace / 'decks' / topic_id
        previous = prior.get(topic_id) or {}
        if wanted and topic_id not in wanted:
            if previous:
                rows.append(previous)
            continue
        try:
            if previous.get('status') == 'passed' and previous.get('input_hashes') == _input_hashes(topic, deck_dir):
                rows.append({**previous, 'reused': True})
                continue
            with provider_receipt_context(workspace_root=workspace, stage_id='executive_claim_fidelity_review',
                                          task_id=TASK_ID, caller=__file__):
                row = review_topic(topic=topic, deck_dir=deck_dir)
        except Exception as exc:
            row = {'topic_id': topic_id, 'status': 'failed', 'findings': [],
                   'error': f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        print(f"{topic_id}: {row['status']} "
              f"({sum(f['severity'] == 'blocker' for f in row['findings'])} blocker, "
              f"{sum(f['severity'] == 'advisory' for f in row['findings'])} advisory)"
              + (f" {row['error']}" if row.get('error') else ''), file=sys.stderr)
    status = 'passed' if rows and all(row['status'] == 'passed' for row in rows) else 'failed'
    write_json_atomic(output, {'schema_version': 1, 'task_id': TASK_ID, 'status': status,
                               'digests': args.digests, 'topics': rows})
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
