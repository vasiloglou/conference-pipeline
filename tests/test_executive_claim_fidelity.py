"""Executive claims are checked against the topic deck's reviewed evidence (ICML 2026 takeaways)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.presentation import executive_claim_fidelity as fidelity

TOPIC = {
    'topic_id': '01_t', 'title': 'Topic one', 'central_takeaway': 'Loops matter.', 'one_liner': 'Loops.',
    'short': {'evidence_slide': {'headline': 'Agents run loops', 'synthesis': 'Rosetta is driven by an agent.'}},
    'long': {'slides': [{'headline': 'Honest loops', 'body': 'Log every test.',
                         'takeaway': 'Logging every test guarantees honest discovery',
                         'diagram': {'caption': 'Biased versus rigorous'}}]},
}


def _deck(tmp_path: Path) -> Path:
    deck = tmp_path / 'decks' / '01_t'
    deck.mkdir(parents=True)
    (deck / 'source_binding_report.json').write_text(json.dumps({'slides': [{'slide_id': 's1', 'visible_references': [
        {'label': 'SRC-001', 'title': 'Hypothesis Validation', 'authors': ['A'], 'contribution_note': 'Logs tests.'}]}]}))
    (deck / '6_presenter_script.ENGAGING.md').write_text('Logging every test reduces selection bias.')
    return deck


def _runner(verdict: dict, calls: list):
    def run(task_id, *, prompt, max_tokens, caller):
        calls.append((task_id, prompt))
        return SimpleNamespace(text=json.dumps(verdict), stop_reason='end', provider='openai', model='m')
    return run


def test_every_executive_claim_is_named_with_its_takeaway():
    refs = {claim['ref']: claim['text'] for claim in fidelity.topic_claims(TOPIC)}
    assert list(refs) == ['topic', 'short.evidence_slide', 'long.0']
    assert refs['long.0']['takeaway'] == 'Logging every test guarantees honest discovery'
    assert refs['long.0']['diagram'] == 'Biased versus rigorous'


def test_an_overclaimed_takeaway_fails_with_the_topic_evidence_in_the_prompt(tmp_path):
    calls: list = []
    finding = {'ref': 'long.0', 'field': 'takeaway', 'severity': 'blocker', 'code': 'overclaim',
               'quote': 'guarantees', 'issue': 'narration says reduces', 'suggested_change': 'reduces selection bias'}
    report = fidelity.review_topic(topic=TOPIC, deck_dir=_deck(tmp_path),
                                   runner=_runner({'status': 'fail', 'findings': [finding]}, calls))
    assert report['status'] == 'failed' and report['findings'][0]['quote'] == 'guarantees'
    task_id, prompt = calls[0]
    assert task_id == fidelity.TASK_ID
    assert 'Logs tests.' in prompt and 'reduces selection bias' in prompt and 'guarantees honest' in prompt


def test_advisory_findings_pass_and_a_status_cannot_hide_a_blocker(tmp_path):
    advisory = {'ref': 'topic', 'field': 'one_liner', 'severity': 'advisory', 'code': 'overclaim',
                'quote': 'Loops.', 'issue': 'terse', 'suggested_change': 'Loops matter.'}
    assert fidelity.review_topic(topic=TOPIC, deck_dir=_deck(tmp_path),
                                 runner=_runner({'status': 'pass', 'findings': [advisory]}, []))['status'] == 'passed'
    blocker = {**advisory, 'severity': 'blocker'}
    assert fidelity.validate_verdict({'status': 'pass', 'findings': [blocker]}, {'topic'})['status'] == 'fail'


@pytest.mark.parametrize('row', [{'ref': 'long.9'}, {'code': 'style'}, {'quote': ''}])
def test_a_malformed_finding_fails_closed(row):
    base = {'ref': 'topic', 'field': 'f', 'severity': 'advisory', 'code': 'overclaim',
            'quote': 'q', 'issue': 'i', 'suggested_change': 's'}
    with pytest.raises(ValueError):
        fidelity.validate_verdict({'status': 'pass', 'findings': [{**base, **row}]}, {'topic'})


def test_the_cli_reuses_an_unchanged_passing_topic_without_a_call(tmp_path, monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/presentations/review_executive_claim_fidelity.py'
    spec = importlib.util.spec_from_file_location('review_executive_claim_fidelity_t', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    _deck(tmp_path)
    (tmp_path / 'decks' / 'executive_topic_digests.json').write_text(json.dumps({'topics': [TOPIC]}))
    calls: list = []
    monkeypatch.setattr(cli, 'review_topic', lambda *, topic, deck_dir: calls.append(1) or {
        'topic_id': topic['topic_id'], 'status': 'passed', 'findings': [],
        'input_hashes': cli._input_hashes(topic, deck_dir)})
    assert cli.main(['--workspace', str(tmp_path)]) == 0
    assert cli.main(['--workspace', str(tmp_path)]) == 0
    assert len(calls) == 1
    report = json.loads((tmp_path / 'decks' / 'executive_claim_fidelity_report.json').read_text())
    assert report['topics'][0]['reused'] is True
