"""Claim fidelity for executive decks, checked against each topic deck's reviewed evidence.

Executive slides are written by a model from the topic narration and have no
Gate A of their own: ICML 2026's 36 long-deck takeaways shipped with only a
layout judge looking at them. Each topic's executive claims are therefore
checked in one governed call against that topic deck's bound source records
and its Gate A-reviewed narration -- the same evidence the topic deck stood on.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from conference_pipeline.model_governance.llm_client import complete
from conference_pipeline.presentation.claim_fidelity import bound_source_records
from conference_pipeline.presentation.claim_fidelity_review import MAX_OUTPUT_TOKENS, TASK_ID

CODES = frozenset({'overclaim', 'misattribution', 'composed_sources', 'position_as_result',
                   'absence_claim', 'unsupported_number', 'unsupported_comparator'})


def topic_claims(topic: dict) -> list[dict]:
    """Every audience-facing claim the executive decks make about one topic."""
    claims = [{'ref': 'topic', 'text': {'central_takeaway': topic.get('central_takeaway'),
                                         'one_liner': topic.get('one_liner')}}]
    short = topic.get('short') or {}
    for name in ('evidence_slide', 'bigpicture_slide'):
        slide = short.get(name) or {}
        if slide:
            claims.append({'ref': f'short.{name}', 'text': {
                key: slide.get(key) for key in ('headline', 'synthesis', 'implication', 'named_sources')
                if slide.get(key)} | ({'diagram': (slide.get('diagram') or {}).get('caption')}
                                      if slide.get('diagram') else {})})
    for index, slide in enumerate((topic.get('long') or {}).get('slides') or []):
        claims.append({'ref': f'long.{index}', 'text': {
            key: slide.get(key) for key in ('headline', 'body', 'takeaway', 'named_sources')
            if slide.get(key)} | ({'diagram': (slide.get('diagram') or {}).get('caption')}
                                  if slide.get('diagram') else {})})
    return claims


def topic_evidence(deck_dir: Path) -> tuple[list[dict], str, dict[str, str]]:
    """The topic deck's bound source records (one per label) and its reviewed narration."""
    paths = {name: deck_dir / name for name in ('source_binding_report.json', '6_presenter_script.ENGAGING.md')}
    texts = {name: path.read_text(encoding='utf-8') for name, path in paths.items()}
    records: dict[str, dict] = {}
    for slide in bound_source_records(json.loads(texts['source_binding_report.json'])):
        for row in slide['sources']:
            records.setdefault(str(row.get('label') or row.get('title')), row)
    if not records:
        raise ValueError(f'no bound source records in {deck_dir.name}')
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    return list(records.values()), texts['6_presenter_script.ENGAGING.md'], hashes


def build_prompt(*, title: str, claims: list[dict], records: list[dict], narration: str) -> str:
    return '''You are checking an executive summary's claims about one conference topic
against the evidence that topic's own deck was reviewed on. The executive text
is a compressed retelling: plain summarising, merging related findings into one
sentence, and naming the field-level pattern are all fine. Contribution notes
are short summaries, not the whole paper, so a missing detail is not evidence
against a claim.

Assign severity "blocker" ONLY to:
(a) a claim that contradicts the narration or a source record, or states a
    result more strongly than either does (a proposal stated as shown, "proves"
    for "suggests", a guarantee for a heuristic, a number the evidence lacks);
(b) misattribution: a result, method or number credited to a work other than
    the one the evidence credits;
(c) composed_sources: two works' methods presented as one system or pipeline
    that neither the narration nor a record describes;
(d) a position paper's proposal stated as a demonstrated result;
(e) an absence claim about a work ("no work reports...").
Everything else is "advisory". Do not review style, length or layout.

Return ONLY JSON {"status":"pass|fail","findings":[{"ref":"<claim ref>",
"field":"<which field>","severity":"blocker|advisory","code":"overclaim|misattribution|composed_sources|position_as_result|absence_claim|unsupported_number|unsupported_comparator",
"quote":"<the exact words at issue>","issue":"...","suggested_change":"<replacement text>"}]}.
Status is fail only if at least one finding is a blocker. Review every claim;
do not manufacture findings.

TOPIC: ''' + title + '\nEXECUTIVE CLAIMS:\n' + json.dumps(claims, ensure_ascii=False) + \
        '\nSOURCE RECORDS:\n' + json.dumps(records, ensure_ascii=False) + '\nREVIEWED NARRATION:\n' + narration


def validate_verdict(value: object, refs: set[str]) -> dict:
    if not isinstance(value, dict) or value.get('status') not in {'pass', 'fail'} or not isinstance(value.get('findings'), list):
        raise ValueError('invalid executive fidelity verdict')
    for row in value['findings']:
        if (not isinstance(row, dict) or row.get('ref') not in refs
                or row.get('severity') not in {'blocker', 'advisory'} or row.get('code') not in CODES
                or not all(isinstance(row.get(k), str) and row[k].strip() for k in ('field', 'quote', 'issue', 'suggested_change'))):
            raise ValueError(f'invalid executive fidelity finding: {row!r}'[:300])
    return {**value, 'status': 'fail' if any(r['severity'] == 'blocker' for r in value['findings']) else 'pass'}


def review_topic(*, topic: dict, deck_dir: Path, runner: Callable = complete) -> dict:
    """One governed call for one topic; refuses before spending if over the task's cap."""
    from conference_pipeline.costs.estimator import load_pricing_catalog
    from conference_pipeline.model_governance.registry import load_registry

    claims = topic_claims(topic)
    records, narration, hashes = topic_evidence(deck_dir)
    prompt = build_prompt(title=str(topic.get('title') or topic.get('topic_id')), claims=claims,
                          records=records, narration=narration)
    policy = load_registry().get(TASK_ID)
    estimated, _, _ = load_pricing_catalog().estimate_tokens(
        policy.provider, policy.model, len(prompt.encode('utf-8')), MAX_OUTPUT_TOKENS)
    if policy.budget_cap_usd is None or estimated > policy.budget_cap_usd:
        raise ValueError('executive fidelity review exceeds governed budget cap')
    response = runner(TASK_ID, prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS, caller=__file__)
    if response.stop_reason != 'end':
        raise ValueError(f'executive fidelity review incomplete: {response.stop_reason}')
    verdict = validate_verdict(json.loads(response.text), {claim['ref'] for claim in claims})
    return {'topic_id': topic.get('topic_id'), 'status': 'passed' if verdict['status'] == 'pass' else 'failed',
            'findings': verdict['findings'], 'provider': response.provider, 'model': response.model,
            'estimated_max_cost_usd': estimated,
            'input_hashes': hashes | {'claims': hashlib.sha256(
                json.dumps(claims, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}}
