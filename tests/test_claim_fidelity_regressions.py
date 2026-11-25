import json

from conference_pipeline.presentation.source_inventory import _transcript_extraction_metadata, _enrich_speech_transcript_metadata_from_local_join


def test_oral_session_exact_segment(tmp_path):
    root = tmp_path / 'storage/extractions/run/llm-full-read/by-unit'
    root.mkdir(parents=True)
    for segment, talk in [('00000', 'talk_001'), ('00003', 'talk_004')]:
        (root / f'speech_transcript_oral3a_segment_{segment}.json').write_text(json.dumps({'record': {'metadata': {'talk_id': talk, 'talk_title': 'Oral 3A ' + talk}}, 'summary': talk}))
    artifact = tmp_path / 'storage/extractions/run/by-artifact/speech_transcript_oral3a.json'
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({'record': {'source_artifact_path': 'session.mp4', 'metadata': {'talk_title': 'Wrong first talk', 'talk_id': 'talk_001'}}}))
    sid = 'speech_transcript:oral3a:segment:00003'
    assert _transcript_extraction_metadata(tmp_path, sid)['talk_title'] == 'Oral 3A talk_004'
    draft = {'source_id': sid, 'resolved_title': 'Oral 3A talk_001', 'talk_id': 'talk_004', 'authors': ['Wrong author'], 'contribution_note': 'First talk note'}
    _enrich_speech_transcript_metadata_from_local_join(tmp_path, {sid: draft})
    assert draft['resolved_title'] == 'Oral 3A talk_004'
    assert draft['contribution_note'] == 'talk_004'
    assert _transcript_extraction_metadata(tmp_path, sid)['paths'] == ['session.mp4']
    assert _transcript_extraction_metadata(tmp_path, sid.replace('00003', '00009')).get('talk_title') is None


import pytest
from conference_pipeline.presentation.claim_fidelity import absence_claims


@pytest.mark.parametrize('sentence', [
    'Neither work reports a benchmark score, so we are comparing designs.',
    'None of these sources reports quantitative results yet.',
    "The sources don't yet give us comparable numbers.",
    'Both hypotheses are presented without controlled numbers.',
    'The paper lacks evaluations.',
])
def test_absence_assertions(sentence):
    assert absence_claims(sentence)


def test_comparison_scope_is_not_source_absence():
    assert not absence_claims("We don't compare scores here. This talk shows the proposed mechanism.")


from conference_pipeline.figures.lint import unsupported_outcomes
from conference_pipeline.figures.spec import FigureSpec


@pytest.mark.parametrize('claim', [
    "Detector finds no signal; text quality is intact", 'Result stays certifiable',
    'Agents certified before market use', 'Kong, Zhang, and Liu certify each instance',
    'Safe behavior recovered', 'Early steering recovers safety that scaling erodes',
    'Weights unchanged', 'Certified output',
])
def test_unsupported_outcome_predicates(claim):
    spec = FigureSpec('f', 's', 'flow', claim, 'Proposed control', None, {'nodes': [{'label': claim}]})
    assert unsupported_outcomes(spec, {'approved_sources': [{'label': 'SRC-014', 'contribution_note': 'The position paper argues for symbolic integration.'}]})
    assert not unsupported_outcomes(spec, {'approved_sources': [{'contribution_note': claim}]})


def test_an_outcome_word_copied_verbatim_from_the_slide_is_not_invented_by_the_figure():
    # ICML 2026 deck 10: a text layout's claim "causal discovery recovers which variables
    # drive which" was rejected, so the author redrew the slide and reintroduced an error.
    sentence = 'Causal discovery recovers which variables drive which.'
    spec = FigureSpec('f', 's', 'key_points', sentence, '', None, {'points': [{'text': sentence}]})
    slide = {'approved_sources': [{'contribution_note': 'A benchmark for causal methods.'}], 'claims': [sentence]}
    assert not unsupported_outcomes(spec, slide)
    invented = FigureSpec('f', 's', 'flow', 'Safe behavior recovered', '', None, {'nodes': [{'label': 'x'}]})
    assert unsupported_outcomes(invented, slide)


def _script(name):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/presentations' / (name + '.py')
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def test_figure_prompts_keep_source_notes_and_work_kind():
    module = _script('generate_figure_specs')
    slides = [{'slide_id': 's'}]
    module.attach_approved_sources(slides, {'slides': [{'slide_id': 's', 'speaker_author_work_mentions': [
        {'label': 'SRC-014', 'title': 'Position: Certified Correctness', 'contribution_note': 'Argues for symbolic integration.'},
        {'label': 'SRC-015', 'title': 'Thought correction', 'work_kind': 'method', 'contribution_note': 'Proposes a plug-in.'},
    ]}]})
    text = module.approved_source_lines(slides[0])
    assert text.count('SOURCE BLOCK') == 2
    assert 'Argues for symbolic integration.' in text and '"work_kind": "position"' in text
    prompt = module.build_critic_prompt(payload={}, slide=slides[0], archetype='flow')
    assert 'composed_sources' in prompt and 'position_as_result' in prompt
    assert 'never against the slide claim' in prompt
    assert 'Argues for symbolic integration.' in prompt


def test_gate_a_evidence_and_prior_sweep(tmp_path):
    from conference_pipeline.presentation.presentation_review import build_review_prompt
    from conference_pipeline.production.presentation_steps import supersede_failed_review
    requirements = tmp_path / 'presentation_blueprint.json'
    requirements.write_text('{}')
    narration = tmp_path / 'script.md'
    narration.write_text('We compare proposals.')
    (tmp_path / 'source_binding_report.json').write_text(json.dumps({'slides': [{'slide_id': 's', 'visible_references': [{'label': 'SRC-014', 'title': 'Position: Certified Correctness', 'contribution_note': 'Argues for symbolic integration.'}]}]}))
    review = tmp_path / 'presentation_review.json'
    review.write_text(json.dumps({'status': 'failed', 'reviews': [{'production_blockers': [{'slide': 20, 'issue': 'Position shown as certified output'}]}]}))
    archived = supersede_failed_review(review)
    assert archived.parent.name == 'gate_a_attempts'
    prompt = build_review_prompt(workspace=tmp_path, presentation_id='test', presentation_type='topic', topic=None, requirements_path=requirements, narration_script_path=narration, screenshots=[])
    for phrase in ['source_fidelity', 'EVERY factual', 'carrying a synthesis label', 'then sweep again', 'Position shown as certified output', 'Argues for symbolic integration.', '"work_kind": "position"']:
        assert phrase in prompt


@pytest.mark.parametrize('title', [
    'Multi-turn jailbreaks: the attack is spread across the conversation',
    'Adversarial fine-tuning can bypass classifiers with no capability tax',
    'Peer-preservation: frontier models shield other agents from shutdown',
    'Release checklist: retest safety after every weight or compute change',
])
def test_titles_are_rewritten_once_not_truncated(title, monkeypatch):
    from conference_pipeline.presentation.blueprint import validate_blueprint_payload
    module = _script('generate_editorial_blueprint')
    payload = {'slides': [{'title': title, 'claims': ['Original claim']}]}
    assert any(e.code == 'BLUEPRINT_TITLE_TOO_LONG' for e in validate_blueprint_payload(payload).errors)
    calls = []
    def call(prompt):
        calls.append(prompt)
        return {'titles': [{'index': 0, 'title': 'Complete rewritten title'}]}
    monkeypatch.setattr(module, '_call', call)
    assert module._cap_title(title) == title
    module.repair_overlength_titles(payload)
    assert len(calls) == 1 and title in calls[0]
    assert payload['slides'][0] == {'title': 'Complete rewritten title', 'claims': ['Original claim']}
    monkeypatch.setattr(module, '_call', lambda prompt: {'titles': [{'index': 0, 'title': title}]})
    with pytest.raises(ValueError, match='title repair failed'):
        module.repair_overlength_titles({'slides': [{'title': title}]})


@pytest.mark.parametrize('sentence', [
    'Neither work supplies a shared numeric failure rate here.',
    'Neither reports a shared failure rate, but both evaluate the executed trajectory.',
    "Neither work reports a benchmark score, so we're comparing designs, not measured effect sizes.",
    'There are no comparative scores for these works yet, so this slide is about framing, not measured performance.',
    'None of these sources reports quantitative results yet, so the next step is to measure this whole chain end to end.',
])
def test_historical_deck03_absences(sentence):
    assert absence_claims(sentence)


def _fidelity_deck(tmp_path):
    (tmp_path / 'figure_specs').mkdir()
    (tmp_path / 'figure_specs/f.json').write_text(json.dumps({'slide_id': 's', 'archetype': 'flow', 'claim': 'Text quality is intact', 'content': {'nodes': [{'label': 'Text quality is intact'}]}}))
    (tmp_path / 'source_binding_report.json').write_text(json.dumps({'slides': [{'slide_id': 's', 'visible_references': [{'label': 'SRC-014', 'title': 'A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness', 'contribution_note': 'The paper critiques the reliability of LLM-as-a-Judge frameworks for safety evaluation, demonstrating that distribution shifts cause performance to degrade to near random chance, and proposes new benchmarks to address these failures.', 'work_kind': 'empirical'}]}]}))
    for name in ['6_presenter_script.ENGAGING.md', '6_presenter_script_notebooklm_2host.ENGAGING.md']:
        (tmp_path / name).write_text('Neither work reports a benchmark score.')


def test_fidelity_one_pass_and_repair_routing(tmp_path):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity, TASK_ID
    _fidelity_deck(tmp_path)
    calls = []
    def runner(task_id, **kwargs):
        calls.append((task_id, kwargs))
        return SimpleNamespace(text=json.dumps({'status': 'fail', 'findings': [
            {'slide_id': 's', 'target': 'figure', 'severity': 'blocker', 'code': 'overclaim', 'source_labels': ['SRC-014'], 'issue': 'Unsupported quality outcome', 'suggested_change': 'Remove quality claim'},
            {'slide_id': 's', 'target': 'narration', 'severity': 'advisory', 'code': 'absence_claim', 'source_labels': ['SRC-014'], 'issue': 'Unsupported absence', 'suggested_change': 'Describe what we compare here'},
        ]}), stop_reason='end', provider='stub', model='fixture')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='narration', runner=runner)
    assert report['status'] == 'failed'
    assert len(calls) == 1 and calls[0][0] == TASK_ID
    assert 'near random chance' in calls[0][1]['prompt']
    assert 'Neither work reports' in calls[0][1]['prompt']
    assert report['findings'][0]['repair_tool'].endswith('patch_figure_spec.py')
    assert report['findings'][1]['repair_tool'].endswith('fix_narration_script.py')
    assert len(report['input_hashes']) == 4
    assert report['estimated_max_cost_usd'] <= report['budget_cap_usd']


def test_fidelity_keeps_a_finding_that_cites_a_label_bound_elsewhere(tmp_path):
    # ICML 2026 deck 03: one such label made the step discard the whole verdict.
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _fidelity_deck(tmp_path)
    runner = lambda task_id, **kwargs: SimpleNamespace(text=json.dumps({'status': 'fail', 'findings': [
        {'slide_id': 's', 'target': 'figure', 'severity': 'blocker', 'code': 'misattribution', 'source_labels': ['SRC-014', 'SRC-099'],
         'issue': 'Credits another slide\'s work', 'suggested_change': 'Cite the bound work'}]}),
        stop_reason='end', provider='stub', model='fixture')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=runner)
    assert report['status'] == 'failed'
    assert report['findings'][0]['unbound_source_labels'] == ['SRC-099']


def test_fidelity_missing_evidence_never_calls_model(tmp_path):
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _fidelity_deck(tmp_path)
    (tmp_path / 'source_binding_report.json').write_text('{"slides": []}')
    with pytest.raises(ValueError, match='missing bound source notes'):
        review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=lambda *a, **k: pytest.fail('must not call'))


@pytest.mark.parametrize('verdict', [
    {}, {'status': 'pass', 'findings': [{'slide_id': 's'}]},
    {'status': 'unknown', 'findings': []},
])
def test_fidelity_malformed_verdict_fails_closed(verdict):
    from conference_pipeline.presentation.claim_fidelity_review import validate_verdict
    with pytest.raises(ValueError):
        validate_verdict(verdict, {'s'})


def test_fidelity_registry_and_chain_contract():
    from pathlib import Path
    from conference_pipeline.model_governance.registry import load_registry
    from conference_pipeline.stage_script_registry import load_stage_script_registry
    policy = load_registry().get('presentation.claim_fidelity_review')
    assert policy.budget_cap_usd == 5
    registry = load_stage_script_registry()
    for task in ['claim_fidelity_review', 'figure_spec_authoring', 'figure_spec_critic', 'engaging_narration']:
        assert registry['task_bindings']['presentation.' + task]['required_script_ids']
    chain = (Path(__file__).resolve().parents[1] / 'src/conference_pipeline/production/presentation_chain.py').read_text()
    assert chain.index('_claim_fidelity_step("figures")') < chain.index('figure_compile_receipt_output =')
    assert chain.index('_claim_fidelity_step("narration")') > chain.index('_rebuild_delivery_and_reconcile("post-gates")')
    assert chain.index('_claim_fidelity_step("narration")') < chain.index('# Gate A:')


def test_fidelity_reads_active_specs_not_critic_sidecars(tmp_path):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _fidelity_deck(tmp_path)
    (tmp_path / 'figure_specs/f.critic.json').write_text('{"verdict":"accept"}')
    (tmp_path / 'figure_specs/old.json').write_text('{"slide_id":"stale"}')
    (tmp_path / 'figure_spec_receipt.json').write_text(json.dumps({'status': 'pass', 'specs': [{'slide_id': 's', 'spec_path': 'figure_specs/f.json'}]}))
    calls = []
    def runner(task, **kwargs):
        calls.append(kwargs['prompt'])
        return SimpleNamespace(text='{"status":"pass","findings":[]}', stop_reason='end', provider='stub', model='fixture')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=runner)
    assert report['status'] == 'passed' and len(calls) == 1
    assert 'stale' not in calls[0] and 'Neither work reports' not in calls[0]
    assert 'figure_spec_receipt.json' in report['input_hashes']


def test_absence_generator_and_both_lints(tmp_path):
    sentence = 'Neither work reports a benchmark score.'
    generator = _script('generate_engaging_narration')
    assert generator.absence_claims({'slides': [{'slide_id': 's', 'two_host': [{'text': sentence}]}]})
    presenter = _script('lint_presenter_narration')
    two_host = _script('lint_narration_script')
    script = tmp_path / 'presenter.md'
    script.write_text('### [SLIDE 1] Control\n\n**SAY:**\n' + sentence + '\n')
    assert any(f['rule'] == 'absence_claim' and f['severity'] == 'fail' for f in presenter.analyze_presenter_narration(script_md=script)['findings'])
    script.write_text('### [SLIDE 1] Control\n\n**DIALOGUE:**\n- **Host:** ' + sentence + '\n')
    assert any(f['rule'] == 'absence_claim' and f['severity'] == 'fail' for f in two_host.analyze_narration_script(script_md=script)['findings'])


def test_fidelity_budget_cap_prevents_call(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    import conference_pipeline.costs.estimator as estimator
    _fidelity_deck(tmp_path)
    monkeypatch.setattr(estimator, 'load_pricing_catalog', lambda: SimpleNamespace(estimate_tokens=lambda *a: (6, [], [])))
    with pytest.raises(ValueError, match='budget cap'):
        review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=lambda *a, **k: pytest.fail('must not spend'))


def test_segment_talk_id_mismatch_is_blocked(tmp_path):
    root = tmp_path / 'storage/extractions/run/llm-full-read/by-unit'
    root.mkdir(parents=True)
    (root / 'speech_transcript_oral3a_segment_00003.json').write_text(json.dumps({'record': {'metadata': {'talk_id': 'talk_004', 'talk_title': 'Fourth talk'}}, 'summary': 'Fourth talk note'}))
    sid = 'speech_transcript:oral3a:segment:00003'
    draft = {'source_id': sid, 'resolved_title': sid, 'talk_id': 'talk_001'}
    _enrich_speech_transcript_metadata_from_local_join(tmp_path, {sid: draft})
    assert draft['status'] == 'title_mismatch'
    assert not draft.get('contribution_note')


# Minimal text copied from decks 03/04's first archived Gate A findings.
# The two semantic cases require a model. Replaying the archived verdict with
# a stub tests enforcement, not the intelligence of a new provider response.
@pytest.mark.parametrize('claim,note,code', [
    ('Detector finds no signal; text quality is intact', 'Linear ensembling removes the distributional watermark signal.', 'overclaim'),
    ('Inaudible adversarial audio', 'Audible audio goes unnoticed because it exploits human selective attention.', 'overclaim'),
    ('Result stays certifiable', 'The position work proposes controlled evaluation.', 'position_as_result'),
    ('Agents certified before market use', 'The position paper argues for certification requirements.', 'position_as_result'),
    ('Kong, Zhang, and Liu certify each instance', 'The position work argues that symbolic integration is required for correctness.', 'position_as_result'),
    ('Diagnostic accuracy paired with CIR/SR causal tests', "CIR/SR are Yu et al.'s measures; OR-Loop Bench is Ao et al.'s solver-in-the-loop benchmark.", 'misattribution'),
    ('Safe behavior recovered', 'Test-time scaling can increase unsafe outputs by reducing candidate diversity; no combined SafeThink experiment is stated here.', 'composed_sources'),
    ('Weights unchanged', 'FG-TTL updates the model on selected tokens at test time.', 'overclaim'),
])
def test_archived_blocker_verdict_replay(tmp_path, claim, note, code):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _fidelity_deck(tmp_path)
    spec_path = tmp_path / 'figure_specs/f.json'
    spec_path.write_text(json.dumps({'slide_id': 's', 'archetype': 'flow', 'claim': claim}))
    binding = json.loads((tmp_path / 'source_binding_report.json').read_text())
    binding['slides'][0]['visible_references'][0]['contribution_note'] = note
    (tmp_path / 'source_binding_report.json').write_text(json.dumps(binding))
    def archived_review(task, **kwargs):
        assert claim in kwargs['prompt'] and note in kwargs['prompt']
        return SimpleNamespace(text=json.dumps({'status': 'fail', 'findings': [{'slide_id': 's', 'target': 'figure', 'severity': 'advisory' if claim == 'Detector finds no signal; text quality is intact' else 'blocker', 'code': code, 'source_labels': ['SRC-014'], 'issue': claim, 'suggested_change': note}]}), stop_reason='end', provider='stub', model='archived-review')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=archived_review)
    # An unlabelled combination blocks; explicit synthesis captions are tested separately.
    advisory = claim == 'Detector finds no signal; text quality is intact'
    assert report['status'] == ('passed' if advisory else 'failed')
    assert report['findings'][0]['code'] == code


def _calibrated_finding(severity='advisory', code='overclaim'):
    return {'slide_id': 's', 'target': 'figure', 'severity': severity, 'code': code,
            'source_labels': ['SRC-014'], 'issue': 'A detail is absent from a short summary.',
            'suggested_change': 'Gate A should verify or dismiss this lead.'}


@pytest.mark.parametrize('severity', [None, '', 'major', 'warning', 1, [], {}])
def test_fidelity_requires_valid_severity(severity):
    from conference_pipeline.presentation.claim_fidelity_review import validate_verdict
    row = _calibrated_finding(severity)
    if severity is None:
        row.pop('severity')
    with pytest.raises(ValueError, match='invalid fidelity finding'):
        validate_verdict({'status': 'pass', 'findings': [row]}, {'s'})


@pytest.mark.parametrize('severities,expected', [(['advisory'], 'passed'),
                                             (['blocker'], 'failed'),
                                             (['advisory', 'blocker'], 'failed')])
def test_fidelity_status_is_derived_from_severity(tmp_path, severities, expected):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity, MAX_OUTPUT_TOKENS
    _fidelity_deck(tmp_path)
    def runner(task, **kwargs):
        assert kwargs['max_tokens'] == MAX_OUTPUT_TOKENS == 32000
        # Intentionally over-strict summary: findings, not this summary, govern blocking.
        return SimpleNamespace(text=json.dumps({'status': 'fail', 'findings': [_calibrated_finding(s) for s in severities]}),
                               stop_reason='end', provider='stub', model='fixture')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=runner)
    assert report['status'] == expected
    assert report['finding_counts'] == {s: severities.count(s) for s in ('blocker', 'advisory')}
    assert len(report['findings']) == len(severities)


def test_ledger_mismatch_is_prominent_and_never_blocks(tmp_path):
    from types import SimpleNamespace
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _fidelity_deck(tmp_path)
    row = _calibrated_finding('blocker', 'ledger_mismatch')
    report = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=lambda *a, **k: SimpleNamespace(
        text=json.dumps({'status': 'fail', 'findings': [row]}), stop_reason='end', provider='stub', model='fixture'))
    assert report['status'] == 'passed'
    assert report['finding_counts'] == {'blocker': 0, 'advisory': 1}
    assert report['ledger_mismatch_findings'] == report['findings']


def test_calibrated_prompt_allows_scoped_explanation_and_synthesis():
    from conference_pipeline.presentation.claim_fidelity_review import build_fidelity_prompt
    prompt = " ".join(build_fidelity_prompt(binding={}, specs=[]).split())
    for wording in ('short summaries', 'not the whole paper', 'NOT evidence',
                    'pedagogical', 'the talk proposes', 'untested end-to-end', 'not reported here',
                    'blocker" ONLY', 'directly contradicting', 'misattribution', 'composed_sources',
                    'position paper', 'ledger_mismatch', 'always advisory'):
        assert wording in prompt


def test_slide_aware_authoring_and_reuse_checks_unlabelled_composition():
    from dataclasses import replace
    module = _script('generate_figure_specs')
    slide = {'approved_sources': [{'label': 'SRC-001', 'title': 'AgentLAB'}, {'label': 'SRC-002', 'title': 'AutoControl Arena'}]}
    spec = FigureSpec('f', 's', 'flow', 'Two methods', 'Compare methods', None,
                      {'nodes': [{'id': 'a', 'label': 'AgentLAB'}, {'id': 'b', 'label': 'AutoControl Arena'}], 'edges': [{'from': 'a', 'to': 'b'}]})
    assert any('unlabeled_composition' in e for e in module._slide_title_errors(spec, slide))
    assert not module._slide_title_errors(replace(spec, synthesis_label="Talk's synthesis"), slide)


def test_the_fidelity_prompt_says_ids_are_slugs_not_visible_text():
    # ICML 2026 deck 09: "sampling-leaks-what-greedy-decoding" in a slide_id was read as the title.
    from conference_pipeline.presentation.claim_fidelity_review import build_fidelity_prompt
    prompt = build_fidelity_prompt(binding={'slides': []}, specs=[])
    assert 'never the ids' in prompt


@pytest.mark.parametrize('sentence', [
    # Read aloud in ICML 2026 deck 09 as the full authorship of a capped six-name list.
    'Antoni Kowalczuk, Dominik Hintersdorf, Lukas Struppek, Kristian Kersting, Adam Dziedzic and Franziska Boenisch show it.',
    'Koromilas, Demou, Oldfield, Panagakis, and Nicolaou find the same.',
])
def test_a_closed_list_of_four_or_more_names_is_flagged(sentence):
    from conference_pipeline.presentation.claim_fidelity import closed_author_lists
    assert closed_author_lists(sentence)


@pytest.mark.parametrize('sentence', [
    'Tao, Weston and colleagues show in Hybrid Reinforcement that it helps.',
    'Koromilas, Demou, Oldfield and colleagues find the same.',
    'Wang, Zhang and Gong report a gain.',
])
def test_open_or_short_author_lists_are_not_flagged(sentence):
    from conference_pipeline.presentation.claim_fidelity import closed_author_lists
    assert closed_author_lists(sentence) == []


def test_closed_author_lists_reach_the_generator_repair_and_both_lints(tmp_path):
    sentence = 'Lee, Bang, Hwang, Lim, and Kim show it.'
    generator = _script('generate_engaging_narration')
    assert generator.closed_author_lists({'slides': [{'slide_id': 's', 'single': sentence}]})
    script = tmp_path / 'presenter.md'
    script.write_text('### [SLIDE 1] Control\n\n**SAY:**\n' + sentence + '\n')
    findings = _script('lint_presenter_narration').analyze_presenter_narration(script_md=script)['findings']
    assert any(f['rule'] == 'closed_author_list' and f['severity'] == 'fail' for f in findings)
    script.write_text('### [SLIDE 1] Control\n\n**DIALOGUE:**\n- **Host:** ' + sentence + '\n')
    findings = _script('lint_narration_script').analyze_narration_script(script_md=script)['findings']
    assert any(f['rule'] == 'closed_author_list' and f['severity'] == 'fail' for f in findings)


def test_source_records_say_how_many_authors_a_work_has():
    from conference_pipeline.presentation.claim_fidelity import source_record
    assert source_record({'title': 't', 'authors': ['A', 'B'], 'author_count': 9})['author_count'] == 9
    assert source_record({'title': 't', 'authors': ['A', 'B']})['author_count'] == 2


def test_the_executive_build_requires_link_maps_unless_told_otherwise(monkeypatch):
    import importlib.util, sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/presentations/build_executive_summary.py'
    spec = importlib.util.spec_from_file_location('build_executive_summary_t', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, 'run', lambda cmd, label: calls.append(cmd))
    monkeypatch.setattr(module.shutil, 'copyfile', lambda *a: None)
    module.main(['--workspace', 'conferences/x', '--scope', 'short'])
    generate = next(c for c in calls if c[0].endswith('generate_executive_overview_decks.py'))
    assert '--require-presentation-link-map' in generate and '--require-source-link-map' in generate
    calls.clear()
    module.main(['--workspace', 'conferences/x', '--scope', 'short', '--allow-local-links'])
    generate = next(c for c in calls if c[0].endswith('generate_executive_overview_decks.py'))
    assert '--require-presentation-link-map' not in generate


def _two_slide_deck(tmp_path):
    (tmp_path / 'figure_specs').mkdir()
    refs = []
    for sid in ('a', 'b'):
        (tmp_path / f'figure_specs/{sid}.json').write_text(json.dumps({'slide_id': sid, 'archetype': 'flow', 'claim': f'Claim {sid}', 'content': {'nodes': [{'label': f'Node {sid}'}]}}))
        refs.append({'slide_id': sid, 'visible_references': [{'label': f'SRC-{sid}', 'title': f'Work {sid}', 'contribution_note': f'Work {sid} proposes a method.'}]})
    (tmp_path / 'source_binding_report.json').write_text(json.dumps({'slides': refs}))
    (tmp_path / 'engaging_narration_raw.json').write_text(json.dumps({'slides': [
        {'slide_id': sid, 'single': f'Spoken {sid}.', 'two_host': [{'speaker': 'Iapetus', 'text': f'Host {sid}?'}]} for sid in ('a', 'b')]}))
    for name in ['6_presenter_script.ENGAGING.md', '6_presenter_script_notebooklm_2host.ENGAGING.md']:
        (tmp_path / name).write_text('Spoken a. Spoken b.')


def _recording_runner(findings=()):
    from types import SimpleNamespace
    calls = []
    def runner(task, **kwargs):
        calls.append(kwargs['prompt'])
        rows = [row for row in findings if f'"slide_id": "{row["slide_id"]}"' in kwargs['prompt']]
        return SimpleNamespace(text=json.dumps({'status': 'fail' if any(r['severity'] == 'blocker' for r in rows) else 'pass', 'findings': rows}),
                               stop_reason='end', provider='stub', model='fixture')
    runner.calls = calls
    return runner


def _finding(slide_id, severity):
    return {'slide_id': slide_id, 'target': 'figure', 'severity': severity, 'code': 'overclaim', 'source_labels': [f'SRC-{slide_id}'],
            'issue': f'Issue on {slide_id}', 'suggested_change': 'Reword'}


def test_an_unchanged_settled_deck_is_not_re_judged(tmp_path):
    # ICML 2026, 2026-11-06: identical specs passed one round and failed the next.
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _two_slide_deck(tmp_path)
    first = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner([_finding('b', 'advisory')]))
    assert first['status'] == 'passed' and set(first['judged_slide_ids']) == {'a', 'b'}
    again = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=lambda *a, **k: pytest.fail('must not re-judge'), prior=first)
    assert again == first
    # After a partial run (a slide edited, then reverted) settled slides carry their advisories.
    spec = json.loads((tmp_path / 'figure_specs/a.json').read_text())
    (tmp_path / 'figure_specs/a.json').write_text(json.dumps({**spec, 'claim': 'Edited'}))
    edited = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner(), prior=first)
    assert edited['judged_slide_ids'] == ['a'] and edited['carried_forward_slide_ids'] == ['b']
    assert [row['issue'] for row in edited['findings']] == ['Issue on b'] and edited['findings'][0]['carried_forward'] is True


def test_only_changed_or_blocked_slides_reach_the_model(tmp_path):
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _two_slide_deck(tmp_path)
    first = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner([_finding('a', 'blocker')]))
    assert first['status'] == 'failed'
    # Slide a is unchanged but still blocked: judged again. Slide b settled: carried.
    runner = _recording_runner()
    second = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=runner, prior=first)
    assert second['judged_slide_ids'] == ['a'] and second['carried_forward_slide_ids'] == ['b']
    assert '"slide_id": "b"' not in runner.calls[0] and 'Work b proposes' not in runner.calls[0]
    assert second['status'] == 'passed'
    # Now edit slide b: only b is judged.
    spec = json.loads((tmp_path / 'figure_specs/b.json').read_text())
    (tmp_path / 'figure_specs/b.json').write_text(json.dumps({**spec, 'claim': 'Edited claim b'}))
    third = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner(), prior=second)
    assert third['judged_slide_ids'] == ['b'] and third['carried_forward_slide_ids'] == ['a']


def test_narration_phase_tracks_each_slides_spoken_text(tmp_path):
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _two_slide_deck(tmp_path)
    first = review_claim_fidelity(deck_dir=tmp_path, phase='narration', runner=_recording_runner())
    raw = json.loads((tmp_path / 'engaging_narration_raw.json').read_text())
    raw['slides'][1]['two_host'][0]['text'] = 'A changed host turn?'
    (tmp_path / 'engaging_narration_raw.json').write_text(json.dumps(raw))
    runner = _recording_runner()
    second = review_claim_fidelity(deck_dir=tmp_path, phase='narration', runner=runner, prior=first)
    assert second['judged_slide_ids'] == ['b'] and second['carried_forward_slide_ids'] == ['a']
    assert 'A changed host turn?' in runner.calls[0] and 'Spoken a.' not in runner.calls[0]


def test_a_rules_change_re_judges_every_slide(tmp_path, monkeypatch):
    from conference_pipeline.presentation import claim_fidelity_review as cf
    _two_slide_deck(tmp_path)
    first = cf.review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner())
    monkeypatch.setattr(cf, 'RULES_SHA256', 'new-rules')
    second = cf.review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner(), prior=first)
    assert set(second['judged_slide_ids']) == {'a', 'b'}


def test_a_schema_1_report_settles_slides_whose_files_are_unchanged(tmp_path):
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _two_slide_deck(tmp_path)
    first = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner([_finding('a', 'blocker')]))
    legacy = {key: value for key, value in first.items() if key not in {'slide_digests', 'judged_slide_ids', 'carried_forward_slide_ids'}}
    second = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner(), prior=legacy)
    assert second['judged_slide_ids'] == ['a'] and second['carried_forward_slide_ids'] == ['b']
    (tmp_path / 'source_binding_report.json').write_text((tmp_path / 'source_binding_report.json').read_text().replace('proposes', 'introduces'))
    third = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner(), prior=legacy)
    assert set(third['judged_slide_ids']) == {'a', 'b'}, 'a changed shared input settles nothing'


def test_an_execution_error_keeps_the_last_verdict_for_the_next_run(tmp_path, monkeypatch):
    import importlib.util
    from conference_pipeline.presentation import claim_fidelity_review as cf
    _two_slide_deck(tmp_path)
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(cf.review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner())))
    spec = importlib.util.spec_from_file_location('rcf', 'scripts/presentations/review_claim_fidelity.py')
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    def boom(**kwargs):
        raise RuntimeError('provider down')
    monkeypatch.setattr(script, 'review_claim_fidelity', boom)
    assert script.main(['--workspace', str(tmp_path), '--deck-dir', str(tmp_path), '--phase', 'figures', '--report-output', str(report_path)]) == 1
    failed = json.loads(report_path.read_text())
    assert failed['error'].startswith('RuntimeError') and failed['last_verdict']['slide_digests']
    seen = {}
    monkeypatch.setattr(script, 'review_claim_fidelity', lambda **kwargs: seen.update(kwargs) or {'status': 'passed'})
    script.main(['--workspace', str(tmp_path), '--deck-dir', str(tmp_path), '--phase', 'figures', '--report-output', str(report_path)])
    assert seen['prior']['slide_digests'], 'the next run sees the last real verdict, not the error'


def test_a_no_op_rerun_returns_the_identical_report(tmp_path):
    # Gate A reads this report: a rewritten-but-equal report would re-run Gate A.
    from conference_pipeline.presentation.claim_fidelity_review import review_claim_fidelity
    _two_slide_deck(tmp_path)
    first = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=_recording_runner())
    again = review_claim_fidelity(deck_dir=tmp_path, phase='figures', runner=lambda *a, **k: pytest.fail('no call'), prior=first)
    assert again == first
