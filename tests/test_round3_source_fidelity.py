"""Offline replays of the source-fidelity calibration cases."""
import json
from dataclasses import replace

import pytest

from conference_pipeline.figures.spec import FigureSpec, Grounding, load_spec, FigureSpecError
from conference_pipeline.figures.lint import source_fidelity_errors
from conference_pipeline.presentation.claim_fidelity_review import validate_verdict, build_fidelity_prompt
from conference_pipeline.presentation import source_inventory as inventory


def _spec(content, **kwargs):
    return FigureSpec('f', 's', 'flow', 'Compare mechanisms', 'Different mechanisms', Grounding('schematic'), content, **kwargs)


def _slide(*names):
    return {'approved_sources': [{'label': f'SRC-{i:03}', 'title': name, 'contribution_note': f'{name} proposes a method.'} for i, name in enumerate(names)]}


@pytest.mark.parametrize('names', [
    ('AgentLAB', 'AutoControl Arena'), ('Surgery', 'Trojan-Speak'),
    ('Jailbreak Foundry', 'A Coin Flip for Safety'), ('SSL', 'LiMuon'),
    ('FlashSinkhorn', 'StitchCUDA'),
])
@pytest.mark.parametrize('shape', ['edge', 'nested', 'sequence'])
def test_composition_requires_explicit_caption(names, shape):
    nodes = [{'id': 'a', 'label': names[0]}, {'id': 'b', 'label': names[1]}]
    content = {'nodes': nodes, 'edges': [{'from': 'a', 'to': 'b'}]}
    spec = _spec(content)
    if shape == 'nested':
        spec = replace(spec, content={'nodes': [{'id': 'a', 'label': names[0], 'children': [nodes[1]]}]})
    if shape == 'sequence':
        spec = replace(spec, archetype='layered_stack', content={'layers': nodes})
    assert 'unlabeled_composition' in {f.code for f in source_fidelity_errors(spec, _slide(*names))}
    assert not source_fidelity_errors(replace(spec, synthesis_label="Talk's synthesis"), _slide(*names))
    assert not source_fidelity_errors(_spec({'nodes': nodes}), _slide(*names))


def test_synthesis_caption_serializes_and_renders():
    from conference_pipeline.figures.compiler import compile_figure
    payload = _spec({'nodes': [{'id': 'a', 'label': 'Input'}, {'id': 'b', 'label': 'Output'}], 'edges': [{'from': 'a', 'to': 'b'}]}, synthesis_label="Talk's synthesis").to_dict()
    assert load_spec(payload).synthesis_label == "Talk's synthesis"
    compiled = compile_figure(payload)
    assert 'synthesis' in compiled.svg
    with pytest.raises(FigureSpecError, match='synthesis_label'):
        load_spec({**payload, 'synthesis_label': []})


def _finding(code):
    return {'slide_id': 's', 'target': 'figure', 'severity': 'advisory', 'code': code, 'source_labels': ['SRC-032'], 'issue': '1.72x speedup vs hand-tuned kernels like FlashSinkhorn', 'suggested_change': 'Say 1.72x over baselines.'}


def test_composition_verdict_depends_on_actual_spec_caption():
    verdict = {'status': 'pass', 'findings': [_finding('composed_sources')]}
    assert validate_verdict(verdict, {'s'})['status'] == 'fail'
    assert validate_verdict(verdict, {'s'}, {'s': True})['status'] == 'pass'


def test_named_comparator_replay():
    verdict = validate_verdict({'status': 'pass', 'findings': [_finding('unsupported_comparator')]}, {'s'})
    assert verdict['status'] == 'fail'
    prompt = build_fidelity_prompt(binding={'slides': [{'slide_id': 's', 'visible_references': [{'label': 'SRC-032', 'contribution_note': '1.72x speedup over baselines'}]}]}, specs=[{'claim': _finding('unsupported_comparator')['issue']}])
    assert 'unsupported_comparator' in prompt and '1.72x speedup over baselines' in prompt


@pytest.mark.parametrize('claim', ['No control is tested at deployment scale yet', 'none has been shown to work at scale'])
def test_matrix_absence_claims(claim):
    spec = _spec({'nodes': [{'label': claim}]})
    assert 'absence_claim' in {f.code for f in source_fidelity_errors(spec, _slide('Control'))}


def test_no_cell_requires_negative_note():
    spec = replace(_spec({'rows': [{'id': 'a', 'label': 'FlashSinkhorn'}], 'cols': ['Deployment'], 'cells': [{'row': 'a', 'col': 'Deployment', 'value': 'no'}]}), archetype='feature_matrix')
    slide = _slide('FlashSinkhorn', 'StitchCUDA')
    assert 'unsupported_negative_cell' in {f.code for f in source_fidelity_errors(spec, slide)}
    slide['approved_sources'][1]['contribution_note'] = 'StitchCUDA is not a deployed system.'
    assert source_fidelity_errors(spec, slide)  # a different work's negative cannot license this cell
    slide['approved_sources'][0]['contribution_note'] = 'FlashSinkhorn does not test deployment.'
    assert not source_fidelity_errors(spec, slide)
    spec = replace(spec, content={**spec.content, 'cells': [{'row': 'a', 'col': 'Deployment', 'value': 'not_reported'}]})
    assert not source_fidelity_errors(spec, _slide('FlashSinkhorn'))


def test_analysis_notes_skip_header_and_cross_work_paragraphs(tmp_path):
    path = tmp_path / 'analysis.md'
    clean = '- CrispEdit improves model editing (`paper_abstract:crisp`).'
    path.write_text('\n'.join([
        '- **Full-read abstract records and paper resources:** `paper_abstract:src001`',
        '- DiSC and ALER improve reasoning. https://openreview.net/forum?id=disc https://openreview.net/forum?id=aler https://openreview.net/forum?id=crisp',
        '- MultiBreak attacks agents. `paper_abstract:multi` `paper_abstract:auto`', clean,
        '- StitchCUDA speeds kernels. `paper_text:stitch`',
    ]))
    notes = inventory._contribution_notes_from_inventory(workspace=tmp_path, source_records=[{'kind': 'analysis_report', 'path': 'analysis.md'}], work_titles={'paper_abstract:crisp': 'CrispEdit', 'paper_abstract:auto': 'AutoControl Arena', 'paper_abstract:multi': 'MultiBreak', 'paper_abstract:disc': 'DiSC', 'paper_abstract:aler': 'ALER'})
    assert 'paper_abstract:src001' not in notes
    assert 'paper_abstract:auto' not in notes
    assert notes['paper_abstract:crisp'] == inventory._clean_contribution_note(clean)
    assert notes['paper_abstract:disc'].startswith('DiSC')
    assert 'paper_text:stitch' in notes


@pytest.mark.parametrize('available, expected', [
    (['summary', 'packet', 'pack', 'abstract'], 'full_read_summary'),
    (['packet', 'pack', 'abstract'], 'packet_claim'),
    (['pack', 'abstract'], 'narrative_pack'), (['abstract'], 'openreview_abstract'), ([], 'none'),
])
def test_rejected_note_fallback_order_and_provenance(tmp_path, monkeypatch, available, expected):
    drafts = {'paper_abstract:crisp': {'source_id': 'paper_abstract:crisp', 'resolved_title': 'CrispEdit', 'authors': ['A Person'], 'contribution_note': 'DiSC and ALER improve reasoning.', 'contribution_note_source': 'analysis_line', '_packet_claim_note': 'CrispEdit packet claim.' if 'packet' in available else ''},
              'paper_abstract:disc': {'source_id': 'paper_abstract:disc', 'resolved_title': 'DiSC', 'contribution_note': 'DiSC clean note.', 'contribution_note_source': 'analysis_line'}}
    monkeypatch.setattr(inventory, '_paper_extraction_summary', lambda *a, **k: 'CrispEdit full summary.' if 'summary' in available else '')
    monkeypatch.setattr(inventory, '_narrative_pack_paper_notes', lambda *a: {'crisp': 'CrispEdit narrative note.'} if 'pack' in available else {})
    monkeypatch.setattr(inventory, '_openreview_abstract_index', lambda *a: {'crisp': 'CrispEdit abstract.'} if 'abstract' in available else {})
    inventory._enrich_paper_contribution_notes(tmp_path, drafts, topic_id='t')
    entries = {e.source_id: e.to_dict() for e in inventory._finalize_ledger_entries(drafts)}
    assert entries['paper_abstract:crisp']['contribution_note_source'] == expected
    assert entries['paper_abstract:disc']['contribution_note'] == 'DiSC clean note.'


def test_summary_title_mismatch_is_rejected(tmp_path):
    root = tmp_path / 'storage/extractions/run/llm-full-read/by-unit'
    root.mkdir(parents=True)
    (root / 'paper_abstract_crisp.json').write_text(json.dumps({'record': {'metadata': {'title': 'DiSC'}}, 'summary': 'DiSC summary.'}))
    assert not inventory._paper_extraction_summary(tmp_path, 'crisp', expected_title='CrispEdit')


def test_nested_group_attribution_does_not_depend_on_json_key_order():
    spec = _spec({'groups': [{'id': 'env', 'label': 'AutoControl Arena', 'members': ['lab']}], 'nodes': [{'id': 'lab', 'label': 'AgentLAB'}]})
    assert 'unlabeled_composition' in {f.code for f in source_fidelity_errors(spec, _slide('AutoControl Arena', 'AgentLAB'))}


def test_bold_work_names_count_as_works_but_a_bold_lead_sentence_does_not():
    # ICML 2026 deck 08 SRC-002: T-measure's note described the evaluability framework.
    from conference_pipeline.presentation.source_inventory import _note_is_not_a_contribution
    note = ('**A metric needs a finite-sample justification, not just an intuitive name.** '
            '**A Theoretical Framework for Statistical Evaluability** examines estimability. '
            '**T-measure** illustrates the complementary problem.')
    assert _note_is_not_a_contribution(note, 'T-measure: A Topology-Consistent Metric')
    lead_only = '**A metric needs justification.** T-measure scores topology in binary segmentation.'
    assert not _note_is_not_a_contribution(lead_only, 'T-measure: A Topology-Consistent Metric')


def test_an_analysis_line_that_never_names_its_work_falls_back(tmp_path, monkeypatch):
    # ICML 2026 deck 08 SRC-021: "A credible program should use both ..." bound to an adversarial-evaluation paper.
    draft = {'source_id': 'paper_abstract:adv', 'resolved_title': 'A New Case for Adversarial Evaluation',
             'contribution_note': 'A credible program should use both, with distinct claims.', 'contribution_note_source': 'analysis_line'}
    monkeypatch.setattr(inventory, '_paper_extraction_summary', lambda *a, **k: 'The paper argues for adversarial evaluation of models.')
    monkeypatch.setattr(inventory, '_narrative_pack_paper_notes', lambda *a: {})
    monkeypatch.setattr(inventory, '_openreview_abstract_index', lambda *a: {})
    inventory._enrich_paper_contribution_notes(tmp_path, {'paper_abstract:adv': draft}, topic_id='t')
    assert draft['contribution_note_source'] == 'full_read_summary'


def test_a_works_own_full_read_summary_beats_a_named_analysis_line(tmp_path, monkeypatch):
    # ICML 2026 deck 10 SRC-018: the analysis line named TICL and Fed-CAW in plain text.
    draft = {'source_id': 'paper_abstract:trace', 'resolved_title': 'TRACE: Trajectory Recovery for Mechanism Evolution',
             'contribution_note': 'Adaptation is entering causal inference. TICL learns at test time; Fed-CAW weights clients; TRACE models evolving mechanisms.',
             'contribution_note_source': 'analysis_line'}
    monkeypatch.setattr(inventory, '_paper_extraction_summary', lambda *a, **k: 'The paper proposes TRACE, a mixture-of-experts method that recovers evolving mechanisms.')
    monkeypatch.setattr(inventory, '_narrative_pack_paper_notes', lambda *a: {})
    monkeypatch.setattr(inventory, '_openreview_abstract_index', lambda *a: {})
    inventory._enrich_paper_contribution_notes(tmp_path, {'paper_abstract:trace': draft}, topic_id='t')
    assert draft['contribution_note'].startswith('The paper proposes TRACE')
    assert draft['contribution_note_source'] == 'full_read_summary'


def test_a_note_shared_by_two_works_falls_back_to_each_works_own_summary(tmp_path, monkeypatch):
    # ICML 2026 deck 07: PolySAE and Ensembling SAEs carried one group sentence.
    shared = 'Sparse Autoencoders can be extended with polynomial decoding, or ensembled to improve stability.'
    drafts = {f'paper_abstract:{pid}': {'source_id': f'paper_abstract:{pid}', 'resolved_title': title, 'contribution_note': shared, 'contribution_note_source': 'packet_claim'}
              for pid, title in (('poly', 'PolySAE: Feature Interactions'), ('ens', 'Ensembling Sparse Autoencoders'))}
    own = {'poly': 'The paper introduces PolySAE, polynomial decoding for SAEs.', 'ens': 'The paper proposes SAE ensembles for better reconstruction.'}
    monkeypatch.setattr(inventory, '_paper_extraction_summary', lambda ws, pid, **k: own[pid])
    monkeypatch.setattr(inventory, '_narrative_pack_paper_notes', lambda *a: {})
    monkeypatch.setattr(inventory, '_openreview_abstract_index', lambda *a: {})
    inventory._enrich_paper_contribution_notes(tmp_path, drafts, topic_id='t')
    assert drafts['paper_abstract:poly']['contribution_note'] == own['poly']
    assert drafts['paper_abstract:ens']['contribution_note'] == own['ens']
    assert {d['contribution_note_source'] for d in drafts.values()} == {'full_read_summary'}


def test_clean_packet_note_is_preserved(tmp_path, monkeypatch):
    draft = {'source_id': 'paper_abstract:crisp', 'resolved_title': 'CrispEdit', 'contribution_note': 'CrispEdit clean single-work note.', 'contribution_note_source': 'packet_claim'}
    monkeypatch.setattr(inventory, '_paper_extraction_summary', lambda *a, **k: pytest.fail('a clean note must not be replaced'))
    inventory._enrich_paper_contribution_notes(tmp_path, {'paper_abstract:crisp': draft}, topic_id='t')
    assert draft['contribution_note'] == 'CrispEdit clean single-work note.'
    assert draft['contribution_note_source'] == 'packet_claim'


def test_transcript_fallback_and_topic_summary_title_check(tmp_path):
    root = tmp_path / 'storage/extractions/run/llm-full-read/by-topic'
    root.mkdir(parents=True)
    sid = 'speech_transcript:oral3a:segment:00003'
    (root / 't.json').write_text(json.dumps({'items': [
        {'unit_id': sid, 'metadata': {'title': 'Other talk'}, 'summary': 'Wrong summary.'},
        {'unit_id': sid, 'metadata': {'title': 'AutoControl Arena'}, 'summary': 'AutoControl Arena evaluates agents.'},
    ]}))
    drafts = {sid: {'source_id': sid, 'resolved_title': 'AutoControl Arena', 'contribution_note': ''}}
    inventory._enrich_paper_contribution_notes(tmp_path, drafts, topic_id='t')
    assert drafts[sid]['contribution_note'] == 'AutoControl Arena evaluates agents.'
    assert drafts[sid]['contribution_note_source'] == 'full_read_summary'


def test_no_note_names_a_different_ledger_work_at_finalization():
    drafts = {'a': {'source_id': 'a', 'resolved_title': 'AutoControl Arena', 'contribution_note': 'MultiBreak attacks agents.'}, 'b': {'source_id': 'b', 'resolved_title': 'MultiBreak', 'contribution_note': 'MultiBreak attacks agents.'}}
    entries = {e.source_id: e.to_dict() for e in inventory._finalize_ledger_entries(drafts)}
    assert not entries['a']['contribution_note']
    assert entries['a']['contribution_note_source'] == 'none'


def test_missing_data_legend_fits_six_column_matrix():
    from conference_pipeline.figures.compiler import compile_figure
    from conference_pipeline.figures.examples import EXAMPLES
    payload = json.loads(json.dumps(EXAMPLES['feature_matrix']))
    payload['content']['rows'] = ['SRC-002', 'SRC-004']
    payload['content']['cols'] = [f'Property {i}' for i in range(6)]
    payload['content']['cells'] = [{'row': row, 'col': col, 'value': ['yes', 'partial', 'no', 'not_reported'][i % 4]} for row in payload['content']['rows'] for i, col in enumerate(payload['content']['cols'])]
    svg = compile_figure(payload).svg
    assert 'not in source summaries' in svg and 'Not Reported' not in svg


def test_ledger_build_repairs_wrong_analysis_notes_without_conference_data(tmp_path):
    (tmp_path / 'analysis/topic_reference_ledgers').mkdir(parents=True)
    ledger = tmp_path / 'analysis/topic_reference_ledgers/t.json'
    ledger.write_text(json.dumps({'entries': [
        {'canonical_id': 'paper_abstract:crisp', 'titles': ['CrispEdit'], 'metadata': {'authors': ['A Person']}},
        {'canonical_id': 'paper_abstract:disc', 'titles': ['DiSC'], 'metadata': {'authors': ['B Person']}},
    ]}))
    (tmp_path / 'analysis/report.md').write_text('- DiSC and ALER improve reasoning. https://openreview.net/forum?id=disc https://openreview.net/forum?id=aler https://openreview.net/forum?id=crisp\n')
    extraction = tmp_path / 'storage/extractions/run/llm-full-read/by-unit'
    extraction.mkdir(parents=True)
    (extraction / 'paper_abstract_crisp.json').write_text(json.dumps({'record': {'metadata': {'title': 'CrispEdit'}}, 'summary': 'CrispEdit improves editing.'}))
    (tmp_path / 'inventory.json').write_text(json.dumps({'topic_id': 't', 'source_records': [
        {'kind': 'reference_ledger', 'path': 'analysis/topic_reference_ledgers/t.json'},
        {'kind': 'analysis_report', 'path': 'analysis/report.md'},
    ]}))
    inventory.build_speaker_author_work_ledger(tmp_path, inventory_json=tmp_path / 'inventory.json', output_json=tmp_path / 'result.json', receipt_path=tmp_path / 'receipt.json')
    entries = {e['source_id']: e for e in json.loads((tmp_path / 'result.json').read_text())['entries']}
    assert entries['paper_abstract:crisp']['contribution_note'] == 'CrispEdit improves editing.'
    assert entries['paper_abstract:crisp']['contribution_note_source'] == 'full_read_summary'
    assert entries['paper_abstract:disc']['contribution_note'].startswith('DiSC and ALER')
    assert entries['paper_abstract:disc']['contribution_note_source'] == 'analysis_line'


def test_negative_cell_bound_by_source_label_uses_its_own_note():
    spec = replace(_spec({'rows': ['SRC-000', 'SRC-001'], 'cols': ['Deployment'], 'cells': [{'row': 'SRC-000', 'col': 'Deployment', 'value': 'no'}]}), archetype='feature_matrix')
    slide = _slide('SSL', 'LiMuon')
    slide['approved_sources'][1]['contribution_note'] = 'LiMuon does not test deployment.'
    assert 'unsupported_negative_cell' in {f.code for f in source_fidelity_errors(spec, slide)}


@pytest.mark.parametrize('note', [
    # ICML 2026 deck 05 SRC-013: a resource line with a bare dash and an OpenReview link.
    '*PLoRA: Efficient Concurrent LoRA Training for Large Language Models* — ; [OpenReview](https://openreview.net/forum?id=khYw62jVZ2).',
    # Deck 03 SRC-006: presentation artifacts with a cached media path.
    'Presentation artifacts: *AgentLAB*, ; *AutoControl Arena*, , `cache/acquisition-downloads/media_asset/poster-63362/slideslive_audio/raw.mp3`',
    # Deck 04 SRC-015: a packet claim that only repeats the title.
    'One-shot Entropy Minimization for Language Model Reasoning',
    'Poster: One-shot Entropy Minimization for Language Model Reasoning',
])
def test_a_resource_line_or_bare_title_is_not_a_contribution_note(note):
    from conference_pipeline.presentation.source_inventory import _note_is_not_a_contribution
    assert _note_is_not_a_contribution(note, 'Poster: One-shot Entropy Minimization for Language Model Reasoning')


def test_a_real_one_sentence_note_still_counts_as_a_contribution():
    from conference_pipeline.presentation.source_inventory import _note_is_not_a_contribution
    note = 'One-shot Entropy Minimization sharpens a model\'s confidence from a single example to improve reasoning.'
    assert not _note_is_not_a_contribution(note, 'One-shot Entropy Minimization for Language Model Reasoning')


def test_a_section_paragraph_naming_several_works_is_not_a_contribution_note():
    # ICML 2026 deck 05 SRC-014: the analysis line for Grouter opened on Spectron.
    from conference_pipeline.presentation.source_inventory import _note_is_not_a_contribution
    note = ('**Model structure can be chosen for efficiency before training ends.** '
            '*Stabilizing Native Low-Rank LLM Pretraining* introduces Spectron. '
            '*Grouter: Decoupling Routing from Representation for Accelerated MoE Train...')  # truncated mid-title
    assert _note_is_not_a_contribution(note, 'Grouter: Decoupling Routing from Representation for Accelerated MoE Training')
    # A bold lead naming only this work is still its own note.
    own = '**LiMuon** approaches optimizer efficiency from a different direction, proposing a lighter optimizer.'
    assert not _note_is_not_a_contribution(own, 'LiMuon: Light and Fast Muon Optimizer for Large Models')


def test_a_session_recording_segment_matches_on_its_talk_title(tmp_path):
    # ICML 2026 deck 03 SRC-030: segment 35 of the Pluralistic Alignment Workshop recording.
    # metadata.title is the session; metadata.talk_title is the talk. The right summary was
    # on disk and the title check, preferring ``title``, rejected it on every segment.
    root = tmp_path / 'storage/extractions/run/llm-full-read/by-unit'
    root.mkdir(parents=True)
    sid = 'speech_transcript:plural:segment:00035'
    (root / 'speech_transcript_plural_segment_00035.json').write_text(json.dumps({
        'record': {'metadata': {'title': 'Workshop: Pluralistic Alignment Workshop',
                                'talk_title': 'Of Myself I Contain Multitudes: Why Value Pluralism Matters for AI'}},
        'summary': "The talk presents a 'Value Compass' benchmark for evaluating AI value orientations."}))
    note = inventory._transcript_extraction_summary(
        tmp_path, sid, expected_title='Of Myself I Contain Multitudes: Why Value Pluralism Matters for AI')
    assert 'Value Compass' in note
    # A different talk from the same session is still refused.
    assert inventory._transcript_extraction_summary(tmp_path, sid, expected_title='Some Other Talk') == ''


def test_a_predicate_that_names_a_bound_works_own_subject_is_supported():
    # ICML 2026 deck 03 slides 20/21: "Symbolic certification" is the subject of the bound
    # position paper "Certified Correctness in Neural Reasoning"; its note says "proposes".
    # Checking the note alone called the hedged slide an unsupported 'certif' outcome.
    from conference_pipeline.figures.lint import unsupported_outcomes
    from conference_pipeline.figures.spec import load_spec
    spec = load_spec({'spec_version': 'figure-spec/1', 'figure_id': 'f', 'slide_id': 's', 'archetype': 'contrast_pair',
                      'title': 'Two routes to safety', 'takeaway': 'Certify each output, or design safety in',
                      'claim': 'Kong, Zhang and Liu argue for certifying each instance via a symbolic checker.',
                      'grounding': {'kind': 'schematic'},
                      'content': {'left': {'id': 'a', 'title': 'Runtime certification (argued)', 'items': ['A position, not a delivered system']},
                                  'right': {'id': 'b', 'title': 'Design-time objective', 'items': ['Safety by design']}}})
    bound = [{'label': 'SRC-028', 'title': 'Poster: Position: Certified Correctness in Neural Reasoning',
              'contribution_note': 'The speaker proposes a multi-agent framework (PVS) that integrates neural proposers with symbolic verifiers.'}]
    assert unsupported_outcomes(spec, {'approved_sources': bound}) == []
    # The same words against a work whose title and note never mention certification still fail.
    other = [{'label': 'SRC-029', 'title': 'Keynote: Scientist AI', 'contribution_note': 'Safety by design through honesty.'}]
    assert [f.code for f in unsupported_outcomes(spec, {'approved_sources': other})] == ['unsupported_outcome'] * 2


def test_an_acronym_title_does_not_claim_an_ordinary_word():
    # ICML 2026 deck 04: "REAL: ..." matched "real-world" and disowned FG-TTL's every note.
    drafts = {
        "paper_abstract:fgttl": {"resolved_title": "Future-Gain Guided Test-Time Learning for Large Language Models"},
        "paper_abstract:real": {"resolved_title": "REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering"},
    }
    note = "The paper proposes FG-TTL, adapting LLMs to distribution shifts during real-world deployment."
    assert inventory._note_belongs_to_entry(note, "paper_abstract:fgttl", drafts)
    assert not inventory._note_belongs_to_entry("REAL aligns reasoning pivots for VQA.", "paper_abstract:fgttl", drafts)
