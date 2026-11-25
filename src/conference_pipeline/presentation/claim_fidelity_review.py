"""One governed, text-only fidelity sweep before expensive artifact review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from conference_pipeline.model_governance.llm_client import complete
from conference_pipeline.presentation.claim_fidelity import bound_source_records

TASK_ID = 'presentation.claim_fidelity_review'
# Reasoning tokens count against this; 8000 truncated a 24-slide deck (ICML 2026 deck 03).
MAX_OUTPUT_TOKENS = 32000


def build_fidelity_prompt(*, binding: dict, specs: list[dict], narration: str = '') -> str:
    return '''Review every slide against its own bound source records. This is an
ADVISORY review with a narrow blocking core. Contribution notes are short
summaries, not the whole paper: absence of a detail in a note is NOT evidence
against it. Do not demand note-level support for every explanatory word.

Assign severity "blocker" ONLY to:
(a) a statement directly contradicting a bound note or title (e.g. "inaudible"
    versus audio people overlook; "weights unchanged" for a method updating weights);
(b) clear misattribution: a result, method or measure credited to a different
    work or author than the one bound;
(c) an unsupported_comparator: a specific baseline/comparator/competitor named
    ("vs X", "beats X", "compared with X", "like X") that the bound note does
    not name. "1.72x over baselines" does not license naming a baseline;
(d) a position paper's proposal stated as a demonstrated result.
Two works' methods drawn or narrated as one pipeline is "composed_sources",
a blocker without the figure spec's synthesis_label. With synthesis_label set,
it is advisory: judge it as the talk's declared synthesis.
Everything else is "advisory". Plain explanatory/pedagogical description of how
a cited method works is fine. Content labelled as the talk's own synthesis or
proposal ("the talk proposes", "untested end-to-end") is fine. Wording scoped
to the talk ("the records we drew on", "not reported here", "we don't compare
scores here") is fine. Do not turn these into blockers or infer paper-wide
absence claims from talk-scoped wording. Missing detail alone cannot justify
an overclaim finding. Flag uncertainty for Gate A to confirm or dismiss.

If a bound record's title and contribution_note appear to describe different
works, use code "ledger_mismatch", always advisory. Surface these prominently
so Gate A can check the ledger rather than blaming the slide for bad evidence.

Return ONLY JSON {"status":"pass|fail", "findings":[{"slide_id":"...",
"target":"figure|narration", "severity":"blocker|advisory",
"code":"overclaim|misattribution|composed_sources|position_as_result|absence_claim|unsupported_number|unsupported_comparator|ledger_mismatch",
"source_labels":["SRC-..."], "issue":"...", "suggested_change":"..."}]}.
Status is fail only if at least one finding is a blocker; advisory findings
remain in a passing report. Review all slides, but do not manufacture findings.
Do not review layout, aesthetics or receipt bookkeeping. slide_id and
figure_id are file slugs fixed when the deck was first drafted, not text the
audience sees; judge the title, claim, takeaway and labels, never the ids.

BOUND SOURCE RECORDS:\n''' + json.dumps(bound_source_records(binding), ensure_ascii=False) + '\nFIGURE SPECS:\n' + json.dumps(specs, ensure_ascii=False) + '\nNARRATION:\n' + narration


ADVISORY_ONLY_CODES = frozenset({'ledger_mismatch'})


def validate_verdict(value: object, slide_ids: set[str], synthesis_labels: dict[str, bool] | None = None) -> dict:
    if not isinstance(value, dict) or value.get('status') not in {'pass', 'fail'} or not isinstance(value.get('findings'), list):
        raise ValueError('invalid fidelity verdict')
    findings = []
    for row in value['findings']:
        if (not isinstance(row, dict) or row.get('slide_id') not in slide_ids
            or row.get('target') not in {'figure', 'narration'}
            or row.get('severity') not in ('blocker', 'advisory')
            or row.get('code') not in {'overclaim', 'misattribution', 'composed_sources', 'position_as_result', 'absence_claim', 'unsupported_number', 'unsupported_comparator', 'ledger_mismatch'}
            or not all(isinstance(row.get(k), str) and row[k].strip() for k in ('issue', 'suggested_change'))
            or not isinstance(row.get('source_labels'), list)
            or not all(isinstance(label, str) and label.strip() for label in row['source_labels'])):
            raise ValueError('invalid fidelity finding')
        severity = row['severity']
        if row['code'] in ADVISORY_ONLY_CODES:
            severity = 'advisory'
        elif row['code'] == 'composed_sources':
            severity = 'advisory' if (synthesis_labels or {}).get(row['slide_id'], False) else 'blocker'
        elif row['code'] == 'unsupported_comparator':
            severity = 'blocker'
        findings.append({**row, 'severity': severity})
    return {**value, 'findings': findings,
            'status': 'fail' if any(row['severity'] == 'blocker' for row in findings) else 'pass'}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# The rules themselves are part of every slide's digest: a prompt change re-judges all slides.
RULES_SHA256 = hashlib.sha256(build_fidelity_prompt(binding={}, specs=[]).encode()).hexdigest()


def slide_narration(deck_dir: Path) -> dict[str, str] | None:
    """Spoken text per slide, both variants, from the narration source of truth."""
    path = deck_dir / 'engaging_narration_raw.json'
    if not path.is_file():
        return None
    out: dict[str, str] = {}
    for slide in json.loads(path.read_text(encoding='utf-8')).get('slides') or []:
        if isinstance(slide, dict) and slide.get('slide_id'):
            turns = ' '.join(f"{t.get('speaker')}: {t.get('text')}" for t in slide.get('two_host') or [] if isinstance(t, dict))
            out[str(slide['slide_id'])] = f"single: {slide.get('single') or ''}\ntwo_host: {turns}"
    return out


def settled_slides(prior: dict | None, digests: dict[str, str], *, file_hashes: dict[str, str] | None = None,
                   spec_files: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """Slides whose inputs are unchanged since a report that found no blocker on them.

    An LLM verdict is not stable across calls: re-judging an unchanged, already
    passing slide only re-rolls it (ICML 2026, 2026-11-06: identical specs passed,
    then failed). Such slides keep their verdict -- and their advisory findings --
    and are not sent to the model again. Maps slide_id -> carried findings.
    """
    if not isinstance(prior, dict) or prior.get('error'):
        return {}
    blocked = {row.get('slide_id') for row in prior.get('findings') or [] if row.get('severity') == 'blocker'}
    if isinstance(prior.get('slide_digests'), dict):
        unchanged = {slide_id for slide_id, digest in digests.items() if prior['slide_digests'].get(slide_id) == digest}
    else:
        # A schema-1 report has only per-file hashes: a slide is unchanged when its
        # own spec file is, and so is every shared input (binding, narration scripts).
        prior_hashes = prior.get('input_hashes') if isinstance(prior.get('input_hashes'), dict) else {}
        current = file_hashes or {}
        spec_names = set((spec_files or {}).values())
        shared = [name for name in current if name not in spec_names and name != 'figure_spec_receipt.json']
        if not prior_hashes or any(prior_hashes.get(name) != current[name] for name in shared):
            return {}
        unchanged = {slide_id for slide_id, name in (spec_files or {}).items()
                     if slide_id in digests and prior_hashes.get(name) == current.get(name)}
    return {slide_id: [dict(row, carried_forward=True) for row in prior.get('findings') or [] if row.get('slide_id') == slide_id]
            for slide_id in sorted(unchanged) if slide_id not in blocked}


def review_claim_fidelity(*, deck_dir: Path, phase: str, runner: Callable = complete, prior: dict | None = None) -> dict:
    """Read exact local inputs and judge the slides that changed; callers write failed receipts too.

    ``prior`` is this phase's previous report: slides it settled (unchanged
    digest, no blocker) are carried forward without a model call.
    """
    from conference_pipeline.model_governance.registry import load_registry
    from conference_pipeline.costs.estimator import load_pricing_catalog

    deck_dir = deck_dir.resolve()
    if phase not in {"figures", "narration"}:
        raise ValueError("unknown fidelity phase")
    spec_paths = sorted(path for path in (deck_dir / 'figure_specs').glob('*.json')
                        if not path.name.endswith('.critic.json'))
    paths = [deck_dir / 'source_binding_report.json']
    receipt_path = deck_dir / 'figure_spec_receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('status') != 'pass' or not isinstance(receipt.get('specs'), list):
            raise ValueError('passing figure authoring receipt required')
        spec_paths = []
        for row in receipt['specs']:
            path = (deck_dir / str(row.get('spec_path') or '')).resolve()
            if not path.is_relative_to((deck_dir / 'figure_specs').resolve()) or path.name.endswith('.critic.json'):
                raise ValueError('invalid active figure spec path')
            spec_paths.append(path)
        paths.append(receipt_path)
    if not spec_paths:
        raise ValueError('figure specs are required for fidelity review')
    paths.extend(spec_paths)
    if phase == 'narration':
        paths.extend([deck_dir / '6_presenter_script.ENGAGING.md', deck_dir / '6_presenter_script_notebooklm_2host.ENGAGING.md'])
    texts = {str(path.relative_to(deck_dir)): path.read_text(encoding='utf-8') for path in paths}
    binding = json.loads(texts['source_binding_report.json'])
    specs = [json.loads(texts[str(path.relative_to(deck_dir))]) for path in paths if path.parent.name == 'figure_specs']
    bound = {row['slide_id']: row['sources'] for row in bound_source_records(binding)}
    for spec in specs:
        sources = bound.get(spec.get('slide_id'))
        if not sources or any(not row['contribution_note'] for row in sources):
            raise ValueError(f"missing bound source notes for {spec.get('slide_id')}")
    if len({spec.get('slide_id') for spec in specs}) != len(specs):
        raise ValueError('duplicate active figure slide ids')
    full_narration = '\n'.join(text for name, text in texts.items() if name.endswith('.md'))
    per_slide = slide_narration(deck_dir) if phase == 'narration' else {}
    digests = {spec['slide_id']: _digest({'rules': RULES_SHA256, 'phase': phase, 'spec': spec,
                                          'sources': bound[spec['slide_id']],
                                          # Without per-slide narration every slide depends on the whole script.
                                          'narration': (per_slide.get(spec['slide_id'], '') if per_slide is not None
                                                        else hashlib.sha256(full_narration.encode()).hexdigest())})
               for spec in specs}
    file_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    spec_files = {json.loads(texts[name]).get('slide_id'): name for name in texts if name.startswith('figure_specs/')}
    settled = settled_slides(prior, digests, file_hashes=file_hashes, spec_files=spec_files)
    to_judge = [spec for spec in specs if spec['slide_id'] not in settled]
    base = {'schema_version': 2, 'task_id': TASK_ID, 'phase': phase, 'slide_digests': digests,
            'judged_slide_ids': [spec['slide_id'] for spec in to_judge],
            'carried_forward_slide_ids': sorted(settled),
            'input_hashes': file_hashes}
    carried = [row for rows in settled.values() for row in rows]
    if not to_judge and isinstance(prior, dict) and prior.get('slide_digests') == digests and prior.get('status') == 'passed':
        # Nothing to judge and nothing moved: return the same verdict unchanged, so
        # downstream steps that read this report (Gate A) see no new input.
        return prior
    if not to_judge:
        return {**base, 'status': 'passed', 'findings': carried, **_counts(carried),
                'provider': None, 'model': None, 'note': 'every slide unchanged since a report with no blocker on it; no model call'}
    if phase == 'narration' and per_slide is not None:
        narration = '\n\n'.join(f"SLIDE {spec['slide_id']}\n{per_slide.get(spec['slide_id'], '')}" for spec in to_judge)
    else:
        narration = full_narration
    judged_ids = {spec['slide_id'] for spec in to_judge}
    prompt = build_fidelity_prompt(binding={'slides': [row for row in binding.get('slides', []) if isinstance(row, dict) and row.get('slide_id') in judged_ids]},
                                   specs=to_judge, narration=narration)
    policy = load_registry().get(TASK_ID)
    # UTF-8 bytes upper-bound input token count conservatively. Refuse oversized
    # requests before spending, using the canonical local price catalog.
    estimated, _, _ = load_pricing_catalog().estimate_tokens(policy.provider, policy.model, len(prompt.encode('utf-8')), MAX_OUTPUT_TOKENS)
    if policy.budget_cap_usd is None or estimated > policy.budget_cap_usd:
        raise ValueError('claim fidelity review exceeds governed budget cap')
    response = runner(TASK_ID, prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS, caller=__file__)
    if response.stop_reason != 'end':
        raise ValueError(f'fidelity review incomplete: {response.stop_reason}')
    synthesis_labels = {spec['slide_id']: bool(str(spec.get('synthesis_label') or '').strip()) for spec in specs}
    verdict = validate_verdict(json.loads(response.text), judged_ids, synthesis_labels)
    # A finding citing a label bound to another slide (or none) is still a finding:
    # discarding the whole verdict over it (ICML 2026 deck 03) hid every real issue.
    findings = []
    for row in verdict['findings']:
        labels = {source['label'] for source in bound[row['slide_id']]}
        unbound = [label for label in row['source_labels'] if label not in labels]
        findings.append({**row,
                         **({'unbound_source_labels': unbound} if unbound else {}),
                         'repair_tool': ('scripts/presentations/patch_figure_spec.py' if row['target'] == 'figure'
                                         else 'scripts/presentations/fix_narration_script.py')})
    findings = carried + findings
    return {**base,
            'status': 'passed' if verdict['status'] == 'pass' else 'failed',
            'findings': findings, **_counts(findings),
            'provider': response.provider, 'model': response.model,
            'budget_cap_usd': policy.budget_cap_usd, 'estimated_max_cost_usd': estimated}


def _counts(findings: list[dict]) -> dict:
    return {'finding_counts': {severity: sum(row['severity'] == severity for row in findings)
                               for severity in ('blocker', 'advisory')},
            'ledger_mismatch_findings': [row for row in findings if row['code'] == 'ledger_mismatch']}
