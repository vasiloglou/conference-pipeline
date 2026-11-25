from pathlib import Path

import pytest

from conference_pipeline.analysis.local_summary_benchmark import (
    Budget,
    digest,
    pack,
    split_units,
    validate_ledger,
    validate_summary,
)


def test_oversized_source_split_preserves_all_characters_and_offsets():
    row = {"unit_id": "u", "source_artifact_id": "a", "text": "abc def " * 50}

    def prompt(rows):
        return "".join(r["text"] for r in rows) + "HEADER"

    parts = split_units([row], prompt, len, 30)
    assert "".join(p["text"] for p in parts) == row["text"]
    assert all(len(prompt([p])) <= 30 for p in parts)
    assert all(row["text"][p["start"] : p["end"]] == p["text"] for p in parts)
    assert len({p["part_id"] for p in parts}) == len(parts)


def test_full_prompt_budget_and_oversized_reduction_fail_closed():
    def prompt(rows):
        return "overhead" + "".join(x["text"] for x in rows)

    rows = [{"text": "a" * 8}, {"text": "b" * 8}]
    assert len(pack(rows, prompt, len, 20)) == 2
    with pytest.raises(ValueError, match="oversized_record"):
        pack([{"text": "x" * 30}], prompt, len, 20)


def test_accounting_and_exact_quotes():
    row = {
        "part_id": "u@10",
        "unit_id": "u",
        "source_artifact_id": "a",
        "text": "result is 42",
        "start": 10,
    }
    valid = {
        "units": [
            {
                "part_id": "u@10",
                "status": "processed",
                "claims": [{"claim": "result", "quote": "42"}],
            }
        ]
    }
    assert validate_ledger(valid, [row])[0]["start"] == 20
    with pytest.raises(ValueError, match="accounting"):
        validate_ledger({"units": []}, [row])
    valid["units"][0]["claims"][0]["quote"] = "43"
    with pytest.raises(ValueError, match="unverified_quote"):
        validate_ledger(valid, [row])


def test_budget_failed_call_reservation_survives_resume(tmp_path):
    b = Budget(tmp_path, dollars=1, hours=1)
    b.reserve("failed_call", 0.8)
    b.charge_seconds(10)
    resumed = Budget(tmp_path, dollars=1, hours=1)
    assert resumed.remaining_seconds() == 3590
    with pytest.raises(RuntimeError, match="api_budget"):
        resumed.reserve("retry", 0.3)
    assert resumed.state["reserved_usd"] == 0.8


def test_summary_unknown_reference_and_schema():
    p = {
        "summary": "s",
        "narrative": "n",
        "key_points": [],
        "notable_evidence": [{"claim": "x", "unit_ids": ["wrong"]}],
        "tensions_or_contrasts": [],
        "open_questions": [],
        "analysis_feed": {
            "what_to_emphasize": [],
            "representative_examples": [],
            "risks_or_caveats": [],
        },
    }
    with pytest.raises(ValueError, match="unknown_citation"):
        validate_summary(p, {"u"})
    p["notable_evidence"][0]["unit_ids"] = ["u"]
    assert validate_summary(p, {"u"}) == p
    del p["analysis_feed"]["risks_or_caveats"]
    with pytest.raises(ValueError, match="analysis_feed"):
        validate_summary(p, {"u"})


def test_content_digest_invalidates_changed_evidence_and_prompt():
    assert digest(["evidence", "v1"]) != digest(["changed", "v1"])
    assert digest(["evidence", "v1"]) != digest(["evidence", "v2"])


def test_quote_alignment_only_repairs_formatting():
    from conference_pipeline.analysis.local_summary_benchmark import align_quote

    source = "There is *intrinsic* resistance at $H(X|Z)$, with gain 42%."
    quote = "There is intrinsic resistance at H(X|Z), with gain 42%."
    a, b = align_quote(quote, source)
    assert source[a:b] == source
    assert align_quote(quote.replace("42", "43"), source) is None
    assert align_quote(quote.replace("resistance", "security"), source) is None


def test_interrupted_local_call_retains_time_reservation(tmp_path):
    b = Budget(tmp_path, hours=1)
    key, seconds = b.begin_local(900)
    assert seconds == 900
    assert Budget(tmp_path, hours=1).remaining_seconds() == 2700
    b.end_local(key, 20)
    assert Budget(tmp_path, hours=1).remaining_seconds() == 3580


def test_sentence_pointer_evidence_recovers_exact_original_spans():
    from conference_pipeline.analysis.local_summary_benchmark import sentence_spans

    text = "Title: Formula\n\nThe $R^2_c$ result is 42%. Limitations apply."
    row = {"unit_id": "u", "part_id": "u@20", "source_artifact_id": "a", "start": 20, "text": text}
    sentences = sentence_spans(text)
    evidence = next(s for s in sentences if "42%" in s["text"])
    p = {
        "units": [
            {
                "part_id": "u@20",
                "status": "processed",
                "claims": [
                    {"claim": "42% result", "evidence_sentence_ids": [evidence["sentence_id"]]}
                ],
            }
        ]
    }
    result = validate_ledger(p, [row])[0]
    assert result["source_spans"][0]["text"] == evidence["text"]
    assert result["source_spans"][0]["start"] == 20 + evidence["start"]
    p["units"][0]["claims"][0]["evidence_sentence_ids"] = ["s9999"]
    with pytest.raises(ValueError, match="sentence_ids"):
        validate_ledger(p, [row])


def test_quote_alignment_does_not_drop_multiplication_or_currency():
    from conference_pipeline.analysis.local_summary_benchmark import align_quote

    assert (
        align_quote(
            "The method computes ab for estimation.", "The method computes a*b for estimation."
        )
        is None
    )
    assert (
        align_quote(
            "The cost was 500 for all experiments.", "The cost was $500 for all experiments."
        )
        is None
    )


def test_content_addressed_calls_resume_without_regenerating(tmp_path):
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import Calls

    generated = []

    def generate(prompt, output_tokens):
        generated.append(prompt)
        return '{"ok":true}', {"finish_reason": "stop", "input_tokens": 3, "output_tokens": 4}

    local = SimpleNamespace(
        alias="qwen36_local", revision="test-revision", count=len, generate=generate
    )
    calls = Calls(tmp_path, local, Budget(tmp_path), 100)

    def validator(p):
        return None if p == {"ok": True} else (_ for _ in ()).throw(ValueError("bad"))

    calls("first", validator)
    calls("first", validator)
    calls("changed", validator)
    assert generated == ["first", "changed"]
    with pytest.raises(ValueError, match="ceiling"):
        calls("x" * 101, validator)


def test_truncated_response_cannot_be_resumed_as_success(tmp_path):
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import Calls

    local = SimpleNamespace(
        alias="qwen36_local",
        revision="test-revision",
        count=len,
        generate=lambda *_: (
            '{"ok":true}',
            {"finish_reason": "length", "input_tokens": 3, "output_tokens": 4},
        ),
    )
    calls = Calls(tmp_path, local, Budget(tmp_path), 100)
    with pytest.raises(ValueError, match="truncated"):
        calls("hello", lambda p: None)
    import json

    receipt = json.loads(next((tmp_path / "calls").glob("*.json")).read_text())
    assert receipt["status"] == "failed"
    assert receipt["raw_text"] == '{"ok":true}'


def test_only_stopped_local_outputs_reuse_a_smaller_cap(tmp_path):
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import Calls

    generated = []

    def generate(prompt, output_tokens):
        generated.append(output_tokens)
        return '{"ok":true}', {"finish_reason": "stop", "output_tokens": 4}

    local = SimpleNamespace(
        alias="qwen36_local", revision="test-revision", count=len, generate=generate
    )
    budget = Budget(tmp_path)
    Calls(tmp_path, local, budget, 100, 4096)("same", lambda p: None)
    larger = Calls(tmp_path, local, budget, 100, 8192)
    assert larger("same", lambda p: None) == {"ok": True}
    assert generated == [4096]
    assert larger.used[0]["cache_hit"] is True
    with pytest.raises(ValueError, match="invalid_payload"):
        larger("same", lambda p: (_ for _ in ()).throw(ValueError("invalid_payload")))


def test_retry_preserves_the_original_failed_receipt(tmp_path):
    import json
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import Calls

    responses = iter(["{broken", '{"ok":true}'])
    local = SimpleNamespace(
        alias="qwen36_local",
        revision="test",
        count=len,
        generate=lambda *_: (next(responses), {"finish_reason": "stop"}),
    )
    calls = Calls(tmp_path, local, Budget(tmp_path), 100)

    def validator(p):
        if p != {"ok": True}:
            raise ValueError("invalid_payload")

    with pytest.raises(ValueError):
        calls("same", validator)
    calls("same", validator)
    archives = list((tmp_path / "calls" / "attempts").glob("*.json"))
    assert len(archives) == 1
    prior = json.loads(archives[0].read_text())
    assert prior["status"] == "failed"
    assert prior["raw_text"] == "{broken"


def test_separate_cloud_evaluation_allocation(tmp_path):
    b = Budget(tmp_path)
    b.reserve("eval1", 14.0, "evaluation")
    b.reserve("synthesis", 9.0, "synthesis")
    with pytest.raises(RuntimeError, match="evaluation_budget"):
        b.reserve("eval2", 2.0, "evaluation")
    with pytest.raises(RuntimeError, match="synthesis_budget"):
        b.reserve("synthesis2", 2.0, "synthesis")
    assert b.state["reserved_usd"] == 23.0


def test_successful_cost_settlement_refunds_unused_tokens_but_retains_other_reservations(tmp_path):
    budget = Budget(tmp_path)
    budget.reserve("completed", 3, "evaluation")
    budget.reserve("failed-unknown", 2, "evaluation")
    budget.settle("completed", 0.25, "evaluation")
    assert budget.state["reserved_usd"] == 2.25
    assert budget.state["by_purpose"]["evaluation"] == 2.25
    assert budget.state["reservations"]["failed-unknown"] == 2
    assert budget.state["original_reservations"]["completed"] == 3
    budget.settle("completed", 0.25, "evaluation")
    assert budget.state["reserved_usd"] == 2.25
    with pytest.raises(ValueError, match="exceeds_reservation"):
        budget.settle("completed", 10, "evaluation")


def test_hybrid_uses_complete_reduction_level_instead_of_partial_ledger(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "benchmark_eval", Path(__file__).parents[1] / "scripts/analysis/evaluate_local_summaries.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = {
        "summary": "s",
        "narrative": "n",
        "key_points": [],
        "notable_evidence": [{"claim": "x", "unit_ids": ["u"]}],
        "tensions_or_contrasts": [],
        "open_questions": [],
        "analysis_feed": {
            "what_to_emphasize": [],
            "representative_examples": [],
            "risks_or_caveats": [],
        },
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(p))
    (tmp_path / "ledger.json").write_text(json.dumps({"claims": [{"claim": "x" * 10000}]}))
    (tmp_path / "synthesis_packs.json").write_text(json.dumps({"levels": [[p, p], [p]]}))

    class FakeCalls:
        count = staticmethod(len)
        ceiling = 5000

        def __call__(self, prompt, validator, **kwargs):
            validator(p)
            return p

    out = tmp_path / "hybrid.json"
    module.hybrid({"title": "test", "rows": [{"unit_id": "u"}]}, FakeCalls(), source, out)
    coverage = json.loads(out.with_suffix(".input-coverage.json").read_text())
    assert coverage["mode"] == "complete_bounded_reduction_level"
    assert coverage["synthesis_pack_count"] == 2
    (tmp_path / "ledger.json").write_text(json.dumps({"claims": [{"claim": "RAW_QUOTE_CANARY"}]}))

    class PackOnlyCalls(FakeCalls):
        def __call__(self, prompt, validator, **kwargs):
            assert "RAW_QUOTE_CANARY" not in prompt
            validator(p)
            return p

    module.hybrid(
        {"title": "test", "rows": [{"unit_id": "u"}]}, PackOnlyCalls(), source, out, "summary-packs"
    )
    coverage = json.loads(out.with_suffix(".input-coverage.json").read_text())
    assert coverage["evidence_mode"] == "summary-packs"


def test_numeric_diagnostic_flags_unseen_numbers_without_claiming_entailment():
    from conference_pipeline.analysis.local_summary_benchmark import numeric_support_flags

    summary = {"notable_evidence": [{"claim": "The result improves by 43%.", "unit_ids": ["u"]}]}
    rows = [{"unit_id": "u", "text": "The measured improvement was 42%."}]
    result = numeric_support_flags(summary, rows)
    assert len(result["flags"]) == 1
    assert result["status"] == "diagnostic_only"
    summary["notable_evidence"][0]["claim"] = "The result improves by 42.0%."
    assert numeric_support_flags(summary, rows)["flags"] == []


def test_recall_interval_requires_matched_findings_and_clusters_by_source():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_report",
        Path(__file__).parents[1] / "scripts/analysis/report_local_summary_benchmark.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packets = [
        {
            "candidate": candidate,
            "findings": [
                {"unit_id": unit, "finding_index": i, "recalled": recalled}
                for unit in ("u", "v")
                for i in (0, 1)
            ],
        }
        for candidate, recalled in [("a", False), ("b", True)]
    ]
    result = module.paired_recall_interval(packets, 100)
    assert result["source_unit_count"] == 2
    assert result["interval_95"] == [1, 1]
    assert result["recall_difference"] == 1
    for packet in packets:
        for finding in packet["findings"]:
            finding["source_artifact_id"] = "same-recording"
    assert module.paired_recall_interval(packets, 100)["source_cluster_count"] == 1
    packets[1]["findings"].pop()
    assert module.paired_recall_interval(packets)["status"] == "unmatched_reference_findings"


def test_complete_summary_records_compress_before_merge_and_require_shrinkage():
    import copy

    from conference_pipeline.analysis.local_summary_benchmark import (
        reduce_summaries,
        summary_prompt,
    )

    payload = {
        "summary": "s" * 1000,
        "narrative": "n" * 1000,
        "key_points": [],
        "notable_evidence": [{"claim": "finding", "unit_ids": ["u"]}],
        "tensions_or_contrasts": [],
        "open_questions": [],
        "analysis_feed": {
            "what_to_emphasize": [],
            "representative_examples": [],
            "risks_or_caveats": [],
        },
    }
    title = "test"
    heading = title + "; merge these synthesis packs, preserving evidence and contrasts"

    class FakeCalls:
        count = staticmethod(len)
        ceiling = len(summary_prompt(heading, [payload])) + 500

        def __init__(self, shrink):
            self.shrink = shrink
            self.prompts = []

        def __call__(self, prompt, validator):
            assert len(prompt) <= self.ceiling
            self.prompts.append(prompt)
            result = copy.deepcopy(payload)
            if self.shrink:
                result["summary"] = "s"
                result["narrative"] = "n"
            validator(result)
            return result

    calls = FakeCalls(True)
    result = reduce_summaries([payload, payload], title, calls, {"u"})
    assert result["summary"] == "s"
    assert len(calls.prompts) == 3
    assert "INTERMEDIATE" in calls.prompts[0]
    assert "n" * 1000 in calls.prompts[0]  # Complete source record reached the compressor.
    with pytest.raises(ValueError, match="did_not_shrink"):
        reduce_summaries([payload, payload], title, FakeCalls(False), {"u"})


def test_critic_can_reread_a_source_missing_from_the_draft(tmp_path):
    import json
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import run_strategy

    rows = [
        {"unit_id": u, "source_artifact_id": u, "text": "A supported finding."}
        for u in ("u1", "u2")
    ]

    class FakeCalls:
        count = staticmethod(len)
        ceiling = 50000
        output_tokens = 8192
        used = []
        local = SimpleNamespace(model_id="fixture", revision="fixture", hardware={})

        def __call__(self, prompt, validator):
            if prompt.startswith("Read every source part"):
                sources = json.loads(prompt.split("SOURCES:\n", 1)[1])
                result = {
                    "units": [
                        {
                            "part_id": r["part_id"],
                            "status": "processed",
                            "reason": "relevant",
                            "claims": [
                                {
                                    "claim": "A supported finding.",
                                    "theme": r["unit_id"],
                                    "caveat": "conditions",
                                    "evidence_sentence_ids": ["s0"],
                                }
                            ],
                        }
                        for r in sources
                    ]
                }
            elif prompt.startswith("Audit this draft"):
                # The draft cites u1 only; the critic must still be allowed to recover u2.
                assert '"unit_ids": ["u2"]' in prompt.split("LEDGER PACKET:")[1]
                result = {"unit_ids": ["u2"], "reasons": ["omitted finding"]}
            else:
                result = {
                    "summary": "s",
                    "narrative": "n",
                    "key_points": [],
                    "notable_evidence": [
                        {
                            "claim": "A supported finding.",
                            "unit_ids": [
                                "u2" if "verify against original evidence" in prompt else "u1"
                            ],
                        }
                    ],
                    "tensions_or_contrasts": [],
                    "open_questions": [],
                    "analysis_feed": {
                        "what_to_emphasize": [],
                        "representative_examples": [],
                        "risks_or_caveats": [],
                    },
                }
            validator(result)
            return result

    run_strategy(
        {"rows": rows, "title": "test", "category": "papers"},
        FakeCalls(),
        "ledger",
        tmp_path,
        reread_rounds=1,
    )
    critic = json.loads((tmp_path / "critic-0.json").read_text())
    assert critic["ledger_claims_reviewed"] == 2
    assert critic["selected"]["unit_ids"] == ["u2"]


def test_paper_only_cloud_scope_excludes_transcripts(tmp_path):
    import importlib.util
    import json
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "benchmark_campaign",
        Path(__file__).parents[1] / "scripts/analysis/run_local_summary_campaign.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cloud_allowed("papers", "papers")
    assert not module.cloud_allowed("papers", "transcripts")
    assert not module.cloud_allowed("none", "papers")
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"category": "transcripts"}))
    result = module.review_job(SimpleNamespace(cloud_scope="papers"), {"input": str(source)})
    assert result["status"] == "not_authorized_for_source_category"


def test_reference_preparation_respects_scope_and_deduplicates_jobs(tmp_path, monkeypatch):
    import importlib.util
    import json
    import time
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "benchmark_campaign",
        Path(__file__).parents[1] / "scripts/analysis/run_local_summary_campaign.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paper, transcript = tmp_path / "papers.json", tmp_path / "transcripts.json"
    paper.write_text(json.dumps({"category": "papers"}))
    transcript.write_text(json.dumps({"category": "transcripts"}))
    jobs = [{"input": str(p)} for p in (paper, transcript, transcript)]
    args = SimpleNamespace(
        root=tmp_path,
        cloud_scope="papers",
        deadline=time.monotonic() + 60,
        credential_file=tmp_path / "trusted.env",
    )
    called = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: called.append(command))
    results = module.prepare_references(args, jobs)
    assert len(results) == len(called) == 1
    assert str(paper) in called[0]
    args.cloud_scope = "all"
    called.clear()
    results = module.prepare_references(args, jobs)
    assert len(results) == len(called) == 2
    assert all(row["status"] == "frozen" for row in results)
    args.cloud_scope = "papers"
    args.transcript_inputs = [transcript]
    assert module.input_cloud_allowed(args, {"category": "transcripts"}, transcript)
    assert not module.input_cloud_allowed(
        args, {"category": "transcripts"}, tmp_path / "other.json"
    )


def test_invalid_batch_bisection_preserves_every_source_and_validates_children():
    import json

    from conference_pipeline.analysis.local_summary_benchmark import validated_batches

    class Calls:
        ceiling = 1000
        count = staticmethod(len)

        def __init__(self):
            self.prompts = []

        def __call__(self, prompt, validate):
            self.prompts.append(prompt)
            rows = json.loads(prompt.split('\n')[0])
            payload = {'ids': rows if len(rows) == 1 else rows[:-1]}
            validate(payload)
            return payload

    def validator_for(rows):
        def validate(payload):
            if payload['ids'] != rows:
                raise ValueError('incomplete_accounting')
        return validate

    calls = Calls()
    result = list(validated_batches(['a', 'b', 'c'], json.dumps, validator_for, calls))
    assert [x for batch, _ in result for x in batch] == ['a', 'b', 'c']
    assert all(payload['ids'] == batch for batch, payload in result)
    assert len(calls.prompts) == 7  # two failed attempts per multi-unit batch, one per leaf


def test_invalid_single_source_still_fails_after_one_repair():
    from conference_pipeline.analysis.local_summary_benchmark import validated_batches

    class Calls:
        ceiling = 1000
        count = staticmethod(len)
        attempts = 0

        def __call__(self, prompt, validate):
            self.attempts += 1
            raise ValueError('invalid_source_evidence')

    calls = Calls()
    with pytest.raises(ValueError, match='invalid_source_evidence'):
        list(validated_batches(['a'], str, lambda rows: lambda payload: None, calls))
    assert calls.attempts == 2


def test_rubric_includes_valid_cited_sources_outside_reference_sample():
    import importlib.util

    path = Path('scripts/analysis/evaluate_local_summaries.py')
    spec = importlib.util.spec_from_file_location('context_rubric_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = {u: {'unit_id': u, 'text': u} for u in ['sampled', 'cited', 'unrelated']}
    result = module.rubric_sources_for_candidates(
        sources, [sources['sampled']], {'candidate': {'unit_ids': ['cited']}}
    )
    assert {r['unit_id'] for r in result} == {'sampled', 'cited'}


def test_extraction_packets_identical_across_synthesis_context_limits(tmp_path):
    import json
    from types import SimpleNamespace

    from conference_pipeline.analysis.local_summary_benchmark import (
        available_unit_ids,
        run_strategy,
    )

    rows = [{'unit_id': f'u{i}', 'source_artifact_id': f'u{i}', 'text': 'A supported finding.'} for i in range(40)]

    class Calls:
        count = staticmethod(len)
        output_tokens = 8192
        used = []
        local = SimpleNamespace(model_id='fixture', revision='fixture', hardware={})

        def __init__(self, ceiling):
            self.ceiling = ceiling
            self.extraction_prompts = []

        def __call__(self, prompt, validate):
            if prompt.startswith('Read every source part'):
                self.extraction_prompts.append(prompt)
                sources = json.loads(prompt.split('SOURCES:\n')[1])
                payload = {'units': [{'part_id': r['part_id'], 'status': 'processed',
                           'reason': 'relevant', 'claims': [{'claim': 'A supported finding.',
                           'theme': 'common', 'evidence_sentence_ids': ['s0'], 'caveat': 'conditions'}]}
                           for r in sources]}
            else:
                evidence = json.loads(prompt.split('EVIDENCE:\n')[1])
                unit = sorted(available_unit_ids(evidence))[0]
                payload = {'summary': 's', 'narrative': 'n', 'key_points': [],
                           'notable_evidence': [{'claim': 'A supported finding.', 'unit_ids': [unit]}],
                           'tensions_or_contrasts': [], 'open_questions': [],
                           'analysis_feed': {'what_to_emphasize': [], 'representative_examples': [], 'risks_or_caveats': []}}
            validate(payload)
            return payload

    prompts = []
    for ceiling in [8192, 57344, 90112]:
        out = tmp_path/str(ceiling)
        out.mkdir()
        calls = Calls(ceiling)
        run_strategy({'rows': rows, 'title': 'test', 'category': 'papers'}, calls,
                     'ledger', out, reread_rounds=0, extraction_ceiling=8192)
        prompts.append(calls.extraction_prompts)
        assert len(json.loads((out/'coverage.json').read_text())['accounting']) == len(rows)
    assert prompts[0] == prompts[1] == prompts[2]


def _fake_chat_client(sent):
    from types import SimpleNamespace

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(
            id="r",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_cloud_calls_are_receipted_and_openai_follows_the_registry_reasoning_contract(tmp_path, monkeypatch):
    """The benchmark used to call the SDK directly: spend that no provider receipt showed, and
    gpt-6-sol without the high reasoning effort the registry requires of it."""

    import json
    from types import SimpleNamespace

    import openai

    from conference_pipeline.analysis import local_summary_benchmark as lsb
    from conference_pipeline.model_governance import model_for_alias

    sent: list[dict] = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _fake_chat_client(sent))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    local = SimpleNamespace(alias="qwen36_local", revision="r1", count=lambda prompt: 10)
    calls = lsb.Calls(tmp_path, local, lsb.Budget(tmp_path), ceiling=10**6, output_tokens=512)

    assert calls("prompt one", lambda payload: None, cloud_alias="kimi_openrouter", purpose="synthesis") == {"ok": True}
    assert calls("prompt two", lambda payload: None, cloud_alias="sol", purpose="evaluation") == {"ok": True}

    openrouter_call, openai_call = sent
    assert openrouter_call["model"] == "moonshotai/kimi-k3", "OpenRouter is sent the bare slug"
    assert openai_call["model"] == model_for_alias("sol")
    assert openai_call["reasoning_effort"] == "high", "the registry requires high effort for gpt-6-sol"
    receipts = [json.loads(path.read_text()) for path in (tmp_path / "storage" / "provider-calls").rglob("*.json")]
    assert sorted((r["provider"], r["model"]) for r in receipts) == sorted([
        ("openrouter", model_for_alias("kimi_openrouter")),
        ("openai", model_for_alias("sol")),
    ]), "each cloud call has a provider receipt under its catalog model name"
    assert {r["task_id"] for r in receipts} == {"analysis.local_summary_benchmark"}
