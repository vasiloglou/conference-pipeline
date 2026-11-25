from __future__ import annotations

import dataclasses

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.analysis import exhaustive_summaries as summaries
from conference_pipeline.analysis.exhaustive_summaries import EvidenceItem
from conference_pipeline.model_governance import ModelGovernanceError
from conference_pipeline.model_governance.registry import load_registry


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _item(unit_id: str, kind: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        unit_id=unit_id,
        input_kind=kind,
        text=text,
        text_tokens=summaries.token_count(text),
        text_chars=len(text),
        source_artifact_id=unit_id.split(":segment:", 1)[0],
        source_artifact_path="source",
        metadata={"talk_title": "Talk"} if kind == "transcript_segment" else {"title": "Paper"},
        llm_summary="prior summary",
        llm_reason="prior reason",
        evidence_phrases=("evidence",),
        confidence=0.9,
    )


def test_chunk_evidence_items_respects_token_budget() -> None:
    items = [
        _item(f"speech_transcript:a:segment:{idx:05d}", "transcript_segment", "reasoning " * 200)
        for idx in range(5)
    ]

    chunks = summaries.chunk_evidence_items(
        topic_id="01_reasoning",
        category="transcripts",
        items=items,
        max_input_tokens=500,
    )

    assert len(chunks) > 1
    assert sum(len(chunk.unit_ids) for chunk in chunks) == 5
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)


def test_parse_json_object_accepts_fenced_json() -> None:
    payload = summaries.parse_json_object('```json\n{"summary": "ok"}\n```')
    assert payload == {"summary": "ok"}


def test_run_exhaustive_topic_summaries_dry_run_writes_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {
            "final_topics": [
                {"topic_id": "01_reasoning", "title": "Reasoning"},
            ]
        },
    )

    def fake_collect(
        workspace: Path,
        *,
        run_id: str,
        topic_id: str,
        category: str,
        unit_index,
    ):
        del workspace, run_id, topic_id, unit_index
        if category == "transcripts":
            return [_item("speech_transcript:a:segment:00000", "transcript_segment", "agents " * 50)]
        return [_item("paper_abstract:P1", "paper_abstract", "abstract " * 50)]

    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})
    monkeypatch.setattr(summaries, "collect_topic_category_items", fake_collect)

    receipt = summaries.run_exhaustive_topic_summaries(
        workspace=workspace,
        run_id="test-run",
        mode="dry-run",
        provider="gemini",
        model="gemini-3.1-pro-preview",
        max_input_tokens=10_000,
    )

    assert receipt["status"] == "planned"
    assert receipt["pair_count"] == 2
    assert receipt["provider_calls_made"] is False
    assert (workspace / "storage/analysis/exhaustive-summaries/test-run/summary_plan.json").exists()


def test_live_gemini_requires_provider_cost_approval(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning"}]},
    )
    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})

    receipt = summaries.run_exhaustive_topic_summaries(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        provider="gemini",
        provider_cost_approved=False,
    )

    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == "provider_cost_approval_required"
    assert receipt["provider_calls_made"] is False


def test_live_openai_requires_provider_cost_approval(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning"}]},
    )
    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})

    receipt = summaries.run_exhaustive_topic_summaries(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        provider="openai",
        model=summaries.DEFAULT_OPENAI_MODEL,
        provider_cost_approved=False,
    )

    assert receipt["status"] == "blocked"
    assert receipt["blocker_code"] == "provider_cost_approval_required"
    assert receipt["provider_calls_made"] is False


class _FakeOpenAICompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        payload = {
            "summary": "openai summary",
            "key_points": ["point"],
            "narrative": "openai narrative",
            "notable_evidence": [{"claim": "claim", "unit_ids": ["paper_abstract:P1"]}],
            "tensions_or_contrasts": [],
            "open_questions": [],
            "analysis_feed": {"what_to_emphasize": ["emphasis"]},
        }
        return SimpleNamespace(
            id="chatcmpl-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload, sort_keys=True))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=45),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FakeOpenAICompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_live_openai_writes_outputs_and_provider_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning"}]},
    )

    def fake_collect(
        workspace: Path,
        *,
        run_id: str,
        topic_id: str,
        category: str,
        unit_index,
    ):
        del workspace, run_id, topic_id, category, unit_index
        return [_item("paper_abstract:P1", "paper_abstract", "abstract " * 50)]

    fake_client = _FakeOpenAIClient()
    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})
    monkeypatch.setattr(summaries, "collect_topic_category_items", fake_collect)

    receipt = summaries.run_exhaustive_topic_summaries(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        provider="openai",
        model=summaries.DEFAULT_OPENAI_MODEL,
        categories=("papers",),
        max_input_tokens=10_000,
        provider_cost_approved=True,
        client=fake_client,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["provider"] == "openai"
    assert receipt["live_provider_calls_made"] is True
    assert receipt["paid_provider_calls_made"] is True
    assert receipt["summary_count"] if "summary_count" in receipt else True
    assert receipt["summary_json_paths"]
    assert receipt["provider_receipt_paths"]
    assert fake_client.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert fake_client.completions.calls[0]["max_completion_tokens"] == summaries.DEFAULT_MAX_OUTPUT_TOKENS


def test_live_resume_reuses_a_pair_only_while_its_inputs_are_unchanged(tmp_path: Path, monkeypatch) -> None:
    """v5.1 sub-step lineage re-stamps the 07A receipt as current, so a cached
    topic x category pair is reused only when its input digest matches."""

    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning"}]},
    )
    text = {"value": "abstract " * 50}

    def fake_collect(workspace: Path, *, run_id: str, topic_id: str, category: str, unit_index):
        del workspace, run_id, topic_id, category, unit_index
        return [_item("paper_abstract:P1", "paper_abstract", text["value"])]

    fake_client = _FakeOpenAIClient()
    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})
    monkeypatch.setattr(summaries, "collect_topic_category_items", fake_collect)

    def run() -> dict:
        return summaries.run_exhaustive_topic_summaries(
            workspace=workspace,
            run_id="test-run",
            mode="live",
            provider="openai",
            model=summaries.DEFAULT_OPENAI_MODEL,
            categories=("papers",),
            max_input_tokens=10_000,
            provider_cost_approved=True,
            client=fake_client,
            resume=True,
        )

    first = run()
    assert first["status"] == "succeeded"
    first_calls = len(fake_client.completions.calls)
    assert first_calls > 0
    # The taxonomy is an input: editing it makes the summaries stale.
    from conference_pipeline import receipt_lineage

    assert first["lineage_unbound_reason"] == "partial_scope"  # one category only
    lineage = receipt_lineage.bind_substep_lineage(workspace, "test-run", "exhaustive-topic-summaries")
    assert lineage["input_artifacts"]["topic_strategy"]["path"] == "analysis/topic_strategy.json"
    assert receipt_lineage.verify_input_artifacts(workspace, lineage, stage_id="07A") == []
    strategy = workspace / "analysis" / "topic_strategy.json"
    original_strategy = strategy.read_text(encoding="utf-8")
    _write_json(strategy, {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning (revised)"}]})
    assert [b["code"] for b in receipt_lineage.verify_input_artifacts(workspace, lineage, stage_id="07A")] == [
        "stage_contract_input_artifact_changed"
    ]
    strategy.write_text(original_strategy, encoding="utf-8")
    assert run()["reused_pair_count"] == 1
    assert len(fake_client.completions.calls) == first_calls, "unchanged inputs: reused"

    # A Markdown that is not the one its JSON was written with is not reused.
    md_path = workspace / "analysis" / "exhaustive_summaries" / "01_reasoning__papers.md"
    md_path.write_text("# half-written re-run\n", encoding="utf-8")
    assert run()["reused_pair_count"] == 0
    first_calls = len(fake_client.completions.calls)
    assert run()["reused_pair_count"] == 1

    text["value"] = "revised abstract " * 50
    changed = run()
    assert changed["reused_pair_count"] == 0
    assert len(fake_client.completions.calls) > first_calls, "changed inputs: summarized again"

    # Same text, corrected attribution: the summarizer sees the metadata too.
    calls_before = len(fake_client.completions.calls)
    original_collect = fake_collect

    def corrected_collect(workspace: Path, *, run_id: str, topic_id: str, category: str, unit_index):
        [item] = original_collect(workspace, run_id=run_id, topic_id=topic_id, category=category, unit_index=unit_index)
        return [dataclasses.replace(item, metadata={**item.metadata, "title": "Corrected Paper Title"})]

    monkeypatch.setattr(summaries, "collect_topic_category_items", corrected_collect)
    corrected = run()
    assert corrected["reused_pair_count"] == 0
    assert len(fake_client.completions.calls) > calls_before, "metadata-only change: summarized again"

    # A rewrite interrupted after the Markdown but before the JSON (which holds
    # the resume key) is never reused on the next resume.
    # (Failing the Markdown render: were the JSON written first, it would
    # already carry the new key next to the old Markdown.)
    text["value"] = "interrupted abstract " * 50
    real_render = summaries._render_markdown

    def failing_render(payload):
        raise OSError("interrupted")

    monkeypatch.setattr(summaries, "_render_markdown", failing_render)
    try:
        run()
    except OSError:
        pass
    monkeypatch.setattr(summaries, "_render_markdown", real_render)
    calls_before_resume = len(fake_client.completions.calls)
    recovered = run()
    assert recovered["reused_pair_count"] == 0
    assert len(fake_client.completions.calls) > calls_before_resume

    # The category loses all its evidence: the old summary must not survive to
    # be read downstream and certified current by the stamp.
    json_path = workspace / "analysis" / "exhaustive_summaries" / "01_reasoning__papers.json"
    assert json_path.exists()
    monkeypatch.setattr(
        summaries, "collect_topic_category_items", lambda workspace, *, run_id, topic_id, category, unit_index: []
    )
    emptied = run()
    assert not json_path.exists() and not json_path.with_suffix(".md").exists()
    assert "analysis/exhaustive_summaries/01_reasoning__papers.json" in emptied["pruned_outputs"]


def test_each_provider_default_is_a_registered_selection_for_that_provider() -> None:
    """The local default was once the task's fallback model (an OpenAI model),
    so --provider local would have asked the local backend for gpt-5.6-sol."""

    policy = load_registry().get(summaries.OWNER_ID)
    registered = {(item.provider, item.model) for item in policy.allowed_selections}
    assert ("gemini", summaries.DEFAULT_GEMINI_MODEL) in registered
    assert ("openai", summaries.DEFAULT_OPENAI_MODEL) in registered
    assert ("openrouter", summaries.DEFAULT_OPENROUTER_MODEL) in registered
    assert ("local", summaries.DEFAULT_LOCAL_MODEL) in registered
    assert summaries.DEFAULT_PROVIDER == policy.provider


def test_unregistered_provider_model_pair_fails_before_any_receipt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    with pytest.raises(ModelGovernanceError, match="not registered"):
        summaries.run_exhaustive_topic_summaries(
            workspace=workspace,
            run_id="test-run",
            mode="dry-run",
            provider="local",
            model=summaries.DEFAULT_OPENAI_MODEL,
        )
    assert not (workspace / "storage").exists()


def _tier_catalog(threshold: int | None):
    from conference_pipeline.costs.estimator import PricingCatalog

    model_data: dict[str, object] = {
        "pricing_unit": "tokens",
        "input_per_1m_usd": 2.0,
        "output_per_1m_usd": 12.0,
    }
    if threshold is not None:
        model_data.update(
            {
                "high_tier_input_token_threshold": threshold,
                "input_per_1m_usd_high": 4.0,
                "output_per_1m_usd_high": 18.0,
            }
        )
    return PricingCatalog({"providers": {"gemini": {"models": {"tiered-test-model": model_data}}}})


def test_tier_aware_ceiling_stays_below_the_catalog_threshold_with_headroom() -> None:
    from conference_pipeline.costs.estimator import tier_aware_input_ceiling

    ceiling = tier_aware_input_ceiling(
        model="tiered-test-model",
        provider="gemini",
        provider_default=400_000,
        catalog=_tier_catalog(200_000),
    )

    assert ceiling.source == "pricing_catalog_high_tier"
    assert ceiling.high_tier_input_token_threshold == 200_000
    # 20% tokenizer headroom and a 4k scaffold allowance below the threshold.
    assert ceiling.max_input_tokens == 156_000
    assert "200,000-token high-tier threshold" in ceiling.reason
    receipt = ceiling.to_receipt()
    assert receipt["max_input_tokens"] == 156_000
    assert receipt["max_input_tokens_source"] == "pricing_catalog_high_tier"


def test_tier_aware_ceiling_honours_an_explicit_value_and_says_when_it_crosses_the_tier() -> None:
    from conference_pipeline.costs.estimator import tier_aware_input_ceiling

    crossing = tier_aware_input_ceiling(
        model="tiered-test-model",
        provider="gemini",
        provider_default=400_000,
        explicit=400_000,
        catalog=_tier_catalog(200_000),
    )
    assert crossing.max_input_tokens == 400_000
    assert crossing.source == "explicit"
    assert "exceeds the 200,000-token high-tier threshold" in crossing.reason

    below = tier_aware_input_ceiling(
        model="tiered-test-model",
        provider="gemini",
        provider_default=400_000,
        explicit=100_000,
        catalog=_tier_catalog(200_000),
    )
    assert below.max_input_tokens == 100_000
    assert "exceeds" not in below.reason


def test_tier_aware_ceiling_falls_back_to_the_provider_default_without_a_tier() -> None:
    from conference_pipeline.costs.estimator import tier_aware_input_ceiling

    untiered = tier_aware_input_ceiling(
        model="tiered-test-model",
        provider="gemini",
        provider_default=400_000,
        catalog=_tier_catalog(None),
    )
    unpriced = tier_aware_input_ceiling(
        model="model-not-in-catalog",
        provider="gemini",
        provider_default=120_000,
        catalog=_tier_catalog(200_000),
    )

    assert (untiered.max_input_tokens, untiered.source) == (400_000, "provider_default")
    assert (unpriced.max_input_tokens, unpriced.source) == (120_000, "provider_default")


def test_default_summary_ceiling_reads_the_governed_models_tier_from_the_pricing_catalog() -> None:
    from conference_pipeline.costs.estimator import load_pricing_catalog

    for provider, model in (
        ("gemini", summaries.DEFAULT_GEMINI_MODEL),
        ("openai", summaries.DEFAULT_OPENAI_MODEL),
    ):
        ceiling = summaries.resolve_max_input_tokens(provider=provider, model=model)
        threshold = load_pricing_catalog().high_tier_input_token_threshold(model, provider)
        assert ceiling.high_tier_input_token_threshold == threshold
        if threshold is None:
            assert ceiling.max_input_tokens == summaries.provider_default_max_input_tokens(provider)
        else:
            assert ceiling.source == "pricing_catalog_high_tier"
            assert ceiling.max_input_tokens < threshold


def test_dry_run_receipt_records_the_tier_aware_ceiling_and_keeps_chunks_below_it(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {"final_topics": [{"topic_id": "01_reasoning", "title": "Reasoning"}]},
    )
    monkeypatch.setattr(summaries, "build_unit_index", lambda workspace: {})
    monkeypatch.setattr(
        summaries,
        "collect_topic_category_items",
        lambda workspace, **kwargs: [
            _item(f"speech_transcript:a:segment:{idx:05d}", "transcript_segment", "agents " * 400)
            for idx in range(6)
        ],
    )
    monkeypatch.setattr(
        summaries,
        "tier_aware_input_ceiling",
        lambda **kwargs: summaries.InputCeiling(
            max_input_tokens=1_000,
            source="pricing_catalog_high_tier",
            reason="kept below the test tier",
            high_tier_input_token_threshold=1_300,
        ),
    )

    receipt = summaries.run_exhaustive_topic_summaries(
        workspace=workspace,
        run_id="test-run",
        mode="dry-run",
        provider="gemini",
        categories=("transcripts",),
    )

    assert receipt["max_input_tokens"] == 1_000
    assert receipt["max_input_tokens_source"] == "pricing_catalog_high_tier"
    assert receipt["max_input_tokens_reason"] == "kept below the test tier"
    assert receipt["high_tier_input_token_threshold"] == 1_300
    assert receipt["chunk_count"] > 1
    plan = json.loads(
        (workspace / "storage/analysis/exhaustive-summaries/test-run/summary_plan.json").read_text()
    )
    assert plan["max_input_tokens_source"] == "pricing_catalog_high_tier"
    assert all(pair["max_input_tokens"] == 1_000 for pair in plan["pairs"])


def test_stage_refuses_and_asks_when_no_summary_model_is_chosen(tmp_path: Path) -> None:
    """07A is a stop point: no --provider and no workflow.exhaustive_summary_model
    means the operator has not chosen, so nothing is written and the error asks."""

    from conference_pipeline.analysis.summary_model_choice import (
        SummaryModelNotChosenError,
        allowed_aliases,
        resolve_summary_model,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(SummaryModelNotChosenError, match="Ask the operator"):
        summaries.run_exhaustive_topic_summaries(workspace=workspace, run_id="test-run", mode="dry-run")
    assert not (workspace / "storage").exists()

    # The registered options are named by catalog alias and include the default,
    # the fallback and the quality option.
    aliases = allowed_aliases()
    assert {"luna", "kimi_openrouter", "sol", "gemini_pro"} <= set(aliases)

    # An explicit provider is the operator's answer; the model defaults to the
    # task's registered selection for that provider.
    choice = resolve_summary_model(workspace, provider="openrouter")
    assert (choice.provider, choice.model, choice.source) == ("openrouter", summaries.DEFAULT_OPENROUTER_MODEL, "cli")

    # The conference config is the durable answer, resolved through the catalog.
    (workspace / "config.yaml").write_text(
        "conference:\n  name: New Conf\nworkflow:\n  exhaustive_summary_model: kimi_openrouter\n"
        "  exhaustive_summary_model_decision: operator test\n",
        encoding="utf-8",
    )
    choice = resolve_summary_model(workspace)
    assert choice.to_payload() == {
        "provider": "openrouter",
        "model": summaries.DEFAULT_OPENROUTER_MODEL,
        "alias": "kimi_openrouter",
        "source": "config",
        "decision": "operator test",
    }

    # An alias the registry does not allow for the task is rejected, not defaulted.
    (workspace / "config.yaml").write_text(
        "conference:\n  name: New Conf\nworkflow:\n  exhaustive_summary_model: opus\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be one of"):
        resolve_summary_model(workspace)


def test_openrouter_summary_call_sends_the_bare_slug_and_receipts_the_catalog_name(tmp_path):
    import json
    from types import SimpleNamespace

    from conference_pipeline.provider_receipts import provider_receipt_context

    sent: dict = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            id="g",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "x"}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with provider_receipt_context(workspace_root=tmp_path, stage_id="07a"):
        payload, receipt_path, cost = summaries._call_openrouter(
            client=client, model="openrouter/moonshotai/kimi-k3", prompt="p", max_output_tokens=123,
        )
    assert payload == {"summary": "x"}
    assert sent["model"] == "moonshotai/kimi-k3"
    assert sent["max_tokens"] == 123 and sent["response_format"] == {"type": "json_object"}
    receipt = json.loads(Path(receipt_path).read_text())
    assert (receipt["provider"], receipt["model"]) == ("openrouter", "openrouter/moonshotai/kimi-k3")
    assert cost and cost > 0, "the pricing catalog prices the OpenRouter model"
