"""A malformed digest reply is repaired in conversation, not re-asked blind.

On 2026-09-25 Opus 5.5 returned a topic digest with an unescaped quote twice in
a row: re-sending the same prompt with "return only JSON" reproduced the same
defect and failed the whole executive build. The retry now shows the model its
own reply and the parser error, and a reply cut off at max_tokens is retried
with a larger budget.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    script_path = REPO_ROOT / "scripts" / "presentations" / "generate_executive_topic_digests.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reply(text: str, stop_reason: str = "end"):
    return SimpleNamespace(
        text=text, stop_reason=stop_reason, block_types=("text",), receipt=None,
        input_tokens=10, output_tokens=5,
    )


@pytest.fixture
def digests(monkeypatch):
    module = _load()
    calls: list[dict] = []
    replies: list = []

    def fake_complete(task_id, **kwargs):
        calls.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(module, "complete", fake_complete)
    return module, calls, replies


def _call(module):
    return module.call_model(system="s", user="u", model="m", max_tokens=100, caller="t")


def test_parse_error_is_repaired_with_the_reply_in_context(digests):
    module, calls, replies = digests
    replies += [_reply('{"one_liner": "the "best" model"}'), _reply('{"one_liner": "the \\"best\\" model"}')]
    digest, usage, _ = _call(module)
    assert digest == {"one_liner": 'the "best" model'}
    repair = calls[1]["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"]
    assert repair[1]["content"] == '{"one_liner": "the "best" model"}'
    assert "not valid JSON" in repair[2]["content"]
    assert usage == {"input_tokens": 20, "output_tokens": 10}


def test_truncated_reply_gets_a_larger_budget(digests):
    module, calls, replies = digests
    replies += [_reply('{"one_liner": "cut', stop_reason="max_tokens"), _reply('{"one_liner": "whole"}')]
    digest, _, _ = _call(module)
    assert digest == {"one_liner": "whole"}
    assert calls[1]["max_tokens"] == 200
    assert calls[1]["messages"] == [{"role": "user", "content": "u"}]


def test_gives_up_after_three_bad_replies(digests):
    module, calls, replies = digests
    replies += [_reply("not json")] * 3
    with pytest.raises(RuntimeError, match="valid JSON"):
        _call(module)
    assert len(calls) == 3


def test_the_digest_model_can_be_any_one_the_task_allows(monkeypatch):
    """Not an Opus-only stage: a Kimi digest model resolves to OpenRouter and its
    credential is the one required -- Anthropic's is not."""

    module = _load()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    module.require_credentials("openrouter/moonshotai/kimi-k3")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        module.require_credentials("openrouter/moonshotai/kimi-k3")
