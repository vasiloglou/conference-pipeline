from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_model_registry_hardcoding import audit  # noqa: E402


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_argparse_default_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
import argparse


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-opus-4-8")
    return p
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert any(f["match"] == "claude-opus-4-8" for f in findings)


def test_model_kwarg_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def run():
    return call_llm(model="gpt-5.2", prompt="hi")
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert any(f["match"] == "gpt-5.2" for f in findings)


def test_model_id_kwarg_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def run():
    return call_llm(model_id="gemini-3.1-pro-preview")
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert any(f["match"] == "gemini-3.1-pro-preview" for f in findings)


def test_cli_model_flag_sibling_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
default_command = ("claude", "--model", "claude-opus-4-8", "--print")
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert any(f["match"] == "claude-opus-4-8" for f in findings)


def test_markdown_command_example_is_flagged(tmp_path: Path) -> None:
    md_root = tmp_path / "skills"
    _write(
        md_root / "some-skill" / "SKILL.md",
        """
Run the owner command:
`python scripts/select_deep_dive_questions.py --run-id x --mode live --model gpt-5.6-luna --summary`
""",
    )
    findings = audit(py_roots=[], md_roots=[md_root])
    assert any(f["match"] == "gpt-5.6-luna" for f in findings)


def test_suppressed_python_line_is_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def run():
    return call_llm(model="claude-opus-4-8")  # model-literal-ok: deliberate opus narration choice
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert findings == []


def test_suppressed_markdown_line_is_not_flagged(tmp_path: Path) -> None:
    md_root = tmp_path / "skills"
    _write(
        md_root / "some-skill" / "SKILL.md",
        """
`python scripts/select_deep_dive_questions.py --model gpt-5.6-luna --summary` (model-literal-ok: illustrative example only, registry governs the real default)
""",
    )
    findings = audit(py_roots=[], md_roots=[md_root])
    assert findings == []


def test_docstring_constant_is_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        '''
"""This module used to call claude-opus-4-8 for narration."""


def run():
    return "ok"
''',
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert findings == []


def test_comment_is_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def run():
    # historically this used gpt-5.2, now routed through the registry
    return call_llm(model=default_model("some.task"))
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert findings == []


def test_membership_test_is_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def run(x):
    return "claude-opus-4-8" in x
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert findings == []


def test_markdown_prose_sentence_is_not_flagged(tmp_path: Path) -> None:
    md_root = tmp_path / "skills"
    _write(
        md_root / "some-skill" / "SKILL.md",
        """
The deep-dive question selector runs with GPT-5.6 Luna to compress summaries,
and a future revision may switch to gemini-3.1-pro-preview if benchmarks favor it.
""",
    )
    findings = audit(py_roots=[], md_roots=[md_root])
    assert findings == []


def test_clean_files_yield_no_findings(tmp_path: Path) -> None:
    src = tmp_path / "src"
    md_root = tmp_path / "skills"
    _write(
        src / "tool.py",
        """
from conference_pipeline.model_governance.registry import default_model


def run():
    return call_llm(model=default_model("some.task"))
""",
    )
    _write(
        md_root / "some-skill" / "SKILL.md",
        """
Run the registry-governed owner: `python scripts/run_stage.py --workspace conferences/<slug>`.
""",
    )
    findings = audit(py_roots=[src], md_roots=[md_root])
    assert findings == []


def test_cli_json_output_is_parseable() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_model_registry_hardcoding.py"), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] in {"pass", "fail"}
    assert result.returncode in (0, 1)


def test_env_override_falling_back_to_a_model_literal_is_flagged(tmp_path: Path) -> None:
    """generate_engaging_narration.py chose its model as
    MODEL_ID = os.environ.get("ENGAGING_NARRATION_MODEL", "<literal>"): an
    uppercase constant AND a Call value, so neither existing rule saw it."""

    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
import os

MODEL_ID = os.environ.get("ENGAGING_NARRATION_MODEL", "claude-opus-4-8")
JUDGE = os.getenv("JUDGE_MODEL", "gemini-3.1-pro-preview")
LABEL = os.environ.get("DECK_TITLE", "claude-opus-4-8 notes")
""",
    )
    matches = {f["match"] for f in audit(py_roots=[src], md_roots=[])}
    assert "claude-opus-4-8" in matches
    assert "gemini-3.1-pro-preview" in matches
    # An env var that does not name a model is not a model default.
    assert len([f for f in audit(py_roots=[src], md_roots=[]) if f["line"] == 6]) == 0


def test_uppercase_and_suffixed_constants_are_flagged(tmp_path: Path) -> None:
    """Module constants such as DEFAULT_MODEL / MODEL_ID / *_MODEL held model
    literals unseen by the lowercase-only name rule until 2026-09-24, including
    ones split across a `+` to dodge the lint."""

    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
DEFAULT_MODEL = "gpt-5.5"
MODEL_ID = "claude-opus-4-8"
SYNTHESIS_MODEL = ("claude-" + "sonnet-5")
JUDGE_MODEL_NAME: str = "gemini-3.1-pro-preview"
REVIEW_MODEL_ID = "gpt-5.6-sol"


def pick(provider):
    summary_model = "gemini-2.5" + "-pro"
    return summary_model
""",
    )
    lines = {f["line"] for f in audit(py_roots=[src], md_roots=[])}
    assert lines == {2, 3, 4, 5, 6, 10}


def test_non_model_suffixes_are_not_flagged(tmp_path: Path) -> None:
    """Names that merely contain "model" (not as a trailing _model/_model_id/
    _model_name) are not functional selection targets."""

    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
MODEL_NOTES = "claude-opus-4-8 was used for the ICLR run"
MODELS_COMPARED = "gpt-5.5"
SUPERMODEL = "gemini-3.1-pro-preview"
""",
    )
    assert audit(py_roots=[src], md_roots=[]) == []


def test_split_literal_is_flagged_in_any_position(tmp_path: Path) -> None:
    """A model id split across `+` exists only to hide it from a scanner, so it
    is flagged even where a plain literal would not be (dict value, positional
    constructor argument, collection element). Found live on 2026-09-24 in
    exhaustive extraction's primary route and the OCR benchmark candidates."""

    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
CONFIG = {"fast_llm": ("gpt-" + "5-mini")}
ROUTE = ("anthropic/" + "claude-" + "haiku-4-5-20251001")


def run(genai):
    return genai.GenerativeModel(("gemini-" + "3-pro-preview"))


CANDIDATES = [Candidate("openai", "gpt-" + "5.4-nano")]
PINNED = ("gpt-" + "5-mini")  # model-literal-ok: fixture pin
AUDIO = ("gemini-" + "2.5-pro-preview-tts")
""",
    )
    findings = audit(py_roots=[src], md_roots=[])
    assert {(f["line"], f["match"]) for f in findings} == {
        (2, "gpt-5-mini"),
        (3, "claude-haiku-4-5-20251001"),
        (7, "gemini-3-pro-preview"),
        (10, "gpt-5.4-nano"),
    }


def test_gemini_id_without_minor_version_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(src / "tool.py", 'DEFAULT_MODEL = "gemini-3-pro-preview"\n')
    assert [f["match"] for f in audit(py_roots=[src], md_roots=[])] == ["gemini-3-pro-preview"]


def test_suffixed_parameter_default_and_keyword_are_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def plan(topic, research_model: str = "gpt-5.6-luna"):
    return topic


def call(client):
    return client.run(judge_model_id="claude-opus-4-8")
""",
    )
    assert {f["line"] for f in audit(py_roots=[src], md_roots=[])} == {2, 7}


# v6.0 Phase 4 D2 (spec 6.6): the public site's two embedding models are governed, so
# their ids are audited like chat-model ids.
def test_embedding_model_constants_are_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
DEFAULT_EMBEDDING_MODEL = "mlx-community/bge-m3-mlx-8bit"
BROWSER_MODEL = "onnx-community/embeddinggemma-300m-ONNX"
""",
    )
    matches = {f["match"] for f in audit(py_roots=[src], md_roots=[])}
    assert "bge-m3-mlx-8bit" in matches
    assert "embeddinggemma-300m-ONNX" in matches


def test_embedding_model_markdown_flag_is_flagged(tmp_path: Path) -> None:
    md_root = tmp_path / "skills"
    _write(
        md_root / "s" / "SKILL.md",
        "`python scripts/x.py --model mlx-community/bge-m3-mlx-8bit`\n"
        "`python scripts/y.py --model onnx-community/embeddinggemma-300m-ONNX`\n",
    )
    assert len(audit(py_roots=[], md_roots=[md_root])) == 2


def test_suppressed_embedding_pin_is_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        'PIN_MODEL = "mlx-community/bge-m3-mlx-8bit"  # model-literal-ok: fixture pins the model id\n',
    )
    assert audit(py_roots=[src], md_roots=[]) == []


# D2 review r1, F1: defaults behind `or`, a conditional, setdefault or a config subscript.
def test_or_fallback_setdefault_and_subscript_defaults_are_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def build(model, config, fast):
    client = make(model=model or "mlx-community/bge-m3-mlx-8bit")
    config.setdefault("model", "onnx-community/embeddinggemma-300m-ONNX")
    config["model"] = "claude-opus-4-8"
    chosen_model = "gpt-5.2" if fast else model
    return client
""",
    )
    matches = {f["match"] for f in audit(py_roots=[src], md_roots=[])}
    assert matches == {"bge-m3-mlx-8bit", "embeddinggemma-300m-ONNX", "claude-opus-4-8", "gpt-5.2"}


def test_non_model_setdefault_and_subscript_are_not_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(
        src / "tool.py",
        """
def build(config):
    config.setdefault("note", "mlx-community/bge-m3-mlx-8bit")
    config["label"] = "gpt-5.2"
""",
    )
    assert audit(py_roots=[src], md_roots=[]) == []


# D2 review r1, F2: a quoted --model argument in a Markdown command example.
def test_quoted_markdown_model_argument_is_flagged(tmp_path: Path) -> None:
    md_root = tmp_path / "skills"
    _write(
        md_root / "s" / "SKILL.md",
        '`x --model "mlx-community/bge-m3-mlx-8bit"`\n'
        "`y --model='onnx-community/embeddinggemma-300m-ONNX'`\n"
        '`z --model "gpt-5.6-luna"`\n',
    )
    assert len(audit(py_roots=[], md_roots=[md_root])) == 3
