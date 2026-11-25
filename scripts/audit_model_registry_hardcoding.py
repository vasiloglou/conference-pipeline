#!/usr/bin/env python3
"""Fail-closed audit for hardcoded LLM model-name literals outside the registry.

`src/conference_pipeline/model_governance/registry.yaml` is the single source of
truth for which model a governed task uses. Code and skills must reference models
by task id through `conference_pipeline.model_governance.registry` accessors
(`default_model`, `default_model_for_provider`, `default_fallback_model`) — never
by hardcoding a model-name literal. This audit scans the reusable Python surface
(`src/conference_pipeline`, `scripts`) and the planning-skill Markdown surface
(`.claude/skills`) for LLM model-family literals appearing in a *functional*
position — an argparse `default=`, a `model=`/`model_id=`/`model_name=` keyword
or assigned default (any case, and any `*_model`/`*_model_id`/`*_model_name`
name), a `"--model", "<literal>"` CLI pair, or a `--model <literal>` Markdown
command example — and exits non-zero when any survive unsuppressed, so a
milestone touching model selection cannot close with unresolved drift.

A model name split across a `+` concatenation (``"claude-" + "sonnet-5"``) is
flagged wherever it sits: no piece matches on its own, so the split exists only
to hide the literal from a scanner.

Run:
    python scripts/audit_model_registry_hardcoding.py            # human report
    python scripts/audit_model_registry_hardcoding.py --json     # machine report

Suppressing an intentional literal: append ``# model-literal-ok: <reason>`` to the
line in a ``.py`` file, or include the ``model-literal-ok`` token anywhere on the
line in a ``.md`` file (with a reason). A concrete reason is required for review;
suppressions are meant to be enumerated by the caller, not hidden.

Fix pattern for a real finding: read the model from the registry instead of
hardcoding it —
``from conference_pipeline.model_governance.registry import default_model,
default_model_for_provider`` — and call ``default_model(task_id)`` or
``default_model_for_provider(task_id, provider)`` for the governed task.

Allowlisted (never scanned or always safe, and not a portability defect):
- Any path containing ``/tests/`` or ``test_`` (test fixtures legitimately pin
  models for reproducibility).
- ``registry.py`` itself (the loader legitimately names allowed models).
- The registry's own data files: ``registry.yaml``, ``pricing_catalog.json``,
  ``costs/task_specs.yaml`` (these are ``.yaml``/``.json`` and are not scanned by
  this ``.py``/``.md`` scanner; listed here for clarity).
- This file (it names model families in its own docstring/patterns).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

# Python source is scanned repo-wide across the reusable library AND scripts —
# this is a cross-branch model-selection governance gate, broader in scope than
# the conference-hardcoding audit's library-only scope.
DEFAULT_PY_SCAN_DIRS = ("src/conference_pipeline", "scripts")

# Planning-skill Markdown command examples are also governed.
DEFAULT_MD_SCAN_DIRS = (".claude/skills",)

SUPPRESS_MARKER = "model-literal-ok"

# LLM model families the registry governs. Deliberately excludes bare `gpt-4o`
# and `*-tts` audio/transcription models — out of scope for this gate.
#
# The `mini`/`nano` gpt-5 suffixes are combined into one `(?:mini|nano)`
# alternation rather than two separate literal patterns: this file lives under
# `scripts/` and is itself scanned by tests/test_model_governance.py's blunt,
# allowlist-free AST literal scan, which (unlike this audit) matches on any
# `gpt-5-mini`/`gpt-5-nano` *substring* regardless of position. Splitting the
# alternation avoids a spurious self-match without changing the trigger scope.
PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"gpt-5\.[0-9]"),
    re.compile(r"gpt-5-(?:mini|nano)"),
    re.compile(r"claude-(?:opus|sonnet|haiku)-[0-9]"),
    # The minor version is optional: `gemini-3-pro-preview` is a real model id.
    re.compile(r"gemini-[0-9](?:\.[0-9])?-(?:pro|flash)"),
    # The public site's two embedding models (v6.0 Phase 4, spec 6.6), governed by
    # public_search.canonical_embedding and public_search.browser_embedding.
    re.compile(r"bge-m3"),
    re.compile(r"embeddinggemma"),
)

# Functional keyword/attribute/variable names a governed model literal may sit
# under, outside the argparse `default=` case (which is scoped to add_argument
# calls specifically).
FUNCTIONAL_NAMES = ("model", "model_id", "model_name")

# Assignment targets are matched case-insensitively and by suffix, so a module
# constant such as `DEFAULT_MODEL`, `MODEL_ID` or `SYNTHESIS_MODEL` holding a
# model literal is governed too (these went unflagged until 2026-09-24).
FUNCTIONAL_NAME_SUFFIXES = ("_model", "_model_id", "_model_name")


def _is_functional_target_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in FUNCTIONAL_NAMES or lowered.endswith(FUNCTIONAL_NAME_SUFFIXES)


def _env_model_default(node: ast.Call) -> ast.expr | None:
    """The literal default of `os.environ.get("..MODEL..", "<literal>")` or
    `os.getenv(...)`: an env override that silently falls back to a hardcoded
    model (how scripts/presentations/generate_engaging_narration.py chose its
    model, unseen by this audit, until 2026-09-24)."""
    func = node.func
    is_env_get = (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
    ) or (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
        isinstance(func, ast.Name) and func.id == "getenv"
    )
    if not is_env_get or len(node.args) < 2:
        return None
    name = _string_value(node.args[0])
    if name is None or "MODEL" not in name.upper():
        return None
    return node.args[1] if _string_value(node.args[1]) is not None else None

# Same patterns, anchored to the token immediately following `--model[= ]` in a
# Markdown command example (never matches free-standing prose).
MD_MODEL_FLAG_RE = re.compile(
    r"--model[= ]+[\"']?("
    r"gpt-5\.[0-9][\w.-]*"
    r"|gpt-5-(?:mini|nano)[\w.-]*"
    r"|claude-(?:opus|sonnet|haiku)-[0-9][\w.-]*"
    r"|gemini-[0-9]\.[0-9]-(?:pro|flash)[\w.-]*"
    r"|[\w.-]*/?bge-m3[\w.-]*"
    r"|[\w.-]*/?embeddinggemma[\w.-]*"
    r")\b"
)


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Ids of Constant string nodes that are module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _string_value(node: ast.expr) -> str | None:
    """Resolve a string literal, including one split across a `+` concatenation
    (e.g. ``"gemini-" + "2.5-pro-preview-tts"``) so that pattern does not evade
    detection by keeping no single AST Constant node long enough to match."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left)
        right = _string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _match_literal(value: str | None) -> str | None:
    if value is None:
        return None
    for pattern in PATTERNS:
        m = pattern.search(value)
        if m:
            # The trigger patterns are deliberately narrow (e.g. a single
            # trailing digit) to stay high-signal; extend the reported match
            # forward through the rest of the hyphen/dot-delimited model
            # identifier (e.g. "claude-opus-4" -> "claude-opus-4-8") purely
            # for a more readable report. Does not change trigger semantics.
            end = m.end()
            while end < len(value) and (value[end].isalnum() or value[end] in ".-"):
                end += 1
            extended = value[m.start():end]
            # Audio/TTS and transcription model ids are out of scope for this
            # gate (see module docstring) even when they contain a governed
            # chat-model family prefix, e.g. "gemini-2.5-pro-preview-tts".
            if extended.endswith("-tts") or extended.endswith("-transcribe") or extended.endswith("-transcribe-diarize"):
                continue
            return extended
    return None


def _consider_sibling_sequence(
    elements: list[ast.expr], found: dict[int, tuple[ast.expr, str]]
) -> None:
    """Flag a string element directly following a `"--model"` literal."""
    for i in range(len(elements) - 1):
        current = elements[i]
        nxt = elements[i + 1]
        if (
            isinstance(current, ast.Constant)
            and current.value == "--model"
            and _string_value(nxt) is not None
            and id(nxt) not in found
        ):
            found[id(nxt)] = (nxt, "cli_model_flag_sibling")


def _literal_operands(node: ast.expr | None) -> list[ast.expr]:
    """The string literals a value can evaluate to: the value itself, or each literal
    operand of an ``x or "<literal>"`` fallback or a conditional expression. A default
    hidden behind ``or`` went unseen until 2026-11-16 (v6.0 Phase 4 D2 review, F1)."""
    if node is None:
        return []
    if _string_value(node) is not None:
        return [node]
    if isinstance(node, ast.BoolOp):
        return [lit for value in node.values for lit in _literal_operands(value)]
    if isinstance(node, ast.IfExp):
        return _literal_operands(node.body) + _literal_operands(node.orelse)
    return []


def _subscript_key(node: ast.expr) -> str | None:
    """The string key of ``config["model"]``-style subscripts."""
    if isinstance(node, ast.Subscript):
        return _string_value(node.slice)
    return None


def _consider_function_defaults(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    found: dict[int, tuple[ast.expr, str]],
) -> None:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    offset = len(positional) - len(defaults)
    for arg, default in zip(positional[offset:], defaults):
        if (
            _is_functional_target_name(arg.arg)
            and _string_value(default) is not None
            and id(default) not in found
        ):
            found[id(default)] = (default, "assigned_default")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if (
            default is not None
            and _is_functional_target_name(arg.arg)
            and _string_value(default) is not None
            and id(default) not in found
        ):
            found[id(default)] = (default, "assigned_default")


def _find_functional_constants(tree: ast.AST) -> list[ast.expr]:
    """Return string-literal expr nodes (plain Constant, or a `+`-concatenation
    of Constants) sitting in a functional model-literal position."""
    found: dict[int, tuple[ast.expr, str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_add_argument = (
                isinstance(func, ast.Attribute) and func.attr == "add_argument"
            ) or (isinstance(func, ast.Name) and func.id == "add_argument")
            for kw in node.keywords:
                if _string_value(kw.value) is not None and kw.arg == "default" and is_add_argument:
                    if id(kw.value) not in found:
                        found[id(kw.value)] = (kw.value, "argparse_default")
                elif kw.arg is not None and _is_functional_target_name(kw.arg):
                    for lit in _literal_operands(kw.value):
                        if id(lit) not in found:
                            found[id(lit)] = (lit, "model_kwarg")
            # config.setdefault("model", "<literal>") (D2 review, F1)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setdefault"
                and len(node.args) >= 2
                and _string_value(node.args[0]) is not None
                and _is_functional_target_name(_string_value(node.args[0]) or "")
            ):
                for lit in _literal_operands(node.args[1]):
                    if id(lit) not in found:
                        found[id(lit)] = (lit, "setdefault_model")
            _consider_sibling_sequence(list(node.args), found)
            env_default = _env_model_default(node)
            if env_default is not None and id(env_default) not in found:
                found[id(env_default)] = (env_default, "env_model_default")
        elif isinstance(node, (ast.List, ast.Tuple)):
            _consider_sibling_sequence(list(node.elts), found)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _consider_function_defaults(node, found)
        elif isinstance(node, ast.Assign):
            if any(
                (isinstance(t, ast.Name) and _is_functional_target_name(t.id))
                or _is_functional_target_name(_subscript_key(t) or "")
                for t in node.targets
            ):
                for lit in _literal_operands(node.value):
                    if id(lit) not in found:
                        found[id(lit)] = (lit, "assigned_default")
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_functional_target_name(node.target.id):
                for lit in _literal_operands(node.value):
                    if id(lit) not in found:
                        found[id(lit)] = (lit, "assigned_default")

    return [constant for constant, _category in found.values()]


def _find_split_literals(tree: ast.AST) -> list[ast.expr]:
    """Outermost `+` concatenations of string constants whose joined value is a
    governed model id while no single piece is: flagged in any position."""
    nested: set[int] = set()
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
            continue
        for child in (node.left, node.right):
            if isinstance(child, ast.BinOp):
                nested.add(id(child))
        if id(node) in nested:
            continue
        if _match_literal(_string_value(node)) is None:
            continue
        pieces = [
            piece.value
            for piece in ast.walk(node)
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
        ]
        if not any(_match_literal(piece) for piece in pieces):
            found.append(node)
    return found


def _allowlisted(path: Path) -> bool:
    # Path-segment/filename checks (not a raw substring search) so a pytest
    # tmp_path fixture such as `.../pytest-1/test_some_case0/src/tool.py` is
    # never mistaken for a real `tests/` directory or `test_*.py` module.
    if any(part == "tests" for part in path.parts):
        return True
    if path.name.startswith("test_"):
        return True
    if path.name in {"registry.py", "audit_model_registry_hardcoding.py"}:
        return True
    return False


def _iter_files(roots: Iterable[Path], suffix: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob(f"*{suffix}")))
        elif root.is_file() and root.suffix == suffix:
            files.append(root)
    return files


def _audit_python_files(roots: Iterable[Path]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in _iter_files(roots, ".py"):
        if _allowlisted(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        source_lines = source.splitlines()
        docstring_ids = _docstring_constant_ids(tree)
        candidates = _find_functional_constants(tree)
        seen = {id(node) for node in candidates}
        candidates += [node for node in _find_split_literals(tree) if id(node) not in seen]
        for constant in candidates:
            if id(constant) in docstring_ids:
                continue
            match = _match_literal(_string_value(constant))
            if not match:
                continue
            lineno = getattr(constant, "lineno", 0)
            line_text = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
            if SUPPRESS_MARKER in line_text:
                continue
            try:
                display_path = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                display_path = path.as_posix()
            findings.append(
                {
                    "file": display_path,
                    "line": lineno,
                    "match": match,
                    "text": line_text.strip()[:200],
                }
            )
    return findings


def _audit_markdown_files(roots: Iterable[Path]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in _iter_files(roots, ".md"):
        if _allowlisted(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(source.splitlines(), start=1):
            if SUPPRESS_MARKER in line:
                continue
            match = MD_MODEL_FLAG_RE.search(line)
            if not match:
                continue
            try:
                display_path = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                display_path = path.as_posix()
            findings.append(
                {
                    "file": display_path,
                    "line": lineno,
                    "match": match.group(1),
                    "text": line.strip()[:200],
                }
            )
    return findings


def audit(
    *,
    py_roots: Iterable[Path] | None = None,
    md_roots: Iterable[Path] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Run the scan. Accepts optional scan roots so tests can point at a fixture."""
    base = repo_root or REPO_ROOT
    if py_roots is None:
        py_roots = [base / rel for rel in DEFAULT_PY_SCAN_DIRS]
    if md_roots is None:
        md_roots = [base / rel for rel in DEFAULT_MD_SCAN_DIRS]
    findings = _audit_python_files(py_roots) + _audit_markdown_files(md_roots)
    findings.sort(key=lambda f: (str(f["file"]), int(f["line"])))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    findings = audit()
    if args.json:
        print(json.dumps({"status": "fail" if findings else "pass", "findings": findings}, indent=2))
    else:
        if not findings:
            print("PASS: no hardcoded model-registry literals found.")
        else:
            print(f"FAIL: {len(findings)} hardcoded model literal(s) outside the registry:\n")
            for f in findings:
                print(f"  {f['file']}:{f['line']}  [{f['match']}]")
                print(f"      {f['text']}")
            print(
                "\nFix: read the model from the registry via "
                "conference_pipeline.model_governance.registry.default_model(task_id) or "
                "default_model_for_provider(task_id, provider). If a literal is genuinely "
                f"intentional, append '# {SUPPRESS_MARKER}: <reason>' (.py) or include "
                f"'{SUPPRESS_MARKER}' with a reason on the line (.md)."
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
