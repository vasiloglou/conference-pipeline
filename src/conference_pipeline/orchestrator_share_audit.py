"""Report-only share audit for per-conference production orchestrators.

Each conference keeps its own production orchestrator (``scripts/run_*_production.py``)
because every conference has acquisition, auth and media quirks. The goal is
that the orchestrator stays *thin*: it imports shared stage owners from
``src/conference_pipeline/`` and implements only its quirks, behind declared
hooks. This audit measures how far each orchestrator is from that shape.

It is static: every orchestrator is parsed with :mod:`ast`, nothing is imported
or executed, and no workspace or provider is touched.

Per top-level unit (``def`` / ``class``, decorators included) it records:

* ``lines`` and the stage it serves, by reachability from the stage dispatch
  in ``main()`` (``stage.id == "<stage>"`` branches), from extra CLI command
  branches (``if args.<flag>:`` before the dispatch), from ``main`` / module
  level, or from a declared hook. A unit reached from more than one pipeline
  stage is ``shared-infra``; a unit reached from nothing is dead. Fallback
  branches of a stage dispatch chain (``else`` / non-stage ``elif``) count as
  dead when every stage in the ``STAGES`` table has an explicit branch.
  Orchestrators that delegate to ``conference_pipeline.production.orchestrator``
  route stages through a declarative dispatch table instead: a dict literal
  whose keys are all stage ids (``handlers={"<stage>": entry, ...}``) counts as
  that stage's branch, and an ``extra_commands={"<dest>": ...}`` dict counts as
  the extra CLI command branches. Neither table's entries count toward the
  enclosing function's own references or adapter size.
* ``classification``: ``hook`` (decorated ``@conference_hook(...)``),
  ``adapter`` (at most ``ADAPTER_MAX_LINES`` non-docstring lines that reference
  a shared component), ``dead``, or ``local`` (undeclared orchestrator logic).
* overlays: ``shadow_of`` (normalized similarity >= threshold to a top-level
  function under ``src/conference_pipeline``) and ``clone_of`` (the same to a
  unit in another audited orchestrator).

Shared components are modules under ``conference_pipeline.production`` plus
any ``src`` module annotated with a module-level ``RECOMMENDED_FOR`` constant
(the reusable acquisition lanes: they are written once and recommended for the
conferences whose quirk they cover). Other ``src`` modules whose file name
carries the orchestrator's own conference token are reported as
conference-local components.

Per stage it reports ``shared``, ``shared+hooks``, ``conference-local``
(delegates to a conference-local component), ``local`` or ``unrouted``.

Conference tokens used for normalization are derived from the orchestrator
file names (``run_<slug>_production.py``), never from literals here.

Run::

    python -m conference_pipeline.orchestrator_share_audit                # markdown
    python -m conference_pipeline.orchestrator_share_audit --format json
    python -m conference_pipeline.orchestrator_share_audit \
        --orchestrator scripts/run_<slug>_production.py --json-out report.json

Report-only: the command always exits 0. A ``--baseline``/``--ratchet`` mode is
deliberately not part of this package.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.registry_diagnostics import attach_agent_diagnostic

TOOL_ID = "orchestrator_share_audit"
SCHEMA_VERSION = 1
MODE_REPORT_ONLY = "report-only"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORCHESTRATOR_GLOB = "scripts/run_*_production.py"
DEFAULT_SRC_ROOT = "src/conference_pipeline"
DEFAULT_REGISTRY = "src/conference_pipeline/stage_owner_registry.json"
RUNNER_SCRIPT_ID_SUFFIX = ".production_runner"

SHARED_PACKAGE_PREFIXES = ("conference_pipeline.production",)
# The shared orchestrator's own CLI/dispatch/hook infrastructure. A unit that
# calls only these is still an adapter, but it routes a stage rather than
# sharing it: stage status looks through such dispatch adapters to the entry
# function they call.
ORCHESTRATION_INFRA_MODULES = frozenset(
    {
        "conference_pipeline.production.orchestrator",
        "conference_pipeline.production.stage_table",
        "conference_pipeline.production.hooks",
    }
)
ANNOTATION_NAME = "RECOMMENDED_FOR"
HOOK_DECORATOR_NAME = "conference_hook"
STAGE_TABLE_NAME = "STAGES"
STAGE_TABLE_NEXT_ACTIONS_KEYWORD = "next_actions"
EXTRA_COMMANDS_KEYWORD = "extra_commands"
MAIN_NAME = "main"

ADAPTER_MAX_LINES = 40
SIMILARITY_THRESHOLD = 0.80
MIN_COMPARABLE_LINES = 4
COMMON_LINE_DF_CAP = 60
MAX_CANDIDATES = 30
CONFERENCE_PLACEHOLDER = "CONF"

RUNNER_SLUG_RE = re.compile(r"^run_(?P<slug>.+)_production$")
SLUG_PART_RE = re.compile(r"([a-z]+)(\d{2,4})?")
STAGE_ID_ATTRS = frozenset({"id", "stage", "stage_id"})

CLASS_HOOK = "hook"
CLASS_ADAPTER = "adapter"
CLASS_DEAD = "dead"
CLASS_LOCAL = "local"

GROUP_SHARED_INFRA = "shared-infra"
GROUP_CLI_EXTRA = "cli-extra"
GROUP_RUNNER_INFRA = "runner-infra"
GROUP_HOOK_ONLY = "hook-only"
GROUP_UNREACHABLE = "unreachable"

STATUS_SHARED = "shared"
STATUS_SHARED_HOOKS = "shared+hooks"
STATUS_CONFERENCE_LOCAL = "conference-local"
STATUS_LOCAL = "local"
STATUS_UNROUTED = "unrouted"
SHARED_STATUSES = frozenset({STATUS_SHARED, STATUS_SHARED_HOOKS})


# --------------------------------------------------------------------------
# Conference-token normalization
# --------------------------------------------------------------------------


def runner_slug(path: Path) -> str | None:
    match = RUNNER_SLUG_RE.match(Path(path).stem)
    return match.group("slug") if match else None


def slug_tokens(slug: str | None) -> tuple[set[str], set[str]]:
    """``(alpha_tokens, two_digit_years)`` found in an orchestrator slug."""

    alphas: set[str] = set()
    years: set[str] = set()
    for alpha, year in SLUG_PART_RE.findall(str(slug or "").lower()):
        if len(alpha) >= 3:
            alphas.add(alpha)
        if year:
            years.add(year[-2:])
    return alphas, years


def conference_token_regex(slugs: Iterable[str | None]) -> re.Pattern[str] | None:
    alphas: set[str] = set()
    years: set[str] = set()
    for slug in slugs:
        slug_alphas, slug_years = slug_tokens(slug)
        alphas |= slug_alphas
        years |= slug_years
    if not alphas:
        return None
    alpha_alt = "|".join(re.escape(item) for item in sorted(alphas, key=lambda value: (-len(value), value)))
    if years:
        year_alt = "|".join(sorted(years))
        pattern = rf"(?:{alpha_alt})(?:[ _-]?(?:20)?(?:{year_alt}))?"
    else:
        pattern = rf"(?:{alpha_alt})"
    return re.compile(pattern, re.IGNORECASE)


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            child.body = body[1:] or [ast.Pass()]
    return node


def normalized_lines(node: ast.AST, token_re: re.Pattern[str] | None) -> tuple[str, ...]:
    """Unit source without docstrings, conference tokens replaced, lines stripped."""

    try:
        text = ast.unparse(_strip_docstrings(copy.deepcopy(node)))
    except Exception:  # pragma: no cover - unparse failures are exotic
        return ()
    if token_re is not None:
        text = token_re.sub(CONFERENCE_PLACEHOLDER, text)
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _body_lines(node: ast.AST, skip: Sequence[ast.AST] = ()) -> int:
    """Lines from the first non-docstring body statement to the end.

    Signature and docstring lines are excluded, so a wrapper with a long
    keyword signature and a one-call body still measures as thin. Lines of
    ``skip`` nodes (declarative dispatch tables) are excluded as well.
    """

    body = list(getattr(node, "body", None) or ())
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return 0
    end = int(getattr(node, "end_lineno", None) or body[-1].lineno)
    skipped = sum(int(item.end_lineno or item.lineno) - int(item.lineno) + 1 for item in skip)
    return max(1, end - int(body[0].lineno) + 1 - skipped)


# --------------------------------------------------------------------------
# Similarity index (exact hash first, then line-bag prefilter, then difflib)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparable:
    key: str
    owner: str
    lines: tuple[str, ...]
    digest: str
    span: int
    counter: Counter = field(compare=False, hash=False)


def make_comparable(key: str, owner: str, lines: Sequence[str], *, span: int) -> Comparable:
    """``span`` is the unit's source line count; units shorter than
    ``MIN_COMPARABLE_LINES`` source lines are never compared (too generic)."""

    lines = tuple(lines)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return Comparable(key=key, owner=owner, lines=lines, digest=digest, span=span, counter=Counter(lines))


def _span(node: ast.AST) -> int:
    start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", ())])
    return int((node.end_lineno or node.lineno) - start + 1)


class SimilarityIndex:
    """Near-duplicate lookup over normalized units.

    Exact matches come from a digest map. Near matches are drawn from an
    inverted line index (lines shared by more than ``df_cap`` units are too
    common to discriminate), bounded by the multiset upper bound of
    ``difflib.SequenceMatcher.ratio`` and then scored by ``ratio`` itself.
    """

    def __init__(self, *, threshold: float = SIMILARITY_THRESHOLD, df_cap: int = COMMON_LINE_DF_CAP) -> None:
        self.threshold = threshold
        self.df_cap = df_cap
        self.items: dict[str, Comparable] = {}
        self.by_digest: dict[str, list[str]] = defaultdict(list)
        self.by_line: dict[str, list[str]] = defaultdict(list)

    def add(self, item: Comparable) -> None:
        if item.span < MIN_COMPARABLE_LINES or not item.lines:
            return
        self.items[item.key] = item
        self.by_digest[item.digest].append(item.key)
        for line in set(item.lines):
            self.by_line[line].append(item.key)

    def best_match(self, item: Comparable, *, exclude_owner: str | None = None) -> tuple[str, float] | None:
        if item.span < MIN_COMPARABLE_LINES or not item.lines:
            return None
        for key in self.by_digest.get(item.digest, ()):
            other = self.items[key]
            if key != item.key and other.owner != exclude_owner:
                return key, 1.0
        shared: Counter = Counter()
        for line in set(item.lines):
            keys = self.by_line.get(line, ())
            if len(keys) > self.df_cap:
                continue
            for key in keys:
                shared[key] += 1
        best: tuple[str, float] | None = None
        size = len(item.lines)
        checked = 0
        for key, _count in shared.most_common():
            if checked >= MAX_CANDIDATES:
                break
            other = self.items[key]
            if key == item.key or other.owner == exclude_owner:
                continue
            other_size = len(other.lines)
            if 2 * min(size, other_size) / (size + other_size) < self.threshold:
                continue
            checked += 1
            overlap = sum((item.counter & other.counter).values())
            if 2 * overlap / (size + other_size) < self.threshold:
                continue
            ratio = difflib.SequenceMatcher(None, item.lines, other.lines, autojunk=False).ratio()
            if ratio >= self.threshold and (best is None or ratio > best[1]):
                best = (key, ratio)
        return best


# --------------------------------------------------------------------------
# Source components (shared vs conference-local)
# --------------------------------------------------------------------------


class ComponentResolver:
    """Resolves dotted import targets to ``src`` modules and classifies them."""

    def __init__(self, src_root: Path, shared_prefixes: Sequence[str] = SHARED_PACKAGE_PREFIXES) -> None:
        self.src_root = Path(src_root)
        self.package = self.src_root.name
        self.shared_prefixes = tuple(shared_prefixes)
        self._annotation_cache: dict[str, tuple[bool, Any]] = {}

    def module_file(self, module: str) -> Path | None:
        parts = module.split(".")
        if not parts or parts[0] != self.package:
            return None
        base = self.src_root.joinpath(*parts[1:])
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
        return None

    def resolve(self, dotted: str) -> str | None:
        parts = dotted.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if self.module_file(candidate) is not None:
                return candidate
        return None

    def annotation(self, module: str) -> tuple[bool, Any]:
        if module in self._annotation_cache:
            return self._annotation_cache[module]
        result: tuple[bool, Any] = (False, None)
        path = self.module_file(module)
        if path is not None:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                tree = None
            for stmt in getattr(tree, "body", ()):
                targets: list[ast.expr] = []
                value: ast.expr | None = None
                if isinstance(stmt, ast.Assign):
                    targets, value = list(stmt.targets), stmt.value
                elif isinstance(stmt, ast.AnnAssign):
                    targets, value = [stmt.target], stmt.value
                if any(isinstance(target, ast.Name) and target.id == ANNOTATION_NAME for target in targets):
                    try:
                        literal = ast.literal_eval(value) if value is not None else None
                    except ValueError:
                        literal = None
                    if isinstance(literal, (tuple, list, set)):
                        literal = sorted(str(item) for item in literal)
                    result = (True, literal)
                    break
        self._annotation_cache[module] = result
        return result

    def is_shared(self, module: str) -> bool:
        if any(module == prefix or module.startswith(prefix + ".") for prefix in self.shared_prefixes):
            return True
        return self.annotation(module)[0]

    def is_conference_local(self, module: str, alpha_tokens: Iterable[str]) -> bool:
        if self.is_shared(module):
            return False
        leaf = module.rsplit(".", 1)[-1].lower()
        return any(token in leaf for token in alpha_tokens)


# --------------------------------------------------------------------------
# Orchestrator parsing
# --------------------------------------------------------------------------


@dataclass
class Unit:
    name: str
    kind: str
    start: int
    end: int
    node: ast.AST
    names: set[str]
    dotted: set[str]
    local_imports: dict[str, str]
    hook_id: str | None
    body_lines: int
    reach: set[str] = field(default_factory=set)
    group: str = GROUP_UNREACHABLE
    classification: str = CLASS_LOCAL
    shared_components: list[str] = field(default_factory=list)
    local_components: list[str] = field(default_factory=list)
    shadow_of: dict[str, Any] | None = None
    clone_of: dict[str, Any] | None = None

    @property
    def lines(self) -> int:
        return self.end - self.start + 1


def _import_bindings(stmts: Iterable[ast.stmt]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    bindings[root] = root
        elif isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bindings[alias.asname or alias.name] = f"{stmt.module}.{alias.name}"
    return bindings


def _dotted_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _collect_names(node: ast.AST, skip: frozenset[int] = frozenset()) -> tuple[set[str], set[str]]:
    """Names and dotted chains under ``node``, ignoring subtrees whose ``id()`` is in ``skip``."""

    names: set[str] = set()
    dotted: set[str] = set()
    stack = [node]
    while stack:
        child = stack.pop()
        if id(child) in skip:
            continue
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            chain = _dotted_chain(child)
            if chain:
                dotted.add(chain)
        stack.extend(ast.iter_child_nodes(child))
    return names, dotted


@dataclass
class DispatchTables:
    """Declarative stage dispatch (``{stage_id: entry}``) and extra-command tables."""

    stage_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    extra_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    covered: list[str] = field(default_factory=list)
    nodes: list[ast.AST] = field(default_factory=list)

    @property
    def skip_ids(self) -> frozenset[int]:
        return frozenset(id(node) for node in self.nodes)


def _string_keys(node: ast.Dict) -> list[str] | None:
    keys: list[str] = []
    for key in node.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return None
        keys.append(key.value)
    return keys or None


def find_dispatch_tables(tree: ast.Module, stage_ids: Iterable[str]) -> DispatchTables:
    """Stage dispatch dicts (every key a stage id) and ``extra_commands=`` dicts."""

    known = set(stage_ids)
    tables = DispatchTables()
    extra_nodes: set[int] = set()
    for child in ast.walk(tree):
        if isinstance(child, ast.keyword) and child.arg == EXTRA_COMMANDS_KEYWORD and isinstance(child.value, ast.Dict):
            keys = _string_keys(child.value)
            if keys is None:
                continue
            extra_nodes.add(id(child.value))
            tables.nodes.append(child.value)
            for key, value in zip(keys, child.value.values):
                tables.extra_refs[key].update(_collect_names(value)[0])
    if not known:
        return tables
    for child in ast.walk(tree):
        if not isinstance(child, ast.Dict) or id(child) in extra_nodes:
            continue
        keys = _string_keys(child)
        if keys is None or not set(keys) <= known:
            continue
        # Only references to entry functions route a stage; a stage-keyed dict
        # of texts (next actions) or of rows (the stage table) does not.
        if not all(isinstance(value, (ast.Name, ast.Attribute, ast.Lambda)) for value in child.values):
            continue
        tables.nodes.append(child)
        for key, value in zip(keys, child.values):
            if key not in tables.covered:
                tables.covered.append(key)
            tables.stage_refs[key].update(_collect_names(value)[0])
    return tables


def _hook_id(node: ast.AST) -> str | None:
    for decorator in getattr(node, "decorator_list", ()):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name != HOOK_DECORATOR_NAME:
            continue
        if isinstance(decorator, ast.Call) and decorator.args:
            first = decorator.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value
        return getattr(node, "name", "")
    return None


def _stage_ids_in_test(test: ast.AST) -> set[str]:
    ids: set[str] = set()
    for child in ast.walk(test):
        if not isinstance(child, ast.Compare) or len(child.ops) != 1:
            continue
        if not isinstance(child.ops[0], (ast.Eq, ast.In)):
            continue
        left, right = child.left, child.comparators[0]
        for subject, other in ((left, right), (right, left)):
            is_stage_subject = (isinstance(subject, ast.Attribute) and subject.attr in STAGE_ID_ATTRS) or (
                isinstance(subject, ast.Name) and subject.id in STAGE_ID_ATTRS
            )
            if not is_stage_subject:
                continue
            if isinstance(other, ast.Constant) and isinstance(other.value, str):
                ids.add(other.value)
            elif isinstance(other, (ast.Tuple, ast.List, ast.Set)):
                ids.update(
                    item.value
                    for item in other.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
    return ids


def _extra_command_flag(test: ast.AST) -> str | None:
    """``args.<flag>`` (optionally combined with ``and``/``or``) -> ``flag``.

    Only branches that return early from ``main`` count as extra commands
    (``--prepare-*`` / ``--authorize-*``); output flags such as ``--summary``
    stay ``main`` references.
    """

    node = test
    if isinstance(node, ast.BoolOp) and node.values:
        node = node.values[0]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    return None


@dataclass
class DispatchScan:
    stage_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    extra_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    main_refs: set[str] = field(default_factory=set)
    fallback_refs: set[str] = field(default_factory=set)
    covered: list[str] = field(default_factory=list)
    has_chain: bool = False


def scan_main(main: ast.AST, skip: frozenset[int] = frozenset()) -> DispatchScan:
    scan = DispatchScan()

    def add(target: set[str], node: ast.AST) -> None:
        names, _dotted = _collect_names(node, skip)
        target.update(names)

    def refs_for(stage: str | None, extra: str | None) -> set[str]:
        if stage is not None:
            return scan.stage_refs[stage]
        if extra is not None:
            return scan.extra_refs[extra]
        return scan.main_refs

    def walk(stmts: Sequence[ast.stmt], stage: str | None, extra: str | None) -> None:
        for stmt in stmts:
            if not isinstance(stmt, ast.If):
                add(refs_for(stage, extra), stmt)
                continue
            if stage is None and _stage_ids_in_test(stmt.test):
                walk_chain(stmt, extra)
                continue
            flag = None
            if stage is None and extra is None and any(isinstance(item, ast.Return) for item in stmt.body):
                flag = _extra_command_flag(stmt.test)
            add(refs_for(stage, extra), stmt.test)
            if flag is not None:
                walk(stmt.body, None, flag)
            else:
                walk(stmt.body, stage, extra)
            walk(stmt.orelse, stage, extra)

    def walk_chain(node: ast.If, extra: str | None) -> None:
        scan.has_chain = True
        current: ast.If | None = node
        while current is not None:
            ids = _stage_ids_in_test(current.test)
            if ids:
                for stage_id in sorted(ids):
                    if stage_id not in scan.covered:
                        scan.covered.append(stage_id)
                    walk(current.body, stage_id, None)
            else:
                add(scan.fallback_refs, current.test)
                for stmt in current.body:
                    add(scan.fallback_refs, stmt)
            orelse = current.orelse
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                current = orelse[0]
                continue
            for stmt in orelse:
                add(scan.fallback_refs, stmt)
            current = None

    walk(list(getattr(main, "body", ())), None, None)
    return scan


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assigned: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assigned[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            assigned[stmt.target.id] = stmt.value
    return assigned


def _registry_stage_table(value: ast.Call, assigned: Mapping[str, ast.expr]) -> list[tuple[str, int | None]]:
    """``STAGES = stage_table.from_registry(next_actions=<dict>)``: ids from the dict keys.

    Orders come from the registry's stage bindings (filled in by the caller).
    """

    for keyword in value.keywords:
        if keyword.arg != STAGE_TABLE_NEXT_ACTIONS_KEYWORD:
            continue
        table = keyword.value
        if isinstance(table, ast.Name):
            table = assigned.get(table.id)
        if isinstance(table, ast.Dict):
            return [(key, None) for key in (_string_keys(table) or [])]
    return []


def _stage_table(tree: ast.Module) -> list[tuple[str, int | None]]:
    assigned = _module_assignments(tree)
    for stmt in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and target.id == STAGE_TABLE_NAME and isinstance(value, ast.Call):
            rows = _registry_stage_table(value, assigned)
            if rows:
                return rows
            continue
        if not (isinstance(target, ast.Name) and target.id == STAGE_TABLE_NAME and isinstance(value, ast.Dict)):
            continue
        rows: list[tuple[str, int | None]] = []
        for key, item in zip(value.keys, value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            order: int | None = None
            if isinstance(item, ast.Dict):
                for item_key, item_value in zip(item.keys, item.values):
                    if (
                        isinstance(item_key, ast.Constant)
                        and item_key.value == "order"
                        and isinstance(item_value, ast.Constant)
                        and isinstance(item_value.value, int)
                    ):
                        order = item_value.value
            if isinstance(item, ast.Call):
                for keyword in item.keywords:
                    if (
                        keyword.arg == "order"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, int)
                    ):
                        order = keyword.value.value
            rows.append((key.value, order))
        return rows
    return []


# --------------------------------------------------------------------------
# Per-orchestrator analysis
# --------------------------------------------------------------------------


@dataclass
class OrchestratorAnalysis:
    path: Path
    display_path: str
    slug: str | None
    source_lines: int
    tree: ast.Module
    units: dict[str, Unit]
    stage_rows: list[tuple[str, int | None]]
    scan: DispatchScan
    module_refs: set[str]
    imports: dict[str, str]
    parse_error: str | None = None


def _display(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_orchestrator(path: Path, repo_root: Path) -> OrchestratorAnalysis:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = _import_bindings(stmt for stmt in tree.body if isinstance(stmt, (ast.Import, ast.ImportFrom)))
    stage_rows = _stage_table(tree)
    tables = find_dispatch_tables(tree, [stage_id for stage_id, _order in stage_rows])
    skip = tables.skip_ids
    units: dict[str, Unit] = {}
    module_refs: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([stmt.lineno] + [decorator.lineno for decorator in stmt.decorator_list])
            # Decorators are excluded: the hook decorator's own module is
            # not a shared component the unit adapts to.
            undecorated = copy.copy(stmt)
            undecorated.decorator_list = []
            names, dotted = _collect_names(undecorated, skip)
            own_tables = [node for node in tables.nodes if stmt.lineno <= node.lineno <= int(stmt.end_lineno or 0)]
            local_imports = _import_bindings(
                child for child in ast.walk(stmt) if isinstance(child, (ast.Import, ast.ImportFrom))
            )
            units[stmt.name] = Unit(
                name=stmt.name,
                kind="class" if isinstance(stmt, ast.ClassDef) else "function",
                start=start,
                end=int(stmt.end_lineno or stmt.lineno),
                node=stmt,
                names=names,
                dotted=dotted,
                local_imports=local_imports,
                hook_id=_hook_id(stmt),
                body_lines=_body_lines(stmt, own_tables),
            )
        elif not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names, _dotted = _collect_names(stmt, skip)
            module_refs.update(names)
    main_unit = units.get(MAIN_NAME)
    scan = scan_main(main_unit.node, skip) if main_unit else DispatchScan()
    if tables.covered or tables.extra_refs:
        for stage_id in tables.covered:
            if stage_id not in scan.covered:
                scan.covered.append(stage_id)
            scan.stage_refs[stage_id].update(tables.stage_refs[stage_id])
        for flag, refs in tables.extra_refs.items():
            scan.extra_refs[flag].update(refs)
        scan.has_chain = scan.has_chain or bool(tables.covered)
    return OrchestratorAnalysis(
        path=path,
        display_path=_display(path, repo_root),
        slug=runner_slug(path),
        source_lines=len(source.splitlines()),
        tree=tree,
        units=units,
        stage_rows=stage_rows,
        scan=scan,
        module_refs=module_refs,
        imports=imports,
    )


def _closure(units: Mapping[str, Unit], seeds: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    frontier = [name for name in seeds if name in units and name != MAIN_NAME]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        for ref in units[name].names:
            if ref in units and ref not in seen and ref != MAIN_NAME:
                frontier.append(ref)
    return seen


def _unit_components(
    unit: Unit, imports: Mapping[str, str], resolver: ComponentResolver, alpha_tokens: set[str]
) -> tuple[list[str], list[str]]:
    bindings = dict(imports)
    bindings.update(unit.local_imports)
    targets: set[str] = set()
    for name in unit.names:
        if name in bindings:
            targets.add(bindings[name])
    for chain in unit.dotted:
        root, _, rest = chain.partition(".")
        if root in bindings:
            targets.add(bindings[root] + ("." + rest if rest else ""))
    shared: set[str] = set()
    local: set[str] = set()
    for target in targets:
        module = resolver.resolve(target)
        if module is None:
            continue
        if resolver.is_shared(module):
            shared.add(module)
        elif resolver.is_conference_local(module, alpha_tokens):
            local.add(module)
    return sorted(shared), sorted(local)


def _names_components(
    names: Iterable[str], imports: Mapping[str, str], resolver: ComponentResolver, alpha_tokens: set[str]
) -> tuple[set[str], set[str]]:
    shared: set[str] = set()
    local: set[str] = set()
    for name in names:
        target = imports.get(name)
        module = resolver.resolve(target) if target else None
        if module is None:
            continue
        if resolver.is_shared(module):
            shared.add(module)
        elif resolver.is_conference_local(module, alpha_tokens):
            local.add(module)
    return shared, local


def _is_dispatch_adapter(unit: Unit) -> bool:
    return (
        unit.classification == CLASS_ADAPTER
        and unit.hook_id is None
        and bool(unit.shared_components)
        and set(unit.shared_components) <= ORCHESTRATION_INFRA_MODULES
    )


def _effective_entry_units(units: Mapping[str, Unit], branch_refs: Iterable[str]) -> list[str]:
    """Stage entry units, looking through dispatch adapters to what they call.

    Units met only through a dispatch adapter that several stages reach
    (``shared-infra`` helpers such as run-directory resolution) are not entries.
    """

    entries: set[str] = set()
    seen: set[str] = set()
    frontier = [(name, True) for name in branch_refs]
    while frontier:
        name, direct = frontier.pop()
        if name in seen or name not in units or name == MAIN_NAME:
            continue
        seen.add(name)
        unit = units[name]
        if _is_dispatch_adapter(unit):
            frontier.extend((ref, False) for ref in unit.names if ref in units and ref != name)
            continue
        if not direct and unit.group == GROUP_SHARED_INFRA:
            continue
        entries.add(name)
    return sorted(entries)


def classify_orchestrator(
    analysis: OrchestratorAnalysis,
    resolver: ComponentResolver,
    *,
    adapter_max_lines: int = ADAPTER_MAX_LINES,
) -> dict[str, Any]:
    units = analysis.units
    scan = analysis.scan
    alpha_tokens, _years = slug_tokens(analysis.slug)
    stage_ids = [stage_id for stage_id, _order in analysis.stage_rows]
    for stage_id in scan.covered:
        if stage_id not in stage_ids:
            stage_ids.append(stage_id)
    orders = dict(analysis.stage_rows)
    fallback_dead = bool(scan.has_chain and stage_ids and set(stage_ids) <= set(scan.covered))

    reach: dict[str, set[str]] = defaultdict(set)
    for stage_id, refs in scan.stage_refs.items():
        for name in _closure(units, refs):
            reach[name].add(f"stage:{stage_id}")
    for flag, refs in scan.extra_refs.items():
        for name in _closure(units, refs):
            reach[name].add(f"extra:{flag}")
    main_seeds = set(scan.main_refs)
    if not fallback_dead:
        main_seeds |= scan.fallback_refs
    for name in _closure(units, main_seeds):
        reach[name].add("main")
    module_seeds = set(analysis.module_refs)
    for name in _closure(units, module_seeds):
        reach[name].add("module")
    if MAIN_NAME in units and (MAIN_NAME in module_seeds or scan.has_chain or scan.main_refs):
        reach[MAIN_NAME].add("main")
    for unit in units.values():
        if unit.hook_id is not None:
            for name in _closure(units, [unit.name]):
                reach[name].add(f"hook:{unit.hook_id}")

    stage_set = set(stage_ids)
    for unit in units.values():
        unit.reach = reach.get(unit.name, set())
        stages = sorted(item.split(":", 1)[1] for item in unit.reach if item.startswith("stage:"))
        hook_stages = sorted(
            {
                item.split(":", 1)[1].split(".", 1)[0]
                for item in unit.reach
                if item.startswith("hook:") and item.split(":", 1)[1].split(".", 1)[0] in stage_set
            }
        )
        if len(stages) > 1:
            unit.group = GROUP_SHARED_INFRA
        elif len(stages) == 1:
            unit.group = stages[0]
        elif len(hook_stages) == 1:
            unit.group = hook_stages[0]
        elif any(item.startswith("extra:") for item in unit.reach):
            unit.group = GROUP_CLI_EXTRA
        elif unit.reach & {"main", "module"}:
            unit.group = GROUP_RUNNER_INFRA
        elif any(item.startswith("hook:") for item in unit.reach):
            unit.group = GROUP_HOOK_ONLY
        else:
            unit.group = GROUP_UNREACHABLE
        unit.shared_components, unit.local_components = _unit_components(
            unit, analysis.imports, resolver, alpha_tokens
        )
        if not unit.reach:
            unit.classification = CLASS_DEAD
        elif unit.hook_id is not None:
            unit.classification = CLASS_HOOK
        elif unit.shared_components and unit.body_lines <= adapter_max_lines:
            unit.classification = CLASS_ADAPTER
        else:
            unit.classification = CLASS_LOCAL

    main_unit = units.get(MAIN_NAME)
    delegated_components: list[str] = []
    if main_unit is not None and not scan.has_chain and main_unit.classification == CLASS_ADAPTER:
        delegated_components = list(main_unit.shared_components)

    stages_report: list[dict[str, Any]] = []
    for stage_id in stage_ids:
        members = [unit for unit in units.values() if unit.group == stage_id]
        local_lines = sum(unit.lines for unit in members if unit.classification == CLASS_LOCAL)
        adapter_lines = sum(unit.lines for unit in members if unit.classification == CLASS_ADAPTER)
        hook_units = [unit for unit in members if unit.classification == CLASS_HOOK]
        branch_refs = scan.stage_refs.get(stage_id, set())
        direct_shared, direct_local = _names_components(branch_refs, analysis.imports, resolver, alpha_tokens)
        shared_components = set(direct_shared)
        local_components = set(direct_local)
        for unit in members:
            shared_components.update(unit.shared_components)
            local_components.update(unit.local_components)
        shared_components -= ORCHESTRATION_INFRA_MODULES
        routed = stage_id in scan.covered
        entry_units = _effective_entry_units(units, branch_refs)
        local_entries = [
            name for name in entry_units if units[name].classification in (CLASS_LOCAL, CLASS_DEAD)
        ]
        if not routed and delegated_components:
            shared_components.update(delegated_components)
            status = STATUS_SHARED
        elif not routed:
            status = STATUS_UNROUTED
        elif local_lines > 0 or local_entries:
            status = STATUS_LOCAL
        elif shared_components:
            status = STATUS_SHARED_HOOKS if hook_units else STATUS_SHARED
        elif local_components:
            status = STATUS_CONFERENCE_LOCAL
        else:
            status = STATUS_LOCAL
        stages_report.append(
            {
                "stage_id": stage_id,
                "order": orders.get(stage_id),
                "status": status,
                "routed": routed,
                "shared_components": sorted(shared_components),
                "conference_local_components": sorted(local_components),
                "hooks": sorted(str(unit.hook_id) for unit in hook_units),
                "entry_units": entry_units,
                "local_entry_units": local_entries,
                "local_lines": local_lines,
                "adapter_lines": adapter_lines,
                "hook_lines": sum(unit.lines for unit in hook_units),
                "unit_count": len(members),
            }
        )

    return {
        "stage_ids": stage_ids,
        "stages": stages_report,
        "extra_commands": sorted(scan.extra_refs),
        "fallback_branches_dead": fallback_dead,
        "delegated_to_shared_orchestrator": delegated_components,
    }


def _unit_record(unit: Unit) -> dict[str, Any]:
    return {
        "name": unit.name,
        "kind": unit.kind,
        "start_line": unit.start,
        "end_line": unit.end,
        "lines": unit.lines,
        "group": unit.group,
        "reachable_from": sorted(unit.reach),
        "classification": unit.classification,
        "hook_id": unit.hook_id,
        "shared_components": unit.shared_components,
        "conference_local_components": unit.local_components,
        "shadow_of": unit.shadow_of,
        "clone_of": unit.clone_of,
    }


def _metrics(analysis: OrchestratorAnalysis, stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = list(analysis.units.values())
    unit_lines = sum(unit.lines for unit in units)

    def lines_where(predicate) -> int:
        return sum(unit.lines for unit in units if predicate(unit))

    local_lines = lines_where(lambda unit: unit.classification == CLASS_LOCAL)
    shared_stage_count = sum(1 for stage in stages if stage["status"] in SHARED_STATUSES)
    clone_by_peer: Counter = Counter()
    for unit in units:
        if unit.clone_of:
            clone_by_peer[str(unit.clone_of["orchestrator"])] += unit.lines
    total = analysis.source_lines
    return {
        "total_lines": total,
        "module_level_lines": max(total - unit_lines, 0),
        "unit_count": len(units),
        "local_lines": local_lines,
        "declared_hook_lines": lines_where(lambda unit: unit.classification == CLASS_HOOK),
        "adapter_lines": lines_where(lambda unit: unit.classification == CLASS_ADAPTER),
        "dead_lines": lines_where(lambda unit: unit.classification == CLASS_DEAD),
        "shadow_lines": lines_where(lambda unit: unit.shadow_of is not None),
        "clone_lines": lines_where(lambda unit: unit.clone_of is not None),
        "clone_lines_by_orchestrator": dict(sorted(clone_by_peer.items())),
        "shared_stage_count": shared_stage_count,
        "stage_count": len(stages),
        "shared_stage_ratio": round(shared_stage_count / len(stages), 4) if stages else 0.0,
        "local_share": round(local_lines / total, 4) if total else 0.0,
    }


def _hooks_report(analysis: OrchestratorAnalysis) -> list[dict[str, Any]]:
    return [
        {"hook_id": unit.hook_id, "function": unit.name, "lines": unit.lines, "group": unit.group}
        for unit in sorted(analysis.units.values(), key=lambda item: item.start)
        if unit.hook_id is not None
    ]


# --------------------------------------------------------------------------
# Discovery and the whole audit
# --------------------------------------------------------------------------


def discover_orchestrators(repo_root: Path, registry_path: Path | None = None) -> list[Path]:
    """``scripts/run_*_production.py`` plus registry ``*.production_runner`` paths."""

    found = {path.resolve() for path in repo_root.glob(DEFAULT_ORCHESTRATOR_GLOB) if path.is_file()}
    registry_path = registry_path if registry_path is not None else repo_root / DEFAULT_REGISTRY
    try:
        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registry = {}
    scripts = registry.get("scripts") if isinstance(registry, Mapping) else None
    if isinstance(scripts, Mapping):
        for script_id, entry in scripts.items():
            if not str(script_id).endswith(RUNNER_SCRIPT_ID_SUFFIX) or not isinstance(entry, Mapping):
                continue
            raw = entry.get("path")
            if isinstance(raw, str) and raw:
                candidate = (repo_root / raw).resolve()
                if candidate.is_file():
                    found.add(candidate)
    return sorted(found)


def registry_stage_orders(repo_root: Path, registry_path: Path | None = None) -> dict[str, int]:
    """``{stage_id: stage_order}`` from the registry's stage bindings (empty when unreadable)."""

    registry_path = registry_path if registry_path is not None else repo_root / DEFAULT_REGISTRY
    try:
        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    bindings = registry.get("stage_bindings") if isinstance(registry, Mapping) else None
    orders: dict[str, int] = {}
    if isinstance(bindings, Mapping):
        for stage_id, binding in bindings.items():
            order = binding.get("stage_order") if isinstance(binding, Mapping) else None
            if isinstance(order, int) and not isinstance(order, bool):
                orders[str(stage_id)] = order
    return orders


def _src_functions(src_root: Path) -> Iterable[tuple[str, ast.AST]]:
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(src_root.parent).as_posix()
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{rel}::{stmt.name}", stmt


def run_audit(
    *,
    repo_root: Path = REPO_ROOT,
    orchestrators: Sequence[Path] | None = None,
    src_root: Path | None = None,
    registry_path: Path | None = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    adapter_max_lines: int = ADAPTER_MAX_LINES,
    include_units: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    src_root = Path(src_root) if src_root is not None else repo_root / DEFAULT_SRC_ROOT
    warnings: list[dict[str, Any]] = []
    if orchestrators:
        paths = []
        for raw in orchestrators:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.is_file():
                paths.append(candidate.resolve())
            else:
                warnings.append(
                    {
                        "code": "orchestrator_share_audit_path_missing",
                        "observed": str(raw),
                        "next_action": "Pass an existing orchestrator path with --orchestrator.",
                    }
                )
        discovery = "explicit"
    else:
        paths = discover_orchestrators(repo_root, registry_path)
        discovery = "discovered"

    analyses: list[OrchestratorAnalysis] = []
    for path in paths:
        try:
            analyses.append(parse_orchestrator(path, repo_root))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            warnings.append(
                {
                    "code": "orchestrator_share_audit_parse_failed",
                    "observed": f"{_display(path, repo_root)}: {exc}",
                    "next_action": "Fix the orchestrator so it parses, then rerun the share audit.",
                }
            )

    if any(order is None for analysis in analyses for _stage_id, order in analysis.stage_rows):
        registry_orders = registry_stage_orders(repo_root, registry_path)
        for analysis in analyses:
            analysis.stage_rows = [
                (stage_id, order if order is not None else registry_orders.get(stage_id))
                for stage_id, order in analysis.stage_rows
            ]

    token_re = conference_token_regex(analysis.slug for analysis in analyses)
    resolver = ComponentResolver(src_root)

    src_index = SimilarityIndex(threshold=similarity_threshold)
    src_function_count = 0
    if src_root.is_dir():
        for key, node in _src_functions(src_root):
            src_function_count += 1
            src_index.add(make_comparable(key, "src", normalized_lines(node, token_re), span=_span(node)))

    runner_index = SimilarityIndex(threshold=similarity_threshold)
    comparables: dict[tuple[str, str], Comparable] = {}
    for analysis in analyses:
        for unit in analysis.units.values():
            item = make_comparable(
                f"{analysis.display_path}::{unit.name}",
                analysis.display_path,
                normalized_lines(unit.node, token_re),
                span=unit.lines,
            )
            comparables[(analysis.display_path, unit.name)] = item
            runner_index.add(item)

    reports: list[dict[str, Any]] = []
    for analysis in analyses:
        classified = classify_orchestrator(analysis, resolver, adapter_max_lines=adapter_max_lines)
        for unit in analysis.units.values():
            item = comparables[(analysis.display_path, unit.name)]
            shadow = src_index.best_match(item)
            if shadow is not None:
                path, _, function = shadow[0].partition("::")
                unit.shadow_of = {"path": path, "function": function, "similarity": round(shadow[1], 4)}
            clone = runner_index.best_match(item, exclude_owner=analysis.display_path)
            if clone is not None:
                path, _, function = clone[0].partition("::")
                unit.clone_of = {"orchestrator": path, "unit": function, "similarity": round(clone[1], 4)}
        report: dict[str, Any] = {
            "path": analysis.display_path,
            "runner_slug": analysis.slug,
            "metrics": _metrics(analysis, classified["stages"]),
            "stages": classified["stages"],
            "extra_commands": classified["extra_commands"],
            "fallback_branches_dead": classified["fallback_branches_dead"],
            "delegated_to_shared_orchestrator": classified["delegated_to_shared_orchestrator"],
            "hooks": _hooks_report(analysis),
            "conference_local_components": sorted(
                {module for unit in analysis.units.values() for module in unit.local_components}
            ),
            "top_shadows": _top(analysis, "shadow_of"),
            "top_clones": _top(analysis, "clone_of"),
        }
        if include_units:
            report["units"] = [
                _unit_record(unit) for unit in sorted(analysis.units.values(), key=lambda item: item.start)
            ]
        reports.append(report)

    annotated = sorted(
        {
            module
            for analysis in analyses
            for unit in analysis.units.values()
            for module in unit.shared_components
            if resolver.annotation(module)[0]
        }
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "mode": MODE_REPORT_ONLY,
        "status": "pass",
        "blockers": [],
        "warnings": warnings,
        "discovery": discovery,
        "thresholds": {
            "similarity": similarity_threshold,
            "adapter_max_lines": adapter_max_lines,
            "min_comparable_lines": MIN_COMPARABLE_LINES,
        },
        "shared_component_rules": {
            "package_prefixes": list(SHARED_PACKAGE_PREFIXES),
            "annotation_constant": ANNOTATION_NAME,
            "annotated_modules_referenced": [
                {"module": module, "recommended_for": resolver.annotation(module)[1]} for module in annotated
            ],
        },
        "src_root": _display(src_root, repo_root),
        "src_function_count": src_function_count,
        "orchestrators": reports,
    }
    return attach_agent_diagnostic(
        payload,
        tool=TOOL_ID,
        checked_inputs={
            "orchestrators": [report["path"] for report in reports],
            "src_root": payload["src_root"],
        },
        accepted_evidence=[
            f"{len(reports)} orchestrator(s) parsed statically; nothing was executed.",
            "Report-only: no baseline or ratchet is enforced.",
        ],
        rerun_command="python -m conference_pipeline.orchestrator_share_audit --format json",
        pass_next_actions=[
            "Use the per-stage table to pick the next lift; shared stages stay adapters.",
        ],
    )


def _top(analysis: OrchestratorAnalysis, attr: str, limit: int = 10) -> list[dict[str, Any]]:
    rows = [unit for unit in analysis.units.values() if getattr(unit, attr) is not None]
    rows.sort(key=lambda unit: (-unit.lines, unit.name))
    return [{"unit": unit.name, "lines": unit.lines, "match": getattr(unit, attr)} for unit in rows[:limit]]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    out: list[str] = ["# Orchestrator share audit (report-only)", ""]
    out.append(
        "| orchestrator | total | module-level | local | hooks | adapters | dead | shadow of src | "
        "cross-runner clone | shared stages | local share |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report.get("orchestrators") or ():
        metrics = item["metrics"]
        out.append(
            f"| `{item['path']}` | {metrics['total_lines']:,} | {metrics['module_level_lines']:,} | "
            f"{metrics['local_lines']:,} | {metrics['declared_hook_lines']:,} | {metrics['adapter_lines']:,} | "
            f"{metrics['dead_lines']:,} | {metrics['shadow_lines']:,} | {metrics['clone_lines']:,} | "
            f"{metrics['shared_stage_count']}/{metrics['stage_count']} | {metrics['local_share']:.1%} |"
        )
    out += [
        "",
        "Local lines per stage count units reached only from that stage; units reached from",
        "several stages are shared-infra. A stage is local when it has such lines or its",
        "dispatch branch calls a local orchestrator unit directly (local entries).",
    ]
    for item in report.get("orchestrators") or ():
        out += ["", f"## `{item['path']}`", ""]
        out.append(
            "| stage | status | shared components | hooks | local lines | adapter lines | local entries |"
        )
        out.append("|---|---|---|---|---:|---:|---|")
        for stage in item["stages"]:
            components = ", ".join(f"`{module}`" for module in stage["shared_components"]) or "-"
            if stage["conference_local_components"]:
                components += " (local: " + ", ".join(
                    f"`{module}`" for module in stage["conference_local_components"]
                ) + ")"
            hooks = ", ".join(f"`{hook}`" for hook in stage["hooks"]) or "-"
            local_entries = ", ".join(f"`{name}`" for name in stage["local_entry_units"]) or "-"
            out.append(
                f"| {stage['stage_id']} | {stage['status']} | {components} | {hooks} | "
                f"{stage['local_lines']:,} | {stage['adapter_lines']:,} | {local_entries} |"
            )
        if item.get("extra_commands"):
            out.append("")
            out.append("Extra CLI commands: " + ", ".join(f"`--{flag}`" for flag in item["extra_commands"]))
        clones = item["metrics"].get("clone_lines_by_orchestrator") or {}
        if clones:
            out.append("")
            out.append(
                "Clone lines by peer: " + ", ".join(f"`{peer}` {lines:,}" for peer, lines in clones.items())
            )
    warnings = report.get("warnings") or ()
    if warnings:
        out += ["", "## Warnings", ""]
        out += [f"- {warning.get('code')}: {warning.get('observed')}" for warning in warnings]
    out.append("")
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--orchestrator",
        action="append",
        type=Path,
        default=None,
        help="Orchestrator path (repeatable). Default: discover scripts/run_*_production.py.",
    )
    parser.add_argument("--src-root", type=Path, default=None, help="Default: <repo-root>/src/conference_pipeline")
    parser.add_argument("--registry", type=Path, default=None, help="Stage owner registry used for discovery.")
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--adapter-max-lines", type=int, default=ADAPTER_MAX_LINES)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--no-units", action="store_true", help="Omit the per-unit list from JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        repo_root=args.repo_root,
        orchestrators=args.orchestrator,
        src_root=args.src_root,
        registry_path=args.registry,
        similarity_threshold=args.similarity_threshold,
        adapter_max_lines=args.adapter_max_lines,
        include_units=not args.no_units,
    )
    markdown = render_markdown(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(markdown)
    # Report-only in this package: never fail the caller.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
