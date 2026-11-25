"""Declared conference quirks: the hook interface a thin orchestrator implements.

Each conference keeps its own production orchestrator because every
conference has quirks (OpenReview auth, SlidesLive recovery, organizer YouTube
channels, registration media, ...). The shared stages own the common
behaviour; an orchestrator plugs its quirks in through :class:`ConferenceHooks`
and marks every function that implements one with :func:`conference_hook`.
The orchestrator share audit counts decorated functions as declared quirks and
everything else orchestrator-local as undeclared logic.

Every hook defaults to ``None`` or empty, and the shared orchestrator behaves
correctly with ``ConferenceHooks()``: a conference with no quirks declares a
stage table, its stage entry functions and nothing else.

Hook ids are ``<scope>.<name>``. The scope is a stage id from the
orchestrator's stage table (``download-corpus.acquisition_lane``) or ``cli``
for command-line extras (``cli.extra_cli_args``).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

HOOK_ATTRIBUTE = "conference_hook_id"
CLI_HOOK_SCOPE = "cli"
HOOK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_.-]*$")

F = TypeVar("F", bound=Callable[..., Any])


class HookDeclarationError(ValueError):
    """A hook id or hook set that does not fit the orchestrator's stage table."""


def conference_hook(hook_id: str) -> Callable[[F], F]:
    """Mark an orchestrator function as the implementation of a declared hook.

    The function is returned unchanged; only ``conference_hook_id`` is set on
    it, so decorating never alters behaviour.
    """

    if not isinstance(hook_id, str) or not HOOK_ID_RE.match(hook_id):
        raise HookDeclarationError(f"hook id must look like '<stage-or-cli>.<name>': {hook_id!r}")

    def decorate(function: F) -> F:
        existing = getattr(function, HOOK_ATTRIBUTE, None)
        if existing is not None and existing != hook_id:
            raise HookDeclarationError(f"{function!r} is already declared as hook {existing!r}")
        setattr(function, HOOK_ATTRIBUTE, hook_id)
        return function

    return decorate


def hook_id_of(function: Any) -> str | None:
    value = getattr(function, HOOK_ATTRIBUTE, None)
    return value if isinstance(value, str) else None


def hook_scope(hook_id: str) -> str:
    return hook_id.split(".", 1)[0]


def declared_hooks(namespace: Mapping[str, Any] | Any) -> dict[str, Callable[..., Any]]:
    """``{hook_id: function}`` for every decorated function in a module or mapping."""

    items = namespace.items() if isinstance(namespace, Mapping) else vars(namespace).items()
    found: dict[str, Callable[..., Any]] = {}
    for _name, value in items:
        hook_id = hook_id_of(value)
        if hook_id is None or not callable(value):
            continue
        previous = found.get(hook_id)
        if previous is not None and previous is not value:
            raise HookDeclarationError(f"hook {hook_id!r} is declared by more than one function")
        found[hook_id] = value
    return found


# --------------------------------------------------------------------------
# Hook payloads
# --------------------------------------------------------------------------


LANE_PHASE_ACQUIRE = "acquire"
LANE_PHASE_GATE = "gate"
LANE_PHASES = (LANE_PHASE_ACQUIRE, LANE_PHASE_GATE)


@dataclass(frozen=True)
class LaneResult:
    """What one acquisition lane contributed to Stage 4.

    ``artifacts`` are direct downloads the frame appends to the stage's
    artifact lists after the lane returns (a lane that must interleave its
    records with shared state records them through the lane context instead);
    ``blockers`` are appended in lane order; ``receipt_fields`` are merged into
    the Stage 4 ledger (frozen-corpus format) or the stage checks
    (program-ledger format). ``completed=False`` records the lane's declared
    ``incomplete_blocker`` when the stage closes; ``resolves`` names blocker
    codes this outcome supersedes (removed after its own blockers are added).
    """

    artifacts: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    receipt_fields: dict[str, Any] = field(default_factory=dict)
    completed: bool = True
    resolves: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcquisitionLane:
    """A conference-specific Stage 4 acquisition lane (zero or more per conference).

    ``run(ctx, frozen_corpus, prior_inventory)`` returns a :class:`LaneResult`.
    A ``paid`` lane must be authorized by the preflight stage before it runs;
    the shared frame skips an unauthorized one. ``phase`` orders the lane:
    every ``acquire`` lane runs (in declaration order) before every ``gate``
    lane (coverage checks over what was acquired). ``incomplete_blocker`` is
    the blocker code recorded when the lane did not complete.
    """

    lane_id: str
    run: Callable[..., LaneResult]
    paid: bool = False
    phase: str = LANE_PHASE_ACQUIRE
    incomplete_blocker: str | None = None


@dataclass(frozen=True)
class ExtraCommand:
    """A non-stage command (``--prepare-*`` / ``--authorize-*``) an orchestrator adds.

    ``run(args)`` returns a JSON-serialisable payload printed on stdout (exit
    0); an ``OSError``/``ValueError`` is reported as ``"<failure_label>:
    <error>"`` on stderr (exit 1). ``validate(parser, args, raw_argv)`` runs at
    parse time instead of the stage-dispatch checks.
    """

    run: Callable[[argparse.Namespace], Any]
    failure_label: str
    validate: Callable[[argparse.ArgumentParser, argparse.Namespace, Sequence[str]], None] | None = None


_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class ConferenceHooks:
    """Every quirk a conference orchestrator may plug into the shared stages.

    The stage hooks (Stages 1-7) are consumed by the shared stage modules as
    they land (W4+); the CLI hooks are consumed by
    :mod:`conference_pipeline.production.orchestrator` today.
    """

    # Stage 1: ``provider_probes(ctx)`` replaces the planned provider probe
    # specs; each ``extra_preflight_checks(ctx)`` returns blocker dicts that
    # block the preflight before any environment reload or probe.
    provider_probes: Callable[..., Sequence[Any]] | None = None
    extra_preflight_checks: Sequence[Callable[..., list[dict[str, Any]]]] = ()
    # Stage 2: ``source_family_probe(url, timeout_seconds=, recovery=)`` probes
    # one source-intake family URL (the probed source-intake format).
    source_families: Callable[..., Sequence[Any]] | None = None
    source_family_probe: Callable[..., dict[str, Any]] | None = None
    # Stage 3: ``corpus_exclusions(inventory)`` returns ``(policy_exclusions,
    # blockers)``: locked source-policy postures that stay auditable without
    # removing program items.
    corpus_exclusions: Callable[..., Any] | None = None
    # Stage 4
    auth: Callable[..., Any] | None = None
    acquisition_lanes: Sequence[AcquisitionLane] = ()
    paper_priority_key: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None
    request_pacer: Callable[..., Any] | None = None
    # Stage 5: ``transcription_lanes`` are
    # ``transcription_ocr_stage.TranscriptionLane`` values (a conference with
    # none runs the default local lanes); ``retry_authorization`` is a
    # ``transcription_ocr_stage.RetryAuthorization`` for one operator-authorized
    # retry of a blocked Stage 5.
    transcription_lanes: Sequence[Any] = ()
    retry_authorization: Any | None = None
    # Stage 6: ``record_builders`` are ``corpus_index_stage.RecordBuilder``
    # steps run in declaration order per phase (admit, build, curate, gate);
    # a conference with none runs its Stage 6 format's default builders.
    record_builders: Sequence[Any] = ()
    # Stage 7
    source_pack_policy: Any | None = None
    matched_papers_loader: Callable[..., list[dict[str, Any]]] | None = None
    # CLI: flags added after the shared core flags, checks run after the
    # shared live-only/--stage checks, and non-stage commands keyed by their
    # argparse dest (checked and run in declaration order).
    extra_cli_args: Callable[[argparse.ArgumentParser], None] | None = None
    validate_args: Callable[[argparse.ArgumentParser, argparse.Namespace, Sequence[str]], None] | None = None
    extra_commands: Mapping[str, ExtraCommand] = _EMPTY


NO_HOOKS = ConferenceHooks()


def validate_hooks(
    hooks: ConferenceHooks,
    stage_ids: Iterable[str],
    *,
    declared: Mapping[str, Callable[..., Any]] | None = None,
) -> list[str]:
    """Return every way ``hooks`` (and decorated functions) disagree with the stage table."""

    problems: list[str] = []
    known = set(stage_ids) | {CLI_HOOK_SCOPE}
    lane_ids = [lane.lane_id for lane in hooks.acquisition_lanes]
    duplicates = sorted({lane_id for lane_id in lane_ids if lane_ids.count(lane_id) > 1})
    if duplicates:
        problems.append("duplicate acquisition lane ids: " + ", ".join(duplicates))
    for lane in hooks.acquisition_lanes:
        if not lane.lane_id:
            problems.append("acquisition lane with an empty lane_id")
        if not callable(lane.run):
            problems.append(f"acquisition lane {lane.lane_id!r} run is not callable")
        if lane.phase not in LANE_PHASES:
            problems.append(f"acquisition lane {lane.lane_id!r} has an unknown phase {lane.phase!r}")
    if hooks.acquisition_lanes and "download-corpus" not in known:
        problems.append("acquisition lanes declared but the stage table has no download-corpus stage")
    builder_ids = [getattr(builder, "builder_id", None) for builder in hooks.record_builders]
    duplicate_builders = sorted({str(item) for item in builder_ids if builder_ids.count(item) > 1})
    if duplicate_builders:
        problems.append("duplicate record builder ids: " + ", ".join(duplicate_builders))
    if any(not builder_id for builder_id in builder_ids):
        problems.append("record builder with an empty builder_id")
    if hooks.record_builders and "corpus-inventory-index" not in known:
        problems.append("record builders declared but the stage table has no corpus-inventory-index stage")
    for dest, command in hooks.extra_commands.items():
        if not dest.isidentifier():
            problems.append(f"extra command key {dest!r} is not an argparse dest")
        if not callable(command.run):
            problems.append(f"extra command {dest!r} run is not callable")
    for hook_id in sorted(declared or {}):
        if hook_scope(hook_id) not in known:
            problems.append(f"hook {hook_id!r} names no stage in the stage table")
    return problems
