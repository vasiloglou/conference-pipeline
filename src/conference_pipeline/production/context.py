"""The runner-supplied context every shared production stage runs under.

A conference runner builds one :class:`ProductionRunnerContext` per
invocation. It carries the three things the shared stages must never guess:
which workspace and run they act on, which stage table (ids, orders, receipt
filenames) the run follows, and what identity the runner records on every
receipt it writes (receipt type and code identity for lineage). Everything
conference-specific -- the slug, the executive deck ids, the podcast layout --
comes from the workspace or the runner through this object, so the stage
modules themselves carry no conference literal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    conference_name,
    load_conference_identity,
)
from conference_pipeline.receipt_lineage import (
    finalize_receipt,
    receipt_path_resolver_from_registry,
    summarize_lineage_blockers,
    upstream_freshness_blockers,
)
from conference_pipeline.stage_contracts import StageContractResult, verify_stage_contract
from conference_pipeline.workflow_runs import validate_run_id

UPSTREAM_LINEAGE_NOT_FRESH = "upstream_lineage_not_fresh"
UPSTREAM_LINEAGE_REMEDY = (
    "An upstream stage is stale or unbound: run `conference-pipeline workflow stage-status "
    "--workspace <workspace> --run-id <run>` and re-run its next_stage_to_run first."
)

DEFAULT_EXECUTIVE_OVERVIEW_DECKS: tuple[tuple[str, str], ...] = (
    ("executive_summary_short", "Executive Summary -- Short"),
    ("executive_summary_long", "Executive Summary -- Long"),
)


@runtime_checkable
class StageSpec(Protocol):
    """The shape every runner's stage-table row already has."""

    id: str
    order: int
    receipt_filename: str
    implemented: bool
    next_action: str


@dataclass(frozen=True)
class ProductionRunnerContext:
    """Everything a shared stage needs from the runner that invoked it."""

    workspace: Path
    run_id: str
    stages: Mapping[str, StageSpec]
    runner_id: str
    runner_path: str
    receipt_type: str
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    conference_slug: str | None = None
    executive_decks: tuple[tuple[str, str], ...] = DEFAULT_EXECUTIVE_OVERVIEW_DECKS
    decks_dir: Path = Path("decks")
    podcast_dir: Path = Path("audio")
    analysis_manifest: Path = Path("analysis") / "analysis_manifest.json"
    analysis_executive_summary: Path = Path("analysis") / "executive_summary.md"
    analysis_trend_report: Path = Path("analysis") / "trend_report.md"
    notebooklm_subdir: str = "notebooklm_podcast"
    notebooklm_audio_filename: str = "podcast.m4a"
    # A computer-vision conference expects visual-domain figures (frames,
    # images, 3-D, tokens) in every deck; the figure compiler's variety gate
    # enforces a minimum only when this is set (CVPR). Off for ML venues.
    figure_vision_deck: bool = False
    # Runner conventions the shared stages call back into. ``env_loader``
    # loads an explicit --env-file under the runner's own allowlist and
    # returns status flags only (never a value). ``summary_writer`` receives
    # each written stage receipt and returns the run summary the runner wants
    # reported (None: the stage's own compact summary is reported and the
    # runner writes summary.json itself). ``provider_receipt_caller`` is the
    # path provider receipts attribute calls to (the runner script, not this
    # package).
    env_loader: Callable[[Path | None], dict[str, bool]] | None = None
    summary_writer: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    provider_receipt_caller: str | None = None
    # ``cost_approval(flag)`` returns the runner's provider-cost approval
    # record (its own env variable and estimate commands); None falls back to
    # the command-line flag alone.
    cost_approval: Callable[[bool], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        object.__setattr__(self, "run_id", validate_run_id(str(self.run_id)))
        if self.conference_slug is None:
            try:
                slug = load_conference_identity(self.workspace).slug
            except ConferenceIdentityError:
                slug = self.workspace.name
            object.__setattr__(self, "conference_slug", slug)

    # -- identity --------------------------------------------------------

    @property
    def code_identity(self) -> dict[str, str]:
        return {"runner": self.runner_id, "runner_path": self.runner_path}

    @property
    def slug(self) -> str:
        return str(self.conference_slug)

    def conference_display_name(self, default: str = "") -> str:
        return conference_name(self.workspace, default=default)

    def load_env(self, env_file: Path | None) -> dict[str, bool]:
        """Load ``env_file`` through the runner's allowlisted loader; flags only."""

        if self.env_loader is None or env_file is None:
            return {"attempted": env_file is not None, "exists": bool(env_file and env_file.exists()), "loaded": False}
        return self.env_loader(env_file)

    @property
    def receipt_caller(self) -> str:
        return self.provider_receipt_caller or self.runner_path

    def provider_cost_approval(self, approved_flag: bool) -> dict[str, Any]:
        """The runner's approval record for paid provider calls, flags only."""

        if self.cost_approval is not None:
            return self.cost_approval(bool(approved_flag))
        return {
            "approved": bool(approved_flag),
            "approval_sources": ["--provider-cost-approved"] if approved_flag else [],
            "required_before_paid_provider_calls": True,
            "secret_values_recorded": False,
        }

    # -- paths -----------------------------------------------------------

    def run_dir(self) -> Path:
        target = (self.workspace / "storage" / "workflow-runs" / self.run_id).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover - validate_run_id guards normal inputs
            raise ValueError("workflow run directory must remain inside the workspace") from exc
        return target

    def stage(self, stage_id: str) -> StageSpec:
        try:
            return self.stages[stage_id]
        except KeyError as exc:
            raise KeyError(f"runner {self.runner_id!r} declares no stage {stage_id!r}") from exc

    def receipt_path(self, stage_id: str) -> Path:
        return self.run_dir() / self.stage(stage_id).receipt_filename

    def deck_dir(self, deck_id: str) -> Path:
        return self.workspace / self.decks_dir / deck_id

    def implemented_stage_receipts(
        self, *, min_order: int | None = None, max_order_exclusive: int | None = None
    ) -> dict[str, Path]:
        """Receipt path per implemented stage in an order window (both bounds optional)."""

        run_dir = self.run_dir()
        selected: dict[str, Path] = {}
        for stage_id, entry in self.stages.items():
            if not entry.implemented:
                continue
            if min_order is not None and entry.order < min_order:
                continue
            if max_order_exclusive is not None and entry.order >= max_order_exclusive:
                continue
            selected[stage_id] = run_dir / entry.receipt_filename
        return selected

    # -- receipts and lineage (v5.1) --------------------------------------

    def finalize(
        self,
        receipt: dict[str, Any],
        *,
        stage_id: str,
        lineage: Mapping[str, Any] | None = None,
        bind: bool = True,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """Attach output artifacts, ``output_digest`` and ``lineage``.

        A fresh run binds; a resume path hands back the receipt's own recorded
        lineage and never binds afresh, so a stale receipt can only fail
        verification, never heal.
        """

        return finalize_receipt(
            receipt,
            workspace=self.workspace,
            run_id=self.run_id,
            stage_id=stage_id,
            run_dir=self.run_dir(),
            lineage=lineage,
            bind=bind,
            profile=profile,
            code_identity=self.code_identity,
        )

    def verify_contract(
        self,
        *,
        stage_id: str,
        receipt: Mapping[str, Any] | None,
        bind: bool,
        mode: str = "live",
        strict_execution: bool = False,
        profile: str | None = None,
        lineage: Mapping[str, Any] | None = None,
        registry_path: str | Path | None = None,
        required_owner_ids: Sequence[str] | None = None,
    ) -> StageContractResult:
        """``verify_stage_contract`` with lineage: bind on a fresh run's
        provisional receipt, verify as-is on a persisted one (``bind=False``),
        always requiring lineage."""

        payload: dict[str, Any] = dict(receipt or {})
        if bind or lineage is not None:
            payload = self.finalize(payload, stage_id=stage_id, lineage=lineage, bind=bind, profile=profile)
        return verify_stage_contract(
            workspace=self.workspace,
            run_id=self.run_id,
            stage_id=stage_id,
            receipt=payload,
            mode=mode,
            strict_execution=strict_execution,
            profile=profile,
            registry_path=registry_path,
            required_owner_ids=required_owner_ids,
            require_lineage=True,
        )

    def upstream_lineage_checks(self, stage_id: str) -> dict[str, Any] | None:
        """Pre-spend lineage gate: ``None`` when every upstream is fresh, else
        the checks a blocked receipt records."""

        blockers = upstream_freshness_blockers(
            self.workspace,
            self.run_id,
            stage_id,
            receipt_path_for_stage=receipt_path_resolver_from_registry(self.workspace, self.run_dir()),
        )
        if not blockers:
            return None
        return {
            "upstream_lineage_blockers": summarize_lineage_blockers(blockers),
            "upstream_lineage_remedy": UPSTREAM_LINEAGE_REMEDY,
        }
