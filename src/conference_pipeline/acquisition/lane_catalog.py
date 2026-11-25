"""Catalog of every Stage 4 acquisition lane, with the conferences it is recommended for.

Each conference orchestrator composes its ``download-corpus`` stage from the
shared frame (:mod:`conference_pipeline.production.download_corpus_stage`) and
a sequence of declared lanes (``ConferenceHooks.acquisition_lanes``, plus the
``download-corpus.auth`` / ``paper_priority_key`` / ``request_pacer`` hooks).
The lanes are written once, in capability-named modules under
``conference_pipeline.acquisition``; this catalog lists them so that planning a
new conference starts from the lanes recommended for it (a new CVPR gets the
CVPR-recommended lanes, a new ICML the ICML-recommended ones) instead of
copying a previous runner.

``recommended_for`` names conference *families* (``CVPR``, ``ICML``, ...),
matched against a conference name or slug by its leading letters
(``"CVPR 2027"``, ``"cvpr2027"`` -> ``CVPR``). ``may_fit`` lists other
families the lane plausibly serves; it is advisory and never selects a lane.
A module that declares ``RECOMMENDED_FOR`` must agree with its catalog entries
(:func:`catalog_problems`, enforced by the tests).

Run::

    python -m conference_pipeline.acquisition.lane_catalog                       # every lane
    python -m conference_pipeline.acquisition.lane_catalog --conference "CVPR 2027"
    python -m conference_pipeline.acquisition.lane_catalog --format json
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

STAGE_DOWNLOAD_CORPUS = "download-corpus"

KIND_LANE = "lane"  # an AcquisitionLane factory (acquire phase)
KIND_GATE = "gate"  # an AcquisitionLane factory (gate phase)
KIND_EXCLUSIVE_LANE = "exclusive-lane"  # a mutually exclusive Stage 4 lane owner (select_lane route)
KIND_AUTH = "auth"  # a download-corpus.auth building block
KIND_COMPONENT = "component"  # a lane building block the orchestrator binds (owner, transport, pacer, priority)
KINDS = (KIND_LANE, KIND_GATE, KIND_EXCLUSIVE_LANE, KIND_AUTH, KIND_COMPONENT)

_FAMILY_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class LaneCatalogEntry:
    """One acquisition lane (or lane building block) and where it is recommended."""

    lane_id: str
    module: str
    entry_point: str
    kind: str
    recommended_for: tuple[str, ...]
    purpose: str
    quirk: str
    may_fit: tuple[str, ...] = ()
    paid: bool = False
    stage_id: str = STAGE_DOWNLOAD_CORPUS


_ACQ = "conference_pipeline.acquisition"

LANE_CATALOG: tuple[LaneCatalogEntry, ...] = (
    # ---- CVPR-recommended (D4: moved out of the CVPR runner) -------------
    LaneCatalogEntry(
        lane_id="public-program-items",
        module=f"{_ACQ}.program_item_acquisition",
        entry_point="public_acquisition_lane",
        kind=KIND_LANE,
        recommended_for=("CVPR",),
        purpose="Acquire every frozen program item's public sources into the per-item Stage 4 ledger (resume, byte reservation, bounded transport).",
        quirk="CVPR r3 public lane over a pinned 14k-item program freeze",
        may_fit=("ICCV", "WACV", "ECCV"),
    ),
    LaneCatalogEntry(
        lane_id="cvf-workshop-owner",
        module=f"{_ACQ}.cvf_proceedings",
        entry_point="workshop_owner",
        kind=KIND_COMPONENT,
        recommended_for=("CVPR",),
        purpose="Route workshop requests once per CVF parent venue to the registered OpenReview workshop owner; credential-free public transport.",
        quirk="CVPR workshops live under the CVF OpenReview parent venue",
        may_fit=("ICCV", "WACV"),
    ),
    LaneCatalogEntry(
        lane_id="cvf-openaccess-workshop-papers",
        module=f"{_ACQ}.cvf_openaccess_workshops",
        entry_point="run",
        kind=KIND_COMPONENT,
        recommended_for=("CVPR",),
        purpose="Download archival workshop papers from CVF Open Access that OpenReview leaves as external-site stubs.",
        quirk="CVPR workshop proceedings not on OpenReview",
        may_fit=("ICCV", "WACV"),
    ),
    LaneCatalogEntry(
        lane_id="organizer-channel-talks",
        module=f"{_ACQ}.youtube_channel_harvest",
        entry_point="talk_lane",
        kind=KIND_LANE,
        recommended_for=("CVPR",),
        purpose="Harvest priority talks (keynote/invited/oral) from public organizer YouTube channels, captions only, bound to the program ledger.",
        quirk="CVPR talks are not on CVF/SlidesLive; they live on organizer YouTube channels",
        may_fit=("ICCV", "WACV", "ECCV"),
    ),
    LaneCatalogEntry(
        lane_id="priority-talk-gap-search",
        module=f"{_ACQ}.talk_gap_search",
        entry_point="talk_gap_lane",
        kind=KIND_LANE,
        recommended_for=("CVPR",),
        purpose="Escalate only the exact unmatched priority-talk frontier to a governed paid web search (frontier-scoped authority, fresh preflight per call).",
        quirk="CVPR Tavily priority-gap escalation",
        may_fit=("ICCV", "WACV", "ECCV"),
        paid=True,
    ),
    LaneCatalogEntry(
        lane_id="program-talk-spine",
        module=f"{_ACQ}.program_spine",
        entry_point="build_pinned_program_spine",
        kind=KIND_COMPONENT,
        recommended_for=("CVPR",),
        purpose="Derive the source-bound talk spine from the pinned program freeze for the talk lane.",
        quirk="CVPR schedule pages carry family paths, not a joined talk table",
        may_fit=("ICCV", "WACV"),
    ),
    LaneCatalogEntry(
        lane_id="cached-slideslive-focused-recovery",
        module=f"{_ACQ}.program_spine",
        entry_point="focused_recovery_lane",
        kind=KIND_LANE,
        recommended_for=("CVPR",),
        purpose="Assess cached item-page probes for SlidesLive embeds that need the focused recovery owner.",
        quirk="CVPR r3 focused-recovery disposition",
        may_fit=("ICCV", "WACV"),
    ),
    LaneCatalogEntry(
        lane_id="registration-media",
        module=f"{_ACQ}.registration_media",
        entry_point="run_registration_media_lane",
        kind=KIND_EXCLUSIVE_LANE,
        recommended_for=("CVPR",),
        purpose="Recover official media links from registration-gated virtual-site pages behind one authenticated sentinel (overlay ledger).",
        quirk="CVPR r4 registration media",
        may_fit=("ICCV", "WACV"),
    ),
    LaneCatalogEntry(
        lane_id="event-media",
        module=f"{_ACQ}.event_media",
        entry_point="run_event_media_lane",
        kind=KIND_EXCLUSIVE_LANE,
        recommended_for=("CVPR",),
        purpose="Census every virtual-program event page and fetch caption sidecars only for its official YouTube recordings.",
        quirk="CVPR r5 event media (keynotes, tutorials, workshops, orals, sessions)",
        may_fit=("ICCV", "WACV", "ECCV"),
    ),
    # ---- ICML-recommended -------------------------------------------------
    LaneCatalogEntry(
        lane_id="openreview-bearer-auth",
        module="conference_pipeline.openreview_auth",
        entry_point="load_openreview_auth",
        kind=KIND_AUTH,
        recommended_for=("ICML",),
        purpose="Authenticate OpenReview API/PDF requests with a Bearer token (secret-free receipt fields) for the download-corpus.auth hook.",
        quirk="ICML OpenReview bot gate on notes/PDFs",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="slideslive-carry-forward",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="prior_artifact_carry_forward_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Carry a focused recovery owner's governed records (SlidesLive audio) from the prior inventory into a resumed Stage 4.",
        quirk="ICML SlidesLive recovery owner runs between Stage 4 attempts",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="openreview-workshops-first",
        module=f"{_ACQ}.openreview_accepted_papers",
        entry_point="workshop_first_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Acquire OpenReview workshop venues (uncapped, rate-limited) before the broad conference paper lane.",
        quirk="ICML workshop-first acquisition order",
        may_fit=("ICLR", "NeurIPS", "CVPR"),
    ),
    LaneCatalogEntry(
        lane_id="virtual-site-source-pages",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="source_page_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Download each frozen source family's virtual-site page (session cookie on the site host) and record the targets it links.",
        quirk="ICML virtual-site family pages",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="virtual-site-direct-targets",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="direct_target_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Download every directly downloadable target (PDFs, archives, media files) the source pages link.",
        quirk="ICML virtual-site direct assets",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="openreview-accepted-papers",
        module=f"{_ACQ}.openreview_accepted_papers",
        entry_point="accepted_papers_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Import accepted-paper metadata from the OpenReview API, order by a priority keyword manifest, download every PDF (bounded futures).",
        quirk="ICML OpenReview auth + physics-first paper priority",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="physics-paper-priority",
        module=f"{_ACQ}.openreview_accepted_papers",
        entry_point="keyword_priority_key",
        kind=KIND_COMPONENT,
        recommended_for=("ICML",),
        purpose="Order papers by title, then abstract/keyword hits of a workspace priority manifest (the download-corpus.paper_priority_key hook).",
        quirk="ICML physics-first ordering",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="virtual-site-item-page-expansion",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="item_page_expansion_lane",
        kind=KIND_LANE,
        recommended_for=("ICML",),
        purpose="Probe every discovered item page through a global request pacer, retry auth-gated probes slower, download exposed media.",
        quirk="ICML server-side burst throttle on item pages",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="request-pacer",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="RequestPacer",
        kind=KIND_COMPONENT,
        recommended_for=("ICML",),
        purpose="Bound the aggregate request rate across a worker pool (the download-corpus.request_pacer hook).",
        quirk="ICML item-page throttling",
        may_fit=("ICLR", "NeurIPS"),
    ),
    LaneCatalogEntry(
        lane_id="virtual-site-media-family-coverage",
        module=f"{_ACQ}.virtual_site_crawl",
        entry_point="media_family_coverage_lane",
        kind=KIND_GATE,
        recommended_for=("ICML",),
        purpose="Fail closed when a declared media-bearing program family (tutorials, invited talks, orals, workshops, expo) yielded no item page.",
        quirk="ICML media-family coverage defect (v4.0 recovery)",
        may_fit=("ICLR", "NeurIPS"),
    ),
)


def conference_family(name: str) -> str:
    """``"CVPR 2027"`` / ``"cvpr2027"`` / ``"CVPR"`` -> ``"CVPR"`` (leading letters, upper-cased)."""

    match = _FAMILY_RE.match(str(name or "").strip())
    return match.group(0).upper() if match else ""


def lanes_recommended_for(conference: str, catalog: Iterable[LaneCatalogEntry] = LANE_CATALOG) -> list[LaneCatalogEntry]:
    """Catalog entries recommended for ``conference`` (by family), in catalog order."""

    family = conference_family(conference)
    return [entry for entry in catalog if family and family in {item.upper() for item in entry.recommended_for}]


def lanes_that_may_fit(conference: str, catalog: Iterable[LaneCatalogEntry] = LANE_CATALOG) -> list[LaneCatalogEntry]:
    """Entries not recommended for ``conference`` but listing its family in ``may_fit``."""

    family = conference_family(conference)
    recommended = {entry.lane_id for entry in lanes_recommended_for(conference, catalog)}
    return [
        entry
        for entry in catalog
        if family and entry.lane_id not in recommended and family in {item.upper() for item in entry.may_fit}
    ]


def module_recommended_for(module: str) -> tuple[str, ...] | None:
    """The module's own ``RECOMMENDED_FOR`` annotation, or ``None`` when it declares none."""

    value = getattr(importlib.import_module(module), "RECOMMENDED_FOR", None)
    return tuple(value) if value is not None else None


def catalog_problems(catalog: Sequence[LaneCatalogEntry] = LANE_CATALOG) -> list[str]:
    """Every way the catalog disagrees with itself or with the modules it names."""

    problems: list[str] = []
    seen: set[str] = set()
    for entry in catalog:
        if entry.lane_id in seen:
            problems.append(f"duplicate lane id {entry.lane_id!r}")
        seen.add(entry.lane_id)
        if entry.kind not in KINDS:
            problems.append(f"{entry.lane_id}: unknown kind {entry.kind!r}")
        if not entry.recommended_for:
            problems.append(f"{entry.lane_id}: recommended_for is empty")
        if not entry.purpose.strip():
            problems.append(f"{entry.lane_id}: purpose is empty")
        try:
            module = importlib.import_module(entry.module)
        except ImportError as exc:
            problems.append(f"{entry.lane_id}: module {entry.module} does not import ({exc})")
            continue
        if not hasattr(module, entry.entry_point):
            problems.append(f"{entry.lane_id}: {entry.module} has no {entry.entry_point}")
        declared = getattr(module, "RECOMMENDED_FOR", None)
        if declared is not None and tuple(declared) != tuple(entry.recommended_for):
            problems.append(
                f"{entry.lane_id}: catalog recommends {list(entry.recommended_for)} but "
                f"{entry.module}.RECOMMENDED_FOR is {list(declared)}"
            )
    return problems


def render_markdown(entries: Sequence[LaneCatalogEntry], *, title: str = "Acquisition lane catalog") -> str:
    lines = [
        f"# {title}",
        "",
        "| lane | kind | recommended for | may fit | module | purpose |",
        "|---|---|---|---|---|---|",
    ]
    for entry in entries:
        paid = " (paid)" if entry.paid else ""
        lines.append(
            f"| `{entry.lane_id}`{paid} | {entry.kind} | {', '.join(entry.recommended_for)} | "
            f"{', '.join(entry.may_fit) or '-'} | `{entry.module}.{entry.entry_point}` | {entry.purpose} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the Stage 4 acquisition lanes and where they are recommended.")
    parser.add_argument("--conference", default=None, help='Conference name or slug (its family is the leading letters).')
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    if args.conference:
        recommended = lanes_recommended_for(args.conference)
        may_fit = lanes_that_may_fit(args.conference)
    else:
        recommended, may_fit = list(LANE_CATALOG), []
    if args.format == "json":
        payload = {
            "conference": args.conference,
            "family": conference_family(args.conference) if args.conference else None,
            "recommended": [asdict(entry) for entry in recommended],
            "may_fit": [asdict(entry) for entry in may_fit],
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    if args.conference:
        family = conference_family(args.conference)
        sys.stdout.write(render_markdown(recommended, title=f"Lanes recommended for {family}"))
        if may_fit:
            sys.stdout.write("\n" + render_markdown(may_fit, title=f"Other lanes that may fit {family}"))
        if not recommended:
            sys.stdout.write(
                "\nNo lane is recommended for this conference family yet: start from the "
                "frozen-corpus frame with no lanes (ConferenceHooks()) and add lanes from "
                "the full catalog only where the analysis spec records the quirk.\n"
            )
        return 0
    sys.stdout.write(render_markdown(recommended))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
