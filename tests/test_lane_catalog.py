"""Acquisition lane catalog and the ``RECOMMENDED_FOR`` annotations (D4).

The CVPR quirk lanes moved out of the CVPR runner into capability-named
shared acquisition modules; ICML's lanes did the same. Each such module
declares ``RECOMMENDED_FOR`` and documents the stage/quirk it covers and the
other conferences it may fit; the lane catalog lists every lane so planning a
new conference can propose the recommended ones; the old ``cvpr_*`` import
paths stay as aliases of the very same module objects; the share audit counts
the annotated modules as shared components.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from conference_pipeline.acquisition import lane_catalog
from conference_pipeline.orchestrator_share_audit import ComponentResolver
from conference_pipeline.production.hooks import AcquisitionLane

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "conference_pipeline"

# Shared lane modules introduced or renamed by W8 / D4.
CVPR_LANE_MODULES = (
    "program_item_acquisition",
    "program_spine",
    "registration_media",
    "event_media",
    "youtube_channel_harvest",
    "talk_gap_search",
    "cvf_proceedings",
    "cvf_openaccess_workshops",
)
ICML_LANE_MODULES = ("virtual_site_crawl", "openreview_accepted_papers")
COMPATIBILITY_ALIASES = {
    "cvpr_corpus_download": "program_item_acquisition",
    "cvpr_program_spine": "program_spine",
    "cvpr_registration_media": "registration_media",
    "cvpr_event_media": "event_media",
}


def test_catalog_is_consistent_with_itself_and_the_modules_it_names() -> None:
    assert lane_catalog.catalog_problems() == []


@pytest.mark.parametrize(
    ("modules", "family"),
    [(CVPR_LANE_MODULES, "CVPR"), (ICML_LANE_MODULES, "ICML")],
)
def test_lane_modules_declare_where_they_are_recommended(modules: tuple[str, ...], family: str) -> None:
    for name in modules:
        module = importlib.import_module(f"conference_pipeline.acquisition.{name}")
        assert module.RECOMMENDED_FOR == (family,), name
        doc = module.__doc__ or ""
        assert "Recommended for" in doc and "RECOMMENDED_FOR" in doc, name
        assert "stage and quirk covered" in doc, name
        assert "May also fit" in doc, name


def test_every_annotated_lane_module_is_in_the_catalog() -> None:
    cataloged = {entry.module for entry in lane_catalog.LANE_CATALOG}
    for name in (*CVPR_LANE_MODULES, *ICML_LANE_MODULES):
        assert f"conference_pipeline.acquisition.{name}" in cataloged, name


def test_lane_factories_build_acquisition_lanes() -> None:
    for entry in lane_catalog.LANE_CATALOG:
        if entry.kind not in {lane_catalog.KIND_LANE, lane_catalog.KIND_GATE}:
            continue
        factory = getattr(importlib.import_module(entry.module), entry.entry_point)
        assert callable(factory), entry.lane_id
        assert factory.__annotations__.get("return") in {"AcquisitionLane", "Any", AcquisitionLane}, entry.lane_id


def test_paid_lanes_are_marked_paid_in_the_catalog() -> None:
    paid = {entry.lane_id for entry in lane_catalog.LANE_CATALOG if entry.paid}
    assert paid == {"priority-talk-gap-search"}


@pytest.mark.parametrize("old,new", sorted(COMPATIBILITY_ALIASES.items()))
def test_old_cvpr_import_paths_are_the_same_module_objects(old: str, new: str) -> None:
    alias = importlib.import_module(f"conference_pipeline.acquisition.{old}")
    target = importlib.import_module(f"conference_pipeline.acquisition.{new}")
    assert alias is target
    source = (SRC_ROOT / "acquisition" / f"{old}.py").read_text(encoding="utf-8")
    assert "sys.modules[__name__]" in source and len(source.splitlines()) < 20


@pytest.mark.parametrize("name", ["CVPR 2027", "cvpr2027", "CVPR"])
def test_a_new_cvpr_gets_every_cvpr_recommended_lane(name: str) -> None:
    recommended = {entry.lane_id for entry in lane_catalog.lanes_recommended_for(name)}
    assert {
        "public-program-items",
        "organizer-channel-talks",
        "priority-talk-gap-search",
        "cached-slideslive-focused-recovery",
        "registration-media",
        "event-media",
        "cvf-workshop-owner",
    } <= recommended
    assert not any(entry.module.endswith("virtual_site_crawl") for entry in lane_catalog.lanes_recommended_for(name))


def test_a_new_icml_gets_every_icml_recommended_lane() -> None:
    recommended = {entry.lane_id for entry in lane_catalog.lanes_recommended_for("ICML 2027")}
    assert {
        "openreview-bearer-auth",
        "slideslive-carry-forward",
        "openreview-workshops-first",
        "openreview-accepted-papers",
        "virtual-site-item-page-expansion",
        "virtual-site-media-family-coverage",
    } <= recommended


def test_an_unknown_conference_family_gets_no_recommendation_but_may_fit_hints() -> None:
    assert lane_catalog.lanes_recommended_for("Wombat 2031") == []
    assert lane_catalog.lanes_that_may_fit("Wombat 2031") == []
    iccv = {entry.lane_id for entry in lane_catalog.lanes_that_may_fit("ICCV 2027")}
    assert "public-program-items" in iccv and "organizer-channel-talks" in iccv


def test_catalog_cli_prints_markdown_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert lane_catalog.main(["--conference", "CVPR 2027"]) == 0
    assert "Lanes recommended for CVPR" in capsys.readouterr().out
    assert lane_catalog.main(["--conference", "Wombat 2031"]) == 0
    assert "No lane is recommended" in capsys.readouterr().out
    assert lane_catalog.main(["--format", "json", "--conference", "icml2027"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "ICML" and payload["recommended"]


def test_share_audit_counts_the_lane_modules_as_shared_components() -> None:
    resolver = ComponentResolver(SRC_ROOT)
    for name in (*CVPR_LANE_MODULES, *ICML_LANE_MODULES):
        assert resolver.is_shared(f"conference_pipeline.acquisition.{name}"), name
    assert resolver.is_shared("conference_pipeline.production.download_corpus_stage")
    for old in COMPATIBILITY_ALIASES:
        # The thin aliases are not themselves annotated: runners import the new names.
        assert not resolver.is_shared(f"conference_pipeline.acquisition.{old}")
