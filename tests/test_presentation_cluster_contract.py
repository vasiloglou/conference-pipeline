import json
import pathlib

from conference_pipeline.presentation.cluster_contract import (
    validate_cluster_to_slide_breaks,
    validate_missing_section_membership,
    validate_orphaned_references,
    validate_section_spine,
    validate_stable_reference_ids,
)

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "presentations" / "phase16"


def _load(filename: str) -> dict:
    return json.loads((_FIXTURES / filename).read_text())


def test_cluster_blocks_orphaned_references():
    fixture = _load("material_clusters.json")
    refs = fixture["clusters"]  # clusters act as refs — each has stable_ref_id
    valid_clusters = [{"reference_set": fixture["section_membership"]}]
    result = validate_orphaned_references(refs, valid_clusters)
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "cluster_orphaned_reference" in codes


def test_cluster_blocks_missing_section_membership():
    clusters = [{"cluster_id": "CLU-X", "section_id": "SEC-NONEXISTENT"}]
    sections = [{"section_id": "SEC-1"}]
    result = validate_missing_section_membership(clusters, sections)
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "cluster_missing_section_membership" in codes


def test_cluster_blocks_slide_breaks():
    clusters = [{"cluster_id": "CLU-MULTI", "multi_slide": True, "slide_boundary": None}]
    result = validate_cluster_to_slide_breaks(clusters)
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "cluster_missing_slide_break" in codes


def test_section_spine_requires_all_fields():
    fixture = _load("section_spine.json")
    result = validate_section_spine(fixture["sections"])
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "section_spine_missing_field" in codes


def test_stable_reference_ids_rejects_duplicates():
    fixture = _load("stable_references.json")
    result = validate_stable_reference_ids(fixture["references"])
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "cluster_duplicate_stable_ref_id" in codes
