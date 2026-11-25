"""v6.1 Phase 2: the seed vocabulary (spec 6.5, tests T17-T18)."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest
import yaml

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily

SEED = resources.files("conference_pipeline.ontology") / "vocabulary" / "seed-v1.orm.yaml"


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


needs_gnd = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")


def _seed() -> dict:
    return yaml.safe_load(SEED.read_text(encoding="utf-8"))


def _constraints(model):
    yield from model.get("constraints") or []
    for fact_type in model.get("fact_types") or []:
        yield from fact_type.get("constraints") or []


def test_the_seed_is_includable_and_holds_only_identity_and_assertions():
    model = _seed()
    assert model["version"] <= 4
    assert not model.get("operations") and not model.get("rejected_proposals")
    assert model["source"]["mode"] != "unattended" and not model.get("includes")
    assert not model.get("reference_populations")
    names = {o["name"] for o in model["object_types"]}
    assert {"WorkKey", "VersionId", "PassageId", "AssertionId", "Predicate", "Polarity", "Attribution"} <= names
    assert not names & {"Method", "Dataset", "Metric", "Result", "Benchmark"}


def test_every_seed_constraint_carries_the_owners_ruling():
    for constraint in _constraints(_seed()):
        assert (constraint["source"], constraint["status"], constraint["modality"]) == (
            "explicit", "confirmed", "alethic"), constraint["id"]
        assert constraint["provenance"]["user_quote"] and constraint["provenance"]["reviewed_in"]
        assert constraint["rationale"]


# T17
@needs_gnd
def test_gnd_validate_accepts_the_seed(tmp_path):
    model = tmp_path / "seed.orm.yaml"
    model.write_bytes(SEED.read_bytes())
    result = gnd_tool.run_gnd(["validate", str(model)], family=CommandFamily.CHECK, cwd=tmp_path)
    assert result.exit_code == 0, result.stdout + result.stderr


# T18
@needs_gnd
def test_a_version_5_root_can_include_the_seed(tmp_path):
    seed = tmp_path / "vocabulary" / "seed.orm.yaml"
    seed.parent.mkdir()
    seed.write_bytes(SEED.read_bytes())
    root = {"version": 5, "model_id": "urn:test:root@1", "source": _seed()["source"],
            "object_types": [], "fact_types": [], "constraints": [],
            "includes": [{"id": "seed", "path": "vocabulary/seed.orm.yaml",
                          "sha256": hashlib.sha256(seed.read_bytes()).hexdigest()}]}
    (tmp_path / "root.orm.json").write_text(json.dumps(root))
    result = gnd_tool.run_gnd(["validate", str(tmp_path / "root.orm.json")], family=CommandFamily.CHECK, cwd=tmp_path)
    assert result.exit_code == 0, result.stdout + result.stderr
    seed.write_bytes(SEED.read_bytes() + b"# changed\n")
    stale = gnd_tool.run_gnd(["validate", str(tmp_path / "root.orm.json")], family=CommandFamily.CHECK, cwd=tmp_path)
    assert stale.exit_code == 3
