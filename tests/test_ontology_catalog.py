"""v6.1 Phase 2: the sharded reference catalog, checked by GND (spec 6.4, tests T14-T16)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conference_pipeline.ontology import catalog as cat
from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology import work_families as wf
from conference_pipeline.ontology.gnd_tool import CommandFamily
from tests._ontology_fixtures import aid, build_workspace


def _gnd_ready() -> str | None:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable as exc:
        return str(exc)
    return None


needs_gnd = pytest.mark.skipif(_gnd_ready() is not None, reason=f"gnd {gnd_tool.GND_VERSION} unavailable")


@pytest.fixture
def roster(tmp_path):
    return wf.build_roster(build_workspace(tmp_path / "ws"), run_id="t")


def _validate(model: Path):
    return gnd_tool.run_gnd(["validate", str(model)], family=CommandFamily.CHECK, cwd=model.parent)


def _rows(index, out, fact_type):
    rows = []
    for shard in index["shards"]:
        rows += json.loads((out / shard["shard"] / cat.REFERENCE_DIR / f"{fact_type}.json").read_text())["facts"][fact_type]
    return rows


def test_population_states_the_program_and_companions_carry_their_works_authors(roster, tmp_path):
    index = cat.build_catalog(roster, tmp_path / "c")
    out = tmp_path / "c"
    authors = [r for r in _rows(index, out, "work_has_author_at_position") if r["work"] == "openreview:FORUMAAAA1"]
    assert authors == [{"work": "openreview:FORUMAAAA1", "author": "Ann Alpha", "position": 1},
                       {"work": "openreview:FORUMAAAA1", "author": "Bo Beta", "position": 2}]
    versions = {r["version"]: r for r in _rows(index, out, "work_has_version")}
    assert versions[aid("paper_text:p01")] == {"work": "openreview:FORUMAAAA1", "version": aid("paper_text:p01"),
                                               "kind": "paper_text:slides:poster"}
    sessions = _rows(index, out, "work_presented_in_session")
    assert {"work": "openreview:FORUMDDDD4", "session": "event:session-900"} in sessions
    assert index["works"] == len(roster["families"]) and index["shard_count"] == 1


def test_a_subset_catalog_holds_only_the_named_works(roster, tmp_path):
    index = cat.build_catalog(roster, tmp_path / "c", works=frozenset({"openreview:FORUMAAAA1"}))
    titles = _rows(index, tmp_path / "c", "work_has_title")
    assert [r["work"] for r in titles] == ["openreview:FORUMAAAA1"] and index["subset"] is True


def test_the_catalog_shards_when_a_limit_would_be_crossed(roster, tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "GND_REFERENCE_MAX_ENTRIES", 30)
    index = cat.build_catalog(roster, tmp_path / "c")
    assert index["shard_count"] > 1
    assert all(s["entries"] <= 30 * cat.SHARD_HEADROOM for s in index["shards"])
    assert sum(s["works"] for s in index["shards"]) == len(roster["families"])
    for shard in index["shards"]:
        for row in _rows({"shards": [shard]}, tmp_path / "c", "work_has_kind"):
            assert cat.shard_of(row["work"], index["shard_count"]) == int(shard["shard"].split("-")[1])


def test_a_work_too_large_for_any_shard_is_refused_before_writing(roster, tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "GND_REFERENCE_MAX_ENTRIES", 2)
    with pytest.raises(cat.CatalogTooLargeError):
        cat.build_catalog(roster, tmp_path / "c")
    assert not (tmp_path / "c").exists()


# T14
@needs_gnd
def test_every_catalog_shard_validates(roster, tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "GND_REFERENCE_MAX_ENTRIES", 30)
    index = cat.build_catalog(roster, tmp_path / "c")
    for shard in index["shards"]:
        result = _validate(tmp_path / "c" / shard["model"])
        assert result.exit_code == 0, result.stdout + result.stderr


_SIDECAR = """
import hashlib, json, sys
from gnd.anchors import vocabulary_digest
model, population, document, out = sys.argv[1:5]
m = json.load(open(model))
sidecar = {"anchors": [], "schema_version": 2, "vocabulary_digest": vocabulary_digest(m),
           "population_sha256": hashlib.sha256(open(population, "rb").read()).hexdigest(),
           "document_sha256": hashlib.sha256(open(document, "rb").read()).hexdigest()}
open(out, "w").write(json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\\n")
"""


def _one_source(root: Path, model: Path) -> Path:
    """A composition directory with one empty anchored source for ``model``."""
    sources = root / "sources"
    (sources / "s1").mkdir(parents=True)
    (sources / "manifest.json").write_text(json.dumps({"schema": "gnd-composition-manifest/1", "sources": ["s1"],
                                                       "renamings": [], "deduplications": "identical-rows",
                                                       "scope_tags": []}))
    population = sources / "s1" / "population.json"
    population.write_text(json.dumps({"facts": {}, "population": {"covers": [], "model": model.name,
                                                                   "source": "agent-output"}}))
    (sources / "s1" / "document.txt").write_text("A source with no facts.\n")
    result = gnd_tool.run_gnd_python(["-I", "-c", _SIDECAR, str(model), str(population),
                                      str(sources / "s1" / "document.txt"), str(population) + ".anchors.json"],
                                     family=CommandFamily.OTHER, cwd=root)
    assert result.exit_code == 0, result.stderr
    return sources


# T15
@needs_gnd
def test_gnd_compose_loads_the_catalog_and_binds_its_digests(roster, tmp_path):
    index = cat.build_catalog(roster, tmp_path / "c")
    model = tmp_path / "c" / index["shards"][0]["model"]
    sources = _one_source(tmp_path / "compose", model)
    record = tmp_path / "compose" / "record.json"
    result = gnd_tool.run_gnd(["compose", str(model), str(sources), "--out", str(tmp_path / "compose" / "out.json"),
                               "--record", str(record)], family=CommandFamily.OTHER, cwd=tmp_path)
    assert result.exit_code == 0, result.stdout + result.stderr
    bound = {r["id"]: r["raw_sha256"] for r in json.loads(record.read_text())["references"]}
    assert bound == {ft: p["sha256"] for ft, p in index["shards"][0]["populations"].items()}


# T16
@needs_gnd
@pytest.mark.parametrize("command", ["validate", "compose"])
def test_one_changed_catalog_byte_is_refused_at_exit_3(roster, tmp_path, command):
    index = cat.build_catalog(roster, tmp_path / "c")
    model = tmp_path / "c" / index["shards"][0]["model"]
    population = model.parent / cat.REFERENCE_DIR / "work_has_title.json"
    raw = population.read_bytes()
    population.write_bytes(raw.replace(b"Alpha", b"Alpho", 1))
    assert hashlib.sha256(population.read_bytes()).hexdigest() != index["shards"][0]["populations"]["work_has_title"]["sha256"]
    if command == "validate":
        result = _validate(model)
    else:
        sources = _one_source(tmp_path / "compose", model)
        result = gnd_tool.run_gnd(["compose", str(model), str(sources), "--out", str(tmp_path / "o.json"),
                                   "--record", str(tmp_path / "r.json")], family=CommandFamily.OTHER, cwd=tmp_path)
    assert result.exit_code == 3
    assert "reference raw hash does not match declared sha256" in result.stdout + result.stderr
