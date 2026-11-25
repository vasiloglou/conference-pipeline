from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from conference_pipeline.source_discovery import SafeSourceFetcher, canonical_json_digest

from rederive_source_discovery_authors import (  # noqa: E402
    RederiveError,
    assert_identity_preserved,
    main,
    run_rederive,
)

_TWO_PAPER_HTML_BODY = """
    <dt class="ptitle"><a href="/content/CVPR2026/html/Alpha_CVPR_2026_paper.html">Paper Alpha</a></dt>
    <dd>
        <form id="form-a1" action="/CVPR2026" method="post" class="authsearch">
        <input type="hidden" name="query_author" value="Zhengxian Yang">
        <a href="#" onclick="document.getElementById('form-a1').submit();">Zhengxian Yang</a>,
        </form>
        <form id="form-a2" action="/CVPR2026" method="post" class="authsearch">
        <input type="hidden" name="query_author" value="Wei Chen">
        <a href="#" onclick="document.getElementById('form-a2').submit();">Wei Chen</a>
        </form>
        [<a href="/content/CVPR2026/papers/Alpha_CVPR_2026_paper.pdf">pdf</a>]
    </dd>
    <dt class="ptitle"><a href="/content/CVPR2026/html/Beta_CVPR_2026_paper.html">Paper Beta</a></dt>
    <dd>
        <form id="form-b1" action="/CVPR2026" method="post" class="authsearch">
        <input type="hidden" name="query_author" value="Ada Lovelace">
        <a href="#" onclick="document.getElementById('form-b1').submit();">Ada Lovelace</a>
        </form>
        [<a href="/content/CVPR2026/papers/Beta_CVPR_2026_paper.pdf">pdf</a>]
    </dd>
""".encode("utf-8")

_ALPHA_SOURCE_RECORD_ID = "cvpr2026:source:cvf-openaccess-day-2026-06-05:alpha"
_BETA_SOURCE_RECORD_ID = "cvpr2026:source:cvf-openaccess-day-2026-06-05:beta"


def _fixture_source_records() -> list[dict[str, object]]:
    return [
        {
            "source_record_id": _ALPHA_SOURCE_RECORD_ID,
            "native_ids": ["native:paper:Alpha_CVPR_2026_paper"],
            "aliases": ["https://openaccess.thecvf.com/content/CVPR2026/papers/Alpha_CVPR_2026_paper.pdf"],
            "family_id": "cvf-openaccess-day-2026-06-05",
            "snapshot_sha256": "a" * 64,
            "title": "Paper Alpha",
        },
        {
            "source_record_id": _BETA_SOURCE_RECORD_ID,
            "native_ids": ["native:paper:Beta_CVPR_2026_paper"],
            "aliases": ["https://openaccess.thecvf.com/content/CVPR2026/papers/Beta_CVPR_2026_paper.pdf"],
            "family_id": "cvf-openaccess-day-2026-06-05",
            "snapshot_sha256": "b" * 64,
            "title": "Paper Beta",
        },
    ]


def _fixture_program_items(source_records: list[dict[str, object]]) -> list[dict[str, object]]:
    items = []
    for record in source_records:
        items.append(
            {
                "program_item_id": f"cvpr2026:program-item:{record['source_record_id']}",
                "aliases": list(record["aliases"]),
                "source_record_ids": [record["source_record_id"]],
                "field_evidence": {
                    "title": {
                        "authority": record["family_id"],
                        "source_record_id": record["source_record_id"],
                        "snapshot_sha256": record["snapshot_sha256"],
                    }
                },
            }
        )
    return items


def _write_fixture_workspace(tmp_path: Path, *, run_id: str = "fixture-run") -> Path:
    """A tmp_path workspace with one cached CVF day-listing body and a
    matching, pre-authors inventory -- exactly the shape a frozen corpus
    predating the extractor change would have."""

    workspace = tmp_path / "workspace"
    cache_dir = (
        workspace / "cache" / "source-discovery" / run_id / "cvf-openaccess-day-2026-06-05" / "snap-1"
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / "body.bin").write_bytes(_TWO_PAPER_HTML_BODY)

    source_records = _fixture_source_records()
    program_items = _fixture_program_items(source_records)
    inventory: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_id": "source-discovery",
        "status": "prepared",
        "blockers": [],
        "generated_at": "2026-07-31T00:00:00+00:00",
        "source_records": source_records,
        "program_items": program_items,
    }
    digest_payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = canonical_json_digest(digest_payload)

    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workspace


def _read_inventory(workspace: Path) -> dict[str, object]:
    path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rederive_adds_authors_and_symmetric_program_item_field_evidence(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)

    summary = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=False)

    assert summary["records_authors_added"] == 2
    assert summary["program_items_authors_added"] == 2
    inventory = _read_inventory(workspace)
    records_by_id = {r["source_record_id"]: r for r in inventory["source_records"]}
    assert records_by_id[_ALPHA_SOURCE_RECORD_ID]["authors"] == ["Zhengxian Yang", "Wei Chen"]
    assert records_by_id[_BETA_SOURCE_RECORD_ID]["authors"] == ["Ada Lovelace"]
    items_by_source_id = {
        item["source_record_ids"][0]: item for item in inventory["program_items"]
    }
    alpha_evidence = items_by_source_id[_ALPHA_SOURCE_RECORD_ID]["field_evidence"]
    assert set(alpha_evidence["authors"]) == {"authority", "source_record_id", "snapshot_sha256"}
    assert alpha_evidence["authors"]["source_record_id"] == _ALPHA_SOURCE_RECORD_ID


def test_record_ids_and_snapshot_digests_are_stable_across_the_run(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)
    pre_ids = {r["source_record_id"] for r in _fixture_source_records()}
    pre_digests = {r["snapshot_sha256"] for r in _fixture_source_records()}

    summary = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=False)

    assert summary["record_ids_stable"] is True
    assert summary["snapshot_digests_stable"] is True
    inventory = _read_inventory(workspace)
    post_ids = {r["source_record_id"] for r in inventory["source_records"]}
    post_digests = {r["snapshot_sha256"] for r in inventory["source_records"]}
    assert post_ids == pre_ids
    assert post_digests == pre_digests


def test_zero_network_calls_even_if_the_fetcher_is_monkeypatched_to_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof of the zero-network claim: break the one network entry
    point this codebase's source-discovery layer uses, and confirm the
    rederive command still succeeds, because it never calls it."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("rederive_source_discovery_authors must never fetch")

    monkeypatch.setattr(SafeSourceFetcher, "fetch", _raise)
    workspace = _write_fixture_workspace(tmp_path)

    summary = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=False)

    assert summary["network_calls"] == 0
    assert summary["records_authors_added"] == 2


def test_dry_run_reports_counts_and_writes_nothing(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)
    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    pre_bytes = inventory_path.read_bytes()

    summary = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=True)

    assert summary["dry_run"] is True
    assert summary["records_authors_added"] == 2
    assert summary["program_items_authors_added"] == 2
    assert summary["wrote_file"] is False
    post_bytes = inventory_path.read_bytes()
    assert post_bytes == pre_bytes
    import hashlib

    assert hashlib.sha256(pre_bytes).hexdigest() == hashlib.sha256(post_bytes).hexdigest()


def test_second_run_is_idempotent_and_leaves_the_file_byte_identical(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)
    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"

    first = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=False)
    assert first["records_authors_added"] == 2
    after_first = inventory_path.read_bytes()

    second = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=False)

    assert second["records_authors_added"] == 0
    assert second["program_items_authors_added"] == 0
    assert second["wrote_file"] is False
    after_second = inventory_path.read_bytes()
    assert after_second == after_first


def test_cache_symlink_escape_is_refused_before_any_read(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "cvf-openaccess-day-2026-06-05").mkdir()

    run_id = "escape-run"
    escape_target = workspace / "cache" / "source-discovery" / run_id
    escape_target.symlink_to(outside)

    with pytest.raises(RederiveError) as excinfo:
        run_rederive(workspace=workspace, run_id=run_id, dry_run=True)

    assert excinfo.value.code == "rederive_path_escapes_workspace"


def test_json_summary_includes_all_required_keys_and_downstream_bindings(tmp_path: Path) -> None:
    workspace = _write_fixture_workspace(tmp_path)

    summary = run_rederive(workspace=workspace, run_id="fixture-run", dry_run=True)

    for key in (
        "records_scanned",
        "records_authors_added",
        "program_items_authors_added",
        "record_ids_stable",
        "snapshot_digests_stable",
        "network_calls",
        "pre_inventory_sha256",
        "post_inventory_sha256",
        "downstream_digest_bindings",
    ):
        assert key in summary, key
    assert summary["network_calls"] == 0
    # The real production runner pins the whole inventory's own digest
    # (PINNED_CORPUS_SOURCE_INVENTORY_SHA256) -- this scan must find that
    # binding for real, not assume it away.
    assert summary["downstream_digest_bindings"], (
        "expected at least one inventory-digest binding in "
        "scripts/run_cvpr2026_production.py"
    )


def test_assert_identity_preserved_raises_when_the_id_set_moves() -> None:
    pre = [{"source_record_id": "a", "snapshot_sha256": "1" * 64}]
    post = [{"source_record_id": "b", "snapshot_sha256": "1" * 64}]

    with pytest.raises(RederiveError) as excinfo:
        assert_identity_preserved(pre, post)

    assert excinfo.value.code == "rederive_identity_violation"


def test_identity_violation_blocks_the_write_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Simulate the corrupted-identity case end-to-end through main(): the
    guard fires before any write, so the file on disk must be untouched and
    the process must exit non-zero with the named error code."""

    workspace = _write_fixture_workspace(tmp_path)
    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    pre_bytes = inventory_path.read_bytes()

    import rederive_source_discovery_authors as rederive_module

    def _always_violates(_pre: object, _post: object) -> None:
        raise RederiveError("rederive_identity_violation", "simulated identity drift")

    monkeypatch.setattr(rederive_module, "assert_identity_preserved", _always_violates)

    exit_code = main(
        ["--workspace", str(workspace), "--run-id", "fixture-run", "--json"]
    )

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["code"] == "rederive_identity_violation"
    assert inventory_path.read_bytes() == pre_bytes


def test_dry_run_cli_invocation_against_the_real_cvpr2026_workspace_is_network_free() -> None:
    """A lighter-weight in-process check mirroring the plan's required real
    ``--dry-run`` invocation against the actual CVPR2026 workspace, so this
    exact behavior is covered by the test suite (not only the manual verify
    command)."""

    workspace = REPO_ROOT / "conferences" / "cvpr2026"
    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    if not inventory_path.is_file():
        pytest.skip("real CVPR2026 inventory not present in this checkout")
    pre_bytes = inventory_path.read_bytes()

    summary = run_rederive(
        workspace=workspace,
        run_id="cvpr2026-live-production-r10-main-paper-corpus",
        dry_run=True,
    )

    assert summary["network_calls"] == 0
    assert summary["record_ids_stable"] is True
    assert summary["snapshot_digests_stable"] is True
    assert inventory_path.read_bytes() == pre_bytes
