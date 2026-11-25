from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "refresh_analysis_manifest_digests.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("refresh_analysis_manifest_digests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _workspace(tmp_path: Path, *, lineage: dict | None) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    topic = workspace / "analysis" / "topics" / "01.md"
    topic.parent.mkdir(parents=True, exist_ok=True)
    topic.write_text("# corrected topic\n", encoding="utf-8")
    _write_json(
        workspace / "analysis" / "analysis_manifest.json",
        {
            "content_sha256": "0" * 64,
            "topics": [{"id": "01", "markdown_path": "analysis/topics/01.md", "markdown_sha256": "1" * 64}],
        },
    )
    receipt = {
        "stage_id": "analysis",
        "stage_order": 7,
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": "r" * 64,
        "required_script_ids": [],
        "owner_execution_evidence": {},
        "checks": {
            "analysis_manifest_path": "analysis/analysis_manifest.json",
            "analysis_manifest_content_sha256": "0" * 64,
        },
        "output_digest": "old-output-digest",
    }
    if lineage is not None:
        receipt["lineage"] = lineage
    stage07 = workspace / "storage" / "workflow-runs" / "run-1" / "stage-07-analysis.json"
    _write_json(stage07, receipt)
    return workspace, stage07


def test_refresh_carries_recorded_lineage_and_updates_output_digest(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    recorded = {"schema_version": 1, "upstream_receipts": {}, "input_artifacts": {}, "lineage_digest": "L"}
    workspace, stage07 = _workspace(tmp_path, lineage=recorded)
    seen: list[dict] = []

    def fake_verify(**kwargs):
        seen.append(kwargs["receipt"])
        return SimpleNamespace(to_dict=lambda: {"status": "pass"})

    monkeypatch.setattr(module, "verify_stage_contract", fake_verify)
    assert module.main(["--workspace", str(workspace), "--run-id", "run-1"]) == 0

    verified = seen[0]
    assert verified["lineage"] == recorded  # carried through, never re-bound
    assert verified["output_digest"] and verified["output_digest"] != "old-output-digest"
    written = json.loads(stage07.read_text(encoding="utf-8"))
    assert written["lineage"] == recorded
    assert written["output_digest"] == verified["output_digest"]
    assert written["checks"]["analysis_manifest_content_sha256"] != "0" * 64
    copies = list(stage07.parent.glob("stage-07-analysis.superseded-*.json"))
    assert len(copies) == 1
    assert json.loads(copies[0].read_text(encoding="utf-8"))["output_digest"] == "old-output-digest"


def test_refresh_refuses_a_receipt_without_lineage_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    workspace, stage07 = _workspace(tmp_path, lineage=None)
    manifest = workspace / "analysis" / "analysis_manifest.json"
    before = (manifest.read_bytes(), stage07.read_bytes())
    monkeypatch.setattr(module, "verify_stage_contract", lambda **_: pytest.fail("must not verify"))
    assert module.main(["--workspace", str(workspace), "--run-id", "run-1"]) == 1
    assert (manifest.read_bytes(), stage07.read_bytes()) == before
