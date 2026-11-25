from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import model_catalog_upgrade as upgrade  # noqa: E402


@pytest.mark.parametrize(
    ("model_id", "family", "version", "stable"),
    [
        ("claude-opus-4-8", "claude-opus", (4, 8), True),
        ("claude-sonnet-5", "claude-sonnet", (5,), True),
        ("gemini-3.1-pro-preview", "gemini-pro", (3, 1), False),
        ("gpt-5.6-sol", "gpt-sol", (5, 6), True),
        ("gpt-4o-mini-tts", "gpt-o-mini-tts", (4,), True),
        ("o3-deep-research-2025-06-26", "o-deep-research", (3,), True),
        ("openrouter/anthropic/claude-sonnet-5", "openrouter/anthropic/claude-sonnet", (5,), True),
    ],
)
def test_model_shape_splits_family_from_version(model_id, family, version, stable) -> None:
    shape = upgrade.model_shape(model_id)
    assert (shape.family, shape.version, shape.stable) == (family, version, stable)


def test_newer_means_same_family_higher_version_or_ga_of_a_preview() -> None:
    newer = lambda a, b: upgrade.is_newer(upgrade.model_shape(a), upgrade.model_shape(b))  # noqa: E731
    assert newer("claude-opus-5-5", "claude-opus-4-8")
    assert not newer("claude-sonnet-5", "claude-opus-4-8")  # another tier is not an upgrade
    assert newer("gemini-3.1-pro", "gemini-3.1-pro-preview")  # GA of the same preview
    assert not newer("gemini-3.2-pro-preview", "gemini-3.1-pro")  # never GA -> preview
    assert not newer("claude-opus-5-5-20260901", "claude-opus-4-8")  # dated snapshot
    assert not newer("gpt-5.6-terra", "gpt-5.6-sol")


@pytest.fixture()
def governed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tool at copies of the real governed files."""

    for name, source in {
        "models.yaml": upgrade.MODELS_YAML,
        "registry.yaml": upgrade.REGISTRY_YAML,
        "task_specs.yaml": upgrade.TASK_SPECS_YAML,
        "pricing_catalog.json": upgrade.PRICING_JSON,
    }.items():
        shutil.copy(source, tmp_path / name)
    monkeypatch.setattr(upgrade, "MODELS_YAML", tmp_path / "models.yaml")
    monkeypatch.setattr(upgrade, "REGISTRY_YAML", tmp_path / "registry.yaml")
    monkeypatch.setattr(upgrade, "TASK_SPECS_YAML", tmp_path / "task_specs.yaml")
    monkeypatch.setattr(upgrade, "PRICING_JSON", tmp_path / "pricing_catalog.json")
    monkeypatch.setattr(upgrade, "LITERAL_SCAN_DIRS", ())
    return tmp_path


def _opus(governed: Path) -> str:
    return yaml.safe_load((governed / "models.yaml").read_text())["models"]["opus"]["model"]


def test_propose_finds_the_newer_tier_member_and_its_blast_radius(governed: Path) -> None:
    old = _opus(governed)
    discovered = {"providers": {"anthropic": [{"id": old}, {"id": "claude-opus-9-9"}, {"id": "claude-sonnet-9"}]}}
    proposals, _ = upgrade.build_proposals(discovered, include_legacy=False)
    opus = next(p for p in proposals if p.alias == "opus")
    assert opus.candidate == "claude-opus-9-9"
    assert opus.tasks_default and opus.registry_literals > 0
    assert opus.candidate_price is None  # the tool must say so, not assume $0


def test_apply_refuses_an_unpriced_model(governed: Path) -> None:
    with pytest.raises(SystemExit, match="no price on file"):
        upgrade.main(["apply", "--alias", "opus", "--to", "claude-opus-9-9"])
    assert _opus(governed) != "claude-opus-9-9"


def test_apply_moves_the_alias_retires_the_old_model_and_prices_the_new(governed: Path, monkeypatch) -> None:
    old = _opus(governed)
    # The post-apply load_registry() check reads the real package files; skip it here.
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})())
    assert upgrade.main([
        "apply", "--alias", "opus", "--to", "claude-opus-9-9",
        "--input-price", "5", "--output-price", "25", "--price-source", "anthropic_pricing",
    ]) == 0
    models = yaml.safe_load((governed / "models.yaml").read_text())["models"]
    assert models["opus"]["model"] == "claude-opus-9-9"
    retired = [e for e in models.values() if e["model"] == old]
    assert retired and all(e["status"] == "retired" for e in retired)
    registry_text = (governed / "registry.yaml").read_text()
    assert upgrade._bounded(old).search(registry_text) is None
    assert "claude-opus-9-9" in registry_text
    pricing = json.loads((governed / "pricing_catalog.json").read_text())
    assert pricing["providers"]["anthropic"]["models"]["claude-opus-9-9"]["input_per_1m_usd"] == 5.0
    # Comments in the catalog survive the edit.
    assert "Single-source-of-truth catalog" in (governed / "models.yaml").read_text()


def test_dry_run_changes_nothing(governed: Path) -> None:
    before = {p.name: p.read_text() for p in governed.iterdir()}
    upgrade.main([
        "apply", "--alias", "opus", "--to", "claude-opus-9-9", "--dry-run",
        "--input-price", "5", "--output-price", "25", "--price-source", "anthropic_pricing",
    ])
    assert {p.name: p.read_text() for p in governed.iterdir()} == before


def test_repoint_moves_tasks_onto_the_alias_and_retires_the_old_one(governed: Path, monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "_run_load_registry", lambda: 0)
    # A throwaway alias (independent of the live catalog's state) owning one task.
    models = governed / "models.yaml"
    models.write_text(models.read_text().rstrip("\n") + "\n  tmp_judge:\n    provider: gemini\n"
                      "    model: gemini-9.9-pro-preview\n    status: active\n")
    registry = governed / "registry.yaml"
    registry.write_text(registry.read_text().replace(
        "  presentation.slide_reflow:\n", "  presentation.slide_reflow:\n    # test\n", 1))
    text = registry.read_text()
    start = text.index("  presentation.slide_reflow:")
    ref = text.index("model_ref:", start)
    line_end = text.index("\n", ref)
    registry.write_text(text[:ref] + "model_ref: tmp_judge" + text[line_end:])
    new_model = yaml.safe_load(models.read_text())["models"]["gemini_pro"]["model"]

    assert upgrade.main(["repoint", "--alias", "tmp_judge", "--into", "gemini_pro"]) == 0
    assert yaml.safe_load(models.read_text())["models"]["tmp_judge"]["status"] == "retired"
    tasks = yaml.safe_load(registry.read_text())["tasks"]
    assert tasks["presentation.slide_reflow"]["model_ref"] == "gemini_pro"
    assert new_model in registry.read_text()


def test_repoint_refuses_to_change_the_provider_route(governed: Path) -> None:
    with pytest.raises(SystemExit, match="provider route"):
        upgrade.main(["repoint", "--alias", "gpt5_mini_openrouter", "--into", "sonnet"])


def test_price_refreshes_an_existing_entry(governed: Path) -> None:
    upgrade.main(["price", "--provider", "anthropic", "--model", "claude-opus-9-9",
                  "--input-price", "4", "--output-price", "20", "--price-source", "anthropic_pricing"])
    entry = json.loads((governed / "pricing_catalog.json").read_text())["providers"]["anthropic"]["models"]["claude-opus-9-9"]
    assert (entry["input_per_1m_usd"], entry["output_per_1m_usd"], entry["source"]) == (4.0, 20.0, "anthropic_pricing")


def test_env_overrides_report_keys_pinning_a_model(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-secret\nOCR_MODEL=old-model-1\nexport VIDEO_DETECTION_MODEL='old-model-1'\nOTHER=old-model-1\n")
    assert upgrade.env_overrides("old-model-1", env) == ["OCR_MODEL", "VIDEO_DETECTION_MODEL"]
