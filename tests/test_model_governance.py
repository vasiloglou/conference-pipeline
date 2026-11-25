from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conference_pipeline.config import PipelineSettings, load_settings
from conference_pipeline.benchmarks.receipt_schema import BenchmarkReceipt, receipt_to_dict
from conference_pipeline.model_governance import (
    LocalCandidateRecord,
    ModelGovernanceError,
    ModelRegistry,
    ModelSelection,
    approved_local_selection,
    default_model,
    default_model_for_provider,
    default_provider,
    load_model_catalog,
    load_registry,
    validate_selection,
)
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile, load_profiles


REPO_ROOT = Path(__file__).resolve().parent.parent
INVENTORY = REPO_ROOT / "src" / "conference_pipeline" / "model_governance" / "callsite_inventory.json"
EXCEPTIONS = REPO_ROOT / "src" / "conference_pipeline" / "model_governance" / "migration_exceptions.yaml"
REGISTRY = REPO_ROOT / "src" / "conference_pipeline" / "model_governance" / "registry.yaml"
AGENTIC_SLIDES = REPO_ROOT / "scripts" / "generate_slides_agentic.py"
ANALYSIS_SCAFFOLD = REPO_ROOT / ".claude" / "skills" / "plan-conference-analysis" / "scripts" / "scaffold_plan.py"
PRODUCTION_SCAFFOLD = (
    REPO_ROOT
    / ".claude"
    / "skills"
    / "plan-conference-production"
    / "scripts"
    / "scaffold_production_plan.py"
)
MODEL_LITERAL_RE = re.compile(
    r"\b(?:"
    r"gpt-\d(?:\.\d+)?-[a-z0-9][a-z0-9.-]*|"
    r"gemini-\d(?:\.\d+)?-[a-z0-9][a-z0-9.-]*|"
    r"claude-[a-z0-9.-]+-\d{4,}|"
    r"o3-[a-z0-9.-]+-\d{4}-\d{2}-\d{2}|"
    r"openrouter/(?:anthropic|google|openai)/[a-z0-9.-]+"
    r")\b"
)
SCAN_ROOTS = (REPO_ROOT / "src" / "conference_pipeline", REPO_ROOT / "scripts")


def test_model_registry_loads_required_task_contracts() -> None:
    registry = load_registry()

    required = {
        "transcription.session_speech",
        "ocr.slide_image_parse",
        "analysis.source_reading",
        "analysis.deep_narrative_synthesis",
        "source_reading.categorization",
        "interest_group.research",
        "podcast.script_generation",
        "podcast.tts",
        "presentation.deck_generation",
        "presentation.blueprint_generation",
        "presentation.svg_figure_generation",
        "presentation.visual_qc",
        "presentation.repair",
        "vision.figure_caption",
        "ranking.paper_reference_frequency",
        "ranking.fact_importance",
        "podcast.focused_script_generation",
        "podcast.longform_script_generation",
        "synthesis.exhaustive_topic_summary",
        "vector_store.query_answer",
        "research.deep_research",
        "topic_video.scene_generation",
        "qa.paper_content_verification",
        "presentation.audio_sync",
        "source_discovery.paper_source_ranking",
        "source_discovery.cross_source_join",
        "analysis.workshop_segmentation",
        "transcription_ocr.workshop_talk_boundary_discovery",
        "presentation.agentic_deck_generation",
        "presentation.agentic_vision_crop",
        "presentation.presenter_script_generation",
        "presentation.presenter_dialogue_generation",
        "presentation.presenter_tts",
        "research.gpt_researcher_fast",
        "research.gpt_researcher_smart",
        "research.gemini_deep_research_fast",
        "research.gemini_deep_research_strategic",
        "research.claude_deep_research",
        "research.claude_deep_synthesis",
        "exhaustive_extraction.cost_estimation",
        "exhaustive_extraction.gpt5_mini",
        "exhaustive_extraction.gpt52_quote_fallback",
        "exhaustive_extraction.claude_sonnet",
        "exhaustive_extraction.claude_haiku",
        "exhaustive_extraction.gemini_pro",
        "exhaustive_extraction.openrouter_sonnet",
        "exhaustive_extraction.openrouter_gemini",
        "exhaustive_extraction.openrouter_gpt5_mini",
        "exhaustive_extraction.openrouter_haiku",
        "qa.repair",
        "synthesis.executive_summary",
    }
    assert required <= set(registry.tasks)

    for task_id in required:
        policy = registry.get(task_id)
        assert policy.current_research["required"] is True
        assert policy.benchmark["required"] is True
        assert policy.benchmark["receipt_required"] is True
        assert policy.provenance_required is True
        if task_id in {"research.gpt_researcher_fast", "research.gpt_researcher_smart"}:
            assert policy.local_runtime_policy.startswith("approved_for_corpus_only_gpt_researcher")
        else:
            assert policy.local_runtime_policy.startswith("deferred")


def test_source_discovery_cross_source_join_has_a_governed_receipt_bound_policy() -> None:
    policy = load_registry().get("source_discovery.cross_source_join")

    assert policy.stage == "source-discovery"
    assert policy.budget_cap_usd is not None and policy.budget_cap_usd > 0
    assert policy.current_research["required"] is True
    assert policy.current_research["status"] == "current"
    assert policy.benchmark["required"] is True
    assert policy.benchmark["receipt_required"] is True
    assert policy.schema_validity_required is True
    assert policy.provenance_required is True
    assert policy.confidence_threshold == pytest.approx(0.90)
    assert default_provider("source_discovery.cross_source_join") == policy.provider
    assert default_model("source_discovery.cross_source_join") == policy.model
    assert policy.fallback_selection is not None
    assert default_model_for_provider(
        "source_discovery.cross_source_join", policy.fallback_selection.provider
    ) == policy.fallback_selection.model


def test_pipeline_settings_defaults_are_registry_backed() -> None:
    settings = PipelineSettings()

    assert settings.ocr_provider == default_provider("ocr.slide_image_parse")
    assert settings.ocr_model == default_model("ocr.slide_image_parse")
    assert settings.openai_model == default_model("transcription.session_speech")
    assert settings.anthropic_model == default_model("analysis.source_reading")
    assert settings.video_detection_model == default_model("video.session_detection")


def test_gpt_researcher_registry_has_corpus_only_local_candidate() -> None:
    registry = load_registry()

    for task_id in ("research.gpt_researcher_fast", "research.gpt_researcher_smart"):
        policy = registry.get(task_id)
        candidates = {
            candidate.model_id: candidate for candidate in policy.local_candidates
        }
        local_candidate = candidates["mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"]
        assert local_candidate.status == "approved"
        assert local_candidate.runtime_kind == "mlx_lm"
        assert "corpus-only" in (local_candidate.rejection_reason or "")
        assert policy.default_selection.provider == "local"
        assert policy.default_selection.model == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
        assert policy.fallback_selection is not None
        assert policy.fallback_selection.provider == "openai"


def test_provider_specific_default_models_are_registered() -> None:
    assert default_model_for_provider("ocr.slide_image_parse", "gemini") == CURRENT_GENERATION["gemini_flash_lite"][1]
    assert default_model_for_provider("ocr.slide_image_parse", "anthropic") == "claude-sonnet-5"


def test_opus_is_default_for_requested_editorial_tasks() -> None:
    for task_id in (
        "podcast.script_generation",
        "podcast.focused_script_generation",
        "synthesis.executive_summary",
        "analysis.deep_dive_question_selection",
        "analysis.topic_strategy_generation",
    ):
        assert _task_refs(task_id)["model_ref"] == "opus"
        assert (default_provider(task_id), default_model(task_id)) == CURRENT_GENERATION["opus"]


def test_ocr_benchmark_authority_models_are_registered() -> None:
    registry = load_registry()
    policy = registry.get("ocr.slide_image_parse")

    for selection in policy.benchmark["authority_selections"]:
        validate_selection(
            "ocr.slide_image_parse",
            provider=selection["provider"],
            model=selection["model"],
        )


def test_registry_rejects_unregistered_benchmark_selections() -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))

    payload["tasks"]["ocr.slide_image_parse"]["benchmark"]["candidates"] = [
        "unregistered-candidate-model"
    ]
    with pytest.raises(ModelGovernanceError, match="benchmark.candidates"):
        ModelRegistry.from_mapping(payload)

    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    payload["tasks"]["ocr.slide_image_parse"]["benchmark"]["authority_selections"] = [
        {"provider": "openai", "model": "unregistered-authority-model"}
    ]
    with pytest.raises(ModelGovernanceError, match="benchmark.authority_selections"):
        ModelRegistry.from_mapping(payload)


def test_load_settings_uses_provider_specific_ocr_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "gemini")
    monkeypatch.delenv("OCR_MODEL", raising=False)

    settings = load_settings()

    assert settings.ocr_provider == "gemini"
    assert settings.ocr_model == CURRENT_GENERATION["gemini_flash_lite"][1]


def test_unregistered_model_selection_fails_closed() -> None:
    with pytest.raises(ModelGovernanceError):
        validate_selection(
            "ocr.slide_image_parse",
            provider="openai",
            model="unregistered-current-looking-model",
        )

    with pytest.raises(ModelGovernanceError):
        PipelineSettings(ocr_provider="openai", ocr_model="unregistered-model")  # type: ignore[arg-type]


def test_video_detection_provider_selection_fails_closed() -> None:
    with pytest.raises(ModelGovernanceError):
        PipelineSettings(
            video_detection_provider="anthropic",  # type: ignore[arg-type]
            video_detection_model=default_model("presentation.repair"),
        )


def test_agentic_deck_generation_rejects_unregistered_claude_model() -> None:
    spec = importlib.util.spec_from_file_location("generate_slides_agentic", AGENTIC_SLIDES)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.raises(ModelGovernanceError):
        module.call_llm("Create a deck.", default_model("presentation.repair"), max_retries=0)


def test_callsite_inventory_and_exception_allowlist_are_machine_readable() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == 1
    entries = inventory["entries"]
    assert any(entry["integration_status"] == "registry_backed" for entry in entries)
    assert any("ocr.slide_image_parse" in entry["task_ids"] for entry in entries)
    assert any("plan-conference-production" in entry["path"] for entry in entries)

    text = EXCEPTIONS.read_text(encoding="utf-8")
    assert "owner:" in text
    assert "reason:" in text
    assert "revisit:" in text


def _sample_brief() -> dict[str, object]:
    return {
        "platform_version": "v1.3",
        "conference_name": "ICLR Scaffold 2027",
        "conference_identity": {"slug": "iclr-scaffold-2027"},
        "source_urls": [
            "https://example.org/iclr-scaffold-2027/schedule",
            "https://example.org/iclr-scaffold-2027/proceedings",
        ],
        "auth_notes": "Public scaffold URLs only; no cookies or provider secrets.",
        "target_outputs": ["analysis", "QA", "podcast", "presentation production"],
        "proof_mode": "dry-run/mock only",
        "topic_strategy": "start with a small representative topic set, then expand after proof",
        "provider_preferences": "mock providers for UAT; live providers need live proof",
        "provider_cost_approval_posture": "estimate required before paid provider work",
        "storage_share_targets": "local workspace only",
        "acceptance_criteria": [
            "generated roadmap covers source intake through proof/closeout",
            "unsupported live support is labeled needs live proof",
        ],
    }


def test_production_scaffold_emitted_task_ids_match_inventory_and_registry(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(_sample_brief(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis_root = tmp_path / "analysis-drafts"
    production_root = tmp_path / "production-drafts"

    analysis_result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_SCAFFOLD),
            "--brief",
            str(brief_path),
            "--draft-root",
            str(analysis_root),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert analysis_result.returncode == 0, analysis_result.stderr + analysis_result.stdout
    spec_root = analysis_root / "iclr-scaffold-2027"

    production_result = subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCAFFOLD),
            "--spec-root",
            str(spec_root),
            "--draft-root",
            str(production_root),
            "--registry-path",
            str(REGISTRY),
            "--inventory-path",
            str(INVENTORY),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert production_result.returncode == 0, production_result.stderr + production_result.stdout
    approval = json.loads(
        (production_root / "iclr-scaffold-2027-production" / "approval-summary.json").read_text(
            encoding="utf-8"
        )
    )

    emitted_task_ids = set(approval["registry task ids"])
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in inventory["entries"]
        if item["path"] == ".claude/skills/plan-conference-production/scripts/scaffold_production_plan.py"
    )
    inventory_task_ids = set(entry["task_ids"])
    assert emitted_task_ids <= inventory_task_ids
    assert {
        "interest_group.research",
        "video.session_detection",
        "vision.figure_caption",
        "presentation.visual_qc",
    } <= inventory_task_ids

    registry = load_registry()
    for task_id in inventory_task_ids:
        registry.get(task_id)


def _joined_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _joined_string(node.left), _joined_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def test_active_python_does_not_hardcode_current_model_literals() -> None:
    """Keep active Python defaults pointed at the model-governance registry.

    A `+`-joined string is checked as one value, so splitting a model id does
    not hide it. A line carrying `# model-literal-ok: <reason>` is a reviewed,
    deliberate pin (the same marker audit_model_registry_hardcoding.py honors);
    without that exemption the only way to keep a pin was to split the literal.
    """
    offenders: list[str] = []

    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT)
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            tree = ast.parse(source, filename=str(relative))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Constant, ast.BinOp)):
                    continue
                value = _joined_string(node)
                if value is None or not MODEL_LITERAL_RE.search(value):
                    continue
                if "model-literal-ok" in lines[node.lineno - 1]:
                    continue
                offenders.append(f"{relative}:{node.lineno}: {value!r}")

    assert offenders == []


_ALLOWED_HARDWARE_FIT_STATUSES = {"unknown", "pending", "verified", "rejected"}
_ALLOWED_APPROVAL_STATUSES = {"deferred", "benchmark_pending", "approved", "blocked"}
_KNOWN_MODEL_PROVIDER_PATTERNS = re.compile(
    r"\b(?:gpt|gemini|claude|qwen|llama|phi|gemma|mistral)\b",
    re.IGNORECASE,
)


def test_local_runtime_profiles_yaml_loads() -> None:
    profiles = load_profiles()
    assert isinstance(profiles, dict)
    assert len(profiles) >= 2, "expected at least two example profiles"
    for profile_id, profile in profiles.items():
        assert isinstance(profile, LocalRuntimeProfile), f"{profile_id} is not a LocalRuntimeProfile"
        assert isinstance(profile.manage_server, bool), f"{profile_id}: manage_server must be bool"
        assert profile.hardware_fit_status in _ALLOWED_HARDWARE_FIT_STATUSES, (
            f"{profile_id}: hardware_fit_status {profile.hardware_fit_status!r} not in allowed set"
        )
        assert profile.approval_status in _ALLOWED_APPROVAL_STATUSES, (
            f"{profile_id}: approval_status {profile.approval_status!r} not in allowed set"
        )


def test_local_runtime_profile_has_required_fields() -> None:
    profiles = load_profiles()
    required_str_fields = [
        "profile_id", "runtime_kind", "base_url", "lock_scope",
        "hardware_fit_status", "approval_status",
    ]
    for profile_id, profile in profiles.items():
        for field in required_str_fields:
            value = getattr(profile, field)
            assert isinstance(value, str) and value, (
                f"{profile_id}: field {field!r} must be a non-empty string, got {value!r}"
            )
        assert isinstance(profile.optional_deps, list), (
            f"{profile_id}: optional_deps must be a list"
        )
        assert profile.startup_timeout_s > 0, f"{profile_id}: startup_timeout_s must be positive"
        assert profile.request_timeout_s > 0, f"{profile_id}: request_timeout_s must be positive"
        assert profile.max_concurrency >= 1, f"{profile_id}: max_concurrency must be >= 1"
        assert profile.max_batch_size >= 1, f"{profile_id}: max_batch_size must be >= 1"
        assert hasattr(profile, "model_id"), f"{profile_id}: model_id field must exist"
        assert profile.model_id is None or isinstance(profile.model_id, str)


def test_local_runtime_profiles_contain_no_production_model_defaults() -> None:
    profiles_path = (
        REPO_ROOT
        / "src"
        / "conference_pipeline"
        / "model_governance"
        / "local_runtime_profiles.yaml"
    )
    raw_text = profiles_path.read_text(encoding="utf-8")
    profiles = load_profiles()
    for profile_id, profile in profiles.items():
        model_id = getattr(profile, "model_id", None)
        if model_id is not None:
            assert not _KNOWN_MODEL_PROVIDER_PATTERNS.search(str(model_id)), (
                f"{profile_id}: model_id {model_id!r} appears to be a hardcoded production "
                "default; model selection must come from Phase 8 source research"
            )
    assert "model_id:" not in raw_text or all(
        "<set-by" in line or "#" in line.strip()[0:1]
        for line in raw_text.splitlines()
        if "model_id:" in line
    ), (
        "local_runtime_profiles.yaml contains a non-placeholder model_id; "
        "use '<set-by-Phase-8-research>' or omit the field entirely"
    )


# ---------------------------------------------------------------------------
# Local candidate record tests (Phase 08-02)
# ---------------------------------------------------------------------------

def _minimal_registry_yaml(
    task_local_candidates: str = "",
    *,
    local_runtime_profile_id: str | None = None,
) -> str:
    """Return a minimal valid registry YAML with optional local_candidates block."""
    candidates_block = f"\n    local_candidates:\n{task_local_candidates}" if task_local_candidates else ""
    profile_block = (
        f"    local_runtime_profile_id: {local_runtime_profile_id}\n"
        if local_runtime_profile_id
        else ""
    )
    return f"""schema_version: 1
tasks:
  test.task:
    task_family: test
    stage: test-stage
    model_ref: gpt54_mini_legacy
    current_research:
      required: true
      status: pending
    benchmark:
      required: true
      receipt_required: true
      candidates: [gpt-5.4-mini]
    quality_threshold: any
    schema_validity_required: false
    provenance_required: true
    local_runtime_policy: deferred_until_receipts
{profile_block}    selection_status: planning_default
    allowed_selections:
      - {{provider: openai, model: gpt-5.4-mini, role: default}}{candidates_block}
"""


def test_local_candidates_parse_from_registry() -> None:
    registry = load_registry()
    tasks_with_candidates = [
        (task_id, task)
        for task_id, task in registry.tasks.items()
        if len(task.local_candidates) > 0
    ]
    assert tasks_with_candidates, "expected at least one task with local_candidates in real registry"
    for task_id, task in tasks_with_candidates:
        for record in task.local_candidates:
            assert isinstance(record, LocalCandidateRecord), (
                f"{task_id}: expected LocalCandidateRecord, got {type(record)}"
            )
            if record.status == "benchmark_pending":
                assert record.model_id is not None, (
                    f"{task_id}: benchmark_pending candidate must have model_id"
                )
            if record.status == "no_local_candidate":
                assert record.rejection_reason is not None, (
                    f"{task_id}: no_local_candidate must have rejection_reason"
                )


def test_local_candidate_preserves_benchmark_receipt_ref(tmp_path: Path) -> None:
    candidates = """\
      - model_id: some-local-model
        checked_date: "2026-04-27"
        runtime_support: mlx_community_tested
        status: benchmark_pending
        benchmark_receipt_ref: receipts/some-local-model.json
"""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(_minimal_registry_yaml(candidates), encoding="utf-8")
    registry = load_registry(registry_path)
    candidate = registry.get("test.task").local_candidates[0]
    assert candidate.benchmark_receipt_ref == "receipts/some-local-model.json"


def test_task_policy_parses_local_runtime_profile_id(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        _minimal_registry_yaml(local_runtime_profile_id="mlx_light"),
        encoding="utf-8",
    )
    registry = load_registry(registry_path)
    assert registry.get("test.task").local_runtime_profile_id == "mlx_light"


def test_task_policy_rejects_unknown_local_runtime_profile_id(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        _minimal_registry_yaml(local_runtime_profile_id="not_a_real_profile"),
        encoding="utf-8",
    )
    with pytest.raises(ModelGovernanceError) as exc_info:
        load_registry(registry_path)
    msg = str(exc_info.value)
    assert "local_runtime_profile_id" in msg
    assert "not_a_real_profile" in msg


def test_local_candidate_approved_without_receipt_raises(tmp_path: Path) -> None:
    candidates = """\
      - model_id: some-local-model
        checked_date: "2026-04-27"
        runtime_support: mlx_verified
        status: approved
"""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(_minimal_registry_yaml(candidates), encoding="utf-8")
    with pytest.raises(ModelGovernanceError) as exc_info:
        load_registry(registry_path)
    msg = str(exc_info.value)
    assert "approved" in msg
    assert "benchmark_receipt_ref" in msg


def test_local_candidate_missing_checked_date_raises(tmp_path: Path) -> None:
    candidates = """\
      - model_id: some-local-model
        runtime_support: mlx_verified
        status: benchmark_pending
"""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(_minimal_registry_yaml(candidates), encoding="utf-8")
    with pytest.raises(ModelGovernanceError, match="checked_date"):
        load_registry(registry_path)


def test_local_candidate_invalid_status_raises(tmp_path: Path) -> None:
    candidates = """\
      - model_id: some-local-model
        checked_date: "2026-04-27"
        runtime_support: mlx_verified
        status: production_ready
"""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(_minimal_registry_yaml(candidates), encoding="utf-8")
    with pytest.raises(ModelGovernanceError, match="status"):
        load_registry(registry_path)


def test_benchmark_pending_local_candidate_does_not_become_default(tmp_path: Path) -> None:
    candidates = """\
      - model_id: some-local-model
        checked_date: "2026-04-27"
        runtime_support: mlx_community_tested
        status: benchmark_pending
"""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(_minimal_registry_yaml(candidates), encoding="utf-8")
    registry = load_registry(registry_path)
    task = registry.get("test.task")
    assert task.default_selection.provider == "openai"
    assert task.default_selection.model == "gpt-5.4-mini"
    with pytest.raises(ModelGovernanceError):
        registry.validate("test.task", provider="local", model="some-local-model")


def test_local_candidates_absent_returns_empty_tuple() -> None:
    registry = load_registry()
    for task in registry.tasks.values():
        assert isinstance(task.local_candidates, tuple)


# ---------------------------------------------------------------------------
# approved_local_selection tests (Phase 10-01)
# ---------------------------------------------------------------------------

# Companion catalog for every ad-hoc test registry below that references a
# model not meant to exist in the real production catalog (a fake/synthetic
# test identifier, or a historical model kept only for fixture stability).
# `registry.py` resolves a registry's model catalog from a sibling
# `models.yaml` next to it when one exists, falling back to the real
# installed catalog otherwise — so writing this alongside a tmp_path registry
# keeps these fixtures hermetic instead of depending on (or polluting) the
# real src/conference_pipeline/model_governance/models.yaml.
_TEST_MODEL_CATALOG_YAML = """\
schema_version: 1
models:
  test_sonnet:
    provider: anthropic
    model: claude-sonnet-4-6-20251001
    status: active
  test_local:
    provider: local
    model: mlx-community/TestModel-4bit
    status: active
  test_whisper_local:
    provider: local
    model: mlx-community/whisper-large-v3-mlx
    status: active
  test_gpt4o_transcribe_diarize:
    provider: openai
    model: gpt-4o-transcribe-diarize
    status: active
"""


def _write_test_model_catalog(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(_TEST_MODEL_CATALOG_YAML, encoding="utf-8")


_MINIMAL_REGISTRY_FULL_PROOF = """
schema_version: 1
tasks:
  test.task:
    task_family: analysis
    stage: analysis
    model_ref: test_sonnet
    current_research: {required: true}
    benchmark:
      required: true
      candidates: [claude-sonnet-4-6-20251001]
      authority_selections:
        - {provider: anthropic, model: claude-sonnet-4-6-20251001}
    quality_threshold: high
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_benchmarks
    selection_status: active
    local_proof:
      research_ref: "path/to/research"
      hardware_fit_ref: "path/to/hardware"
      benchmark_receipt_ref: "path/to/receipt"
      operator_approval_ref: "path/to/approval"
      approved_date: "2026-04-27T12:00:00Z"
      scope: "test.task for unit testing only"
    allowed_selections:
      - {provider: anthropic, model: claude-sonnet-4-6-20251001, role: default}
      - {provider: local, model: mlx-community/TestModel-4bit, role: approved_local}
"""


def _write_registry(tmp_path: Path, yaml_text: str) -> Path:
    _write_test_model_catalog(tmp_path)
    p = tmp_path / "r.yaml"
    p.write_text(yaml_text.strip(), encoding="utf-8")
    return p


def _write_passed_local_receipt(
    tmp_path: Path,
    *,
    task_id: str = "test.task",
    model_id: str = "mlx-community/TestModel-4bit",
    selection_decision: str = "passed",
) -> Path:
    output_path = tmp_path / "local-output.txt"
    output_path.write_text("source evidence from paper", encoding="utf-8")
    receipt = BenchmarkReceipt(
        workspace="test",
        task_id=task_id,
        task_family="analysis",
        stage="analysis",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-04-27T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id=model_id,
        runtime_id="mlx",
        hardware_profile="mlx:unit-test",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://analysis",
        output_ref=str(output_path),
        raw_output_snippet="source evidence from paper",
        rubric_scores={"quality": 1.0, "schema_ok": 1.0},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=0.0,
        latency_ms=0.0,
        cost_usd=0.0,
        budget_violations=[],
        selection_decision=selection_decision,
        failure_reason=None if selection_decision == "passed" else "unit-test failure",
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt_to_dict(receipt)), encoding="utf-8")
    return path


def _registry_with_receipt_ref(receipt_path: Path) -> str:
    return _MINIMAL_REGISTRY_FULL_PROOF.replace(
        'benchmark_receipt_ref: "path/to/receipt"',
        f'benchmark_receipt_ref: "{receipt_path}"',
    )


def test_approved_local_selection_passes_with_all_four_refs(tmp_path: Path) -> None:
    receipt_path = _write_passed_local_receipt(tmp_path)
    reg_path = _write_registry(tmp_path, _registry_with_receipt_ref(receipt_path))
    result = approved_local_selection("test.task", registry_path=reg_path)
    assert isinstance(result, ModelSelection)
    assert result.role == "approved_local"


def test_approved_local_selection_rejects_missing_receipt(tmp_path: Path) -> None:
    missing = tmp_path / "missing-receipt.json"
    reg_path = _write_registry(tmp_path, _registry_with_receipt_ref(missing))
    with pytest.raises(ModelGovernanceError, match="benchmark_receipt_ref"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_rejects_failed_receipt(tmp_path: Path) -> None:
    receipt_path = _write_passed_local_receipt(tmp_path, selection_decision="failed")
    reg_path = _write_registry(tmp_path, _registry_with_receipt_ref(receipt_path))
    with pytest.raises(ModelGovernanceError, match="selection_decision"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_rejects_receipt_task_mismatch(tmp_path: Path) -> None:
    receipt_path = _write_passed_local_receipt(tmp_path, task_id="other.task")
    reg_path = _write_registry(tmp_path, _registry_with_receipt_ref(receipt_path))
    with pytest.raises(ModelGovernanceError, match="task_id"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_rejects_receipt_model_mismatch(tmp_path: Path) -> None:
    receipt_path = _write_passed_local_receipt(tmp_path, model_id="mlx-community/OtherModel")
    reg_path = _write_registry(tmp_path, _registry_with_receipt_ref(receipt_path))
    with pytest.raises(ModelGovernanceError, match="model_id"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_fails_missing_research_ref(tmp_path: Path) -> None:
    yaml_text = _MINIMAL_REGISTRY_FULL_PROOF.replace(
        'research_ref: "path/to/research"', 'research_ref: ""'
    )
    reg_path = _write_registry(tmp_path, yaml_text)
    with pytest.raises(ModelGovernanceError, match="research_ref"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_fails_missing_hardware_fit_ref(tmp_path: Path) -> None:
    yaml_text = _MINIMAL_REGISTRY_FULL_PROOF.replace(
        'hardware_fit_ref: "path/to/hardware"', 'hardware_fit_ref: ""'
    )
    reg_path = _write_registry(tmp_path, yaml_text)
    with pytest.raises(ModelGovernanceError, match="hardware_fit_ref"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_fails_missing_benchmark_receipt_ref(tmp_path: Path) -> None:
    yaml_text = _MINIMAL_REGISTRY_FULL_PROOF.replace(
        'benchmark_receipt_ref: "path/to/receipt"', 'benchmark_receipt_ref: ""'
    )
    reg_path = _write_registry(tmp_path, yaml_text)
    with pytest.raises(ModelGovernanceError, match="benchmark_receipt_ref"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_fails_missing_operator_approval_ref(tmp_path: Path) -> None:
    yaml_text = _MINIMAL_REGISTRY_FULL_PROOF.replace(
        'operator_approval_ref: "path/to/approval"', 'operator_approval_ref: ""'
    )
    reg_path = _write_registry(tmp_path, yaml_text)
    with pytest.raises(ModelGovernanceError, match="operator_approval_ref"):
        approved_local_selection("test.task", registry_path=reg_path)


def test_approved_local_selection_fails_stale_research(tmp_path: Path) -> None:
    yaml_text = _MINIMAL_REGISTRY_FULL_PROOF.replace(
        "local_proof:\n      research_ref: \"path/to/research\"\n"
        "      hardware_fit_ref: \"path/to/hardware\"\n"
        "      benchmark_receipt_ref: \"path/to/receipt\"\n"
        "      operator_approval_ref: \"path/to/approval\"\n"
        "      approved_date: \"2026-04-27T12:00:00Z\"\n"
        "      scope: \"test.task for unit testing only\"",
        "local_proof: {}",
    )
    reg_path = _write_registry(tmp_path, yaml_text)
    with pytest.raises(ModelGovernanceError):
        approved_local_selection("test.task", registry_path=reg_path)


def test_raw_local_model_string_rejected_by_validate(tmp_path: Path) -> None:
    reg_path = _write_registry(tmp_path, _MINIMAL_REGISTRY_FULL_PROOF)
    with pytest.raises(ModelGovernanceError):
        validate_selection(
            "test.task",
            provider="local",
            model="raw-local-model-name",
            registry_path=reg_path,
        )


def test_local_proof_disagree_fails_closed() -> None:
    registry = load_registry()
    task_id = next(iter(registry.tasks))
    with pytest.raises(ModelGovernanceError):
        approved_local_selection(task_id)


# ---------------------------------------------------------------------------
# Task 2 (03-01): Generic operator-gated promotion tests
# ---------------------------------------------------------------------------


def _make_temp_registry_with_pending_candidate(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal temp registry with a benchmark_pending candidate and a
    passing benchmark receipt suitable for generic promotion tests."""
    registry_yaml = """\
schema_version: 1
tasks:
  transcription.session_speech:
    task_family: transcription
    stage: transcription-ocr
    model_ref: test_gpt4o_transcribe_diarize
    current_research:
      required: true
      status: pending
    benchmark:
      required: true
      receipt_required: true
      candidates: [gpt-4o-transcribe-diarize]
    quality_threshold: any
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_local_runtime_receipts
    selection_status: planning_default_requires_stage_receipt
    local_proof: {}
    allowed_selections:
      - {provider: openai, model: gpt-4o-transcribe-diarize, role: default}
      - {provider: local, model: mlx-community/whisper-large-v3-mlx, role: approved_local}
    local_candidates:
      - model_id: "mlx-community/whisper-large-v3-mlx"
        checked_date: "2026-06-25"
        runtime_support: mlx_community_tested
        status: benchmark_pending
"""
    _write_test_model_catalog(tmp_path)
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(registry_yaml, encoding="utf-8")

    # A passing live benchmark receipt for the candidate
    output_path = tmp_path / "local-output.txt"
    output_path.write_text("hello world transcription output", encoding="utf-8")
    receipt_data = {
        "workspace": "test",
        "task_id": "transcription.session_speech",
        "task_family": "transcription",
        "stage": "transcription-ocr",
        "run_id": "00000000-0000-0000-0000-000000000001",
        "timestamp": "2026-06-25T00:00:00+00:00",
        "execution_mode": "live",
        "provider": "local",
        "model_id": "mlx-community/whisper-large-v3-mlx",
        "runtime_id": "mlx",
        "hardware_profile": "mlx:mlx_light:local",
        "prompt_digest": "a" * 64,
        "config_digest": "b" * 64,
        "input_sample_ref": "fixture://transcription",
        "output_ref": str(output_path),
        "raw_output_snippet": "hello world transcription output",
        "rubric_scores": {
            "wer": 0.1,
            "cer": 0.05,
            "quality": 0.9,
            "schema_ok": 1.0,
        },
        "schema_ok": True,
        "provenance_ok": True,
        "quality_delta": 0.0,
        "latency_ms": 500.0,
        "cost_usd": 0.0,
        "budget_violations": [],
        "selection_decision": "passed",
        "failure_reason": None,
    }
    receipt_path = tmp_path / "whisper-receipt.json"
    import json as _json
    receipt_path.write_text(_json.dumps(receipt_data), encoding="utf-8")
    return reg_path, receipt_path


def test_promote_local_candidate_populates_full_approved_local_proof(tmp_path: Path) -> None:
    """After generic promote_local_candidate(), approved_local_selection() must return
    a ModelSelection with provider=='local' and role=='approved_local'. (LOCAL-01, D-03)"""
    from conference_pipeline.benchmarks.promotion import promote_local_candidate

    reg_path, receipt_path = _make_temp_registry_with_pending_candidate(tmp_path)
    research_ref = str(tmp_path / "research.md")
    hardware_fit_ref = str(tmp_path / "hardware.md")
    operator_approval_ref = str(tmp_path / "approval.md")
    # Write placeholder proof files
    for ref in (research_ref, hardware_fit_ref, operator_approval_ref):
        Path(ref).write_text("placeholder", encoding="utf-8")

    diff = promote_local_candidate(
        task_id="transcription.session_speech",
        model_id="mlx-community/whisper-large-v3-mlx",
        receipt_path=receipt_path,
        research_ref=research_ref,
        hardware_fit_ref=hardware_fit_ref,
        operator_approval_ref=operator_approval_ref,
        registry_path=reg_path,
        dry_run=False,
    )

    assert diff.previous_status == "benchmark_pending"
    assert diff.new_status == "approved"

    # Clear registry cache so approved_local_selection sees updated YAML
    load_registry.cache_clear()
    selection = approved_local_selection(
        "transcription.session_speech", registry_path=reg_path
    )
    assert selection.provider == "local"
    assert selection.role == "approved_local"


def test_promote_local_candidate_without_operator_approval_ref_fails_before_mutation(
    tmp_path: Path,
) -> None:
    """Promotion without operator_approval_ref must fail before writing the registry. (D-03)"""
    from conference_pipeline.benchmarks.promotion import (
        promote_local_candidate,
        PromotionRequiresOperatorApprovalError,
    )

    reg_path, receipt_path = _make_temp_registry_with_pending_candidate(tmp_path)
    before_text = reg_path.read_text(encoding="utf-8")

    with pytest.raises(PromotionRequiresOperatorApprovalError):
        promote_local_candidate(
            task_id="transcription.session_speech",
            model_id="mlx-community/whisper-large-v3-mlx",
            receipt_path=receipt_path,
            research_ref="ref/research.md",
            hardware_fit_ref="ref/hardware.md",
            operator_approval_ref=None,  # missing
            registry_path=reg_path,
        )

    # Registry must be unchanged on failure.
    assert reg_path.read_text(encoding="utf-8") == before_text


def test_promote_local_candidate_dry_run_leaves_registry_unchanged(tmp_path: Path) -> None:
    """dry_run=True must return RegistryDiff but leave registry text unchanged. (T-03-03)"""
    from conference_pipeline.benchmarks.promotion import promote_local_candidate

    reg_path, receipt_path = _make_temp_registry_with_pending_candidate(tmp_path)
    before_text = reg_path.read_text(encoding="utf-8")

    diff = promote_local_candidate(
        task_id="transcription.session_speech",
        model_id="mlx-community/whisper-large-v3-mlx",
        receipt_path=receipt_path,
        research_ref="ref/research.md",
        hardware_fit_ref="ref/hardware.md",
        operator_approval_ref="ref/approval.md",
        registry_path=reg_path,
        dry_run=True,
    )

    assert diff.new_status == "approved"
    assert reg_path.read_text(encoding="utf-8") == before_text


def test_promote_local_candidate_rejects_dry_run_receipt_with_high_rubric(
    tmp_path: Path,
) -> None:
    """A *dry-run* receipt (execution_mode='dry-run', selection_decision='skipped')
    with rubric scores above threshold must be refused — the registry must not be
    mutated. The gate validates receipt provenance, not just rubric_scores. (T-03-03)"""
    import json as _json

    from conference_pipeline.benchmarks.promotion import (
        PromotionError,
        promote_local_candidate,
    )

    reg_path, receipt_path = _make_temp_registry_with_pending_candidate(tmp_path)

    # Tamper the receipt into a dry-run / skipped proof but keep the high rubric
    # scores (quality=0.9 >> the 0.6 transcription floor). The old rubric-only
    # gate would have promoted this; the provenance gate must refuse it.
    receipt_data = _json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_data["execution_mode"] = "dry-run"
    receipt_data["selection_decision"] = "skipped"
    receipt_data["failure_reason"] = "execution_mode=dry-run"
    assert receipt_data["rubric_scores"]["quality"] >= 0.6
    receipt_path.write_text(_json.dumps(receipt_data), encoding="utf-8")

    before_text = reg_path.read_text(encoding="utf-8")

    with pytest.raises(PromotionError, match="passing live local proof"):
        promote_local_candidate(
            task_id="transcription.session_speech",
            model_id="mlx-community/whisper-large-v3-mlx",
            receipt_path=receipt_path,
            research_ref="ref/research.md",
            hardware_fit_ref="ref/hardware.md",
            operator_approval_ref="ref/approval.md",
            registry_path=reg_path,
            dry_run=False,
        )

    # The registry text must be unchanged — fail closed before any mutation.
    assert reg_path.read_text(encoding="utf-8") == before_text


# ---------------------------------------------------------------------------
# Task 1 (03-02): Local-first routing resolver tests
# ---------------------------------------------------------------------------


def _make_approved_registry_and_receipt(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal registry with full local proof + a passing receipt."""
    receipt_output = tmp_path / "local-output.txt"
    receipt_output.write_text("test output", encoding="utf-8")
    receipt = BenchmarkReceipt(
        workspace="test",
        task_id="test.task",
        task_family="analysis",
        stage="analysis",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-06-25T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id="mlx-community/TestModel-4bit",
        runtime_id="mlx",
        hardware_profile="mlx:unit-test",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://test",
        output_ref=str(receipt_output),
        raw_output_snippet="test output",
        rubric_scores={"quality": 1.0, "schema_ok": 1.0},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=0.0,
        latency_ms=0.0,
        cost_usd=0.0,
        budget_violations=[],
        selection_decision="passed",
        failure_reason=None,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt_to_dict(receipt)), encoding="utf-8")

    registry_yaml = f"""\
schema_version: 1
tasks:
  test.task:
    task_family: analysis
    stage: analysis
    model_ref: test_sonnet
    fallback_model_ref: test_sonnet
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [claude-sonnet-4-6-20251001]
      authority_selections:
        - {{provider: anthropic, model: claude-sonnet-4-6-20251001}}
    quality_threshold: high
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_benchmarks
    selection_status: active
    local_proof:
      research_ref: "ref/research"
      hardware_fit_ref: "ref/hardware"
      benchmark_receipt_ref: "{receipt_path}"
      operator_approval_ref: "ref/approval"
    allowed_selections:
      - {{provider: anthropic, model: claude-sonnet-4-6-20251001, role: default}}
      - {{provider: local, model: mlx-community/TestModel-4bit, role: approved_local}}
"""
    _write_test_model_catalog(tmp_path)
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(registry_yaml, encoding="utf-8")
    return reg_path, receipt_path


def test_local_first_resolver_uses_approved_local(tmp_path: Path) -> None:
    """Resolver prefers the approved local selection when all proofs are present."""
    from conference_pipeline.model_governance.local_routing import (
        LocalFirstRoutingDecision,
        resolve_local_first_selection,
    )

    reg_path, _ = _make_approved_registry_and_receipt(tmp_path)
    load_registry.cache_clear()
    decision = resolve_local_first_selection(
        "test.task",
        local_first_enabled=True,
        registry_path=reg_path,
    )

    assert isinstance(decision, LocalFirstRoutingDecision)
    assert decision.provider == "local"
    assert decision.role == "approved_local"
    assert decision.used_local is True
    assert decision.fallback_used is False
    assert decision.fallback_reason is None
    assert decision.task_id == "test.task"


def test_local_first_resolver_falls_back_when_readiness_fails(tmp_path: Path) -> None:
    """Resolver falls back to paid when readiness check raises ModelGovernanceError."""
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )

    reg_path, _ = _make_approved_registry_and_receipt(tmp_path)
    load_registry.cache_clear()

    def failing_readiness(task_id: str) -> bool:
        raise ModelGovernanceError("not installed")

    decision = resolve_local_first_selection(
        "test.task",
        local_first_enabled=True,
        registry_path=reg_path,
        readiness_check=failing_readiness,
    )

    assert decision.fallback_used is True
    assert decision.fallback_reason == "local_readiness_failed"
    assert decision.used_local is False
    # Paid fallback provider is the registry fallback
    assert decision.provider in {"anthropic", "openai", "gemini"}


def test_local_first_resolver_honors_task_force_paid(tmp_path: Path) -> None:
    """A task with local_first_enabled: false forces paid even when local proof exists."""
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )

    registry_yaml = f"""\
schema_version: 1
tasks:
  test.task:
    task_family: analysis
    stage: analysis
    model_ref: test_sonnet
    fallback_model_ref: test_sonnet
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [claude-sonnet-4-6-20251001]
      authority_selections:
        - {{provider: anthropic, model: claude-sonnet-4-6-20251001}}
    quality_threshold: high
    schema_validity_required: false
    provenance_required: false
    local_runtime_policy: deferred_until_benchmarks
    local_first_enabled: false
    selection_status: active
    local_proof:
      research_ref: "ref/research"
      hardware_fit_ref: "ref/hardware"
      benchmark_receipt_ref: "ref/receipt"
      operator_approval_ref: "ref/approval"
    allowed_selections:
      - {{provider: anthropic, model: claude-sonnet-4-6-20251001, role: default}}
      - {{provider: local, model: mlx-community/TestModel-4bit, role: approved_local}}
"""
    _write_test_model_catalog(tmp_path)
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(registry_yaml, encoding="utf-8")
    load_registry.cache_clear()

    decision = resolve_local_first_selection(
        "test.task",
        local_first_enabled=True,
        registry_path=reg_path,
    )

    assert decision.fallback_used is True
    assert decision.fallback_reason == "task_forced_paid"
    assert decision.used_local is False


def test_local_first_resolver_writes_fallback_receipt(tmp_path: Path) -> None:
    """Resolver writes a LocalFallbackReceipt when workspace + readiness fails. (T-03-02)"""
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts

    reg_path, _ = _make_approved_registry_and_receipt(tmp_path)
    load_registry.cache_clear()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def failing_readiness(task_id: str) -> bool:
        raise ModelGovernanceError("not installed")

    decision = resolve_local_first_selection(
        "test.task",
        local_first_enabled=True,
        registry_path=reg_path,
        readiness_check=failing_readiness,
        workspace=workspace,
        run_id="run-fallback-receipt-test",
        stage_id="test-stage",
    )

    assert decision.fallback_used is True
    assert decision.fallback_reason == "local_readiness_failed"

    receipts = scan_local_fallback_receipts(workspace)
    assert len(receipts) == 1
    r = receipts[0]
    assert r["fallback_reason"] == "local_readiness_failed"
    assert r["task_id"] == "test.task"


# ---------------------------------------------------------------------------
# v3.0 03-04: benchmark_receipt_ref exposure (LOCAL-04 / T-03-04)
# ---------------------------------------------------------------------------


def test_local_first_decision_exposes_benchmark_receipt_ref(tmp_path: Path) -> None:
    """LOCAL-04 / T-03-04: when the resolver returns an approved local decision,
    benchmark_receipt_ref must be non-empty so downstream can stamp receipts."""
    from conference_pipeline.model_governance.local_routing import (
        LocalFirstRoutingDecision,
        resolve_local_first_selection,
    )

    reg_path, receipt_path = _make_approved_registry_and_receipt(tmp_path)
    load_registry.cache_clear()

    decision = resolve_local_first_selection(
        "test.task",
        local_first_enabled=True,
        registry_path=reg_path,
    )

    assert isinstance(decision, LocalFirstRoutingDecision)
    assert decision.used_local is True, "decision must use local for an approved task"
    assert decision.benchmark_receipt_ref is not None, (
        "approved local decision must expose benchmark_receipt_ref (T-03-04)"
    )
    assert decision.benchmark_receipt_ref != "", (
        "benchmark_receipt_ref must not be empty for an approved local decision"
    )
    # The ref should point to the receipt path we registered.
    assert str(receipt_path) in decision.benchmark_receipt_ref or \
        decision.benchmark_receipt_ref == str(receipt_path), (
        f"benchmark_receipt_ref should reference the registry receipt: "
        f"got {decision.benchmark_receipt_ref!r}, expected to contain {receipt_path!s}"
    )


# ---------------------------------------------------------------------------
# Regression guard: 2026-07-15 merge (1cdaa5a2e) silently reverted 27 tasks'
# provider/model/fallback fields from the corrected GPT-5.6 Sol/Terra/Luna
# tier assignments back to a stale pre-fix generation. registry.yaml now
# references models.yaml catalog aliases instead of repeating literal model
# strings across dozens of independent (and independently mergeable) sites;
# these assertions pin the exact resolved model per task so a future merge
# that silently reintroduces the same class of staleness fails a test run
# immediately rather than only showing up in a live production receipt.
# ---------------------------------------------------------------------------

# The catalog generation this repository currently runs on -- the ONE place
# a deliberate model upgrade updates (scripts/model_catalog_upgrade.py and the
# model-catalog-upgrade skill tell you to). A merge that reverts models.yaml
# to an older generation fails here; a merge that reverts a task's alias
# assignment fails the task-level pins below.
CURRENT_GENERATION = {
    "sol": ("openai", "gpt-6-sol"),
    "terra": ("openai", "gpt-5.6-terra"),
    "luna": ("openai", "gpt-6-luna"),
    "kimi_openrouter": ("openrouter", "openrouter/moonshotai/kimi-k3"),
    "opus": ("anthropic", "claude-opus-5-5"),
    "sonnet": ("anthropic", "claude-sonnet-5"),
    "gemini_pro": ("gemini", "gemini-3.1-pro-preview"),
    "gemini_flash_lite": ("gemini", "gemini-3.5-flash-lite"),
    "gemini_tts": ("gemini", "gemini-3.8-flash-tts"),
    "gemini_flash_legacy": ("gemini", "gemini-2.0-flash"),
    "gpt4o_mini_tts": ("openai", "gpt-4o-mini-tts"),
    "qwen36_local": ("local", "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"),
}

# Task -> alias for the 27 tasks the 2026-07-15 merge reverted. Pinned by alias
# (what the merge actually broke), so a generation upgrade that moves an alias
# does not touch these, while a reverted task assignment still fails.
_CORRECTED_TASK_DEFAULTS = {
    "analysis.deep_narrative_synthesis": "sol",
    # 2026-10-01: deliberately moved off gemini_pro after the ICML 2026 topic-09
    # benchmark (07A-model-benchmark-20261001.json); the merge-regression value
    # was gemini_pro, the governed value is now kimi_openrouter.
    "analysis.exhaustive_topic_summarization": "kimi_openrouter",
    "analysis.topic_reconciliation": "qwen36_local",
    "analysis.topic_reconciliation_reference_workflow": "qwen36_local",
    "interest_group.research": "gemini_pro",
    "ocr.slide_image_parse": "luna",
    "podcast.longform_script_generation": "gemini_pro",
    "presentation.agentic_deck_generation": "gemini_pro",
    "presentation.agentic_vision_crop": "luna",
    "presentation.deck_generation": "gemini_pro",
    "presentation.presentation_review": "gemini_pro",
    "presentation.presenter_tts": "gemini_tts",
    "presentation.repair": "sonnet",
    "presentation.visual_qc": "sol",
    "ranking.fact_importance": "gemini_flash_legacy",
    "ranking.paper_reference_frequency": "gemini_flash_legacy",
    "research.gpt_researcher_fast": "qwen36_local",
    "research.gpt_researcher_smart": "qwen36_local",
    "research.paper_matching_keypoints": "sol",
    "source_discovery.paper_source_ranking": "terra",
    "synthesis.exhaustive_topic_summary": "gemini_pro",
    "topic_video.scene_generation": "luna",
    "vector_store.query_answer": "luna",
    "video.session_detection": "luna",
    "vision.chart_summary": "luna",
    "vision.figure_caption": "sol",
    "vision.slide_analysis": "luna",
}

_CORRECTED_TASK_FALLBACKS = {
    "analysis.deep_narrative_synthesis": "terra",
    "analysis.exhaustive_topic_summarization": "luna",
    "analysis.topic_reconciliation": "sol",
    "analysis.topic_reconciliation_reference_workflow": "sol",
    "interest_group.research": "luna",
    "ocr.slide_image_parse": "gemini_flash_lite",
    "podcast.longform_script_generation": "luna",
    "presentation.agentic_deck_generation": "luna",
    "presentation.agentic_vision_crop": "terra",
    "presentation.deck_generation": "sol",
    "presentation.presentation_review": "sol",
    "presentation.presenter_tts": "gpt4o_mini_tts",
    "presentation.repair": "luna",
    "presentation.visual_qc": "terra",
    "ranking.fact_importance": "luna",
    "ranking.paper_reference_frequency": "luna",
    "research.gpt_researcher_fast": "luna",
    "research.gpt_researcher_smart": "luna",
    "research.paper_matching_keypoints": "luna",
    "source_discovery.paper_source_ranking": "gemini_flash_lite",
    "synthesis.exhaustive_topic_summary": "luna",
    "topic_video.scene_generation": "terra",
    "vector_store.query_answer": "terra",
    "video.session_detection": None,
    "vision.chart_summary": None,
    "vision.figure_caption": None,
    "vision.slide_analysis": None,
}


def _task_refs(task_id: str) -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["tasks"][task_id]


@pytest.mark.parametrize("task_id", sorted(_CORRECTED_TASK_DEFAULTS))
def test_merge_regressed_tasks_resolve_to_corrected_default(task_id: str) -> None:
    """Pin the 2026-07-15 merge-regression fix: each of the 27 affected tasks
    keeps its corrected alias, and resolves to that alias's current model."""
    alias = _CORRECTED_TASK_DEFAULTS[task_id]
    assert _task_refs(task_id)["model_ref"] == alias
    assert (default_provider(task_id), default_model(task_id)) == CURRENT_GENERATION[alias]


@pytest.mark.parametrize("task_id", sorted(_CORRECTED_TASK_FALLBACKS))
def test_merge_regressed_tasks_resolve_to_corrected_fallback(task_id: str) -> None:
    alias = _CORRECTED_TASK_FALLBACKS[task_id]
    policy = load_registry().get(task_id)
    if alias is None:
        assert policy.fallback_provider is None
        assert policy.fallback_model is None
    else:
        assert _task_refs(task_id).get("fallback_model_ref") == alias
        assert (policy.fallback_provider, policy.fallback_model) == CURRENT_GENERATION[alias]


# ---------------------------------------------------------------------------
# Catalog fail-closed behavior (models.yaml)
# ---------------------------------------------------------------------------


def test_model_catalog_loads_and_every_entry_is_well_formed() -> None:
    catalog = load_model_catalog()
    assert catalog, "expected at least one catalog entry"
    for alias, entry in catalog.items():
        assert isinstance(entry.get("provider"), str) and entry["provider"], alias
        assert isinstance(entry.get("model"), str) and entry["model"], alias
        assert entry.get("status") in {"active", "retired"}, alias


def test_model_catalog_is_on_the_current_generation() -> None:
    """Every pinned alias exists, is active, and names the current model; a
    catalog reverted by a merge (or a model dropped from it) fails here even
    before any task-level assertion runs. Update CURRENT_GENERATION only as
    part of an operator-approved upgrade (model-catalog-upgrade skill)."""
    catalog = load_model_catalog()
    for alias, (provider, model) in CURRENT_GENERATION.items():
        entry = catalog.get(alias)
        assert entry is not None, alias
        assert entry.get("status") == "active", alias
        assert (entry["provider"], entry["model"]) == (provider, model), alias


def test_task_with_unknown_model_ref_raises(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "tasks": {
            "test.task": {
                "task_family": "test",
                "stage": "test-stage",
                "model_ref": "this_alias_does_not_exist_in_any_catalog",
                "current_research": {"required": True},
                "benchmark": {"required": True, "candidates": []},
                "quality_threshold": "test",
                "schema_validity_required": False,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default",
                "allowed_selections": [
                    {"provider": "openai", "model": "gpt-5.4-mini", "role": "default"},
                ],
            },
        },
    }
    with pytest.raises(ModelGovernanceError, match="unknown model-catalog alias"):
        ModelRegistry.from_mapping(payload)


def test_task_with_retired_model_ref_raises() -> None:
    payload = {
        "schema_version": 1,
        "tasks": {
            "test.task": {
                "task_family": "test",
                "stage": "test-stage",
                "model_ref": "haiku_retired",
                "current_research": {"required": True},
                "benchmark": {"required": True, "candidates": []},
                "quality_threshold": "test",
                "schema_validity_required": False,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default",
                "allowed_selections": [
                    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "role": "default"},
                ],
            },
        },
    }
    with pytest.raises(ModelGovernanceError, match="retired model-catalog alias"):
        ModelRegistry.from_mapping(payload)


def test_allowed_selections_with_retired_model_raises() -> None:
    payload = {
        "schema_version": 1,
        "tasks": {
            "test.task": {
                "task_family": "test",
                "stage": "test-stage",
                "model_ref": "sonnet",
                "current_research": {"required": True},
                "benchmark": {"required": True, "candidates": ["claude-sonnet-5"]},
                "quality_threshold": "test",
                "schema_validity_required": False,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default",
                "allowed_selections": [
                    {"provider": "anthropic", "model": "claude-sonnet-5", "role": "default"},
                    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "role": "legacy_candidate"},
                ],
            },
        },
    }
    with pytest.raises(ModelGovernanceError, match="allowed_selections references retired model"):
        ModelRegistry.from_mapping(payload)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5-mini", True),
        ("gpt-5.6-luna", True),
        ("gpt-6-sol", True),  # the generation check must survive the next upgrade
        ("gpt-7-astra", True),
        ("openrouter/openai/gpt-6-luna", True),
        ("o3-deep-research-2025-06-26", True),
        ("gpt-4o", False),
        ("gpt-4o-mini-tts", False),
        ("claude-opus-5-5", False),
        ("gemini-3.8-flash", False),
    ],
)
def test_openai_reasoning_params_follow_the_generation_not_the_name(model: str, expected: bool) -> None:
    from conference_pipeline.model_governance.model_families import uses_openai_reasoning_params

    assert uses_openai_reasoning_params(model) is expected


def test_public_search_embedding_tasks_are_governed_without_fallback() -> None:
    """v6.0 Phase 4 D2 (spec 6.6): both site embedding models come from the registry,
    with no fallback, and public_index's defaults are those registry values."""
    from conference_pipeline.model_governance import registry as reg
    from conference_pipeline.publishing import public_index

    canonical = reg.load_registry().get("public_search.canonical_embedding")
    browser = reg.load_registry().get("public_search.browser_embedding")
    assert (canonical.provider, canonical.fallback_model) == ("mlx", None)
    assert (browser.provider, browser.fallback_model) == ("browser", None)
    assert public_index.DEFAULT_EMBEDDING_PROVIDER == canonical.provider
    assert public_index.DEFAULT_EMBEDDING_MODEL == canonical.model == reg.model_for_alias("bge_m3_mlx")
    assert public_index.DEFAULT_BROWSER_EMBEDDING_MODEL == browser.model == reg.model_for_alias("embeddinggemma_onnx_q8")


def test_mlx_embedding_defaults_come_from_the_catalog(monkeypatch) -> None:
    """D2 review r1, F1: the factory and the resolver default to the catalog's bge-m3 entry."""
    from conference_pipeline.indexing import embedding_resolver
    from conference_pipeline.model_governance import registry as reg

    config = embedding_resolver.resolve_embedding_config(env={"EMBEDDING_PROVIDER": "mlx"})
    assert config["model"] == reg.model_for_alias("bge_m3_mlx")
    monkeypatch.setattr(reg, "model_for_alias", lambda alias: f"catalog::{alias}")
    assert embedding_resolver.resolve_embedding_config(env={"EMBEDDING_PROVIDER": "mlx"})["model"] == "catalog::bge_m3_mlx"
