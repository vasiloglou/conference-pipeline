"""BenchmarkHarness: run matrix construction and dry-run execution."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from conference_pipeline.model_governance import local_preflight
from conference_pipeline.model_governance.local_runtime import LocalRuntimeProfile, load_profiles
from conference_pipeline.model_governance.mlx_client import ManagedMlxServer, MlxServerClient
from conference_pipeline.model_governance.registry import ModelRegistry
from conference_pipeline.model_governance.registry import ModelGovernanceError
from conference_pipeline.model_governance.registry import TaskPolicy
from conference_pipeline.model_governance.runtime_lock import HeavyRuntimeLock
from conference_pipeline.benchmarks.receipt_schema import (
    BenchmarkReceipt,
    receipt_to_dict,
    validate_receipt,
)
from conference_pipeline.benchmarks.transcription_scoring import score_transcription_output


_PROVENANCE_TOKENS = ("source", "citation", "evidence", "paper", "session", "url", "doi", "openreview")
# v1.61 (Phase 67) lifted "qwen3-vl" from this list now that the local vision
# lane (vllm-mlx / mlx-vlm + FastVLM native) lands. Per-model gating is
# enforced via registry.yaml `local_candidates[*].status` (`benchmark_pending`
# blocks until promoted to `approved` per VLM-BENCH-03).
# v3.0 Phase 03-01: "whisper" lifted — transcription benchmarking now supported.
# kokoro (TTS) remains unsupported.
_UNSUPPORTED_LOCAL_MODEL_TOKENS = ("kokoro",)
# v1.61 (Phase 67) lifted "ocr" and "vision" from this set.
# v3.0 Phase 03-01: lifted "transcription" — whisper-large-v3-mlx is benchmarkable.
# tts and audio remain blocked (separate future milestone).
_UNSUPPORTED_LOCAL_TASK_FAMILIES = {"tts", "audio"}


def _provider_for_model(policy: TaskPolicy, model_id: str) -> str | None:
    """Return the registered provider for a benchmark candidate model."""
    selections = list(policy.allowed_selections)
    selections.append(policy.default_selection)
    fallback = policy.fallback_selection
    if fallback is not None:
        selections.append(fallback)
    for selection in selections:
        if selection.model == model_id:
            return selection.provider
    return None


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def _prompt_for_input(input_sample: dict) -> str:
    return json.dumps(input_sample, sort_keys=True)


def _unsupported_local_runtime_shape(policy: TaskPolicy, model_id: str) -> bool:
    task_family = policy.task_family.casefold()
    model = model_id.casefold()
    if task_family in _UNSUPPORTED_LOCAL_TASK_FAMILIES:
        return True
    return any(token in model for token in _UNSUPPORTED_LOCAL_MODEL_TOKENS)


def _lock_metadata(task_id: str, profile_id: str, model_id: str) -> dict:
    return {
        "task_id": task_id,
        "profile_id": profile_id,
        "model_id": model_id,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _live_hardware_profile(profile: LocalRuntimeProfile) -> str:
    return f"{profile.runtime_kind}:{profile.profile_id}:local"


def _make_rubric_scores(policy: TaskPolicy, input_sample: dict) -> dict:
    """Return task-family-appropriate rubric scores for a dry-run/fixture receipt.

    For transcription tasks (D-02): compute WER/CER/quality when reference_text
    is provided in the input_sample; fall back to fixture defaults otherwise.
    For all other task families: return the standard quality/schema_ok defaults.
    """
    if policy.task_family == "transcription":
        reference_text = str(input_sample.get("reference_text") or "")
        # Fixture candidate output: deterministic mock text matching the audio URL
        candidate_text = str(input_sample.get("transcript") or "")
        scores = score_transcription_output(reference_text, candidate_text)
        scores["schema_ok"] = 1.0
        return scores
    return {"quality": 0.0, "schema_ok": 1.0}


def _score_output(output: str, policy: TaskPolicy) -> tuple[bool, bool, float]:
    schema_ok = bool(output.strip())
    if policy.provenance_required:
        lowered = output.casefold()
        provenance_ok = any(token in lowered for token in _PROVENANCE_TOKENS)
    else:
        provenance_ok = True
    quality = 1.0 if schema_ok and provenance_ok else 0.0
    return schema_ok, provenance_ok, quality


def _safe_failure_reason(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    text = text.splitlines()[0]
    for bad in ("token", "secret", "password", "api_key", "apikey"):
        text = text.replace(bad, "[redacted]")
        text = text.replace(bad.upper(), "[redacted]")
    return text[:240]


@dataclass
class RunSpec:
    task_id: str
    provider: str
    model_id: str
    runtime_id: str
    role: str  # "authority" | "cloud_candidate" | "local_candidate"
    runtime_profile_id: str | None = None


class BenchmarkHarness:
    def __init__(
        self,
        registry: ModelRegistry,
        output_dir: Path,
        *,
        workspace: str = "workstream",
        local_runtime_profile_id: str | None = None,
        profiles_path: Path | None = None,
        local_profiles: dict[str, LocalRuntimeProfile] | None = None,
        lock_path: Path | None = None,
        preflight_runner=None,
        runtime_lock_factory=None,
        managed_server_factory=None,
        client_factory=None,
    ) -> None:
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.workspace = workspace
        self.local_runtime_profile_id = local_runtime_profile_id
        self.profiles_path = profiles_path
        self.local_profiles = local_profiles
        self.lock_path = lock_path
        self.preflight_runner = preflight_runner or local_preflight.run_all
        self.runtime_lock_factory = runtime_lock_factory or HeavyRuntimeLock
        self.managed_server_factory = managed_server_factory or ManagedMlxServer
        self.client_factory = client_factory or MlxServerClient

    def build_run_matrix(self, task_id: str) -> list[RunSpec]:
        """Return a RunSpec list for all authority, candidate, and local roles."""
        policy = self.registry.get(task_id)
        benchmark = policy.benchmark
        specs: list[RunSpec] = []

        # Authority selections
        authority_selections = benchmark.get("authority_selections") or []
        if authority_selections:
            for item in authority_selections:
                specs.append(
                    RunSpec(
                        task_id=task_id,
                        provider=str(item["provider"]),
                        model_id=str(item["model"]),
                        runtime_id="mock",
                        role="authority",
                    )
                )
        else:
            # Fall back to the task's registered default as authority baseline
            specs.append(
                RunSpec(
                    task_id=task_id,
                    provider=policy.provider,
                    model_id=policy.model,
                    runtime_id="mock",
                    role="authority",
                )
            )

        # Candidate selections (explicit cloud candidates)
        seen_candidates: set[tuple[str, str]] = set()
        for item in benchmark.get("candidate_selections") or []:
            pair = (str(item["provider"]), str(item["model"]))
            seen_candidates.add(pair)
            specs.append(
                RunSpec(
                    task_id=task_id,
                    provider=pair[0],
                    model_id=pair[1],
                    runtime_id="mock",
                    role="cloud_candidate",
                )
            )
        for model_id in benchmark.get("candidates") or []:
            model_id = str(model_id)
            provider = _provider_for_model(policy, model_id)
            if provider is None or provider == "local":
                continue
            pair = (provider, model_id)
            if pair in seen_candidates:
                continue
            specs.append(
                RunSpec(
                    task_id=task_id,
                    provider=provider,
                    model_id=model_id,
                    runtime_id="mock",
                    role="cloud_candidate",
                )
            )
            seen_candidates.add(pair)

        # Local candidates awaiting benchmarking
        for candidate in policy.local_candidates:
            if candidate.status == "benchmark_pending" and candidate.model_id:
                specs.append(
                    RunSpec(
                        task_id=task_id,
                        provider="local",
                        model_id=candidate.model_id,
                        runtime_id="mlx",
                        role="local_candidate",
                    )
                )

        return specs

    def run_task(
        self,
        task_id: str,
        spec: RunSpec,
        input_sample: dict,
        *,
        mode: str = "dry-run",
    ) -> BenchmarkReceipt:
        """Execute one benchmark run, producing and persisting a receipt."""
        policy = self.registry.get(task_id)
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        if mode == "live" and spec.role == "local_candidate":
            return self._run_live_local(task_id, policy, spec, input_sample, task_dir)

        key_data = json.dumps(
            {"task_id": task_id, "model_id": spec.model_id, "input": input_sample},
            sort_keys=True,
        )
        mock_output = hashlib.sha256(key_data.encode()).hexdigest()[:32]

        prompt_digest = hashlib.sha256(b"mock-prompt").hexdigest()
        config_digest = hashlib.sha256(
            json.dumps({"task_id": task_id, "model_id": spec.model_id}, sort_keys=True).encode()
        ).hexdigest()

        rubric_scores = _make_rubric_scores(policy, input_sample)

        receipt = BenchmarkReceipt(
            workspace=self.workspace,
            task_id=task_id,
            task_family=policy.task_family,
            stage=policy.stage,
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_mode=mode,
            provider=spec.provider,
            model_id=spec.model_id,
            runtime_id=spec.runtime_id,
            hardware_profile="mock",
            prompt_digest=prompt_digest,
            config_digest=config_digest,
            input_sample_ref=f"fixture://{policy.task_family}",
            output_ref=None,
            raw_output_snippet=mock_output,
            rubric_scores=rubric_scores,
            schema_ok=True,
            provenance_ok=True,
            quality_delta=0.0,
            latency_ms=0.0,
            cost_usd=0.0,
            budget_violations=[],
            selection_decision="skipped",
            failure_reason=f"execution_mode={mode}",
        )

        self._persist_receipt(receipt, task_dir, policy)
        return receipt

    def _run_live_local(
        self,
        task_id: str,
        policy: TaskPolicy,
        spec: RunSpec,
        input_sample: dict,
        task_dir: Path,
    ) -> BenchmarkReceipt:
        """Run a live local benchmark through the typed MLX runtime stack."""
        profile, failure_reason = self._resolve_runtime_profile(policy, spec)
        model_id = (profile.model_id or spec.model_id) if profile else spec.model_id
        if failure_reason:
            receipt = self._make_failed_live_local_receipt(
                task_id, policy, spec, input_sample, failure_reason, model_id=model_id
            )
            self._persist_receipt(receipt, task_dir, policy)
            return receipt

        assert profile is not None
        if _unsupported_local_runtime_shape(policy, model_id):
            receipt = self._make_failed_live_local_receipt(
                task_id,
                policy,
                spec,
                input_sample,
                "unsupported_local_runtime_shape",
                model_id=model_id,
            )
            self._persist_receipt(receipt, task_dir, policy)
            return receipt

        lock = self._make_runtime_lock()
        server = None
        lock_acquired = False
        try:
            effective_profile = self.preflight_runner(profile, model_id, lock)
            client = self.client_factory(
                effective_profile.base_url,
                timeout_s=effective_profile.request_timeout_s,
            )
            if effective_profile.manage_server:
                server = self.managed_server_factory()
                server.start(effective_profile, model_id, lock=lock, task_id=task_id)
            else:
                lock.acquire(_lock_metadata(task_id, effective_profile.profile_id, model_id))
                lock_acquired = True
                if not client.health_check():
                    raise ModelGovernanceError(
                        f"MLX server health check failed for profile {effective_profile.profile_id!r}"
                    )

            output = client.complete(_prompt_for_input(input_sample), model=model_id)
            receipt = self._receipt_from_live_output(
                task_id, policy, spec, input_sample, model_id, effective_profile, output, task_dir
            )
        except Exception as exc:
            receipt = self._make_failed_live_local_receipt(
                task_id,
                policy,
                spec,
                input_sample,
                _safe_failure_reason(exc),
                model_id=model_id,
            )
        finally:
            if server is not None:
                server.stop()
            elif lock_acquired:
                lock.release()

        self._persist_receipt(receipt, task_dir, policy)
        return receipt

    def _resolve_runtime_profile(
        self,
        policy: TaskPolicy,
        spec: RunSpec,
    ) -> tuple[LocalRuntimeProfile | None, str | None]:
        profile_id = (
            spec.runtime_profile_id
            or self.local_runtime_profile_id
            or policy.local_runtime_profile_id
        )
        if not profile_id:
            return None, "local_runtime_profile_id required"
        profiles = self.local_profiles if self.local_profiles is not None else load_profiles(self.profiles_path)
        profile = profiles.get(profile_id)
        if profile is None:
            return None, f"unknown local_runtime_profile_id: {profile_id}"
        return profile, None

    def _make_runtime_lock(self):
        lock_path = self.lock_path or (self.output_dir / "local-runtime.lock")
        return self.runtime_lock_factory(lock_path)

    def _receipt_from_live_output(
        self,
        task_id: str,
        policy: TaskPolicy,
        spec: RunSpec,
        input_sample: dict,
        model_id: str,
        profile: LocalRuntimeProfile,
        output: str,
        task_dir: Path,
    ) -> BenchmarkReceipt:
        output_text = output.strip()
        if not output_text:
            return self._make_failed_live_local_receipt(
                task_id, policy, spec, input_sample, "missing_local_output", model_id=model_id
            )

        output_path = task_dir / f"{_safe_model_id(model_id)}.output.txt"
        output_path.write_text(output_text, encoding="utf-8")
        schema_ok, provenance_ok, quality = _score_output(output_text, policy)
        passed = schema_ok and provenance_ok and quality > 0
        receipt = self._make_live_local_receipt(
            task_id=task_id,
            policy=policy,
            spec=spec,
            input_sample=input_sample,
            model_id=model_id,
            hardware_profile=_live_hardware_profile(profile),
            output_ref=str(output_path),
            raw_output_snippet=output_text[:500],
            rubric_scores={"quality": quality, "schema_ok": 1.0 if schema_ok else 0.0},
            schema_ok=schema_ok,
            provenance_ok=provenance_ok,
            quality_delta=0.0 if passed else None,
            selection_decision="passed" if passed else "failed",
            failure_reason=None if passed else "quality_or_provenance_gate_failed",
        )
        try:
            validate_receipt(receipt, task_policy=policy)
            return receipt
        except ValueError as exc:
            return self._make_failed_live_local_receipt(
                task_id,
                policy,
                spec,
                input_sample,
                f"receipt_validation_failed: {_safe_failure_reason(exc)}",
                model_id=model_id,
                output_ref=str(output_path),
                raw_output_snippet=output_text[:500],
            )

    def _make_failed_live_local_receipt(
        self,
        task_id: str,
        policy: TaskPolicy,
        spec: RunSpec,
        input_sample: dict,
        failure_reason: str,
        *,
        model_id: str | None = None,
        output_ref: str | None = None,
        raw_output_snippet: str = "",
    ) -> BenchmarkReceipt:
        return self._make_live_local_receipt(
            task_id=task_id,
            policy=policy,
            spec=spec,
            input_sample=input_sample,
            model_id=model_id or spec.model_id,
            hardware_profile="mock",
            output_ref=output_ref,
            raw_output_snippet=raw_output_snippet,
            rubric_scores={"quality": 0.0, "schema_ok": 0.0},
            schema_ok=False,
            provenance_ok=False,
            quality_delta=None,
            selection_decision="failed",
            failure_reason=failure_reason,
        )

    def _make_live_local_receipt(
        self,
        *,
        task_id: str,
        policy: TaskPolicy,
        spec: RunSpec,
        input_sample: dict,
        model_id: str,
        hardware_profile: str,
        output_ref: str | None,
        raw_output_snippet: str,
        rubric_scores: dict,
        schema_ok: bool,
        provenance_ok: bool,
        quality_delta: float | None,
        selection_decision: str,
        failure_reason: str | None,
    ) -> BenchmarkReceipt:
        prompt = _prompt_for_input(input_sample)
        return BenchmarkReceipt(
            workspace=self.workspace,
            task_id=task_id,
            task_family=policy.task_family,
            stage=policy.stage,
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_mode="live",
            provider="local",
            model_id=model_id,
            runtime_id=spec.runtime_id,
            hardware_profile=hardware_profile,
            prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
            config_digest=hashlib.sha256(
                json.dumps({"task_id": task_id, "model_id": model_id}, sort_keys=True).encode()
            ).hexdigest(),
            input_sample_ref=f"fixture://{policy.task_family}",
            output_ref=output_ref,
            raw_output_snippet=raw_output_snippet,
            rubric_scores=rubric_scores,
            schema_ok=schema_ok,
            provenance_ok=provenance_ok,
            quality_delta=quality_delta,
            latency_ms=0.0,
            cost_usd=0.0,
            budget_violations=[],
            selection_decision=selection_decision,
            failure_reason=failure_reason,
        )

    def _persist_receipt(
        self,
        receipt: BenchmarkReceipt,
        task_dir: Path,
        policy: TaskPolicy,
    ) -> None:
        validate_receipt(receipt, task_policy=policy)
        receipt_path = task_dir / f"{_safe_model_id(receipt.model_id)}.json"
        receipt_path.write_text(
            json.dumps(receipt_to_dict(receipt), indent=2), encoding="utf-8"
        )

    def run_all(
        self,
        *,
        task_ids: list[str] | None = None,
        mode: str = "dry-run",
    ) -> list[BenchmarkReceipt]:
        """Run benchmark across all (or specified) tasks and return receipts."""
        from conference_pipeline.benchmarks.fixtures import BENCHMARK_FIXTURES

        all_task_ids = list(task_ids) if task_ids is not None else list(self.registry.tasks.keys())
        receipts: list[BenchmarkReceipt] = []

        for tid in all_task_ids:
            policy = self.registry.get(tid)
            input_sample = BENCHMARK_FIXTURES.get(
                policy.task_family, BENCHMARK_FIXTURES["default"]
            )
            specs = self.build_run_matrix(tid)
            for spec in specs:
                receipts.append(self.run_task(tid, spec, input_sample, mode=mode))

        return receipts
