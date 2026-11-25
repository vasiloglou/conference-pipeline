"""Whole-pipeline provider cost estimates for conference workspaces."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

SCENARIOS = ("low", "base", "high")
DEFAULT_SCOPE = "full"
DEFAULT_SCENARIO = "base"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus"}
TEXT_EXTENSIONS = {".md", ".txt", ".json", ".jsonl"}


@dataclass
class Workload:
    """Measured or inferred workspace quantities used for cost estimates."""

    workspace: str
    audio_files: int = 0
    audio_minutes: float = 0.0
    audio_duration_missing_files: int = 0
    image_files: int = 0
    openai_image_tokens: int = 0
    anthropic_image_tokens: int = 0
    pdf_files: int = 0
    pdf_pages: int = 0
    text_files: int = 0
    text_chars: int = 0
    text_tokens: int = 0
    topic_count: int = 0
    presentation_targets: int = 0
    source_discovery_candidate_pairs: int = 0
    tavily_priority_gap_queries: int = 0
    # Per topic/category pair input tokens for the exhaustive summary pass, and
    # where they were measured ("summary_plan", "llm_full_read_records", or ""
    # when nothing in the workspace sizes the pass yet).
    exhaustive_summary_pair_tokens: list[int] = field(default_factory=list)
    exhaustive_summary_quantity_basis: str = ""
    # Per topic artifact-findings input tokens for the paper-matching keypoint
    # call, measured from existing findings or bounded by the extractor's cap.
    paper_matching_keypoint_input_tokens: list[int] = field(default_factory=list)
    paper_matching_keypoint_quantity_basis: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audio_hours"] = round(self.audio_minutes / 60, 4)
        return payload


@dataclass(frozen=True)
class ReceiptBoundAudioWorkload:
    """Finite speech workload derived from declared acquired-audio receipt rows."""

    receipt_sha256: str
    item_count: int
    duration_seconds: float
    item_ids: tuple[str, ...]

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_sha256": self.receipt_sha256,
            "item_count": self.item_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "duration_minutes": round(self.duration_minutes, 6),
            "item_ids": list(self.item_ids),
        }


@dataclass
class CostEstimate:
    """A single task-level estimate."""

    task_id: str
    family: str
    stage: str
    provider: str
    model: str
    pricing_unit: str
    enabled: bool
    quantity: float
    input_tokens: int = 0
    output_tokens: int = 0
    audio_minutes: float = 0.0
    tts_minutes: float = 0.0
    credits: float = 0.0
    estimated_cost_usd: float = 0.0
    # "measured" when the quantity came from the workspace; "declared_fallback_budget"
    # when it was not derivable and the spec's declared budget was priced instead.
    quantity_basis: str = "measured"
    source_urls: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimated_cost_usd"] = round(self.estimated_cost_usd, 6)
        payload["quantity"] = round(self.quantity, 4)
        payload["audio_minutes"] = round(self.audio_minutes, 4)
        payload["tts_minutes"] = round(self.tts_minutes, 4)
        payload["credits"] = round(self.credits, 4)
        return payload


@dataclass
class EstimateReport:
    """Serializable whole-pipeline cost report."""

    mode: str
    scope: str
    scenario: str
    workspace: str
    generated_at: str
    pricing_refreshed_at: str
    workload: Workload
    tasks: list[CostEstimate]
    actual_usage: dict[str, Any] | None = None

    @property
    def total_estimated_cost_usd(self) -> float:
        return sum(task.estimated_cost_usd for task in self.tasks)

    def totals_by_provider(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for task in self.tasks:
            totals[task.provider] = totals.get(task.provider, 0.0) + task.estimated_cost_usd
        return {provider: round(total, 6) for provider, total in sorted(totals.items())}

    def totals_by_family(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for task in self.tasks:
            totals[task.family] = totals.get(task.family, 0.0) + task.estimated_cost_usd
        return {family: round(total, 6) for family, total in sorted(totals.items())}

    def task_family_ids(self) -> list[str]:
        return sorted({task.family for task in self.tasks})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "scope": self.scope,
            "scenario": self.scenario,
            "workspace": self.workspace,
            "generated_at": self.generated_at,
            "pricing_refreshed_at": self.pricing_refreshed_at,
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 6),
            "totals_by_provider": self.totals_by_provider(),
            "totals_by_family": self.totals_by_family(),
            "task_family_ids": self.task_family_ids(),
            "workload": self.workload.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
        }
        if self.actual_usage is not None:
            payload["actual_usage"] = self.actual_usage
        return payload


class PricingCatalog:
    """Lookup and math helpers for the static provider pricing registry."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.sources = data.get("sources", {})
        self.aliases = data.get("aliases", {})
        self._model_index: dict[str, tuple[str, dict[str, Any]]] = {}
        for provider, provider_data in data.get("providers", {}).items():
            for model, model_data in provider_data.get("models", {}).items():
                self._model_index[model] = (provider, model_data)

    @property
    def refreshed_at(self) -> str:
        return str(self.data.get("refreshed_at", "unknown"))

    def resolve_model(
        self, model: str, provider: str | None = None
    ) -> tuple[str, str, dict[str, Any]]:
        if provider:
            provider_models = self.data.get("providers", {}).get(provider, {}).get("models", {})
            if model in provider_models:
                return provider, model, provider_models[model]

        canonical = self.aliases.get(model, model)
        if canonical in self._model_index:
            resolved_provider, model_data = self._model_index[canonical]
            return resolved_provider, canonical, model_data

        if "/" in canonical and not canonical.startswith("openrouter/"):
            _, short_model = canonical.split("/", 1)
            if short_model in self._model_index:
                resolved_provider, model_data = self._model_index[short_model]
                return resolved_provider, short_model, model_data

        raise KeyError(f"No pricing entry for model {model!r}")

    def high_tier_input_token_threshold(self, model: str, provider: str | None = None) -> int | None:
        """Per-request input size above which the model bills at its high tier.

        Returns ``None`` when the model is unpriced or has no large-prompt tier.
        """

        try:
            _, _, model_data = self.resolve_model(model, provider)
        except KeyError:
            return None
        threshold = model_data.get("high_tier_input_token_threshold")
        if threshold is None:
            return None
        try:
            value = int(threshold)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def source_urls_for(self, model_data: dict[str, Any]) -> list[str]:
        source = model_data.get("source")
        if source is None:
            return []
        if source in self.sources:
            return [self.sources[source]]
        if source == "local":
            return []
        return [str(source)]

    def estimate_audio_minutes(
        self, provider: str, model: str, minutes: float
    ) -> tuple[float, list[str], list[str]]:
        _, _, model_data = self.resolve_model(model, provider)
        price = float(model_data.get("audio_minute_usd", 0.0))
        return minutes * price, self.source_urls_for(model_data), []

    def estimate_tokens(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, list[str], list[str]]:
        resolved_provider, _, model_data = self.resolve_model(model, provider)
        input_price = float(model_data.get("input_per_1m_usd", 0.0))
        output_price = float(model_data.get("output_per_1m_usd", 0.0))
        warnings: list[str] = []
        # Large-prompt tiering: some providers (e.g. Gemini 2.5/3.x Pro) bill a
        # request at a higher per-token rate for BOTH input and output once the
        # request's input exceeds a threshold. Applied per-request against this
        # call's input_tokens, matching how the provider invoices each call.
        threshold = model_data.get("high_tier_input_token_threshold")
        if threshold is not None and input_tokens > int(threshold):
            input_price = float(model_data.get("input_per_1m_usd_high", input_price))
            output_price = float(model_data.get("output_per_1m_usd_high", output_price))
            warnings.append(
                f"Large-prompt tier applied: input {input_tokens:,} tokens exceeds "
                f"{int(threshold):,}; billed at the >{int(threshold):,}-token rate."
            )
        subtotal = (input_tokens / 1_000_000) * input_price
        subtotal += (output_tokens / 1_000_000) * output_price
        subtotal *= 1.0 + float(model_data.get("platform_fee_rate", 0.0))
        if resolved_provider == "openrouter":
            warnings.append("OpenRouter route prices can change; refresh against the live catalog before approval.")
        if model_data.get("estimated_from_model"):
            warnings.append(
                f"Pricing uses {model_data['estimated_from_model']} as a static fallback."
            )
        if model_data.get("notes"):
            warnings.append(str(model_data["notes"]))
        return subtotal, self.source_urls_for(model_data), warnings

    def estimate_tts_minutes(
        self, provider: str, model: str, minutes: float
    ) -> tuple[float, list[str], list[str]]:
        _, _, model_data = self.resolve_model(model, provider)
        price = float(model_data.get("tts_minute_usd", 0.0))
        warnings: list[str] = []
        if model_data.get("estimate_method"):
            warnings.append(str(model_data["estimate_method"]))
        if model_data.get("notes"):
            warnings.append(str(model_data["notes"]))
        return minutes * price, self.source_urls_for(model_data), warnings

    def estimate_gpu_cost(
        self, provider: str, model: str, gpu_seconds: float
    ) -> tuple[float, list[str], list[str]]:
        """Self-hosted endpoints billed by GPU wall time (``pricing_unit: gpu_second``)."""
        _, _, model_data = self.resolve_model(model, provider)
        if model_data.get("pricing_unit") != "gpu_second":
            raise KeyError(f"{model!r} is not priced per GPU second")
        warnings = [f"Billed by GPU time: {gpu_seconds:.3f} s on {model_data.get('gpu', 'GPU')}."]
        if model_data.get("notes"):
            warnings.append(str(model_data["notes"]))
        return gpu_seconds * float(model_data["gpu_second_usd"]), self.source_urls_for(model_data), warnings

    def estimate_credits(
        self, provider: str, model: str, credits: float
    ) -> tuple[float, list[str], list[str]]:
        _, _, model_data = self.resolve_model(model, provider)
        credit_price = float(model_data.get("credit_usd", 0.0))
        warnings: list[str] = []
        if model_data.get("notes"):
            warnings.append(str(model_data["notes"]))
        return credits * credit_price, self.source_urls_for(model_data), warnings


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def build_receipt_bound_audio_workload(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipt_sha256: str,
) -> ReceiptBoundAudioWorkload:
    """Build a finite workload only from explicit acquired-audio receipt rows.

    This is intentionally not a path scanner: callers must pass the rows from
    one declared receipt and bind that receipt's byte digest.  A non-acquired
    row cannot be silently included in paid speech scope.
    """

    if not _is_sha256(receipt_sha256):
        raise ValueError("receipt_sha256 must be a SHA-256 digest")
    if not rows:
        raise ValueError("receipt-bound audio workload requires declared rows")
    seen_item_ids: set[str] = set()
    seen_video_ids: set[str] = set()
    duration_seconds = 0.0
    item_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("receipt row must be an object, not a glob/path")
        item_id = row.get("item_id")
        video_id = row.get("video_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("receipt row has missing item_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("receipt row has missing video_id")
        if item_id in seen_item_ids or video_id in seen_video_ids:
            raise ValueError("receipt rows contain duplicate item_id or video_id")
        if row.get("terminal_state") != "acquired":
            raise ValueError("receipt workload accepts acquired rows only")
        if not _is_sha256(row.get("sha256")):
            raise ValueError("receipt row has missing or invalid audio digest")
        try:
            duration = float(row.get("duration_seconds"))
        except (TypeError, ValueError) as exc:
            raise ValueError("receipt row has missing duration") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("receipt row has nonfinite duration")
        seen_item_ids.add(item_id)
        seen_video_ids.add(video_id)
        item_ids.append(item_id)
        duration_seconds += duration
    return ReceiptBoundAudioWorkload(
        receipt_sha256=receipt_sha256,
        item_count=len(item_ids),
        duration_seconds=duration_seconds,
        item_ids=tuple(item_ids),
    )


def estimate_receipt_bound_audio_workload(
    workload: ReceiptBoundAudioWorkload,
    *,
    provider: str,
    model: str,
    catalog: PricingCatalog | None = None,
) -> dict[str, Any]:
    """Price a finite receipt workload and reject models without audio pricing."""

    if workload.item_count <= 0 or not math.isfinite(workload.duration_minutes):
        raise ValueError("receipt workload must have a finite non-empty duration")
    pricing = catalog or load_pricing_catalog()
    resolved_provider, resolved_model, model_data = pricing.resolve_model(model, provider)
    if "audio_minute_usd" not in model_data:
        raise ValueError("selected model has no audio-minute price")
    price = float(model_data["audio_minute_usd"])
    if not math.isfinite(price) or price < 0:
        raise ValueError("selected model has an invalid audio-minute price")
    cost, sources, warnings = pricing.estimate_audio_minutes(
        resolved_provider,
        resolved_model,
        workload.duration_minutes,
    )
    return {
        "receipt_sha256": workload.receipt_sha256,
        "item_count": workload.item_count,
        "duration_seconds": round(workload.duration_seconds, 6),
        "duration_minutes": round(workload.duration_minutes, 6),
        "provider": resolved_provider,
        "model": resolved_model,
        "estimated_cost_usd": round(cost, 6),
        "pricing_sources": sources,
        "warnings": warnings,
    }


def _load_package_text(filename: str) -> str:
    return resources.files("conference_pipeline.costs").joinpath(filename).read_text()


def load_pricing_catalog(path: Path | None = None) -> PricingCatalog:
    if path is None:
        data = json.loads(_load_package_text("pricing_catalog.json"))
    else:
        data = json.loads(path.read_text())
    return PricingCatalog(data)


def load_task_specs(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return yaml.safe_load(_load_package_text("task_specs.yaml"))
    return yaml.safe_load(path.read_text())


def estimate_text_model_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str | None = None,
    catalog: PricingCatalog | None = None,
) -> float:
    """Estimate text-model cost from the canonical pricing catalog."""

    pricing = catalog or load_pricing_catalog()
    resolved_provider, resolved_model, _ = pricing.resolve_model(model, provider)
    cost, _, _ = pricing.estimate_tokens(
        resolved_provider,
        resolved_model,
        input_tokens,
        output_tokens,
    )
    return cost


# Artifact findings are truncated to this many characters before the
# paper-matching keypoint prompt (paper_matching.matcher keeps the same value).
PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP = 24_000

# Headroom between a locally counted chunk and the provider's billed prompt.
# On the ICML 2026 07A run the provider's tokenizer billed 3.94-5.16 chars per
# token against the local counter's ~4.5, i.e. up to ~14% more tokens than we
# counted, so a chunk sized right at the threshold would still cross it.
TIER_CEILING_TOKENIZER_HEADROOM = 0.20
# Fixed allowance for the instructions, schema hint, and headers wrapped
# around the evidence content of each chunk prompt.
TIER_CEILING_PROMPT_SCAFFOLD_TOKENS = 4_000


@dataclass(frozen=True)
class InputCeiling:
    """A per-call input-token ceiling and why it was chosen."""

    max_input_tokens: int
    source: str  # "explicit" | "pricing_catalog_high_tier" | "provider_default"
    reason: str
    high_tier_input_token_threshold: int | None = None

    def to_receipt(self) -> dict[str, Any]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_input_tokens_source": self.source,
            "max_input_tokens_reason": self.reason,
            "high_tier_input_token_threshold": self.high_tier_input_token_threshold,
        }


def tier_aware_input_ceiling(
    *,
    model: str,
    provider: str | None,
    provider_default: int,
    explicit: int | None = None,
    catalog: PricingCatalog | None = None,
    headroom: float = TIER_CEILING_TOKENIZER_HEADROOM,
    scaffold_tokens: int = TIER_CEILING_PROMPT_SCAFFOLD_TOKENS,
) -> InputCeiling:
    """Choose a chunk ceiling that keeps each call below the model's high price tier.

    The threshold is read from the pricing catalog for the selected model, so
    no model name is hardcoded here. An explicit ceiling always wins (and the
    reason says whether it crosses the tier); without a tier the provider
    default is used unchanged.
    """

    pricing = catalog or load_pricing_catalog()
    threshold = pricing.high_tier_input_token_threshold(model, provider)
    if explicit is not None:
        explicit = int(explicit)
        if explicit <= 0:
            raise ValueError("max_input_tokens must be positive")
        if threshold is not None and explicit > threshold:
            reason = (
                f"explicit --max-input-tokens {explicit:,} exceeds the {threshold:,}-token "
                "high-tier threshold; chunks near the ceiling bill at the high tier"
            )
        else:
            reason = f"explicit --max-input-tokens {explicit:,}"
        return InputCeiling(explicit, "explicit", reason, threshold)
    if threshold is None:
        return InputCeiling(
            int(provider_default),
            "provider_default",
            "pricing catalog lists no high-tier input threshold for this model; provider default used",
            None,
        )
    tier_ceiling = int(threshold * (1.0 - float(headroom))) - int(scaffold_tokens)
    tier_ceiling = max(1_000, tier_ceiling)
    ceiling = min(int(provider_default), tier_ceiling)
    reason = (
        f"kept below the {threshold:,}-token high-tier threshold from the pricing catalog "
        f"({headroom:.0%} tokenizer headroom, {int(scaffold_tokens):,}-token prompt scaffold)"
    )
    return InputCeiling(ceiling, "pricing_catalog_high_tier", reason, threshold)


def estimate_openai_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate OpenAI vision input tokens using the published tiled image rule."""

    if width <= 0 or height <= 0:
        return 0
    if detail == "low":
        return 85

    max_side = max(width, height)
    if max_side > 2048:
        scale = 2048 / max_side
        width = max(1, math.floor(width * scale))
        height = max(1, math.floor(height * scale))

    min_side = min(width, height)
    if min_side > 768:
        scale = 768 / min_side
        width = max(1, math.floor(width * scale))
        height = max(1, math.floor(height * scale))

    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + tiles * 170


def estimate_anthropic_image_tokens(width: int, height: int) -> int:
    """Approximate Anthropic vision tokens from pixel count."""

    if width <= 0 or height <= 0:
        return 0
    return math.ceil((width * height) / 750)


def _scenario_value(value: Any, scenario: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        if scenario in value:
            return value[scenario]
        if DEFAULT_SCENARIO in value:
            return value[DEFAULT_SCENARIO]
        if value:
            return next(iter(value.values()))
        return default
    if value is None:
        return default
    return value


def _run_text_command(args: list[str], timeout: float = 8.0) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _audio_duration_seconds(path: Path) -> float | None:
    ffprobe_output = _run_text_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=10.0,
    )
    if ffprobe_output:
        try:
            value = float(ffprobe_output.splitlines()[-1])
            if value > 0:
                return value
        except ValueError:
            pass

    if platform.system() == "Darwin":
        mdls_output = _run_text_command(
            ["mdls", "-name", "kMDItemDurationSeconds", "-raw", str(path)],
            timeout=5.0,
        )
        if mdls_output and mdls_output != "(null)":
            try:
                value = float(mdls_output.splitlines()[-1])
                if value > 0:
                    return value
            except ValueError:
                pass
    return None


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        pass

    output = _run_text_command(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)])
    if not output:
        return None
    width: int | None = None
    height: int | None = None
    for line in output.splitlines():
        if "pixelWidth:" in line:
            width = int(line.rsplit(":", 1)[1].strip())
        if "pixelHeight:" in line:
            height = int(line.rsplit(":", 1)[1].strip())
    if width and height:
        return width, height
    return None


def _pdf_page_count(path: Path) -> int | None:
    if platform.system() == "Darwin":
        output = _run_text_command(["mdls", "-name", "kMDItemNumberOfPages", "-raw", str(path)])
        if output and output != "(null)":
            try:
                return int(float(output.splitlines()[-1]))
            except ValueError:
                pass
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def _is_under(path: Path, part: str) -> bool:
    return part in path.parts


def _unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen and path.exists():
            seen.add(resolved)
            unique.append(path)
    return unique


def _discover_audio_paths(workspace: Path) -> list[Path]:
    patterns = [
        "cache/acquisition-downloads/media/**/audio/raw.*",
        "storage/acquisition-downloads/media/**/audio/raw.*",
        "media/**/audio/raw.*",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in workspace.glob(pattern) if path.suffix.lower() in AUDIO_EXTENSIONS)
    return _unique_existing(paths)


def _discover_image_paths(workspace: Path) -> list[Path]:
    patterns = [
        "cache/acquisition-downloads/media/**/poster/*",
        "storage/acquisition-downloads/media/**/poster/*",
        "slides/**/*",
        "decks/**/*",
        "presentations/**/*",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in workspace.glob(pattern) if path.suffix.lower() in IMAGE_EXTENSIONS)
    return _unique_existing(paths)


def _discover_pdf_paths(workspace: Path) -> list[Path]:
    patterns = [
        "cache/acquisition-downloads/media/**/*.pdf",
        "storage/acquisition-downloads/media/**/*.pdf",
        "papers/**/*.pdf",
        "storage/corpus/**/*.pdf",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in workspace.glob(pattern) if path.is_file())
    return _unique_existing(paths)


def _discover_text_paths(workspace: Path) -> list[Path]:
    candidate_roots = [
        workspace / "analysis",
        workspace / "storage" / "corpus",
        workspace / "storage" / "transcription-ocr",
        workspace / "transcripts",
        workspace / "papers",
        workspace / "decks",
        workspace / "presentations",
    ]
    paths: list[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*") if path.suffix.lower() in TEXT_EXTENSIONS)
    return _unique_existing(paths)


def _count_topics(workspace: Path) -> int:
    manifest_candidates = [
        workspace / "analysis" / "analysis_manifest.json",
        workspace / "storage" / "analysis" / "analysis_manifest.json",
    ]
    for manifest in manifest_candidates:
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        topics = data.get("topics") or data.get("reports") or data.get("analysis_topics")
        if isinstance(topics, list):
            return len(topics)
        if isinstance(topics, dict):
            return len(topics)

    analysis_dir = workspace / "analysis"
    if analysis_dir.exists():
        topic_docs = [
            path
            for path in analysis_dir.glob("[0-9][0-9]_*.md")
            if not path.name.endswith("_checklist.md")
        ]
        if topic_docs:
            return len(topic_docs)
    return 0


def _count_presentation_targets(workspace: Path, topic_count: int) -> int:
    manifest_candidates = [
        workspace / "presentations" / "presentation_manifest.json",
        workspace / "storage" / "presentation" / "presentation_manifest.json",
        workspace / "storage" / "presentations" / "presentation_manifest.json",
    ]
    for manifest in manifest_candidates:
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        presentations = data.get("presentations") or data.get("items") or data.get("topics")
        if isinstance(presentations, list):
            return len(presentations)
        if isinstance(presentations, dict):
            return len(presentations)

    presentation_dirs = [
        path
        for path in (workspace / "presentations").glob("*")
        if path.is_dir()
    ] if (workspace / "presentations").exists() else []
    if presentation_dirs:
        return len(presentation_dirs)
    return topic_count


def _count_source_discovery_candidate_pairs(workspace: Path) -> int:
    """Count only the bounded frontier pairs in the canonical control inventory."""

    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    if not inventory_path.exists():
        return 0
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source-discovery inventory is unreadable for join costing") from exc
    if isinstance(inventory, dict) and "candidate_frontier" not in inventory:
        # An inventory written before the join frontier existed (schema without
        # the key, e.g. ICML 2026's July run): discovery finished with no LLM
        # join step, so there are no pending pairs to price. A frontier that is
        # present but malformed still refuses below.
        return 0
    frontier = inventory.get("candidate_frontier") if isinstance(inventory, dict) else None
    pairs = frontier.get("pairs") if isinstance(frontier, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("source-discovery inventory is missing candidate frontier pairs for join costing")
    if not all(isinstance(pair, dict) and isinstance(pair.get("canonical_pair_key"), str) for pair in pairs):
        raise ValueError("source-discovery candidate frontier has an invalid join pair")
    # Only pending_llm pairs are ever billable: "included" pairs were resolved
    # deterministically (same-corpus title join) and "excluded" pairs never
    # reach the LLM boundary at all -- counting anything else as billable
    # overstates the join cost by every deterministically-resolved pair.
    return sum(pair.get("inclusion_status") == "pending_llm" for pair in pairs)


def _count_tavily_priority_gap_queries(workspace: Path) -> int:
    """Return only a complete digest-bound Stage-4 priority-gap frontier.

    The public harvest writes this control record before any paid escalation.
    Missing control data means no paid task is estimated; malformed or changed
    data is an error, never a convenient fallback to a broad search count.
    """

    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    if not ledger_path.exists():
        return 0
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Tavily priority-gap ledger is unreadable for costing") from exc
    talk_harvest = ledger.get("talk_harvest") if isinstance(ledger, dict) else None
    frontier = talk_harvest.get("priority_gap_frontier") if isinstance(talk_harvest, dict) else None
    if frontier is None:
        return 0
    if not isinstance(frontier, dict):
        raise ValueError("Tavily priority-gap frontier is invalid")
    required = {
        "program_spine_sha256",
        "harvest_inventory_sha256",
        "query_policy_sha256",
        "priority_program_item_ids",
        "priority_gap_quantity",
        "priority_gap_frontier_sha256",
    }
    if not required.issubset(frontier):
        raise ValueError("Tavily priority-gap frontier is incomplete")
    item_ids = frontier.get("priority_program_item_ids")
    quantity = frontier.get("priority_gap_quantity")
    if (
        not isinstance(item_ids, list)
        or not all(isinstance(item_id, str) and item_id for item_id in item_ids)
        or item_ids != sorted(item_ids)
        or len(set(item_ids)) != len(item_ids)
        or not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity != len(item_ids)
        or quantity < 0
    ):
        raise ValueError("Tavily priority-gap frontier has an invalid quantity")
    core = {
        key: frontier[key]
        for key in (
            "program_spine_sha256",
            "harvest_inventory_sha256",
            "query_policy_sha256",
            "priority_program_item_ids",
            "priority_gap_quantity",
        )
    }
    digest = __import__("hashlib").sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if frontier.get("priority_gap_frontier_sha256") != digest:
        raise ValueError("Tavily priority-gap frontier digest is incompatible")
    return quantity


# Conservative chars-per-token for sizing prompts from character counts: the
# provider tokenizer billed 3.94-5.16 chars/token on the ICML 2026 summary pass.
PROMPT_CHARS_PER_TOKEN = 4.0
# Fixed per-unit framing in an exhaustive-summary evidence block
# ("### Evidence Unit", the remaining JSON header keys, "TEXT:").
_SUMMARY_BLOCK_FRAMING_CHARS = 250
# Mirrors the full-read input kinds the summary owner groups into categories
# (analysis.exhaustive_summaries.KIND_BY_CATEGORY).
_SUMMARY_CATEGORY_BY_KIND = {
    "transcript_segment": "transcripts",
    "paper_abstract": "papers",
}


def _latest_dir(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_dir()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, path.name))


def _exhaustive_summary_pair_tokens(workspace: Path) -> tuple[list[int], str]:
    """Size the exhaustive topic-summary pass per topic/category pair.

    Prefers the latest summary plan (exact, counted by the summary owner; a
    dry run writes it for free), then the completed llm-full-read by-topic and
    by-unit records. Returns ``([], "")`` when neither exists.
    """

    plans = sorted(
        workspace.glob("storage/analysis/exhaustive-summaries/*/summary_plan.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    for plan_path in reversed(plans):
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pairs = plan.get("pairs") if isinstance(plan, dict) else None
        if not isinstance(pairs, list):
            continue
        tokens = [
            int(pair["input_tokens"])
            for pair in pairs
            if isinstance(pair, dict)
            and isinstance(pair.get("input_tokens"), int)
            and not isinstance(pair.get("input_tokens"), bool)
            and pair["input_tokens"] > 0
        ]
        if tokens:
            return tokens, "summary_plan"

    run_dir = _latest_dir(
        [path.parent for path in workspace.glob("storage/extractions/*/llm-full-read/by-topic")]
    )
    if run_dir is None:
        return [], ""
    by_unit = run_dir / "by-unit"
    unit_chars: dict[str, int] = {}

    def _unit_block_chars(unit_id: str, item: Mapping[str, Any]) -> int:
        if unit_id in unit_chars:
            return unit_chars[unit_id]
        from conference_pipeline.extractions.records import safe_artifact_filename

        record: Any = None
        try:
            record = json.loads(
                (by_unit / f"{safe_artifact_filename(unit_id)}.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            record = None
        source = record if isinstance(record, dict) else item
        text_length = source.get("text_length")
        if not isinstance(text_length, int) or isinstance(text_length, bool):
            text_length = 0
        from conference_pipeline.analysis.exhaustive_summaries import _metadata_summary

        metadata = source.get("metadata")
        header_chars = len(
            json.dumps(
                {
                    "unit_id": unit_id,
                    "source_artifact_id": source.get("source_artifact_id"),
                    "source_artifact_path": source.get("source_artifact_path"),
                    "metadata": _metadata_summary(metadata) if isinstance(metadata, Mapping) else {},
                    "summary": source.get("summary"),
                    "reason": source.get("reason"),
                    "evidence_phrases": source.get("evidence_phrases"),
                },
                ensure_ascii=True,
                default=str,
            )
        )
        size = text_length + header_chars + _SUMMARY_BLOCK_FRAMING_CHARS
        unit_chars[unit_id] = size
        return size

    pair_chars: dict[tuple[str, str], int] = {}
    for topic_path in sorted((run_dir / "by-topic").glob("*.json")):
        try:
            payload = json.loads(topic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            category = _SUMMARY_CATEGORY_BY_KIND.get(str(item.get("input_kind") or ""))
            unit_id = str(item.get("unit_id") or "").strip()
            if category is None or not unit_id:
                continue
            key = (topic_path.stem, category)
            pair_chars[key] = pair_chars.get(key, 0) + _unit_block_chars(unit_id, item)
    tokens = [
        math.ceil(chars / PROMPT_CHARS_PER_TOKEN)
        for _, chars in sorted(pair_chars.items())
        if chars > 0
    ]
    return (tokens, "llm_full_read_records") if tokens else ([], "")


def _paper_matching_keypoint_input_tokens(
    workspace: Path, topic_count: int
) -> tuple[list[int], str]:
    """Size one keypoint-extraction call per topic.

    Uses the latest artifact findings per topic when paper matching has run,
    and otherwise the extractor's findings cap (the prompt is truncated to it)
    for each counted topic. Returns ``([], "")`` when no topic is known.
    """

    char_cap = PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP
    findings: dict[str, int] = {}
    paths = sorted(
        workspace.glob("storage/paper-matching/*/*/artifact_gpt_researcher_findings.md"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    for path in paths:
        try:
            findings[path.parent.name] = min(len(path.read_text(encoding="utf-8")), char_cap)
        except OSError:
            continue
    if findings:
        tokens = [math.ceil(chars / PROMPT_CHARS_PER_TOKEN) for _, chars in sorted(findings.items())]
        missing = max(topic_count - len(tokens), 0)
        tokens.extend([math.ceil(char_cap / PROMPT_CHARS_PER_TOKEN)] * missing)
        return tokens, "paper_matching_findings"
    if topic_count > 0:
        return [math.ceil(char_cap / PROMPT_CHARS_PER_TOKEN)] * topic_count, "topic_count_x_findings_cap"
    return [], ""


def collect_workspace_workload(workspace: Path) -> Workload:
    resolved = workspace.resolve()
    workload = Workload(workspace=str(resolved))
    if not workspace.exists():
        workload.warnings.append(f"Workspace does not exist: {workspace}")
        return workload

    audio_paths = _discover_audio_paths(workspace)
    workload.audio_files = len(audio_paths)
    for path in audio_paths:
        duration = _audio_duration_seconds(path)
        if duration is None:
            workload.audio_duration_missing_files += 1
            continue
        workload.audio_minutes += duration / 60

    image_paths = _discover_image_paths(workspace)
    workload.image_files = len(image_paths)
    for path in image_paths:
        dimensions = _image_dimensions(path)
        if dimensions is None:
            workload.warnings.append(f"Could not read image dimensions for {path}")
            continue
        width, height = dimensions
        workload.openai_image_tokens += estimate_openai_image_tokens(width, height)
        workload.anthropic_image_tokens += estimate_anthropic_image_tokens(width, height)

    pdf_paths = _discover_pdf_paths(workspace)
    workload.pdf_files = len(pdf_paths)
    for path in pdf_paths:
        page_count = _pdf_page_count(path)
        if page_count is None:
            workload.warnings.append(f"Could not read PDF page count for {path}")
            continue
        workload.pdf_pages += page_count

    text_paths = _discover_text_paths(workspace)
    workload.text_files = len(text_paths)
    for path in text_paths:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            workload.warnings.append(f"Could not read text file for token estimate: {path}")
            continue
        workload.text_chars += len(text)
    workload.text_tokens = math.ceil(workload.text_chars / 4)

    workload.topic_count = _count_topics(workspace)
    workload.presentation_targets = _count_presentation_targets(workspace, workload.topic_count)
    workload.source_discovery_candidate_pairs = _count_source_discovery_candidate_pairs(workspace)
    workload.tavily_priority_gap_queries = _count_tavily_priority_gap_queries(workspace)
    (
        workload.exhaustive_summary_pair_tokens,
        workload.exhaustive_summary_quantity_basis,
    ) = _exhaustive_summary_pair_tokens(workspace)
    (
        workload.paper_matching_keypoint_input_tokens,
        workload.paper_matching_keypoint_quantity_basis,
    ) = _paper_matching_keypoint_input_tokens(workspace, workload.topic_count)
    return workload


def _scope_includes(task: dict[str, Any], scope: str) -> bool:
    if scope == "full":
        return True
    normalized = scope.replace("_", "-")
    return normalized in {
        str(task.get("stage", "")).replace("_", "-"),
        str(task.get("family", "")).replace("_", "-"),
        str(task.get("task_id", "")).replace("_", "-"),
    }


def _enabled_for_scenario(
    task: dict[str, Any], scenario: str, defaults: dict[str, Any], workload: Workload
) -> bool:
    enabled_scenarios = task.get("enabled_scenarios")
    if enabled_scenarios and scenario not in set(enabled_scenarios):
        return False
    if task.get("quantity_source") == "pdf_pages_as_images":
        return bool(defaults.get("pdf_visual_ocr_enabled", False))
    if task.get("quantity_source") == "source_discovery_candidate_pairs":
        return workload.source_discovery_candidate_pairs > 0
    if task.get("quantity_source") == "tavily_priority_gap_queries":
        return workload.tavily_priority_gap_queries > 0
    return True


def _task_model(task: dict[str, Any], scenario: str) -> str:
    return str(_scenario_value(task.get("models", {}), scenario, "local-only"))


def _base_quantity_count(workload: Workload, quantity_source: str) -> int:
    if quantity_source == "topic_count":
        return max(workload.topic_count, 1)
    if quantity_source == "presentation_targets":
        return max(workload.presentation_targets, workload.topic_count, 1)
    return 1


def _estimate_task(
    task: dict[str, Any],
    scenario: str,
    defaults: dict[str, Any],
    workload: Workload,
    catalog: PricingCatalog,
) -> CostEstimate:
    task_id = str(task.get("task_id"))
    provider = str(task.get("provider", "local"))
    model = _task_model(task, scenario)
    pricing_unit = str(task.get("pricing_unit", "tokens"))
    quantity_source = str(task.get("quantity_source", "fixed"))
    enabled = _enabled_for_scenario(task, scenario, defaults, workload)
    retry_multiplier = float(defaults.get("retry_multiplier", 1.0))

    estimate = CostEstimate(
        task_id=task_id,
        family=str(task.get("family", task_id)),
        stage=str(task.get("stage", "unknown")),
        provider=provider,
        model=model,
        pricing_unit=pricing_unit,
        enabled=enabled,
        quantity=0.0,
        assumptions=[],
    )
    if task.get("notes"):
        estimate.assumptions.append(str(task["notes"]))
    if not enabled:
        if task.get("quantity_source") == "source_discovery_candidate_pairs":
            estimate.warnings.append("Disabled because no candidate frontier pairs are available.")
            return estimate
        if task.get("quantity_source") == "tavily_priority_gap_queries":
            estimate.warnings.append("Disabled because no priority-gap frontier is available.")
            return estimate
        estimate.warnings.append(f"Disabled in {scenario} scenario.")
        return estimate

    try:
        if pricing_unit == "none":
            estimate.quantity = 1
            estimate.estimated_cost_usd = float(task.get("fixed_usd", 0.0))
            return estimate

        if quantity_source == "audio_minutes":
            minutes = workload.audio_minutes * retry_multiplier
            estimate.quantity = minutes
            estimate.audio_minutes = minutes
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_audio_minutes(
                provider, model, minutes
            )
            estimate.warnings.extend(warnings)
            if workload.audio_duration_missing_files:
                estimate.warnings.append(
                    f"{workload.audio_duration_missing_files} audio files lacked readable duration metadata."
                )
            return estimate

        if quantity_source == "openai_image_tokens":
            output_per_image = int(defaults.get("ocr_output_tokens_per_image", 1000))
            fallback_share = task.get("fallback_share")
            image_share = float(_scenario_value(fallback_share, scenario, 1.0))
            estimate.input_tokens = int(workload.openai_image_tokens * retry_multiplier * image_share)
            estimate.output_tokens = int(
                workload.image_files * output_per_image * retry_multiplier * image_share
            )
            estimate.quantity = workload.image_files * image_share
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            if fallback_share is not None:
                estimate.assumptions.append(f"Estimated fallback share: {image_share:.0%}.")
            return estimate

        if quantity_source == "pdf_pages_as_images":
            output_per_page = int(defaults.get("ocr_output_tokens_per_image", 1000))
            image_tokens_per_page = int(task.get("image_tokens_per_page", 1105))
            estimate.input_tokens = int(workload.pdf_pages * image_tokens_per_page * retry_multiplier)
            estimate.output_tokens = int(workload.pdf_pages * output_per_page * retry_multiplier)
            estimate.quantity = workload.pdf_pages
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            return estimate

        if quantity_source == "text_tokens":
            multiplier = float(task.get("input_multiplier", 1.0))
            estimate.input_tokens = int(workload.text_tokens * multiplier * retry_multiplier)
            estimate.output_tokens = int(float(task.get("output_tokens", 0)) * retry_multiplier)
            estimate.quantity = estimate.input_tokens
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            return estimate

        if quantity_source == "exhaustive_extraction_tokens":
            fallback_input = int(task.get("fallback_input_tokens", 0))
            fallback_output = int(task.get("fallback_output_tokens", 0))
            estimate.input_tokens = int(max(workload.text_tokens, fallback_input) * retry_multiplier)
            estimate.output_tokens = int(fallback_output * retry_multiplier)
            estimate.quantity = estimate.input_tokens
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            if workload.text_tokens < fallback_input:
                estimate.assumptions.append(
                    f"Uses fallback corpus budget of {fallback_input:,} input tokens."
                )
            return estimate

        if quantity_source == "exhaustive_extraction_failover_tokens":
            share = float(_scenario_value(task.get("failover_share", {}), scenario, 0.1))
            fallback_input = 26_000_000
            fallback_output = 1_600_000
            estimate.input_tokens = int(max(workload.text_tokens, fallback_input) * share * retry_multiplier)
            estimate.output_tokens = int(fallback_output * share * retry_multiplier)
            estimate.quantity = share
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            estimate.assumptions.append(f"OpenRouter failover share: {share:.0%}.")
            return estimate

        if quantity_source == "embedding_tokens":
            multiplier = float(task.get("input_multiplier", 1.0))
            estimate.input_tokens = int(workload.text_tokens * multiplier)
            estimate.output_tokens = 0
            estimate.quantity = estimate.input_tokens
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            return estimate

        if quantity_source in {"topic_count", "presentation_targets"}:
            unit_count = _base_quantity_count(workload, quantity_source)
            run_count = int(_scenario_value(task.get("run_count"), scenario, 1))
            effective_units = unit_count * run_count
            estimate.quantity = effective_units

            if pricing_unit == "credits":
                credits_per_unit = float(_scenario_value(task.get("credits_per_unit"), scenario, 0))
                estimate.credits = effective_units * credits_per_unit
                estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_credits(
                    provider, model, estimate.credits
                )
                estimate.warnings.extend(warnings)
                return estimate

            if pricing_unit == "tts_minute_estimate":
                target_minutes = float(_scenario_value(task.get("target_minutes_per_unit"), scenario, 0))
                estimate.tts_minutes = effective_units * target_minutes
                estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tts_minutes(
                    provider, model, estimate.tts_minutes
                )
                estimate.warnings.extend(warnings)
                return estimate

            input_per_unit = int(task.get("input_tokens_per_unit", 0))
            output_per_unit = int(task.get("output_tokens_per_unit", 0))
            estimate.input_tokens = int(effective_units * input_per_unit * retry_multiplier)
            estimate.output_tokens = int(effective_units * output_per_unit * retry_multiplier)
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            if run_count > 1:
                estimate.assumptions.append(f"Repeated generation count: {run_count}.")
            return estimate

        if quantity_source == "source_discovery_candidate_pairs":
            max_quantity = int(task.get("max_quantity", 0))
            if max_quantity <= 0:
                raise ValueError("source-discovery join projection requires a positive max_quantity")
            effective_units = workload.source_discovery_candidate_pairs
            if effective_units > max_quantity:
                raise ValueError(
                    "source-discovery candidate frontier exceeds the bounded join quantity"
                )
            estimate.quantity = effective_units
            input_per_unit = int(task.get("input_tokens_per_unit", 0))
            output_per_unit = int(task.get("output_tokens_per_unit", 0))
            if input_per_unit <= 0 or output_per_unit <= 0:
                raise ValueError("source-discovery join projection requires finite token bounds")
            estimate.input_tokens = int(effective_units * input_per_unit * retry_multiplier)
            estimate.output_tokens = int(effective_units * output_per_unit * retry_multiplier)
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_tokens(
                provider, model, estimate.input_tokens, estimate.output_tokens
            )
            estimate.warnings.extend(warnings)
            if not math.isfinite(estimate.estimated_cost_usd):
                raise ValueError("source-discovery join projection must be finite")
            estimate.assumptions.append(f"Bounded candidate frontier: {effective_units}/{max_quantity} pairs.")
            return estimate

        if quantity_source == "tavily_priority_gap_queries":
            effective_units = workload.tavily_priority_gap_queries
            if effective_units <= 0:
                raise ValueError("Tavily priority-gap projection requires a positive exact quantity")
            if pricing_unit != "credits":
                raise ValueError("Tavily priority-gap projection requires credit pricing")
            credits_per_unit = float(_scenario_value(task.get("credits_per_unit"), scenario, 0))
            if not math.isfinite(credits_per_unit) or credits_per_unit <= 0:
                raise ValueError("Tavily priority-gap projection requires finite credit pricing")
            estimate.quantity = effective_units
            estimate.credits = effective_units * credits_per_unit
            estimate.estimated_cost_usd, estimate.source_urls, warnings = catalog.estimate_credits(
                provider, model, estimate.credits
            )
            estimate.warnings.extend(warnings)
            if not math.isfinite(estimate.estimated_cost_usd) or estimate.estimated_cost_usd < 0:
                raise ValueError("Tavily priority-gap projection must be finite")
            estimate.assumptions.append(
                f"Exact priority-gap frontier: {effective_units} governed Tavily queries."
            )
            return estimate

        if quantity_source == "fixed":
            estimate.quantity = 1
            estimate.estimated_cost_usd = float(task.get("fixed_usd", 0.0))
            return estimate

        if quantity_source == "exhaustive_summary_tokens":
            return _estimate_exhaustive_summary(
                estimate, task, workload, catalog, retry_multiplier=retry_multiplier
            )

        if quantity_source == "paper_matching_keypoint_tokens":
            return _estimate_paper_matching_keypoints(
                estimate, task, workload, catalog, retry_multiplier=retry_multiplier
            )

        if task_id in {"source_discovery_cross_source_join", "cvpr2026_tavily_priority_gap_search"}:
            raise ValueError("source-discovery join projection has an unknown quantity source")
        if provider != "local":
            # A paid task must never be summed into totals at a silent $0.
            return _price_declared_fallback_budget(
                estimate,
                task,
                catalog,
                retry_multiplier=retry_multiplier,
                reason=f"unknown quantity source {quantity_source!r}",
            )
        estimate.warnings.append(f"Unknown quantity source: {quantity_source}")
        return estimate
    except KeyError as exc:
        if quantity_source in {"exhaustive_summary_tokens", "paper_matching_keypoint_tokens"}:
            raise ValueError(f"{task_id}: projection is unpriced ({exc})") from exc
        if task_id == "source_discovery_cross_source_join":
            raise ValueError("source-discovery join projection is unpriced") from exc
        if task_id == "cvpr2026_tavily_priority_gap_search":
            raise ValueError("Tavily priority-gap projection is unpriced") from exc
        estimate.warnings.append(str(exc))
        return estimate


UNMEASURED_WARNING_PREFIX = "UNMEASURED QUANTITY"


def _price_declared_fallback_budget(
    estimate: CostEstimate,
    task: Mapping[str, Any],
    catalog: PricingCatalog,
    *,
    retry_multiplier: float,
    reason: str,
    calls: int = 1,
) -> CostEstimate:
    """Price a paid task's declared fallback budget, loudly, or refuse.

    Used when the workspace cannot size the task. A spec without a declared
    budget is refused rather than summed into totals as $0.
    """

    fallback_input = int(task.get("fallback_input_tokens") or 0)
    fallback_output = int(task.get("fallback_output_tokens") or 0)
    if fallback_input <= 0:
        raise ValueError(
            f"{estimate.task_id}: paid task quantity cannot be derived ({reason}) and the "
            "task spec declares no fallback_input_tokens budget; refusing to price it at $0"
        )
    calls = max(int(calls), 1)
    per_call_input = math.ceil(fallback_input / calls)
    per_call_output = math.ceil(fallback_output / calls)
    total = 0.0
    sources: list[str] = []
    warnings: list[str] = []
    for _ in range(calls):
        cost, sources, warnings = catalog.estimate_tokens(
            estimate.provider, estimate.model, per_call_input, per_call_output
        )
        total += cost
    estimate.quantity_basis = "declared_fallback_budget"
    estimate.quantity = calls
    estimate.input_tokens = int(per_call_input * calls * retry_multiplier)
    estimate.output_tokens = int(per_call_output * calls * retry_multiplier)
    estimate.estimated_cost_usd = total * retry_multiplier
    estimate.source_urls = sources
    estimate.warnings.extend(dict.fromkeys(warnings))
    estimate.warnings.insert(
        0,
        f"{UNMEASURED_WARNING_PREFIX}: {reason}; priced the task spec's declared fallback "
        f"budget ({fallback_input:,} input / {fallback_output:,} output tokens over "
        f"{calls} call(s)) instead of a workspace measurement.",
    )
    return estimate


def _resolved_provider(catalog: PricingCatalog, model: str, provider: str) -> str:
    try:
        return catalog.resolve_model(model, provider)[0]
    except KeyError:
        return provider


def _estimate_exhaustive_summary(
    estimate: CostEstimate,
    task: Mapping[str, Any],
    workload: Workload,
    catalog: PricingCatalog,
    *,
    retry_multiplier: float,
) -> CostEstimate:
    """Price the exhaustive topic-summary pass call by call.

    Each topic/category pair is chunked at the same tier-aware ceiling the
    summary owner uses by default, and every chunk (plus one merge call per
    multi-chunk pair) is priced per request, so the high-tier split is real.
    """

    provider = _resolved_provider(catalog, estimate.model, estimate.provider)
    provider_defaults = task.get("provider_default_max_input_tokens") or {}
    provider_default = int(
        provider_defaults.get(provider, provider_defaults.get("default", 400_000))
        if isinstance(provider_defaults, Mapping)
        else provider_defaults
    )
    ceiling = tier_aware_input_ceiling(
        model=estimate.model,
        provider=provider,
        provider_default=provider_default,
        catalog=catalog,
    )
    # Plan sizes are local token counts and need the provider-tokenizer
    # inflation; record-based sizes are already provider-scale (chars / 4).
    inflation = (
        float(task.get("provider_token_inflation", 1.0))
        if workload.exhaustive_summary_quantity_basis == "summary_plan"
        else 1.0
    )
    scaffold = int(task.get("prompt_scaffold_tokens_per_call", 0))
    output_per_call = int(task.get("output_tokens_per_call", 0))
    if output_per_call <= 0:
        raise ValueError(f"{estimate.task_id}: output_tokens_per_call must be positive")
    estimate.assumptions.append(f"Chunk ceiling: {ceiling.max_input_tokens:,} tokens ({ceiling.reason}).")
    pairs = list(workload.exhaustive_summary_pair_tokens)
    if not pairs:
        calls = max(math.ceil(int(task.get("fallback_input_tokens") or 0) / ceiling.max_input_tokens), 1)
        return _price_declared_fallback_budget(
            estimate,
            task,
            catalog,
            retry_multiplier=retry_multiplier,
            reason=(
                "no summary plan or llm-full-read records in the workspace "
                "(a free --mode dry-run of scripts/summarize_llm_full_read_topics.py sizes it)"
            ),
            calls=calls,
        )
    call_sizes: list[int] = []
    for pair_tokens in pairs:
        chunk_count = max(math.ceil(pair_tokens / ceiling.max_input_tokens), 1)
        per_chunk = math.ceil(pair_tokens / chunk_count)
        call_sizes.extend([math.ceil(per_chunk * inflation) + scaffold] * chunk_count)
        if chunk_count > 1:
            call_sizes.append(chunk_count * output_per_call + scaffold)
    total = 0.0
    over_tier_calls = 0
    over_tier_cost = 0.0
    sources: list[str] = []
    seen_warnings: dict[str, None] = {}
    for size in call_sizes:
        cost, sources, warnings = catalog.estimate_tokens(
            estimate.provider, estimate.model, size, output_per_call
        )
        total += cost
        if (
            ceiling.high_tier_input_token_threshold is not None
            and size > ceiling.high_tier_input_token_threshold
        ):
            over_tier_calls += 1
            over_tier_cost += cost
        for warning in warnings:
            if not warning.startswith("Large-prompt tier applied"):
                seen_warnings[warning] = None
    estimate.quantity = len(call_sizes)
    estimate.input_tokens = int(sum(call_sizes) * retry_multiplier)
    estimate.output_tokens = int(len(call_sizes) * output_per_call * retry_multiplier)
    estimate.estimated_cost_usd = total * retry_multiplier
    estimate.source_urls = sources
    estimate.warnings.extend(seen_warnings)
    estimate.assumptions.append(
        f"Sized from {workload.exhaustive_summary_quantity_basis}: {len(pairs)} topic/category pairs, "
        f"{len(call_sizes)} calls (chunks plus merges), x{inflation:g} provider-tokenizer inflation."
    )
    if ceiling.high_tier_input_token_threshold is not None:
        estimate.assumptions.append(
            f"High-tier split: {over_tier_calls} of {len(call_sizes)} calls above "
            f"{ceiling.high_tier_input_token_threshold:,} input tokens "
            f"(${over_tier_cost * retry_multiplier:,.2f} of ${estimate.estimated_cost_usd:,.2f})."
        )
    return estimate


def _estimate_paper_matching_keypoints(
    estimate: CostEstimate,
    task: Mapping[str, Any],
    workload: Workload,
    catalog: PricingCatalog,
    *,
    retry_multiplier: float,
) -> CostEstimate:
    """Price one keypoint-extraction call per topic from its findings size."""

    scaffold = int(task.get("prompt_scaffold_tokens_per_call", 0))
    output_per_call = int(task.get("output_tokens_per_call", 0))
    if output_per_call <= 0:
        raise ValueError(f"{estimate.task_id}: output_tokens_per_call must be positive")
    sizes = [tokens + scaffold for tokens in workload.paper_matching_keypoint_input_tokens]
    if not sizes:
        return _price_declared_fallback_budget(
            estimate,
            task,
            catalog,
            retry_multiplier=retry_multiplier,
            reason="no topics or paper-matching findings in the workspace",
        )
    total = 0.0
    sources: list[str] = []
    seen_warnings: dict[str, None] = {}
    for size in sizes:
        cost, sources, warnings = catalog.estimate_tokens(
            estimate.provider, estimate.model, size, output_per_call
        )
        total += cost
        seen_warnings.update(dict.fromkeys(warnings))
    estimate.quantity = len(sizes)
    estimate.input_tokens = int(sum(sizes) * retry_multiplier)
    estimate.output_tokens = int(len(sizes) * output_per_call * retry_multiplier)
    estimate.estimated_cost_usd = total * retry_multiplier
    estimate.source_urls = sources
    estimate.warnings.extend(seen_warnings)
    estimate.assumptions.append(
        f"Sized from {workload.paper_matching_keypoint_quantity_basis}: {len(sizes)} keypoint "
        f"calls (one per topic), findings capped at {PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP:,} chars."
    )
    return estimate


def scan_actual_usage(workspace: Path) -> dict[str, Any]:
    """Best-effort receipt scanner; estimates remain separate from invoices."""

    from conference_pipeline.provider_receipts import (
        aggregate_provider_call_receipts,
        scan_provider_call_receipts,
    )

    canonical_receipts = scan_provider_call_receipts(workspace)
    canonical_usage = aggregate_provider_call_receipts(canonical_receipts)

    cost_fields = {"cost_usd", "estimated_cost_usd", "actual_cost_usd", "total_cost_usd"}
    token_fields = {
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
    }
    scanned_files = 0
    cost_values: list[float] = []
    token_totals: dict[str, int] = {field: 0 for field in token_fields}

    for path in (workspace / "storage").rglob("*.json") if (workspace / "storage").exists() else []:
        try:
            if path.stat().st_size > 10_000_000:
                continue
            data = json.loads(path.read_text(errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        scanned_files += 1
        stack = [data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if key in cost_fields and isinstance(value, (int, float)):
                        cost_values.append(float(value))
                    if key in token_fields and isinstance(value, int):
                        token_totals[key] += value
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(current, list):
                stack.extend(item for item in current if isinstance(item, (dict, list)))

    return {
        **canonical_usage,
        "receipt_files_scanned": scanned_files,
        "cost_fields_found": len(cost_values),
        "reported_cost_usd_sum": round(sum(cost_values), 6),
        "token_totals": {key: value for key, value in sorted(token_totals.items()) if value},
        "legacy_scan": {
            "receipt_files_scanned": scanned_files,
            "cost_fields_found": len(cost_values),
            "reported_cost_usd_sum": round(sum(cost_values), 6),
            "token_totals": {key: value for key, value in sorted(token_totals.items()) if value},
        },
        "note": "Canonical provider receipts are precise per-call local estimates; legacy broad scan is retained as a fallback and can double-count nested rollups. Provider invoices remain authoritative.",
    }


def estimate_workspace(
    workspace: Path,
    *,
    scope: str = DEFAULT_SCOPE,
    scenario: str = DEFAULT_SCENARIO,
    mode: str = "estimate",
    pricing_path: Path | None = None,
    task_specs_path: Path | None = None,
) -> EstimateReport:
    if scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario {scenario!r}; expected one of {', '.join(SCENARIOS)}")
    if mode not in {"estimate", "actual"}:
        raise ValueError("mode must be estimate or actual")

    catalog = load_pricing_catalog(pricing_path)
    specs = load_task_specs(task_specs_path)
    workload = collect_workspace_workload(workspace)
    actual_usage = scan_actual_usage(workspace) if mode == "actual" else None
    return _build_report_from_workload(
        workspace,
        workload,
        catalog,
        specs,
        scope=scope,
        scenario=scenario,
        mode=mode,
        actual_usage=actual_usage,
    )


def _build_report_from_workload(
    workspace: Path,
    workload: Workload,
    catalog: PricingCatalog,
    specs: dict[str, Any],
    *,
    scope: str,
    scenario: str,
    mode: str,
    actual_usage: dict[str, Any] | None = None,
) -> EstimateReport:
    scenario_defaults = specs.get("scenario_defaults", {}).get(scenario, {})
    tasks = [
        _estimate_task(task, scenario, scenario_defaults, workload, catalog)
        for task in specs.get("tasks", [])
        if _scope_includes(task, scope)
    ]
    return EstimateReport(
        mode=mode,
        scope=scope,
        scenario=scenario,
        workspace=str(workspace),
        generated_at=datetime.now(timezone.utc).isoformat(),
        pricing_refreshed_at=catalog.refreshed_at,
        workload=workload,
        tasks=tasks,
        actual_usage=actual_usage,
    )


def compare_workspace(
    workspace: Path,
    *,
    scope: str = DEFAULT_SCOPE,
    pricing_path: Path | None = None,
    task_specs_path: Path | None = None,
) -> dict[str, Any]:
    catalog = load_pricing_catalog(pricing_path)
    specs = load_task_specs(task_specs_path)
    workload = collect_workspace_workload(workspace)
    reports = {
        scenario: _build_report_from_workload(
            workspace,
            workload,
            catalog,
            specs,
            scope=scope,
            scenario=scenario,
            mode="estimate",
        )
        for scenario in SCENARIOS
    }
    return {
        "mode": "compare",
        "scope": scope,
        "workspace": str(workspace),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {
            scenario: {
                "total_estimated_cost_usd": round(report.total_estimated_cost_usd, 6),
                "totals_by_provider": report.totals_by_provider(),
                "totals_by_family": report.totals_by_family(),
                "workload": report.workload.to_dict(),
                "tasks": [task.to_dict() for task in report.tasks],
            }
            for scenario, report in reports.items()
        },
    }


def format_report(report: EstimateReport) -> str:
    lines = [
        f"Provider cost estimate ({report.scenario})",
        f"Workspace: {report.workspace}",
        f"Scope: {report.scope}",
        f"Pricing refreshed: {report.pricing_refreshed_at}",
        f"Total: ${report.total_estimated_cost_usd:,.2f}",
        "",
        "By provider:",
    ]
    for provider, total in report.totals_by_provider().items():
        lines.append(f"  {provider}: ${total:,.2f}")
    lines.append("")
    lines.append("By family:")
    for family, total in report.totals_by_family().items():
        lines.append(f"  {family}: ${total:,.2f}")
    lines.append("")
    lines.append("Largest tasks:")
    for task in sorted(report.tasks, key=lambda item: item.estimated_cost_usd, reverse=True)[:10]:
        lines.append(
            f"  {task.task_id} ({task.model}): ${task.estimated_cost_usd:,.2f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m conference_pipeline.costs",
        description="Estimate, reconcile, and compare conference pipeline provider costs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("estimate", "actual"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--workspace", required=True, type=Path)
        command_parser.add_argument("--scope", default=DEFAULT_SCOPE)
        command_parser.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=SCENARIOS)
        command_parser.add_argument("--pricing-catalog", type=Path)
        command_parser.add_argument("--task-specs", type=Path)
        command_parser.add_argument("--json", action="store_true", dest="json_output")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--workspace", required=True, type=Path)
    compare_parser.add_argument("--scope", default=DEFAULT_SCOPE)
    compare_parser.add_argument("--pricing-catalog", type=Path)
    compare_parser.add_argument("--task-specs", type=Path)
    compare_parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        payload = compare_workspace(
            args.workspace,
            scope=args.scope,
            pricing_path=args.pricing_catalog,
            task_specs_path=args.task_specs,
        )
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for scenario, scenario_payload in payload["scenarios"].items():
                print(f"{scenario}: ${scenario_payload['total_estimated_cost_usd']:,.2f}")
        return 0

    report = estimate_workspace(
        args.workspace,
        scope=args.scope,
        scenario=args.scenario,
        mode=args.command,
        pricing_path=args.pricing_catalog,
        task_specs_path=args.task_specs,
    )
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
