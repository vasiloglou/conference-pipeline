"""Configuration management for the NeurIPS pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from conference_pipeline.model_governance import (
    default_fallback_model,
    default_fallback_provider,
    default_model,
    default_model_for_provider,
    default_provider,
    validate_selection,
)
from dotenv import load_dotenv


@dataclass
class DomainRateLimits:
    """Per-domain rate limiting configuration."""

    neurips: int = 3  # Max concurrent requests to neurips.cc
    slideslive: int = 3  # Max concurrent requests to slideslive.com
    arxiv: int = 3  # Max concurrent requests to arxiv.org
    crossref: int = 5  # Max concurrent requests to crossref.org
    semantic_scholar: int = 5  # Max concurrent requests to semanticscholar.org
    default: int = 4  # Default for other domains


@dataclass
class PipelineSettings:
    """Main configuration for the pipeline."""

    # Conference settings
    year: int = 2025
    neurips_base_url: str = "https://neurips.cc"  # conference-literal-ok: historical NeurIPS fallback; config-driven via load_settings(config_path=...)
    virtual_base_url: str = "https://neurips.cc/virtual/2025"  # conference-literal-ok: historical NeurIPS fallback; config-driven via load_settings(config_path=...)
    slideslive_base_url: str = "https://slideslive.com"

    # Storage
    storage_root: Path = field(default_factory=lambda: Path("./data/neurips_2025"))

    # Concurrency and rate limiting
    max_concurrent_requests: int = 4
    domain_rate_limits: DomainRateLimits = field(default_factory=DomainRateLimits)
    request_delay_seconds: float = 0.5
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    # Audio settings
    audio_chunk_max_bytes: int = 23 * 1024 * 1024  # 23 MB for OpenAI limit
    audio_sample_rate: int = 16000
    audio_format: str = "mp3"

    # OCR settings
    ocr_language: str = "eng"
    use_llm_ocr: bool = False
    ocr_provider: Literal["openai", "anthropic", "gemini"] = field(
        default_factory=lambda: default_provider("ocr.slide_image_parse")  # type: ignore[return-value]
    )
    ocr_model: str = field(default_factory=lambda: default_model("ocr.slide_image_parse"))
    ocr_fallback_provider: Literal["none", "openai", "anthropic", "gemini"] = field(
        default_factory=lambda: default_fallback_provider("ocr.slide_image_parse")  # type: ignore[return-value]
    )
    ocr_fallback_model: str = field(
        default_factory=lambda: default_fallback_model("ocr.slide_image_parse")
    )

    # Browser automation
    use_playwright: bool = True
    headless: bool = True
    browser_timeout: int = 30000  # ms

    # Transcription
    transcription_provider: Literal["openai", "anthropic", "local", "external"] = "openai"
    openai_model: str = field(default_factory=lambda: default_model("transcription.session_speech"))
    transcription_timeout_seconds: float = 600.0
    openai_api_key: str = ""

    # Anthropic (for LLM-based OCR, transcription, or reference normalization)
    anthropic_api_key: str = ""
    anthropic_model: str = field(default_factory=lambda: default_model("analysis.source_reading"))
    google_api_key: str = ""

    # NeurIPS credentials
    neurips_email: str = ""
    neurips_password: str = ""

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None
    use_structured_logging: bool = True  # Use structlog if available

    # LLM Reference Normalization
    use_llm_reference_normalization: bool = False

    # LLM-assisted video detection (optional)
    use_llm_video_detection: bool = False
    video_detection_provider: Literal["openai", "anthropic"] = "openai"
    video_detection_model: str = field(default_factory=lambda: default_model("video.session_detection"))

    # External transcription service (e.g., AssemblyAI)
    external_transcription_api_key: str = ""
    external_transcription_base_url: str = ""
    external_transcription_model: str = "default"

    # Local-first routing (v3.0 / LOCAL-02, D-08)
    # One dedicated global switch for the local-first routing resolver. Default
    # ON: the live spine prefers an approved local model, falling back to paid
    # automatically. Actual local use is still gated by the registry's
    # approved_local proof + readiness, so with no host-approved model the
    # pipeline resolves to paid and nothing breaks. This is NOT the
    # local-vision (--local / GPTR_LOCAL_VISION_*) flag.
    local_first_enabled: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.storage_root, str):
            self.storage_root = Path(self.storage_root)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        self.validate_model_policy()

    def validate_model_policy(self) -> None:
        """Validate configured provider/model defaults against the registry."""
        validate_selection(
            "ocr.slide_image_parse",
            provider=self.ocr_provider,
            model=self.ocr_model,
        )
        if self.ocr_fallback_provider != "none":
            validate_selection(
                "ocr.slide_image_parse",
                provider=self.ocr_fallback_provider,
                model=self.ocr_fallback_model,
            )
        if self.transcription_provider == "openai":
            validate_selection(
                "transcription.session_speech",
                provider="openai",
                model=self.openai_model,
            )
        validate_selection(
            "video.session_detection",
            provider=self.video_detection_provider,
            model=self.video_detection_model,
        )

    def get_domain_limit(self, domain: str) -> int:
        """Get rate limit for a specific domain.

        Args:
            domain: Domain name (e.g., 'neurips.cc', 'arxiv.org').

        Returns:
            Maximum concurrent requests for the domain.
        """
        domain_lower = domain.lower()
        from urllib.parse import urlsplit

        conference_host = (urlsplit(self.neurips_base_url).hostname or "").lower()
        conference_host = conference_host.removeprefix("www.")
        if conference_host and conference_host in domain_lower:
            return self.domain_rate_limits.neurips
        if "neurips" in domain_lower:
            return self.domain_rate_limits.neurips
        elif "slideslive" in domain_lower:
            return self.domain_rate_limits.slideslive
        elif "arxiv" in domain_lower:
            return self.domain_rate_limits.arxiv
        elif "crossref" in domain_lower:
            return self.domain_rate_limits.crossref
        elif "semanticscholar" in domain_lower:
            return self.domain_rate_limits.semantic_scholar
        return self.domain_rate_limits.default


def _conference_config_defaults(config_path: str | Path | None) -> dict[str, object]:
    """Read conference base/virtual URL and year from a workspace config.yaml.

    Returns a dict with any of ``base_url``, ``virtual_url``, ``year`` that the
    config declares. Reusable scraper defaults should come from the conference
    workspace, not a hardcoded neurips.cc literal. Missing/unreadable config
    yields an empty dict (callers keep the historical NeurIPS fallback).
    """
    if not config_path:
        return {}
    import yaml

    path = Path(config_path)
    if path.name != "config.yaml":
        path = path / "config.yaml"
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    conference = payload.get("conference") if isinstance(payload.get("conference"), dict) else {}
    urls = payload.get("urls") if isinstance(payload.get("urls"), dict) else {}
    out: dict[str, object] = {}
    base = str(urls.get("base") or "").strip()
    virtual = str(urls.get("virtual") or "").strip()
    if base:
        out["base_url"] = base
    if virtual:
        out["virtual_url"] = virtual
    try:
        out["year"] = int(str(conference.get("year")).strip())
    except (TypeError, ValueError):
        pass
    return out


def load_settings(
    env_file: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> PipelineSettings:
    """Load settings from environment variables and optional .env file.

    Args:
        env_file: Path to .env file. If None, looks for .env in current directory.
        config_path: Optional conference workspace (or config.yaml path). When
            given, conference base/virtual URL and year are sourced from it so
            the legacy scraper serves any conference, not just NeurIPS. Explicit
            environment variables still win over config values.

    Returns:
        Configured PipelineSettings instance.
    """
    # Keep track of operator-supplied environment values before .env expansion.
    # This lets an explicit OCR_PROVIDER override pick that provider's registry
    # default when OCR_MODEL was not explicitly supplied in the shell.
    pre_dotenv_keys = set(os.environ)

    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    def get_int(key: str, default: int) -> int:
        val = os.getenv(key)
        return int(val) if val else default

    def get_float(key: str, default: float) -> float:
        val = os.getenv(key)
        return float(val) if val else default

    def get_bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    def get_path(key: str, default: Path | None) -> Path | None:
        val = os.getenv(key)
        return Path(val) if val else default

    # Load domain rate limits
    domain_limits = DomainRateLimits(
        neurips=get_int("RATE_LIMIT_NEURIPS", 3),
        slideslive=get_int("RATE_LIMIT_SLIDESLIVE", 3),
        arxiv=get_int("RATE_LIMIT_ARXIV", 3),
        crossref=get_int("RATE_LIMIT_CROSSREF", 5),
        semantic_scholar=get_int("RATE_LIMIT_SEMANTIC_SCHOLAR", 5),
        default=get_int("RATE_LIMIT_DEFAULT", 4),
    )

    # Conference identity defaults sourced from the workspace config.yaml when
    # provided, so the legacy scraper is not pinned to NeurIPS. Env vars win.
    conf_defaults = _conference_config_defaults(config_path)
    conf_base_default = str(conf_defaults.get("base_url") or "https://neurips.cc")  # conference-literal-ok: guarded fallback when no workspace config is supplied
    conf_virtual_default = str(conf_defaults.get("virtual_url") or "")
    conf_year_default = int(conf_defaults.get("year") or 2025)

    # Get year first since other settings depend on it
    year = get_int("CONFERENCE_YEAR", 0) or get_int("NEURIPS_YEAR", conf_year_default)
    ocr_provider_default = default_provider("ocr.slide_image_parse")
    ocr_provider = os.getenv("OCR_PROVIDER", ocr_provider_default)
    ocr_model_env = os.getenv("OCR_MODEL")
    if "OCR_PROVIDER" in pre_dotenv_keys and "OCR_MODEL" not in pre_dotenv_keys:
        ocr_model_env = None
    ocr_model = ocr_model_env or default_model_for_provider("ocr.slide_image_parse", ocr_provider)
    ocr_fallback_provider = os.getenv(
        "OCR_FALLBACK_PROVIDER",
        default_fallback_provider("ocr.slide_image_parse"),
    )
    ocr_fallback_model = (
        ""
        if ocr_fallback_provider == "none"
        else os.getenv(
            "OCR_FALLBACK_MODEL",
            default_model_for_provider("ocr.slide_image_parse", ocr_fallback_provider),
        )
    )

    conference_base_url = (
        os.getenv("CONFERENCE_BASE_URL")
        or os.getenv("NEURIPS_BASE_URL")
        or conf_base_default
    )
    from urllib.parse import urlsplit as _urlsplit

    _conf_host = (_urlsplit(conference_base_url).hostname or "neurips.cc")  # conference-literal-ok: guarded fallback host when config lacks a base URL
    virtual_default = conf_virtual_default or f"https://{_conf_host}/virtual/{year}"
    _storage_slug = _conf_host.split(".")[0] or "conference"
    return PipelineSettings(
        year=year,
        neurips_base_url=conference_base_url,
        virtual_base_url=(
            os.getenv("CONFERENCE_VIRTUAL_URL")
            or os.getenv("NEURIPS_VIRTUAL_URL")
            or virtual_default
        ),
        slideslive_base_url=os.getenv("SLIDESLIVE_BASE_URL", "https://slideslive.com"),
        storage_root=get_path("STORAGE_ROOT", Path(f"./data/{_storage_slug}_{year}"))
        or Path(f"./data/{_storage_slug}_{year}"),
        max_concurrent_requests=get_int("MAX_CONCURRENT_REQUESTS", 4),
        domain_rate_limits=domain_limits,
        request_delay_seconds=get_float("REQUEST_DELAY_SECONDS", 0.5),
        max_retries=get_int("MAX_RETRIES", 3),
        retry_backoff_factor=get_float("RETRY_BACKOFF_FACTOR", 2.0),
        audio_chunk_max_bytes=get_int("AUDIO_CHUNK_MAX_BYTES", 23 * 1024 * 1024),
        audio_sample_rate=get_int("AUDIO_SAMPLE_RATE", 16000),
        audio_format=os.getenv("AUDIO_FORMAT", "mp3"),
        ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
        use_llm_ocr=get_bool("USE_LLM_OCR", False),
        ocr_provider=ocr_provider,  # type: ignore
        ocr_model=ocr_model,
        ocr_fallback_provider=ocr_fallback_provider,  # type: ignore
        ocr_fallback_model=ocr_fallback_model,
        use_playwright=get_bool("USE_PLAYWRIGHT", True),
        headless=get_bool("HEADLESS", True),
        browser_timeout=get_int("BROWSER_TIMEOUT", 30000),
        transcription_provider=os.getenv("TRANSCRIPTION_PROVIDER", "openai"),  # type: ignore
        openai_model=os.getenv("OPENAI_MODEL", default_model("transcription.session_speech")),
        transcription_timeout_seconds=get_float("TRANSCRIPTION_TIMEOUT_SECONDS", 600.0),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", default_model("analysis.source_reading")),
        google_api_key=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        neurips_email=os.getenv("CONFERENCE_EMAIL") or os.getenv("NEURIPS_EMAIL", ""),
        neurips_password=os.getenv("CONFERENCE_PASSWORD") or os.getenv("NEURIPS_PASSWORD", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=get_path("LOG_FILE", None),
        use_structured_logging=get_bool("USE_STRUCTURED_LOGGING", True),
        use_llm_reference_normalization=get_bool("USE_LLM_REFERENCE_NORMALIZATION", False),
        use_llm_video_detection=get_bool("USE_LLM_VIDEO_DETECTION", False),
        video_detection_provider=os.getenv("VIDEO_DETECTION_PROVIDER", "openai"),  # type: ignore
        video_detection_model=os.getenv("VIDEO_DETECTION_MODEL", default_model("video.session_detection")),
        local_first_enabled=get_bool("LOCAL_FIRST_ENABLED", True),
    )
