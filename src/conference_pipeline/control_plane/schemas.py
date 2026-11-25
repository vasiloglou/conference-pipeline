"""Control-plane request and response records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from conference_pipeline.smoke_test import (
    DEFAULT_SMOKE_TEST_MODE,
    DEFAULT_SMOKE_TEST_STAGES,
    DEFAULT_SMOKE_TEST_TOPIC,
    DEFAULT_SMOKE_TEST_WORKSPACE,
    SMALL_NEURIPS_SMOKETEST_ID,
)

DEFAULT_WORKSPACE = DEFAULT_SMOKE_TEST_WORKSPACE
DEFAULT_MODE = DEFAULT_SMOKE_TEST_MODE
DEFAULT_STAGE_IDS = DEFAULT_SMOKE_TEST_STAGES
CANONICAL_STAGE_IDS = ("discover", "download", "transcribe", "analyze", "podcast", "deck")
SMOKE_TEST_PRESET_ID = SMALL_NEURIPS_SMOKETEST_ID
SMOKE_TEST_TOPIC = DEFAULT_SMOKE_TEST_TOPIC


def default_stages_csv() -> str:
    return ",".join(DEFAULT_STAGE_IDS)


@dataclass(frozen=True)
class LaunchRequest:
    workspace: str | Path = DEFAULT_WORKSPACE
    mode: str = DEFAULT_MODE
    stages: str = default_stages_csv()
    topic: str | None = None
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class LaunchResult:
    accepted: bool
    run_id: str


@dataclass(frozen=True)
class RunPreset:
    preset_id: str
    label: str
    workspace: Path
    mode: str
    stages: tuple[str, ...]
    topic: str | None
    resume: bool = False

    @property
    def stages_csv(self) -> str:
        return ",".join(self.stages)


SMOKE_TEST_PRESET = RunPreset(
    preset_id=SMOKE_TEST_PRESET_ID,
    label="Launch Smoke Test",
    workspace=DEFAULT_WORKSPACE,
    mode=DEFAULT_MODE,
    stages=DEFAULT_STAGE_IDS,
    topic=SMOKE_TEST_TOPIC,
    resume=False,
)
