"""NeurIPS 2025 Video + Slides Data Acquisition Pipeline."""

__version__ = "0.1.0"

from conference_pipeline.config import PipelineSettings, load_settings
from conference_pipeline.models import (
    AudioFile,
    MediaPaths,
    OcrResult,
    Paper,
    Presentation,
    PresentationStatus,
    Provenance,
    ReferenceCandidate,
    SlideImage,
    SlidesLiveMetadata,
    TranscriptChunk,
    VideoFile,
)
__all__ = [
    # Config
    "PipelineSettings",
    "load_settings",
    # Models
    "AudioFile",
    "MediaPaths",
    "OcrResult",
    "Paper",
    "Presentation",
    "PresentationStatus",
    "Provenance",
    "ReferenceCandidate",
    "SlideImage",
    "SlidesLiveMetadata",
    "TranscriptChunk",
    "VideoFile",
    # Pipeline
    "NeuripsPipeline",
    "run_pipeline",
]


def __getattr__(name: str):
    if name in {"NeuripsPipeline", "run_pipeline"}:
        from conference_pipeline.pipeline import NeuripsPipeline, run_pipeline

        exports = {
            "NeuripsPipeline": NeuripsPipeline,
            "run_pipeline": run_pipeline,
        }
        return exports[name]
    raise AttributeError(name)
