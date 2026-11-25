import json
import sys
import types
from pathlib import Path

import pytest

# Make scripts importable
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Stub external SDKs (anthropic, google.generativeai) to avoid network deps
class _DummyAnthropicClient:
    def __init__(self, *a, **k):
        pass


anthropic_stub = types.SimpleNamespace(
    Anthropic=_DummyAnthropicClient,
    APIConnectionError=Exception,
    APITimeoutError=Exception,
    RateLimitError=Exception,
)
sys.modules.setdefault("anthropic", anthropic_stub)


class _DummyGenAIModel:
    def __init__(self, *a, **k):
        pass

    def generate_content(self, *a, **k):
        return types.SimpleNamespace(text="{}")


google_genai_stub = types.SimpleNamespace(
    configure=lambda **kwargs: None,
    GenerationConfig=type("GenerationConfig", (), {}),
    GenerativeModel=_DummyGenAIModel,
)


class _DummyGoogleGenAIClient:
    def __init__(self, *a, **k):
        pass


google_genai_v2_stub = types.SimpleNamespace(
    Client=_DummyGoogleGenAIClient,
    types=types.SimpleNamespace(),
)
google_root_stub = sys.modules.setdefault("google", types.SimpleNamespace())
setattr(google_root_stub, "generativeai", google_genai_stub)
setattr(google_root_stub, "genai", google_genai_v2_stub)
sys.modules.setdefault("google.generativeai", google_genai_stub)
sys.modules.setdefault("google.genai", google_genai_v2_stub)
sys.modules.setdefault("google.genai.types", google_genai_v2_stub.types)

from exhaustive_extraction.qa_validator import PresentationQAValidator
from exhaustive_extraction.schemas import ContentExtraction, SlideReference
from exhaustive_extraction.processor import PresentationProcessor
from exhaustive_extraction.storage import ProgressTracker


def hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_slide_file(base: Path, slide_id: str, timestamp_sec: float) -> str:
    slide_dir = base / "decks" / "pres_1" / "ocr"
    slide_dir.mkdir(parents=True, exist_ok=True)
    slide_path = slide_dir / f"{slide_id}.json"
    slide_path.write_text(json.dumps({"timestamp_sec": timestamp_sec, "slide_text": {"title": "t"}}))
    # return path relative to conf_base_dir
    return str(slide_path.relative_to(base))


def make_content_extraction(
    *,
    source_id="pres_1",
    topic_id="01",
    ts_start=1.0,
    ts_end=2.0,
    seg_start=0.0,
    seg_end=10.0,
    quote="demo transcript",
    summary="valid summary text",
    slide_ref: SlideReference | None = None,
):
    content_hash = ContentExtraction.compute_content_hash(
        source_id=source_id,
        segment_index=0,
        topic_id=topic_id,
        verbatim_quote=quote,
        summary=summary,
    )
    slide_refs = [slide_ref] if slide_ref else []
    return ContentExtraction(
        extraction_id="ex1",
        content_hash=content_hash,
        source_type="presentation",
        source_id=source_id,
        source_title="Demo",
        source_coverage=None,
        segment_index=0,
        segment_start_sec=seg_start,
        segment_end_sec=seg_end,
        topic_id=topic_id,
        topic_name="Tools",
        timestamp_start_sec=ts_start,
        timestamp_end_sec=ts_end,
        timestamp_start=hms(ts_start),
        timestamp_end=hms(ts_end),
        extraction_type="verbatim_quote",
        verbatim_quote=quote,
        summary=summary,
        slide_references=slide_refs,
        relevance_score=0.9,
        transcript_quality="high",
        model_used="claude-sonnet-4-6",
    )


def test_qa_good_extraction_passes(tmp_path: Path):
    slide_path = make_slide_file(tmp_path, "slide_001", timestamp_sec=2.0)
    slide_ref = SlideReference(
        slide_id="slide_001",
        slide_path=slide_path,
        slide_timestamp_sec=2.0,
        ocr_confidence="high",
        slide_content_summary="t",
        timestamp_aligned=True,
    )
    extraction = make_content_extraction(slide_ref=slide_ref)
    qa = PresentationQAValidator(tmp_path)
    results = qa.validate_extraction(extraction, transcript="demo transcript text")
    assert results["timestamp_check"].valid is True
    assert results["quote_check"].valid is True
    assert all(r.valid for r in results["slide_checks"])


def test_qa_quote_missing_fails(tmp_path: Path):
    extraction = make_content_extraction(quote="missing quote", summary="valid")
    qa = PresentationQAValidator(tmp_path)
    result = qa.check_quote_in_transcript(extraction, transcript="other text")
    assert result.valid is False
    assert "not found" in result.reason


def test_qa_slide_out_of_bounds_fails(tmp_path: Path):
    slide_path = make_slide_file(tmp_path, "slide_001", timestamp_sec=50.0)
    slide_ref = SlideReference(
        slide_id="slide_001",
        slide_path=slide_path,
        slide_timestamp_sec=50.0,
        ocr_confidence="high",
        slide_content_summary="t",
        timestamp_aligned=True,
    )
    with pytest.raises(ValueError):
        make_content_extraction(seg_end=10.0, ts_end=2.0, slide_ref=slide_ref)


def test_qa_hallucination_generic_summary_fails(tmp_path: Path):
    extraction = make_content_extraction(summary="they discussed ai")
    qa = PresentationQAValidator(tmp_path)
    result = qa.detect_hallucination(extraction, "demo transcript")
    assert result.valid is False
    assert "Generic" in result.reason


def test_guardrail_required_cues_enforced(tmp_path: Path):
    guardrails = {"01": {"required_cues": ["infrastructure"], "allow_generic": True}}
    qa = PresentationQAValidator(tmp_path, guardrails=guardrails)
    extraction = make_content_extraction(summary="this is a valid summary text")
    result = qa._apply_topic_guardrails(extraction)
    assert result.valid is False
    extraction.summary = "this touches compute infrastructure scaling"
    result2 = qa._apply_topic_guardrails(extraction)
    assert result2.valid is True


def test_guardrail_min_lengths(tmp_path: Path):
    guardrails = {"19": {"min_quote_len": 5, "min_summary_len": 6}}
    qa = PresentationQAValidator(tmp_path, guardrails=guardrails)
    extraction = make_content_extraction(topic_id="19", quote="short", summary="too short")
    result = qa._apply_topic_guardrails(extraction)
    assert result.valid is False
    extraction.verbatim_quote = "this is a long enough quote"
    extraction.summary = "this is a long enough summary"
    result2 = qa._apply_topic_guardrails(extraction)
    assert result2.valid is True


def test_process_segment_stores_only_valid(tmp_path: Path, monkeypatch):
    # Setup minimal conf dir
    conf_dir = tmp_path / "conf"
    pres_dir = conf_dir / "decks" / "pres_1"
    (pres_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (pres_dir / "meta.json").write_text(json.dumps({"title": "Demo Pres"}))
    (pres_dir / "transcripts" / "segments.json").write_text(
        json.dumps([{"start": 0.0, "end": 10.0, "text": "demo transcript text"}])
    )

    # Slide files
    good_slide_rel = make_slide_file(conf_dir, "slide_001", timestamp_sec=2.0)
    bad_slide_rel = make_slide_file(conf_dir, "slide_999", timestamp_sec=50.0)

    class DummyStorage:
        def __init__(self):
            self.saved = []

        def append_extraction(self, extraction, topic_id):
            self.saved.append(extraction)

        def shutdown(self):
            pass

    storage = DummyStorage()
    tracker = ProgressTracker(conf_dir / "progress.json")
    qa = PresentationQAValidator(conf_dir)
    processor = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=tracker,
        qa_validator=qa,
    )

    # Fake runner with one valid and one invalid extraction
    processor.extraction_runner = type(
        "Runner",
        (),
        {
            "run_extraction": lambda self, prompt: (
                {
                    "extractions": [
                        {
                            "topic_id": "01",
                            "topic_name": "Tools",
                                "timestamp_start_sec": 1.0,
                                "timestamp_end_sec": 2.0,
                                "timestamp_start": "00:00:01",
                                "timestamp_end": "00:00:02",
                                "verbatim_quote": "demo transcript text",
                                "summary": "this is a valid summary text",
                                "slide_references": [
                                    {
                                        "slide_id": "slide_001",
                                        "slide_timestamp_sec": 2.0,
                                        "ocr_confidence": "high",
                                    "timestamp_aligned": True,
                                    "slide_path": good_slide_rel,
                                    "slide_content_summary": "t",
                                }
                            ],
                        },
                        {
                            "topic_id": "01",
                            "topic_name": "Tools",
                                "timestamp_start_sec": 1.0,
                                "timestamp_end_sec": 2.0,
                                "timestamp_start": "00:00:01",
                                "timestamp_end": "00:00:02",
                                "verbatim_quote": "demo transcript text",
                                "summary": "they discussed ai systems",
                                "slide_references": [
                                    {
                                        "slide_id": "slide_002",
                                        "slide_timestamp_sec": 2.5,
                                        "ocr_confidence": "high",
                                        "timestamp_aligned": True,
                                        "slide_path": good_slide_rel,
                                        "slide_content_summary": "t",
                                    }
                                ],
                            },
                    ]
                },
                "claude",
                10,
                5,
            )
        },
    )()

    segments = processor.load_segments("pres_1")
    extracted, _, _, topic_dist = processor.process_segment("pres_1", "Demo", segments[0])
    # Only the valid extraction should be stored/validated
    assert extracted == 1
    assert len(storage.saved) == 1
    assert topic_dist == {"01": 1}
