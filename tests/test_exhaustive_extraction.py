import json
import time
from pathlib import Path

import pytest

from scripts.exhaustive_extraction.schemas import ContentExtraction, SlideReference
from scripts.exhaustive_extraction.qa_validator import PresentationQAValidator, QAResult
from scripts.exhaustive_extraction.storage import QueuedTopicStorage


# ---------- Helpers ----------

def make_slide_ref(
    slide_id="slide_001",
    slide_path="slides/slide_001.json",
    slide_timestamp_sec=10.0,
    segment_start_sec=0.0,
    segment_end_sec=100.0,
    aligned=True,
    ocr_confidence="high",
):
    return SlideReference(
        slide_id=slide_id,
        slide_path=slide_path,
        slide_timestamp_sec=slide_timestamp_sec,
        ocr_confidence=ocr_confidence,
        slide_content_summary="demo slide",
        timestamp_aligned=aligned,
    )


def make_presentation_extraction(
    *,
    extraction_id="ex_1",
    source_id="pres_1",
    segment_index=0,
    segment_start_sec=0.0,
    segment_end_sec=100.0,
    ts_start=10.0,
    ts_end=20.0,
    slide_refs=None,
    timestamp_aligned=True,
    quote_text="quote text",
    summary_text="summary text",
):
    slide_refs = slide_refs or []
    timestamp_start_hms = seconds_to_hms(ts_start)
    timestamp_end_hms = seconds_to_hms(ts_end)
    content_hash = ContentExtraction.compute_content_hash(
        source_id, segment_index, "01", quote_text, summary_text
    )
    return ContentExtraction(
        extraction_id=extraction_id,
        content_hash=content_hash,
        source_type="presentation",
        source_id=source_id,
        source_title="Demo Presentation",
        source_coverage=None,
        segment_index=segment_index,
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        topic_id="01",
        topic_name="Tools",
        timestamp_start_sec=ts_start,
        timestamp_end_sec=ts_end,
        timestamp_start=timestamp_start_hms,
        timestamp_end=timestamp_end_hms,
        extraction_type="verbatim_quote",
        verbatim_quote=quote_text,
        summary=summary_text,
        slide_references=slide_refs,
        relevance_score=0.9,
        transcript_quality="high",
        model_used="claude-sonnet-4-6",
    )


def make_paper_extraction(
    *,
    extraction_id="paper_ex_1",
    source_id="paper_1",
    quote="paper quote",
    summary="paper summary",
):
    content_hash = ContentExtraction.compute_content_hash(
        source_id, None, "02", quote, summary
    )
    return ContentExtraction(
        extraction_id=extraction_id,
        content_hash=content_hash,
        source_type="paper",
        source_id=source_id,
        source_title="Paper Title",
        source_coverage="abstract_only",
        segment_index=None,
        segment_start_sec=None,
        segment_end_sec=None,
        topic_id="02",
        topic_name="New Architectures",
        timestamp_start_sec=None,
        timestamp_end_sec=None,
        timestamp_start=None,
        timestamp_end=None,
        extraction_type="verbatim_quote",
        verbatim_quote=quote,
        summary=summary,
        slide_references=[],
        relevance_score=0.8,
        transcript_quality=None,
        model_used="claude-sonnet-4-6",
    )


# ---------- Schema tests ----------


def test_presentation_requires_timestamps_and_bounds_missing_fails():
    with pytest.raises(ValueError):
        ContentExtraction(
            extraction_id="ex_miss",
            content_hash="sha256:" + "0" * 64,
            source_type="presentation",
            source_id="pres",
            source_title="Title",
            topic_id="01",
            topic_name="Tools",
            extraction_type="summary",
            summary="s",
            verbatim_quote=None,
            relevance_score=0.5,
            slide_references=[],
            model_used="claude-sonnet-4-6",
        )


def test_presentation_timestamp_out_of_bounds_start_fails():
    with pytest.raises(ValueError):
        make_presentation_extraction(
            ts_start=-1.0, ts_end=10.0, segment_start_sec=0.0, segment_end_sec=20.0
        )


def test_presentation_timestamp_out_of_bounds_end_fails():
    with pytest.raises(ValueError):
        make_presentation_extraction(
            ts_start=5.0, ts_end=25.0, segment_start_sec=0.0, segment_end_sec=20.0
        )


def test_presentation_timestamp_order_fails():
    with pytest.raises(ValueError):
        make_presentation_extraction(ts_start=15.0, ts_end=10.0)


def test_slide_timestamp_mismatch_when_marked_aligned_fails():
    slide = make_slide_ref(
        slide_timestamp_sec=150.0,
        segment_start_sec=0.0,
        segment_end_sec=100.0,
        aligned=True,
    )
    with pytest.raises(ValueError):
        make_presentation_extraction(
            slide_refs=[slide], segment_start_sec=0.0, segment_end_sec=100.0
        )


def test_slide_timestamp_out_of_bounds_allowed_when_not_aligned_flag():
    slide = make_slide_ref(
        slide_timestamp_sec=150.0,
        segment_start_sec=0.0,
        segment_end_sec=100.0,
        aligned=False,
    )
    extraction = make_presentation_extraction(
        slide_refs=[slide], segment_start_sec=0.0, segment_end_sec=100.0
    )
    assert extraction.slide_references[0].timestamp_aligned is False


def test_compute_content_hash_full_length_and_prefix():
    hash_value = ContentExtraction.compute_content_hash(
        "src", 1, "01", "quote", "summary"
    )
    assert hash_value.startswith("sha256:")
    assert len(hash_value.split(":")[1]) == 64


def test_compute_content_hash_deterministic_and_changes_with_quote():
    h1 = ContentExtraction.compute_content_hash("src", 1, "01", "q1", "s")
    h2 = ContentExtraction.compute_content_hash("src", 1, "01", "q1", "s")
    h3 = ContentExtraction.compute_content_hash("src", 1, "01", "q2", "s")
    assert h1 == h2
    assert h1 != h3


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------- QA validator tests ----------


def test_qa_validator_missing_required_fields_fails():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    extraction = make_presentation_extraction()
    extraction.segment_start_sec = None
    result = validator.check_timestamps(extraction)
    assert result.valid is False
    assert "Missing required" in result.reason


def test_qa_validator_bounds_start_fail():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    with pytest.raises(ValueError):
        make_presentation_extraction(ts_start=-1.0, ts_end=5.0)


def test_qa_validator_bounds_end_fail():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    with pytest.raises(ValueError):
        make_presentation_extraction(ts_start=1.0, ts_end=200.0)


def test_qa_validator_valid_passes():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    extraction = make_presentation_extraction(ts_start=1.0, ts_end=5.0)
    result = validator.check_timestamps(extraction)
    assert result.valid is True


def test_qa_validator_generic_summary_detected():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    extraction = make_presentation_extraction(ts_start=1.0, ts_end=5.0)
    extraction.summary = "they discussed ai"
    result = validator.detect_hallucination(extraction, "ai content")
    assert result.valid is False
    assert "Generic" in result.reason


def test_qa_validator_slide_missing_file():
    validator = PresentationQAValidator(conf_base_dir=Path("."))
    slide = make_slide_ref(slide_path="nonexistent.json")
    extraction = make_presentation_extraction(slide_refs=[slide])
    results = validator.check_slides_exist(extraction)
    assert any(r.valid is False for r in results)


# ---------- Storage tests ----------


def test_storage_routes_presentations(tmp_path: Path):
    storage = QueuedTopicStorage(tmp_path)
    extraction = make_presentation_extraction()
    storage.append_extraction(extraction, topic_id="01")
    time.sleep(0.1)
    storage.shutdown()
    out_file = tmp_path / "presentations" / "by_topic" / "01.jsonl"
    assert out_file.exists()
    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["source_type"] == "presentation"


def test_storage_routes_papers(tmp_path: Path):
    storage = QueuedTopicStorage(tmp_path)
    extraction = make_paper_extraction()
    storage.append_extraction(extraction, topic_id="02")
    time.sleep(0.1)
    storage.shutdown()
    out_file = tmp_path / "papers" / "by_topic" / "02.jsonl"
    assert out_file.exists()
    stored = json.loads(out_file.read_text().strip())
    assert stored["source_type"] == "paper"


def test_storage_dedup_skips_duplicates(tmp_path: Path):
    storage = QueuedTopicStorage(tmp_path)
    extraction = make_presentation_extraction()
    storage.append_extraction(extraction, topic_id="01")
    storage.append_extraction(extraction, topic_id="01")  # duplicate
    time.sleep(0.2)
    storage.shutdown()
    out_file = tmp_path / "presentations" / "by_topic" / "01.jsonl"
    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 1  # duplicate skipped


def test_storage_flushes_on_shutdown(tmp_path: Path):
    storage = QueuedTopicStorage(tmp_path)
    extraction1 = make_presentation_extraction(extraction_id="ex1", ts_start=1.0, ts_end=2.0)
    extraction2 = make_presentation_extraction(
        extraction_id="ex2",
        ts_start=3.0,
        ts_end=4.0,
        quote_text="quote2",
        summary_text="summary2",
    )
    storage.append_extraction(extraction1, topic_id="01")
    storage.append_extraction(extraction2, topic_id="01")
    time.sleep(0.1)
    storage.shutdown()
    out_file = tmp_path / "presentations" / "by_topic" / "01.jsonl"
    lines = out_file.read_text().strip().splitlines()
    ids = [json.loads(line)["extraction_id"] for line in lines]
    assert set(ids) == {"ex1", "ex2"}
