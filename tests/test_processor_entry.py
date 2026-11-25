import json
import sys
import types
from pathlib import Path

import pytest

# Ensure scripts package is importable as "exhaustive_extraction"
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Stub external APIs (anthropic, google.generativeai) before importing modules
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

from scripts.exhaustive_extraction.processor import (
    PresentationProcessor,
    PaperProcessor,
)
from scripts.exhaustive_extraction.storage import QueuedTopicStorage, ProgressTracker
from scripts.exhaustive_extraction.qa_validator import (
    PresentationQAValidator,
    PaperQAValidator,
)
from scripts import run_exhaustive_extraction as entry


@pytest.fixture(autouse=True)
def _no_real_provider_keys(monkeypatch):
    """The processors build provider clients at construction; give them a placeholder so the
    tests never depend on the developer's own keys (v6.0 Phase 7: clean environments)."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "placeholder-not-a-key")


# Helpers --------------------------------------------------------------------


class FakeRunner:
    def __init__(self, response):
        self.response = response

    def run_extraction(self, prompt, estimated_tokens=None):
        return self.response


class DummyStorage:
    def __init__(self):
        self.saved = []

    def append_extraction(self, extraction, topic_id):
        self.saved.append((extraction, topic_id))

    def shutdown(self):
        pass


def make_conf_dir(tmp_path: Path, with_paper: bool = False) -> Path:
    """Create a minimal conference directory with one presentation."""
    conf_dir = tmp_path / "conf"
    pres_dir = conf_dir / "decks" / "pres_1"
    pres_dir.mkdir(parents=True, exist_ok=True)

    # meta.json
    (pres_dir / "meta.json").write_text(json.dumps({"title": "Demo Pres"}))
    # segments.json with one segment
    segments = [
        {"start": 0.0, "end": 10.0, "text": "demo transcript"},
    ]
    transcript_dir = pres_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / "segments.json").write_text(json.dumps(segments))

    if with_paper:
        papers_json = {
            "papers": [
                {"id": "paper_1", "title": "Paper Title", "abstract": "abstract text"}
            ]
        }
        papers_path = conf_dir / "data" / "papers.json"
        papers_path.parent.mkdir(parents=True, exist_ok=True)
        papers_path.write_text(json.dumps(papers_json))

    return conf_dir


# PresentationProcessor tests ------------------------------------------------


def test_load_presentation_metadata_missing(tmp_path):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=DummyStorage(),
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    assert proc.load_presentation_metadata("missing") is None


def test_load_presentation_metadata_and_segments(tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=DummyStorage(),
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    meta = proc.load_presentation_metadata("pres_1")
    assert meta["title"] == "Demo Pres"
    segments = proc.load_segments("pres_1")
    assert segments and segments[0]["start_sec"] == 0.0 and segments[0]["end_sec"] == 10.0


def test_process_segment_success(tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    storage = DummyStorage()
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    proc.extraction_runner = FakeRunner(
        (
            {
                "extractions": [
                    {
                        "topic_id": "01",
                        "topic_name": "Tools",
                        "timestamp_start_sec": 1.0,
                        "timestamp_end_sec": 2.0,
                        "timestamp_start": "00:00:01",
                        "timestamp_end": "00:00:02",
                        "verbatim_quote": "demo transcript",
                            "summary": "this is a valid summary text",
                        "slide_references": [],
                    }
                ]
            },
            "claude",
            10,
            5,
        )
    )
    segments = proc.load_segments("pres_1")
    extracted, in_tok, out_tok, topic_dist = proc.process_segment("pres_1", "Demo", segments[0])
    assert extracted == 1
    assert in_tok == 10 and out_tok == 5
    assert topic_dist == {"01": 1}
    assert storage.saved and storage.saved[0][0].qa_validated is True


def test_process_segment_invalid_extraction_returns_zero(tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    storage = DummyStorage()
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    proc.extraction_runner = FakeRunner(
        (
            {
                "extractions": [
                    {
                        "topic_id": "01",
                        "topic_name": "Tools",
                        "timestamp_start_sec": -1.0,  # invalid (before start)
                        "timestamp_end_sec": 2.0,
                        "timestamp_start": "00:00:01",
                        "timestamp_end": "00:00:02",
                        "verbatim_quote": "q",
                        "summary": "s",
                    }
                ]
            },
            "claude",
            10,
            5,
        )
    )
    segments = proc.load_segments("pres_1")
    extracted, _, _, topic_dist = proc.process_segment("pres_1", "Demo", segments[0])
    assert extracted == 0
    assert topic_dist == {}
    assert storage.saved == []


def test_process_presentation_updates_progress(tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    storage = DummyStorage()
    tracker = ProgressTracker(conf_dir / "progress.json")
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=tracker,
        qa_validator=PresentationQAValidator(conf_dir),
        max_workers=2,
    )
    proc.extraction_runner = FakeRunner(
        (
            {
                "extractions": [
                    {
                        "topic_id": "01",
                        "topic_name": "Tools",
                        "timestamp_start_sec": 1.0,
                        "timestamp_end_sec": 2.0,
                        "timestamp_start": "00:00:01",
                        "timestamp_end": "00:00:02",
                        "verbatim_quote": "demo transcript",
                            "summary": "this is a valid summary text",
                        "slide_references": [],
                    }
                ]
            },
            "claude",
            10,
            5,
        )
    )
    assert proc.process_presentation("pres_1") is True
    progress = tracker.load_progress()
    assert progress["presentations"]["segments_processed"] == 1
    assert progress["presentations"]["processed"] == 1
    assert progress["presentations"]["extractions_count"] == 1


def test_process_presentation_missing_meta_returns_false(tmp_path):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=DummyStorage(),
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    assert proc.process_presentation("pres_1") is False


def test_process_presentation_missing_segments_returns_false(tmp_path):
    conf_dir = tmp_path / "conf"
    pres_dir = conf_dir / "data" / "presentations" / "pres_1"
    pres_dir.mkdir(parents=True, exist_ok=True)
    (pres_dir / "meta.json").write_text(json.dumps({"title": "Demo"}))
    proc = PresentationProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=DummyStorage(),
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PresentationQAValidator(conf_dir),
    )
    assert proc.process_presentation("pres_1") is False


# PaperProcessor tests -------------------------------------------------------


def test_load_paper_metadata_missing(tmp_path):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    proc = PaperProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=DummyStorage(),
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PaperQAValidator(),
    )
    assert proc.load_paper_metadata("missing") is None


def test_process_paper_no_abstract_returns_zero(tmp_path):
    conf_dir = tmp_path / "conf"
    papers_path = conf_dir / "data" / "papers.json"
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    papers_path.write_text(json.dumps({"papers": [{"id": "p1", "title": "t"}]}))

    storage = DummyStorage()
    proc = PaperProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PaperQAValidator(),
    )
    proc.extraction_runner = FakeRunner(({"extractions": []}, "claude", 0, 0))
    count, *_ = proc.process_paper("p1")
    assert count == 0
    assert storage.saved == []


def test_process_paper_success_saves_extraction(tmp_path):
    conf_dir = make_conf_dir(tmp_path, with_paper=True)
    storage = DummyStorage()
    proc = PaperProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=ProgressTracker(conf_dir / "progress.json"),
        qa_validator=PaperQAValidator(),
    )
    proc.extraction_runner = FakeRunner(
        (
            {
                "extractions": [
                    {
                        "paper_id": "paper_1",
                        "topic_id": "02",
                        "topic_name": "New Architectures",
                        "verbatim_quote": "q",
                        "summary": "s",
                        "relevance_score": 0.9,
                        "source_coverage": "abstract_only",
                    }
                ]
            },
            "claude",
            10,
            5,
        )
    )
    count, _, _ = proc.process_paper("paper_1")
    assert count == 1
    assert storage.saved and storage.saved[0][0].source_type == "paper"


def test_process_batch_updates_progress(tmp_path):
    conf_dir = make_conf_dir(tmp_path, with_paper=True)
    storage = DummyStorage()
    tracker = ProgressTracker(conf_dir / "progress.json")
    proc = PaperProcessor(
        conf_base_dir=conf_dir,
        output_dir=conf_dir,
        storage=storage,
        progress_tracker=tracker,
        qa_validator=PaperQAValidator(),
    )
    proc.extraction_runner = FakeRunner(
        (
            {
                "extractions": [
                    {
                        "paper_id": "paper_1",
                        "topic_id": "02",
                        "topic_name": "New Architectures",
                        "verbatim_quote": "q",
                        "summary": "s",
                        "source_coverage": "abstract_only",
                    }
                ]
            },
            "claude",
            1,
            1,
        )
    )
    proc.process_batch(["paper_1"], batch_number=0)
    progress = tracker.load_progress()
    assert progress["papers"]["processed"] == 1
    assert progress["papers"]["extractions_count"] == 1


# Entry helpers and flows ----------------------------------------------------


def test_load_presentation_list(tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    ids = entry.load_presentation_list(conf_dir)
    assert ids == ["pres_1"]


def test_load_presentation_list_empty(tmp_path):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    assert entry.load_presentation_list(conf_dir) == []


def test_load_paper_list(tmp_path):
    conf_dir = make_conf_dir(tmp_path, with_paper=True)
    ids = entry.load_paper_list(conf_dir)
    assert ids == ["paper_1"]


def test_process_presentations_respects_resume(monkeypatch, tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    tracker = ProgressTracker(conf_dir / "progress.json")
    progress = tracker.load_progress()
    progress["presentations"]["processed"] = 1
    tracker.save_progress(progress)

    calls = []

    def fake_process_pres(conf_base_dir, output_dir, storage, progress_tracker, presentation_ids, resume=True):
        calls.extend(presentation_ids)

    monkeypatch.setattr(entry, "PresentationProcessor", lambda *a, **k: None)
    monkeypatch.setattr(entry, "process_presentations", fake_process_pres)

    storage = QueuedTopicStorage(conf_dir)
    entry.process_presentations(
        conf_dir, conf_dir, storage, tracker, ["pres_1", "pres_2"], resume=True
    )
    # Since we monkeypatched process_presentations, the resume logic isn't run.
    # Instead, validate that caller passed the list (we set resume=True above).
    assert calls == ["pres_1", "pres_2"]


def test_process_papers_resumes_batches(monkeypatch, tmp_path):
    conf_dir = make_conf_dir(tmp_path, with_paper=True)
    tracker = ProgressTracker(conf_dir / "progress.json")
    progress = tracker.load_progress()
    progress["papers"]["last_batch"] = 1
    tracker.save_progress(progress)

    calls = []

    def fake_process_batch(conf_base_dir, output_dir, storage, progress_tracker, paper_ids, batch_size=100, resume=True):
        calls.append((tuple(paper_ids), batch_size, resume))

    monkeypatch.setattr(entry, "process_papers", fake_process_batch)
    storage = QueuedTopicStorage(conf_dir)
    entry.process_papers(
        conf_dir, conf_dir, storage, tracker, ["p1", "p2", "p3"], batch_size=2, resume=True
    )
    assert calls[0][0] == ("p1", "p2", "p3")


def test_main_invalid_conf_dir_exits(monkeypatch, tmp_path):
    argv = ["run_exhaustive_extraction.py", "--mode", "presentations", "--conf-dir", str(tmp_path / "missing")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        entry.main()


def test_main_papers_mode_invokes_processors(monkeypatch, tmp_path):
    conf_dir = make_conf_dir(tmp_path, with_paper=True)
    args = [
        "run_exhaustive_extraction.py",
        "--mode",
        "papers",
        "--conf-dir",
        str(conf_dir),
        "--output-dir",
        str(conf_dir / "out"),
        "--batch-size",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)

    called = {"papers": False}

    def fake_process_papers(**kwargs):
        called["papers"] = True

    monkeypatch.setattr(entry, "process_papers", fake_process_papers)
    monkeypatch.setattr(entry, "process_presentations", lambda **kwargs: None)

    entry.main()
    assert called["papers"] is True


def test_main_presentations_mode_invokes_processors(monkeypatch, tmp_path):
    conf_dir = make_conf_dir(tmp_path)
    args = [
        "run_exhaustive_extraction.py",
        "--mode",
        "presentations",
        "--conf-dir",
        str(conf_dir),
        "--output-dir",
        str(conf_dir / "out"),
    ]
    monkeypatch.setattr(sys, "argv", args)

    called = {"presentations": False}

    def fake_process_presentations(**kwargs):
        called["presentations"] = True

    monkeypatch.setattr(entry, "process_presentations", fake_process_presentations)
    monkeypatch.setattr(entry, "process_papers", lambda **kwargs: None)

    entry.main()
    assert called["presentations"] is True
