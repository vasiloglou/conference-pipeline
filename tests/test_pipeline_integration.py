"""Integration tests for the NeurIPS pipeline.

These tests run the pipeline on a small number of mock presentations
to verify end-to-end functionality without hitting external services.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import (
    OcrResult,
    Presentation,
    PresentationStatus,
    ReferenceCandidate,
    SlideImage,
    SlidesLiveMetadata,
    TranscriptChunk,
)
from conference_pipeline.pipeline import NeuripsPipeline
from conference_pipeline.storage import StorageBackend


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_storage_dir):
    """Create test settings with temporary storage."""
    return PipelineSettings(
        storage_root=temp_storage_dir,
        neurips_email="test@example.com",
        neurips_password="testpass",
        use_playwright=False,
        headless=True,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_presentations():
    """Create sample presentations for testing."""
    return [
        Presentation(
            id="paper-001",
            title="Attention Is All You Need",
            authors=["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
            session_title="Poster Session 1",
            track="main",
            neurips_url="https://neurips.cc/virtual/2025/paper/001",
            slideslive_id="38943210",
            slideslive_url="https://slideslive.com/38943210",
            has_video=True,
            has_slides=True,
            status=PresentationStatus.DISCOVERED,
        ),
        Presentation(
            id="paper-002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin, J.", "Chang, M."],
            session_title="Poster Session 2",
            track="main",
            neurips_url="https://neurips.cc/virtual/2025/paper/002",
            slideslive_id="38943211",
            slideslive_url="https://slideslive.com/38943211",
            has_video=True,
            has_slides=True,
            status=PresentationStatus.DISCOVERED,
        ),
        Presentation(
            id="paper-003",
            title="GPT-3: Language Models are Few-Shot Learners",
            authors=["Brown, T.", "Mann, B."],
            session_title="Spotlight Session",
            track="main",
            neurips_url="https://neurips.cc/virtual/2025/paper/003",
            slideslive_id="38943212",
            slideslive_url="https://slideslive.com/38943212",
            has_video=True,
            has_slides=False,
            status=PresentationStatus.DISCOVERED,
        ),
    ]


@pytest.fixture
def sample_slideslive_metadata():
    """Create sample SlidesLive metadata."""
    return SlidesLiveMetadata(
        slideslive_id="38943210",
        title="Attention Is All You Need",
        video_url="https://example.com/video.mp4",
        slides_pdf_url="https://example.com/slides.pdf",
        slide_image_urls=[
            "https://example.com/slide_1.jpg",
            "https://example.com/slide_2.jpg",
            "https://example.com/slide_3.jpg",
        ],
        duration_seconds=1800.0,
    )


@pytest.fixture
def sample_transcript_chunks():
    """Create sample transcript chunks."""
    return [
        TranscriptChunk(
            index=0,
            audio_path=Path("audio.mp3"),
            text="Hello and welcome to our presentation on attention mechanisms.",
            start_time=0.0,
            end_time=10.0,
        ),
        TranscriptChunk(
            index=1,
            audio_path=Path("audio.mp3"),
            text="We build on work by Bahdanau et al., 2015, which introduced "
                 "attention for machine translation. See arxiv:1409.0473.",
            start_time=10.0,
            end_time=25.0,
        ),
        TranscriptChunk(
            index=2,
            audio_path=Path("audio.mp3"),
            text="Our transformer architecture is described in arxiv:1706.03762.",
            start_time=25.0,
            end_time=40.0,
        ),
    ]


@pytest.fixture
def sample_ocr_results():
    """Create sample OCR results with references."""
    return [
        OcrResult(
            presentation_id="paper-001",
            slide_index=0,
            image_path=Path("slide_1.jpg"),
            raw_text="Attention Is All You Need\nVaswani et al.",
            engine="tesseract",
        ),
        OcrResult(
            presentation_id="paper-001",
            slide_index=1,
            image_path=Path("slide_2.jpg"),
            raw_text="Related Work\n- Bahdanau et al., 2015\n- arxiv:1409.0473\n- DOI: 10.48550/arXiv.1706.03762",
            engine="tesseract",
        ),
        OcrResult(
            presentation_id="paper-001",
            slide_index=2,
            image_path=Path("slide_3.jpg"),
            raw_text="References\nhttps://arxiv.org/abs/1706.03762\nhttps://doi.org/10.1145/3295222",
            engine="tesseract",
        ),
    ]


# ============================================================================
# Storage Tests
# ============================================================================


class TestStorageBackend:
    """Tests for storage operations."""

    def test_storage_initialization(self, test_settings):
        """Test that storage initializes correctly."""
        storage = StorageBackend(test_settings)
        assert storage.root.exists()
        assert storage.db_path.exists()
        assert (storage.root / "presentations").exists()

    def test_save_and_load_presentation(self, test_settings, sample_presentations):
        """Test saving and loading presentations."""
        storage = StorageBackend(test_settings)
        pres = sample_presentations[0]

        storage.save_presentation(pres)
        loaded = storage.load_presentation(pres.id)

        assert loaded is not None
        assert loaded.id == pres.id
        assert loaded.title == pres.title
        assert loaded.status == pres.status

    def test_load_presentations_by_status(self, test_settings, sample_presentations):
        """Test loading presentations filtered by status."""
        storage = StorageBackend(test_settings)

        # Save presentations with different statuses
        for i, pres in enumerate(sample_presentations):
            if i == 0:
                pres.status = PresentationStatus.DISCOVERED
            elif i == 1:
                pres.status = PresentationStatus.ASSETS_DOWNLOADED
            else:
                pres.status = PresentationStatus.ERROR
            storage.save_presentation(pres)

        discovered = storage.load_presentations_by_status(PresentationStatus.DISCOVERED)
        assert len(discovered) == 1
        assert discovered[0].id == "paper-001"

        downloaded = storage.load_presentations_by_status(PresentationStatus.ASSETS_DOWNLOADED)
        assert len(downloaded) == 1
        assert downloaded[0].id == "paper-002"

        errors = storage.load_presentations_by_status(PresentationStatus.ERROR)
        assert len(errors) == 1
        assert errors[0].id == "paper-003"

    def test_save_and_load_transcript(self, test_settings, sample_presentations, sample_transcript_chunks):
        """Test saving and loading transcripts."""
        storage = StorageBackend(test_settings)
        pres = sample_presentations[0]
        storage.save_presentation(pres)

        storage.save_transcript(pres.id, sample_transcript_chunks)
        loaded = storage.load_transcript(pres.id)

        assert len(loaded) == len(sample_transcript_chunks)
        assert loaded[0].text == sample_transcript_chunks[0].text

    def test_save_and_load_ocr_results(self, test_settings, sample_presentations, sample_ocr_results):
        """Test saving and loading OCR results."""
        storage = StorageBackend(test_settings)
        pres = sample_presentations[0]
        storage.save_presentation(pres)

        for ocr in sample_ocr_results:
            storage.save_ocr_result(ocr)

        loaded = storage.load_ocr_results(pres.id)
        assert len(loaded) == len(sample_ocr_results)

    def test_get_stats(self, test_settings, sample_presentations):
        """Test getting pipeline statistics."""
        storage = StorageBackend(test_settings)
        for pres in sample_presentations:
            storage.save_presentation(pres)

        stats = storage.get_stats()
        assert stats["total_presentations"] == 3
        assert "presentations_by_status" in stats


# ============================================================================
# Reference Extraction Tests
# ============================================================================


class TestReferenceExtractionIntegration:
    """Tests for reference extraction from OCR and transcripts."""

    def test_extract_references_from_ocr(self, test_settings, sample_presentations, sample_ocr_results):
        """Test extracting references from OCR results."""
        from conference_pipeline.reference_extractor import ReferenceExtractor

        storage = StorageBackend(test_settings)
        pres = sample_presentations[0]
        storage.save_presentation(pres)

        for ocr in sample_ocr_results:
            storage.save_ocr_result(ocr)

        extractor = ReferenceExtractor()
        all_candidates = []

        for ocr in sample_ocr_results:
            candidates = extractor.extract_from_ocr(ocr)
            all_candidates.extend(candidates)

        # Should find arXiv IDs, DOIs, and URLs
        arxiv_ids = [c.arxiv_id for c in all_candidates if c.arxiv_id]
        dois = [c.doi for c in all_candidates if c.doi]
        urls = [c.detected_url for c in all_candidates if c.detected_url]

        assert len(arxiv_ids) >= 1
        assert len(dois) >= 1 or len(urls) >= 1

    def test_extract_references_from_transcript(self, sample_transcript_chunks):
        """Test extracting references from transcripts."""
        from conference_pipeline.reference_extractor import ReferenceExtractor

        extractor = ReferenceExtractor()
        candidates = extractor.extract_from_transcript(sample_transcript_chunks, "paper-001")

        arxiv_ids = [c.arxiv_id for c in candidates if c.arxiv_id]
        assert "1409.0473" in arxiv_ids or "1706.03762" in arxiv_ids


# ============================================================================
# Pipeline Resumability Tests
# ============================================================================


class TestPipelineResumability:
    """Tests for pipeline resume functionality."""

    def test_skip_completed_presentations(self, test_settings, sample_presentations):
        """Test that completed presentations are skipped."""
        storage = StorageBackend(test_settings)

        # Mark one as completed
        pres1 = sample_presentations[0]
        pres1.status = PresentationStatus.PAPERS_DOWNLOADED
        storage.save_presentation(pres1)

        # Mark one as in progress
        pres2 = sample_presentations[1]
        pres2.status = PresentationStatus.TRANSCRIBED
        storage.save_presentation(pres2)

        all_presentations = storage.load_all_presentations()
        to_process = [
            p for p in all_presentations
            if p.status != PresentationStatus.PAPERS_DOWNLOADED
        ]

        assert len(to_process) == 1
        assert to_process[0].id == "paper-002"

    def test_retry_errored_presentations(self, test_settings, sample_presentations):
        """Test that errored presentations can be retried."""
        storage = StorageBackend(test_settings)

        # Mark one as errored
        pres = sample_presentations[0]
        pres.status = PresentationStatus.ERROR
        pres.error_message = "Download failed"
        storage.save_presentation(pres)

        # Simulate retry logic
        loaded = storage.load_presentation(pres.id)
        assert loaded.status == PresentationStatus.ERROR

        loaded.status = PresentationStatus.DISCOVERED
        loaded.error_message = None
        storage.save_presentation(loaded)

        reloaded = storage.load_presentation(pres.id)
        assert reloaded.status == PresentationStatus.DISCOVERED
        assert reloaded.error_message is None

    def test_status_progression(self, test_settings, sample_presentations):
        """Test that status progresses correctly through stages."""
        storage = StorageBackend(test_settings)
        pres = sample_presentations[0]

        # Simulate pipeline progression
        statuses = [
            PresentationStatus.DISCOVERED,
            PresentationStatus.ASSETS_DOWNLOADED,
            PresentationStatus.TRANSCRIBED,
            PresentationStatus.SLIDES_EXTRACTED,
            PresentationStatus.OCR_DONE,
            PresentationStatus.REFERENCES_EXTRACTED,
            PresentationStatus.PAPERS_DOWNLOADED,
        ]

        for status in statuses:
            pres.status = status
            storage.save_presentation(pres)
            loaded = storage.load_presentation(pres.id)
            assert loaded.status == status


# ============================================================================
# Pipeline Integration Tests with Mocks
# ============================================================================


class TestPipelineWithMocks:
    """Integration tests using mocked external services."""

    @pytest.mark.asyncio
    async def test_process_presentation_mocked(
        self,
        test_settings,
        sample_presentations,
        sample_slideslive_metadata,
        sample_transcript_chunks,
        sample_ocr_results,
    ):
        """Test processing a presentation with mocked services."""
        pipeline = NeuripsPipeline(test_settings)

        # Initialize storage and save presentation
        pipeline.storage = StorageBackend(test_settings)
        pres = sample_presentations[0]
        pipeline.storage.save_presentation(pres)

        # Mock the SlidesLive client
        mock_slideslive = AsyncMock()
        mock_slideslive.fetch_metadata_auto.return_value = sample_slideslive_metadata
        mock_slideslive.get_yt_dlp_url.return_value = "https://slideslive.com/38943210"
        pipeline._slideslive_client = mock_slideslive

        # Mock the media downloader
        mock_downloader = AsyncMock()
        mock_downloader.download_file.return_value = Path("test.pdf")
        pipeline._media_downloader = mock_downloader

        # Mock slides downloader - create an actual PDF file to simulate download
        pdf_path = test_settings.storage_root / "presentations" / pres.id / "slides" / "raw.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 test")  # Fake but valid-looking PDF header

        async def mock_download_pdf(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4 test")
            return dest

        mock_slides_downloader = AsyncMock()
        mock_slides_downloader.download_pdf.side_effect = mock_download_pdf
        pipeline._slides_downloader = mock_slides_downloader

        # Mock yt-dlp (returns None to simulate audio not available)
        pipeline._yt_dlp_downloader = None

        # Mock audio preprocessor
        mock_audio_preprocessor = MagicMock()
        mock_audio_preprocessor.chunk_audio_if_needed.return_value = [Path("chunk.mp3")]
        pipeline._audio_preprocessor = mock_audio_preprocessor

        # Mock transcription service
        mock_transcription = AsyncMock()
        mock_transcription.transcribe.return_value = sample_transcript_chunks
        pipeline._transcription_service = mock_transcription

        # Mock slide extractor
        mock_slide_extractor = MagicMock()
        mock_slide_extractor.extract_from_pdf.return_value = [
            SlideImage(presentation_id=pres.id, slide_index=i, image_path=Path(f"slide_{i}.jpg"))
            for i in range(3)
        ]
        pipeline._slide_extractor = mock_slide_extractor

        # Mock OCR engine
        mock_ocr = MagicMock()
        mock_ocr.run_ocr.side_effect = sample_ocr_results
        pipeline._ocr_engine = mock_ocr

        # Mock reference extractor
        from conference_pipeline.reference_extractor import ReferenceExtractor
        pipeline._ref_extractor = ReferenceExtractor()

        # Mock paper resolver and downloader
        mock_resolver = AsyncMock()
        mock_resolver.resolve_many.return_value = []
        pipeline._paper_resolver = mock_resolver

        mock_paper_downloader = AsyncMock()
        mock_paper_downloader.download_many.return_value = []
        pipeline._paper_downloader = mock_paper_downloader

        # Test asset download (first stage)
        pres = await pipeline.download_assets_for_presentation(pres)
        assert pres.status == PresentationStatus.ASSETS_DOWNLOADED

        # Reload from storage to verify persistence
        loaded = pipeline.storage.load_presentation(pres.id)
        assert loaded.status == PresentationStatus.ASSETS_DOWNLOADED


class TestSmallScalePipelineRun:
    """Test running the pipeline on a small number of presentations."""

    @pytest.mark.asyncio
    async def test_run_on_three_presentations(self, test_settings, sample_presentations):
        """Test running pipeline on 3 sample presentations."""
        pipeline = NeuripsPipeline(test_settings)
        pipeline.storage = StorageBackend(test_settings)

        # Save all presentations
        for pres in sample_presentations:
            pipeline.storage.save_presentation(pres)

        # Verify all were saved
        all_pres = pipeline.storage.load_all_presentations()
        assert len(all_pres) == 3

        # Verify status breakdown
        stats = pipeline.storage.get_stats()
        assert stats["total_presentations"] == 3
        assert stats["presentations_by_status"].get("discovered", 0) == 3

    @pytest.mark.asyncio
    async def test_partial_progress_resume(self, test_settings, sample_presentations):
        """Test resuming from partial progress."""
        storage = StorageBackend(test_settings)

        # Simulate partial progress: 1 complete, 1 in progress, 1 not started
        pres1 = sample_presentations[0]
        pres1.status = PresentationStatus.PAPERS_DOWNLOADED
        storage.save_presentation(pres1)

        pres2 = sample_presentations[1]
        pres2.status = PresentationStatus.OCR_DONE
        storage.save_presentation(pres2)

        pres3 = sample_presentations[2]
        pres3.status = PresentationStatus.DISCOVERED
        storage.save_presentation(pres3)

        # Count what needs processing
        all_pres = storage.load_all_presentations()
        to_process = [
            p for p in all_pres
            if p.status not in [PresentationStatus.PAPERS_DOWNLOADED, PresentationStatus.ERROR]
        ]

        assert len(to_process) == 2  # pres2 and pres3 need work
        assert "paper-002" in [p.id for p in to_process]
        assert "paper-003" in [p.id for p in to_process]

    @pytest.mark.asyncio
    async def test_error_handling_and_continue(self, test_settings, sample_presentations):
        """Test that pipeline continues after errors."""
        storage = StorageBackend(test_settings)

        # Mark first as error
        pres1 = sample_presentations[0]
        pres1.status = PresentationStatus.ERROR
        pres1.error_message = "Network timeout"
        storage.save_presentation(pres1)

        # Others are ready
        for pres in sample_presentations[1:]:
            storage.save_presentation(pres)

        # Get non-errored presentations
        all_pres = storage.load_all_presentations()
        to_process = [p for p in all_pres if p.status != PresentationStatus.ERROR]

        assert len(to_process) == 2
        assert all(p.status == PresentationStatus.DISCOVERED for p in to_process)


# ============================================================================
# End-to-End Workflow Test
# ============================================================================


class TestEndToEndWorkflow:
    """Test complete workflow from discovery to references."""

    def test_complete_workflow_simulation(
        self,
        test_settings,
        sample_presentations,
        sample_transcript_chunks,
        sample_ocr_results,
    ):
        """Simulate complete workflow without external calls."""
        from conference_pipeline.reference_extractor import ReferenceExtractor

        storage = StorageBackend(test_settings)

        # Step 1: Discovery - save presentations
        for pres in sample_presentations:
            storage.save_presentation(pres)

        discovered = storage.load_presentations_by_status(PresentationStatus.DISCOVERED)
        assert len(discovered) == 3

        # Step 2: Asset download (simulated)
        for pres in discovered[:2]:  # Process only 2 for speed
            pres.status = PresentationStatus.ASSETS_DOWNLOADED
            storage.save_presentation(pres)

        downloaded = storage.load_presentations_by_status(PresentationStatus.ASSETS_DOWNLOADED)
        assert len(downloaded) == 2

        # Step 3: Transcription (simulated)
        for pres in downloaded:
            storage.save_transcript(pres.id, sample_transcript_chunks)
            pres.status = PresentationStatus.TRANSCRIBED
            storage.save_presentation(pres)

        transcribed = storage.load_presentations_by_status(PresentationStatus.TRANSCRIBED)
        assert len(transcribed) == 2

        # Step 4: Slide extraction (simulated)
        for pres in transcribed:
            pres.status = PresentationStatus.SLIDES_EXTRACTED
            storage.save_presentation(pres)

        # Step 5: OCR (simulated)
        slides_extracted = storage.load_presentations_by_status(PresentationStatus.SLIDES_EXTRACTED)
        for pres in slides_extracted:
            for ocr in sample_ocr_results:
                ocr.presentation_id = pres.id
                storage.save_ocr_result(ocr)
            pres.status = PresentationStatus.OCR_DONE
            storage.save_presentation(pres)

        # Step 6: Reference extraction
        extractor = ReferenceExtractor()
        ocr_done = storage.load_presentations_by_status(PresentationStatus.OCR_DONE)
        for pres in ocr_done:
            all_candidates = []
            ocr_results = storage.load_ocr_results(pres.id)
            for ocr in ocr_results:
                candidates = extractor.extract_from_ocr(ocr)
                all_candidates.extend(candidates)

            transcript = storage.load_transcript(pres.id)
            if transcript:
                candidates = extractor.extract_from_transcript(transcript, pres.id)
                all_candidates.extend(candidates)

            if all_candidates:
                storage.save_reference_candidates(all_candidates)

            pres.status = PresentationStatus.REFERENCES_EXTRACTED
            storage.save_presentation(pres)

        # Verify final state
        refs_extracted = storage.load_presentations_by_status(PresentationStatus.REFERENCES_EXTRACTED)
        assert len(refs_extracted) == 2

        # Check statistics
        stats = storage.get_stats()
        assert stats["total_presentations"] == 3
        assert stats["presentations_by_status"]["references_extracted"] == 2
        assert stats["presentations_by_status"]["discovered"] == 1
