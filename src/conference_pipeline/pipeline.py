"""Main pipeline orchestrator."""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path

import httpx

from conference_pipeline.audio_preprocessor import AudioPreprocessor
from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession
from conference_pipeline.config import PipelineSettings
from conference_pipeline.downloader import MediaDownloader, SlidesDownloader, YtDlpDownloader
from conference_pipeline.models import (
    AudioFile,
    Presentation,
    PresentationStatus,
    Provenance,
    SlideImage,
    VideoFile,
    __pipeline_version__,
)
from conference_pipeline.ocr_engine import OcrEngine, create_ocr_engine
from conference_pipeline.paper_resolver import PaperDownloader, PaperResolver
from conference_pipeline.presentation_parser import PresentationPageParser
from conference_pipeline.reference_extractor import LLMReferenceNormalizer, ReferenceExtractor
from conference_pipeline.schedule_scraper import ScheduleScraper
from conference_pipeline.slides_extractor import SlideExtractor
from conference_pipeline.slides_scraper import SlidesScraper
from conference_pipeline.slideslive_client import SlidesLiveClient
from conference_pipeline.storage import StorageBackend
from conference_pipeline.transcription import TranscriptionService, create_transcription_service
from utils.http import create_client
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _create_provenance(source_url: str, file_path: Path | None = None) -> Provenance:
    """Create provenance metadata for a downloaded file."""
    prov = Provenance(
        source_url=source_url,
        downloaded_at=datetime.utcnow(),
        pipeline_version=__pipeline_version__,
    )
    if file_path and file_path.exists():
        prov.file_hash = _compute_file_hash(file_path)
        prov.file_size = file_path.stat().st_size
    return prov


class NeuripsPipeline:
    """Orchestrates the complete NeurIPS data acquisition pipeline."""

    def __init__(
        self,
        settings: PipelineSettings,
        session: NeuripsSession | None = None,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize pipeline.

        Args:
            settings: Pipeline settings.
            session: Optional pre-authenticated HTTP session.
            browser_session: Optional pre-authenticated browser session.
        """
        self.settings = settings
        self.session = session
        self.browser_session = browser_session

        # Initialize storage
        self.storage = StorageBackend(settings)

        # HTTP client for general use
        self.http_client: httpx.AsyncClient | None = None

        # Components (initialized lazily)
        self._slideslive_client: SlidesLiveClient | None = None
        self._media_downloader: MediaDownloader | None = None
        self._slides_downloader: SlidesDownloader | None = None
        self._yt_dlp_downloader: YtDlpDownloader | None = None
        self._audio_preprocessor: AudioPreprocessor | None = None
        self._transcription_service: TranscriptionService | None = None
        self._slide_extractor: SlideExtractor | None = None
        self._slides_scraper: SlidesScraper | None = None
        self._ocr_engine: OcrEngine | None = None
        self._ref_extractor: ReferenceExtractor | None = None
        self._ref_normalizer: LLMReferenceNormalizer | None = None
        self._paper_resolver: PaperResolver | None = None
        self._paper_downloader: PaperDownloader | None = None

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        setup_logging(
            level=self.settings.log_level,
            log_file=self.settings.log_file,
            use_structlog=self.settings.use_structured_logging,
        )

        # Create HTTP client
        self.http_client = create_client()

        # Initialize components
        self._slideslive_client = SlidesLiveClient(self.http_client, self.settings)
        self._media_downloader = MediaDownloader(
            self.http_client,
            self.settings.max_concurrent_requests,
        )
        self._slides_downloader = SlidesDownloader(self.http_client, self.settings)
        self._audio_preprocessor = AudioPreprocessor(self.settings)
        self._slide_extractor = SlideExtractor(self.settings)
        if self.browser_session:
            self._slides_scraper = SlidesScraper(self.settings, self.browser_session)
        self._ref_extractor = ReferenceExtractor()
        self._paper_resolver = PaperResolver(self.http_client)

        # Initialize LLM reference normalizer if configured
        if self.settings.use_llm_reference_normalization:
            try:
                self._ref_normalizer = LLMReferenceNormalizer(self.settings)
            except Exception as e:
                logger.warning(f"LLM reference normalizer not available: {e}")
        self._paper_downloader = PaperDownloader(self.http_client, self.settings)

        # Try to initialize yt-dlp
        try:
            self._yt_dlp_downloader = YtDlpDownloader(self.settings)
        except RuntimeError as e:
            logger.warning(f"yt-dlp not available: {e}")

        # Initialize transcription service
        try:
            self._transcription_service = create_transcription_service(self.settings)
        except Exception as e:
            logger.warning(f"Transcription service not available: {e}")

        # Initialize OCR engine
        try:
            self._ocr_engine = create_ocr_engine(self.settings)
        except Exception as e:
            logger.warning(f"OCR engine not available: {e}")

        logger.info("Pipeline initialized")

    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.http_client:
            await self.http_client.aclose()
        if self.session:
            await self.session.close()
        if self.browser_session:
            await self.browser_session.close()

    async def login(self) -> None:
        """Authenticate with NeurIPS."""
        if self.session is None:
            logger.info("Logging into NeurIPS...")
            self.session = await NeuripsSession.login(self.settings)
            logger.info("Login successful")

        if self.settings.use_playwright and self.browser_session is None:
            try:
                logger.info("Starting browser session...")
                self.browser_session = await NeuripsBrowserSession.login(self.settings)
                logger.info("Browser session started")
            except Exception as e:
                logger.warning(f"Browser session failed: {e}")

    async def discover_presentations(self) -> list[Presentation]:
        """Discover all presentations from NeurIPS schedule.

        Returns:
            List of discovered presentations.
        """
        if not self.session:
            await self.login()

        scraper = ScheduleScraper(
            self.session,
            self.settings,
            self.browser_session,
        )

        logger.info("Starting presentation discovery...")
        presentations = await scraper.scrape_all_presentations()

        # Enrich with SlidesLive info
        parser = PresentationPageParser(
            self.session,
            self.settings,
            self.browser_session,
        )

        enriched = []
        for pres in presentations:
            try:
                pres = await parser.enrich_presentation(pres)
                self.storage.save_presentation(pres)
                enriched.append(pres)

                await asyncio.sleep(self.settings.request_delay_seconds)

            except Exception as e:
                logger.error(f"Error enriching presentation {pres.id}: {e}")
                pres.status = PresentationStatus.ERROR
                pres.error_message = str(e)
                self.storage.save_presentation(pres)
                enriched.append(pres)

        logger.info(f"Discovered {len(enriched)} presentations")
        self.storage.export_presentations_json()

        return enriched

    async def download_assets_for_presentation(self, pres: Presentation) -> Presentation:
        """Download audio and slides for a presentation.

        Args:
            pres: Presentation to download assets for.

        Returns:
            Updated presentation.
        """
        if not pres.slideslive_id:
            logger.debug(f"No SlidesLive ID for {pres.id}, skipping asset download")
            return pres

        paths = self.storage.ensure_presentation_dirs(pres)
        assets_downloaded = False

        try:
            # Get SlidesLive metadata (use browser if available for better URL discovery)
            metadata = await self._slideslive_client.fetch_metadata_auto(pres.slideslive_id)
            slideslive_url = self._slideslive_client.get_yt_dlp_url(pres.slideslive_id)

            # Download audio using yt-dlp
            if self._yt_dlp_downloader and pres.has_video:
                try:
                    audio_path = paths.audio_dir / "raw"
                    downloaded_audio = self._yt_dlp_downloader.download_audio(
                        slideslive_url,
                        audio_path,
                        self.settings.audio_format,
                    )

                    # Save audio file with provenance
                    if downloaded_audio and downloaded_audio.exists():
                        audio_file = AudioFile(
                            presentation_id=pres.id,
                            file_path=downloaded_audio,
                            format=self.settings.audio_format,
                            provenance=_create_provenance(slideslive_url, downloaded_audio),
                        )
                        self.storage.save_audio_file(audio_file)
                        assets_downloaded = True
                        logger.info(f"Saved audio with provenance for {pres.id}")

                except Exception as e:
                    logger.warning(f"Audio download failed for {pres.id}: {e}")

            # Download slides using slides.json API (preferred, most reliable)
            if pres.slideslive_id and not pres.has_slides:
                try:
                    logger.info(f"Fetching slides via slides.json API for {pres.id}")
                    # Get slide info with timestamps for audio/video synchronization
                    slide_info = await self._slideslive_client.get_all_slide_info(
                        pres.slideslive_id,
                        height=1080,
                        format="jpeg",
                    )

                    if slide_info:
                        slide_urls = [info["url"] for info in slide_info]
                        logger.info(f"Downloading {len(slide_urls)} slides from slides.json API for {pres.id}")
                        downloaded_images = await self._slides_downloader.download_slide_images(
                            slide_urls,
                            paths.slides_images_dir,
                        )

                        # Save slide images with provenance and timing info
                        for i, img_path in enumerate(downloaded_images, 1):
                            info = slide_info[i - 1] if i <= len(slide_info) else {}
                            source_url = info.get("url", slideslive_url)
                            slide = SlideImage(
                                presentation_id=pres.id,
                                slide_index=i,
                                image_path=img_path,
                                timestamp_ms=info.get("timestamp_ms"),
                                source_type="slides_json_api",
                                provenance=_create_provenance(source_url, img_path),
                            )
                            self.storage.save_slide_image(slide)

                        if downloaded_images:
                            pres.has_slides = True
                            assets_downloaded = True
                            # Export sync.json with slide timestamps
                            self.storage.export_slide_sync_data(pres)
                            logger.info(f"Downloaded {len(downloaded_images)} slides with sync timestamps via API for {pres.id}")
                    else:
                        logger.warning(f"No slides found in slides.json for {pres.id}")

                except Exception as e:
                    logger.warning(f"Slide download via API failed for {pres.id}: {e}")

            # Fallback: Scrape slides by navigating through SlidesLive player
            if not pres.has_slides and self._slides_scraper and pres.has_video and pres.slideslive_id:
                try:
                    logger.info(f"Scraping slides from SlidesLive player for {pres.id}")

                    # Scrape slides by clicking through the player (use NeurIPS URL for auth)
                    scraped_slides = await self._slides_scraper.scrape_slides_from_player(
                        pres.slideslive_id,
                        paths.slides_images_dir,
                        pres.id,
                        neurips_url=pres.neurips_url,  # Use authenticated NeurIPS page
                        max_slides=500,
                        crop_bottom_pixels=80,
                    )

                    # Save scraped slide images with provenance
                    for slide in scraped_slides:
                        slide.provenance = _create_provenance(slideslive_url, slide.image_path)
                        self.storage.save_slide_image(slide)

                    if scraped_slides:
                        pres.has_slides = True
                        assets_downloaded = True
                        logger.info(f"Scraped {len(scraped_slides)} slide images from player for {pres.id}")

                except Exception as e:
                    logger.warning(f"Slide scraping failed for {pres.id}: {e}")

            # Download slides - try PDF first
            if metadata.slides_pdf_url:
                try:
                    pdf_path = paths.slides_dir / "raw.pdf"
                    await self._slides_downloader.download_pdf(metadata.slides_pdf_url, pdf_path)

                    if pdf_path.exists():
                        pres.has_slides = True
                        assets_downloaded = True
                        logger.info(f"Downloaded PDF slides for {pres.id}")

                except Exception as e:
                    logger.warning(f"PDF download failed for {pres.id}: {e}")

            # Try slide images if no PDF
            if not pres.has_slides and metadata.slide_image_urls:
                try:
                    downloaded_images = await self._slides_downloader.download_slide_images(
                        metadata.slide_image_urls,
                        paths.slides_images_dir,
                    )

                    # Save slide images with provenance
                    for i, img_path in enumerate(downloaded_images, 1):
                        source_url = (
                            metadata.slide_image_urls[i - 1]
                            if i <= len(metadata.slide_image_urls)
                            else slideslive_url
                        )
                        slide = SlideImage(
                            presentation_id=pres.id,
                            slide_index=i,
                            image_path=img_path,
                            source_type="download",
                            provenance=_create_provenance(source_url, img_path),
                        )
                        self.storage.save_slide_image(slide)

                    if downloaded_images:
                        pres.has_slides = True
                        assets_downloaded = True
                        logger.info(f"Saved {len(downloaded_images)} slide images with provenance for {pres.id}")

                except Exception as e:
                    logger.warning(f"Slide images download failed for {pres.id}: {e}")

            # Only mark as ASSETS_DOWNLOADED if we actually got something
            if assets_downloaded:
                pres.status = PresentationStatus.ASSETS_DOWNLOADED
                self.storage.save_presentation(pres)
            else:
                logger.warning(f"No assets downloaded for {pres.id}")

        except Exception as e:
            logger.error(f"Asset download failed for {pres.id}: {e}")
            pres.status = PresentationStatus.ERROR
            pres.error_message = str(e)
            self.storage.save_presentation(pres)

        return pres

    async def transcribe_presentation(self, pres: Presentation) -> Presentation:
        """Transcribe audio for a presentation.

        Supports resuming partial transcriptions - if some chunks were already
        transcribed, only the remaining chunks will be processed.

        Args:
            pres: Presentation to transcribe.

        Returns:
            Updated presentation.
        """
        # Skip if already transcribed or further along
        if pres.status in [
            PresentationStatus.TRANSCRIBED,
            PresentationStatus.SLIDES_EXTRACTED,
            PresentationStatus.OCR_DONE,
            PresentationStatus.REFERENCES_EXTRACTED,
            PresentationStatus.PAPERS_DOWNLOADED,
        ]:
            logger.debug(f"Skipping transcription for {pres.id} (status: {pres.status.value})")
            return pres

        if not self._transcription_service:
            logger.warning("Transcription service not available")
            return pres

        paths = self.storage.get_media_paths(pres)

        # Find audio file
        audio_files = list(paths.audio_dir.glob("raw.*"))
        if not audio_files:
            logger.debug(f"No audio file found for {pres.id}")
            return pres

        audio_path = audio_files[0]

        try:
            # Chunk audio if needed
            audio_chunks = self._audio_preprocessor.chunk_audio_if_needed(audio_path)
            total_chunks = len(audio_chunks)

            # Check for existing partial transcription (resume support)
            completed_indices = self.storage.get_completed_chunk_indices(pres.id)
            existing_chunks = self.storage.load_transcript(pres.id)

            # If all chunks are done, mark as complete
            if len(completed_indices) >= total_chunks and existing_chunks:
                logger.info(f"All {total_chunks} chunks already transcribed for {pres.id}")
                pres.status = PresentationStatus.TRANSCRIBED
                self.storage.save_presentation(pres)
                return pres

            # Find chunks that still need transcription
            chunks_to_transcribe = []
            chunk_indices_to_transcribe = []
            for i, chunk_path in enumerate(audio_chunks):
                if i not in completed_indices:
                    chunks_to_transcribe.append(chunk_path)
                    chunk_indices_to_transcribe.append(i)

            if not chunks_to_transcribe:
                logger.info(f"No chunks to transcribe for {pres.id}")
                pres.status = PresentationStatus.TRANSCRIBED
                self.storage.save_presentation(pres)
                return pres

            # Build context for better transcription accuracy
            context = self._build_transcription_context(pres)

            # Mark as in progress if we have chunks to process
            if completed_indices and len(completed_indices) < total_chunks:
                logger.info(
                    f"Resuming transcription for {pres.id}: "
                    f"{len(completed_indices)}/{total_chunks} chunks done, "
                    f"{len(chunks_to_transcribe)} remaining"
                )
            else:
                logger.info(f"Transcribing {pres.id} ({total_chunks} chunks)")

            pres.status = PresentationStatus.TRANSCRIPTION_IN_PROGRESS
            self.storage.save_presentation(pres)

            # Transcribe remaining chunks with incremental saving
            from conference_pipeline.transcription import AnthropicTranscriptionService

            for chunk_path, original_index in zip(chunks_to_transcribe, chunk_indices_to_transcribe):
                try:
                    # Use context-aware transcription if available
                    if context and isinstance(self._transcription_service, AnthropicTranscriptionService):
                        chunk_results = await self._transcription_service.transcribe_with_context(
                            [chunk_path], context
                        )
                    else:
                        chunk_results = await self._transcription_service.transcribe([chunk_path])

                    if chunk_results:
                        chunk_result = chunk_results[0]
                        chunk_result.index = original_index
                        # Save immediately for resume capability
                        self.storage.save_transcript_chunk(pres.id, chunk_result)
                        logger.debug(f"Saved chunk {original_index + 1}/{total_chunks} for {pres.id}")

                except Exception as e:
                    logger.error(f"Failed to transcribe chunk {original_index} for {pres.id}: {e}")
                    # Continue with next chunk instead of failing entirely
                    from conference_pipeline.models import TranscriptChunk
                    error_chunk = TranscriptChunk(
                        index=original_index,
                        audio_path=chunk_path,
                        text=f"[Transcription error: {e}]",
                    )
                    self.storage.save_transcript_chunk(pres.id, error_chunk)

            # Load all chunks and save final merged transcript
            all_chunks = self.storage.load_transcript(pres.id)
            successful_chunks = [c for c in all_chunks if not c.text.startswith("[Transcription error:")]

            if successful_chunks:
                # Re-save to update the merged transcript.txt
                self.storage.save_transcript(pres.id, all_chunks)
                pres.status = PresentationStatus.TRANSCRIBED
                logger.info(f"Transcription complete for {pres.id}: {len(successful_chunks)}/{total_chunks} chunks successful")
            else:
                pres.status = PresentationStatus.ERROR
                pres.error_message = "All transcription chunks failed"
                logger.error(f"All transcription chunks failed for {pres.id}")

            self.storage.save_presentation(pres)

        except Exception as e:
            logger.error(f"Transcription failed for {pres.id}: {e}")
            pres.status = PresentationStatus.ERROR
            pres.error_message = str(e)
            self.storage.save_presentation(pres)

        return pres

    def _build_transcription_context(self, pres: Presentation) -> str:
        """Build context string for transcription from presentation metadata.

        Args:
            pres: Presentation to build context for.

        Returns:
            Context string with title, authors, and other metadata.
        """
        parts = []

        if pres.title:
            parts.append(f"Presentation title: {pres.title}")

        if pres.authors:
            parts.append(f"Speakers/Authors: {', '.join(pres.authors)}")

        if pres.session_title:
            parts.append(f"Session: {pres.session_title}")

        if pres.track:
            parts.append(f"Track: {pres.track}")

        # Add OCR text from slides if available for technical terms
        ocr_results = self.storage.load_ocr_results(pres.id)
        if ocr_results:
            # Get unique technical terms from first few slides
            slide_text = " ".join(r.raw_text for r in ocr_results[:5])
            if slide_text:
                # Truncate to avoid making context too long
                if len(slide_text) > 500:
                    slide_text = slide_text[:500] + "..."
                parts.append(f"Key terms from slides: {slide_text}")

        return "\n".join(parts) if parts else ""

    async def process_slides_for_presentation(self, pres: Presentation) -> Presentation:
        """Extract and OCR slides for a presentation.

        Tries multiple sources in order:
        1. PDF file (if available)
        2. Already downloaded images
        3. Video frames (fallback when no PDF/images)

        Args:
            pres: Presentation to process.

        Returns:
            Updated presentation.
        """
        paths = self.storage.get_media_paths(pres)
        slides = []

        try:
            # Check for PDF first
            pdf_path = paths.slides_dir / "raw.pdf"
            if pdf_path.exists():
                logger.info(f"Extracting slides from PDF for {pres.id}")
                slides = self._slide_extractor.extract_from_pdf(
                    pdf_path,
                    paths.slides_images_dir,
                    pres.id,
                )

            # Check for already downloaded images
            if not slides:
                image_files = list(paths.slides_images_dir.glob("*.jpg")) + \
                             list(paths.slides_images_dir.glob("*.png"))
                if image_files:
                    logger.info(f"Using existing slide images for {pres.id}")
                    slides = self._slide_extractor.extract_from_image_urls(
                        image_files,
                        paths.slides_images_dir,
                        pres.id,
                    )

            # Fallback: extract from video if no slides found
            if not slides and pres.has_video:
                video_files = (
                    list(paths.video_dir.glob("*.mp4")) +
                    list(paths.video_dir.glob("*.mkv")) +
                    list(paths.video_dir.glob("*.webm"))
                )
                if video_files:
                    video_path = video_files[0]
                    logger.info(f"Extracting slides from video for {pres.id}: {video_path}")
                    try:
                        slides = self._slide_extractor.extract_from_video(
                            video_path,
                            paths.slides_images_dir,
                            pres.id,
                        )
                        # Mark slides with video source
                        for slide in slides:
                            slide.source_type = "video"
                    except Exception as e:
                        logger.warning(f"Video slide extraction failed for {pres.id}: {e}")

            if not slides:
                logger.debug(f"No slides found for {pres.id}")
                return pres

            # Save slide metadata with provenance
            for slide in slides:
                self.storage.save_slide_image(slide)

            pres.status = PresentationStatus.SLIDES_EXTRACTED
            self.storage.save_presentation(pres)

            # Run OCR if available
            if self._ocr_engine and slides:
                logger.info(f"Running OCR on {len(slides)} slides for {pres.id}")
                for slide in slides:
                    try:
                        result = self._ocr_engine.run_ocr(slide)
                        self.storage.save_ocr_result(result)
                    except Exception as e:
                        logger.warning(f"OCR failed for slide {slide.slide_index}: {e}")

                pres.status = PresentationStatus.OCR_DONE
                self.storage.save_presentation(pres)

        except Exception as e:
            logger.error(f"Slide processing failed for {pres.id}: {e}")
            pres.error_message = str(e)

        return pres

    async def extract_references_for_presentation(self, pres: Presentation) -> Presentation:
        """Extract references from slides and transcript.

        If LLM reference normalization is enabled, normalizes extracted
        references for better paper resolution accuracy.

        Args:
            pres: Presentation to extract references from.

        Returns:
            Updated presentation.
        """
        try:
            all_candidates = []

            # Extract from OCR
            ocr_results = self.storage.load_ocr_results(pres.id)
            for ocr in ocr_results:
                candidates = self._ref_extractor.extract_from_ocr(ocr)
                all_candidates.extend(candidates)

            # Extract from transcript
            transcript = self.storage.load_transcript(pres.id)
            if transcript:
                candidates = self._ref_extractor.extract_from_transcript(transcript, pres.id)
                all_candidates.extend(candidates)

            # Deduplicate
            if all_candidates:
                unique_candidates = self._ref_extractor.merge_candidates(all_candidates, [])

                # Apply LLM normalization if enabled and available
                if self._ref_normalizer and self.settings.use_llm_reference_normalization:
                    logger.info(f"Normalizing {len(unique_candidates)} references with LLM for {pres.id}")
                    try:
                        unique_candidates = await self._ref_normalizer.normalize_batch(
                            unique_candidates,
                            provider="anthropic" if self.settings.anthropic_api_key else "openai",
                            max_concurrent=3,
                        )
                    except Exception as e:
                        logger.warning(f"LLM normalization failed for {pres.id}: {e}")

                self.storage.save_reference_candidates(unique_candidates)
                logger.info(f"Extracted {len(unique_candidates)} references for {pres.id}")

            pres.status = PresentationStatus.REFERENCES_EXTRACTED
            self.storage.save_presentation(pres)

        except Exception as e:
            logger.error(f"Reference extraction failed for {pres.id}: {e}")
            pres.error_message = str(e)

        return pres

    async def download_papers_for_presentation(self, pres: Presentation) -> Presentation:
        """Resolve references and download papers.

        Args:
            pres: Presentation to download papers for.

        Returns:
            Updated presentation.
        """
        try:
            paths = self.storage.get_media_paths(pres)

            # Load reference candidates
            candidates = self.storage.load_reference_candidates(pres.id)
            if not candidates:
                logger.debug(f"No references to resolve for {pres.id}")
                return pres

            # Resolve references - returns (paper, reference_candidate_id) tuples
            resolved = await self._paper_resolver.resolve_many(candidates)
            logger.info(f"Resolved {len(resolved)} papers for {pres.id}")

            # Download PDFs
            papers_only = [paper for paper, _ in resolved]
            downloaded_papers = await self._paper_downloader.download_many(
                papers_only,
                paths.papers_dir,
            )

            # Create mapping from paper.id to reference_candidate_id
            ref_candidate_map = {paper.id: ref_id for paper, ref_id in resolved}

            # Save papers with reference candidate linkage
            for paper in downloaded_papers:
                ref_candidate_id = ref_candidate_map.get(paper.id)
                self.storage.save_paper(
                    paper,
                    presentation_ids=[pres.id],
                    reference_candidate_id=ref_candidate_id,
                )

            pres.status = PresentationStatus.PAPERS_DOWNLOADED
            self.storage.save_presentation(pres)

        except Exception as e:
            logger.error(f"Paper download failed for {pres.id}: {e}")
            pres.error_message = str(e)

        return pres

    async def process_presentation(self, pres: Presentation) -> Presentation:
        """Run full processing pipeline for a single presentation.

        Args:
            pres: Presentation to process.

        Returns:
            Processed presentation.
        """
        logger.info(f"Processing presentation: {pres.id} - {pres.title}")

        # Download assets
        if pres.status == PresentationStatus.DISCOVERED:
            pres = await self.download_assets_for_presentation(pres)

        # Transcribe
        if pres.status == PresentationStatus.ASSETS_DOWNLOADED:
            pres = await self.transcribe_presentation(pres)

        # Process slides (extract + OCR)
        if pres.status in [
            PresentationStatus.TRANSCRIBED,
            PresentationStatus.ASSETS_DOWNLOADED,
        ]:
            pres = await self.process_slides_for_presentation(pres)

        # Extract references
        if pres.status == PresentationStatus.OCR_DONE:
            pres = await self.extract_references_for_presentation(pres)

        # Download papers
        if pres.status == PresentationStatus.REFERENCES_EXTRACTED:
            pres = await self.download_papers_for_presentation(pres)

        return pres

    async def run_full_pipeline(
        self,
        presentation_ids: list[str] | None = None,
        retry_errors: bool = False,
        skip_completed: bool = True,
    ) -> None:
        """Run the complete pipeline with full resumability support.

        The pipeline automatically resumes from where it left off:
        - Presentations are processed based on their current status
        - Completed presentations (PAPERS_DOWNLOADED) are skipped by default
        - Error presentations are skipped unless retry_errors=True
        - Progress is saved after each stage, so interruptions are safe

        Args:
            presentation_ids: Optional list of specific presentation IDs to process.
                If None, processes all discovered presentations.
            retry_errors: If True, retry presentations that previously errored.
                Their status will be reset to DISCOVERED before processing.
            skip_completed: If True (default), skip presentations that have
                already reached PAPERS_DOWNLOADED status.
        """
        logger.info("Starting full pipeline run (resumable)")

        # Initialize
        await self.initialize()

        try:
            # Discover or load presentations
            if presentation_ids:
                presentations = [
                    self.storage.load_presentation(pid)
                    for pid in presentation_ids
                ]
                presentations = [p for p in presentations if p is not None]
            else:
                presentations = self.storage.load_all_presentations()

            if not presentations:
                logger.info("No presentations loaded, starting discovery")
                presentations = await self.discover_presentations()

            # Count statistics for logging
            total = len(presentations)
            skipped_completed = 0
            skipped_errors = 0
            to_process = 0

            # Process each presentation
            for i, pres in enumerate(presentations, 1):
                # Handle completed presentations
                if pres.status == PresentationStatus.PAPERS_DOWNLOADED:
                    if skip_completed:
                        skipped_completed += 1
                        logger.debug(f"Skipping completed presentation: {pres.id}")
                        continue
                    else:
                        logger.info(f"Re-processing completed presentation: {pres.id}")

                # Handle error presentations
                if pres.status == PresentationStatus.ERROR:
                    if retry_errors:
                        logger.info(f"Retrying errored presentation: {pres.id}")
                        pres.status = PresentationStatus.DISCOVERED
                        pres.error_message = None
                        self.storage.save_presentation(pres)
                    else:
                        skipped_errors += 1
                        logger.warning(f"Skipping errored presentation: {pres.id}")
                        continue

                to_process += 1
                logger.info(
                    f"Processing presentation {to_process}/{total - skipped_completed - skipped_errors}: "
                    f"{pres.id} (status: {pres.status.value})"
                )

                try:
                    await self.process_presentation(pres)
                except Exception as e:
                    logger.error(f"Error processing {pres.id}: {e}")
                    pres.status = PresentationStatus.ERROR
                    pres.error_message = str(e)
                    self.storage.save_presentation(pres)

            # Export final state
            self.storage.export_presentations_json()

            # Log stats
            stats = self.storage.get_stats()
            logger.info(
                f"Pipeline complete. Processed: {to_process}, "
                f"Skipped completed: {skipped_completed}, "
                f"Skipped errors: {skipped_errors}"
            )
            logger.info(f"Final stats: {stats}")

        finally:
            await self.cleanup()


async def run_pipeline(
    settings: PipelineSettings | None = None,
    presentation_ids: list[str] | None = None,
    retry_errors: bool = False,
    skip_completed: bool = True,
) -> None:
    """Convenience function to run the pipeline.

    Args:
        settings: Optional settings (loads from env if None).
        presentation_ids: Optional specific presentations to process.
        retry_errors: If True, retry presentations that previously errored.
        skip_completed: If True (default), skip completed presentations.
    """
    from conference_pipeline.config import load_settings

    if settings is None:
        settings = load_settings()

    pipeline = NeuripsPipeline(settings)
    await pipeline.run_full_pipeline(
        presentation_ids,
        retry_errors=retry_errors,
        skip_completed=skip_completed,
    )
