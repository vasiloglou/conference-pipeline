"""Slide extraction from PDFs, archives, and videos."""

import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import SlideImage
from utils.logging import get_logger

logger = get_logger(__name__)


class SlideExtractorError(Exception):
    """Raised when slide extraction fails."""

    pass


class SlideExtractor:
    """Extracts slides as JPEG images from various formats."""

    def __init__(self, settings: PipelineSettings):
        """Initialize slide extractor.

        Args:
            settings: Pipeline settings.
        """
        self.settings = settings

    def extract_from_pdf(
        self,
        pdf_path: Path,
        out_dir: Path,
        presentation_id: str,
        dpi: int = 150,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract slides from a PDF file.

        Args:
            pdf_path: Path to PDF file.
            out_dir: Output directory for images.
            presentation_id: Presentation ID for metadata.
            dpi: Resolution for rendering.
            image_format: Output image format.

        Returns:
            List of SlideImage objects.

        Raises:
            SlideExtractorError: If extraction fails.
        """
        if not pdf_path.exists():
            raise SlideExtractorError(f"PDF file not found: {pdf_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        slides = []

        # Try PyMuPDF first (faster)
        try:
            slides = self._extract_with_pymupdf(
                pdf_path, out_dir, presentation_id, dpi, image_format
            )
            if slides:
                return slides
        except ImportError:
            logger.debug("PyMuPDF not available, trying pdf2image")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")

        # Fallback to pdf2image
        try:
            slides = self._extract_with_pdf2image(
                pdf_path, out_dir, presentation_id, dpi, image_format
            )
            return slides
        except ImportError:
            raise SlideExtractorError(
                "Neither PyMuPDF nor pdf2image available. "
                "Install with: pip install pymupdf or pip install pdf2image"
            )

    def _extract_with_pymupdf(
        self,
        pdf_path: Path,
        out_dir: Path,
        presentation_id: str,
        dpi: int,
        image_format: str,
    ) -> list[SlideImage]:
        """Extract using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        slides = []
        doc = fitz.open(pdf_path)

        try:
            zoom = dpi / 72  # Convert DPI to zoom factor
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)

                ext = "jpg" if image_format == "jpeg" else image_format
                image_path = out_dir / f"slide_{page_num + 1:03d}.{ext}"

                if image_format == "jpeg":
                    pix.save(image_path, output="jpeg", jpg_quality=95)
                else:
                    pix.save(image_path)

                slides.append(
                    SlideImage(
                        presentation_id=presentation_id,
                        slide_index=page_num + 1,
                        image_path=image_path,
                    )
                )

                logger.debug(f"Extracted slide {page_num + 1}/{len(doc)}")

        finally:
            doc.close()

        logger.info(f"Extracted {len(slides)} slides from {pdf_path}")
        return slides

    def _extract_with_pdf2image(
        self,
        pdf_path: Path,
        out_dir: Path,
        presentation_id: str,
        dpi: int,
        image_format: str,
    ) -> list[SlideImage]:
        """Extract using pdf2image (requires poppler)."""
        from pdf2image import convert_from_path

        slides = []

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=image_format,
        )

        for i, image in enumerate(images, 1):
            ext = "jpg" if image_format == "jpeg" else image_format
            image_path = out_dir / f"slide_{i:03d}.{ext}"

            if image_format == "jpeg":
                image.save(image_path, "JPEG", quality=95)
            else:
                image.save(image_path)

            slides.append(
                SlideImage(
                    presentation_id=presentation_id,
                    slide_index=i,
                    image_path=image_path,
                )
            )

            logger.debug(f"Extracted slide {i}/{len(images)}")

        logger.info(f"Extracted {len(slides)} slides from {pdf_path}")
        return slides

    def extract_from_images_archive(
        self,
        archive_path: Path,
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Extract slides from an archive of images.

        Args:
            archive_path: Path to ZIP/archive file.
            out_dir: Output directory.
            presentation_id: Presentation ID for metadata.

        Returns:
            List of SlideImage objects.

        Raises:
            SlideExtractorError: If extraction fails.
        """
        if not archive_path.exists():
            raise SlideExtractorError(f"Archive not found: {archive_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        slides = []

        if archive_path.suffix.lower() == ".zip":
            slides = self._extract_from_zip(archive_path, out_dir, presentation_id)
        else:
            raise SlideExtractorError(
                f"Unsupported archive format: {archive_path.suffix}"
            )

        return slides

    def _extract_from_zip(
        self,
        zip_path: Path,
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Extract images from ZIP archive."""
        slides = []
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get image files sorted by name
            image_files = sorted([
                name for name in zf.namelist()
                if Path(name).suffix.lower() in image_extensions
            ])

            for i, name in enumerate(image_files, 1):
                # Extract with new name
                ext = Path(name).suffix.lower()
                image_path = out_dir / f"slide_{i:03d}{ext}"

                with zf.open(name) as src, open(image_path, "wb") as dst:
                    dst.write(src.read())

                slides.append(
                    SlideImage(
                        presentation_id=presentation_id,
                        slide_index=i,
                        image_path=image_path,
                    )
                )

        logger.info(f"Extracted {len(slides)} slides from {zip_path}")
        return slides

    def extract_from_image_urls(
        self,
        image_paths: list[Path],
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Create SlideImage objects from already downloaded images.

        Args:
            image_paths: List of image file paths.
            out_dir: Output directory (for renaming if needed).
            presentation_id: Presentation ID for metadata.

        Returns:
            List of SlideImage objects.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        slides = []

        for i, src_path in enumerate(sorted(image_paths), 1):
            if not src_path.exists():
                logger.warning(f"Image not found: {src_path}")
                continue

            # Rename to consistent naming
            ext = src_path.suffix
            dest_path = out_dir / f"slide_{i:03d}{ext}"

            if src_path != dest_path:
                import shutil
                shutil.copy2(src_path, dest_path)
                image_path = dest_path
            else:
                image_path = src_path

            slides.append(
                SlideImage(
                    presentation_id=presentation_id,
                    slide_index=i,
                    image_path=image_path,
                )
            )

        logger.info(f"Processed {len(slides)} slide images")
        return slides

    def convert_images_to_jpeg(
        self,
        slides: list[SlideImage],
        quality: int = 95,
    ) -> list[SlideImage]:
        """Convert slide images to JPEG format.

        Args:
            slides: List of SlideImage objects.
            quality: JPEG quality (1-100).

        Returns:
            List of SlideImage objects with JPEG paths.
        """
        from PIL import Image

        converted = []

        for slide in slides:
            if not slide.image_path.exists():
                logger.warning(f"Image not found: {slide.image_path}")
                continue

            if slide.image_path.suffix.lower() in {".jpg", ".jpeg"}:
                converted.append(slide)
                continue

            # Convert to JPEG
            jpeg_path = slide.image_path.with_suffix(".jpg")

            try:
                with Image.open(slide.image_path) as img:
                    # Convert RGBA to RGB if needed
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(img, mask=img.split()[-1])
                        img = background

                    img.save(jpeg_path, "JPEG", quality=quality)

                converted.append(
                    SlideImage(
                        presentation_id=slide.presentation_id,
                        slide_index=slide.slide_index,
                        image_path=jpeg_path,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to convert {slide.image_path}: {e}")
                converted.append(slide)  # Keep original

        return converted

    def extract_from_video(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        sample_interval: float = 1.0,
        similarity_threshold: float = 0.95,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract unique slide frames from a video.

        Uses scene change detection to identify when slides change
        and extracts only unique frames.

        Args:
            video_path: Path to video file.
            out_dir: Output directory for extracted frames.
            presentation_id: Presentation ID for metadata.
            sample_interval: Interval in seconds between frame samples.
            similarity_threshold: Threshold for considering frames similar (0-1).
            image_format: Output image format (jpeg, png).

        Returns:
            List of SlideImage objects for unique slides.

        Raises:
            SlideExtractorError: If extraction fails.
        """
        if not video_path.exists():
            raise SlideExtractorError(f"Video file not found: {video_path}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # First try using ffmpeg scene detection
        try:
            return self._extract_with_ffmpeg_scene(
                video_path, out_dir, presentation_id, similarity_threshold, image_format
            )
        except Exception as e:
            logger.warning(f"FFmpeg scene detection failed: {e}, trying sampling approach")

        # Fall back to sampling and deduplication
        return self._extract_with_sampling(
            video_path, out_dir, presentation_id, sample_interval, similarity_threshold, image_format
        )

    def extract_from_slide_videos(
        self,
        video_files: list[Path],
        out_dir: Path,
        presentation_id: str,
        crop_bottom_pixels: int = 80,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract clean slide images from slide video segments.

        Each slide video is a short clip showing a single slide. This extracts
        one frame from the middle of each video and crops out the player controls.

        Args:
            video_files: List of slide video files (sorted).
            out_dir: Output directory for extracted images.
            presentation_id: Presentation ID for metadata.
            crop_bottom_pixels: Number of pixels to crop from bottom (removes player controls).
            image_format: Output image format (jpeg, png).

        Returns:
            List of SlideImage objects.

        Raises:
            SlideExtractorError: If extraction fails.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if image_format == "jpeg" else image_format

        slide_images = []

        for i, video_path in enumerate(video_files, 1):
            if not video_path.exists():
                logger.warning(f"Slide video not found: {video_path}")
                continue

            output_path = out_dir / f"slide_{i:03d}.{ext}"

            # Get video duration to extract from middle
            try:
                duration_cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ]
                duration_result = subprocess.run(
                    duration_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,  # 30 second timeout for ffprobe
                )
                duration = float(duration_result.stdout.strip())
                timestamp = duration / 2  # Extract from middle
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout getting duration for {video_path}, using 1s")
                timestamp = 1.0
            except Exception as e:
                logger.warning(f"Could not get duration for {video_path}: {e}, using 1s")
                timestamp = 1.0

            # Extract frame with cropping
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),  # Seek to middle
                "-i", str(video_path),
                "-vf", f"crop=in_w:in_h-{crop_bottom_pixels}:0:0",  # Crop bottom
                "-frames:v", "1",  # Extract 1 frame
                "-q:v", "2",  # High quality
                str(output_path),
                "-y",
            ]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,  # 60 second timeout for frame extraction
                )

                if output_path.exists():
                    slide_image = SlideImage(
                        presentation_id=presentation_id,
                        slide_index=i,
                        image_path=output_path,
                        source_type="video_extraction",
                    )
                    slide_images.append(slide_image)
                    logger.debug(f"Extracted slide {i} from {video_path.name}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout extracting frame from {video_path}")
                continue
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract frame from {video_path}: {e.stderr}")
                continue

        logger.info(f"Extracted {len(slide_images)} slides from {len(video_files)} videos")
        return slide_images

    def extract_from_video_url_with_chapters(
        self,
        video_url: str,
        chapters: list[dict],
        out_dir: Path,
        presentation_id: str,
        crop_bottom_pixels: int = 80,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract slides from video URL using chapter timestamps.

        This uses ffmpeg to extract frames at specific timestamps where slides change,
        as provided by the chapters metadata from yt-dlp.

        Args:
            video_url: Direct video URL (from yt-dlp).
            chapters: List of chapter dicts with 'title', 'start_time', 'end_time'.
            out_dir: Output directory for extracted frames.
            presentation_id: Presentation ID for metadata.
            crop_bottom_pixels: Number of pixels to crop from bottom (removes player controls).
            image_format: Output image format (jpeg, png).

        Returns:
            List of SlideImage objects.

        Raises:
            SlideExtractorError: If extraction fails.
        """
        if not chapters:
            logger.warning("No chapters provided, cannot extract slides")
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if image_format == "jpeg" else image_format

        slide_images = []

        logger.info(f"Extracting {len(chapters)} slides from video using chapter timestamps")

        for i, chapter in enumerate(chapters, 1):
            # Extract timestamp (use start_time, or middle if available)
            timestamp = chapter.get('start_time', 0)
            if timestamp is None:
                timestamp = 0

            # Add small offset to avoid transition frames
            timestamp += 0.5

            output_path = out_dir / f"slide_{i:03d}.{ext}"

            # Extract frame at timestamp with cropping
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),  # Seek to timestamp
                "-i", video_url,        # Input URL (can be streaming)
                "-vf", f"crop=in_w:in_h-{crop_bottom_pixels}:0:0",  # Crop bottom
                "-frames:v", "1",       # Extract 1 frame
                "-q:v", "2",            # High quality
                str(output_path),
                "-y",                   # Overwrite
                "-loglevel", "error",   # Quiet unless error
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout per frame
                )

                if result.returncode != 0:
                    logger.warning(f"Failed to extract slide {i}: {result.stderr}")
                    continue

                if output_path.exists():
                    slide_image = SlideImage(
                        presentation_id=presentation_id,
                        slide_index=i,
                        image_path=output_path,
                        source_type="video_chapter_extraction",
                    )
                    slide_images.append(slide_image)
                else:
                    logger.warning(f"Output file not created for slide {i}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout extracting slide {i}")
                continue
            except Exception as e:
                logger.warning(f"Error extracting slide {i}: {e}")
                continue

        logger.info(f"Successfully extracted {len(slide_images)}/{len(chapters)} slides")
        return slide_images

    def _extract_with_ffmpeg_scene(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        similarity_threshold: float,
        image_format: str,
    ) -> list[SlideImage]:
        """Extract frames using ffmpeg scene change detection."""
        ext = "jpg" if image_format == "jpeg" else image_format
        output_pattern = str(out_dir / f"frame_%04d.{ext}")

        # Scene detection threshold (lower = more sensitive)
        scene_threshold = 1.0 - similarity_threshold

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            "-q:v", "2",
            output_pattern,
            "-y",
        ]

        logger.info(f"Extracting frames with scene detection: {video_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,  # 5 minute timeout for scene detection (can be slow for long videos)
            )
            logger.debug(f"ffmpeg output: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            raise SlideExtractorError(f"FFmpeg scene extraction timed out after 5 minutes") from e
        except subprocess.CalledProcessError as e:
            raise SlideExtractorError(f"FFmpeg scene extraction failed: {e.stderr}") from e

        # Collect extracted frames
        slides = []
        for i, frame_path in enumerate(sorted(out_dir.glob(f"frame_*.{ext}")), 1):
            # Rename to consistent slide naming
            new_path = out_dir / f"slide_{i:03d}.{ext}"
            frame_path.rename(new_path)

            slides.append(
                SlideImage(
                    presentation_id=presentation_id,
                    slide_index=i,
                    image_path=new_path,
                    source_type="video",
                )
            )

        logger.info(f"Extracted {len(slides)} unique frames from video")
        return slides

    def _extract_with_sampling(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        sample_interval: float,
        similarity_threshold: float,
        image_format: str,
    ) -> list[SlideImage]:
        """Extract frames by sampling and deduplication."""
        try:
            from PIL import Image
            import imagehash
        except ImportError:
            raise SlideExtractorError(
                "PIL and imagehash required for video sampling. "
                "Install with: pip install Pillow imagehash"
            )

        ext = "jpg" if image_format == "jpeg" else image_format

        # Create temp dir for raw frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frame_pattern = str(temp_path / f"frame_%06d.{ext}")

            # Extract frames at interval
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"fps=1/{sample_interval}",
                "-q:v", "2",
                frame_pattern,
                "-y",
            ]

            logger.info(f"Sampling video frames every {sample_interval}s: {video_path}")

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,  # 5 minute timeout for video sampling
                )
            except subprocess.TimeoutExpired as e:
                raise SlideExtractorError(f"FFmpeg sampling timed out after 5 minutes") from e
            except subprocess.CalledProcessError as e:
                raise SlideExtractorError(f"FFmpeg sampling failed: {e.stderr}") from e

            # Find unique frames using perceptual hashing
            frame_files = sorted(temp_path.glob(f"frame_*.{ext}"))
            unique_frames = []
            last_hash = None

            for frame_file in frame_files:
                try:
                    with Image.open(frame_file) as img:
                        current_hash = imagehash.phash(img)

                        if last_hash is None:
                            unique_frames.append(frame_file)
                            last_hash = current_hash
                        else:
                            # Calculate similarity (0 = identical)
                            diff = current_hash - last_hash
                            max_diff = len(current_hash.hash) * len(current_hash.hash[0])
                            similarity = 1 - (diff / max_diff)

                            if similarity < similarity_threshold:
                                unique_frames.append(frame_file)
                                last_hash = current_hash
                except Exception as e:
                    logger.debug(f"Error processing frame {frame_file}: {e}")

            # Copy unique frames to output
            slides = []
            for i, frame_file in enumerate(unique_frames, 1):
                dest_path = out_dir / f"slide_{i:03d}.{ext}"

                with Image.open(frame_file) as img:
                    if image_format == "jpeg":
                        img.save(dest_path, "JPEG", quality=95)
                    else:
                        img.save(dest_path)

                slides.append(
                    SlideImage(
                        presentation_id=presentation_id,
                        slide_index=i,
                        image_path=dest_path,
                        source_type="video",
                    )
                )

            logger.info(
                f"Extracted {len(slides)} unique frames from "
                f"{len(frame_files)} sampled frames"
            )
            return slides

    def extract_keyframes(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract only keyframes (I-frames) from video.

        This is faster than scene detection but may miss slide transitions.

        Args:
            video_path: Path to video file.
            out_dir: Output directory.
            presentation_id: Presentation ID.
            image_format: Output format.

        Returns:
            List of SlideImage objects.
        """
        if not video_path.exists():
            raise SlideExtractorError(f"Video file not found: {video_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if image_format == "jpeg" else image_format
        output_pattern = str(out_dir / f"slide_%03d.{ext}")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", "select='eq(pict_type,I)'",
            "-vsync", "vfr",
            "-q:v", "2",
            output_pattern,
            "-y",
        ]

        logger.info(f"Extracting keyframes from: {video_path}")

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,  # 5 minute timeout for keyframe extraction
            )
        except subprocess.TimeoutExpired as e:
            raise SlideExtractorError(f"FFmpeg keyframe extraction timed out after 5 minutes") from e
        except subprocess.CalledProcessError as e:
            raise SlideExtractorError(f"FFmpeg keyframe extraction failed: {e.stderr}") from e

        slides = []
        for i, frame_path in enumerate(sorted(out_dir.glob(f"slide_*.{ext}")), 1):
            slides.append(
                SlideImage(
                    presentation_id=presentation_id,
                    slide_index=i,
                    image_path=frame_path,
                    source_type="video",
                )
            )

        logger.info(f"Extracted {len(slides)} keyframes from video")
        return slides
