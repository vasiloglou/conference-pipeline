# slides_extractor.py - Slide Extraction

Extract slide images from PDFs, image archives, and videos.

## Location

`src/conference_pipeline/slides_extractor.py`

## Classes

### SlideExtractor

```python
class SlideExtractor:
    def __init__(self, settings: PipelineSettings):
        """Initialize slide extractor."""

    def extract_from_pdf(
        self,
        pdf_path: Path,
        out_dir: Path,
        presentation_id: str,
        dpi: int = 150,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Render PDF pages as images."""

    def extract_from_images_archive(
        self,
        archive_path: Path,
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Extract slides from a ZIP archive of images."""

    def extract_from_image_urls(
        self,
        image_paths: list[Path],
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Normalize already-downloaded images into slide_### names."""

    def convert_images_to_jpeg(
        self,
        slides: list[SlideImage],
        quality: int = 95,
    ) -> list[SlideImage]:
        """Convert slides to JPEG format."""

    def extract_from_video(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        sample_interval: float = 1.0,
        similarity_threshold: float = 0.95,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract unique slide frames using scene detection or sampling."""

    def extract_from_slide_videos(
        self,
        video_files: list[Path],
        out_dir: Path,
        presentation_id: str,
        crop_bottom_pixels: int = 80,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract one frame from each slide video segment."""

    def extract_from_video_url_with_chapters(
        self,
        video_url: str,
        chapters: list[dict],
        out_dir: Path,
        presentation_id: str,
        crop_bottom_pixels: int = 80,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract frames at chapter timestamps."""

    def extract_keyframes(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """Extract I-frames only."""
```

## Dependencies

- PDF: `pymupdf` preferred, `pdf2image` + `poppler` fallback.
- Video: `ffmpeg` required for frame extraction.
- Sampling fallback: `Pillow` + `imagehash`.

## Usage Example

```python
from conference_pipeline.slides_extractor import SlideExtractor
from pathlib import Path

extractor = SlideExtractor(settings)
slides = extractor.extract_from_pdf(
    Path("slides/raw.pdf"),
    Path("slides/images"),
    presentation_id="paper-123",
)
```

## Source References

- `src/conference_pipeline/slides_extractor.py`
- `src/conference_pipeline/models.py`
