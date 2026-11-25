"""Storage backend for the NeurIPS pipeline."""

import json
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from conference_pipeline.config import PipelineSettings
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
    TranscriptChunk,
    VideoFile,
    __pipeline_version__,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a filesystem-safe slug.

    Args:
        text: Text to convert to slug.
        max_length: Maximum length of the slug (default 80).

    Returns:
        Lowercase slug with only alphanumeric chars and hyphens.
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace special characters with spaces
    slug = re.sub(r'[^\w\s-]', ' ', slug)
    # Replace whitespace with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    # Truncate to max_length, avoiding cutting in middle of word
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit('-', 1)[0]
    return slug


class StorageBackend:
    """Manages storage of pipeline artifacts and metadata."""

    def __init__(self, settings: PipelineSettings):
        """Initialize storage backend.

        Args:
            settings: Pipeline settings with storage configuration.
        """
        self.settings = settings
        self.root = settings.storage_root
        self.db_path = self.root / "metadata.sqlite"
        self._ensure_directories()
        self._init_database()

    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "presentations").mkdir(exist_ok=True)

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Presentations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS presentations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                session_title TEXT,
                track TEXT,
                date TEXT,
                neurips_url TEXT,
                slideslive_id TEXT,
                slideslive_url TEXT,
                has_video INTEGER,
                has_slides INTEGER,
                status TEXT,
                error_message TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Audio files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                format TEXT,
                duration_seconds REAL,
                sample_rate INTEGER,
                source_url TEXT,
                downloaded_at TEXT,
                pipeline_version TEXT,
                file_hash TEXT,
                file_size INTEGER,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id)
            )
        """)

        # Video files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                format TEXT,
                duration_seconds REAL,
                source_url TEXT,
                downloaded_at TEXT,
                pipeline_version TEXT,
                file_hash TEXT,
                file_size INTEGER,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id)
            )
        """)

        # Slide images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS slide_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                slide_index INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                timestamp_ms INTEGER,
                source_type TEXT,
                source_url TEXT,
                downloaded_at TEXT,
                pipeline_version TEXT,
                file_hash TEXT,
                file_size INTEGER,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id),
                UNIQUE(presentation_id, slide_index)
            )
        """)

        # Add timestamp_ms column if it doesn't exist (migration for existing databases)
        try:
            cursor.execute("ALTER TABLE slide_images ADD COLUMN timestamp_ms INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Transcripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                audio_path TEXT,
                text TEXT,
                start_time REAL,
                end_time REAL,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id)
            )
        """)

        # OCR results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                slide_index INTEGER NOT NULL,
                image_path TEXT,
                raw_text TEXT,
                engine TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id)
            )
        """)

        # Reference candidates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_candidates (
                id TEXT PRIMARY KEY,
                presentation_id TEXT NOT NULL,
                slide_index INTEGER,
                source_type TEXT,
                raw_text TEXT,
                detected_url TEXT,
                arxiv_id TEXT,
                doi TEXT,
                title TEXT,
                authors TEXT,
                normalized INTEGER DEFAULT 0,
                normalized_title TEXT,
                normalized_authors TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (presentation_id) REFERENCES presentations(id)
            )
        """)

        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                venue TEXT,
                year INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                url TEXT,
                pdf_url TEXT,
                local_pdf_path TEXT,
                source_url TEXT,
                downloaded_at TEXT,
                pipeline_version TEXT,
                file_hash TEXT,
                file_size INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Paper-presentation mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS presentation_papers (
                presentation_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                reference_candidate_id TEXT,
                PRIMARY KEY (presentation_id, paper_id),
                FOREIGN KEY (presentation_id) REFERENCES presentations(id),
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)

        # Session checkpoints table for resume capability
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                session_id TEXT PRIMARY KEY,
                session_title TEXT,
                session_type TEXT,
                status TEXT,
                presentation_count INTEGER DEFAULT 0,
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.debug(f"Database initialized at {self.db_path}")

    def get_presentation_dir_name(self, pres: Presentation) -> str:
        """Get the directory name for a presentation.

        Uses a slugified title for readability, falling back to ID if no title.

        Args:
            pres: Presentation to get directory name for.

        Returns:
            Directory name string (e.g., "building-foundational-models-for-robotics").
        """
        if pres.title:
            return slugify(pres.title)
        return pres.id

    def get_media_paths(self, pres: Presentation) -> MediaPaths:
        """Get paths for media artifacts of a presentation.

        Args:
            pres: Presentation to get paths for.

        Returns:
            MediaPaths with all artifact directories.
        """
        dir_name = self.get_presentation_dir_name(pres)
        base = self.root / "presentations" / dir_name
        return MediaPaths(
            audio_dir=base / "audio",
            video_dir=base / "video",
            slides_dir=base / "slides",
            slides_images_dir=base / "slides" / "images",
            transcripts_dir=base / "transcripts",
            ocr_dir=base / "ocr",
            papers_dir=base / "papers",
        )

    def ensure_presentation_dirs(self, pres: Presentation) -> MediaPaths:
        """Create directories for a presentation's artifacts.

        Args:
            pres: Presentation to create directories for.

        Returns:
            MediaPaths with created directories.
        """
        paths = self.get_media_paths(pres)
        paths.audio_dir.mkdir(parents=True, exist_ok=True)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        paths.slides_dir.mkdir(parents=True, exist_ok=True)
        paths.slides_images_dir.mkdir(parents=True, exist_ok=True)
        paths.transcripts_dir.mkdir(parents=True, exist_ok=True)
        paths.ocr_dir.mkdir(parents=True, exist_ok=True)
        paths.papers_dir.mkdir(parents=True, exist_ok=True)
        return paths

    def save_presentation(self, pres: Presentation) -> None:
        """Save or update a presentation in the database.

        Args:
            pres: Presentation to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        authors_json = json.dumps(pres.authors)
        date_str = pres.date.isoformat() if pres.date else None

        cursor.execute(
            """
            INSERT INTO presentations
            (id, title, authors, session_title, track, date, neurips_url,
             slideslive_id, slideslive_url, has_video, has_slides, status,
             error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                session_title=excluded.session_title,
                track=excluded.track,
                date=excluded.date,
                neurips_url=excluded.neurips_url,
                slideslive_id=excluded.slideslive_id,
                slideslive_url=excluded.slideslive_url,
                has_video=excluded.has_video,
                has_slides=excluded.has_slides,
                status=excluded.status,
                error_message=excluded.error_message,
                updated_at=excluded.updated_at
        """,
            (
                pres.id,
                pres.title,
                authors_json,
                pres.session_title,
                pres.track,
                date_str,
                pres.neurips_url,
                pres.slideslive_id,
                pres.slideslive_url,
                int(pres.has_video),
                int(pres.has_slides),
                pres.status.value,
                pres.error_message,
                now,
                now,
            ),
        )

        conn.commit()
        conn.close()

        # Also save JSON sidecar
        self._save_presentation_json(pres)

    def _save_presentation_json(self, pres: Presentation) -> None:
        """Save presentation metadata as JSON file."""
        dir_name = self.get_presentation_dir_name(pres)
        base = self.root / "presentations" / dir_name
        base.mkdir(parents=True, exist_ok=True)
        meta_path = base / "meta.json"

        data = asdict(pres)
        data["status"] = pres.status.value
        if pres.date:
            data["date"] = pres.date.isoformat()
        data["pipeline_version"] = __pipeline_version__

        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_presentation(self, pres_id: str) -> Presentation | None:
        """Load a presentation from the database.

        Args:
            pres_id: Presentation ID.

        Returns:
            Presentation if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM presentations WHERE id = ?", (pres_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_presentation(row)

    def _row_to_presentation(self, row: tuple) -> Presentation:
        """Convert database row to Presentation object."""
        (
            id_,
            title,
            authors_json,
            session_title,
            track,
            date_str,
            neurips_url,
            slideslive_id,
            slideslive_url,
            has_video,
            has_slides,
            status,
            error_message,
            created_at,
            updated_at,
        ) = row

        return Presentation(
            id=id_,
            title=title,
            authors=json.loads(authors_json) if authors_json else [],
            session_title=session_title,
            track=track,
            date=datetime.fromisoformat(date_str) if date_str else None,
            neurips_url=neurips_url or "",
            slideslive_id=slideslive_id,
            slideslive_url=slideslive_url,
            has_video=bool(has_video),
            has_slides=bool(has_slides),
            status=PresentationStatus(status),
            error_message=error_message,
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
        )

    def load_all_presentations(self) -> list[Presentation]:
        """Load all presentations from the database.

        Returns:
            List of all presentations.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM presentations ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_presentation(row) for row in rows]

    def load_presentations_by_status(self, status: PresentationStatus) -> list[Presentation]:
        """Load presentations with a specific status.

        Args:
            status: Status to filter by.

        Returns:
            List of matching presentations.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM presentations WHERE status = ? ORDER BY id",
            (status.value,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_presentation(row) for row in rows]

    # Audio file methods
    def save_audio_file(self, audio: AudioFile) -> None:
        """Save audio file metadata.

        Args:
            audio: AudioFile to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        prov = audio.provenance

        cursor.execute(
            """
            INSERT INTO audio_files
            (presentation_id, file_path, format, duration_seconds, sample_rate,
             source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                audio.presentation_id,
                str(audio.file_path),
                audio.format,
                audio.duration_seconds,
                audio.sample_rate,
                prov.source_url if prov else None,
                prov.downloaded_at.isoformat() if prov else None,
                prov.pipeline_version if prov else None,
                prov.file_hash if prov else None,
                prov.file_size if prov else None,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def load_audio_files(self, pres_id: str) -> list[AudioFile]:
        """Load audio files for a presentation.

        Args:
            pres_id: Presentation ID.

        Returns:
            List of AudioFile objects.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT presentation_id, file_path, format, duration_seconds, sample_rate,
                   source_url, downloaded_at, pipeline_version, file_hash, file_size
            FROM audio_files WHERE presentation_id = ?
        """,
            (pres_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            provenance = None
            if row[5]:  # source_url
                provenance = Provenance(
                    source_url=row[5],
                    downloaded_at=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
                    pipeline_version=row[7] or __pipeline_version__,
                    file_hash=row[8],
                    file_size=row[9],
                )
            results.append(
                AudioFile(
                    presentation_id=row[0],
                    file_path=Path(row[1]),
                    format=row[2] or "mp3",
                    duration_seconds=row[3],
                    sample_rate=row[4],
                    provenance=provenance,
                )
            )
        return results

    # Video file methods
    def save_video_file(self, video: VideoFile) -> None:
        """Save video file metadata.

        Args:
            video: VideoFile to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        prov = video.provenance

        cursor.execute(
            """
            INSERT INTO video_files
            (presentation_id, file_path, format, duration_seconds,
             source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                video.presentation_id,
                str(video.file_path),
                video.format,
                video.duration_seconds,
                prov.source_url if prov else None,
                prov.downloaded_at.isoformat() if prov else None,
                prov.pipeline_version if prov else None,
                prov.file_hash if prov else None,
                prov.file_size if prov else None,
                now,
            ),
        )

        conn.commit()
        conn.close()

    # Slide image methods
    def save_slide_image(self, slide: SlideImage) -> None:
        """Save slide image metadata.

        Args:
            slide: SlideImage to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        prov = slide.provenance

        cursor.execute(
            """
            INSERT OR REPLACE INTO slide_images
            (presentation_id, slide_index, image_path, width, height, timestamp_ms, source_type,
             source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                slide.presentation_id,
                slide.slide_index,
                str(slide.image_path),
                slide.width,
                slide.height,
                slide.timestamp_ms,
                slide.source_type,
                prov.source_url if prov else None,
                prov.downloaded_at.isoformat() if prov else None,
                prov.pipeline_version if prov else None,
                prov.file_hash if prov else None,
                prov.file_size if prov else None,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def save_slide_images(self, slides: list[SlideImage]) -> None:
        """Save multiple slide images.

        Args:
            slides: List of SlideImage objects.
        """
        for slide in slides:
            self.save_slide_image(slide)

    def load_slide_images(self, pres_id: str) -> list[SlideImage]:
        """Load slide images for a presentation.

        Args:
            pres_id: Presentation ID.

        Returns:
            List of SlideImage objects.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT presentation_id, slide_index, image_path, width, height, timestamp_ms, source_type,
                   source_url, downloaded_at, pipeline_version, file_hash, file_size
            FROM slide_images WHERE presentation_id = ?
            ORDER BY slide_index
        """,
            (pres_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            provenance = None
            if row[7]:  # source_url
                provenance = Provenance(
                    source_url=row[7],
                    downloaded_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                    pipeline_version=row[9] or __pipeline_version__,
                    file_hash=row[10],
                    file_size=row[11],
                )
            results.append(
                SlideImage(
                    presentation_id=row[0],
                    slide_index=row[1],
                    image_path=Path(row[2]),
                    width=row[3],
                    height=row[4],
                    timestamp_ms=row[5],
                    source_type=row[6] or "pdf",
                    provenance=provenance,
                )
            )
        return results

    def export_slide_sync_data(self, pres: Presentation) -> Path | None:
        """Export slide synchronization data as a JSON file.

        Creates a sync.json file with slide timing information for
        synchronizing slides with audio/video.

        Args:
            pres: Presentation to export sync data for.

        Returns:
            Path to the sync.json file, or None if no slides.
        """
        slides = self.load_slide_images(pres.id)
        if not slides:
            return None

        paths = self.get_media_paths(pres)
        sync_path = paths.slides_dir / "sync.json"

        sync_data = {
            "presentation_id": pres.id,
            "title": pres.title,
            "slideslive_id": pres.slideslive_id,
            "slides": [
                {
                    "index": s.slide_index,
                    "image": s.image_path.name,
                    "timestamp_ms": s.timestamp_ms,
                    "timestamp_sec": s.timestamp_ms / 1000 if s.timestamp_ms is not None else None,
                }
                for s in slides
            ],
        }

        with open(sync_path, "w") as f:
            json.dump(sync_data, f, indent=2)

        logger.info(f"Exported slide sync data to {sync_path}")
        return sync_path

    def save_transcript(self, pres_id: str, chunks: list[TranscriptChunk]) -> None:
        """Save transcript chunks for a presentation.

        Args:
            pres_id: Presentation ID.
            chunks: List of transcript chunks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        # Delete existing chunks
        cursor.execute("DELETE FROM transcripts WHERE presentation_id = ?", (pres_id,))

        # Insert new chunks
        for chunk in chunks:
            cursor.execute(
                """
                INSERT INTO transcripts
                (presentation_id, chunk_index, audio_path, text, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pres_id,
                    chunk.index,
                    str(chunk.audio_path),
                    chunk.text,
                    chunk.start_time,
                    chunk.end_time,
                    now,
                ),
            )

        conn.commit()
        conn.close()

        # Save JSON sidecar
        paths = self.get_media_paths(Presentation(id=pres_id, title=""))
        paths.transcripts_dir.mkdir(parents=True, exist_ok=True)

        transcript_json = paths.transcripts_dir / "transcript.json"
        with open(transcript_json, "w") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, default=str)

        # Save plain text version (merged)
        transcript_txt = paths.transcripts_dir / "transcript.txt"
        merged_text = self.merge_transcript_chunks(chunks)
        with open(transcript_txt, "w") as f:
            f.write(merged_text)

    def save_transcript_chunk(self, pres_id: str, chunk: TranscriptChunk) -> None:
        """Save a single transcript chunk (for incremental/resume support).

        Args:
            pres_id: Presentation ID.
            chunk: Single transcript chunk to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO transcripts
            (presentation_id, chunk_index, audio_path, text, start_time, end_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pres_id,
                chunk.index,
                str(chunk.audio_path),
                chunk.text,
                chunk.start_time,
                chunk.end_time,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_completed_chunk_indices(self, pres_id: str) -> set[int]:
        """Get indices of chunks that have been successfully transcribed.

        Args:
            pres_id: Presentation ID.

        Returns:
            Set of completed chunk indices.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_index FROM transcripts
            WHERE presentation_id = ? AND text NOT LIKE '[Transcription error:%'
            ORDER BY chunk_index
        """,
            (pres_id,),
        )
        indices = {row[0] for row in cursor.fetchall()}
        conn.close()

        return indices

    def merge_transcript_chunks(self, chunks: list[TranscriptChunk]) -> str:
        """Merge transcript chunks into a single coherent transcript.

        Handles overlap between chunks by removing duplicate sentences at boundaries.

        Args:
            chunks: List of transcript chunks in order.

        Returns:
            Merged transcript text.
        """
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0].text.strip()

        # Sort by index to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.index)

        merged_parts = []
        for i, chunk in enumerate(sorted_chunks):
            text = chunk.text.strip()

            # Skip error chunks
            if text.startswith("[Transcription error:"):
                continue

            if i == 0:
                # First chunk - use full text
                merged_parts.append(text)
            else:
                # For subsequent chunks, try to remove overlap
                # Split into sentences and find overlap with previous chunk
                prev_text = merged_parts[-1] if merged_parts else ""
                text = self._remove_overlap(prev_text, text)
                if text:
                    merged_parts.append(text)

        return "\n\n".join(merged_parts)

    def _remove_overlap(self, prev_text: str, curr_text: str) -> str:
        """Remove overlapping content between consecutive chunks.

        Args:
            prev_text: Text from previous chunk.
            curr_text: Text from current chunk.

        Returns:
            Current text with overlap removed.
        """
        if not prev_text or not curr_text:
            return curr_text

        # Get last few sentences from previous chunk
        prev_sentences = self._split_sentences(prev_text)
        curr_sentences = self._split_sentences(curr_text)

        if not prev_sentences or not curr_sentences:
            return curr_text

        # Look for overlap in the last N sentences of prev and first N of curr
        overlap_window = min(5, len(prev_sentences), len(curr_sentences))

        for overlap_size in range(overlap_window, 0, -1):
            prev_end = prev_sentences[-overlap_size:]
            curr_start = curr_sentences[:overlap_size]

            # Check if they match (case-insensitive, whitespace-normalized)
            prev_normalized = [s.lower().strip() for s in prev_end]
            curr_normalized = [s.lower().strip() for s in curr_start]

            if prev_normalized == curr_normalized:
                # Found overlap - return text after the overlap
                return " ".join(curr_sentences[overlap_size:])

        # No clear overlap found - return full text
        return curr_text

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting on common terminators
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def load_transcript(self, pres_id: str) -> list[TranscriptChunk]:
        """Load transcript chunks for a presentation.

        Args:
            pres_id: Presentation ID.

        Returns:
            List of transcript chunks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_index, audio_path, text, start_time, end_time
            FROM transcripts WHERE presentation_id = ?
            ORDER BY chunk_index
        """,
            (pres_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            TranscriptChunk(
                index=row[0],
                audio_path=Path(row[1]) if row[1] else Path(),
                text=row[2] or "",
                start_time=row[3],
                end_time=row[4],
            )
            for row in rows
        ]

    def save_ocr_result(self, result: OcrResult) -> None:
        """Save OCR result for a slide.

        Args:
            result: OCR result to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO ocr_results
            (presentation_id, slide_index, image_path, raw_text, engine, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.presentation_id,
                result.slide_index,
                str(result.image_path),
                result.raw_text,
                result.engine,
                result.confidence,
                now,
            ),
        )

        conn.commit()
        conn.close()

        # Save text file
        paths = self.get_media_paths(Presentation(id=result.presentation_id, title=""))
        paths.ocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_file = paths.ocr_dir / f"slide_{result.slide_index:03d}.txt"
        with open(ocr_file, "w") as f:
            f.write(result.raw_text)

    def load_ocr_results(self, pres_id: str) -> list[OcrResult]:
        """Load all OCR results for a presentation.

        Args:
            pres_id: Presentation ID.

        Returns:
            List of OCR results.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT presentation_id, slide_index, image_path, raw_text, engine, confidence
            FROM ocr_results WHERE presentation_id = ?
            ORDER BY slide_index
        """,
            (pres_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            OcrResult(
                presentation_id=row[0],
                slide_index=row[1],
                image_path=Path(row[2]) if row[2] else Path(),
                raw_text=row[3] or "",
                engine=row[4] or "tesseract",
                confidence=row[5],
            )
            for row in rows
        ]

    def save_reference_candidates(self, refs: list[ReferenceCandidate]) -> None:
        """Save reference candidates.

        Args:
            refs: List of reference candidates to save.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        for ref in refs:
            authors_json = json.dumps(ref.authors) if ref.authors else None
            norm_authors_json = json.dumps(ref.normalized_authors) if ref.normalized_authors else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO reference_candidates
                (id, presentation_id, slide_index, source_type, raw_text,
                 detected_url, arxiv_id, doi, title, authors,
                 normalized, normalized_title, normalized_authors, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    ref.id,
                    ref.presentation_id,
                    ref.slide_index,
                    ref.source_type,
                    ref.raw_text,
                    ref.detected_url,
                    ref.arxiv_id,
                    ref.doi,
                    ref.title,
                    authors_json,
                    int(ref.normalized),
                    ref.normalized_title,
                    norm_authors_json,
                    ref.confidence,
                    now,
                ),
            )

        conn.commit()
        conn.close()

    def load_reference_candidates(self, pres_id: str) -> list[ReferenceCandidate]:
        """Load reference candidates for a presentation.

        Args:
            pres_id: Presentation ID.

        Returns:
            List of reference candidates.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, presentation_id, slide_index, source_type, raw_text,
                   detected_url, arxiv_id, doi, title, authors,
                   normalized, normalized_title, normalized_authors, confidence
            FROM reference_candidates WHERE presentation_id = ?
        """,
            (pres_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            ReferenceCandidate(
                id=row[0],
                presentation_id=row[1],
                slide_index=row[2],
                source_type=row[3],
                raw_text=row[4],
                detected_url=row[5],
                arxiv_id=row[6],
                doi=row[7],
                title=row[8],
                authors=json.loads(row[9]) if row[9] else None,
                normalized=bool(row[10]),
                normalized_title=row[11],
                normalized_authors=json.loads(row[12]) if row[12] else None,
                confidence=row[13],
            )
            for row in rows
        ]

    def save_paper(
        self,
        paper: Paper,
        presentation_ids: list[str] | None = None,
        reference_candidate_id: str | None = None,
    ) -> None:
        """Save a paper and optionally link to presentations.

        Args:
            paper: Paper to save.
            presentation_ids: Optional list of presentation IDs to link.
            reference_candidate_id: Optional reference candidate ID that led to this paper.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        authors_json = json.dumps(paper.authors)
        local_path = str(paper.local_pdf_path) if paper.local_pdf_path else None
        prov = paper.provenance

        cursor.execute(
            """
            INSERT INTO papers
            (id, title, authors, venue, year, doi, arxiv_id, url, pdf_url, local_pdf_path,
             source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                venue=excluded.venue,
                year=excluded.year,
                doi=excluded.doi,
                arxiv_id=excluded.arxiv_id,
                url=excluded.url,
                pdf_url=excluded.pdf_url,
                local_pdf_path=excluded.local_pdf_path,
                source_url=excluded.source_url,
                downloaded_at=excluded.downloaded_at,
                pipeline_version=excluded.pipeline_version,
                file_hash=excluded.file_hash,
                file_size=excluded.file_size,
                updated_at=excluded.updated_at
        """,
            (
                paper.id,
                paper.title,
                authors_json,
                paper.venue,
                paper.year,
                paper.doi,
                paper.arxiv_id,
                paper.url,
                paper.pdf_url,
                local_path,
                prov.source_url if prov else None,
                prov.downloaded_at.isoformat() if prov else None,
                prov.pipeline_version if prov else None,
                prov.file_hash if prov else None,
                prov.file_size if prov else None,
                now,
                now,
            ),
        )

        # Link to presentations with reference candidate tracking
        if presentation_ids:
            for pres_id in presentation_ids:
                cursor.execute(
                    """
                    INSERT INTO presentation_papers (presentation_id, paper_id, reference_candidate_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(presentation_id, paper_id) DO UPDATE SET
                        reference_candidate_id = COALESCE(excluded.reference_candidate_id, presentation_papers.reference_candidate_id)
                """,
                    (pres_id, paper.id, reference_candidate_id),
                )

        conn.commit()
        conn.close()

    def save_papers(self, papers: list[Paper]) -> None:
        """Save multiple papers.

        Args:
            papers: List of papers to save.
        """
        for paper in papers:
            self.save_paper(paper)

    def load_paper(self, paper_id: str) -> Paper | None:
        """Load a paper by ID.

        Args:
            paper_id: Paper ID.

        Returns:
            Paper if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, title, authors, venue, year, doi, arxiv_id, url, pdf_url, local_pdf_path,
                   source_url, downloaded_at, pipeline_version, file_hash, file_size
            FROM papers WHERE id = ?
        """,
            (paper_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        provenance = None
        if row[10]:  # source_url
            provenance = Provenance(
                source_url=row[10],
                downloaded_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
                pipeline_version=row[12] or __pipeline_version__,
                file_hash=row[13],
                file_size=row[14],
            )

        return Paper(
            id=row[0],
            title=row[1],
            authors=json.loads(row[2]) if row[2] else [],
            venue=row[3],
            year=row[4],
            doi=row[5],
            arxiv_id=row[6],
            url=row[7],
            pdf_url=row[8],
            local_pdf_path=Path(row[9]) if row[9] else None,
            provenance=provenance,
        )

    def export_presentations_json(self) -> None:
        """Export all presentations to a JSON file."""
        presentations = self.load_all_presentations()
        output_path = self.root / "presentations.json"

        data = []
        for pres in presentations:
            pres_dict = asdict(pres)
            pres_dict["status"] = pres.status.value
            if pres.date:
                pres_dict["date"] = pres.date.isoformat()
            pres_dict["pipeline_version"] = __pipeline_version__
            data.append(pres_dict)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(data)} presentations to {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with various statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Presentation counts by status
        cursor.execute(
            """
            SELECT status, COUNT(*) FROM presentations GROUP BY status
        """
        )
        stats["presentations_by_status"] = dict(cursor.fetchall())

        # Total counts
        cursor.execute("SELECT COUNT(*) FROM presentations")
        stats["total_presentations"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT presentation_id) FROM transcripts")
        stats["presentations_with_transcripts"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT presentation_id) FROM ocr_results")
        stats["presentations_with_ocr"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM slide_images")
        stats["total_slide_images"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM audio_files")
        stats["total_audio_files"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM papers")
        stats["total_papers"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM papers WHERE local_pdf_path IS NOT NULL")
        stats["papers_with_pdfs"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reference_candidates")
        stats["total_references"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reference_candidates WHERE normalized = 1")
        stats["normalized_references"] = cursor.fetchone()[0]

        conn.close()
        return stats

    # Session checkpoint methods for resume capability
    def mark_session_started(self, session_id: str, session_title: str, session_type: str) -> None:
        """Mark a session as started for checkpoint tracking.

        Args:
            session_id: Session identifier.
            session_title: Session title.
            session_type: Session type.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO session_checkpoints
            (session_id, session_title, session_type, status, started_at)
            VALUES (?, ?, ?, 'in_progress', ?)
        """,
            (session_id, session_title, session_type, now),
        )

        conn.commit()
        conn.close()
        logger.debug(f"Marked session {session_id} as started")

    def mark_session_completed(
        self, session_id: str, presentation_count: int, error_message: str | None = None
    ) -> None:
        """Mark a session as completed.

        Args:
            session_id: Session identifier.
            presentation_count: Number of presentations found.
            error_message: Optional error message if session failed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        status = "error" if error_message else "completed"

        cursor.execute(
            """
            UPDATE session_checkpoints
            SET status = ?, presentation_count = ?, error_message = ?, completed_at = ?
            WHERE session_id = ?
        """,
            (status, presentation_count, error_message, now, session_id),
        )

        conn.commit()
        conn.close()
        logger.debug(f"Marked session {session_id} as {status}")

    def is_session_completed(self, session_id: str) -> bool:
        """Check if a session has been completed.

        Args:
            session_id: Session identifier.

        Returns:
            True if session is completed, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status FROM session_checkpoints WHERE session_id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row and row[0] == "completed":
            return True
        return False

    def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint progress.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Session counts by status
        cursor.execute(
            """
            SELECT status, COUNT(*) FROM session_checkpoints GROUP BY status
        """
        )
        stats["sessions_by_status"] = dict(cursor.fetchall())

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM session_checkpoints")
        stats["total_sessions"] = cursor.fetchone()[0]

        # Total presentations from completed sessions
        cursor.execute(
            """
            SELECT SUM(presentation_count) FROM session_checkpoints WHERE status = 'completed'
        """
        )
        result = cursor.fetchone()[0]
        stats["total_presentations_discovered"] = result if result else 0

        conn.close()
        return stats

    def get_pending_sessions(self, all_sessions: list) -> list:
        """Get sessions that haven't been completed yet.

        Args:
            all_sessions: List of all Session objects to check.

        Returns:
            List of sessions that need to be processed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completed_ids = set()
        cursor.execute(
            """
            SELECT session_id FROM session_checkpoints WHERE status = 'completed'
        """
        )
        for row in cursor.fetchall():
            completed_ids.add(row[0])

        conn.close()

        # Return sessions not in completed set
        pending = [s for s in all_sessions if s.id not in completed_ids]
        logger.info(
            f"Found {len(pending)} pending sessions out of {len(all_sessions)} total"
        )
        return pending
