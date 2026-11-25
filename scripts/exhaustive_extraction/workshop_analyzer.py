"""
Workshop analyzer module for identifying individual talks within workshops.

Uses Gemini's large context window to analyze entire workshop transcripts
and identify individual talk boundaries, speakers, and topics.

Two-phase approach:
1. Phase 1: Analyze full workshop with Gemini to identify talks
2. Phase 2: Process 5-minute batches with talk context
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
)

# Gemini model for large context analysis
GEMINI_MODEL = default_model_for_provider("analysis.workshop_segmentation", "gemini")
# Lower threshold to avoid API timeout (60s default)
# 40K tokens processes in ~30-40 seconds typically
MAX_CONTEXT_TOKENS = 40_000
OVERLAP_SEC = 600  # 10-minute overlap between chunks


@dataclass
class Talk:
    """Represents an individual talk within a workshop."""
    talk_id: str
    title: str
    speaker: str
    start_sec: float
    end_sec: float
    description: str


@dataclass
class WorkshopAnalysis:
    """Result of workshop analysis."""
    workshop_id: str
    workshop_title: str
    talks: List[Talk]
    total_duration_sec: float
    analysis_chunks: int  # How many chunks were needed


class WorkshopAnalyzer:
    """Analyze workshops to identify individual talks."""

    WORKSHOP_INDICATORS = [
        "workshop",
        "symposium",
        "tutorial",
        "bootcamp",
    ]

    # Minimum segment count to consider as workshop (assumes ~10s per segment)
    MIN_WORKSHOP_SEGMENTS = 500  # ~1.4 hours

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini API.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
        """
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def is_workshop(self, pres_id: str, segment_count: int) -> bool:
        """Detect if presentation is a workshop.

        Args:
            pres_id: Presentation ID
            segment_count: Number of transcript segments

        Returns:
            True if presentation should be treated as a workshop
        """
        # Check name for workshop indicators
        pres_lower = pres_id.lower()
        if any(ind in pres_lower for ind in self.WORKSHOP_INDICATORS):
            return True

        # Check length (workshops typically >1 hour)
        if segment_count > self.MIN_WORKSHOP_SEGMENTS:
            return True

        return False

    def analyze_workshop(
        self,
        pres_id: str,
        transcript: str,
        slides: List[Dict],
        title: Optional[str] = None
    ) -> WorkshopAnalysis:
        """Analyze workshop to identify individual talks.

        Uses Gemini's large context to process full transcript.
        Falls back to chunking with overlap if transcript exceeds context.

        Args:
            pres_id: Workshop presentation ID
            transcript: Full workshop transcript
            slides: List of slides with OCR text and timestamps
            title: Workshop title (optional)

        Returns:
            WorkshopAnalysis with identified talks
        """
        # Estimate token count (rough: ~1.3 tokens per word)
        estimated_tokens = len(transcript.split()) * 1.3
        slides_text = self._format_slides(slides)
        estimated_tokens += len(slides_text.split()) * 1.3

        # Check if we need chunking
        if estimated_tokens < MAX_CONTEXT_TOKENS:
            # Single pass analysis
            talks = self._analyze_single_pass(pres_id, transcript, slides_text, title)
            chunks_used = 1
        else:
            # Chunk and merge analysis
            talks = self._analyze_with_chunks(pres_id, transcript, slides, title)
            chunks_used = max(1, int(estimated_tokens / MAX_CONTEXT_TOKENS) + 1)

        # Calculate total duration
        total_duration = max(t.end_sec for t in talks) if talks else 0

        return WorkshopAnalysis(
            workshop_id=pres_id,
            workshop_title=title or pres_id,
            talks=talks,
            total_duration_sec=total_duration,
            analysis_chunks=chunks_used
        )

    def _analyze_single_pass(
        self,
        pres_id: str,
        transcript: str,
        slides_text: str,
        title: Optional[str]
    ) -> List[Talk]:
        """Analyze workshop in a single Gemini call.

        Args:
            pres_id: Workshop ID
            transcript: Full transcript
            slides_text: Formatted slides text
            title: Workshop title

        Returns:
            List of identified talks
        """
        prompt = self._build_analysis_prompt(transcript, slides_text, title)

        try:
            response = gemini_generate_content(self.client, caller=__file__,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=8000,
                    temperature=0.0
                )
            )
            if response.text is None:
                raise ValueError("Gemini returned empty response (response.text is None)")
            return self._parse_talks_response(response.text, pres_id)
        except Exception as e:
            print(f"      ⚠️  Workshop analysis failed: {e}")
            # Return single "Unknown Talk" covering entire workshop
            return [Talk(
                talk_id="talk_001",
                title="Full Workshop",
                speaker="Unknown",
                start_sec=0,
                end_sec=float("inf"),
                description="Workshop analysis failed - treating as single talk"
            )]

    def _analyze_with_chunks(
        self,
        pres_id: str,
        transcript: str,
        slides: List[Dict],
        title: Optional[str]
    ) -> List[Talk]:
        """Analyze workshop in chunks with overlap and merge results.

        Args:
            pres_id: Workshop ID
            transcript: Full transcript
            slides: Slides list
            title: Workshop title

        Returns:
            Merged list of talks
        """
        chunks = self._build_context_chunks(transcript, slides)
        all_talks = []

        for i, chunk in enumerate(chunks):
            print(f"      📊 Analyzing chunk {i + 1}/{len(chunks)}...")
            slides_text = self._format_slides(chunk["slides"])
            talks = self._analyze_single_pass(
                pres_id,
                chunk["transcript"],
                slides_text,
                title
            )
            # Adjust talk timestamps by chunk offset
            for talk in talks:
                talk.start_sec += chunk["offset_sec"]
                talk.end_sec += chunk["offset_sec"]
            all_talks.extend(talks)

        # Merge overlapping talks from different chunks
        return self._merge_overlapping_talks(all_talks)

    def _build_context_chunks(
        self,
        transcript: str,
        slides: List[Dict]
    ) -> List[Dict]:
        """Split transcript and slides into chunks with overlap.

        Splits on sentence/paragraph boundaries near time markers.

        Args:
            transcript: Full transcript with timestamps
            slides: Slides list

        Returns:
            List of chunks with transcript, slides, offset_sec
        """
        # Parse transcript to find timestamp markers
        # Assuming format like "[00:15:30] Text here..."
        lines = transcript.split("\n")
        chunks = []
        current_chunk_lines = []
        current_offset = 0
        chunk_start_sec = 0
        estimated_tokens = 0

        for line in lines:
            # Try to extract timestamp from line
            ts_sec = self._extract_timestamp_from_line(line)
            if ts_sec is not None:
                current_offset = ts_sec

            line_tokens = len(line.split()) * 1.3
            estimated_tokens += line_tokens
            current_chunk_lines.append(line)

            # Check if we should start a new chunk
            if estimated_tokens > MAX_CONTEXT_TOKENS * 0.9:  # 90% of limit
                # Find slides in this time range
                chunk_slides = [
                    s for s in slides
                    if s.get("timestamp_sec", 0) >= chunk_start_sec
                    and s.get("timestamp_sec", 0) < current_offset
                ]

                chunks.append({
                    "transcript": "\n".join(current_chunk_lines),
                    "slides": chunk_slides,
                    "offset_sec": chunk_start_sec,
                    "end_sec": current_offset
                })

                # Start new chunk with overlap
                overlap_start = max(0, current_offset - OVERLAP_SEC)
                chunk_start_sec = overlap_start
                current_chunk_lines = []
                estimated_tokens = 0

        # Add final chunk
        if current_chunk_lines:
            chunk_slides = [
                s for s in slides
                if s.get("timestamp_sec", 0) >= chunk_start_sec
            ]
            chunks.append({
                "transcript": "\n".join(current_chunk_lines),
                "slides": chunk_slides,
                "offset_sec": chunk_start_sec,
                "end_sec": current_offset
            })

        return chunks

    def _extract_timestamp_from_line(self, line: str) -> Optional[float]:
        """Extract timestamp in seconds from transcript line.

        Handles formats: [HH:MM:SS], (HH:MM:SS), HH:MM:SS:

        Args:
            line: Transcript line

        Returns:
            Timestamp in seconds or None
        """
        import re
        patterns = [
            r"\[(\d{1,2}):(\d{2}):(\d{2})\]",
            r"\((\d{1,2}):(\d{2}):(\d{2})\)",
            r"^(\d{1,2}):(\d{2}):(\d{2}):",
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return h * 3600 + m * 60 + s

        return None

    def _merge_overlapping_talks(self, talks: List[Talk]) -> List[Talk]:
        """Merge talks that likely refer to the same talk from different chunks.

        Uses fuzzy matching on title/speaker within overlap windows.

        Args:
            talks: List of talks from all chunks

        Returns:
            Merged list of unique talks
        """
        if not talks:
            return []

        # Sort by start time
        talks.sort(key=lambda t: t.start_sec)

        merged = [talks[0]]
        for talk in talks[1:]:
            last = merged[-1]
            # Check if talks overlap in time and have similar titles
            time_overlap = (
                last.end_sec > talk.start_sec - OVERLAP_SEC and
                talk.start_sec < last.end_sec + OVERLAP_SEC
            )
            title_similar = self._titles_similar(last.title, talk.title)

            if time_overlap and title_similar:
                # Merge: extend end time if needed
                last.end_sec = max(last.end_sec, talk.end_sec)
                # Keep more specific speaker if available
                if last.speaker == "Unknown" and talk.speaker != "Unknown":
                    last.speaker = talk.speaker
            else:
                merged.append(talk)

        # Renumber talk IDs
        for i, talk in enumerate(merged, 1):
            talk.talk_id = f"talk_{i:03d}"

        return merged

    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar (fuzzy match).

        Args:
            title1: First title
            title2: Second title

        Returns:
            True if titles are likely the same talk
        """
        # Simple word overlap check
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2)
        max_len = max(len(words1), len(words2))
        return overlap / max_len > 0.5

    def _format_slides(self, slides: List[Dict]) -> str:
        """Format slides list for prompt.

        Args:
            slides: List of slide dicts with ocr_text, timestamp_sec

        Returns:
            Formatted string for prompt
        """
        if not slides:
            return "(No slides available)"

        lines = []
        for slide in slides:
            ts = slide.get("timestamp_sec", 0)
            ts_str = f"{int(ts // 3600):02d}:{int((ts % 3600) // 60):02d}:{int(ts % 60):02d}"
            slide_id = slide.get("slide_id", "unknown")
            ocr_text = slide.get("ocr_text", "")[:200]  # Truncate long OCR
            lines.append(f"[{ts_str}] {slide_id}: {ocr_text}")

        return "\n".join(lines)

    def _build_analysis_prompt(
        self,
        transcript: str,
        slides_text: str,
        title: Optional[str]
    ) -> str:
        """Build prompt for talk identification.

        Args:
            transcript: Workshop transcript
            slides_text: Formatted slides
            title: Workshop title

        Returns:
            Prompt string
        """
        title_line = f"Workshop Title: {title}\n" if title else ""

        return f"""WORKSHOP TALK SEGMENTATION TASK

{title_line}
Analyze this workshop transcript and identify individual talks/presentations.

TRANSCRIPT:
{transcript}

SLIDES (with timestamps):
{slides_text}

INSTRUCTIONS:
1. Identify each distinct talk/presentation in this workshop
2. Look for speaker introductions, topic transitions, "thank you" moments, and Q&A sessions
3. For each talk, provide:
   - title: The talk title (from slides or introduction)
   - speaker: Speaker name if mentioned, otherwise "Speaker N"
   - start_sec: Start time in seconds from beginning
   - end_sec: End time in seconds from beginning
   - description: Brief 1-2 sentence description of the talk content

OUTPUT FORMAT (JSON):
{{
  "talks": [
    {{
      "title": "Opening Remarks",
      "speaker": "Workshop Organizers",
      "start_sec": 0,
      "end_sec": 600,
      "description": "Introduction to workshop themes and logistics"
    }},
    {{
      "title": "Scaling Multimodal Models",
      "speaker": "Dr. Jane Smith",
      "start_sec": 600,
      "end_sec": 2400,
      "description": "Discussion of scaling laws for vision-language models"
    }}
  ]
}}

IMPORTANT:
- Include ALL talks, even short ones (opening remarks, Q&A sessions)
- Timestamps must be in seconds from the start
- Be conservative: if unsure about boundaries, use the clearest transition points
- Use "Speaker N" (e.g., "Speaker 1", "Speaker 2") if name is not mentioned

Respond with ONLY the JSON object, no other text."""

    def _parse_talks_response(self, response_text: str, pres_id: str) -> List[Talk]:
        """Parse Gemini response into Talk objects.

        Args:
            response_text: JSON response from Gemini
            pres_id: Workshop ID for error messages

        Returns:
            List of Talk objects
        """
        # Extract JSON from potential markdown wrapping
        text = response_text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
            talks_data = data.get("talks", [])

            talks = []
            for i, t in enumerate(talks_data, 1):
                talks.append(Talk(
                    talk_id=f"talk_{i:03d}",
                    title=t.get("title", f"Talk {i}"),
                    speaker=t.get("speaker", f"Speaker {i}"),
                    start_sec=float(t.get("start_sec", 0)),
                    end_sec=float(t.get("end_sec", 0)),
                    description=t.get("description", "")
                ))

            return talks

        except (json.JSONDecodeError, KeyError) as e:
            print(f"      ⚠️  Failed to parse talks JSON for {pres_id}: {e}")
            return [Talk(
                talk_id="talk_001",
                title="Full Workshop",
                speaker="Unknown",
                start_sec=0,
                end_sec=float("inf"),
                description="Failed to parse talk structure"
            )]


def find_talks_for_batch(
    talks: List[Talk],
    batch_start_sec: float,
    batch_end_sec: float
) -> List[Talk]:
    """Find all talks that overlap with a time batch.

    Args:
        talks: List of workshop talks
        batch_start_sec: Batch start time
        batch_end_sec: Batch end time

    Returns:
        List of talks overlapping this batch (may be multiple)
    """
    overlapping = []
    for talk in talks:
        # Check for any overlap
        if not (talk.end_sec <= batch_start_sec or talk.start_sec >= batch_end_sec):
            overlapping.append(talk)
    return overlapping


def save_talks_json(talks: List[Talk], output_path: Path, workshop_info: Dict) -> None:
    """Save talks analysis to JSON file.

    Args:
        talks: List of talks
        output_path: Path to save JSON
        workshop_info: Additional workshop metadata
    """
    data = {
        "workshop_id": workshop_info.get("workshop_id"),
        "workshop_title": workshop_info.get("workshop_title"),
        "total_duration_sec": workshop_info.get("total_duration_sec"),
        "analysis_chunks": workshop_info.get("analysis_chunks"),
        "talks": [
            {
                "talk_id": t.talk_id,
                "title": t.title,
                "speaker": t.speaker,
                "start_sec": t.start_sec,
                "end_sec": t.end_sec,
                "description": t.description
            }
            for t in talks
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_talks_json(talks_path: Path) -> Optional[List[Talk]]:
    """Load previously saved talks analysis.

    Args:
        talks_path: Path to talks.json

    Returns:
        List of Talk objects or None if file doesn't exist
    """
    if not talks_path.exists():
        return None

    with open(talks_path) as f:
        data = json.load(f)

    return [
        Talk(
            talk_id=t["talk_id"],
            title=t["title"],
            speaker=t["speaker"],
            start_sec=t["start_sec"],
            end_sec=t["end_sec"],
            description=t["description"]
        )
        for t in data.get("talks", [])
    ]
