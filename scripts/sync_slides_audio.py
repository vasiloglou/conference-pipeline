#!/usr/bin/env python3
"""Synchronize slides with audio segments using Gemini AI."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

load_dotenv()

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with `pip install google-genai`."
    ) from exc


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path)
        ],
        capture_output=True,
        text=True
    )
    return float(result.stdout.strip())


def parse_slides(slides_path: Path) -> list[dict]:
    """Parse slides.md and extract slide content."""
    content = slides_path.read_text()

    # Split by --- (slide separators)
    slides_raw = re.split(r'\n---\n', content)

    slides = []
    for i, slide_content in enumerate(slides_raw):
        if i == 0:  # Skip frontmatter
            continue

        # Extract title (first heading)
        title_match = re.search(r'^#+\s*(.+)$', slide_content, re.MULTILINE)
        title = title_match.group(1) if title_match else f"Slide {i}"

        # Extract comments (speaker notes)
        comments = re.findall(r'<!--\s*(.+?)\s*-->', slide_content, re.DOTALL)

        # Check if it's an image slide
        has_image = '![' in slide_content

        # Check for stat slides
        is_stat = '_class: stat' in slide_content
        stat_value = None
        if is_stat:
            stat_match = re.search(r'#\s*(\+?\d+\.?\d*%?)', slide_content)
            if stat_match:
                stat_value = stat_match.group(1)

        slides.append({
            "index": i,
            "title": title.strip(),
            "comments": comments,
            "has_image": has_image,
            "is_stat": is_stat,
            "stat_value": stat_value,
            "content_preview": slide_content[:200].strip()
        })

    return slides


def create_sync_with_gemini(
    slides: list[dict],
    segments: list[dict],
    segment_durations: list[float],
    model_name: str = default_model_for_provider("presentation.audio_sync", "gemini"),
) -> list[dict]:
    """Use Gemini to create intelligent slide-audio synchronization."""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    # Build prompt with context
    slides_summary = []
    for s in slides:
        notes = "; ".join(s["comments"]) if s["comments"] else "no notes"
        slides_summary.append(f"Slide {s['index']}: {s['title']} (notes: {notes[:100]})")

    segments_summary = []
    for i, seg in enumerate(segments):
        segments_summary.append(f"Segment {i+1} ({segment_durations[i]:.1f}s): {seg['speaker']}: {seg['text'][:150]}...")

    prompt = f"""You are helping synchronize a podcast presentation.

SLIDES (total {len(slides)} slides):
{chr(10).join(slides_summary)}

AUDIO SEGMENTS (total {len(segments)} segments):
{chr(10).join(segments_summary)}

Create a mapping of which slide(s) should be shown during each audio segment.

Rules:
1. The title/intro slide (slide 1-2) should show during segments 1-2
2. When speakers discuss a specific paper or topic, show the relevant slide
3. Image slides should be shown when visuals are referenced
4. Stat slides (+3.2%, 20%, etc.) should show when those statistics are mentioned
5. Section title slides should show briefly when transitioning topics
6. Multiple segments can map to the same slide if the discussion continues
7. Some slides may be skipped if not referenced in the audio

Return a JSON array where each object has:
- "segment": segment number (1-indexed)
- "slide": slide number to show (1-indexed)
- "notes": brief reason for the mapping

Output ONLY valid JSON, no other text."""

    response = gemini_generate_content(client, caller=__file__,
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 8000,
        }
    )

    # Parse response
    text = response.text.strip()

    # Remove markdown code blocks if present
    if "```" in text:
        # Find JSON within code blocks
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Try removing just the markers
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

    # Try to extract JSON array from text
    if not text.startswith('['):
        # Find the first [ and last ]
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]

    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error: {e}")
        print(f"Response preview: {text[:500]}...")

        # Try to fix common JSON issues
        # Remove trailing commas
        fixed_text = re.sub(r',\s*([}\]])', r'\1', text)
        # Fix single quotes to double quotes
        fixed_text = fixed_text.replace("'", '"')

        try:
            mapping = json.loads(fixed_text)
            print("Fixed JSON parsing issues")
        except json.JSONDecodeError:
            # Parse line by line to extract as much as possible
            print("Attempting line-by-line extraction...")
            mapping = []
            # More flexible pattern that handles varying whitespace and field order
            pattern = r'"segment"\s*:\s*(\d+).*?"slide"\s*:\s*(\d+)'
            for match in re.finditer(pattern, text, re.DOTALL):
                seg_num = int(match.group(1))
                slide_num = int(match.group(2))
                # Check if we already have this segment
                if not any(m["segment"] == seg_num for m in mapping):
                    mapping.append({
                        "segment": seg_num,
                        "slide": slide_num,
                        "notes": "extracted"
                    })

            if len(mapping) < len(segments) // 2:
                # Not enough valid entries - use smart fallback
                print(f"Only {len(mapping)} entries extracted, using smart fallback mapping...")
                mapping = create_smart_fallback_mapping(slides, segments)
            elif len(mapping) < len(segments):
                print(f"Extracted {len(mapping)} entries via regex")
                # Fill in missing segments by interpolating
                mapping = fill_missing_segments(mapping, len(segments), len(slides))

    return mapping


def fill_missing_segments(partial_mapping: list[dict], total_segments: int, total_slides: int) -> list[dict]:
    """Fill in missing segments by interpolating between known mappings."""
    # Sort by segment number
    partial_mapping.sort(key=lambda x: x["segment"])

    # Create a dict for quick lookup
    known = {m["segment"]: m["slide"] for m in partial_mapping}

    full_mapping = []
    last_known_slide = 1

    for seg in range(1, total_segments + 1):
        if seg in known:
            slide = known[seg]
            last_known_slide = slide
        else:
            # Find next known segment
            next_known = None
            for future_seg in range(seg + 1, total_segments + 1):
                if future_seg in known:
                    next_known = (future_seg, known[future_seg])
                    break

            if next_known:
                # Interpolate
                gap = next_known[0] - seg
                slide_diff = next_known[1] - last_known_slide
                slide = last_known_slide + (slide_diff * 1 // (gap + 1))
            else:
                # Use last known
                slide = last_known_slide

        slide = max(1, min(slide, total_slides))
        full_mapping.append({
            "segment": seg,
            "slide": slide,
            "notes": "interpolated" if seg not in known else "extracted"
        })
        last_known_slide = slide

    return full_mapping


def create_smart_fallback_mapping(slides: list[dict], segments: list[dict]) -> list[dict]:
    """Create a smarter fallback mapping based on content keywords."""
    mapping = []

    # Define slide ranges for different topics based on slide titles
    slide_topics = {}
    for s in slides:
        title = s["title"].lower()
        if "data renaissance" in title or "agenda" in title:
            slide_topics[s["index"]] = "intro"
        elif "hdd" in title or "hyperbolic" in title:
            slide_topics[s["index"]] = "hdd"
        elif "ld3m" in title or "latent diffusion" in title or "diffusion" in title:
            slide_topics[s["index"]] = "ld3m"
        elif "boomerang" in title:
            slide_topics[s["index"]] = "boomerang"
        elif "linear gradient" in title:
            slide_topics[s["index"]] = "lgm"
        elif "cleanpatrick" in title or "prefcleanbench" in title or "cleaning" in title:
            slide_topics[s["index"]] = "cleaning"
        elif "dataset inference" in title or "membership" in title or "ownership" in title:
            slide_topics[s["index"]] = "legal"
        elif "takeaway" in title or "key" in title:
            slide_topics[s["index"]] = "conclusion"
        elif s["is_stat"]:
            slide_topics[s["index"]] = f"stat_{s['stat_value']}"

    # Map segments to slides based on content keywords
    current_slide = 1
    for i, seg in enumerate(segments):
        text = seg["text"].lower()
        segment_num = i + 1

        # Determine which slide to show based on keywords
        if i < 2:
            current_slide = 1  # Title/intro
        elif "agenda" in text or "today we" in text:
            current_slide = 2
        elif "hdd" in text or "hyperbolic dataset distillation" in text:
            # Find HDD slides
            for idx, topic in slide_topics.items():
                if topic == "hdd":
                    current_slide = idx
                    break
        elif "ld3m" in text or "latent diffusion" in text:
            for idx, topic in slide_topics.items():
                if topic == "ld3m":
                    current_slide = idx
                    break
        elif "boomerang" in text:
            for idx, topic in slide_topics.items():
                if topic == "boomerang":
                    current_slide = idx
                    break
        elif "cleanpatrick" in text or "prefcleanbench" in text or "data cleaning" in text:
            for idx, topic in slide_topics.items():
                if topic == "cleaning":
                    current_slide = idx
                    break
        elif "+3.2%" in text or "3.2 percent" in text or "3.2%" in text:
            for idx, topic in slide_topics.items():
                if "3.2" in str(topic):
                    current_slide = idx
                    break
        elif "20%" in text and ("reduction" in text or "core set" in text):
            for idx, topic in slide_topics.items():
                if "20" in str(topic):
                    current_slide = idx
                    break
        elif "dataset inference" in text or "ownership" in text or "membership" in text:
            for idx, topic in slide_topics.items():
                if topic == "legal":
                    current_slide = idx
                    break
        elif "takeaway" in text or "wrap up" in text or "conclusion" in text:
            for idx, topic in slide_topics.items():
                if topic == "conclusion":
                    current_slide = idx
                    break

        mapping.append({
            "segment": segment_num,
            "slide": min(current_slide, len(slides)),
            "notes": "smart_fallback"
        })

    return mapping


def generate_sync_data(
    slides: list[dict],
    segments: list[dict],
    segment_durations: list[float],
    mapping: list[dict]
) -> dict:
    """Generate final synchronization data with timestamps."""

    # Calculate cumulative timestamps
    timestamps = [0.0]
    for dur in segment_durations:
        timestamps.append(timestamps[-1] + dur)

    # Build sync data
    sync_entries = []
    for m in mapping:
        seg_idx = m["segment"] - 1  # Convert to 0-indexed
        if seg_idx < len(segment_durations):
            entry = {
                "segment": m["segment"],
                "slide": m["slide"],
                "start_time": timestamps[seg_idx],
                "end_time": timestamps[seg_idx + 1],
                "duration": segment_durations[seg_idx],
                "notes": m.get("notes", "")
            }
            sync_entries.append(entry)

    return {
        "total_duration": sum(segment_durations),
        "total_slides": len(slides),
        "total_segments": len(segments),
        "sync": sync_entries
    }


def main():
    parser = argparse.ArgumentParser(description="Sync slides with audio segments")
    parser.add_argument("podcast_dir", type=Path, help="Directory containing script.json and audio_segments/")
    parser.add_argument("slides_dir", type=Path, help="Directory containing slides.md")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file for sync data")
    parser.add_argument(
        "--model",
        default=default_model_for_provider("presentation.audio_sync", "gemini"),
        help="Gemini model to use",
    )
    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from podcast_dir/slides_dir to avoid falling back to Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(
        args.podcast_dir
    ) or infer_conference_workspace(args.slides_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "sync_slides_audio")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    # Load data
    script_path = args.podcast_dir / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"script.json not found in {args.podcast_dir}")

    with open(script_path) as f:
        script = json.load(f)
    segments = script["segments"]

    slides_path = args.slides_dir / "slides.md"
    if not slides_path.exists():
        raise FileNotFoundError(f"slides.md not found in {args.slides_dir}")

    slides = parse_slides(slides_path)
    print(f"Parsed {len(slides)} slides")

    # Get audio durations
    audio_dir = args.podcast_dir / "audio_segments"
    segment_durations = []
    for i in range(1, len(segments) + 1):
        audio_path = audio_dir / f"segment_{i:03d}_01.mp3"
        if audio_path.exists():
            dur = get_audio_duration(audio_path)
            segment_durations.append(dur)
        else:
            print(f"Warning: {audio_path} not found")
            segment_durations.append(5.0)  # Default

    print(f"Got durations for {len(segment_durations)} segments")
    print(f"Total audio duration: {sum(segment_durations):.1f}s")

    # Use Gemini to create mapping
    print(f"Using {args.model} to create slide-audio mapping...")
    mapping = create_sync_with_gemini(slides, segments, segment_durations, args.model)
    print(f"Created mapping for {len(mapping)} segments")

    # Generate sync data
    sync_data = generate_sync_data(slides, segments, segment_durations, mapping)

    # Output
    output_path = args.output or args.slides_dir / "sync_data.json"
    with open(output_path, "w") as f:
        json.dump(sync_data, f, indent=2)

    print(f"Sync data written to {output_path}")

    return sync_data


if __name__ == "__main__":
    main()
