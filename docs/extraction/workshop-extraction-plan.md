# Workshop Extraction Plan: Talk-Aware Processing

## Overview

Workshops contain multiple talks by different speakers. Current approach treats them as one long presentation, losing talk-level context. This plan introduces a two-phase approach:

1. **Phase 1**: Use Gemini's large context to analyze entire workshop and identify individual talks
2. **Phase 2**: Process 5-minute batches with talk context, referencing as "Workshop/{talk_id}-Talk:timestamp"

## Current Issues to Fix

### 1. Validation Errors

#### Issue A: Missing `slide_path` in slide_references
```
slide_references.0.slide_path
  Field required [type=missing, input_value={'slide_id': 'slide_014',...
```

**Root Cause**: Model returns slide_references without slide_path field
**Fix**: Make slide_path optional in schema AND auto-populate from slide_id (or slide cache) when absent

#### Issue B: Timestamp bounds exceeded
```
timestamp_end_sec (8730.64) is after segment_end_sec (8392.81)
```

**Root Cause**: Model extracts timestamps beyond batched segment boundaries
**Fix**: Clip both start and end timestamps to segment bounds (no validation failure)

### 2. Batch Duration
- **Current**: 600s (10 minutes)
- **New**: 300s (5 minutes) - better granularity for talk-level extraction

## New Architecture

### Phase 1: Workshop Analysis (Gemini)

```
┌─────────────────────────────────────────────────────────────────┐
│  Workshop Recording (~20 hours)                                  │
│  - Full transcript (all segments concatenated)                   │
│  - All slide OCR text with timestamps                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ Send to Gemini (1M context)
┌─────────────────────────────────────────────────────────────────┐
│  Talk Segmentation Prompt:                                       │
│  "Analyze this workshop transcript and identify individual       │
│   talks. For each talk, provide:                                │
│   - Talk title                                                   │
│   - Speaker name (if mentioned)                                 │
│   - Start timestamp (HH:MM:SS)                                  │
│   - End timestamp (HH:MM:SS)                                    │
│   - Brief description (1-2 sentences)"                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: talks.json                                              │
│  {                                                               │
│    "workshop_id": "2nd-workshop-on-multi-modal...",             │
│    "workshop_title": "Multi-Modal Foundation Models",           │
│    "talks": [                                                    │
│      {                                                           │
│        "talk_id": "talk_001",                                   │
│        "title": "Opening Remarks",                              │
│        "speaker": "Workshop Organizers",                        │
│        "start_sec": 0,                                          │
│        "end_sec": 600,                                          │
│        "description": "Introduction to workshop themes"         │
│      },                                                          │
│      {                                                           │
│        "talk_id": "talk_002",                                   │
│        "title": "Scaling Multimodal Models",                    │
│        "speaker": "Dr. Jane Smith",                             │
│        "start_sec": 600,                                        │
│        "end_sec": 2400,                                         │
│        "description": "Discussion of scaling laws for VLMs"     │
│      },                                                          │
│      ...                                                         │
│    ]                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Talk-Aware Extraction (Sonnet)

```
┌─────────────────────────────────────────────────────────────────┐
│  For each 5-minute batch:                                        │
│  1. Look up which talk(s) this batch belongs to                 │
│  2. Include talk context in extraction prompt                    │
│  3. Structure extraction IDs as:                                │
│     "workshop-name/{talk_id}-talk-name:HH:MM:SS"                │
└─────────────────────────────────────────────────────────────────┘
```

### Updated Extraction Schema

```python
class ContentExtraction:
    # Existing fields...

    # New workshop/talk fields
    workshop_id: Optional[str] = None        # If from workshop
    talk_id: Optional[str] = None            # Talk within workshop
    talk_title: Optional[str] = None         # Human-readable talk title
    talk_speaker: Optional[str] = None       # Speaker name if known

    # Updated reference format
    source_reference: str  # "workshop-name/{talk_id}-Talk Title:HH:MM:SS"
```

## Implementation Steps

### Step 1: Fix Validation Issues

**File**: `scripts/exhaustive_extraction/schemas.py`

```python
# Make slide_path optional
class SlideReference(BaseModel):
    slide_id: str
    slide_path: Optional[str] = None  # Make optional
    timestamp_aligned: bool = False

# Add timestamp clipping option or relax validation
class ContentExtraction(BaseModel):
    # Add validator to clip timestamps
    @model_validator(mode='after')
    def clip_timestamps(self) -> 'ContentExtraction':
        if self.segment_start_sec is not None and self.timestamp_start_sec is not None:
            if self.timestamp_start_sec < self.segment_start_sec:
                self.timestamp_start_sec = self.segment_start_sec
        if self.segment_end_sec is not None and self.timestamp_end_sec is not None:
            if self.timestamp_end_sec > self.segment_end_sec:
                self.timestamp_end_sec = self.segment_end_sec
        return self
```

**File**: `scripts/exhaustive_extraction/processor.py`

```python
def _hydrate_slide_paths(self, slide_refs: list[SlideReference]) -> list[SlideReference]:
    for ref in slide_refs:
        if ref.slide_path:
            continue
        ref.slide_path = self.slides_cache.get(ref.slide_id) or f"{ref.slide_id}.pdf"
    return slide_refs
```

Call `_hydrate_slide_paths` immediately after loading slide references so downstream consumers always get a path.

### Step 2: Create Workshop Detection Module

**New File**: `scripts/exhaustive_extraction/workshop_analyzer.py`

```python
class WorkshopAnalyzer:
    """Analyze workshops to identify individual talks."""

    WORKSHOP_INDICATORS = [
        "workshop",
        "symposium",
        "tutorial",
        "bootcamp"
    ]

    def is_workshop(self, pres_id: str, segment_count: int) -> bool:
        """Detect if presentation is a workshop."""
        # Check name
        if any(ind in pres_id.lower() for ind in self.WORKSHOP_INDICATORS):
            return True
        # Check length (workshops typically >1 hour = >360 segments at 10s each)
        if segment_count > 500:
            return True
        return False

    def analyze_workshop(
        self,
        pres_id: str,
        transcript: str,
        slides: List[Dict]
    ) -> Dict:
        """Use Gemini to identify talks within workshop with chunking fallback."""
        chunks = self._build_context_chunks(transcript, slides, max_tokens=900_000, overlap_sec=600)
        talks = []
        for chunk in chunks:
            prompt = self._build_analysis_prompt(chunk["transcript"], chunk["slides"])
            result = self.gemini.generate_content(prompt)
            talks.extend(self._parse_talks(result.text))
        return self._merge_overlapping_talks(talks)

    def _build_context_chunks(self, transcript, slides, max_tokens: int, overlap_sec: int) -> List[Dict]:
        """Split transcript+slides on time boundaries with overlap to stay within context."""
        ...
```

### Step 3: Update Processor for Talk Context

**File**: `scripts/exhaustive_extraction/processor.py`

```python
class PresentationProcessor:
    def process_presentation(self, pres_id: str) -> bool:
        # Load segments
        segments = self.load_segments(pres_id)

        # Check if workshop
        if self.workshop_analyzer.is_workshop(pres_id, len(segments)):
            return self.process_workshop(pres_id, segments)
        else:
            return self.process_regular_presentation(pres_id, segments)

    def process_workshop(self, pres_id: str, segments: List) -> bool:
        # Phase 1: Analyze workshop structure
        full_transcript = self._concatenate_transcripts(segments)
        talks = self.workshop_analyzer.analyze_workshop(
            pres_id, full_transcript, self.slides_cache
        )

        # Save talks structure
        self._save_talks_json(pres_id, talks)

        # Phase 2: Process batches with talk context
        batched = self._batch_segments(segments)  # 5-minute batches

        for batch in batched:
            # Find all talks overlapping this batch (duplicate extraction)
            batch_talks = self._find_talks_for_batch(
                talks, batch["start_sec"], batch["end_sec"]
            )
            for talk in batch_talks:
                self.process_segment_with_talk_context(
                    pres_id, batch, talk
                )

    def _find_talks_for_batch(self, talks: List[Dict], start: float, end: float) -> List[Dict]:
        """Return all talks overlapping the batch window."""
        return [t for t in talks if not (t["end_sec"] <= start or t["start_sec"] >= end)]
```

### Step 4: Update Batch Duration

**File**: `scripts/run_exhaustive_extraction.py`

```python
parser.add_argument(
    "--batch-duration-limit",
    type=int,
    default=300,  # Changed from 600 to 300 (5 minutes)
    help="Max duration in seconds per batched segment (default: 300 = 5 minutes)"
)
```

### Step 5: Update Extraction Prompt

**File**: `scripts/exhaustive_extraction/prompts.py`

Add talk context to prompt:

```python
def build_workshop_extraction_prompt(
    workshop_title: str,
    talk_title: str,
    talk_speaker: str,
    segment_start_hms: str,
    segment_end_hms: str,
    transcript_text: str,
    slides_list_with_ocr: str,
    topic_guardrails: str
) -> str:
    return f"""
WORKSHOP SEGMENT EXTRACTION

Workshop: {workshop_title}
Current Talk: {talk_title}
Speaker: {talk_speaker}
Segment: {segment_start_hms} - {segment_end_hms}

TRANSCRIPT:
{transcript_text}

SLIDES IN THIS SEGMENT:
{slides_list_with_ocr}

{topic_guardrails}

Extract structured content from this workshop segment.
Reference format: Use "{workshop_title}/{talk_id}-{talk_title}:{segment_start_hms}" for citations.
...
"""
```

## File Changes Summary

| File | Change |
|------|--------|
| `schemas.py` | Make slide_path optional, clip start/end timestamps |
| `processor.py` | Hydrate missing slide_path, duplicate extraction for overlapping talks |
| `prompts.py` | Add workshop-specific prompt with talk context and talk_id-based references |
| `run_exhaustive_extraction.py` | Change default batch to 300s |
| **NEW** `workshop_analyzer.py` | Workshop analysis using Gemini with chunking fallback |

## Testing Plan

1. **Unit test workshop detection**: Test `is_workshop()` with various presentation IDs
2. **Schema validators**: Ensure start/end timestamps clip to bounds; `slide_path` auto-populates from cache/slide_id
3. **Overlap policy**: Verify batches spanning two talks are extracted twice with distinct `source_reference`
4. **Integration test talk analysis**: Run Gemini analysis on one workshop with and without chunking
5. **End-to-end test**: Process one workshop with new pipeline

## Cost Estimate

Use the canonical registry for current prices:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope exhaustive-extraction
```

### Phase 1 (Workshop Analysis)
- Input: ~500K tokens (full workshop transcript + slides)
- Output: ~2K tokens (talks.json)
- Cost per workshop: estimate with `src/conference_pipeline/costs/pricing_catalog.json`
- Total for 2 workshops: estimate with `conference_pipeline.costs`

### Phase 2 (Extraction)
- With 5-minute batches: ~240 batches per 20-hour workshop
- Per-batch provider cost depends on the selected direct/fallback model
- Total for 2 workshops: estimate with `conference_pipeline.costs`

### Total Workshop Cost: ~$2.74

## Timeline

1. **Fix validation issues**: 30 min
2. **Create workshop_analyzer.py**: 1 hour
3. **Update processor.py**: 1 hour
4. **Update prompts.py**: 30 min
5. **Testing**: 1 hour

**Total**: ~4 hours implementation

## Questions to Resolve

1. **Talk boundaries**: How to handle segments that span two talks?
   - **Decision**: Duplicate extraction for each overlapping talk

2. **Missing speaker names**: What if Gemini can't identify speaker?
   - Use "Unknown Speaker" or talk number
   - **Recommendation**: "Speaker {N}" where N is talk number

3. **Context overflow**: What if workshop analysis exceeds context?
   - **Decision**: Chunk transcript+slides on time boundaries with small overlaps; re-run analysis per chunk and merge talks

4. **Nested workshops**: Some workshops have sub-sessions
   - Flatten to single level for now
   - **Recommendation**: Keep simple, revisit if needed
