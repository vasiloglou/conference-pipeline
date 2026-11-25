# Exhaustive NeurIPS 2025 Analysis - Implementation Plan

## Overview

Build a **push-based extraction system** that scans all 9,825 papers + 131 presentations and extracts specific content relevant to each of the 20 research areas defined in NeurIPS_Analysis_Plan.md.

**Critical Requirement (REVISED):**
This is NOT just classification - it's **content extraction with timestamps**.
- For presentations: Extract specific segments (quotes/summaries) relevant to each topic WITH precise timestamp ranges
- For papers: Classify and extract relevant sentences from abstracts
- Store factual statements, not just labels

**User Requirements (Confirmed):**
- Budget: No strict limit - optimize for quality
- Multi-area: Content can be relevant to multiple topics - extract for each
- Integration: Keep separate from existing analysis (new `exhaustive_analysis/` directory)
- Workshops: Process in manageable chunks to extract relevant segments

**Estimated Cost:** ~$80-120 total (higher due to extraction vs classification)
**Estimated Time:** 4-6 hours processing time

## Core Strategy

### 1. Paper Processing: BATCH CLASSIFICATION + SENTENCE EXTRACTION
- **Strategy:** Process 20-50 papers per API call
- **Task:** For each paper, identify relevant topics AND extract specific sentences from the abstract
- **Model:** Claude Sonnet 4.6 for all papers
- **Output per paper:** List of (topic_id, relevant_sentences, confidence) tuples
- **Cost:** ~$25 for all 9,825 papers
- **Time:** ~15-25 minutes (197 batches)

### 2. Presentation Processing: SEGMENT-BY-SEGMENT EXTRACTION
- **Strategy:** Process presentations in ~2-5 minute chunks (using existing segments.json)
- **Task:** For each segment, identify if relevant to any topics and extract:
  - Verbatim quotes from transcript
  - Summary of key points
  - Slide references (which slides, what they show)
  - Precise timestamp ranges (start-end)
- **Model Selection:**
  - Use Claude Sonnet 4.6 for most segments
  - For very dense segments with many slides, use Gemini 2.5 Pro
- **Rationale:** Chunking allows precise timestamp extraction and manageable context
- **Cost:** ~$80-100 for all 131 presentations (higher due to extraction detail)
- **Time:** ~4-5 hours

### 3. Multi-Topic Extraction
- A single presentation segment can contain content for multiple topics
- Extract separately for each topic with its own timestamps/quotes
- Store all extractions in their respective topic files
- Each extraction is a standalone factual statement with full source attribution

### 4. Output Format (Per Extraction)
```json
{
  "source_type": "presentation",
  "source_id": "dl-code-agentic-era",
  "source_title": "Deep Learning for Code in the Agentic Era",
  "topic_id": "18",
  "topic_name": "Agentic AI and Autonomous Systems",
  "timestamp_start": "00:15:30",
  "timestamp_end": "00:17:45",
  "slide_references": ["slide_042", "slide_043"],
  "extraction_type": "verbatim_quote",
  "content": "We call this the demo soup problem - where you have many demos but no systematic way to evaluate them.",
  "summary": "Speaker introduces the 'demo soup problem' in agent evaluation",
  "relevance_score": 0.95,
  "model_used": "claude-sonnet-4-6"
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Day 1 Morning)

**Files to create:**
1. `scripts/exhaustive_classification/prompts.py`
   - Extraction prompt template with all 20 area definitions
   - Paper variant: "Extract sentences relevant to each topic"
   - Presentation segment variant: "Extract quotes/summaries with timestamps for each topic"

2. `scripts/exhaustive_classification/models.py`
   - Token counting utility
   - Model selection logic (Sonnet/Gemini based on segment density)
   - API wrappers with retry logic (reuse from claude_deep_research.py:47-70)

3. `scripts/exhaustive_classification/schemas.py`
   - Pydantic models for ContentExtraction (not just Classification)
   - Timestamp validation (HH:MM:SS format)
   - Slide reference validation

4. `scripts/exhaustive_classification/storage.py`
   - Multi-topic storage manager (extractions, not classifications)
   - JSONL writers (one extraction per line)
   - Progress tracking by presentation/segment

### Phase 2: Validation Run (Day 1 Afternoon)

**Validate on small sample:**
- 10 oral papers: Check that relevant sentences are extracted correctly
- 2 short presentations (1-2 segments each): Check timestamp precision and quote extraction
- Manual review: Verify timestamps match actual content, quotes are verbatim, summaries are accurate

**Success criteria:**
- Timestamps are precise (within 5 seconds of actual content)
- Quotes are verbatim from transcripts
- Slide references are correct
- Extractions are factual and relevant (not generic)
- Multi-topic extractions make sense

### Phase 3: High-Value Processing (Day 2)

**Process in order:**
1. Oral papers (77) - 2 batches, ~5 min
2. Spotlight papers (688) - 14 batches, ~25 min
3. Short presentations (80) - ~500 segments total, ~2 hours
4. Medium presentations (35) - ~1,000 segments total, ~3 hours

**Checkpoints:**
- Review extraction quality: Are timestamps precise? Are quotes accurate?
- Check topic distribution: Which topics have most extractions?
- Validate slide references match OCR data
- Adjust prompts if extraction quality issues found

### Phase 4: Bulk Processing (Day 3+)

**Complete remaining content:**
1. Poster papers (4,525) - 91 batches, ~2 hours
2. Workshop papers (3,944) - 79 batches, ~1.5 hours
3. Long workshops (16) - ~5,000 segments total, ~6-8 hours

**Monitoring:**
- Track segments processed per presentation
- Monitor extraction yield (extractions per segment)
- Cost tracking per topic
- Failure logging with segment-level resume capability

### Phase 5: Report Generation (Day 3 Afternoon)

**Generate outputs:**
1. Reorganize results into by_topic/ JSONL files
2. Create summary statistics per area
3. Generate cross-topic overlap analysis
4. Create markdown reports for each of 20 areas

## Directory Structure

```
conferences/neurips2025/exhaustive_analysis/
├── README.md                          # Documentation
├── progress.json                      # Real-time progress tracking
├── config.json                        # Classification config
│
├── papers/
│   ├── by_batch/                      # Raw classification results
│   │   ├── batch_0000_0049.json
│   │   └── ...
│   ├── by_topic/                      # Organized by topic
│   │   ├── 01_tools_and_open_source.jsonl
│   │   └── ...
│   └── failed_papers.json
│
├── presentations/
│   ├── by_presentation/               # Individual results
│   ├── by_topic/                      # Organized by topic
│   └── failed_presentations.json
│
├── reports/
│   ├── 01_tools_and_open_source/
│   │   ├── summary.md                 # Executive summary
│   │   ├── top_papers.md              # High-confidence papers
│   │   └── statistics.json
│   └── ...
│
├── cross_cutting/
│   ├── multi_topic_papers.jsonl       # Papers in 2+ topics
│   └── topic_overlap_matrix.json      # Topic relationships
│
└── logs/
    ├── paper_processing.log
    └── presentation_processing.log
```

## Critical Implementation Details

### 1. Extraction Prompt Template (For Presentation Segments)

**Structure:**
- Description of all 20 research areas with key indicators
- Instructions: "Extract ALL content relevant to each area - don't just classify"
- For each relevant topic, provide:
  - Verbatim quotes from transcript (with exact timestamps)
  - Summary of key points
  - Slide references and what they show
  - Timestamp range for each extraction

**Prompt example:**
```
You are analyzing a segment from a NeurIPS 2025 presentation.

SEGMENT INFO:
Title: "Deep Learning for Code in the Agentic Era"
Timestamp: 00:15:00 - 00:20:00
Transcript: [full transcript text]
Slides visible: slide_042.json, slide_043.json
Slide content: [OCR text from slides]

TASK:
For each of the 20 research areas below, extract ANY relevant content from this segment.
For each extraction, provide:
1. Exact verbatim quote from transcript
2. Timestamp range (start-end) where this was discussed
3. Which slides support this (if any)
4. Brief summary of the key point

Output as JSON array of extractions.
```

**Output format:**
```json
{
  "extractions": [
    {
      "topic_id": "18",
      "topic_name": "Agentic AI and Autonomous Systems",
      "timestamp_start": "00:15:30",
      "timestamp_end": "00:17:45",
      "verbatim_quote": "We call this the demo soup problem...",
      "summary": "Introduces demo soup problem in agent evaluation",
      "slide_references": ["slide_042"],
      "relevance_score": 0.95
    },
    {
      "topic_id": "01",
      "topic_name": "Tools, Open Source, and Infrastructure",
      "timestamp_start": "00:18:12",
      "timestamp_end": "00:19:30",
      "verbatim_quote": "SWE-Bench Verified is now available...",
      "summary": "Announces SWE-Bench Verified benchmark",
      "slide_references": ["slide_043"],
      "relevance_score": 0.90
    }
  ]
}
```

### 2. Retry Logic (Reuse Pattern)

From `claude_deep_research.py:47-70`:
```python
def api_call_with_retry(func, max_retries=5, initial_delay=5.0):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
```

### 3. Progress Tracking

Real-time `progress.json`:
```json
{
  "papers": {
    "total": 9825,
    "processed": 2450,
    "failed": 12,
    "last_batch": 49,
    "extractions_count": 4523
  },
  "presentations": {
    "total": 131,
    "processed": 45,
    "total_segments": 6234,
    "segments_processed": 2156,
    "segments_failed": 23,
    "extractions_count": 8945
  },
  "cost_tracking": {
    "total_tokens_input": 5234567,
    "total_tokens_output": 1234567,
    "estimated_cost_usd": 67.43
  },
  "topic_distribution": {
    "01": 412,
    "02": 834,
    ...
  },
  "extraction_yield": {
    "avg_extractions_per_segment": 4.1,
    "segments_with_no_extractions": 234
  }
}
```

### 4. Resume Capability

**Implementation:**
- Load `progress.json` to find last processed presentation and segment
- For presentations: Resume from last successful segment (not whole presentation)
- Skip already-processed segments by checking extraction files
- Continue segment-by-segment processing
- Critical for long workshops that may take hours to process

## Critical Files

**To create:**
1. `scripts/exhaustive_classification/__init__.py`
2. `scripts/exhaustive_classification/prompts.py` - Prompt templates
3. `scripts/exhaustive_classification/models.py` - Model selection + API wrappers
4. `scripts/exhaustive_classification/schemas.py` - Pydantic models
5. `scripts/exhaustive_classification/storage.py` - Multi-area storage
6. `scripts/exhaustive_classification/processor.py` - Main processing logic
7. `scripts/run_exhaustive_classification.py` - Entry point

**To reference:**
1. `scripts/claude_deep_research.py` - Retry logic pattern (lines 47-70)
2. `conferences/neurips2025/data/papers.json` - Paper source data
3. `conferences/neurips2025/decks/*/transcripts/segments.json` - Presentation data
4. `conferences/neurips2025/analysis/NeurIPS_Analysis_Plan.md` - Topic definitions

**Output location:**
1. `conferences/neurips2025/exhaustive_analysis/` - All results
2. `docs/analysis-plans/exhaustive-analysis-plan.md` - Detailed documentation (copy of this plan)

## Success Metrics

**Minimum viable:**
- 95%+ presentation segments processed successfully
- 90%+ papers processed with relevant sentence extraction
- All 20 topics have extractions
- Timestamp accuracy: within 10 seconds of actual content
- Quotes are verbatim (spot-check 50 random quotes)
- Costs within ~$120

**Stretch goals:**
- 99%+ segments processed
- Timestamp accuracy: within 5 seconds
- 95%+ quote accuracy (verbatim)
- Avg 3+ extractions per relevant segment
- Complete in < 8 hours total processing time

## Next Steps

1. Create script structure with extraction-focused architecture
2. Build prompts that emphasize extraction over classification
3. Implement segment-by-segment processing for presentations
4. Run validation: Check timestamp accuracy and quote fidelity
5. Process 2-3 full presentations to test extraction yield
6. Review extraction quality before bulk processing
7. Process high-value content with careful monitoring
8. Run bulk processing with segment-level resume capability
9. Generate topic-organized extraction files (JSONL)

## Key Differences from Original Plan

**BEFORE (Classification):**
- "This presentation is about agentic AI" ✗
- Whole-presentation processing
- Simple topic labels

**AFTER (Extraction):**
- "At 00:15:30-00:17:45: [verbatim quote about demo soup problem]" ✓
- Segment-by-segment processing
- Factual statements with timestamps and slide references
- Can extract multiple topics from same segment
- Higher cost but much more valuable output
