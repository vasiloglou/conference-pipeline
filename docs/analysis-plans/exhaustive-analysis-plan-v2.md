# Exhaustive NeurIPS 2025 Analysis - Implementation Plan (v2)

## Overview

Build a **push-based extraction system** that scans all 9,825 papers + 131 presentations and extracts specific content relevant to each of the 20 research areas defined in NeurIPS_Analysis_Plan.md.

**Critical Requirement:**
This is **content extraction with timestamps**, not classification:
- For presentations: Extract specific segments (quotes/summaries) WITH precise timestamp ranges
- For papers: Extract relevant sentences from abstracts
- Store factual statements with full provenance (timestamps, slide references, hashes)

**User Requirements:**
- Budget: No strict limit - optimize for quality
- Multi-area: Content can be relevant to multiple topics - extract for each with deduplication
- Integration: Keep separate from existing analysis (`exhaustive_analysis/` directory)
- Workshops: Process in manageable chunks to extract relevant segments

**Estimated Cost:** ~$80-150 total (includes QA and retry budget)
**Estimated Time:** 6-8 hours processing time (includes validation loops)

---

## Core Strategy

### 1. Paper Processing: BATCH EXTRACTION

⚠️ **IMPORTANT LIMITATION: Abstract-Only Coverage**

Papers are processed using **abstracts only**, not full text. This has significant implications:

- **Expected recall:** ~60-70% of relevant papers (vs ~90%+ with full text)
- **Detail quality:** Abstract mentions may be too brief for quality extraction
- **Papers missed:** Content discussed in body but not abstract will be missed

**Mitigation:**
- All paper extractions marked with `"source_coverage": "abstract_only"`
- Reports include disclaimer about abstract-only limitation
- **Planned re-run:** When full PDFs become available, re-run extraction on full text

**Processing Details:**
- **Strategy:** Process 20-50 papers per API call
- **Task:** Extract relevant sentences from abstracts for each topic
- **Model:** Claude Sonnet 4.6 (with Haiku fallback if rate-limited)
- **Output:** List of (topic_id, extracted_sentences, confidence, source_coverage) per paper
- **Token Budget:** ~4.9M tokens total, $20-25
- **Time:** ~20-30 minutes (197 batches)

### 2. Presentation Processing: SEGMENT-BY-SEGMENT EXTRACTION
- **Strategy:** Process ~2-5 minute chunks from segments.json
- **Task:** For each segment, extract:
  - Verbatim quotes from transcript (anchored to segment start offset)
  - Summary of key points
  - Slide references with OCR content validation
  - Precise timestamp ranges (relative to segment start)
- **Model Fallback Chain:**
  1. Primary: Claude Sonnet 4.6
  2. Fallback: Gemini 2.5 Pro (if Sonnet rate-limited or very long segments)
  3. Last resort: Claude Haiku 4.5 (if both unavailable)
- **Guardrails:**
  - Skip segments with corrupted/missing transcripts
  - Handle OCR noise gracefully (mark as "OCR uncertain" if confidence low)
  - "No evidence → no extraction" (don't hallucinate content)
  - Validate slide references exist in OCR directory
- **Token Budget:** ~9.7M tokens total, $80-100
- **Time:** ~5-7 hours (6,000-7,000 segments)

### 3. Multi-Topic Extraction with Deduplication
- A segment can yield extractions for multiple topics
- **Deduplication Rules:**
  - Same verbatim_quote + same topic_id = duplicate (keep first, track source)
  - Same segment hash + same topic_id = already processed (skip)
  - Different topics, same quote = allowed (different semantic relevance)
- **Merge Strategy:**
  - Store each extraction in its topic file
  - Track cross-references in `multi_topic_map.json`
  - No merging of content - keep extractions atomic

### 4. Output Format (Per Extraction)

```json
{
  "extraction_id": "pres_dl-code_seg_042_topic_18_001",
  "content_hash": "sha256:7a8f3c2e1d9b4a6f5e3c8d7b2a1f9e4c3d8b7a6f5e4d3c2b1a9f8e7d6c5b4a3f2e1d0",
  "source_type": "presentation",
  "source_id": "dl-code-agentic-era",
  "source_title": "Deep Learning for Code in the Agentic Era",
  "segment_index": 42,
  "segment_start_sec": 900.0,
  "segment_end_sec": 1020.0,
  "topic_id": "18",
  "topic_name": "Agentic AI and Autonomous Systems",
  "timestamp_start_sec": 930.0,
  "timestamp_end_sec": 1015.0,
  "timestamp_start": "00:15:30",
  "timestamp_end": "00:16:55",
  "slide_references": [
    {
      "slide_id": "slide_042",
      "slide_path": "presentations/dl-code-agentic-era/ocr/slide_042.json",
      "slide_timestamp_sec": 905.0,
      "ocr_confidence": "high",
      "slide_content_summary": "Diagram showing agent evaluation framework",
      "timestamp_aligned": true
    }
  ],
  "extraction_type": "verbatim_quote",
  "verbatim_quote": "We call this the demo soup problem - where you have many demos but no systematic way to evaluate them.",
  "summary": "Speaker introduces the 'demo soup problem' in agent evaluation",
  "relevance_score": 0.95,
  "model_used": "claude-sonnet-4-6",
  "extracted_at": "2026-01-05T14:23:45Z",
  "qa_validated": false
}
```

**Note:** Timestamps are validated to be within segment bounds (900-1020s). Out-of-bounds timestamps hard-fail validation and are rejected (no auto-clipping).

---

## Implementation Phases

### Phase 1: Core Infrastructure + QA Framework (Day 1 Morning)

**Files to create:**

1. `scripts/exhaustive_extraction/__init__.py`

2. `scripts/exhaustive_extraction/prompts.py`
   - Extraction prompt template with all 20 area definitions
   - **Guardrails built into prompts:**
     - "If transcript contains '[inaudible]' or '[crosstalk]', skip that portion"
     - "Anchor all timestamps relative to segment start time: {segment_start_sec}"
     - "If no clear evidence for a topic, return empty array for that topic"
     - "Validate slide references: only cite slides that were actually shown during this segment"
   - Paper variant: "Extract sentences relevant to each topic"
   - Segment variant: "Extract quotes/summaries with timestamp anchoring"

3. `scripts/exhaustive_extraction/models.py`
   - Token counting utility
   - **Model selection with fallback chain:**
     ```python
     def select_model(segment):
         if segment.token_count < 50000 and sonnet_available():
             return "claude-sonnet-4-6"
         elif segment.token_count < 1000000 and gemini_available():
             return "gemini-2.5-pro"
         else:
             return "claude-haiku-4.5"
     ```
   - API wrappers with retry logic (from claude_deep_research.py:47-70)
   - **Rate limiting:** Max 50 req/min for Sonnet, 30 req/min for Gemini
   - **Concurrency:** 3 parallel workers max (to respect rate limits)

4. `scripts/exhaustive_extraction/schemas.py`
   - Pydantic models for `ContentExtraction`
   - **Timestamp validation (STRICT - no auto-clipping):**
     - Parse HH:MM:SS format
     - Convert to seconds and validate against segment bounds
     - **REJECT** (raise ValidationError) if timestamps out of bounds - no auto-clipping
     - Ensure timestamp_start <= timestamp_end
     - Out-of-bounds extractions logged to QA artifacts for debugging
   - **Slide reference validation:**
     - Check slide file exists: `presentations/{source_id}/ocr/{slide_id}.json`
     - Parse OCR confidence from slide file
     - Validate slide_timestamp_sec is within segment bounds (hard-fail if out of range)
     - **Validate timestamp alignment:** slide_timestamp_sec must be within segment bounds
     - Set `timestamp_aligned: false` if slide timestamp outside segment (QA tracks this)
   - **Content hash (FULL SHA256 - no truncation):**
     ```python
     def compute_hash(extraction):
         # Hash: source_id + segment_index + topic_id + verbatim_quote
         key = f"{extraction.source_id}_{extraction.segment_index}_{extraction.topic_id}_{extraction.verbatim_quote or extraction.summary}"
         # CRITICAL: Use FULL hash (64 chars), not truncated
         # Truncation to 16 chars = collision risk with 100K+ extractions
         full_hash = hashlib.sha256(key.encode()).hexdigest()
         return f"sha256:{full_hash}"  # 71 chars total
     ```

5. `scripts/exhaustive_extraction/storage.py`
   - **Queue-Based Storage (Recommended for 3 workers):**
     - Single writer thread per topic (no locking overhead)
     - In-memory hash index for O(1) dedup checks (not O(n) file scans)
     - Batch writes (100 extractions) with single fsync
     - Workers submit to queue, writer drains to disk
     ```python
     class QueuedTopicStorage:
         def __init__(self, output_dir: Path):
             self.output_dir = output_dir
             self.queues: Dict[str, queue.Queue] = {}
             self.hash_indices: Dict[str, Set[str]] = {}
             self.writers: Dict[str, threading.Thread] = {}

         def start_writer(self, topic_id: str):
             """Initialize queue and start background writer thread."""
             if topic_id in self.queues:
                 return  # Already started

             q = queue.Queue(maxsize=1000)
             self.queues[topic_id] = q
             # Note: hash_indices[topic_id] initialized by writer thread after loading from file

             writer = threading.Thread(
                 target=self._writer_loop,
                 args=(topic_id, q),
                 daemon=True
             )
             writer.start()
             self.writers[topic_id] = writer

         def append_extraction(self, extraction, topic_id):
             # Start writer on first use
             if topic_id not in self.queues:
                 self.start_writer(topic_id)

             # O(1) dedup check via in-memory hash set
             # Use .get() to handle race where writer hasn't loaded hashes yet
             if extraction.content_hash in self.hash_indices.get(topic_id, set()):
                 return  # Duplicate

             # Submit to queue for async batch write
             self.queues[topic_id].put(extraction)

         def _writer_loop(self, topic_id, queue):
             # Background thread: drain queue, batch write
             topic_file = self.output_dir / f"by_topic/{topic_id}.jsonl"
             topic_file.parent.mkdir(parents=True, exist_ok=True)

             # Load existing hashes from file
             self.hash_indices[topic_id] = self._load_hashes(topic_file)

             batch = []
             while True:
                 try:
                     extraction = queue.get(timeout=1.0)
                     if extraction is None:  # Shutdown signal
                         if batch:
                             self._flush_batch(topic_file, batch)
                         break

                     if extraction.content_hash not in self.hash_indices[topic_id]:
                         batch.append(extraction)
                         self.hash_indices[topic_id].add(extraction.content_hash)

                     # Flush every 100 items
                     if len(batch) >= 100:
                         self._flush_batch(topic_file, batch)
                         batch = []
                 except queue.Empty:
                     # Flush on timeout
                     if batch:
                         self._flush_batch(topic_file, batch)
                         batch = []

         def _load_hashes(self, topic_file: Path) -> Set[str]:
             """Load existing content hashes from JSONL file."""
             if not topic_file.exists():
                 return set()
             hashes = set()
             with open(topic_file) as f:
                 for line in f:
                     extraction = json.loads(line)
                     hashes.add(extraction['content_hash'])
             return hashes

         def _flush_batch(self, topic_file: Path, batch: List[Any]):
             """Write batch to disk with single fsync."""
             with open(topic_file, 'a') as f:
                 for extraction in batch:
                     f.write(json.dumps(extraction.dict()) + '\n')
                 f.flush()
                 os.fsync(f.fileno())
     ```
   - **Production Default: Use `QueuedTopicStorage` for 3-worker parallel processing (best performance)**
   - **Alternative: File Locking (for single-process or simpler deployments):**
     - Use `fcntl.flock()` for multi-process safety
     - Still use in-memory hash index to avoid O(n) scans
     - See addendum for full implementation
     - Simpler to debug but ~30% slower due to lock contention
   - **Crash-safe progress tracking with file locking:**
     ```python
     import fcntl

     def save_progress(progress):
         lock_file = Path("progress.json.lock")
         # Ensure lock file exists (touch)
         lock_file.touch(exist_ok=True)

         with open(lock_file, 'r+') as lock:
             fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
             try:
                 temp = Path("progress.json.tmp")
                 with open(temp, 'w') as f:
                     json.dump(progress, f, indent=2)
                     f.flush()
                     os.fsync(f.fileno())
                 temp.replace(Path("progress.json"))  # Atomic
             finally:
                 fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
     ```

6. `scripts/exhaustive_extraction/qa_validator.py`
   - **Automated QA checks:**
     - Timestamp consistency: Is timestamp within segment bounds?
     - Quote verbatim check: Does quote appear in transcript?
     - Slide reference validation: Does slide file exist and timestamp match?
     - Hallucination detection: Check for generic statements not in source
   - **Sampling protocol:**
     - Sample 50 random extractions every 500 segments
     - Run automated checks
     - Flag issues for manual review

7. `scripts/exhaustive_extraction/processor.py`
   - Main processing logic with segment-level resume
   - Per-segment checkpointing

---

### Phase 2: Validation Run with QA Loop (Day 1 Afternoon)

**Validate on small sample:**
- 10 oral papers (high-quality known content)
- 2 full presentations (~10-20 segments total)

**QA Loop Protocol:**

1. **Run extraction** on validation sample (10 papers + 2 presentations)

2. **Automated QA checks - PRESENTATIONS:**
   - Timestamp accuracy: Run `qa_validator.check_timestamps()`
   - Quote verbatim: Run `qa_validator.check_quotes_in_transcript()`
   - Slide validation: Run `qa_validator.check_slides_exist()` + timestamp alignment + OCR confidence

3. **Automated QA checks - PAPERS:**
   - Quote in abstract: Check extracted quotes actually appear in abstract
   - Hallucination detection: Check summary concepts present in abstract
   - Generic statement detection: Flag generic phrases like "this paper discusses"
   ```python
   # Sample 50 paper extractions
   paper_qa = PaperQAValidator().validate_batch(paper_extractions, sample_size=50)
   ```

4. **Calculate QA metrics:**
   ```python
   qa_metrics = {
       # Presentation metrics
       "pres_timestamp_accuracy": 0.94,  # % within 5 seconds
       "pres_quote_verbatim_rate": 0.89,  # % exact matches
       "pres_slide_accuracy": 0.97,  # % valid references with timestamp alignment
       "pres_hallucination_rate": 0.03,  # % generic/unsupported claims

       # Paper metrics (NEW)
       "paper_quote_in_abstract": 0.92,  # % quotes found in abstract
       "paper_hallucination_rate": 0.04,  # % content not in abstract
       "paper_generic_rate": 0.08,  # % generic statements
   }
   ```

5. **Acceptance Thresholds:**
   - **Presentations:**
     - Timestamp accuracy: >= 90% (within 10 sec) OR >= 80% (within 5 sec)
     - Quote verbatim: >= 85%
     - Slide accuracy: >= 95%
     - Hallucination rate: <= 5%
   - **Papers (NEW):**
     - Quote in abstract: >= 90%
     - Hallucination rate: <= 5%
     - Generic rate: <= 10%

6. **If ANY threshold fails:**
   - **Rollback:** Discard validation batch
   - **Revise prompts:** Add more explicit guardrails
   - **Retry:** Run validation again
   - **Document:** Log prompt version and results in `qa_artifacts/validation_runs/`

7. **If ALL thresholds pass:**
   - Proceed to Phase 3
   - Save successful prompt version as `prompts_v{N}_validated.py`

**QA Artifacts Generated:**
```
exhaustive_analysis/qa_artifacts/
├── validation_runs/
│   ├── run_001_failed.json       # Metrics, failed thresholds
│   ├── run_002_passed.json
│   └── prompt_versions/
│       ├── prompts_v1.py
│       └── prompts_v2.py
├── sampled_extractions/
│   ├── sample_001.json           # 50 random extractions
│   └── sample_001_qa_report.json # QA check results
└── bad_cases/
    ├── hallucinated_quotes.jsonl
    ├── wrong_timestamps.jsonl
    └── invalid_slides.jsonl
```

---

### Phase 3: High-Value Processing (Day 2)

**Process in order with continuous QA sampling:**
1. Oral papers (77) - 2 batches, ~5 min
2. Spotlight papers (688) - 14 batches, ~25 min
3. Short presentations (80) - ~500 segments, ~2 hours
4. Medium presentations (35) - ~1,000 segments, ~3 hours

**Continuous QA:**
- Every 500 segments: sample 50 extractions, run QA checks
- If drift detected (metrics drop >5%), pause and review
- Adjust prompts if needed, resume

**Checkpoints:**
- After each batch/presentation: save progress with fsync
- Track QA metrics over time in `qa_artifacts/metrics_timeline.jsonl`

---

### Phase 4: Bulk Processing (Day 3+)

**Process remaining content:**
1. Poster papers (4,525) - 91 batches, ~2 hours
2. Workshop papers (3,944) - 79 batches, ~1.5 hours
3. Long workshops (16) - ~5,000 segments, ~6-8 hours

**Concurrency:**
- 3 parallel workers (respect rate limits)
- Each worker processes different presentations
- Shared progress tracking with file locking

**Monitoring:**
- Real-time dashboard showing:
  - Segments/second processing rate
  - Current QA metrics (rolling avg last 500 segments)
  - Cost accumulation
  - Model distribution (% Sonnet vs Gemini vs Haiku)
  - Error rates by type

---

### Phase 5: Report Generation & Final QA (Day 3-4)

**Deduplication pass:**
- Scan all topic JSONL files
- Identify exact duplicate extractions (same content_hash)
- Keep first occurrence, log duplicates in `qa_artifacts/duplicates.jsonl`

**Generate outputs:**
1. Reorganize into by_topic/ JSONL files (with dedup)
2. Create summary statistics per topic
3. Generate cross-topic overlap matrix
4. Create markdown reports for each of 20 topics

**Final QA Report:**
- Total extractions: X
- Duplicates removed: Y
- QA metrics across all data:
  - Avg timestamp accuracy
  - Avg quote verbatim rate
  - Slide reference accuracy
  - Hallucination rate
- Topics with most extractions
- Presentations with highest extraction yield

---

## Directory Structure (Revised)

```
conferences/neurips2025/exhaustive_analysis/
├── README.md
├── progress.json                      # Crash-safe progress
├── config.json                        # Extraction config
│
├── papers/
│   ├── by_batch/                      # Raw extraction results
│   │   ├── batch_0000_0049.json
│   │   └── ...
│   ├── by_topic/                      # Topic-organized (deduplicated)
│   │   ├── 01_tools_and_open_source.jsonl
│   │   └── ...
│   └── failed_papers.json
│
├── presentations/
│   ├── by_segment/                    # Segment-level results
│   │   ├── {presentation_id}/
│   │   │   ├── segment_000.json
│   │   │   └── ...
│   ├── by_topic/                      # Topic-organized (deduplicated)
│   │   ├── 01_tools_and_open_source.jsonl
│   │   └── ...
│   └── failed_segments.json
│
├── qa_artifacts/                      # QA and audit data
│   ├── validation_runs/
│   │   ├── run_001_failed.json
│   │   └── prompt_versions/
│   ├── sampled_extractions/
│   │   ├── sample_001.json
│   │   └── sample_001_qa_report.json
│   ├── bad_cases/
│   │   ├── hallucinated_quotes.jsonl
│   │   ├── wrong_timestamps.jsonl
│   │   └── invalid_slides.jsonl
│   ├── metrics_timeline.jsonl         # QA metrics over time
│   └── duplicates.jsonl               # Removed duplicates log
│
├── reports/
│   ├── 01_tools_and_open_source/
│   │   ├── summary.md
│   │   ├── top_extractions.md
│   │   └── statistics.json
│   └── ...
│
├── cross_cutting/
│   ├── multi_topic_extractions.jsonl # Extractions in 2+ topics
│   ├── topic_overlap_matrix.json     # Topic co-occurrence
│   └── deduplication_report.json
│
└── logs/
    ├── paper_processing.log
    ├── presentation_processing.log
    └── qa_validation.log
```

---

## Token Budget & Rate Limits

### Papers (9,825 total)

| Batch Size | Batches | Avg Tokens/Batch | Total Tokens | Model | Cost |
|------------|---------|------------------|--------------|-------|------|
| 50 | 197 | 25,000 | 4.9M | Sonnet 4.6 | Estimate with `conference_pipeline.costs` |

**Rate limit:** 50 req/min (Sonnet) → ~4 min total (well within limits)

### Presentations (131 total, ~6,500 segments)

| Segment Type | Count | Avg Tokens/Seg | Total Tokens | Primary Model | Fallback | Cost |
|--------------|-------|----------------|--------------|---------------|----------|------|
| Short (<2K tok) | 4,000 | 1,500 | 6.0M | Sonnet 4.6 | Haiku | Estimate with registry |
| Medium (2-10K) | 2,000 | 5,000 | 10.0M | Sonnet 4.6 | Haiku | Estimate with registry |
| Long (>10K) | 500 | 20,000 | 10.0M | Gemini 2.5 Pro | Sonnet | Estimate with registry |
| **Total** | **6,500** | - | **26.0M** | - | - | Estimate with registry |

**Rate limits:**
- Sonnet: 50 req/min → ~130 min (2.2 hours) sequential
- With 3 parallel workers: ~45 min actual time
- Gemini: 30 req/min → ~17 min sequential for long segments

**Retry budget:** configured in `src/conference_pipeline/costs/task_specs.yaml`.

Estimate current costs with:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope exhaustive-extraction \
  --scenario base
```

---

## Prompt Guardrails (Enhanced)

### Presentation Segment Extraction Prompt

```
You are analyzing a segment from a NeurIPS 2025 presentation.

SEGMENT INFO:
Presentation: "{presentation_title}"
Segment index: {segment_index}
Segment timestamp: {segment_start_sec}s - {segment_end_sec}s ({segment_start_hms} - {segment_end_hms})
Duration: {segment_duration_sec}s

TRANSCRIPT:
{transcript_text}

SLIDES SHOWN DURING THIS SEGMENT:
{slides_list_with_ocr}

CRITICAL GUARDRAILS:
1. Timestamp anchoring: All timestamps MUST be relative to segment start ({segment_start_sec}s).
   - If you extract content at second 15 of the transcript, and segment starts at 900s, output timestamp should be 915s (900 + 15).
   - Extraction timestamps MUST be within segment bounds [{segment_start_sec}s, {segment_end_sec}s]

2. OCR handling with confidence assessment:
   - Assess OCR quality for each slide reference:
     * "missing": No slide_text field present
     * "low": Many "[OCR uncertain]" markers or very short text (<20 chars)
     * "medium": Some text, few uncertainties
     * "high": Clean text, no uncertainty markers
   - Do NOT invent slide content not present in OCR
   - If OCR confidence is "low" or "missing", include this in slide_reference metadata
   - Mark extraction with appropriate ocr_confidence level

3. Slide timestamp alignment (ENHANCED):
   - For EACH slide reference, you MUST provide:
     * slide_id: The slide identifier (e.g., "slide_042")
     * slide_timestamp_sec: When this slide was shown (must be within segment bounds)
     * ocr_confidence: Assessment of OCR quality (missing/low/medium/high)
     * timestamp_aligned: true if slide_timestamp_sec is within [{segment_start_sec}, {segment_end_sec}]
   - ONLY cite slides actually shown during this segment (from the list provided)
   - Verify slide timestamp falls within segment bounds - if not, mark timestamp_aligned: false

4. Transcript noise:
   - Skip portions marked as "[inaudible]", "[crosstalk]", or "[unclear]"
   - If transcript quality is poor, mark extraction with "transcript_quality": "low"

5. No evidence = no extraction:
   - If you cannot find clear evidence for a topic in THIS segment, return empty array for that topic.
   - Do NOT extract generic statements like "They discussed AI" - be specific.

6. Verbatim quotes:
   - Quotes must be EXACT text from transcript
   - Use [...] for omitted portions
   - If paraphrasing, use extraction_type: "summary", not "verbatim_quote"

TASK:
For each of the 20 research areas below, extract ANY relevant content from this segment.
For each extraction, provide:
1. Exact verbatim quote from transcript OR a summary (mark type)
2. Timestamp range IN SECONDS (relative to segment start: {segment_start_sec}s)
3. Which slides support this (IDs only, from list above)
4. Brief summary of the key point
5. Confidence/relevance score (0-1)

If a topic has NO clear evidence in this segment, return empty array for that topic.

[20 research area definitions follow...]

Output JSON:
{
  "extractions": [
    {
      "topic_id": "18",
      "timestamp_start_sec": 930.0,  // segment_start + offset
      "timestamp_end_sec": 1015.0,  // MUST be within segment bounds
      "verbatim_quote": "...",
      "extraction_type": "verbatim_quote",
      "summary": "...",
      "slide_references": [
        {
          "slide_id": "slide_042",
          "slide_timestamp_sec": 905.0,  // When this slide was shown
          "ocr_confidence": "high",  // missing/low/medium/high
          "timestamp_aligned": true  // Is slide_timestamp within segment bounds?
        }
      ],
      "relevance_score": 0.95,
      "transcript_quality": "high"
    }
  ]
}
```

---

## Success Metrics (Updated)

**Minimum viable (QA enforced):**
- 95%+ segments processed successfully
- 90%+ papers processed with extraction
- All 20 topics have extractions
- **QA thresholds (automated):**
  - Timestamp accuracy: >= 90% within 10 sec
  - Quote verbatim rate: >= 85%
  - Slide reference accuracy: >= 95%
  - Hallucination rate: <= 5%
- Costs within ~$150

**Stretch goals:**
- 99%+ segments processed
- Timestamp accuracy: >= 90% within 5 seconds
- Quote verbatim: >= 95%
- Avg 3+ extractions per relevant segment
- Complete in < 8 hours total

---

## Operational Details

### Model Fallback Chain

```python
FALLBACK_CHAIN = [
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 200_000,
        "rate_limit_rpm": 50,
        "pricing_source": "src/conference_pipeline/costs/pricing_catalog.json",
        "conditions": ["segment_tokens < 50000", "sonnet_available()"]
    },
    {
        "model": "gemini-2.5-pro",
        "max_tokens": 2_000_000,
        "rate_limit_rpm": 30,
        "pricing_source": "src/conference_pipeline/costs/pricing_catalog.json",
        "conditions": ["segment_tokens < 1000000", "gemini_available()"]
    },
    {
        "model": "claude-haiku-4.5",
        "max_tokens": 200_000,
        "rate_limit_rpm": 100,
        "pricing_source": "src/conference_pipeline/costs/pricing_catalog.json",
        "conditions": ["always"]  # Last resort
    }
]
```

### Concurrency & Rate Limiting

```python
# Global rate limiters
sonnet_limiter = RateLimiter(max_requests=50, window_seconds=60)
gemini_limiter = RateLimiter(max_requests=30, window_seconds=60)

# Worker pool
MAX_WORKERS = 3  # Conservative to respect rate limits

def process_presentations_parallel(presentations):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_presentation, pres): pres
            for pres in presentations
        }
        for future in as_completed(futures):
            # Handle results
            pass
```

### Degraded Mode & Failure Handling

**Note:** Cost estimates assume normal API quotas and network access. In constrained environments:

**Degraded Mode Strategy:**
```python
class DegradedModeConfig:
    # Rate limit detection
    RATE_LIMIT_BACKOFF_MULTIPLIER = 2.0
    MAX_BACKOFF_SECONDS = 300  # 5 min max

    # Degraded concurrency
    NORMAL_WORKERS = 3
    DEGRADED_WORKERS = 1  # Fall back to sequential

    # Offline queue
    ENABLE_OFFLINE_QUEUE = True
    OFFLINE_QUEUE_DIR = Path("exhaustive_analysis/offline_queue/")

class RateLimitHandler:
    def handle_rate_limit(self, error):
        self.consecutive_rate_limits += 1

        if self.consecutive_rate_limits >= 3:
            # Reduce workers: 3 → 1
            logger.warning("Entering degraded mode: reducing to 1 worker")
            self.workers = 1

        if self.consecutive_rate_limits >= 5:
            # Enable offline queue for severe rate limiting
            logger.warning("Severe rate limiting: switching to offline queue")
            return "offline_queue"

        # Exponential backoff
        self.current_backoff = min(
            self.current_backoff * 2.0,
            300  # Max 5 min
        )
        time.sleep(self.current_backoff)
        return "retry"
```

**Pre-run API Quota Check:**
```python
def check_api_quotas() -> Dict[str, Any]:
    """Check API quotas before bulk processing."""
    quotas = {}

    # Test Anthropic
    try:
        test_response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        quotas['anthropic'] = {'available': True}
    except RateLimitError:
        quotas['anthropic'] = {'available': False}
        logger.error("Anthropic API rate limited. Consider degraded mode.")

    return quotas

# Run before bulk processing:
quotas = check_api_quotas()
if not quotas['anthropic']['available']:
    # Options: fallback to Gemini, offline queue, or dry-run
    pass
```

**Dry-Run Mode** (for testing/quota conservation):
- Process only 100 segments
- Estimate full costs from sample
- Cost: ~$10 vs ~$150 for full run

---

## Critical Files

**To create:**
1. `scripts/exhaustive_extraction/__init__.py`
2. `scripts/exhaustive_extraction/prompts.py` - Enhanced with guardrails
3. `scripts/exhaustive_extraction/models.py` - Model selection + fallback + rate limiting
4. `scripts/exhaustive_extraction/schemas.py` - Pydantic models with hashing
5. `scripts/exhaustive_extraction/storage.py` - Idempotent writers with crash safety
6. `scripts/exhaustive_extraction/qa_validator.py` - Automated QA checks
7. `scripts/exhaustive_extraction/processor.py` - Main processing with concurrency
8. `scripts/run_exhaustive_extraction.py` - Entry point

**To reference:**
1. `scripts/claude_deep_research.py` - Retry logic (lines 47-70)
2. `conferences/neurips2025/data/papers.json` - Paper source
3. `conferences/neurips2025/decks/*/transcripts/segments.json` - Segments
4. `conferences/neurips2025/decks/*/ocr/slide_*.json` - OCR data
5. `conferences/neurips2025/analysis/NeurIPS_Analysis_Plan.md` - Topic definitions

---

## Next Steps

1. Create `exhaustive_extraction` module with QA framework
2. Implement prompts with all guardrails (OCR, timestamps, no-hallucination)
3. Build idempotent storage with crash safety
4. Run validation with QA loop until all thresholds pass
5. Process 2-3 full presentations, review QA metrics
6. If metrics acceptable, proceed to bulk processing
7. Continuous QA sampling every 500 segments
8. Final deduplication and report generation

---

## Key Improvements from V1

1. **Module renamed:** `exhaustive_classification` → `exhaustive_extraction`
2. **QA loop added:** Automated validation with acceptance thresholds and rollback
3. **Prompt guardrails:** OCR noise, timestamp anchoring, no-hallucination rules
4. **Idempotent writes:** Per-segment hashes, atomic writes, crash-safe progress
5. **QA artifacts:** Dedicated folder for validation runs, bad cases, metrics timeline
6. **Dedup rules:** Explicit content-hash-based deduplication with merge strategy
7. **Token budgets:** Detailed table with rate limits and concurrency plan
8. **Model fallbacks:** Explicit fallback chain with availability checks
9. **Operational details:** Rate limiters, concurrency, cost breakdown
