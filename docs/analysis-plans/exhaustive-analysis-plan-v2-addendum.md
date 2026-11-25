# Exhaustive Analysis Plan v2.1 - Technical Corrections & Addendum

**Date:** 2026-01-05
**Status:** Critical fixes to v2 before implementation

---

## Critical Issues Identified & Fixes

### Issue 1: Timestamp Validation Beyond Segment Bounds

**Problem:** Example shows `timestamp_end_sec: 1065` but segment ends at `1020`. Schema doesn't reject this.

**Fix:**
```python
class ContentExtraction(BaseModel):
    timestamp_start_sec: float
    timestamp_end_sec: float
    segment_start_sec: float
    segment_end_sec: float

    @validator('timestamp_end_sec')
    def validate_timestamp_bounds(cls, v, values):
        segment_end = values.get('segment_end_sec')
        timestamp_start = values.get('timestamp_start_sec')
        segment_start = values.get('segment_start_sec')

        # Ensure timestamp is within segment
        if v > segment_end:
            # CLIP to segment end, log warning
            logger.warning(f"Timestamp {v} exceeds segment end {segment_end}, clipping")
            v = segment_end

        # Ensure start <= end
        if timestamp_start and v < timestamp_start:
            raise ValueError(f"timestamp_end ({v}) < timestamp_start ({timestamp_start})")

        # Ensure timestamps are within segment
        if timestamp_start and timestamp_start < segment_start:
            raise ValueError(f"timestamp_start ({timestamp_start}) < segment_start ({segment_start})")

        return v
```

**QA Check:**
```python
def qa_check_timestamp_bounds(extraction):
    """Reject extractions with timestamps outside segment bounds."""
    if extraction.timestamp_start_sec < extraction.segment_start_sec:
        return QAResult(valid=False, reason="Timestamp before segment start")
    if extraction.timestamp_end_sec > extraction.segment_end_sec:
        return QAResult(valid=False, reason="Timestamp after segment end")
    return QAResult(valid=True)
```

---

### Issue 2: Hash Collision Risk & Undefined Variable

**Problem:**
- `segment_idx` used but field is named `segment_index`
- SHA256 truncated to only 16 chars (high collision risk with 10K+ extractions)

**Fix:**
```python
def compute_extraction_hash(extraction):
    """
    Compute deterministic hash for deduplication.
    Use full SHA256 (64 chars) to avoid collisions.
    """
    # Use consistent field names
    key_components = [
        extraction.source_id,
        str(extraction.segment_index),  # FIX: use segment_index, not segment_idx
        extraction.topic_id,
        extraction.verbatim_quote or extraction.summary  # Handle both types
    ]
    key = "_".join(key_components)

    # FIX: Use full hash (64 chars), not truncated
    full_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return f"sha256:{full_hash}"  # Full 64-character hash
```

**Collision Analysis:**
- 16 chars hex = 2^64 possible values
- With 100K extractions, birthday paradox gives ~0.0027% collision chance
- 64 chars hex = 2^256 possible values = virtually zero collision risk

---

### Issue 3: Dedup Performance & Race Conditions

**Problem:**
- Scanning entire JSONL file on every append is O(n) per write
- No file locking with 3 workers → race conditions and lost writes
- Slow for large files (10K+ lines)

**Fix: Per-Topic Hash Index + File Locking**

```python
import fcntl
from pathlib import Path
from typing import Set

class TopicStorage:
    """Thread-safe storage with hash-based deduplication."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        # In-memory hash index per topic (loaded at startup)
        self.hash_indices: Dict[str, Set[str]] = {}
        self.locks: Dict[str, threading.Lock] = {}

    def load_hash_index(self, topic_id: str) -> Set[str]:
        """Load all existing hashes from topic file."""
        topic_file = self.output_dir / f"by_topic/{topic_id}.jsonl"
        if not topic_file.exists():
            return set()

        hashes = set()
        with open(topic_file, 'r') as f:
            for line in f:
                extraction = json.loads(line)
                hashes.add(extraction['content_hash'])
        return hashes

    def append_extraction(self, extraction: ContentExtraction, topic_id: str):
        """
        Append extraction with deduplication and file locking.
        O(1) dedup check using in-memory hash index.
        """
        # Ensure hash index loaded
        if topic_id not in self.hash_indices:
            self.hash_indices[topic_id] = self.load_hash_index(topic_id)
            self.locks[topic_id] = threading.Lock()

        # Fast O(1) dedup check
        if extraction.content_hash in self.hash_indices[topic_id]:
            logger.debug(f"Skipping duplicate: {extraction.content_hash}")
            return False  # Already exists

        topic_file = self.output_dir / f"by_topic/{topic_id}.jsonl"
        topic_file.parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe write with file locking
        with self.locks[topic_id]:  # Python threading lock
            with open(topic_file, 'a') as f:
                # OS-level file lock (for multi-process safety)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Write and flush
                    f.write(json.dumps(extraction.dict()) + '\n')
                    f.flush()
                    os.fsync(f.fileno())

                    # Update in-memory index
                    self.hash_indices[topic_id].add(extraction.content_hash)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return True  # Successfully added

# Alternative: Single-Writer Queue Per Topic (Better for High Concurrency)
class QueuedTopicStorage:
    """
    Single writer thread per topic to avoid locking overhead.
    Workers submit to queue, writer drains to disk.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.queues: Dict[str, queue.Queue] = {}
        self.writers: Dict[str, threading.Thread] = {}
        self.hash_indices: Dict[str, Set[str]] = {}

    def start_writer(self, topic_id: str):
        """Start background writer thread for topic."""
        q = queue.Queue(maxsize=1000)
        self.queues[topic_id] = q

        writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(topic_id, q),
            daemon=True
        )
        writer_thread.start()
        self.writers[topic_id] = writer_thread

    def _writer_loop(self, topic_id: str, q: queue.Queue):
        """Background writer: drain queue and batch writes."""
        topic_file = self.output_dir / f"by_topic/{topic_id}.jsonl"
        topic_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing hashes
        self.hash_indices[topic_id] = self._load_hashes(topic_file)

        batch = []
        while True:
            try:
                extraction = q.get(timeout=1.0)
                if extraction is None:  # Shutdown signal
                    break

                # Dedup check
                if extraction.content_hash not in self.hash_indices[topic_id]:
                    batch.append(extraction)
                    self.hash_indices[topic_id].add(extraction.content_hash)

                # Batch write every 100 items
                if len(batch) >= 100:
                    self._flush_batch(topic_file, batch)
                    batch = []
            except queue.Empty:
                # Flush on timeout
                if batch:
                    self._flush_batch(topic_file, batch)
                    batch = []

    def _flush_batch(self, topic_file: Path, batch: List[ContentExtraction]):
        """Write batch to disk with single fsync."""
        with open(topic_file, 'a') as f:
            for extraction in batch:
                f.write(json.dumps(extraction.dict()) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def append_extraction(self, extraction: ContentExtraction, topic_id: str):
        """Submit to queue for async writing."""
        if topic_id not in self.queues:
            self.start_writer(topic_id)

        self.queues[topic_id].put(extraction)
```

**Recommendation:** Use `QueuedTopicStorage` for production (better performance with 3 workers).

---

### Issue 4: Model Naming Inconsistency

**Problem:** Stale previous-generation Sonnet identifiers drifted from the canonical
`claude-sonnet-5` provider contract.

**Fix: Canonical Model IDs**

```python
# models.py - Define canonical IDs
CANONICAL_MODEL_IDS = {
    "sonnet": "claude-sonnet-5",  # Consistent: dashes everywhere
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
    "gemini_pro": "gemini-2.5-pro",
    "gemini_flash": "gemini-2.0-flash"
}

def get_model_id(model_alias: str) -> str:
    """Get canonical model ID from alias."""
    return CANONICAL_MODEL_IDS.get(model_alias, model_alias)

# Usage:
model_id = get_model_id("sonnet")  # Always returns "claude-sonnet-5"
```

**Update all references in plan and code to use canonical IDs.**

---

### Issue 5: Papers Coverage Limitation

**Problem:** Abstract-only extraction has lower recall than full-text. Users may over-trust coverage.

**Fix: Explicit Limitation Notice**

```markdown
## Papers Processing - Coverage Limitation

**IMPORTANT:** Paper extractions are based on **abstracts only**, not full text.

**Expected Limitations:**
- **Lower recall:** Papers discussing topics in body but not abstract will be missed
- **Lower precision on details:** Abstract mentions may be too brief for quality extraction
- **Estimated coverage:** ~60-70% of relevant papers (vs ~90%+ with full text)

**Mitigation:**
1. Mark all paper extractions with `source_coverage: "abstract_only"`
2. Add disclaimer to all paper-based reports
3. **Planned re-run:** When full PDFs become available (post-conference), re-run extraction on full text

**User Notice in Reports:**
```
⚠️  Paper-based findings are extracted from abstracts only.
Full-text analysis planned when PDFs become available.
Treat paper counts and coverage as lower bounds.
```

**Tracking:**
```json
{
  "extraction_metadata": {
    "source_coverage": "abstract_only",
    "full_text_available": false,
    "rerun_planned": true,
    "estimated_recall": 0.65
  }
}
```
```

---

### Issue 6: Cross-Worker Locking for Progress & JSONL

**Problem:** No explicit locking mechanism for `progress.json` or topic JSONL files with 3 parallel workers.

**Fix: Explicit Locking Strategy**

```python
import fcntl
import json
from pathlib import Path
from typing import Dict, Any

class ProgressTracker:
    """Thread-safe and process-safe progress tracking."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.lock_file = Path(str(progress_file) + ".lock")

    def update_progress(self, updates: Dict[str, Any]):
        """
        Update progress with file locking.
        Safe for multi-process workers.
        """
        # Create lock file if doesn't exist
        self.lock_file.touch(exist_ok=True)

        with open(self.lock_file, 'r+') as lock:
            # Acquire exclusive lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            try:
                # Load current progress
                if self.progress_file.exists():
                    with open(self.progress_file, 'r') as f:
                        progress = json.load(f)
                else:
                    progress = {}

                # Merge updates (deep merge for nested dicts)
                progress = self._deep_merge(progress, updates)

                # Atomic write
                temp_file = Path(str(self.progress_file) + ".tmp")
                with open(temp_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (replaces old file)
                temp_file.replace(self.progress_file)

            finally:
                # Release lock
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _deep_merge(self, base: dict, updates: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

# Usage in workers:
progress_tracker = ProgressTracker(Path("exhaustive_analysis/progress.json"))

# Worker 1:
progress_tracker.update_progress({
    "presentations": {
        "segments_processed": 100
    }
})

# Worker 2 (concurrent):
progress_tracker.update_progress({
    "presentations": {
        "segments_processed": 150
    }
})
# Result: segments_processed = 150 (last write wins, but atomic)

# Better: Increment counter
def increment_counter(field_path: List[str], amount: int = 1):
    """Thread-safe counter increment."""
    with lock:
        current = get_nested(progress, field_path)
        set_nested(progress, field_path, current + amount)
```

**For JSONL files:** Use `QueuedTopicStorage` from Issue 3 (single writer per topic = no locking needed).

---

### Issue 7: QA for Papers

**Problem:** QA loop only validates presentations. Need paper QA too.

**Fix: Paper-Specific QA Checks**

```python
class PaperQAValidator:
    """QA checks specific to paper extractions."""

    def validate_paper_extraction(self, extraction: ContentExtraction, paper: dict) -> QAResult:
        """
        Validate paper extraction quality.

        Checks:
        1. Extracted sentences actually appear in abstract
        2. Not hallucinating content not in abstract
        3. Relevance to topic is justified
        """
        abstract = paper['abstract']

        # Check 1: Verbatim quote in abstract?
        if extraction.verbatim_quote:
            if extraction.verbatim_quote not in abstract:
                # Try fuzzy match (allow minor OCR differences)
                similarity = self._fuzzy_match(extraction.verbatim_quote, abstract)
                if similarity < 0.9:
                    return QAResult(
                        valid=False,
                        reason=f"Quote not found in abstract (similarity: {similarity:.2f})",
                        severity="high"
                    )

        # Check 2: Hallucination detection
        if extraction.summary:
            # Check if summary concepts appear in abstract
            concepts = self._extract_concepts(extraction.summary)
            abstract_concepts = self._extract_concepts(abstract)

            overlap = len(concepts & abstract_concepts) / len(concepts) if concepts else 0
            if overlap < 0.5:
                return QAResult(
                    valid=False,
                    reason=f"Summary concepts not in abstract (overlap: {overlap:.2f})",
                    severity="medium"
                )

        # Check 3: Generic statements detection
        generic_phrases = [
            "this paper discusses",
            "the authors present",
            "this work introduces",
            "we propose"
        ]
        if any(phrase in extraction.summary.lower() for phrase in generic_phrases):
            return QAResult(
                valid=True,
                reason="Generic phrasing detected - review recommended",
                severity="low",
                needs_review=True
            )

        return QAResult(valid=True)

# QA Sampling for Papers
def qa_sample_papers(batch_results: List[dict], sample_size: int = 50):
    """Sample and validate paper extractions."""
    import random

    samples = random.sample(batch_results, min(sample_size, len(batch_results)))

    qa_results = []
    for sample in samples:
        paper = load_paper(sample['paper_id'])
        for extraction in sample['extractions']:
            result = PaperQAValidator().validate_paper_extraction(extraction, paper)
            qa_results.append(result)

    # Calculate metrics
    metrics = {
        "quote_accuracy": sum(1 for r in qa_results if r.valid and "quote" in r.reason) / len(qa_results),
        "hallucination_rate": sum(1 for r in qa_results if not r.valid and "not in abstract" in r.reason) / len(qa_results),
        "generic_rate": sum(1 for r in qa_results if r.needs_review) / len(qa_results)
    }

    return metrics

# Add to Phase 2 validation:
# After processing 10 oral papers:
paper_qa_metrics = qa_sample_papers(paper_batch_results, sample_size=50)

# Acceptance thresholds:
assert paper_qa_metrics['quote_accuracy'] >= 0.90, "Paper quote accuracy too low"
assert paper_qa_metrics['hallucination_rate'] <= 0.05, "Paper hallucination rate too high"
```

---

### Issue 8: Degraded Mode for Rate Limits / Network Issues

**Problem:** Plan assumes generous rate limits and network access. Need fallback for restrictions.

**Fix: Degraded Mode Strategy**

```python
class DegradedModeConfig:
    """Configuration for degraded operation under constraints."""

    # Rate limit detection
    RATE_LIMIT_BACKOFF_MULTIPLIER = 2.0
    MAX_BACKOFF_SECONDS = 300  # 5 min max

    # Degraded concurrency
    NORMAL_WORKERS = 3
    DEGRADED_WORKERS = 1  # Fall back to sequential

    # Offline queue
    ENABLE_OFFLINE_QUEUE = True
    OFFLINE_QUEUE_DIR = Path("exhaustive_analysis/offline_queue/")

    # Dry-run mode
    DRY_RUN_SAMPLE_SIZE = 100  # Process only 100 segments in dry-run

class RateLimitHandler:
    """Handle rate limiting gracefully."""

    def __init__(self):
        self.consecutive_rate_limits = 0
        self.current_backoff = 5.0
        self.workers = DegradedModeConfig.NORMAL_WORKERS

    def handle_rate_limit(self, error):
        """Adjust behavior on rate limit."""
        self.consecutive_rate_limits += 1

        if self.consecutive_rate_limits >= 3:
            logger.warning("Entering degraded mode: reducing workers")
            self.workers = DegradedModeConfig.DEGRADED_WORKERS

        if self.consecutive_rate_limits >= 5:
            logger.warning("Severe rate limiting: switching to offline queue")
            return "offline_queue"

        # Exponential backoff
        self.current_backoff = min(
            self.current_backoff * DegradedModeConfig.RATE_LIMIT_BACKOFF_MULTIPLIER,
            DegradedModeConfig.MAX_BACKOFF_SECONDS
        )

        logger.info(f"Rate limited. Backing off for {self.current_backoff}s")
        time.sleep(self.current_backoff)

        return "retry"

    def handle_success(self):
        """Reset on successful request."""
        self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - 1)
        self.current_backoff = max(5.0, self.current_backoff / 2)

class OfflineQueueProcessor:
    """Queue requests for offline processing."""

    def enqueue_request(self, segment_id: str, segment_data: dict):
        """Save request for later processing."""
        queue_file = DegradedModeConfig.OFFLINE_QUEUE_DIR / f"{segment_id}.json"
        queue_file.parent.mkdir(parents=True, exist_ok=True)

        with open(queue_file, 'w') as f:
            json.dump(segment_data, f)

        logger.info(f"Queued {segment_id} for offline processing")

    def process_queue(self):
        """Process offline queue when rate limits clear."""
        queue_files = list(DegradedModeConfig.OFFLINE_QUEUE_DIR.glob("*.json"))

        logger.info(f"Processing {len(queue_files)} queued requests")

        for queue_file in queue_files:
            with open(queue_file, 'r') as f:
                segment_data = json.load(f)

            # Process with rate limit handling
            try:
                result = process_segment(segment_data)
                queue_file.unlink()  # Remove from queue on success
            except RateLimitError:
                logger.warning("Still rate limited, pausing queue processing")
                break

# Pre-run quota check:
def check_api_quotas() -> Dict[str, Any]:
    """Check API quotas before bulk processing."""
    quotas = {}

    # Test Anthropic quota
    try:
        test_response = anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        quotas['anthropic'] = {'available': True, 'tested_at': datetime.now()}
    except RateLimitError as e:
        quotas['anthropic'] = {'available': False, 'error': str(e)}

    # Test Gemini quota
    try:
        # Similar test for Gemini
        quotas['gemini'] = {'available': True, 'tested_at': datetime.now()}
    except Exception as e:
        quotas['gemini'] = {'available': False, 'error': str(e)}

    return quotas

# Usage:
quotas = check_api_quotas()
if not quotas['anthropic']['available']:
    logger.error("Anthropic API not available. Switch to degraded mode or abort.")
    # Options:
    # 1. Fall back to Gemini
    # 2. Enable offline queue
    # 3. Dry-run mode
```

---

### Issue 9: Slide Validation - Timestamp Alignment & OCR Confidence

**Problem:** Slide validation only checks file existence. Should also validate:
- Slide timestamp within segment bounds
- OCR confidence level

**Fix: Enhanced Slide Validation**

```python
class SlideValidator:
    """Validate slide references comprehensively."""

    def validate_slide_reference(
        self,
        slide_id: str,
        slide_path: Path,
        extraction: ContentExtraction
    ) -> QAResult:
        """
        Validate slide reference is legitimate.

        Checks:
        1. Slide file exists
        2. Slide timestamp within segment bounds
        3. OCR confidence is acceptable
        """
        # Check 1: File existence
        if not slide_path.exists():
            return QAResult(
                valid=False,
                reason=f"Slide file not found: {slide_path}",
                severity="high"
            )

        # Load slide data
        with open(slide_path, 'r') as f:
            slide_data = json.load(f)

        # Check 2: Timestamp alignment
        slide_timestamp = slide_data.get('timestamp_sec', 0)

        if not (extraction.segment_start_sec <= slide_timestamp <= extraction.segment_end_sec):
            return QAResult(
                valid=False,
                reason=f"Slide timestamp {slide_timestamp} outside segment bounds "
                       f"[{extraction.segment_start_sec}, {extraction.segment_end_sec}]",
                severity="medium"
            )

        # Check 3: OCR confidence
        ocr_confidence = self._assess_ocr_confidence(slide_data)

        if ocr_confidence == "missing":
            return QAResult(
                valid=True,
                reason="OCR data missing for slide - visual only",
                severity="low",
                needs_review=True,
                metadata={"ocr_confidence": "missing"}
            )

        if ocr_confidence == "low":
            return QAResult(
                valid=True,
                reason="Low OCR confidence - may have inaccuracies",
                severity="low",
                needs_review=True,
                metadata={"ocr_confidence": "low"}
            )

        return QAResult(
            valid=True,
            metadata={"ocr_confidence": ocr_confidence}
        )

    def _assess_ocr_confidence(self, slide_data: dict) -> str:
        """
        Assess OCR confidence level.

        Heuristics:
        - "missing": No slide_text field
        - "low": Many [OCR uncertain] markers or very short text
        - "medium": Some text, few uncertainties
        - "high": Rich text, no uncertainties
        """
        slide_text = slide_data.get('slide_text', {})

        # No OCR data
        if not slide_text or all(not v for v in slide_text.values()):
            return "missing"

        # Extract all text
        all_text = " ".join([
            str(slide_text.get('title', '')),
            " ".join(slide_text.get('bullet_points', [])),
            " ".join(slide_text.get('equations', [])),
            str(slide_text.get('other_text', ''))
        ])

        # Count uncertainty markers
        uncertainty_markers = all_text.count('[OCR uncertain]')
        uncertainty_ratio = uncertainty_markers / max(1, len(all_text.split()))

        # Classify confidence
        if uncertainty_ratio > 0.1 or len(all_text) < 20:
            return "low"
        elif uncertainty_ratio > 0.05:
            return "medium"
        else:
            return "high"

# Integration in extraction schema:
class ContentExtraction(BaseModel):
    # ... existing fields ...

    slide_references: List[SlideReference] = []

    @validator('slide_references')
    def validate_slides(cls, v, values):
        """Validate all slide references."""
        validator = SlideValidator()

        for slide_ref in v:
            slide_path = Path(slide_ref.slide_path)

            # Create pseudo-extraction for validation
            extraction_stub = type('obj', (object,), {
                'segment_start_sec': values.get('segment_start_sec'),
                'segment_end_sec': values.get('segment_end_sec')
            })

            result = validator.validate_slide_reference(
                slide_ref.slide_id,
                slide_path,
                extraction_stub
            )

            if not result.valid:
                logger.warning(f"Invalid slide reference: {result.reason}")
                # Remove invalid slide
                v.remove(slide_ref)
            else:
                # Annotate with OCR confidence
                slide_ref.ocr_confidence = result.metadata.get('ocr_confidence', 'unknown')

        return v

class SlideReference(BaseModel):
    slide_id: str
    slide_path: str
    slide_content_summary: str
    ocr_confidence: str = "unknown"  # NEW: Track OCR quality
```

---

## Summary of Changes

| Issue | Impact | Fix Complexity | Priority |
|-------|--------|----------------|----------|
| 1. Timestamp bounds | High (invalid data) | Low (validator) | Critical |
| 2. Hash collision | Medium (dedup fails) | Low (use full hash) | High |
| 3. Dedup performance | High (slow + races) | Medium (queue system) | Critical |
| 4. Model naming | Medium (API errors) | Low (constants) | High |
| 5. Papers coverage | Low (user expectation) | Low (documentation) | Medium |
| 6. Cross-worker locking | High (data corruption) | Medium (file locks) | Critical |
| 7. QA for papers | Medium (quality drift) | Medium (new validator) | High |
| 8. Degraded mode | Medium (fails on limits) | High (fallback logic) | High |
| 9. Slide validation | Medium (false positives) | Medium (enhanced check) | Medium |

---

## Implementation Priority

**Before any coding:**
1. Fix Issue 3 (dedup + locking) - prevents data corruption
2. Fix Issue 6 (progress locking) - prevents lost progress
3. Fix Issue 1 (timestamp validation) - prevents invalid data

**During validation phase:**
4. Fix Issue 7 (paper QA) - catch quality issues early
5. Fix Issue 9 (slide validation) - improve reference quality

**Before bulk processing:**
6. Fix Issue 8 (degraded mode) - handle rate limits gracefully
7. Fix Issue 2 (hash collision) - ensure dedup works at scale
8. Fix Issue 4 (model naming) - prevent API routing errors

**Documentation:**
9. Fix Issue 5 (papers coverage note) - set user expectations

---

## Updated Cost Estimate

With degraded mode fallbacks and retry budgets:
- **Optimistic (no rate limits):** $141
- **Realistic (some retries):** $160
- **Pessimistic (frequent rate limits, offline queue):** $200
- **Dry-run mode:** $10 (100 segments only)

**Recommendation:** Budget $200, expect ~$150-170 actual spend.
