"""
Queue-based storage for multi-topic extractions with deduplication.

Implements:
- Queue-based writer (one thread per topic)
- In-memory hash index for O(1) dedup
- Batch writes with fsync for durability
- Crash-safe progress tracking with file locking
"""

import fcntl
import json
import os
import queue
import threading
from pathlib import Path
from typing import Dict, Set, List, Any, Optional

from .schemas import ContentExtraction


class QueuedTopicStorage:
    """
    Queue-based storage with single writer thread per topic.

    Production default for 3-worker parallel processing.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize storage manager.

        Args:
            output_dir: Base directory for outputs
        """
        self.output_dir = Path(output_dir)
        self.queues: Dict[str, queue.Queue] = {}
        self.hash_indices: Dict[str, Set[str]] = {}
        self.writers: Dict[str, threading.Thread] = {}

    def _queue_key(self, source_type: str, topic_id: str) -> str:
        """Build a unique queue key per source type and topic."""
        return f"{source_type}:{topic_id}"

    def start_writer(self, source_type: str, topic_id: str):
        """
        Initialize queue and start background writer thread.

        Args:
            source_type: "presentation" or "paper"
            topic_id: Topic ID (e.g., "01")
        """
        queue_key = self._queue_key(source_type, topic_id)
        if queue_key in self.queues:
            return  # Already started

        q = queue.Queue(maxsize=1000)
        self.queues[queue_key] = q
        # Note: hash_indices[queue_key] initialized by writer thread after loading from file

        writer = threading.Thread(
            target=self._writer_loop,
            args=(source_type, topic_id, q),
            daemon=True
        )
        writer.start()
        self.writers[queue_key] = writer

    def append_extraction(self, extraction: ContentExtraction, topic_id: str):
        """
        Submit extraction to queue for async writing.

        Args:
            extraction: ContentExtraction object
            topic_id: Topic ID (e.g., "01")
        """
        source_type = extraction.source_type
        queue_key = self._queue_key(source_type, topic_id)

        # Start writer on first use
        if queue_key not in self.queues:
            self.start_writer(source_type, topic_id)

        # O(1) dedup check via in-memory hash set
        # Use .get() to handle race where writer hasn't loaded hashes yet
        if extraction.content_hash in self.hash_indices.get(queue_key, set()):
            return  # Duplicate

        # Submit to queue for async batch write
        self.queues[queue_key].put(extraction)

    def _writer_loop(self, source_type: str, topic_id: str, q: queue.Queue):
        """
        Background writer: drain queue and batch write.

        Args:
            source_type: "presentation" or "paper"
            topic_id: Topic ID
            q: Queue for this topic
        """
        # Background thread: drain queue, batch write
        if source_type == "presentation":
            base_dir = self.output_dir / "presentations"
        else:
            base_dir = self.output_dir / "papers"
        topic_file = base_dir / "by_topic" / f"{topic_id}.jsonl"
        topic_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing hashes from file
        queue_key = self._queue_key(source_type, topic_id)
        self.hash_indices[queue_key] = self._load_hashes(topic_file)

        batch = []
        while True:
            try:
                extraction = q.get(timeout=1.0)
                if extraction is None:  # Shutdown signal
                    if batch:
                        self._flush_batch(topic_file, batch)
                    break

                if extraction.content_hash not in self.hash_indices[queue_key]:
                    batch.append(extraction)
                    self.hash_indices[queue_key].add(extraction.content_hash)

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
        """
        Load existing content hashes from JSONL file.

        Args:
            topic_file: Path to topic JSONL file

        Returns:
            Set of content hashes
        """
        if not topic_file.exists():
            return set()
        hashes = set()
        with open(topic_file) as f:
            for line in f:
                try:
                    extraction = json.loads(line)
                    hashes.add(extraction['content_hash'])
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed lines
                    pass
        return hashes

    def _flush_batch(self, topic_file: Path, batch: List[ContentExtraction]):
        """
        Write batch to disk with single fsync.

        Args:
            topic_file: Path to topic file
            batch: List of extractions to write
        """
        with open(topic_file, 'a') as f:
            for extraction in batch:
                f.write(json.dumps(extraction.model_dump()) + '\n')
                f.flush()
                os.fsync(f.fileno())

    def shutdown(self):
        """Gracefully shutdown all writer threads."""
        # Send shutdown signal to all queues
        for q in self.queues.values():
            q.put(None)

        # Wait for all writers to finish
        for writer in self.writers.values():
            writer.join(timeout=5.0)


class ProgressTracker:
    """
    Crash-safe progress tracking with file locking.

    Ensures atomic updates to progress.json even with multiple workers.
    """

    def __init__(self, progress_file: Path):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to progress.json
        """
        self.progress_file = Path(progress_file)
        self.lock_file = self.progress_file.parent / f"{self.progress_file.name}.lock"
        self.lock_file.touch(exist_ok=True)

        # Initialize progress file if doesn't exist
        if not self.progress_file.exists():
            self.save_progress({
                "papers": {
                    "total": 0,
                    "processed": 0,
                    "failed": 0,
                    "last_batch": -1,
                    "extractions_count": 0
                },
                "presentations": {
                    "total": 0,
                    "processed": 0,
                    "total_segments": 0,
                    "segments_processed": 0,
                    "segments_failed": 0,
                    "extractions_count": 0
                },
                "cost_tracking": {
                    "total_tokens_input": 0,
                    "total_tokens_output": 0,
                    "estimated_cost_usd": 0.0
                },
                "topic_distribution": {},
                "extraction_yield": {
                    "avg_extractions_per_segment": 0.0,
                    "segments_with_no_extractions": 0
                },
                "pre_filter_stats": {
                    "invalid_extraction_type": 0,
                    "invalid_timestamps": 0
                },
                "failed_segments": [],  # List of {"pres_id": str, "segment_index": int, "error": str, "timestamp": str}
                "qa_failures": []  # List of {"extraction_id": str, "pres_id": str, "segment_index": int, "reasons": list, "timestamp": str}
            })

    def load_progress(self) -> Dict[str, Any]:
        """
        Load progress with file locking.

        Returns:
            Progress dict
        """
        with open(self.lock_file, 'r+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            try:
                if self.progress_file.exists():
                    with open(self.progress_file) as f:
                        return json.load(f)
                return {}
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def save_progress(self, progress: Dict[str, Any]):
        """
        Save progress with file locking and atomic write.

        Args:
            progress: Progress dict to save
        """
        with open(self.lock_file, 'r+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
            try:
                temp = self.progress_file.parent / f"{self.progress_file.name}.tmp"
                with open(temp, 'w') as f:
                    json.dump(progress, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                temp.replace(self.progress_file)  # Atomic rename
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def update_paper_progress(
        self,
        processed: int = 0,
        failed: int = 0,
        extractions: int = 0,
        last_batch: Optional[int] = None
    ):
        """
        Update paper processing progress.

        Args:
            processed: Papers processed in this update
            failed: Papers failed in this update
            extractions: Extractions added in this update
            last_batch: Last batch number processed
        """
        progress = self.load_progress()

        progress["papers"]["processed"] += processed
        progress["papers"]["failed"] += failed
        progress["papers"]["extractions_count"] += extractions
        if last_batch is not None:
            progress["papers"]["last_batch"] = last_batch

        self.save_progress(progress)

    def update_presentation_progress(
        self,
        presentations_processed: int = 0,
        segments_processed: int = 0,
        segments_failed: int = 0,
        extractions: int = 0,
        topic_distribution: Optional[Dict[str, int]] = None
    ):
        """
        Update presentation processing progress.

        Args:
            presentations_processed: Presentations completed
            segments_processed: Segments processed
            segments_failed: Segments failed
            extractions: Extractions added
            topic_distribution: Topic counts update
        """
        progress = self.load_progress()

        progress["presentations"]["processed"] += presentations_processed
        progress["presentations"]["segments_processed"] += segments_processed
        progress["presentations"]["segments_failed"] += segments_failed
        progress["presentations"]["extractions_count"] += extractions

        # Update topic distribution
        if topic_distribution:
            for topic_id, count in topic_distribution.items():
                current = progress["topic_distribution"].get(topic_id, 0)
                progress["topic_distribution"][topic_id] = current + count

        self.save_progress(progress)

    def update_cost_tracking(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float
    ):
        """
        Update cost tracking.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost_usd: Cost in USD
        """
        progress = self.load_progress()

        progress["cost_tracking"]["total_tokens_input"] += input_tokens
        progress["cost_tracking"]["total_tokens_output"] += output_tokens
        progress["cost_tracking"]["estimated_cost_usd"] += cost_usd

        self.save_progress(progress)

    def update_pre_filter_stats(
        self,
        invalid_extraction_type: int = 0,
        invalid_timestamps: int = 0
    ):
        """
        Update pre-filter statistics for invalid extractions.

        Args:
            invalid_extraction_type: Count of extractions with invalid type (e.g., "none")
            invalid_timestamps: Count of extractions with invalid timestamps (start >= end)
        """
        progress = self.load_progress()

        # Initialize pre_filter_stats if not present (for older progress files)
        if "pre_filter_stats" not in progress:
            progress["pre_filter_stats"] = {
                "invalid_extraction_type": 0,
                "invalid_timestamps": 0
            }

        progress["pre_filter_stats"]["invalid_extraction_type"] += invalid_extraction_type
        progress["pre_filter_stats"]["invalid_timestamps"] += invalid_timestamps

        self.save_progress(progress)

    def record_failed_segment(
        self,
        pres_id: str,
        segment_index: int,
        error: str
    ):
        """
        Record a failed segment for later retry.

        Args:
            pres_id: Presentation ID
            segment_index: Segment index that failed
            error: Error message
        """
        from datetime import datetime, timezone

        progress = self.load_progress()

        # Initialize failed_segments if not present (for older progress files)
        if "failed_segments" not in progress:
            progress["failed_segments"] = []

        # Check if this segment is already recorded (avoid duplicates)
        existing = [
            f for f in progress["failed_segments"]
            if f["pres_id"] == pres_id and f["segment_index"] == segment_index
        ]
        if existing:
            # Update existing entry with new error
            for f in existing:
                f["error"] = error
                f["timestamp"] = datetime.now(timezone.utc).isoformat()
        else:
            # Add new failure record
            progress["failed_segments"].append({
                "pres_id": pres_id,
                "segment_index": segment_index,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        self.save_progress(progress)

    def get_failed_segments(self) -> List[Dict[str, Any]]:
        """
        Get list of all failed segments.

        Returns:
            List of failed segment records
        """
        progress = self.load_progress()
        return progress.get("failed_segments", [])

    def clear_failed_segment(self, pres_id: str, segment_index: int):
        """
        Remove a segment from the failed list (after successful retry).

        Args:
            pres_id: Presentation ID
            segment_index: Segment index
        """
        progress = self.load_progress()

        if "failed_segments" not in progress:
            return

        progress["failed_segments"] = [
            f for f in progress["failed_segments"]
            if not (f["pres_id"] == pres_id and f["segment_index"] == segment_index)
        ]

        self.save_progress(progress)

    def clear_all_failed_segments(self):
        """Clear all failed segment records."""
        progress = self.load_progress()
        progress["failed_segments"] = []
        self.save_progress(progress)

    def record_qa_failure(
        self,
        extraction_id: str,
        pres_id: str,
        segment_index: int,
        reasons: List[str],
        extraction_data: Dict[str, Any] = None
    ):
        """
        Record a QA validation failure for later review.

        Args:
            extraction_id: Extraction ID that failed QA
            pres_id: Presentation ID
            segment_index: Segment index
            reasons: List of failure reasons
            extraction_data: Optional full extraction data for debugging
        """
        from datetime import datetime, timezone

        progress = self.load_progress()

        # Initialize qa_failures if not present (for older progress files)
        if "qa_failures" not in progress:
            progress["qa_failures"] = []

        # Add new failure record (don't dedupe - may want to see all failures)
        failure_record = {
            "extraction_id": extraction_id,
            "pres_id": pres_id,
            "segment_index": segment_index,
            "reasons": reasons,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Optionally include extraction data for debugging
        if extraction_data:
            failure_record["extraction_data"] = extraction_data

        progress["qa_failures"].append(failure_record)

        self.save_progress(progress)

    def get_qa_failures(self) -> List[Dict[str, Any]]:
        """
        Get list of all QA failures.

        Returns:
            List of QA failure records
        """
        progress = self.load_progress()
        return progress.get("qa_failures", [])

    def clear_all_qa_failures(self):
        """Clear all QA failure records."""
        progress = self.load_progress()
        progress["qa_failures"] = []
        self.save_progress(progress)
