# Exhaustive Extraction System - User Guide

## Overview

The exhaustive extraction system performs automated content extraction from conference presentations and papers, organizing findings by research topic. It processes all available content systematically with quality assurance validation and crash-safe progress tracking.

### What It Does

- **Processes presentations**: Extracts content from video transcripts and slide OCR, segment by segment
- **Processes papers**: Extracts content from paper abstracts and metadata
- **Multi-topic classification**: Each segment/paper can yield extractions for multiple topics (1 segment → up to 20 topics)
- **Quality assurance**: Validates timestamps, quotes, slide references, and detects hallucinations
- **Topic guardrails**: Enforces topic-specific keyword requirements and quality thresholds
- **Deduplication**: Prevents duplicate extractions using content hashing
- **Cost tracking**: Monitors token usage and estimates API costs
- **Resume capability**: Can pause and resume processing without data loss

### Architecture

```
Presentations (129)          Papers (9,825)
      ↓                            ↓
  Segments.json              papers.json
  (2-5 min chunks)           (abstracts)
      ↓                            ↓
  ┌─────────────────────────────────────┐
  │   Model Selection + Failover        │
  │   Sonnet < 50K → OpenRouter Sonnet  │
  │   Gemini 50K-1M → OpenRouter Gemini │
  │   Haiku (final fallback)            │
  └─────────────────────────────────────┘
      ↓                            ↓
  ┌─────────────────────────────────────┐
  │   Extraction Prompts                │
  │   + Topic Definitions               │
  │   + Guardrail Requirements          │
  └─────────────────────────────────────┘
      ↓                            ↓
  ┌─────────────────────────────────────┐
  │   QA Validation                     │
  │   - Timestamp accuracy              │
  │   - Quote verbatim check            │
  │   - Slide reference validation      │
  │   - Topic guardrail enforcement     │
  │   - Hallucination detection         │
  └─────────────────────────────────────┘
      ↓ (pass)              ↓ (fail)
  ┌──────────────────┐  ┌─────────────────┐
  │ Storage (Queue)  │  │ QA Failure Log  │
  │ by_topic/*.jsonl │  │ progress.json   │
  └──────────────────┘  └─────────────────┘
```

## Prerequisites

### API Keys

You need at least one API key:

- **Required**: `ANTHROPIC_API_KEY` (for Claude Sonnet/Haiku)
- **Recommended**: `GOOGLE_API_KEY` (for Gemini, handles large segments 50K-1M tokens)
- **Optional**: `OPENROUTER_API_KEY` (enables failover when direct APIs fail)

Set these in your `.env` file:

```bash
# Claude API key (required)
ANTHROPIC_API_KEY="sk-ant-api03-..."

# Google API key (recommended for large segments)
GOOGLE_API_KEY="AIzaSy..."

# OpenRouter API key (optional - enables automatic failover)
OPENROUTER_API_KEY="sk-or-v1-..."
```

### Data Requirements

The system expects this directory structure:

```
conferences/neurips2025/
├── data/
│   ├── presentations/
│   │   ├── <presentation-id>/
│   │   │   ├── meta.json
│   │   │   ├── transcripts/
│   │   │   │   ├── transcript.txt
│   │   │   │   └── segments.json
│   │   │   └── ocr/
│   │   │       ├── slide_001.json
│   │   │       ├── slide_002.json
│   │   │       └── ...
│   │   └── ...
│   └── papers.json
└── analysis/
    ├── NeurIPS_Analysis_Plan.md  (topic definitions)
    └── topic_guardrails.json     (quality rules)
```

## Quick Start

### Basic Usage

```bash
# Process all presentations (recommended for first run)
python scripts/run_exhaustive_extraction.py --mode presentations

# Process all papers
python scripts/run_exhaustive_extraction.py --mode papers

# Process everything
python scripts/run_exhaustive_extraction.py --mode all
```

### Test Run (Small Sample)

Before processing everything, test with a small sample:

```bash
# Test with 1 presentation and 10 papers
python scripts/run_exhaustive_extraction.py \
  --mode all \
  --limit-presentations 1 \
  --limit-papers 10
```

This validates:
- API keys work
- Guardrails are loading correctly
- QA validation is functioning
- Output format is correct

Expected cost: run the registry-backed estimator before approval:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope exhaustive-extraction
```

### Resume After Interruption

The system automatically tracks progress. If interrupted (Ctrl+C or crash):

```bash
# Resume from last checkpoint (default behavior)
python scripts/run_exhaustive_extraction.py --mode presentations

# Start fresh (clear progress and reprocess everything)
python scripts/run_exhaustive_extraction.py --mode presentations --no-resume
```

## Command-Line Options

```bash
python scripts/run_exhaustive_extraction.py [OPTIONS]

Required:
  --mode {presentations,papers,all}
                        What to process

Optional:
  --conf-dir PATH       Conference directory
                        (default: conferences/neurips2025)

  --output-dir PATH     Output directory
                        (default: conferences/neurips2025/exports/exhaustive-extractions)

  --batch-size INT      Papers per batch (for paper processing)
                        (default: 100)

  --no-resume           Start from scratch, ignore existing progress

  --limit-presentations INT
                        Process only first N presentations (for testing)

  --limit-papers INT    Process only first N papers (for testing)

  --max-workers INT     Parallel workers for segment processing
                        (default: 3)

  --batch-segments      Enable segment batching for long presentations

Failure Management:
  --list-failed         List all failed segments and exit
  --retry-failed        Retry all previously failed segments
  --clear-failed        Clear failed segment records

  --list-qa-failures    List all QA validation failures and exit
  --clear-qa-failures   Clear QA failure records
```

For the NeurIPS 2025 analysis plan, use:

```bash
--output-dir conferences/neurips2025/exports/exhaustive-extractions
```

## Configuration

### Topic Guardrails

Topic-specific quality rules are defined in `conferences/<conf>/analysis/topic_guardrails.json`:

```json
{
  "01": {
    "required_cues": ["benchmark", "framework", "infrastructure"],
    "allow_generic": false,
    "min_summary_len": 6
  },
  "19": {
    "required_cues": [],
    "allow_generic": false,
    "min_quote_len": 12,
    "min_summary_len": 10
  }
}
```

#### Guardrail Parameters

- **`required_cues`**: Keywords that must appear in extraction (case-insensitive substring match)
- **`allow_generic`**: If `false`, rejects generic phrases like "they discussed" or "scaling laws"
- **`min_quote_len`**: Minimum word count for verbatim quotes
- **`min_summary_len`**: Minimum word count for summaries

#### How Guardrails Work

1. **Prompt Integration**: Guardrails are injected into extraction prompts, guiding the model to focus on topic-specific content
2. **QA Validation**: After extraction, QA validator checks:
   - Required cues are present in quote or summary
   - Quote/summary meet minimum length requirements
   - Generic phrases are absent (if `allow_generic: false`)
3. **Rejection**: Extractions failing guardrails are rejected and not stored

### Model Selection

The system automatically selects the best model based on segment size, with automatic failover:

| Token Count | Primary Model | Failover (if primary fails) |
|------------|--------------|----------------------------|
| < 50,000 | Claude Sonnet 4.6 | OpenRouter Sonnet → Haiku |
| 50,000 - 1M | Gemini 2.5 Pro | OpenRouter Gemini |
| ≥ 1M | **Error** | Segment too large |

**API Key Priority**:
1. `ANTHROPIC_API_KEY` - Required for Sonnet/Haiku
2. `GOOGLE_API_KEY` - Required for Gemini (large segments)
3. `OPENROUTER_API_KEY` - Enables automatic failover when direct APIs fail

**Important**: If `GOOGLE_API_KEY` is not set, segments with 50K-1M tokens will fail with a clear error message. Set `OPENROUTER_API_KEY` for automatic failover when direct APIs are rate-limited or unavailable.

## Output Structure

### Directory Layout

```
conferences/neurips2025/exports/exhaustive-extractions/
├── progress.json                    # Progress tracking and stats
├── presentations/
│   └── by_topic/
│       ├── 01.jsonl                 # Tools & Infrastructure
│       ├── 02.jsonl                 # GenAI Architectures
│       ├── ...
│       └── 20.jsonl                 # Small Language Models
└── papers/
    └── by_topic/
        ├── 01.jsonl
        ├── 02.jsonl
        ├── ...
        └── 20.jsonl
```

### Extraction Record Format

Each line in `*.jsonl` files contains one extraction:

```json
{
  "extraction_id": "agentic-ai-rl_seg003_17_extract001",
  "content_hash": "sha256:a3f2c1...",
  "source_type": "presentation",
  "source_id": "agentic-ai-rl",
  "source_title": "Agentic Development at the Frontier",
  "topic_id": "17",
  "topic_name": "LLM Scaling Laws & Efficiency",
  "extraction_type": "insight",
  "verbatim_quote": "We reduced inference latency by 40% using speculative decoding...",
  "summary": "Speculative decoding technique reduced inference latency by 40%...",
  "relevance_score": 0.85,
  "timestamp_start": "00:05:23",
  "timestamp_end": "00:05:47",
  "timestamp_start_sec": 323.5,
  "timestamp_end_sec": 347.2,
  "segment_index": 3,
  "segment_start_sec": 300.0,
  "segment_end_sec": 420.0,
  "slide_references": [
    {
      "slide_id": "slide_008",
      "timestamp_sec": 325.1,
      "timestamp_aligned": true,
      "ocr_confidence": "high"
    }
  ],
  "model_used": "claude-sonnet-4-6",
  "extracted_at": "2026-01-05T12:34:56.789Z",
  "qa_validated": true
}
```

### Progress Tracking

`progress.json` contains real-time statistics and failure records:

```json
{
  "papers": {
    "total": 9825,
    "processed": 500,
    "failed": 2,
    "last_batch": 4,
    "extractions_count": 2341
  },
  "presentations": {
    "total": 129,
    "processed": 45,
    "total_segments": 2160,
    "segments_processed": 850,
    "segments_failed": 3,
    "extractions_count": 1879
  },
  "cost_tracking": {
    "total_tokens_input": 1250000,
    "total_tokens_output": 185000,
    "estimated_cost_usd": 6.525
  },
  "topic_distribution": {
    "01": 143,
    "02": 267,
    "03": 892,
    "17": 1024,
    "18": 651
  },
  "failed_segments": [
    {
      "pres_id": "ai-workshop-2025",
      "segment_index": 42,
      "error": "Both primary (claude-sonnet-4-5) and fallback failed...",
      "timestamp": "2026-01-05T10:23:45.123456+00:00"
    }
  ],
  "qa_failures": [
    {
      "extraction_id": "ai-workshop-2025_talk_003_17_sha256:a3f2c1...",
      "pres_id": "ai-workshop-2025",
      "segment_index": 42,
      "reasons": [
        "quote_check: Verbatim quote not found in transcript (severity: high)"
      ],
      "timestamp": "2026-01-05T10:23:45.123456+00:00",
      "extraction_data": { "...full extraction for debugging..." }
    }
  ]
}
```

## Monitoring and Troubleshooting

### Real-Time Monitoring

Watch progress in real-time:

```bash
# Run extraction in one terminal
python scripts/run_exhaustive_extraction.py --mode presentations

# Monitor progress in another terminal
watch -n 5 'cat conferences/neurips2025/exports/exhaustive-extractions/progress.json | jq'
```

### Understanding Output

During processing, you'll see:

```
[12/129]
  Processing: agentic-ai-rl
  Title: Agentic Development at the Frontier
  Segments: 48

  Segment 5: 00:10:15 - 00:12:45 (150s)
    Model: claude-sonnet-4-6
    Tokens: 12,450 → 1,234
    Extractions: 3
      ✓ Valid extractions: 3/3

  Segment 6: 00:12:45 - 00:15:20 (155s)
    Model: claude-sonnet-4-6
    Tokens: 13,200 → 987
    Extractions: 2
      ⚠️  QA failed for extraction: agentic-ai-rl_seg006_19_extract001
         - topic_guardrail_check: Quote too short (<12 words) (severity: medium)
      ✓ Valid extractions: 1/2
```

### Common Issues

#### 1. Missing API Key

```
RuntimeError: No available models - check API keys
```

**Solution**: Set `ANTHROPIC_API_KEY` in `.env`

#### 2. Segment Too Large

```
RuntimeError: Segment too large: 1,250,000 tokens exceeds maximum supported limit
without Gemini API (1,000,000 tokens). Set GOOGLE_API_KEY to process large segments.
```

**Solution**: Add `GOOGLE_API_KEY` to `.env`, or contact maintainer to split segment

#### 3. Import Errors

```
ModuleNotFoundError: No module named 'exhaustive_extraction'
```

**Solution**: Run from repository root, not from `scripts/` directory

#### 4. QA Validation Failures

High QA failure rate (>50%) indicates:
- Guardrails too strict for the content
- Model not following prompt instructions
- Source data quality issues (poor OCR, bad transcripts)

**Solution**:
1. Review `topic_guardrails.json` - adjust `required_cues` or `min_*_len`
2. Inspect rejected extractions to understand patterns
3. Consider relaxing `allow_generic: true` for broad topics

### Resume Stability Warnings

```
⚠️  WARNING: Resume assumes stable presentation ID ordering
```

This means:
- Resume relies on presentation/paper IDs staying in the same order
- If filesystem ordering changes between runs, resume may skip/duplicate items
- For production at scale, consider persisting processed ID sets instead of positional indexing

## Cost Estimation

Pricing is centralized in
`src/conference_pipeline/costs/pricing_catalog.json` and task assumptions are in
`src/conference_pipeline/costs/task_specs.yaml`. Use the estimator instead of
copying model prices into this guide:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope exhaustive-extraction \
  --scenario base
```

OpenRouter fallback estimates are included in the full-pipeline scope; refresh
against the live OpenRouter catalog before final approval because route prices
can change.

### Historical NeurIPS 2025 Test Run

**Test**: 1 presentation (48 segments) + 10 papers

- Input tokens: 75,508
- Output tokens: 12,478
- Cost: **$0.41 USD**
- Extractions: 132 (79 presentation + 53 papers)

**Historical full-run planning note**: older NeurIPS notes estimated 129
presentations plus 9,825 papers at roughly 10M input tokens, 1.6M output
tokens, and about `$54`. Treat that as historical run evidence only; new
approvals should cite the current estimator output.

### Cost Optimization

1. **Use Gemini for large segments**: Set `GOOGLE_API_KEY` (Gemini is cheaper than Sonnet for 50K+ token contexts)
2. **Batch papers**: Default batch size of 100 is optimal for cost/speed balance
3. **Test first**: Always run `--limit-presentations 1 --limit-papers 10` before full run
4. **Monitor progress.json**: Check `estimated_cost_usd` field regularly

## Quality Assurance

### Validation Checks

Every extraction undergoes these checks:

#### 1. Timestamp Validation (Presentations Only)

- Timestamps must be within segment bounds
- Start must be < End
- If slide reference marked as `timestamp_aligned: true`, slide timestamp must match extraction timestamp (±5s tolerance)

**Failure**: Extraction rejected with severity `high`

#### 2. Quote Verbatim Check

- If `verbatim_quote` provided, must appear in source transcript/abstract (exact substring match, case-insensitive)

**Failure**: Extraction marked low confidence, severity `medium`

#### 3. Slide Reference Validation (Presentations Only)

- Referenced slide files must exist at specified paths
- Slide timestamps should align with extraction timestamps

**Failure**: Severity `low` (warning only)

#### 4. Hallucination Detection

- Rejects extractions with generic phrases: "they discussed", "the paper talks about", "this presentation covers"
- Rejects summaries < 5 words (too vague)
- Disabled if topic allows generic content (`allow_generic: true`)

**Failure**: Severity `medium`

#### 5. Topic Guardrail Enforcement

- Required cues must be present (e.g., "benchmark" for Topic 01)
- Minimum quote/summary lengths must be met
- Generic content blocked unless `allow_generic: true`

**Failure**: Severity `high`

### QA Failure Handling

Extractions failing QA are:
- **Not stored** (rejected completely)
- **Logged** to console with failure reason and severity
- **Counted** in progress stats (see `segments_failed`)

Example QA failure log:

```
⚠️  QA failed for extraction: agentic-ai-rl_seg006_19_extract001
   - topic_guardrail_check: Quote too short (<12 words) (severity: medium)
   - hallucination_check: Generic statement detected: 'they discussed' (severity: medium)
```

## Reusability for Other Conferences

The system is designed to be conference-agnostic. To use for a new conference:

### Step 1: Prepare Conference Data

Create directory structure:

```
conferences/myconf2026/
├── data/
│   ├── presentations/
│   │   └── <presentation-id>/
│   │       ├── meta.json
│   │       ├── transcripts/segments.json
│   │       └── ocr/*.json
│   └── papers.json
└── analysis/
    ├── MyConf_Analysis_Plan.md
    └── topic_guardrails.json
```

### Step 2: Define Topics

In `MyConf_Analysis_Plan.md`, define your research topics:

```markdown
## Research Topics

### 01. Topic Name Here
Brief description of what this topic covers...

### 02. Another Topic
Description...
```

### Step 3: Configure Guardrails

In `topic_guardrails.json`, add quality rules:

```json
{
  "01": {
    "required_cues": ["keyword1", "keyword2"],
    "allow_generic": false,
    "min_summary_len": 6
  }
}
```

### Step 4: Run Extraction

```bash
python scripts/run_exhaustive_extraction.py \
  --mode all \
  --conf-dir conferences/myconf2026 \
  --output-dir conferences/myconf2026/exhaustive_extractions
```

That's it! The system will:
- Load topics from your analysis plan
- Apply your guardrails
- Process all content
- Output extractions to your conference directory

## Failure Mitigation and Recovery

The extraction system implements multiple layers of failure mitigation to ensure robust processing even when APIs fail, segments have issues, or extractions fail QA validation.

### Multi-Level Failure Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 1: API Failover                                                       │
│  Sonnet → OpenRouter Sonnet → Haiku → OpenRouter Haiku                      │
│  Gemini → OpenRouter Gemini                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 2: Retry with Exponential Backoff                                    │
│  5 attempts, initial delay 5s, doubles each retry                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 3: Segment Failure Tracking                                          │
│  Failed segments recorded for later --retry-failed                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 4: QA Failure Recording                                              │
│  QA failures stored with full extraction data for analysis                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 5: Crash-Safe Resume                                                 │
│  Atomic writes, file locking, graceful shutdown                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Level 1: API Failover (OpenRouter)

When direct API calls fail, the system automatically falls back to OpenRouter:

| Primary Model | OpenRouter Fallback |
|---------------|---------------------|
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` |
| Claude Haiku 4.5 | `anthropic/claude-3-5-haiku` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |

**Configuration**: Set `OPENROUTER_API_KEY` in `.env` to enable failover:

```bash
# OpenRouter API key (enables failover when direct APIs fail)
OPENROUTER_API_KEY="sk-or-v1-..."
```

**Example output during failover**:
```
⚠️  Primary model failed: RateLimitError
🔄 Trying OpenRouter fallback: anthropic/claude-sonnet-4
✅ OpenRouter fallback succeeded
```

### Level 2: Retry with Exponential Backoff

Every API call includes automatic retry logic:

- **Max retries**: 5 attempts
- **Initial delay**: 5 seconds
- **Backoff**: Doubles each attempt (5s → 10s → 20s → 40s → 80s)
- **Handles**: Connection errors, timeouts, rate limits

**Example output during retry**:
```
API error (attempt 1/5): RateLimitError
Retrying in 5.0s...
API error (attempt 2/5): RateLimitError
Retrying in 10.0s...
[success on attempt 3]
```

### Level 3: Segment Failure Tracking

When a segment fails after all retries, it's recorded for later retry:

**View failed segments**:
```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --list-failed \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

**Output**:
```
================================================================================
FAILED SEGMENTS (3 total)
================================================================================

📁 ai-workshop-2025 (2 failed segments):
   Segment 42: Both primary (claude-sonnet-4-5) and fallback failed...
      Failed at: 2026-01-05T10:23:45.123456+00:00
   Segment 87: JSONDecodeError: Expecting property name...
      Failed at: 2026-01-05T10:45:12.987654+00:00

📁 ml-tutorial (1 failed segment):
   Segment 15: Connection timeout after 120s...
      Failed at: 2026-01-05T11:02:33.456789+00:00

================================================================================
To retry: python scripts/run_exhaustive_extraction.py --mode presentations --retry-failed
================================================================================
```

**Retry failed segments**:
```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --retry-failed \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

**Clear failed segments** (after manual investigation):
```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --clear-failed \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

### Level 4: QA Failure Recording

Extractions that fail QA validation are recorded with full context for later analysis:

**View QA failures**:
```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --list-qa-failures \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

**Output**:
```
================================================================================
QA VALIDATION FAILURES (47 total)
================================================================================

📁 agentic-ai-workshop (12 QA failures):

   Extraction: agentic-ai-workshop_talk_003_17_sha256:a3f2c1...
   Segment: 42
   Timestamp: 2026-01-05T10:23:45.123456+00:00
   Reasons:
      - quote_check: Verbatim quote not found in transcript (severity: high)

   Extraction: agentic-ai-workshop_talk_003_19_sha256:b4e3d2...
   Segment: 42
   Timestamp: 2026-01-05T10:23:46.234567+00:00
   Reasons:
      - topic_guardrail_check: Quote too short (<12 words) (severity: medium)
      - hallucination_check: Generic statement detected (severity: medium)

================================================================================
FAILURE TYPE SUMMARY
================================================================================
  quote_check: 23 failures
  topic_guardrail_check: 15 failures
  hallucination_check: 7 failures
  slide_checks: 2 failures

================================================================================
```

**Clear QA failures** (after reviewing and adjusting guardrails):
```bash
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --clear-qa-failures \
  --output-dir conferences/neurips2025/exports/exhaustive-extractions
```

**Analyzing QA failures**:

The QA failure records include full extraction data, enabling deep analysis:

```python
import json

# Load progress file
with open("conferences/neurips2025/exports/exhaustive-extractions/progress.json") as f:
    progress = json.load(f)

# Analyze QA failures
qa_failures = progress.get("qa_failures", [])
for failure in qa_failures[:5]:
    print(f"Extraction: {failure['extraction_id']}")
    print(f"Reasons: {failure['reasons']}")
    if "extraction_data" in failure:
        print(f"Summary: {failure['extraction_data']['summary']}")
        print(f"Quote: {failure['extraction_data'].get('verbatim_quote', 'N/A')}")
    print()
```

### Level 5: Crash-Safe Resume

The system ensures no data loss even with unexpected termination:

**File Locking**:
- Uses `fcntl` locks for atomic progress.json updates
- Prevents corruption from concurrent writes

**Atomic Writes**:
- Progress written to temp file first
- Atomically renamed to final location
- Ensures complete file or no change

**Queue-Based Storage**:
- Background writer threads per topic
- Batched writes with fsync
- Graceful shutdown flushes all pending data

**Graceful Shutdown**:
```
^C

⚠️  Interrupted by user
Progress saved. Run again with same command to resume.

Shutting down storage...
✓ Storage shutdown complete

================================================================================
EXTRACTION SUMMARY
================================================================================
...
```

### CLI Options for Failure Management

| Option | Description |
|--------|-------------|
| `--list-failed` | List all failed segments with error details |
| `--retry-failed` | Retry all previously failed segments |
| `--clear-failed` | Clear failed segment records |
| `--list-qa-failures` | List all QA validation failures with reasons |
| `--clear-qa-failures` | Clear QA failure records |
| `--no-resume` | Start fresh (ignore saved progress) |

### Summary: Final Status Output

After each run, the summary includes failure counts:

```
================================================================================
EXTRACTION SUMMARY
================================================================================

Presentations:
  Processed: 45/129
  Segments: 850/2160
  Failed segments: 3
  Extractions: 1879

Papers:
  Processed: 500/9825
  Failed: 2
  Extractions: 2341

Cost Tracking:
  Input tokens: 1,250,000
  Output tokens: 185,000
  Estimated cost: $6.52

Top 10 Topics by Extraction Count:
  Topic 17: 1024 extractions
  Topic 03: 892 extractions
  ...

⚠️  Failed Segments: 3
   Run with --list-failed to see details
   Run with --retry-failed to retry them

⚠️  QA Failures: 47
   Run with --list-qa-failures to see details
   Run with --clear-qa-failures to clear them

================================================================================
```

## Advanced Features

### Deduplication

The system prevents duplicate extractions using:

1. **Content hashing**: SHA256 hash of (source_id + segment_index + topic_id + content)
2. **O(1) dedup**: In-memory hash index per topic for fast lookups
3. **Cross-run dedup**: Hash index persisted across runs, prevents re-processing on resume

### Crash Safety

- **File locking**: `fcntl` locks prevent concurrent writes to progress.json
- **Atomic updates**: Progress written atomically using temp file + rename
- **Graceful shutdown**: Ctrl+C triggers clean shutdown, flushes all pending writes
- **Queue-based storage**: Background writer threads ensure no data loss

### Batch Processing

Papers are processed in batches (default 100) to:
- Reduce API overhead
- Enable efficient progress tracking
- Allow partial failures without losing entire batch

### Segment-by-Segment Processing

Presentations are processed segment-by-segment (not whole transcript) because:
- **Token limits**: Segments fit within model context windows
- **Timestamp accuracy**: Segment-relative timestamps for precise attribution
- **Parallel potential**: Segments can be parallelized in future versions
- **Resume granularity**: Can resume mid-presentation

## Best Practices

### Before Running

1. ✅ **Test with small sample** (`--limit-presentations 1 --limit-papers 10`)
2. ✅ **Verify API keys** are set in `.env`
3. ✅ **Check guardrails** are appropriate for your conference topics
4. ✅ **Estimate costs** using test run metrics

### During Processing

1. ✅ **Monitor progress.json** regularly
2. ✅ **Check QA failure rates** - high rates indicate guardrail/data issues
3. ✅ **Don't kill process forcefully** - use Ctrl+C for graceful shutdown
4. ✅ **Monitor API rate limits** - built-in retry with exponential backoff

### After Completion

1. ✅ **Verify extraction counts** match expected ranges
2. ✅ **Spot-check topic files** for quality (sample 2-3 extractions per topic)
3. ✅ **Review cost tracking** to validate estimates
4. ✅ **Check for failed segments** in progress.json

## Performance

### Throughput

Based on NeurIPS 2025 test run:

- **Presentations**: ~2-3 minutes per segment (model-dependent)
- **Papers**: ~10-20 papers per minute (batch processing)
- **Concurrency**: Single-threaded (safe for API rate limits)

### Bottlenecks

1. **API latency**: 1-3 seconds per request
2. **Model processing**: Proportional to token count
3. **QA validation**: Negligible (<0.1s per extraction)

### Optimization Opportunities

- Parallel segment processing (requires queue refactoring)
- Faster models for simple topics (Haiku for all)
- Caching repeated prompts (topic definitions)

## Troubleshooting Checklist

| Issue | Check | Solution |
|-------|-------|----------|
| Import errors | Running from repo root? | `cd /path/to/neurips-foundation-conf-pipeline` |
| API errors | Keys in `.env`? | `grep API_KEY .env` |
| Segment too large | Gemini key set? | Add `GOOGLE_API_KEY` to `.env` |
| High QA failures | Guardrails too strict? | Adjust `topic_guardrails.json` |
| Low extraction yield | Guardrails too loose? | Add more `required_cues` |
| Duplicate extractions | Hash collision? | Check `content_hash` uniqueness |
| Progress not saving | File permissions? | Check write access to output dir |
| Resume not working | Started with `--no-resume`? | Remove flag to enable resume |
| Rate limit errors | Direct API saturated? | Add `OPENROUTER_API_KEY` for failover |
| Segments failing | Check failed segments | `--list-failed` then `--retry-failed` |
| QA rejections high | Review QA failure patterns | `--list-qa-failures` to analyze |
| Want fresh start | Clear all tracking | `--clear-failed` and `--clear-qa-failures` |

## Support and Feedback

- **Documentation**: `docs/analysis-plans/exhaustive-analysis-plan-v2.md` (detailed design)
- **Code**: `scripts/exhaustive_extraction/` (all modules)
- **Tests**: `tests/test_exhaustive_extraction.py` (validation examples)
- **Issues**: Check existing error messages - they include actionable guidance

## Appendix: File Reference

### Core Modules

| File | Purpose |
|------|---------|
| `run_exhaustive_extraction.py` | Entry point, CLI |
| `schemas.py` | Pydantic models, validation |
| `prompts.py` | Extraction prompt templates |
| `models.py` | Model selection, API wrappers |
| `storage.py` | Queue-based storage, deduplication |
| `qa_validator.py` | Quality assurance checks |
| `processor.py` | Presentation/paper processing logic |
| `guardrails.py` | Guardrail loading |

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API keys |
| `conferences/<conf>/analysis/NeurIPS_Analysis_Plan.md` | Topic definitions |
| `conferences/<conf>/analysis/topic_guardrails.json` | Quality rules |

### Output Files

| File | Purpose |
|------|---------|
| `progress.json` | Real-time progress tracking |
| `presentations/by_topic/*.jsonl` | Presentation extractions by topic |
| `papers/by_topic/*.jsonl` | Paper extractions by topic |

## Source References

- scripts/run_exhaustive_extraction.py
- scripts/exhaustive_extraction/processor.py
- scripts/exhaustive_extraction/schemas.py
- scripts/exhaustive_extraction/qa_validator.py
- scripts/exhaustive_extraction/guardrails.py
- scripts/list_exhaustive_sources.py
