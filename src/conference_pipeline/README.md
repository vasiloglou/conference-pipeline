# NeurIPS Conference Pipeline

A comprehensive data acquisition and analysis pipeline for extracting, processing, and analyzing content from NeurIPS conference presentations.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA ACQUISITION PHASE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. discover_sessions.py    ──►  sessions.json (manual filtering)           │
│           │                                                                 │
│           ▼                                                                 │
│  2. download_from_sessions.py  ──►  audio/ + slides/                        │
│           │                                                                 │
│           ├──────────────────────────┐                                      │
│           ▼                          ▼                                      │
│  3. transcribe_sessions.py     cleanup_slides.py (optional)                 │
│           │                          │                                      │
│           ▼                          ▼                                      │
│  cleanup_transcripts.py (opt)  detect_reference_slides.py (optional)        │
│           │                          │                                      │
│           ▼                          │                                      │
│  transcripts/segments.json           │                                      │
│           │                          │                                      │
│           └──────────────────────────┤                                      │
│                                      ▼                                      │
│                            4. ocr_slides.py  ──►  ocr/slide_XXX.json        │
│                                                                             │
│  [PARALLEL PATH - Can run anytime]                                          │
│  collect_papers.py  ──►  papers.json                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  5. setup_vector_store_enhanced.py  ──►  ChromaDB + entity_topic_index.json │
│           │                                                                 │
│           ▼                                                                 │
│  [Optional: setup BM25 for hybrid search]                                   │
│  query_hybrid_bm25.py --rebuild     ──►  BM25 index cache                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  6. query_gpt_researcher_hybrid.py  ──►  Research reports (per topic)       │
│           │                                                                 │
│           ▼                                                                 │
│  [Write analysis documents: 01_tools.md, 02_architectures.md, etc.]         │
│           │                                                                 │
│           ▼                                                                 │
│  7. analyze_presentation_coverage.py ──►  coverage_report.md                │
│           │                                                                 │
│           ▼                                                                 │
│  8. map_presentations_to_topics.py  ──►  presentation_mappings.json         │
│           │                                                                 │
│           ▼                                                                 │
│  9. fill_coverage_gaps.py          ──►  Updated topic documents             │
│           │                                                                 │
│           ▼                                                                 │
│  10. qa_analysis_docs.py            ──►  qa_report.md                       │
│           │                                                                 │
│           ▼                                                                 │
│  11. traceability_recovery.py       ──►  Addenda with source citations      │
│           │                                                                 │
│           ▼                                                                 │
│  12. phase8_paper_analysis.py       ──►  Paper scoring + classification     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT PHASE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  13. generate_podcast.py            ──►  Audio podcast from topics          │
│  14. generate_topic_video.py        ──►  Video summaries with slides        │
│  15. export_for_kg_construction.py  ──►  Knowledge graph JSONL export       │
│  16. sync_to_gdrive.py              ──►  Backup to Google Drive             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Execution Order

Execute scripts in this order. Each step shows dependencies and outputs.

### Stage 1: Data Collection (Run Once)

```bash
# 1.1 Collect paper metadata from OpenReview (can run in parallel with sessions)
python scripts/collect_papers.py --year 2025
# Output: conferences/neurips2025/data/papers.json

# 1.2 Discover all sessions from NeurIPS virtual site
python scripts/discover_sessions.py --year 2025 --output sessions.json
# Output: sessions.json
# MANUAL STEP: Edit sessions.json, set "chosen": "Y" for sessions to process
```

### Stage 2: Asset Download

```bash
# 2.1 Download audio + slides for chosen sessions
python scripts/download_from_sessions.py --sessions-file sessions.json
# Output: data/presentations/{id}/audio/, data/presentations/{id}/slides/
# Resumable: re-run to continue from interruption
```

### Stage 3: Content Processing (Can Partially Parallelize)

```bash
# 3.1 Transcribe audio (requires: audio downloaded)
python scripts/transcribe_sessions.py --parallel 3
# Output: data/presentations/{id}/transcripts/segments.json
# Resumable: skips already transcribed

# 3.2 [Optional] Detect reference slides to skip during OCR
python scripts/detect_reference_slides.py
# Output: data/reference_slides/

# 3.3 [Optional] Mark duplicate slides to skip during OCR
python scripts/cleanup_slides.py
# Output: Updates sync.json with skip_ocr flags

# 3.4 OCR slides (requires: slides downloaded, benefits from transcripts)
python scripts/ocr_slides.py --provider openai --model gpt-5.4-mini --parallel 5
# Output: data/presentations/{id}/ocr/slide_XXX.json
# Uses transcripts for context when available
```

### Stage 4: Indexing (Run After Processing)

```bash
# 4.1 Build vector store with all content
python scripts/setup_vector_store_enhanced.py
# Requires: papers.json, transcripts, OCR results
# Output: vector_store/chroma_db/, entity_topic_index.json

# 4.2 [Optional] Build BM25 index for hybrid search
python scripts/query_hybrid_bm25.py --rebuild
# Output: vector_store/bm25_cache/
```

### Stage 5: Research & Analysis (Iterative)

```bash
# 5.1 Query for research (repeat for each topic)
python scripts/query_gpt_researcher_hybrid.py "What are the advances in [TOPIC]?"
# Output: vector_store/reports/{timestamp}_{query}/

# [MANUAL STEP: Write analysis documents based on query results]
# Output: analysis/01_tools_and_open_source.md, etc.

# 5.2 Check presentation coverage
python scripts/analyze_presentation_coverage.py
# Requires: analysis documents exist
# Output: analysis/coverage_report.md

# 5.3 Map unreferenced presentations to topics
python scripts/map_presentations_to_topics.py
# Requires: coverage_report.md
# Output: analysis/presentation_mappings.json

# 5.4 Fill coverage gaps automatically
python scripts/fill_coverage_gaps.py
# Requires: presentation_mappings.json
# Output: Updates analysis documents with new references

# 5.5 Quality assurance check
python scripts/qa_analysis_docs.py
# Requires: analysis documents
# Output: analysis/qa_report.md

# 5.6 [Optional] Recover missing source citations
python scripts/traceability_recovery.py
# Output: Addenda files with traced sources

# 5.7 [Optional] Analyze poster-only papers
python scripts/phase8_paper_analysis.py --all
# Requires: papers.json, vector_store
# Output: importance_scores.json, paper_topic_mapping.json
```

### Stage 6: Output Generation (Final)

```bash
# 6.1 Generate podcast from topic document
python scripts/generate_podcast.py analysis/09_genai_in_science.md
# Requires: OpenAI API, analysis document
# Output: podcasts/{topic}_podcast.mp3

# 6.2 Generate video from topic document
python scripts/generate_topic_video.py analysis/02_genai_architectures.md
# Requires: OpenAI API, analysis document, slide images
# Output: videos/{topic}_video.mp4, videos/{topic}_deck.pdf

# 6.3 Export for knowledge graph
python scripts/export_for_kg_construction.py
# Requires: transcripts, OCR, papers
# Output: export.jsonl

# 6.4 Backup to Google Drive
python scripts/sync_to_gdrive.py
# Output: Syncs to Google Drive mount point
```

---

## Dependency Graph

```
                    collect_papers.py
                          │
                          ▼
                     papers.json ─────────────────────┐
                                                      │
discover_sessions.py                                  │
        │                                             │
        ▼                                             │
   sessions.json                                      │
        │                                             │
        ▼ [manual: set chosen=Y]                      │
        │                                             │
download_from_sessions.py                             │
        │                                             │
        ▼                                             │
   audio/ + slides/                                   │
        │                                             │
        ├─────────────────┐                           │
        │                 │                           │
        ▼                 ▼                           │
transcribe_sessions.py  cleanup_slides.py (opt)      │
        │                 │                           │
        ▼                 ▼                           │
   segments.json     detect_reference_slides.py      │
        │                 │                           │
        └────────┬────────┘                           │
                 │                                    │
                 ▼                                    │
           ocr_slides.py                             │
                 │                                    │
                 ▼                                    │
           ocr/slide_XXX.json                        │
                 │                                    │
                 └────────────────┬───────────────────┘
                                  │
                                  ▼
                   setup_vector_store_enhanced.py
                                  │
                                  ▼
                         ChromaDB + BM25
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
query_gpt_researcher.py   query_vector_store.py   query_hybrid_bm25.py
            │
            ▼
      research reports
            │
            ▼ [manual: write analysis docs]
            │
analyze_presentation_coverage.py
            │
            ▼
map_presentations_to_topics.py
            │
            ▼
fill_coverage_gaps.py
            │
            ▼
qa_analysis_docs.py
            │
            ▼
   ┌────────┼────────┐
   │        │        │
   ▼        ▼        ▼
podcast   video    KG export
```

---

## Step-by-Step Execution Guide

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Required API keys in the repository .env (then: set -a; source .env; set +a):
export OPENAI_API_KEY="sk-..."      # For transcription + OCR + embeddings
export TAVILY_API_KEY="tvly-..."    # For web search (optional)
export NEURIPS_EMAIL="..."          # NeurIPS login (for authenticated content)
export NEURIPS_PASSWORD="..."
```

---

### Phase 1: Data Acquisition

#### Step 1: Discover Sessions

Scrapes the NeurIPS virtual site to find all presentations (workshops, tutorials, keynotes, etc.)

```bash
python scripts/discover_sessions.py --year 2025 --output sessions.json
```

**Output:** `sessions.json` with all discovered sessions

**Manual Step:** Open `sessions.json` and set `"chosen": "Y"` for sessions you want to process.

---

#### Step 2: Download Assets

Downloads audio (MP3) and slides (PDF/images) for chosen sessions.

```bash
python scripts/download_from_sessions.py --sessions-file sessions.json
```

**Options:**
- `--limit 5` - Process only first 5 sessions (for testing)
- `--retry-errors` - Retry previously failed downloads

**Output per presentation:**
```
data/presentations/{pres-id}/
├── audio/
│   └── audio.mp3
├── slides/
│   └── slide_001.png, slide_002.png, ...
└── meta.json
```

---

#### Step 3: Transcribe Audio

Transcribes audio using OpenAI Whisper API with automatic chunking for large files.

```bash
python scripts/transcribe_sessions.py --parallel 3
```

**Options:**
- `--parallel N` - Process N sessions in parallel (default: 1)
- `--retry-errors` - Retry failed transcriptions
- `--limit 5` - Process only first 5 presentations

**Output per presentation:**
```
data/presentations/{pres-id}/
└── transcripts/
    ├── transcript.txt      # Full text
    └── segments.json       # Timestamped segments
```

---

#### Step 4: OCR Slides

Extracts text, visual elements, and references from slides using a configured
vision model.

```bash
python scripts/ocr_slides.py --provider openai --model gpt-5.4-mini --parallel 5
```

**Options:**
- `--provider openai|anthropic|gemini` - Vision model provider
- `--model` - Primary OCR model override
- `--fallback-provider` / `--fallback-model` - Quality fallback for low-confidence or failed parses
- `--parallel N` - Concurrent API requests (default: 5)
- `--retry-errors` - Retry failed OCRs
- `--limit 5` - Process only first 5 presentations

**Output per presentation:**
```
data/presentations/{pres-id}/
└── ocr/
    ├── slide_001.json      # Structured OCR result
    ├── slide_001.md        # Human-readable markdown
    └── ...
```

**OCR JSON structure:**
```json
{
  "slide_index": 1,
  "timestamp_sec": 123.45,
  "image_path": "slide_001.png",
  "slide_text": {
    "title": "...",
    "bullet_points": [...],
    "equations": [...],
    "other_text": "..."
  },
  "visual_elements": [...],
  "key_concepts": [...],
  "references": [...]
}
```

---

### Phase 2: Indexing

#### Step 5: Build Vector Store

Creates ChromaDB vector store with smart chunking for semantic search.

```bash
python scripts/setup_vector_store_enhanced.py
```

**What gets indexed:**
- **Papers:** Title + abstract as single chunk (9,825 papers)
- **Transcripts:** ~2 minute semantic segments preserving timestamps
- **Slides:** Individual slides with title + content together

**Output:**
```
data/vector_store/
├── chroma_db/              # Persistent ChromaDB database
├── entity_topic_index.json # Entity tracking (tools, models, methods)
└── vector_store_metadata.json
```

---

### Phase 3: Analysis

#### Step 6: Research Queries

Query the vector store using GPT Researcher with hybrid search (local + web).

```bash
# Hybrid search (local ChromaDB + Tavily web)
python scripts/query_gpt_researcher_hybrid.py "What are the new attention mechanisms?"

# Local only (no web search)
python scripts/query_gpt_researcher_hybrid.py --no-web "PINN advances at NeurIPS"

# More results
python scripts/query_gpt_researcher_hybrid.py --max-results 25 "Your query"
```

**Output:**
```
data/vector_store/reports/{timestamp}_{query}/
├── report.md           # Generated research report
├── sources.md          # Human-readable sources
├── sources.json        # Machine-readable sources
└── raw_context.json    # Retrieved chunks
```

---

#### Step 7: Coverage Analysis

Identifies which presentations are underreferenced in analysis documents.

```bash
python scripts/analyze_presentation_coverage.py
```

**Output:** `analysis/coverage_report.md` with:
- Coverage scores per presentation (duration-adjusted)
- Unreferenced presentations prioritized by length
- Underreferenced presentations (coverage < 50%)

---

#### Step 8: Quality Assurance

Validates analysis document formatting and reference quality.

```bash
python scripts/qa_analysis_docs.py
```

**Checks:**
- Source/timestamp format compliance
- Verbatim quote presence
- Cross-references between documents
- Missing sections

---

### Phase 4: Output Generation

#### Step 9: Generate Podcast

Creates audio podcast from topic analysis documents.

```bash
python scripts/generate_podcast.py --topic 09_genai_in_science
```

##### Gemini multi-topic podcast (length control)

Uses Gemini to generate a single episode across topics 01-20 (or a single file via `--input`).

```bash
python scripts/generate_podcast_gemini.py \
  --analysis-dir conferences/neurips2025/analysis \
  --output-dir conferences/neurips2025/audio/neurips2025_topics_01_20_gemini
```

**Length tuning knobs (approximate):**
- `--target-minutes` controls the word count target in the prompt (130 wpm heuristic).
- `--max-output-tokens` caps the model response size; increase if outputs are short.
- `--max-segment-chars` raises the per-segment size limit to allow longer segments.

Example for ~25–30 minutes:

```bash
python scripts/generate_podcast_gemini.py \
  --target-minutes 28 \
  --max-output-tokens 12000 \
  --max-segment-chars 1200
```

---

#### Step 10: Generate Topic Video

Creates video summaries with slide clips.

```bash
python scripts/generate_topic_video.py --topic 02_genai_architectures
```

---

#### Step 11: Export for Knowledge Graph

Exports structured data for knowledge graph construction.

```bash
python scripts/export_for_kg_construction.py
```

---

## Project Structure: `src/` vs `scripts/`

The codebase separates **reusable library code** from **executable scripts**:

```
neurips-foundation-conf-pipeline/
├── src/conference_pipeline/    # LIBRARY: Reusable modules (importable)
│   ├── pipeline.py             # Core orchestration logic
│   ├── transcription.py        # Whisper API client
│   ├── slide_ocr.py            # Vision LLM OCR
│   └── ...                     # Other modules
│
└── scripts/                    # EXECUTABLES: Command-line entry points
    ├── discover_sessions.py    # Run: python scripts/discover_sessions.py
    ├── transcribe_sessions.py  # Run: python scripts/transcribe_sessions.py
    └── ...                     # Other scripts
```

**Why this separation?**

| Aspect | `src/conference_pipeline/` | `scripts/` |
|--------|---------------------------|------------|
| **Purpose** | Reusable library code | Command-line entry points |
| **Usage** | `from conference_pipeline import X` | `python scripts/X.py` |
| **State** | Stateless functions/classes | Handles CLI args, logging, progress |
| **Testing** | Unit testable | Integration/E2E testing |
| **Dependencies** | Minimal, focused | Can import from src/ |

**Benefits:**
1. **Modularity:** Library code can be imported by multiple scripts
2. **Testing:** Core logic is unit-testable without CLI overhead
3. **Flexibility:** Scripts can combine modules in different ways
4. **Clarity:** Clear separation of "what" (library) vs "how to run" (scripts)

---

## Scripts Reference

All scripts are in the `scripts/` directory and run as: `python scripts/<script>.py`

### Data Acquisition Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `discover_sessions.py` | Scrape NeurIPS schedule, output sessions.json | `--year`, `--output` |
| `download_from_sessions.py` | Download audio + slides for chosen sessions | `--limit`, `--retry-errors` |
| `transcribe_sessions.py` | Transcribe audio via Whisper API | `--parallel`, `--retry-errors` |
| `cleanup_transcripts.py` | Detect/fix corrupted transcript segments | `--analyze`, `--fix`, `--dry-run` |
| `ocr_slides.py` | OCR slides via vision LLMs | `--provider`, `--model`, `--parallel` |
| `collect_papers.py` | Collect paper metadata from OpenReview | `--year` |
| `cleanup_slides.py` | Mark duplicate/generic slides to skip OCR | `--threshold` |
| `detect_reference_slides.py` | Detect reference/bibliography slides | - |

#### Transcript Cleanup Details

The `cleanup_transcripts.py` script detects and removes corrupted segments from Whisper transcriptions:

**Common corruption patterns:**
- Music symbols (♪, 🎵) during intro/outro music
- Non-Latin text (Sinhala, Georgian, Russian) from wrong language detection
- Repeated short words ("You", "So") from silence misdetection

**Usage:**
```bash
# Analyze all transcripts
python scripts/cleanup_transcripts.py --analyze

# Analyze specific presentation
python scripts/cleanup_transcripts.py --analyze --presentation machine-learning-and-the-physical-sciences

# Preview fixes (dry run)
python scripts/cleanup_transcripts.py --fix --dry-run

# Apply fixes (creates .backup files)
python scripts/cleanup_transcripts.py --fix

# Restore from backup (if needed)
cd conferences/neurips2025/decks/<presentation-id>/transcripts
cp segments.json.backup segments.json
cp transcript.txt.backup transcript.txt
```

**After fixing transcripts**, rebuild the vector store to use cleaned data:
```bash
python scripts/setup_vector_store_enhanced.py
```

### Indexing Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `setup_vector_store.py` | Create basic ChromaDB index | - |
| `setup_vector_store_enhanced.py` | Create ChromaDB with smart chunking + entity tracking | - |
| `setup_gpt_researcher.py` | Configure GPT Researcher | - |
| `setup_gpt_researcher_with_vectorstore.py` | Configure GPT Researcher with local vector store | - |

### Query Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `query_gpt_researcher_hybrid.py` | Hybrid search (local + web) via GPT Researcher | `--no-web`, `--max-results` |
| `query_gpt_researcher.py` | Basic GPT Researcher query | - |
| `query_vector_store.py` | Direct ChromaDB query | `--top-k` |
| `query_hybrid_bm25.py` | BM25 + vector hybrid search | - |

### Analysis Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `analyze_presentation_coverage.py` | Calculate coverage metrics per presentation | - |
| `qa_analysis_docs.py` | Quality assurance for analysis documents | - |
| `map_presentations_to_topics.py` | Map presentations to topic documents | - |
| `fill_coverage_gaps.py` | Add missing presentation references | - |
| `traceability_recovery.py` | Recover missing source references | - |
| `phase8_paper_analysis.py` | Score and classify poster-only papers | - |

### Output Generation Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `generate_podcast.py` | Generate audio podcast from topic | `--topic` |
| `generate_podcast_gemini.py` | Generate multi-topic podcast with Gemini | `--target-minutes`, `--max-output-tokens`, `--input`, `--analysis-dir`, `--output-dir` |
| `generate_topic_video.py` | Generate video summary with slides | `--topic` |
| `export_for_kg_construction.py` | Export data for knowledge graph | - |
| `sync_to_gdrive.py` | Sync data to Google Drive | - |

---

## Core Pipeline Modules

Located in `src/conference_pipeline/`:

| Module | Purpose |
|--------|---------|
| `cli.py` | Typer CLI entry point with all commands |
| `pipeline.py` | Main NeuripsPipeline orchestrator |
| `session_scraper.py` | Discovers sessions from NeurIPS virtual site |
| `presentation_parser.py` | Parses presentation pages for SlidesLive IDs |
| `downloader.py` | Downloads audio/video from SlidesLive |
| `transcription.py` | Whisper API transcription with chunking |
| `slide_ocr.py` | Vision LLM-based OCR pipeline |
| `slides_extractor.py` | Extracts slide images from PDFs/videos |
| `reference_extractor.py` | Extracts paper references from content |
| `paper_resolver.py` | Resolves references to actual papers |
| `storage.py` | Storage backend for presentations |
| `auth.py` | NeurIPS authentication (HTTP + browser) |
| `config.py` | Configuration and settings |
| `models.py` | Data models (Presentation, Paper, etc.) |

---

## CLI Commands (via `neurips-pipeline`)

```bash
# Discovery
neurips-pipeline discover              # Scrape and build presentation list
neurips-pipeline discover-sessions     # Discover sessions to JSON

# Processing
neurips-pipeline download-assets       # Download audio/video/slides
neurips-pipeline transcribe            # Transcribe audio
neurips-pipeline extract-slides        # Extract slide images from PDFs
neurips-pipeline ocr-slides            # Run OCR on slides
neurips-pipeline extract-refs          # Extract paper references
neurips-pipeline download-papers       # Download referenced papers

# Full pipeline
neurips-pipeline run-all               # Run complete pipeline
neurips-pipeline resume                # Resume from last checkpoint

# Management
neurips-pipeline stats                 # Show pipeline statistics
neurips-pipeline list-presentations    # List presentations by status
neurips-pipeline reset-errors          # Reset errored presentations
```

---

## Data Directory Structure

```
conferences/neurips2025/
├── data/
│   ├── papers.json                    # All conference papers
│   └── presentations/
│       └── {presentation-id}/
│           ├── meta.json              # Presentation metadata
│           ├── audio/
│           │   └── audio.mp3
│           ├── slides/
│           │   └── slide_XXX.png
│           ├── transcripts/
│           │   ├── transcript.txt
│           │   └── segments.json
│           └── ocr/
│               ├── slide_XXX.json
│               └── slide_XXX.md
├── vector_store/
│   ├── chroma_db/                     # ChromaDB persistent store
│   ├── entity_topic_index.json        # Entity tracking
│   └── reports/                       # Query results
└── analysis/
    ├── 01_tools_and_open_source.md
    ├── 02_genai_architectures.md
    ├── ...
    ├── 20_small_language_models.md
    └── NeurIPS_Analysis_Plan.md
```

---

## Status Tracking

Each presentation progresses through statuses:

```
DISCOVERED → ASSETS_DOWNLOADED → TRANSCRIBED → SLIDES_EXTRACTED → OCR_DONE → REFERENCES_EXTRACTED → PAPERS_DOWNLOADED
                                                                                                      ↓
                                                                                                   COMPLETED
```

Status is tracked in:
- `meta.json` per presentation
- `transcription_status.json` (transcription progress)
- `ocr_status.json` (OCR progress)

---

## Resume & Error Handling

All scripts support automatic resume:

1. **Interrupt anytime** - Progress is saved after each item
2. **Re-run the script** - Already completed items are skipped
3. **Retry errors** - Use `--retry-errors` flag

```bash
# Example: Resume transcription after interruption
python scripts/transcribe_sessions.py  # Continues from where it left off

# Retry failed transcriptions
python scripts/transcribe_sessions.py --retry-errors
```

---

## Provider Costs

Use the whole-pipeline cost estimator before approving paid provider work:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/iclr2025 \
  --scope full \
  --scenario base \
  --json
```

Pricing is centralized in `src/conference_pipeline/costs/pricing_catalog.json`
and task assumptions are centralized in
`src/conference_pipeline/costs/task_specs.yaml`. The estimator covers ASR,
OCR, extraction, GPT Researcher, embeddings, Tavily credits, rankings,
podcasts, decks, presenter videos, OpenRouter failover, and `$0 API cost`
local-only work.

For the slide/poster OCR model experiment, start with the dry-run benchmark and
add `--live` only after provider and budget approval:

```bash
.venv/bin/python -m conference_pipeline.ocr_benchmark run \
  --workspace conferences/iclr2025 \
  --sample-size 12 \
  --budget-cap-usd 30 \
  --per-call-timeout-seconds 60 \
  --json
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No SlidesLive embed found" | Session may not have video; check manually |
| Transcription timeout | Use `--parallel 1` for large files |
| OCR rate limits | Reduce `--parallel` and `--max-rpm` |
| ChromaDB not found | Run `setup_vector_store_enhanced.py` first |
| Auth failures | Check NEURIPS_EMAIL/PASSWORD in .env |

---

## Version

- **Pipeline Version:** 0.1.0
- **NeurIPS Year:** 2025
- **Last Updated:** December 2025
