# Getting Started

This guide walks you through installing and configuring the NeurIPS data acquisition pipeline.

## Prerequisites

- Python 3.11 or higher
- FFmpeg (for audio/video processing)
- Tesseract OCR (optional, for local OCR)

### System Dependencies

**macOS:**
```bash
brew install ffmpeg tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg tesseract-ocr
```

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd neurips-foundation-conf-pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### 2. Configure Environment

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Required: NeurIPS credentials
NEURIPS_EMAIL=your-email@example.com
NEURIPS_PASSWORD=your-password

# Optional: API keys for transcription and LLM features
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Recommended for NeurIPS 2025 analysis workflow
TAVILY_API_KEY=tvly-...
STORAGE_ROOT=conferences/neurips2025/data
```

### 3. Verify Installation (Optional)

Smoke-test discovery (no browser):

```bash
python3 scripts/discover_sessions.py --year 2024 --output /tmp/sessions.json --no-browser
```

## Basic Usage

### Discover Presentations

Scrape the NeurIPS schedule to find all presentations:

```bash
python -m conference_pipeline.cli discover
```

This creates a database of presentations with metadata.

### Download Assets

Download videos, audio, and slides for all presentations:

```bash
python -m conference_pipeline.cli download-assets
```

Or for a specific presentation:

```bash
python -m conference_pipeline.cli download-assets --pres-id paper-12345
```

### Run Full Pipeline

Execute all stages in sequence:

```bash
python -m conference_pipeline.cli run-all
```

### Resume After Interruption

If the pipeline is interrupted, resume from where it left off:

```bash
python -m conference_pipeline.cli resume
```

To also retry failed presentations:

```bash
python -m conference_pipeline.cli resume --retry-errors
```

## Pipeline Stages

The pipeline processes presentations through these stages:

1. **DISCOVERED** - Presentation found in schedule
2. **ASSETS_DOWNLOADED** - Video/audio/slides downloaded
3. **TRANSCRIPTION_IN_PROGRESS** - Partial transcription
4. **TRANSCRIBED** - Audio transcribed to text
5. **SLIDES_EXTRACTED** - Slide images extracted
6. **OCR_DONE** - Text extracted from slides
7. **REFERENCES_EXTRACTED** - Paper references identified
8. **PAPERS_DOWNLOADED** - Referenced papers downloaded (open access)

Each presentation tracks its current status, allowing the pipeline to resume from any point.

## Data Output

Downloaded data is organized in this structure (example uses `STORAGE_ROOT=conferences/neurips2025/data`):

```
conferences/neurips2025/data/
├── metadata.sqlite          # SQLite database
├── presentations.json       # JSON export
└── presentations/
    └── {presentation_id}/
        ├── meta.json        # Presentation metadata
        ├── audio/           # Audio files
        ├── video/           # Video files
        ├── slides/          # Slide deck and images
        ├── transcripts/     # Transcription output
        ├── ocr/             # OCR results
        └── papers/          # Downloaded papers
```

## Next Steps

- [Configuration](configuration.md) - Detailed configuration options
- [CLI Reference](cli.md) - All available commands
- [Architecture](architecture.md) - How the pipeline works
- [Data + Analysis Workflow](workflows/pipeline-workflow.md) - End-to-end NeurIPS 2025 runbook

## Source References

- scripts/discover_sessions.py
- src/conference_pipeline/cli.py
- src/conference_pipeline/models.py
- src/conference_pipeline/storage.py
