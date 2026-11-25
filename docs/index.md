# Conference Intelligence Platform

A Python platform for acquiring, processing, analyzing, and repackaging foundation-conference content across conference workspaces.

## Documentation

### Core

- [Getting Started](getting-started.md) - Installation and configuration
- [Configuration](configuration.md) - Settings and environment variables
- [Architecture](architecture.md) - System design and data flow
- [CLI Reference](cli.md) - Command-line interface
- [Examples](examples.md) - Common tasks and usage
- [API Reference](api/index.md) - Module documentation
- [Workflow Module API](api/workflow.md) - Stage registry, workspace validation, and workflow dispatch helpers
- [Source Of Truth Contract](source-of-truth.md) - Docs ownership and conference context rules
- [Pipeline Capabilities](reference/pipeline-capabilities.md) - Code-grounded reusable capability catalog

### Workflows

- [Canonical Workflow CLI](workflows/pipeline-workflow.md#canonical-workflow-cli) - Named workflow stages and workspace-targeted stage execution
- [Conference Analysis Playbook](workflows/conference-analysis-playbook.md) - Reusable lifecycle for conference-local context and proof
- [Operations Runbooks](operations/index.md) - Public release, private backup, cleanup, and recovery workflows
- [Data + Analysis Workflow](workflows/pipeline-workflow.md) - End-to-end workflow and receipt contract
- [Deep Research Export](workflows/pipeline-workflow.md#phase-1b-deep-research-export-optional) - Package presentations and papers for NotebookLM and similar tools
- NeurIPS 2025 Context (`.planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md`) - Historical pilot context and local analysis entrypoints
- ICLR 2025 Context (`.planning/milestones/v1.2-ICLR-2025-CONTEXT.md`) - v1.2 proof workspace context

### Integrations

- [Gemini Automation](integrations/gemini-automation.md) - Browser automation options
- [GPT Researcher Quick Start](integrations/gpt-researcher/quickstart.md) - Fast setup
- [GPT Researcher Integration Guide](integrations/gpt-researcher/integration-guide.md) - Full setup
- [Vector Store Guide](integrations/gpt-researcher/vector-store-guide.md) - Approaches and tradeoffs

### Extraction

- [Exhaustive Extraction Guide](extraction/exhaustive-extraction-guide.md) - System walkthrough
- [Guardrails System](extraction/guardrails-system.md) - QA and enforcement
- [Workshop Extraction Plan](extraction/workshop-extraction-plan.md) - Talk-aware processing

### Analysis Plans

- [Exhaustive Analysis Plan (v1)](analysis-plans/exhaustive-analysis-plan.md)
- [Exhaustive Analysis Plan (v2)](analysis-plans/exhaustive-analysis-plan-v2.md)
- [Exhaustive Analysis Plan (v2.1 addendum)](analysis-plans/exhaustive-analysis-plan-v2-addendum.md)

### Design

- [Pipeline Design Spec](design/pipeline-design.md)

## Quick Start

For a new conference analysis, start with the
[Conference Analysis Playbook](workflows/conference-analysis-playbook.md). The
normal flow is `$plan-conference-analysis` to create the conference analysis
specification, then `$plan-conference-production` to create live-only production
work from the finalized spec. Do not treat dry-run/mock planning proof as live
production completion.

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Configure credentials
cp .env.example .env
# Edit .env with conference credentials and API keys as needed

# Run the pipeline
python -m conference_pipeline.cli workflow list --workspace conferences/neurips2025
python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages discover,download,transcribe
python -m conference_pipeline.cli conference-smoketest --workspace conferences/iclr2025 --stages analyze,deck --topic 01 --mode dry-run --run-id phase20-iclr2025-smoketest
```

The committed `conferences/neurips2025/` workspace holds the lane-based
analysis, deck, and audio artifacts. Acquisition state such as `sessions.json`,
`metadata.sqlite`, and runtime logs lives under `STORAGE_ROOT` rather than an
obsolete nested `data/` tree under the workspace.

## Features

- **Authentication**: Cookie or provider-specific auth for protected conference content
- **Discovery**: Scrape conference schedule to find all presentations
- **Asset Download**: Download videos, slides, and audio from SlidesLive
- **Transcription**: Convert audio to text using OpenAI, Anthropic, or local Whisper
- **OCR**: Extract text from slide images using Tesseract or LLM-based vision
- **Reference Extraction**: Identify paper references (arXiv, DOI, citations)
- **Paper Download**: Resolve and download referenced papers (open access only)
- **Resumability**: Pipeline can resume from any stage if interrupted

## Project Structure

```
conference-pipeline/
├── src/conference_pipeline/   # Main package
│   ├── auth.py             # NeurIPS authentication
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── downloader.py       # Media downloading
│   ├── models.py           # Data models
│   ├── ocr_engine.py       # OCR engines
│   ├── paper_resolver.py   # Paper resolution & download
│   ├── pipeline.py         # Main orchestration
│   ├── presentation_parser.py  # Parse presentation pages
│   ├── reference_extractor.py  # Extract references
│   ├── schedule_scraper.py # Scrape conference schedule
│   ├── slides_extractor.py # Extract slides from PDFs/video
│   ├── slideslive_client.py # SlidesLive API client
│   ├── storage.py          # Data persistence
│   └── transcription.py    # Audio transcription
├── utils/                  # Shared utilities
│   ├── http.py             # HTTP client with retries
│   ├── logging.py          # Structured logging
│   └── concurrency.py      # Rate limiting
├── prompts/                # LLM prompts
│   ├── ocr.py
│   ├── transcription.py
│   └── reference_normalization.py
├── tests/                  # Test suite
├── docs/                   # Platform documentation
└── conferences/            # Conference-local workspaces and context
    ├── neurips2025/
    └── iclr2025/
```

## License

This project is for educational and research purposes. Users must comply with the Terms of Service of NeurIPS, SlidesLive, and all content providers.

## Source References

- docs/source-of-truth.md
- docs/reference/pipeline-capabilities.md
- docs/workflows/pipeline-workflow.md
- .planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md
- .planning/milestones/v1.2-ICLR-2025-CONTEXT.md
- src/conference_pipeline/cli.py
