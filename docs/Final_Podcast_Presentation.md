# Final Podcast Presentation Design Document

> Historical NeurIPS-era design note. This file is not a current operator
> runbook. Several scripts named below are now deprecated or rejected as
> reusable production completion owners; use the registry-owned
> [Stage Script Registry Tool Usage Index](reference/stage-owner-audit.md#stage-script-registry-tool-usage-index)
> for current tool instructions.

## Overview

This document outlines the design for creating high-quality, focused podcast presentations from NeurIPS 2025 analysis documents. The goal is to identify and prioritize the most important facts/ideas from each topic, then generate both audio podcasts and visual slide presentations.

## Problem Statement

Current podcast generation produces scripts that cover only 40-75% of source material depending on target length. We need a systematic way to:
1. **Identify the most important facts** in each topic document
2. **Rank ideas by importance** rather than just including them by order of appearance
3. **Generate focused podcasts** that prioritize high-value content
4. **Create companion slide presentations** for visual consumption

## Proposed Approaches

### Approach 1: Telescopic Generation

**Concept**: Generate podcasts at multiple durations (10, 20, 30, 40 min) and rank facts by the stage at which they appear.

```
Duration    Coverage    Interpretation
10 min      ~40%        Tier 1: Essential facts (always included)
20 min      ~55%        Tier 2: Important facts
30 min      ~75%        Tier 3: Supporting details
40 min      ~90%        Tier 4: Comprehensive coverage
```

**Implementation**:

```python
# scripts/rank_facts_telescopic.py
def rank_facts_telescopic(source_path: Path, output_dir: Path):
    """Generate telescopic podcasts and rank facts by inclusion level."""

    durations = [10, 20, 30, 40]
    scripts = {}

    for duration in durations:
        script = generate_podcast_script(
            source_path=source_path,
            target_minutes=duration,
            model="gemini-2.0-flash",
        )
        scripts[duration] = extract_facts(script)

    # Rank facts: Tier 1 = in 10min, Tier 2 = in 20min but not 10min, etc.
    fact_ranks = {}
    for fact in scripts[40]:
        for tier, duration in enumerate([10, 20, 30, 40], start=1):
            if fact in scripts[duration]:
                fact_ranks[fact] = tier
                break

    return fact_ranks
```

**Pros**:
- Simple to implement
- Natural ranking based on model's inherent prioritization
- Single pass per duration level

**Cons**:
- Order of facts in source affects selection
- Model may have biases toward early content
- Only 4 data points per fact

---

### Approach 2: Frequency Analysis with Scrambling

**Concept**: Run 10-minute podcast generation N times with scrambled input order, then score facts by appearance frequency.

```
Run 1 (order A): Facts [1, 3, 5, 7, 9]
Run 2 (order B): Facts [2, 3, 5, 8, 9]
Run 3 (order C): Facts [1, 3, 6, 7, 9]
...
Frequency: Fact 3 = 100%, Fact 9 = 100%, Fact 5 = 67%, ...
```

**Implementation**:

```python
# scripts/rank_facts_frequency.py
import random
from collections import Counter

def scramble_sections(source_text: str) -> str:
    """Scramble major sections while preserving internal structure."""
    sections = split_into_sections(source_text)
    random.shuffle(sections)
    return "\n\n".join(sections)

def rank_facts_frequency(source_path: Path, output_dir: Path, n_runs: int = 10):
    """Run multiple generations with scrambled input and count fact frequency."""

    source_text = source_path.read_text()
    all_facts = Counter()

    for run in range(n_runs):
        scrambled = scramble_sections(source_text)
        script = generate_podcast_script(
            source_text=scrambled,
            target_minutes=10,
            model="gemini-2.0-flash",
        )
        facts = extract_facts(script)
        all_facts.update(facts)

    # Normalize to frequency (0-100%)
    fact_scores = {
        fact: (count / n_runs) * 100
        for fact, count in all_facts.items()
    }

    return sorted(fact_scores.items(), key=lambda x: -x[1])
```

**Pros**:
- Eliminates order bias through scrambling
- Statistical robustness with multiple samples
- Can use confidence intervals

**Cons**:
- Expensive (10x API calls per topic)
- Requires reliable fact extraction/matching
- Scrambling may break context for some facts

---

### Approach 3: Hybrid (Recommended)

**Concept**: Combine both approaches - use frequency analysis for robust ranking, then validate with telescopic generation.

```
Phase 1: Frequency Analysis (5 runs, 10-min each)
         → Preliminary fact scores

Phase 2: Telescopic Validation (10, 20, 30 min)
         → Adjust scores based on inclusion tier

Phase 3: Final Ranking
         → Combined score = 0.6 * frequency + 0.4 * tier_score
```

---

## Slide Generation (Podcast Decks)

We support two slide generation paths:

### Path A: Heuristic (no images)

**Script:** `scripts/generate_slides_from_script.py`

**How it works:**
1. Parse `script.json` into segments.
2. Detect topics, quotes, stats, and key points with regex heuristics.
3. Plan a deck: title → topic intros → key points → (optional) quote/stat → summary.
4. Render Marp markdown and optionally a PDF via `marp` CLI.

**Pros:** fast, deterministic, no external dependencies.  
**Cons:** no paper/talk images, topic detection is heuristic.

### Path B: Agentic (LLM + assets)

**Script:** `scripts/generate_slides_agentic.py`

**How it works:**
1. LLM reads `script.json` + analysis source to identify papers/talks.
2. Paper assets are downloaded and rendered:
   - First page PNG via `pdftoppm`.
   - "Central diagram" via page-selection heuristic (see below).
3. Talk slide images are copied from `conferences/neurips2025/decks/*/slides/images`.
4. LLM generates Marp markdown with embedded images and speaker notes.

**Pros:** rich decks with figures and talk slides.  
**Cons:** relies on LLM quality and external tools (pdftoppm, pdftotext, marp).

### Central Diagram Extraction (Current)

In `generate_slides_agentic.py`, the "central diagram" is **not cropped**:
it selects a likely page (caption keyword heuristic) and renders the full page
as `*_diagram.png`.

### Central Diagram Extraction (Vision LLM Bounding Box - Optional)

It is possible to improve diagram extraction by using a vision LLM to return a
bounding box over the main figure, then cropping locally.

**Workflow:**
1. Render page to PNG at high DPI (e.g., 300).
2. Send the PNG to a vision LLM with strict JSON output:
   ```json
   {"bbox":[x1,y1,x2,y2],"confidence":0.0-1.0}
   ```
3. Clamp bbox to image bounds and crop via PIL or ImageMagick.
4. If confidence is low or bbox is invalid, fall back to full page or caption-based cropping.

**Important note:** Vision LLMs can return bounding boxes when prompted, but
this is **best-effort** and not a guaranteed API feature. Accuracy varies by
model and document layout. Always validate the box and keep a fallback.

---

## Paper-Based Podcast Generation

Generate podcasts from influential papers curated by autonomous agents (Claude and Codex).

### Paper Sources

Papers are sourced from two autonomous analysis pipelines:

```
conferences/neurips2025/agent-work/claude/paper-influence/reports/
conferences/neurips2025/agent-work/codex/paper-influence/reports/
```

Each report contains 20 papers with:
- Title, authors, venue, URL, PDF link
- Citation counts and mention counts
- "Why Selected" explanation
- Key contributions

### Paper Ranking Algorithm

**Goal:** Rank papers by how prominently they are referenced in generated podcast scripts.

**Algorithm:**

```
1. LOAD papers from claude-autonomy and codex-autonomy reports
2. DEDUPLICATE: If paper appears in both sources → weight = 2, else weight = 1
3. DOWNLOAD PDFs and extract content up to "References" section
   (exclude appendices to focus on core contribution)
4. FOR run in 1..N (frequency runs):
   a. Shuffle paper order
   b. Generate podcast script from papers + report content
   c. FOR each paper:
      - Count words in script that reference this paper
      - Add to paper's word_count accumulator
5. SCORE = total_word_count × weight
6. RANK papers by score (descending)
```

**Deduplication Logic:**

Papers are matched by normalized title (lowercase, stripped punctuation). If a paper appears in both claude-autonomy and codex-autonomy reports, it receives 2x weight in scoring, reflecting independent validation of importance.

**Content Extraction:**

Paper content is extracted from PDFs up to the "References" section:
- Pages 1-12 typically contain introduction, method, and results
- Appendices often contain supplementary material less suitable for podcasts
- Uses `pdftotext` with page limits and reference detection

### Implementation

**Script:** `scripts/rank_papers.py`

```python
# scripts/rank_papers.py
"""
Rank influential papers by podcast reference frequency.

Algorithm:
1. Load papers from claude-autonomy and codex-autonomy reports
2. Deduplicate with 2x weighting for papers in both sources
3. Download PDFs and extract content (up to references)
4. Generate N podcast scripts with shuffled paper order
5. Count word-based references per paper in scripts
6. Score = word_count × weight, rank by score
"""

from pathlib import Path
import re
import json

def normalize_title(title: str) -> str:
    """Normalize paper title for deduplication."""
    return re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()

def load_papers_from_report(report_path: Path) -> list[dict]:
    """Parse influential papers report markdown."""
    text = report_path.read_text()
    papers = []

    # Extract paper blocks (## N. Title format)
    pattern = r'## (\d+)\. (.+?)\n\n(.*?)(?=\n## \d+\.|---|\Z)'
    for match in re.finditer(pattern, text, re.DOTALL):
        rank, title, content = match.groups()
        paper = {
            'rank': int(rank),
            'title': title.strip(),
            'source_report': report_path.name,
        }
        # Extract URL
        url_match = re.search(r'\*\*URL:\*\* (.+)', content)
        if url_match:
            paper['url'] = url_match.group(1).strip()
        # Extract PDF
        pdf_match = re.search(r'\*\*PDF:\*\* (.+)', content)
        if pdf_match:
            paper['pdf_url'] = pdf_match.group(1).strip()
        # Extract authors
        authors_match = re.search(r'\*\*Authors:\*\* (.+)', content)
        if authors_match:
            paper['authors'] = authors_match.group(1).strip()
        papers.append(paper)

    return papers

def deduplicate_papers(claude_papers: list, codex_papers: list) -> list[dict]:
    """Merge papers with 2x weight for duplicates."""
    by_title = {}

    for p in claude_papers:
        key = normalize_title(p['title'])
        by_title[key] = {'paper': p, 'sources': ['claude'], 'weight': 1}

    for p in codex_papers:
        key = normalize_title(p['title'])
        if key in by_title:
            by_title[key]['sources'].append('codex')
            by_title[key]['weight'] = 2  # Appears in both
        else:
            by_title[key] = {'paper': p, 'sources': ['codex'], 'weight': 1}

    return [
        {**entry['paper'], 'weight': entry['weight'], 'sources': entry['sources']}
        for entry in by_title.values()
    ]

def extract_paper_content(pdf_path: Path, max_pages: int = 12) -> str:
    """Extract text from PDF up to References section."""
    import subprocess

    # Extract first N pages
    result = subprocess.run(
        ['pdftotext', '-f', '1', '-l', str(max_pages), str(pdf_path), '-'],
        capture_output=True, text=True
    )
    text = result.stdout

    # Truncate at References section
    refs_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
    ]
    for pattern in refs_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break

    return text.strip()

def count_paper_references(script: dict, paper: dict) -> int:
    """Count words in script segments referencing a paper."""
    # Build search patterns from paper title and key terms
    title_words = paper['title'].lower().split()
    significant_words = [w for w in title_words if len(w) > 4]

    word_count = 0
    for segment in script.get('segments', []):
        text = segment.get('text', '').lower()
        # Check if segment references this paper
        if any(w in text for w in significant_words[:3]):
            word_count += len(text.split())

    return word_count
```

### CLI Usage

```bash
# Rank papers for topic 01 with 5 frequency runs
python scripts/rank_papers.py \
  --topic 01_tools_and_open_source \
  --claude-reports conferences/neurips2025/agent-work/claude/paper-influence/reports \
  --codex-reports conferences/neurips2025/agent-work/codex/paper-influence/reports \
  --output-dir conferences/neurips2025/audio/01_tools_and_open_source/paper_ranks \
  --frequency-runs 5 \
  --target-minutes 10

# Generate podcast from ranked papers
python scripts/generate_focused_podcast.py \
  conferences/neurips2025/audio/01_tools_and_open_source/paper_ranks/ranked_papers.json \
  --output-dir conferences/neurips2025/audio/01_tools_and_open_source_papers \
  --target-minutes 15
```

### Output Structure

```
conferences/neurips2025/audio/01_tools_and_open_source/paper_ranks/
├── ranked_papers.json       # Final ranked paper list with scores
├── paper_contents/          # Extracted paper text (up to references)
│   ├── paper_001.txt
│   └── ...
├── frequency_runs/          # Generated scripts per run
│   ├── run_01.json
│   └── ...
└── analysis.md              # Human-readable ranking report
```

### Output Format

**ranked_papers.json:**
```json
{
  "papers": [
    {
      "rank": 1,
      "title": "OpenHands: An Open Platform for AI Software Developers",
      "authors": "Xingyao Wang et al.",
      "weight": 2,
      "sources": ["claude", "codex"],
      "word_count": 847,
      "score": 1694,
      "pdf_url": "https://arxiv.org/pdf/2407.16741"
    }
  ],
  "metadata": {
    "frequency_runs": 5,
    "total_papers": 32,
    "duplicates_found": 8
  }
}
```

---

## Legacy: Extract Influential Papers (Deprecated)

The original approach extracted papers from analysis document tables:

```python
# scripts/extract_influential_papers.py (deprecated)
def extract_papers(source_path: Path) -> list[dict]:
    """Extract influential papers table from source."""
    text = source_path.read_text()

    # Find the Influential Papers section
    match = re.search(r"## Influential Papers.*?\n\|.*?\|.*?\n((?:\|.*?\n)+)", text, re.DOTALL)
    if not match:
        return []

    # Parse markdown table
    papers = []
    for row in match.group(1).strip().split("\n"):
        cols = [c.strip() for c in row.split("|")[1:-1]]
        if len(cols) >= 5:
            papers.append({
                "rank": int(cols[0]),
                "title": cols[1],
                "authors": cols[2],
                "year": cols[3],
                "why_influential": cols[4],
            })

    return papers
```

This approach is superseded by the autonomous agent-based paper curation in `claude-autonomy` and `codex-autonomy` directories.

---

## Slides from Script Agent (Agentic LLM Approach)

### Design Philosophy

The slide generation **must be agentic** — using Claude or Codex to intelligently:
1. Analyze the podcast script and cross-reference with source documents
2. Identify papers and talks mentioned in the script
3. Extract relevant diagrams/figures from those papers
4. Generate rich slides with actual visual assets from the source material

This approach produces professional slides with real paper figures and talk screenshots, not just bullet points.

### Inputs and Outputs

```
Inputs:
  - script.json (podcast transcript)
  - source_analysis.md (analysis document with paper/talk references)
  - conferences/neurips2025/decks/*/slides/images/ (talk slides)
  - Paper PDFs (downloaded on demand)

Outputs:
  - slides/ directory with:
    - slides.md (Marp markdown with embedded images)
    - slides.pdf (rendered presentation)
    - sources.json (mapping from script segments to papers/talks)
    - bullet_points.md (script converted to bullet points)
    - assets/
      - papers/ (downloaded PDFs)
      - paper_images/ (first page + central diagram crops)
      - slides/ (talk slide PNGs)
```

### Architecture (Agentic)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC SLIDE GENERATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐    │
│  │   Script    │───▶│  LLM Analysis   │───▶│   Source Cross-Reference │    │
│  │   + Source  │    │  (Claude/Codex) │    │   • Identify papers      │    │
│  │   Document  │    │                 │    │   • Identify talks       │    │
│  └─────────────┘    │  Prompt:        │    │   • Map segments→sources │    │
│                     │  "Analyze this  │    └───────────┬──────────────┘    │
│                     │   script and    │                │                    │
│                     │   identify all  │                ▼                    │
│                     │   papers/talks" │    ┌──────────────────────────┐    │
│                     └─────────────────┘    │   Asset Extraction       │    │
│                                            │   • Download paper PDFs  │    │
│                                            │   • Extract diagrams     │    │
│                                            │   • Copy talk slides     │    │
│                                            └───────────┬──────────────┘    │
│                                                        │                    │
│                                                        ▼                    │
│                     ┌─────────────────┐    ┌──────────────────────────┐    │
│                     │  LLM Slide      │◀───│   Content Assembly       │    │
│                     │  Generation     │    │   • Script segments      │    │
│                     │  (Claude/Codex) │    │   • Extracted images     │    │
│                     │                 │    │   • Source metadata      │    │
│                     │  Prompt:        │    └──────────────────────────┘    │
│                     │  "Generate Marp │                                     │
│                     │   slides with   │                                     │
│                     │   these assets" │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│                              ▼                                              │
│                     ┌─────────────────┐                                     │
│                     │   slides.md     │──▶ marp-cli ──▶ slides.pdf         │
│                     └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Slide Types

| Type | Content | Layout | Visual Assets |
|------|---------|--------|---------------|
| Title | Episode title, summary | Centered, large text | Optional banner image |
| Topic Intro | Theme name, brief description | Left-aligned | Talk slide screenshot |
| Key Points | 3-5 bullet points | Bulleted list | — |
| Paper Highlight | Paper title, key contribution | Two-column | Paper's central diagram |
| Quote | Direct quote with attribution | Large quote, small source | — |
| Statistic | Single impactful number | Large number, context below | — |
| Architecture | System/method overview | Full-width image | Paper figure (architecture diagram) |
| Comparison | Before/after, benchmark results | Two-column or table | Paper figure (results table) |
| Summary | Key takeaways | Numbered list | — |

### Diagram Extraction Workflow

When extracting figures from papers, use a **central figure heuristic**:

**Definition of "central" figure:**
- Referenced more than once in the paper, OR
- Caption indicates architecture/workflow/overview of the proposed solution
- Usually appears in early pages (1–3) where the method is introduced

**Strong caption keywords (prioritize):**
- overview, architecture, pipeline, workflow, system, construction, schematic

**Weak caption keywords (secondary):**
- framework, design, methodology

**Extraction procedure:**
1. Read early pages (1–12) with `pdftotext` and collect figure captions
2. Score figures using caption keyword hits, mention count, early-page bonus
3. If no strong-keyword figure exists, default to Figure 1
4. Use `pdftotext -bbox` to locate caption, crop region above it
5. Refine with OCR (`tesseract`) if headers/captions remain in crop

```bash
# Example extraction commands
pdftotext -f 1 -l 4 paper.pdf - | rg -n "^Figure|^Fig\\."
pdftoppm -r 300 -f 2 -l 2 -png -singlefile paper.pdf /tmp/paper_page
pdftotext -bbox -f 2 -l 2 paper.pdf /tmp/page_bbox.html
tesseract /tmp/paper_page.png stdout -l eng hocr > /tmp/page.hocr
```

### Implementation

```python
# scripts/generate_slides_agentic.py

"""
Agentic slide generation using Claude or Codex.

This script uses an LLM to:
1. Analyze the podcast script and identify referenced papers/talks
2. Extract relevant diagrams from papers
3. Generate rich Marp slides with embedded images
"""

import anthropic
import openai
from pathlib import Path

def generate_slides_agentic(
    script_path: Path,
    source_path: Path,
    output_dir: Path,
    model: str = "claude-sonnet-5",  # or "gpt-5.6-luna"
) -> Path:
    """Generate slides using an agentic LLM approach."""

    script = json.loads(script_path.read_text())
    source = source_path.read_text()

    # Phase 1: Analyze script and identify sources
    sources_prompt = f"""Analyze this podcast script and identify all papers and talks mentioned.

SCRIPT:
{json.dumps(script, indent=2)}

SOURCE DOCUMENT:
{source[:50000]}

For each mentioned paper/talk, extract:
- title
- authors (if mentioned)
- what aspect is discussed in the script
- which script segment(s) reference it

Output as JSON:
{{"papers": [...], "talks": [...], "segment_mappings": [...]}}
"""

    sources_response = call_llm(sources_prompt, model)
    sources_data = json.loads(sources_response)

    # Phase 2: Extract assets
    assets_dir = output_dir / "assets"
    paper_images = extract_paper_diagrams(sources_data["papers"], assets_dir)
    talk_slides = copy_talk_slides(sources_data["talks"], assets_dir)

    # Phase 3: Generate slides
    slides_prompt = f"""Generate a Marp markdown presentation for this podcast.

SCRIPT:
{json.dumps(script, indent=2)}

AVAILABLE IMAGES:
{json.dumps(paper_images + talk_slides, indent=2)}

REQUIREMENTS:
1. Create one slide per major topic transition
2. Include relevant paper diagrams where the paper is discussed
3. Include talk slides where the talk is referenced
4. Use proper Marp syntax with image embedding
5. Add speaker notes from the script segments

Output the complete Marp markdown file.
"""

    slides_md = call_llm(slides_prompt, model)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slides.md").write_text(slides_md)
    (output_dir / "sources.json").write_text(json.dumps(sources_data, indent=2))

    # Render PDF
    if shutil.which("marp"):
        subprocess.run([
            "marp", str(output_dir / "slides.md"),
            "--pdf", "--allow-local-files",
            "-o", str(output_dir / "slides.pdf")
        ])

    return output_dir / "slides.md"


def call_llm(prompt: str, model: str) -> str:
    """Call Claude or OpenAI based on model name."""
    if model.startswith("claude"):
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def extract_paper_diagrams(papers: list[dict], assets_dir: Path) -> list[dict]:
    """Download papers and extract central diagrams."""
    paper_images = []
    papers_dir = assets_dir / "papers"
    images_dir = assets_dir / "paper_images"
    papers_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        # Download PDF if URL available
        # Extract first page and central diagram
        # ... (implementation using pdftotext, pdftoppm, tesseract)
        pass

    return paper_images


def copy_talk_slides(talks: list[dict], assets_dir: Path) -> list[dict]:
    """Copy relevant talk slides from presentations directory."""
    talk_images = []
    slides_dir = assets_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    # Search in conferences/neurips2025/decks/*/slides/images/
    # ... (implementation)

    return talk_images
```

### Marp Template with Images

```markdown
---
marp: true
theme: default
paginate: true
style: |
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  img { max-height: 400px; }
---

# {title}

{summary}

---

## {topic_title}

![bg right:40%](assets/slides/talk_screenshot.png)

- {point_1}
- {point_2}
- {point_3}

<!--
Speaker notes from script segment
-->

---

## Paper: {paper_title}

<div class="columns">
<div>

**Key Contribution:**
{contribution_summary}

**Why It Matters:**
{significance}

</div>
<div>

![](assets/paper_images/paper_diagram.png)

</div>
</div>

---

<!-- _class: quote -->

> "{quote_text}"

— {attribution}, {paper_or_talk}

---

## Key Takeaways

1. {takeaway_1}
2. {takeaway_2}
3. {takeaway_3}

---
```

### CLI Interface

```bash
# Agentic slide generation with Claude
python scripts/generate_slides_agentic.py \
  --script conferences/neurips2025/audio/01_tools/script.json \
  --source conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --output-dir conferences/neurips2025/audio/01_tools/slides \
  --model claude-sonnet-5 \
  --render-pdf

# With OpenAI
python scripts/generate_slides_agentic.py \
  --script conferences/neurips2025/audio/01_tools/script.json \
  --source conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --output-dir conferences/neurips2025/audio/01_tools/slides \
  --model gpt-4o \
  --render-pdf
```

---

## Full Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FACT RANKING PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Source Doc ──▶ Split Sections ──▶ Scramble ──▶ Generate (N times)      │
│                                         │                                │
│                                         ▼                                │
│                              Extract Facts from Scripts                  │
│                                         │                                │
│                                         ▼                                │
│                              Count Frequency ──▶ Rank Facts              │
│                                                        │                 │
└────────────────────────────────────────────────────────┼─────────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FOCUSED PODCAST GENERATION                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Ranked Facts ──▶ Select Top N ──▶ Generate Script ──▶ Synthesize Audio │
│                                                               │          │
│                     Influential Papers ──────────────────────▶│          │
│                                                               │          │
└───────────────────────────────────────────────────────────────┼──────────┘
                                                                │
                                                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      SLIDE GENERATION PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Script ──▶ Parse Segments ──▶ Plan Slides ──▶ Render Markdown ──▶ PDF  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Interface

```bash
# Step 1: Rank facts in a topic document
python scripts/rank_facts.py \
  --input conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --output-dir conferences/neurips2025/fact_rankings/01_tools \
  --method hybrid \
  --n-runs 5

# Step 2: Generate focused podcast from ranked facts
python scripts/generate_focused_podcast.py \
  --input conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --fact-ranks conferences/neurips2025/fact_rankings/01_tools/ranks.json \
  --output-dir conferences/neurips2025/audio/01_tools_focused \
  --target-minutes 20 \
  --top-n-facts 50

# Step 3: Generate slides from podcast script
python scripts/generate_slides.py \
  --input conferences/neurips2025/audio/01_tools_focused/script.json \
  --output-dir conferences/neurips2025/audio/01_tools_focused/slides \
  --format marp \
  --render-pdf

# All-in-one pipeline
python scripts/podcast_pipeline.py \
  --input conferences/neurips2025/analysis/01_tools_and_open_source.md \
  --output-dir conferences/neurips2025/final_presentations/01_tools \
  --rank-method hybrid \
  --target-minutes 20 \
  --generate-slides \
  --generate-audio
```

---

## Output Structure

```
conferences/neurips2025/final_presentations/01_tools/
├── fact_ranking/
│   ├── ranks.json           # Fact importance scores
│   ├── frequency_runs/      # Individual run outputs (if frequency method)
│   │   ├── run_01.json
│   │   ├── run_02.json
│   │   └── ...
│   └── analysis.md          # Human-readable ranking report
├── podcast/
│   ├── script.json          # Podcast script
│   ├── script.md            # Markdown version
│   ├── show_notes.md        # Show notes with sources
│   ├── audio_segments/      # Individual audio files
│   └── podcast.mp3          # Final merged audio
├── slides/
│   ├── slides.md            # Marp/Reveal markdown
│   ├── slides.pdf           # Rendered PDF
│   └── assets/              # Diagrams, images
└── influential_papers/
    ├── papers.json          # Extracted papers data
    ├── podcast/             # Separate papers podcast
    └── slides/              # Papers-specific slides
```

---

## Implementation Phases

### Phase 1: Fact Ranking (Week 1) ✅ COMPLETE
- [x] Implement fact extraction from scripts
- [x] Implement frequency analysis with scrambling
- [x] Implement telescopic generation
- [x] Create hybrid ranking algorithm
- [x] Add CLI for ranking (`scripts/rank_facts.py`)

### Phase 2: Focused Podcast Generation (Week 2) ✅ COMPLETE
- [x] Modify prompt to accept fact importance hints
- [x] Implement fact-prioritized script generation (`scripts/generate_focused_podcast.py`)
- [ ] Separate influential papers extraction
- [ ] Test with multiple topics

### Phase 3: Slide Generation Agent (Week 3) 🔄 IN PROGRESS
- [x] Basic heuristic slide generator (`scripts/generate_slides_from_script.py`) - fallback
- [ ] **Agentic slide generation with Claude/Codex** (`scripts/generate_slides_agentic.py`)
  - [ ] LLM-based script analysis and source cross-reference
  - [ ] Paper PDF download and diagram extraction
  - [ ] Talk slide screenshot collection
  - [ ] LLM-based Marp slide generation with embedded images
- [ ] PDF generation with marp-cli
- [ ] Integration with podcast pipeline

### Phase 4: Full Pipeline Integration (Week 4)
- [ ] End-to-end pipeline script
- [ ] Batch processing for all 20 topics
- [ ] Quality metrics and validation
- [ ] Documentation and examples

---

## Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Fact coverage | >80% of Tier 1 facts | Automated comparison |
| Audio duration accuracy | ±10% of target | ffprobe duration |
| Slide count | 1 slide per 2 min | Slide count / duration |
| Quote accuracy | 100% exact | String matching |
| PDF rendering | No errors | marp-cli exit code |

---

## Dependencies

```
# Python packages
google-genai>=0.5.0
httpx>=0.25.0
python-dotenv>=1.0.0
tiktoken>=0.5.0  # for token counting

# System tools
ffmpeg          # audio processing
marp-cli        # slide rendering (npm install -g @marp-team/marp-cli)
```

---

## References

- Existing podcast scripts: `conferences/neurips2025/audio/` (ignored workspace data, present only in a checkout that holds the NeurIPS 2025 outputs; see `conferences/neurips2025/DRIVE.md`)
- Example slides: `conferences/neurips2025/audio/03_data_preparation/focused_podcast/slides_from_script/` (same)
- Podcast generation: [scripts/generate_podcast_gemini.py](../scripts/generate_podcast_gemini.py)
- Audio synthesis: [scripts/synthesize_audio_from_script.py](../scripts/synthesize_audio_from_script.py)
