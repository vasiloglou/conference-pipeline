# Figure Extraction Tool

> Historical/deprecated NeurIPS-era reference. This file is not a current
> operator runbook for reusable presentation production; any references to
> `scripts/generate_slides_agentic.py` are legacy context only and are rejected
> as completion-owner evidence by the Stage Script Registry.

Extract figures from academic papers and generate captions using vision LLMs.

## Overview

This tool automatically:
1. Detects figure regions in PDF papers by finding and clustering embedded images
2. Extracts figures as high-resolution PNGs
3. Generates academic captions using the configured vision model, defaulting to GPT-5.4 mini
4. Optionally creates composite images with captions rendered underneath

## Installation

```bash
pip install pymupdf openai pillow
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

### Command Line

```bash
# Basic usage - extract figures until References section
python -m conference_pipeline.figure_extractor paper.pdf

# Specify output directory
python -m conference_pipeline.figure_extractor paper.pdf -o ./output

# Process entire paper including appendix
python -m conference_pipeline.figure_extractor paper.pdf --no-stop-refs

# Use a different vision model (if explicitly overriding)
python -m conference_pipeline.figure_extractor paper.pdf -m gpt-5.6-luna
```

### Python API

```python
from conference_pipeline.figure_extractor import extract_figures_from_paper

results = extract_figures_from_paper(
    pdf_path="paper.pdf",
    output_dir="./figures",
    model="gpt-5.6-luna",
    stop_at_references=True
)

# Results is a list of dicts with figure info and captions
for fig in results:
    print(f"Page {fig['page']}: {fig['llm_response']['caption']}")
```

## How It Works

### 1. Image Detection
The tool uses PyMuPDF to find all embedded images in each PDF page:
```python
image_list = page.get_images(full=True)
for img in image_list:
    rects = page.get_image_rects(img[0])
```

### 2. Bounding Box Clustering
Nearby image elements are clustered together to handle multi-part figures:
```python
# Images within 30 points of each other are grouped
clusters = cluster_rects(all_rects, threshold=30)
```

### 3. Union of Bounding Boxes
Each cluster is combined into a single bounding box:
```python
x0 = min(r.x0 for r in rects)  # leftmost
y0 = min(r.y0 for r in rects)  # topmost
x1 = max(r.x1 for r in rects)  # rightmost
y1 = max(r.y1 for r in rects)  # bottommost
```

### 4. Figure Rendering
Figures are rendered at 2x resolution for quality:
```python
mat = fitz.Matrix(2, 2)  # 2x scale
pix = page.get_pixmap(matrix=mat, clip=region)
```

### 5. Caption Generation
Each figure is sent to the configured vision model, defaulting to GPT-5.4 mini:
```python
response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]
    }]
)
```

## Output Format

### Directory Structure
```
output_dir/
├── figure_01_page2.png      # Extracted figure images
├── figure_02_page3.png
├── figure_03_page12.png
└── extraction_results.json  # Full results with captions
```

### JSON Schema
```json
[
  {
    "page": 2,
    "figure_index": 1,
    "global_figure_number": 1,
    "bbox": {
      "x0": 73.91,
      "y0": 411.31,
      "x1": 538.09,
      "y1": 582.39,
      "width": 464.18,
      "height": 171.08
    },
    "image_path": "/path/to/figure_01_page2.png",
    "image_filename": "figure_01_page2.png",
    "llm_response": {
      "description": "Detailed description of what's in the image...",
      "caption": "Figure 1: Academic caption for the figure...",
      "figure_type": "graph/plot"
    }
  }
]
```

### Figure Types
The LLM classifies figures into these categories:
- `diagram` - System diagrams, block diagrams
- `flowchart` - Process flows, pipelines
- `graph/plot` - Line graphs, bar charts, scatter plots
- `table` - Tabular data
- `photo_grid` - Image grids, sample visualizations
- `architecture` - Neural network architectures
- `equation` - Mathematical formulas
- `other` - Unclassified

## API Reference

### `extract_figures_from_paper()`

```python
def extract_figures_from_paper(
    pdf_path: str,
    output_dir: str = None,
    model: str = "gpt-5.6-luna",
    min_figure_size: int = 50,
    cluster_threshold: int = 30,
    render_scale: int = 2,
    stop_at_references: bool = True
) -> list:
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | str | required | Path to the PDF file |
| `output_dir` | str | None | Output directory (default: `{pdf_stem}_figures`) |
| `model` | str | "gpt-5.6-luna" | Vision LLM model to use |
| `min_figure_size` | int | 50 | Minimum width/height in points to consider as figure |
| `cluster_threshold` | int | 30 | Distance threshold for clustering image elements |
| `render_scale` | int | 2 | Scale factor for rendering (2 = 144 DPI) |
| `stop_at_references` | bool | True | Stop processing at References section |

**Returns:** List of dictionaries containing figure metadata and LLM responses.

### Helper Functions

```python
# Find where References section starts
find_references_page(doc) -> int

# Get all image bounding boxes from a page
get_image_bboxes(page) -> list[fitz.Rect]

# Cluster nearby rectangles
cluster_rects(rects, threshold=50) -> list[list[fitz.Rect]]

# Take union of rectangles
union_rects(rects) -> fitz.Rect

# Generate caption with vision LLM
generate_caption_with_llm(image_path, client, model) -> dict
```

## Creating Composite Images with Captions

To render captions underneath figures:

```python
from PIL import Image, ImageDraw, ImageFont
import textwrap

def add_caption_to_figure(image_path, caption, output_path):
    img = Image.open(image_path)

    # Calculate space needed for caption
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    wrapped = textwrap.fill(caption, width=80)

    # Create composite image
    padding = 30
    caption_height = 120
    composite = Image.new('RGB',
        (img.width + 2*padding, img.height + caption_height + 3*padding),
        'white')

    # Paste figure and draw caption
    composite.paste(img, (padding, padding))
    draw = ImageDraw.Draw(composite)
    draw.text((padding, img.height + padding + 25), wrapped, fill='black', font=font)

    composite.save(output_path)
```

## Limitations

- Vector-only figures are detected via caption heuristics and can be missed in unusual layouts
- References detection uses simple text matching
- Multi-column layouts may require tuning `cluster_threshold`
- Very small images (< 50 points) are filtered out by default

## Examples

### Extract from Single Paper
```bash
python -m conference_pipeline.figure_extractor paper.pdf -o ./paper_figures
```

### Batch Processing
```python
from pathlib import Path
from conference_pipeline.figure_extractor import extract_figures_from_paper

papers = Path("./papers").glob("*.pdf")
for paper in papers:
    extract_figures_from_paper(paper, output_dir=f"./figures/{paper.stem}")
```

### Custom Post-Processing
```python
results = extract_figures_from_paper("paper.pdf")

# Filter to only graphs/plots
graphs = [r for r in results if r['llm_response']['figure_type'] == 'graph/plot']

# Get figures from specific pages
page_5_figs = [r for r in results if r['page'] == 5]
```

## Integration with Slide Generation

The figure extraction module is integrated into the agentic slide generation pipeline (`scripts/generate_slides_agentic.py`).

### Automatic Integration

When generating slides, figures are automatically extracted from referenced papers:

```bash
python scripts/generate_slides_agentic.py \
    --script podcast/script.json \
    --source analysis/topic.md \
    --output-dir output/slides \
    --use-pymupdf
```

The `--use-pymupdf` flag (enabled by default) uses the bounding box union method for accurate figure extraction. Use `--no-pymupdf` to fall back to the legacy `pdftoppm` method.

### How It Works in the Pipeline

1. **Phase 1**: Script analysis identifies papers and talks mentioned in the podcast
   - Extracts paper titles, arXiv URLs, and key points
   - Extracts presentation IDs and specific slide numbers from source references
   - Source documents contain structured refs like: `[Source: ...; Presentation ID: some-id; Slide: 799; ...]`

2. **Phase 2**: Asset extraction
   - **Papers**: Downloads PDFs from arXiv, extracts figures using PyMuPDF
   - **Talks**: Finds presentations by ID in `--presentations-dir`, copies specific slides mentioned

3. **Phase 3**: LLM generates Marp slides with embedded figures and talk slides

### Configuration Options

```bash
# Use PyMuPDF extraction with vision-based captions
python scripts/generate_slides_agentic.py \
    --script script.json \
    --source source.md \
    --output-dir output \
    --use-pymupdf \
    --vision-crop

# Disable PyMuPDF, use legacy extraction
python scripts/generate_slides_agentic.py \
    --script script.json \
    --source source.md \
    --output-dir output \
    --no-pymupdf
```

### Module Location

The figure extraction module is located at:

```text
src/conference_pipeline/figure_extractor.py
```

Import in your own scripts:
```python
from conference_pipeline.figure_extractor import (
    extract_figures_from_paper,
    extract_figure_from_page,
    get_image_bboxes,
    union_rects,
)
```
