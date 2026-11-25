# topic_video.py - Topic Video Builder

Generate a narrated slide deck + video from a topic document.

## Location

`src/conference_pipeline/topic_video.py`

## Data Classes

```python
@dataclass
class SlideEvidence:
    source_title: str
    presentation_id: str | None = None
    slide_number: int | None = None
    timestamp: str | None = None
    quote: str | None = None
    section: str | None = None
    image_path: Path | None = None
    transcript_excerpt: str | None = None
    slide_text: str | None = None
    paper_id: str | None = None
    paper_url: str | None = None
    paper_title: str | None = None
    paper_abstract: str | None = None

@dataclass
class Scene:
    scene_type: str
    narration: str
    speaker: str | None = None
    slide_index: int | None = None
    filler_title: str | None = None
    filler_bullets: list[str] = field(default_factory=list)
    on_screen: str | None = None
```

### TopicVideoConfig

```python
@dataclass
class TopicVideoConfig:
    title: str
    model: str = "gpt-5.6-luna"
    tts_model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    voice_b: str = "nova"
    two_voices: bool = False
    output_dir: Path = Path("topic_videos")
    width: int = 1280
    height: int = 720
    max_source_chars: int = 60000
    max_tts_chars: int = 900
    temperature: float = 0.4
    intro_path: Path | None = None
    outro_path: Path | None = None
    music_bed_path: Path | None = None
    normalize_audio: bool = True
```

## Classes

### TopicVideoBuilder

```python
class TopicVideoBuilder:
    def __init__(self, config: TopicVideoConfig, api_key: str | None = None) -> None:
        """Initialize using OPENAI_API_KEY."""

    def parse_topic_document(self, path: Path) -> tuple[str, list[SlideEvidence]]:
        """Parse topic doc into title + evidence list."""

    def map_presentations(self, evidence: list[SlideEvidence], presentations_root: Path) -> None:
        """Resolve presentation IDs, slide images, OCR text, transcript excerpts."""

    def build_scenes(self, title: str, evidence: list[SlideEvidence]) -> list[Scene]:
        """Build narration scenes (LLM or heuristic)."""

    def render_slides(self, evidence: list[SlideEvidence], scenes: list[Scene]) -> list[Path]:
        """Render slides for each scene into output_dir/slides."""

    def synthesize_audio(self, scenes: list[Scene]) -> list[Path]:
        """Generate TTS audio into output_dir/audio."""

    def assemble_video(self, slide_paths: list[Path], audio_paths: list[Path]) -> Path | None:
        """Create topic.mp4/topic_normalized.mp4 from slides + audio."""

    def assemble_deck_pdf(self, slide_paths: list[Path]) -> Path:
        """Export rendered slides to deck.pdf."""

    def write_narration(self, scenes: list[Scene]) -> Path:
        """Write narration.md."""

    def write_scenes_json(self, scenes: list[Scene]) -> Path:
        """Write scenes.json."""
```

## Inputs and Outputs

Inputs expected under `presentations_root/<presentation_id>/`:
- `slides/images/slide_###.{png|jpg}`
- `ocr/slide_###.json` (structured OCR from `slide_ocr.py`)
- `transcripts/segments.json` (timestamped transcript segments)

Outputs under `config.output_dir`:
- `slides/scene_###.png`
- `audio/scene_###.mp3`
- `segments/segment_###.mp4`
- `topic.mp4` or `topic_normalized.mp4`
- `deck.pdf`
- `narration.md`
- `scenes.json`

## Usage Example

```python
from conference_pipeline.topic_video import TopicVideoConfig, TopicVideoBuilder
from pathlib import Path

config = TopicVideoConfig(title="NeurIPS 2025: Foundation Models")
builder = TopicVideoBuilder(config)

_, evidence = builder.parse_topic_document(Path("topic.md"))
builder.map_presentations(evidence, Path("conferences/neurips2025/decks"))
scenes = builder.build_scenes(config.title, evidence)
slides = builder.render_slides(evidence, scenes)
audio = builder.synthesize_audio(scenes)
video = builder.assemble_video(slides, audio)
```

## Source References

- `src/conference_pipeline/topic_video.py`
