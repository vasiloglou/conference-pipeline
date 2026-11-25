# podcast.py - Podcast Generation

Generate a narrated podcast from a source document using OpenAI chat + TTS.

## Location

`src/conference_pipeline/podcast.py`

## Data Classes

```python
@dataclass
class PodcastSegment:
    speaker: str
    text: str

@dataclass
class PodcastScript:
    title: str
    summary: str
    segments: list[PodcastSegment]
    show_notes: list[str] = field(default_factory=list)

@dataclass
class PodcastConfig:
    title: str
    host_a: str = "Host A"
    host_b: str = "Host B"
    two_hosts: bool = True
    target_minutes: int = 8
    model: str = "gpt-5.6-luna"
    tts_model: str = "gpt-4o-mini-tts"
    voice_a: str = "alloy"
    voice_b: str = "nova"
    max_segment_chars: int = 800
    max_tts_chars: int = 900
    temperature: float = 0.4
    max_source_chars: int = 60000
    include_show_notes: bool = True
    output_dir: Path = Path("podcast_output")
    source_path: Path | None = None
    source_paths: list[Path] = field(default_factory=list)
    openai_base_url: str = "https://api.openai.com/v1"
    intro_path: Path | None = None
    outro_path: Path | None = None
    normalize_audio: bool = True
```

## Classes

### PodcastGenerator

```python
class PodcastGenerator:
    def __init__(self, config: PodcastConfig, api_key: str | None = None) -> None:
        """Initialize generator using OPENAI_API_KEY."""

    def read_source(self, source_path: Path) -> str:
        """Read and truncate source text."""

    def read_sources(self, source_paths: list[Path]) -> str:
        """Read and combine multiple sources."""

    def generate_script(self, source_text: str) -> PodcastScript:
        """Generate structured script via chat completion."""

    def write_outputs(self, script: PodcastScript) -> dict[str, Path]:
        """Write script.json, script.md, show_notes.md."""

    def synthesize_audio(self, script: PodcastScript) -> list[Path]:
        """Generate MP3 segments via TTS."""

    def assemble_audio(self, segment_paths: list[Path]) -> Path | None:
        """Combine segments into podcast.mp3 (ffmpeg required)."""
```

## Output Files

- `script.json`
- `script.md`
- `show_notes.md`
- `audio_segments/segment_###_##.mp3`
- `podcast.mp3` (if ffmpeg available)

## Usage Example

```python
from conference_pipeline.podcast import PodcastConfig, PodcastGenerator
from pathlib import Path

config = PodcastConfig(title="NeurIPS 2025 Highlights")
generator = PodcastGenerator(config)

source_text = generator.read_source(Path("notes.md"))
script = generator.generate_script(source_text)
paths = generator.write_outputs(script)
segments = generator.synthesize_audio(script)
final = generator.assemble_audio(segments)
```

## Source References

- `src/conference_pipeline/podcast.py`
