"""Generate a narrated slide deck and video from a topic document."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

from conference_pipeline.config import load_settings
from conference_pipeline.model_governance import default_model, validate_selection
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = default_model("topic_video.scene_generation")
DEFAULT_TTS_MODEL = default_model("podcast.tts")
DEFAULT_VOICE = "alloy"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


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


@dataclass
class TopicVideoConfig:
    title: str
    model: str = DEFAULT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_VOICE
    voice_b: str = "nova"
    two_voices: bool = False
    output_dir: Path = field(default_factory=lambda: Path("topic_videos"))
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    max_source_chars: int = 60000
    max_tts_chars: int = 900
    temperature: float = 0.4
    intro_path: Path | None = None
    outro_path: Path | None = None
    music_bed_path: Path | None = None
    music_bed_gain: float = 0.18
    normalize_audio: bool = True
    loudnorm_i: str = "-16"
    loudnorm_tp: str = "-1.5"
    loudnorm_lra: str = "11"
    openai_base_url: str = "https://api.openai.com/v1"
    prefer_playwright: bool = True
    use_llm_scenes: bool = False
    context_slide_radius: int = 1
    transcript_window_sec: int = 18


class TopicVideoError(Exception):
    """Raised when topic video generation fails."""


class OpenAIClient:
    """Minimal OpenAI client via HTTPX."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120)

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def tts(self, model: str, voice: str, text: str) -> bytes:
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "mp3",
        }
        response = self.client.post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.content


class TopicVideoBuilder:
    """Build slides + narration + video for a topic document."""

    def __init__(self, config: TopicVideoConfig, api_key: str | None = None) -> None:
        settings = load_settings()
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise TopicVideoError("OPENAI_API_KEY not configured.")
        validate_selection("topic_video.scene_generation", provider="openai", model=config.model)
        validate_selection("podcast.tts", provider="openai", model=config.tts_model)
        self.config = config
        self.client = OpenAIClient(api_key=api_key, base_url=config.openai_base_url)
        self._presentation_index: dict[str, str] | None = None
        self._presentation_title_cache: dict[str, str] = {}
        self._slide_sync_cache: dict[str, list[dict[str, Any]]] = {}
        self._ocr_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._transcript_cache: dict[str, list[dict[str, Any]]] = {}
        self._paper_index: dict[str, dict[str, Any]] | None = None

    def parse_topic_document(self, path: Path) -> tuple[str, list[SlideEvidence]]:
        text = path.read_text(encoding="utf-8")
        title = self._extract_title(text) or path.stem.replace("_", " ").title()

        evidence: list[SlideEvidence] = []
        current_section: str | None = None
        current_source: str | None = None
        current_presentation_id: str | None = None
        current_slide: int | None = None
        current_timestamp: str | None = None
        current_paper_id: str | None = None
        current_paper_url: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
            elif line.startswith("### "):
                current_section = line[4:].strip()

            if not line:
                continue

            source_title = self._match_source(line)
            if source_title:
                current_source = source_title
                current_presentation_id = None
                current_slide = None
                current_timestamp = None
                current_paper_id = None
                current_paper_url = None
                continue

            presentation_id = self._match_presentation_id(line)
            if presentation_id:
                current_presentation_id = presentation_id

            slide_number = self._match_slide_number(line)
            if slide_number is not None:
                current_slide = slide_number

            timestamp = self._match_timestamp(line)
            if timestamp:
                current_timestamp = timestamp

            paper_id = self._match_paper_id(line)
            if paper_id:
                current_paper_id = paper_id

            paper_url = self._match_paper_url(line)
            if paper_url:
                current_paper_url = paper_url

            quote, inline = self._parse_quote_line(line)
            if not quote:
                continue

            source_title = inline.get("source_title") or current_source or "Unknown Source"
            presentation_id = inline.get("presentation_id") or current_presentation_id
            slide_number = inline.get("slide_number") or current_slide
            timestamp = inline.get("timestamp") or current_timestamp
            paper_id = inline.get("paper_id") or current_paper_id
            paper_url = inline.get("paper_url") or current_paper_url

            evidence.append(
                SlideEvidence(
                    source_title=source_title,
                    presentation_id=presentation_id,
                    slide_number=slide_number,
                    timestamp=timestamp,
                    quote=quote,
                    section=current_section,
                    paper_id=paper_id,
                    paper_url=paper_url,
                )
            )

            if inline.get("source_title"):
                current_source = inline["source_title"]
            if inline.get("presentation_id"):
                current_presentation_id = inline["presentation_id"]
            if inline.get("slide_number") is not None:
                current_slide = inline["slide_number"]
            if inline.get("timestamp"):
                current_timestamp = inline["timestamp"]
            if inline.get("paper_id"):
                current_paper_id = inline["paper_id"]
            if inline.get("paper_url"):
                current_paper_url = inline["paper_url"]

        return title, evidence

    def map_presentations(self, evidence: list[SlideEvidence], presentations_root: Path) -> None:
        if self._presentation_index is None:
            self._presentation_index = self._build_presentation_index(presentations_root)
        paper_index = self._load_paper_index()

        for item in evidence:
            self._enrich_evidence_item(item, presentations_root, self._presentation_index, paper_index)

        if self.config.context_slide_radius > 0:
            expanded = self._expand_context_slides(evidence, presentations_root, paper_index)
            if expanded is not evidence:
                evidence[:] = expanded

    def build_scenes(self, title: str, evidence: list[SlideEvidence]) -> list[Scene]:
        if self.config.use_llm_scenes:
            prompt = self._build_scene_prompt(title, evidence)
            messages = [
                {"role": "system", "content": "You are a concise video narrator and editor."},
                {"role": "user", "content": prompt},
            ]
            raw = self.client.chat(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
            )
            payload = self._parse_json_payload(raw)
            scenes = self._coerce_scenes(payload, len(evidence))
            if scenes:
                return scenes
        return self._fallback_scenes(evidence)

    def render_slides(
        self, evidence: list[SlideEvidence], scenes: list[Scene]
    ) -> list[Path]:
        slides_dir = self.config.output_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)

        rendered: list[Path] = []
        for idx, scene in enumerate(scenes, start=1):
            slide_path = slides_dir / f"scene_{idx:03d}.png"
            if scene.scene_type == "slide" and scene.slide_index is not None:
                evidence_item = evidence[scene.slide_index]
                if not evidence_item.image_path:
                    title, bullets = self._filler_content_from_evidence(evidence_item, scene)
                    self._render_filler_slide(slide_path, title, bullets)
                else:
                    self._render_annotated_slide(
                        evidence_item.image_path,
                        slide_path,
                        scene.on_screen or self._scene_on_screen(evidence_item) or "",
                    )
            else:
                self._render_filler_slide(
                    slide_path,
                    scene.filler_title or "Transition",
                    scene.filler_bullets or ["Key takeaway", "Supporting detail", "Next idea"],
                )
            rendered.append(slide_path)
        return rendered

    def synthesize_audio(self, scenes: list[Scene]) -> list[Path]:
        audio_dir = self.config.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for idx, scene in enumerate(scenes, start=1):
            chunks = self._split_text(scene.narration, self.config.max_tts_chars)
            chunk_paths: list[Path] = []
            for chunk_idx, chunk in enumerate(chunks, start=1):
                filename = f"scene_{idx:03d}_{chunk_idx:02d}.mp3"
                path = audio_dir / filename
                voice = self._voice_for_scene(scene, idx)
                audio = self.client.tts(
                    model=self.config.tts_model,
                    voice=voice,
                    text=chunk,
                )
                path.write_bytes(audio)
                chunk_paths.append(path)

            if len(chunk_paths) == 1:
                outputs.append(chunk_paths[0])
            else:
                merged = self._merge_audio_chunks(chunk_paths, audio_dir / f"scene_{idx:03d}.mp3")
                outputs.append(merged)
        return outputs

    def assemble_video(self, slide_paths: list[Path], audio_paths: list[Path]) -> Path | None:
        if not shutil.which("ffmpeg"):
            raise TopicVideoError("ffmpeg is required to assemble the video.")

        if len(slide_paths) != len(audio_paths):
            raise TopicVideoError("Slide count does not match audio count.")

        segments_dir = self.config.output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        segment_paths: list[Path] = []

        if self.config.intro_path:
            intro_slide = segments_dir / "intro_slide.png"
            self._render_filler_slide(intro_slide, "Intro", ["NeurIPS 2025", self.config.title])  # conference-literal-ok: legacy per-topic video generator, not on the multi-conference path
            segment_paths.append(
                self._build_static_segment(
                    intro_slide,
                    self.config.intro_path,
                    segments_dir / "segment_intro.mp4",
                )
            )
        for idx, (slide_path, audio_path) in enumerate(zip(slide_paths, audio_paths), start=1):
            duration = self._probe_duration(audio_path)
            segment_path = segments_dir / f"segment_{idx:03d}.mp4"
            audio_input = audio_path
            if self.config.music_bed_path:
                audio_input = self._mix_music_bed(
                    audio_path,
                    self.config.music_bed_path,
                    duration,
                    segments_dir / f"segment_{idx:03d}_mixed.m4a",
                )
            cmd = [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(slide_path),
                "-i",
                str(audio_input),
                "-t",
                f"{duration:.3f}",
                "-vf",
                f"scale={self.config.width}:{self.config.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.config.width}:{self.config.height}:(ow-iw)/2:(oh-ih)/2",
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-c:a",
                "aac",
                "-shortest",
                str(segment_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            segment_paths.append(segment_path)

        if self.config.outro_path:
            outro_slide = segments_dir / "outro_slide.png"
            self._render_filler_slide(outro_slide, "Outro", ["Thanks for listening"])
            segment_paths.append(
                self._build_static_segment(
                    outro_slide,
                    self.config.outro_path,
                    segments_dir / "segment_outro.mp4",
                )
            )

        return self._concat_segments(segment_paths)

    def assemble_deck_pdf(self, slide_paths: list[Path]) -> Path:
        output_path = self.config.output_dir / "deck.pdf"
        images = [Image.open(path).convert("RGB") for path in slide_paths]
        if not images:
            raise TopicVideoError("No slides available for PDF.")
        images[0].save(output_path, save_all=True, append_images=images[1:])
        return output_path

    def write_narration(self, scenes: list[Scene]) -> Path:
        narration_path = self.config.output_dir / "narration.md"
        lines = ["# Narration", ""]
        for idx, scene in enumerate(scenes, start=1):
            lines.append(f"## Scene {idx}")
            lines.append(scene.narration)
            lines.append("")
        narration_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return narration_path

    def write_scenes_json(self, scenes: list[Scene]) -> Path:
        path = self.config.output_dir / "scenes.json"
        payload = [
            {
                "scene_type": scene.scene_type,
                "slide_index": scene.slide_index,
                "narration": scene.narration,
                "speaker": scene.speaker,
                "filler_title": scene.filler_title,
                "filler_bullets": scene.filler_bullets,
                "on_screen": scene.on_screen,
            }
            for scene in scenes
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _extract_title(self, text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _match_source(self, line: str) -> str | None:
        if re.match(r"^\*\*Source Details:\*\*\s*$", line, re.IGNORECASE):
            return None
        patterns = [
            r'^\*\*Source:\*\*\s*"?(.+?)"?$',
            r'^Source:\s*"?(.+?)"?$',
        ]
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match and match.groups():
                return match.group(1).strip()
        if line.lower().startswith("presentation:"):
            return line.split(":", 1)[1].strip().strip("\"")
        return None

    def _match_slide_number(self, line: str) -> int | None:
        match = re.search(r"Slide\s*[:#]\s*(\d+)", line, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _match_timestamp(self, line: str) -> str | None:
        match = re.search(r"Timestamp\s*[:#]\s*([^|]+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _match_presentation_id(self, line: str) -> str | None:
        match = re.search(r"Presentation ID\s*[:#]\s*([^|\]]+)", line, re.IGNORECASE)
        if match:
            return self._clean_reference_value(match.group(1))
        return None

    def _match_paper_id(self, line: str) -> str | None:
        match = re.search(r"Paper ID\s*[:#]\s*([^|\]]+)", line, re.IGNORECASE)
        if match:
            return self._clean_reference_value(match.group(1))
        return None

    def _match_paper_url(self, line: str) -> str | None:
        match = re.search(r"URL\s*[:#]\s*(https?://[^\s\]]+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _clean_reference_value(self, value: str) -> str:
        return value.strip().strip('"').strip()

    def _parse_quote_line(self, line: str) -> tuple[str | None, dict[str, Any]]:
        inline = {}
        ref_match = re.search(r"\[([^\]]*Source:[^\]]+)\]", line)
        if not ref_match:
            ref_match = re.search(r"\[([^\]]*(?:Presentation ID|Paper ID):[^\]]+)\]", line)
        if ref_match:
            inline = self._parse_inline_reference(ref_match.group(1))
            line = (line[: ref_match.start()] + line[ref_match.end() :]).rstrip()

        is_blockquote = line.lstrip().startswith(">")
        if not is_blockquote and not inline:
            return None, {}

        content = line.lstrip("> ").strip()
        if not content and inline:
            return None, inline

        quote = self._extract_quote_text(content)
        if not quote:
            return None, inline
        return quote, inline

    def _extract_quote_text(self, content: str) -> str | None:
        normalized = self._strip_markdown_wrapping(content)
        matches = re.findall(r"\"([^\"]+)\"", normalized)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"“([^”]+)”", normalized)
        if matches:
            return matches[-1].strip()
        return normalized.strip() or None

    def _strip_markdown_wrapping(self, text: str) -> str:
        stripped = text.strip()
        while stripped.startswith(("*", "_")) and stripped.endswith(("*", "_")):
            stripped = stripped.strip("*_").strip()
        return stripped

    def _parse_inline_reference(self, text: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        parts = [part.strip() for part in text.split(";") if part.strip()]
        for part in parts:
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = self._clean_reference_value(value)
            if not value:
                continue
            if key == "source":
                data["source_title"] = value
            elif key == "presentation id":
                data["presentation_id"] = value
            elif key == "slide":
                if value.isdigit():
                    data["slide_number"] = int(value)
                else:
                    match = re.search(r"\d+", value)
                    if match:
                        data["slide_number"] = int(match.group(0))
            elif key == "timestamp":
                data["timestamp"] = value
            elif key == "paper id":
                data["paper_id"] = value
            elif key in {"url", "paper url"}:
                data["paper_url"] = value
        return data

    def _build_presentation_index(self, presentations_root: Path) -> dict[str, str]:
        index: dict[str, str] = {}
        for meta_path in presentations_root.glob("*/meta.json"):
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            title = str(data.get("title", "")).strip()
            if title:
                presentation_id = meta_path.parent.name
                index[title.lower()] = presentation_id
                self._presentation_title_cache.setdefault(presentation_id, title)
        return index

    def _resolve_presentation_id(self, source_title: str, index: dict[str, str]) -> str | None:
        key = source_title.lower()
        if key in index:
            return index[key]
        # Partial match fallback
        for title, pid in index.items():
            if key in title or title in key:
                return pid
        return None

    def _load_presentation_title(self, presentations_root: Path, presentation_id: str) -> str | None:
        if presentation_id in self._presentation_title_cache:
            return self._presentation_title_cache[presentation_id]
        meta_path = presentations_root / presentation_id / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        title = str(data.get("title", "")).strip()
        if title:
            self._presentation_title_cache[presentation_id] = title
            return title
        return None

    def _load_paper_index(self) -> dict[str, dict[str, Any]]:
        if self._paper_index is not None:
            return self._paper_index
        papers_path = Path("conferences/neurips2025/data/papers.json")
        if not papers_path.exists():
            self._paper_index = {}
            return self._paper_index
        data = json.loads(papers_path.read_text(encoding="utf-8"))
        papers = data.get("papers", [])
        index: dict[str, dict[str, Any]] = {}
        for paper in papers:
            paper_id = str(paper.get("id", "")).strip()
            if paper_id:
                index[paper_id] = paper
        self._paper_index = index
        return index

    def _enrich_evidence_item(
        self,
        item: SlideEvidence,
        presentations_root: Path,
        index: dict[str, str],
        paper_index: dict[str, dict[str, Any]],
    ) -> None:
        if not item.presentation_id and item.source_title:
            item.presentation_id = self._resolve_presentation_id(item.source_title, index)
        if not item.source_title and item.presentation_id:
            item.source_title = (
                self._load_presentation_title(presentations_root, item.presentation_id)
                or item.presentation_id
            )

        if item.presentation_id is None and item.paper_id:
            self._attach_paper_metadata(item, paper_index)
            return

        if item.presentation_id:
            if item.source_title == "Unknown Source":
                item.source_title = (
                    self._load_presentation_title(presentations_root, item.presentation_id)
                    or item.presentation_id
                )
            slides = self._load_slide_sync(presentations_root, item.presentation_id)

            timestamp_sec = self._parse_timestamp_to_seconds(item.timestamp)
            if item.slide_number is None and timestamp_sec is not None:
                item.slide_number = self._find_slide_by_timestamp(slides, timestamp_sec)

            if item.slide_number is not None:
                if item.timestamp is None:
                    slide_timestamp = self._slide_timestamp_for_number(slides, item.slide_number)
                    if slide_timestamp is not None:
                        item.timestamp = self._format_timestamp(slide_timestamp)
                        timestamp_sec = slide_timestamp
                item.image_path = self._find_slide_image(
                    presentations_root, item.presentation_id, item.slide_number
                )
                if item.image_path is None:
                    logger.warning(
                        "Slide image not found",
                        extra={"presentation_id": item.presentation_id, "slide": item.slide_number},
                    )
                item.slide_text = self._load_ocr_slide_text(
                    presentations_root, item.presentation_id, item.slide_number
                )

            if timestamp_sec is None and item.slide_number is not None:
                timestamp_sec = self._slide_timestamp_for_number(slides, item.slide_number)

            if timestamp_sec is not None:
                segments = self._load_transcript_segments(presentations_root, item.presentation_id)
                item.transcript_excerpt = self._extract_transcript_excerpt(
                    segments, timestamp_sec, self.config.transcript_window_sec
                )

        if item.paper_id:
            self._attach_paper_metadata(item, paper_index)

    def _attach_paper_metadata(
        self, item: SlideEvidence, paper_index: dict[str, dict[str, Any]]
    ) -> None:
        if not item.paper_id:
            return
        data = paper_index.get(item.paper_id)
        if not data:
            logger.warning("Paper not found for id", extra={"paper_id": item.paper_id})
            return
        item.paper_title = str(data.get("title", "")).strip() or item.paper_title
        item.paper_abstract = str(data.get("abstract", "")).strip() or item.paper_abstract
        if not item.paper_url:
            item.paper_url = data.get("openreview_url") or data.get("pdf_url")

    def _expand_context_slides(
        self,
        evidence: list[SlideEvidence],
        presentations_root: Path,
        paper_index: dict[str, dict[str, Any]],
    ) -> list[SlideEvidence]:
        if self.config.context_slide_radius <= 0:
            return evidence
        expanded: list[SlideEvidence] = []
        seen_context: set[tuple[str, int]] = set()
        slide_indices_cache: dict[str, set[int]] = {}

        for item in evidence:
            if item.presentation_id and item.slide_number is not None:
                if item.presentation_id not in slide_indices_cache:
                    slides = self._load_slide_sync(presentations_root, item.presentation_id)
                    slide_indices_cache[item.presentation_id] = {
                        s.get("index") for s in slides if s.get("index") is not None
                    }
                slide_indices = slide_indices_cache[item.presentation_id]
                start = item.slide_number - self.config.context_slide_radius
                end = item.slide_number + self.config.context_slide_radius
                for slide_number in range(start, item.slide_number):
                    if slide_number in slide_indices:
                        key = (item.presentation_id, slide_number)
                        if key not in seen_context:
                            context_item = SlideEvidence(
                                source_title=item.source_title or "Unknown Source",
                                presentation_id=item.presentation_id,
                                slide_number=slide_number,
                                section=item.section,
                                paper_id=item.paper_id,
                                paper_url=item.paper_url,
                            )
                            self._enrich_evidence_item(
                                context_item, presentations_root, self._presentation_index or {}, paper_index
                            )
                            expanded.append(context_item)
                            seen_context.add(key)

            expanded.append(item)

            if item.presentation_id and item.slide_number is not None:
                slide_indices = slide_indices_cache.get(item.presentation_id, set())
                end = item.slide_number + self.config.context_slide_radius
                for slide_number in range(item.slide_number + 1, end + 1):
                    if slide_number in slide_indices:
                        key = (item.presentation_id, slide_number)
                        if key not in seen_context:
                            context_item = SlideEvidence(
                                source_title=item.source_title or "Unknown Source",
                                presentation_id=item.presentation_id,
                                slide_number=slide_number,
                                section=item.section,
                                paper_id=item.paper_id,
                                paper_url=item.paper_url,
                            )
                            self._enrich_evidence_item(
                                context_item, presentations_root, self._presentation_index or {}, paper_index
                            )
                            expanded.append(context_item)
                            seen_context.add(key)

        return expanded

    def _load_slide_sync(self, presentations_root: Path, presentation_id: str) -> list[dict[str, Any]]:
        if presentation_id in self._slide_sync_cache:
            return self._slide_sync_cache[presentation_id]
        sync_path = presentations_root / presentation_id / "slides" / "sync.json"
        if not sync_path.exists():
            self._slide_sync_cache[presentation_id] = []
            return self._slide_sync_cache[presentation_id]
        data = json.loads(sync_path.read_text(encoding="utf-8"))
        slides = data.get("slides", [])
        self._slide_sync_cache[presentation_id] = slides
        return slides

    def _slide_timestamp_for_number(self, slides: list[dict[str, Any]], slide_number: int) -> float | None:
        for slide in slides:
            if slide.get("index") == slide_number:
                return slide.get("timestamp_sec")
        return None

    def _find_slide_by_timestamp(
        self, slides: list[dict[str, Any]], timestamp_sec: float
    ) -> int | None:
        candidates = [s for s in slides if s.get("timestamp_sec") is not None]
        if not candidates:
            return None
        before = [s for s in candidates if s["timestamp_sec"] <= timestamp_sec]
        if before:
            best = max(before, key=lambda s: s["timestamp_sec"])
            return best.get("index")
        closest = min(candidates, key=lambda s: abs(s["timestamp_sec"] - timestamp_sec))
        return closest.get("index")

    def _parse_timestamp_to_seconds(self, timestamp: str | None) -> float | None:
        if not timestamp:
            return None
        raw = timestamp.strip()
        if not raw:
            return None
        raw = re.split(r"\s*[-–]\s*", raw)[0].strip()
        parts = [p for p in raw.split(":") if p != ""]
        if not parts or any(not p.isdigit() for p in parts):
            return None
        values = [int(p) for p in parts]
        if len(values) == 2:
            minutes, seconds = values
            return minutes * 60 + seconds
        if len(values) == 3:
            hours, minutes, seconds = values
            return hours * 3600 + minutes * 60 + seconds
        return None

    def _format_timestamp(self, timestamp_sec: float) -> str:
        total = int(timestamp_sec)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _load_ocr_slide_text(
        self, presentations_root: Path, presentation_id: str, slide_number: int
    ) -> str | None:
        ocr_cache = self._ocr_cache.setdefault(presentation_id, {})
        if slide_number in ocr_cache:
            return self._format_slide_text(ocr_cache[slide_number])

        ocr_dir = presentations_root / presentation_id / "ocr"
        if not ocr_dir.exists():
            return None
        patterns = [
            ocr_dir / f"slide_{slide_number:03d}.json",
            ocr_dir / f"slide_{slide_number:04d}.json",
            ocr_dir / f"slide_{slide_number}.json",
        ]
        for path in patterns:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                ocr_cache[slide_number] = data
                return self._format_slide_text(data)
        return None

    def _format_slide_text(self, data: dict[str, Any]) -> str | None:
        slide_text = data.get("slide_text") or {}
        parts: list[str] = []
        title = slide_text.get("title")
        if title:
            parts.append(str(title).strip())
        bullet_points = slide_text.get("bullet_points") or []
        for bullet in bullet_points:
            if bullet:
                parts.append(str(bullet).strip())
        other_text = slide_text.get("other_text")
        if other_text:
            if isinstance(other_text, list):
                for entry in other_text:
                    if entry:
                        parts.append(str(entry).strip())
            else:
                for line in str(other_text).splitlines():
                    if line.strip():
                        parts.append(line.strip())
        return "\n".join(parts).strip() or None

    def _load_transcript_segments(
        self, presentations_root: Path, presentation_id: str
    ) -> list[dict[str, Any]]:
        if presentation_id in self._transcript_cache:
            return self._transcript_cache[presentation_id]
        transcript_path = presentations_root / presentation_id / "transcripts" / "segments.json"
        if not transcript_path.exists():
            self._transcript_cache[presentation_id] = []
            return self._transcript_cache[presentation_id]
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        self._transcript_cache[presentation_id] = data
        return data

    def _extract_transcript_excerpt(
        self, segments: list[dict[str, Any]], timestamp_sec: float, window_sec: int
    ) -> str | None:
        if not segments:
            return None
        half_window = window_sec / 2
        start_time = max(0, timestamp_sec - half_window)
        end_time = timestamp_sec + half_window
        snippets: list[str] = []
        for segment in segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            if seg_end >= start_time and seg_start <= end_time:
                text = segment.get("text", "").strip()
                if text:
                    snippets.append(text)
        if not snippets:
            return None
        return " ".join(snippets).strip()

    def _find_slide_image(
        self, presentations_root: Path, presentation_id: str, slide_number: int
    ) -> Path | None:
        base = presentations_root / presentation_id / "slides" / "images"
        if not base.exists():
            return None
        patterns = [
            base / f"slide_{slide_number:03d}.png",
            base / f"slide_{slide_number:04d}.png",
            base / f"slide_{slide_number:03d}.jpg",
            base / f"slide_{slide_number:04d}.jpg",
        ]
        for candidate in patterns:
            if candidate.exists():
                return candidate
        for candidate in base.glob(f"slide_{slide_number}.*"):
            return candidate
        return None

    def _build_scene_prompt(self, title: str, evidence: list[SlideEvidence]) -> str:
        items = []
        for idx, item in enumerate(evidence):
            items.append(
                {
                    "slide_index": idx,
                    "section": item.section,
                    "source_title": item.source_title,
                    "presentation_id": item.presentation_id,
                    "slide_number": item.slide_number,
                    "timestamp": item.timestamp,
                    "quote": self._clip_text(item.quote or "", 240),
                    "transcript_excerpt": self._clip_text(item.transcript_excerpt or "", 240),
                    "slide_text": self._clip_text(item.slide_text or "", 240),
                    "paper_title": item.paper_title,
                    "paper_id": item.paper_id,
                }
            )
        payload = json.dumps(items, indent=2)
        if len(payload) > self.config.max_source_chars:
            trimmed: list[dict[str, Any]] = []
            total = 0
            for item in items:
                item_json = json.dumps(item)
                if total + len(item_json) > self.config.max_source_chars:
                    break
                trimmed.append(item)
                total += len(item_json)
            items = trimmed
        voice_rule = "Use a single speaker name." if not self.config.two_voices else "Alternate speakers: Host A, Host B."
        return f"""Build a narrated scene plan for a video on: {title}.

Use the evidence list below. Only reference slides by their slide_index.
Provide short, smooth narration and add filler scenes between major sections.
{voice_rule}
Output JSON only with schema:
{{
  "scenes": [
    {{
      "scene_type": "slide" | "filler",
      "slide_index": 0,
      "narration": "...",
      "speaker": "Host A",
      "on_screen": "...",
      "filler_title": "...",
      "filler_bullets": ["...", "..."]
    }}
  ]
}}

Evidence:
{json.dumps(items, indent=2)}
"""

    def _parse_json_payload(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"{.*}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise TopicVideoError("Failed to parse JSON response from model.")

    def _coerce_scenes(self, payload: dict[str, Any], evidence_count: int) -> list[Scene]:
        scenes_raw = payload.get("scenes", [])
        scenes: list[Scene] = []
        for raw in scenes_raw:
            scene_type = str(raw.get("scene_type", "slide")).strip().lower()
            narration = str(raw.get("narration", "")).strip()
            if not narration:
                continue
            slide_index = raw.get("slide_index")
            if scene_type == "slide" and slide_index is None:
                continue
            if scene_type == "slide" and (slide_index < 0 or slide_index >= evidence_count):
                continue
            filler_title = raw.get("filler_title")
            filler_bullets = raw.get("filler_bullets", []) or []
            on_screen = raw.get("on_screen")
            speaker = raw.get("speaker")
            scenes.append(
                Scene(
                    scene_type=scene_type,
                    narration=narration,
                    slide_index=slide_index,
                    filler_title=filler_title,
                    filler_bullets=filler_bullets,
                    on_screen=on_screen,
                    speaker=speaker,
                )
            )
        return scenes

    def _fallback_scenes(self, evidence: list[SlideEvidence]) -> list[Scene]:
        return self._build_default_scenes(evidence)

    def _build_default_scenes(self, evidence: list[SlideEvidence]) -> list[Scene]:
        scenes: list[Scene] = []
        current_section: str | None = None
        for idx, item in enumerate(evidence):
            if item.section and item.section != current_section:
                if current_section is not None:
                    bullets = self._section_bullets(evidence, idx, item.section)
                    narration = self._clip_text(
                        f"Next, {item.section}. " + ", ".join(bullets),
                        self.config.max_tts_chars,
                    )
                    scenes.append(
                        Scene(
                            scene_type="filler",
                            narration=narration,
                            filler_title=item.section,
                            filler_bullets=bullets,
                        )
                    )
                current_section = item.section

            narration = self._scene_narration(item)
            on_screen = self._scene_on_screen(item)
            scenes.append(
                Scene(
                    scene_type="slide",
                    narration=narration,
                    slide_index=idx,
                    on_screen=on_screen,
                    speaker=self.config.voice if not self.config.two_voices else None,
                )
            )
        return scenes

    def _section_bullets(
        self, evidence: list[SlideEvidence], start_idx: int, section: str
    ) -> list[str]:
        bullets: list[str] = []
        for item in evidence[start_idx:]:
            if item.section != section:
                if bullets:
                    break
                continue
            candidate = item.quote or item.slide_text or item.paper_title
            if candidate:
                bullets.append(self._clip_text(self._first_sentence(candidate), 120))
            if len(bullets) >= 3:
                break
        if not bullets:
            bullets = ["Key highlights", "Evidence from presentations", "Notable results"]
        return bullets

    def _scene_narration(self, item: SlideEvidence) -> str:
        max_chars = min(self.config.max_tts_chars, 700)
        if item.transcript_excerpt:
            return self._take_sentences(item.transcript_excerpt, 2, max_chars)
        if item.quote:
            return self._clip_text(item.quote, max_chars)
        if item.paper_abstract:
            prefix = f"{item.paper_title}. " if item.paper_title else ""
            return self._take_sentences(prefix + item.paper_abstract, 2, max_chars)
        if item.slide_text:
            return self._take_sentences(item.slide_text, 2, max_chars)
        if item.source_title:
            return f"From {item.source_title}."
        return "Key takeaway."

    def _scene_on_screen(self, item: SlideEvidence) -> str | None:
        if item.quote:
            return self._clip_text(item.quote, 160)
        if item.slide_text:
            return self._clip_text(self._first_sentence(item.slide_text), 140)
        if item.paper_title:
            return self._clip_text(item.paper_title, 140)
        if item.section:
            return item.section
        return item.source_title

    def _first_sentence(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return sentences[0].strip() if sentences and sentences[0].strip() else cleaned

    def _take_sentences(self, text: str, max_sentences: int, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        collected: list[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            tentative = " ".join(collected + [sentence]).strip()
            if len(tentative) > max_chars and collected:
                break
            collected.append(sentence)
            if len(collected) >= max_sentences:
                break
        if not collected:
            return self._clip_text(cleaned, max_chars)
        return " ".join(collected).strip()

    def _clip_text(self, text: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if len(cleaned) <= max_chars:
            return cleaned
        clipped = cleaned[: max_chars - 1].rsplit(" ", 1)[0]
        return f"{clipped}..." if clipped else cleaned[: max_chars - 1] + "..."

    def _render_annotated_slide(self, image_path: Path, output_path: Path, caption: str) -> None:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))
        if caption:
            caption = self._clip_text(caption, 180)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            wrapped = self._wrap_text(self._style_caption(caption), 70)
            text = "\n".join(wrapped)
            bbox = draw.multiline_textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            margin = 20
            box_height = text_height + margin * 2
            box = (0, self.config.height - box_height, self.config.width, self.config.height)
            draw.rectangle(box, fill=(0, 0, 0, 180))
            draw.multiline_text(
                (margin, self.config.height - box_height + margin),
                text,
                fill=(255, 255, 255),
                font=font,
            )
        image.save(output_path)

    def _render_filler_slide(self, output_path: Path, title: str, bullets: list[str]) -> None:
        if self.config.prefer_playwright and self._render_html_filler(output_path, title, bullets):
            return
        self._render_pil_filler(output_path, title, bullets)

    def _filler_content_from_evidence(
        self, item: SlideEvidence, scene: Scene
    ) -> tuple[str, list[str]]:
        title = (
            scene.filler_title
            or scene.on_screen
            or item.paper_title
            or item.section
            or item.source_title
            or "Slide"
        )
        text_source = item.slide_text or item.paper_abstract or item.quote or item.transcript_excerpt
        bullets = self._bullets_from_text(text_source, max_items=4)
        if not bullets:
            bullets = ["Context slide", "Supporting evidence", "See narration for details"]
        return title, bullets

    def _bullets_from_text(self, text: str | None, max_items: int) -> list[str]:
        if not text:
            return []
        if "\n" in text:
            parts = [line.strip(" -•\t") for line in text.splitlines()]
        else:
            parts = re.split(r"(?<=[.!?])\s+", text)
        bullets: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            bullets.append(self._clip_text(cleaned, 120))
            if len(bullets) >= max_items:
                break
        return bullets

    def _render_html_filler(self, output_path: Path, title: str, bullets: list[str]) -> bool:
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return False

        html = self._build_filler_html(title, bullets)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": self.config.width, "height": self.config.height})
            page.set_content(html)
            page.wait_for_timeout(500)
            page.screenshot(path=str(output_path))
            browser.close()
        return True

    def _build_filler_html(self, title: str, bullets: list[str]) -> str:
        bullet_html = "".join([f"<li>{self._escape_html(b)}</li>" for b in bullets])
        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body {{
        margin: 0;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #f7f7f7;
        background: linear-gradient(120deg, #0f172a, #1d4ed8);
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .card {{
        width: 82%;
        padding: 48px;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.7);
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
      }}
      h1 {{
        font-size: 48px;
        margin: 0 0 20px 0;
      }}
      ul {{
        font-size: 26px;
        line-height: 1.5;
        margin: 0;
        padding-left: 28px;
      }}
      .accent {{
        display: inline-block;
        margin-bottom: 16px;
        font-size: 16px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #a5b4fc;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="accent">Transition</div>
      <h1>{self._escape_html(title)}</h1>
      <ul>{bullet_html}</ul>
    </div>
    <script>
      const hues = [210, 230, 250, 270];
      const hue = hues[Math.floor(Math.random() * hues.length)];
      document.body.style.background = `linear-gradient(120deg, hsl(${{hue}}, 80%, 20%), hsl(${{hue + 30}}, 75%, 35%))`;
    </script>
  </body>
</html>
"""

    def _render_pil_filler(self, output_path: Path, title: str, bullets: list[str]) -> None:
        image = Image.new("RGB", (self.config.width, self.config.height), (15, 23, 42))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((60, 60), title, fill=(255, 255, 255), font=font)
        y = 140
        for bullet in bullets:
            line = f"- {bullet}"
            draw.text((80, y), line, fill=(200, 200, 255), font=font)
            y += 28
        image.save(output_path)

    def _escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _wrap_text(self, text: str, width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            if len(" ".join(current + [word])) > width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines

    def _split_text(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if not sentence:
                continue
            if current_len + len(sentence) + 1 > max_chars and current:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _probe_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _build_static_segment(self, slide_path: Path, audio_path: Path, output_path: Path) -> Path:
        duration = self._probe_duration(audio_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(slide_path),
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"scale={self.config.width}:{self.config.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.config.width}:{self.config.height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _voice_for_scene(self, scene: Scene, index: int) -> str:
        if not self.config.two_voices:
            return self.config.voice
        if scene.speaker:
            speaker = str(scene.speaker).strip().lower()
            if speaker in {"host b", "b", "second"}:
                return self.config.voice_b
            return self.config.voice
        return self.config.voice if index % 2 == 1 else self.config.voice_b

    def _style_caption(self, caption: str) -> str:
        return self._emphasize_numbers(caption)

    def _emphasize_numbers(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            value = match.group(1)
            suffix = match.group(2) or ""
            if suffix:
                return f"{value}×"
            return value
        return re.sub(r"(\b\d+(?:\.\d+)?)(x|×)?", repl, text)

    def _mix_music_bed(
        self, narration_path: Path, music_path: Path, duration: float, output_path: Path
    ) -> Path:
        if not shutil.which("ffmpeg"):
            return narration_path
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(narration_path),
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            f"[1:a]volume={self.config.music_bed_gain}[bed];[0:a][bed]amix=inputs=2:duration=first",
            "-c:a",
            "aac",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _concat_segments(self, segment_paths: list[Path]) -> Path:
        concat_list = self.config.output_dir / "concat.txt"
        concat_list.write_text(
            "\n".join([f"file '{path.resolve().as_posix()}'" for path in segment_paths]),
            encoding="utf-8",
        )

        output_path = self.config.output_dir / "topic.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if self.config.normalize_audio:
            normalized = self.config.output_dir / "topic_normalized.mp4"
            loudnorm = (
                f"loudnorm=I={self.config.loudnorm_i}:"
                f"TP={self.config.loudnorm_tp}:"
                f"LRA={self.config.loudnorm_lra}"
            )
            cmd_norm = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-filter:a",
                loudnorm,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(normalized),
            ]
            subprocess.run(cmd_norm, check=True, capture_output=True)
            return normalized

        return output_path

    def _merge_audio_chunks(self, chunk_paths: list[Path], output_path: Path) -> Path:
        if not shutil.which("ffmpeg"):
            raise TopicVideoError("ffmpeg is required to merge audio chunks.")
        concat_list = output_path.parent / f"{output_path.stem}_concat.txt"
        concat_list.write_text(
            "\n".join([f"file '{path.resolve().as_posix()}'" for path in chunk_paths]),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
