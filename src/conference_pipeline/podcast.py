"""Podcast generation utilities for turning a text document into audio."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from conference_pipeline.config import load_settings
from conference_pipeline.model_governance import default_model, default_provider, validate_selection
from conference_pipeline.provider_receipts import anthropic_messages_create, record_provider_call
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROVIDER = default_provider("podcast.script_generation")
DEFAULT_MODEL = default_model("podcast.script_generation")
DEFAULT_TTS_MODEL = default_model("podcast.tts")
DEFAULT_VOICE_A = "alloy"
DEFAULT_VOICE_B = "nova"


@dataclass
class PodcastSegment:
    """A single narrated segment in the podcast."""

    speaker: str
    text: str


@dataclass
class PodcastScript:
    """Structured podcast script."""

    title: str
    summary: str
    segments: list[PodcastSegment]
    show_notes: list[str] = field(default_factory=list)


@dataclass
class PodcastConfig:
    """Configuration for podcast generation."""

    title: str
    host_a: str = "Host A"
    host_b: str = "Host B"
    two_hosts: bool = True
    target_minutes: int = 8
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    voice_a: str = DEFAULT_VOICE_A
    voice_b: str = DEFAULT_VOICE_B
    max_segment_chars: int = 800
    max_tts_chars: int = 900
    temperature: float = 0.4
    max_source_chars: int = 60000
    include_show_notes: bool = True
    output_dir: Path = field(default_factory=lambda: Path("podcast_output"))
    source_path: Path | None = None
    source_paths: list[Path] = field(default_factory=list)
    openai_base_url: str = "https://api.openai.com/v1"
    intro_path: Path | None = None
    outro_path: Path | None = None
    normalize_audio: bool = True
    loudnorm_i: str = "-16"
    loudnorm_tp: str = "-1.5"
    loudnorm_lra: str = "11"


class PodcastError(Exception):
    """Raised when podcast generation fails."""


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

    def _raw_tts_post(self, model: str, voice: str, text: str) -> bytes:
        """Execute raw TTS HTTP POST; extracted for record_provider_call wrapping."""
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

    def tts(self, model: str, voice: str, text: str) -> bytes:
        # D-12: char→minute proxy matching catalog estimate_method for tts_minute_estimate
        tts_minutes = len(text) / 1000 / 3.0
        result = record_provider_call(
            lambda: self._raw_tts_post(model, voice, text),
            provider="openai",
            model=model,
            operation="audio.speech",
            tts_minutes=tts_minutes,
            caller=__file__,
        )
        return result.response


class AnthropicClient:
    """Minimal Anthropic messages client for governed podcast scripts."""

    def __init__(self, api_key: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency boundary.
            raise PodcastError("anthropic package is required for podcast scripts") from exc
        self.client = Anthropic(api_key=api_key, timeout=300)

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float) -> str:
        del temperature
        system = "You are a meticulous podcast producer. Return one JSON object only."
        user_messages = [message for message in messages if message.get("role") != "system"]
        response = anthropic_messages_create(
            self.client,
            caller=__file__,
            model=model,
            max_tokens=16000,
            system=system,
            messages=user_messages,
        )
        return "".join(
            str(block.get("text") or "")
            if isinstance(block, dict)
            else str(getattr(block, "text", "") or "")
            for block in (getattr(response, "content", None) or [])
        )


class PodcastGenerator:
    """Generate a podcast script and audio from a text document."""

    def __init__(self, config: PodcastConfig, api_key: str | None = None) -> None:
        settings = load_settings()
        validate_selection("podcast.script_generation", provider=config.provider, model=config.model)
        validate_selection("podcast.tts", provider="openai", model=config.tts_model)
        self.config = config
        if config.provider == "anthropic":
            script_api_key = api_key or settings.anthropic_api_key
            if not script_api_key:
                raise PodcastError("ANTHROPIC_API_KEY not configured.")
            self.script_client: Any = AnthropicClient(api_key=script_api_key)
        elif config.provider == "openai":
            script_api_key = api_key or settings.openai_api_key
            if not script_api_key:
                raise PodcastError("OPENAI_API_KEY not configured.")
            self.script_client = OpenAIClient(
                api_key=script_api_key,
                base_url=config.openai_base_url,
            )
        else:
            raise PodcastError(f"Unsupported podcast script provider: {config.provider}")

        tts_api_key = settings.openai_api_key
        if not tts_api_key:
            raise PodcastError("OPENAI_API_KEY not configured for podcast TTS.")
        self.tts_client = OpenAIClient(api_key=tts_api_key, base_url=config.openai_base_url)

    def read_source(self, source_path: Path) -> str:
        text = source_path.read_text(encoding="utf-8")
        if len(text) > self.config.max_source_chars:
            logger.warning(
                "Source text truncated for prompt size",
                extra={"limit": self.config.max_source_chars, "original": len(text)},
            )
            text = text[: self.config.max_source_chars]
        return text

    def read_sources(self, source_paths: list[Path]) -> str:
        parts: list[str] = []
        for path in source_paths:
            text = path.read_text(encoding="utf-8")
            parts.append(f"[SOURCE: {path.as_posix()}]\n{text}")
        combined = "\n\n".join(parts)
        if len(combined) > self.config.max_source_chars:
            logger.warning(
                "Combined source text truncated for prompt size",
                extra={"limit": self.config.max_source_chars, "original": len(combined)},
            )
            combined = combined[: self.config.max_source_chars]
        return combined

    def generate_script(self, source_text: str) -> PodcastScript:
        prompt = self._build_prompt(source_text)
        messages = [
            {"role": "system", "content": "You are a meticulous podcast producer."},
            {"role": "user", "content": prompt},
        ]
        raw = self.script_client.chat(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        payload = self._parse_json_payload(raw)
        return self._coerce_script(payload)

    def write_outputs(self, script: PodcastScript) -> dict[str, Path]:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "script.json"
        json_path.write_text(self._script_to_json(script), encoding="utf-8")

        md_path = output_dir / "script.md"
        md_path.write_text(self._script_to_markdown(script), encoding="utf-8")

        notes_path = output_dir / "show_notes.md"
        notes_path.write_text(self._show_notes(script), encoding="utf-8")

        return {
            "script_json": json_path,
            "script_markdown": md_path,
            "show_notes": notes_path,
        }

    def synthesize_audio(self, script: PodcastScript) -> list[Path]:
        audio_dir = self.config.output_dir / "audio_segments"
        audio_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []
        for idx, segment in enumerate(script.segments, start=1):
            voice = self._voice_for_speaker(segment.speaker)
            chunks = self._split_text(segment.text, self.config.max_tts_chars)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                filename = f"segment_{idx:03d}_{chunk_idx:02d}.mp3"
                path = audio_dir / filename
                audio = self.tts_client.tts(
                    model=self.config.tts_model,
                    voice=voice,
                    text=chunk,
                )
                path.write_bytes(audio)
                outputs.append(path)
        return outputs

    def assemble_audio(self, segment_paths: list[Path]) -> Path | None:
        if not segment_paths:
            return None

        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found; leaving audio segments unmerged.")
            return None

        inputs: list[Path] = []
        if self.config.intro_path:
            inputs.append(self.config.intro_path)
        inputs.extend(segment_paths)
        if self.config.outro_path:
            inputs.append(self.config.outro_path)

        output_path = self.config.output_dir / "podcast.mp3"
        filter_inputs = "".join([f"[{idx}:a]" for idx in range(len(inputs))])
        concat_filter = f"{filter_inputs}concat=n={len(inputs)}:v=0:a=1[a]"

        if self.config.normalize_audio:
            loudnorm = (
                f"loudnorm=I={self.config.loudnorm_i}:"
                f"TP={self.config.loudnorm_tp}:"
                f"LRA={self.config.loudnorm_lra}"
            )
            filter_complex = f"{concat_filter};[a]{loudnorm}[aout]"
            map_target = "[aout]"
        else:
            filter_complex = concat_filter
            map_target = "[a]"

        cmd = ["ffmpeg", "-y"]
        for path in inputs:
            cmd.extend(["-i", str(path)])
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                map_target,
                "-ar",
                "44100",
                "-ac",
                "2",
                str(output_path),
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _build_prompt(self, source_text: str) -> str:
        host_a = self.config.host_a
        host_b = self.config.host_b
        if self.config.two_hosts:
            host_line = f"Hosts: {host_a} and {host_b}."
            speaker_rule = f"Use both hosts and alternate every 1-2 segments."
        else:
            host_line = f"Single host: {host_a}."
            speaker_rule = f"Use only {host_a} in every segment."
        target_words = int(self.config.target_minutes * 130)
        return f"""Create a podcast script based ONLY on the source text below.

{host_line}
Target length: about {target_words} words.
Max segment length: {self.config.max_segment_chars} characters.
Tone: concise, engaging, and factual.
Rules:
- Do not add facts not present in the source.
- If quoting the source, quote exactly.
- Output JSON only.
- {speaker_rule}

JSON schema:
{{
  "title": "...",
  "summary": "...",
  "segments": [
    {{"speaker": "{host_a}", "text": "..."}},
    {{"speaker": "{host_b}", "text": "..."}}
  ],
  "show_notes": ["..."]
}}

SOURCE TEXT:
{source_text}
"""

    def _parse_json_payload(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"{.*}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise PodcastError("Failed to parse JSON response from model.")

    def _coerce_script(self, payload: dict[str, Any]) -> PodcastScript:
        title = payload.get("title", self.config.title).strip()
        summary = payload.get("summary", "").strip()
        segments_raw = payload.get("segments", [])

        if not segments_raw:
            raise PodcastError("No segments returned in script payload.")

        segments: list[PodcastSegment] = []
        for seg in segments_raw:
            speaker = str(seg.get("speaker", self.config.host_a)).strip()
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segments.append(PodcastSegment(speaker=speaker, text=text))

        if not segments:
            raise PodcastError("All segments were empty after parsing.")

        show_notes = payload.get("show_notes", [])
        if not isinstance(show_notes, list):
            show_notes = []

        return PodcastScript(title=title, summary=summary, segments=segments, show_notes=show_notes)

    def _script_to_json(self, script: PodcastScript) -> str:
        payload = {
            "title": script.title,
            "summary": script.summary,
            "segments": [
                {"speaker": seg.speaker, "text": seg.text} for seg in script.segments
            ],
            "show_notes": script.show_notes,
        }
        return json.dumps(payload, indent=2)

    def _script_to_markdown(self, script: PodcastScript) -> str:
        lines = [
            f"# {script.title}",
            "",
            "## Summary",
            script.summary,
            "",
            "## Script",
            "",
        ]
        for seg in script.segments:
            lines.append(f"{seg.speaker}: {seg.text}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _show_notes(self, script: PodcastScript) -> str:
        lines = ["# Show Notes", ""]
        sources = []
        if self.config.source_paths:
            sources.extend(self.config.source_paths)
        elif self.config.source_path:
            sources.append(self.config.source_path)
        if sources:
            lines.append("Sources:")
            for path in sources:
                lines.append(f"- {path.as_posix()}")
            lines.append("")
        if script.show_notes:
            for note in script.show_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- Generated from the source document.")
        return "\n".join(lines).strip() + "\n"

    def _voice_for_speaker(self, speaker: str) -> str:
        if speaker.strip().lower() == self.config.host_b.lower().strip() and self.config.two_hosts:
            return self.config.voice_b
        return self.config.voice_a

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

        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), max_chars):
                    final_chunks.append(chunk[i : i + max_chars])
        return final_chunks
