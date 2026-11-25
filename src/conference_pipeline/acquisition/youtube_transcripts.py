"""Tiered transcript acquisition for public YouTube talk videos.

Two tiers, in order:

1. **YouTube captions (free).** yt-dlp pulls the manual track before the
   auto-generated one (``youtube_caption_manual`` / ``youtube_caption_auto``).
   When YouTube serves its "confirm you're not a bot" gate the lane is marked
   blocked for the rest of the run: the gate is an IP-level block, so further
   requests only extend it and a failure there says nothing about captions.
2. **Gemini on the YouTube URL (paid, governed).** Google fetches the public
   video server-side, so the local IP block does not apply. Gemini does not read
   the caption track; it transcribes the speech. The recording is split into
   time windows (``video_metadata`` offsets) at a very low frame rate because
   only the speech matters. Provenance is ``gemini_youtube_url_transcription``.

The model comes from the model-governance registry task
``transcription.youtube_url``; every call goes through
``record_provider_call`` so it lands in the canonical provider-receipt spend,
and a run-level budget halts (never auto-raises) before a window that would
cross it.

Window offsets past the end of a video make Gemini return a bare HTTP 500 that
is indistinguishable from a transient server error, so the duration must be
known up front. It is read from the public watch page (``lengthSeconds``),
which still answers while the caption endpoints are bot-gated.

Nothing here is conference-specific.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conference_pipeline.acquisition.embedded_media import sanitize_media_url
from conference_pipeline.acquisition.youtube_talks import (
    DEFAULT_TRANSCRIPT_LANGS,
    HarvestedVideo,
    Runner,
    YouTubeBotCheckBlocked,
    _default_runner,
    download_captions,
    resolve_yt_dlp_cmd,
    write_transcript_sidecar,
)

TRANSCRIPTION_TASK_ID = "transcription.youtube_url"
GEMINI_TRANSCRIPT_SOURCE = "gemini_youtube_url_transcription"
GEMINI_OPERATION = "youtube_url_transcription"

# 15-minute windows: on CVPR 2026 recordings Gemini returned ~11-12 chars/s
# of English from 30-minute windows but 15-16 chars/s from 15-minute ones of
# the same talk, i.e. long windows silently drop speech.
DEFAULT_WINDOW_SECONDS = 900
MIN_WINDOW_SECONDS = 300
# A final window shorter than this is merged into the previous one.
MIN_TAIL_SECONDS = 60
# Listed durations can exceed the real stream by a second or two, and an end
# offset past the end returns a bare HTTP 500; stop this far short of it.
END_GUARD_SECONDS = 2
# Normal speech yields ~2.5-3 output tokens per second of video in English,
# Chinese, Japanese, and Korean alike. A window far below that (one keynote
# came back at 0.6 with only its closing Q&A) is re-asked in halves.
MIN_OUTPUT_TOKENS_PER_SECOND = 1.2
# One frame every 50 s: speech is what matters. Measured on a CVPR 2026
# tutorial clip, this cut prompt tokens ~3.4x versus the 1 fps default with an
# essentially identical transcript.
DEFAULT_GEMINI_FPS = 0.02
# Conservative planning rates for the budget preflight (measured ~27 input
# tokens/s at fps 0.02 and ~3 output tokens/s of speech).
PLANNING_INPUT_TOKENS_PER_SECOND = 40
PLANNING_OUTPUT_TOKENS_PER_SECOND = 5

NO_SPEECH_MARKER = "[NO SPEECH]"
# A window whose output repeats one sentence many times is a decoding loop (one
# English-translation window of a 7.5-minute span came back as 322k characters,
# one sentence 1,352 times). Such a window is re-asked in halves down to
# LOOP_MIN_SPLIT_SECONDS; at that size the repeats are collapsed and flagged.
LOOP_REPEAT_THRESHOLD = 5
LOOP_MAX_OUTPUT_TOKENS_PER_SECOND = 20.0
LOOP_MIN_SPLIT_SECONDS = 60
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def strip_no_speech_markers(text: str) -> str:
    """Remove ``[NO SPEECH]`` markers the model interleaves with real speech."""

    return re.sub(r"[ \t]*\[NO SPEECH\][ \t]*\n?", "", text).strip()


def repetition_loop_count(text: str) -> int:
    """Occurrences of the most repeated sentence (20+ characters)."""

    counts: dict[str, int] = {}
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if len(sentence) >= 20:
            counts[sentence] = counts.get(sentence, 0) + 1
    return max(counts.values(), default=0)


def collapse_repeated_sentences(text: str) -> str:
    """Keep each 20+ character sentence once, in first-seen order."""

    seen: set[str] = set()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        key = sentence.strip()
        if len(key) >= 20:
            if key in seen:
                continue
            seen.add(key)
        kept.append(sentence)
    return " ".join(part.strip() for part in kept if part.strip())
GEMINI_PROMPT = (
    "Transcribe the spoken audio of this video segment verbatim, in the "
    "language spoken. Output only the transcript text: no timestamps, speaker "
    "labels, headings, summaries, or commentary. Keep technical terms, model "
    "names, and numbers exactly as spoken. Never read out on-screen text, "
    "slides, or captions: a silent slide or a music-only teaser has no "
    "transcript. If nobody speaks in the segment, output exactly "
    f"{NO_SPEECH_MARKER}."
)

# English output for talks given in another language: the corpus is searched
# in English, and a Japanese or Chinese transcript embeds and retrieves poorly
# next to it. The source language is kept in the sidecar.
GEMINI_ENGLISH_PROMPT = (
    "Transcribe the spoken audio of this video segment and translate it into "
    "English as you go. Output only the English transcript text: no "
    "timestamps, speaker labels, headings, summaries, or commentary, and no "
    "text in the original language. Translate faithfully and completely, "
    "sentence by sentence; do not summarize or skip. Keep technical terms, "
    "model and method names, and numbers exactly as spoken (romanize names). "
    "Never read out on-screen text, slides, or captions: a silent slide or a "
    "music-only teaser has no transcript. If nobody speaks in the segment, "
    f"output exactly {NO_SPEECH_MARKER}."
)


def gemini_prompt(output_language: str | None) -> str:
    """Prompt for verbatim transcription, or transcription into English."""

    if output_language is None:
        return GEMINI_PROMPT
    if output_language != "en":
        raise ValueError(f"unsupported Gemini output language {output_language!r}; only 'en'")
    return GEMINI_ENGLISH_PROMPT


TIER_YOUTUBE_CAPTION = "youtube_caption"
TIER_GEMINI = "gemini"

_LENGTH_SECONDS_RE = re.compile(r'"lengthSeconds"\s*:\s*"(\d+)"')
_CAPTION_TRACKS_RE = re.compile(r'"captionTracks"\s*:\s*\[')
_PLAYABILITY_RE = re.compile(
    r'"playabilityStatus"\s*:\s*\{\s*"status"\s*:\s*"(\w+)"(?:\s*,\s*"reason"\s*:\s*"([^"]*)")?'
)


class GeminiBudgetExhausted(RuntimeError):
    """The next Gemini window would cross the run budget: halt for approval."""


class GeminiTranscriptionError(RuntimeError):
    """A Gemini window failed after retries."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Watch-page metadata (duration, caption-track presence)
# ---------------------------------------------------------------------------


HttpGet = Callable[[str], str]


def _default_http_get(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https host
        return response.read().decode("utf-8", "replace")


def fetch_watch_page_metadata(video_id: str, *, http_get: HttpGet = _default_http_get) -> dict[str, Any]:
    """Read ``lengthSeconds`` and caption-track presence from the watch page.

    Returns ``{"duration_seconds", "caption_tracks_listed", "playability_status",
    "playability_reason"}``; the playability fields distinguish a taken-down
    (``ERROR``) or private (``LOGIN_REQUIRED``) video from a page that merely
    lacks a length. Network or parse failures yield ``None`` values, never an
    exception.
    """

    empty = {
        "duration_seconds": None,
        "caption_tracks_listed": None,
        "playability_status": None,
        "playability_reason": None,
    }
    try:
        page = http_get(f"https://www.youtube.com/watch?v={video_id}")
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return empty
    match = _LENGTH_SECONDS_RE.search(page)
    duration = int(match.group(1)) if match else None
    playability = _PLAYABILITY_RE.search(page)
    return {
        "duration_seconds": duration if duration and duration > 0 else None,
        "caption_tracks_listed": bool(_CAPTION_TRACKS_RE.search(page)) if page else None,
        "playability_status": playability.group(1) if playability else None,
        "playability_reason": playability.group(2) if playability else None,
    }


# ---------------------------------------------------------------------------
# Gemini tier
# ---------------------------------------------------------------------------


def plan_windows(duration_seconds: float, window_seconds: int) -> list[tuple[int, int]]:
    """Split ``[0, duration - END_GUARD_SECONDS)`` into contiguous windows.

    A tail shorter than ``MIN_TAIL_SECONDS`` is folded into the previous
    window so no request covers only the last few (possibly nonexistent)
    seconds of a video.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    end_total = int(duration_seconds)
    if end_total > END_GUARD_SECONDS + 1:
        end_total -= END_GUARD_SECONDS
    end_total = max(end_total, 1)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < end_total:
        end = min(start + window_seconds, end_total)
        windows.append((start, end))
        start = end
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < MIN_TAIL_SECONDS:
        tail_end = windows.pop()[1]
        windows[-1] = (windows[-1][0], tail_end)
    return windows


@dataclass
class GeminiWindowResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    truncated: bool
    estimated_cost_usd: float | None
    receipt_path: str | None = None


# (video_url, start_seconds | None, end_seconds | None) -> GeminiWindowResult
# ``None`` offsets mean the whole video: Gemini needs no duration, which keeps
# the lane from touching YouTube at all (watch-page reads get 429-ed when many
# videos run in parallel).
WindowTranscriber = Callable[[str, "int | None", "int | None"], GeminiWindowResult]


def _make_sdk_window_transcriber(
    *,
    api_key: str,
    model: str,
    fps: float,
    workspace: Path | None,
    run_id: str | None,
    caller: str,
    output_language: str | None = None,
) -> WindowTranscriber:  # pragma: no cover - live SDK path, exercised by operator runs
    from google import genai
    from google.genai import types

    from conference_pipeline.provider_receipts import (
        gemini_generate_content,
        gemini_output_tokens,
        provider_receipt_context,
    )

    client = genai.Client(api_key=api_key)

    def transcribe_window(
        video_url: str, start: int | None, end: int | None
    ) -> GeminiWindowResult:
        metadata = (
            types.VideoMetadata(fps=fps)
            if start is None or end is None
            else types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s", fps=fps)
        )
        part = types.Part(
            file_data=types.FileData(file_uri=video_url),
            video_metadata=metadata,
        )
        contents = types.Content(parts=[part, types.Part(text=gemini_prompt(output_language))])
        config = types.GenerateContentConfig(
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            temperature=0,
        )
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id="acquisition",
            caller=caller,
            task_id=TRANSCRIPTION_TASK_ID,
        ):
            result = gemini_generate_content(
                client,
                model=model,
                contents=contents,
                config=config,
                operation=GEMINI_OPERATION,
                caller=caller,
                receipt_request_kwargs={
                    "video_url": sanitize_media_url(video_url),
                    "start_seconds": start,
                    "end_seconds": end,
                    "whole_video": start is None or end is None,
                    "fps": fps,
                    "media_resolution": "low",
                    "output_language": output_language,
                },
            )
        response = result.response
        usage = getattr(response, "usage_metadata", None)
        finish = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(getattr(candidates[0], "finish_reason", None), "name", None)
        receipt = result.receipt
        return GeminiWindowResult(
            text=(getattr(response, "text", None) or "").strip(),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=gemini_output_tokens(usage),
            truncated=finish == "MAX_TOKENS",
            estimated_cost_usd=getattr(receipt, "estimated_cost_usd", None),
            receipt_path=getattr(receipt, "receipt_path", None),
        )

    return transcribe_window


@dataclass
class GeminiTranscriber:
    """Budget-governed Gemini transcription of a public YouTube URL."""

    model: str
    budget_usd: float
    window_transcriber: WindowTranscriber
    provider: str = "gemini"
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    fps: float = DEFAULT_GEMINI_FPS
    max_retries: int = 2
    retry_sleep_seconds: float = 10.0
    sleeper: Callable[[float], None] = time.sleep
    price_estimator: Callable[[int, int], float] | None = None
    # None: verbatim in the language spoken. "en": transcribed into English.
    output_language: str | None = None
    spent_usd: float = 0.0
    calls: int = 0
    failed_calls: int = 0
    halted: bool = False
    # Guards the budget and the counters when videos are transcribed in
    # parallel: the cap must never be crossed by two threads reserving at once.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def from_registry(
        cls,
        *,
        api_key: str,
        budget_usd: float,
        workspace: Path | None = None,
        run_id: str | None = None,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        fps: float = DEFAULT_GEMINI_FPS,
        caller: str = "acquisition.youtube_talk_harvest",
        output_language: str | None = None,
    ) -> "GeminiTranscriber":  # pragma: no cover - live wiring
        from conference_pipeline.model_governance.registry import (
            default_model,
            default_provider,
            load_registry,
        )

        provider = default_provider(TRANSCRIPTION_TASK_ID)
        if provider != "gemini":
            raise GeminiTranscriptionError(
                f"{TRANSCRIPTION_TASK_ID} must resolve to a gemini model, got {provider}"
            )
        gemini_prompt(output_language)  # rejects an unsupported language before any call
        model = default_model(TRANSCRIPTION_TASK_ID)
        cap = load_registry().get(TRANSCRIPTION_TASK_ID).budget_cap_usd
        if cap is not None and budget_usd > cap:
            raise GeminiBudgetExhausted(
                f"requested Gemini budget ${budget_usd:.2f} exceeds the registry cap "
                f"${cap:.2f} for {TRANSCRIPTION_TASK_ID}; raising it needs operator approval"
            )
        return cls(
            model=model,
            budget_usd=budget_usd,
            window_transcriber=_make_sdk_window_transcriber(
                api_key=api_key,
                model=model,
                fps=fps,
                workspace=workspace,
                run_id=run_id,
                caller=caller,
                output_language=output_language,
            ),
            window_seconds=window_seconds,
            fps=fps,
            price_estimator=_catalog_price_estimator(provider, model),
            output_language=output_language,
        )

    # -- budget -----------------------------------------------------------

    def estimate_window_cost(self, seconds: int) -> float:
        input_tokens = int(seconds * PLANNING_INPUT_TOKENS_PER_SECOND)
        output_tokens = int(seconds * PLANNING_OUTPUT_TOKENS_PER_SECOND)
        if self.price_estimator is None:
            raise GeminiTranscriptionError(
                "no price estimator configured; refusing an unpriced paid call"
            )
        return float(self.price_estimator(input_tokens, output_tokens))

    def _charge_locked(self, result: GeminiWindowResult, seconds: int) -> float:
        cost = result.estimated_cost_usd
        if cost is None and self.price_estimator is not None and (
            result.input_tokens is not None or result.output_tokens is not None
        ):
            cost = float(
                self.price_estimator(int(result.input_tokens or 0), int(result.output_tokens or 0))
            )
        if cost is None:
            # Fail closed: charge the planning estimate rather than zero.
            cost = self.estimate_window_cost(seconds)
        self.spent_usd += cost
        return cost

    def _charge(self, result: GeminiWindowResult, seconds: int) -> float:
        with self._lock:
            return self._charge_locked(result, seconds)

    def _preflight(self, seconds: int) -> None:
        with self._lock:
            projected = self.spent_usd + self.estimate_window_cost(seconds)
            if projected > self.budget_usd:
                self.halted = True
                raise GeminiBudgetExhausted(
                    f"next {seconds}s Gemini window would bring spend to ${projected:.4f}, "
                    f"over the ${self.budget_usd:.2f} run budget; halting for approval"
                )

    # -- transcription ----------------------------------------------------

    def _call_with_retries(self, video_url: str, start: int, end: int) -> GeminiWindowResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.window_transcriber(video_url, start, end)
                with self._lock:
                    self.calls += 1
                return result
            except GeminiBudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - provider SDK error types vary
                with self._lock:
                    self.failed_calls += 1
                last_error = exc
                if attempt < self.max_retries:
                    self.sleeper(self.retry_sleep_seconds * (attempt + 1))
        raise GeminiTranscriptionError(
            f"Gemini window {start}-{end}s failed after {self.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {str(last_error)[:200]}"
        )

    def _transcribe_span(
        self, video_url: str, start: int, end: int, segments: list[dict[str, Any]], usage: dict[str, Any]
    ) -> None:
        seconds = end - start
        self._preflight(seconds)
        result = self._call_with_retries(video_url, start, end)
        cost = self._charge(result, seconds)
        usage["input_tokens"] += int(result.input_tokens or 0)
        usage["output_tokens"] += int(result.output_tokens or 0)
        usage["estimated_cost_usd"] += cost
        if result.receipt_path:
            usage["provider_receipts"].append(result.receipt_path)
        text = strip_no_speech_markers(result.text)
        no_speech = not text
        density = (
            float(result.output_tokens) / seconds
            if result.output_tokens is not None and seconds > 0
            else None
        )
        sparse = bool(
            not no_speech and density is not None and density < MIN_OUTPUT_TOKENS_PER_SECOND
        )
        looping = repetition_loop_count(text) >= LOOP_REPEAT_THRESHOLD or bool(
            density is not None and density > LOOP_MAX_OUTPUT_TOKENS_PER_SECOND
        )
        if looping and seconds >= 2 * LOOP_MIN_SPLIT_SECONDS:
            usage["split_windows"] += 1
            usage["loop_splits"] = int(usage.get("loop_splits") or 0) + 1
            middle = start + seconds // 2
            self._transcribe_span(video_url, start, middle, segments, usage)
            self._transcribe_span(video_url, middle, end, segments, usage)
            return
        repetition_collapsed = False
        if looping:
            text = collapse_repeated_sentences(text)
            repetition_collapsed = True
        if (result.truncated or sparse) and seconds >= 2 * MIN_WINDOW_SECONDS:
            usage["split_windows"] += 1
            if sparse and not result.truncated:
                usage["sparse_splits"] += 1
            middle = start + seconds // 2
            self._transcribe_span(video_url, start, middle, segments, usage)
            self._transcribe_span(video_url, middle, end, segments, usage)
            return
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": "" if no_speech else text,
                "no_speech": no_speech,
                "truncated": bool(result.truncated),
                "sparse": sparse,
                "repetition_collapsed": repetition_collapsed,
                "output_tokens_per_second": round(density, 3) if density is not None else None,
            }
        )

    def transcribe(
        self,
        video: HarvestedVideo,
        *,
        duration_seconds: float | None,
        transcripts_dir: Path,
    ) -> dict[str, Any]:
        """Transcribe ``video`` and write its sidecar.

        With a known duration the recording is split into windows. Without one
        (``duration_seconds is None``) Gemini is asked for the whole video in a
        single call, so no YouTube metadata read is needed; the result is
        rejected as ``whole_video_incomplete`` if it comes back truncated,
        since there is no duration to split against.
        """

        segments: list[dict[str, Any]] = []
        usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "split_windows": 0,
            "sparse_splits": 0,
            "provider_receipts": [],
        }
        if duration_seconds is None:
            self._transcribe_whole(video.url, segments, usage)
        else:
            for start, end in plan_windows(duration_seconds, self.window_seconds):
                self._transcribe_span(video.url, start, end, segments, usage)
        text = " ".join(segment["text"] for segment in segments if segment["text"])
        record = {
            "video_id": video.video_id,
            "video_url": sanitize_media_url(video.url),
            "transcript_source": GEMINI_TRANSCRIPT_SOURCE,
            "language": self.output_language,
            "output_language": self.output_language,
            "translated_by_model": self.output_language is not None,
            "provider": self.provider,
            "model": self.model,
            "task_id": TRANSCRIPTION_TASK_ID,
            "duration_seconds": duration_seconds,
            "whole_video_call": duration_seconds is None,
            "window_seconds": self.window_seconds,
            "fps": self.fps,
            "segment_count": len(segments),
            "word_count": len(text.split()),
            # Whitespace word counts undercount Chinese/Japanese; keep chars too.
            "char_count": len(text),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "truncated_windows": sum(1 for segment in segments if segment["truncated"]),
            "sparse_windows": sum(1 for segment in segments if segment["sparse"]),
            "usage": {**usage, "estimated_cost_usd": round(usage["estimated_cost_usd"], 6)},
            "generated_at": _utc_now_iso(),
            "segments": segments,
            "text": text,
        }
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_path = transcripts_dir / f"{video.video_id}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {
            "transcript_path": str(out_path),
            "transcript_source": GEMINI_TRANSCRIPT_SOURCE,
            "language": self.output_language,
            "output_language": self.output_language,
            "segment_count": record["segment_count"],
            "word_count": record["word_count"],
            "char_count": record["char_count"],
            "text_sha256": record["text_sha256"],
            "model": self.model,
            "estimated_cost_usd": record["usage"]["estimated_cost_usd"],
            "truncated_windows": record["truncated_windows"],
            "sparse_windows": record["sparse_windows"],
        }

    def _transcribe_whole(
        self, video_url: str, segments: list[dict[str, Any]], usage: dict[str, Any]
    ) -> None:
        """One call for the entire video: no duration, no YouTube metadata read."""

        self._preflight(self.window_seconds)
        result = self._call_with_retries(video_url, None, None)
        cost = self._charge(result, self.window_seconds)
        usage["input_tokens"] += int(result.input_tokens or 0)
        usage["output_tokens"] += int(result.output_tokens or 0)
        usage["estimated_cost_usd"] += cost
        if result.receipt_path:
            usage["provider_receipts"].append(result.receipt_path)
        if result.truncated:
            raise GeminiTranscriptionError(
                "whole_video_incomplete: the single-call transcript was truncated and "
                "there is no duration to split the video against"
            )
        text = strip_no_speech_markers(result.text)
        if repetition_loop_count(text) >= LOOP_REPEAT_THRESHOLD:
            raise GeminiTranscriptionError(
                "whole_video_repetition_loop: the single-call transcript repeats itself and "
                "there is no duration to split the video against"
            )
        no_speech = not text
        segments.append(
            {
                "start": 0.0,
                "end": None,
                "text": "" if no_speech else text,
                "no_speech": no_speech,
                "truncated": False,
                "sparse": False,
                "whole_video": True,
                "output_tokens_per_second": None,
            }
        )

    def lane_summary(self) -> dict[str, Any]:
        with self._lock:
            return self._lane_summary_locked()

    def _lane_summary_locked(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "task_id": TRANSCRIPTION_TASK_ID,
            "provider": self.provider,
            "model": self.model,
            "window_seconds": self.window_seconds,
            "fps": self.fps,
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "halted_on_budget": self.halted,
        }


def _catalog_price_estimator(provider: str, model: str) -> Callable[[int, int], float]:
    from conference_pipeline.costs.estimator import load_pricing_catalog

    catalog = load_pricing_catalog()
    catalog.resolve_model(model, provider)  # fail closed on an unpriced model

    def estimate(input_tokens: int, output_tokens: int) -> float:
        cost, _sources, _warnings = catalog.estimate_tokens(provider, model, input_tokens, output_tokens)
        return float(cost)

    return estimate


# ---------------------------------------------------------------------------
# Tier orchestration
# ---------------------------------------------------------------------------


@dataclass
class YouTubeLaneState:
    """Run-scoped state for the free YouTube caption tier."""

    blocked: bool = False
    blocked_at_video_id: str | None = None
    blocked_at: str | None = None
    caption_requests: int = 0


def acquire_transcript(
    video: HarvestedVideo,
    *,
    captions_dir: Path,
    transcripts_dir: Path,
    runner: Runner,
    yt_dlp_cmd: Sequence[str],
    youtube_state: YouTubeLaneState,
    langs: Sequence[str] = DEFAULT_TRANSCRIPT_LANGS,
    gemini: GeminiTranscriber | None = None,
    http_get: HttpGet = _default_http_get,
    known_no_caption: bool = False,
    gemini_only: bool = False,
) -> dict[str, Any]:
    """Get one video's transcript: YouTube captions, then Gemini.

    Returns ``{"tier", "status", "transcript", ...}`` where ``status`` is
    ``transcribed`` or a gap reason (``no_caption_gemini_disabled``,
    ``youtube_blocked_gemini_disabled``, ``video_unavailable``,
    ``duration_unknown``, ``gemini_failed``). :class:`GeminiBudgetExhausted` propagates so the run
    halts.
    """

    youtube_outcome = "skipped_lane_blocked" if youtube_state.blocked else None
    if gemini_only:
        # Paid tier only: YouTube is never contacted, so nothing can bot-gate
        # the run and videos may be transcribed in parallel.
        youtube_outcome = "skipped_gemini_only"
    elif known_no_caption:
        youtube_outcome = "no_caption_track"
    elif not youtube_state.blocked:
        youtube_state.caption_requests += 1
        try:
            caption_info = download_captions(
                video, captions_dir, runner=runner, yt_dlp_cmd=yt_dlp_cmd, langs=langs
            )
        except YouTubeBotCheckBlocked:
            youtube_state.blocked = True
            youtube_state.blocked_at_video_id = video.video_id
            youtube_state.blocked_at = _utc_now_iso()
            youtube_outcome = "bot_check_blocked"
        else:
            if caption_info is not None:
                return {
                    "tier": TIER_YOUTUBE_CAPTION,
                    "status": "transcribed",
                    "youtube_outcome": "caption_found",
                    "transcript": write_transcript_sidecar(video, caption_info, transcripts_dir),
                }
            youtube_outcome = "no_caption_track"

    if gemini is None:
        status = (
            "no_caption_gemini_disabled"
            if youtube_outcome == "no_caption_track"
            else "youtube_blocked_gemini_disabled"
        )
        return {
            "tier": None,
            "status": status,
            "youtube_outcome": youtube_outcome,
            "transcript": {"transcript_source": "asr_fallback_required"},
        }

    duration = video.duration_seconds
    duration_source = "input" if duration else None
    if not duration and not gemini_only:
        # Serial lane: one watch-page read per video is safe and lets a
        # taken-down video be named as such before any paid call.
        metadata = fetch_watch_page_metadata(video.video_id, http_get=http_get)
        playability = metadata["playability_status"]
        if playability and playability != "OK":
            # Taken down or private: Gemini only reaches public videos.
            return {
                "tier": None,
                "status": "video_unavailable",
                "youtube_outcome": youtube_outcome,
                "playability_status": playability,
                "playability_reason": metadata["playability_reason"],
                "transcript": {"transcript_source": "asr_fallback_required"},
            }
        duration = metadata["duration_seconds"]
        duration_source = "watch_page" if duration else None
        if not duration:
            return {
                "tier": TIER_GEMINI,
                "status": "duration_unknown",
                "youtube_outcome": youtube_outcome,
                "transcript": {"transcript_source": "asr_fallback_required"},
            }
    if not duration:
        duration_source = "whole_video_call"
    try:
        transcript = gemini.transcribe(
            video,
            duration_seconds=float(duration) if duration else None,
            transcripts_dir=transcripts_dir,
        )
    except GeminiTranscriptionError as exc:
        # 403 PERMISSION_DENIED is Gemini saying it cannot fetch the video at
        # all (private, age-restricted, region-locked) — a gap, not a fault of
        # the lane, and retrying it costs time for nothing.
        text = str(exc).lower()
        unavailable = any(
            marker in text
            for marker in ("not available", "private", "permission_denied", "403")
        )
        return {
            "tier": TIER_GEMINI,
            "status": "video_unavailable" if unavailable else "gemini_failed",
            "youtube_outcome": youtube_outcome,
            "error": str(exc)[:300],
            "transcript": {"transcript_source": "asr_fallback_required"},
        }
    return {
        "tier": TIER_GEMINI,
        "status": "transcribed",
        "youtube_outcome": youtube_outcome,
        "duration_source": duration_source,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Video-list mode (no program spine)
# ---------------------------------------------------------------------------


@dataclass
class VideoListConfig:
    """Transcribe an explicit list of public YouTube videos."""

    workspace: Path
    videos: Sequence[Mapping[str, Any]]
    output_dir: Path
    run_id: str | None = None
    langs: Sequence[str] = DEFAULT_TRANSCRIPT_LANGS
    resume: bool = True
    min_sleep_seconds: float = 15.0
    max_sleep_seconds: float = 30.0
    gemini: GeminiTranscriber | None = None
    # Paid tier only: skip the free caption tier entirely. Nothing contacts
    # YouTube, so there is no bot gate to pace against and videos can run in
    # parallel (``concurrency`` > 1 requires this).
    gemini_only: bool = False
    concurrency: int = 1
    inventory_name: str = "video_transcript_inventory.json"
    receipt_name: str = "video_transcript_receipt.json"


def load_video_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("videos") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("video list must be a JSON list or {'videos': [...]}")
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not str(row.get("video_id") or "").strip():
            raise ValueError(f"video list row {index} lacks video_id")
        video_id = str(row["video_id"]).strip()
        if video_id in seen:
            continue
        seen.add(video_id)
        videos.append(dict(row))
    return videos


def _video_from_row(row: Mapping[str, Any]) -> HarvestedVideo:
    video_id = str(row["video_id"])
    duration = row.get("duration_seconds")
    return HarvestedVideo(
        video_id=video_id,
        title=str(row.get("title") or ""),
        url=f"https://www.youtube.com/watch?v={video_id}",
        channel=row.get("channel"),
        duration_seconds=float(duration) if duration else None,
        source="video_list",
    )


def _swallow_budget_halt(
    work: Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], HarvestedVideo, dict[str, Any]]],
) -> Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], HarvestedVideo, dict[str, Any]]]:
    """Let a budget halt end the pool without losing the rows already done."""

    def wrapped(row: Mapping[str, Any]):
        video = _video_from_row(row)
        try:
            return work(row)
        except GeminiBudgetExhausted as exc:
            return row, video, {"status": "gemini_budget_halted", "error": str(exc), "tier": None}

    return wrapped


def transcribe_video_list(
    config: VideoListConfig,
    *,
    runner: Runner = _default_runner,
    yt_dlp_cmd: Sequence[str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    http_get: HttpGet = _default_http_get,
) -> dict[str, Any]:
    """Run the tiered transcript lane over a video list, checkpointing each row."""

    if yt_dlp_cmd is None:
        yt_dlp_cmd = resolve_yt_dlp_cmd()
    output_dir = config.output_dir
    captions_dir = output_dir / "captions"
    transcripts_dir = output_dir / "transcripts"
    inventory_path = output_dir / config.inventory_name
    receipt_path = output_dir / config.receipt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, Any]] = {}
    if config.resume and inventory_path.exists():
        try:
            prior = json.loads(inventory_path.read_text(encoding="utf-8"))
            rows = {str(row["video_id"]): dict(row) for row in prior.get("videos") or []}
        except (OSError, json.JSONDecodeError, KeyError):
            rows = {}

    youtube_state = YouTubeLaneState()
    halted_reason: str | None = None
    touched_youtube = False

    def write_inventory(status: str) -> dict[str, Any]:
        counts: dict[str, int] = {}
        tiers: dict[str, int] = {}
        for row in rows.values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row.get("status") == "transcribed":
                tier = str(row.get("tier"))
                tiers[tier] = tiers.get(tier, 0) + 1
        inventory = {
            "schema_version": 1,
            "run_id": config.run_id,
            "generated_at": _utc_now_iso(),
            "video_count": len(config.videos),
            "videos": list(rows.values()),
        }
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "stage": "acquisition.youtube_talk_harvest",
            "mode": "video_list",
            "status": status,
            "run_id": config.run_id,
            "generated_at": _utc_now_iso(),
            "workspace": str(config.workspace),
            "inventory_path": str(inventory_path),
            "allowed_hosts": ["youtube.com", "youtu.be"],
            "counts": {
                "videos": len(config.videos),
                "processed": len(rows),
                "by_status": counts,
                "transcribed_by_tier": tiers,
            },
            "youtube_lane": {
                "blocked": youtube_state.blocked,
                "blocked_at_video_id": youtube_state.blocked_at_video_id,
                "blocked_at": youtube_state.blocked_at,
                "caption_requests": youtube_state.caption_requests,
            },
            "gemini_lane": config.gemini.lane_summary() if config.gemini else {"enabled": False},
            "halted_reason": halted_reason,
            "boundaries": {
                "audio_downloaded": False,
                "video_downloaded": False,
                "cookies_used": False,
                "paywalled_sources_touched": False,
                "transcript_first": True,
            },
        }
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        receipt["receipt_path"] = str(receipt_path)
        return receipt

    if config.concurrency > 1 and not config.gemini_only:
        raise ValueError("concurrency > 1 requires gemini_only: the caption tier must stay paced and serial")

    def pending_rows() -> list[Mapping[str, Any]]:
        out = []
        for row in config.videos:
            prior = rows.get(str(row["video_id"]))
            if prior and prior.get("status") == "transcribed":
                continue
            # A watch-page playability check already proved these gone; do not
            # spend retries re-asking Gemini for a video nobody can fetch.
            if prior and prior.get("status") == "video_unavailable":
                continue
            # A confirmed caption-less video is not re-asked of YouTube while
            # the Gemini tier is still off; blocked or failed rows are retried.
            if prior and prior.get("status") == "no_caption_gemini_disabled" and config.gemini is None:
                continue
            out.append(row)
        return out

    def record(row: Mapping[str, Any], video: HarvestedVideo, outcome: Mapping[str, Any]) -> None:
        rows[video.video_id] = {
            "video_id": video.video_id,
            "title": video.title,
            "channel": video.channel,
            "video_url": sanitize_media_url(video.url),
            **{key: value for key, value in row.items() if key not in {"video_id", "title", "channel"}},
            **outcome,
            "processed_at": _utc_now_iso(),
        }

    def transcribe_one(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], HarvestedVideo, dict[str, Any]]:
        video = _video_from_row(row)
        prior = rows.get(video.video_id)
        known_no_caption = bool(prior and prior.get("youtube_outcome") == "no_caption_track")
        outcome = acquire_transcript(
            video,
            captions_dir=captions_dir,
            transcripts_dir=transcripts_dir,
            runner=runner,
            yt_dlp_cmd=yt_dlp_cmd,
            youtube_state=youtube_state,
            langs=config.langs,
            gemini=config.gemini,
            http_get=http_get,
            known_no_caption=known_no_caption,
            gemini_only=config.gemini_only,
        )
        return row, video, outcome

    if config.concurrency > 1:
        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            for row, video, outcome in pool.map(_swallow_budget_halt(transcribe_one), pending_rows()):
                if outcome.get("status") == "gemini_budget_halted":
                    halted_reason = str(outcome.get("error") or "Gemini budget exhausted")
                    continue
                with write_lock:
                    record(row, video, outcome)
                    write_inventory("running")
    else:
        for row in pending_rows():
            prior = rows.get(str(row["video_id"]))
            if (
                touched_youtube
                and not config.gemini_only
                and not youtube_state.blocked
                and not (prior and prior.get("youtube_outcome") == "no_caption_track")
            ):
                sleeper(random.uniform(config.min_sleep_seconds, config.max_sleep_seconds))
            will_touch_youtube = (
                not config.gemini_only
                and not youtube_state.blocked
                and not (prior and prior.get("youtube_outcome") == "no_caption_track")
            )
            try:
                row, video, outcome = transcribe_one(row)
            except GeminiBudgetExhausted as exc:
                halted_reason = str(exc)
                break
            touched_youtube = touched_youtube or will_touch_youtube
            record(row, video, outcome)
            write_inventory("running")
            if youtube_state.blocked and config.gemini is None:
                halted_reason = (
                    "YouTube bot check blocked the caption lane and Gemini fallback is "
                    "disabled; resume later or enable the Gemini tier"
                )
                break

    remaining = [
        row for row in config.videos
        if rows.get(str(row["video_id"]), {}).get("status") != "transcribed"
    ]
    if halted_reason:
        status = "halted"
    elif remaining:
        status = "partial"
    else:
        status = "succeeded"
    return write_inventory(status)
