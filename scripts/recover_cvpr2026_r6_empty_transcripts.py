#!/usr/bin/env python3
"""Definitive recovery + honest gap-marking for empty r6 transcripts.

The bulk ASR driver recorded some acquired audios as succeeded with EMPTY
text: OpenAI returned no transcript. Two distinct causes exist and must not be
conflated:

  * Genuinely silent audio (yt-dlp captured a silent track): no transcript is
    possible, ever.
  * Real audio that produced no speech transcript on the first upload.

For every currently-empty transcript this pass re-encodes the source to clean
16 kHz mono AAC (which also sidesteps any fragmented-container upload quirk)
and re-transcribes. If speech comes back, the real transcript replaces the
empty sidecar. If it is still empty, the sidecar is stamped with an explicit
``empty_reason`` (``silent_audio`` when the source max volume is below -60 dB,
else ``no_speech_detected``) so downstream treats it as a coverage gap, never
as a real transcript. Idempotent: already-populated sidecars are left alone.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.config import load_settings
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.transcription import create_transcription_service

TASK_ID = "transcription.session_speech"
SILENCE_DB = -60.0


def _max_volume_db(path: Path) -> float | None:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    m = re.search(r"max_volume:\s*([\-0-9.]+)\s*dB", out)
    return float(m.group(1)) if m else None


async def _main() -> int:
    contract = R6ArtifactContract.production()
    workspace = contract.workspace
    out_dir = workspace / contract.relative_path("speech_text")
    recs = {
        r["item_id"]: r
        for r in json.loads(contract.path("audio_recovery_receipt").read_bytes())["records"]
        if r.get("terminal_state") == "acquired"
    }

    empty_files = []
    for f in sorted(out_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("text") and d.get("transcript_source") == "asr_fallback":
            empty_files.append((f, d))
    print(f"empty transcripts to recover: {len(empty_files)}")

    model = default_model_for_provider(TASK_ID, "openai")
    settings = load_settings()
    settings.openai_model = model
    service = create_transcription_service(settings)
    ffmpeg = shutil.which("ffmpeg")

    recovered = 0
    silent = 0
    no_speech = 0
    with provider_receipt_context(
        workspace_root=workspace, run_id=contract.run_id,
        stage_id="transcription-ocr-r6", caller="transcription_ocr.cvpr2026.r6.empty_recovery",
        task_id=TASK_ID,
    ):
        for out_path, payload in empty_files:
            row = recs.get(payload["item_id"])
            if row is None:
                continue
            src = workspace / row["local_media_path"] if not Path(row["local_media_path"]).is_absolute() else Path(row["local_media_path"])
            max_db = _max_volume_db(src)
            text = ""
            tmp = Path(tempfile.mktemp(suffix=".m4a"))
            try:
                proc = subprocess.run(
                    [ffmpeg, "-nostdin", "-v", "error", "-i", str(src),
                     "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "96k", str(tmp), "-y"],
                    capture_output=True, text=True,
                )
                if proc.returncode == 0 and tmp.is_file():
                    chunk = await service.transcribe_single(tmp)
                    t = chunk.text or ""
                    if not t.startswith("[Transcription error"):
                        text = t
            finally:
                tmp.unlink(missing_ok=True)

            if text:
                payload["text"] = text
                payload["char_count"] = len(text)
                payload.pop("empty_reason", None)
                recovered += 1
                print(f"  recovered {payload['video_id']} ({len(text)} chars)", flush=True)
            else:
                reason = "silent_audio" if (max_db is not None and max_db < SILENCE_DB) else "no_speech_detected"
                payload["char_count"] = 0
                payload["empty_reason"] = reason
                payload["source_max_volume_db"] = max_db
                if reason == "silent_audio":
                    silent += 1
                else:
                    no_speech += 1
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "empty_examined": len(empty_files),
        "recovered_with_speech": recovered,
        "silent_audio_gaps": silent,
        "no_speech_detected_gaps": no_speech,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
