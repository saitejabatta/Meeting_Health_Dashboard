# assistant/transcriber.py
"""Live Whisper transcription with rolling transcript deduplication."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".webm"}


@dataclass
class TranscriptSegment:
    """A timestamped transcript segment with optional speaker and emotion metadata."""

    start: float
    end: float
    text: str
    speaker_label: str = "Speaker_A"
    emotion_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this segment."""
        return asdict(self)


class LiveTranscriber:
    """
    Transcribes audio chunks in real time using faster-whisper or openai-whisper.

    The class maintains a rolling transcript and removes duplicate text caused
    by overlapping audio chunks using SequenceMatcher.
    """

    def __init__(self) -> None:
        """Load the configured Whisper backend if available."""
        if load_dotenv is not None:
            load_dotenv()
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self._segments: list[TranscriptSegment] = []
        self._backend = "none"
        self._model = None
        self._load_model()

    def transcribe_chunk(self, audio_bytes: bytes, offset_seconds: float) -> list[TranscriptSegment]:
        """Transcribe a WAV chunk and return timestamp-adjusted transcript segments."""
        if not audio_bytes:
            return []
        if self._model is None:
            logger.warning("No Whisper backend is available; returning no transcript segments.")
            return []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = Path(temp_audio.name)
        try:
            if self._backend == "faster-whisper":
                new_segments = self._transcribe_with_faster_whisper(temp_path, offset_seconds)
            else:
                new_segments = self._transcribe_with_openai_whisper(temp_path, offset_seconds)
            added = self._add_segments_deduplicated(new_segments)
            return added
        except Exception as exc:
            logger.warning("Whisper transcription failed; returning no new segments: %s", exc)
            return []
        finally:
            temp_path.unlink(missing_ok=True)

    def get_full_transcript(self) -> list[TranscriptSegment]:
        """Return the complete transcript so far, deduplicated and sorted."""
        self._segments.sort(key=lambda segment: (segment.start, segment.end))
        return list(self._segments)

    def get_recent_context(self, seconds: int = 120) -> str:
        """Return the last N seconds of transcript formatted by speaker."""
        if not self._segments:
            return ""
        latest_end = max(segment.end for segment in self._segments)
        cutoff = latest_end - seconds
        recent = [segment for segment in self.get_full_transcript() if segment.end >= cutoff]
        return "\n".join(f"{segment.speaker_label}: {segment.text}" for segment in recent if segment.text.strip())

    def save_transcript_json(self, path: Path) -> None:
        """Save the current transcript to JSON for debugging or handoff."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([segment.to_dict() for segment in self.get_full_transcript()], indent=2))

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            self._backend = "faster-whisper"
            return
        except Exception as exc:
            logger.info("faster-whisper unavailable; trying openai-whisper: %s", exc)

        try:
            import whisper

            self._model = whisper.load_model(self.model_name)
            self._backend = "openai-whisper"
        except Exception as exc:
            logger.warning("No Whisper model could be loaded: %s", exc)
            self._model = None
            self._backend = "none"

    def _transcribe_with_faster_whisper(self, audio_path: Path, offset_seconds: float) -> list[TranscriptSegment]:
        raw_segments, _ = self._model.transcribe(str(audio_path), vad_filter=True)
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            text = str(raw.text).strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start=float(raw.start) + offset_seconds,
                        end=float(raw.end) + offset_seconds,
                        text=text,
                    )
                )
        return segments

    def _transcribe_with_openai_whisper(self, audio_path: Path, offset_seconds: float) -> list[TranscriptSegment]:
        result = self._model.transcribe(str(audio_path), fp16=False)
        segments: list[TranscriptSegment] = []
        for raw in result.get("segments", []):
            text = str(raw.get("text", "")).strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start=float(raw.get("start", 0.0)) + offset_seconds,
                        end=float(raw.get("end", 0.0)) + offset_seconds,
                        text=text,
                    )
                )
        if not segments and str(result.get("text", "")).strip():
            text = str(result["text"]).strip()
            segments.append(TranscriptSegment(start=offset_seconds, end=offset_seconds, text=text))
        return segments

    def _add_segments_deduplicated(self, new_segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        added: list[TranscriptSegment] = []
        for segment in new_segments:
            if not segment.text.strip():
                continue
            duplicate = False
            for existing in self._segments[-12:]:
                similarity = SequenceMatcher(None, existing.text.lower(), segment.text.lower()).ratio()
                overlaps_in_time = segment.start <= existing.end + 5 and segment.end >= existing.start - 5
                if similarity >= 0.85 and overlaps_in_time:
                    duplicate = True
                    break
            if not duplicate:
                self._segments.append(segment)
                added.append(segment)
        self._segments.sort(key=lambda item: (item.start, item.end))
        return added


def normalize_audio_file_to_wav(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert wav, mp3, mp4, m4a, or webm audio to 16kHz mono WAV using ffmpeg."""
    input_path = input_path.expanduser().resolve()
    if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {input_path.suffix}. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".16k_mono.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to convert audio files. Install ffmpeg and try again.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to convert {input_path}: {exc.stderr}") from exc
    return output_path
