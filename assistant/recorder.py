# assistant/recorder.py
"""Meeting audio recording, transcript export, and session metadata saves."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, is_dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.briefing import Briefing, briefing_to_dict
from assistant.memory import RuntimeMemory, runtime_memory_to_dict
from assistant.transcriber import TranscriptSegment


class Recorder:
    """
    Saves raw audio, compressed audio, structured transcripts, and session metadata.

    Audio is stored under RECORDINGS_DIR/session_id, defaulting to ./recordings.
    Raw WAV is preserved for the existing dashboard pipeline, while MP3 is
    created for compact playback when ffmpeg is available.
    """

    def __init__(self, recordings_dir: Path | str | None = None) -> None:
        """Initialise recorder paths and WAV writer state."""
        if load_dotenv is not None:
            load_dotenv()
        self.recordings_dir = Path(recordings_dir or os.getenv("RECORDINGS_DIR", "./recordings")).expanduser()
        self.session_dir: Path | None = None
        self.audio_wav_path: Path | None = None
        self._wav_file: wave.Wave_write | None = None

    def start_recording(self, session_id: str) -> None:
        """Open recordings/{session_id}/audio_raw.wav for appending live audio chunks."""
        self.session_dir = self.recordings_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.audio_wav_path = self.session_dir / "audio_raw.wav"
        self._wav_file = wave.open(str(self.audio_wav_path), "wb")
        self._wav_file.setnchannels(1)
        self._wav_file.setsampwidth(2)
        self._wav_file.setframerate(16000)

    def write_chunk(self, audio_bytes: bytes) -> None:
        """Append a WAV or raw 16kHz mono PCM audio chunk to the raw WAV file."""
        if self._wav_file is None:
            raise RuntimeError("Recording has not been started.")
        pcm = _extract_pcm_frames(audio_bytes)
        if pcm:
            self._wav_file.writeframes(pcm)

    def stop_recording(self) -> Path:
        """Finalize the raw WAV file, convert it to MP3, and return the MP3 path."""
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None
        if self.audio_wav_path is None:
            raise RuntimeError("Recording has not been started.")
        mp3_path = self.audio_wav_path.with_name("audio.mp3")
        _convert_wav_to_mp3(self.audio_wav_path, mp3_path)
        return mp3_path

    def save_transcript(self, segments: list[TranscriptSegment], session_id: str) -> dict[str, Path]:
        """Save transcript as JSON, TXT, and SRT files."""
        session_dir = self._ensure_session_dir(session_id)
        ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
        json_path = session_dir / "transcript.json"
        txt_path = session_dir / "transcript.txt"
        srt_path = session_dir / "transcript.srt"
        _atomic_write_text(json_path, json.dumps([segment.to_dict() for segment in ordered], indent=2))
        _atomic_write_text(txt_path, _format_transcript_txt(ordered))
        _atomic_write_text(srt_path, _format_transcript_srt(ordered))
        return {"json": json_path, "txt": txt_path, "srt": srt_path}

    def save_session_metadata(
        self,
        briefing: Briefing,
        runtime_memory: RuntimeMemory,
        session_id: str,
        checklist: object | None = None,
    ) -> Path:
        """Save session_metadata.json with briefing, stats, responses, checklist, and talking-point state."""
        session_dir = self._ensure_session_dir(session_id)
        transcript = runtime_memory.full_transcript
        duration_seconds = _duration_seconds(transcript)
        metadata = {
            "session_id": session_id,
            "briefing": briefing_to_dict(briefing),
            "duration_seconds": duration_seconds,
            "speaker_count": len({segment.speaker_label for segment in transcript if segment.speaker_label}),
            "total_words_spoken": sum(len(segment.text.split()) for segment in transcript),
            "agent_responses_given": {
                "count": len(runtime_memory.agent_responses_given),
                "log": [_to_primitive(item) for item in runtime_memory.agent_responses_given],
            },
            "checklist": _checklist_to_dict(checklist),
            "talking_points_status": {
                "raised": list(runtime_memory.talking_points_raised),
                "pending": list(runtime_memory.talking_points_pending),
            },
            "runtime_memory": runtime_memory_to_dict(runtime_memory),
        }
        path = session_dir / "session_metadata.json"
        _atomic_write_text(path, json.dumps(metadata, indent=2, default=str))
        return path

    def _ensure_session_dir(self, session_id: str) -> Path:
        if self.session_dir is None or self.session_dir.name != session_id:
            self.session_dir = self.recordings_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir


def generate_session_id(meeting_title: str, started_at: datetime | None = None) -> str:
    """Generate a session id as YYYYMMDD_HHMM_meeting_title_slug."""
    timestamp = started_at or datetime.now()
    slug = slugify(meeting_title) or "untitled_meeting"
    return f"{timestamp:%Y%m%d_%H%M}_{slug}"


def slugify(value: str) -> str:
    """Convert text to a lowercase filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return re.sub(r"_+", "_", slug)[:80]


def _extract_pcm_frames(audio_bytes: bytes) -> bytes:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as input_wav:
            frames = input_wav.readframes(input_wav.getnframes())
            if input_wav.getnchannels() == 1 and input_wav.getsampwidth() == 2 and input_wav.getframerate() == 16000:
                return frames
    except Exception:
        return audio_bytes
    return frames


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        shutil.copyfile(wav_path, mp3_path.with_suffix(".wav.copy"))
        return
    temp_mp3 = mp3_path.with_suffix(".tmp.mp3")
    command = ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "128k", str(temp_mp3)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temp_mp3.replace(mp3_path)
    except subprocess.CalledProcessError as exc:
        if temp_mp3.exists():
            temp_mp3.unlink()
        raise RuntimeError(f"ffmpeg failed to convert WAV to MP3: {exc.stderr}") from exc


def _format_transcript_txt(segments: list[TranscriptSegment]) -> str:
    lines = []
    for segment in segments:
        lines.append(f"[{_format_timestamp_txt(segment.start)}] {segment.speaker_label}: {segment.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _format_transcript_srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp_srt(segment.start)
        end = _format_timestamp_srt(max(segment.end, segment.start + 0.5))
        blocks.append(f"{index}\n{start} --> {end}\n{segment.speaker_label}: {segment.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _format_timestamp_txt(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_timestamp_srt(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _duration_seconds(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(segment.end for segment in segments) - min(segment.start for segment in segments)


def _checklist_to_dict(checklist: object | None) -> object | None:
    if checklist is None:
        return None
    if hasattr(checklist, "to_dict"):
        return checklist.to_dict()
    return _to_primitive(checklist)


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)
