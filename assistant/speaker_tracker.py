# assistant/speaker_tracker.py
"""Live speaker diarization and speaker statistics for meeting audio."""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from array import array
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


@dataclass
class SpeakerStats:
    """Aggregate speaking statistics for one speaker label."""

    total_talk_time: float = 0.0
    turn_count: int = 0
    avg_turn_duration: float = 0.0
    last_spoke_at: float = 0.0


class SpeakerTracker:
    """
    Assigns stable speaker labels to live transcript segments.

    pyannote.audio is used when installed and HF_TOKEN is configured. If it is
    unavailable, a single-speaker energy-based fallback keeps timestamps and
    stats usable for downstream features.
    """

    def __init__(self) -> None:
        """Initialise diarization backend, label map, and stats store."""
        if load_dotenv is not None:
            load_dotenv()
        self.sample_rate = 16000
        self._pipeline = None
        self._next_label_index = 0
        self._backend_label_map: dict[str, str] = {}
        self._name_map: dict[str, str] = {}
        self._stats: dict[str, SpeakerStats] = {}
        self._load_pipeline()

    def process_chunk(self, audio_bytes: bytes, offset_seconds: float) -> dict[float, str]:
        """Return a timestamp-to-speaker-label mapping for an audio chunk."""
        if not audio_bytes:
            return {}
        if self._pipeline is None:
            return self._process_with_energy_fallback(audio_bytes, offset_seconds)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = Path(temp_audio.name)
        try:
            diarization = self._pipeline(str(temp_path))
            mapping: dict[float, str] = {}
            for turn, _, backend_label in diarization.itertracks(yield_label=True):
                stable_label = self._stable_label(str(backend_label))
                display_label = self._name_map.get(stable_label, stable_label)
                start = float(turn.start) + offset_seconds
                end = float(turn.end) + offset_seconds
                mapping[start] = display_label
                self._update_stats(display_label, max(0.0, end - start), end)
            return dict(sorted(mapping.items()))
        except Exception as exc:
            logger.warning("pyannote diarization failed; using fallback speaker labels: %s", exc)
            return self._process_with_energy_fallback(audio_bytes, offset_seconds)
        finally:
            temp_path.unlink(missing_ok=True)

    def get_speaker_stats(self) -> dict[str, SpeakerStats]:
        """Return per-speaker talk-time, turn count, average turn duration, and recency."""
        return dict(self._stats)

    def map_names(self, name_map: dict[str, str]) -> None:
        """Rename stable labels such as Speaker_A to confirmed attendee names."""
        cleaned = {key.strip(): value.strip() for key, value in name_map.items() if key.strip() and value.strip()}
        if not cleaned:
            return
        self._name_map.update(cleaned)
        for old_label, new_label in cleaned.items():
            if old_label in self._stats:
                self._stats[new_label] = self._stats.pop(old_label)

    def current_active_speaker(self) -> str | None:
        """Return the speaker who most recently spoke, if known."""
        if not self._stats:
            return None
        return max(self._stats.items(), key=lambda item: item[1].last_spoke_at)[0]

    def _load_pipeline(self) -> None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.info("HF_TOKEN not configured; speaker tracking will use local fallback.")
            return
        try:
            from pyannote.audio import Pipeline

            model_name = os.getenv("PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
            self._pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        except Exception as exc:
            logger.warning("Could not load pyannote diarization pipeline: %s", exc)
            self._pipeline = None

    def _stable_label(self, backend_label: str) -> str:
        if backend_label in self._backend_label_map:
            return self._backend_label_map[backend_label]
        label = f"Speaker_{chr(ord('A') + self._next_label_index)}"
        self._next_label_index += 1
        self._backend_label_map[backend_label] = label
        return label

    def _process_with_energy_fallback(self, audio_bytes: bytes, offset_seconds: float) -> dict[float, str]:
        duration = _wav_duration_seconds(audio_bytes)
        has_voice = _wav_has_voice(audio_bytes)
        if not has_voice or duration <= 0.0:
            return {}
        label = self._name_map.get("Speaker_A", "Speaker_A")
        self._update_stats(label, duration, offset_seconds + duration)
        return {offset_seconds: label}

    def _update_stats(self, speaker_label: str, duration: float, last_spoke_at: float) -> None:
        stats = self._stats.setdefault(speaker_label, SpeakerStats())
        stats.total_talk_time += duration
        stats.turn_count += 1
        stats.last_spoke_at = max(stats.last_spoke_at, last_spoke_at)
        stats.avg_turn_duration = stats.total_talk_time / stats.turn_count if stats.turn_count else 0.0


def assign_speakers_to_segments(timestamp_map: dict[float, str], segment_starts: list[float]) -> dict[float, str]:
    """Assign each segment start timestamp to the nearest known speaker label at or before it."""
    if not timestamp_map:
        return {start: "Speaker_A" for start in segment_starts}
    sorted_points = sorted(timestamp_map.items())
    assignments: dict[float, str] = {}
    for start in segment_starts:
        label = sorted_points[0][1]
        for timestamp, candidate_label in sorted_points:
            if timestamp <= start:
                label = candidate_label
            else:
                break
        assignments[start] = label
    return assignments


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / rate if rate else 0.0
    except Exception:
        return 0.0


def _wav_has_voice(audio_bytes: bytes, threshold: float = 250.0) -> bool:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
        if not frames:
            return False
        samples = array("h")
        samples.frombytes(frames)
        if not samples:
            return False
        square_sum = sum(float(sample) * float(sample) for sample in samples)
        rms = (square_sum / len(samples)) ** 0.5
        return rms >= threshold
    except Exception:
        return bool(audio_bytes)
