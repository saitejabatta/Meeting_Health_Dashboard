# assistant/responder.py
"""Response timing, text-to-speech, and private advisor output."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.briefing import AgentMode

logger = logging.getLogger(__name__)


class Responder:
    """
    Controls when and how the agent speaks.

    Speech uses ElevenLabs when configured, then pyttsx3 as a local fallback,
    then a terminal print fallback. Advisor messages are always private text.
    """

    def __init__(self, audio_capture: object | None = None) -> None:
        """Create a responder with optional access to recent audio levels."""
        if load_dotenv is not None:
            load_dotenv()
        self.audio_capture = audio_capture
        self.virtual_audio_device_name = os.getenv("VIRTUAL_AUDIO_DEVICE_NAME", "").strip()
        self.silence_rms_threshold = float(os.getenv("SILENCE_RMS_THRESHOLD", "350"))
        self.spoken_log: list[dict[str, object]] = []

    def wait_for_speaking_gap(self, min_silence_ms: int = 800) -> None:
        """Block briefly until recent captured audio appears silent."""
        deadline = time.time() + max(0.8, min_silence_ms / 1000 * 5)
        required_seconds = min_silence_ms / 1000
        silent_since: float | None = None
        while time.time() < deadline:
            if self._recent_audio_rms() <= self.silence_rms_threshold:
                if silent_since is None:
                    silent_since = time.time()
                if time.time() - silent_since >= required_seconds:
                    return
            else:
                silent_since = None
            time.sleep(0.05)

    def speak(self, text: str, mode: AgentMode) -> None:
        """Speak aloud for voice modes, print privately for advisor mode, and no-op for silent mode."""
        cleaned = text.strip()
        if not cleaned or mode == AgentMode.SILENT_OBSERVER:
            return
        if mode == AgentMode.ADVISOR:
            self.whisper_to_user(cleaned)
            return
        self.wait_for_speaking_gap()
        audio_path = self._synthesize_speech(cleaned)
        if audio_path is not None:
            self._play_audio(audio_path)
        else:
            print(f"\nAgent says: {cleaned}\n")
        self.spoken_log.append({"timestamp": time.time(), "mode": mode.value, "text": cleaned})

    def whisper_to_user(self, tip: str) -> None:
        """Print a private formatted tip to the user's terminal."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            Console().print(Panel(tip, title="Assistant", border_style="cyan"))
        except Exception:
            print(f"\n[Assistant] {tip}\n")

    def _synthesize_speech(self, text: str) -> Path | None:
        elevenlabs_path = self._synthesize_with_elevenlabs(text)
        if elevenlabs_path is not None:
            return elevenlabs_path
        return self._synthesize_with_pyttsx3(text)

    def _synthesize_with_elevenlabs(self, text: str) -> Path | None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if not api_key or not voice_id:
            return None
        try:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=api_key)
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
                text=text,
            )
            path = Path(tempfile.mkstemp(suffix=".mp3")[1])
            with path.open("wb") as handle:
                for chunk in audio:
                    handle.write(chunk)
            return path
        except Exception as exc:
            logger.warning("ElevenLabs TTS failed; falling back locally: %s", exc)
            return None

    def _synthesize_with_pyttsx3(self, text: str) -> Path | None:
        try:
            import pyttsx3

            path = Path(tempfile.mkstemp(suffix=".wav")[1])
            engine = pyttsx3.init()
            engine.save_to_file(text, str(path))
            engine.runAndWait()
            return path if path.exists() and path.stat().st_size > 0 else None
        except Exception as exc:
            logger.warning("pyttsx3 TTS failed; using terminal fallback: %s", exc)
            return None

    def _play_audio(self, audio_path: Path) -> None:
        try:
            from pydub import AudioSegment
            from pydub.playback import play

            play(AudioSegment.from_file(audio_path))
        except Exception as exc:
            logger.warning("Audio playback failed; speech text was logged instead: %s", exc)

    def _recent_audio_rms(self) -> float:
        if self.audio_capture is None or not hasattr(self.audio_capture, "get_full_recording"):
            return 0.0
        try:
            audio = self.audio_capture.get_full_recording()
            with wave.open(_BytesReader(audio), "rb") as wav_file:
                frame_count = min(wav_file.getnframes(), int(wav_file.getframerate() * 0.5))
                wav_file.setpos(max(0, wav_file.getnframes() - frame_count))
                frames = wav_file.readframes(frame_count)
            return _rms_int16(frames)
        except Exception:
            return 0.0


class _BytesReader:
    """Small seekable byte reader for wave.open."""

    def __init__(self, data: bytes) -> None:
        """Store bytes for file-like reads."""
        self._data = data
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the current position."""
        if size < 0:
            size = len(self._data) - self._position
        start = self._position
        end = min(len(self._data), start + size)
        self._position = end
        return self._data[start:end]

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek within the byte stream."""
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = len(self._data) + offset
        self._position = max(0, min(len(self._data), self._position))
        return self._position

    def tell(self) -> int:
        """Return the current byte position."""
        return self._position


def _rms_int16(frames: bytes) -> float:
    if not frames:
        return 0.0
    sample_count = len(frames) // 2
    if sample_count == 0:
        return 0.0
    total = 0.0
    for index in range(0, len(frames) - 1, 2):
        sample = int.from_bytes(frames[index : index + 2], "little", signed=True)
        total += sample * sample
    return (total / sample_count) ** 0.5
