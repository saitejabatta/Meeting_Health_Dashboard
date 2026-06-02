# assistant/audio_capture.py
"""Real-time microphone and system audio capture for meeting sessions."""

from __future__ import annotations

import io
import logging
import os
import queue
import threading
import wave
from dataclasses import dataclass
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


class AudioCaptureError(RuntimeError):
    """Raised when live audio capture cannot be started or read."""


@dataclass
class AudioCaptureStats:
    """Small diagnostic snapshot for live audio capture."""

    started: bool
    captured_seconds: float
    queued_chunks: int
    rms: float
    peak: int


class AudioCapture:
    """
    Captures audio from microphone and/or system audio.

    Captured audio is converted to 16kHz mono 16-bit PCM and emitted as WAV
    chunks through a thread-safe queue. System capture uses the virtual audio
    device named by VIRTUAL_AUDIO_DEVICE_NAME.
    """

    def __init__(self) -> None:
        """Initialise capture buffers, queue, and environment-driven settings."""
        if load_dotenv is not None:
            load_dotenv()
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2
        self.chunk_seconds = max(1, int(os.getenv("AUDIO_CHUNK_SECONDS", "30")))
        self.overlap_seconds = max(0, min(5, int(os.getenv("AUDIO_OVERLAP_SECONDS", "5"))))
        self.virtual_audio_device_name = os.getenv("VIRTUAL_AUDIO_DEVICE_NAME", "").strip()
        self._chunk_bytes = self.chunk_seconds * self.sample_rate * self.sample_width
        self._overlap_bytes = self.overlap_seconds * self.sample_rate * self.sample_width
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._streams: list[object] = []
        self._lock = threading.Lock()
        self._capture_buffer = bytearray()
        self._full_pcm = bytearray()
        self._started = False

    def start(self, source: Literal["mic", "system", "both"]) -> None:
        """Open audio streams for mic, system audio, or both."""
        if source not in {"mic", "system", "both"}:
            raise AudioCaptureError(f"Unsupported audio source: {source}")
        if self._started:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioCaptureError(
                "sounddevice is required for live capture. Install requirements_assistant.txt "
                "and ensure PortAudio is available on your system."
            ) from exc

        selected_devices: list[int | None] = []
        if source in {"mic", "both"}:
            selected_devices.append(None)
        if source in {"system", "both"}:
            selected_devices.append(self._find_system_audio_device(sd))

        try:
            for device in selected_devices:
                stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    blocksize=0,
                    device=device,
                    callback=self._on_audio_data,
                )
                stream.start()
                self._streams.append(stream)
            self._started = True
            logger.info("Audio capture started for source=%s", source)
        except Exception as exc:
            self.stop()
            raise AudioCaptureError(f"Could not start audio capture: {exc}") from exc

    def stop(self) -> None:
        """Flush queued audio and close any open streams."""
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                logger.warning("Error closing audio stream: %s", exc)
        self._streams = []
        self._started = False
        with self._lock:
            if self._capture_buffer:
                self._queue.put(_pcm_to_wav(bytes(self._capture_buffer), self.sample_rate, self.channels, self.sample_width))
                self._capture_buffer.clear()

    def get_chunk(self) -> bytes | None:
        """Return the next WAV chunk, or None if no complete chunk is available."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def get_full_recording(self) -> bytes:
        """Return the full captured session as a 16kHz mono WAV byte string."""
        with self._lock:
            return _pcm_to_wav(bytes(self._full_pcm), self.sample_rate, self.channels, self.sample_width)

    def get_stats(self) -> AudioCaptureStats:
        """Return capture diagnostics without consuming queued audio."""
        with self._lock:
            pcm = bytes(self._full_pcm)
        frame_count = len(pcm) // self.sample_width // self.channels
        return AudioCaptureStats(
            started=self._started,
            captured_seconds=frame_count / self.sample_rate if self.sample_rate else 0.0,
            queued_chunks=self._queue.qsize(),
            rms=_rms_int16(pcm[-self.sample_rate * self.sample_width * self.channels :]),
            peak=_peak_int16(pcm[-self.sample_rate * self.sample_width * self.channels :]),
        )

    def _on_audio_data(self, indata: bytes, frames: int, time_info: object, status: object) -> None:
        if status:
            logger.debug("Audio stream status: %s", status)
        with self._lock:
            self._capture_buffer.extend(indata)
            self._full_pcm.extend(indata)
            while len(self._capture_buffer) >= self._chunk_bytes:
                chunk_pcm = bytes(self._capture_buffer[: self._chunk_bytes])
                self._queue.put(_pcm_to_wav(chunk_pcm, self.sample_rate, self.channels, self.sample_width))
                keep_start = max(0, self._chunk_bytes - self._overlap_bytes)
                self._capture_buffer = self._capture_buffer[keep_start:]

    def _find_system_audio_device(self, sounddevice_module: object) -> int:
        if not self.virtual_audio_device_name:
            raise AudioCaptureError(
                "VIRTUAL_AUDIO_DEVICE_NAME is required for system audio capture. "
                "Use BlackHole on macOS, VB-Cable on Windows, or a PulseAudio null sink on Linux."
            )
        devices = sounddevice_module.query_devices()
        needle = self.virtual_audio_device_name.lower()
        for index, device in enumerate(devices):
            name = str(device.get("name", "")).lower()
            max_input_channels = int(device.get("max_input_channels", 0))
            if needle in name and max_input_channels > 0:
                return index
        available = ", ".join(str(device.get("name", "")) for device in devices if int(device.get("max_input_channels", 0)) > 0)
        raise AudioCaptureError(
            f"Virtual audio input device '{self.virtual_audio_device_name}' was not found. "
            f"Available input devices: {available or 'none detected'}"
        )


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


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


def _peak_int16(frames: bytes) -> int:
    peak = 0
    for index in range(0, len(frames) - 1, 2):
        sample = abs(int.from_bytes(frames[index : index + 2], "little", signed=True))
        peak = max(peak, sample)
    return peak
