# integrations/audio_router.py
"""Virtual audio device discovery and routing diagnostics."""

from __future__ import annotations

import math
import os
import platform
import struct
import sys
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@dataclass
class AudioRoutingStatus:
    """Result of checking virtual audio routing configuration."""

    ok: bool
    platform_name: str
    device_name: str
    message: str


class AudioRouter:
    """Abstracts virtual audio device setup across macOS, Windows, and Linux."""

    def __init__(self) -> None:
        """Load environment and infer platform-specific virtual device names."""
        if load_dotenv is not None:
            load_dotenv()
        self.platform_name = platform.system()
        self.device_name = os.getenv("VIRTUAL_AUDIO_DEVICE_NAME", self.default_device_name()).strip()

    def default_device_name(self) -> str:
        """Return the conventional virtual audio device name for this OS."""
        if self.platform_name == "Darwin":
            return "BlackHole 2ch"
        if self.platform_name == "Windows":
            return "CABLE Output (VB-Audio Virtual Cable)"
        if self.platform_name == "Linux":
            return "meeting_monitor.monitor"
        return ""

    def setup_instructions(self) -> str:
        """Return setup instructions for the current platform."""
        if self.platform_name == "Darwin":
            return (
                "Install BlackHole-2ch with Homebrew, create a Multi-Output Device in Audio MIDI Setup "
                "combining BlackHole and your speakers, then set VIRTUAL_AUDIO_DEVICE_NAME=BlackHole 2ch."
            )
        if self.platform_name == "Windows":
            return (
                "Install VB-Cable, route meeting output to CABLE Input, and set "
                "VIRTUAL_AUDIO_DEVICE_NAME=CABLE Output (VB-Audio Virtual Cable)."
            )
        if self.platform_name == "Linux":
            return (
                "Run: pactl load-module module-null-sink sink_name=meeting_monitor "
                "sink_properties=device.description=meeting_monitor; then set "
                "VIRTUAL_AUDIO_DEVICE_NAME=meeting_monitor.monitor."
            )
        return "Configure a virtual audio input device and set VIRTUAL_AUDIO_DEVICE_NAME in .env."

    def find_device(self) -> int | None:
        """Return the sounddevice input index for the configured virtual device, if found."""
        try:
            import sounddevice as sd
        except ImportError:
            return None
        needle = self.device_name.lower()
        for index, device in enumerate(sd.query_devices()):
            name = str(device.get("name", "")).lower()
            if needle in name and int(device.get("max_input_channels", 0)) > 0:
                return index
        return None

    def check(self) -> AudioRoutingStatus:
        """Check whether the configured virtual input device is visible."""
        if not self.device_name:
            return AudioRoutingStatus(False, self.platform_name, "", self.setup_instructions())
        device_index = self.find_device()
        if device_index is None:
            return AudioRoutingStatus(
                False,
                self.platform_name,
                self.device_name,
                f"Virtual audio device '{self.device_name}' was not found. {self.setup_instructions()}",
            )
        return AudioRoutingStatus(True, self.platform_name, self.device_name, f"Found virtual audio device at index {device_index}.")

    def test_audio_routing(self) -> AudioRoutingStatus:
        """Play a one-second tone and verify the virtual input device is configured."""
        status = self.check()
        if not status.ok:
            return status
        try:
            import sounddevice as sd

            sample_rate = 16000
            tone = [
                struct.unpack("<h", struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))))[0]
                for i in range(sample_rate)
            ]
            sd.play(tone, samplerate=sample_rate)
            sd.wait()
            return AudioRoutingStatus(True, self.platform_name, self.device_name, "Played a one-second routing test tone.")
        except Exception as exc:
            return AudioRoutingStatus(False, self.platform_name, self.device_name, f"Could not play test tone: {exc}")


def test_audio_routing() -> AudioRoutingStatus:
    """Run a virtual audio routing check and tone test."""
    return AudioRouter().test_audio_routing()


def main() -> None:
    """CLI entry point for audio routing diagnostics."""
    status = test_audio_routing()
    print(status.message)
    raise SystemExit(0 if status.ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        main()
    main()
