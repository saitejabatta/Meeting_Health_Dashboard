# integrations/teams_bot.py
"""Microsoft Teams joining helper with desktop/manual fallback."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from integrations.audio_router import AudioRouter


@dataclass
class JoinResult:
    """Result of attempting to join a Teams meeting."""

    joined: bool
    message: str


class TeamsBot:
    """Launches Teams meeting links and provides virtual-audio fallback steps."""

    def __init__(self) -> None:
        """Load Teams credentials and audio routing configuration."""
        if load_dotenv is not None:
            load_dotenv()
        self.email = os.getenv("TEAMS_EMAIL", "")
        self.password = os.getenv("TEAMS_PASSWORD", "")
        self.audio_router = AudioRouter()

    def join(self, meeting_url: str, display_name: str = "") -> JoinResult:
        """Open a Teams meeting URL using the OS URL handler."""
        try:
            subprocess.Popen(["open", meeting_url])
            credentials_note = " Work SSO or MFA may require manual sign-in." if not (self.email and self.password) else ""
            name_note = f" Join with display name '{display_name}'." if display_name else ""
            routing = self.audio_router.check()
            routing_note = (
                f" Route Teams audio to {routing.device_name}."
                if routing.ok
                else f" After Teams opens, set VIRTUAL_AUDIO_DEVICE_NAME={self.audio_router.device_name} and confirm the device is visible."
            )
            return JoinResult(True, f"Opened Teams meeting URL.{name_note}{routing_note}{credentials_note}")
        except Exception as exc:
            return JoinResult(False, self.manual_join_instructions(meeting_url, str(exc), display_name=display_name))

    def leave(self) -> JoinResult:
        """Return manual leave instructions for Teams."""
        return JoinResult(True, "Leave the Teams meeting from the Teams client when !end is called.")

    def manual_join_instructions(self, meeting_url: str, reason: str = "", display_name: str = "") -> str:
        """Return manual Teams joining and routing instructions."""
        reason_text = f" Automatic launch failed: {reason}" if reason else ""
        name_text = f" Use display name '{display_name}' so participants know an assistant is attending." if display_name else ""
        return (
            f"Join Teams manually at {meeting_url}.{reason_text}{name_text} "
            f"Set meeting audio output to the virtual device and keep VIRTUAL_AUDIO_DEVICE_NAME={self.audio_router.device_name}."
        )
