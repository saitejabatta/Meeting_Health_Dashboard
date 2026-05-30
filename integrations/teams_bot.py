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

    def join(self, meeting_url: str) -> JoinResult:
        """Open a Teams meeting URL using the OS URL handler."""
        routing = self.audio_router.check()
        if not routing.ok:
            return JoinResult(False, routing.message)
        try:
            subprocess.Popen(["open", meeting_url])
            credentials_note = " Work SSO or MFA may require manual sign-in." if not (self.email and self.password) else ""
            return JoinResult(True, f"Opened Teams meeting URL. Route Teams audio to {routing.device_name}.{credentials_note}")
        except Exception as exc:
            return JoinResult(False, self.manual_join_instructions(meeting_url, str(exc)))

    def leave(self) -> JoinResult:
        """Return manual leave instructions for Teams."""
        return JoinResult(True, "Leave the Teams meeting from the Teams client when !end is called.")

    def manual_join_instructions(self, meeting_url: str, reason: str = "") -> str:
        """Return manual Teams joining and routing instructions."""
        reason_text = f" Automatic launch failed: {reason}" if reason else ""
        return (
            f"Join Teams manually at {meeting_url}.{reason_text} "
            f"Set meeting audio output to the virtual device and keep VIRTUAL_AUDIO_DEVICE_NAME={self.audio_router.device_name}."
        )
