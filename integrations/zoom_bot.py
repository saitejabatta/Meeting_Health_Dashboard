# integrations/zoom_bot.py
"""Zoom joining helper with virtual-audio fallback instructions."""

from __future__ import annotations

import os
import subprocess
import urllib.parse
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from integrations.audio_router import AudioRouter


@dataclass
class JoinResult:
    """Result of attempting to join a virtual meeting."""

    joined: bool
    message: str


class ZoomBot:
    """Launches Zoom meetings when possible and explains manual fallback steps."""

    def __init__(self) -> None:
        """Load credentials and audio routing configuration."""
        if load_dotenv is not None:
            load_dotenv()
        self.email = os.getenv("ZOOM_EMAIL", "")
        self.password = os.getenv("ZOOM_PASSWORD", "")
        self.audio_router = AudioRouter()

    def join(self, meeting_url: str, display_name: str = "") -> JoinResult:
        """Join a Zoom meeting URL using the Zoom URL scheme when available."""
        routing = self.audio_router.check()
        if not routing.ok:
            return JoinResult(False, routing.message)
        zoom_url = _to_zoom_url_scheme(meeting_url, display_name=display_name)
        try:
            subprocess.Popen(["open", zoom_url])
            credentials_note = " SSO or 2FA may require manual sign-in." if not (self.email and self.password) else ""
            name_note = f" Requested display name '{display_name}'." if display_name else ""
            return JoinResult(True, f"Opened Zoom via URL scheme.{name_note} Route Zoom audio to {routing.device_name}.{credentials_note}")
        except Exception as exc:
            return JoinResult(False, self.manual_join_instructions(meeting_url, str(exc), display_name=display_name))

    def leave(self) -> JoinResult:
        """Return manual leave instructions for Zoom."""
        return JoinResult(True, "Leave the Zoom meeting from the Zoom client when !end is called.")

    def manual_join_instructions(self, meeting_url: str, reason: str = "", display_name: str = "") -> str:
        """Return instructions for manual Zoom joining and virtual audio routing."""
        reason_text = f" Automatic launch failed: {reason}" if reason else ""
        name_text = f" Use display name '{display_name}' so participants know an assistant is attending." if display_name else ""
        return (
            f"Join Zoom manually at {meeting_url}.{reason_text}{name_text} "
            f"Set speaker output to your virtual device and set VIRTUAL_AUDIO_DEVICE_NAME={self.audio_router.device_name}."
        )


def _to_zoom_url_scheme(meeting_url: str, display_name: str = "") -> str:
    if meeting_url.startswith("zoommtg://"):
        return meeting_url
    parsed = urllib.parse.urlparse(meeting_url)
    query = urllib.parse.parse_qs(parsed.query)
    meeting_id = ""
    if "/j/" in parsed.path:
        meeting_id = parsed.path.rsplit("/j/", 1)[-1].split("/")[0]
    password = query.get("pwd", [""])[0]
    params = {"confno": meeting_id}
    if password:
        params["pwd"] = password
    if display_name:
        params["uname"] = display_name
    return "zoommtg://zoom.us/join?" + urllib.parse.urlencode(params)
