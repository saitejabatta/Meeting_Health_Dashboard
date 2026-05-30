# integrations/__init__.py
"""Virtual meeting integrations for the AI Meeting Assistant."""

from integrations.audio_router import AudioRouter, test_audio_routing
from integrations.gmeet_bot import GoogleMeetBot
from integrations.teams_bot import TeamsBot
from integrations.zoom_bot import ZoomBot

__all__ = ["AudioRouter", "GoogleMeetBot", "TeamsBot", "ZoomBot", "test_audio_routing"]
