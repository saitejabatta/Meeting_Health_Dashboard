# assistant/__init__.py
"""AI Meeting Assistant package for the Meeting Health Dashboard."""

__all__ = [
    "ActionItem",
    "AgentMemory",
    "AgentMode",
    "AudioCapture",
    "Briefing",
    "ChecklistItem",
    "CommandParser",
    "DoNotMissChecklist",
    "Flag",
    "KeyMoment",
    "LiveTranscriber",
    "MeetingAgent",
    "MeetingSession",
    "Recorder",
    "Responder",
    "RuntimeMemory",
    "SessionReport",
    "SpeakerStats",
    "SpeakerTracker",
    "SpeakDecision",
    "TranscriptSegment",
    "collect_briefing_interactive",
    "generate_session_id",
    "learn_from_briefing",
    "parse_briefing_from_text",
    "slugify",
]

_LAZY_IMPORTS = {
    "ActionItem": ("assistant.memory", "ActionItem"),
    "AgentMemory": ("assistant.memory", "AgentMemory"),
    "AgentMode": ("assistant.briefing", "AgentMode"),
    "AudioCapture": ("assistant.audio_capture", "AudioCapture"),
    "Briefing": ("assistant.briefing", "Briefing"),
    "ChecklistItem": ("assistant.session", "ChecklistItem"),
    "CommandParser": ("assistant.session", "CommandParser"),
    "DoNotMissChecklist": ("assistant.session", "DoNotMissChecklist"),
    "Flag": ("assistant.agent", "Flag"),
    "KeyMoment": ("assistant.memory", "KeyMoment"),
    "LiveTranscriber": ("assistant.transcriber", "LiveTranscriber"),
    "MeetingAgent": ("assistant.agent", "MeetingAgent"),
    "MeetingSession": ("assistant.session", "MeetingSession"),
    "Recorder": ("assistant.recorder", "Recorder"),
    "Responder": ("assistant.responder", "Responder"),
    "RuntimeMemory": ("assistant.memory", "RuntimeMemory"),
    "SessionReport": ("assistant.session", "SessionReport"),
    "SpeakerStats": ("assistant.speaker_tracker", "SpeakerStats"),
    "SpeakerTracker": ("assistant.speaker_tracker", "SpeakerTracker"),
    "SpeakDecision": ("assistant.agent", "SpeakDecision"),
    "TranscriptSegment": ("assistant.transcriber", "TranscriptSegment"),
    "collect_briefing_interactive": ("assistant.briefing", "collect_briefing_interactive"),
    "generate_session_id": ("assistant.recorder", "generate_session_id"),
    "learn_from_briefing": ("assistant.memory", "learn_from_briefing"),
    "parse_briefing_from_text": ("assistant.briefing", "parse_briefing_from_text"),
    "slugify": ("assistant.recorder", "slugify"),
}


def __getattr__(name: str) -> object:
    """Load assistant exports lazily to avoid import cycles between packages."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'assistant' has no attribute {name!r}")
    module_name, attribute = _LAZY_IMPORTS[name]
    module = __import__(module_name, fromlist=[attribute])
    value = getattr(module, attribute)
    globals()[name] = value
    return value
