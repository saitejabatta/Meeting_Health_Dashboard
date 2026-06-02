# assistant/briefing.py
"""Pre-meeting briefing models, parsing, and interactive collection."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Supported operating modes for the meeting assistant."""

    FULL_PROXY = "FULL_PROXY"
    SILENT_OBSERVER = "SILENT_OBSERVER"
    ADVISOR = "ADVISOR"
    PARTICIPATOR = "PARTICIPATOR"


@dataclass
class Briefing:
    """Structured pre-meeting instructions used to configure the assistant."""

    meeting_title: str
    meeting_objective: str
    attendees: list[str]
    user_role: str
    agent_mode: AgentMode
    talking_points: list[str]
    topics_to_avoid: list[str]
    background_context: str
    response_style: str
    custom_instructions: list[str]
    max_speaking_turns: int
    auto_respond: bool
    user_name: str = ""
    speaking_persona: str = ""
    decision_authority: str = ""
    escalation_rules: list[str] = field(default_factory=list)
    assistant_display_name: str = ""


def parse_briefing_from_text(raw_input: str) -> Briefing:
    """Parse free-form user briefing text into a complete Briefing object."""
    _load_environment()
    defaults = _default_briefing(raw_input)
    llm_payload = _parse_with_llm(raw_input)
    if llm_payload is None:
        logger.info("Using rule-based briefing parser fallback.")
        llm_payload = _parse_with_rules(raw_input)
    merged = _merge_briefing_data(asdict(defaults), llm_payload)
    return _briefing_from_mapping(merged)


def collect_briefing_interactive() -> Briefing:
    """Run a CLI wizard that collects and confirms a pre-meeting briefing."""
    while True:
        answers: list[tuple[str, str]] = []
        questions = [
            ("meeting", "What is this meeting about? (title and objective)"),
            ("attendees", "Who will be attending? (names/roles, or skip)"),
            ("identity", "What name and role should I use if I speak on your behalf?"),
            ("mode", f"What role should I play? Options: {_mode_options()}"),
            ("persona", "How should I sound if I speak as you? (tone, phrases, style, or skip)"),
            ("authority", "What am I allowed to decide or commit to on your behalf?"),
            ("points", "Are there specific points you want me to raise? (list them or skip)"),
            ("avoid", "Any topics I should absolutely avoid?"),
            ("escalation", "When should I defer or ask you before answering?"),
            ("context", "Any background context I should know?"),
            ("style", "How should I sound? (formal / casual / concise / detailed)"),
            ("respond", "Should I respond automatically, or only when you prompt me?"),
        ]
        for key, question in questions:
            answer = input(f"{question}\n> ").strip()
            answers.append((key, answer))

        raw = "\n".join(f"{key}: {answer}" for key, answer in answers if answer)
        briefing = parse_briefing_from_text(raw)
        print(_format_briefing_summary(briefing))
        confirmation = input("Does this look correct? (yes / edit)\n> ").strip().lower()
        if confirmation in {"yes", "y"}:
            return briefing
        print("No problem. Let's collect the briefing again.")


def briefing_to_dict(briefing: Briefing) -> dict[str, Any]:
    """Convert a Briefing to a JSON-serializable dictionary."""
    data = asdict(briefing)
    data["agent_mode"] = briefing.agent_mode.value
    return data


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _default_briefing(raw_input: str) -> Briefing:
    title = _first_non_empty_line(raw_input) or "Untitled Meeting"
    return Briefing(
        meeting_title=_clean_title(title),
        meeting_objective="Capture the discussion and identify important outcomes.",
        attendees=[],
        user_role="participant",
        agent_mode=AgentMode.SILENT_OBSERVER,
        talking_points=[],
        topics_to_avoid=[],
        background_context=raw_input.strip(),
        response_style="concise",
        custom_instructions=[],
        max_speaking_turns=0,
        auto_respond=False,
        user_name="",
        speaking_persona="",
        decision_authority="Do not make binding commitments unless explicitly authorized in the briefing.",
        escalation_rules=[],
        assistant_display_name="",
    )


def _parse_with_llm(raw_input: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("OpenAI package unavailable for briefing parsing: %s", exc)
        return None

    schema_description = {
        "meeting_title": "short title",
        "meeting_objective": "expected meeting outcome",
        "attendees": ["names or roles"],
        "user_name": "the name the agent should use when speaking as the user",
        "user_role": "role the human user wants to play",
        "agent_mode": [mode.value for mode in AgentMode],
        "talking_points": ["points the user wants raised"],
        "topics_to_avoid": ["topics the user forbids or wants avoided"],
        "background_context": "free-text background",
        "response_style": "concise, detailed, formal, casual, or another style",
        "custom_instructions": ["per-meeting rules"],
        "max_speaking_turns": "integer, 0 means silent observer",
        "auto_respond": "boolean",
        "speaking_persona": "how the user naturally sounds, including tone, phrases, preferences",
        "decision_authority": "what the agent may decide or commit to on the user's behalf",
        "escalation_rules": ["situations where the agent should defer, ask for help, or avoid answering"],
        "assistant_display_name": "transparent meeting display name, for example 'Sai - AI Assistant'",
    }
    prompt = (
        "Extract a complete AI meeting assistant briefing from the user's raw text. "
        "Return strict JSON using exactly these keys. Use sensible defaults for missing fields. "
        "If the user wants silence or observation, set agent_mode to SILENT_OBSERVER, "
        "auto_respond to false, and max_speaking_turns to 0.\n\n"
        f"Schema guidance:\n{json.dumps(schema_description, indent=2)}\n\n"
        f"Raw briefing:\n{raw_input}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_BRIEFING_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[
                {
                    "role": "system",
                    "content": "You produce only valid JSON for meeting-assistant configuration.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.warning("LLM briefing parse failed; falling back to rules: %s", exc)
        return None


def _parse_with_rules(raw_input: str) -> dict[str, Any]:
    text = raw_input.strip()
    lower = text.lower()
    data: dict[str, Any] = {}
    data["agent_mode"] = _detect_mode(lower).value
    data["auto_respond"] = _detect_auto_respond(lower, data["agent_mode"])
    data["max_speaking_turns"] = (
        _extract_labeled_value(text, ["max speaking turns", "speaking turns", "max turns"])
        or _detect_max_turns(lower, data["agent_mode"])
    )
    data["response_style"] = _detect_response_style(lower)
    data["meeting_title"] = _extract_labeled_value(text, ["title", "meeting"]) or _clean_title(_first_non_empty_line(text))
    data["meeting_objective"] = _extract_labeled_value(text, ["objective", "goal", "outcome"]) or ""
    data["attendees"] = _extract_list(text, ["attendees", "attending", "who"])
    data["user_name"] = _extract_labeled_value(text, ["user name", "my name", "name"]) or ""
    data["talking_points"] = _extract_list(text, ["talking points", "points", "raise"])
    data["topics_to_avoid"] = _extract_list(text, ["avoid", "topics to avoid", "do not discuss"])
    data["background_context"] = _extract_labeled_value(text, ["context", "background"]) or text
    data["user_role"] = _extract_labeled_value(text, ["user role", "my role", "role"]) or ""
    data["custom_instructions"] = _extract_list(text, ["instructions", "rules"])
    data["speaking_persona"] = _extract_labeled_value(text, ["speaking persona", "persona", "sound like me", "speaking style"]) or ""
    data["decision_authority"] = _extract_labeled_value(text, ["decision authority", "authority", "can decide", "commitments"]) or ""
    data["escalation_rules"] = _extract_list(text, ["escalation rules", "escalate", "ask me when", "defer"])
    data["assistant_display_name"] = _extract_labeled_value(
        text,
        ["assistant display name", "meeting display name", "display name", "join name"],
    ) or ""
    return data


def _merge_briefing_data(defaults: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in incoming.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        merged[key] = value
    return merged


def _briefing_from_mapping(data: dict[str, Any]) -> Briefing:
    mode = _coerce_agent_mode(data.get("agent_mode"))
    max_turns = _coerce_int(data.get("max_speaking_turns"), default=0)
    auto_respond = _coerce_bool(data.get("auto_respond"), default=False)
    if mode == AgentMode.SILENT_OBSERVER:
        max_turns = 0
        auto_respond = False
    return Briefing(
        meeting_title=str(data.get("meeting_title") or "Untitled Meeting").strip(),
        meeting_objective=str(data.get("meeting_objective") or "Capture important meeting outcomes.").strip(),
        attendees=_coerce_str_list(data.get("attendees")),
        user_role=str(data.get("user_role") or "participant").strip(),
        agent_mode=mode,
        talking_points=_coerce_str_list(data.get("talking_points")),
        topics_to_avoid=_coerce_str_list(data.get("topics_to_avoid")),
        background_context=str(data.get("background_context") or "").strip(),
        response_style=str(data.get("response_style") or "concise").strip(),
        custom_instructions=_coerce_str_list(data.get("custom_instructions")),
        max_speaking_turns=max(0, max_turns),
        auto_respond=auto_respond,
        user_name=str(data.get("user_name") or "").strip(),
        speaking_persona=str(data.get("speaking_persona") or "").strip(),
        decision_authority=str(
            data.get("decision_authority")
            or "Do not make binding commitments unless explicitly authorized in the briefing."
        ).strip(),
        escalation_rules=_coerce_str_list(data.get("escalation_rules")),
        assistant_display_name=str(data.get("assistant_display_name") or "").strip(),
    )


def _coerce_agent_mode(value: Any) -> AgentMode:
    if isinstance(value, AgentMode):
        return value
    normalized = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "FULL": AgentMode.FULL_PROXY,
        "PROXY": AgentMode.FULL_PROXY,
        "FULL_PROXY": AgentMode.FULL_PROXY,
        "SILENT": AgentMode.SILENT_OBSERVER,
        "OBSERVER": AgentMode.SILENT_OBSERVER,
        "SILENT_OBSERVER": AgentMode.SILENT_OBSERVER,
        "ADVISOR": AgentMode.ADVISOR,
        "ADVISER": AgentMode.ADVISOR,
        "PARTICIPATOR": AgentMode.PARTICIPATOR,
        "PARTICIPANT": AgentMode.PARTICIPATOR,
    }
    return aliases.get(normalized, AgentMode.SILENT_OBSERVER)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[\n;,]+", value)
    elif isinstance(value, (list, tuple, set)):
        pieces = [str(item) for item in value]
    else:
        pieces = [str(value)]
    return [piece.strip(" -\t") for piece in pieces if piece and piece.strip(" -\t")]


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y", "1", "auto", "automatically", "automatic"}:
        return True
    if normalized in {"false", "no", "n", "0", "manual", "prompt", "only when prompted"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detect_mode(lower_text: str) -> AgentMode:
    if any(term in lower_text for term in ["full proxy", "speak on my behalf", "as me", "proxy"]):
        return AgentMode.FULL_PROXY
    if any(term in lower_text for term in ["attend on my behalf", "represent me", "stand in for me"]):
        return AgentMode.FULL_PROXY
    if any(term in lower_text for term in ["advisor", "adviser", "whisper", "suggestions", "text-only", "text only"]):
        return AgentMode.ADVISOR
    if any(term in lower_text for term in ["participate", "contribute", "join naturally", "participant"]):
        return AgentMode.PARTICIPATOR
    return AgentMode.SILENT_OBSERVER


def _detect_auto_respond(lower_text: str, mode: str) -> bool:
    if mode == AgentMode.SILENT_OBSERVER.value:
        return False
    if any(term in lower_text for term in ["automatically", "auto respond", "speak up", "autonomous"]):
        return True
    if any(term in lower_text for term in ["only when", "when prompted", "manual", "do not respond"]):
        return False
    return False


def _detect_max_turns(lower_text: str, mode: str) -> int:
    if mode == AgentMode.SILENT_OBSERVER.value:
        return 0
    match = re.search(r"(?:max|maximum|up to)\s+(\d+)\s+(?:speaking\s+)?turns?", lower_text)
    if match:
        return int(match.group(1))
    if mode == AgentMode.ADVISOR.value:
        return 0
    return 3


def _detect_response_style(lower_text: str) -> str:
    for style in ["concise", "detailed", "formal", "casual"]:
        if style in lower_text:
            return style
    return "concise"


def _extract_labeled_value(text: str, labels: list[str]) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        for label in labels:
            pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return ""


def _extract_list(text: str, labels: list[str]) -> list[str]:
    value = _extract_labeled_value(text, labels)
    if not value or value.lower() in {"skip", "none", "n/a", "na"}:
        return []
    return _coerce_str_list(value)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _clean_title(title: str) -> str:
    cleaned = re.sub(r"^(meeting|title)\s*:\s*", "", title or "", flags=re.IGNORECASE).strip()
    return cleaned[:120] if cleaned else "Untitled Meeting"


def _mode_options() -> str:
    return ", ".join(mode.value for mode in AgentMode)


def _format_briefing_summary(briefing: Briefing) -> str:
    data = briefing_to_dict(briefing)
    lines = ["\nBriefing summary:"]
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
